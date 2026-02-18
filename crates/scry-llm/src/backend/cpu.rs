use crate::backend::{DeviceBackend, MathBackend};
use crate::tensor::shape::Shape;

/// CPU reference backend. Correctness over performance.
pub struct CpuBackend;

impl DeviceBackend for CpuBackend {
    type Storage = Vec<f32>;
    type Stream = ();

    fn zeros(shape: &Shape) -> Vec<f32> {
        vec![0.0; shape.numel()]
    }

    fn ones(shape: &Shape) -> Vec<f32> {
        vec![1.0; shape.numel()]
    }

    fn from_vec(data: Vec<f32>, _shape: &Shape) -> Vec<f32> {
        data
    }

    fn to_vec(storage: &Vec<f32>) -> Vec<f32> {
        storage.clone()
    }

    fn clone_storage(storage: &Vec<f32>) -> Vec<f32> {
        storage.clone()
    }
}

impl MathBackend for CpuBackend {
    fn matmul(
        a: &Vec<f32>,
        b: &Vec<f32>,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> Vec<f32> {
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for p in 0..k {
                    let a_val = if trans_a {
                        a[p * m + i] // A is [K, M] stored row-major
                    } else {
                        a[i * k + p] // A is [M, K] stored row-major
                    };
                    let b_val = if trans_b {
                        b[j * k + p] // B is [N, K] stored row-major
                    } else {
                        b[p * n + j] // B is [K, N] stored row-major
                    };
                    sum += f64::from(a_val) * f64::from(b_val);
                }
                c[i * n + j] = sum as f32;
            }
        }
        c
    }

    fn add(
        a: &Vec<f32>,
        b: &Vec<f32>,
        a_shape: &Shape,
        b_shape: &Shape,
        out_shape: &Shape,
    ) -> Vec<f32> {
        let out_numel = out_shape.numel();
        let mut result = vec![0.0; out_numel];

        let a_strides = a_shape.broadcast_strides(out_shape);
        let b_strides = b_shape.broadcast_strides(out_shape);
        let out_strides = out_shape.strides();
        let ndim = out_shape.ndim();

        for idx in 0..out_numel {
            // Decompose flat index into multi-dim coords
            let mut remaining = idx;
            let mut a_offset = 0;
            let mut b_offset = 0;
            for d in 0..ndim {
                let coord = remaining / out_strides[d];
                remaining %= out_strides[d];
                a_offset += coord * a_strides[d];
                b_offset += coord * b_strides[d];
            }
            result[idx] = a[a_offset] + b[b_offset];
        }
        result
    }

    fn softmax(input: &Vec<f32>, shape: &Shape) -> Vec<f32> {
        let dims = shape.dims();
        let last = *dims.last().unwrap();
        let batch = input.len() / last;
        let mut output = vec![0.0; input.len()];

        for b in 0..batch {
            let start = b * last;
            let slice = &input[start..start + last];

            let max_val = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f64;
            for i in 0..last {
                let e = f64::from((slice[i] - max_val).exp());
                output[start + i] = e as f32;
                sum += e;
            }
            for i in 0..last {
                output[start + i] = (f64::from(output[start + i]) / sum) as f32;
            }
        }
        output
    }

    fn layernorm(
        input: &Vec<f32>,
        gamma: &Vec<f32>,
        beta: &Vec<f32>,
        shape: &Shape,
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let dims = shape.dims();
        let d = *dims.last().unwrap();
        let n = input.len() / d;
        let mut output = vec![0.0; input.len()];
        let mut means = vec![0.0; n];
        let mut rstds = vec![0.0; n];

        for i in 0..n {
            let start = i * d;
            let slice = &input[start..start + d];

            // Mean (f64 accumulation)
            let mean = slice.iter().map(|&x| f64::from(x)).sum::<f64>() / d as f64;
            means[i] = mean as f32;

            // Variance
            let var = slice
                .iter()
                .map(|&x| {
                    let diff = f64::from(x) - mean;
                    diff * diff
                })
                .sum::<f64>()
                / d as f64;

            let rstd = 1.0 / (var + f64::from(eps)).sqrt();
            rstds[i] = rstd as f32;

            // Normalize + affine
            for j in 0..d {
                let norm = (f64::from(slice[j]) - mean) * rstd;
                output[start + j] = (norm * f64::from(gamma[j]) + f64::from(beta[j])) as f32;
            }
        }

        (output, means, rstds)
    }

    fn gelu(input: &Vec<f32>) -> Vec<f32> {
        input.iter().map(|&x| gelu_scalar(x)).collect()
    }

    fn cross_entropy(logits: &Vec<f32>, targets: &[usize], batch: usize, vocab: usize) -> f32 {
        let mut total_loss = 0.0f64;
        for b in 0..batch {
            let start = b * vocab;
            let slice = &logits[start..start + vocab];
            let target = targets[b];

            // log-softmax with max-subtract trick
            let max_val = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f64 = slice.iter().map(|&x| f64::from((x - max_val).exp())).sum();
            let log_sum_exp = f64::from(max_val) + sum_exp.ln();
            let log_prob = f64::from(slice[target]) - log_sum_exp;
            total_loss -= log_prob;
        }
        (total_loss / batch as f64) as f32
    }

    fn embedding(weight: &Vec<f32>, indices: &[usize], vocab: usize, dim: usize) -> Vec<f32> {
        assert!(
            indices.iter().all(|&i| i < vocab),
            "embedding index out of range: max index {}, vocab size {}",
            indices.iter().copied().max().unwrap_or(0),
            vocab
        );
        let mut output = vec![0.0; indices.len() * dim];
        for (i, &idx) in indices.iter().enumerate() {
            let src = &weight[idx * dim..(idx + 1) * dim];
            output[i * dim..(i + 1) * dim].copy_from_slice(src);
        }
        output
    }

    fn sum(input: &Vec<f32>) -> f32 {
        let s: f64 = input.iter().map(|&x| f64::from(x)).sum();
        s as f32
    }

    // ---- Backward ops ----

    fn matmul_backward(
        d_out: &Vec<f32>,
        a: &Vec<f32>,
        b: &Vec<f32>,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        // d_out is [M, N]
        // Need d_a and d_b
        match (trans_a, trans_b) {
            (false, false) => {
                // C = A @ B → dA = dC @ B^T, dB = A^T @ dC
                let d_a = Self::matmul(d_out, b, m, n, k, false, true);
                let d_b = Self::matmul(a, d_out, k, m, n, true, false);
                (d_a, d_b)
            }
            (true, false) => {
                // C = A^T @ B (A stored as [K,M]) → dA = B @ dC^T, dB = A @ dC
                let d_a = Self::matmul(b, d_out, k, n, m, false, true);
                let d_b = Self::matmul(a, d_out, k, m, n, false, false);
                (d_a, d_b)
            }
            (false, true) => {
                // C = A @ B^T (B stored as [N,K]) → dA = dC @ B, dB = dC^T @ A
                let d_a = Self::matmul(d_out, b, m, n, k, false, false);
                let d_b = Self::matmul(d_out, a, n, m, k, true, false);
                (d_a, d_b)
            }
            (true, true) => {
                // C = A^T @ B^T → dA = B^T @ dC^T = (dC @ B)^T, dB = dC^T @ A^T = (A @ dC)^T
                let d_a = Self::matmul(b, d_out, k, n, m, true, true);
                let d_b = Self::matmul(d_out, a, n, m, k, true, true);
                (d_a, d_b)
            }
        }
    }

    fn add_backward(
        d_out: &Vec<f32>,
        a_shape: &Shape,
        b_shape: &Shape,
        out_shape: &Shape,
    ) -> (Vec<f32>, Vec<f32>) {
        let d_a = reduce_broadcast(d_out, out_shape, a_shape);
        let d_b = reduce_broadcast(d_out, out_shape, b_shape);
        (d_a, d_b)
    }

    fn softmax_backward(d_out: &Vec<f32>, output: &Vec<f32>, shape: &Shape) -> Vec<f32> {
        let dims = shape.dims();
        let last = *dims.last().unwrap();
        let batch = output.len() / last;
        let mut d_input = vec![0.0; output.len()];

        for b in 0..batch {
            let start = b * last;
            // dot = sum(dY * Y)
            let mut dot = 0.0f64;
            for i in 0..last {
                dot += f64::from(d_out[start + i]) * f64::from(output[start + i]);
            }
            // dX = Y * (dY - dot)
            for i in 0..last {
                d_input[start + i] =
                    (f64::from(output[start + i]) * (f64::from(d_out[start + i]) - dot)) as f32;
            }
        }
        d_input
    }

    fn layernorm_backward(
        d_out: &Vec<f32>,
        input: &Vec<f32>,
        gamma: &Vec<f32>,
        mean: &Vec<f32>,
        rstd: &Vec<f32>,
        shape: &Shape,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let dims = shape.dims();
        let d = *dims.last().unwrap();
        let n = input.len() / d;

        let mut d_input = vec![0.0f32; input.len()];
        let mut d_gamma = vec![0.0f64; d];
        let mut d_beta = vec![0.0f64; d];

        for i in 0..n {
            let start = i * d;
            let m = f64::from(mean[i]);
            let rs = f64::from(rstd[i]);

            // dnorm = d_out * gamma
            // norm = (input - mean) * rstd
            // sum_dnorm = sum(dnorm)
            // sum_dnorm_norm = sum(dnorm * norm)
            let mut sum_dnorm = 0.0f64;
            let mut sum_dnorm_norm = 0.0f64;

            for j in 0..d {
                let dnorm = f64::from(d_out[start + j]) * f64::from(gamma[j]);
                let norm = (f64::from(input[start + j]) - m) * rs;
                sum_dnorm += dnorm;
                sum_dnorm_norm += dnorm * norm;

                d_gamma[j] += f64::from(d_out[start + j]) * norm;
                d_beta[j] += f64::from(d_out[start + j]);
            }

            // dx = (1/D) * rstd * (D*dnorm - sum_dnorm - norm*sum_dnorm_norm)
            let inv_d = 1.0 / d as f64;
            for j in 0..d {
                let dnorm = f64::from(d_out[start + j]) * f64::from(gamma[j]);
                let norm = (f64::from(input[start + j]) - m) * rs;
                d_input[start + j] =
                    (inv_d * rs * (d as f64 * dnorm - sum_dnorm - norm * sum_dnorm_norm)) as f32;
            }
        }

        let d_gamma_f32: Vec<f32> = d_gamma.iter().map(|&x| x as f32).collect();
        let d_beta_f32: Vec<f32> = d_beta.iter().map(|&x| x as f32).collect();
        (d_input, d_gamma_f32, d_beta_f32)
    }

    fn gelu_backward(d_out: &Vec<f32>, input: &Vec<f32>) -> Vec<f32> {
        d_out
            .iter()
            .zip(input.iter())
            .map(|(&dy, &x)| dy * gelu_derivative(x))
            .collect()
    }

    fn cross_entropy_backward(
        logits: &Vec<f32>,
        targets: &[usize],
        batch: usize,
        vocab: usize,
    ) -> Vec<f32> {
        let mut d_logits = vec![0.0; batch * vocab];

        for b in 0..batch {
            let start = b * vocab;
            let slice = &logits[start..start + vocab];
            let target = targets[b];

            // Compute softmax
            let max_val = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f64;
            for &x in slice {
                sum_exp += f64::from((x - max_val).exp());
            }

            for j in 0..vocab {
                let prob = f64::from((slice[j] - max_val).exp()) / sum_exp;
                let target_val = if j == target { 1.0 } else { 0.0 };
                d_logits[start + j] = ((prob - target_val) / batch as f64) as f32;
            }
        }
        d_logits
    }

    fn embedding_backward(
        d_out: &Vec<f32>,
        indices: &[usize],
        vocab: usize,
        dim: usize,
    ) -> Vec<f32> {
        let mut d_weight = vec![0.0; vocab * dim];
        for (i, &idx) in indices.iter().enumerate() {
            for j in 0..dim {
                d_weight[idx * dim + j] += d_out[i * dim + j];
            }
        }
        d_weight
    }

    fn mul_elementwise(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
    }

    fn scale(a: &Vec<f32>, scalar: f32) -> Vec<f32> {
        a.iter().map(|&x| x * scalar).collect()
    }

    fn add_inplace(a: &mut Vec<f32>, b: &Vec<f32>) {
        for (x, y) in a.iter_mut().zip(b.iter()) {
            *x += *y;
        }
    }

    fn adamw_step(
        param: &mut Vec<f32>,
        grad: &Vec<f32>,
        m: &mut Vec<f32>,
        v: &mut Vec<f32>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: u32,
    ) {
        let beta1_64 = f64::from(beta1);
        let beta2_64 = f64::from(beta2);
        #[allow(clippy::cast_possible_wrap)]
        let step_i32 = step as i32;
        let bc1 = 1.0 - beta1_64.powi(step_i32);
        let bc2 = 1.0 - beta2_64.powi(step_i32);

        for i in 0..param.len() {
            let g = f64::from(grad[i]);

            // Moment updates
            m[i] = (f64::from(beta1) * f64::from(m[i]) + (1.0 - f64::from(beta1)) * g) as f32;
            v[i] = (f64::from(beta2) * f64::from(v[i]) + (1.0 - f64::from(beta2)) * g * g) as f32;

            // Bias-corrected moments
            let m_hat = f64::from(m[i]) / bc1;
            let v_hat = f64::from(v[i]) / bc2;

            // Weight decay + Adam update
            let update = m_hat / (v_hat.sqrt() + f64::from(eps));
            param[i] = (f64::from(param[i])
                - f64::from(lr) * (update + f64::from(weight_decay) * f64::from(param[i])))
                as f32;
        }
    }
}

// ---- Helper functions ----

/// `GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044_715 * x^3)))`
fn gelu_scalar(x: f32) -> f32 {
    let x64 = f64::from(x);
    let sqrt_2_over_pi: f64 = (2.0 / std::f64::consts::PI).sqrt();
    let inner = sqrt_2_over_pi * (x64 + 0.044_715 * x64 * x64 * x64);
    (0.5 * x64 * (1.0 + inner.tanh())) as f32
}

/// Derivative of GELU (tanh approximation).
fn gelu_derivative(x: f32) -> f32 {
    let x64 = f64::from(x);
    let sqrt_2_over_pi: f64 = (2.0 / std::f64::consts::PI).sqrt();
    let kappa = 0.044_715;
    let inner = sqrt_2_over_pi * (x64 + kappa * x64 * x64 * x64);
    let tanh_val = inner.tanh();
    let sech2 = 1.0 - tanh_val * tanh_val;
    let d_inner = sqrt_2_over_pi * (1.0 + 3.0 * kappa * x64 * x64);
    (0.5 * (1.0 + tanh_val) + 0.5 * x64 * sech2 * d_inner) as f32
}

/// Reduce `d_out` from `out_shape` back to `target_shape` by summing along broadcast dims.
fn reduce_broadcast(d_out: &[f32], out_shape: &Shape, target_shape: &Shape) -> Vec<f32> {
    let target_numel = target_shape.numel();

    // If shapes match, just clone
    if out_shape == target_shape {
        return d_out.to_vec();
    }

    let mut result = vec![0.0f32; target_numel];
    let target_strides = target_shape.broadcast_strides(out_shape);
    let out_strides = out_shape.strides();
    let ndim = out_shape.ndim();

    for idx in 0..out_shape.numel() {
        let mut remaining = idx;
        let mut target_offset = 0;
        for d in 0..ndim {
            let coord = remaining / out_strides[d];
            remaining %= out_strides[d];
            target_offset += coord * target_strides[d];
        }
        result[target_offset] += d_out[idx];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_identity() {
        // 2x2 identity @ [1,2; 3,4] = [1,2; 3,4]
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let c = CpuBackend::matmul(&a, &b, 2, 2, 2, false, false);
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn softmax_basic() {
        let input = vec![1.0, 2.0, 3.0];
        let shape = Shape::new(&[1, 3]);
        let output = CpuBackend::softmax(&input, &shape);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(output[2] > output[1]);
        assert!(output[1] > output[0]);
    }

    #[test]
    fn gelu_zero() {
        assert!((gelu_scalar(0.0)).abs() < 1e-7);
    }

    #[test]
    fn cross_entropy_perfect() {
        // logits where correct class has very high score
        let logits = vec![10.0, -10.0, -10.0];
        let targets = vec![0usize];
        let loss = CpuBackend::cross_entropy(&logits, &targets, 1, 3);
        assert!(loss < 0.001);
    }
}
