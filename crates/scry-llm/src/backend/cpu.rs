use rayon::prelude::*;

use crate::backend::{DeviceBackend, MathBackend};
use crate::tensor::shape::Shape;

/// CPU reference backend. Correctness over performance.
pub struct CpuBackend;

/// Minimum number of rows (row-tiles) before engaging rayon for matmul.
#[cfg(not(feature = "blas"))]
const MATMUL_PAR_THRESHOLD: usize = 128;
/// Minimum number of rows before parallelizing row-wise ops.
const ROW_PAR_THRESHOLD: usize = 64;
/// Minimum element count before parallelizing element-wise ops.
const ELEM_PAR_THRESHOLD: usize = 8192;

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
        matmul_tiled(a, b, m, k, n, trans_a, trans_b)
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
        let mut output = vec![0.0f32; input.len()];

        let process_row = |out_row: &mut [f32], in_row: &[f32]| {
            let max_val = in_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f64;
            for i in 0..last {
                let e = f64::from((in_row[i] - max_val).exp());
                out_row[i] = e as f32;
                sum += e;
            }
            for i in 0..last {
                out_row[i] = (f64::from(out_row[i]) / sum) as f32;
            }
        };

        if batch >= ROW_PAR_THRESHOLD {
            output
                .par_chunks_mut(last)
                .zip(input.par_chunks(last))
                .for_each(|(o, i)| process_row(o, i));
        } else {
            output
                .chunks_mut(last)
                .zip(input.chunks(last))
                .for_each(|(o, i)| process_row(o, i));
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
        let mut output = vec![0.0f32; input.len()];
        let mut means = vec![0.0f32; n];
        let mut rstds = vec![0.0f32; n];

        let process_row = |i: usize, out_row: &mut [f32], mean_out: &mut f32, rstd_out: &mut f32| {
            let start = i * d;
            let slice = &input[start..start + d];

            let mean = slice.iter().map(|&x| f64::from(x)).sum::<f64>() / d as f64;
            *mean_out = mean as f32;

            let var = slice
                .iter()
                .map(|&x| {
                    let diff = f64::from(x) - mean;
                    diff * diff
                })
                .sum::<f64>()
                / d as f64;

            let rstd = 1.0 / (var + f64::from(eps)).sqrt();
            *rstd_out = rstd as f32;

            for j in 0..d {
                let norm = (f64::from(slice[j]) - mean) * rstd;
                out_row[j] = (norm * f64::from(gamma[j]) + f64::from(beta[j])) as f32;
            }
        };

        if n >= ROW_PAR_THRESHOLD {
            output
                .par_chunks_mut(d)
                .zip(means.par_iter_mut().zip(rstds.par_iter_mut()))
                .enumerate()
                .for_each(|(i, (out_row, (mean_out, rstd_out)))| {
                    process_row(i, out_row, mean_out, rstd_out);
                });
        } else {
            output
                .chunks_mut(d)
                .zip(means.iter_mut().zip(rstds.iter_mut()))
                .enumerate()
                .for_each(|(i, (out_row, (mean_out, rstd_out)))| {
                    process_row(i, out_row, mean_out, rstd_out);
                });
        }

        (output, means, rstds)
    }

    fn gelu(input: &Vec<f32>) -> Vec<f32> {
        if input.len() >= ELEM_PAR_THRESHOLD {
            input.par_iter().map(|&x| gelu_scalar(x)).collect()
        } else {
            input.iter().map(|&x| gelu_scalar(x)).collect()
        }
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

    fn mul_elementwise(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
    }

    fn scale(a: &Vec<f32>, scalar: f32) -> Vec<f32> {
        a.iter().map(|&x| x * scalar).collect()
    }

    fn concat_rows(
        a: &Vec<f32>,
        b: &Vec<f32>,
        _a_rows: usize,
        _b_rows: usize,
        _cols: usize,
    ) -> Vec<f32> {
        let mut out = Vec::with_capacity(a.len() + b.len());
        out.extend_from_slice(a);
        out.extend_from_slice(b);
        out
    }

    // ---- Llama-specific ops ----

    fn rmsnorm(
        input: &Vec<f32>,
        weight: &Vec<f32>,
        shape: &Shape,
        eps: f32,
    ) -> Vec<f32> {
        let dims = shape.dims();
        let d = *dims.last().unwrap();
        let n = input.len() / d;
        let mut output = vec![0.0f32; input.len()];

        for i in 0..n {
            let start = i * d;
            let slice = &input[start..start + d];

            // mean(x^2)
            let mean_sq = slice
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum::<f64>()
                / d as f64;

            let rstd = 1.0 / (mean_sq + f64::from(eps)).sqrt();

            for j in 0..d {
                output[start + j] = (f64::from(slice[j]) * rstd * f64::from(weight[j])) as f32;
            }
        }

        output
    }

    fn rope(
        input: &Vec<f32>,
        shape: &Shape,
        pos: usize,
        head_dim: usize,
        theta: f32,
    ) -> Vec<f32> {
        let dims = shape.dims();
        let total_elements = input.len();
        let last_dim = *dims.last().unwrap();
        let n_rows = total_elements / last_dim;
        let mut output = input.clone();

        let theta_f64 = f64::from(theta);

        for row in 0..n_rows {
            let row_start = row * last_dim;
            // Apply RoPE to each pair of dimensions within each head
            let n_heads_in_row = last_dim / head_dim;
            for h in 0..n_heads_in_row {
                let head_start = row_start + h * head_dim;
                for i in 0..head_dim / 2 {
                    let freq = 1.0 / theta_f64.powf(2.0 * i as f64 / head_dim as f64);
                    let angle = pos as f64 * freq;
                    let cos_val = angle.cos() as f32;
                    let sin_val = angle.sin() as f32;

                    let idx0 = head_start + 2 * i;
                    let idx1 = head_start + 2 * i + 1;
                    let x0 = input[idx0];
                    let x1 = input[idx1];
                    output[idx0] = x0 * cos_val - x1 * sin_val;
                    output[idx1] = x0 * sin_val + x1 * cos_val;
                }
            }
        }

        output
    }

    fn rope_with_freqs_preloaded(
        input: &Vec<f32>,
        seq: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
        freqs: &Vec<f32>,
    ) -> Vec<f32> {
        let total_dim = n_heads * head_dim;
        let mut output = input.clone();
        for s in 0..seq {
            let pos = (start_pos + s) as f64;
            let row_start = s * total_dim;
            for h in 0..n_heads {
                let head_start = row_start + h * head_dim;
                for (i, &freq) in freqs.iter().enumerate() {
                    let angle = pos * f64::from(freq);
                    let cos_val = angle.cos() as f32;
                    let sin_val = angle.sin() as f32;
                    let idx0 = head_start + 2 * i;
                    let idx1 = head_start + 2 * i + 1;
                    let x0 = input[idx0];
                    let x1 = input[idx1];
                    output[idx0] = x0 * cos_val - x1 * sin_val;
                    output[idx1] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
        output
    }

    fn swiglu(gate: &Vec<f32>, up: &Vec<f32>) -> Vec<f32> {
        if gate.len() >= ELEM_PAR_THRESHOLD {
            gate.par_iter()
                .zip(up.par_iter())
                .map(|(&g, &u)| silu_scalar(g) * u)
                .collect()
        } else {
            gate.iter()
                .zip(up.iter())
                .map(|(&g, &u)| silu_scalar(g) * u)
                .collect()
        }
    }

    fn repeat_kv(
        input: &Vec<f32>,
        n_kv_heads: usize,
        n_q_heads: usize,
        seq: usize,
        d_head: usize,
    ) -> Vec<f32> {
        let n_rep = n_q_heads / n_kv_heads;
        if n_rep == 1 {
            return input.clone();
        }

        let head_size = seq * d_head;
        let mut output = vec![0.0f32; n_q_heads * head_size];

        for kv_h in 0..n_kv_heads {
            let src_start = kv_h * head_size;
            let src = &input[src_start..src_start + head_size];
            for r in 0..n_rep {
                let dst_start = (kv_h * n_rep + r) * head_size;
                output[dst_start..dst_start + head_size].copy_from_slice(src);
            }
        }

        output
    }

    fn gather_reshape_repeat_kv(
        cache: &Vec<f32>,
        _max_seq: usize,
        cached_len: usize,
        n_kv_heads: usize,
        n_q_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let n_rep = n_q_heads / n_kv_heads;
        let kv_dim = n_kv_heads * head_dim;
        let total = n_q_heads * cached_len * head_dim;
        let mut out = vec![0.0f32; total];

        for q_head in 0..n_q_heads {
            let kv_head = q_head / n_rep;
            for s in 0..cached_len {
                let dst = (q_head * cached_len + s) * head_dim;
                let src = s * kv_dim + kv_head * head_dim;
                out[dst..dst + head_dim].copy_from_slice(&cache[src..src + head_dim]);
            }
        }

        out
    }
}

// ---- Helper functions ----

/// Compute one row-tile of C for the tiled matmul.
#[cfg(not(feature = "blas"))]
fn matmul_tile_rows(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
    i_tile: usize,
    i_end: usize,
) -> Vec<f64> {
    const T: usize = 32;
    let rows = i_end - i_tile;
    let mut local = vec![0.0f64; rows * n];

    let mut p_tile = 0;
    while p_tile < k {
        let p_end = (p_tile + T).min(k);
        let mut j_tile = 0;
        while j_tile < n {
            let j_end = (j_tile + T).min(n);

            for i in i_tile..i_end {
                let li = i - i_tile;
                for p in p_tile..p_end {
                    let a_val = if trans_a {
                        f64::from(a[p * m + i])
                    } else {
                        f64::from(a[i * k + p])
                    };
                    for j in j_tile..j_end {
                        let b_val = if trans_b {
                            f64::from(b[j * k + p])
                        } else {
                            f64::from(b[p * n + j])
                        };
                        local[li * n + j] += a_val * b_val;
                    }
                }
            }

            j_tile += T;
        }
        p_tile += T;
    }
    local
}

/// Tiled matmul with optional rayon parallelism.
fn matmul_tiled(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) -> Vec<f32> {
    #[cfg(feature = "blas")]
    {
        return matmul_cblas(a, b, m, k, n, trans_a, trans_b);
    }

    #[cfg(not(feature = "blas"))]
    {
        matmul_tiled_fallback(a, b, m, k, n, trans_a, trans_b)
    }
}

#[cfg(not(feature = "blas"))]
fn matmul_tiled_fallback(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) -> Vec<f32> {
    const T: usize = 32;
    let i_tiles: Vec<usize> = (0..m).step_by(T).collect();

    let row_chunks: Vec<Vec<f64>> = if m >= MATMUL_PAR_THRESHOLD {
        i_tiles
            .par_iter()
            .map(|&i_tile| {
                let i_end = (i_tile + T).min(m);
                matmul_tile_rows(a, b, m, k, n, trans_a, trans_b, i_tile, i_end)
            })
            .collect()
    } else {
        i_tiles
            .iter()
            .map(|&i_tile| {
                let i_end = (i_tile + T).min(m);
                matmul_tile_rows(a, b, m, k, n, trans_a, trans_b, i_tile, i_end)
            })
            .collect()
    };

    let mut c = Vec::with_capacity(m * n);
    for chunk in &row_chunks {
        c.extend(chunk.iter().map(|&x| x as f32));
    }
    c
}

/// BLAS-accelerated matmul via cblas_sgemm.
#[cfg(feature = "blas")]
fn matmul_cblas(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) -> Vec<f32> {
    use cblas_sys::{cblas_sgemm, CBLAS_LAYOUT, CBLAS_TRANSPOSE};

    let mut c = vec![0.0f32; m * n];

    let (transa, lda) = if trans_a {
        (CBLAS_TRANSPOSE::CblasTrans, m as i32)
    } else {
        (CBLAS_TRANSPOSE::CblasNoTrans, k as i32)
    };

    let (transb, ldb) = if trans_b {
        (CBLAS_TRANSPOSE::CblasTrans, k as i32)
    } else {
        (CBLAS_TRANSPOSE::CblasNoTrans, n as i32)
    };

    unsafe {
        cblas_sgemm(
            CBLAS_LAYOUT::CblasRowMajor,
            transa,
            transb,
            m as i32,   // M
            n as i32,   // N
            k as i32,   // K
            1.0,        // alpha
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            0.0,        // beta
            c.as_mut_ptr(),
            n as i32,   // ldc
        );
    }

    c
}

/// `GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044_715 * x^3)))`
fn gelu_scalar(x: f32) -> f32 {
    let x64 = f64::from(x);
    let sqrt_2_over_pi: f64 = (2.0 / std::f64::consts::PI).sqrt();
    let inner = sqrt_2_over_pi * (x64 + 0.044_715 * x64 * x64 * x64);
    (0.5 * x64 * (1.0 + inner.tanh())) as f32
}

/// `SiLU(x) = x / (1 + exp(-x))`
fn silu_scalar(x: f32) -> f32 {
    let x64 = f64::from(x);
    (x64 / (1.0 + (-x64).exp())) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_identity() {
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
    fn rmsnorm_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let shape = Shape::new(&[1, 4]);
        let output = CpuBackend::rmsnorm(&input, &weight, &shape, 1e-5);
        // mean(x^2) = (1+4+9+16)/4 = 7.5, rstd = 1/sqrt(7.5) ≈ 0.3651
        let expected_rstd = 1.0 / (7.5f64 + 1e-5).sqrt();
        for i in 0..4 {
            let expected = (i + 1) as f64 * expected_rstd;
            assert!((f64::from(output[i]) - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn swiglu_basic() {
        let gate = vec![0.0, 1.0, -1.0];
        let up = vec![1.0, 1.0, 1.0];
        let output = CpuBackend::swiglu(&gate, &up);
        // silu(0) = 0, silu(1) ≈ 0.7311, silu(-1) ≈ -0.2689
        assert!(output[0].abs() < 1e-6);
        assert!((output[1] - 0.7311).abs() < 0.001);
        assert!((output[2] - (-0.2689)).abs() < 0.001);
    }

    #[test]
    fn repeat_kv_passthrough() {
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 1 head, seq=2, d_head=2
        let output = CpuBackend::repeat_kv(&input, 1, 1, 2, 2);
        assert_eq!(output, input);
    }

    #[test]
    fn repeat_kv_doubles() {
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 1 kv_head, seq=2, d_head=2
        let output = CpuBackend::repeat_kv(&input, 1, 2, 2, 2);
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rope_basic() {
        // Just verify it doesn't panic and modifies the values
        let input = vec![1.0, 0.0, 1.0, 0.0]; // 1 row, head_dim=4
        let shape = Shape::new(&[1, 4]);
        let output = CpuBackend::rope(&input, &shape, 1, 4, 10000.0);
        // At pos=1, first pair rotated by theta=1.0*1/(10000^0) = 1.0
        // cos(1) ≈ 0.5403, sin(1) ≈ 0.8415
        assert!((output[0] - 0.5403_f32).abs() < 0.001);
        assert!((output[1] - 0.8415_f32).abs() < 0.001);
    }

    #[test]
    fn gather_reshape_repeat_kv_matches_3step() {
        // Simulate a cache [max_seq=4, n_kv=2, hd=2] with cached_len=3
        // n_q_heads=4 (n_rep=2)
        let max_seq = 4;
        let n_kv = 2;
        let n_q = 4;
        let hd = 2;
        let kv_dim = n_kv * hd;
        let cached_len = 3;

        // Fill cache: row s, col c = (s * kv_dim + c) as f32
        let mut cache = vec![0.0f32; max_seq * kv_dim];
        for s in 0..max_seq {
            for c in 0..kv_dim {
                cache[s * kv_dim + c] = (s * kv_dim + c) as f32;
            }
        }

        // Fused path
        let fused = CpuBackend::gather_reshape_repeat_kv(&cache, max_seq, cached_len, n_kv, n_q, hd);

        // 3-step reference path
        let gathered = CpuBackend::gather_rows(&cache, max_seq, kv_dim, 0, cached_len);
        let reshaped = CpuBackend::reshape_for_heads(&gathered, 1, cached_len, n_kv, hd);
        let reference = CpuBackend::repeat_kv(&reshaped, n_kv, n_q, cached_len, hd);

        assert_eq!(fused.len(), reference.len());
        for (i, (&f, &r)) in fused.iter().zip(reference.iter()).enumerate() {
            assert!(
                (f - r).abs() < 1e-6,
                "mismatch at index {i}: fused={f}, reference={r}"
            );
        }
    }
}
