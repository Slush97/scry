pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod kernels;

use crate::tensor::shape::Shape;

/// Memory management trait — allocate, copy, transfer between host/device.
pub trait DeviceBackend: Sized {
    type Storage: Clone;
    type Stream;

    fn zeros(shape: &Shape) -> Self::Storage;
    fn ones(shape: &Shape) -> Self::Storage;
    fn from_vec(data: Vec<f32>, shape: &Shape) -> Self::Storage;
    fn to_vec(storage: &Self::Storage) -> Vec<f32>;
    fn clone_storage(storage: &Self::Storage) -> Self::Storage;
}

/// Math operations trait — all operations needed for forward passes.
pub trait MathBackend: DeviceBackend {
    // ---- Core forward ops ----

    /// General matrix multiply: `C = alpha * op(A) * op(B) + beta * C`
    fn matmul(
        a: &Self::Storage,
        b: &Self::Storage,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> Self::Storage;

    /// Elementwise add with broadcasting.
    fn add(
        a: &Self::Storage,
        b: &Self::Storage,
        a_shape: &Shape,
        b_shape: &Shape,
        out_shape: &Shape,
    ) -> Self::Storage;

    /// Softmax along the last axis.
    fn softmax(input: &Self::Storage, shape: &Shape) -> Self::Storage;

    /// Layer normalization along the last axis.
    /// Returns `(output, mean, rstd)`.
    fn layernorm(
        input: &Self::Storage,
        gamma: &Self::Storage,
        beta: &Self::Storage,
        shape: &Shape,
        eps: f32,
    ) -> (Self::Storage, Self::Storage, Self::Storage);

    /// GELU activation (tanh approximation).
    fn gelu(input: &Self::Storage) -> Self::Storage;

    /// Embedding lookup: gather rows by indices.
    fn embedding(
        weight: &Self::Storage,
        indices: &[usize],
        vocab: usize,
        dim: usize,
    ) -> Self::Storage;

    /// Sum all elements to scalar.
    fn sum(input: &Self::Storage) -> f32;

    /// Elementwise multiply: `a * b` (same shape, no broadcast).
    fn mul_elementwise(a: &Self::Storage, b: &Self::Storage) -> Self::Storage;

    /// Scale all elements: `a * scalar`.
    fn scale(a: &Self::Storage, scalar: f32) -> Self::Storage;

    /// Concatenate two row-major matrices along rows (axis 0).
    fn concat_rows(
        a: &Self::Storage,
        b: &Self::Storage,
        a_rows: usize,
        b_rows: usize,
        cols: usize,
    ) -> Self::Storage;

    // ---- Llama-specific ops ----

    /// RMS normalization: `out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]`
    fn rmsnorm(
        input: &Self::Storage,
        weight: &Self::Storage,
        shape: &Shape,
        eps: f32,
    ) -> Self::Storage;

    /// Rotary position embeddings.
    fn rope(
        input: &Self::Storage,
        shape: &Shape,
        pos: usize,
        head_dim: usize,
        theta: f32,
    ) -> Self::Storage;

    /// Rotary position embeddings with pre-computed frequencies.
    ///
    /// Applies RoPE to `[seq, n_heads * head_dim]` using pre-computed `freqs`
    /// (one per dimension pair, len = head_dim/2). This avoids GPU→CPU→GPU
    /// roundtrips by keeping the operation entirely on-device.
    fn rope_with_freqs(
        input: &Self::Storage,
        seq: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
        freqs: &[f64],
    ) -> Self::Storage;

    /// `SwiGLU`: `silu(gate) * up` where `silu(x) = x / (1 + exp(-x))`
    fn swiglu(gate: &Self::Storage, up: &Self::Storage) -> Self::Storage;

    /// Repeat KV heads for GQA: expand `[n_kv_heads, seq, d_head]` to `[n_q_heads, seq, d_head]`.
    fn repeat_kv(
        input: &Self::Storage,
        n_kv_heads: usize,
        n_q_heads: usize,
        seq: usize,
        d_head: usize,
    ) -> Self::Storage;

    // ---- Attention helpers ----

    /// Extract columns `[col_start..col_start+col_count)` from a `[rows, total_cols]` matrix.
    fn gather_columns(
        storage: &Self::Storage,
        rows: usize,
        total_cols: usize,
        col_start: usize,
        col_count: usize,
    ) -> Self::Storage {
        let data = Self::to_vec(storage);
        let mut out = vec![0.0f32; rows * col_count];
        for r in 0..rows {
            for c in 0..col_count {
                out[r * col_count + c] = data[r * total_cols + col_start + c];
            }
        }
        Self::from_vec(out, &Shape::new(&[rows, col_count]))
    }

    /// Scatter (additive) a `[rows, col_count]` source into columns of a destination.
    fn scatter_columns(
        dst: &mut Self::Storage,
        src: &Self::Storage,
        rows: usize,
        total_cols: usize,
        col_start: usize,
        col_count: usize,
    ) {
        let mut dst_vec = Self::to_vec(dst);
        let src_vec = Self::to_vec(src);
        for r in 0..rows {
            for c in 0..col_count {
                dst_vec[r * total_cols + col_start + c] += src_vec[r * col_count + c];
            }
        }
        *dst = Self::from_vec(dst_vec, &Shape::new(&[rows, total_cols]));
    }

    /// Extract rows `[row_start..row_start+row_count)` from a `[total_rows, cols]` matrix.
    fn gather_rows(
        storage: &Self::Storage,
        total_rows: usize,
        cols: usize,
        row_start: usize,
        row_count: usize,
    ) -> Self::Storage {
        let _ = total_rows;
        let data = Self::to_vec(storage);
        let start = row_start * cols;
        let end = start + row_count * cols;
        Self::from_vec(data[start..end].to_vec(), &Shape::new(&[row_count, cols]))
    }

    /// Write rows into a destination matrix (overwrite).
    fn scatter_rows(
        dst: &mut Self::Storage,
        src: &Self::Storage,
        total_rows: usize,
        cols: usize,
        row_start: usize,
        row_count: usize,
    ) {
        let mut dst_vec = Self::to_vec(dst);
        let src_vec = Self::to_vec(src);
        let start = row_start * cols;
        dst_vec[start..start + row_count * cols].copy_from_slice(&src_vec[..row_count * cols]);
        *dst = Self::from_vec(dst_vec, &Shape::new(&[total_rows, cols]));
    }

    /// Apply causal mask and scale to a `[seq_len, seq_len]` matrix in-place.
    fn apply_causal_mask_and_scale(
        scores: &mut Self::Storage,
        seq_len: usize,
        scale: f32,
        mask_value: f32,
    ) {
        let mut data = Self::to_vec(scores);
        for s in 0..seq_len {
            for t in 0..seq_len {
                if t > s {
                    data[s * seq_len + t] = mask_value;
                } else {
                    data[s * seq_len + t] *= scale;
                }
            }
        }
        *scores = Self::from_vec(data, &Shape::new(&[seq_len, seq_len]));
    }

    /// Reshape `[B*S, H*d_head]` → `[B*H, S, d_head]` for batched multi-head attention.
    fn reshape_for_heads(
        storage: &Self::Storage,
        batch: usize,
        seq: usize,
        n_heads: usize,
        d_head: usize,
    ) -> Self::Storage {
        let data = Self::to_vec(storage);
        let d_model = n_heads * d_head;
        let total = batch * n_heads * seq * d_head;
        let mut out = vec![0.0f32; total];
        for b in 0..batch {
            for h in 0..n_heads {
                for s in 0..seq {
                    for d in 0..d_head {
                        out[(b * n_heads + h) * seq * d_head + s * d_head + d] =
                            data[(b * seq + s) * d_model + h * d_head + d];
                    }
                }
            }
        }
        Self::from_vec(out, &Shape::new(&[batch * n_heads, seq, d_head]))
    }

    /// Reshape `[B*H, S, d_head]` → `[B*S, H*d_head]` (reverse of `reshape_for_heads`).
    fn reshape_from_heads(
        storage: &Self::Storage,
        batch: usize,
        seq: usize,
        n_heads: usize,
        d_head: usize,
    ) -> Self::Storage {
        let data = Self::to_vec(storage);
        let d_model = n_heads * d_head;
        let total = batch * seq * d_model;
        let mut out = vec![0.0f32; total];
        for b in 0..batch {
            for h in 0..n_heads {
                for s in 0..seq {
                    for d in 0..d_head {
                        out[(b * seq + s) * d_model + h * d_head + d] =
                            data[(b * n_heads + h) * seq * d_head + s * d_head + d];
                    }
                }
            }
        }
        Self::from_vec(out, &Shape::new(&[batch * seq, d_model]))
    }

    /// Strided batched matrix multiply.
    #[allow(clippy::too_many_arguments)]
    fn matmul_strided_batched(
        a: &Self::Storage,
        b: &Self::Storage,
        batch_count: usize,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> Self::Storage {
        let a_stride = m * k;
        let b_stride = k * n;
        let c_stride = m * n;
        let total = batch_count * c_stride;
        let a_vec = Self::to_vec(a);
        let b_vec = Self::to_vec(b);
        let mut c_vec = vec![0.0f32; total];
        for i in 0..batch_count {
            let a_slice = Self::from_vec(
                a_vec[i * a_stride..(i + 1) * a_stride].to_vec(),
                &Shape::new(&[m, k]),
            );
            let b_slice = Self::from_vec(
                b_vec[i * b_stride..(i + 1) * b_stride].to_vec(),
                &Shape::new(&[k, n]),
            );
            let c_slice = Self::matmul(&a_slice, &b_slice, m, k, n, trans_a, trans_b);
            let c_data = Self::to_vec(&c_slice);
            c_vec[i * c_stride..(i + 1) * c_stride].copy_from_slice(&c_data);
        }
        Self::from_vec(c_vec, &Shape::new(&[batch_count * m, n]))
    }

    /// Apply causal mask and scale to batched matrices.
    fn apply_batched_causal_mask_and_scale(
        scores: &mut Self::Storage,
        num_matrices: usize,
        seq_len: usize,
        scale: f32,
        mask_value: f32,
    ) {
        let _ = num_matrices;
        let mut data = Self::to_vec(scores);
        let mat_size = seq_len * seq_len;
        let total = data.len() / mat_size;
        for m in 0..total {
            let off = m * mat_size;
            for s in 0..seq_len {
                for t in 0..seq_len {
                    let idx = off + s * seq_len + t;
                    if t > s {
                        data[idx] = mask_value;
                    } else {
                        data[idx] *= scale;
                    }
                }
            }
        }
        let n = data.len();
        *scores = Self::from_vec(data, &Shape::new(&[n]));
    }
}
