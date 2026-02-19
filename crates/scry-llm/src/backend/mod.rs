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

/// Math operations trait — all operations needed for forward and backward passes.
pub trait MathBackend: DeviceBackend {
    // ---- Forward ops ----

    /// General matrix multiply: `C = alpha * op(A) * op(B) + beta * C`
    /// `A` is `[M, K]` (or `[K, M]` if `trans_a`), `B` is `[K, N]` (or `[N, K]` if `trans_b`)
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
    /// Returns `(output, mean, rstd)` for backward pass.
    fn layernorm(
        input: &Self::Storage,
        gamma: &Self::Storage,
        beta: &Self::Storage,
        shape: &Shape,
        eps: f32,
    ) -> (Self::Storage, Self::Storage, Self::Storage);

    /// GELU activation (tanh approximation).
    fn gelu(input: &Self::Storage) -> Self::Storage;

    /// Cross-entropy loss from logits.
    /// `logits`: `[B, V]`, `targets`: `[B]` (class indices as `f32`).
    /// Returns scalar loss.
    fn cross_entropy(logits: &Self::Storage, targets: &[usize], batch: usize, vocab: usize) -> f32;

    /// Embedding lookup: gather rows by indices.
    /// `weight`: `[V, D]`, `indices`: `[N]` → output: `[N, D]`
    fn embedding(
        weight: &Self::Storage,
        indices: &[usize],
        vocab: usize,
        dim: usize,
    ) -> Self::Storage;

    /// Sum all elements to scalar.
    fn sum(input: &Self::Storage) -> f32;

    // ---- Backward ops ----

    /// Backward for matmul. Returns `(d_a, d_b)`.
    fn matmul_backward(
        d_out: &Self::Storage,
        a: &Self::Storage,
        b: &Self::Storage,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> (Self::Storage, Self::Storage);

    /// Backward for add with broadcasting. Returns `(d_a, d_b)`.
    fn add_backward(
        d_out: &Self::Storage,
        a_shape: &Shape,
        b_shape: &Shape,
        out_shape: &Shape,
    ) -> (Self::Storage, Self::Storage);

    /// Backward for softmax. Returns `d_input`.
    fn softmax_backward(
        d_out: &Self::Storage,
        output: &Self::Storage,
        shape: &Shape,
    ) -> Self::Storage;

    /// Backward for layernorm. Returns `(d_input, d_gamma, d_beta)`.
    fn layernorm_backward(
        d_out: &Self::Storage,
        input: &Self::Storage,
        gamma: &Self::Storage,
        mean: &Self::Storage,
        rstd: &Self::Storage,
        shape: &Shape,
    ) -> (Self::Storage, Self::Storage, Self::Storage);

    /// Backward for GELU. Returns `d_input`.
    fn gelu_backward(d_out: &Self::Storage, input: &Self::Storage) -> Self::Storage;

    /// Backward for `cross_entropy`. Returns `d_logits` `[B, V]`.
    fn cross_entropy_backward(
        logits: &Self::Storage,
        targets: &[usize],
        batch: usize,
        vocab: usize,
    ) -> Self::Storage;

    /// Backward for embedding. Returns `d_weight` `[V, D]`.
    fn embedding_backward(
        d_out: &Self::Storage,
        indices: &[usize],
        vocab: usize,
        dim: usize,
    ) -> Self::Storage;

    /// Elementwise multiply: `a * b` (same shape, no broadcast).
    fn mul_elementwise(a: &Self::Storage, b: &Self::Storage) -> Self::Storage;

    /// Scale all elements: `a * scalar`.
    fn scale(a: &Self::Storage, scalar: f32) -> Self::Storage;

    /// Elementwise add in place: `a += b`. Shapes must match.
    fn add_inplace(a: &mut Self::Storage, b: &Self::Storage);

    /// L2 norm of all elements: `sqrt(sum(x^2))`.
    fn norm(storage: &Self::Storage) -> f32;

    /// In-place scalar multiply: `a[i] *= scalar` for all i.
    fn scale_inplace(a: &mut Self::Storage, scalar: f32);

    /// Concatenate two row-major matrices along rows (axis 0).
    /// `a`: `[a_rows, cols]`, `b`: `[b_rows, cols]` → `[a_rows + b_rows, cols]`.
    fn concat_rows(
        a: &Self::Storage,
        b: &Self::Storage,
        a_rows: usize,
        b_rows: usize,
        cols: usize,
    ) -> Self::Storage;

    /// Invalidate any cached bf16 shadow for this storage.
    /// Called after optimizer updates so the shadow is re-created on next forward pass.
    /// Default is a no-op (CPU backend has no bf16 shadow).
    fn invalidate_bf16_cache(_storage: &Self::Storage) {}

    /// `AdamW` optimizer step (fused). Updates `param` in place.
    #[allow(clippy::too_many_arguments)]
    fn adamw_step(
        param: &mut Self::Storage,
        grad: &Self::Storage,
        m: &mut Self::Storage,
        v: &mut Self::Storage,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: u32,
    );

    // ---- Attention helpers (GPU-optimized, CPU fallback via to_vec/from_vec) ----

    /// Extract columns `[col_start..col_start+col_count)` from a `[rows, total_cols]` matrix.
    /// Returns a `[rows, col_count]` storage.
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

    /// Scatter (additive) a `[rows, col_count]` source into columns
    /// `[col_start..col_start+col_count)` of a `[rows, total_cols]` destination.
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
    /// Returns a `[row_count, cols]` storage.
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

    /// Write a `[row_count, cols]` source into rows
    /// `[row_start..row_start+row_count)` of a `[total_rows, cols]` destination (overwrite).
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

    /// GPU-accelerated dropout: generates mask and applies in a single kernel.
    /// Returns `(output, mask)` where mask contains the inverted-dropout scale values.
    /// Default implementation falls back to CPU.
    fn dropout(
        input: &Self::Storage,
        n: usize,
        p: f32,
        seed: u64,
    ) -> (Self::Storage, Self::Storage) {
        let data = Self::to_vec(input);
        let scale = 1.0 / (1.0 - p);
        let mut mask = vec![0.0f32; n];
        let mut output = vec![0.0f32; n];
        // Replicate the GPU's philox RNG on CPU for consistency
        // (but in practice this path is only used by CpuBackend)
        let mut rng = fastrand::Rng::with_seed(seed);
        for i in 0..n {
            if rng.f32() >= p {
                mask[i] = scale;
                output[i] = data[i] * scale;
            }
        }
        let shape = crate::tensor::shape::Shape::new(&[n]);
        (Self::from_vec(output, &shape), Self::from_vec(mask, &shape))
    }

    /// Apply causal mask and scale to a `[seq_len, seq_len]` matrix in-place.
    /// Upper triangle is set to `mask_value` (e.g. `-inf` for forward, `0.0` for backward).
    /// Lower triangle (including diagonal) is multiplied by `scale`.
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
    /// Default CPU implementation transposes the data.
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

    /// Strided batched matrix multiply: `batch_count` independent GEMMs on contiguous slices.
    /// `a`: `[batch_count, M, K]`, `b`: `[batch_count, K, N]` → `c`: `[batch_count, M, N]`
    /// (with optional transposes). Default falls back to a loop of `matmul` calls.
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
        // CPU fallback: loop over batches
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

    /// Apply causal mask and scale to `num_matrices` copies of `[seq_len, seq_len]`
    /// stored contiguously. Default calls `apply_causal_mask_and_scale` in a loop.
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
