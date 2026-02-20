use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

/// All custom CUDA kernels compiled once at init time.
pub struct KernelCache {
    pub gelu_fwd: CudaFunction,
    pub gelu_bwd: CudaFunction,
    pub add_broadcast_2d: CudaFunction,
    pub softmax_fwd: CudaFunction,
    pub softmax_bwd: CudaFunction,
    pub layernorm_fwd: CudaFunction,
    pub layernorm_bwd: CudaFunction,
    pub cross_entropy_fwd: CudaFunction,
    pub cross_entropy_bwd: CudaFunction,
    pub embedding_fwd: CudaFunction,
    pub embedding_bwd: CudaFunction,
    pub adamw_step: CudaFunction,
    pub scale_kernel: CudaFunction,
    pub mul_elementwise: CudaFunction,
    pub add_inplace_kernel: CudaFunction,
    pub gather_columns: CudaFunction,
    pub scatter_columns: CudaFunction,
    pub causal_mask_scale: CudaFunction,
    pub reduce_sum: CudaFunction,
    pub dot_self: CudaFunction,
    pub scale_inplace_kernel: CudaFunction,
    pub reduce_rows: CudaFunction,
    pub dropout_fwd: CudaFunction,
    pub reshape_bsh_to_bnsh: CudaFunction,
    pub reshape_bnsh_to_bsh: CudaFunction,
    pub batched_causal_mask_scale: CudaFunction,
    pub residual_add_layernorm_fwd: CudaFunction,
    pub residual_add_layernorm_bwd: CudaFunction,
    pub split_qkv_to_heads: CudaFunction,
    pub merge_heads_to_qkv: CudaFunction,
    pub fused_bias_gelu_fwd: CudaFunction,
    pub fused_bias_gelu_bwd: CudaFunction,
    pub fused_bias_dropout_residual_fwd: CudaFunction,
    pub flash_attention_fwd: CudaFunction,
    pub flash_attention_bwd: CudaFunction,
    pub multi_dot_self: CudaFunction,
    pub fused_mul_reduce_rows: CudaFunction,
    pub broadcast_scalar: CudaFunction,
    pub cross_entropy_fwd_bwd: CudaFunction,

    // BF16 kernels (cast + flash attention — element-wise ops use f32 path)
    #[cfg(feature = "bf16")]
    pub f32_to_bf16: CudaFunction,
    #[cfg(feature = "bf16")]
    pub bf16_to_f32: CudaFunction,
    #[cfg(feature = "bf16")]
    pub flash_attention_fwd_bf16: CudaFunction,
    #[cfg(feature = "bf16")]
    pub flash_attention_bwd_bf16: CudaFunction,
}

impl KernelCache {
    pub fn compile(ctx: &Arc<cudarc::driver::CudaContext>) -> Self {
        let opts = CompileOptions {
            use_fast_math: Some(true),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(KERNEL_SOURCE, opts)
            .expect("failed to compile CUDA kernels");
        let module: Arc<CudaModule> = ctx.load_module(ptx)
            .expect("failed to load CUDA module");

        #[cfg(feature = "bf16")]
        let bf16_module = {
            let bf16_opts = CompileOptions {
                use_fast_math: Some(true),
                ..Default::default()
            };
            let bf16_ptx = compile_ptx_with_opts(BF16_KERNEL_SOURCE, bf16_opts)
                .expect("failed to compile BF16 CUDA kernels");
            ctx.load_module(bf16_ptx)
                .expect("failed to load BF16 CUDA module")
        };

        Self {
            gelu_fwd: module.load_function("gelu_fwd").unwrap(),
            gelu_bwd: module.load_function("gelu_bwd").unwrap(),
            add_broadcast_2d: module.load_function("add_broadcast_2d").unwrap(),
            softmax_fwd: module.load_function("softmax_fwd").unwrap(),
            softmax_bwd: module.load_function("softmax_bwd").unwrap(),
            layernorm_fwd: module.load_function("layernorm_fwd").unwrap(),
            layernorm_bwd: module.load_function("layernorm_bwd").unwrap(),
            cross_entropy_fwd: module.load_function("cross_entropy_fwd").unwrap(),
            cross_entropy_bwd: module.load_function("cross_entropy_bwd").unwrap(),
            embedding_fwd: module.load_function("embedding_fwd").unwrap(),
            embedding_bwd: module.load_function("embedding_bwd").unwrap(),
            adamw_step: module.load_function("adamw_step").unwrap(),
            scale_kernel: module.load_function("scale_kernel").unwrap(),
            mul_elementwise: module.load_function("mul_elementwise").unwrap(),
            add_inplace_kernel: module.load_function("add_inplace_kernel").unwrap(),
            gather_columns: module.load_function("gather_columns").unwrap(),
            scatter_columns: module.load_function("scatter_columns").unwrap(),
            causal_mask_scale: module.load_function("causal_mask_scale").unwrap(),
            reduce_sum: module.load_function("reduce_sum").unwrap(),
            dot_self: module.load_function("dot_self").unwrap(),
            scale_inplace_kernel: module.load_function("scale_inplace_kernel").unwrap(),
            reduce_rows: module.load_function("reduce_rows").unwrap(),
            dropout_fwd: module.load_function("dropout_fwd").unwrap(),
            reshape_bsh_to_bnsh: module.load_function("reshape_bsh_to_bnsh").unwrap(),
            reshape_bnsh_to_bsh: module.load_function("reshape_bnsh_to_bsh").unwrap(),
            batched_causal_mask_scale: module.load_function("batched_causal_mask_scale").unwrap(),
            residual_add_layernorm_fwd: module.load_function("residual_add_layernorm_fwd").unwrap(),
            residual_add_layernorm_bwd: module.load_function("residual_add_layernorm_bwd").unwrap(),
            split_qkv_to_heads: module.load_function("split_qkv_to_heads").unwrap(),
            merge_heads_to_qkv: module.load_function("merge_heads_to_qkv").unwrap(),
            fused_bias_gelu_fwd: module.load_function("fused_bias_gelu_fwd").unwrap(),
            fused_bias_gelu_bwd: module.load_function("fused_bias_gelu_bwd").unwrap(),
            fused_bias_dropout_residual_fwd: module.load_function("fused_bias_dropout_residual_fwd").unwrap(),
            flash_attention_fwd: module.load_function("flash_attention_fwd").unwrap(),
            flash_attention_bwd: module.load_function("flash_attention_bwd").unwrap(),
            multi_dot_self: module.load_function("multi_dot_self").unwrap(),
            fused_mul_reduce_rows: module.load_function("fused_mul_reduce_rows").unwrap(),
            broadcast_scalar: module.load_function("broadcast_scalar").unwrap(),
            cross_entropy_fwd_bwd: module.load_function("cross_entropy_fwd_bwd").unwrap(),

            #[cfg(feature = "bf16")]
            f32_to_bf16: bf16_module.load_function("f32_to_bf16").unwrap(),
            #[cfg(feature = "bf16")]
            bf16_to_f32: bf16_module.load_function("bf16_to_f32").unwrap(),
            #[cfg(feature = "bf16")]
            flash_attention_fwd_bf16: bf16_module.load_function("flash_attention_fwd_bf16").unwrap(),
            #[cfg(feature = "bf16")]
            flash_attention_bwd_bf16: bf16_module.load_function("flash_attention_bwd_bf16").unwrap(),
        }
    }
}

const KERNEL_SOURCE: &str = r#"
// ============================================================
// Simple element-wise kernels
// ============================================================

extern "C" __global__ void gelu_fwd(float* out, const float* x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
    out[i] = 0.5f * v * (1.0f + tanhf(inner));
}

extern "C" __global__ void gelu_bwd(float* dx, const float* dy, const float* x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float kappa = 0.044715f;
    float s2pi = 0.7978845608f;
    float inner = s2pi * (v + kappa * v * v * v);
    float tanh_val = tanhf(inner);
    float sech2 = 1.0f - tanh_val * tanh_val;
    float d_inner = s2pi * (1.0f + 3.0f * kappa * v * v);
    dx[i] = dy[i] * (0.5f * (1.0f + tanh_val) + 0.5f * v * sech2 * d_inner);
}

extern "C" __global__ void scale_kernel(float* out, const float* a, float scalar, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] * scalar;
}

extern "C" __global__ void mul_elementwise(float* out, const float* a, const float* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] * b[i];
}

extern "C" __global__ void add_inplace_kernel(float* a, const float* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] += b[i];
}

// ============================================================
// Broadcast add: a[rows, cols] + b[1, cols] = out[rows, cols]
// ============================================================

extern "C" __global__ void add_broadcast_2d(
    float* out, const float* a, const float* b,
    size_t rows, size_t cols
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    size_t col = idx % cols;
    out[idx] = a[idx] + b[col];
}

// ============================================================
// Softmax: row-wise with shared memory reduction
// One block per row. blockDim.x threads cooperate on one row.
// ============================================================

extern "C" __global__ void softmax_fwd(float* out, const float* inp, size_t rows, size_t cols) {
    extern __shared__ float sdata[];

    size_t row = blockIdx.x;
    if (row >= rows) return;
    const float* row_in = inp + row * cols;
    float* row_out = out + row * cols;
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    // Find max
    float local_max = -1e30f;
    for (size_t j = tid; j < cols; j += stride) {
        float v = row_in[j];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    // Compute exp and sum
    float local_sum = 0.0f;
    for (size_t j = tid; j < cols; j += stride) {
        float e = expf(row_in[j] - max_val);
        row_out[j] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sum = sdata[0];
    __syncthreads();

    // Normalize
    for (size_t j = tid; j < cols; j += stride) {
        row_out[j] /= sum;
    }
}

extern "C" __global__ void softmax_bwd(
    float* dx, const float* dy, const float* y,
    size_t rows, size_t cols
) {
    extern __shared__ float sdata[];

    size_t row = blockIdx.x;
    if (row >= rows) return;
    size_t off = row * cols;
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    // dot = sum(dy * y)
    float local_dot = 0.0f;
    for (size_t j = tid; j < cols; j += stride) {
        local_dot += dy[off + j] * y[off + j];
    }
    sdata[tid] = local_dot;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float dot = sdata[0];
    __syncthreads();

    // dx = y * (dy - dot)
    for (size_t j = tid; j < cols; j += stride) {
        dx[off + j] = y[off + j] * (dy[off + j] - dot);
    }
}

// ============================================================
// LayerNorm: one block per row
// ============================================================

extern "C" __global__ void layernorm_fwd(
    float* out, float* mean_out, float* rstd_out,
    const float* inp, const float* gamma, const float* beta,
    size_t rows, size_t d, float eps
) {
    extern __shared__ float sdata[];

    size_t row = blockIdx.x;
    if (row >= rows) return;
    size_t off = row * d;
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    // Mean
    float local_sum = 0.0f;
    for (size_t j = tid; j < d; j += stride) {
        local_sum += inp[off + j];
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / (float)d;
    __syncthreads();

    // Variance
    float local_var = 0.0f;
    for (size_t j = tid; j < d; j += stride) {
        float diff = inp[off + j] - mean;
        local_var += diff * diff;
    }
    sdata[tid] = local_var;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float var = sdata[0] / (float)d;
    float rstd = rsqrtf(var + eps);
    __syncthreads();

    if (tid == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }

    // Normalize + affine
    for (size_t j = tid; j < d; j += stride) {
        float norm = (inp[off + j] - mean) * rstd;
        out[off + j] = norm * gamma[j] + beta[j];
    }
}

extern "C" __global__ void layernorm_bwd(
    float* dx, float* dgamma, float* dbeta,
    const float* dy, const float* x, const float* gamma,
    const float* mean, const float* rstd,
    size_t rows, size_t d
) {
    // One block per row for dx; dgamma/dbeta accumulated via atomicAdd
    extern __shared__ float sdata[];
    // sdata layout: [0..blockDim.x) = sum_dnorm, [blockDim.x..2*blockDim.x) = sum_dnorm_norm

    size_t row = blockIdx.x;
    if (row >= rows) return;
    size_t off = row * d;
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;
    float m = mean[row];
    float rs = rstd[row];

    float local_sum_dnorm = 0.0f;
    float local_sum_dnorm_norm = 0.0f;

    for (size_t j = tid; j < d; j += stride) {
        float dnorm = dy[off + j] * gamma[j];
        float norm = (x[off + j] - m) * rs;
        local_sum_dnorm += dnorm;
        local_sum_dnorm_norm += dnorm * norm;
        // dgamma, dbeta via atomicAdd across rows
        atomicAdd(&dgamma[j], dy[off + j] * norm);
        atomicAdd(&dbeta[j], dy[off + j]);
    }

    sdata[tid] = local_sum_dnorm;
    sdata[blockDim.x + tid] = local_sum_dnorm_norm;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[blockDim.x + tid] += sdata[blockDim.x + tid + s];
        }
        __syncthreads();
    }
    float sum_dnorm = sdata[0];
    float sum_dnorm_norm = sdata[blockDim.x];
    __syncthreads();

    float inv_d = 1.0f / (float)d;
    for (size_t j = tid; j < d; j += stride) {
        float dnorm = dy[off + j] * gamma[j];
        float norm = (x[off + j] - m) * rs;
        dx[off + j] = inv_d * rs * ((float)d * dnorm - sum_dnorm - norm * sum_dnorm_norm);
    }
}

// ============================================================
// Cross-entropy: one block per batch item
// ============================================================

extern "C" __global__ void cross_entropy_fwd(
    float* loss_out, const float* logits, const unsigned int* targets,
    size_t batch, size_t vocab
) {
    extern __shared__ float sdata[];

    size_t b = blockIdx.x;
    if (b >= batch) return;
    size_t off = b * vocab;
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;
    unsigned int target = targets[b];

    // Max
    float local_max = -1e30f;
    for (size_t j = tid; j < vocab; j += stride) {
        float v = logits[off + j];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    // Sum exp
    float local_sum = 0.0f;
    for (size_t j = tid; j < vocab; j += stride) {
        local_sum += expf(logits[off + j] - max_val);
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sum_exp = sdata[0];

    if (tid == 0) {
        float log_sum_exp = max_val + logf(sum_exp);
        float log_prob = logits[off + target] - log_sum_exp;
        atomicAdd(loss_out, -log_prob / (float)batch);
    }
}

extern "C" __global__ void cross_entropy_bwd(
    float* d_logits, const float* logits, const unsigned int* targets,
    size_t batch, size_t vocab, const float* d_out_scalar
) {
    extern __shared__ float sdata[];

    size_t b = blockIdx.x;
    if (b >= batch) return;
    size_t off = b * vocab;
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;
    unsigned int target = targets[b];

    // Read upstream scalar gradient from device memory (no D2H sync)
    float d_out_s = d_out_scalar[0];

    // Max
    float local_max = -1e30f;
    for (size_t j = tid; j < vocab; j += stride) {
        float v = logits[off + j];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    // Sum exp
    float local_sum = 0.0f;
    for (size_t j = tid; j < vocab; j += stride) {
        local_sum += expf(logits[off + j] - max_val);
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sum_exp = sdata[0];
    __syncthreads();

    float inv_batch = 1.0f / (float)batch;
    float combined_scale = inv_batch * d_out_s;
    for (size_t j = tid; j < vocab; j += stride) {
        float prob = expf(logits[off + j] - max_val) / sum_exp;
        float target_val = (j == target) ? 1.0f : 0.0f;
        d_logits[off + j] = (prob - target_val) * combined_scale;
    }
}

// Broadcast a 1-element scalar to fill an N-element output.
// Reads scalar from device memory — no D2H sync needed.
extern "C" __global__ void broadcast_scalar(float* out, const float* scalar, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = scalar[0];
}

// ============================================================
// Fused cross-entropy forward+backward: one softmax pass total.
// Computes loss and caches d_logits = (softmax - one_hot) / batch.
// Backward just scales the cached grad by upstream d_out.
// One block per batch row. Three passes over vocab:
//   1. Find max (shared mem reduction)
//   2. Compute sum_exp (shared mem reduction)
//   3. Write grad + accumulate loss via atomicAdd
// ============================================================

extern "C" __global__ void cross_entropy_fwd_bwd(
    float* loss_out, float* d_logits,
    const float* logits, const unsigned int* targets,
    size_t batch, size_t vocab
) {
    extern __shared__ float sdata[];

    size_t b = blockIdx.x;
    if (b >= batch) return;
    size_t off = b * vocab;
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;
    unsigned int target = targets[b];

    // Pass 1: find max
    float local_max = -1e30f;
    for (size_t j = tid; j < vocab; j += stride) {
        float v = logits[off + j];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    // Pass 2: sum exp
    float local_sum = 0.0f;
    for (size_t j = tid; j < vocab; j += stride) {
        local_sum += expf(logits[off + j] - max_val);
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sum_exp = sdata[0];
    __syncthreads();

    // Pass 3: write grad and accumulate loss
    float inv_batch = 1.0f / (float)batch;
    float local_loss = 0.0f;
    for (size_t j = tid; j < vocab; j += stride) {
        float prob = expf(logits[off + j] - max_val) / sum_exp;
        float target_val = (j == target) ? 1.0f : 0.0f;
        d_logits[off + j] = (prob - target_val) * inv_batch;
        if (j == target) {
            local_loss = -logf(prob) * inv_batch;
        }
    }

    // Reduce local_loss across threads (only one thread has nonzero loss per row)
    sdata[tid] = local_loss;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(loss_out, sdata[0]);
    }
}

// ============================================================
// Embedding
// ============================================================

extern "C" __global__ void embedding_fwd(
    float* out, const float* weight, const unsigned int* indices,
    size_t n_indices, size_t dim
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_indices * dim) return;
    size_t token = idx / dim;
    size_t d = idx % dim;
    out[token * dim + d] = weight[indices[token] * dim + d];
}

extern "C" __global__ void embedding_bwd(
    float* d_weight, const float* d_out, const unsigned int* indices,
    size_t n_indices, size_t dim
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_indices * dim) return;
    size_t token = idx / dim;
    size_t d = idx % dim;
    atomicAdd(&d_weight[indices[token] * dim + d], d_out[token * dim + d]);
}

// ============================================================
// AdamW: one thread per parameter
// ============================================================

extern "C" __global__ void adamw_step(
    float* param, const float* grad, float* m, float* v,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bc1, float bc2, size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grad[i];
    float mi = beta1 * m[i] + (1.0f - beta1) * g;
    float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;
    float m_hat = mi / bc1;
    float v_hat = vi / bc2;
    float update = m_hat / (sqrtf(v_hat) + eps);
    param[i] -= lr * (update + weight_decay * param[i]);
}

// ============================================================
// Attention helpers: gather/scatter columns, causal mask+scale
// ============================================================

// Gather columns: extract column range [col_start, col_start+col_count) from
// a [rows, total_cols] matrix into a [rows, col_count] output.
extern "C" __global__ void gather_columns(
    float* out, const float* src,
    size_t rows, size_t total_cols, size_t col_start, size_t col_count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * col_count) return;
    size_t r = idx / col_count;
    size_t c = idx % col_count;
    out[idx] = src[r * total_cols + col_start + c];
}

// Scatter columns: write a [rows, col_count] matrix into column range
// [col_start, col_start+col_count) of a [rows, total_cols] matrix (additive).
extern "C" __global__ void scatter_columns(
    float* dst, const float* src,
    size_t rows, size_t total_cols, size_t col_start, size_t col_count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * col_count) return;
    size_t r = idx / col_count;
    size_t c = idx % col_count;
    dst[r * total_cols + col_start + c] += src[idx];
}

// Apply causal mask and scale to a [seq, seq] matrix.
// Upper triangle set to mask_value (-inf for forward, 0 for backward).
// Lower triangle (incl diagonal) multiplied by scale.
extern "C" __global__ void causal_mask_scale(
    float* scores, size_t seq_len, float scale, float mask_value
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * seq_len) return;
    size_t s = idx / seq_len;
    size_t t = idx % seq_len;
    if (t > s) {
        scores[idx] = mask_value;
    } else {
        scores[idx] *= scale;
    }
}

// ============================================================
// Reductions: sum, dot_self (for L2 norm), reduce_rows
// ============================================================

// Tree reduction: sum all elements into out[0]. Single block.
// Handles arbitrary N via strided accumulation into shared memory.
extern "C" __global__ void reduce_sum(float* out, const float* x, size_t n) {
    extern __shared__ float sdata[];
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    float local_sum = 0.0f;
    for (size_t i = tid; i < n; i += stride) {
        local_sum += x[i];
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[0] = sdata[0];
}

// Sum of squares: out[0] = sum(x[i]^2). Single block, for L2 norm.
extern "C" __global__ void dot_self(float* out, const float* x, size_t n) {
    extern __shared__ float sdata[];
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    float local_sum = 0.0f;
    for (size_t i = tid; i < n; i += stride) {
        float v = x[i];
        local_sum += v * v;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[0] = sdata[0];
}

// Multi-tensor dot_self with multi-block-per-tensor support.
// Grid: (n_tensors, blocks_per_tensor, 1).
// Each block processes a slice of one tensor and atomicAdds its partial sum.
// out[] MUST be zeroed before launch.
extern "C" __global__ void multi_dot_self(
    float* out,
    const float* const* ptrs,
    const size_t* lens,
    size_t n_tensors
) {
    extern __shared__ float sdata[];
    size_t tensor_idx = blockIdx.x;
    if (tensor_idx >= n_tensors) return;

    const float* x = ptrs[tensor_idx];
    size_t n = lens[tensor_idx];
    size_t tid = threadIdx.x;

    // Each block in the y-dimension handles a strided slice
    size_t blocks_y = gridDim.y;
    size_t block_id = blockIdx.y;
    size_t global_stride = blockDim.x * blocks_y;
    size_t start = tid + block_id * blockDim.x;

    float local_sum = 0.0f;
    for (size_t i = start; i < n; i += global_stride) {
        float v = x[i];
        local_sum += v * v;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(&out[tensor_idx], sdata[0]);
}

// ============================================================
// Dropout with philox-style counter-based RNG (no cuRAND dependency)
// Each thread generates its own random number from seed + thread_index.
// ============================================================

__device__ __forceinline__ unsigned int philox_round(unsigned int ctr, unsigned int key) {
    // Philox 4x32-10 simplified to single counter
    unsigned long long product = (unsigned long long)ctr * 0xD2511F53u;
    return (unsigned int)(product >> 32) ^ (unsigned int)product ^ key;
}

__device__ __forceinline__ float rand_uniform(unsigned long long seed, size_t idx) {
    // Mix seed and index through multiple rounds
    unsigned int ctr = (unsigned int)idx;
    unsigned int key = (unsigned int)(seed >> 32) ^ (unsigned int)seed;
    ctr = philox_round(ctr, key);
    ctr = philox_round(ctr, key + 1u);
    ctr = philox_round(ctr, key + 2u);
    ctr = philox_round(ctr, key + 3u);
    // Convert to float in (0, 1)
    return ((float)(ctr & 0x00FFFFFFu)) / 16777216.0f;
}

// Dropout forward: out[i] = inp[i] * mask[i], where mask[i] is scale if rand >= p, else 0.
extern "C" __global__ void dropout_fwd(
    float* out, float* mask,
    const float* inp,
    float p, float scale,
    unsigned long long seed,
    size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float r = rand_uniform(seed, i);
    float m = (r >= p) ? scale : 0.0f;
    mask[i] = m;
    out[i] = inp[i] * m;
}

// ============================================================
// Reshape kernels for batched multi-head attention
// ============================================================

// Reshape [B*S, H*d_head] -> [B*H, S, d_head]
// src layout: element at (b*S+s, h*d_head+d) = src[(b*S+s)*D + h*d_head+d]  where D=H*d_head
// dst layout: element at (b*H+h, s, d) = dst[(b*H+h)*S*d_head + s*d_head + d]
extern "C" __global__ void reshape_bsh_to_bnsh(
    float* dst, const float* src,
    size_t batch, size_t seq, size_t n_heads, size_t d_head
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * n_heads * seq * d_head;
    if (idx >= total) return;
    // Decompose dst index: idx = (b*H+h)*S*d_head + s*d_head + d
    size_t d = idx % d_head;
    size_t tmp = idx / d_head;
    size_t s = tmp % seq;
    size_t bh = tmp / seq;
    size_t h = bh % n_heads;
    size_t b = bh / n_heads;
    size_t D = n_heads * d_head;
    dst[idx] = src[(b * seq + s) * D + h * d_head + d];
}

// Reshape [B*H, S, d_head] -> [B*S, H*d_head]
// Reverse of above.
extern "C" __global__ void reshape_bnsh_to_bsh(
    float* dst, const float* src,
    size_t batch, size_t seq, size_t n_heads, size_t d_head
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * n_heads * seq * d_head;
    if (idx >= total) return;
    // Decompose src index: idx = (b*H+h)*S*d_head + s*d_head + d
    size_t d = idx % d_head;
    size_t tmp = idx / d_head;
    size_t s = tmp % seq;
    size_t bh = tmp / seq;
    size_t h = bh % n_heads;
    size_t b = bh / n_heads;
    size_t D = n_heads * d_head;
    dst[(b * seq + s) * D + h * d_head + d] = src[idx];
}

// Batched causal mask + scale: apply to num_matrices copies of [S, S].
// Same logic as causal_mask_scale but for a contiguous [num_matrices*S*S] buffer.
extern "C" __global__ void batched_causal_mask_scale(
    float* scores, size_t num_matrices, size_t seq_len, float scale, float mask_value
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = num_matrices * seq_len * seq_len;
    if (idx >= total) return;
    size_t local = idx % (seq_len * seq_len);
    size_t s = local / seq_len;
    size_t t = local % seq_len;
    if (t > s) {
        scores[idx] = mask_value;
    } else {
        scores[idx] *= scale;
    }
}

// In-place scale: a[i] *= scalar.
extern "C" __global__ void scale_inplace_kernel(float* a, float scalar, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] *= scalar;
}

// Reduce rows: out[col] = sum over rows of x[row * cols + col].
// One block per column, shared memory reduction.
extern "C" __global__ void reduce_rows(float* out, const float* x, size_t rows, size_t cols) {
    extern __shared__ float sdata[];
    size_t col = blockIdx.x;
    if (col >= cols) return;
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    float local_sum = 0.0f;
    for (size_t r = tid; r < rows; r += stride) {
        local_sum += x[r * cols + col];
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[col] = sdata[0];
}

// ============================================================
// Fused residual-add + LayerNorm: one block per row
// out_sum[i] = residual[i] + sublayer[i]
// out_norm = layernorm(out_sum, gamma, beta)
// Two output buffers, reads each input once.
// ============================================================

extern "C" __global__ void residual_add_layernorm_fwd(
    float* out_norm, float* out_sum,
    float* mean_out, float* rstd_out,
    const float* residual, const float* sublayer,
    const float* gamma, const float* beta,
    size_t rows, size_t d, float eps
) {
    extern __shared__ float sdata[];

    size_t row = blockIdx.x;
    if (row >= rows) return;
    size_t off = row * d;
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    // Pass 1: compute sum = residual + sublayer, write to out_sum, accumulate mean
    float local_sum = 0.0f;
    for (size_t j = tid; j < d; j += stride) {
        float s = residual[off + j] + sublayer[off + j];
        out_sum[off + j] = s;
        local_sum += s;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / (float)d;
    __syncthreads();

    // Pass 2: variance
    float local_var = 0.0f;
    for (size_t j = tid; j < d; j += stride) {
        float diff = out_sum[off + j] - mean;
        local_var += diff * diff;
    }
    sdata[tid] = local_var;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float var = sdata[0] / (float)d;
    float rstd = rsqrtf(var + eps);
    __syncthreads();

    if (tid == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }

    // Pass 3: normalize + affine
    for (size_t j = tid; j < d; j += stride) {
        float norm = (out_sum[off + j] - mean) * rstd;
        out_norm[off + j] = norm * gamma[j] + beta[j];
    }
}

// Backward for fused residual-add + LayerNorm.
// Recomputes sum = residual + sublayer in-kernel to avoid saving it.
// Outputs d_input (shared for both residual and sublayer), d_gamma, d_beta.
extern "C" __global__ void residual_add_layernorm_bwd(
    float* dx, float* dgamma, float* dbeta,
    const float* dy, const float* residual, const float* sublayer,
    const float* gamma, const float* mean, const float* rstd,
    size_t rows, size_t d
) {
    extern __shared__ float sdata[];

    size_t row = blockIdx.x;
    if (row >= rows) return;
    size_t off = row * d;
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;
    float m = mean[row];
    float rs = rstd[row];

    float local_sum_dnorm = 0.0f;
    float local_sum_dnorm_norm = 0.0f;

    for (size_t j = tid; j < d; j += stride) {
        float x_val = residual[off + j] + sublayer[off + j];
        float dnorm = dy[off + j] * gamma[j];
        float norm = (x_val - m) * rs;
        local_sum_dnorm += dnorm;
        local_sum_dnorm_norm += dnorm * norm;
        atomicAdd(&dgamma[j], dy[off + j] * norm);
        atomicAdd(&dbeta[j], dy[off + j]);
    }

    sdata[tid] = local_sum_dnorm;
    sdata[blockDim.x + tid] = local_sum_dnorm_norm;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[blockDim.x + tid] += sdata[blockDim.x + tid + s];
        }
        __syncthreads();
    }
    float sum_dnorm = sdata[0];
    float sum_dnorm_norm = sdata[blockDim.x];
    __syncthreads();

    float inv_d = 1.0f / (float)d;
    for (size_t j = tid; j < d; j += stride) {
        float x_val = residual[off + j] + sublayer[off + j];
        float dnorm = dy[off + j] * gamma[j];
        float norm = (x_val - m) * rs;
        dx[off + j] = inv_d * rs * ((float)d * dnorm - sum_dnorm - norm * sum_dnorm_norm);
    }
}

// ============================================================
// Fused QKV split + reshape to heads (replaces 3× gather_columns + 3× reshape)
// Input:  qkv [B*S, 3*D]  where D = H * d_head
// Output: q [B*H, S, d_head], k [B*H, S, d_head], v [B*H, S, d_head]
// Total threads: B * H * S * d_head (same size as one output)
// ============================================================

extern "C" __global__ void split_qkv_to_heads(
    float* q, float* k, float* v,
    const float* qkv,
    size_t batch, size_t seq, size_t n_heads, size_t d_head
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * n_heads * seq * d_head;
    if (idx >= total) return;

    // Decompose idx into (b, h, s, d) in [B*H, S, d_head] layout
    size_t d = idx % d_head;
    size_t tmp = idx / d_head;
    size_t s = tmp % seq;
    size_t bh = tmp / seq;
    size_t h = bh % n_heads;
    size_t b = bh / n_heads;

    size_t D = n_heads * d_head;
    // qkv layout: row = b*seq+s, col = [Q_0..Q_D-1, K_0..K_D-1, V_0..V_D-1]
    size_t row = (b * seq + s) * (3 * D);
    size_t col_in_head = h * d_head + d;

    q[idx] = qkv[row + col_in_head];
    k[idx] = qkv[row + D + col_in_head];
    v[idx] = qkv[row + 2 * D + col_in_head];
}

// ============================================================
// Fused merge heads + scatter to QKV (replaces 3× reshape_from_heads + 3× scatter_columns)
// Input: dq [B*H, S, d_head], dk [B*H, S, d_head], dv [B*H, S, d_head]
// Output: d_qkv [B*S, 3*D]  (additive scatter)
// ============================================================

extern "C" __global__ void merge_heads_to_qkv(
    float* d_qkv,
    const float* dq, const float* dk, const float* dv,
    size_t batch, size_t seq, size_t n_heads, size_t d_head
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * n_heads * seq * d_head;
    if (idx >= total) return;

    size_t d = idx % d_head;
    size_t tmp = idx / d_head;
    size_t s = tmp % seq;
    size_t bh = tmp / seq;
    size_t h = bh % n_heads;
    size_t b = bh / n_heads;

    size_t D = n_heads * d_head;
    size_t row = (b * seq + s) * (3 * D);
    size_t col_in_head = h * d_head + d;

    // Additive scatter (multiple heads write to same row, but different columns
    // within a head — no conflict since each (h, d) pair is unique)
    d_qkv[row + col_in_head] += dq[idx];
    d_qkv[row + D + col_in_head] += dk[idx];
    d_qkv[row + 2 * D + col_in_head] += dv[idx];
}

// ============================================================
// Fused bias + GELU: out[i] = gelu(matmul_out[i] + bias[i % cols])
// Replaces separate bias_add + gelu kernels. Saves one full tensor read+write.
// ============================================================

extern "C" __global__ void fused_bias_gelu_fwd(
    float* out,
    const float* inp, const float* bias,
    size_t rows, size_t cols
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    size_t col = idx % cols;
    float v = inp[idx] + bias[col];
    float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
    out[idx] = 0.5f * v * (1.0f + tanhf(inner));
}

// Backward for fused bias+GELU. Produces d_input (same shape as input) which is
// the gradient w.r.t. the pre-bias-add matmul output. Bias gradient is obtained
// by reduce_rows on d_input.
// Needs the pre-GELU value (matmul_out + bias) to compute GELU derivative.
extern "C" __global__ void fused_bias_gelu_bwd(
    float* d_input,
    const float* d_out,
    const float* inp, const float* bias,
    size_t rows, size_t cols
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    size_t col = idx % cols;
    float v = inp[idx] + bias[col];
    float kappa = 0.044715f;
    float s2pi = 0.7978845608f;
    float inner = s2pi * (v + kappa * v * v * v);
    float tanh_val = tanhf(inner);
    float sech2 = 1.0f - tanh_val * tanh_val;
    float d_inner = s2pi * (1.0f + 3.0f * kappa * v * v);
    d_input[idx] = d_out[idx] * (0.5f * (1.0f + tanh_val) + 0.5f * v * sech2 * d_inner);
}

// ============================================================
// Fused bias + dropout + residual add:
// out[i] = residual[i] + dropout(matmul_out[i] + bias[i % cols], p, seed)
// Replaces separate bias_add + dropout + residual_add (3 kernels → 1).
// Also outputs the dropout mask for backward.
// ============================================================

extern "C" __global__ void fused_bias_dropout_residual_fwd(
    float* out, float* mask_out,
    const float* matmul_out, const float* bias, const float* residual,
    size_t rows, size_t cols,
    float p, float scale,
    unsigned long long seed
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = rows * cols;
    if (idx >= n) return;
    size_t col = idx % cols;
    float biased = matmul_out[idx] + bias[col];
    float r = rand_uniform(seed, idx);
    float m = (r >= p) ? scale : 0.0f;
    mask_out[idx] = m;
    out[idx] = residual[idx] + biased * m;
}

// ============================================================
// FlashAttention forward kernel — warp-cooperative tiled
// Grid: (ceil(S/FA_BR), B*H, 1)
// Block: (FA_BR, 32, 1) — each warp (threadIdx.x) handles one Q row,
//   32 lanes (threadIdx.y) cooperate on the D dimension via shuffle reduction.
//
// Shared memory: K_tile[FA_BC * D] + V_tile[FA_BC * D]
// Each block owns FA_BR rows of Q/O, iterates over K/V in tiles of FA_BC.
// ============================================================

#define FA_BR 32
#define FA_BC 32
#define FA_MAX_DPT 4  // max ceil(D/32), supports D up to 128

extern "C" __global__ void flash_attention_fwd(
    float* O, float* lse_out,
    const float* Q, const float* K, const float* V,
    int S, int D, float scale, int is_causal
) {
    extern __shared__ float smem[];
    float* K_smem = smem;              // [FA_BC * D]
    float* V_smem = smem + FA_BC * D;  // [FA_BC * D]

    int tile_row = blockIdx.x;
    int bh = blockIdx.y;
    int row_in_tile = threadIdx.x;  // [0, FA_BR) — which Q row
    int lane = threadIdx.y;          // [0, 32) — D-dimension lane

    int row = tile_row * FA_BR + row_in_tile;
    // Entire warp shares same row — safe for warp-coherent early exit
    if (row >= S) return;

    const float* Q_bh = Q + (size_t)bh * S * D;
    const float* K_bh = K + (size_t)bh * S * D;
    const float* V_bh = V + (size_t)bh * S * D;
    float* O_bh = O + (size_t)bh * S * D;
    float* lse_bh = lse_out + (size_t)bh * S;

    int dpt = (D + 31) / 32;  // elements per lane (interleaved)

    // Load Q row into registers (pre-scaled), interleaved across lanes
    float q_reg[FA_MAX_DPT];
    for (int dd = 0; dd < dpt; dd++) {
        int d = dd * 32 + lane;
        q_reg[dd] = (d < D) ? Q_bh[row * D + d] * scale : 0.0f;
    }

    // Online softmax state (warp-uniform after each reduction)
    float row_max = -1e30f;
    float row_sum = 0.0f;
    float acc[FA_MAX_DPT];
    for (int dd = 0; dd < dpt; dd++) acc[dd] = 0.0f;

    int n_kv_tiles = (S + FA_BC - 1) / FA_BC;
    int max_kv_tile = is_causal ? ((row / FA_BC) + 1) : n_kv_tiles;

    // Flat thread ID for cooperative loading (1024 threads)
    int tid = row_in_tile * 32 + lane;
    int nthreads = FA_BR * 32;

    for (int kv_tile = 0; kv_tile < max_kv_tile; kv_tile++) {
        int kv_start = kv_tile * FA_BC;
        int tile_size = (kv_start + FA_BC > S) ? (S - kv_start) : FA_BC;
        int tile_elems = tile_size * D;

        // Cooperative load K and V tiles (1024 threads — 16x faster than old 64)
        for (int i = tid; i < tile_elems; i += nthreads) {
            K_smem[i] = K_bh[kv_start * D + i];
        }
        for (int i = tid; i < tile_elems; i += nthreads) {
            V_smem[i] = V_bh[kv_start * D + i];
        }
        __syncthreads();

        // Process each K vector in tile — warp-cooperative dot products
        for (int j = 0; j < tile_size; j++) {
            int kv_global = kv_start + j;
            if (is_causal && kv_global > row) break;

            // Warp-cooperative dot product: Q[row] · K[j]
            float partial = 0.0f;
            for (int dd = 0; dd < dpt; dd++) {
                int d = dd * 32 + lane;
                if (d < D) partial += q_reg[dd] * K_smem[j * D + d];
            }
            // Butterfly warp reduction — all lanes get full dot product
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                partial += __shfl_xor_sync(0xffffffff, partial, offset);
            }
            float s = partial;

            // Online softmax update (warp-uniform: all lanes have same s)
            float old_max = row_max;
            if (s > row_max) row_max = s;
            float correction = expf(old_max - row_max);
            row_sum = row_sum * correction + expf(s - row_max);

            // V accumulation: each lane handles its D slice
            float w = expf(s - row_max);
            for (int dd = 0; dd < dpt; dd++) {
                int d = dd * 32 + lane;
                if (d < D) {
                    acc[dd] = acc[dd] * correction + w * V_smem[j * D + d];
                }
            }
        }
        __syncthreads();
    }

    // Finalize: O[row] = acc / row_sum, each lane writes its D slice
    float inv_sum = 1.0f / row_sum;
    for (int dd = 0; dd < dpt; dd++) {
        int d = dd * 32 + lane;
        if (d < D) O_bh[row * D + d] = acc[dd] * inv_sum;
    }

    if (lane == 0) {
        lse_bh[row] = row_max + logf(row_sum);
    }
}

// ============================================================
// FlashAttention backward kernel — warp-cooperative KV-tile-centric
// Grid: (ceil(S/FA_BWD_BC), B*H, 1)
// Block: (FA_BWD_BC, 32, 1) — each warp (threadIdx.x) owns one K/V row,
//   32 lanes (threadIdx.y) cooperate on D-dimension dot products and accumulation.
//
// dK and dV accumulated in registers (zero atomicAdd).
// Only dQ uses atomicAdd (BC-way contention, not S-way).
// ============================================================

#define FA_BWD_BC 32

extern "C" __global__ void flash_attention_bwd(
    float* dQ, float* dK, float* dV,
    const float* dO,
    const float* Q, const float* K, const float* V,
    const float* O, const float* lse,
    int S, int D, float scale, int is_causal
) {
    extern __shared__ float smem[];
    float* K_smem = smem;                    // [FA_BWD_BC * D]
    float* V_smem = smem + FA_BWD_BC * D;   // [FA_BWD_BC * D]

    int kv_tile = blockIdx.x;
    int bh = blockIdx.y;
    int kv_in_tile = threadIdx.x;  // [0, FA_BWD_BC)
    int lane = threadIdx.y;         // [0, 32)

    int kv_start = kv_tile * FA_BWD_BC;
    int my_kv_row = kv_start + kv_in_tile;
    if (my_kv_row >= S) return;

    const float* Q_bh  = Q  + (size_t)bh * S * D;
    const float* K_bh  = K  + (size_t)bh * S * D;
    const float* V_bh  = V  + (size_t)bh * S * D;
    const float* O_bh  = O  + (size_t)bh * S * D;
    const float* dO_bh = dO + (size_t)bh * S * D;
    const float* lse_bh = lse + (size_t)bh * S;
    float* dQ_bh = dQ + (size_t)bh * S * D;
    float* dK_bh = dK + (size_t)bh * S * D;
    float* dV_bh = dV + (size_t)bh * S * D;

    int dpt = (D + 31) / 32;

    // Load my K and V row into shared memory — lanes cooperate
    for (int dd = 0; dd < dpt; dd++) {
        int d = dd * 32 + lane;
        if (d < D) {
            K_smem[kv_in_tile * D + d] = K_bh[my_kv_row * D + d];
            V_smem[kv_in_tile * D + d] = V_bh[my_kv_row * D + d];
        }
    }
    __syncthreads();

    // Accumulators for dK and dV — each lane holds its D slice
    float dk_acc[FA_MAX_DPT];
    float dv_acc[FA_MAX_DPT];
    for (int dd = 0; dd < dpt; dd++) {
        dk_acc[dd] = 0.0f;
        dv_acc[dd] = 0.0f;
    }

    int q_start = is_causal ? my_kv_row : 0;

    for (int i = q_start; i < S; i++) {
        float lse_i = lse_bh[i];

        // Load Q[i], dO[i], O[i] into registers — each lane loads its D slice
        float q_r[FA_MAX_DPT], do_r[FA_MAX_DPT], o_r[FA_MAX_DPT];
        for (int dd = 0; dd < dpt; dd++) {
            int d = dd * 32 + lane;
            if (d < D) {
                q_r[dd]  = Q_bh[i * D + d];
                do_r[dd] = dO_bh[i * D + d];
                o_r[dd]  = O_bh[i * D + d];
            } else {
                q_r[dd] = 0.0f; do_r[dd] = 0.0f; o_r[dd] = 0.0f;
            }
        }

        // Fused warp-cooperative reductions: score, Di, dov
        float s_part = 0.0f, di_part = 0.0f, dov_part = 0.0f;
        for (int dd = 0; dd < dpt; dd++) {
            int d = dd * 32 + lane;
            if (d < D) {
                float k_d = K_smem[kv_in_tile * D + d];
                float v_d = V_smem[kv_in_tile * D + d];
                s_part   += q_r[dd] * scale * k_d;
                di_part  += do_r[dd] * o_r[dd];
                dov_part += do_r[dd] * v_d;
            }
        }
        // 3 butterfly reductions
        float s = s_part, Di = di_part, dov = dov_part;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            s   += __shfl_xor_sync(0xffffffff, s,   offset);
            Di  += __shfl_xor_sync(0xffffffff, Di,  offset);
            dov += __shfl_xor_sync(0xffffffff, dov, offset);
        }

        float p_ij = expf(s - lse_i);
        float ds_ij = p_ij * (dov - Di);

        // Accumulate dK, dV, dQ — each lane handles its D slice
        for (int dd = 0; dd < dpt; dd++) {
            int d = dd * 32 + lane;
            if (d < D) {
                dk_acc[dd] += ds_ij * scale * q_r[dd];
                dv_acc[dd] += p_ij * do_r[dd];
                atomicAdd(&dQ_bh[i * D + d], ds_ij * scale * K_smem[kv_in_tile * D + d]);
            }
        }
    }

    // Write dK and dV — each lane writes its D slice
    for (int dd = 0; dd < dpt; dd++) {
        int d = dd * 32 + lane;
        if (d < D) {
            dK_bh[my_kv_row * D + d] = dk_acc[dd];
            dV_bh[my_kv_row * D + d] = dv_acc[dd];
        }
    }
}

// ============================================================
// Fused elementwise multiply + row reduction:
// out[col] = sum_over_rows(a[row * cols + col] * b[row * cols + col])
// Replaces separate mul_elementwise + reduce_rows (2 kernels + 1 alloc → 1 kernel).
// One block per column, shared memory reduction.
// ============================================================

extern "C" __global__ void fused_mul_reduce_rows(
    float* out, const float* a, const float* b, size_t rows, size_t cols
) {
    extern __shared__ float sdata[];
    size_t col = blockIdx.x;
    if (col >= cols) return;
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    float local_sum = 0.0f;
    for (size_t r = tid; r < rows; r += stride) {
        size_t idx = r * cols + col;
        local_sum += a[idx] * b[idx];
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[col] = sdata[0];
}
"#;

#[cfg(feature = "bf16")]
const BF16_KERNEL_SOURCE: &str = r#"
// ============================================================
// BF16 helpers: bitwise cast (no cuda_bf16.h needed for NVRTC)
// ============================================================

typedef unsigned short bf16_t;

__device__ __forceinline__ float bf16_to_float(bf16_t x) {
    unsigned int bits = ((unsigned int)x) << 16;
    float r;
    memcpy(&r, &bits, 4);
    return r;
}

__device__ __forceinline__ bf16_t float_to_bf16(float x) {
    unsigned int bits;
    memcpy(&bits, &x, 4);
    // Round-to-nearest-even
    return (bf16_t)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

// ============================================================
// Cast kernels (only these are needed — element-wise ops use f32 path,
// matmul uses cublasGemmEx with bf16 shadow inputs and f32 output)
// ============================================================

extern "C" __global__ void f32_to_bf16(bf16_t* out, const float* in, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = float_to_bf16(in[i]);
}

extern "C" __global__ void bf16_to_f32(float* out, const bf16_t* in, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = bf16_to_float(in[i]);
}

// ============================================================
// BF16 FlashAttention forward kernel — warp-cooperative mixed-precision
// Q/K/V inputs: bf16, K/V shared tiles: bf16
// Score accumulation, online softmax, O, lse: f32
// Grid: (ceil(S/FA_BF16_BR), B*H, 1)
// Block: (FA_BF16_BR, 32, 1) — warp-cooperative on D dimension
// ============================================================

#define FA_BF16_BR 32
#define FA_BF16_BC 64
#define FA_BF16_MAX_DPT 4

extern "C" __global__ void flash_attention_fwd_bf16(
    float* O, float* lse_out,
    const bf16_t* Q, const bf16_t* K, const bf16_t* V,
    int S, int D, float scale, int is_causal
) {
    extern __shared__ bf16_t smem_bf16[];
    bf16_t* K_smem = smem_bf16;                      // [FA_BF16_BC * D]
    bf16_t* V_smem = smem_bf16 + FA_BF16_BC * D;     // [FA_BF16_BC * D]

    int tile_row = blockIdx.x;
    int bh = blockIdx.y;
    int row_in_tile = threadIdx.x;  // [0, FA_BF16_BR)
    int lane = threadIdx.y;          // [0, 32)

    int row = tile_row * FA_BF16_BR + row_in_tile;
    if (row >= S) return;

    const bf16_t* Q_bh = Q + (size_t)bh * S * D;
    const bf16_t* K_bh = K + (size_t)bh * S * D;
    const bf16_t* V_bh = V + (size_t)bh * S * D;
    float* O_bh = O + (size_t)bh * S * D;
    float* lse_bh = lse_out + (size_t)bh * S;

    int dpt = (D + 31) / 32;

    // Load Q row into f32 registers (pre-scaled), interleaved
    float q_reg[FA_BF16_MAX_DPT];
    for (int dd = 0; dd < dpt; dd++) {
        int d = dd * 32 + lane;
        q_reg[dd] = (d < D) ? bf16_to_float(Q_bh[row * D + d]) * scale : 0.0f;
    }

    float row_max = -1e30f;
    float row_sum = 0.0f;
    float acc[FA_BF16_MAX_DPT];
    for (int dd = 0; dd < dpt; dd++) acc[dd] = 0.0f;

    int n_kv_tiles = (S + FA_BF16_BC - 1) / FA_BF16_BC;
    int max_kv_tile = is_causal ? ((row / FA_BF16_BC) + 1) : n_kv_tiles;

    int tid = row_in_tile * 32 + lane;
    int nthreads = FA_BF16_BR * 32;

    for (int kv_tile = 0; kv_tile < max_kv_tile; kv_tile++) {
        int kv_start = kv_tile * FA_BF16_BC;
        int tile_size = (kv_start + FA_BF16_BC > S) ? (S - kv_start) : FA_BF16_BC;
        int tile_elems = tile_size * D;

        for (int i = tid; i < tile_elems; i += nthreads) {
            K_smem[i] = K_bh[kv_start * D + i];
        }
        for (int i = tid; i < tile_elems; i += nthreads) {
            V_smem[i] = V_bh[kv_start * D + i];
        }
        __syncthreads();

        for (int j = 0; j < tile_size; j++) {
            int kv_global = kv_start + j;
            if (is_causal && kv_global > row) break;

            float partial = 0.0f;
            for (int dd = 0; dd < dpt; dd++) {
                int d = dd * 32 + lane;
                if (d < D) partial += q_reg[dd] * bf16_to_float(K_smem[j * D + d]);
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                partial += __shfl_xor_sync(0xffffffff, partial, offset);
            }
            float s = partial;

            float old_max = row_max;
            if (s > row_max) row_max = s;
            float correction = expf(old_max - row_max);
            row_sum = row_sum * correction + expf(s - row_max);

            float w = expf(s - row_max);
            for (int dd = 0; dd < dpt; dd++) {
                int d = dd * 32 + lane;
                if (d < D) {
                    acc[dd] = acc[dd] * correction + w * bf16_to_float(V_smem[j * D + d]);
                }
            }
        }
        __syncthreads();
    }

    float inv_sum = 1.0f / row_sum;
    for (int dd = 0; dd < dpt; dd++) {
        int d = dd * 32 + lane;
        if (d < D) O_bh[row * D + d] = acc[dd] * inv_sum;
    }

    if (lane == 0) {
        lse_bh[row] = row_max + logf(row_sum);
    }
}

// ============================================================
// BF16 FlashAttention backward — warp-cooperative KV-tile-centric
// Q/K/V/dO: bf16, O/lse: f32, dQ/dK/dV: f32
// Grid: (ceil(S/FA_BF16_BWD_BC), B*H, 1)
// Block: (FA_BF16_BWD_BC, 32, 1)
// ============================================================

#define FA_BF16_BWD_BC 32

extern "C" __global__ void flash_attention_bwd_bf16(
    float* dQ, float* dK, float* dV,
    const bf16_t* dO,
    const bf16_t* Q, const bf16_t* K, const bf16_t* V,
    const float* O, const float* lse,
    int S, int D, float scale, int is_causal
) {
    extern __shared__ bf16_t smem_bf16_bwd[];
    bf16_t* K_smem = smem_bf16_bwd;                          // [FA_BF16_BWD_BC * D]
    bf16_t* V_smem = smem_bf16_bwd + FA_BF16_BWD_BC * D;     // [FA_BF16_BWD_BC * D]

    int kv_tile = blockIdx.x;
    int bh = blockIdx.y;
    int kv_in_tile = threadIdx.x;  // [0, FA_BF16_BWD_BC)
    int lane = threadIdx.y;         // [0, 32)

    int kv_start = kv_tile * FA_BF16_BWD_BC;
    int my_kv_row = kv_start + kv_in_tile;
    if (my_kv_row >= S) return;

    const bf16_t* Q_bh  = Q  + (size_t)bh * S * D;
    const bf16_t* K_bh  = K  + (size_t)bh * S * D;
    const bf16_t* V_bh  = V  + (size_t)bh * S * D;
    const float*  O_bh  = O  + (size_t)bh * S * D;
    const bf16_t* dO_bh = dO + (size_t)bh * S * D;
    const float*  lse_bh = lse + (size_t)bh * S;
    float* dQ_bh = dQ + (size_t)bh * S * D;
    float* dK_bh = dK + (size_t)bh * S * D;
    float* dV_bh = dV + (size_t)bh * S * D;

    int dpt = (D + 31) / 32;

    // Load my K and V row into bf16 shared memory — lanes cooperate
    for (int dd = 0; dd < dpt; dd++) {
        int d = dd * 32 + lane;
        if (d < D) {
            K_smem[kv_in_tile * D + d] = K_bh[my_kv_row * D + d];
            V_smem[kv_in_tile * D + d] = V_bh[my_kv_row * D + d];
        }
    }
    __syncthreads();

    float dk_acc[FA_BF16_MAX_DPT];
    float dv_acc[FA_BF16_MAX_DPT];
    for (int dd = 0; dd < dpt; dd++) {
        dk_acc[dd] = 0.0f;
        dv_acc[dd] = 0.0f;
    }

    int q_start = is_causal ? my_kv_row : 0;

    for (int i = q_start; i < S; i++) {
        float lse_i = lse_bh[i];

        // Load Q[i], dO[i], O[i] — each lane loads its D slice
        float q_r[FA_BF16_MAX_DPT], do_r[FA_BF16_MAX_DPT], o_r[FA_BF16_MAX_DPT];
        for (int dd = 0; dd < dpt; dd++) {
            int d = dd * 32 + lane;
            if (d < D) {
                q_r[dd]  = bf16_to_float(Q_bh[i * D + d]);
                do_r[dd] = bf16_to_float(dO_bh[i * D + d]);
                o_r[dd]  = O_bh[i * D + d];
            } else {
                q_r[dd] = 0.0f; do_r[dd] = 0.0f; o_r[dd] = 0.0f;
            }
        }

        // Fused warp-cooperative reductions
        float s_part = 0.0f, di_part = 0.0f, dov_part = 0.0f;
        for (int dd = 0; dd < dpt; dd++) {
            int d = dd * 32 + lane;
            if (d < D) {
                float k_d = bf16_to_float(K_smem[kv_in_tile * D + d]);
                float v_d = bf16_to_float(V_smem[kv_in_tile * D + d]);
                s_part   += q_r[dd] * scale * k_d;
                di_part  += do_r[dd] * o_r[dd];
                dov_part += do_r[dd] * v_d;
            }
        }
        float s = s_part, Di = di_part, dov = dov_part;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            s   += __shfl_xor_sync(0xffffffff, s,   offset);
            Di  += __shfl_xor_sync(0xffffffff, Di,  offset);
            dov += __shfl_xor_sync(0xffffffff, dov, offset);
        }

        float p_ij = expf(s - lse_i);
        float ds_ij = p_ij * (dov - Di);

        for (int dd = 0; dd < dpt; dd++) {
            int d = dd * 32 + lane;
            if (d < D) {
                dk_acc[dd] += ds_ij * scale * q_r[dd];
                dv_acc[dd] += p_ij * do_r[dd];
                atomicAdd(&dQ_bh[i * D + d], ds_ij * scale * bf16_to_float(K_smem[kv_in_tile * D + d]));
            }
        }
    }

    for (int dd = 0; dd < dpt; dd++) {
        int d = dd * 32 + lane;
        if (d < D) {
            dK_bh[my_kv_row * D + d] = dk_acc[dd];
            dV_bh[my_kv_row * D + d] = dv_acc[dd];
        }
    }
}
"#;
