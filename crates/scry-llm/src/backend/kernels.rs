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

    // BF16 kernels (only cast kernels needed — element-wise ops use f32 path)
    #[cfg(feature = "bf16")]
    pub f32_to_bf16: CudaFunction,
    #[cfg(feature = "bf16")]
    pub bf16_to_f32: CudaFunction,
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

            #[cfg(feature = "bf16")]
            f32_to_bf16: bf16_module.load_function("f32_to_bf16").unwrap(),
            #[cfg(feature = "bf16")]
            bf16_to_f32: bf16_module.load_function("bf16_to_f32").unwrap(),
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
    __syncthreads();

    float inv_batch = 1.0f / (float)batch;
    for (size_t j = tid; j < vocab; j += stride) {
        float prob = expf(logits[off + j] - max_val) / sum_exp;
        float target_val = (j == target) ? 1.0f : 0.0f;
        d_logits[off + j] = (prob - target_val) * inv_batch;
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
"#;
