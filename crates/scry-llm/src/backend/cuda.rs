use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublaslt::{CudaBlasLT, Matmul as CublasLtMatmul, MatmulConfig as LtMatmulConfig};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::backend::{DeviceBackend, MathBackend};
use crate::tensor::shape::Shape;

use super::kernels::KernelCache;

// ---- CUDA caching memory allocator ----

/// Pool of device buffers bucketed by `next_power_of_two()` element count.
/// Eliminates per-op `cuMemAlloc`/`cuMemFree` round-trips to the CUDA driver.
struct GpuPool {
    free: HashMap<usize, Vec<CudaSlice<f32>>>,
    hits: u64,
    misses: u64,
}

impl GpuPool {
    fn new() -> Self {
        Self {
            free: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// Allocate a device buffer of at least `n` f32 elements. Uninitialized.
    /// Checks the pool first; falls back to `cuMemAlloc` on miss.
    fn alloc(&mut self, stream: &Arc<CudaStream>, n: usize) -> CudaSlice<f32> {
        let bucket = n.next_power_of_two().max(1);
        if let Some(list) = self.free.get_mut(&bucket) {
            if let Some(buf) = list.pop() {
                self.hits += 1;
                return buf;
            }
        }
        self.misses += 1;
        unsafe { stream.alloc::<f32>(bucket).unwrap() }
    }

    /// Allocate a zeroed device buffer of at least `n` f32 elements.
    /// Pool buffer + `memset_zeros`, or fresh `alloc_zeros`.
    fn alloc_zeros(&mut self, stream: &Arc<CudaStream>, n: usize) -> CudaSlice<f32> {
        let bucket = n.next_power_of_two().max(1);
        if let Some(list) = self.free.get_mut(&bucket) {
            if let Some(mut buf) = list.pop() {
                self.hits += 1;
                stream.memset_zeros(&mut buf).unwrap();
                return buf;
            }
        }
        self.misses += 1;
        stream.alloc_zeros::<f32>(bucket).unwrap()
    }

    /// Return a buffer to the pool for reuse.
    fn recycle(&mut self, buf: CudaSlice<f32>, len: usize) {
        let bucket = len.next_power_of_two().max(1);
        self.free.entry(bucket).or_default().push(buf);
    }

    /// Pool hit/miss statistics. Returns `(hits, misses)`.
    pub fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }

    /// Drop all pooled buffers (for VRAM pressure).
    #[allow(dead_code)]
    pub fn trim(&mut self) {
        self.free.clear();
    }
}

// ---- BF16 mode flag ----

#[cfg(feature = "bf16")]
thread_local! {
    static BF16_MODE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(feature = "bf16")]
fn bf16_enabled() -> bool {
    BF16_MODE.with(std::cell::Cell::get)
}


/// Thread-local GPU context: device, stream, cuBLAS handle, compiled kernels, memory pool.
struct GpuCtx {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    blas_lt: CudaBlasLT,
    kernels: KernelCache,
    pool: RefCell<GpuPool>,
}

thread_local! {
    static GPU_CTX: RefCell<Option<GpuCtx>> = const { RefCell::new(None) };
}

/// Initialize the GPU backend on the current thread. Must be called before any
/// `CudaBackend` operations. Compiles all NVRTC kernels and creates a cuBLAS handle.
///
/// # Panics
///
/// Panics if the CUDA device cannot be initialized.
pub fn init_gpu(device_id: usize) {
    GPU_CTX.with(|cell| {
        let ctx = CudaContext::new(device_id).expect("failed to create CUDA context");
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).expect("failed to create cuBLAS handle");

        // Enable TF32 tensor cores for fp32 sgemm. ~2-3x faster with 10-bit
        // mantissa precision (vs 23-bit). PyTorch enables this by default.
        unsafe {
            use cudarc::cublas::sys::{cublasSetMathMode, cublasMath_t};
            cublasSetMathMode(
                *blas.handle(),
                cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH,
            );
        }

        let blas_lt = CudaBlasLT::new(stream.clone()).expect("failed to create cuBLASLt handle");
        let kernels = KernelCache::compile(&ctx);
        *cell.borrow_mut() = Some(GpuCtx {
            ctx,
            stream,
            blas,
            blas_lt,
            kernels,
            pool: RefCell::new(GpuPool::new()),
        });
    });
}

/// Initialize the GPU backend in BF16 mixed-precision mode. All compute ops
/// will cast f32→bf16, run on tensor cores, and cast back to f32 storage.
/// `AdamW` stays fully f32 (master weights).
#[cfg(feature = "bf16")]
pub fn init_gpu_bf16(device_id: usize) {
    init_gpu(device_id);
    BF16_MODE.with(|c| c.set(true));
}

/// Access the thread-local GPU context. Panics if `init_gpu` was not called.
fn with_gpu<R>(f: impl FnOnce(&GpuCtx) -> R) -> R {
    GPU_CTX.with(|cell| {
        let borrow = cell.borrow();
        let gpu = borrow.as_ref().expect("GPU not initialized — call init_gpu() first");
        f(gpu)
    })
}

/// Allocate from the pool (uninitialized). Use only when the kernel fully overwrites.
fn pool_alloc(gpu: &GpuCtx, n: usize) -> CudaSlice<f32> {
    gpu.pool.borrow_mut().alloc(&gpu.stream, n)
}

/// Allocate from the pool (zeroed). Use for atomicAdd targets, reductions, etc.
fn pool_alloc_zeros(gpu: &GpuCtx, n: usize) -> CudaSlice<f32> {
    gpu.pool.borrow_mut().alloc_zeros(&gpu.stream, n)
}

/// Query pool statistics. Returns `(hits, misses)`.
pub fn pool_stats() -> (u64, u64) {
    with_gpu(|gpu| gpu.pool.borrow().stats())
}

/// Access the thread-local GPU context mutably (for cuBLAS which needs &mut).
fn with_gpu_mut<R>(f: impl FnOnce(&mut GpuCtx) -> R) -> R {
    GPU_CTX.with(|cell| {
        let mut borrow = cell.borrow_mut();
        let gpu = borrow.as_mut().expect("GPU not initialized — call init_gpu() first");
        f(gpu)
    })
}

/// GPU storage: a device-side f32 buffer with optional bf16 shadow for mixed-precision.
///
/// Uses `Option<CudaSlice<f32>>` so `Drop` can take ownership and recycle to the pool.
#[derive(Debug)]
pub struct GpuStorage {
    buf: Option<CudaSlice<f32>>,
    pub(crate) len: usize,
    /// Cached bf16 copy, created lazily on first bf16 matmul access.
    /// Invalidated after each optimizer step so it stays in sync with f32 master weights.
    #[cfg(feature = "bf16")]
    pub(crate) bf16_shadow: std::cell::RefCell<Option<CudaSlice<half::bf16>>>,
}

impl Clone for GpuStorage {
    fn clone(&self) -> Self {
        // Route clone through the pool: alloc from pool + D2D copy
        with_gpu(|gpu| {
            let mut cloned = gpu.pool.borrow_mut().alloc(&gpu.stream, self.len);
            gpu.stream.memcpy_dtod(self.inner(), &mut cloned).unwrap();
            Self {
                buf: Some(cloned),
                len: self.len,
                #[cfg(feature = "bf16")]
                bf16_shadow: std::cell::RefCell::new(None),
            }
        })
    }
}

impl Drop for GpuStorage {
    fn drop(&mut self) {
        if let Some(slice) = self.buf.take() {
            // Try to recycle into the thread-local pool.
            // If the thread-local is already torn down (e.g. during thread exit),
            // the CudaSlice drops normally via cuMemFree.
            let len = self.len;
            let _ = GPU_CTX.try_with(|cell| {
                if let Ok(borrow) = cell.try_borrow() {
                    if let Some(gpu) = borrow.as_ref() {
                        gpu.pool.borrow_mut().recycle(slice, len);
                        return;
                    }
                }
                // Pool not available — let CudaSlice::drop handle it
                drop(slice);
            });
        }
    }
}

impl GpuStorage {
    /// Create a new `GpuStorage` from a `CudaSlice<f32>` and length.
    fn new(inner: CudaSlice<f32>, len: usize) -> Self {
        Self {
            buf: Some(inner),
            len,
            #[cfg(feature = "bf16")]
            bf16_shadow: std::cell::RefCell::new(None),
        }
    }

    /// Access the underlying `CudaSlice`.
    #[inline]
    pub(crate) fn inner(&self) -> &CudaSlice<f32> {
        self.buf.as_ref().expect("GpuStorage already recycled")
    }

    /// Mutable access to the underlying `CudaSlice`.
    #[inline]
    pub(crate) fn inner_mut(&mut self) -> &mut CudaSlice<f32> {
        self.buf.as_mut().expect("GpuStorage already recycled")
    }

    /// Invalidate the bf16 shadow cache (called after optimizer updates f32 master weights).
    #[cfg(feature = "bf16")]
    pub fn invalidate_bf16_shadow(&self) {
        *self.bf16_shadow.borrow_mut() = None;
    }

    /// Get or create the bf16 shadow. Returns a reference-counted clone of the shadow slice.
    #[cfg(feature = "bf16")]
    fn ensure_bf16_shadow(&self) -> CudaSlice<half::bf16> {
        let mut shadow = self.bf16_shadow.borrow_mut();
        if let Some(ref s) = *shadow {
            return s.try_clone().expect("failed to clone bf16 shadow");
        }
        // Create the shadow by casting f32 → bf16
        let s = with_gpu(|gpu| bf16_ops::cast_f32_to_bf16(gpu, self.inner(), self.len));
        let cloned = s.try_clone().expect("failed to clone bf16 shadow");
        *shadow = Some(s);
        cloned
    }
}

/// CUDA backend: all ops run on GPU via cuBLAS + custom NVRTC kernels.
pub struct CudaBackend;

/// Helper: compute grid size for n elements at given block size.
fn grid_for(n: usize, block: u32) -> u32 {
    (n as u32).div_ceil(block)
}

/// Block size for simple element-wise kernels.
const BLOCK: u32 = 256;
/// Block size for row-reduction kernels (softmax, layernorm). Must be power of 2.
const ROW_BLOCK: u32 = 256;

impl DeviceBackend for CudaBackend {
    type Storage = GpuStorage;
    type Stream = Arc<CudaStream>;

    fn zeros(shape: &Shape) -> GpuStorage {
        let n = shape.numel();
        with_gpu(|gpu| {
            let inner = pool_alloc_zeros(gpu, n);
            GpuStorage::new(inner, n )
        })
    }

    fn ones(shape: &Shape) -> GpuStorage {
        let n = shape.numel();
        let data = vec![1.0f32; n];
        with_gpu(|gpu| {
            let mut inner = pool_alloc(gpu, n);
            gpu.stream.memcpy_htod(&data, &mut inner).unwrap();
            GpuStorage::new(inner, n)
        })
    }

    fn from_vec(data: Vec<f32>, _shape: &Shape) -> GpuStorage {
        let len = data.len();
        with_gpu(|gpu| {
            let mut inner = pool_alloc(gpu, len);
            gpu.stream.memcpy_htod(&data, &mut inner).unwrap();
            GpuStorage::new(inner, len)
        })
    }

    fn to_vec(storage: &GpuStorage) -> Vec<f32> {
        with_gpu(|gpu| {
            let full = gpu.stream.clone_dtoh(storage.inner()).unwrap();
            // Pool allocations may be larger than logical size (next_power_of_two bucketing).
            // Truncate to the actual element count.
            if full.len() > storage.len {
                full[..storage.len].to_vec()
            } else {
                full
            }
        })
    }

    fn clone_storage(storage: &GpuStorage) -> GpuStorage {
        storage.clone()
    }
}

impl MathBackend for CudaBackend {
    fn matmul(
        a: &GpuStorage,
        b: &GpuStorage,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> GpuStorage {
        #[cfg(feature = "bf16")]
        if bf16_enabled() {
            return bf16_ops::matmul_bf16(a, b, m, k, n, trans_a, trans_b);
        }
        with_gpu_mut(|gpu| {
            let mut c = pool_alloc(gpu, m * n);

            // Row-major trick: C = A @ B in row-major ↔ C^T = B^T @ A^T in col-major
            // cuBLAS sees column-major, so we pass B as first arg, A as second.
            //
            // For row-major A[M,K] (no trans): col-major sees A^T[K,M], so cuBLAS "A" = A^T
            //   → to get A in cuBLAS we use OP_T
            // For row-major A^T[K,M] (trans_a): col-major sees (A^T)^T = A[M,K]
            //   → cuBLAS already has what we want with OP_N
            //
            // Similarly for B.
            // But we swap A/B for the row-major trick, so:
            //   cuBLAS_A = B, cuBLAS_B = A
            //   cuBLAS m=N, cuBLAS n=M, cuBLAS k=K
            let (transa, lda) = if trans_b {
                // B is [N,K] row-major. cuBLAS sees [K,N] col-major. We want B^T = [K,N].
                // That's OP_N on the col-major [K,N]. lda = K.
                (cublasOperation_t::CUBLAS_OP_T, k as i32)
            } else {
                // B is [K,N] row-major. cuBLAS sees [N,K] col-major. We want B = [K,N] col-major.
                // That's OP_N on [N,K]? No — we want the row-major B as-is.
                // Row-major [K,N] = col-major [N,K]. cuBLAS "A" = [N,K] col-major.
                // We want cuBLAS to compute with B (not B^T), so OP_N. lda = N.
                (cublasOperation_t::CUBLAS_OP_N, n as i32)
            };
            let (transb, ldb) = if trans_a {
                // A is [K,M] row-major. cuBLAS sees [M,K] col-major. Want A^T.
                // OP_T on [M,K] gives [K,M]. ldb = M.
                (cublasOperation_t::CUBLAS_OP_T, m as i32)
            } else {
                // A is [M,K] row-major = [K,M] col-major. Want A. OP_N. ldb = K.
                (cublasOperation_t::CUBLAS_OP_N, k as i32)
            };

            unsafe {
                gpu.blas
                    .gemm(
                        GemmConfig {
                            transa,
                            transb,
                            m: n as i32,
                            n: m as i32,
                            k: k as i32,
                            alpha: 1.0f32,
                            lda,
                            ldb,
                            beta: 0.0f32,
                            ldc: n as i32,
                        },
                        b.inner(),
                        a.inner(),
                        &mut c,
                    )
                    .expect("cuBLAS sgemm failed");
            }

            GpuStorage::new(c, m * n )
        })
    }

    fn matmul_bias(
        a: &GpuStorage,
        b: &GpuStorage,
        bias: &GpuStorage,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> GpuStorage {
        with_gpu(|gpu| {
            let mut c = pool_alloc(gpu, m * n);

            // Row-major trick: C = A @ B + bias in row-major
            //   ↔ C^T = B^T @ A^T + bias in col-major (cuBLASLt perspective)
            // cuBLASLt A = our B, cuBLASLt B = our A, m↔n swapped.
            // Bias is [N] in our layout = [cuBLAS_m] in col-major → EPILOGUE_BIAS.
            let (transa, lda) = if trans_b {
                (true, k as i64)
            } else {
                (false, n as i64)
            };
            let (transb, ldb) = if trans_a {
                (true, m as i64)
            } else {
                (false, k as i64)
            };

            unsafe {
                <CudaBlasLT as CublasLtMatmul<f32>>::matmul(
                    &gpu.blas_lt,
                    LtMatmulConfig {
                        transa,
                        transb,
                        transc: false,
                        m: n as u64,
                        n: m as u64,
                        k: k as u64,
                        alpha: 1.0,
                        lda,
                        ldb,
                        beta: 0.0,
                        ldc: n as i64,
                        stride_a: None,
                        stride_b: None,
                        stride_c: None,
                        stride_bias: None,
                        batch_size: None,
                    },
                    b.inner(),   // cuBLAS A = our B
                    a.inner(),   // cuBLAS B = our A
                    &mut c,
                    Some(bias.inner()),  // bias [N] = [cuBLAS_m]
                    None,                // no activation epilogue
                )
                .expect("cuBLASLt matmul_bias failed");
            }

            GpuStorage::new(c, m * n)
        })
    }

    fn add(
        a: &GpuStorage,
        b: &GpuStorage,
        a_shape: &Shape,
        b_shape: &Shape,
        out_shape: &Shape,
    ) -> GpuStorage {

        // Fast path: same shape → element-wise add (copy a, then add b in-place)
        if a_shape == b_shape {
            let n = a.len;
            return with_gpu(|gpu| {
                let mut out = pool_alloc(gpu, n);
                unsafe {
                    gpu.stream.memcpy_dtod(a.inner(), &mut out).unwrap();
                    gpu.stream
                        .launch_builder(&gpu.kernels.add_inplace_kernel)
                        .arg(&mut out)
                        .arg(b.inner())
                        .arg(&n)
                        .launch(LaunchConfig {
                            grid_dim: (grid_for(n, BLOCK), 1, 1),
                            block_dim: (BLOCK, 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .unwrap();
                }
                GpuStorage::new(out, n )
            });
        }

        // Common broadcast: [rows, cols] + [1, cols] or [cols]
        let out_dims = out_shape.dims();
        let b_dims = b_shape.dims();
        if out_dims.len() == 2
            && (b_dims == [1, out_dims[1]] || b_dims == [out_dims[1]])
        {
            let rows = out_dims[0];
            let cols = out_dims[1];
            let n = rows * cols;
            return with_gpu(|gpu| {
                let mut out = pool_alloc(gpu, n);
                unsafe {
                    gpu.stream
                        .launch_builder(&gpu.kernels.add_broadcast_2d)
                        .arg(&mut out)
                        .arg(a.inner())
                        .arg(b.inner())
                        .arg(&rows)
                        .arg(&cols)
                        .launch(LaunchConfig {
                            grid_dim: (grid_for(n, BLOCK), 1, 1),
                            block_dim: (BLOCK, 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .unwrap();
                }
                GpuStorage::new(out, n )
            });
        }

        // General fallback: transfer to CPU, compute, transfer back
        let a_vec = Self::to_vec(a);
        let b_vec = Self::to_vec(b);
        let result =
            crate::backend::cpu::CpuBackend::add(&a_vec, &b_vec, a_shape, b_shape, out_shape);
        Self::from_vec(result, out_shape)
    }

    fn softmax(input: &GpuStorage, shape: &Shape) -> GpuStorage {
        let dims = shape.dims();
        let cols = *dims.last().unwrap();
        let rows = input.len / cols;
        with_gpu(|gpu| {
            let mut out = pool_alloc(gpu, input.len);
            let threads = ROW_BLOCK.min(cols.next_power_of_two() as u32);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.softmax_fwd)
                    .arg(&mut out)
                    .arg(input.inner())
                    .arg(&rows)
                    .arg(&cols)
                    .launch(LaunchConfig {
                        grid_dim: (rows as u32, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: threads * 4,
                    })
                    .unwrap();
            }
            GpuStorage::new(out, input.len)
        })
    }

    fn layernorm(
        input: &GpuStorage,
        gamma: &GpuStorage,
        beta: &GpuStorage,
        shape: &Shape,
        eps: f32,
    ) -> (GpuStorage, GpuStorage, GpuStorage) {
        let dims = shape.dims();
        let d = *dims.last().unwrap();
        let rows = input.len / d;
        with_gpu(|gpu| {
            let mut out = pool_alloc(gpu, input.len);
            let mut mean_out = pool_alloc(gpu, rows);
            let mut rstd_out = pool_alloc(gpu, rows);
            let threads = ROW_BLOCK.min(d.next_power_of_two() as u32);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.layernorm_fwd)
                    .arg(&mut out)
                    .arg(&mut mean_out)
                    .arg(&mut rstd_out)
                    .arg(input.inner())
                    .arg(gamma.inner())
                    .arg(beta.inner())
                    .arg(&rows)
                    .arg(&d)
                    .arg(&eps)
                    .launch(LaunchConfig {
                        grid_dim: (rows as u32, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: threads * 4,
                    })
                    .unwrap();
            }
            (
                GpuStorage::new(out, input.len),
                GpuStorage::new(mean_out, rows),
                GpuStorage::new(rstd_out, rows),
            )
        })
    }

    fn gelu(input: &GpuStorage) -> GpuStorage {
        let n = input.len;
        with_gpu(|gpu| {
            let mut out = pool_alloc(gpu, n);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.gelu_fwd)
                    .arg(&mut out)
                    .arg(input.inner())
                    .arg(&n)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(n, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            GpuStorage::new(out, n )
        })
    }

    fn cross_entropy(
        logits: &GpuStorage,
        targets: &[usize],
        batch: usize,
        vocab: usize,
    ) -> GpuStorage {
        with_gpu(|gpu| {
            let mut loss = pool_alloc_zeros(gpu, 1);
            let targets_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();
            let targets_dev = gpu.stream.clone_htod(&targets_u32).unwrap();
            let threads = ROW_BLOCK.min(vocab.next_power_of_two() as u32);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.cross_entropy_fwd)
                    .arg(&mut loss)
                    .arg(logits.inner())
                    .arg(&targets_dev)
                    .arg(&batch)
                    .arg(&vocab)
                    .launch(LaunchConfig {
                        grid_dim: (batch as u32, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: threads * 4,
                    })
                    .unwrap();
            }
            GpuStorage::new(loss, 1)
        })
    }

    fn cross_entropy_fwd_bwd(
        logits: &GpuStorage,
        targets: &[usize],
        batch: usize,
        vocab: usize,
    ) -> (GpuStorage, GpuStorage) {
        let n = batch * vocab;
        with_gpu(|gpu| {
            let mut loss = pool_alloc_zeros(gpu, 1);
            let mut d_logits = pool_alloc(gpu, n);
            let targets_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();
            let targets_dev = gpu.stream.clone_htod(&targets_u32).unwrap();
            let threads = ROW_BLOCK.min(vocab.next_power_of_two() as u32);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.cross_entropy_fwd_bwd)
                    .arg(&mut loss)
                    .arg(&mut d_logits)
                    .arg(logits.inner())
                    .arg(&targets_dev)
                    .arg(&batch)
                    .arg(&vocab)
                    .launch(LaunchConfig {
                        grid_dim: (batch as u32, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: threads * 4,
                    })
                    .unwrap();
            }
            (GpuStorage::new(loss, 1), GpuStorage::new(d_logits, n))
        })
    }

    fn embedding(
        weight: &GpuStorage,
        indices: &[usize],
        _vocab: usize,
        dim: usize,
    ) -> GpuStorage {
        let n_indices = indices.len();
        let total = n_indices * dim;
        with_gpu(|gpu| {
            let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
            let indices_dev = gpu.stream.clone_htod(&indices_u32).unwrap();
            let mut out = pool_alloc(gpu, total);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.embedding_fwd)
                    .arg(&mut out)
                    .arg(weight.inner())
                    .arg(&indices_dev)
                    .arg(&n_indices)
                    .arg(&dim)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(total, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            GpuStorage::new(out, total)
        })
    }

    fn sum(input: &GpuStorage) -> GpuStorage {
        with_gpu(|gpu| {
            let mut out = pool_alloc_zeros(gpu, 1);
            let n = input.len;
            let threads = ROW_BLOCK.min(n.next_power_of_two() as u32);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.reduce_sum)
                    .arg(&mut out)
                    .arg(input.inner())
                    .arg(&n)
                    .launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: threads * 4,
                    })
                    .unwrap();
            }
            GpuStorage::new(out, 1)
        })
    }

    // ---- Backward ops ----
    // Note: matmul_backward, add_backward dispatch through the forward ops which already
    // have bf16 dispatch, so no extra gating needed here.

    fn matmul_backward(
        d_out: &GpuStorage,
        a: &GpuStorage,
        b: &GpuStorage,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> (GpuStorage, GpuStorage) {
        match (trans_a, trans_b) {
            (false, false) => {
                let d_a = Self::matmul(d_out, b, m, n, k, false, true);
                let d_b = Self::matmul(a, d_out, k, m, n, true, false);
                (d_a, d_b)
            }
            (true, false) => {
                let d_a = Self::matmul(b, d_out, k, n, m, false, true);
                let d_b = Self::matmul(a, d_out, k, m, n, false, false);
                (d_a, d_b)
            }
            (false, true) => {
                let d_a = Self::matmul(d_out, b, m, n, k, false, false);
                let d_b = Self::matmul(d_out, a, n, m, k, true, false);
                (d_a, d_b)
            }
            (true, true) => {
                let d_a = Self::matmul(b, d_out, k, n, m, true, true);
                let d_b = Self::matmul(d_out, a, n, m, k, true, true);
                (d_a, d_b)
            }
        }
    }

    fn add_backward(
        d_out: &GpuStorage,
        a_shape: &Shape,
        b_shape: &Shape,
        out_shape: &Shape,
    ) -> (GpuStorage, GpuStorage) {
        // BF16 dispatch for bias reduction path

        // Fast path: [rows, cols] + [1, cols] or [cols] (bias add pattern)
        let out_dims = out_shape.dims();
        let b_dims = b_shape.dims();
        if a_shape == out_shape
            && out_dims.len() == 2
            && (b_dims == [1, out_dims[1]] || b_dims == [out_dims[1]])
        {
            let rows = out_dims[0];
            let cols = out_dims[1];
            let d_a = d_out.clone();
            let d_b = with_gpu(|gpu| {
                let mut out = pool_alloc(gpu, cols);
                let threads = ROW_BLOCK.min(rows.next_power_of_two() as u32);
                unsafe {
                    gpu.stream
                        .launch_builder(&gpu.kernels.reduce_rows)
                        .arg(&mut out)
                        .arg(d_out.inner())
                        .arg(&rows)
                        .arg(&cols)
                        .launch(LaunchConfig {
                            grid_dim: (cols as u32, 1, 1),
                            block_dim: (threads, 1, 1),
                            shared_mem_bytes: threads * 4,
                        })
                        .unwrap();
                }
                GpuStorage::new(out, cols)
            });
            return (d_a, d_b);
        }

        // Same-shape: both grads are just clones of d_out
        if a_shape == b_shape {
            return (d_out.clone(), d_out.clone());
        }

        // General fallback: transfer to CPU, compute, transfer back
        let d_out_vec = Self::to_vec(d_out);
        let (d_a, d_b) =
            crate::backend::cpu::CpuBackend::add_backward(&d_out_vec, a_shape, b_shape, out_shape);
        (
            Self::from_vec(d_a, a_shape),
            Self::from_vec(d_b, b_shape),
        )
    }

    fn softmax_backward(
        d_out: &GpuStorage,
        output: &GpuStorage,
        shape: &Shape,
    ) -> GpuStorage {
        let dims = shape.dims();
        let cols = *dims.last().unwrap();
        let rows = output.len / cols;
        with_gpu(|gpu| {
            let mut dx = pool_alloc(gpu, output.len);
            let threads = ROW_BLOCK.min(cols.next_power_of_two() as u32);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.softmax_bwd)
                    .arg(&mut dx)
                    .arg(d_out.inner())
                    .arg(output.inner())
                    .arg(&rows)
                    .arg(&cols)
                    .launch(LaunchConfig {
                        grid_dim: (rows as u32, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: threads * 4,
                    })
                    .unwrap();
            }
            GpuStorage::new(dx, output.len)
        })
    }

    fn layernorm_backward(
        d_out: &GpuStorage,
        input: &GpuStorage,
        gamma: &GpuStorage,
        mean: &GpuStorage,
        rstd: &GpuStorage,
        shape: &Shape,
    ) -> (GpuStorage, GpuStorage, GpuStorage) {
        let dims = shape.dims();
        let d = *dims.last().unwrap();
        let rows = input.len / d;
        with_gpu(|gpu| {
            let mut dx = pool_alloc(gpu, input.len);
            let mut dgamma = pool_alloc_zeros(gpu, d);
            let mut dbeta = pool_alloc_zeros(gpu, d);
            let threads = ROW_BLOCK.min(d.next_power_of_two() as u32);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.layernorm_bwd)
                    .arg(&mut dx)
                    .arg(&mut dgamma)
                    .arg(&mut dbeta)
                    .arg(d_out.inner())
                    .arg(input.inner())
                    .arg(gamma.inner())
                    .arg(mean.inner())
                    .arg(rstd.inner())
                    .arg(&rows)
                    .arg(&d)
                    .launch(LaunchConfig {
                        grid_dim: (rows as u32, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: threads * 4 * 2, // two shared arrays
                    })
                    .unwrap();
            }
            (
                GpuStorage::new(dx, input.len),
                GpuStorage::new(dgamma, d),
                GpuStorage::new(dbeta, d),
            )
        })
    }

    fn gelu_backward(d_out: &GpuStorage, input: &GpuStorage) -> GpuStorage {
        let n = input.len;
        with_gpu(|gpu| {
            let mut dx = pool_alloc(gpu, n);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.gelu_bwd)
                    .arg(&mut dx)
                    .arg(d_out.inner())
                    .arg(input.inner())
                    .arg(&n)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(n, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            GpuStorage::new(dx, n )
        })
    }

    fn cross_entropy_backward(
        logits: &GpuStorage,
        targets: &[usize],
        batch: usize,
        vocab: usize,
        d_out_scalar: &GpuStorage,
    ) -> GpuStorage {
        let n = batch * vocab;
        with_gpu(|gpu| {
            let targets_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();
            let targets_dev = gpu.stream.clone_htod(&targets_u32).unwrap();
            let mut d_logits = pool_alloc(gpu, n);
            let threads = ROW_BLOCK.min(vocab.next_power_of_two() as u32);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.cross_entropy_bwd)
                    .arg(&mut d_logits)
                    .arg(logits.inner())
                    .arg(&targets_dev)
                    .arg(&batch)
                    .arg(&vocab)
                    .arg(d_out_scalar.inner())
                    .launch(LaunchConfig {
                        grid_dim: (batch as u32, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: threads * 4,
                    })
                    .unwrap();
            }
            GpuStorage::new(d_logits, n)
        })
    }

    fn embedding_backward(
        d_out: &GpuStorage,
        indices: &[usize],
        vocab: usize,
        dim: usize,
    ) -> GpuStorage {
        let n_indices = indices.len();
        let total = n_indices * dim;
        let weight_size = vocab * dim;
        with_gpu(|gpu| {
            let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
            let indices_dev = gpu.stream.clone_htod(&indices_u32).unwrap();
            let mut d_weight = pool_alloc_zeros(gpu, weight_size);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.embedding_bwd)
                    .arg(&mut d_weight)
                    .arg(d_out.inner())
                    .arg(&indices_dev)
                    .arg(&n_indices)
                    .arg(&dim)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(total, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            GpuStorage::new(d_weight, weight_size)
        })
    }

    fn mul_elementwise(a: &GpuStorage, b: &GpuStorage) -> GpuStorage {
        let n = a.len;
        with_gpu(|gpu| {
            let mut out = pool_alloc(gpu, n);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.mul_elementwise)
                    .arg(&mut out)
                    .arg(a.inner())
                    .arg(b.inner())
                    .arg(&n)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(n, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            GpuStorage::new(out, n )
        })
    }

    fn scale(a: &GpuStorage, scalar: f32) -> GpuStorage {
        let n = a.len;
        with_gpu(|gpu| {
            let mut out = pool_alloc(gpu, n);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.scale_kernel)
                    .arg(&mut out)
                    .arg(a.inner())
                    .arg(&scalar)
                    .arg(&n)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(n, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            GpuStorage::new(out, n )
        })
    }

    fn add_inplace(a: &mut GpuStorage, b: &GpuStorage) {
        let n = a.len;
        with_gpu(|gpu| {
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.add_inplace_kernel)
                    .arg(a.inner_mut())
                    .arg(b.inner())
                    .arg(&n)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(n, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
        });
    }

    fn broadcast_scalar(scalar: &GpuStorage, n: usize) -> GpuStorage {
        with_gpu(|gpu| {
            let mut out = pool_alloc(gpu, n);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.broadcast_scalar)
                    .arg(&mut out)
                    .arg(scalar.inner())
                    .arg(&n)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(n, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            GpuStorage::new(out, n)
        })
    }

    fn norm(storage: &GpuStorage) -> f32 {
        with_gpu(|gpu| {
            let mut out = pool_alloc_zeros(gpu, 1);
            let n = storage.len;
            let threads = ROW_BLOCK.min(n.next_power_of_two() as u32);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.dot_self)
                    .arg(&mut out)
                    .arg(storage.inner())
                    .arg(&n)
                    .launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: threads * 4,
                    })
                    .unwrap();
            }
            let result = gpu.stream.clone_dtoh(&out).unwrap();
            (result[0] as f64).sqrt() as f32
        })
    }

    fn scale_inplace(a: &mut GpuStorage, scalar: f32) {
        let n = a.len;
        with_gpu(|gpu| {
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.scale_inplace_kernel)
                    .arg(a.inner_mut())
                    .arg(&scalar)
                    .arg(&n)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(n, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
        });
    }

    fn concat_rows(
        a: &GpuStorage,
        b: &GpuStorage,
        _a_rows: usize,
        _b_rows: usize,
        _cols: usize,
    ) -> GpuStorage {
        let total = a.len + b.len;
        with_gpu(|gpu| {
            let mut out = pool_alloc(gpu, total);
            gpu.stream
                .memcpy_dtod(a.inner(), &mut out.slice_mut(0..a.len))
                .unwrap();
            gpu.stream
                .memcpy_dtod(b.inner(), &mut out.slice_mut(a.len..total))
                .unwrap();
            GpuStorage::new(out, total)
        })
    }

    #[cfg(feature = "bf16")]
    fn invalidate_bf16_cache(storage: &GpuStorage) {
        storage.invalidate_bf16_shadow();
    }

    fn multi_norm_squared(storages: &[&GpuStorage]) -> Vec<f64> {
        use cudarc::driver::DevicePtr;

        if storages.is_empty() {
            return Vec::new();
        }
        let n_tensors = storages.len();
        with_gpu(|gpu| {
            // Collect raw device pointers and lengths
            let mut ptrs_host: Vec<u64> = Vec::with_capacity(n_tensors);
            let mut guards = Vec::with_capacity(n_tensors);
            let mut lens_host: Vec<u64> = Vec::with_capacity(n_tensors);

            for s in storages {
                let (ptr, guard) = s.inner().device_ptr(&gpu.stream);
                ptrs_host.push(ptr);
                guards.push(guard);
                lens_host.push(s.len as u64);
            }

            // Upload pointer + length arrays to GPU
            let ptrs_dev = gpu.stream.clone_htod(&ptrs_host).unwrap();
            let lens_dev = gpu.stream.clone_htod(&lens_host).unwrap();
            let mut out_dev = pool_alloc_zeros(gpu, n_tensors);

            // Use multiple blocks per tensor for large tensors.
            // Target: each thread processes ~1024 elements max for good throughput.
            let threads = ROW_BLOCK; // 256 threads per block
            let max_len = storages.iter().map(|s| s.len).max().unwrap_or(1);
            let elems_per_block = threads as usize * 1024;
            let blocks_per_tensor = ((max_len + elems_per_block - 1) / elems_per_block).max(1) as u32;

            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.multi_dot_self)
                    .arg(&mut out_dev)
                    .arg(&ptrs_dev)
                    .arg(&lens_dev)
                    .arg(&n_tensors)
                    .launch(LaunchConfig {
                        grid_dim: (n_tensors as u32, blocks_per_tensor, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: threads * 4,
                    })
                    .unwrap();
            }

            // Single D2H transfer — one sync point instead of 125
            let results = gpu.stream.clone_dtoh(&out_dev).unwrap();
            drop(guards); // keep SyncOnDrop alive until after kernel completes
            results.into_iter().map(|x| f64::from(x)).collect()
        })
    }

    fn adamw_step(
        param: &mut GpuStorage,
        grad: &GpuStorage,
        m: &mut GpuStorage,
        v: &mut GpuStorage,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: u32,
    ) {
        let n = param.len;
        let bc1 = 1.0f32 - beta1.powi(step as i32);
        let bc2 = 1.0f32 - beta2.powi(step as i32);
        with_gpu(|gpu| {
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.adamw_step)
                    .arg(param.inner_mut())
                    .arg(grad.inner())
                    .arg(m.inner_mut())
                    .arg(v.inner_mut())
                    .arg(&lr)
                    .arg(&beta1)
                    .arg(&beta2)
                    .arg(&eps)
                    .arg(&weight_decay)
                    .arg(&bc1)
                    .arg(&bc2)
                    .arg(&n)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(n, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
        });
    }


    fn gather_columns(
        storage: &GpuStorage,
        rows: usize,
        total_cols: usize,
        col_start: usize,
        col_count: usize,
    ) -> GpuStorage {
        let n = rows * col_count;
        with_gpu(|gpu| {
            let mut out = pool_alloc(gpu, n);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.gather_columns)
                    .arg(&mut out)
                    .arg(storage.inner())
                    .arg(&rows)
                    .arg(&total_cols)
                    .arg(&col_start)
                    .arg(&col_count)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(n, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            GpuStorage::new(out, n )
        })
    }

    fn scatter_columns(
        dst: &mut GpuStorage,
        src: &GpuStorage,
        rows: usize,
        total_cols: usize,
        col_start: usize,
        col_count: usize,
    ) {
        let n = rows * col_count;
        with_gpu(|gpu| {
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.scatter_columns)
                    .arg(dst.inner_mut())
                    .arg(src.inner())
                    .arg(&rows)
                    .arg(&total_cols)
                    .arg(&col_start)
                    .arg(&col_count)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(n, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
        });
    }

    fn dropout(
        input: &GpuStorage,
        n: usize,
        p: f32,
        seed: u64,
    ) -> (GpuStorage, GpuStorage) {
        let scale = 1.0 / (1.0 - p);
        with_gpu(|gpu| {
            let mut out = pool_alloc(gpu, n);
            let mut mask = pool_alloc(gpu, n);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.dropout_fwd)
                    .arg(&mut out)
                    .arg(&mut mask)
                    .arg(input.inner())
                    .arg(&p)
                    .arg(&scale)
                    .arg(&seed)
                    .arg(&n)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(n, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            (GpuStorage::new(out, n), GpuStorage::new(mask, n))
        })
    }

    fn gather_rows(
        storage: &GpuStorage,
        _total_rows: usize,
        cols: usize,
        row_start: usize,
        row_count: usize,
    ) -> GpuStorage {
        let n = row_count * cols;
        let start = row_start * cols;
        with_gpu(|gpu| {
            let mut out = pool_alloc(gpu, n);
            gpu.stream
                .memcpy_dtod(&storage.inner().slice(start..start + n), &mut out)
                .unwrap();
            GpuStorage::new(out, n)
        })
    }

    fn scatter_rows(
        dst: &mut GpuStorage,
        src: &GpuStorage,
        _total_rows: usize,
        cols: usize,
        row_start: usize,
        row_count: usize,
    ) {
        let start = row_start * cols;
        let n = row_count * cols;
        with_gpu(|gpu| {
            gpu.stream
                .memcpy_dtod(src.inner(), &mut dst.inner_mut().slice_mut(start..start + n))
                .unwrap();
        });
    }

    fn apply_causal_mask_and_scale(
        scores: &mut GpuStorage,
        seq_len: usize,
        scale: f32,
        mask_value: f32,
    ) {
        let n = seq_len * seq_len;
        with_gpu(|gpu| {
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.causal_mask_scale)
                    .arg(scores.inner_mut())
                    .arg(&seq_len)
                    .arg(&scale)
                    .arg(&mask_value)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(n, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
        });
    }

    fn reshape_for_heads(
        storage: &GpuStorage,
        batch: usize,
        seq: usize,
        n_heads: usize,
        d_head: usize,
    ) -> GpuStorage {
        let total = batch * n_heads * seq * d_head;
        with_gpu(|gpu| {
            let mut out = pool_alloc(gpu, total);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.reshape_bsh_to_bnsh)
                    .arg(&mut out)
                    .arg(storage.inner())
                    .arg(&batch)
                    .arg(&seq)
                    .arg(&n_heads)
                    .arg(&d_head)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(total, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            GpuStorage::new(out, total)
        })
    }

    fn reshape_from_heads(
        storage: &GpuStorage,
        batch: usize,
        seq: usize,
        n_heads: usize,
        d_head: usize,
    ) -> GpuStorage {
        let total = batch * seq * n_heads * d_head;
        with_gpu(|gpu| {
            let mut out = pool_alloc(gpu, total);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.reshape_bnsh_to_bsh)
                    .arg(&mut out)
                    .arg(storage.inner())
                    .arg(&batch)
                    .arg(&seq)
                    .arg(&n_heads)
                    .arg(&d_head)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(total, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            GpuStorage::new(out, total)
        })
    }

    fn matmul_strided_batched(
        a: &GpuStorage,
        b: &GpuStorage,
        batch_count: usize,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> GpuStorage {
        #[cfg(feature = "bf16")]
        if bf16_enabled() {
            return bf16_ops::matmul_strided_batched_bf16(
                a, b, batch_count, m, k, n, trans_a, trans_b,
            );
        }
        let total = batch_count * m * n;
        with_gpu_mut(|gpu| {
            let mut c = pool_alloc(gpu, total);

            // Row-major trick: same as matmul but with strides.
            // C = A @ B in row-major ↔ C^T = B^T @ A^T in col-major
            let (transa, lda) = if trans_b {
                (cublasOperation_t::CUBLAS_OP_T, k as i32)
            } else {
                (cublasOperation_t::CUBLAS_OP_N, n as i32)
            };
            let (transb, ldb) = if trans_a {
                (cublasOperation_t::CUBLAS_OP_T, m as i32)
            } else {
                (cublasOperation_t::CUBLAS_OP_N, k as i32)
            };

            let stride_a = if trans_a { k * m } else { m * k };
            let stride_b = if trans_b { n * k } else { k * n };
            let stride_c = m * n;

            unsafe {
                gpu.blas
                    .gemm_strided_batched(
                        cudarc::cublas::StridedBatchedConfig {
                            gemm: cudarc::cublas::GemmConfig {
                                transa,
                                transb,
                                m: n as i32,
                                n: m as i32,
                                k: k as i32,
                                alpha: 1.0f32,
                                lda,
                                ldb,
                                beta: 0.0f32,
                                ldc: n as i32,
                            },
                            batch_size: batch_count as i32,
                            stride_a: stride_b as i64,  // swapped: cuBLAS A = our B
                            stride_b: stride_a as i64,  // swapped: cuBLAS B = our A
                            stride_c: stride_c as i64,
                        },
                        b.inner(),  // cuBLAS A = our B (row-major trick)
                        a.inner(),  // cuBLAS B = our A
                        &mut c,
                    )
                    .expect("cuBLAS sgemm_strided_batched failed");
            }

            GpuStorage::new(c, total)
        })
    }

    fn residual_add_layernorm(
        residual: &GpuStorage,
        sublayer: &GpuStorage,
        gamma: &GpuStorage,
        beta: &GpuStorage,
        shape: &Shape,
        eps: f32,
    ) -> (GpuStorage, GpuStorage, GpuStorage, GpuStorage) {
        let dims = shape.dims();
        let d = *dims.last().unwrap();
        let rows = residual.len / d;
        with_gpu(|gpu| {
            let mut out_norm = pool_alloc(gpu, residual.len);
            let mut out_sum = pool_alloc(gpu, residual.len);
            let mut mean_out = pool_alloc(gpu, rows);
            let mut rstd_out = pool_alloc(gpu, rows);
            let threads = ROW_BLOCK.min(d.next_power_of_two() as u32);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.residual_add_layernorm_fwd)
                    .arg(&mut out_norm)
                    .arg(&mut out_sum)
                    .arg(&mut mean_out)
                    .arg(&mut rstd_out)
                    .arg(residual.inner())
                    .arg(sublayer.inner())
                    .arg(gamma.inner())
                    .arg(beta.inner())
                    .arg(&rows)
                    .arg(&d)
                    .arg(&eps)
                    .launch(LaunchConfig {
                        grid_dim: (rows as u32, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: threads * 4,
                    })
                    .unwrap();
            }
            (
                GpuStorage::new(out_sum, residual.len),
                GpuStorage::new(out_norm, residual.len),
                GpuStorage::new(mean_out, rows),
                GpuStorage::new(rstd_out, rows),
            )
        })
    }

    fn residual_add_layernorm_backward(
        d_out: &GpuStorage,
        residual: &GpuStorage,
        sublayer: &GpuStorage,
        gamma: &GpuStorage,
        mean: &GpuStorage,
        rstd: &GpuStorage,
        shape: &Shape,
    ) -> (GpuStorage, GpuStorage, GpuStorage) {
        let dims = shape.dims();
        let d = *dims.last().unwrap();
        let rows = residual.len / d;
        with_gpu(|gpu| {
            let mut dx = pool_alloc(gpu, residual.len);
            let mut dgamma = pool_alloc_zeros(gpu, d);
            let mut dbeta = pool_alloc_zeros(gpu, d);
            let threads = ROW_BLOCK.min(d.next_power_of_two() as u32);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.residual_add_layernorm_bwd)
                    .arg(&mut dx)
                    .arg(&mut dgamma)
                    .arg(&mut dbeta)
                    .arg(d_out.inner())
                    .arg(residual.inner())
                    .arg(sublayer.inner())
                    .arg(gamma.inner())
                    .arg(mean.inner())
                    .arg(rstd.inner())
                    .arg(&rows)
                    .arg(&d)
                    .launch(LaunchConfig {
                        grid_dim: (rows as u32, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: threads * 4 * 2, // two shared arrays
                    })
                    .unwrap();
            }
            (
                GpuStorage::new(dx, residual.len),
                GpuStorage::new(dgamma, d),
                GpuStorage::new(dbeta, d),
            )
        })
    }

    fn apply_batched_causal_mask_and_scale(
        scores: &mut GpuStorage,
        num_matrices: usize,
        seq_len: usize,
        scale: f32,
        mask_value: f32,
    ) {
        let total = num_matrices * seq_len * seq_len;
        with_gpu(|gpu| {
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.batched_causal_mask_scale)
                    .arg(scores.inner_mut())
                    .arg(&num_matrices)
                    .arg(&seq_len)
                    .arg(&scale)
                    .arg(&mask_value)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(total, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
        });
    }

    fn split_qkv_to_heads(
        qkv: &GpuStorage,
        batch: usize,
        seq: usize,
        n_heads: usize,
        d_head: usize,
    ) -> (GpuStorage, GpuStorage, GpuStorage) {
        let per_output = batch * n_heads * seq * d_head;
        with_gpu(|gpu| {
            let mut q = pool_alloc(gpu, per_output);
            let mut k = pool_alloc(gpu, per_output);
            let mut v = pool_alloc(gpu, per_output);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.split_qkv_to_heads)
                    .arg(&mut q)
                    .arg(&mut k)
                    .arg(&mut v)
                    .arg(qkv.inner())
                    .arg(&batch)
                    .arg(&seq)
                    .arg(&n_heads)
                    .arg(&d_head)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(per_output, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            (
                GpuStorage::new(q, per_output),
                GpuStorage::new(k, per_output),
                GpuStorage::new(v, per_output),
            )
        })
    }

    fn merge_heads_to_qkv(
        dq: &GpuStorage,
        dk: &GpuStorage,
        dv: &GpuStorage,
        batch: usize,
        seq: usize,
        n_heads: usize,
        d_head: usize,
    ) -> GpuStorage {
        let d_model = n_heads * d_head;
        let total_tokens = batch * seq;
        let per_head = batch * n_heads * seq * d_head;
        with_gpu(|gpu| {
            let mut d_qkv = pool_alloc_zeros(gpu, total_tokens * 3 * d_model);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.merge_heads_to_qkv)
                    .arg(&mut d_qkv)
                    .arg(dq.inner())
                    .arg(dk.inner())
                    .arg(dv.inner())
                    .arg(&batch)
                    .arg(&seq)
                    .arg(&n_heads)
                    .arg(&d_head)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(per_head, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            GpuStorage::new(d_qkv, total_tokens * 3 * d_model)
        })
    }

    fn fused_bias_gelu(
        input: &GpuStorage,
        bias: &GpuStorage,
        rows: usize,
        cols: usize,
    ) -> GpuStorage {
        let n = rows * cols;
        with_gpu(|gpu| {
            let mut out = pool_alloc(gpu, n);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.fused_bias_gelu_fwd)
                    .arg(&mut out)
                    .arg(input.inner())
                    .arg(bias.inner())
                    .arg(&rows)
                    .arg(&cols)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(n, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            GpuStorage::new(out, n)
        })
    }

    fn fused_bias_gelu_backward(
        d_out: &GpuStorage,
        input: &GpuStorage,
        bias: &GpuStorage,
        rows: usize,
        cols: usize,
    ) -> GpuStorage {
        let n = rows * cols;
        with_gpu(|gpu| {
            let mut dx = pool_alloc(gpu, n);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.fused_bias_gelu_bwd)
                    .arg(&mut dx)
                    .arg(d_out.inner())
                    .arg(input.inner())
                    .arg(bias.inner())
                    .arg(&rows)
                    .arg(&cols)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(n, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            GpuStorage::new(dx, n)
        })
    }

    fn fused_bias_dropout_residual(
        matmul_out: &GpuStorage,
        bias: &GpuStorage,
        residual: &GpuStorage,
        rows: usize,
        cols: usize,
        p: f32,
        seed: u64,
    ) -> (GpuStorage, GpuStorage) {
        let n = rows * cols;
        let scale = 1.0 / (1.0 - p);
        with_gpu(|gpu| {
            let mut out = pool_alloc(gpu, n);
            let mut mask = pool_alloc(gpu, n);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.fused_bias_dropout_residual_fwd)
                    .arg(&mut out)
                    .arg(&mut mask)
                    .arg(matmul_out.inner())
                    .arg(bias.inner())
                    .arg(residual.inner())
                    .arg(&rows)
                    .arg(&cols)
                    .arg(&p)
                    .arg(&scale)
                    .arg(&seed)
                    .launch(LaunchConfig {
                        grid_dim: (grid_for(n, BLOCK), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            (GpuStorage::new(out, n), GpuStorage::new(mask, n))
        })
    }

    fn flash_attention_forward(
        q: &GpuStorage,
        k: &GpuStorage,
        v: &GpuStorage,
        batch_heads: usize,
        seq_len: usize,
        d_head: usize,
        scale: f32,
        is_causal: bool,
    ) -> (GpuStorage, GpuStorage) {
        #[cfg(feature = "bf16")]
        if bf16_enabled() {
            return bf16_ops::flash_attention_forward_bf16(
                q, k, v, batch_heads, seq_len, d_head, scale, is_causal,
            );
        }
        let total_output = batch_heads * seq_len * d_head;
        let total_lse = batch_heads * seq_len;
        let is_causal_i32: i32 = if is_causal { 1 } else { 0 };
        let s_i32 = seq_len as i32;
        let d_i32 = d_head as i32;
        let br: u32 = 32;  // FA_BR — warp-cooperative (each warp = 1 Q row)
        let bc: u32 = 32;  // FA_BC

        // Shared memory: K_tile[BC * D] + V_tile[BC * D]
        let smem_bytes = 2 * bc * (d_head as u32) * 4;

        with_gpu(|gpu| {
            let mut output = pool_alloc(gpu, total_output);
            let mut lse = pool_alloc(gpu, total_lse);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.flash_attention_fwd)
                    .arg(&mut output)
                    .arg(&mut lse)
                    .arg(q.inner())
                    .arg(k.inner())
                    .arg(v.inner())
                    .arg(&s_i32)
                    .arg(&d_i32)
                    .arg(&scale)
                    .arg(&is_causal_i32)
                    .launch(LaunchConfig {
                        grid_dim: (
                            (seq_len as u32).div_ceil(br),
                            batch_heads as u32,
                            1,
                        ),
                        block_dim: (br, 32, 1),
                        shared_mem_bytes: smem_bytes,
                    })
                    .unwrap();
            }
            (GpuStorage::new(output, total_output), GpuStorage::new(lse, total_lse))
        })
    }

    fn flash_attention_backward(
        d_out: &GpuStorage,
        q: &GpuStorage,
        k: &GpuStorage,
        v: &GpuStorage,
        output: &GpuStorage,
        lse: &GpuStorage,
        batch_heads: usize,
        seq_len: usize,
        d_head: usize,
        scale: f32,
        is_causal: bool,
    ) -> (GpuStorage, GpuStorage, GpuStorage) {
        #[cfg(feature = "bf16")]
        if bf16_enabled() {
            return bf16_ops::flash_attention_backward_bf16(
                d_out, q, k, v, output, lse, batch_heads, seq_len, d_head, scale, is_causal,
            );
        }
        let total = batch_heads * seq_len * d_head;
        let is_causal_i32: i32 = if is_causal { 1 } else { 0 };
        let s_i32 = seq_len as i32;
        let d_i32 = d_head as i32;
        let bwd_bc: u32 = 32;  // FA_BWD_BC — warp-cooperative KV-tile-centric

        // Shared memory: K_tile[BC * D] + V_tile[BC * D]
        let smem_bytes = 2 * bwd_bc * (d_head as u32) * 4;

        with_gpu(|gpu| {
            let mut dq = pool_alloc_zeros(gpu, total);  // atomicAdd target
            let mut dk = pool_alloc(gpu, total);
            let mut dv = pool_alloc(gpu, total);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.flash_attention_bwd)
                    .arg(&mut dq)
                    .arg(&mut dk)
                    .arg(&mut dv)
                    .arg(d_out.inner())
                    .arg(q.inner())
                    .arg(k.inner())
                    .arg(v.inner())
                    .arg(output.inner())
                    .arg(lse.inner())
                    .arg(&s_i32)
                    .arg(&d_i32)
                    .arg(&scale)
                    .arg(&is_causal_i32)
                    .launch(LaunchConfig {
                        grid_dim: (
                            (seq_len as u32).div_ceil(bwd_bc),
                            batch_heads as u32,
                            1,
                        ),
                        block_dim: (bwd_bc, 32, 1),
                        shared_mem_bytes: smem_bytes,
                    })
                    .unwrap();
            }
            (
                GpuStorage::new(dq, total),
                GpuStorage::new(dk, total),
                GpuStorage::new(dv, total),
            )
        })
    }

    fn fused_mul_reduce_rows(
        a: &GpuStorage,
        b: &GpuStorage,
        rows: usize,
        cols: usize,
    ) -> GpuStorage {
        with_gpu(|gpu| {
            let mut out = pool_alloc(gpu, cols);
            let threads = ROW_BLOCK.min(rows.next_power_of_two() as u32);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.fused_mul_reduce_rows)
                    .arg(&mut out)
                    .arg(a.inner())
                    .arg(b.inner())
                    .arg(&rows)
                    .arg(&cols)
                    .launch(LaunchConfig {
                        grid_dim: (cols as u32, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: threads * 4,
                    })
                    .unwrap();
            }
            GpuStorage::new(out, cols)
        })
    }
}

impl CudaBackend {
    /// Synchronize the GPU stream (wait for all queued operations to complete).
    pub fn synchronize() {
        with_gpu(|gpu| {
            gpu.stream.synchronize().expect("CUDA sync failed");
        });
    }
}

// ============================================================
// BF16 helpers and dispatch
// ============================================================

#[cfg(feature = "bf16")]
#[allow(clippy::wildcard_imports, clippy::doc_markdown, clippy::cast_lossless)]
mod bf16_ops {
    use super::*;

    /// Cast a f32 device buffer to bf16.
    pub fn cast_f32_to_bf16(gpu: &GpuCtx, src: &CudaSlice<f32>, n: usize) -> CudaSlice<half::bf16> {
        let mut dst = gpu.stream.alloc_zeros::<half::bf16>(n).unwrap();
        unsafe {
            gpu.stream
                .launch_builder(&gpu.kernels.f32_to_bf16)
                .arg(&mut dst)
                .arg(src)
                .arg(&n)
                .launch(LaunchConfig {
                    grid_dim: (grid_for(n, BLOCK), 1, 1),
                    block_dim: (BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        dst
    }

    /// BF16 matmul via cublasGemmEx: bf16 inputs, f32 accumulation, f32 output.
    ///
    /// Uses bf16 shadow caches on inputs to avoid per-op f32→bf16 casts.
    /// Output is f32 directly (no cast-back kernel needed).
    pub fn matmul_bf16(
        a: &GpuStorage,
        b: &GpuStorage,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> GpuStorage {
        // Get bf16 shadows (cached, or created + cached lazily)
        let a_bf16 = a.ensure_bf16_shadow();
        let b_bf16 = b.ensure_bf16_shadow();

        with_gpu_mut(|gpu| {
            let mut c_f32 = pool_alloc(gpu, m * n);

            // Row-major trick: C = A @ B ↔ C^T = B^T @ A^T (col-major for cuBLAS)
            let (transa, lda) = if trans_b {
                (cublasOperation_t::CUBLAS_OP_T, k as i32)
            } else {
                (cublasOperation_t::CUBLAS_OP_N, n as i32)
            };
            let (transb, ldb) = if trans_a {
                (cublasOperation_t::CUBLAS_OP_T, m as i32)
            } else {
                (cublasOperation_t::CUBLAS_OP_N, k as i32)
            };

            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;

            // Call cublasGemmEx directly: bf16 inputs → f32 output, f32 accumulation
            unsafe {
                use cudarc::cublas::sys::{
                    cublasComputeType_t, cublasGemmAlgo_t, cudaDataType_t,
                };
                use cudarc::driver::{DevicePtr, DevicePtrMut};

                let (a_ptr, _a_rec) = a_bf16.device_ptr(&gpu.stream);
                let (b_ptr, _b_rec) = b_bf16.device_ptr(&gpu.stream);
                let (c_ptr, _c_rec) = c_f32.device_ptr_mut(&gpu.stream);

                cudarc::cublas::result::gemm_ex(
                    *gpu.blas.handle(),
                    transa,
                    transb,
                    n as i32,  // cuBLAS m = our N (row-major trick)
                    m as i32,  // cuBLAS n = our M
                    k as i32,
                    (&alpha) as *const f32 as *const _,
                    b_ptr as *const _,           // cuBLAS A = our B
                    cudaDataType_t::CUDA_R_16BF, // input type: bf16
                    lda,
                    a_ptr as *const _,           // cuBLAS B = our A
                    cudaDataType_t::CUDA_R_16BF, // input type: bf16
                    ldb,
                    (&beta) as *const f32 as *const _,
                    c_ptr as *mut _,
                    cudaDataType_t::CUDA_R_32F,  // output type: f32
                    n as i32,                    // ldc
                    cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
                .expect("cublasGemmEx bf16→f32 failed");
            }

            GpuStorage::new(c_f32, m * n)
        })
    }

    /// BF16 strided batched matmul via `cublasGemmStridedBatchedEx`:
    /// bf16 inputs, f32 accumulation, f32 output.
    pub fn matmul_strided_batched_bf16(
        a: &GpuStorage,
        b: &GpuStorage,
        batch_count: usize,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> GpuStorage {
        let a_bf16 = a.ensure_bf16_shadow();
        let b_bf16 = b.ensure_bf16_shadow();

        let total = batch_count * m * n;
        with_gpu_mut(|gpu| {
            let mut c_f32 = pool_alloc(gpu, total);

            // Row-major trick: C = A @ B ↔ C^T = B^T @ A^T (col-major for cuBLAS)
            let (transa, lda) = if trans_b {
                (cublasOperation_t::CUBLAS_OP_T, k as i32)
            } else {
                (cublasOperation_t::CUBLAS_OP_N, n as i32)
            };
            let (transb, ldb) = if trans_a {
                (cublasOperation_t::CUBLAS_OP_T, m as i32)
            } else {
                (cublasOperation_t::CUBLAS_OP_N, k as i32)
            };

            let stride_a = if trans_a { k * m } else { m * k };
            let stride_b = if trans_b { n * k } else { k * n };
            let stride_c = m * n;

            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;

            unsafe {
                use cudarc::cublas::sys::{
                    cublasComputeType_t, cublasGemmAlgo_t, cudaDataType_t,
                };
                use cudarc::driver::{DevicePtr, DevicePtrMut};

                let (a_ptr, _a_rec) = a_bf16.device_ptr(&gpu.stream);
                let (b_ptr, _b_rec) = b_bf16.device_ptr(&gpu.stream);
                let (c_ptr, _c_rec) = c_f32.device_ptr_mut(&gpu.stream);

                // Row-major trick: cuBLAS A = our B, cuBLAS B = our A
                cudarc::cublas::result::gemm_strided_batched_ex(
                    *gpu.blas.handle(),
                    transa,
                    transb,
                    n as i32,  // cuBLAS m = our N
                    m as i32,  // cuBLAS n = our M
                    k as i32,
                    (&alpha) as *const f32 as *const _,
                    b_ptr as *const _,           // cuBLAS A = our B
                    cudaDataType_t::CUDA_R_16BF,
                    lda,
                    stride_b as i64,             // swapped strides
                    a_ptr as *const _,           // cuBLAS B = our A
                    cudaDataType_t::CUDA_R_16BF,
                    ldb,
                    stride_a as i64,             // swapped strides
                    (&beta) as *const f32 as *const _,
                    c_ptr as *mut _,
                    cudaDataType_t::CUDA_R_32F,
                    n as i32,                    // ldc
                    stride_c as i64,
                    batch_count as i32,
                    cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
                .expect("cublasGemmStridedBatchedEx bf16→f32 failed");
            }

            GpuStorage::new(c_f32, total)
        })
    }

    /// BF16 flash attention forward: bf16 Q/K/V inputs, f32 output/lse.
    /// Uses bf16 shared memory tiles (BC=64, doubled from FP32's 32).
    pub fn flash_attention_forward_bf16(
        q: &GpuStorage,
        k: &GpuStorage,
        v: &GpuStorage,
        batch_heads: usize,
        seq_len: usize,
        d_head: usize,
        scale: f32,
        is_causal: bool,
    ) -> (GpuStorage, GpuStorage) {
        let q_bf16 = q.ensure_bf16_shadow();
        let k_bf16 = k.ensure_bf16_shadow();
        let v_bf16 = v.ensure_bf16_shadow();

        let total_output = batch_heads * seq_len * d_head;
        let total_lse = batch_heads * seq_len;
        let is_causal_i32: i32 = if is_causal { 1 } else { 0 };
        let s_i32 = seq_len as i32;
        let d_i32 = d_head as i32;
        let br: u32 = 32; // FA_BF16_BR — warp-cooperative
        let bc: u32 = 64; // FA_BF16_BC

        // Shared memory: K_tile[BC*D] + V_tile[BC*D] as bf16 (2 bytes each)
        let smem_bytes = 2 * bc * (d_head as u32) * 2;

        with_gpu(|gpu| {
            let mut output = pool_alloc(gpu, total_output);
            let mut lse = pool_alloc(gpu, total_lse);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.flash_attention_fwd_bf16)
                    .arg(&mut output)
                    .arg(&mut lse)
                    .arg(&q_bf16)
                    .arg(&k_bf16)
                    .arg(&v_bf16)
                    .arg(&s_i32)
                    .arg(&d_i32)
                    .arg(&scale)
                    .arg(&is_causal_i32)
                    .launch(LaunchConfig {
                        grid_dim: (
                            (seq_len as u32).div_ceil(br),
                            batch_heads as u32,
                            1,
                        ),
                        block_dim: (br, 32, 1),
                        shared_mem_bytes: smem_bytes,
                    })
                    .unwrap();
            }
            (
                GpuStorage::new(output, total_output),
                GpuStorage::new(lse, total_lse),
            )
        })
    }

    /// BF16 flash attention backward: bf16 dO/Q/K/V inputs, f32 O/lse, f32 dQ/dK/dV outputs.
    /// Halves inner-loop bandwidth for Q and dO reads (the dominant cost).
    pub fn flash_attention_backward_bf16(
        d_out: &GpuStorage,
        q: &GpuStorage,
        k: &GpuStorage,
        v: &GpuStorage,
        output: &GpuStorage,
        lse: &GpuStorage,
        batch_heads: usize,
        seq_len: usize,
        d_head: usize,
        scale: f32,
        is_causal: bool,
    ) -> (GpuStorage, GpuStorage, GpuStorage) {
        let d_out_bf16 = d_out.ensure_bf16_shadow();
        let q_bf16 = q.ensure_bf16_shadow();
        let k_bf16 = k.ensure_bf16_shadow();
        let v_bf16 = v.ensure_bf16_shadow();

        let total = batch_heads * seq_len * d_head;
        let is_causal_i32: i32 = if is_causal { 1 } else { 0 };
        let s_i32 = seq_len as i32;
        let d_i32 = d_head as i32;
        let bwd_bc: u32 = 32; // FA_BF16_BWD_BC — warp-cooperative

        // Shared memory: K_tile[BC*D] + V_tile[BC*D] as bf16 (2 bytes each)
        let smem_bytes = 2 * bwd_bc * (d_head as u32) * 2;

        with_gpu(|gpu| {
            let mut dq = pool_alloc_zeros(gpu, total);  // atomicAdd target
            let mut dk = pool_alloc(gpu, total);
            let mut dv = pool_alloc(gpu, total);
            unsafe {
                gpu.stream
                    .launch_builder(&gpu.kernels.flash_attention_bwd_bf16)
                    .arg(&mut dq)
                    .arg(&mut dk)
                    .arg(&mut dv)
                    .arg(&d_out_bf16)
                    .arg(&q_bf16)
                    .arg(&k_bf16)
                    .arg(&v_bf16)
                    .arg(output.inner())
                    .arg(lse.inner())
                    .arg(&s_i32)
                    .arg(&d_i32)
                    .arg(&scale)
                    .arg(&is_causal_i32)
                    .launch(LaunchConfig {
                        grid_dim: (
                            (seq_len as u32).div_ceil(bwd_bc),
                            batch_heads as u32,
                            1,
                        ),
                        block_dim: (bwd_bc, 32, 1),
                        shared_mem_bytes: smem_bytes,
                    })
                    .unwrap();
            }
            (
                GpuStorage::new(dq, total),
                GpuStorage::new(dk, total),
                GpuStorage::new(dv, total),
            )
        })
    }
}
