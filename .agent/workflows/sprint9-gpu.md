---
description: Sprint 9 — GPU acceleration with wgpu and optional CUDA for ML
---

# Sprint 9 — GPU Acceleration

> **Goal**: Break through the CPU rendering ceiling with wgpu. Add optional CUDA acceleration for scry-learn's heaviest operations.

// turbo-all

## Session 9A: Architecture & wgpu Setup (1 session)

**Estimated effort:** 1 session (3-4 hours)

### Context Files to Read
- `src/rasterize/skia.rs` — current tiny-skia rasterizer
- `src/rasterize/mod.rs` — rasterizer trait/dispatch
- `src/rasterize/batch.rs` — command batching
- `Cargo.toml` — feature flags

### Step 1: Add wgpu Feature Flag

**Modify:** `Cargo.toml`
```toml
[features]
gpu = ["wgpu", "bytemuck"]

[dependencies]
wgpu = { version = "24", optional = true }
bytemuck = { version = "1", features = ["derive"], optional = true }
```

### Step 2: Create GPU Rasterizer Trait

**New file:** `src/rasterize/gpu.rs`

Define the abstraction:
```rust
pub trait GpuRasterizer {
    fn rasterize_scene(&self, commands: &[DrawCommand], width: u32, height: u32) -> Vec<u8>;
    fn supports_dirty_tiles(&self) -> bool;
}
```

### Step 3: Implement wgpu Backend

**New file:** `src/rasterize/wgpu_backend.rs`

Initial implementation:
1. Initialize wgpu adapter/device/queue
2. Create render pipeline with vertex + fragment shaders
3. Implement shape primitives as GPU draw calls:
   - Circles → instanced quad + SDF in fragment shader
   - Rectangles → simple quads
   - Lines → triangle strips with AA
   - Paths → tessellate to triangles on CPU, render on GPU
4. Anti-aliasing via MSAA (4x or 8x)
5. Output to texture → readback to CPU buffer

### Verification
```bash
cargo check --features gpu
cargo test --features gpu --lib
```

---

## Session 9B: wgpu Shape Rendering (2 sessions)

**Estimated effort:** 2 sessions (6-8 hours)

### Compute Shader Rasterization
- Signed Distance Field (SDF) evaluation in fragment shader
- Handle: circle, rect, rounded_rect, ellipse, line, arc
- Gradient support (linear, radial) as uniform buffer data
- Alpha blending in correct order (painter's algorithm)

### Dirty Tile Integration
- Reuse existing CPU-side dirty tile detection
- Only submit GPU draw calls for dirty tiles
- Tile-level render targets (64×64 tiles)

### Performance Target
- ≥10x throughput vs tiny-skia at 1920×1080
- ≥20x at 3840×2160 (GPU shines at high resolution)

### Verification
```bash
cargo bench --bench engine_throughput --features gpu
# Compare vs non-GPU baseline
cargo bench --bench engine_throughput
```

---

## Session 9C: CUDA for scry-learn (optional, 2 sessions)

**Estimated effort:** 2 sessions (6-8 hours)

### Prerequisites
- NVIDIA GPU with CUDA toolkit installed
- `nvcc` available in PATH

### Step 1: CUDA Feature Flag

**Modify:** `crates/scry-learn/Cargo.toml`
```toml
[features]
cuda = ["cudarc"]

[dependencies]
cudarc = { version = "0.12", optional = true }
```

### Step 2: Matrix Operations

**New directory:** `crates/scry-learn/src/accel/`
**New files:** `mod.rs`, `cuda.rs`

Accelerate:
1. **Matrix multiply** — used in linear regression, PCA, kernel SVM
2. **Distance matrix** — used in KNN, KMeans, DBSCAN
3. **Histogram accumulation** — used in HistGBT binning

Each operation has a CPU fallback:
```rust
pub fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    #[cfg(feature = "cuda")]
    if cuda_available() {
        return cuda::matmul(a, b, m, k, n);
    }
    cpu_matmul(a, b, m, k, n)
}
```

### Step 3: Benchmarks

Compare:
- Linear regression fit: CPU vs CUDA at 1K/10K/100K/1M rows × 10/100/1000 features
- KNN predict: CPU vs CUDA distance computation
- HistGBT: CPU vs CUDA histogram binning

### Verification
```bash
cargo test -p scry-learn --features cuda
cargo bench --bench ml_algorithms -p scry-learn --features cuda
```
