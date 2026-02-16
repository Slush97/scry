---
description: Handoff instructions for the next agent — current state, what's done, what's left
---

# Next Agent Handoff — Full Status & Instructions

> **Last updated**: 2026-02-16 03:49 PST

---

## 1. Project Overview

**scry** is a Rust workspace with five crates:

| Crate | Purpose | Maturity |
|-------|---------|----------|
| `scry-engine` | Low-level 2D rasterizer (shapes, text, gradients, animation) | Stable |
| `scry-chart` | Chart builder library (19 chart types + 3D scatter, 6 themes) | Stable |
| `scry-learn` | Machine learning library (23+ models, search, preprocessing) | Stable |
| `scry-cli` | CLI for chart rendering | Stable |
| `scry-pipe` | Feature engineering compiler (Phase 1 done) | Beta |

---

## 2. What Is Complete (Sprints 1–7 + 8A + 8.5A–D + 8C + 9B + 9C + 9D-1 + 9D-2)

**Everything through Sprint 9D-2 GPU Compute Wiring is done.**

### Sprint 8.5A ✅ 3D Scene Graph & Camera

- `Camera3D` with arcball quaternion rotation (gimbal-lock-free orbit/pan/zoom)
- `Camera3D::orbiting()` — spherical coordinate constructor for interactive loops
- `PerspectiveProjection` with depth sorting (painter's algorithm)
- `Scene3D` with point clouds, axis lines, grid planes, billboard labels
- `Rasterizer3D` trait (architecture boundary for backend swaps)
- `SkiaRasterizer3D` v1 backend (tiny-skia + fontdue)
- `Chart3D::scatter()` builder with `render_to_png()` / `save_png()`

### Sprint 8.5B ✅ Two Rendering Frontends (Terminal Integration)

| Mode | API | Dependency | How to use |
|------|-----|------------|------------|
| **Inline** | `Chart3D::show()` | `widget` feature (crossterm) | Renders directly to stdout via Kitty/Sixel/halfblock. Scoped raw mode for keyboard controls (WASD/arrows rotate, +/- zoom, Q quit). |
| **TUI** | `Chart3DWidget` + `Chart3DState` | `widget` feature (ratatui) | `StatefulWidget` — compose with other ratatui widgets. Same camera controls in your event loop. |

**Bridge method:** `Chart3D::render_to_canvas(w, h)` — converts RGBA output to `PixelCanvas` via `ImageData`. Used by both modes internally.

**Example:** `cargo run --example scatter3d` (inline) or `cargo run --example scatter3d -- --tui` (TUI)

### Sprint 8.5C ✅ CLI Integration & ML Hooks

- `scry viz 3d-scatter` subcommand with `--x`, `--y`, `--z`, `--color-by`, `--output`
- `Chart3D::color_by_labels()` and `color_by_class()` builder methods
- `default_palette()` shared 6-color palette function
- `scatter3d_data()` and `scatter3d_chart()` in scry-learn `viz.rs`

### Sprint 8C ✅ ratatui Feature-Gated

- `ratatui` + `crossterm` moved to optional deps behind `widget` feature flag (default-on)
- `pub mod widget` gated with `#[cfg(feature = "widget")]`
- `Chart3D::show()` gated with `#[cfg(feature = "widget")]`
- Prelude re-exports (`ChartWidget`, `ChartState`, `Chart3DWidget`, `Chart3DState`) conditional
- Headless users: `cargo add scry-chart --no-default-features` drops ~30 transitive deps

### Sprint 8.5D ✅ Performance Targets

| Scenario | Target | Actual Mean |
|----------|--------|:-----------:|
| 1K pts, 800×600 | 16.7ms (60fps) | **1.21ms** (~827fps) |
| 5K pts, 1080p | 33.3ms (30fps) | **4.96ms** (~202fps) |
| 10K pts, 1080p | 66.7ms (15fps) | **10.4ms** (~96fps) |

### Sprint 9B ✅ GPU Acceleration v1 (wgpu)

Implemented `WgpuRasterizer3D` — a GPU-accelerated backend behind `gpu` feature flag (opt-in, NOT default).

**Architecture:**
- `WgpuRasterizer3D` implements `Rasterizer3D` trait — drop-in swap for `SkiaRasterizer3D`
- `Chart3D::render_gpu(w, h)` convenience method (feature-gated)
- Headless wgpu: `Instance → Adapter → Device` (no Surface/window needed)
- Deferred batching: all `draw_*` calls record batches, `finish()` submits one render pass + readback
- WGSL shaders: instanced circle SDF (points) + anti-aliased line quads (segments)
- Text stays CPU-side: fontdue rasterizes, blits to GPU output after readback

**Dependencies added (all optional, behind `gpu` feature):**
- `wgpu = "24"`, `pollster = "0.4"`, `bytemuck = "1"`

**Benchmark results (RTX 5070 Ti, 1080p):**

| Scenario | CPU (tiny-skia) | GPU (wgpu v1) | Notes |
|----------|----------------|---------------|-------|
| 50K pts | 21.9ms | 104ms | GPU slower due to per-call device init overhead |
| 100K pts | ~44ms | 107ms | GPU rendering scales flat (+3ms for 2× points) |

**Key insight:** The ~100ms constant overhead is `request_adapter` + `request_device` + pipeline creation + readback sync. The actual GPU rendering at 100K points is only ~7ms. Sprint 9C caches the device.

**Test results:** 304/304 scry-learn tests passing. 172/172 scry-chart tests passing. Clippy clean `--all-features`.

### Sprint 9C ✅ GPU Device Caching (scry-chart 3D)

Extracted wgpu device, queue, and pipelines into a reusable `WgpuContext` struct.

**New APIs:**
- `WgpuContext::new()` — one-time expensive init (~100ms)
- `WgpuRasterizer3D::with_context(&ctx, w, h, bg)` — fast per-frame rasterizer
- `Chart3D::render_gpu_with_context(&ctx, w, h)` — convenience method

**Architecture:**
- `DeviceRef` / `QueueRef` / `PipelineRef` enums — owned (one-shot) or borrowed (cached)
- `create_frame_resources()` — creates only per-frame texture + uniform buffer
- Existing `WgpuRasterizer3D::new()` and `Chart3D::render_gpu()` still work unchanged

**Benchmark results (RTX 5070 Ti, 1080p):**

| Scenario | CPU (tiny-skia) | GPU uncached | GPU **cached** | vs uncached | vs CPU |
|----------|:-:|:-:|:-:|:-:|:-:|
| 50K pts | 22.2ms | 107ms | **3.46ms** | **31×** | **6.4×** |
| 100K pts | 45.6ms | 109ms | **6.59ms** | **17×** | **6.9×** |

**Test results:** 172/172 tests passing (164 existing + 8 GPU). Clippy clean `--all-features`.

### Sprint 9D-1 ✅ GPU Compute Backend for scry-learn

Implemented `ComputeBackend` trait with CPU and wgpu GPU implementations for accelerated linear algebra.

**Architecture:**
- `ComputeBackend` trait — matmul, XᵀX/Xᵀy, pairwise distances
- `CpuBackend` — pure Rust (always available)
- `GpuBackend` — wgpu compute shaders (behind `gpu` feature flag)
- `accel::auto()` — runtime auto-detection (GPU → CPU fallback)
- Size thresholds: GPU only used when matrices are large enough to offset overhead

**WGSL Compute Shaders:**
- `matmul.wgsl` — tiled 16×16 matrix multiply with shared memory
- `distance.wgsl` — pairwise squared Euclidean distance (256-thread workgroups)

**Dependencies added (all optional, behind `gpu` feature):**
- `wgpu = "24"`, `pollster = "0.4"`, `bytemuck = "1"`

**Test results:** 304/304 tests passing (297 existing + 4 GPU compute + 3 GPU wiring). Clippy clean `--all-features`.

### Sprint 9D-2 ✅ Wire GPU into Model Training

Replaced manual linear algebra loops with `ComputeBackend` calls in all three models:

**LinearRegression.fit():**
- Replaced 24-line XᵀX/Xᵀy loop with `accel::auto().xtx_xty()` (1 line)
- `data.features` is already column-major — zero conversion needed

**KnnClassifier.compute_votes():**
- Batched brute-force distances via `pairwise_distances_squared()` for Euclidean metric
- Gated on: Euclidean metric, no KD-tree, `n_q × n_t ≥ 256`

**KnnRegressor.predict():**
- Same batched distance refactoring as classifier

**Extracted shared helpers:** `scalar_brute_force()`, `batched_brute_force_neighbors()`, `aggregate_votes()`, `aggregate_regression()`

**Test results:** 304/304 scry-learn tests passing (301 existing + 3 GPU wiring). Clippy clean `--all-features`.

---

## 3. What To Do Next

> **Read `.agent/ROADMAP.md` for the full plan.**

### PRIORITY 1: Sprint 9C — scry-engine GPU 2D Rasterizer

Add a GPU-accelerated 2D rasterizer to scry-engine. The current `Rasterizer` in `src/rasterize/skia.rs` uses `tiny-skia` (CPU). This sprint adds a wgpu alternative behind the `gpu` feature flag.

**Scope:**

1. **Add `gpu` feature to `Cargo.toml`** — `wgpu = "24"`, `pollster = "0.4"`, `bytemuck = "1"` (all optional)

2. **Create `src/rasterize/wgpu.rs`** — a `WgpuRasterizer` implementing the same API as `Rasterizer`:
   - `rasterize(&PixelCanvas) → Pixmap`
   - `rasterize_into(&PixelCanvas, &mut Pixmap)`

3. **WGSL shaders to implement:**

   | DrawCommand variant | GPU shader approach |
   |---|---|
   | `Clear` | `clear_value` on render pass |
   | `Circle` | Instanced SDF quad (same pattern as 3D `point.wgsl`) |
   | `Rectangle` | Instanced box SDF or direct quad geometry |
   | `Ellipse` | SDF quad with non-uniform scale |
   | `Line` | Anti-aliased line quads (same pattern as 3D `line.wgsl`) |
   | `Polyline` | Decompose into line segments, batch |
   | `Arc` | SDF fragment shader (angular test) |
   | `Gradient` | Fragment shader with linear/radial interpolation |
   | `Image` | Texture sample + alpha blend |
   | `Path` | CPU tessellation → GPU triangles (or fallback to CPU) |
   | `Text` | CPU rasterization (fontdue) → GPU blit (same as 3D approach) |
   | `Group` | Offscreen render target + composite |

4. **Device caching** — reuse the `WgpuContext` pattern from scry-chart's 3D backend:
   - `WgpuContext::new()` for one-time init
   - `WgpuRasterizer::with_context()` for per-frame reuse
   - Reference: `crates/scry-chart/src/chart3d/wgpu_backend.rs`

5. **`rasterize_auto()`** — runtime detection function:
   - If `gpu` feature enabled + GPU available → `WgpuRasterizer`
   - Otherwise → `Rasterizer` (tiny-skia)

6. **Integration with existing caching** — `RasterCache` still works at the hash level. Only the rasterization backend changes.

**Reference implementation pattern:**
- Study `crates/scry-chart/src/chart3d/wgpu_backend.rs` — same project, same wgpu version, same headless rendering approach
- The `WgpuRasterizer3D` is a working blueprint: headless instance → adapter → device → render pass → texture readback

**Key decision: Path rendering**
- Complex Bézier paths (`DrawCommand::Path`) are hard to GPU-rasterize directly
- Recommended: use cpu-side tessellation (e.g., `lyon` crate or convert `tiny_skia::Path` to triangles) then draw triangles on GPU
- Alternative: fall back to CPU (`Rasterizer`) for Path commands, GPU for everything else
- Start simple — get shapes and lines on GPU first, handle paths later

**Current scry-engine DrawCommand variants (11 total):**
`Clear`, `Circle`, `Rectangle`, `Ellipse`, `Line`, `Path`, `Polyline`, `Gradient`, `Arc`, `Image`, `Text` (text feature-gated), `Group` (nested commands)

**Target:** ≥10× throughput improvement at 4K resolution for 2D charts

### PRIORITY 2: Housekeeping (P2)

- Git commit workflow → `.agent/workflows/git-commit.md` (conventional commits)
- `cargo-semver-checks` → add to CI + Sprint 13A
- `cargo-audit` → add to CI alongside `cargo deny`
- Remove `suggestions.md` before publishing (internal audit)

---

## 4. Known Issues

| Issue | Severity | Details |
|-------|----------|---------|
| ~~GPU v1 per-call overhead~~ | ~~High~~ | ~~Fixed in Sprint 9C — cached: 3.5ms/6.6ms (was 107ms/109ms)~~ |
| Gaussian NB digits gap | Medium | −2.2% vs sklearn (improved from −3.3% with var_smoothing fix) |
| KNN iris gap | Low | −2.7% — inherent to 150-sample dataset, not a bug |
| `determinism_rf_same_seed` flaky | Low | Passes reliably in `--release`; debug-only thread scheduling |
| 8 bench tests `#[ignore]`d | Info | Run: `--release -- --ignored` |

---

## 5. Key Files

| Purpose | Path |
|---------|------|
| Product roadmap | `.agent/ROADMAP.md` |
| Agent context | `.agent/CONTEXT.md` |
| **2D rasterizer (CPU)** | `src/rasterize/skia.rs` (1187 lines) |
| **2D rasterize module** | `src/rasterize/mod.rs` |
| **Command batching** | `src/rasterize/batch.rs` |
| **Content-hash cache** | `src/rasterize/cache.rs` |
| **Performance profiler** | `src/rasterize/profiler.rs` |
| **Scene commands** | `src/scene/command.rs` (DrawCommand enum, 11 variants) |
| **Scene canvas** | `src/scene/mod.rs` (PixelCanvas builder) |
| **Style types** | `src/scene/style.rs` (Color, Rect, ShapeStyle, GradientDef, etc.) |
| Engine Cargo.toml | `Cargo.toml` (workspace root = scry-engine) |
| **3D wgpu backend (reference)** | `crates/scry-chart/src/chart3d/wgpu_backend.rs` |
| **3D point shader (reference)** | `crates/scry-chart/src/chart3d/shaders/point.wgsl` |
| **3D line shader (reference)** | `crates/scry-chart/src/chart3d/shaders/line.wgsl` |
| GPU compute backend | `crates/scry-learn/src/accel/mod.rs` |
| GPU backend (wgpu) | `crates/scry-learn/src/accel/gpu.rs` |
| CPU backend | `crates/scry-learn/src/accel/cpu.rs` |
| Linear regression (GPU-wired) | `crates/scry-learn/src/linear/regression.rs` |
| KNN (GPU-wired) | `crates/scry-learn/src/neighbors/knn.rs` |
| Benchmarks doc | `BENCHMARKS.md` |

## 6. Feature Flag Architecture

| Crate | Feature | What it gates |
|-------|---------|---------------|
| `scry-engine` | `widget` (default) | `ratatui` dep, `PixelCanvasWidget` |
| `scry-engine` | `kitty` (default) | Kitty graphics protocol |
| `scry-engine` | `text` | fontdue text rendering |
| `scry-engine` | `svg` | resvg SVG rendering |
| `scry-engine` | **`gpu`** (to add) | **wgpu + pollster + bytemuck, `WgpuRasterizer`, `rasterize_auto()`** |
| `scry-chart` | `widget` (default) | `ratatui` + `crossterm` deps, `mod widget`, `Chart3D::show()` |
| `scry-chart` | **`gpu`** (opt-in) | **`wgpu` + `pollster` + `bytemuck`, `WgpuRasterizer3D`, `Chart3D::render_gpu()`** |
| `scry-learn` | **`gpu`** (opt-in) | **`wgpu` + `pollster` + `bytemuck`, `GpuBackend`, `accel::auto()`** |
| `scry-chart` | `serde` | Serialize/Deserialize derives |

## 7. Verification Commands

// turbo-all

```bash
# All scry-engine tests
cargo test -p scry-engine --lib --all-features

# Clippy (must be 0 warnings)
cargo clippy -p scry-engine --all-features -- -D warnings

# Headless build (no optional features)
cargo check -p scry-engine --no-default-features

# Default features build
cargo check -p scry-engine

# Workspace check
cargo check --workspace

# Rasterize benchmarks
cargo bench --bench rasterize -p scry-engine

# All scry-chart tests (includes GPU)
cargo test -p scry-chart --lib --all-features

# All scry-learn tests — USE --release for speed
cargo test -p scry-learn --lib --all-features --release
```

## 8. Code Quality Rules

- **Zero clippy warnings** — always run with `-D warnings`
- **All tests must pass** — never break existing functionality
- **Use --release for tests** — debug mode is 50x slower for ML tests
- **Builder pattern** — config types use consuming builder methods
- **Feature gates** — widget/TUI code behind `#[cfg(feature = "widget")]`, GPU behind `#[cfg(feature = "gpu")]`
- **Serde** — new public types derive Serialize/Deserialize under `#[cfg(feature = "serde")]`
- **Doc comments** — all public APIs need `///` doc comments
- **Error handling** — `Result<T, E>` with crate-specific error types, never panic
- **Pure Rust** — no BLAS, nalgebra, or ndarray in default dependencies (optional feature only)
