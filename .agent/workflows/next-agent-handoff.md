---
description: Handoff instructions for the next agent — current state, what's done, what's left
---

# Next Agent Handoff — Full Status & Instructions

> **Last updated**: 2026-02-16 22:03 PST

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

## 2. What Was Just Completed — Gap Analysis

A deep code-validated analysis of 6 honest gaps was completed. Full report at: **`.gemini/antigravity/brain/6f9ada49-436d-4867-8561-857a2a180a75/gap_analysis.md`**

### Summary of Findings

| # | Gap | Severity | Validated? | Key Discovery |
|---|-----|----------|:----------:|---------------|
| 1 | No BLAS/LAPACK (custom DenseMatrix) | High | ✅ | `accel` module is architecturally ready — new backend slots in with zero model changes |
| 2 | Pipeline manual PipelineModel impls | Medium | ✅ | **22** identical impls (not 15) — macro or supertrait fixes it |
| 3 | `Vec<Vec<f64>>` predict API surface | Medium | ✅ | `Dataset` has `from_matrix()` + `flat_feature_matrix()`, but predict bypasses it |
| 4 | Simplified SMO (j-index heuristic) | Medium | ✅ | Deterministic rotation at line 456, no shrinking, O(n²) precomputed kernel matrix |
| 5 | polars/mmap not wired into public API | Medium | ✅ | Both modules fully built+tested, just not exported from `lib.rs` |
| 6 | f64-only (no generic numerics) | Low | ✅ | Intentional for v1 — sklearn is also f64 internally |

### Previous Audit Fixes (All Complete)

All 8 engineering audit findings (P1 + P2) resolved:
- Wire `stream` module, width/height args, ragged CSV rows, division-by-zero guards
- Multi-series bar, candlestick x=0, Cargo.toml docs
- **`cargo test -p scry-pipe`** — 74/74 pass ✅
- **`cargo check -p scry-cli`** — compiles clean ✅

---

## 3. What To Do Next

> **Read `.agent/ROADMAP.md` for the full sprint plan.**

### IMMEDIATE: Quick Wins from Gap Analysis (< 1 hour total)

These are mechanical fixes that dramatically improve surface quality:

#### Quick Win 1: Wire polars/mmap into `lib.rs` (~15 min)

**File:** `crates/scry-learn/src/lib.rs`

Add at the end of the module declarations (after line 66):

```rust
#[cfg(feature = "polars")]
pub mod polars_interop;

#[cfg(feature = "mmap")]
pub mod mmap;
```

Also add to prelude (behind feature gates):
```rust
#[cfg(feature = "mmap")]
pub use crate::mmap::MmapDataset;
```

**Verify:** `cargo check -p scry-learn --all-features` compiles. Check that `polars` and `mmap` features exist in `crates/scry-learn/Cargo.toml` — if not, add them with the appropriate deps (`polars`, `memmap2`).

#### Quick Win 2: Macro for PipelineModel impls (~30 min)

**File:** `crates/scry-learn/src/pipeline.rs`

Replace the 22 identical impl blocks (lines 50-157) with a declarative macro:

```rust
macro_rules! impl_pipeline_model {
    ($($ty:ty),* $(,)?) => {
        $(
            impl PipelineModel for $ty {
                fn fit(&mut self, data: &Dataset) -> Result<()> { self.fit(data) }
                fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> { self.predict(features) }
            }
        )*
    };
}

impl_pipeline_model! {
    crate::tree::DecisionTreeClassifier,
    crate::tree::RandomForestClassifier,
    crate::linear::LinearRegression,
    crate::linear::LogisticRegression,
    crate::neighbors::KnnClassifier,
    crate::naive_bayes::GaussianNb,
    crate::tree::DecisionTreeRegressor,
    crate::tree::RandomForestRegressor,
    crate::tree::GradientBoostingClassifier,
    crate::tree::GradientBoostingRegressor,
    crate::linear::LassoRegression,
    crate::linear::ElasticNet,
    crate::svm::LinearSVC,
    crate::svm::LinearSVR,
    crate::svm::KernelSVC,
    crate::svm::KernelSVR,
    crate::naive_bayes::BernoulliNB,
    crate::naive_bayes::MultinomialNB,
    crate::tree::HistGradientBoostingClassifier,
    crate::tree::HistGradientBoostingRegressor,
    crate::neural::MLPClassifier,
    crate::neural::MLPRegressor,
}
```

**Verify:** `cargo test -p scry-learn --lib --all-features --release`

---

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

### PRIORITY 2: Sprint 10 Gap Fixes (from Gap Analysis)

| Fix | Effort | Impact | Details |
|-----|--------|--------|---------|
| BLAS backend via `ComputeBackend` trait | 2 days | Closes ~1.7% R² gap + 2-4× speed | Add `faer` or `openblas-src` behind `blas` feature flag |
| MVP working-set selection for SMO | 1 day | Major convergence improvement | Replace line 456 heuristic in `kernel.rs` with max-violating-pair scan |
| `predict_matrix()` zero-copy path | 1 day | Eliminates allocation-per-sample | Add overloaded predict accepting `&DenseMatrix` |

### PRIORITY 3: Housekeeping

- Git commit workflow → `.agent/workflows/git-commit.md` (conventional commits)
- `cargo-semver-checks` → add to CI + Sprint 13A
- `cargo-audit` → add to CI alongside `cargo deny`

---

## 4. Known Issues

| Issue | Severity | Details |
|-------|----------|---------|
| Gaussian NB digits gap | Medium | −2.2% vs sklearn (improved from −3.3% with var_smoothing fix) |
| KNN iris gap | Low | −2.7% — inherent to 150-sample dataset, not a bug |
| `determinism_rf_same_seed` flaky | Low | Passes reliably in `--release`; debug-only thread scheduling |
| 8 bench tests `#[ignore]`d | Info | Run: `--release -- --ignored` |
| Full workspace clippy not run | Low | Audit fixes session didn't run full clippy due to compile time |
| 22 manual PipelineModel impls | Medium | Quick fix: macro (see Quick Win 2 above) |
| polars/mmap not exported | Medium | Quick fix: 2 lines in lib.rs (see Quick Win 1 above) |

---

## 5. Key Files

| Purpose | Path |
|---------|------|
| Product roadmap | `.agent/ROADMAP.md` |
| Agent context | `.agent/CONTEXT.md` |
| **Gap analysis report** | `.gemini/antigravity/.../gap_analysis.md` |
| **2D rasterizer (CPU)** | `src/rasterize/skia.rs` (1187 lines) |
| **2D rasterize module** | `src/rasterize/mod.rs` |
| **Command batching** | `src/rasterize/batch.rs` |
| **Content-hash cache** | `src/rasterize/cache.rs` |
| **Performance profiler** | `src/rasterize/profiler.rs` |
| **Scene commands** | `src/scene/command.rs` (DrawCommand enum, 11 variants) |
| **Scene canvas** | `src/scene/mod.rs` (PixelCanvas builder) |
| **Style types** | `src/scene/style.rs` (Color, Rect, ShapeStyle, GradientDef, etc.) |
| Engine Cargo.toml | `Cargo.toml` (workspace root = scry-engine) |
| **Pipeline (22 manual impls)** | `crates/scry-learn/src/pipeline.rs` |
| **DenseMatrix** | `crates/scry-learn/src/matrix.rs` (333 lines) |
| **Compute backend trait** | `crates/scry-learn/src/accel/mod.rs` |
| **GPU backend (wgpu)** | `crates/scry-learn/src/accel/gpu.rs` |
| **CPU backend** | `crates/scry-learn/src/accel/cpu.rs` |
| **SMO solver** | `crates/scry-learn/src/svm/kernel.rs` (lines 409-544) |
| **polars interop (not exported)** | `crates/scry-learn/src/polars_interop.rs` |
| **mmap dataset (not exported)** | `crates/scry-learn/src/mmap.rs` (595 lines) |
| **lib.rs (module exports)** | `crates/scry-learn/src/lib.rs` |
| **3D wgpu backend (reference)** | `crates/scry-chart/src/chart3d/wgpu_backend.rs` |
| **3D point shader (reference)** | `crates/scry-chart/src/chart3d/shaders/point.wgsl` |
| **3D line shader (reference)** | `crates/scry-chart/src/chart3d/shaders/line.wgsl` |
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
| `scry-learn` | `polars` (opt-in) | polars interop (module exists, **not yet exported**) |
| `scry-learn` | `mmap` (opt-in) | memmap2 dataset (module exists, **not yet exported**) |
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

# scry-pipe tests (fast, no GPU deps)
cargo test -p scry-pipe
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

## 9. Architecture Insight: ComputeBackend as BLAS Gateway

The `accel` module in scry-learn already provides the abstraction layer for plugging in optimized backends:

```
ComputeBackend (trait)
├── CpuBackend       ← current default (scalar loops)
├── GpuBackend       ← wgpu compute shaders (exists, feature-gated)
└── BlasBackend      ← TODO: faer/openblas (would slot in here)
```

**Key methods:** `matmul()`, `xtx_xty()`, `pairwise_distances_squared()`, `xtx_xty_contiguous()`

Models call `accel::auto()` which returns the best available backend. Adding BLAS requires:
1. New `BlasBackend` struct implementing `ComputeBackend`
2. Update `auto()` priority: GPU → BLAS → CPU
3. Feature flag: `blas` in `Cargo.toml`
4. **Zero model code changes** — the trait abstraction handles it
