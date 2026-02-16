# scry — Product Roadmap

> **Updated**: 2026-02-16 | v0.7.0 tagged | 450+ tests, 0 clippy warnings
> **Previous sprints 1–7: ALL COMPLETE.** Sprint 8A benchmarks complete + integrity overhaul.

---

## Current State

| Crate | Key Stats | Grade |
|-------|-----------|:-----:|
| scry-engine | 4 protocols, 20+ easings, WASM, SVG, Miri 125/126 | A- |
| scry-chart | 19 chart types, 6 themes, PNG/SVG, 120+ tests | A- |
| scry-learn | 23+ models, 428 tests, GridSearchCV, HistGBT | A- |
| scry-pipe | Phase 1 done (IR + engine + codegen, ~1800 lines) | B |
| scry-cli | 9 source files, chart/image/animation commands | B |

---

## Sprint 8 — Competitive Hardening & Benchmarks

> **Goal**: Identify every weakness vs industry leaders. Prove parity or document gaps.

### 8A. Cross-Competitor Benchmark Suite

**Priority: P0** | 3 sessions | ~9h | **COMPLETE** ✅ — BENCHMARKS.md published, integrity overhaul done

Build a rigorous, reproducible benchmark suite comparing scry against every relevant competitor across all crates.

#### Session 1 Results: scry-learn vs scikit-learn 1.8.0 (5-Fold Stratified CV)

| Dataset | Wins | Ties | Losses | Biggest Gap |
|---------|:----:|:----:|:------:|-------------|
| iris (150×4, 3c) | 2 | 0 | 6 | LogReg −8.7% |
| wine (178×13, 3c) | 4 | 1 | 3 | GBT +2.2% |
| breast_cancer (569×30, 2c) | 3 | 1 | 4 | LinearSVC +1.2% |
| digits (1797×64, 10c) | 3 | 0 | 5 | LogReg −7.0% |
| **TOTAL** | **12** | **2** | **18** | |

**Key findings:**
- ✅ **Tree models at parity** — RF, GBT, HistGBT within ±1% on most datasets
- ✅ **LinearSVC competitive** — wins breast_cancer, ties/wins elsewhere
- ⚠️ **Logistic Regression** is the biggest weakness (−8.7% iris, −7.0% digits) — SGD solver vs sklearn's L-BFGS
- ⚠️ **KNN** slightly behind (−4% iris, −0.5% digits) — likely distance/weight handling differences

**Deliverables created:**
- `benches/industry_benchmark.rs` — 4 Criterion groups (accuracy CV, training scale, prediction latency, scaling curves)
- `benches/python/bench_sklearn.py` — sklearn baseline (ran, JSON saved)
- `benches/python/bench_xgboost.py` — XGBoost baseline (created, needs `pip install xgboost`)
- `benches/python/bench_lightgbm.py` — LightGBM baseline (created, needs `pip install lightgbm`)
- `examples/industry_report.rs` — formatted table report

#### Session 2 TODO: XGBoost/LightGBM Head-to-Head

| Benchmark | What to measure |
|-----------|-----------------|
| **HistGBT vs XGBoost/LightGBM** | Head-to-head accuracy + throughput on UCI datasets |
| **Training speed** | Wall clock at 1K/10K/100K/1M rows — Criterion vs Python timing |
| **Memory footprint** | Peak RSS during fit + model size after serialize |
| **Scaling curves** | Generated comparison charts (scry-chart eating its own dogfood) |

#### Session 3: Benchmark Integrity Overhaul ✅ DONE

Removed all bias/marketing from benchmark suite, added scientific rigor:

| Change | File | Status |
|--------|------|--------|
| Removed 4 Feature Gap Analysis marketing tables (✅/❌ grids) | `benchmark_audit.rs` | ✅ |
| Relabeled train=test accuracy as "timing only — NOT generalization" | `benchmark_audit.rs` | ✅ |
| Added prominent caveat to GBT-vs-RF apples-to-oranges comparison | `benchmark_audit.rs` | ✅ |
| Added FNV-1a `prediction_checksum()` for cross-machine verification | `benchmark_audit.rs` | ✅ |
| Created `tests/numerical_stability.rs` (8 tests) | NEW | ✅ |
| Created `tests/convergence.rs` (6 tests) | NEW | ✅ |
| Created `tests/regression_audit.rs` (4 tests, proper 80/20 splits) | NEW | ✅ |
| Added `bench_thread_scaling` criterion group ([1,2,4,8] threads) | `ml_algorithms.rs` | ✅ |
| Fixed LogReg panic on single-class input (returned Err instead) | `logistic.rs` | ✅ |

#### Session 3 Measured Results: Cross-Library Regression Audit (80/20 split, seed=42)

| Model | scry R² | smartcore R² | linfa R² | Notes |
|-------|---------|-------------|----------|-------|
| LinearRegression | 0.999993 | 0.999993 | 0.999993 | All at parity |
| DTRegressor (depth=8) | 0.591 | 0.578 | — | scry slightly better |
| Lasso (α=0.1) | 0.9999 | — | 0.249 | linfa penalty mismatch? |
| ElasticNet (α=0.1) | 0.9999 | — | 0.249 | linfa penalty mismatch? |

#### Identified Performance Gaps (Rust-vs-Rust)

| Category | scry | Competitor | Gap | Root Cause |
|----------|------|-----------|-----|-----------|
| PCA Transform (2K×20→5) | 1069µs | linfa 268µs | **4× slower** | `Vec<Vec<f64>>` scattered reads in centering — 20 separate heap allocs, CPU prefetcher fails |
| LinearRegression fit (2K×10) | 387µs | linfa 198µs | **2× slower** | Cache-unfriendly XᵀX in `accel/cpu.rs` — same `Vec<Vec<f64>>` problem, 72% of fit time |
| DTRegressor fit (2K×10, d=8) | 16327µs | smartcore 6522µs | **2.5× slower** | Membership bitset overhead, per-threshold MSE recomputation, per-call Vec allocations |

**Root cause analysis:**
- **PCA Transform + LinearRegression**: Both caused by `Vec<Vec<f64>>` column-major layout with separate heap allocations per feature. Sprint 12 (DenseMatrix) fixes both to parity. Short-term: row-major conversion in `xtx_xty()` gives LinearRegression parity (~50 LOC).
- **DTRegressor**: Algorithmic — membership bitset with scattered lookups (35%), no MSE caching (40%), per-call Vec allocations (15%), repeated O(N) index collection (10%). Fix: pre-filtered index arrays + buffer reuse → 2-2.5× speedup. Independent of Sprint 12.

#### scry-chart vs {plotters, D3.js, matplotlib, Plotly}

| Benchmark | What to measure |
|-----------|-----------------|
| **Render speed** | Time to render 10K/100K/1M data points per chart type |
| **Output quality** | Side-by-side screenshot comparison at same data |
| **Feature matrix** | Tick: what we have, gap: what they have that we don't |
| **SVG compliance** | Validate generated SVG against W3C spec |

#### scry-engine vs {plotters-backend, viuer, ratatui-image}

| Benchmark | What to measure |
|-----------|-----------------|
| **Throughput** | Frames per second at 1080p / 4K / 8K |
| **Protocol efficiency** | Bytes-on-wire per frame (Kitty vs Sixel vs iTerm2) |
| **Latency** | First-frame-to-display time |
| **Dirty tile hit rate** | % of tiles skipped in animation loop |

---

### 8B. Weakness Audit & Gap Analysis ✅ DONE

**Priority: P0** | 2 sessions | ~6h | **COMPLETE**

Key fixes shipped:
- ✅ LogReg L-BFGS solver (now default, closing −8% gap)
- ✅ KNN GPU-accelerated distances + tie-breaking fix
- ✅ MLP neural networks → Sprint 11 (in progress)
- ✅ LogReg single-class panic fix (now returns `Err(InvalidParameter)` instead of index OOB)

#### scry-learn Remaining Gaps (vs scikit-learn)

| Gap | Impact | Effort |
|-----|--------|--------|
| ~~No neural network / MLP~~ | ~~High~~ | → **Sprint 11 (in progress)** |
| No BLAS acceleration path | Medium — limits large-matrix perf | 2 sessions |
| Linear regression: normal equation only (no SVD/QR) | Medium — ill-conditioned fails | 1 session |
| No model persistence to ONNX | Medium — limits interop | 2 sessions |
| No warm_start for iterative models | Low | 1 session |
| No sample_weight for all regressors | Low | 1 session |

#### scry-chart Gaps (vs matplotlib/Plotly)

| Gap | Impact | Effort |
|-----|--------|--------|
| ~~No 3D charts (surface, 3D scatter)~~ | ~~Medium~~ | → **Sprint 8.5 (P0)** |
| No geographic/map charts | Low | Long-term |
| No animated transitions between datasets | Medium | 2 sessions |
| No interactive tooltip content callbacks | Low | 1 session |
| Statistical overlays (CI bands, regression lines) | Medium | 1 session |

#### scry-engine Gaps (vs GPU renderers)

| Gap | Impact | Effort |
|-----|--------|--------|
| No GPU acceleration (CUDA/wgpu) | High — ceiling on throughput | Sprint 9 (unlocks 3D crazy mode) |
| WASM limited to basic shapes | Medium | 1 session |
| No text shaping (harfbuzz) | Low — mono fonts only | 2 sessions |

---

### 8C. Test Infrastructure & Code Health

**Priority: P0** | 2 sessions | ~6h | **Fuzz, Miri, numerical stability, convergence: COMPLETE ✅**

#### Fuzz & Miri Coverage

| Crate | Fuzz | Miri | Status |
|-------|:----:|:----:|--------|
| scry-engine | ✅ 4 targets | ✅ 125/126 | Done |
| scry-chart | ✅ 2 targets | ✅ 9/9 | Done |
| scry-learn | ✅ 3 fuzz + 8 numerical stability + 6 convergence | ✅ CI added | **Done** — fuzz, miri, numerical_stability.rs, convergence.rs |
| scry-pipe | ❌ 0 | ❌ not run | Remaining — IR deserialization, codegen |

**scry-learn fuzz targets (9 total, 3 new):**
- `fuzz_cart_predict` — structurally valid FlatTree with fuzz thresholds/values (1.8M runs/11s clean)
- `fuzz_scaler_chain` — StandardScaler/MinMaxScaler/RobustScaler on degenerate data (223K runs/11s clean)
- `fuzz_neural_forward` — random MLP architectures + fuzz data through fit/predict (49K runs/11s clean)

**scry-pipe fuzz targets (TODO):**
- `fuzz_pipeline_transform` — random pipeline JSON → engine transform
- `fuzz_ir_roundtrip` — random JSON → PipelineDef → JSON → assert equal

**Miri CI:**
- ✅ `cargo +nightly miri test -p scry-learn -- --skip gpu --skip viz` — added to `.github/workflows/ci.yml`
- ❌ `cargo +nightly miri test -p scry-pipe` — TODO

#### Large File Refactoring

Files past maintainability thresholds that need decomposition:

| File | Lines | Split Into |
|------|------:|-----------|
| `src/rasterize/skia.rs` | 1,187 | `skia/{shapes,gradients,text,mod}.rs` |
| `scry-learn/src/tree/cart.rs` | 2,040 | `cart/{node,builder,predict,pruning}.rs` |
| `scry-learn/src/search.rs` | ~1,400 | `search/{grid,random,tunable,results}.rs` |
| `scry-chart/src/formatter.rs` | ~1,300 | `formatter/{numeric,date,locale,semantic}.rs` |
| `scry-chart/src/layout/mod.rs` | ~1,200 | Extract `render_context.rs`, `common_overlays.rs` |

**Rules:** Zero public API change, re-export from parent, all 428 tests pass, clippy clean.

#### Dependency Audit ⚠️

Prod dependencies are lean, but **ratatui gating needs work**:

| Crate | Prod Deps | Notes |
|-------|:---------:|-------|
| scry-engine | tiny-skia, fontdue, ratatui*(opt)*, flate2, base64, png | ✅ ratatui is optional (`widget` feature) |
| scry-chart | scry-engine, tiny-skia, fontdue, **ratatui, crossterm**, thiserror | ⚠️ ratatui is **hard** — should be optional |
| scry-learn | csv, fastrand, rayon, scry-chart, thiserror | ✅ 5 deps only |
| scry-pipe | rayon, serde, serde_json, thiserror | ✅ 4 deps, standalone |

No BLAS, no nalgebra, no ndarray in production. Competitor deps (linfa-*, smartcore) are dev-only.

#### Make ratatui Optional in scry-chart (NEW — P1)

**Problem:** `scry-chart` hard-depends on `ratatui` + `crossterm`, pulling in **~30 transitive crates** (cassowary, compact_str, mio, parking_lot, signal-hook, darling, itertools, lru, strum, unicode-*, etc.) that are completely unnecessary for headless/export use cases.

**scry-engine already does this correctly** — `ratatui` is behind `widget` feature flag.

**Changes needed:**

| File | Change |
|------|--------|
| `crates/scry-chart/Cargo.toml` | Move `ratatui` and `crossterm` to optional deps, add `widget` feature flag |
| `crates/scry-chart/src/widget.rs` | Gate entire module with `#[cfg(feature = "widget")]` |
| `crates/scry-chart/src/lib.rs` | Conditional `mod widget` + conditional re-exports in `prelude` |
| `crates/scry-chart/src/cursor.rs` | Audit for crossterm usage — gate or abstract |
| `crates/scry-chart/src/chart3d/mod.rs` | Gate `Chart3DWidget`/`Chart3DState` re-exports |
| Root `Cargo.toml` (dev-deps) | Ensure examples still compile with explicit feature activation |

**Verification:** `cargo check -p scry-chart --no-default-features` must succeed (no ratatui). All existing tests must pass with `--all-features`.

---

## Sprint 8.5 — Interactive 3D ML Visualization ⭐

> **Goal**: Ship interactive 3D scatter/surface plots in the terminal. The differentiator feature.

**Priority: P0** | 4 sessions | ~12h

This is the feature that makes scry **the** terminal ML tool. Nobody else does live, rotatable 3D ML visualization natively in Kitty.

### 8.5A. 3D Scene Graph & Camera (2 sessions)

New module: `scry-chart/src/chart3d/`

| File | Purpose |
|------|---------|
| `camera.rs` | `Camera3D` — position, target, up, FOV, arcball quaternion rotation |
| `scene.rs` | `Scene3D` — point clouds, axis lines, grid planes, labels |
| `projection.rs` | Perspective 3D→2D transform, depth sorting (painter's algorithm) |
| `interaction.rs` | Input→camera mapping: orbit (drag/WASD), pan (shift+drag), zoom (scroll/+/-) |
| `mod.rs` | `Chart3D` builder — public API |

**Critical architecture rule:** The `Rasterizer3D` trait abstracts rendering. Scene/camera logic does NOT depend on tiny-skia or any renderer:

```rust
pub trait Rasterizer3D {
    fn draw_points(&mut self, points: &[ProjectedPoint], colors: &[Color], sizes: &[f32]);
    fn draw_line_segments(&mut self, segments: &[(ProjectedPoint, ProjectedPoint)], color: Color);
    fn draw_axes(&mut self, axes: &AxisConfig3D);
    fn finish(self) -> PixelCanvas;
}

pub struct SkiaRasterizer3D { pixmap: tiny_skia::Pixmap }  // v1: ships now
// pub struct WgpuRasterizer3D { ... }                       // Sprint 9: swaps in
```

### 8.5B. Two Rendering Frontends (1 session)

| Mode | Flag | Behavior |
|------|------|----------|
| **Inline Kitty** | default | Streams frames to stdout via Kitty protocol. Keyboard controls (WASD rotate, +/- zoom, Q quit). No terminal takeover — user stays in their shell. |
| **TUI** | `--tui` | Full Ratatui `StatefulWidget` with mouse drag rotation, scroll zoom, side panels for metrics/feature importance. |

Inline mode uses scoped `enable_raw_mode()` for key capture only — NOT a full TUI takeover.

### 8.5C. CLI Integration & ML Hooks (1 session)

```bash
# Direct CSV visualization
scry viz 3d-scatter data.csv --x col1 --y col2 --z col3 --color-by species

# Pipe from model predictions
scry learn predict --model rf --data test.csv | scry viz 3d-scatter --color-by prediction

# Decision boundary surface
scry viz decision-boundary model.json --features f1,f2,f3 --resolution 50
```

ML integration:
- `scry-learn` models expose `extract_3d_data()` → `(Vec<[f64; 3]>, Vec<usize>)` (points, labels)
- Color by class/cluster, point size by confidence/distance
- Axis labels from column names

### 8.5D. Performance Targets

| Scenario | Target (SkiaRasterizer3D) |
|----------|:-------------------------:|
| 1K points, 800×600 | 60 fps |
| 5K points, 1080p | 30 fps |
| 10K points, 1080p | 15 fps (acceptable for v1) |
| 50K+ points | Requires GPU (Sprint 9) |

### 8.5E. Concrete Deliverables

1. `Camera3D` with arcball orbit (quaternion, no gimbal lock)
2. `Chart3D::scatter()` builder producing a `PixelCanvas`
3. Billboard axis labels (always face camera)
4. Inline Kitty rendering loop with keyboard controls
5. `scry viz 3d-scatter` CLI subcommand
6. Integration test: iris dataset 3D scatter colored by species
7. Example: `examples/ml_3d_scatter.rs`

**Existing code to build on:**
- `examples/cube_3d.rs` — basic 3D→2D projection
- `scry-chart/src/interactive.rs` — zoom/pan interaction model
- `src/kitty.rs` — Kitty protocol transport
- `crates/scry-cli/` — CLI structure

---

## Sprint 8.6 — Algorithm Visualization Gallery ⭐ ✅ DONE

> **Goal**: Ship a `.viz()` method on every scry-learn model that produces scry-chart outputs. Every algorithm family gets canonical visualizations.

**Priority: P1** | 3 sessions | ~9h | **COMPLETE**

Full `Visualize` trait with `.viz()` on all model families: trees (feature importance), linear (coefficients), clustering (scatter), PCA (scree), GaussianNb (density), classifiers (confusion matrix), regressors (residual + prediction error).

### API Pattern

```rust
let model = RandomForestClassifier::fit(&x, &y)?;
model.viz().feature_importance();          // → PixelCanvas bar chart
model.viz().confusion_matrix(&test_x, &test_y);  // → PixelCanvas heatmap
model.viz().learning_curve(&val_x, &val_y);       // → PixelCanvas line chart
```

### Visualization Matrix

| Model Family | Visualizations |
|-------------|----------------|
| Decision Tree | Tree structure diagram (splits + leaf values), feature importance bar chart |
| Random Forest | Feature importance bar chart, OOB error curve, individual tree overlay |
| Gradient Boosting / HistGBT | Learning curves (train vs val loss), staged prediction evolution |
| Logistic Regression / Linear | Coefficient bar chart, regularization path plot |
| KNN | 2D decision boundary plot, distance heatmap |
| SVM (Linear + Kernel) | Support vector overlay on scatter, 3D decision boundary surface (Chart3D) |
| Gaussian NB | Per-class Gaussian density curves |
| KMeans / DBSCAN | Cluster assignment scatter, silhouette plot, elbow curve |
| PCA | Explained variance scree plot, biplot |
| All classifiers | Confusion matrix heatmap, ROC curve, PR curve, calibration curve |
| All regressors | Residual plot, Q-Q plot, predicted-vs-actual scatter |

### Implementation

- Add `Visualize` trait in scry-learn that returns a builder
- Each model implements it
- Charts produced via scry-chart

---

## Sprint 9 — GPU Acceleration (wgpu)

> **Goal**: Break through the CPU ceiling. Unlock "crazy mode" for 3D viz + ML compute.

**Priority: P0** | 5 sessions | ~15-20h

With Sprint 8.5's `Rasterizer3D` trait in place, GPU is a backend swap — not a rewrite.

### 9A. Architecture Decision — wgpu vs CUDA

| Option | Pros | Cons |
|--------|------|------|
| **wgpu** | Cross-platform (Vulkan/Metal/DX12/WebGPU), Rust-native, works on WASM | Compute shader complexity, less mature for compute workloads |
| **CUDA** | Maximum NVIDIA perf, mature ecosystem, cuBLAS interop | NVIDIA-only, C FFI, no WASM, licensing |
| **Hybrid** | wgpu for rendering, optional CUDA for ML training | 2x implementation cost |

**Recommendation:** wgpu for rasterization, optional CUDA feature for scry-learn matrix operations.

### 9B. WgpuRasterizer3D — GPU Rendering Backend

- New feature flag: `--features gpu`
- Implements `Rasterizer3D` trait from Sprint 8.5
- Instanced point rendering — 100K+ points at 60fps
- Fragment shader for anti-aliased shape rendering
- Render to offscreen texture → read back → Kitty protocol
- Fallback to `SkiaRasterizer3D` when no GPU available
- **Target:** ≥10x throughput improvement, 100K points at 60fps

### 9C. wgpu 2D Rasterization Backend

- `src/rasterize/wgpu.rs` — GPU-accelerated 2D chart rendering
- Batch all shapes into GPU draw calls
- Dirty-tile detection still on CPU, GPU renders only dirty tiles
- **Target:** ≥10x throughput at 4K resolution for 2D charts

### 9D. GPU Compute for scry-learn ✅ DONE

Implemented via wgpu (not CUDA) for cross-platform support:

- Feature flag: `--features gpu`
- `crates/scry-learn/src/accel/gpu.rs` — wgpu compute shaders for matmul + distances
- `ComputeBackend` trait with auto-detection (GPU → CPU fallback)
- Matmul: 16×16 workgroups, f32 precision, threshold `m*k*n ≥ 4096`
- Pairwise distances: 256-thread workgroups, threshold `n_q*n_t ≥ 1024`
- Used by KNN, PCA, and now MLP forward pass

### 9E. "Crazy Mode" Unlocks (stretch goals)

- Live training visualization — watch gradient descent on loss surface in real-time
- 1M-point 3D scatter at 60fps
- Volumetric density plots (ray marching in fragment shader)
- Animated cluster convergence with particle trails
- `scry demo galaxy` — the viral demo (100K synthetic galaxy, DBSCAN clustering, 3D orbit)

---

## Sprint 10 — scry-pipe Phase 2 + Interop

> **Goal**: Complete the train→serve pipeline story.

**Priority: P1** | 5 sessions | ~15h

### 10A. PyO3 Python SDK (3 sessions)

- `pip install scry-pipe`
- Python API: `Pipeline.fit(X, y).export("pipeline.json")` → JSON
- Rust: `scry_pipe::PipelineDef::from_json("pipeline.json")` → compiled binary
- **Parity guarantee:** identical float output between Python and Rust to 1e-12

### 10B. WASM Codegen Target (1 session)

- `RustCodegen` → `WasmCodegen` — emit `.wasm` binary from pipeline definition
- Run feature pipelines in browser without server roundtrip

### 10C. DataFrame / Polars Interop (1 session)

- `impl From<polars::DataFrame> for Dataset`
- `impl From<Dataset> for polars::DataFrame`
- Feature flag: `--features polars`

---

## Sprint 11 — Neural Networks & Deep Learning Basics ⭐ CURRENT SPRINT

> **Goal**: MLP and basic neural network to close the biggest algorithm gap.

**Priority: P0** | 3 sessions | ~10h | **IN PROGRESS**

### 11A. Multi-Layer Perceptron

- `MLPClassifier` / `MLPRegressor` with sklearn-compatible API
- Hidden layers: `&[100, 50]`, default `[100]`
- Activations: ReLU (default), Sigmoid, Tanh, Identity
- Optimizers: Adam (default, β₁=0.9, β₂=0.999), SGD with Nesterov momentum
- Manual backprop (zero extra deps), He/Xavier init
- GPU matmul for forward pass via `ComputeBackend`
- Early stopping with best-weight restoration
- L2 regularization, mini-batch training
- `.viz().learning_curve()` + `.viz().weight_heatmap()`
- `impl Tunable` for GridSearchCV integration

**Files:**
- `crates/scry-learn/src/neural/mod.rs` — module root
- `crates/scry-learn/src/neural/activation.rs` — forward/backward
- `crates/scry-learn/src/neural/optimizer.rs` — SGD + Adam
- `crates/scry-learn/src/neural/layer.rs` — DenseLayer
- `crates/scry-learn/src/neural/network.rs` — forward/backward chain
- `crates/scry-learn/src/neural/classifier.rs` — MLPClassifier
- `crates/scry-learn/src/neural/regressor.rs` — MLPRegressor

### 11B. ONNX Export (stretch)

- Export trained models to ONNX format for cross-framework interop
- Import ONNX models for prediction (limited op coverage)

---

## Sprint 12 — Scaling Foundation: Contiguous Memory Layout

> **Goal**: Replace `Vec<Vec<f64>>` with a contiguous, stride-based matrix to unlock cache-efficient computation at scale. This is the prerequisite for every other scaling improvement.

> [!IMPORTANT]
> **SWITCH TO DESKTOP (Ryzen 9800X3D + RTX 5070 Ti) for this sprint and all subsequent sprints.**
> The 96MB 3D V-Cache is critical for cache-sensitive benchmarks. The 5070 Ti is needed for wgpu compute validation. Laptop is fine for Sprints 8C/11 but Sprint 12+ needs the real hardware.

**Priority: P0** | 4 sessions | ~12h

### Why This Matters — Measured Evidence

The current `Vec<Vec<f64>>` column-major layout means every column is a separate heap allocation. **Measured impact from 8A benchmarks:**

| Operation | scry (scattered) | linfa (contiguous) | Gap |
|-----------|-----------------|-------------------|-----|
| PCA Transform (2K×20→5) | 1069µs | 268µs | 4× — centering step chases 20 pointers |
| LinearRegression fit (2K×10) | 387µs | 198µs | 2× — XᵀX computation 72% of cost |
| PCA Fit (2K×20) | 2153µs | 26172µs | scry 12× faster (blocked matmul compensates) |

The PCA Fit is already fast because the blocked matmul copies into a contiguous buffer first. But the centering/transform step can't hide the layout cost. At 100K+ rows, the gap widens further.

NumPy/sklearn get their speed from contiguous memory — this is the single highest-impact change for scaling.

### 12A. Core Matrix Type (2 sessions)

New internal type: `crates/scry-learn/src/matrix.rs`

| Component | Design |
|-----------|--------|
| `DenseMatrix` | Contiguous `Vec<f64>` with `(n_rows, n_cols, stride)`, column-major layout |
| Indexing | `matrix.col(j) -> &[f64]` (zero-cost slice), `matrix.row(i) -> StrideIter` |
| Ownership | Owned (`DenseMatrix`) and borrowed (`MatrixView<'a>`) variants |
| Construction | `from_col_major(data, n_rows, n_cols)`, `from_row_major(...)` (transposes once) |
| Interop | `impl From<Vec<Vec<f64>>> for DenseMatrix` for backwards compat during migration |

**Critical constraint:** The `Dataset` public API should remain stable. `DenseMatrix` is the internal storage; `Dataset::new()` still accepts `Vec<Vec<f64>>` but converts internally. New `Dataset::from_matrix()` for zero-copy path.

### 12B. Migrate Dataset & All Models (2 sessions)

| Area | Change |
|------|--------|
| `dataset.rs` | Store `DenseMatrix` internally, keep public `Vec<Vec<f64>>` accessors (deprecated) |
| Tree models (CART, RF, GBT, HistGBT) | Replace `&[Vec<f64>]` feature access with `matrix.col(j)` slices |
| Linear models | Replace manual dot products with contiguous column slices |
| KNN / distance computation | Contiguous row iteration for pairwise distances |
| PCA | Already does blocked matmul — wire to contiguous backing |
| Preprocessing (scalers) | Operate on contiguous column slices |
| GPU compute | Pass contiguous buffer directly to wgpu — no gather step |

**Verification:**
- All 428+ tests pass with identical results (bitwise f64 equality)
- Benchmark before/after at 10K, 100K, 1M rows
- No public API breakage (semver safe via deprecation)

### 12C. Large-Scale Benchmarks (extend 8A)

Add benchmark tiers that stress the new layout:

| Scale | Rows | Features | What it tests |
|-------|-----:|:--------:|---------------|
| Medium | 100K | 50 | Cache efficiency, rayon scaling |
| Large | 1M | 100 | Memory throughput, allocation pressure |
| Wide | 10K | 10K | High-dimensional regime (PCA, linear) |

New benchmark file: `benches/scaling_benchmark.rs`
- Compare scry-learn vs sklearn at each tier
- Report wall clock, peak RSS, throughput (rows/sec)

---

## Sprint 12.5 — CART Builder Optimization

> **Goal**: Close the 2.5× DTRegressor performance gap vs smartcore. Algorithmic fix, independent of DenseMatrix.

**Priority: P1** | 1 session | ~4h

### Root Cause (measured in 8A regression_audit)

scry DTRegressor fit: 16327µs vs smartcore 6522µs on 2000×10, max_depth=8.

| Bottleneck | % of overhead | Fix |
|-----------|:------------:|-----|
| Membership bitset with scattered index lookups | 35% | Pre-filter sorted arrays at partition time, pass index subsets to children |
| Per-threshold MSE recomputation (no caching) | 40% | Incremental variance (Welford's) or histogram binning |
| Per-call `Vec<usize>` allocation for feature indices | 15% | Cache in `self` or preallocate at tree start |
| Repeated O(N) active index collection per node | 10% | Partition returns filtered subsets directly |

### Implementation

**File:** `crates/scry-learn/src/tree/cart/builder.rs`

1. Replace `membership: Vec<bool>` with pre-filtered `active_indices: Vec<usize>` passed to recursive calls
2. At partition time, split `active_indices` into `left_indices` / `right_indices` (single pass)
3. Iterate `active_indices` directly instead of checking `membership[idx]` in tight loops
4. Preallocate feature selection buffer once at tree root

**Expected result:** 2.0-2.5× speedup → parity with smartcore.

**Verification:** `regression_audit.rs` DTRegressor test must show scry fit time within 1.5× of smartcore.

---

## Sprint 13 — SVD/QR Solvers & Numerical Robustness

> **Goal**: Close the numerical stability gap. The normal equation fails on ill-conditioned and wide matrices — real-world data hits this constantly.

**Priority: P1** | 2 sessions | ~6h

### 13A. SVD Solver for Linear/Ridge Regression (1 session)

| Component | Design |
|-----------|--------|
| `svd.rs` | Golub-Kahan bidiagonalization → SVD (pure Rust, no deps) |
| `qr.rs` | Householder QR decomposition (fallback for overdetermined systems) |
| Solver selection | `LinearRegression::new().solver(Solver::Svd)` / `Solver::Qr` / `Solver::Normal` (default stays Normal for speed, SVD for robustness) |
| Auto-detect | If `n_features > n_samples` or condition number > threshold, warn and suggest SVD |

### 13B. Condition Number Diagnostics (1 session)

- `Dataset::condition_number()` — compute and warn on ill-conditioned feature matrices
- `LinearRegression::fit()` returns `ScryWarning::IllConditioned` when κ > 1e12
- Docs: when to use which solver, with examples of failure modes

**Verification:**
- Test on Hilbert matrix (classic ill-conditioned case) — Normal equation diverges, SVD converges
- Test on wide matrix (p >> n) — QR/SVD solve, Normal panics/NaN
- Accuracy parity with sklearn's `LinearRegression(svd)` on UCI datasets

---

## Sprint 14 — Sparse Matrix Support

> **Goal**: Enable scry-learn to handle high-dimensional sparse data (text, categoricals, recommender systems) without blowing up memory.

**Priority: P1** | 4 sessions | ~12h

### Why This Matters

Real production ML is dominated by sparse data. A text classification task with 50K vocabulary and 100K documents is a 5 billion element matrix — but >99% zeros. Without sparse support, scry-learn can't touch NLP, recommender systems, or high-cardinality categorical features.

### 14A. Sparse Matrix Types (1 session)

New module: `crates/scry-learn/src/sparse.rs`

| Type | Layout | Use case |
|------|--------|----------|
| `CsrMatrix` | Compressed Sparse Row | Row iteration (KNN, tree predict) |
| `CscMatrix` | Compressed Sparse Column | Column iteration (tree fit, linear algebra) |
| Conversion | `csr.to_csc()`, `csc.to_csr()` — one-time O(nnz) transpose |
| Construction | `CsrMatrix::from_triplets(rows, cols, vals, shape)` |

### 14B. Sparse-Aware Algorithms (2 sessions)

| Model | Sparse adaptation |
|-------|-------------------|
| Linear/Logistic/Lasso/ElasticNet | Sparse dot product, skip zero columns in gradient |
| Decision Trees | Sparse split finding — only iterate non-zero entries per feature |
| Random Forest / GBT | Inherit from tree changes |
| KNN | Sparse distance computation (only sum non-zero diffs) |
| Naive Bayes | Sparse likelihood — natural fit for text classification |
| StandardScaler | Sparse-aware mean/std (compute from nnz, don't densify) |

### 14C. Sparse Dataset Integration (1 session)

| Component | Change |
|-----------|--------|
| `Dataset` | `enum Storage { Dense(DenseMatrix), Sparse(CscMatrix) }` |
| `Dataset::from_sparse(csc, target, names, target_name)` | Constructor |
| `Dataset::is_sparse() -> bool` | Runtime check |
| Models | `fit()` dispatches to dense or sparse path internally |
| CSV loading | Detect sparsity ratio, auto-convert if >80% zeros |

**Verification:**
- 20 Newsgroups text classification benchmark (TF-IDF, ~20K docs × ~130K features)
- Memory: sparse should use <1% of dense memory
- Accuracy: identical to dense path on same data
- Speed: sparse KNN/NB/linear should be faster than dense on sparse data

---

## Sprint 15 — Out-of-Core & Streaming Fit

> **Goal**: Train on datasets larger than RAM. Enable `partial_fit` for incremental learning.

**Priority: P2** | 3 sessions | ~9h

### 15A. partial_fit API (1 session)

| Model | partial_fit support |
|-------|-------------------|
| SGD-based (LogReg, Linear with SGD) | Natural — already mini-batch internally |
| Mini-batch K-Means | Update centroids incrementally |
| MLP | Already mini-batch — expose `partial_fit(batch)` |
| Naive Bayes | Accumulate sufficient statistics |
| Trees / RF / GBT | **Not supported** — inherently batch algorithms (document this) |

```rust
let mut model = LogisticRegression::new().solver(Solver::Sgd);
for batch in data_stream.chunks(10_000) {
    model.partial_fit(&batch)?;
}
```

### 15B. Memory-Mapped Dataset Loading (1 session)

| Component | Design |
|-----------|--------|
| `MmapDataset` | Memory-mapped file backing via `memmap2` crate |
| Format | Custom binary format: header (schema) + contiguous f64 columns |
| `Dataset::from_mmap(path)` | Zero-copy load, OS manages paging |
| Parquet support | Feature-gated `parquet` dep for direct columnar read |

### 15C. Streaming Data Sources (1 session)

- `StreamingDataset` iterator adapter — wraps any `Iterator<Item = Vec<f64>>`
- Integrates with `partial_fit` models
- Backpressure: configurable buffer size
- Progress reporting: rows processed, estimated time remaining

---

## Sprint 16 — Streaming & Live Data (Charts)

> **Goal**: Real-time charting and streaming data support.

**Priority: P2** | 3 sessions | ~9h | ⚠️ **Partially exists — needs audit**

> [!NOTE]
> Some streaming chart functionality may already exist in the codebase.
> Audit what's done vs what remains before starting new work.

### 16A. Streaming Charts

- `StreamingChart` with ring buffer backing
- Auto-scrolling time axis with tick anchoring and hysteresis
- Configurable window: last N points or last T seconds
- Append-only API: `chart.push(timestamp, value)`

### 16B. Live Data Sources

- stdin pipe: `cat /proc/stat | scry stream --chart line`
- WebSocket: `scry stream --ws ws://metrics:8080/cpu --chart gauge`

---

## Sprint 17 — Publication & Ecosystem

> **Goal**: Ship to crates.io and establish the ecosystem presence.

**Priority: P1** | 2 sessions | ~6h

### 17A. Pre-Publish Checklist

- [ ] Per-crate README.md (learn, pipe, cli) — currently engine and chart only
- [ ] `cargo publish --dry-run` for all 5 crates
- [ ] License headers in all source files
- [ ] API stability audit — mark `#[non_exhaustive]` on all public error types
- [ ] `CHANGELOG.md` audit — ensure all changes since 0.7.0 are captured
- [ ] `cargo deny check` — license and advisory audit

### 17B. The scry Handbook

- [ ] mdBook-based documentation site (`book/`)
- [ ] **Part 1 — Engine**: rendering pipeline, protocol selection, animation system, WASM deployment
- [ ] **Part 2 — Charts**: chart type catalog with screenshots, theming guide, formatter reference, export options
- [ ] **Part 3 — ML**: algorithm selection guide, preprocessing pipeline cookbook, hyperparameter tuning patterns, visualization gallery
- [ ] **Part 4 — Pipe**: feature engineering tutorial, Python→Rust deployment walkthrough, codegen internals
- [ ] **Part 5 — CLI**: command reference, piping patterns, scripting recipes
- [ ] Gallery page with rendered chart screenshots for all 19 types
- [ ] sklearn migration guide (mapping sklearn API → scry-learn API)
- [ ] Performance tuning guide (when to use HistGBT vs GBT, batch sizes, Rayon tuning)

---

## Versioning Strategy

| Milestone | Version | Gate |
|:---------:|:-------:|------|
| ~~Everything through Sprint 7~~ | ~~0.7.0~~ | ✅ Tagged |
| Industry benchmark suite published | 0.8.0 | BENCHMARKS.md with reproducible results |
| **3D interactive viz (inline + TUI)** | **0.8.5** | **`scry viz 3d-scatter` works with rotation/zoom in Kitty** |
| Algorithm viz gallery (`.viz()`) | 0.8.6 | `model.viz().feature_importance()` works for all model families |
| GPU backend MVP | 0.9.0 | wgpu compute for learn + 2D rasterizer for engine |
| scry-pipe Phase 2 (Python SDK) | 0.10.0 | `pip install scry-pipe` works |
| **MLP neural networks** | **0.11.0** | **MLPClassifier/MLPRegressor pass XOR + regression tests** |
| **Contiguous memory layout** | **0.12.0** | **DenseMatrix backing, 1M-row benchmarks pass** |
| **SVD/QR solvers** | **0.13.0** | **Hilbert matrix test passes, wide matrix regression works** |
| **Sparse matrix support** | **0.14.0** | **CsrMatrix/CscMatrix, sparse NB on 20 Newsgroups** |
| Out-of-core / partial_fit | 0.15.0 | LogReg/KMeans/MLP partial_fit works on streaming data |
| Streaming charts | 0.16.0 | `scry stream` with live updating charts |
| Handbook published | 0.17.0 | mdBook deployed |
| API freeze + crates.io publish | **1.0.0** | SemVer enforced, public API committed |

---

## Execution Priority

```
8A. Benchmark Suite         ██████████  P0  ← DONE — integrity overhaul, regression audit, checksums
8B. Weakness Audit          ██████████  P0  ← DONE — L-BFGS default solver, KNN GPU + tie-breaking
8.5 3D Interactive Viz      ██████████  P0  ← DONE (8.5A-D complete)
8.6 Algorithm Viz Gallery   ██████████  P1  ← DONE — full .viz() trait, all model families
8C. Test Infra (fuzz/miri)  ██████████  P0  ← DONE — 3 fuzz, Miri CI, numerical_stability, convergence
8C. Ratatui Feature-Gating  ██████████  P1  ← DONE
8C. Large File Refactoring  ░░░░░░░░░░  P1  ← IN PROGRESS (other agent) — cart.rs, search.rs, formatter.rs
9A-B. GPU (wgpu + 3D)       ██████████  P0  ← DONE — 9B+9C complete
9D. GPU Compute (learn)     ██████████  P0  ← DONE — wgpu matmul + distances implemented
11. Neural Networks (MLP)   ████████░░  P0  ← IN PROGRESS (other agent) — 2175 LOC, 34 tests
12. Contiguous Memory       ░░░░░░░░░░  P0  ← NOT STARTED — fixes PCA Transform 4× + LinReg 2× gaps
12.5 CART Builder Optim     ░░░░░░░░░░  P1  ← NOT STARTED — fixes DTRegressor 2.5× gap (algorithmic)
13. SVD/QR Solvers          ░░░░░░░░░░  P1  ← NOT STARTED — numerical robustness
14. Sparse Matrices         ░░░░░░░░░░  P1  ← NOT STARTED — unlocks NLP/recommender workloads
15. Out-of-Core / partial   ░░░░░░░░░░  P2  ← NOT STARTED — depends on 12+14
17A. Pre-Publish            ██████░░░░  P1
10A. PyO3 SDK               █████░░░░░  P1
10B-C. WASM + Polars        ███░░░░░░░  P1
16. Streaming Charts        ██░░░░░░░░  P2  ← needs audit (may be partial)
```
