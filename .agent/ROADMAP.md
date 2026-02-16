# scry — Product Roadmap

> **Updated**: 2026-02-16 | v0.7.0 tagged | 428 tests, 0 clippy warnings
> **Previous sprints 1–7: ALL COMPLETE.** Sprint 8A-1 benchmarks complete.

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

**Priority: P0** | 3 sessions | ~9h | **Session 1 COMPLETE ✅**

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

#### Session 3 TODO: Fix Gaps + Publish BENCHMARKS.md

| Task | Priority |
|------|---------|
| **Fix LogReg solver** — implement L-BFGS or Newton-CG to close −8% gap | P0 |
| **KNN accuracy audit** — investigate distance/weight handling | P1 |
| **Publish BENCHMARKS.md** with reproducible results | P0 |
| **Generated comparison charts** — scry-chart bar charts from JSON data | P1 |

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

### 8B. Weakness Audit & Gap Analysis

**Priority: P0** | 2 sessions | ~6h

Systematic audit of what competitors have that we don't.

#### scry-learn Gaps (vs scikit-learn)

| Gap | Impact | Effort |
|-----|--------|--------|
| No neural network / MLP | High — basic NN is expected | 3 sessions |
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

**Priority: P0** | 2 sessions | ~6h

#### Fuzz & Miri Coverage Gaps

Current coverage is **engine + chart only**:

| Crate | Fuzz | Miri | Gap |
|-------|:----:|:----:|-----|
| scry-engine | ✅ 4 targets | ✅ 125/126 | — |
| scry-chart | ✅ 2 targets | ✅ 9/9 | — |
| scry-learn | ❌ 0 | ❌ not run | **Critical** — tree `unsafe` predict, kernel SVM, matrix ops |
| scry-pipe | ❌ 0 | ❌ not run | **Important** — IR deserialization, codegen |

**New fuzz targets:**
- `fuzz_cart_predict` — random FlatTree + random input (catches OOB in unsafe predict)
- `fuzz_pipeline_transform` — random pipeline JSON → engine transform
- `fuzz_ir_roundtrip` — random JSON → PipelineDef → JSON → assert equal
- `fuzz_scaler_chain` — random data through scaler pipeline

**Miri targets:**
- `cargo +nightly miri test -p scry-learn` — especially tree module
- `cargo +nightly miri test -p scry-pipe`

**CI update:** Add to `.github/workflows/ci.yml` Miri and fuzz jobs.

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

## Sprint 8.6 — Algorithm Visualization Gallery ⭐

> **Goal**: Ship a `.viz()` method on every scry-learn model that produces scry-chart outputs. Every algorithm family gets canonical visualizations.

**Priority: P1** | 3 sessions | ~9h

This is the dogfooding moment where learn + chart become inseparable.

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

### 9D. CUDA Acceleration for scry-learn (optional)

- New feature flag: `--features cuda`
- `crates/scry-learn/src/accel/cuda.rs` — cuBLAS matrix multiply wrapper
- Accelerates: linear regression fit, PCA eigendecomposition, KNN distance matrix
- Does NOT replace pure Rust by default — opt-in only
- **Target:** ≥5x speedup on datasets with >50K rows × >100 features

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

## Sprint 11 — Neural Networks & Deep Learning Basics

> **Goal**: MLP and basic neural network to close the biggest algorithm gap.

**Priority: P2** | 3 sessions | ~10h

### 11A. Multi-Layer Perceptron

- `MLPClassifier` / `MLPRegressor`
- Configurable hidden layers: `vec![64, 32, 16]`
- Activations: ReLU, Sigmoid, Tanh
- Optimizers: SGD (momentum), Adam
- Backpropagation with auto-diff (manual)
- **Target:** Competitive with sklearn MLPClassifier on MNIST subset

### 11B. ONNX Export (stretch)

- Export trained models to ONNX format for cross-framework interop
- Import ONNX models for prediction (limited op coverage)

---

## Sprint 12 — Streaming & Live Data

> **Goal**: Real-time charting and streaming data support.

**Priority: P2** | 3 sessions | ~9h | ⚠️ **Partially exists — needs audit**

> [!NOTE]
> Some streaming chart functionality may already exist in the codebase.
> Audit what's done vs what remains before starting new work.

### 12A. Streaming Charts

- `StreamingChart` with ring buffer backing
- Auto-scrolling time axis with tick anchoring and hysteresis
- Configurable window: last N points or last T seconds
- Append-only API: `chart.push(timestamp, value)`

### 12B. Live Data Sources

- stdin pipe: `cat /proc/stat | scry stream --chart line`
- WebSocket: `scry stream --ws ws://metrics:8080/cpu --chart gauge`

---

## Sprint 13 — Publication & Ecosystem

> **Goal**: Ship to crates.io and establish the ecosystem presence.

**Priority: P1** | 2 sessions | ~6h

### 13A. Pre-Publish Checklist

- [ ] Per-crate README.md (learn, pipe, cli) — currently engine and chart only
- [ ] `cargo publish --dry-run` for all 5 crates
- [ ] License headers in all source files
- [ ] API stability audit — mark `#[non_exhaustive]` on all public error types
- [ ] `CHANGELOG.md` audit — ensure all changes since 0.7.0 are captured
- [ ] `cargo deny check` — license and advisory audit

### 13B. The scry Handbook

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
| MLP + ONNX export | 0.11.0 | MNIST benchmark passes |
| Handbook published | 0.12.0 | mdBook deployed |
| API freeze + crates.io publish | **1.0.0** | SemVer enforced, public API committed |

---

## Execution Priority

```
8A. Benchmark Suite         ████████░░  P0  ← Session 1 DONE, 2 remaining
8B. Weakness Audit          ████████░░  P0  ← LogReg gap identified (fix in 8A-3)
8.5 3D Interactive Viz      ██████████  P0  ← DONE (8.5A-D complete)
8.6 Algorithm Viz Gallery   ░░░░░░░░░░  P1  ← .viz() on every model
8C. Ratatui Feature-Gating  ██████████  P1  ← DONE
9A-B. GPU (wgpu + 3D)       ██████████  P0  ← 9B+9C DONE, 9D in progress
9D. GPU Compute (learn)     ████░░░░░░  P0  ← wgpu matmul + distances
13A. Pre-Publish            ██████░░░░  P1
10A. PyO3 SDK               █████░░░░░  P1
10B-C. WASM + Polars        ███░░░░░░░  P1
11. Neural Networks         ███░░░░░░░  P2
12. Streaming               ██░░░░░░░░  P2  ← needs audit (may be partial)
```
