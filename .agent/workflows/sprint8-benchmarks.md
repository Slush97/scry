---
description: Sprint 8 — Industry benchmark suite and competitive gap analysis workflow
---

# Sprint 8 — Competitive Hardening & Benchmarks

> **Goal**: Identify every weakness vs industry leaders. Prove parity or document gaps.

// turbo-all

## Session 8A: Cross-Competitor Benchmark Suite (3 sessions)

### Pre-requisites

Ensure Python environment has competitors installed:
```bash
pip install scikit-learn xgboost lightgbm numpy pandas
```

Ensure Rust competitors are in dev-dependencies:
```bash
# Verify these exist in crates/scry-learn/Cargo.toml [dev-dependencies]
# smartcore, linfa-trees, linfa-logistic, linfa-clustering, ndarray
```

---

### Session 8A-1: ML Accuracy & Speed Benchmarks

**Estimated effort:** 1 session (3-4 hours)

#### Context Files to Read
- `crates/scry-learn/benches/ml_algorithms.rs` — existing Criterion benchmarks
- `crates/scry-learn/benches/competitor_bench.rs` — existing competitor comparisons
- `crates/scry-learn/tests/benchmark_audit.rs` — existing 3-way timings
- `crates/scry-learn/tests/fixtures/` — UCI dataset fixtures

#### Step 1: Create Python Baseline Scripts

**New file:** `crates/scry-learn/benches/python/bench_sklearn.py`

Generate ground truth for accuracy and timing:
```python
# For each model × dataset combination:
# 1. 5-fold CV accuracy (mean ± std)
# 2. Training time (mean over 5 runs)
# 3. Prediction latency (p50, p95, p99 over 10K single-row predicts)
# 4. Model size (pickle bytes)
```

Datasets: iris, wine, breast_cancer, digits, california_housing
Models: DecisionTree, RandomForest, GradientBoosting, HistGBT, LogisticRegression, KNN, LinearSVC, KMeans, GaussianNB

**New file:** `crates/scry-learn/benches/python/bench_xgboost.py`

Head-to-head with HistGBT:
- Accuracy on larger datasets (covertype, higgs subset)
- Training throughput (rows/sec)
- Prediction latency

**New file:** `crates/scry-learn/benches/python/bench_lightgbm.py`

Same as XGBoost comparison.

#### Step 2: Create Rust Benchmark Suite

**New file:** `crates/scry-learn/benches/industry_benchmark.rs`

```rust
// Criterion groups:
// - "accuracy_parity/{model}_{dataset}" — 5-fold CV, compare to sklearn JSON
// - "training/{model}_{size}" — wall clock at 1K/10K/100K rows
// - "prediction/{model}" — single-row latency
// - "memory/{model}_{size}" — peak RSS measurement
// - "scaling/{model}" — time vs N plot data
```

#### Step 3: Create Comparison Chart Generator

**New file:** `crates/scry-learn/examples/industry_report.rs`

Uses scry-chart to generate comparison charts from benchmark results:
- Grouped bar charts: scry vs sklearn vs linfa vs smartcore
- Scaling curves: time vs N_samples
- Latency distribution histograms

#### Verification
```bash
cargo bench --bench industry_benchmark -p scry-learn
python3 crates/scry-learn/benches/python/bench_sklearn.py
cargo run --example industry_report -p scry-learn --release
```

---

### Session 8A-2: Chart & Engine Benchmarks

**Estimated effort:** 1 session (2-3 hours)

#### Context Files to Read
- `crates/scry-chart/src/layout/mod.rs` — rendering pipeline
- `src/rasterize/skia.rs` — rasterization
- `crates/scry-chart/tests/render_tests.rs` — existing render tests

#### Step 1: Chart Render Benchmarks

**New file:** `crates/scry-chart/benches/render_benchmark.rs`

```rust
// For each of the 19 chart types:
// - Time to render at 100, 1K, 10K, 100K data points
// - PNG export time at 800x600, 1920x1080, 3840x2160
// - SVG export time + file size
```

#### Step 2: Engine Throughput Benchmarks

**New file:** `benches/engine_throughput.rs`

```rust
// - FPS: render loop at 800x600, 1920x1080, 3840x2160
// - Dirty tile efficiency: % tiles skipped during animation
// - Protocol overhead: bytes/frame for Kitty, Sixel, iTerm2
// - Shape throughput: shapes/sec at 100, 1K, 10K shapes
```

#### Step 3: Feature Matrix

**New file:** `BENCHMARKS.md`

Publish all results in a structured markdown document with:
- Performance tables
- Feature coverage matrix (us vs every competitor)
- Charts (embedded PNG from our own library)

#### Verification
```bash
cargo bench --bench render_benchmark -p scry-chart
cargo bench --bench engine_throughput
```

---

### Session 8A-3: HistGBT Deep Dive vs XGBoost/LightGBM

**Estimated effort:** 1 session (2-3 hours)

Focus on the flagship algorithm that differentiates scry-learn.

#### Benchmarks
- **Accuracy:** 5-fold CV on Higgs (100K subset), Covertype, airline-delays
- **Training speed:** rows/sec at 10K, 100K, 1M
- **Prediction:** single-row and batch (1K rows)
- **Feature handling:** categorical features, missing values
- **Hyperparameter sensitivity:** learning_rate × n_estimators × max_depth grid

#### Deliverable
Add HistGBT section to `BENCHMARKS.md` with head-to-head tables and charts.

---

## Session 8B: Weakness Audit & Gap Analysis (2 sessions)

### Session 8B-1: scry-learn Gap Analysis

**Estimated effort:** 1 session (3 hours)

#### Method
1. Read scikit-learn's full API reference page-by-page
2. For every module/class in sklearn, check if scry-learn has an equivalent
3. Rate each gap: Critical (must have for 1.0) / Important / Nice-to-have

#### Audit Categories

```
sklearn.linear_model      vs  scry-learn/src/linear/
sklearn.tree              vs  scry-learn/src/tree/
sklearn.ensemble          vs  scry-learn/src/tree/ + src/ensemble/
sklearn.svm               vs  scry-learn/src/svm/
sklearn.neighbors          vs  scry-learn/src/neighbors/
sklearn.cluster           vs  scry-learn/src/cluster/
sklearn.naive_bayes       vs  scry-learn/src/naive_bayes/
sklearn.neural_network    vs  (MISSING)
sklearn.preprocessing     vs  scry-learn/src/preprocess/
sklearn.model_selection   vs  scry-learn/src/search.rs + src/split.rs
sklearn.metrics           vs  scry-learn/src/metrics/
sklearn.pipeline          vs  scry-learn/src/pipeline.rs
sklearn.decomposition     vs  scry-learn/src/preprocess/pca.rs
sklearn.feature_selection vs  scry-learn/src/feature_selection.rs
sklearn.manifold          vs  (MISSING — t-SNE, UMAP)
sklearn.mixture           vs  (MISSING — GMM)
sklearn.calibration       vs  (MISSING — CalibratedClassifierCV)
```

#### Deliverable
**New file:** `crates/scry-learn/GAP_ANALYSIS.md`

Table format: `| sklearn class | scry-learn equivalent | Status | Priority | Effort |`

---

### Session 8B-2: Chart & Engine Gap Analysis

**Estimated effort:** 1 session (2 hours)

#### Method
1. Walk through matplotlib, Plotly, D3.js feature lists
2. For every chart type and feature, check scry-chart coverage
3. Walk through terminal graphics competitors feature lists

#### Deliverables
- Add chart gaps to `GAP_ANALYSIS.md`
- Add engine gaps to `GAP_ANALYSIS.md`
- Prioritized action items for each gap

---

## Verification Commands (all sessions)

```bash
# Run all existing tests (must not regress)
cargo test -p scry-learn
cargo test -p scry-chart
cargo test --workspace

# Clippy
cargo clippy --workspace -- -D warnings

# Benchmarks
cargo bench --bench industry_benchmark -p scry-learn
cargo bench --bench render_benchmark -p scry-chart
cargo bench --bench engine_throughput
```
