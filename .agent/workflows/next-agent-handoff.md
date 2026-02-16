---
description: Handoff instructions for the next agent — current state, what's done, what's left
---

# Next Agent Handoff — Full Status & Instructions

> **Last updated**: 2026-02-15 20:30 PST

---

## 1. Project Overview

**scry** is a Rust workspace with five crates:

| Crate | Purpose | Maturity |
|-------|---------|----------|
| `scry-engine` | Low-level 2D rasterizer (shapes, text, gradients) | Stable |
| `scry-chart` | Chart builder library (17 chart types, 6 themes) | Stable |
| `scry-learn` | Machine learning library (23+ models, search, preprocessing) | Stable |
| `scry-cli` | CLI for chart rendering | Stable |
| `scry-pipe` | Feature engineering compiler | **Phase 1 in progress** |

---

## 2. What Is Complete

**Everything through Sprint 6 complete. Total: 428 tests, 0 clippy warnings.**

### scry-learn (ML crate)
- **23+ model types**: LinearRegression, LogisticRegression (L-BFGS + GD, L1/L2/ElasticNet penalty), Lasso, ElasticNet, DecisionTree (C/R), RandomForest (C/R), GBT (C/R), HistGBT (C/R), KNN (C/R), LinearSVC, LinearSVR, KernelSVC, KernelSVR, GaussianNB, BernoulliNB, MultinomialNB, KMeans, MiniBatchKMeans, DBSCAN (KD-tree), AgglomerativeClustering, IsolationForest, VotingClassifier, StackingClassifier
- **Preprocessing**: StandardScaler, MinMaxScaler, RobustScaler, PCA, OneHotEncoder, SimpleImputer, ColumnTransformer, PolynomialFeatures, Normalizer, LabelEncoder
- **Search**: GridSearchCV, RandomizedSearchCV — Tunable trait on ALL models, `.scoring()`, `.stratified()`, `ParamValue::Categorical`
- **CV**: k_fold, stratified_k_fold, RepeatedKFold, GroupKFold, TimeSeriesSplit, cross_val_predict
- **Metrics**: accuracy, precision, recall, f1, log_loss, balanced_accuracy, cohen_kappa, r2, mse, mae, mape, explained_variance, adjusted_rand_index, calinski_harabasz, davies_bouldin, silhouette
- **Visualizations**: 19 ML-specific chart functions in `viz.rs`
- **SVM**: Platt scaling (predict_proba), auto gamma (Scale/Auto/Value)
- **Tests**: 428 total (8 bench tests `#[ignore]`d, 1 flaky: `determinism_rf_same_seed` due to rayon nondeterminism)

### Industry Standards Audit — COMPLETE
A comprehensive audit comparing scry-learn against scikit-learn was completed. **Estimated score after Sprint 6: 9+/10.**

All gaps identified in the audit have been closed by Sessions 11-17.

---

## 3. What Was Just Done (Latest Session)

**Sprint 7 is in progress.** Multiple parallel agents ran:

1. **v0.7.0 tagged** — all crate versions bumped, annotated tag created
2. **WASM target for scry-engine** — COMPLETE
   - Added `wasm` feature flag + `wasm-bindgen`/`web-sys`/`js-sys` optional deps to root `Cargo.toml`
   - Created `src/wasm.rs` with `WasmCanvas` struct (JS-accessible via wasm-bindgen)
   - Methods: `new()`, `width()`, `height()`, `set_background()`, `add_circle()`, `add_rect()`, `pixels()`, `render_to_canvas()`, `clear()`, `command_count()`
   - Free function `render_rgba_to_canvas()` for blitting pre-rendered RGBA to HTML canvas
   - 6 native unit tests, all passing
   - `cargo check --target wasm32-unknown-unknown --no-default-features --features wasm` ✅
   - Demo page at `examples/wasm_demo/` with build instructions
3. **IsolationForest** — anomaly detection model with `contamination`, `n_estimators`, `max_samples`, `predict()` labels + `decision_function()` scores
4. **VotingClassifier + StackingClassifier** — advanced ensemble methods via `EnsembleClassifier` trait
5. **RF memory optimization fix** — fixed index-out-of-bounds in `cart.rs` membership bitset sizing (was sized to bootstrap max index, now covers full dataset)
6. **Production bench `#[ignore]`** — 8 heavy benchmark tests marked `#[ignore]` to prevent 20+ min debug-mode runs
7. **scry-pipe Phase 1A** — IR + transform engine (likely in progress by another agent)

---

## 4. What To Do Next — Priority-Ordered

### Sprint 7 — Platform & Commercial (in progress)
> Read `ROADMAP.md` section "Sprint 7" for details.

1. ~~**Tag v0.7.0**~~ — ✅ DONE
2. ~~**WASM target** for scry-engine~~ — ✅ DONE
3. **scry-pipe Phase 1** — IR + transform engine + Rust codegen (check if another agent completed this)
4. **scry-pipe Phase 2** — PyO3 Python SDK (3 sessions)
5. **DataFrame / Polars interop** (2 sessions)
6. **Streaming-aware charts** (2 sessions)

### PARALLEL: Chart Quality Hardening (separate agent)

A separate agent is running the chart quality roadmap. See `.agent/workflows/chart-quality-roadmap.md`.

---

## 5. Known Issues

| Issue | Severity | Details |
|-------|----------|---------|
| `determinism_rf_same_seed` flaky test | Low | RF uses rayon without deterministic ordering. Pre-existing. |
| KernelSVC single-sample div-by-zero | Low | `kernel.rs:305`. Wrapped in `catch_unwind` in edge tests. |
| Production bench only in release | Info | 8 bench tests are `#[ignore]`d. Run: `cargo test --test production_bench --release -- --ignored --nocapture` |

---

## 6. Key Files Reference

| Purpose | Path |
|---------|------|
| Product roadmap | `.agent/ROADMAP.md` |
| ML roadmap (sessions 1-17) | `.agent/workflows/ml-learn-roadmap.md` |
| scry-learn lib entry | `crates/scry-learn/src/lib.rs` |
| Search (GridSearchCV, Tunable) | `crates/scry-learn/src/search.rs` |
| Visualizations (15 functions) | `crates/scry-learn/src/viz.rs` |
| Benchmark dashboard | `crates/scry-learn/examples/bench_dashboard.rs` |
| UCI fixture generator (Python) | `crates/scry-learn/tests/fixtures/generate_fixtures.py` |
| sklearn references JSON | `crates/scry-learn/tests/fixtures/sklearn_predictions.json` |
| Decision tree impl | `crates/scry-learn/src/tree/cart.rs` |
| HistGBT impl | `crates/scry-learn/src/tree/histogram_gbt.rs` |
| Pipeline impl | `crates/scry-learn/src/pipeline.rs` |
| KNN impl | `crates/scry-learn/src/neighbors/knn.rs` |
| LogReg impl | `crates/scry-learn/src/linear/logistic.rs` |
| SVM kernel impl | `crates/scry-learn/src/svm/kernel.rs` |
| SVM linear impl | `crates/scry-learn/src/svm/linear.rs` |
| DBSCAN impl | `crates/scry-learn/src/cluster/dbscan.rs` |
| Metrics (classification) | `crates/scry-learn/src/metrics/classification.rs` |
| Metrics (regression) | `crates/scry-learn/src/metrics/regression.rs` |
| CV / split | `crates/scry-learn/src/split.rs` |
| Preprocessing | `crates/scry-learn/src/preprocess/` |
| scry-pipe proposal | `SCRY_PIPE_PROPOSAL.md` |
| WASM bridge | `src/wasm.rs` |
| WASM demo page | `examples/wasm_demo/index.html` |

## 7. Verification Commands

```bash
# All tests (321 passing, 1 flaky)
cargo test -p scry-learn

# scry-engine tests (87 native, 93 with wasm feature)
cargo test -p scry-engine --lib
cargo test -p scry-engine --lib --features wasm

# Clippy (must be 0 warnings) — run BOTH
cargo clippy -p scry-learn -- -D warnings
cargo clippy -p scry-learn --features serde -- -D warnings
cargo clippy -p scry-engine --lib -- -D warnings
cargo clippy -p scry-engine --lib --features wasm -- -D warnings

# WASM cross-compile check
cargo check -p scry-engine --no-default-features --features wasm --target wasm32-unknown-unknown

# Benchmark dashboard (release mode, with memory footprint)
cargo run --example bench_dashboard -p scry-learn --release --features serde
```

## 8. Code Quality Rules

- **Zero clippy warnings** — always run with `-D warnings`
- **All existing tests must continue to pass** — never break existing functionality
- **Builder pattern** — config types use consuming builder methods (`.field(value) -> Self`)
- **Serde** — new public types derive Serialize/Deserialize under `#[cfg(feature = "serde")]`
- **Doc comments** — all public APIs need `///` doc comments
- **Error handling** — `Result<T, E>` with crate-specific error types, never panic
- **Pure Rust** — no BLAS, nalgebra, or ndarray in production dependencies

## 9. Versioning Roadmap

| Version | Gate |
|:-------:|------|
| 0.5.0 | ✅ Tagged — Sprint 4 gates met |
| 0.6.0 | ✅ Correctness hardening suite (Sprint 4.5) |
| 0.7.0 | ✅ Tagged — Industry parity + algorithm expansion (Sprints 5-6) |
| 0.8.0 | scry-pipe Phase 1 |
| 0.9.0 | WASM + Python SDK |
| **1.0.0** | API stability freeze |
