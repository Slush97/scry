---
description: Handoff instructions for the next agent — current state, what's done, what's left
---

# Next Agent Handoff — Full Status & Instructions

> **Last updated**: 2026-02-15 02:00 PST

---

## 1. Project Overview

**scry** is a Rust workspace with five crates:

| Crate | Purpose | Maturity |
|-------|---------|----------|
| `scry-engine` | Low-level 2D rasterizer (shapes, text, gradients) | Stable |
| `scry-chart` | Chart builder library (17 chart types, 6 themes) | Stable |
| `scry-learn` | Machine learning library (19 models, search, preprocessing) | Stable |
| `scry-cli` | CLI for chart rendering | Stable |
| `scry-pipe` | Feature engineering compiler | **Empty stub** |

---

## 2. What Is Complete

**Everything through Sprint 6 complete. Total: 388 tests, 0 clippy warnings.**

### scry-learn (ML crate)
- **21+ model types**: LinearRegression, LogisticRegression (L-BFGS + GD, L1/L2/ElasticNet penalty), Lasso, ElasticNet, DecisionTree (C/R), RandomForest (C/R), GBT (C/R), HistGBT (C/R), KNN (C/R), LinearSVC, LinearSVR, KernelSVC, KernelSVR, GaussianNB, BernoulliNB, MultinomialNB, KMeans, MiniBatchKMeans, DBSCAN (KD-tree), AgglomerativeClustering
- **Preprocessing**: StandardScaler, MinMaxScaler, RobustScaler, PCA, OneHotEncoder, SimpleImputer, ColumnTransformer, PolynomialFeatures, Normalizer, LabelEncoder
- **Search**: GridSearchCV, RandomizedSearchCV — Tunable trait on ALL models, `.scoring()`, `.stratified()`, `ParamValue::Categorical`
- **CV**: k_fold, stratified_k_fold, RepeatedKFold, GroupKFold, TimeSeriesSplit, cross_val_predict
- **Metrics**: accuracy, precision, recall, f1, log_loss, balanced_accuracy, cohen_kappa, r2, mse, mae, mape, explained_variance, adjusted_rand_index, calinski_harabasz, davies_bouldin, silhouette
- **Visualizations**: 19 ML-specific chart functions in `viz.rs`
- **SVM**: Platt scaling (predict_proba), auto gamma (Scale/Auto/Value)
- **Tests**: 388 total (1 flaky: `determinism_rf_same_seed` due to rayon nondeterminism)

### Industry Standards Audit — COMPLETE
A comprehensive audit comparing scry-learn against scikit-learn was completed. **Estimated score after Sprint 6: 9+/10.**

All gaps identified in the audit have been closed by Sessions 11-17.

---

## 3. What Was Just Done (Latest Session)

1. **Verified full project state** — 388 tests passing, 0 clippy warnings
2. **Confirmed all Sprints 1-6 complete** including Sprint 4.5 correctness hardening
3. **Updated `ROADMAP.md`** — marked Sprints 4.5, 5, 6 as COMPLETE, consolidated versioning (v0.7.0 ready to tag)
4. **Updated priority matrix** — removed all completed items, Sprint 7 (Platform & Commercial) is next

---

## 4. What To Do Next — Priority-Ordered

### IMMEDIATE: Tag v0.7.0

All sprint gates through Sprint 6 are met. Tag the release.

### THEN: Sprint 7 — Platform & Commercial
> Read `ROADMAP.md` section "Sprint 7" for details.

1. **scry-pipe Phase 1** — IR + transform engine + Rust codegen (3 sessions)
2. **WASM target** for scry-engine (2 sessions)
3. **scry-pipe Phase 2** — PyO3 Python SDK (3 sessions)
4. **DataFrame / Polars interop** (2 sessions)
5. **Streaming-aware charts** (2 sessions)

### PARALLEL: Chart Quality Hardening (separate agent)

A separate agent is running the chart quality roadmap. See `.agent/workflows/chart-quality-roadmap.md`.

---

## 5. Known Issues

| Issue | Severity | Details |
|-------|----------|---------|
| `determinism_rf_same_seed` flaky test | Low | RF uses rayon without deterministic ordering. Pre-existing. |
| KernelSVC single-sample div-by-zero | Low | `kernel.rs:305`. Wrapped in `catch_unwind` in edge tests. |

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

## 7. Verification Commands

```bash
# All tests (321 passing, 1 flaky)
cargo test -p scry-learn

# Clippy (must be 0 warnings) — run BOTH
cargo clippy -p scry-learn -- -D warnings
cargo clippy -p scry-learn --features serde -- -D warnings

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
