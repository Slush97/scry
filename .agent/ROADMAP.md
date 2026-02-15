# scry — Product Roadmap

> **Updated**: 2026-02-15 | Sprints 1-6 complete | 388 tests, 0 clippy warnings

---

## Completed (Since Last Sync)

These items were on the previous roadmap and are now **done**:

| Item | Product | Status |
|------|---------|--------|
| Model serialization (serde on all types) | scry-learn | ✅ |
| Lasso + ElasticNet | scry-learn | ✅ |
| Hyperparameter search (GridSearchCV, RandomizedSearchCV) | scry-learn | ✅ |
| Feature selection (VarianceThreshold, SelectKBest) | scry-learn | ✅ |
| Class weights for imbalanced data | scry-learn | ✅ |
| KNN improvements (distance weighting, regressor, predict_proba) | scry-learn | ✅ |
| SVM (LinearSVC, LinearSVR, KernelSVC with SMO) | scry-learn | ✅ |
| Subplot / multi-panel layout (SubplotGrid, shared axes) | scry-chart | ✅ |
| Serde for chart configs | scry-chart | ✅ |
| Configurable DPI export | scry-chart | ✅ |
| Semantic zoom formatting | scry-chart | ✅ |
| CLI surface parity (17 chart types, 6 themes) | scry-cli | ✅ |
| CI/CD pipeline (GitHub Actions) | All | ✅ |
| RF oob_score real implementation | scry-learn | ✅ |
| Kill all clippy warnings (scry-learn: 67 → 0) | scry-learn | ✅ |
| README + stability map update | All | ✅ |
| SimpleImputer + RobustScaler + ColumnTransformer | scry-learn | ✅ |
| Tree pruning (ccp_alpha) + GBT loss functions (Huber, Quantile) | scry-learn | ✅ |
| Clustering (n_init, MiniBatchKMeans, silhouette) + BernoulliNB/MultinomialNB | scry-learn | ✅ |
| Benchmark expansion (all models, competitors, scaling) | scry-learn | ✅ |
| Histogram GBT (O(n) splits, LightGBM-class) | scry-learn | ✅ |
| Expand Tunable to all 19 model types (GridSearchCV universally usable) | scry-learn | ✅ |
| Golden reference tests (12 UCI dataset proofs) | scry-learn | ✅ |
| Multi-seed statistical robustness (13 tests) | scry-learn | ✅ |
| Edge case battery (37 degenerate input tests) | scry-learn | ✅ |
| Industry benchmark parity report (5-fold CV dashboard) | scry-learn | ✅ |
| Metrics expansion (log_loss, balanced_accuracy, cohen_kappa, MAPE, clustering metrics) | scry-learn | ✅ |
| Visualization expansion (validation_curve, partial_dependence, cv_boxplot, decision_boundary) | scry-learn | ✅ |
| LogReg L1 penalty + Penalty enum | scry-learn | ✅ |
| SVM completion (KernelSVR, Platt scaling, auto gamma) | scry-learn | ✅ |
| CV infrastructure (RepeatedKFold, GroupKFold, TimeSeriesSplit, cross_val_predict) | scry-learn | ✅ |
| Preprocessing expansion (PolynomialFeatures, Normalizer) | scry-learn | ✅ |
| AgglomerativeClustering + DBSCAN KD-tree optimization | scry-learn | ✅ |

---

## Priority Matrix (Current)

| Priority | What | Why | Product | Est. |
|:---:|------|-----|---------|:---:|
| **P0** | Tag v0.7.0 release | All sprint gates met through Sprint 6 | scry-learn | 1h |
| **P1** | Chart quality hardening (4 sessions) | Audit found 15 gaps vs D3/matplotlib | scry-chart | 4 sessions |
| **P2** | scry-pipe Phase 1 (IR + engine + codegen) | Highest commercial potential | scry-pipe | 3 sessions |
| **P2** | WASM rasterization target | Browser deployment | scry-engine | 2 sessions |
| **P3** | scry-pipe Phase 2 (PyO3 Python SDK) | Data scientist adoption | scry-pipe | 3 sessions |
| **P3** | Streaming-aware axis | Live data visualization | scry-chart | 2 sessions |
| **P3** | GPU rasterization | Performance ceiling lift | scry-engine | 3 sessions |
| **P3** | DataFrame abstraction | Polars interop | scry-learn | 2 sessions |
| **P3** | MLP / Neural network basics | Algorithm breadth | scry-learn | 2 sessions |
| **P3** | Isolation Forest | Anomaly detection | scry-learn | 1 session |
| **P3** | Stacking/Voting ensembles | Advanced ensemble methods | scry-learn | 1 session |

---

## Recommended Execution Order

### ~~Sprint 1 — Hardening~~ ✅ COMPLETE

> CI, clippy (0 warnings), OOB score, README update — all done.

### ~~Sprint 2 — ML Completeness (Sessions 6-9)~~ ✅ COMPLETE

> SimpleImputer, RobustScaler, ColumnTransformer, tree pruning, GBT losses,
> clustering improvements, NB variants, benchmark expansion — all done.

### ~~Sprint 3 — Differentiation (Session 10)~~ ✅ COMPLETE

> Histogram GBT with O(n) splits, binning, leaf-wise growth — done.

### ~~Sprint 4 — Features & Dashboard~~ ✅ COMPLETE

> Unique features + benchmark visibility.

11. ~~Expand `Tunable` to all model types~~ ✅
12. ~~Null/gap handling in line charts (GapPolicy: skip, interpolate, zero)~~ ✅
13. ~~Benchmark dashboard (warmup, ±σ, legend, GBT scry-only)~~ ✅
14. ~~L-BFGS optimizer for LogisticRegression (Solver enum, default)~~ ✅ (bonus)
15. ~~Bar chart legend rendering for multi-series~~ ✅ (bonus)

### ~~Sprint 4.5 — Correctness Hardening~~ ✅ COMPLETE

> Proved algorithmic correctness, robustness, and reliability.

- ~~4.5A. LogReg Optimization~~ ✅ — L-BFGS implemented with Solver enum
- ~~4.5B. Golden Reference Tests~~ ✅ — 12 tests against UCI datasets with sklearn JSON refs
- ~~4.5C. Multi-Seed Statistical Testing~~ ✅ — 5 multi-seed tests (RF, GBT, KMeans)
- ~~4.5D. Edge Case Battery~~ ✅ — 37 tests (empty, single-sample, NaN, Inf, extreme scales)
- ~~4.5E. Scaling & Overfitting Tests~~ ✅ — scaling, regularization, and overfitting proofs
- ~~4.5F. Convergence & Determinism Tests~~ ✅ — solver agreement, seed determinism
- ~~4.5G. Industry Benchmark Parity Report~~ ✅ — 5-fold CV parity table in dashboard

### ~~Sprint 5 — Industry Parity (Sessions 11-14)~~ ✅ COMPLETE

> Closed every gap identified in the industry standards audit.

- ~~Session 11: Metrics & Search~~ ✅ — log_loss, balanced_accuracy, cohen_kappa, clustering metrics, ParamValue::Categorical, stratified CV
- ~~Session 12: Visualization~~ ✅ — validation_curve, partial_dependence, cv_boxplot, decision_boundary
- ~~Session 13: Linear Models~~ ✅ — LogReg L1 penalty, Penalty enum, Ridge alias
- ~~Session 14: SVM Completion~~ ✅ — KernelSVR, Platt scaling, auto gamma

### ~~Sprint 6 — Algorithm Expansion (Sessions 15-17)~~ ✅ COMPLETE

> Filled remaining algorithm family gaps.

- ~~Session 15: CV Infrastructure~~ ✅ — RepeatedKFold, GroupKFold, TimeSeriesSplit, cross_val_predict
- ~~Session 16: Preprocessing~~ ✅ — PolynomialFeatures, Normalizer (LabelEncoder pre-existing)
- ~~Session 17: Clustering~~ ✅ — AgglomerativeClustering, DBSCAN KD-tree, DBSCAN predict()

### Sprint 7 — Platform & Commercial ⬅️ NEXT

> Goal: unique commercial features, broader platform reach

18. scry-pipe Phase 1 — IR + transform engine + Rust codegen
19. scry-pipe Phase 2 — PyO3 Python SDK
20. WASM target for scry-engine
21. DataFrame / Polars interop layer
22. Streaming-aware charts

---

## Versioning Strategy

| Milestone | Version | Gate |
|-----------|:-------:|------|
| ~~Hardening (CI, clippy, README, oob_score)~~ | ~~0.2.0~~ | ✅ All tests green, zero warnings |
| ~~ML Sessions 6-9 complete~~ | ~~0.3.0~~ | ✅ Full model inventory |
| ~~Histogram GBT~~ | ~~0.4.0~~ | ✅ Novel algorithm, 270+ tests |
| ~~Tunable + chart gaps + dashboard + L-BFGS~~ | ~~0.5.0~~ | ✅ All gates met |
| ~~Correctness hardening suite~~ | ~~0.6.0~~ | ✅ Golden refs match, edge case battery, multi-seed stats |
| ~~Industry parity + Algorithm expansion~~ | ~~0.7.0~~ | ✅ Tagged |
| scry-pipe Phase 1 ships | 0.8.0 | IR → Rust codegen e2e passing |
| WASM + Python SDK | 0.9.0 | Browser demo + pip install works |
| API stability freeze | **1.0.0** | Public API committed, SemVer enforced |
