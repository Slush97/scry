# Agent Update Log

> Each agent appends a status update here when finished. Project manager reads this to track progress.

## Format

```
### Agent N — [name] — [status]
**Completed:** YYYY-MM-DD
**Files changed:** list
**Tests:** pass/fail count
**Notes:** anything the PM needs to know
```

---

## Batch 1 (Agents 1–4) — Committed `cb0b1ea`

### Agent 1 — DenseMatrix Core (12A) — DONE
**Completed:** 2026-02-16
**Files changed:** `src/matrix.rs` (NEW), `src/dataset.rs`, `src/lib.rs`
**Notes:** Contiguous column-major `Vec<f64>` with `(n_rows, n_cols)`. `col(j)` returns zero-cost `&[f64]` slice. `Dataset` stores `Option<DenseMatrix>` with `.matrix()` accessor.

### Agent 2 — CART Optimization (12.5) — DONE
**Completed:** 2026-02-16
**Files changed:** `src/tree/cart/builder.rs`
**Notes:** Replaced membership bitset with pre-filtered index arrays, incremental variance, buffer reuse.

### Agent 3 — SVD/QR Solvers (13) — DONE
**Completed:** 2026-02-16
**Files changed:** `src/linear/svd.rs` (NEW), `src/linear/qr.rs` (NEW), `src/linear/regression.rs`, `src/linear/mod.rs`
**Notes:** Golub-Kahan SVD + Householder QR. `Solver::Svd` / `Solver::Qr` / `Solver::Normal` selection.

### Agent 4 — Large File Refactoring (8C) — DONE
**Completed:** 2026-02-16
**Files changed:** `src/tree/cart/` (directory split), `src/search/` (directory split)
**Notes:** cart.rs → cart/{mod,node,flat,builder}.rs. search.rs → search/{mod,grid,random,tunable}.rs. Zero public API changes.

## Batch 2 (Agents 5–8) — Committed `e63fee5`

### Agent 5 — Model Migration (12B) — DONE
**Completed:** 2026-02-16
**Files changed:** `src/preprocess/pca.rs`, `src/preprocess/scaler.rs`, `src/accel/cpu.rs`, `src/linear/regression.rs`, `src/naive_bayes/gaussian.rs`, + others
**Notes:** Migrated PCA, scalers, linear, NB to use `data.matrix().col(j)` instead of `data.features[j]`.

### Agent 6 — Chart Fixes — DONE
**Completed:** 2026-02-16
**Files changed:** `crates/scry-chart/src/chart/line.rs`, `crates/scry-chart/src/layout/common_overlays.rs`, snapshots
**Notes:** Added missing `margin()`/`y_inverted()` to LineChart builder. Fixed axis label/tick collision spacing.

### Agent 7 — Sparse Matrix Types (14A) — DONE
**Completed:** 2026-02-16
**Files changed:** `src/sparse.rs` (NEW, 866 lines), `src/lib.rs`
**Notes:** CsrMatrix + CscMatrix with full API: from_triplets, row/col views, dot_vec, CSR↔CSC conversion, arithmetic.

### Agent 8 — Pipe Fuzz + Miri — DONE
**Completed:** 2026-02-16
**Files changed:** `fuzz/fuzz_targets/fuzz_ir_roundtrip.rs` (NEW), `fuzz/fuzz_targets/fuzz_pipeline_transform.rs` (NEW), `fuzz/Cargo.toml`, `crates/scry-pipe/src/ir.rs`, `crates/scry-pipe/src/engine.rs`, `crates/scry-pipe/src/codegen.rs`
**Notes:** 2 fuzz targets for scry-pipe, PartialEq derives added, test coverage improvements.

## Batch 3 (Agents 9–12) — IN PROGRESS

### Agent 9 — Sparse-Aware Algorithms (14B) — DONE
**Completed:** 2026-02-16
**Files changed:** `src/linear/regression.rs`, `src/linear/logistic.rs`, `src/linear/lasso.rs`, `src/linear/elastic_net.rs`, `src/naive_bayes/gaussian.rs`, `src/naive_bayes/multinomial.rs`, `src/preprocess/scaler.rs`, `src/neighbors/knn.rs`
**Tests:** 16 new sparse tests, all passing. 425 total lib tests pass.
**Notes:** Added `fit_sparse(CscMatrix)` + `predict_sparse(CsrMatrix)` to: LinearRegression (sparse XᵀX via column scatter), LogisticRegression (sparse GD), LassoRegression (sparse coord descent with residual tracking), ElasticNet (same), GaussianNB (zero-entry-aware mean/var), MultinomialNB (sparse count accumulation). Added `fit_sparse` + `transform_sparse` to StandardScaler (std-only, no centering for sparsity) and MinMaxScaler. Added `predict_sparse` to KnnClassifier and KnnRegressor (densify-then-brute-force). All round-trip parity tests verify sparse matches dense results. Also fixed pre-existing `let rng` → `let mut rng` in logistic partial_fit test.

### Agent 10 — Sparse Dataset Integration (14C) — DONE
**Completed:** 2026-02-16
**Files changed:** `src/dataset.rs`, `tests/sparse_dataset.rs` (NEW)
**Tests:** 14 unit tests (8 new sparse tests) + 2 integration tests, all passing
**Notes:** Added `Storage` enum (Dense/Sparse), `from_sparse()` constructor, `is_sparse()`/`sparse_csc()`/`sparse_csr()`/`ensure_dense()` accessors, sparse-aware `subset()` (train_test_split works automatically). Used `CscMatrix::from_dense` for subset to work around a pre-existing dedup bug in `CscMatrix::from_triplets` (cross-row merge in CSR builder). Skipped model dispatch wiring (Agent 9 territory). Skipped CSV auto-detection (stretch goal, not worth the complexity).

### Agent 11 — Large-Scale Benchmarks (12C) — DONE
**Completed:** 2026-02-16
**Files changed:** `benches/scaling_benchmark.rs` (NEW, 237 lines), `Cargo.toml`
**Tests:** 6 benchmark test-runs pass (100k tier). 1M and 10K×10K tiers compile and are available for full runs.
**Notes:** 4 benchmark groups: PCA transform scaling (100K/1M/10K×10K), LinearRegression fit scaling (same tiers), tree fit scaling (DT + RF at 100K/1M), throughput metrics (bytes/sec for PCA fit, rows/sec for LinReg predict). All use `fastrand::Rng::with_seed(42)` for deterministic data. 0 clippy warnings.

### Agent 12 — partial_fit API (15A) — DONE
**Completed:** 2026-02-16
**Files changed:** `src/partial_fit.rs` (NEW), `src/lib.rs`, `src/linear/logistic.rs`, `src/naive_bayes/gaussian.rs`, `src/cluster/mini_batch_kmeans.rs`, `src/neural/classifier.rs`, `src/neural/regressor.rs`, `tests/partial_fit_streaming.rs` (NEW)
**Tests:** 13 new unit tests + 3 integration tests, all passing. Full suite passes (425+ tests).
**Notes:** Created `PartialFit` trait with `partial_fit(&mut self, &Dataset)` and `is_initialized()`. Implemented for 5 models: LogisticRegression (one-pass GD per batch, dynamic class growth), GaussianNb (sufficient statistics accumulation with dynamic class growth), MiniBatchKMeans (streaming centroid updates), MLPClassifier (one epoch mini-batch SGD), MLPRegressor (one epoch mini-batch SGD with MSE loss). Integration test simulates 10K samples in 10 streaming batches. No conflicts with Agent 9's sparse additions — all PartialFit impls are separate impl blocks.

## Batch 4 (Agents 13–16) — IN PROGRESS

### Agent 13 — Sparse Polish (bug fix + dispatch + KNN) — DONE
**Completed:** 2026-02-16
**Files changed:** `src/sparse.rs`, `src/linear/regression.rs`, `src/linear/logistic.rs`, `src/linear/lasso.rs`, `src/linear/elastic_net.rs`, `src/naive_bayes/gaussian.rs`, `src/preprocess/scaler.rs`, `src/neighbors/knn.rs`
**Tests:** 12 new tests added. 437 lib unit tests + 221 integration tests pass. 0 new clippy warnings.
**Notes:**
1. **Fixed from_triplets dedup bug:** The dedup merge in `CsrMatrix::from_triplets` checked `final_indices.last()` globally across all rows. When row i-1 ended with col X and row i started with col X, they merged across row boundaries. Fixed by tracking `row_start` position so dedup only happens within each row.
2. **Auto-dispatch in fit():** Added `if let Some(csc) = data.sparse_csc() { return self.fit_sparse(...) }` at the top of `fit()` for LinearRegression, LogisticRegression, LassoRegression, ElasticNet, GaussianNb, StandardScaler (Transformer::fit), MinMaxScaler (Transformer::fit). Models now transparently use sparse paths when fed a sparse Dataset.
3. **True sparse KNN distance:** Replaced `sparse_row_to_dense()` densification with merge-join sparse distance functions: `sparse_euclidean_sq` (||a||²+||b||²-2a·b), `sparse_manhattan` (merge-join |a-b|), `sparse_cosine` (1-dot/norms). Added `Option<CsrMatrix>` field to KnnClassifier/KnnRegressor for sparse training data storage. KNN `fit()` now auto-stores CSR when given sparse Dataset. `predict_sparse()` uses `sparse_brute_force()` when sparse training data is available. Added `indices()` and `values()` public accessors to `SparseRow`.
4. **High-dimensional test:** 100×5000 matrix with 2% density completes without OOM (would require 400KB per query if densifying).

### Agent 14 — Predict Latency + Memory Benchmarks — DONE
**Completed:** 2026-02-16
**Files changed:** `benches/predict_latency.rs` (NEW, ~280 lines), `Cargo.toml` (added `[[bench]]` entry)
**Tests:** 27 benchmark test-runs pass (10 single-predict, 9 batch-predict, 8 model-memory).
**Notes:** 3 Criterion benchmark groups: (1) single-sample predict latency for 10 models (DT, RF, GBT, HistGBT, LinReg, LogReg, KNN, MLP, GaussianNB, LinearSVC), (2) batch predict throughput at 1K/10K/100K rows for RF, LinReg, MLP with `Throughput::Elements` reporting, (3) model memory footprint via `/proc/self/statm` RSS delta for RF, GBT, KNN, MLP at 10K/100K rows. All use `fastrand::Rng::with_seed(42)` deterministic data and `std::hint::black_box()`. 0 clippy warnings from benchmark file.

### Agent 15 — Pre-Publish Checklist (17A) — DONE
**Completed:** 2026-02-16
**Files changed:** `crates/scry-learn/README.md` (NEW), `crates/scry-pipe/README.md` (NEW), 189 `.rs` files (SPDX headers), `crates/scry-learn/Cargo.toml`, `crates/scry-chart/Cargo.toml`, `crates/scry-cli/Cargo.toml` (version + readme metadata), `CHANGELOG.md`, `crates/scry-chart/src/svg_export.rs` (wildcard match arms), 34 public enums across all crates (added `#[non_exhaustive]`).
**Tests:** Full workspace builds, all lib tests pass, no new clippy warnings.
**Notes:** SPDX `MIT OR Apache-2.0` headers added to all 189 production `.rs` files. Added `version` fields to path dependencies for publish compatibility. `cargo publish --dry-run` succeeds for scry-engine and scry-pipe; scry-chart/scry-learn/scry-cli fail only because upstream workspace deps aren't on crates.io yet (expected). Added `#[non_exhaustive]` to 34 public enums missing it (skipped scry-cli binary enums). CHANGELOG updated with scry-learn and scry-pipe entries covering Sprints 11-15A.

### Agent 16 — Streaming Charts (16A) — DONE
**Completed:** 2026-02-16
**Files changed:** `crates/scry-chart/src/streaming.rs` (NEW, ~320 lines), `crates/scry-chart/src/lib.rs`
**Tests:** 14 unit tests, all passing. Full scry-chart test suite passes.
**Notes:** Created `StreamingChart` with generic ring buffer (`RingBuffer<T>`), builder pattern (window_size, title, y_range, n_series, theme, labels), push/push_now/push_series API, multi-series support, auto-scrolling X-axis via snapshot→LineChart conversion. `snapshot()` produces a `Chart::Line` with the visible window. `render()` and `render_rgba()` produce PNG/RGBA output via existing export infrastructure. No existing streaming code found in codebase — built from scratch. No modifications to existing chart types or engine.
