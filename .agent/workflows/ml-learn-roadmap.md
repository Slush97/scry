---
description: Multi-session implementation roadmap for scry-learn ML library feature gaps
---

# scry-learn Feature Roadmap — Cross-Session Workflow

## Status

**Current Phase: ALL SESSIONS COMPLETE (1-17)** (2026-02-15)

| Session | Focus | Status |
|---------|-------|--------|
| 1 | Serialization + Lasso/ElasticNet | ✅ Complete |
| 2 | Hyperparameter Search + Feature Selection | ✅ Complete |
| 3 | class_weight + Imbalanced Data | ✅ Complete |
| 4 | KNN Improvements + KNN Regressor | ✅ Complete |
| 5 | SVM (Linear + Kernel) | ✅ Complete |
| 6 | Missing Preprocessing + Imputer | ✅ Complete |
| 7 | Tree Pruning + GBT Loss Functions | ✅ Complete |
| 8 | Clustering Improvements + NB Variants | ✅ Complete |
| 9 | Benchmark Expansion | ✅ Complete |
| 10 | Histogram GBT (novel in Rust ecosystem) | ✅ Complete |
| 11 | Metrics & Search Enhancements | ✅ Complete |
| 12 | Visualization Expansion | ✅ Complete |
| 13 | Linear Model Enhancements | ✅ Complete |
| 14 | SVM Completion | ✅ Complete |
| 15 | CV Infrastructure | ✅ Complete |
| 16 | Preprocessing Expansion | ✅ Complete |
| 17 | Clustering & DBSCAN Optimization | ✅ Complete |

## Progress Tracker

The canonical audit is at:
`<appDataDir>/brain/*/ml_audit.md` (latest conversation)

**Every agent working on this project MUST:**
1. Read this workflow file first
2. Read the session they're implementing IN FULL before writing any code
3. Run verification commands after each major change
4. Update the Status table after completing each session
5. Read Context Files listed below before touching any module

## Context Files to Read

### Core Architecture
- `crates/scry-learn/src/lib.rs` — Prelude re-exports, module structure
- `crates/scry-learn/src/error.rs` — `ScryLearnError` enum
- `crates/scry-learn/src/dataset.rs` — `Dataset` struct (column-major storage)
- `crates/scry-learn/src/pipeline.rs` — `Pipeline`, `PipelineModel`, `TransformerBox` traits
- `crates/scry-learn/Cargo.toml` — Dependencies and features

### Algorithm Modules (read the one you're modifying)
- `crates/scry-learn/src/tree/cart.rs` — CART with FlatTree (1,287 lines)
- `crates/scry-learn/src/tree/random_forest.rs` — RF classifier + regressor (513 lines)
- `crates/scry-learn/src/tree/gradient_boosting.rs` — GBT classifier + regressor (1,148 lines)
- `crates/scry-learn/src/linear/regression.rs` — Linear Regression with OLS (215 lines)
- `crates/scry-learn/src/linear/logistic.rs` — Logistic Regression (257 lines)
- `crates/scry-learn/src/neighbors/knn.rs` — KNN classifier brute-force (172 lines)
- `crates/scry-learn/src/cluster/kmeans.rs` — K-Means with k-means++ (285 lines)
- `crates/scry-learn/src/cluster/dbscan.rs` — DBSCAN (175 lines)
- `crates/scry-learn/src/naive_bayes/gaussian.rs` — Gaussian NB (170 lines)

### Preprocessing & Metrics
- `crates/scry-learn/src/preprocess/scaler.rs` — StandardScaler, MinMaxScaler
- `crates/scry-learn/src/preprocess/pca.rs` — PCA with Jacobi eigendecomposition
- `crates/scry-learn/src/preprocess/one_hot.rs` — OneHotEncoder
- `crates/scry-learn/src/split.rs` — train_test_split, cross_val_score, k_fold
- `crates/scry-learn/src/metrics/` — classification, regression, roc

### Tests & Benchmarks
- `crates/scry-learn/tests/correctness.rs` — sklearn correctness proofs (1399 lines)
- `crates/scry-learn/tests/benchmark_audit.rs` — 3-way competitor timing (547 lines)
- `crates/scry-learn/tests/cv_benchmark.rs` — Cross-validation benchmarks
- `crates/scry-learn/benches/ml_algorithms.rs` — Criterion benchmarks (557 lines)
- `crates/scry-learn/benches/competitor_bench.rs` — vs smartcore/linfa (267 lines)

## Verification Commands

// turbo-all

1. Run unit tests:
```bash
cargo test -p scry-learn --lib
```

2. Run integration tests:
```bash
cargo test -p scry-learn --test correctness
```

3. Run full test suite:
```bash
cargo test -p scry-learn
```

4. Run clippy:
```bash
cargo clippy -p scry-learn -- -D warnings
```

5. Workspace check:
```bash
cargo check --workspace
```

6. Run benchmarks (release mode):
```bash
cargo bench --bench ml_algorithms -p scry-learn
```

7. Run competitor audit (release mode):
```bash
cargo test --test benchmark_audit -p scry-learn --release -- --nocapture
```

---

## Session 1: Serialization + Lasso/ElasticNet

**Goal:** Enable model save/load and add L1/L1+L2 regularized linear models.

**Estimated effort:** 1 session (2-3 hours)

### 1A. Model Serialization (`serde` feature)

`serde` is already an optional dep in `Cargo.toml` with `features = ["derive"]`.

**Files to modify:**
- `src/tree/cart.rs` — Add `#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]` to `FlatNode`, `FlatTree`, `DecisionTreeClassifier`, `DecisionTreeRegressor`
- `src/tree/random_forest.rs` — Same for `RandomForestClassifier`, `RandomForestRegressor`, `MaxFeatures`
- `src/tree/gradient_boosting.rs` — Same for `GradientBoostingClassifier`, `GradientBoostingRegressor`
- `src/linear/regression.rs` — Same for `LinearRegression`
- `src/linear/logistic.rs` — Same for `LogisticRegression`
- `src/neighbors/knn.rs` — Same for `KnnClassifier`
- `src/cluster/kmeans.rs` — Same for `KMeans`
- `src/cluster/dbscan.rs` — Same for `Dbscan`
- `src/naive_bayes/gaussian.rs` — Same for `GaussianNb`
- `src/preprocess/scaler.rs` — Same for `StandardScaler`, `MinMaxScaler`
- `src/preprocess/pca.rs` — Same for `Pca`
- `src/preprocess/one_hot.rs` — Same for `OneHotEncoder`
- `src/dataset.rs` — Same for `Dataset`

**Note:** The `serde` derive only activates when users enable `serde` feature. No runtime cost otherwise.

**Tests to add:** A `tests/serialization.rs` with `#[cfg(feature = "serde")]`:
- Roundtrip test: train DT → serialize to JSON → deserialize → predict → assert same results
- Same for RF, LogReg, KMeans, Scaler
- Run with: `cargo test -p scry-learn --test serialization --features serde`

### 1B. Lasso Regression (L1 regularization via coordinate descent)

**New file:** `src/linear/lasso.rs`

**Implementation:**
- `LassoRegression` struct with `alpha`, `max_iter`, `tol` parameters
- Coordinate descent optimizer (standard algorithm)
- `fit()`, `predict()`, `coefficients()`, `intercept()` methods
- Sparse solution — coefficients driven exactly to zero

**New file:** `src/linear/elastic_net.rs`

**Implementation:**
- `ElasticNet` struct with `alpha`, `l1_ratio`, `max_iter`, `tol`
- When `l1_ratio=1.0` → pure Lasso, `l1_ratio=0.0` → pure Ridge
- Same coordinate descent with mixed penalty

**Modify:** `src/linear/mod.rs` — Add `pub mod lasso; pub mod elastic_net;`
**Modify:** `src/lib.rs` — Add to prelude

**Tests to add in `correctness.rs`:**
- Lasso on known coefficients (y = 2x₁ + 0x₂ + 3x₃ + 0x₄ + 1) — verify x₂,x₄ coefficients → 0
- ElasticNet on same data — verify sparse solution
- Run with: `cargo test -p scry-learn --test correctness -- prove_lasso`

### Session 1 Verification Checklist
- [x] `cargo test -p scry-learn --lib` passes
- [x] `cargo test -p scry-learn --test correctness` passes
- [x] `cargo test -p scry-learn --test serialization --features serde` passes
- [ ] `cargo clippy -p scry-learn -- -D warnings` clean
- [x] `cargo check --workspace` clean

---

## Session 2: Hyperparameter Search + Feature Selection

**Goal:** Add GridSearchCV, RandomizedSearchCV, and basic feature selectors.

**Estimated effort:** 1 session (2-3 hours)

### 2A. Hyperparameter Search

**New file:** `src/search.rs`

**Implementation:**
- `ParamGrid` — HashMap<&str, Vec<ParamValue>> where ParamValue is enum { Int(usize), Float(f64), Bool(bool) }
- `GridSearchCV` — exhaustive search over param grid + k-fold CV
  - `.fit(dataset)` → tries all combinations, returns best params + best score
  - `.best_params()`, `.best_score()`, `.cv_results()` accessors
  - Uses the existing `cross_val_score` infrastructure from `split.rs`
- `RandomizedSearchCV` — samples n_iter random combinations from param distributions
- Both should accept a `Scorer` (function pointer `fn(&[f64], &[f64]) -> f64`)

**Key design decision:** Need a `ModelFactory` trait or closure that builds a model from params. Suggest:
```rust
pub trait Tunable: PipelineModel {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()>;
    fn clone_with_params(&self) -> Box<dyn Tunable>;
}
```

**Modify:** `src/lib.rs` — Add `pub mod search;` and prelude exports

**Tests:**
- GridSearchCV on Decision Tree with max_depth ∈ {2,4,6,8} on Iris → assert best params found
- RandomizedSearchCV on RF with n_iter=10 → assert returns valid results
- Run with: `cargo test -p scry-learn --lib -- search::tests`

### 2B. Feature Selection

**New file:** `src/feature_selection.rs`

**Implementation:**
- `VarianceThreshold` — drops features with variance below threshold
  - `fit(dataset)` → compute variances
  - `transform(dataset)` → remove low-variance columns
  - Implements `Transformer` trait for pipeline integration
- `SelectKBest` — select top-k features by scoring function
  - Scoring functions: `f_classif` (ANOVA F-value), `mutual_info_classif` (mutual information)
  - `fit()` → compute scores, `transform()` → keep top-k
  - `get_support()` → boolean mask of selected features

**Modify:** `src/lib.rs` — Add `pub mod feature_selection;`

**Tests:**
- VarianceThreshold on data with one constant column → assert it's removed
- SelectKBest(k=2) on Iris → assert petal features rank highest (they have the most discriminative power)
- Pipeline: VarianceThreshold → StandardScaler → DecisionTree → assert works end-to-end
- Run with: `cargo test -p scry-learn --lib -- feature_selection::tests`

### Session 2 Verification Checklist
- [ ] `cargo test -p scry-learn --lib` passes
- [ ] `cargo test -p scry-learn --test correctness` passes (no regressions)
- [ ] `cargo clippy -p scry-learn -- -D warnings` clean
- [ ] `cargo check --workspace` clean

---

## Session 3: class_weight + Imbalanced Data Support

**Goal:** Add sample/class weighting to all classifiers.

**Estimated effort:** 1 session (1.5-2 hours)

### 3A. Weight Infrastructure

**New file:** `src/weights.rs`

**Implementation:**
- `ClassWeight` enum: `Uniform`, `Balanced`, `Custom(HashMap<usize, f64>)`
- `compute_sample_weights(targets: &[f64], class_weight: &ClassWeight) -> Vec<f64>`
  - `Balanced`: weight_c = n_samples / (n_classes * n_c) (sklearn formula)
  - `Custom`: user-specified per-class weights
  - Returns per-sample weight vector

### 3B. Integrate into Classifiers

**Files to modify:**
- `src/tree/cart.rs` — Accept `sample_weights: Option<&[f64]>` in fit, use in impurity calculation (weighted Gini/Entropy)
- `src/tree/random_forest.rs` — Pass `class_weight` to each tree's fit
- `src/tree/gradient_boosting.rs` — Weight gradient computation
- `src/linear/logistic.rs` — Weight loss function in gradient descent
- `src/naive_bayes/gaussian.rs` — Weight in mean/variance computation
- `src/neighbors/knn.rs` — Weight in majority vote

Each classifier gets a `.class_weight(ClassWeight)` builder method.

**Tests:**
- Train DT on imbalanced 90/10 dataset without weights → assert poor minority recall
- Train DT on same data with `Balanced` weights → assert improved minority recall
- Run with: `cargo test -p scry-learn --test correctness -- prove_class_weight`

### Session 3 Verification Checklist
- [ ] `cargo test -p scry-learn --lib` passes
- [ ] `cargo test -p scry-learn --test correctness` passes
- [ ] `cargo clippy -p scry-learn -- -D warnings` clean

---

## Session 4: KNN Improvements + KNN Regressor

**Goal:** Add distance-weighted voting, KNN Regressor, and predict_proba.

**Estimated effort:** 1 session (1.5-2 hours)

### 4A. KNN Enhancements

**Modify:** `src/neighbors/knn.rs`

**Implementation:**
- `WeightFunction` enum: `Uniform`, `Distance` (weight by 1/distance)
- `.weights(WeightFunction)` builder method
- `predict_proba()` — return class probability distribution
- Cosine distance metric option

### 4B. KNN Regressor

**New file or extend:** `src/neighbors/knn.rs` or `src/neighbors/knn_regressor.rs`

**Implementation:**
- `KnnRegressor` struct — same API shape as classifier
- Predicts mean (or distance-weighted mean) of k-nearest targets
- `fit()`, `predict()` methods

**Modify:** `src/neighbors/mod.rs` — export `KnnRegressor`
**Modify:** `src/lib.rs` — prelude

**Tests:**
- KNN classifier with distance weights on Iris → assert ≥90% accuracy
- KNN regressor on known linear function → assert R² > 0.9
- predict_proba on Iris → assert probabilities sum to 1.0
- Run with: `cargo test -p scry-learn --test correctness -- prove_knn`

### Session 4 Verification Checklist
- [ ] `cargo test -p scry-learn --lib` passes
- [ ] `cargo test -p scry-learn --test correctness` passes
- [ ] `cargo clippy -p scry-learn -- -D warnings` clean

---

## Session 5: SVM (Linear + Kernel)

**Goal:** Add Support Vector Machine with linear and RBF kernels.

**Estimated effort:** 1 long session (3-4 hours)

### 5A. Linear SVM

**New directory:** `src/svm/`
**New files:** `src/svm/mod.rs`, `src/svm/linear.rs`

**Implementation:**
- `LinearSVC` — Linear Support Vector Classifier
  - Hinge loss with L2 penalty
  - Stochastic Gradient Descent solver (Pegasos algorithm or similar)
  - Parameters: `C` (regularization), `max_iter`, `tol`
  - `fit()`, `predict()`, `decision_function()` methods
- `LinearSVR` — Linear Support Vector Regressor
  - Epsilon-insensitive loss
  - Same SGD solver

### 5B. Kernel SVM (stretch goal)

**New file:** `src/svm/kernel.rs`

**Implementation:**
- `KernelSVC` with `Kernel` enum: `Linear`, `RBF { gamma }`, `Polynomial { degree, coef0 }`
- Sequential Minimal Optimization (SMO) solver
- This is the most complex algorithm — may need to defer to a follow-up session

**Tests:**
- LinearSVC on Iris → assert ≥85% accuracy
- LinearSVC on XOR-like data → assert poor (validates linear limitation)
- RBF SVC on XOR-like data → assert good (validates kernel power)
- Run with: `cargo test -p scry-learn --test correctness -- prove_svm`

### Session 5 Verification Checklist ✅ Complete
- [x] `cargo test -p scry-learn --lib` passes (137 tests)
- [x] `cargo test -p scry-learn --test correctness` passes (22 proofs)
- [x] No SVM-specific clippy warnings
- [x] LinearSVC correctness proof passes on Iris (≥85%)
- [x] LinearSVC XOR negative proof (≤60%)
- [x] KernelSVC RBF XOR proof (≥90%)

---

## Session 6: Missing Preprocessing + Imputer

**Goal:** Add SimpleImputer, RobustScaler, and ColumnTransformer.

**Estimated effort:** 1 session (2 hours)

### 6A. SimpleImputer

**New file:** `src/preprocess/imputer.rs`

**Implementation:**
- `SimpleImputer` with `Strategy` enum: `Mean`, `Median`, `MostFrequent`, `Constant(f64)`
- `fit()` → compute fill values per feature
- `transform()` → replace NaN with fill values
- Implements `Transformer` trait

### 6B. RobustScaler

**Modify:** `src/preprocess/scaler.rs`

**Implementation:**
- `RobustScaler` — uses median and IQR instead of mean/std
- Robust to outliers
- `fit()`, `transform()`, `inverse_transform()` methods
- Implements `Transformer` trait

### 6C. ColumnTransformer (if time permits)

**New file:** `src/preprocess/column_transformer.rs`

**Implementation:**
- `ColumnTransformer` — apply different transformers to different column subsets
- `add_transformer(cols: &[usize], transformer: impl Transformer)` builder
- `fit()` → fit each sub-transformer on its columns
- `transform()` → apply each, concatenate results

**Tests:**
- SimpleImputer: dataset with NaN values → assert filled correctly
- RobustScaler: dataset with outliers → assert median/IQR normalization
- ColumnTransformer: scale cols 0-1, one-hot col 2 → assert combined output
- Run with: `cargo test -p scry-learn --lib -- preprocess::tests`

### Session 6 Verification Checklist ✅ Complete
- [x] `cargo test -p scry-learn --lib` passes (149 tests)
- [x] `cargo test -p scry-learn --test correctness` passes (25 proofs)
- [x] No Session-6-specific clippy warnings

---

## Session 7: Tree Pruning + GBT Loss Functions

**Goal:** Add cost-complexity pruning to DT and loss function selection to GBT.

**Estimated effort:** 1 session (2-3 hours)

### 7A. Cost-Complexity Pruning

**Modify:** `src/tree/cart.rs`

**Implementation:**
- `.ccp_alpha(f64)` builder method
- Post-training minimal cost-complexity pruning (MCCP):
  1. Compute effective alpha for each internal subtree
  2. Prune subtrees where effective_alpha < ccp_alpha
  3. Rebuild FlatTree after pruning
- `.cost_complexity_pruning_path()` → returns (alphas, impurities) for elbow selection

### 7B. GBT Loss Functions

**Modify:** `src/tree/gradient_boosting.rs`

**Implementation:**
- `Loss` enum for regression: `SquaredError`, `AbsoluteError`, `Huber { alpha }`, `Quantile { alpha }`
- Each loss implements: `negative_gradient()`, `initial_prediction()`, `update_terminal_regions()`
- `.loss(Loss)` builder method
- `Huber` loss: combines squared error for small residuals, linear for large (robust to outliers)
- `Quantile` loss: for prediction intervals

**Tests:**
- DT with ccp_alpha=0.01 → assert shallower tree than unbounded
- GBT with Huber loss on data with outliers → assert lower MAE than squared error
- Run with: `cargo test -p scry-learn --test correctness -- prove_pruning`

### Session 7 Verification Checklist
- [ ] `cargo test -p scry-learn --lib` passes
- [ ] `cargo test -p scry-learn --test correctness` passes
- [ ] `cargo clippy -p scry-learn -- -D warnings` clean

---

## Session 8: Clustering Improvements

**Goal:** Add n_init for K-Means, silhouette score, MiniBatchKMeans, and BernoulliNB/MultinomialNB.

**Estimated effort:** 1 session (2-3 hours)

### 8A. K-Means Improvements

**Modify:** `src/cluster/kmeans.rs`

- `.n_init(usize)` — run K-Means n_init times, keep best (lowest inertia)
- `.transform()` — return distance to each centroid (for pipeline integration)
- `silhouette_score()` function in `src/metrics/` or `src/cluster/`

### 8B. MiniBatchKMeans

**New file or extend:** `src/cluster/kmeans.rs`

- `MiniBatchKMeans` — use random mini-batches for centroid updates
- `batch_size` parameter
- Much faster on large datasets, slightly worse quality

### 8C. Naive Bayes Variants

**New files:** `src/naive_bayes/bernoulli.rs`, `src/naive_bayes/multinomial.rs`

- `BernoulliNB` — for binary/boolean features
- `MultinomialNB` — for count data (text classification)

**Tests:**
- K-Means n_init=10 → assert lower inertia than n_init=1
- Silhouette score on well-separated clusters → assert > 0.7
- BernoulliNB on binary features → assert reasonable accuracy
- Run with: `cargo test -p scry-learn --lib -- cluster::tests`

### Session 8 Verification Checklist ✅ Complete
- [x] `cargo test -p scry-learn --lib` passes (174 tests)
- [x] `cargo test -p scry-learn --test correctness` passes (31 proofs)
- [x] `cargo check --workspace` clean

---

## Session 9: Benchmark Expansion

**Goal:** Expand benchmarks to cover all new algorithms and add missing competitor comparisons.

**Estimated effort:** 1 session (2 hours)

### 9A. New Criterion Benchmarks

**Modify:** `benches/ml_algorithms.rs`

- Add Lasso, ElasticNet, SVM training/prediction benchmarks
- Add multiclass benchmark (10-class synthetic dataset)
- Add 50K/100K sample scaling test for DT and RF
- Add 100/500 feature scaling test

### 9B. New Competitor Benchmarks

**Modify:** `benches/competitor_bench.rs`

- Logistic Regression: scry vs smartcore vs linfa-logistic
- KNN: scry vs smartcore
- K-Means: scry vs linfa-clustering

### 9C. New Correctness Tests

**Modify:** `tests/correctness.rs`

- DBSCAN correctness (currently has NO test)
- Lasso sparsity proof
- SVM margin proof
- Serialization roundtrip (if not done in Session 1)

### 9D. Benchmark Audit Expansion

**Modify:** `tests/benchmark_audit.rs`

- Add Logistic Regression 3-way timing
- Add KNN 3-way timing
- Add K-Means 3-way timing

### Session 9 Verification Checklist
- [ ] `cargo bench --bench ml_algorithms -p scry-learn` runs successfully
- [ ] `cargo bench --bench competitor_bench -p scry-learn` runs successfully
- [ ] `cargo test --test benchmark_audit -p scry-learn --release -- --nocapture` passes
- [ ] `cargo test --test correctness -p scry-learn` passes

---

## Session 10: Histogram-Based Gradient Boosting (Novel in Rust)

**Goal:** Implement histogram-based GBT — O(n) split finding, 5-10x faster than current GBT on large datasets. Neither linfa nor smartcore has this.

**Estimated effort:** 2 sessions (4-6 hours)

### Why This Matters

Current GBT sorts features per split → O(n log n). Histogram GBT bins features into 256 uint8 bins → O(n) once for binning, O(256) per split. This is the core innovation behind XGBoost/LightGBM/CatBoost. Implementing it in pure Rust makes scry-learn the **only Rust library with production-grade boosting**.

### 10A. Feature Binning

**New file:** `src/tree/binning.rs`

**Implementation:**
- `FeatureBinner` — quantile-based binning into 256 bins (uint8)
- `fit()` → compute bin edges from training data
- `transform()` → map f64 features to u8 bin indices
- Handle missing values as a separate bin (bin 0 or bin 255)

### 10B. Histogram Accumulation

**New file:** `src/tree/histogram_gbt.rs`

**Implementation:**
- `Histogram` struct — gradient/hessian sums per bin (256 entries per feature)
- Histogram subtraction trick: parent - left = right (halves computation)
- `HistGradientBoostingClassifier` — leaf-wise tree growth
- `HistGradientBoostingRegressor` — same algorithm, regression losses
- Parameters: `max_bins` (default 256), `max_leaf_nodes`, `min_samples_leaf`, `learning_rate`, `n_estimators`, `max_iter`

### 10C. Optimizations

- **SIMD-friendly layout:** histograms are contiguous arrays, auto-vectorizable
- **Leaf-wise growth:** grow the leaf with the highest gain (LightGBM strategy)
- **Rayon parallelism:** bin features in parallel, build histograms in parallel

**Tests:**
- HistGBT on Iris → assert ≥95% accuracy
- HistGBT on 50K synthetic data → assert faster than standard GBT
- HistGBT with missing values → assert correct handling
- Run with: `cargo test -p scry-learn --test correctness -- prove_hist_gbt`

**Benchmarks:**
- HistGBT vs standard GBT on 10K, 50K, 100K rows → expect 5-10x speedup
- HistGBT vs smartcore GBT (if available)
- Run with: `cargo bench --bench ml_algorithms -p scry-learn -- hist_gbt`

### Session 10 Verification Checklist
- [ ] `cargo test -p scry-learn --lib` passes
- [ ] `cargo test -p scry-learn --test correctness` passes
- [ ] `cargo clippy -p scry-learn -- -D warnings` clean
- [ ] Benchmark shows ≥5x speedup over standard GBT on 50K rows

---

## Session 11: Metrics & Search Enhancements (Sprint 5 — Industry Parity)

**Goal:** Add missing critical metrics and improve hyperparameter search flexibility.

**Estimated effort:** 1 session (2-3 hours)

**Context:** The industry standards audit scored metrics at 7/10 and search at 5/10. This session
closes the most impactful gaps: `log_loss` (required for probabilistic model scoring),
`balanced_accuracy` (imbalanced data), and `ParamValue::Categorical` (string hyperparameters
like `criterion="gini"|"entropy"`).

### 11A. Classification Metrics

**Modify:** `src/metrics/classification.rs`

**Implementation:**
- `log_loss(y_true, y_prob)` — negative log-likelihood for probability predictions
  - Accepts `&[f64]` true labels and `&[Vec<f64>]` probability vectors
  - Clip probabilities to `[eps, 1-eps]` to avoid log(0)
  - Return `ScoringFn`-compatible wrapper for GridSearchCV
- `balanced_accuracy(y_true, y_pred)` — macro recall (mean per-class recall)
  - For imbalanced datasets where accuracy is misleading
- `cohen_kappa_score(y_true, y_pred)` — inter-rater agreement coefficient
  - Accounts for chance agreement, ranges from -1 to 1

### 11B. Regression Metrics

**Modify:** `src/metrics/regression.rs`

**Implementation:**
- `explained_variance_score(y_true, y_pred)` — 1 - Var(residuals) / Var(y_true)
- `mean_absolute_percentage_error(y_true, y_pred)` — MAPE
  - Handle zero values gracefully (skip or epsilon)

### 11C. Clustering Metrics

**New file:** `src/metrics/clustering.rs`

**Implementation:**
- `adjusted_rand_index(labels_true, labels_pred)` — similarity between two clusterings
  - Adjusted for chance, range [-1, 1]
- `calinski_harabasz_score(features, labels)` — ratio of between/within cluster dispersion
- `davies_bouldin_score(features, labels)` — average similarity between clusters

**Modify:** `src/metrics/mod.rs` — export new submodule

### 11D. Search Enhancements

**Modify:** `src/search.rs`

**Implementation:**
- Add `ParamValue::Categorical(String)` variant to the `ParamValue` enum
  - Update `Display` impl
  - Update all `set_param` impls for models that accept categorical params (e.g., DecisionTreeClassifier for `criterion`)
- Add `.stratified(bool)` builder method to `GridSearchCV` and `RandomizedSearchCV`
  - When true, use `stratified_k_fold` instead of `k_fold` internally
  - Default: `false` (backward-compatible)

**Tests:**
- `log_loss` on perfect predictions → assert near 0
- `log_loss` on random predictions → assert > 0
- `balanced_accuracy` on imbalanced data → assert differs from raw accuracy
- `cohen_kappa` on perfect agreement → assert = 1.0
- `adjusted_rand_index` on identical clusterings → assert = 1.0
- Grid search with `stratified(true)` → assert runs without error
- Grid search with `Categorical("gini")` → assert DT criterion is set
- Run with: `cargo test -p scry-learn --lib -- metrics::tests`

### Session 11 Verification Checklist
- [ ] `cargo test -p scry-learn --lib` passes
- [ ] `cargo test -p scry-learn --test correctness` passes (no regressions)
- [ ] `cargo clippy -p scry-learn -- -D warnings` clean
- [ ] `cargo check --workspace` clean

---

## Session 12: Visualization Expansion (Sprint 5 — Industry Parity)

**Goal:** Add the 4 most impactful missing ML visualizations.

**Estimated effort:** 1 session (2-3 hours)

**Context:** viz.rs already has 15 functions (9/10 rating). These 4 additions complete the
sklearn/yellowbrick parity for model evaluation and interpretation.

### 12A. Validation Curve

**Modify:** `src/viz.rs`

**Implementation:**
- `validation_curve(param_name, param_range, train_scores, val_scores)` → `Chart`
  - Line chart: x = parameter values, y = mean score
  - Two series: "Training" and "Validation"
  - Similar to `learning_curve` but x-axis is a hyperparameter, not dataset size

### 12B. Partial Dependence Plot

**Modify:** `src/viz.rs`

**Implementation:**
- `partial_dependence_chart(feature_values, pdp_values, feature_name)` → `Chart`
  - Line chart showing marginal effect of one feature on prediction
  - User computes PDP values externally; viz function just renders
  - Optionally show ICE (Individual Conditional Expectation) lines

### 12C. CV Box Plot

**Modify:** `src/viz.rs`

**Implementation:**
- `cv_boxplot(model_names, cv_scores)` → `Chart`
  - Box plot comparing CV score distributions across models
  - Each model = one box with median, Q1, Q3, whiskers
  - Requires `BoxChart` type from scry-chart (verify it exists)

### 12D. Decision Boundary Chart

**Modify:** `src/viz.rs`

**Implementation:**
- `decision_boundary_chart(x, y, labels, predict_fn, resolution)` → `Chart`
  - Generate a mesh grid over the 2D feature space
  - Call `predict_fn` on each grid point to get class
  - Render as colored scatter with mesh background
  - Good for visual debugging of classifiers on 2D data

**Tests:**
- `validation_curve` with mock data → assert builds `Line` chart
- `partial_dependence_chart` with linear data → assert builds `Line` chart
- `cv_boxplot` with 3 models → assert correct chart type
- `decision_boundary_chart` with simple data → assert builds chart
- Run with: `cargo test -p scry-learn --lib -- viz::tests`

### Session 12 Verification Checklist
- [ ] `cargo test -p scry-learn --lib` passes
- [ ] `cargo clippy -p scry-learn -- -D warnings` clean
- [ ] `cargo check --workspace` clean

---

## Session 13: Linear Model Enhancements (Sprint 5 — Industry Parity)

**Goal:** Add L1 penalty for LogisticRegression and improve API consistency.

**Estimated effort:** 1 session (1.5-2 hours)

**Context:** LogReg scored 7/10 — missing L1 for sparse feature selection.
sklearn supports `penalty={'l1','l2','elasticnet','none'}` with solver-specific mechanics.
We add a `Penalty` enum and coordinate descent for L1.

### 13A. Penalty Enum + L1 Support

**Modify:** `src/linear/logistic.rs`

**Implementation:**
- Add `Penalty` enum: `None`, `L1`, `L2` (default), `ElasticNet(f64)` (l1_ratio)
- `.penalty(Penalty)` builder method
- L1 implementation: proximal gradient descent (ISTA)
  - After each gradient step, apply soft-thresholding
  - Or use coordinate descent on the logistic loss (sklearn's liblinear approach)
- L-BFGS solver only works with L2 or None (error on L1)
- GD solver works with all penalties
- `ElasticNet(l1_ratio)` mixes L1 and L2 penalties

### 13B. Ridge Type Alias

**New file or modify:** `src/linear/mod.rs`

**Implementation:**
- `pub type Ridge = LinearRegression;` — type alias with documentation
- Add doc comment explaining it's LinearRegression with alpha > 0
- Add `Ridge::new(alpha)` convenience constructor

### 13C. Naming Convention Documentation

**Modify:** Module-level documentation in `src/linear/mod.rs`

- Document that scry-learn uses `alpha` (regularization strength) convention
- Note the relationship: sklearn's `C = 1/alpha` for SVM and LogReg
- Add migration guide comment for sklearn users

**Tests:**
- LogReg L1 on data with irrelevant features → assert irrelevant coefficients → 0
- LogReg L2 on same data → assert non-zero coefficients for all features
- LogReg ElasticNet → assert sparse but not as sparse as pure L1
- Ridge alias → assert same behavior as LinearRegression(alpha=1.0)
- Run with: `cargo test -p scry-learn --test correctness -- prove_logistic_l1`

### Session 13 Verification Checklist
- [ ] `cargo test -p scry-learn --lib` passes
- [ ] `cargo test -p scry-learn --test correctness` passes
- [ ] `cargo clippy -p scry-learn -- -D warnings` clean

---

## Session 14: SVM Completion (Sprint 5 — Industry Parity)

**Goal:** Complete the SVM model family with kernel regression and probability estimates.

**Estimated effort:** 1 session (2-3 hours)

**Context:** SVM scored 6/10 — the weakest algorithm family. Missing KernelSVR
and predict_proba (Platt scaling). The SMO solver is already implemented for
KernelSVC; KernelSVR reuses it with epsilon-insensitive loss.

### 14A. KernelSVR

**New file:** `src/svm/kernel_svr.rs`

**Implementation:**
- `KernelSVR` struct — mirrors `KernelSVC` but for regression
- Epsilon-insensitive loss: L(y, f(x)) = max(0, |y - f(x)| - ε)
- SMO solver adapted for regression (dual variables α_i and α*_i)
- Parameters: `kernel`, `C`, `epsilon`, `tol`, `max_iter`
- `fit()`, `predict()` methods
- Reuse `Kernel` enum from `kernel.rs`

**Modify:** `src/svm/mod.rs` — export `KernelSVR`
**Modify:** `src/lib.rs` — add to prelude
**Modify:** `src/pipeline.rs` — impl `PipelineModel` for `KernelSVR`
**Modify:** `src/search.rs` — impl `Tunable` for `KernelSVR`

### 14B. Platt Scaling (predict_proba)

**Modify:** `src/svm/kernel.rs` and `src/svm/linear.rs`

**Implementation:**
- After fitting SVM, optionally fit a sigmoid (A, B) via Platt's method:
  - Minimize: -Σ [t_i log(p_i) + (1-t_i) log(1-p_i)]
  - Where p_i = 1 / (1 + exp(A * f(x_i) + B))
  - t_i = (y_i + 1) / 2 with label smoothing
- `.probability(true)` builder method enables Platt scaling during fit
- `predict_proba()` returns calibrated probabilities
- Only for binary classification initially

### 14C. Auto Gamma

**Modify:** `src/svm/kernel.rs`

**Implementation:**
- Add `Gamma` enum: `Scale`, `Auto`, `Value(f64)`
- `Scale` = 1.0 / (n_features * feature_variance) — sklearn default
- `Auto` = 1.0 / n_features
- `.gamma(Gamma)` builder method
- Compute during `fit()` when set to Scale or Auto

**Tests:**
- KernelSVR on linear data → assert R² > 0.8
- KernelSVR with RBF on nonlinear data → assert R² > 0.9
- predict_proba on KernelSVC → assert probabilities sum to 1.0
- predict_proba calibration → assert better than raw sigmoid on decision values
- Auto gamma → assert gamma = 1/n_features
- Run with: `cargo test -p scry-learn --test correctness -- prove_kernel_svr`

### Session 14 Verification Checklist
- [ ] `cargo test -p scry-learn --lib` passes
- [ ] `cargo test -p scry-learn --test correctness` passes
- [ ] `cargo clippy -p scry-learn -- -D warnings` clean
- [ ] `cargo check --workspace` clean

---

## Session 15: CV Infrastructure (Sprint 6 — Algorithm Expansion)

**Goal:** Add essential cross-validation strategies missing from `split.rs`.

**Estimated effort:** 1 session (1.5-2 hours)

**Context:** CV scored 8/10. Missing GroupKFold (data leakage prevention),
TimeSeriesSplit (temporal data), and cross_val_predict (OOF predictions).

### 15A. RepeatedKFold

**Modify:** `src/split.rs`

**Implementation:**
- `RepeatedKFold { n_splits, n_repeats, seed }` — repeat k-fold N times with different shuffles
- Returns `n_splits * n_repeats` (train, test) pairs
- Each repeat uses `seed + repeat_idx` for reproducibility
- `repeated_cross_val_score()` convenience function

### 15B. GroupKFold

**Modify:** `src/split.rs`

**Implementation:**
- `group_k_fold(data, groups, k)` → `Vec<(Dataset, Dataset)>`
- `groups: &[usize]` — group label per sample (e.g., patient ID)
- Ensure no group appears in both train and test within a fold
- Assign groups to folds round-robin by group index

### 15C. TimeSeriesSplit

**Modify:** `src/split.rs`

**Implementation:**
- `time_series_split(data, n_splits)` → `Vec<(Dataset, Dataset)>`
- Expanding window: fold k uses samples [0..i] for train, [i..j] for test
- No shuffling — respects temporal order
- Optional `max_train_size` to cap training set

### 15D. cross_val_predict

**Modify:** `src/split.rs`

**Implementation:**
- `cross_val_predict(model, data, k, seed)` → `Result<Vec<f64>>`
- Returns out-of-fold predictions for every sample
- Train on k-1 folds, predict on held-out fold, reassemble in original order

**Tests:**
- RepeatedKFold(5, 3) → assert 15 total folds
- GroupKFold with 3 groups → assert no group leakage
- TimeSeriesSplit → assert train indices always < test indices
- cross_val_predict → assert output length == n_samples
- Run with: `cargo test -p scry-learn --lib -- split::tests`

### Session 15 Verification Checklist
- [ ] `cargo test -p scry-learn --lib` passes
- [ ] `cargo test -p scry-learn --test correctness` passes (no regressions)
- [ ] `cargo clippy -p scry-learn -- -D warnings` clean

---

## Session 16: Preprocessing Expansion (Sprint 6 — Algorithm Expansion)

**Goal:** Add missing preprocessing transformers for feature engineering.

**Estimated effort:** 1 session (1.5-2 hours)

**Context:** Preprocessing scored 7/10. Missing PolynomialFeatures (interaction/polynomial
terms), Normalizer (row-wise L1/L2), and LabelEncoder.

### 16A. PolynomialFeatures

**New file:** `src/preprocess/polynomial.rs`

**Implementation:**
- `PolynomialFeatures` struct with `degree` (default 2), `interaction_only`, `include_bias`
- `fit()` → compute output feature count
- `transform()` → expand features to include polynomial combinations
  - Degree 2 with 3 features: [x1, x2, x3] → [1, x1, x2, x3, x1², x1·x2, x1·x3, x2², x2·x3, x3²]
  - `interaction_only=true` skips powers (x1², x2², x3²)
- Implements `Transformer` trait

### 16B. Normalizer

**New file or modify:** `src/preprocess/normalizer.rs`

**Implementation:**
- `Normalizer` struct with `norm` parameter: `L1`, `L2` (default), `Max`
- Row-wise normalization (each sample scaled independently)
- `L1`: divide by sum of absolute values
- `L2`: divide by Euclidean norm
- `Max`: divide by max absolute value
- Implements `Transformer` trait (fit is a no-op)

### 16C. LabelEncoder

**New file:** `src/preprocess/label_encoder.rs`

**Implementation:**
- `LabelEncoder` — maps categorical string labels to integer indices
- `fit(labels: &[String])` → build mapping
- `transform(labels: &[String]) -> Vec<f64>` → encode to integers
- `inverse_transform(encoded: &[f64]) -> Vec<String>` → decode back
- `OrdinalEncoder` — same but for feature columns, not target

### 16D. FunctionTransformer

**New file:** `src/preprocess/function_transformer.rs`

**Implementation:**
- `FunctionTransformer<F>` where `F: Fn(&mut Dataset) -> Result<()>`
- Wraps a user-defined closure as a `Transformer`
- fit is a no-op, transform calls the function
- Useful for custom preprocessing in pipelines

**Modify:** `src/preprocess/mod.rs` — export new modules
**Modify:** `src/lib.rs` — add to prelude

**Tests:**
- PolynomialFeatures(degree=2) on [[1,2],[3,4]] → verify expanded output
- PolynomialFeatures(interaction_only=true) → verify no self-powers
- Normalizer L2 → verify each row has unit norm
- LabelEncoder roundtrip → assert encode then decode = original
- FunctionTransformer(log) → assert all values log-transformed
- Pipeline: PolynomialFeatures → StandardScaler → DT → assert works
- Run with: `cargo test -p scry-learn --lib -- preprocess::tests`

### Session 16 Verification Checklist
- [ ] `cargo test -p scry-learn --lib` passes
- [ ] `cargo test -p scry-learn --test correctness` passes (no regressions)
- [ ] `cargo clippy -p scry-learn -- -D warnings` clean
- [ ] `cargo check --workspace` clean

---

## Session 17: Clustering & DBSCAN Optimization (Sprint 6 — Algorithm Expansion)

**Goal:** Add hierarchical clustering and fix DBSCAN's O(n²) scaling.

**Estimated effort:** 1 session (2-3 hours)

**Context:** Clustering scored 6/10 — the weakest category after SVMs.
DBSCAN's O(n²) neighbor computation limits usability to ~10K samples.
AgglomerativeClustering is a widely-used algorithm missing entirely.

### 17A. AgglomerativeClustering

**New file:** `src/cluster/agglomerative.rs`

**Implementation:**
- `AgglomerativeClustering` struct
- Parameters: `n_clusters`, `linkage` (Single, Complete, Average, Ward)
- `Linkage` enum controls inter-cluster distance measure:
  - `Single` = min distance between any pair
  - `Complete` = max distance between any pair
  - `Average` = mean distance between all pairs
  - `Ward` = minimize within-cluster variance (most common)
- Algorithm: bottom-up agglomerative (O(n² log n) with priority queue)
  1. Start with each sample as its own cluster
  2. Merge closest pair of clusters
  3. Repeat until n_clusters remain
- `fit()`, `labels()`, `n_clusters()` methods
- Return merge history as `children_` for dendrogram visualization

### 17B. DBSCAN Spatial Index

**Modify:** `src/cluster/dbscan.rs`

**Implementation:**
- Replace brute-force neighbor search with KD-tree lookup
- Reuse `KDTree` from `src/neighbors/knn.rs` (extract if needed)
- `eps_neighbors(point, eps)` → returns all points within eps radius
- Complexity: O(n log n) average case vs current O(n²)
- Fallback to brute-force for high-dimensional data (d > 20)

### 17C. DBSCAN Extensions

**Modify:** `src/cluster/dbscan.rs`

**Implementation:**
- `predict(features)` → assign new points to nearest core point's cluster or noise
- `metric` parameter — `DistanceMetric` enum (Euclidean, Manhattan, Cosine)
  - Reuse from `knn.rs` if already defined

**Modify:** `src/cluster/mod.rs` — export `AgglomerativeClustering`
**Modify:** `src/lib.rs` — add to prelude
**Modify:** `src/pipeline.rs` — impl `PipelineModel` for `AgglomerativeClustering` (if applicable)

**Tests:**
- Agglomerative on well-separated 3-cluster data → assert correct labels
- Different linkages → assert Ward gives lowest inertia-like score
- DBSCAN with KD-tree on 10K samples → assert same labels as brute-force
- DBSCAN predict on new points → assert assigned to nearest cluster
- DBSCAN with Manhattan metric → assert different eps behavior
- Run with: `cargo test -p scry-learn --lib -- cluster::tests`

**Benchmarks:**
- DBSCAN KD-tree vs brute-force on 5K, 10K, 20K rows → expect ≥3x speedup
- Run with: `cargo bench --bench ml_algorithms -p scry-learn -- dbscan`

### Session 17 Verification Checklist
- [ ] `cargo test -p scry-learn --lib` passes
- [ ] `cargo test -p scry-learn --test correctness` passes
- [ ] `cargo clippy -p scry-learn -- -D warnings` clean
- [ ] `cargo check --workspace` clean
- [ ] DBSCAN benchmark shows ≥3x speedup with KD-tree on 10K rows

---

## Code Quality Rules

All sessions must follow these rules:

1. **Documentation:** Every public type and function gets a doc comment with `# Examples` section
2. **Error handling:** Use `ScryLearnError` — add enum variants as needed in `error.rs`
3. **Builder pattern:** All model constructors use the fluent builder pattern (`.param(value)`)
4. **Trait implementation:** Models implement `PipelineModel`, preprocessors implement `Transformer`
5. **Prelude exports:** New public types added to `src/lib.rs` prelude
6. **Serde support:** New types get `#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]`
7. **No BLAS:** All implementations must be pure Rust (no nalgebra, ndarray in prod deps)
8. **Tests:** Every new feature gets at least one unit test AND one correctness proof

## Known Issues & Gotchas

- `Dataset.features` is `pub` — be careful not to assume invariants on feature dimensions
- `Pipeline` holds `Box<dyn TransformerBox>` — new transformers need `TransformerBox` impl
- `viz` module is now unconditionally compiled (was behind feature gate, fixed in Sessions 1-5)
- `ScoringFn` type is `fn(&[f64], &[f64]) -> f64` — `log_loss` needs a wrapper since it takes probabilities
- DBSCAN O(n²) neighbor search — **Session 17 fix target**

### Fixed in Sessions 1-5
- ~~`sigmoid()` in `logistic.rs` is dead code~~ → Removed
- ~~`euclidean_sq` is duplicated in `kmeans.rs` and `dbscan.rs`~~ → Extracted to `src/distance.rs`
- ~~`LinearRegression` doc claims gradient descent solver~~ → Fixed doc to say OLS only
- ~~Default build broken (viz feature gate)~~ → viz now unconditionally compiled
- ~~No SVM support~~ → LinearSVC, LinearSVR, KernelSVC implemented
- ~~No serialization support~~ → serde on all model types
- ~~No hyperparameter search~~ → GridSearchCV + RandomizedSearchCV
- ~~No feature selection~~ → VarianceThreshold + SelectKBest
- ~~No class weights~~ → ClassWeight integrated across all classifiers

### Fixed in Session 8
- ~~No n_init for K-Means~~ → n_init=10 default, best-of-N
- ~~No silhouette score~~ → silhouette_score + silhouette_samples in cluster module
- ~~No MiniBatchKMeans~~ → Implemented with streaming centroid updates
- ~~Only GaussianNB~~ → Added BernoulliNB + MultinomialNB
- ~~KMeans has no transform()~~ → Returns distance-to-centroid matrix

### Fixed in Sprint 4
- ~~oob_score field on RF is never computed~~ → Real OOB implementation
- ~~Tunable only for DT + RF~~ → Expanded to all 19 model types
- ~~No L-BFGS optimizer~~ → Solver enum with L-BFGS default for LogReg
