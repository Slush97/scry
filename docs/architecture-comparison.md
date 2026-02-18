# scry-learn Architecture: Comparison with linfa, smartcore, and scikit-learn

## 1. Data Representation

### 1.1 Primary Data Containers

| Library | Core Type | Backing Storage | Layout |
|---------|-----------|----------------|--------|
| **scry-learn** | `Dataset` | `Vec<Vec<f64>>` + `DenseMatrix` | Column-major |
| **linfa** | `DatasetBase<R, T>` | `ndarray::Array2<F>` | Row-major (C-order) |
| **smartcore** | `DenseMatrix<T>` | `Vec<T>` flat buffer | Row-major |
| **scikit-learn** | numpy `ndarray` | C-contiguous buffer | Row-major (C-order) |

#### scry-learn

```rust
// Column-major: features[feature_idx][sample_idx]
let data = Dataset::new(
    vec![
        vec![1.0, 2.0, 3.0],  // feature 0: 3 samples
        vec![4.0, 5.0, 6.0],  // feature 1: 3 samples
    ],
    vec![0.0, 1.0, 0.0],      // target
    vec!["x1".into(), "x2".into()],
    "y",
);
```

The `Dataset` struct bundles features, target, metadata (feature names, target name, class labels), and optional backing stores:
- `DenseMatrix`: contiguous column-major `Vec<f64>` (`data[col * n_rows + row]`) for cache-friendly column scans
- `row_major_cache`: lazily computed flat buffer for row-oriented algorithms (KNN, neural nets)
- `CscMatrix` / `CsrMatrix`: custom compressed sparse column/row formats

#### linfa

```rust
use linfa::prelude::*;
use ndarray::array;

let features = array![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]; // row-major
let targets = array![0.0, 1.0, 0.0];
let dataset = Dataset::new(features, targets);
```

`DatasetBase<R, T>` is generic over the record type `R` (typically `Array2<f64>`) and target type `T` (can be `f64`, `usize`, `bool`, or label types). Hard dependency on the `ndarray` crate.

#### smartcore

```rust
use smartcore::linalg::basic::matrix::DenseMatrix;

let x = DenseMatrix::from_2d_array(&[
    &[1.0, 4.0], &[2.0, 5.0], &[3.0, 6.0]  // row-major
]);
let y = vec![0.0, 1.0, 0.0];
```

`DenseMatrix<T>` is row-major (`data[row * n_cols + col]`). Also supports `ndarray::Array2` and `nalgebra::DMatrix` via the `BaseMatrix` trait abstraction. No built-in feature names or metadata.

#### scikit-learn

```python
import numpy as np
X = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])  # row-major
y = np.array([0.0, 1.0, 0.0])
```

No unified dataset struct. `X` and `y` are separate numpy arrays passed independently. `feature_names_in_` attribute populated after `fit()` (since v1.0).

### 1.2 Memory Layout Tradeoffs

scry-learn is the only library in this comparison that defaults to **column-major** storage. This is a deliberate architectural choice:

**Column-major advantages:**
- Tree split evaluation scans a single feature column — column-major keeps this contiguous in cache
- Coordinate descent (Lasso, ElasticNet) processes one feature at a time
- Variance/mean computation per feature is cache-friendly
- CSC sparse format aligns naturally with column-major dense layout

**Column-major disadvantages:**
- Row-oriented algorithms (KNN distance computation, neural net forward pass) require transposition
- mitigated by the lazy `row_major_cache` computed on first access via `flat_feature_matrix()`
- Prediction input is row-major (`&[Vec<f64>]`), creating an API asymmetry

**Row-major advantages (linfa, smartcore, sklearn):**
- Each sample is contiguous — natural for per-sample operations
- Consistent with numpy/BLAS conventions
- No transposition needed for sample-oriented access

Note: sklearn internally converts to Fortran-order (column-major) in many tree implementations for split finding, arriving at a similar layout to scry-learn's default.

### 1.3 Sparse Data Support

| Library | Sparse Formats | Implementation |
|---------|---------------|----------------|
| **scry-learn** | CSC, CSR | Custom (`sparse.rs`), zero external deps |
| **linfa** | None built-in | External `sprs` crate integration |
| **smartcore** | None | — |
| **scikit-learn** | CSR, CSC, COO, BSR, LIL, DOK | `scipy.sparse` |

scry-learn implements CSC and CSR from scratch:
- `CscMatrix`: compressed sparse column — used for fitting (column-oriented coordinate descent, tree splits)
- `CsrMatrix`: compressed sparse row — used for prediction (row-oriented dot products)
- `CscMatrix::from_dense()` and `CsrMatrix::from_dense()` for conversion
- Algorithms auto-dispatch to sparse kernels when `data.sparse_csc()` returns `Some`

### 1.4 Type System

| Library | Numeric Type | Flexibility |
|---------|-------------|-------------|
| **scry-learn** | `f64` only | Simplest API, no generic parameters |
| **linfa** | `F: Float` (f32/f64) | Compile-time float precision choice |
| **smartcore** | `T: RealNumber` | Generic over numeric types |
| **scikit-learn** | numpy dtype | Runtime dtype flexibility |

scry-learn's `f64`-only approach eliminates generic type parameters from all APIs, resulting in simpler function signatures and error messages. The tradeoff is no f32 support for memory-constrained or GPU-optimized workloads (though the wgpu GPU backend handles precision internally).

---

## 2. API Design Patterns

### 2.1 Model Lifecycle

#### scry-learn: Mutable self, builder pattern

```rust
let mut model = RandomForestClassifier::new()
    .n_estimators(100)
    .max_depth(10);

model.fit(&train)?;                           // mutates self
let preds = model.predict(&test_features)?;   // borrows self
```

- `fit(&mut self, data: &Dataset) -> Result<()>` — mutates the model in-place
- `predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>>` — immutable borrow
- Runtime `NotFitted` error if predict called before fit
- API input asymmetry: `fit` takes `&Dataset`, `predict` takes `&[Vec<f64>]` (row-major)

#### linfa: Type-state pattern, immutable models

```rust
let model = DecisionTree::params()
    .max_depth(Some(10))
    .fit(&dataset)?;              // returns NEW fitted model

let predictions = model.predict(&test);
```

- Unfitted params struct and fitted model are **different types** (e.g., `DecisionTreeParams` → `DecisionTree`)
- **Compile-time guarantee**: cannot call predict on unfitted model
- `Fit<R, T, E>` trait returns a new fitted model (value semantics)
- `Predict<R, T>` or `PredictInplace<R, T>` for inference

#### smartcore: Static method, no unfitted state

```rust
let model = RandomForestClassifier::fit(
    &x, &y, Default::default()
)?;                                // returns fitted model directly

let predictions = model.predict(&x_test)?;
```

- `Type::fit(&x, &y, params)` — static associated function, returns fitted model
- No separate unfitted state — model is either constructed (fitted) or doesn't exist
- Parameters passed as a struct (often `Default::default()`)

#### scikit-learn: Mutable self (similar to scry-learn)

```python
model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)       # mutates self, returns self
predictions = model.predict(X_test)
```

- `fit(X, y)` mutates in-place, returns `self` for chaining
- Runtime `NotFittedError` if predict called before fit

### 2.2 Comparison Summary

| Aspect | scry-learn | linfa | smartcore | scikit-learn |
|--------|-----------|-------|-----------|-------------|
| Fit mutates self | Yes | No (new type) | No (static method) | Yes |
| Compile-time fit check | No (`NotFitted` error) | Yes (type system) | Yes (no unfitted state) | No (exception) |
| Shared trait | `PipelineModel` (macro) | `Fit`/`Predict` | None | `BaseEstimator` (class) |
| Config style | Builder methods | `with_*` params | Params struct | Constructor kwargs |
| X/y bundling | `Dataset` (bundled) | `DatasetBase` (bundled) | Separate arrays | Separate arrays |

### 2.3 Error Handling

```rust
// scry-learn: typed error enum
#[non_exhaustive]
pub enum ScryLearnError {
    EmptyDataset,
    NotFitted,
    ShapeMismatch { expected: usize, got: usize },
    InvalidParameter(String),
    ConvergenceFailure(String),
    SchemaVersionMismatch { model: u32, current: u32 },
    // ...
}
```

| Library | Error Type | Pattern |
|---------|-----------|---------|
| **scry-learn** | `ScryLearnError` enum | `Result<T, ScryLearnError>` |
| **linfa** | Per-crate error types | `Result<T, linfa_trees::Error>` etc. |
| **smartcore** | `Failed` | `Result<T, Failed>` |
| **scikit-learn** | Python exceptions | `ValueError`, `NotFittedError` etc. |

scry-learn uses a single error enum across all algorithms, simplifying error handling in pipelines. linfa's per-crate errors are more granular but require mapping when combining algorithms from different crates.

### 2.4 Trait System

| Library | Trait Architecture |
|---------|-------------------|
| **scry-learn** | Concrete types. `PipelineModel` trait implemented via `impl_pipeline_model!` macro for all 21+ model types. `Transformer` trait for preprocessors. `PartialFit` for online learning. `Tunable` for hyperparameter search. |
| **linfa** | Rich trait hierarchy: `Fit`, `FitWith`, `Predict`, `PredictInplace`, `Transformer`. Enables generic pipeline composition. |
| **smartcore** | Concrete types with inherent methods. No shared trait hierarchy. |
| **scikit-learn** | Duck typing with `BaseEstimator` mixin providing `get_params()`/`set_params()`. `RegressorMixin`/`ClassifierMixin` add `score()`. |

---

## 3. Algorithm Implementation Comparison

### 3.1 Solver Choices

| Algorithm | scry-learn | linfa | smartcore | scikit-learn |
|-----------|-----------|-------|-----------|-------------|
| Linear Regression | Normal eq (Gauss-Jordan), QR, SVD, Auto | SVD | SVD | SVD, Cholesky |
| Ridge | Normal eq with $\alpha I$ | SVD with $\alpha I$ | SVD | SVD, Cholesky, sparse CG, SAG, SAGA, LSQR |
| Logistic Regression | L-BFGS (default), GD | IRLS | LBFGS | L-BFGS, liblinear, SAG, SAGA, newton-cg, newton-cholesky |
| Lasso | Coordinate descent | Coordinate descent | Coordinate descent | Coordinate descent (Cython) |
| ElasticNet | Coordinate descent | Coordinate descent | — | Coordinate descent (Cython) |
| SVM (kernel) | Simplified SMO | SMO (libsvm-like) | SMO | libsvm |
| SVM (linear) | Pegasos SGD | — | — | liblinear |
| Decision Tree | CART, pre-sorted indices | CART | CART | CART (Cython, pre-sorted) |
| Random Forest | Parallel CART (rayon) | CART ensemble | CART ensemble | Parallel CART (joblib + Cython) |
| Gradient Boosting | Sequential CART + Newton-Raphson leaves | — | — | Sequential CART, histogram-based |
| Hist. Gradient Boosting | Histogram-based (256 bins, GPU) | — | — | Histogram-based (Cython) |
| K-Means | Lloyd's + k-means++ | Lloyd's + k-means++ | Lloyd's + k-means++ | Lloyd's, Elkan, mini-batch |
| PCA | Jacobi rotation eigendecomposition | SVD (ndarray-linalg) | SVD | LAPACK SVD, randomized SVD |
| KNN | Brute + KD-tree | Ball tree, KD-tree | Brute, cover tree | Brute, ball tree, KD-tree |
| Gaussian NB | Single-pass mean/var | Incremental | Naive computation | Incremental |

### 3.2 Parallelism

| Library | Framework | Scope |
|---------|-----------|-------|
| **scry-learn** | `rayon` | Random Forest tree training, batch predictions, cross-validation, histogram construction |
| **linfa** | `rayon` (optional) | Select algorithms |
| **smartcore** | None | Single-threaded throughout |
| **scikit-learn** | `joblib` (process/thread) + OpenMP (Cython) | Most ensemble methods, cross-validation, KNN |

scry-learn's Random Forest uses rayon for parallel tree fitting with atomic OOB vote accumulation — each tree's OOB predictions are aggregated via `AtomicU64` arrays shared across threads.

### 3.3 GPU Acceleration

| Library | GPU Support | Technology |
|---------|-------------|------------|
| **scry-learn** | Feature-gated (`gpu`) | wgpu compute shaders |
| **linfa** | None | — |
| **smartcore** | None | — |
| **scikit-learn** | None native | cuML (RAPIDS) as separate library |

scry-learn's GPU acceleration is available for:
- `matmul`: Neural network forward/backward pass (dispatch threshold: `batch * max_dim >= 4096`)
- `xtx_xty`: Linear regression Gram matrix computation
- `pairwise_distances_squared`: KNN distance computation
- `build_histograms`: Histogram gradient boosting bin accumulation

The `ComputeBackend` trait abstracts CPU vs GPU, with `accel::auto()` selecting the backend at runtime.

---

## 4. Preprocessing Pipeline

### 4.1 Available Transformers

| Transformer | scry-learn | linfa | smartcore | scikit-learn |
|------------|-----------|-------|-----------|-------------|
| StandardScaler | Yes | Yes | Yes | Yes |
| MinMaxScaler | Yes | Yes | Yes | Yes |
| RobustScaler | Yes | — | — | Yes |
| Normalizer | Yes (L1/L2/Max) | — | — | Yes |
| PCA | Yes (Jacobi) | Yes (SVD) | Yes (SVD) | Yes (LAPACK/randomized) |
| PolynomialFeatures | Yes | — | — | Yes |
| LabelEncoder | Yes | — | — | Yes |
| OneHotEncoder | Yes | — | — | Yes |
| SimpleImputer | Yes (mean/median/constant) | — | — | Yes |
| ColumnTransformer | Yes | — | — | Yes |
| CountVectorizer | Yes | — | — | Yes |
| TfidfVectorizer | Yes | — | — | Yes |
| SelectKBest | Yes | — | — | Yes |
| VarianceThreshold | Yes | — | — | Yes |

### 4.2 Pipeline Composition

```rust
// scry-learn
let pipeline = Pipeline::new()
    .add_transformer(StandardScaler::new())
    .set_model(RandomForestClassifier::new());
pipeline.fit(&train)?;
let preds = pipeline.predict(&test)?;
```

| Library | Pipeline | ColumnTransformer | FeatureUnion |
|---------|----------|-------------------|--------------|
| **scry-learn** | `Pipeline` (trait objects) | `ColumnTransformer` | — |
| **linfa** | Trait-based `Transformer` chaining | — | — |
| **smartcore** | Manual chaining only | — | — |
| **scikit-learn** | `Pipeline`, `make_pipeline` | `ColumnTransformer` | `FeatureUnion` |

scry-learn's `Pipeline` uses trait objects (`Box<dyn TransformerBox>` + `Box<dyn PipelineModel>`) for heterogeneous composition. The `impl_pipeline_model!` macro implements `PipelineModel` for all 21+ model types in a single invocation.

---

## 5. Model Selection and Evaluation

### 5.1 Cross-Validation

| Strategy | scry-learn | linfa | smartcore | scikit-learn |
|----------|-----------|-------|-----------|-------------|
| train_test_split | Yes | Yes (`split_with_ratio`) | — | Yes |
| Stratified split | Yes | — | — | Yes |
| K-Fold | Yes | Yes | — | Yes |
| Stratified K-Fold | Yes | — | — | Yes |
| Group K-Fold | Yes | — | — | Yes |
| Time Series Split | Yes | — | — | Yes |
| Repeated K-Fold | Yes | — | — | Yes |
| cross_val_score | Yes | Yes | — | Yes |
| cross_val_predict | Yes | — | — | Yes |

scry-learn implements 7 splitting strategies, closely mirroring sklearn's `model_selection` module. All splitters materialize folds eagerly as `Vec<(Dataset, Dataset)>` (sklearn uses lazy generators yielding index arrays).

### 5.2 Hyperparameter Search

| Search Method | scry-learn | linfa | smartcore | scikit-learn |
|--------------|-----------|-------|-----------|-------------|
| GridSearchCV | Yes | — | — | Yes |
| RandomizedSearchCV | Yes | — | — | Yes |
| BayesSearchCV | Yes | — | — | Yes (scikit-optimize) |
| HalvingGridSearchCV | — | — | — | Yes |

scry-learn's hyperparameter search uses the `Tunable` trait:

```rust
pub trait Tunable {
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()>;
}
```

This allows `GridSearchCV`, `RandomizedSearchCV`, and `BayesSearchCV` to work with any model implementing `Tunable + PipelineModel`.

### 5.3 Metrics

| Category | scry-learn | linfa | smartcore | scikit-learn |
|----------|-----------|-------|-----------|-------------|
| Classification | accuracy, balanced_accuracy, precision, recall, f1_score, cohen_kappa, log_loss, confusion_matrix, classification_report, ROC AUC, PR curve | accuracy, confusion_matrix | accuracy | 30+ metrics |
| Regression | r2_score, MSE, MAPE, explained_variance | r2_score, MSE | r2_score, MSE | 15+ metrics |
| Clustering | adjusted_rand_index, silhouette_score, calinski_harabasz, davies_bouldin | — | — | 10+ metrics |

---

## 6. Serialization and Versioning

| Library | Serialization | Model Versioning | Cross-Language Export |
|---------|--------------|-----------------|---------------------|
| **scry-learn** | serde (feature-gated) | `_schema_version` field | ONNX export |
| **linfa** | serde (partial, per-crate) | — | — |
| **smartcore** | serde (always-on) | — | — |
| **scikit-learn** | pickle/joblib | — | ONNX (via skl2onnx), PMML |

scry-learn's model versioning prevents stale model deserialization:

```rust
// Every model struct has:
#[cfg_attr(feature = "serde", serde(default))]
_schema_version: u32,

// Checked at predict time:
pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
    crate::version::check_schema_version(self._schema_version)?;
    // ...
}
```

If the serialized model's schema version doesn't match the current library version, prediction returns `SchemaVersionMismatch` instead of silently producing incorrect results.

---

## 7. Unique Features

### scry-learn

- **Terminal visualization**: Confusion matrices, ROC curves, learning curves, feature importance plots, and 12 other chart types rendered directly in the terminal via `scry-chart`
- **ONNX export**: Manual protobuf serialization (no external deps) for LinearRegression, LogisticRegression, StandardScaler, MLPClassifier/Regressor, all tree models
- **SHAP explainability**: TreeSHAP for tree-based models, permutation importance for any model
- **Prediction checksums**: FNV-1a hash on f64 prediction bits for cross-machine reproducibility verification
- **NaN/Inf rejection**: `validate_finite()` at `fit()` time prevents silent corruption from invalid data
- **Schema versioning**: Prevents stale deserialized model from producing wrong predictions
- **GPU acceleration**: wgpu compute shaders for matrix multiply, distance computation, histogram construction
- **Column-major layout**: Optimized for tree and coordinate descent workloads
- **Text processing**: CountVectorizer and TfidfVectorizer with sparse output
- **`#![deny(unsafe_code)]`**: No unsafe code allowed in the ML crate

### linfa

- **Type-state pattern**: Compile-time guarantee that predict cannot be called on unfitted models
- **Generic float types**: `F: Float` supports both f32 and f64
- **Rich trait hierarchy**: `Fit`, `Predict`, `Transformer` enable generic algorithm composition
- **ndarray ecosystem**: Direct interop with the Rust numerical computing ecosystem
- **Modular crates**: Each algorithm family is a separate crate with independent versioning

### smartcore

- **Simple API**: Closest to sklearn's feel among Rust libraries
- **Multiple matrix backends**: `BaseMatrix` trait accepts DenseMatrix, ndarray, or nalgebra
- **Low learning curve**: Familiar patterns for sklearn users migrating to Rust
- **Always-on serde**: All models are serializable without feature flags

### scikit-learn

- **Most comprehensive**: 100+ estimators spanning every major ML paradigm
- **Mature ecosystem**: 15+ years of production hardening, extensive documentation
- **Cython optimizations**: Critical inner loops compiled to C for performance
- **Rich pipeline system**: Pipeline, ColumnTransformer, FeatureUnion for complex workflows
- **Community**: Largest user base and contributor community of any ML library

---

## 8. Performance Characteristics

### 8.1 Memory Layout Impact

For **tree-based algorithms** (the most common ML workload), scry-learn's column-major layout provides a theoretical advantage: finding the best split for a feature requires scanning the entire column, which is contiguous in memory. In row-major layouts (linfa, smartcore, sklearn), this access pattern has stride `n_features`, causing cache misses. sklearn mitigates this by internally converting to Fortran-order for tree algorithms.

For **distance-based algorithms** (KNN, K-Means), row-major is naturally faster since computing distance between two samples accesses contiguous memory. scry-learn compensates with the lazy `row_major_cache`.

### 8.2 Abstraction Overhead

| Library | Matrix Abstraction | Overhead |
|---------|-------------------|----------|
| **scry-learn** | Direct `Vec<f64>` indexing | Minimal — no trait dispatch, no bounds-checked iterators |
| **linfa** | ndarray views/slicing | Low — ndarray is well-optimized but adds indirection |
| **smartcore** | `BaseMatrix` trait | Moderate — dynamic dispatch possible through trait objects |
| **scikit-learn** | numpy C API | Minimal — compiled Cython loops bypass Python overhead |

### 8.3 Solver Complexity

| Operation | scry-learn | Best Alternative |
|-----------|-----------|-----------------|
| Linear regression (normal eq.) | $O(nm^2 + m^3)$ Gauss-Jordan | $O(nm^2)$ Cholesky (sklearn) |
| PCA | $O(m^3 \cdot \text{iterations})$ Jacobi rotations | $O(nm^2)$ SVD (sklearn/linfa) |
| KNN search | $O(n \log n)$ KD-tree build, $O(\log n)$ query | Same complexity, but ball tree handles higher dimensions better |
| SVM | $O(n^2)$–$O(n^3)$ SMO | Same — all implementations use SMO variants |

scry-learn's PCA via Jacobi rotations avoids external BLAS/LAPACK dependencies but is asymptotically slower than SVD-based approaches for large feature counts. For typical ML datasets ($m < 1000$), the difference is negligible.

### 8.4 Parallelism Utilization

scry-learn and sklearn parallelize the most computationally expensive operations (ensemble training, cross-validation). linfa offers optional rayon support. smartcore is entirely single-threaded, which can be a significant bottleneck for ensemble methods on multi-core systems.

scry-learn's GPU acceleration via wgpu is unique among Rust ML libraries and provides significant speedup for large matrix operations (neural network training, KNN on large datasets, histogram GBT).

---

## 9. Algorithm Coverage Summary

| Category | scry-learn | linfa | smartcore | scikit-learn |
|----------|-----------|-------|-----------|-------------|
| Linear models | 5 | 5 | 5 | 10+ |
| Tree ensembles | 8 (incl. Hist-GBT) | 2 | 2 | 8+ |
| SVM | 4 | 1-2 | 2 | 4 |
| Neural networks | 2 + CNN layers | — | — | 2 |
| Naive Bayes | 3 | 1 | 2 | 5 |
| Neighbors | 2 | 2 | 2 | 4+ |
| Clustering | 4 | 4 | 2 | 10+ |
| Anomaly detection | 1 | — | — | 5+ |
| Ensemble meta-learners | 2 | — | — | 5+ |
| Preprocessing | 10 | 3 | 3 | 20+ |
| Text processing | 2 | — | — | 5+ |
| Feature selection | 3 | — | — | 10+ |
| Explainability | 2 | — | — | 2+ |
| Hyperparameter search | 3 | — | — | 4+ |
| **Total models** | **31+** | **~15** | **~12** | **100+** |

---

## 10. Key Architectural Differences Summary

| Decision | scry-learn | Why |
|----------|-----------|-----|
| Column-major default | Optimizes tree split evaluation and coordinate descent | Most common ML workloads are tree-based |
| No external matrix deps | Avoids ndarray/nalgebra compile time and API coupling | Self-contained, minimal dependency tree |
| `f64` only | Simplifies all APIs, eliminates generic type noise | ML workloads rarely benefit from f32 precision |
| `#![deny(unsafe_code)]` | Safety guarantee at the cost of some optimization potential | Correctness-first philosophy |
| Bundled `Dataset` | Prevents shape mismatches, carries metadata | Similar to linfa, unlike sklearn's separate X/y |
| Builder pattern | Ergonomic configuration without separate params struct | Matches Rust ecosystem conventions |
| Runtime fit check | Simpler API than linfa's type-state, familiar to sklearn users | Tradeoff: compile-time safety vs API simplicity |
| Integrated visualization | Direct terminal rendering without external tools | Unique capability among ML libraries |
| Schema versioning | Prevents stale model deserialization bugs | Production safety feature |
