# scry — Roadmap

> **Updated**: 2026-02-17 | v0.7.0

## Current State

| Crate | Status |
|-------|--------|
| scry-engine | Stable — 6 transports (Kitty, Sixel, iTerm2, halfblock, window, SHM), GPU rasterizer (wgpu), SDF raytracer (CPU+GPU), WASM, animation |
| scry-chart | Stable — 18 chart types (16 2D + 3D scatter + streaming), 6 themes, PNG/SVG export |
| scry-learn | Stable — 31 models, 15 transformers/search, sparse (CSR/CSC), neural (MLP/Conv2D), GPU compute, partial_fit, SVD/QR/L-BFGS solvers, ONNX export |
| scry-pipe | Beta — Phase 1 done (IR + engine + codegen) |
| scry-cli | Beta — chart/image/animation commands |

### scry-learn Algorithm Coverage

| Family | Models |
|--------|--------|
| Trees | DecisionTree (C/R), RandomForest (C/R), GBT (C/R), HistGBT (C/R) |
| Linear | LinearRegression, Ridge, LogisticRegression, Lasso, ElasticNet |
| SVM | LinearSVC, LinearSVR, KernelSVC, KernelSVR |
| Neighbors | KnnClassifier, KnnRegressor (KdTree + brute-force) |
| Naive Bayes | GaussianNB, BernoulliNB, MultinomialNB |
| Clustering | KMeans, MiniBatchKMeans, DBSCAN, Agglomerative |
| Neural | MLPClassifier, MLPRegressor, Conv2D |
| Anomaly | IsolationForest |
| Ensemble | StackingClassifier, VotingClassifier |
| Preprocessing | StandardScaler, MinMaxScaler, RobustScaler, PCA, OneHot, SimpleImputer, ColumnTransformer, Normalizer, PolynomialFeatures, LabelEncoder |
| Feature Selection | VarianceThreshold, SelectKBest (f_classif) |
| Search | GridSearchCV, RandomizedSearchCV, BayesSearchCV |
| Explain | TreeSHAP, PermutationImportance |
| Export | ONNX |

## Recently Completed

- **Serde serialization**: conditional derives on all model structs (`--features serde`)
- **Lasso + ElasticNet**: coordinate descent with L1/L2 regularization
- **Hyperparameter search**: GridSearchCV, RandomizedSearchCV, BayesSearchCV with `Tunable` trait
- **Class weighting**: `ClassWeight` (Uniform/Balanced/Custom) on DT, RF, GBT, LogReg
- **SVM**: LinearSVC/SVR + KernelSVC/SVR with Gaussian, Polynomial, Linear, Sigmoid kernels
- **HistGBT**: histogram-based gradient boosting (classifier + regressor)
- **GBT loss functions**: SquaredError, AbsoluteError, Huber, Quantile
- **KNN improvements**: KdTree spatial index, KnnRegressor, distance-weighted predictions
- **Naive Bayes**: BernoulliNB, MultinomialNB added alongside GaussianNB
- **Preprocessing**: SimpleImputer, RobustScaler, ColumnTransformer, PolynomialFeatures, Normalizer
- **Feature selection**: VarianceThreshold, SelectKBest with ANOVA f_classif
- **Ensembles**: StackingClassifier, VotingClassifier (hard/soft)
- **Anomaly detection**: IsolationForest
- **Tree improvements**: ccp_alpha pruning, max_leaf_nodes constraint
- **Explainability**: TreeSHAP, ensemble TreeSHAP, permutation importance
- **ONNX export**: model serialization to ONNX format
- **Neural networks**: MLP (Classifier/Regressor), Conv2D, MaxPool2D
- **Text**: CountVectorizer
- **Clustering**: MiniBatchKMeans, AgglomerativeClustering (Ward/Complete/Average/Single), n_init for KMeans
- **Metrics**: balanced_accuracy, cohen_kappa, log_loss, ARI, Calinski-Harabasz, Davies-Bouldin, MAPE, explained_variance
- **Formatter refactor**: split from 915→533 lines (locale, date, zoom, semantic extracted)
- **SVD/QR/L-BFGS solvers**: multiple solver backends for linear models
- **Polars interop + MmapDataset**: implemented behind feature flags

## Not Started

### Probability Calibration

`PlattScaling` (sigmoid fit) and `IsotonicRegression` (PAV algorithm) for calibrating classifier probabilities. `CalibratedClassifierCV` wrapper with cross-validated calibration.

### HDBSCAN

Hierarchical density-based clustering (Campello et al. 2013). Automatically determines clusters from varying-density data without manual epsilon tuning.

### Spectral Clustering

Graph Laplacian-based clustering for non-convex cluster shapes.

### Dimensionality Reduction for Visualization

t-SNE and/or UMAP for 2D embedding of high-dimensional data.

### scry-pipe Phase 2 — PyO3 Python SDK (10A)

`pip install scry-pipe` with pandas/numpy interop. Parity guarantee: identical float output to 1e-12.

### WASM Codegen Target (10B)

Emit `.wasm` binary from pipeline definition.

### Out-of-Core & Streaming Fit (15)

`partial_fit` is done. Remaining: streaming data sources, Parquet support.

### Streaming & Live Data Charts (16)

`StreamingChart` exists. Remaining: stdin pipe integration (`scry stream`), WebSocket sources.

### Publication & Handbook (17)

- `cargo publish --dry-run` passes for engine + pipe; others blocked on upstream
- mdBook documentation site
- sklearn migration guide
- Performance tuning guide

## Lower Priority

- `warm_start` for RandomForest (incremental tree addition)
- Multi-output prediction support
- ndarray interop layer

## Versioning

| Milestone | Version |
|-----------|---------|
| Current | 0.7.0 |
| Calibration + HDBSCAN | 0.8.0 |
| Streaming charts | 0.9.0 |
| Handbook + crates.io publish | 1.0.0 |

## Known Performance Gaps

| Operation | Gap | Notes |
|-----------|-----|-------|
| KNN iris accuracy | -2.7% vs sklearn | Inherent to 150-sample dataset |
| Gaussian NB digits | -2.2% vs sklearn | var_smoothing differences |
