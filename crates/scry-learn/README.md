# scry-learn

Production-grade machine learning toolkit in pure Rust with built-in
[scry-chart](https://docs.rs/scry-chart) visualization.

## Features

- 23+ models: decision trees, random forests, gradient boosting, linear/logistic regression, Lasso, ElasticNet, Ridge, SVM, KNN, naive Bayes, MLP neural networks, K-Means, DBSCAN, agglomerative clustering, isolation forest
- Preprocessing: StandardScaler, MinMaxScaler, RobustScaler, PCA, one-hot encoding, polynomial features, imputation, normalization
- Model selection: GridSearchCV, RandomizedSearchCV, cross-validation, stratified splits
- Sparse matrix support (CSR/CSC) for NLP and recommender workloads
- Incremental learning via `partial_fit` for streaming data
- Built-in visualization (confusion matrices, ROC curves, feature importances, residual plots)
- GPU acceleration (optional, via wgpu)
- No BLAS/LAPACK dependency

## Quick Start

```rust
use scry_learn::prelude::*;

let data = Dataset::from_csv("iris.csv", "species")?;
let (train, test) = train_test_split(&data, 0.2, 42);

let mut model = RandomForestClassifier::new()
    .n_estimators(100)
    .max_depth(10);
model.fit(&train)?;

let preds = model.predict(&test)?;
let report = classification_report(&test.target, &preds);
println!("{report}");
```

## License

MIT OR Apache-2.0
