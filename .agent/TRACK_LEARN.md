# scry-learn — Development Tracker

> **Crate**: `crates/scry-learn` | **Tier**: Advanced Beta | **Version**: 0.1.0
> **Updated**: 2026-02-15

## Current Inventory

### Supervised Models (15)
- **Trees**: DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
- **Linear**: LinearRegression, LogisticRegression, LassoRegression, ElasticNet
- **Neighbors**: KnnClassifier, KnnRegressor (distance-weighted, cosine metric)
- **SVM**: LinearSVC, LinearSVR, KernelSVC (RBF/Polynomial/Linear kernels via SMO)
- **Naive Bayes**: GaussianNb

### Unsupervised Models (2)
- KMeans (k-means++ init), Dbscan

### Preprocessing (10)
- StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder, Pca
- SimpleImputer (Mean, Median, MostFrequent, Constant strategies)
- ColumnTransformer (compose transformers on column subsets)
- VarianceThreshold, SelectKBest (f_classif)

### Infrastructure
- Pipeline (fit → transform → predict chain)
- GridSearchCV, RandomizedSearchCV (Tunable trait — currently DT + RF only)
- ClassWeight (Balanced, Custom — integrated across all classifiers)
- WeightFunction (Uniform, Distance — for KNN)
- train_test_split, stratified_split, cross_val_score, k-fold, stratified k-fold
- Full metrics: accuracy, precision, recall, F1, confusion_matrix, ROC/AUC, MSE, R²
- 14 built-in viz functions (confusion matrix, ROC, PR, feature importance, etc.)
- Serde support on all model types (opt-in via `--features serde`)

### Test Coverage
- 149+ unit tests, 25 correctness proofs, 4 benchmark audits
- Competitor benchmarks vs scikit-learn, linfa, smartcore with charts

### Benchmark Highlights

| Benchmark | scry-learn | scikit-learn | linfa | smartcore |
|---|---|---|---|---|
| DT predict (shallow) | **1.28 µs** | 67.7 µs (53×) | ~1.26 µs | 18 µs (14×) |
| RF predict (100 trees) | **286 µs** | 4,096 µs (14×) | 3,306 µs (11.5×) | 3,680 µs (12.9×) |
| PCA fit (2K×20) | **456 µs** | — | 3,174 µs (7×) | 4,471 µs (9.8×) |

## ML Roadmap Status

| Session | Focus | Status |
|---------|-------|--------|
| 1 | Serialization + Lasso/ElasticNet | ✅ Complete |
| 2 | Hyperparameter Search + Feature Selection | ✅ Complete |
| 3 | ClassWeight + Imbalanced Data | ✅ Complete |
| 4 | KNN Improvements + KNN Regressor | ✅ Complete |
| 5 | SVM (Linear + Kernel) | ✅ Complete |
| 6 | SimpleImputer + RobustScaler + ColumnTransformer | ✅ Complete |
| 7 | Tree Pruning + GBT Loss Functions | ✅ Complete |
| 8 | Clustering + NB Variants | ⬜ |
| 9 | Benchmark Expansion | ⬜ |
| 10 | Histogram GBT (new) | ⬜ |

## Known Issues
- `oob_score` field on RF exists but is never computed → implement or remove
- `Tunable` trait only implemented for DT + RF → expand to all models
- 33 pre-existing clippy warnings → cleanup pass needed

## Key Files
- `src/tree/cart.rs` — CART + FlatTree (16B DFS nodes)
- `src/tree/random_forest.rs` — RF with rayon + FlatTree predict
- `src/tree/gradient_boosting.rs` — GBT classifier + regressor (Newton-Raphson leaves)
- `src/svm/` — LinearSVC, LinearSVR, KernelSVC
- `src/linear/` — OLS, Logistic, Lasso, ElasticNet
- `src/neighbors/` — KNN classifier + regressor
- `src/preprocess/` — scalers, encoders, PCA
- `src/feature_selection.rs` — VarianceThreshold, SelectKBest
- `src/search.rs` — GridSearchCV, RandomizedSearchCV
- `src/viz.rs` — 14 ML visualization functions
- `src/pipeline.rs` — fit → transform → predict chain
