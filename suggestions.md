# Scry Audit Reports

Prepared on: **2026-02-15**
Workspace: `/home/esoc/code/pixelcanvas`

---

## Report 1: Application Libraries and Architecture Audit

### 1) Scope and Method

This report covers the full workspace architecture and library health across:

- `scry-engine` (root crate)
- `crates/scry-chart`
- `crates/scry-cli`
- `crates/scry-learn`
- `crates/scry-pipe`

Methods used:

- Static source and manifest review
- Workspace build validation
- API and module boundary inspection
- Release/readiness and integration assessment

Key evidence points:

- Workspace manifest: `Cargo.toml:1`
- Core architecture docs: `README.md:56`
- `scry-learn` default build failure: `crates/scry-learn/src/lib.rs:57`, `crates/scry-learn/src/lib.rs:88`
- `scry-pipe` implementation status: `crates/scry-pipe/src/lib.rs:1`, `SCRY_PIPE_PROPOSAL.md:3`

---

### 2) Executive Summary

**Overall architecture grade: `B-`**

The workspace has a strong modular design and clear product layering, but two high-severity maturity gaps lower production readiness:

- `scry-learn` fails under default build configuration (feature-gating regression).
- `scry-pipe` is currently proposal/stub, not implemented.

These are fixable and do not require architectural replacement.

---

### 3) Findings by Severity

#### Critical

1. **Workspace build is blocked by `scry-learn` default feature mismatch**
- `cargo check --workspace` fails because `prelude` exports `viz` symbols even when `viz` feature is disabled.
- Evidence: `crates/scry-learn/src/lib.rs:57`, `crates/scry-learn/src/lib.rs:88`.
- Impact: prevents standard workspace builds and CI reliability.

#### High

2. **`scry-pipe` is packaged but not implemented**
- Crate source file is empty.
- Evidence: `crates/scry-pipe/src/lib.rs:1`.
- Proposal explicitly says pre-implementation.
- Evidence: `SCRY_PIPE_PROPOSAL.md:3`.
- Impact: roadmap claims around train-serving parity are not yet available.

3. **Public-facing maturity gap between proposal and runnable product surface**
- Proposal is detailed and commercially strong, but no executable MVP in `scry-pipe`.
- Impact: risk of expectation mismatch for users and stakeholders.

#### Medium

4. **Warning debt in `scry-chart`**
- Unused macro/import and unreachable pattern warnings during checks.
- Evidence from workspace check output.
- Impact: not breaking, but increases maintenance cost and may hide real regressions.

5. **Cross-crate dependency coupling is practical but creates upgrade coordination pressure**
- `scry-chart` depends on root `scry-engine` with enabled features.
- Evidence: `crates/scry-chart/Cargo.toml:19`.
- Impact: refactors in engine can cascade into chart/ML outputs unless CI matrices are broad.

#### Low

6. **Documentation describes strong architecture correctly but some crate capabilities are in transition**
- Core layering docs are good and clear.
- Evidence: `README.md:56`.
- Impact: low, but should include explicit “stable/beta/experimental” per crate.

---

### 4) Architecture Strengths

1. **Clear layered architecture in `scry-engine`**
- Drawing, transport, and widget boundaries are explicit.
- Evidence: `README.md:56`.

2. **Good crate decomposition**
- Separation of rendering (`scry-engine`), charts (`scry-chart`), CLI (`scry-cli`), ML (`scry-learn`), and future feature compiler (`scry-pipe`) is clean.

3. **Modern Rust hygiene posture**
- Strong lint baselines and explicit clippy policy across crates.
- Example: `Cargo.toml:82`, `crates/scry-chart/Cargo.toml:31`, `crates/scry-learn/Cargo.toml:53`.

4. **Product-oriented examples and artifacts**
- Extensive examples and benchmark coverage suggest strong developer feedback loops.

---

### 5) Architecture Weaknesses

1. **Feature gating discipline broke at crate boundary**
- Public prelude currently not resilient to disabled optional modules.

2. **Release-readiness inconsistency across crates**
- Some crates look production-grade while `scry-pipe` is still conceptual.

3. **CI/build matrix likely under-specified for feature combinations**
- Default and optional feature permutations should be validated systematically.

4. **Insufficient crate-level stability signaling**
- Consumers need explicit maturity labels (e.g., stable vs preview vs experimental).

---

### 6) Per-Crate Rating

| Crate | Purpose | Maturity | Architecture | Build Health | Rating |
|---|---|---:|---:|---:|---:|
| `scry-engine` | terminal vector engine | high | high | good | **A-** |
| `scry-chart` | charting on engine | medium-high | high | warnings only | **B+** |
| `scry-cli` | user-facing CLI | medium | medium-high | blocked indirectly by workspace | **B** |
| `scry-learn` | ML toolkit | medium | medium-high | **default build broken** | **C+** |
| `scry-pipe` | feature pipeline compiler | low (proposal) | high concept | not implemented | **D** |

---

### 7) Comparison to Industry Library Standards (Application-Level)

Current industry standards for serious Rust libraries/platforms generally include:

- clean default build and test matrix
- feature-flag-safe public APIs
- clear crate maturity labels
- stable CI gates on workspace + all feature sets
- progressive rollout from proposal to alpha to stable

Compared to that:

- **You are ahead** on architecture modularity and product ambition.
- **You are behind** on release hardening for cross-feature compilation and staged maturity signaling.

---

### 8) Priority Recommendations

1. **Fix default feature build blocker immediately**
- Add `#[cfg(feature = "viz")]` around `prelude` viz re-exports in `crates/scry-learn/src/lib.rs`.

2. **Add workspace feature matrix CI**
- `--workspace` default.
- `-p scry-learn --features viz`.
- `-p scry-learn --features serde`.
- combined feature set smoke build.

3. **Mark `scry-pipe` explicitly as experimental/proposal**
- Update top-level docs and crate docs until MVP exists.

4. **Close warning debt in `scry-chart`**
- Remove unreachable match arms and unused macro/import artifacts.

5. **Publish a stability map**
- One table in root README: crate, maturity, support scope, breaking-change policy.

---

## Report 2: Deep ML Algorithms and Capability Audit (`scry-learn`)

### 1) Scope and Method

Reviewed:

- Model implementations and APIs in `crates/scry-learn/src`
- Preprocessing, feature selection, CV/search, metrics, and pipeline
- Build/test/benchmark harness and competitor comparisons
- External ecosystem references (Rust ML + industry-standard stacks)

---

### 2) Executive Summary

**ML architecture grade: `C+`**

`scry-learn` is a solid classical ML toolkit with strong tree/performance engineering and good usability primitives, but it is not yet competitive with ecosystem leaders in breadth of algorithms, preprocessing depth, and full data science lifecycle support.

**Direct answer to key question:**  
**No** — this library does **not** currently provide more ML algorithms than leading Rust ecosystems (notably SmartCore and Linfa family).

---

### 3) Implemented Capability Inventory

#### Supervised Models

- Linear: `LinearRegression`, `LogisticRegression`, `LassoRegression`, `ElasticNet`
- Trees: `DecisionTreeClassifier/Regressor`, `RandomForestClassifier/Regressor`, `GradientBoostingClassifier/Regressor`
- Neighbors: `KnnClassifier`, `KnnRegressor`
- SVM: `LinearSVC`, `LinearSVR`, `KernelSVC`
- Naive Bayes: `GaussianNb`

Evidence:

- Public exports: `crates/scry-learn/src/lib.rs:67`, `crates/scry-learn/src/lib.rs:73`, `crates/scry-learn/src/lib.rs:75`

#### Unsupervised Models

- `KMeans`
- `Dbscan`

Evidence:

- `crates/scry-learn/src/lib.rs:76`

#### Preprocessing and Feature Engineering

- `StandardScaler`, `MinMaxScaler`
- `LabelEncoder`, `OneHotEncoder`
- `Pca`
- `VarianceThreshold`, `SelectKBest (FClassif)`

Evidence:

- `crates/scry-learn/src/preprocess/mod.rs:11`
- `crates/scry-learn/src/feature_selection.rs:52`, `crates/scry-learn/src/feature_selection.rs:172`

#### Validation / Search / Metrics

- splits: train-test, stratified split, k-fold, stratified k-fold
- CV scoring helpers
- `GridSearchCV`, `RandomizedSearchCV`
- classification/regression/ROC/PR metrics

Evidence:

- `crates/scry-learn/src/split.rs:16`
- `crates/scry-learn/src/search.rs:271`, `crates/scry-learn/src/search.rs:396`
- `crates/scry-learn/src/metrics/mod.rs:7`

---

### 4) High-Impact ML Findings

#### Critical

1. **Default crate build failure**
- same issue as Report 1; blocks out-of-box use.
- Evidence: `crates/scry-learn/src/lib.rs:57`, `crates/scry-learn/src/lib.rs:88`.

#### High

2. **Hyperparameter search is narrowly usable**
- `Tunable` only implemented for two classifier types.
- Evidence: `crates/scry-learn/src/search.rs:125`, `crates/scry-learn/src/search.rs:177`.

3. **`oob_score()` appears unsupported in practice**
- field exists but is never populated.
- Evidence: `crates/scry-learn/src/tree/random_forest.rs:62`, `crates/scry-learn/src/tree/random_forest.rs:269`.

4. **Data ingestion handles non-numeric features by `NaN` fallback without integrated imputation**
- Evidence: `crates/scry-learn/src/dataset.rs:110`.

#### Medium

5. **Solver/numerical robustness choices are pragmatic but can underperform in hard regimes**
- Linear regression uses normal equations + Gauss-Jordan.
- Evidence: `crates/scry-learn/src/linear/regression.rs:39`.

6. **Scalability limits in some algorithms**
- Kernel SVC precomputes full kernel matrix (`O(n^2)` memory).
- Evidence: `crates/scry-learn/src/svm/kernel.rs:282`.

7. **Unsafe hot path in tree traversal despite crate-level deny posture**
- Local `#[allow(unsafe_code)]` used for optimized predict traversal.
- Evidence: `crates/scry-learn/src/tree/cart.rs:149`.

---

### 5) What `scry-learn` Does Well

1. **Strong classical model core**
- Broad practical baseline coverage for many tabular tasks.

2. **Performance-oriented tree implementation**
- Flat tree representation and optimized prediction pathways.
- Evidence: `crates/scry-learn/src/tree/cart.rs:142`.

3. **Parallelism where it matters**
- Random forest training/predict uses Rayon.
- Evidence: `crates/scry-learn/src/tree/random_forest.rs:10`, `crates/scry-learn/src/tree/random_forest.rs:143`.

4. **Useful ML UX layer**
- Pipeline abstraction, CV helpers, chart-native visualization integration.

5. **Benchmark culture exists**
- Internal and competitor benches are present and fairly broad.
- Evidence: `crates/scry-learn/benches/competitor_bench.rs:1`, `crates/scry-learn/tests/benchmark_audit.rs:1`.

---

### 6) What Is Missing in the Full Data Science Process

Current DS lifecycle expectations usually include:

- robust missing-data handling
- rich transformations for non-Gaussian data
- broad model-selection and calibration toolset
- strong model explainability and diagnostics
- reproducible experiment tracking and registry
- production monitoring/drift detection integrations

Key gaps here:

1. **Preprocessing breadth**
- Missing robust scaler, quantile transform, power transform, generalized imputers, richer column-wise transform composition.

2. **Model family breadth**
- No histogram-boosting family, no XGBoost/LightGBM/CatBoost class, limited anomaly-detection and probabilistic families.

3. **Model selection depth**
- Search API exists but limited estimator compatibility.

4. **MLOps lifecycle**
- No integrated experiment tracking, model registry, drift monitoring, or serving governance layer in current implementation.

5. **Data engineering lifecycle**
- `scry-pipe` proposal is promising but presently not implemented, so train-serving parity tooling is absent today.

---

### 7) Industry and Ecosystem Comparison

#### Rust Ecosystem

- **SmartCore** and **Linfa ecosystem** currently expose broader algorithm families and decomposition/reduction breadth than current `scry-learn`.
- Your current strengths are implementation clarity and tree-centric performance emphasis, not maximum breadth.

#### Industry Leaders (Python/C++ stacks)

- **scikit-learn** exceeds current coverage in transformations, composition, model selection depth, and lifecycle ergonomics.
- **XGBoost/LightGBM/CatBoost** exceed current gradient boosting sophistication and production-scale optimization.

Bottom line:

- **Position today:** strong, modern classical ML crate in Rust workspace context.
- **Not yet:** top-of-market breadth or full lifecycle platform.

---

### 8) Ratings (ML-Specific)

| Dimension | Score (/10) | Notes |
|---|---:|---|
| Algorithm Breadth | 6.5 | Good classical core, but below ecosystem leaders |
| Preprocessing Breadth | 5.0 | Essential basics only |
| Model Selection/Tuning | 4.5 | Search exists, limited model compatibility |
| Performance Engineering | 8.0 | Strong tree/RF attention |
| Numerical Robustness | 6.0 | Mostly practical; room for harder cases |
| API/Usability | 7.0 | Prelude + pipeline + metrics are usable |
| Build/Release Reliability | 4.0 | Default feature compile blocker |
| End-to-End DS Lifecycle | 3.5 | Significant gaps in MLOps/serving process |

**Overall `scry-learn` grade: `C+`**

---

### 9) Highest-ROI Improvement Roadmap

1. **Fix build reliability**
- feature-gate prelude `viz` exports correctly.

2. **Generalize tuning framework**
- add `Tunable` support across linear, SVM, KNN, GBT, clustering where applicable.

3. **Fill preprocessing essentials**
- imputation suite, robust/quantile transforms, richer encoding and column transformations.

4. **Improve diagnostics truthfulness**
- implement true OOB score for RF or remove API until implemented.

5. **Add one major algorithm family**
- histogram GBDT or integration path for external booster.

6. **Advance `scry-pipe` from proposal to MVP**
- land minimal executable compiler path to close train-serving skew story.

---

### 10) References Used for Comparative Benchmarking

- SmartCore docs: https://docs.rs/smartcore/latest/smartcore/
- SmartCore user guide: https://smartcorelib.org/user_guide/quick_start.html
- Linfa ecosystem overview: https://www.linfa.dev/about/
- scikit-learn preprocessing docs: https://scikit-learn.org/stable/modules/preprocessing.html
- scikit-learn compose docs: https://scikit-learn.org/stable/modules/compose.html
- scikit-learn model selection docs: https://scikit-learn.org/stable/api/sklearn.model_selection.html
- scikit-learn imputation docs: https://scikit-learn.org/stable/modules/impute.html
- XGBoost docs: https://xgboost.readthedocs.io/en/stable/
- LightGBM features: https://lightgbm.readthedocs.io/en/stable/Features.html
- CatBoost categorical features: https://catboost.ai/docs/en/features/categorical-features.html

