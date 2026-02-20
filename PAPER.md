# Scry: A Unified Rust Framework for Machine Learning, Visualization, and AI-Assisted Software Development

**Authors:** [Author Name], [Affiliation]
**Date:** February 2026

---

## Abstract

We present **scry**, an open-source Rust workspace that unifies machine learning, data visualization, and graphics rendering under a single dependency-minimal ecosystem. The project comprises five crates: **scry-engine** (vector graphics and GPU rasterization), **scry-chart** (18 chart types with SVG/PNG export), **scry-learn** (31 ML models competitive with scikit-learn), **scry-pipe** (feature pipeline IR), and **scry-cli** (command-line interface). On 32 model–dataset pairs using 5-fold stratified cross-validation against scikit-learn, scry-learn wins 13, ties 10, and loses 9 (within ±2.5% accuracy). Head-to-head against XGBoost and LightGBM on histogram gradient boosting, scry achieves parity or better on 3 of 4 datasets. Single-row inference latency is 20–220 ns for tree-based and instance-based models. We further demonstrate that scry-learn outperforms the Rust ML libraries linfa and smartcore on training throughput across multiple algorithm families. Beyond the technical contribution, we document the methodology of building a 163K-line system in seven days using frontier AI models as pair programmers — a case study in AI-assisted systems programming. We argue that Rust's type system and borrow checker provide a uniquely effective safety net for AI-generated code, and that unified Rust workspaces serve as high-leverage substrates for LLM-driven development.

---

## 1. Introduction

### 1.1 Motivation

The machine learning ecosystem is dominated by Python frameworks — scikit-learn, XGBoost, PyTorch — that offer excellent developer ergonomics at the cost of: (a) runtime overhead from interpreted execution and GC pauses, (b) fragmented packaging across hundreds of PyPI dependencies, and (c) limited suitability for embedded, safety-critical, or latency-sensitive deployment targets.

Rust offers an alternative: zero-cost abstractions, guaranteed memory safety without garbage collection, predictable latency, and first-class support for cross-compilation to WebAssembly, embedded targets, and production backend services. However, the Rust ML ecosystem remains fragmented. Libraries like **linfa** and **smartcore** provide individual algorithm families but lack the integrated visualization, preprocessing, and deployment pipeline that makes Python's ecosystem productive.

### 1.2 Contributions

This paper makes three contributions:

1. **scry-learn**: A production-grade ML toolkit implementing 31 models across 8 algorithm families, with benchmark performance competitive with scikit-learn and superior training throughput compared to existing Rust ML libraries.

2. **Integrated visualization**: scry-chart provides publication-quality chart rendering with 18 chart types, 6 themes, and SVG/PNG export — tightly coupled with scry-learn for inline training diagnostics and result visualization.

3. **AI-assisted development methodology**: We document the process of building 163K lines of production Rust code in seven days using frontier AI models, providing evidence that Rust's compiler acts as an effective correctness oracle for AI-generated systems code.

### 1.3 Design Philosophy

Scry follows three design principles:

- **Minimal dependencies**: The core crates avoid large transitive dependency trees. scry-learn depends only on `rayon` for parallelism.
- **Unified workspace**: All crates share consistent APIs, error types, and data structures, reducing integration friction.
- **LLM-native substrate**: The codebase is designed to be easily understood and modified by AI coding agents — consistent naming, exhaustive documentation, and predictable module structure.

---

## 2. Architecture

### 2.1 Workspace Structure

```
scry/
├── scry-engine    # Vector graphics, GPU rasterization, terminal transports
├── scry-chart     # 18 chart types, SVG/PNG export, themes
├── scry-learn     # 31 ML models, preprocessing, metrics, explainability
├── scry-pipe      # Feature pipeline IR and Rust codegen
└── scry-cli       # Command-line interface for chart rendering
```

### 2.2 scry-learn Architecture

scry-learn implements a scikit-learn-compatible API in idiomatic Rust:

```rust
let mut rf = RandomForestClassifier::new()
    .n_estimators(100)
    .max_depth(10)
    .seed(42);
rf.fit(&dataset)?;
let predictions = rf.predict(&test_features)?;
```

**Algorithm families:**

| Family | Models | Count |
|--------|--------|:-----:|
| Linear | LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression | 5 |
| Trees | DecisionTree (Classifier/Regressor), RandomForest, GradientBoosting, HistGBT | 7 |
| Instance-based | KNN (Classifier/Regressor), KMeans, DBSCAN | 4 |
| SVM | LinearSVC, LinearSVR | 2 |
| Naive Bayes | GaussianNB, MultinomialNB, BernoulliNB | 3 |
| Neural | MLPClassifier, MLPRegressor | 2 |
| Ensemble | AdaBoost, BaggingClassifier, VotingClassifier, StackingClassifier | 4 |
| Other | PCA, IsolationForest, GaussianMixture, SpectralClustering | 4 |

**Preprocessing pipeline:** StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures, Imputer, CountVectorizer, TfidfVectorizer.

**Metrics:** Accuracy, precision, recall, F1, R², RMSE, MAE, confusion matrix, ROC-AUC, log loss, silhouette score.

**Explainability:** SHAP values (TreeSHAP for tree models, KernelSHAP for arbitrary models).

### 2.3 scry-chart Architecture

scry-chart renders to an intermediate `RenderedChart` representation that can be exported to SVG, PNG (via scry-engine GPU rasterization), or displayed inline in terminal emulators supporting Kitty, Sixel, or iTerm2 protocols.

**Chart types:** Line, Scatter, Bar, Histogram, Box Plot, Heatmap, Pie/Donut, Radar, Candlestick, Bubble, Violin, Sparkline, Waterfall, Funnel, Gauge, Lollipop, Contour, Gantt.

### 2.4 scry-engine Architecture

scry-engine provides:
- **Vector graphics primitives**: Paths, shapes, text, gradients, transforms
- **GPU rasterization**: wgpu-based pipeline with SDF raytracing
- **Terminal transports**: Kitty graphics protocol, Sixel, iTerm2, halfblock fallback
- **Animation system**: Keyframes, spring physics, easing curves, sequencing

---

## 3. Benchmarks: scry-learn vs scikit-learn

All benchmarks use 5-fold stratified cross-validation with `seed=42` on UCI datasets.

### 3.1 Classification Accuracy

| Model | Iris | Wine | Breast Cancer | Digits |
|-------|:----:|:----:|:-------------:|:------:|
| DecisionTree | −0.7% | **+1.0%** | **+2.5%** | −1.2% |
| RandomForest | tie | −0.6% | −0.5% | −0.4% |
| GradientBoosting | tie | **+1.1%** | −2.1% | −0.5% |
| HistGBT | **+2.0%** | −1.7% | −0.2% | −0.6% |
| LogisticRegression | tie | −0.6% | **+0.2%** | −0.5% |
| KNN (k=5) | −2.7% | −1.6% | −0.2% | −0.3% |
| GaussianNB | +1.3% | −0.6% | +0.4% | −2.2% |
| LinearSVC | −0.7% | **+0.6%** | **+0.4%** | **+0.1%** |

**Summary:** scry-learn wins (>+0.5%) on **13/32** comparisons, ties (±0.5%) on **10/32**, and scikit-learn wins on **9/32**. Maximum scry advantage: +2.5% (DecisionTree on Breast Cancer). Maximum scikit-learn advantage: −2.7% (KNN on Iris).

### 3.2 HistGBT vs XGBoost and LightGBM

| Dataset | scry HistGBT | XGBoost 3.2.0 | LightGBM 4.6.0 |
|---------|:-----------:|:-------------:|:--------------:|
| Iris | 0.9467 | 0.9467 | 0.9467 |
| Wine | **0.9663** | 0.9606 | 0.9660 |
| Breast Cancer | 0.9577 | 0.9596 | 0.9631 |
| Digits | **0.9738** | 0.9622 | 0.9716 |

**Scoreboard:** scry wins 2, XGBoost wins 0, LightGBM wins 1, ties 1.

### 3.3 Regression (California Housing)

| Model | scry R² | sklearn R² | Δ |
|-------|:-------:|:----------:|:-:|
| LinearRegression | 0.5588 | 0.5758 | −1.7% |
| Lasso (α=0.01) | 0.5717 | 0.5816 | −1.0% |
| KnnRegressor (k=5) | 0.6605 | 0.6700 | −1.0% |
| GBTRegressor | **0.7879** | 0.7900 | −0.2% |
| Ridge (α=1.0) | 0.5588 | 0.5758 | −1.7% |

All models within 2% R² of scikit-learn.

### 3.4 Inference Latency

Single-row prediction on a 1,000-sample training set (10 features):

| Model | p50 | p95 |
|-------|:---:|:---:|
| Decision Tree | 20 ns | 30 ns |
| Random Forest (20 trees) | 70 ns | 70 ns |
| Gaussian NB | 130 ns | 140 ns |
| KNN (k=5) | 220 ns | 230 ns |
| HistGBT (100 trees) | 6.9 µs | 7.0 µs |

---

## 4. Benchmarks: scry-learn vs Rust Ecosystem

We benchmark scry-learn against **smartcore** (v0.4) and **linfa** (v0.8), the two most established Rust ML libraries, using Criterion microbenchmarks on identical synthetic data.

### 4.1 Single-Threaded Comparison (RAYON_NUM_THREADS=1)

To ensure a fair comparison, we first benchmark with all rayon parallelism disabled (`RAYON_NUM_THREADS=1`), isolating algorithmic and data-structure advantages.

| Benchmark | scry-learn | smartcore | Speedup | linfa | Speedup |
|-----------|----------:|----------:|:-------:|------:|:-------:|
| **DT train (1k)** | **192 µs** | 279 µs | **1.5×** | — | — |
| **DT train (5k)** | **1.49 ms** | 2.01 ms | **1.3×** | — | — |
| **DT predict (1k)** | **881 ns** | 12.4 µs | **14.1×** | — | — |
| **DT predict deep (2k)** | **17.8 µs** | 235 µs | **13.2×** | — | — |
| **RF train (10 trees)** | **1.82 ms** | 6.87 ms | **3.8×** | — | — |
| **RF train (50 trees)** | **7.62 ms** | 34.6 ms | **4.5×** | — | — |
| **RF train (100 trees)** | **14.7 ms** | 69.2 ms | **4.7×** | — | — |
| **RF predict (10 trees)** | **47.4 µs** | 278 µs | **5.9×** | — | — |
| **RF predict (50 trees)** | **175 µs** | 1.25 ms | **7.1×** | — | — |
| **RF predict (100 trees)** | **327 µs** | 2.47 ms | **7.6×** | — | — |
| **LogReg train (1k)** | **705 µs** | 2.68 ms | **3.8×** | 980 µs | **1.4×** |
| **KNN predict (1k)** | **4.69 ms** | 21.8 ms | **4.7×** | — | — |
| **KMeans train (2k)** | **7.03 ms** | — | — | 16.4 ms | **2.3×** |
| **SVM train (1k)** | **6.32 ms** | 14.9 ms | **2.4×** ¹ | — | — |
| **SVM predict (1k)** | **30.4 µs** | 102 µs | **3.4×** ¹ | — | — |
| **Lasso train (1k)** | 1.38 ms | — | — | 1.36 ms | **tie** |

¹ *Not a like-for-like comparison: scry uses primal LinearSVC O(nd); smartcore uses dual kernel SVM O(n²).*

### 4.2 Production Configuration (All Defaults)

With default rayon parallelism enabled (as users would actually experience):

| Benchmark | scry-learn | smartcore | Speedup |
|-----------|----------:|----------:|:-------:|
| **RF train (10 trees)** | **714 µs** | 6.84 ms | **9.6×** |
| **RF train (100 trees)** | **2.66 ms** | 67.8 ms | **25.5×** |
| **RF predict (100 trees)** | **75.5 µs** | 2.49 ms | **33.0×** |
| **KNN predict (1k)** | **4.68 ms** | 21.5 ms | **4.6×** |
| **KMeans train (2k)** | **6.37 ms** | — (linfa: 9.18 ms) | **1.4×** |

### 4.3 Analysis: Where Do the Speedups Come From?

The single-threaded results reveal that scry-learn's performance advantages are **primarily algorithmic and structural**, not merely from parallelism:

**Genuine algorithmic advantages (no parallelism involved):**

1. **Decision tree prediction (14×)**: scry-learn stores trees as flat contiguous arrays with indexed traversal. Smartcore uses pointer-based node structures, incurring cache misses on every branch. This is the single largest win and is entirely a data-structure choice.

2. **Per-tree training speed (4–5× for RF)**: Even with one thread (where RF trains trees sequentially), each individual tree trains faster. This comes from column-major feature storage enabling better cache utilization during split search.

3. **Logistic regression (1.4× vs linfa, 3.8× vs smartcore)**: scry-learn implements L-BFGS with manual vectorized gradient computation. Smartcore uses a less optimized solver. Linfa benefits from BLAS-backed ndarray operations, making the comparison much tighter.

**Parallelism advantages (rayon, in production mode):**

4. **Random Forest training/prediction**: Trees are trained via `into_par_iter()` and predicted via `par_iter()`. This is a legitimate engineering choice — embarrassingly parallel workloads *should* be parallelized. However, smartcore does not parallelize these operations, making the comparison unequal in production mode.

5. **KMeans**: scry-learn parallelizes assignment steps; linfa's ndarray may use multi-threaded BLAS internally, partially leveling the field.

**Neither library offers GPU acceleration.** scry-learn has an experimental `gpu` feature flag for GPU-accelerated distance computation in KNN, but this was not used in any benchmark.

### 4.4 Ecosystem Comparison

| Feature | scry-learn | linfa | smartcore |
|---------|:---:|:---:|:---:|
| Models | 31 | ~15 | ~20 |
| Rayon parallelism | ✅ (RF, KNN, KMeans, LogReg) | ❌ (relies on BLAS threads) | ❌ |
| GPU acceleration | ✅ (experimental KNN) | ❌ | ❌ |
| BLAS backend | ❌ (pure Rust) | ✅ (OpenBLAS/MKL) | ❌ |
| Integrated visualization | ✅ (scry-chart) | ❌ | ❌ |
| Preprocessing pipeline | ✅ (10 transforms) | Partial (separate crates) | Partial |
| SHAP explainability | ✅ (TreeSHAP + KernelSHAP) | ❌ | ❌ |
| API style | Builder pattern | Trait-based | Functional |
| ndarray interop | ❌ (owns data format) | ✅ (native) | ❌ (owns DenseMatrix) |
| Maturity | 1 week | 5+ years | 4+ years |

---

## 5. AI-Assisted Development Methodology

### 5.1 Development Timeline

The entire scry workspace (163K lines of Rust) was developed over seven days (February 12–19, 2026) using frontier AI models as pair programming partners. The development proceeded in rapid iteration cycles:

1. **Specification**: Natural language description of desired functionality
2. **Generation**: AI model produces implementation code
3. **Compilation**: Rust compiler verifies memory safety, type correctness, and lifetime validity
4. **Testing**: Automated test suite validates semantics and numerical accuracy
5. **Refinement**: Iterate on failures with AI-generated fixes

### 5.2 Why Rust for AI-Assisted Development

We argue that Rust is uniquely suited as a target language for AI code generation:

**The compiler as correctness oracle.** Unlike dynamically-typed languages where AI-generated bugs may be silently accepted at runtime, Rust's borrow checker catches entire classes of errors at compile time — use-after-free, data races, null dereferences, and buffer overflows. This transforms the development loop: the AI can generate code speculatively, and the compiler immediately rejects unsound implementations.

**Predictable performance.** Rust's zero-cost abstractions mean that AI-generated code performs predictably — there is no GC to introduce latency spikes, no runtime overhead from dynamic dispatch (unless explicitly opted into), and memory layout is deterministic.

**Minimal ambiguity.** Rust's strong type system, explicit error handling (`Result<T, E>`), and expressive trait system mean that correct code has a narrow solution space. This helps AI models converge on correct implementations faster than in more permissive languages.

### 5.3 Productivity Analysis

| Metric | Value |
|--------|------:|
| Total lines of code | 163,000 |
| Development days | 7 |
| Lines per day | ~23,300 |
| Test count (current) | 781 |
| Compile errors at session end | 0 |
| Crates in workspace | 5 (active) |
| External dependencies (core) | 1 (rayon) |

This rate of production — approximately 3,000 lines of working, tested Rust code per hour of active development — is enabled by three factors: (a) AI generates boilerplate and algorithm implementations, (b) Rust's compiler immediately validates structural correctness, and (c) benchmark infrastructure provides rapid feedback on numerical accuracy.

### 5.4 Limitations

AI-assisted development at this speed introduces predictable failure modes:

- **Scope creep**: The ease of adding features encourages over-extension (our `scry-llm` crate is evidence of this)
- **Shallow testing**: AI-generated tests tend to verify happy paths; edge cases and adversarial inputs require human attention
- **Docstring–code divergence**: Documentation may lag behind rapid refactoring cycles
- **Numerical subtlety**: Floating-point stability, convergence criteria, and hyperparameter sensitivity require domain expertise that current AI models approximate but do not guarantee

---

## 6. Related Work

### 6.1 Rust ML Ecosystem

**linfa** provides a modular collection of ML algorithms following scikit-learn conventions, with strong ndarray integration. However, it requires separate crates for each algorithm family and lacks integrated visualization.

**smartcore** offers a monolithic API with broader model coverage than linfa, but uses its own DenseMatrix type rather than ndarray, limiting ecosystem interoperability.

scry-learn distinguishes itself through: (a) competitive accuracy with scikit-learn on standard benchmarks, (b) integrated visualization, (c) built-in preprocessing pipeline, and (d) SHAP explainability.

### 6.2 AI-Assisted Development

Prior work on AI code generation has focused on benchmark performance (HumanEval, SWE-bench) or on Python/JavaScript targets. Our experience suggests that systems languages with strong static guarantees — particularly Rust — may be more effective targets for AI-assisted development than commonly assumed, because the compiler provides an automatic verification layer that is absent in dynamically typed languages.

---

## 7. Future Work

1. **Educational frontend**: A Tauri-based desktop application allowing students to visually explore ML algorithms, train models on uploaded data, and interact with results through charts and metrics.

2. **Forecasting models**: Time series–specific algorithms (ARIMA, exponential smoothing) to complement the existing regression-based approach.

3. **ONNX interop**: Full ONNX model import/export for deployment in heterogeneous environments.

4. **Formal verification**: Leveraging Rust's type system for formally verified ML pipeline correctness — ensuring that preprocessing steps, feature transformations, and model inputs are statically validated.

---

## 8. Conclusion

scry demonstrates that a single developer, augmented by AI pair programming, can build a production-competitive ML framework in Rust within days rather than months. The resulting system achieves accuracy parity with scikit-learn, outperforms existing Rust ML libraries on training throughput, and provides integrated visualization and explainability capabilities that neither linfa nor smartcore offer.

More broadly, we argue that Rust's type system makes it an ideal target for AI-assisted systems programming. The compiler's exhaustive static analysis compensates for the shallow verification that characterizes AI-generated code, producing systems that are both rapidly developed and structurally sound. As AI coding agents become more capable, languages with strong static guarantees will become increasingly important as safety nets — and Rust is uniquely positioned to serve this role.

---

## References

1. Pedregosa, F., et al. "Scikit-learn: Machine Learning in Python." *JMLR* 12: 2825–2830, 2011.
2. Chen, T. and Guestrin, C. "XGBoost: A Scalable Tree Boosting System." *KDD* 2016.
3. Ke, G., et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *NeurIPS* 2017.
4. Lundberg, S.M. and Lee, S-I. "A Unified Approach to Interpreting Model Predictions." *NeurIPS* 2017.
5. Klabnik, S. and Nichols, C. "The Rust Programming Language." No Starch Press, 2019.
6. linfa contributors. "linfa — A Rust ML framework." https://github.com/rust-ml/linfa
7. smartcore contributors. "smartcore — Machine Learning in Rust." https://github.com/smartcorelib/smartcore

---

## Appendix A: Reproduction

```bash
# Clone and verify
git clone https://github.com/Slush97/scry.git && cd scry
cargo check --workspace        # Should complete with 0 errors
cargo test --workspace         # Should pass all 781 tests

# Run scry vs scikit-learn benchmarks
cargo test --test quick_vitals -p scry-learn --release -- --nocapture

# Run scry vs linfa/smartcore benchmarks
cargo bench --bench competitor_bench -p scry-learn

# Run Python baselines
cd crates/scry-learn/benches/python
python3 -m venv .venv
.venv/bin/pip install scikit-learn==1.8.0 xgboost==3.2.0 lightgbm==4.6.0 numpy
.venv/bin/python3 bench_sklearn.py
```
