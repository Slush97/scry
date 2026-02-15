//! Head-to-head benchmark: scry-learn vs. Rust ML ecosystem competitors.
//!
//! Compares training throughput and prediction latency for algorithms
//! that multiple libraries implement. Uses identical data generation.
//!
//! Run:  cargo bench --bench competitor_bench -p scry-learn

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

// ── Data generation (shared across all libraries) ─────────────────
fn gen_classification(n: usize, n_features: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = fastrand::Rng::with_seed(42);
    let half = n / 2;
    let mut features_col_major = vec![vec![0.0; n]; n_features];
    let mut target = vec![0.0; n];

    for j in 0..n_features {
        let offset = 3.0 + j as f64 * 0.5;
        for i in 0..half {
            features_col_major[j][i] = rng.f64() * 2.0;
        }
        for i in half..n {
            features_col_major[j][i] = rng.f64() * 2.0 + offset;
            target[i] = 1.0;
        }
    }

    let row_major: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n_features).map(|j| features_col_major[j][i]).collect())
        .collect();

    (row_major, target)
}

fn transpose(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if rows.is_empty() { return vec![]; }
    let n_cols = rows[0].len();
    let n_rows = rows.len();
    (0..n_cols).map(|j| (0..n_rows).map(|i| rows[i][j]).collect()).collect()
}

// ── Benchmarks ───────────────────────────────────────────────────

fn bench_dt_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("vs/decision_tree/train");
    group.sample_size(20);

    for &n in &[1000usize, 5000] {
        let (features, target) = gen_classification(n, 10);
        let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();

        group.bench_with_input(BenchmarkId::new("scry-learn", n), &n, |b, _| {
            b.iter(|| {
                let data = scry_learn::prelude::Dataset::new(
                    transpose(&features), target.clone(),
                    (0..10).map(|i| format!("f{i}")).collect(), "target",
                );
                let mut dt = scry_learn::prelude::DecisionTreeClassifier::new();
                dt.fit(black_box(&data)).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("smartcore", n), &n, |b, _| {
            b.iter(|| {
                let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&features).unwrap();
                let _ = smartcore::tree::decision_tree_classifier::DecisionTreeClassifier::fit(
                    black_box(&x), black_box(&target_i32), Default::default(),
                ).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_dt_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("vs/decision_tree/predict");
    group.sample_size(50);

    let (features, target) = gen_classification(1000, 10);
    let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();

    let data = scry_learn::prelude::Dataset::new(
        transpose(&features), target.clone(),
        (0..10).map(|i| format!("f{i}")).collect(), "target",
    );
    let mut scry_dt = scry_learn::prelude::DecisionTreeClassifier::new();
    scry_dt.fit(&data).unwrap();

    let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&features).unwrap();
    let smart_dt = smartcore::tree::decision_tree_classifier::DecisionTreeClassifier::fit(
        &x, &target_i32, Default::default(),
    ).unwrap();

    group.bench_function("scry-learn/1k", |b| {
        b.iter(|| scry_dt.predict(black_box(&features)).unwrap());
    });
    group.bench_function("smartcore/1k", |b| {
        b.iter(|| smart_dt.predict(black_box(&x)).unwrap());
    });

    group.finish();
}

fn bench_rf_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("vs/random_forest/train");
    group.sample_size(10);

    let (features, target) = gen_classification(2000, 10);
    let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();

    for &n_trees in &[10usize, 50, 100] {
        group.bench_with_input(BenchmarkId::new("scry-learn", n_trees), &n_trees, |b, &nt| {
            b.iter(|| {
                let data = scry_learn::prelude::Dataset::new(
                    transpose(&features), target.clone(),
                    (0..10).map(|i| format!("f{i}")).collect(), "target",
                );
                let mut rf = scry_learn::prelude::RandomForestClassifier::new()
                    .n_estimators(nt).max_depth(8);
                rf.fit(black_box(&data)).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("smartcore", n_trees), &n_trees, |b, &nt| {
            b.iter(|| {
                let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&features).unwrap();
                let params = smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters::default()
                    .with_n_trees(nt as u16)
                    .with_max_depth(8);
                let _ = smartcore::ensemble::random_forest_classifier::RandomForestClassifier::fit(
                    black_box(&x), black_box(&target_i32), params,
                ).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_rf_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("vs/random_forest/predict");
    group.sample_size(20);

    let (features, target) = gen_classification(2000, 10);
    let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();

    for &n_trees in &[10usize, 50, 100] {
        // scry-learn
        let data = scry_learn::prelude::Dataset::new(
            transpose(&features), target.clone(),
            (0..10).map(|i| format!("f{i}")).collect(), "target",
        );
        let mut scry_rf = scry_learn::prelude::RandomForestClassifier::new()
            .n_estimators(n_trees).max_depth(8);
        scry_rf.fit(&data).unwrap();

        // smartcore
        let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&features).unwrap();
        let params = smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters::default()
            .with_n_trees(n_trees as u16)
            .with_max_depth(8);
        let smart_rf = smartcore::ensemble::random_forest_classifier::RandomForestClassifier::fit(
            &x, &target_i32, params,
        ).unwrap();

        group.bench_with_input(BenchmarkId::new("scry-learn", n_trees), &n_trees, |b, _| {
            b.iter(|| scry_rf.predict(black_box(&features)).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("smartcore", n_trees), &n_trees, |b, _| {
            b.iter(|| smart_rf.predict(black_box(&x)).unwrap());
        });
    }

    group.finish();
}

// ── Deep-tree benchmark (noisy overlapping data) ───────────────
// The standard gen_classification produces trivially separable data
// (offset 3.0+) → depth 1-2 trees. This version uses tiny offset
// with heavy overlap → deep trees that stress cache layout.

fn gen_noisy_classification(n: usize, n_features: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = fastrand::Rng::with_seed(7);
    let half = n / 2;
    let mut features_col_major = vec![vec![0.0; n]; n_features];
    let mut target = vec![0.0; n];

    for j in 0..n_features {
        for i in 0..half {
            features_col_major[j][i] = rng.f64() * 4.0 - 2.0;
        }
        for i in half..n {
            features_col_major[j][i] = rng.f64() * 4.0 - 2.0 + 0.5;
            target[i] = 1.0;
        }
    }

    let row_major: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n_features).map(|j| features_col_major[j][i]).collect())
        .collect();
    (row_major, target)
}

fn bench_dt_predict_deep(c: &mut Criterion) {
    let mut group = c.benchmark_group("vs/decision_tree/predict_deep");
    group.sample_size(50);

    let (features, target) = gen_noisy_classification(2000, 10);
    let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();

    let data = scry_learn::prelude::Dataset::new(
        transpose(&features), target.clone(),
        (0..10).map(|i| format!("f{i}")).collect(), "target",
    );
    let mut scry_dt = scry_learn::prelude::DecisionTreeClassifier::new();
    scry_dt.fit(&data).unwrap();

    let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&features).unwrap();
    let smart_dt = smartcore::tree::decision_tree_classifier::DecisionTreeClassifier::fit(
        &x, &target_i32, Default::default(),
    ).unwrap();

    let scry_depth = scry_dt.flat_tree().map(|ft| ft.depth()).unwrap_or(0);
    let scry_leaves = scry_dt.flat_tree().map(|ft| ft.n_leaves()).unwrap_or(0);
    eprintln!("Deep tree bench: scry depth={scry_depth}, leaves={scry_leaves}");

    group.bench_function("scry-learn/2k_deep", |b| {
        b.iter(|| scry_dt.predict(black_box(&features)).unwrap());
    });
    group.bench_function("smartcore/2k_deep", |b| {
        b.iter(|| smart_dt.predict(black_box(&x)).unwrap());
    });

    group.finish();
}

fn bench_confusion_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("vs/metrics/confusion_matrix");
    group.sample_size(50);

    let n = 10_000;
    let mut rng = fastrand::Rng::with_seed(42);
    let y_true: Vec<f64> = (0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }).collect();
    let y_pred: Vec<f64> = y_true.iter()
        .map(|&t| if rng.f64() < 0.9 { t } else { 1.0 - t })
        .collect();

    group.bench_function("scry-learn/10k", |b| {
        b.iter(|| scry_learn::prelude::confusion_matrix(black_box(&y_true), black_box(&y_pred)));
    });

    group.finish();
}

// ── Logistic Regression: scry vs smartcore vs linfa-logistic ────

fn bench_logreg_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("vs/logistic_regression/train");
    group.sample_size(10);

    let (features, target) = gen_classification(1000, 10);
    let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();
    let target_bool: Vec<bool> = target.iter().map(|&t| t > 0.5).collect();

    group.bench_function("scry-learn/1k", |b| {
        b.iter(|| {
            let data = scry_learn::prelude::Dataset::new(
                transpose(&features), target.clone(),
                (0..10).map(|i| format!("f{i}")).collect(), "target",
            );
            let mut lr = scry_learn::prelude::LogisticRegression::new()
                .max_iter(200)
                .learning_rate(0.1);
            lr.fit(black_box(&data)).unwrap();
        });
    });

    group.bench_function("smartcore/1k", |b| {
        b.iter(|| {
            let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&features).unwrap();
            let _ = smartcore::linear::logistic_regression::LogisticRegression::fit(
                black_box(&x), black_box(&target_i32), Default::default(),
            ).unwrap();
        });
    });

    group.bench_function("linfa-logistic/1k", |b| {
        use linfa::prelude::Fit;
        let flat: Vec<f64> = features.iter().flat_map(|r| r.iter().copied()).collect();
        let x = ndarray::Array2::from_shape_vec((1000, 10), flat).unwrap();
        let y = ndarray::Array1::from_vec(target_bool.clone());
        let ds = linfa::Dataset::new(x, y);
        b.iter(|| {
            let _ = linfa_logistic::LogisticRegression::default()
                .max_iterations(200)
                .fit(black_box(&ds))
                .unwrap();
        });
    });

    group.finish();
}

// ── KNN: scry vs smartcore ─────────────────────────────────────

fn bench_knn_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("vs/knn/predict");
    group.sample_size(20);

    let (features, target) = gen_classification(1000, 10);
    let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();

    // scry-learn
    let data = scry_learn::prelude::Dataset::new(
        transpose(&features), target.clone(),
        (0..10).map(|i| format!("f{i}")).collect(), "target",
    );
    let mut scry_knn = scry_learn::prelude::KnnClassifier::new().k(5);
    scry_knn.fit(&data).unwrap();

    // smartcore
    let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&features).unwrap();
    let smart_knn = smartcore::neighbors::knn_classifier::KNNClassifier::fit(
        &x, &target_i32,
        smartcore::neighbors::knn_classifier::KNNClassifierParameters::default().with_k(5),
    ).unwrap();

    let test_features = data.feature_matrix();

    group.bench_function("scry-learn/1k", |b| {
        b.iter(|| scry_knn.predict(black_box(&test_features)).unwrap());
    });

    group.bench_function("smartcore/1k", |b| {
        b.iter(|| smart_knn.predict(black_box(&x)).unwrap());
    });

    group.finish();
}

// ── K-Means: scry vs linfa-clustering ──────────────────────────

fn bench_kmeans_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("vs/kmeans/train");
    group.sample_size(10);

    let (features, target) = gen_classification(2000, 10);

    group.bench_function("scry-learn/2k", |b| {
        b.iter(|| {
            let data = scry_learn::prelude::Dataset::new(
                transpose(&features), target.clone(),
                (0..10).map(|i| format!("f{i}")).collect(), "target",
            );
            let mut km = scry_learn::prelude::KMeans::new(3).seed(42).max_iter(100);
            km.fit(black_box(&data)).unwrap();
        });
    });

    group.bench_function("linfa-clustering/2k", |b| {
        use linfa::prelude::Fit;
        let flat: Vec<f64> = features.iter().flat_map(|r| r.iter().copied()).collect();
        let x = ndarray::Array2::from_shape_vec((2000, 10), flat).unwrap();
        let ds = linfa::DatasetBase::from(x);
        b.iter(|| {
            let _ = linfa_clustering::KMeans::params_with_rng(3, rand::thread_rng())
                .max_n_iterations(100)
                .fit(black_box(&ds))
                .unwrap();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_dt_training,
    bench_dt_predict,
    bench_dt_predict_deep,
    bench_rf_training,
    bench_rf_predict,
    bench_confusion_matrix,
    bench_logreg_training,
    bench_knn_predict,
    bench_kmeans_training,
);
criterion_main!(benches);

