//! Benchmark dashboard generator — produces a self-contained HTML report
//! comparing scry-learn against linfa and smartcore.
//!
//! Includes: speed benchmarks, inference latency (p50/p99), training scaling
//! curves, memory footprint, pipeline benchmark, and 15-dataset parity table.
//!
//! Dogfoods `scry-chart` for SVG chart generation (bar + line charts).
//!
//! Run:
//! ```bash
//! cargo run --example bench_dashboard -p scry-learn --release --features serde
//! ```
//!
//! Outputs `bench_dashboard.html` in the current directory.

use std::fmt::Write as _;
use std::time::Instant;

use linfa::prelude::{Fit, Predict};
use scry_chart::chart::{BarChart, LineChart};
use scry_chart::data::Series;
use scry_chart::svg_export::render_to_svg;
use scry_chart::theme::Theme;
use smartcore::linalg::basic::matrix::DenseMatrix;

// ═══════════════════════════════════════════════════════════════════════════
// Data types
// ═══════════════════════════════════════════════════════════════════════════

/// A single benchmark measurement.
struct BenchResult {
    algorithm: &'static str,
    library: &'static str,
    time_us: f64,
    time_std: f64,
    accuracy: Option<f64>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Data generation (mirrors benchmark_audit.rs)
// ═══════════════════════════════════════════════════════════════════════════

#[allow(clippy::needless_range_loop)]
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

#[allow(clippy::needless_range_loop)]
fn gen_regression(n: usize, n_features: usize) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let mut rng = fastrand::Rng::with_seed(42);
    let mut col_major = vec![vec![0.0; n]; n_features];
    let mut target = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n_features {
            let v = rng.f64() * 10.0;
            col_major[j][i] = v;
            sum += v * (j as f64 + 1.0);
        }
        target[i] = sum + rng.f64() * 0.1;
    }
    let row_major: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n_features).map(|j| col_major[j][i]).collect())
        .collect();
    (col_major, target, row_major)
}

fn transpose(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if rows.is_empty() {
        return vec![];
    }
    let n_cols = rows[0].len();
    let n_rows = rows.len();
    (0..n_cols)
        .map(|j| (0..n_rows).map(|i| rows[i][j]).collect())
        .collect()
}

fn accuracy_f64(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(&t, &p)| (t - p).abs() < 1e-9)
        .count();
    correct as f64 / y_true.len() as f64
}

fn to_linfa_dataset(
    features: &[Vec<f64>],
    target: &[f64],
) -> linfa::DatasetBase<ndarray::Array2<f64>, ndarray::Array1<usize>> {
    let n = features.len();
    let m = features[0].len();
    let flat: Vec<f64> = features.iter().flat_map(|r| r.iter().copied()).collect();
    let x = ndarray::Array2::from_shape_vec((n, m), flat).unwrap();
    let y = ndarray::Array1::from_vec(target.iter().map(|&t| t as usize).collect());
    linfa::Dataset::new(x, y)
}

// ═══════════════════════════════════════════════════════════════════════════
// Timing helpers
// ═══════════════════════════════════════════════════════════════════════════

const WARMUP_ITERS: usize = 2;

/// Compute mean and standard deviation from a slice of durations (in µs).
fn mean_std(times: &[f64]) -> (f64, f64) {
    let n = times.len() as f64;
    let mean = times.iter().sum::<f64>() / n;
    let variance = times.iter().map(|&t| (t - mean).powi(2)).sum::<f64>() / n;
    (mean, variance.sqrt())
}

// ═══════════════════════════════════════════════════════════════════════════
// Benchmark runners
// ═══════════════════════════════════════════════════════════════════════════

fn bench_decision_tree() -> Vec<BenchResult> {
    let (features, target) = gen_classification(1000, 10);
    let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();
    let n_iters = 200;

    // ── scry-learn ──
    let data = scry_learn::prelude::Dataset::new(
        transpose(&features),
        target.clone(),
        (0..10).map(|i| format!("f{i}")).collect(),
        "target",
    );
    let mut scry_dt = scry_learn::prelude::DecisionTreeClassifier::new();
    scry_dt.fit(&data).unwrap();
    let scry_preds = scry_dt.predict(&features).unwrap();
    let scry_acc = accuracy_f64(&target, &scry_preds);

    // Warmup
    for _ in 0..WARMUP_ITERS {
        std::hint::black_box(scry_dt.predict(std::hint::black_box(&features)).unwrap());
    }
    let mut scry_times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let t0 = Instant::now();
        std::hint::black_box(scry_dt.predict(std::hint::black_box(&features)).unwrap());
        scry_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    let (scry_us, scry_std) = mean_std(&scry_times);

    // ── smartcore ──
    let x = DenseMatrix::from_2d_vec(&features).unwrap();
    let smart_dt =
        smartcore::tree::decision_tree_classifier::DecisionTreeClassifier::fit(
            &x,
            &target_i32,
            smartcore::tree::decision_tree_classifier::DecisionTreeClassifierParameters::default(),
        )
        .unwrap();
    let smart_preds_i32: Vec<i32> = smart_dt.predict(&x).unwrap();
    let smart_preds: Vec<f64> = smart_preds_i32.iter().map(|&p| p as f64).collect();
    let smart_acc = accuracy_f64(&target, &smart_preds);

    for _ in 0..WARMUP_ITERS {
        std::hint::black_box(smart_dt.predict(std::hint::black_box(&x)).unwrap());
    }
    let mut smart_times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let t0 = Instant::now();
        std::hint::black_box(smart_dt.predict(std::hint::black_box(&x)).unwrap());
        smart_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    let (smart_us, smart_std) = mean_std(&smart_times);

    // ── linfa ──
    let linfa_ds = to_linfa_dataset(&features, &target);
    let linfa_dt = linfa_trees::DecisionTree::params().fit(&linfa_ds).unwrap();
    let linfa_preds_arr = linfa_dt.predict(&linfa_ds);
    let linfa_preds: Vec<f64> = linfa_preds_arr.iter().map(|&p| p as f64).collect();
    let linfa_acc = accuracy_f64(&target, &linfa_preds);

    for _ in 0..WARMUP_ITERS {
        std::hint::black_box(linfa_dt.predict(std::hint::black_box(&linfa_ds)));
    }
    let mut linfa_times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let t0 = Instant::now();
        std::hint::black_box(linfa_dt.predict(std::hint::black_box(&linfa_ds)));
        linfa_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    let (linfa_us, linfa_std) = mean_std(&linfa_times);

    vec![
        BenchResult { algorithm: "DT Predict", library: "scry-learn", time_us: scry_us, time_std: scry_std, accuracy: Some(scry_acc) },
        BenchResult { algorithm: "DT Predict", library: "smartcore", time_us: smart_us, time_std: smart_std, accuracy: Some(smart_acc) },
        BenchResult { algorithm: "DT Predict", library: "linfa", time_us: linfa_us, time_std: linfa_std, accuracy: Some(linfa_acc) },
    ]
}

fn bench_random_forest() -> Vec<BenchResult> {
    let (features, target) = gen_classification(2000, 10);
    let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();
    let n_trees = 100;
    let n_train_iters = 5;

    // ── scry-learn train ──
    // Warmup
    for _ in 0..WARMUP_ITERS {
        let d = scry_learn::prelude::Dataset::new(
            transpose(&features), target.clone(),
            (0..10).map(|i| format!("f{i}")).collect(), "target",
        );
        let mut rf = scry_learn::prelude::RandomForestClassifier::new()
            .n_estimators(n_trees).max_depth(8);
        rf.fit(std::hint::black_box(&d)).unwrap();
    }
    let mut scry_times = Vec::with_capacity(n_train_iters);
    for _ in 0..n_train_iters {
        let d = scry_learn::prelude::Dataset::new(
            transpose(&features), target.clone(),
            (0..10).map(|i| format!("f{i}")).collect(), "target",
        );
        let mut rf = scry_learn::prelude::RandomForestClassifier::new()
            .n_estimators(n_trees).max_depth(8);
        let t0 = Instant::now();
        rf.fit(std::hint::black_box(&d)).unwrap();
        scry_times.push(t0.elapsed().as_micros() as f64);
    }
    let (scry_train_us, scry_train_std) = mean_std(&scry_times);

    // Get accuracy
    let data = scry_learn::prelude::Dataset::new(
        transpose(&features), target.clone(),
        (0..10).map(|i| format!("f{i}")).collect(), "target",
    );
    let mut scry_rf = scry_learn::prelude::RandomForestClassifier::new()
        .n_estimators(n_trees).max_depth(8);
    scry_rf.fit(&data).unwrap();
    let scry_preds = scry_rf.predict(&features).unwrap();
    let scry_acc = accuracy_f64(&target, &scry_preds);

    // ── smartcore train ──
    for _ in 0..WARMUP_ITERS {
        let x2 = DenseMatrix::from_2d_vec(&features).unwrap();
        let p = smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters::default()
            .with_n_trees(n_trees as u16).with_max_depth(8);
        let _ = smartcore::ensemble::random_forest_classifier::RandomForestClassifier::fit(
            std::hint::black_box(&x2), std::hint::black_box(&target_i32), p,
        ).unwrap();
    }
    let mut smart_times = Vec::with_capacity(n_train_iters);
    for _ in 0..n_train_iters {
        let x2 = DenseMatrix::from_2d_vec(&features).unwrap();
        let p = smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters::default()
            .with_n_trees(n_trees as u16).with_max_depth(8);
        let t0 = Instant::now();
        let _ = smartcore::ensemble::random_forest_classifier::RandomForestClassifier::fit(
            std::hint::black_box(&x2), std::hint::black_box(&target_i32), p,
        ).unwrap();
        smart_times.push(t0.elapsed().as_micros() as f64);
    }
    let (smart_train_us, smart_train_std) = mean_std(&smart_times);

    let x = DenseMatrix::from_2d_vec(&features).unwrap();
    let smart_params = smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters::default()
        .with_n_trees(n_trees as u16).with_max_depth(8);
    let smart_rf = smartcore::ensemble::random_forest_classifier::RandomForestClassifier::fit(
        &x, &target_i32, smart_params,
    ).unwrap();
    let smart_preds_i32: Vec<i32> = smart_rf.predict(&x).unwrap();
    let smart_preds: Vec<f64> = smart_preds_i32.iter().map(|&p| p as f64).collect();
    let smart_acc = accuracy_f64(&target, &smart_preds);

    // ── linfa train ──
    for _ in 0..WARMUP_ITERS {
        let ds = to_linfa_dataset(&features, &target);
        let _ = linfa_ensemble::RandomForestParams::new(
            linfa_trees::DecisionTree::params().max_depth(Some(8)),
        )
        .ensemble_size(n_trees)
        .bootstrap_proportion(0.7)
        .feature_proportion(0.3)
        .fit(std::hint::black_box(&ds))
        .unwrap();
    }
    let mut linfa_times = Vec::with_capacity(n_train_iters);
    for _ in 0..n_train_iters {
        let ds = to_linfa_dataset(&features, &target);
        let t0 = Instant::now();
        let _ = linfa_ensemble::RandomForestParams::new(
            linfa_trees::DecisionTree::params().max_depth(Some(8)),
        )
        .ensemble_size(n_trees)
        .bootstrap_proportion(0.7)
        .feature_proportion(0.3)
        .fit(std::hint::black_box(&ds))
        .unwrap();
        linfa_times.push(t0.elapsed().as_micros() as f64);
    }
    let (linfa_train_us, linfa_train_std) = mean_std(&linfa_times);

    let linfa_ds = to_linfa_dataset(&features, &target);
    let linfa_rf = linfa_ensemble::RandomForestParams::new(
        linfa_trees::DecisionTree::params().max_depth(Some(8)),
    )
    .ensemble_size(n_trees)
    .bootstrap_proportion(0.7)
    .feature_proportion(0.3)
    .fit(&linfa_ds)
    .unwrap();
    let linfa_preds_arr = linfa_rf.predict(&linfa_ds);
    let linfa_preds: Vec<f64> = linfa_preds_arr.iter().map(|&p| p as f64).collect();
    let linfa_acc = accuracy_f64(&target, &linfa_preds);

    vec![
        BenchResult { algorithm: "RF Train",  library: "scry-learn", time_us: scry_train_us,  time_std: scry_train_std,  accuracy: Some(scry_acc) },
        BenchResult { algorithm: "RF Train",  library: "smartcore",  time_us: smart_train_us, time_std: smart_train_std, accuracy: Some(smart_acc) },
        BenchResult { algorithm: "RF Train",  library: "linfa",      time_us: linfa_train_us, time_std: linfa_train_std, accuracy: Some(linfa_acc) },
    ]
}

fn bench_gbt_regressor() -> Vec<BenchResult> {
    let n_estimators = 100;
    let n_iters = 3;
    let (col_major, target, _row_major) = gen_regression(2000, 10);

    // ── scry-learn (exclusive — no competitor has GBT) ──
    let scry_ds = scry_learn::dataset::Dataset::new(
        col_major, target,
        (0..10).map(|i| format!("f{i}")).collect(), "y",
    );

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let mut gbr = scry_learn::tree::GradientBoostingRegressor::new()
            .n_estimators(n_estimators).learning_rate(0.1).max_depth(3);
        gbr.fit(std::hint::black_box(&scry_ds)).unwrap();
    }
    let mut scry_times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let mut gbr = scry_learn::tree::GradientBoostingRegressor::new()
            .n_estimators(n_estimators).learning_rate(0.1).max_depth(3);
        let t0 = Instant::now();
        gbr.fit(std::hint::black_box(&scry_ds)).unwrap();
        scry_times.push(t0.elapsed().as_micros() as f64);
    }
    let (scry_us, scry_std) = mean_std(&scry_times);

    vec![
        BenchResult { algorithm: "GBT Train", library: "scry-learn", time_us: scry_us, time_std: scry_std, accuracy: None },
    ]
}

fn bench_hist_gbt() -> Vec<BenchResult> {
    let n_estimators = 100;
    let n_iters = 3;
    let (col_major, target, _row_major) = gen_regression(2000, 10);

    let scry_ds = scry_learn::dataset::Dataset::new(
        col_major, target,
        (0..10).map(|i| format!("f{i}")).collect(), "y",
    );

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let mut hgbt = scry_learn::tree::HistGradientBoostingRegressor::new()
            .n_estimators(n_estimators).learning_rate(0.1);
        hgbt.fit(std::hint::black_box(&scry_ds)).unwrap();
    }
    let mut scry_times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let mut hgbt = scry_learn::tree::HistGradientBoostingRegressor::new()
            .n_estimators(n_estimators).learning_rate(0.1);
        let t0 = Instant::now();
        hgbt.fit(std::hint::black_box(&scry_ds)).unwrap();
        scry_times.push(t0.elapsed().as_micros() as f64);
    }
    let (scry_us, scry_std) = mean_std(&scry_times);

    vec![
        BenchResult { algorithm: "HistGBT Train", library: "scry-learn", time_us: scry_us, time_std: scry_std, accuracy: None },
    ]
}

fn bench_logistic_regression() -> Vec<BenchResult> {
    let (features, target) = gen_classification(1000, 10);
    let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();
    let target_bool: Vec<bool> = target.iter().map(|&t| t > 0.5).collect();
    let n_iters = 20;

    // ── scry-learn ──
    // Warmup
    for _ in 0..WARMUP_ITERS {
        let d = scry_learn::prelude::Dataset::new(
            transpose(&features), target.clone(),
            (0..10).map(|i| format!("f{i}")).collect(), "target",
        );
        let mut lr = scry_learn::prelude::LogisticRegression::new()
            .max_iter(200);
        lr.fit(std::hint::black_box(&d)).unwrap();
    }
    let mut scry_times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let d = scry_learn::prelude::Dataset::new(
            transpose(&features), target.clone(),
            (0..10).map(|i| format!("f{i}")).collect(), "target",
        );
        let mut lr = scry_learn::prelude::LogisticRegression::new()
            .max_iter(200);
        let t0 = Instant::now();
        lr.fit(std::hint::black_box(&d)).unwrap();
        scry_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    let (scry_us, scry_std) = mean_std(&scry_times);

    // ── smartcore ──
    for _ in 0..WARMUP_ITERS {
        let x2 = DenseMatrix::from_2d_vec(&features).unwrap();
        let _ = smartcore::linear::logistic_regression::LogisticRegression::fit(
            std::hint::black_box(&x2),
            std::hint::black_box(&target_i32),
            smartcore::linear::logistic_regression::LogisticRegressionParameters::default(),
        ).unwrap();
    }
    let mut smart_times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let x2 = DenseMatrix::from_2d_vec(&features).unwrap();
        let t0 = Instant::now();
        let _ = smartcore::linear::logistic_regression::LogisticRegression::fit(
            std::hint::black_box(&x2),
            std::hint::black_box(&target_i32),
            smartcore::linear::logistic_regression::LogisticRegressionParameters::default(),
        ).unwrap();
        smart_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    let (smart_us, smart_std) = mean_std(&smart_times);

    // ── linfa-logistic ──
    let flat: Vec<f64> = features.iter().flat_map(|r| r.iter().copied()).collect();
    let x_nd = ndarray::Array2::from_shape_vec((1000, 10), flat).unwrap();
    let y_nd = ndarray::Array1::from_vec(target_bool);
    let linfa_ds = linfa::Dataset::new(x_nd, y_nd);
    for _ in 0..WARMUP_ITERS {
        let _ = linfa_logistic::LogisticRegression::default()
            .max_iterations(200)
            .fit(std::hint::black_box(&linfa_ds))
            .unwrap();
    }
    let mut linfa_times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let t0 = Instant::now();
        let _ = linfa_logistic::LogisticRegression::default()
            .max_iterations(200)
            .fit(std::hint::black_box(&linfa_ds))
            .unwrap();
        linfa_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    let (linfa_us, linfa_std) = mean_std(&linfa_times);

    vec![
        BenchResult { algorithm: "LogReg Train", library: "scry-learn", time_us: scry_us,  time_std: scry_std,  accuracy: None },
        BenchResult { algorithm: "LogReg Train", library: "smartcore",  time_us: smart_us, time_std: smart_std, accuracy: None },
        BenchResult { algorithm: "LogReg Train", library: "linfa",      time_us: linfa_us, time_std: linfa_std, accuracy: None },
    ]
}

fn bench_knn() -> Vec<BenchResult> {
    let (features, target) = gen_classification(1000, 10);
    let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();
    let n_iters = 50;

    // ── scry-learn ──
    let data = scry_learn::prelude::Dataset::new(
        transpose(&features), target,
        (0..10).map(|i| format!("f{i}")).collect(), "target",
    );
    let mut scry_knn = scry_learn::prelude::KnnClassifier::new().k(5);
    scry_knn.fit(&data).unwrap();
    let matrix = data.feature_matrix();

    for _ in 0..WARMUP_ITERS {
        std::hint::black_box(scry_knn.predict(std::hint::black_box(&matrix)).unwrap());
    }
    let mut scry_times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let t0 = Instant::now();
        std::hint::black_box(scry_knn.predict(std::hint::black_box(&matrix)).unwrap());
        scry_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    let (scry_us, scry_std) = mean_std(&scry_times);

    // ── smartcore ──
    let x = DenseMatrix::from_2d_vec(&features).unwrap();
    let smart_knn = smartcore::neighbors::knn_classifier::KNNClassifier::fit(
        &x, &target_i32,
        smartcore::neighbors::knn_classifier::KNNClassifierParameters::default().with_k(5),
    ).unwrap();

    for _ in 0..WARMUP_ITERS {
        std::hint::black_box(smart_knn.predict(std::hint::black_box(&x)).unwrap());
    }
    let mut smart_times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let t0 = Instant::now();
        std::hint::black_box(smart_knn.predict(std::hint::black_box(&x)).unwrap());
        smart_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    let (smart_us, smart_std) = mean_std(&smart_times);

    vec![
        BenchResult { algorithm: "KNN Predict", library: "scry-learn", time_us: scry_us,  time_std: scry_std,  accuracy: None },
        BenchResult { algorithm: "KNN Predict", library: "smartcore",  time_us: smart_us, time_std: smart_std, accuracy: None },
    ]
}

fn bench_kmeans() -> Vec<BenchResult> {
    let (features, target) = gen_classification(2000, 10);
    let n_iters = 10;

    // ── scry-learn ──
    for _ in 0..WARMUP_ITERS {
        let d = scry_learn::prelude::Dataset::new(
            transpose(&features), target.clone(),
            (0..10).map(|i| format!("f{i}")).collect(), "target",
        );
        let mut km = scry_learn::prelude::KMeans::new(3).seed(42).max_iter(100).n_init(1);
        km.fit(std::hint::black_box(&d)).unwrap();
    }
    let mut scry_times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let d = scry_learn::prelude::Dataset::new(
            transpose(&features), target.clone(),
            (0..10).map(|i| format!("f{i}")).collect(), "target",
        );
        let mut km = scry_learn::prelude::KMeans::new(3).seed(42).max_iter(100).n_init(1);
        let t0 = Instant::now();
        km.fit(std::hint::black_box(&d)).unwrap();
        scry_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    let (scry_us, scry_std) = mean_std(&scry_times);

    // ── linfa-clustering ──
    let flat: Vec<f64> = features.iter().flat_map(|r| r.iter().copied()).collect();
    let x_nd = ndarray::Array2::from_shape_vec((2000, 10), flat).unwrap();
    for _ in 0..WARMUP_ITERS {
        let ds = linfa::DatasetBase::from(x_nd.clone());
        let _ = linfa_clustering::KMeans::params_with_rng(3, rand::thread_rng())
            .max_n_iterations(100)
            .fit(std::hint::black_box(&ds))
            .unwrap();
    }
    let mut linfa_times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let ds = linfa::DatasetBase::from(x_nd.clone());
        let t0 = Instant::now();
        let _ = linfa_clustering::KMeans::params_with_rng(3, rand::thread_rng())
            .max_n_iterations(100)
            .fit(std::hint::black_box(&ds))
            .unwrap();
        linfa_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    let (linfa_us, linfa_std) = mean_std(&linfa_times);

    vec![
        BenchResult { algorithm: "K-Means Train", library: "scry-learn", time_us: scry_us,  time_std: scry_std,  accuracy: None },
        BenchResult { algorithm: "K-Means Train", library: "linfa",      time_us: linfa_us, time_std: linfa_std, accuracy: None },
    ]
}

fn bench_pca() -> Vec<BenchResult> {
    let n_samples = 2000;
    let n_features = 20;
    let n_components = 5;
    let n_iters = 20;

    let (features_row, target) = gen_classification(n_samples, n_features);
    let features_col = transpose(&features_row);

    // ── scry-learn ──
    let scry_ds = scry_learn::prelude::Dataset::new(
        features_col, target,
        (0..n_features).map(|i| format!("f{i}")).collect(), "target",
    );
    for _ in 0..WARMUP_ITERS {
        let mut pca = scry_learn::prelude::Pca::with_n_components(n_components);
        scry_learn::preprocess::Transformer::fit(&mut pca, std::hint::black_box(&scry_ds)).unwrap();
    }
    let mut scry_times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let mut pca = scry_learn::prelude::Pca::with_n_components(n_components);
        let t0 = Instant::now();
        scry_learn::preprocess::Transformer::fit(&mut pca, std::hint::black_box(&scry_ds)).unwrap();
        scry_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    let (scry_us, scry_std) = mean_std(&scry_times);

    // ── smartcore ──
    let x = DenseMatrix::from_2d_vec(&features_row).unwrap();
    let smart_params = smartcore::decomposition::pca::PCAParameters::default()
        .with_n_components(n_components);
    for _ in 0..WARMUP_ITERS {
        let _ = smartcore::decomposition::pca::PCA::fit(
            std::hint::black_box(&x), smart_params.clone(),
        ).unwrap();
    }
    let mut smart_times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let t0 = Instant::now();
        let _ = smartcore::decomposition::pca::PCA::fit(
            std::hint::black_box(&x), smart_params.clone(),
        ).unwrap();
        smart_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    let (smart_us, smart_std) = mean_std(&smart_times);

    // ── linfa-reduction ──
    let flat: Vec<f64> = features_row.iter().flat_map(|r| r.iter().copied()).collect();
    let x_nd = ndarray::Array2::from_shape_vec((n_samples, n_features), flat).unwrap();
    let linfa_ds = linfa::Dataset::new(
        x_nd,
        ndarray::Array1::<usize>::from_vec(scry_ds.target.iter().map(|&t| t as usize).collect()),
    );
    for _ in 0..WARMUP_ITERS {
        let _ = linfa_reduction::Pca::params(n_components)
            .fit(std::hint::black_box(&linfa_ds))
            .unwrap();
    }
    let mut linfa_times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let t0 = Instant::now();
        let _ = linfa_reduction::Pca::params(n_components)
            .fit(std::hint::black_box(&linfa_ds))
            .unwrap();
        linfa_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    let (linfa_us, linfa_std) = mean_std(&linfa_times);

    vec![
        BenchResult { algorithm: "PCA Fit", library: "scry-learn", time_us: scry_us,  time_std: scry_std,  accuracy: None },
        BenchResult { algorithm: "PCA Fit", library: "smartcore",  time_us: smart_us, time_std: smart_std, accuracy: None },
        BenchResult { algorithm: "PCA Fit", library: "linfa",      time_us: linfa_us, time_std: linfa_std, accuracy: None },
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// SVM: scry LinearSVC vs smartcore SVC
// ═══════════════════════════════════════════════════════════════════════════

fn gen_regression_data(n: usize, d: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = fastrand::Rng::with_seed(42);
    let mut col_major = vec![vec![0.0; n]; d];
    let mut target = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..d {
            let v = rng.f64() * 10.0;
            col_major[j][i] = v;
            sum += v * (j as f64 + 1.0);
        }
        target[i] = sum + rng.f64() * 0.1;
    }
    (col_major, target)
}

fn bench_svm() -> Vec<BenchResult> {
    let n_samples = 1000;
    let n_features = 10;
    let n_iters = 20;
    let (features_row, target) = gen_classification(n_samples, n_features);
    let col_major = transpose(&features_row);
    let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();

    // scry-learn LinearSVC
    let scry_ds = scry_learn::prelude::Dataset::new(
        col_major.clone(), target.clone(),
        (0..n_features).map(|i| format!("f{i}")).collect(), "target",
    );
    for _ in 0..WARMUP_ITERS {
        let mut m = scry_learn::prelude::LinearSVC::new();
        m.fit(std::hint::black_box(&scry_ds)).unwrap();
    }
    let mut scry_times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let t0 = Instant::now();
        let mut m = scry_learn::prelude::LinearSVC::new();
        m.fit(std::hint::black_box(&scry_ds)).unwrap();
        scry_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    let (scry_us, scry_std) = mean_std(&scry_times);

    // smartcore SVC with linear kernel
    let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&features_row).unwrap();
    for _ in 0..WARMUP_ITERS {
        let knl = smartcore::svm::Kernels::linear();
        let params = smartcore::svm::svc::SVCParameters::default()
            .with_c(1.0).with_kernel(knl);
        let _ = smartcore::svm::svc::SVC::fit(
            std::hint::black_box(&x), std::hint::black_box(&target_i32),
            std::hint::black_box(&params),
        ).unwrap();
    }
    let mut smart_times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let knl = smartcore::svm::Kernels::linear();
        let params = smartcore::svm::svc::SVCParameters::default()
            .with_c(1.0).with_kernel(knl);
        let t0 = Instant::now();
        let _ = smartcore::svm::svc::SVC::fit(
            std::hint::black_box(&x), std::hint::black_box(&target_i32),
            std::hint::black_box(&params),
        ).unwrap();
        smart_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    let (smart_us, smart_std) = mean_std(&smart_times);

    vec![
        BenchResult { algorithm: "SVM Train", library: "scry-learn", time_us: scry_us,  time_std: scry_std,  accuracy: None },
        BenchResult { algorithm: "SVM Train", library: "smartcore",  time_us: smart_us, time_std: smart_std, accuracy: None },
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// Lasso: scry LassoRegression vs linfa-elasticnet
// ═══════════════════════════════════════════════════════════════════════════

fn bench_lasso() -> Vec<BenchResult> {
    let n_samples = 2000;
    let n_features = 10;
    let n_iters = 20;
    let (col_major, target) = gen_regression_data(n_samples, n_features);
    let row_major: Vec<Vec<f64>> = (0..n_samples)
        .map(|i| (0..n_features).map(|j| col_major[j][i]).collect())
        .collect();

    // scry-learn Lasso
    let scry_ds = scry_learn::prelude::Dataset::new(
        col_major.clone(), target.clone(),
        (0..n_features).map(|i| format!("f{i}")).collect(), "target",
    );
    for _ in 0..WARMUP_ITERS {
        let mut m = scry_learn::prelude::LassoRegression::new().alpha(0.1);
        m.fit(std::hint::black_box(&scry_ds)).unwrap();
    }
    let mut scry_times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let t0 = Instant::now();
        let mut m = scry_learn::prelude::LassoRegression::new().alpha(0.1);
        m.fit(std::hint::black_box(&scry_ds)).unwrap();
        scry_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    let (scry_us, scry_std) = mean_std(&scry_times);

    // linfa-elasticnet
    let flat: Vec<f64> = row_major.iter().flat_map(|r| r.iter().copied()).collect();
    let x = ndarray::Array2::from_shape_vec((n_samples, n_features), flat).unwrap();
    let y = ndarray::Array1::from_vec(target.clone());
    let linfa_ds = linfa::Dataset::new(x, y);

    for _ in 0..WARMUP_ITERS {
        use linfa::prelude::Fit;
        let _ = linfa_elasticnet::ElasticNet::<f64>::lasso()
            .penalty(0.1)
            .fit(std::hint::black_box(&linfa_ds))
            .unwrap();
    }
    let mut linfa_times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        use linfa::prelude::Fit;
        let t0 = Instant::now();
        let _ = linfa_elasticnet::ElasticNet::<f64>::lasso()
            .penalty(0.1)
            .fit(std::hint::black_box(&linfa_ds))
            .unwrap();
        linfa_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    let (linfa_us, linfa_std) = mean_std(&linfa_times);

    vec![
        BenchResult { algorithm: "Lasso Train", library: "scry-learn", time_us: scry_us,  time_std: scry_std,  accuracy: None },
        BenchResult { algorithm: "Lasso Train", library: "linfa",      time_us: linfa_us, time_std: linfa_std, accuracy: None },
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// NEW: Inference latency — single-sample p50/p99
// ═══════════════════════════════════════════════════════════════════════════

/// A single-sample inference latency measurement.
struct InferenceResult {
    model: &'static str,
    p50_ns: u64,
    p99_ns: u64,
    iterations: usize,
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    let idx = ((sorted.len() as f64) * p / 100.0).ceil() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn bench_inference_latency() -> Vec<InferenceResult> {
    let (features, target) = gen_classification(1000, 10);
    let col_major = transpose(&features);
    let sample = vec![features[0].clone()]; // single row for prediction

    let n_iters = 10_000;
    let mut results = Vec::new();

    // ── DecisionTreeClassifier ──
    {
        let data = scry_learn::prelude::Dataset::new(
            col_major.clone(), target.clone(),
            (0..10).map(|i| format!("f{i}")).collect(), "target",
        );
        let mut dt = scry_learn::prelude::DecisionTreeClassifier::new();
        dt.fit(&data).unwrap();
        // Warmup
        for _ in 0..100 {
            std::hint::black_box(dt.predict(std::hint::black_box(&sample)).unwrap());
        }
        let mut latencies = Vec::with_capacity(n_iters);
        for _ in 0..n_iters {
            let t0 = Instant::now();
            std::hint::black_box(dt.predict(std::hint::black_box(&sample)).unwrap());
            latencies.push(t0.elapsed().as_nanos() as u64);
        }
        latencies.sort_unstable();
        results.push(InferenceResult {
            model: "DecisionTree",
            p50_ns: percentile(&latencies, 50.0),
            p99_ns: percentile(&latencies, 99.0),
            iterations: n_iters,
        });
    }

    // ── RandomForestClassifier ──
    {
        let data = scry_learn::prelude::Dataset::new(
            col_major.clone(), target.clone(),
            (0..10).map(|i| format!("f{i}")).collect(), "target",
        );
        let mut rf = scry_learn::prelude::RandomForestClassifier::new()
            .n_estimators(100).max_depth(8);
        rf.fit(&data).unwrap();
        for _ in 0..100 {
            std::hint::black_box(rf.predict(std::hint::black_box(&sample)).unwrap());
        }
        let mut latencies = Vec::with_capacity(n_iters);
        for _ in 0..n_iters {
            let t0 = Instant::now();
            std::hint::black_box(rf.predict(std::hint::black_box(&sample)).unwrap());
            latencies.push(t0.elapsed().as_nanos() as u64);
        }
        latencies.sort_unstable();
        results.push(InferenceResult {
            model: "RandomForest",
            p50_ns: percentile(&latencies, 50.0),
            p99_ns: percentile(&latencies, 99.0),
            iterations: n_iters,
        });
    }

    // ── GradientBoostingClassifier ──
    {
        let data = scry_learn::prelude::Dataset::new(
            col_major.clone(), target.clone(),
            (0..10).map(|i| format!("f{i}")).collect(), "target",
        );
        let mut gbt = scry_learn::prelude::GradientBoostingClassifier::new()
            .n_estimators(100).learning_rate(0.1).max_depth(3);
        gbt.fit(&data).unwrap();
        for _ in 0..100 {
            std::hint::black_box(gbt.predict(std::hint::black_box(&sample)).unwrap());
        }
        let mut latencies = Vec::with_capacity(n_iters);
        for _ in 0..n_iters {
            let t0 = Instant::now();
            std::hint::black_box(gbt.predict(std::hint::black_box(&sample)).unwrap());
            latencies.push(t0.elapsed().as_nanos() as u64);
        }
        latencies.sort_unstable();
        results.push(InferenceResult {
            model: "GBT",
            p50_ns: percentile(&latencies, 50.0),
            p99_ns: percentile(&latencies, 99.0),
            iterations: n_iters,
        });
    }

    // ── HistGradientBoostingClassifier ──
    {
        let data = scry_learn::prelude::Dataset::new(
            col_major.clone(), target.clone(),
            (0..10).map(|i| format!("f{i}")).collect(), "target",
        );
        let mut hgbt = scry_learn::prelude::HistGradientBoostingClassifier::new()
            .n_estimators(100).learning_rate(0.1);
        hgbt.fit(&data).unwrap();
        for _ in 0..100 {
            std::hint::black_box(hgbt.predict(std::hint::black_box(&sample)).unwrap());
        }
        let mut latencies = Vec::with_capacity(n_iters);
        for _ in 0..n_iters {
            let t0 = Instant::now();
            std::hint::black_box(hgbt.predict(std::hint::black_box(&sample)).unwrap());
            latencies.push(t0.elapsed().as_nanos() as u64);
        }
        latencies.sort_unstable();
        results.push(InferenceResult {
            model: "HistGBT",
            p50_ns: percentile(&latencies, 50.0),
            p99_ns: percentile(&latencies, 99.0),
            iterations: n_iters,
        });
    }

    // ── KNN ──
    {
        let data = scry_learn::prelude::Dataset::new(
            col_major.clone(), target.clone(),
            (0..10).map(|i| format!("f{i}")).collect(), "target",
        );
        let mut knn = scry_learn::prelude::KnnClassifier::new().k(5);
        knn.fit(&data).unwrap();
        for _ in 0..100 {
            std::hint::black_box(knn.predict(std::hint::black_box(&sample)).unwrap());
        }
        let mut latencies = Vec::with_capacity(n_iters);
        for _ in 0..n_iters {
            let t0 = Instant::now();
            std::hint::black_box(knn.predict(std::hint::black_box(&sample)).unwrap());
            latencies.push(t0.elapsed().as_nanos() as u64);
        }
        latencies.sort_unstable();
        results.push(InferenceResult {
            model: "KNN (k=5)",
            p50_ns: percentile(&latencies, 50.0),
            p99_ns: percentile(&latencies, 99.0),
            iterations: n_iters,
        });
    }

    // ── LogisticRegression ──
    {
        let data = scry_learn::prelude::Dataset::new(
            col_major.clone(), target.clone(),
            (0..10).map(|i| format!("f{i}")).collect(), "target",
        );
        let mut lr = scry_learn::prelude::LogisticRegression::new().max_iter(200);
        lr.fit(&data).unwrap();
        for _ in 0..100 {
            std::hint::black_box(lr.predict(std::hint::black_box(&sample)).unwrap());
        }
        let mut latencies = Vec::with_capacity(n_iters);
        for _ in 0..n_iters {
            let t0 = Instant::now();
            std::hint::black_box(lr.predict(std::hint::black_box(&sample)).unwrap());
            latencies.push(t0.elapsed().as_nanos() as u64);
        }
        latencies.sort_unstable();
        results.push(InferenceResult {
            model: "LogisticReg",
            p50_ns: percentile(&latencies, 50.0),
            p99_ns: percentile(&latencies, 99.0),
            iterations: n_iters,
        });
    }

    // ── GaussianNB ──
    {
        let data = scry_learn::prelude::Dataset::new(
            col_major.clone(), target.clone(),
            (0..10).map(|i| format!("f{i}")).collect(), "target",
        );
        let mut gnb = scry_learn::prelude::GaussianNb::new();
        gnb.fit(&data).unwrap();
        for _ in 0..100 {
            std::hint::black_box(gnb.predict(std::hint::black_box(&sample)).unwrap());
        }
        let mut latencies = Vec::with_capacity(n_iters);
        for _ in 0..n_iters {
            let t0 = Instant::now();
            std::hint::black_box(gnb.predict(std::hint::black_box(&sample)).unwrap());
            latencies.push(t0.elapsed().as_nanos() as u64);
        }
        latencies.sort_unstable();
        results.push(InferenceResult {
            model: "GaussianNB",
            p50_ns: percentile(&latencies, 50.0),
            p99_ns: percentile(&latencies, 99.0),
            iterations: n_iters,
        });
    }

    // ── LinearSVC ──
    {
        let data = scry_learn::prelude::Dataset::new(
            col_major.clone(), target.clone(),
            (0..10).map(|i| format!("f{i}")).collect(), "target",
        );
        let mut lsvc = scry_learn::prelude::LinearSVC::new();
        lsvc.fit(&data).unwrap();
        for _ in 0..100 {
            std::hint::black_box(lsvc.predict(std::hint::black_box(&sample)).unwrap());
        }
        let mut latencies = Vec::with_capacity(n_iters);
        for _ in 0..n_iters {
            let t0 = Instant::now();
            std::hint::black_box(lsvc.predict(std::hint::black_box(&sample)).unwrap());
            latencies.push(t0.elapsed().as_nanos() as u64);
        }
        latencies.sort_unstable();
        results.push(InferenceResult {
            model: "LinearSVC",
            p50_ns: percentile(&latencies, 50.0),
            p99_ns: percentile(&latencies, 99.0),
            iterations: n_iters,
        });
    }

    // ── BernoulliNB ──
    {
        let data = scry_learn::prelude::Dataset::new(
            col_major, target,
            (0..10).map(|i| format!("f{i}")).collect(), "target",
        );
        let mut bnb = scry_learn::prelude::BernoulliNB::new();
        bnb.fit(&data).unwrap();
        for _ in 0..100 {
            std::hint::black_box(bnb.predict(std::hint::black_box(&sample)).unwrap());
        }
        let mut latencies = Vec::with_capacity(n_iters);
        for _ in 0..n_iters {
            let t0 = Instant::now();
            std::hint::black_box(bnb.predict(std::hint::black_box(&sample)).unwrap());
            latencies.push(t0.elapsed().as_nanos() as u64);
        }
        latencies.sort_unstable();
        results.push(InferenceResult {
            model: "BernoulliNB",
            p50_ns: percentile(&latencies, 50.0),
            p99_ns: percentile(&latencies, 99.0),
            iterations: n_iters,
        });
    }

    results
}

// ═══════════════════════════════════════════════════════════════════════════
// NEW: Training time scaling curves
// ═══════════════════════════════════════════════════════════════════════════

struct ScalingPoint {
    model: &'static str,
    n_samples: usize,
    train_us: f64,
}

fn bench_scaling() -> (Vec<ScalingPoint>, String) {
    let sizes = [100, 1000, 10_000];
    let mut points = Vec::new();

    for &n in &sizes {
        let (features, target) = gen_classification(n, 10);
        let col_major = transpose(&features);

        // ── DT ──
        {
            let data = scry_learn::prelude::Dataset::new(
                col_major.clone(), target.clone(),
                (0..10).map(|i| format!("f{i}")).collect(), "target",
            );
            let n_iter = if n <= 1000 { 10 } else { 3 };
            let mut times = Vec::new();
            for _ in 0..n_iter {
                let mut m = scry_learn::prelude::DecisionTreeClassifier::new();
                let t0 = Instant::now();
                m.fit(std::hint::black_box(&data)).unwrap();
                times.push(t0.elapsed().as_micros() as f64);
            }
            let (mean, _) = mean_std(&times);
            points.push(ScalingPoint { model: "DT", n_samples: n, train_us: mean });
        }

        // ── RF ──
        {
            let data = scry_learn::prelude::Dataset::new(
                col_major.clone(), target.clone(),
                (0..10).map(|i| format!("f{i}")).collect(), "target",
            );
            let n_iter = if n <= 1000 { 5 } else { 2 };
            let mut times = Vec::new();
            for _ in 0..n_iter {
                let mut m = scry_learn::prelude::RandomForestClassifier::new()
                    .n_estimators(50).max_depth(8);
                let t0 = Instant::now();
                m.fit(std::hint::black_box(&data)).unwrap();
                times.push(t0.elapsed().as_micros() as f64);
            }
            let (mean, _) = mean_std(&times);
            points.push(ScalingPoint { model: "RF", n_samples: n, train_us: mean });
        }

        // ── GBT ──
        {
            let data = scry_learn::prelude::Dataset::new(
                col_major.clone(), target.clone(),
                (0..10).map(|i| format!("f{i}")).collect(), "target",
            );
            let n_iter = if n <= 1000 { 5 } else { 2 };
            let mut times = Vec::new();
            for _ in 0..n_iter {
                let mut m = scry_learn::prelude::GradientBoostingClassifier::new()
                    .n_estimators(50).learning_rate(0.1).max_depth(3);
                let t0 = Instant::now();
                m.fit(std::hint::black_box(&data)).unwrap();
                times.push(t0.elapsed().as_micros() as f64);
            }
            let (mean, _) = mean_std(&times);
            points.push(ScalingPoint { model: "GBT", n_samples: n, train_us: mean });
        }

        // ── HistGBT ──
        {
            let data = scry_learn::prelude::Dataset::new(
                col_major.clone(), target.clone(),
                (0..10).map(|i| format!("f{i}")).collect(), "target",
            );
            let n_iter = if n <= 1000 { 5 } else { 2 };
            let mut times = Vec::new();
            for _ in 0..n_iter {
                let mut m = scry_learn::prelude::HistGradientBoostingClassifier::new()
                    .n_estimators(50).learning_rate(0.1);
                let t0 = Instant::now();
                m.fit(std::hint::black_box(&data)).unwrap();
                times.push(t0.elapsed().as_micros() as f64);
            }
            let (mean, _) = mean_std(&times);
            points.push(ScalingPoint { model: "HistGBT", n_samples: n, train_us: mean });
        }

        // ── KNN ──
        {
            let data = scry_learn::prelude::Dataset::new(
                col_major.clone(), target.clone(),
                (0..10).map(|i| format!("f{i}")).collect(), "target",
            );
            let n_iter = if n <= 1000 { 10 } else { 3 };
            let mut times = Vec::new();
            for _ in 0..n_iter {
                let mut m = scry_learn::prelude::KnnClassifier::new().k(5);
                let t0 = Instant::now();
                m.fit(std::hint::black_box(&data)).unwrap();
                times.push(t0.elapsed().as_micros() as f64);
            }
            let (mean, _) = mean_std(&times);
            points.push(ScalingPoint { model: "KNN", n_samples: n, train_us: mean });
        }

        // ── LogReg ──
        {
            let data = scry_learn::prelude::Dataset::new(
                col_major.clone(), target.clone(),
                (0..10).map(|i| format!("f{i}")).collect(), "target",
            );
            let n_iter = if n <= 1000 { 10 } else { 3 };
            let mut times = Vec::new();
            for _ in 0..n_iter {
                let mut m = scry_learn::prelude::LogisticRegression::new().max_iter(200);
                let t0 = Instant::now();
                m.fit(std::hint::black_box(&data)).unwrap();
                times.push(t0.elapsed().as_micros() as f64);
            }
            let (mean, _) = mean_std(&times);
            points.push(ScalingPoint { model: "LogReg", n_samples: n, train_us: mean });
        }

        // ── LinearSVC ──
        {
            let data = scry_learn::prelude::Dataset::new(
                col_major, target,
                (0..10).map(|i| format!("f{i}")).collect(), "target",
            );
            let n_iter = if n <= 1000 { 10 } else { 3 };
            let mut times = Vec::new();
            for _ in 0..n_iter {
                let mut m = scry_learn::prelude::LinearSVC::new();
                let t0 = Instant::now();
                m.fit(std::hint::black_box(&data)).unwrap();
                times.push(t0.elapsed().as_micros() as f64);
            }
            let (mean, _) = mean_std(&times);
            points.push(ScalingPoint { model: "LinearSVC", n_samples: n, train_us: mean });
        }
    }

    // Build line chart (dogfooding scry-chart)
    let models = ["DT", "RF", "GBT", "HistGBT", "KNN", "LogReg", "LinearSVC"];
    let x_vals: Vec<f64> = sizes.iter().map(|&s| s as f64).collect();
    let mut chart = LineChart::new(vec![])
        .x_values(x_vals)
        .title("Training Time Scaling Curves")
        .x_label("Number of Samples (N)")
        .y_label("Training Time (µs)")
        .theme(Theme::dark())
        .with_points();

    for model in &models {
        let values: Vec<f64> = sizes.iter().map(|&n| {
            points.iter()
                .find(|p| p.model == *model && p.n_samples == n)
                .map_or(0.0, |p| p.train_us)
        }).collect();
        chart = chart.add_named_series(*model, &values);
    }

    let svg = render_to_svg(&chart.build(), 700, 400);
    (points, svg)
}

// ═══════════════════════════════════════════════════════════════════════════
// NEW: Memory footprint (serialized model size)
// ═══════════════════════════════════════════════════════════════════════════

struct MemoryResult {
    model: &'static str,
    serialized_bytes: usize,
}

#[cfg(feature = "serde")]
fn bench_model_size() -> Vec<MemoryResult> {
    let (features, target) = gen_classification(1000, 10);
    let col_major = transpose(&features);
    let mut results = Vec::new();

    // ── DT ──
    {
        let data = scry_learn::prelude::Dataset::new(
            col_major.clone(), target.clone(),
            (0..10).map(|i| format!("f{i}")).collect(), "target",
        );
        let mut m = scry_learn::prelude::DecisionTreeClassifier::new();
        m.fit(&data).unwrap();
        let json = serde_json::to_string(&m).unwrap();
        results.push(MemoryResult { model: "DecisionTree", serialized_bytes: json.len() });
    }

    // ── RF ──
    {
        let data = scry_learn::prelude::Dataset::new(
            col_major.clone(), target.clone(),
            (0..10).map(|i| format!("f{i}")).collect(), "target",
        );
        let mut m = scry_learn::prelude::RandomForestClassifier::new()
            .n_estimators(100).max_depth(8);
        m.fit(&data).unwrap();
        let json = serde_json::to_string(&m).unwrap();
        results.push(MemoryResult { model: "RandomForest", serialized_bytes: json.len() });
    }

    // ── GBT ──
    {
        let data = scry_learn::prelude::Dataset::new(
            col_major.clone(), target.clone(),
            (0..10).map(|i| format!("f{i}")).collect(), "target",
        );
        let mut m = scry_learn::prelude::GradientBoostingClassifier::new()
            .n_estimators(100).learning_rate(0.1).max_depth(3);
        m.fit(&data).unwrap();
        let json = serde_json::to_string(&m).unwrap();
        results.push(MemoryResult { model: "GBT", serialized_bytes: json.len() });
    }

    // ── HistGBT ──
    {
        let data = scry_learn::prelude::Dataset::new(
            col_major.clone(), target.clone(),
            (0..10).map(|i| format!("f{i}")).collect(), "target",
        );
        let mut m = scry_learn::prelude::HistGradientBoostingClassifier::new()
            .n_estimators(100).learning_rate(0.1);
        m.fit(&data).unwrap();
        let json = serde_json::to_string(&m).unwrap();
        results.push(MemoryResult { model: "HistGBT", serialized_bytes: json.len() });
    }

    // ── KNN ──
    {
        let data = scry_learn::prelude::Dataset::new(
            col_major.clone(), target.clone(),
            (0..10).map(|i| format!("f{i}")).collect(), "target",
        );
        let mut m = scry_learn::prelude::KnnClassifier::new().k(5);
        m.fit(&data).unwrap();
        let json = serde_json::to_string(&m).unwrap();
        results.push(MemoryResult { model: "KNN (k=5)", serialized_bytes: json.len() });
    }

    // ── LogReg ──
    {
        let data = scry_learn::prelude::Dataset::new(
            col_major.clone(), target.clone(),
            (0..10).map(|i| format!("f{i}")).collect(), "target",
        );
        let mut m = scry_learn::prelude::LogisticRegression::new().max_iter(200);
        m.fit(&data).unwrap();
        let json = serde_json::to_string(&m).unwrap();
        results.push(MemoryResult { model: "LogisticReg", serialized_bytes: json.len() });
    }

    // ── GaussianNB ──
    {
        let data = scry_learn::prelude::Dataset::new(
            col_major.clone(), target.clone(),
            (0..10).map(|i| format!("f{i}")).collect(), "target",
        );
        let mut m = scry_learn::prelude::GaussianNb::new();
        m.fit(&data).unwrap();
        let json = serde_json::to_string(&m).unwrap();
        results.push(MemoryResult { model: "GaussianNB", serialized_bytes: json.len() });
    }

    // ── LinearSVC ──
    {
        let data = scry_learn::prelude::Dataset::new(
            col_major, target,
            (0..10).map(|i| format!("f{i}")).collect(), "target",
        );
        let mut m = scry_learn::prelude::LinearSVC::new();
        m.fit(&data).unwrap();
        let json = serde_json::to_string(&m).unwrap();
        results.push(MemoryResult { model: "LinearSVC", serialized_bytes: json.len() });
    }

    results
}

#[cfg(not(feature = "serde"))]
fn bench_model_size() -> Vec<MemoryResult> {
    Vec::new()
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// NEW: End-to-end pipeline benchmark
// ═══════════════════════════════════════════════════════════════════════════

struct PipelineBenchResult {
    pipeline_us: f64,
    pipeline_std: f64,
    raw_us: f64,
    raw_std: f64,
    overhead_pct: f64,
}

fn bench_pipeline() -> PipelineBenchResult {
    let (features, target) = gen_classification(2000, 10);
    let col_major = transpose(&features);
    let n_iters = 5;

    // ── Pipeline: StandardScaler → RF ──
    let mut pipeline_times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let data = scry_learn::prelude::Dataset::new(
            col_major.clone(), target.clone(),
            (0..10).map(|i| format!("f{i}")).collect(), "target",
        );
        let mut pipe = scry_learn::prelude::Pipeline::new()
            .add_transformer(scry_learn::prelude::StandardScaler::new())
            .set_model(scry_learn::prelude::RandomForestClassifier::new()
                .n_estimators(50).max_depth(8));
        let t0 = Instant::now();
        pipe.fit(std::hint::black_box(&data)).unwrap();
        let _ = pipe.predict(std::hint::black_box(&data)).unwrap();
        pipeline_times.push(t0.elapsed().as_micros() as f64);
    }
    let (pipe_us, pipe_std) = mean_std(&pipeline_times);

    // ── Raw RF (no scaler) ──
    let mut raw_times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let data = scry_learn::prelude::Dataset::new(
            col_major.clone(), target.clone(),
            (0..10).map(|i| format!("f{i}")).collect(), "target",
        );
        let mut rf = scry_learn::prelude::RandomForestClassifier::new()
            .n_estimators(50).max_depth(8);
        let t0 = Instant::now();
        rf.fit(std::hint::black_box(&data)).unwrap();
        let _ = rf.predict(std::hint::black_box(&features)).unwrap();
        raw_times.push(t0.elapsed().as_micros() as f64);
    }
    let (raw_us, raw_std) = mean_std(&raw_times);

    let overhead = if raw_us > 0.0 { ((pipe_us - raw_us) / raw_us) * 100.0 } else { 0.0 };

    PipelineBenchResult {
        pipeline_us: pipe_us,
        pipeline_std: pipe_std,
        raw_us,
        raw_std,
        overhead_pct: overhead,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Chart generation (dogfooding scry-chart)
// ═══════════════════════════════════════════════════════════════════════════

/// Build a grouped bar chart comparing timing results across libraries.
fn make_timing_chart(results: &[BenchResult], title: &str, y_label: &str) -> String {
    // Group by algorithm, preserving order of first appearance.
    let mut algorithms: Vec<&str> = Vec::new();
    for r in results {
        if !algorithms.contains(&r.algorithm) {
            algorithms.push(r.algorithm);
        }
    }

    let mut libraries: Vec<&str> = Vec::new();
    for r in results {
        if !libraries.contains(&r.library) {
            libraries.push(r.library);
        }
    }

    let labels: Vec<String> = algorithms.iter().copied().map(String::from).collect();

    // Build one Series per library.
    let series: Vec<Series> = libraries
        .iter()
        .map(|lib| {
            let values: Vec<f64> = algorithms
                .iter()
                .map(|algo| {
                    results
                        .iter()
                        .find(|r| r.algorithm == *algo && r.library == *lib)
                        .map_or(0.0, |r| r.time_us)
                })
                .collect();
            Series::new(*lib, values)
        })
        .collect();

    let chart = BarChart::new(labels, series)
        .title(title)
        .y_label(y_label)
        .theme(Theme::dark())
        .show_values()
        .build();

    render_to_svg(&chart, 700, 400)
}

/// Build a bar chart showing how many × faster scry-learn is vs each competitor.
fn make_speedup_chart(results: &[BenchResult]) -> String {
    let mut algorithms: Vec<&str> = Vec::new();
    for r in results {
        if !algorithms.contains(&r.algorithm) {
            algorithms.push(r.algorithm);
        }
    }

    let mut competitor_names: Vec<&str> = Vec::new();
    for r in results {
        if r.library != "scry-learn" && !competitor_names.contains(&r.library) {
            competitor_names.push(r.library);
        }
    }

    let labels: Vec<String> = algorithms.iter().copied().map(String::from).collect();

    let series: Vec<Series> = competitor_names
        .iter()
        .map(|comp| {
            let values: Vec<f64> = algorithms
                .iter()
                .map(|algo| {
                    let scry_time = results
                        .iter()
                        .find(|r| r.algorithm == *algo && r.library == "scry-learn")
                        .map_or(1.0, |r| r.time_us);
                    let comp_time = results
                        .iter()
                        .find(|r| r.algorithm == *algo && r.library == *comp)
                        .map_or(0.0, |r| r.time_us);
                    if scry_time > 0.0 && comp_time > 0.0 {
                        comp_time / scry_time
                    } else {
                        0.0
                    }
                })
                .collect();
            Series::new(format!("{comp} / scry"), values)
        })
        .collect();

    let chart = BarChart::new(labels, series)
        .title("Competitor Slowdown Factor (higher = scry-learn is faster)")
        .y_label("× slower than scry-learn")
        .theme(Theme::dark())
        .show_values()
        .build();

    render_to_svg(&chart, 700, 400)
}

/// Build a grouped bar chart showing accuracy comparison.
fn make_accuracy_chart(results: &[BenchResult]) -> String {
    let accuracy_results: Vec<&BenchResult> = results
        .iter()
        .filter(|r| r.accuracy.is_some())
        .collect();

    if accuracy_results.is_empty() {
        return String::new();
    }

    let mut algorithms: Vec<&str> = Vec::new();
    for r in &accuracy_results {
        if !algorithms.contains(&r.algorithm) {
            algorithms.push(r.algorithm);
        }
    }

    let mut libraries: Vec<&str> = Vec::new();
    for r in &accuracy_results {
        if !libraries.contains(&r.library) {
            libraries.push(r.library);
        }
    }

    let labels: Vec<String> = algorithms.iter().copied().map(String::from).collect();

    let series: Vec<Series> = libraries
        .iter()
        .map(|lib| {
            let values: Vec<f64> = algorithms
                .iter()
                .map(|algo| {
                    accuracy_results
                        .iter()
                        .find(|r| r.algorithm == *algo && r.library == *lib)
                        .and_then(|r| r.accuracy)
                        .map_or(0.0, |a| a * 100.0)
                })
                .collect();
            Series::new(*lib, values)
        })
        .collect();

    let chart = BarChart::new(labels, series)
        .title("Accuracy Comparison (%)")
        .y_label("Accuracy (%)")
        .y_range(90.0, 101.0)
        .theme(Theme::dark())
        .show_values()
        .build();

    render_to_svg(&chart, 700, 400)
}

// ═══════════════════════════════════════════════════════════════════════════
// Parity table — 5-fold CV on real UCI datasets vs sklearn
// ═══════════════════════════════════════════════════════════════════════════

struct ParityRow {
    dataset: &'static str,
    model: &'static str,
    scry_mean: f64,
    scry_std: f64,
    sklearn_mean: f64,
    sklearn_std: f64,
}

/// Stratified k-fold cross-validation returning (mean_accuracy, std_accuracy).
///
/// Matches sklearn's `StratifiedKFold(n_splits=k)` default (no shuffle):
/// groups indices by class label, distributes each class round-robin across folds
/// so every fold has proportional class representation.
fn kfold_accuracy(
    features_col_major: &[Vec<f64>],
    target: &[f64],
    k: usize,
    fit_predict: impl Fn(&scry_learn::dataset::Dataset, &[Vec<f64>]) -> Vec<f64>,
) -> (f64, f64) {
    let n = target.len();

    // ── Stratified fold assignment ──
    // Group sample indices by class label (preserving order within each class).
    let mut class_indices: std::collections::BTreeMap<i64, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, &t) in target.iter().enumerate() {
        class_indices.entry(t as i64).or_default().push(i);
    }

    // Assign each sample to a fold, distributing each class round-robin.
    let mut fold_of = vec![0usize; n];
    for indices in class_indices.values() {
        for (pos, &idx) in indices.iter().enumerate() {
            fold_of[idx] = pos % k;
        }
    }

    // ── Evaluate each fold ──
    let mut fold_accs = Vec::with_capacity(k);

    for fold in 0..k {
        let test_mask: Vec<bool> = fold_of.iter().map(|&f| f == fold).collect();

        // Split features (column-major)
        let mut train_features: Vec<Vec<f64>> = Vec::new();
        for col in features_col_major {
            let train_col: Vec<f64> = col.iter().zip(&test_mask)
                .filter(|(_, &is_test)| !is_test)
                .map(|(&v, _)| v)
                .collect();
            train_features.push(train_col);
        }

        // Build test rows (row-major for predict)
        let test_features_rows: Vec<Vec<f64>> = (0..n)
            .filter(|&i| test_mask[i])
            .map(|i| features_col_major.iter().map(|col| col[i]).collect())
            .collect();

        let train_target: Vec<f64> = target.iter().zip(&test_mask)
            .filter(|(_, &is_test)| !is_test)
            .map(|(&t, _)| t)
            .collect();
        let test_target: Vec<f64> = target.iter().zip(&test_mask)
            .filter(|(_, &is_test)| is_test)
            .map(|(&t, _)| t)
            .collect();

        let feat_names: Vec<String> = (0..features_col_major.len())
            .map(|j| format!("f{j}"))
            .collect();
        let train_ds = scry_learn::dataset::Dataset::new(
            train_features,
            train_target,
            feat_names,
            "target",
        );

        let preds = fit_predict(&train_ds, &test_features_rows);
        let correct = preds
            .iter()
            .zip(test_target.iter())
            .filter(|(p, t)| (**p - **t).abs() < 0.5)
            .count();
        fold_accs.push(correct as f64 / test_target.len() as f64);
    }

    let mean = fold_accs.iter().sum::<f64>() / fold_accs.len() as f64;
    let variance = fold_accs.iter().map(|a| (a - mean).powi(2)).sum::<f64>()
        / fold_accs.len() as f64;
    (mean, variance.sqrt())
}

/// Load a CSV fixture into column-major features.
fn load_fixture_features(name: &str) -> Vec<Vec<f64>> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name);
    let mut rdr = csv::Reader::from_path(&path)
        .unwrap_or_else(|e| panic!("Cannot open {}: {e}", path.display()));
    let n_cols = rdr.headers().unwrap().len();
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for result in rdr.records() {
        let record = result.unwrap();
        rows.push(record.iter().map(|s| s.parse::<f64>().unwrap()).collect());
    }
    let mut cols = vec![vec![0.0; rows.len()]; n_cols];
    for (i, row) in rows.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            cols[j][i] = val;
        }
    }
    cols
}

fn load_fixture_target(name: &str) -> Vec<f64> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name);
    let mut rdr = csv::Reader::from_path(&path)
        .unwrap_or_else(|e| panic!("Cannot open {}: {e}", path.display()));
    rdr.records()
        .map(|r| r.unwrap()[0].parse::<f64>().unwrap())
        .collect()
}

/// Load sklearn CV results from fixture JSON.
fn load_sklearn_cv(key: &str) -> (f64, f64) {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("sklearn_predictions.json");
    let data = std::fs::read_to_string(&path).unwrap();
    let v: serde_json::Value = serde_json::from_str(&data).unwrap();
    let mean = v[key]["mean"].as_f64().unwrap();
    let std = v[key]["std"].as_f64().unwrap();
    (mean, std)
}

fn compute_parity_table() -> Vec<ParityRow> {
    let mut rows = Vec::new();

    // Helper: add DT + KNN parity rows for a dataset
    fn add_dt_knn_rows(
        rows: &mut Vec<ParityRow>,
        dataset_label: &'static str,
        features_file: &str,
        target_file: &str,
        dt_cv_key: &str,
        knn_cv_key: &str,
    ) {
        let x = load_fixture_features(features_file);
        let y = load_fixture_target(target_file);

        // DT
        let (sk_mean, sk_std) = load_sklearn_cv(dt_cv_key);
        let (scry_mean, scry_std) = kfold_accuracy(&x, &y, 5, |ds, test| {
            let mut dt = scry_learn::tree::DecisionTreeClassifier::new().max_depth(5);
            dt.fit(ds).unwrap();
            dt.predict(test).unwrap()
        });
        rows.push(ParityRow {
            dataset: dataset_label,
            model: "DecisionTree",
            scry_mean, scry_std, sklearn_mean: sk_mean, sklearn_std: sk_std,
        });

        // KNN
        let (sk_mean, sk_std) = load_sklearn_cv(knn_cv_key);
        let (scry_mean, scry_std) = kfold_accuracy(&x, &y, 5, |ds, test| {
            let mut knn = scry_learn::neighbors::KnnClassifier::new().k(5);
            knn.fit(ds).unwrap();
            knn.predict(test).unwrap()
        });
        rows.push(ParityRow {
            dataset: dataset_label,
            model: "KNN (k=5)",
            scry_mean, scry_std, sklearn_mean: sk_mean, sklearn_std: sk_std,
        });
    }

    // ── Original 5 datasets ──
    add_dt_knn_rows(&mut rows, "Iris",          "iris_features.csv",          "iris_target.csv",          "cv_dt_iris",       "cv_knn_iris");
    add_dt_knn_rows(&mut rows, "Wine",          "wine_features.csv",          "wine_target.csv",          "cv_dt_wine",       "cv_knn_wine");
    add_dt_knn_rows(&mut rows, "Breast Cancer", "breast_cancer_features.csv", "breast_cancer_target.csv", "cv_dt_bc",         "cv_knn_bc");

    // Digits: use DT with max_depth=15 matching sklearn fixture
    {
        let dig_x = load_fixture_features("digits_features.csv");
        let dig_y = load_fixture_target("digits_target.csv");

        let (sk_mean, sk_std) = load_sklearn_cv("cv_dt_digits");
        let (scry_mean, scry_std) = kfold_accuracy(&dig_x, &dig_y, 5, |ds, test| {
            let mut dt = scry_learn::tree::DecisionTreeClassifier::new().max_depth(15);
            dt.fit(ds).unwrap();
            dt.predict(test).unwrap()
        });
        rows.push(ParityRow {
            dataset: "Digits", model: "DecisionTree (d=15)",
            scry_mean, scry_std, sklearn_mean: sk_mean, sklearn_std: sk_std,
        });

        let (sk_mean, sk_std) = load_sklearn_cv("cv_knn_digits");
        let (scry_mean, scry_std) = kfold_accuracy(&dig_x, &dig_y, 5, |ds, test| {
            let mut knn = scry_learn::neighbors::KnnClassifier::new().k(5);
            knn.fit(ds).unwrap();
            knn.predict(test).unwrap()
        });
        rows.push(ParityRow {
            dataset: "Digits", model: "KNN (k=5)",
            scry_mean, scry_std, sklearn_mean: sk_mean, sklearn_std: sk_std,
        });
    }

    // ── 10 new OpenML datasets ──
    add_dt_knn_rows(&mut rows, "Adult Census",  "adult_features.csv",        "adult_target.csv",        "cv_dt_adult",       "cv_knn_adult");
    add_dt_knn_rows(&mut rows, "Spambase",      "spambase_features.csv",     "spambase_target.csv",     "cv_dt_spambase",    "cv_knn_spambase");
    add_dt_knn_rows(&mut rows, "Wine Quality",  "wine_quality_features.csv", "wine_quality_target.csv", "cv_dt_wine_quality", "cv_knn_wine_quality");
    add_dt_knn_rows(&mut rows, "Glass",         "glass_features.csv",        "glass_target.csv",        "cv_dt_glass",       "cv_knn_glass");
    add_dt_knn_rows(&mut rows, "Ionosphere",    "ionosphere_features.csv",   "ionosphere_target.csv",   "cv_dt_ionosphere",  "cv_knn_ionosphere");
    add_dt_knn_rows(&mut rows, "Vehicle",       "vehicle_features.csv",      "vehicle_target.csv",      "cv_dt_vehicle",     "cv_knn_vehicle");
    add_dt_knn_rows(&mut rows, "Segment",       "segment_features.csv",      "segment_target.csv",      "cv_dt_segment",     "cv_knn_segment");
    add_dt_knn_rows(&mut rows, "Sonar",         "sonar_features.csv",        "sonar_target.csv",        "cv_dt_sonar",       "cv_knn_sonar");
    add_dt_knn_rows(&mut rows, "Haberman",      "haberman_features.csv",     "haberman_target.csv",     "cv_dt_haberman",    "cv_knn_haberman");
    add_dt_knn_rows(&mut rows, "Ecoli",         "ecoli_features.csv",        "ecoli_target.csv",        "cv_dt_ecoli",       "cv_knn_ecoli");

    rows
}

// ═══════════════════════════════════════════════════════════════════════════
// HTML generation
// ═══════════════════════════════════════════════════════════════════════════


fn generate_html(
    timing_svg: &str,
    speedup_svg: &str,
    accuracy_svg: &str,
    results: &[BenchResult],
    parity_rows: &[ParityRow],
    inference_results: &[InferenceResult],
    scaling_svg: &str,
    memory_results: &[MemoryResult],
    pipeline_result: &PipelineBenchResult,
) -> String {
    // Build results table rows.
    let mut table_rows = String::new();
    for r in results {
        let acc_str = r
            .accuracy
            .map_or_else(|| "—".to_string(), |a| format!("{:.1}%", a * 100.0));
        let _ = writeln!(
            table_rows,
            "        <tr><td>{}</td><td>{}</td><td>{:.1} ± {:.1}</td><td>{}</td></tr>",
            r.algorithm, r.library, r.time_us, r.time_std, acc_str
        );
    }

    let accuracy_section = if accuracy_svg.is_empty() {
        String::new()
    } else {
        format!(
            r#"  <div class="chart-grid"><div class="chart-card">{accuracy_svg}</div></div>"#,
        )
    };

    // Build parity table rows.
    let mut parity_table_rows = String::new();
    for row in parity_rows {
        let delta = row.scry_mean - row.sklearn_mean;
        let delta_class = if delta.abs() < 0.02 {
            "check"
        } else if delta < 0.0 {
            "cross"
        } else {
            "check"
        };
        let delta_str = format!("{:+.1}%", delta * 100.0);
        let _ = writeln!(
            parity_table_rows,
            "        <tr><td>{}</td><td>{}</td><td>{:.1}% ± {:.1}%</td><td>{:.1}% ± {:.1}%</td><td class=\"{}\">{}</td></tr>",
            row.dataset, row.model,
            row.scry_mean * 100.0, row.scry_std * 100.0,
            row.sklearn_mean * 100.0, row.sklearn_std * 100.0,
            delta_class, delta_str
        );
    }

    // Build inference latency table rows
    let mut inference_rows = String::new();
    for r in inference_results {
        let p50_str = if r.p50_ns >= 1000 {
            format!("{:.1} µs", r.p50_ns as f64 / 1000.0)
        } else {
            format!("{} ns", r.p50_ns)
        };
        let p99_str = if r.p99_ns >= 1000 {
            format!("{:.1} µs", r.p99_ns as f64 / 1000.0)
        } else {
            format!("{} ns", r.p99_ns)
        };
        let _ = writeln!(
            inference_rows,
            "        <tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            r.model, p50_str, p99_str, r.iterations
        );
    }

    // Build memory table rows
    let mut memory_rows = String::new();
    if memory_results.is_empty() {
        let _ = writeln!(
            memory_rows,
            "        <tr><td colspan=\"2\" style=\"color: #888\">Run with <code>--features serde</code> to see model sizes</td></tr>"
        );
    } else {
        for r in memory_results {
            let _ = writeln!(
                memory_rows,
                "        <tr><td>{}</td><td>{}</td></tr>",
                r.model, format_bytes(r.serialized_bytes)
            );
        }
    }

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>scry-learn Benchmark Dashboard</title>
<style>
  :root {{
    --bg: #0f0f19;
    --surface: #1a1a2e;
    --border: #2a2a3e;
    --text: #c8c8dc;
    --accent: #63b3ed;
    --green: #86efac;
    --pink: #fc819b;
    --yellow: #fde68a;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 2rem;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{
    font-size: 2rem;
    background: linear-gradient(135deg, var(--accent), var(--pink));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
  }}
  .subtitle {{ color: #888; margin-bottom: 2rem; font-size: 0.95rem; }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 2rem;
  }}
  .card h2 {{
    font-size: 1.2rem;
    color: var(--accent);
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
  }}
  .chart-grid {{
    display: grid;
    grid-template-columns: 1fr;
    gap: 2rem;
    margin-bottom: 2rem;
  }}
  @media (min-width: 900px) {{
    .chart-grid {{ grid-template-columns: 1fr 1fr; }}
  }}
  .chart-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    overflow: hidden;
  }}
  .chart-card svg {{
    width: 100%;
    height: auto;
    display: block;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
  }}
  th, td {{
    padding: 0.6rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  th {{
    color: var(--accent);
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  tr:hover {{ background: rgba(99, 179, 237, 0.05); }}
  .feature-table td:nth-child(2),
  .feature-table td:nth-child(3),
  .feature-table td:nth-child(4) {{
    text-align: center;
  }}
  .check {{ color: var(--green); }}
  .cross {{ color: var(--pink); }}
  .timestamp {{
    text-align: center;
    color: #555;
    font-size: 0.8rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
  }}
  .highlight {{ color: var(--green); font-weight: 600; }}
  .metric-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 1rem;
  }}
  .metric {{
    background: rgba(99, 179, 237, 0.08);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
  }}
  .metric .value {{
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
  }}
  .metric .label {{
    font-size: 0.8rem;
    color: #888;
    margin-top: 0.25rem;
  }}
</style>
</head>
<body>
<div class="container">
  <h1>⚡ scry-learn Benchmark Dashboard</h1>
  <p class="subtitle">
    Head-to-head comparison: scry-learn vs smartcore 0.4.9 vs linfa 0.8
    · Pure Rust · No BLAS · Release mode · {warmup} warmup iters · ±σ reported
  </p>

  <div class="chart-grid">
    <div class="chart-card">
      {timing_svg}
    </div>
    <div class="chart-card">
      {speedup_svg}
    </div>
  </div>

  {accuracy_section}

  <div class="card">
    <h2>📊 Raw Benchmark Results</h2>
    <table>
      <thead>
        <tr><th>Algorithm</th><th>Library</th><th>Time (µs) ± σ</th><th>Accuracy</th></tr>
      </thead>
      <tbody>
{table_rows}      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>⏱️ Single-Sample Inference Latency</h2>
    <p style="color: #888; margin-bottom: 1rem; font-size: 0.9rem;">
      Predict one sample (10 features) — measured over {n_inf} iterations. p50 = median, p99 = 99th percentile.
    </p>
    <table>
      <thead>
        <tr><th>Model</th><th>p50</th><th>p99</th><th>Iterations</th></tr>
      </thead>
      <tbody>
{inference_rows}      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>📈 Training Time Scaling Curves</h2>
    <p style="color: #888; margin-bottom: 1rem; font-size: 0.9rem;">
      Training time (µs) vs number of samples (N=100, 1K, 10K) for each model.
    </p>
    <div class="chart-card" style="border: none; padding: 0;">
      {scaling_svg}
    </div>
  </div>

  <div class="card">
    <h2>💾 Model Memory Footprint</h2>
    <p style="color: #888; margin-bottom: 1rem; font-size: 0.9rem;">
      Serialized model size (JSON via serde_json) after training on 1K samples × 10 features.
    </p>
    <table>
      <thead>
        <tr><th>Model</th><th>Serialized Size</th></tr>
      </thead>
      <tbody>
{memory_rows}      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>🔗 End-to-End Pipeline Benchmark</h2>
    <p style="color: #888; margin-bottom: 1rem; font-size: 0.9rem;">
      StandardScaler → RandomForest (50 trees) on 2K×10. Measures fit + predict overhead of the Pipeline abstraction.
    </p>
    <div class="metric-grid">
      <div class="metric">
        <div class="value">{pipeline_us:.0} µs</div>
        <div class="label">Pipeline (fit+predict)</div>
      </div>
      <div class="metric">
        <div class="value">{raw_us:.0} µs</div>
        <div class="label">Raw RF (no scaler)</div>
      </div>
      <div class="metric">
        <div class="value" style="color: {overhead_color}">{overhead:+.1}%</div>
        <div class="label">Pipeline Overhead</div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>🏆 Feature Comparison</h2>
    <table>
      <thead>
        <tr><th>Feature</th><th>scry-learn</th><th>smartcore</th><th>linfa</th></tr>
      </thead>
      <tbody>
        <tr><td>Decision Tree (C+R)</td><td class="check">✅</td><td class="check">✅</td><td class="check">✅</td></tr>
        <tr><td>Random Forest (C+R)</td><td class="check">✅</td><td class="check">✅</td><td class="check">✅</td></tr>
        <tr><td>Gradient Boosting (C+R)</td><td class="check">✅</td><td class="cross">❌</td><td class="cross">❌</td></tr>
        <tr><td>Histogram GBT</td><td class="check">✅</td><td class="cross">❌</td><td class="cross">❌</td></tr>
        <tr><td>SVM (Linear + Kernel)</td><td class="check">✅</td><td class="check">✅</td><td class="cross">❌</td></tr>
        <tr><td>KNN (C+R, weighted)</td><td class="check">✅</td><td class="check">✅</td><td class="check">✅</td></tr>
        <tr><td>Naive Bayes (3 variants)</td><td class="check">✅</td><td class="check">✅</td><td class="cross">❌</td></tr>
        <tr><td>K-Means (n_init, mini-batch)</td><td class="check">✅</td><td class="cross">❌</td><td class="check">✅</td></tr>
        <tr><td>DBSCAN</td><td class="check">✅</td><td class="check">✅</td><td class="check">✅</td></tr>
        <tr><td>Pipeline (Transform → Fit)</td><td class="check">✅</td><td class="cross">❌</td><td class="check">✅</td></tr>
        <tr><td>GridSearchCV / RandomizedSearchCV</td><td class="check">✅</td><td class="cross">❌</td><td class="cross">❌</td></tr>
        <tr><td>Class Weights (balanced)</td><td class="check">✅</td><td class="cross">❌</td><td class="cross">❌</td></tr>
        <tr><td>Tree Pruning (ccp_alpha)</td><td class="check">✅</td><td class="cross">❌</td><td class="cross">❌</td></tr>
        <tr><td>Model Serialization (serde)</td><td class="check">✅</td><td class="check">✅</td><td class="cross">❌</td></tr>
        <tr><td>Built-in Visualization</td><td class="check">✅</td><td class="cross">❌</td><td class="cross">❌</td></tr>
        <tr><td>Pure Rust (no BLAS/LAPACK)</td><td class="check">✅</td><td class="check">✅</td><td class="cross">❌</td></tr>
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>🎯 sklearn Parity — 5-Fold CV on {n_parity} UCI Datasets</h2>
    <p style="color: #888; margin-bottom: 1rem; font-size: 0.9rem;">
      Compares scry-learn vs scikit-learn 1.8 on identical datasets and hyperparameters.
      Accuracy = mean ± σ across 5 stratified folds.
    </p>
    <table>
      <thead>
        <tr><th>Dataset</th><th>Model</th><th>scry-learn</th><th>sklearn</th><th>Δ</th></tr>
      </thead>
      <tbody>
{parity_table_rows}      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>📝 Methodology</h2>
    <p>All benchmarks run in <strong>release mode</strong> with identical data generation seeds.
    Timing uses <code>std::time::Instant</code> with {warmup} warmup iterations discarded,
    then averaged over multiple timed iterations with ±σ (standard deviation) reported.
    Datasets: binary classification (1–2K samples, 10 features), regression (2K×10).
    Inference latency: single-sample predict over 10K iterations (p50/p99).
    Charts generated by <span class="highlight">scry-chart</span> (dogfooding!).</p>
  </div>

  <p class="timestamp">Generated by bench_dashboard · scry-learn v0.1.0</p>
</div>
</body>
</html>
"#,
        warmup = WARMUP_ITERS,
        parity_table_rows = parity_table_rows,
        n_inf = if inference_results.is_empty() { 0 } else { inference_results[0].iterations },
        n_parity = parity_rows.len(),
        pipeline_us = pipeline_result.pipeline_us,
        raw_us = pipeline_result.raw_us,
        overhead = pipeline_result.overhead_pct,
        overhead_color = if pipeline_result.overhead_pct < 10.0 { "var(--green)" } else { "var(--yellow)" },
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    println!("⚡ scry-learn Benchmark Dashboard Generator");
    println!("============================================\n");

    println!("🔄 Running Decision Tree benchmarks...");
    let mut all_results = bench_decision_tree();

    println!("🔄 Running Random Forest benchmarks...");
    all_results.extend(bench_random_forest());

    println!("🔄 Running GBT Regressor benchmarks (scry-learn only)...");
    all_results.extend(bench_gbt_regressor());

    println!("🔄 Running HistGBT Regressor benchmarks (scry-learn only)...");
    all_results.extend(bench_hist_gbt());

    println!("🔄 Running Logistic Regression benchmarks...");
    all_results.extend(bench_logistic_regression());

    println!("🔄 Running KNN benchmarks...");
    all_results.extend(bench_knn());

    println!("🔄 Running K-Means benchmarks...");
    all_results.extend(bench_kmeans());

    println!("🔄 Running PCA benchmarks...");
    all_results.extend(bench_pca());

    println!("🔄 Running SVM benchmarks (scry LinearSVC vs smartcore SVC)...");
    all_results.extend(bench_svm());

    println!("🔄 Running Lasso benchmarks (scry Lasso vs linfa-elasticnet)...");
    all_results.extend(bench_lasso());

    // NEW: Inference latency
    println!("\n⏱️  Measuring single-sample inference latency (10K iters)...");
    let inference = bench_inference_latency();
    println!("    {:>14} {:>10} {:>10}", "Model", "p50", "p99");
    for r in &inference {
        println!("    {:>14} {:>10} {:>10}", r.model,
            format!("{} ns", r.p50_ns), format!("{} ns", r.p99_ns));
    }

    // NEW: Scaling curves
    println!("\n📈 Measuring training time scaling (N=100, 1K, 10K)...");
    let (scaling_points, scaling_svg) = bench_scaling();
    for p in &scaling_points {
        println!("    {:>8} N={:>5}: {:.0} µs", p.model, p.n_samples, p.train_us);
    }

    // NEW: Memory footprint
    println!("\n💾 Measuring serialized model sizes...");
    let memory = bench_model_size();
    if memory.is_empty() {
        println!("    (serde feature not enabled — skipping)");
    } else {
        for r in &memory {
            println!("    {:>14}: {}", r.model, format_bytes(r.serialized_bytes));
        }
    }

    // NEW: Pipeline benchmark
    println!("\n🔗 Running pipeline benchmark (StandardScaler → RF 50 trees)...");
    let pipeline = bench_pipeline();
    println!("    Pipeline: {:.0} ± {:.0} µs", pipeline.pipeline_us, pipeline.pipeline_std);
    println!("    Raw RF:   {:.0} ± {:.0} µs", pipeline.raw_us, pipeline.raw_std);
    println!("    Overhead: {:+.1}%", pipeline.overhead_pct);

    // Print summary table
    println!("\n{}", "─".repeat(75));
    println!("{:<18} {:<14} {:>12}  {:>10}  {:>8}", "Algorithm", "Library", "Time (µs)", "± σ", "Accuracy");
    println!("{}", "─".repeat(75));
    for r in &all_results {
        let acc = r
            .accuracy
            .map_or_else(|| "—".to_string(), |a| format!("{:.1}%", a * 100.0));
        println!("{:<18} {:<14} {:>12.1}  {:>10.1}  {:>8}", r.algorithm, r.library, r.time_us, r.time_std, acc);
    }
    println!("{}", "─".repeat(75));

    // Generate SVG charts using scry-chart
    println!("\n📊 Generating SVG charts with scry-chart...");
    let timing_svg = make_timing_chart(&all_results, "Training & Prediction Latency", "Time (µs)");
    let speedup_svg = make_speedup_chart(&all_results);
    let accuracy_svg = make_accuracy_chart(&all_results);

    // Compute parity table (5-fold CV on real UCI datasets)
    println!("\n🎯 Computing sklearn parity table (5-fold CV on 15 UCI datasets)...");
    let parity = compute_parity_table();
    println!("\n{}", "─".repeat(90));
    println!("{:<15} {:<20} {:>18} {:>18} {:>8}", "Dataset", "Model", "scry-learn", "sklearn", "Δ");
    println!("{}", "─".repeat(90));
    for row in &parity {
        let delta = (row.scry_mean - row.sklearn_mean) * 100.0;
        println!(
            "{:<15} {:<20} {:>7.1}% ± {:<7.1}% {:>7.1}% ± {:<7.1}% {:>+6.1}%",
            row.dataset, row.model,
            row.scry_mean * 100.0, row.scry_std * 100.0,
            row.sklearn_mean * 100.0, row.sklearn_std * 100.0,
            delta
        );
    }
    println!("{}", "─".repeat(90));

    // Generate HTML
    println!("\n📄 Generating HTML dashboard...");
    let html = generate_html(
        &timing_svg, &speedup_svg, &accuracy_svg,
        &all_results, &parity,
        &inference, &scaling_svg, &memory, &pipeline,
    );

    let output_path = "bench_dashboard.html";
    std::fs::write(output_path, &html).unwrap();
    println!("\n✅ Dashboard written to: {output_path}");
    println!("   Open in a browser to view the benchmark comparison charts.");
    println!("   File size: {:.1} KB", html.len() as f64 / 1024.0);
}

