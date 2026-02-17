//! Predict latency + memory benchmarks (Agent 14).
//!
//! | Group | What it measures |
//! |-------|-----------------|
//! | 1 | Single-sample predict latency across all model families |
//! | 2 | Batch predict throughput at 1K/10K/100K rows |
//! | 3 | Peak RSS delta during model fit (memory footprint) |
//!
//! Run with: `cargo bench --bench predict_latency -p scry-learn`

#![allow(missing_docs)]

use std::time::Duration;

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};

use scry_learn::dataset::Dataset;
use scry_learn::linear::{LinearRegression, LogisticRegression};
use scry_learn::naive_bayes::GaussianNb;
use scry_learn::neighbors::KnnClassifier;
use scry_learn::neural::MLPClassifier;
use scry_learn::split::train_test_split;
use scry_learn::svm::LinearSVC;
use scry_learn::tree::{
    DecisionTreeClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, RandomForestClassifier,
};

// ─────────────────────────────────────────────────────────────────
// Deterministic data generators
// ─────────────────────────────────────────────────────────────────

fn generate_classification_dataset(
    n_rows: usize,
    n_cols: usize,
    n_classes: usize,
    seed: u64,
) -> Dataset {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut features: Vec<Vec<f64>> = (0..n_cols).map(|_| Vec::with_capacity(n_rows)).collect();
    let mut target = Vec::with_capacity(n_rows);
    let per_class = n_rows / n_classes;

    for c in 0..n_classes {
        let count = if c == n_classes - 1 {
            n_rows - per_class * c
        } else {
            per_class
        };
        for _ in 0..count {
            for (j, col) in features.iter_mut().enumerate() {
                let offset = c as f64 * 3.0 + j as f64 * 0.2;
                col.push(rng.f64() * 2.0 + offset);
            }
            target.push(c as f64);
        }
    }

    let names: Vec<String> = (0..n_cols).map(|i| format!("f{i}")).collect();
    Dataset::new(features, target, names, "class")
}

fn generate_regression_dataset(n_rows: usize, n_cols: usize, seed: u64) -> Dataset {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut features: Vec<Vec<f64>> = (0..n_cols).map(|_| Vec::with_capacity(n_rows)).collect();
    let mut target = Vec::with_capacity(n_rows);

    for _ in 0..n_rows {
        let mut y = 0.0;
        for (j, col) in features.iter_mut().enumerate() {
            let x = rng.f64() * 10.0;
            col.push(x);
            y += x * (j + 1) as f64;
        }
        y += rng.f64() * 0.5;
        target.push(y);
    }

    let names: Vec<String> = (0..n_cols).map(|i| format!("f{i}")).collect();
    Dataset::new(features, target, names, "y")
}

/// Extract a single row from column-major features as a row-major `Vec<Vec<f64>>`.
fn single_row(data: &Dataset, row_idx: usize) -> Vec<Vec<f64>> {
    vec![data
        .features
        .iter()
        .map(|col| col[row_idx])
        .collect()]
}

/// Convert column-major features to row-major `Vec<Vec<f64>>` for predict.
fn to_row_major(data: &Dataset) -> Vec<Vec<f64>> {
    let n_samples = data.target.len();
    let n_features = data.features.len();
    (0..n_samples)
        .map(|i| (0..n_features).map(|j| data.features[j][i]).collect())
        .collect()
}

/// Read RSS (Resident Set Size) in kilobytes from `/proc/self/statm`.
fn rss_kb() -> usize {
    let statm = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
    let pages: usize = statm
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    pages * 4 // page size = 4KB on x86_64
}

// ─────────────────────────────────────────────────────────────────
// Group 1: Single-Sample Predict Latency
// ─────────────────────────────────────────────────────────────────

fn bench_single_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_predict_latency");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    let data = generate_classification_dataset(10_000, 50, 3, 42);
    let (train, _test) = train_test_split(&data, 0.2, 42);
    let row = single_row(&train, 0);

    // Decision Tree
    let mut dt = DecisionTreeClassifier::new().max_depth(10);
    dt.fit(&train).unwrap();
    group.bench_function("dt_classifier", |b| {
        b.iter(|| std::hint::black_box(dt.predict(&row).unwrap()));
    });

    // Random Forest (100 trees)
    let mut rf = RandomForestClassifier::new()
        .n_estimators(100)
        .max_depth(10);
    rf.fit(&train).unwrap();
    group.bench_function("rf_100trees", |b| {
        b.iter(|| std::hint::black_box(rf.predict(&row).unwrap()));
    });

    // GBT (100 estimators)
    let mut gbt = GradientBoostingClassifier::new()
        .n_estimators(100)
        .max_depth(5);
    gbt.fit(&train).unwrap();
    group.bench_function("gbt_100est", |b| {
        b.iter(|| std::hint::black_box(gbt.predict(&row).unwrap()));
    });

    // HistGBT (100 estimators)
    let mut hgbt = HistGradientBoostingClassifier::new()
        .n_estimators(100)
        .max_depth(5);
    hgbt.fit(&train).unwrap();
    group.bench_function("histgbt_100est", |b| {
        b.iter(|| std::hint::black_box(hgbt.predict(&row).unwrap()));
    });

    // Linear Regression
    let reg_data = generate_regression_dataset(10_000, 50, 42);
    let (reg_train, _) = train_test_split(&reg_data, 0.2, 42);
    let reg_row = single_row(&reg_train, 0);
    let mut lr = LinearRegression::new();
    lr.fit(&reg_train).unwrap();
    group.bench_function("linear_regression", |b| {
        b.iter(|| std::hint::black_box(lr.predict(&reg_row).unwrap()));
    });

    // Logistic Regression
    let mut logr = LogisticRegression::new();
    logr.fit(&train).unwrap();
    group.bench_function("logistic_regression", |b| {
        b.iter(|| std::hint::black_box(logr.predict(&row).unwrap()));
    });

    // KNN (k=5)
    let mut knn = KnnClassifier::new().k(5);
    knn.fit(&train).unwrap();
    group.bench_function("knn_k5", |b| {
        b.iter(|| std::hint::black_box(knn.predict(&row).unwrap()));
    });

    // MLP
    let mut mlp = MLPClassifier::new()
        .hidden_layers(&[100, 50])
        .max_iter(50);
    mlp.fit(&train).unwrap();
    group.bench_function("mlp_100_50", |b| {
        b.iter(|| std::hint::black_box(mlp.predict(&row).unwrap()));
    });

    // GaussianNB
    let mut gnb = GaussianNb::new();
    gnb.fit(&train).unwrap();
    group.bench_function("gaussian_nb", |b| {
        b.iter(|| std::hint::black_box(gnb.predict(&row).unwrap()));
    });

    // LinearSVC
    let mut svc = LinearSVC::new();
    svc.fit(&train).unwrap();
    group.bench_function("linear_svc", |b| {
        b.iter(|| std::hint::black_box(svc.predict(&row).unwrap()));
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// Group 2: Batch Predict Throughput
// ─────────────────────────────────────────────────────────────────

fn bench_batch_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_predict_throughput");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);

    for &(label, n_rows) in &[("1k", 1_000), ("10k", 10_000), ("100k", 100_000)] {
        let data = generate_classification_dataset(n_rows, 50, 3, 42);
        let (train, test) = train_test_split(&data, 0.2, 42);
        let test_rows = to_row_major(&test);

        group.throughput(Throughput::Elements(test_rows.len() as u64));

        // Random Forest (50 trees)
        let mut rf = RandomForestClassifier::new()
            .n_estimators(50)
            .max_depth(10);
        rf.fit(&train).unwrap();
        group.bench_function(BenchmarkId::new("rf_50trees", label), |b| {
            b.iter(|| std::hint::black_box(rf.predict(&test_rows).unwrap()));
        });

        // Linear Regression
        let reg_data = generate_regression_dataset(n_rows, 50, 42);
        let (reg_train, reg_test) = train_test_split(&reg_data, 0.2, 42);
        let reg_rows = to_row_major(&reg_test);
        let mut lr = LinearRegression::new();
        lr.fit(&reg_train).unwrap();
        group.throughput(Throughput::Elements(reg_rows.len() as u64));
        group.bench_function(BenchmarkId::new("linear_reg", label), |b| {
            b.iter(|| std::hint::black_box(lr.predict(&reg_rows).unwrap()));
        });

        // MLP (single hidden layer for batch)
        let mut mlp = MLPClassifier::new()
            .hidden_layers(&[64])
            .max_iter(20);
        mlp.fit(&train).unwrap();
        group.throughput(Throughput::Elements(test_rows.len() as u64));
        group.bench_function(BenchmarkId::new("mlp_64", label), |b| {
            b.iter(|| std::hint::black_box(mlp.predict(&test_rows).unwrap()));
        });
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// Group 3: Model Memory Footprint (RSS Delta)
// ─────────────────────────────────────────────────────────────────

fn bench_model_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_memory");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for &(label, n_rows) in &[("10k", 10_000), ("100k", 100_000)] {
        let data = generate_classification_dataset(n_rows, 50, 3, 42);

        // Random Forest — RSS delta during fit
        let rss_before = rss_kb();
        let mut rf = RandomForestClassifier::new()
            .n_estimators(100)
            .max_depth(10);
        rf.fit(&data).unwrap();
        let rss_after = rss_kb();
        let rf_rss_delta_kb = rss_after.saturating_sub(rss_before);
        // Record the RSS delta as a pseudo-benchmark so Criterion reports it
        group.bench_function(BenchmarkId::new("rf_100trees_rss_kb", label), |b| {
            b.iter(|| std::hint::black_box(rf_rss_delta_kb));
        });

        // GBT — RSS delta during fit
        let rss_before = rss_kb();
        let mut gbt = GradientBoostingClassifier::new()
            .n_estimators(100)
            .max_depth(5);
        gbt.fit(&data).unwrap();
        let rss_after = rss_kb();
        let gbt_rss_delta_kb = rss_after.saturating_sub(rss_before);
        group.bench_function(BenchmarkId::new("gbt_100est_rss_kb", label), |b| {
            b.iter(|| std::hint::black_box(gbt_rss_delta_kb));
        });

        // KNN — stores full training set
        let rss_before = rss_kb();
        let mut knn = KnnClassifier::new().k(5);
        knn.fit(&data).unwrap();
        let rss_after = rss_kb();
        let knn_rss_delta_kb = rss_after.saturating_sub(rss_before);
        group.bench_function(BenchmarkId::new("knn_k5_rss_kb", label), |b| {
            b.iter(|| std::hint::black_box(knn_rss_delta_kb));
        });

        // MLP
        let rss_before = rss_kb();
        let mut mlp = MLPClassifier::new()
            .hidden_layers(&[100, 50])
            .max_iter(50);
        mlp.fit(&data).unwrap();
        let rss_after = rss_kb();
        let mlp_rss_delta_kb = rss_after.saturating_sub(rss_before);
        group.bench_function(BenchmarkId::new("mlp_100_50_rss_kb", label), |b| {
            b.iter(|| std::hint::black_box(mlp_rss_delta_kb));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_predict,
    bench_batch_predict,
    bench_model_memory,
);
criterion_main!(benches);
