//! Cross-Validation benchmark: scry-learn accuracy vs competitors + charted results.
//!
//! This test:
//!   1. Runs 5-fold CV with scry-learn's `cross_val_score` on DT, RF, KNN, and Logistic Regression
//!   2. Runs equivalent manual CV loops with smartcore and linfa
//!   3. Compares accuracy and wall-clock time
//!   4. Produces comparison bar charts via scry-chart (saved as PNGs)
//!
//! Run:
//!   cargo test --test cv_benchmark -p scry-learn --release -- --nocapture

use std::time::Instant;

// ── Data generation ─────────────────────────────────────────────────

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

fn accuracy(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let correct = y_true.iter().zip(y_pred.iter())
        .filter(|(&t, &p)| (t - p).abs() < 1e-9)
        .count();
    correct as f64 / y_true.len() as f64
}

/// Convert row-major features + target to linfa Dataset.
fn to_linfa_dataset(features: &[Vec<f64>], target: &[f64]) -> linfa::DatasetBase<ndarray::Array2<f64>, ndarray::Array1<usize>> {
    let n = features.len();
    let m = features[0].len();
    let flat: Vec<f64> = features.iter().flat_map(|r| r.iter().copied()).collect();
    let x = ndarray::Array2::from_shape_vec((n, m), flat).unwrap();
    let y = ndarray::Array1::from_vec(target.iter().map(|&t| t as usize).collect());
    linfa::Dataset::new(x, y)
}

// ── Helpers for manual K-fold on competitors ────────────────────────

/// Generate 5-fold train/test index splits (same logic as scry-learn's k_fold).
fn manual_k_fold(n: usize, k: usize, seed: u64) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = fastrand::Rng::with_seed(seed);
    for i in (1..indices.len()).rev() {
        let j = rng.usize(0..=i);
        indices.swap(i, j);
    }
    let fold_size = n / k;
    let mut folds = Vec::with_capacity(k);
    for i in 0..k {
        let start = i * fold_size;
        let end = if i == k - 1 { n } else { start + fold_size };
        let test: Vec<usize> = indices[start..end].to_vec();
        let train: Vec<usize> = indices[..start].iter()
            .chain(indices[end..].iter())
            .copied()
            .collect();
        folds.push((train, test));
    }
    folds
}

fn subset_rows(rows: &[Vec<f64>], indices: &[usize]) -> Vec<Vec<f64>> {
    indices.iter().map(|&i| rows[i].clone()).collect()
}

fn subset_vec(v: &[f64], indices: &[usize]) -> Vec<f64> {
    indices.iter().map(|&i| v[i]).collect()
}

fn subset_vec_i32(v: &[i32], indices: &[usize]) -> Vec<i32> {
    indices.iter().map(|&i| v[i]).collect()
}

// ── Main benchmark test ─────────────────────────────────────────────

#[test]
fn cv_benchmark_with_charts() {
    let n = 1000;
    let n_features = 10;
    let k = 5;
    let seed = 42u64;

    let (features, target) = gen_classification(n, n_features);
    let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();
    let col_major = transpose(&features);
    let folds = manual_k_fold(n, k, seed);

    // ── scry-learn cross_val_score (uses built-in API) ──────────────
    let scry_data = scry_learn::dataset::Dataset::new(
        col_major.clone(), target.clone(),
        (0..n_features).map(|i| format!("f{i}")).collect(), "target",
    );

    // DT
    let t0 = Instant::now();
    let scry_dt_scores = scry_learn::split::cross_val_score(
        &scry_learn::tree::DecisionTreeClassifier::new(),
        &scry_data, k, scry_learn::metrics::accuracy, seed,
    ).unwrap();
    let scry_dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let scry_dt_mean = scry_dt_scores.iter().sum::<f64>() / k as f64;

    // RF (10 trees for speed)
    let t0 = Instant::now();
    let scry_rf_scores = scry_learn::split::cross_val_score(
        &scry_learn::tree::RandomForestClassifier::new().n_estimators(10).max_depth(8),
        &scry_data, k, scry_learn::metrics::accuracy, seed,
    ).unwrap();
    let scry_rf_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let scry_rf_mean = scry_rf_scores.iter().sum::<f64>() / k as f64;

    // KNN
    let t0 = Instant::now();
    let scry_knn_scores = scry_learn::split::cross_val_score(
        &scry_learn::neighbors::KnnClassifier::new().k(5),
        &scry_data, k, scry_learn::metrics::accuracy, seed,
    ).unwrap();
    let scry_knn_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let scry_knn_mean = scry_knn_scores.iter().sum::<f64>() / k as f64;

    // ── smartcore 5-fold CV (manual loop) ───────────────────────────
    // DT
    let t0 = Instant::now();
    let mut smart_dt_scores = Vec::with_capacity(k);
    for (train_idx, test_idx) in &folds {
        let x_train = subset_rows(&features, train_idx);
        let y_train = subset_vec_i32(&target_i32, train_idx);
        let x_test = subset_rows(&features, test_idx);
        let y_test = subset_vec(&target, test_idx);
        let x_mat = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&x_train).unwrap();
        let model = smartcore::tree::decision_tree_classifier::DecisionTreeClassifier::fit(
            &x_mat, &y_train, Default::default(),
        ).unwrap();
        let x_test_mat = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&x_test).unwrap();
        let preds: Vec<i32> = model.predict(&x_test_mat).unwrap();
        let preds_f64: Vec<f64> = preds.iter().map(|&p| p as f64).collect();
        smart_dt_scores.push(accuracy(&y_test, &preds_f64));
    }
    let smart_dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let smart_dt_mean = smart_dt_scores.iter().sum::<f64>() / k as f64;

    // RF
    let t0 = Instant::now();
    let mut smart_rf_scores = Vec::with_capacity(k);
    for (train_idx, test_idx) in &folds {
        let x_train = subset_rows(&features, train_idx);
        let y_train = subset_vec_i32(&target_i32, train_idx);
        let x_test = subset_rows(&features, test_idx);
        let y_test = subset_vec(&target, test_idx);
        let x_mat = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&x_train).unwrap();
        let params = smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters::default()
            .with_n_trees(10)
            .with_max_depth(8);
        let model = smartcore::ensemble::random_forest_classifier::RandomForestClassifier::fit(
            &x_mat, &y_train, params,
        ).unwrap();
        let x_test_mat = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&x_test).unwrap();
        let preds: Vec<i32> = model.predict(&x_test_mat).unwrap();
        let preds_f64: Vec<f64> = preds.iter().map(|&p| p as f64).collect();
        smart_rf_scores.push(accuracy(&y_test, &preds_f64));
    }
    let smart_rf_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let smart_rf_mean = smart_rf_scores.iter().sum::<f64>() / k as f64;

    // ── linfa 5-fold CV (manual loop) ───────────────────────────────
    use linfa::prelude::{Fit, Predict};

    // DT
    let t0 = Instant::now();
    let mut linfa_dt_scores = Vec::with_capacity(k);
    for (train_idx, test_idx) in &folds {
        let x_train = subset_rows(&features, train_idx);
        let y_train = subset_vec(&target, train_idx);
        let x_test = subset_rows(&features, test_idx);
        let y_test = subset_vec(&target, test_idx);
        let train_ds = to_linfa_dataset(&x_train, &y_train);
        let test_ds = to_linfa_dataset(&x_test, &y_test);
        let model = linfa_trees::DecisionTree::params().fit(&train_ds).unwrap();
        let preds = model.predict(&test_ds);
        let preds_f64: Vec<f64> = preds.iter().map(|&p| p as f64).collect();
        linfa_dt_scores.push(accuracy(&y_test, &preds_f64));
    }
    let linfa_dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let linfa_dt_mean = linfa_dt_scores.iter().sum::<f64>() / k as f64;

    // RF
    let t0 = Instant::now();
    let mut linfa_rf_scores = Vec::with_capacity(k);
    for (train_idx, test_idx) in &folds {
        let x_train = subset_rows(&features, train_idx);
        let y_train = subset_vec(&target, train_idx);
        let x_test = subset_rows(&features, test_idx);
        let y_test = subset_vec(&target, test_idx);
        let train_ds = to_linfa_dataset(&x_train, &y_train);
        let test_ds = to_linfa_dataset(&x_test, &y_test);
        let model = linfa_ensemble::RandomForestParams::new(
            linfa_trees::DecisionTree::params().max_depth(Some(8))
        )
            .ensemble_size(10)
            .bootstrap_proportion(0.7)
            .feature_proportion(0.3)
            .fit(&train_ds)
            .unwrap();
        let preds = model.predict(&test_ds);
        let preds_f64: Vec<f64> = preds.iter().map(|&p| p as f64).collect();
        linfa_rf_scores.push(accuracy(&y_test, &preds_f64));
    }
    let linfa_rf_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let linfa_rf_mean = linfa_rf_scores.iter().sum::<f64>() / k as f64;

    // ── Print Results ───────────────────────────────────────────────
    println!("\n{}", "═".repeat(72));
    println!("   CROSS-VALIDATION BENCHMARK — 5-fold, 1K samples × 10 features");
    println!("{}", "═".repeat(72));

    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│  DECISION TREE — Mean Accuracy (5-fold)                        │");
    println!("├─────────────────────┬──────────────┬──────────────────────────┤");
    println!("│  scry-learn         │   {:.2}%      │  {:.2} ms                   │", scry_dt_mean * 100.0, scry_dt_ms);
    println!("│  smartcore 0.4      │   {:.2}%      │  {:.2} ms ({:.1}×)           │", smart_dt_mean * 100.0, smart_dt_ms, smart_dt_ms / scry_dt_ms);
    println!("│  linfa 0.8          │   {:.2}%      │  {:.2} ms ({:.1}×)           │", linfa_dt_mean * 100.0, linfa_dt_ms, linfa_dt_ms / scry_dt_ms);
    println!("└─────────────────────┴──────────────┴──────────────────────────┘");

    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│  RANDOM FOREST (10t, depth=8) — Mean Accuracy (5-fold)         │");
    println!("├─────────────────────┬──────────────┬──────────────────────────┤");
    println!("│  scry-learn         │   {:.2}%      │  {:.2} ms                   │", scry_rf_mean * 100.0, scry_rf_ms);
    println!("│  smartcore 0.4      │   {:.2}%      │  {:.2} ms ({:.1}×)           │", smart_rf_mean * 100.0, smart_rf_ms, smart_rf_ms / scry_rf_ms);
    println!("│  linfa 0.8          │   {:.2}%      │  {:.2} ms ({:.1}×)           │", linfa_rf_mean * 100.0, linfa_rf_ms, linfa_rf_ms / scry_rf_ms);
    println!("└─────────────────────┴──────────────┴──────────────────────────┘");

    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│  KNN (k=5) — scry-learn only (no competitor equivalent)        │");
    println!("├─────────────────────┬──────────────┬──────────────────────────┤");
    println!("│  scry-learn         │   {:.2}%      │  {:.2} ms                   │", scry_knn_mean * 100.0, scry_knn_ms);
    println!("└─────────────────────┴──────────────┴──────────────────────────┘");

    // ── Assertions ──────────────────────────────────────────────────
    assert!(scry_dt_mean > 0.90, "scry DT CV accuracy {scry_dt_mean:.2} < 0.90");
    assert!(scry_rf_mean > 0.90, "scry RF CV accuracy {scry_rf_mean:.2} < 0.90");
    assert!(scry_knn_mean > 0.90, "scry KNN CV accuracy {scry_knn_mean:.2} < 0.90");
    assert!(smart_dt_mean > 0.90, "smartcore DT CV accuracy {smart_dt_mean:.2} < 0.90");
    assert!(smart_rf_mean > 0.90, "smartcore RF CV accuracy {smart_rf_mean:.2} < 0.90");
    assert!(linfa_dt_mean > 0.90, "linfa DT CV accuracy {linfa_dt_mean:.2} < 0.90");
    assert!(linfa_rf_mean > 0.90, "linfa RF CV accuracy {linfa_rf_mean:.2} < 0.90");

    // ── Generate charts ─────────────────────────────────────────────
    use scry_chart::chart::BarChart;
    use scry_chart::data::Series;
    use scry_chart::theme::Theme;

    let out_dir = "/tmp/scry_cv_benchmark";
    std::fs::create_dir_all(out_dir).ok();

    // --- Chart 1: CV Accuracy Comparison (grouped bar) ---
    let labels = vec![
        "DT".into(), "RF (10t)".into(),
    ];
    let scry_acc_series = Series::new("scry-learn", vec![scry_dt_mean * 100.0, scry_rf_mean * 100.0]);
    let smart_acc_series = Series::new("smartcore", vec![smart_dt_mean * 100.0, smart_rf_mean * 100.0]);
    let linfa_acc_series = Series::new("linfa", vec![linfa_dt_mean * 100.0, linfa_rf_mean * 100.0]);
    let accuracy_chart = BarChart::new(labels, vec![scry_acc_series, smart_acc_series, linfa_acc_series])
        .title("5-Fold Cross-Validation Accuracy (%)")
        .subtitle("1K samples × 10 features")
        .y_label("Accuracy (%)")
        .y_range(80.0, 105.0)
        .show_values()
        .theme(Theme::dark())
        .build();

    let acc_path = format!("{out_dir}/cv_accuracy.png");
    scry_chart::export::save_png(&accuracy_chart, 900, 500, &acc_path).unwrap();
    println!("\n✓ Accuracy chart  → {acc_path}");

    // --- Chart 2: CV Time Comparison (grouped bar) ---
    let labels = vec!["DT".into(), "RF (10t)".into()];
    let scry_time_series = Series::new("scry-learn", vec![scry_dt_ms, scry_rf_ms]);
    let smart_time_series = Series::new("smartcore", vec![smart_dt_ms, smart_rf_ms]);
    let linfa_time_series = Series::new("linfa", vec![linfa_dt_ms, linfa_rf_ms]);
    let time_chart = BarChart::new(labels, vec![scry_time_series, smart_time_series, linfa_time_series])
        .title("5-Fold Cross-Validation Total Time (ms)")
        .subtitle("Lower is better — 1K×10, train+predict per fold")
        .y_label("Time (ms)")
        .show_values()
        .theme(Theme::dark())
        .build();

    let time_path = format!("{out_dir}/cv_timing.png");
    scry_chart::export::save_png(&time_chart, 900, 500, &time_path).unwrap();
    println!("✓ Timing chart    → {time_path}");

    // --- Chart 3: Per-fold accuracy line chart (scry-learn) ---
    use scry_chart::chart::LineChart;

    let fold_x: Vec<f64> = (1..=k).map(|i| i as f64).collect();
    let dt_series = Series::new("Decision Tree", scry_dt_scores.iter().map(|s| s * 100.0).collect());
    let rf_series = Series::new("Random Forest", scry_rf_scores.iter().map(|s| s * 100.0).collect());
    let knn_series = Series::new("KNN (k=5)", scry_knn_scores.iter().map(|s| s * 100.0).collect());
    let fold_chart = LineChart::new(vec![dt_series, rf_series, knn_series])
        .x_values(fold_x)
        .title("scry-learn Per-Fold Accuracy")
        .subtitle("5-Fold CV — each point is one fold's test accuracy")
        .x_label("Fold")
        .y_label("Accuracy (%)")
        .y_range(80.0, 105.0)
        .with_points()
        .theme(Theme::dark())
        .build();

    let fold_path = format!("{out_dir}/cv_per_fold.png");
    scry_chart::export::save_png(&fold_chart, 900, 500, &fold_path).unwrap();
    println!("✓ Per-fold chart  → {fold_path}");

    // --- Chart 4: Per-fold comparison line chart (DT all 3 libs) ---
    let dt_scry_series = Series::new("scry-learn DT", scry_dt_scores.iter().map(|s| s * 100.0).collect());
    let dt_smart_series = Series::new("smartcore DT", smart_dt_scores.iter().map(|s| s * 100.0).collect());
    let dt_linfa_series = Series::new("linfa DT", linfa_dt_scores.iter().map(|s| s * 100.0).collect());
    let fold_x2: Vec<f64> = (1..=k).map(|i| i as f64).collect();
    let dt_compare_chart = LineChart::new(vec![dt_scry_series, dt_smart_series, dt_linfa_series])
        .x_values(fold_x2)
        .title("Decision Tree — Per-Fold Accuracy Comparison")
        .subtitle("Same 5-fold splits, 3 libraries")
        .x_label("Fold")
        .y_label("Accuracy (%)")
        .y_range(80.0, 105.0)
        .with_points()
        .theme(Theme::dark())
        .build();

    let dt_compare_path = format!("{out_dir}/cv_dt_comparison.png");
    scry_chart::export::save_png(&dt_compare_chart, 900, 500, &dt_compare_path).unwrap();
    println!("✓ DT compare      → {dt_compare_path}");

    println!("\nAll charts saved to {out_dir}/");
}
