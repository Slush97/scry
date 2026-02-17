//! Fuzz target: Histogram-based Gradient Boosting classifier + regressor.
//!
//! Exercises HistGradientBoostingClassifier and HistGradientBoostingRegressor
//! with small ensembles on fuzz-derived datasets.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_learn::dataset::Dataset;
use scry_learn::tree::{HistGradientBoostingClassifier, HistGradientBoostingRegressor};

fuzz_target!(|data: &[u8]| {
    if data.len() < 10 {
        return;
    }

    let mut cursor = 0;

    let n_features = (data[cursor] % 4).max(1) as usize;
    cursor += 1;
    let n_samples = (data[cursor] % 13).max(4) as usize;
    cursor += 1;
    let dispatch = data[cursor] % 2;
    cursor += 1;

    // Build column-major feature matrix.
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(n_features);
    for _ in 0..n_features {
        let mut col = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            if cursor < data.len() {
                let v = (data[cursor] as f64 / 128.0) - 1.0;
                col.push(v);
                cursor += 1;
            } else {
                col.push(0.0);
            }
        }
        features.push(col);
    }

    let is_classifier = dispatch == 0;
    let target: Vec<f64> = (0..n_samples)
        .map(|i| {
            if is_classifier {
                (i % 2) as f64
            } else if cursor < data.len() {
                let c = cursor;
                cursor += 1;
                (data[c] as f64 / 128.0) - 1.0
            } else {
                (i % 3) as f64
            }
        })
        .collect();

    let feature_names: Vec<String> = (0..n_features).map(|i| format!("f{i}")).collect();
    let dataset = Dataset::new(features, target, feature_names, "target");

    let test_sample: Vec<Vec<f64>> = vec![(0..n_features)
        .map(|_| {
            if cursor < data.len() {
                let v = (data[cursor] as f64 / 128.0) - 1.0;
                cursor += 1;
                v
            } else {
                0.0
            }
        })
        .collect()];

    match dispatch {
        0 => {
            let mut model = HistGradientBoostingClassifier::new()
                .n_estimators(3)
                .max_depth(3)
                .learning_rate(0.1)
                .seed(42);
            if model.fit(&dataset).is_ok() {
                let _ = model.predict(&test_sample);
                let _ = model.predict_proba(&test_sample);
            }
        }
        _ => {
            let mut model = HistGradientBoostingRegressor::new()
                .n_estimators(3)
                .max_depth(3)
                .learning_rate(0.1)
                .seed(42);
            if model.fit(&dataset).is_ok() {
                let _ = model.predict(&test_sample);
            }
        }
    }
});
