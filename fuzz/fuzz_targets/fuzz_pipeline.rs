//! Fuzz target: end-to-end Pipeline.
//!
//! Builds a Pipeline with StandardScaler + LinearRegression and exercises
//! fit + predict on fuzz-derived data.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_learn::dataset::Dataset;
use scry_learn::linear::LinearRegression;
use scry_learn::pipeline::Pipeline;
use scry_learn::preprocess::StandardScaler;

fuzz_target!(|data: &[u8]| {
    if data.len() < 10 {
        return;
    }

    let mut cursor = 0;

    let n_features = (data[cursor] % 4).max(1) as usize;
    cursor += 1;
    let n_samples = (data[cursor] % 13).max(4) as usize;
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

    // Continuous target.
    let target: Vec<f64> = (0..n_samples)
        .map(|i| {
            if cursor < data.len() {
                let c = cursor;
                cursor += 1;
                (data[c] as f64 / 128.0) - 1.0
            } else {
                (i as f64) / n_samples as f64
            }
        })
        .collect();

    let feature_names: Vec<String> = (0..n_features).map(|i| format!("f{i}")).collect();
    let dataset = Dataset::new(features, target, feature_names, "target");

    // Build test dataset for predict.
    let mut test_features: Vec<Vec<f64>> = Vec::with_capacity(n_features);
    for _ in 0..n_features {
        let mut col = Vec::with_capacity(2);
        for _ in 0..2 {
            if cursor < data.len() {
                let v = (data[cursor] as f64 / 128.0) - 1.0;
                col.push(v);
                cursor += 1;
            } else {
                col.push(0.0);
            }
        }
        test_features.push(col);
    }
    let test_target = vec![0.0; 2];
    let test_names: Vec<String> = (0..n_features).map(|i| format!("f{i}")).collect();
    let test_dataset = Dataset::new(test_features, test_target, test_names, "target");

    let mut pipe = Pipeline::new()
        .add_transformer(StandardScaler::new())
        .set_model(LinearRegression::new());

    if pipe.fit(&dataset).is_ok() {
        let _ = pipe.predict(&test_dataset);
    }
});
