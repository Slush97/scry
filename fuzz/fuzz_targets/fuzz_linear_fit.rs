//! Fuzz target: Linear model fit + predict paths.
//!
//! Exercises LinearRegression, LogisticRegression, Lasso, ElasticNet,
//! and Ridge on fuzz-derived datasets. Tests that no combination of
//! inputs causes panics or UB.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_learn::dataset::Dataset;
use scry_learn::linear::{
    ElasticNet, LassoRegression, LinearRegression, LogisticRegression, Ridge, Solver,
};

fuzz_target!(|data: &[u8]| {
    if data.len() < 10 {
        return;
    }

    let mut cursor = 0;

    // Parse dimensions.
    let n_features = (data[cursor] % 6).max(1) as usize;
    cursor += 1;
    let n_samples = (data[cursor] % 17).max(4) as usize;
    cursor += 1;
    let dispatch = data[cursor] % 5;
    cursor += 1;

    // Build column-major feature matrix.
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(n_features);
    for _ in 0..n_features {
        let mut col = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            if cursor < data.len() {
                let v = (data[cursor] as f64 / 128.0) - 1.0; // [-1, ~1]
                col.push(v.clamp(-1e6, 1e6));
                cursor += 1;
            } else {
                col.push(0.0);
            }
        }
        features.push(col);
    }

    // Target: regression values or binary classes depending on model.
    let is_classifier = dispatch == 2;
    let target: Vec<f64> = (0..n_samples)
        .map(|i| {
            if is_classifier {
                // LogisticRegression needs class labels.
                (i % 2) as f64
            } else if cursor < data.len() {
                let c = cursor;
                cursor += 1;
                let v = (data[c] as f64 / 128.0) - 1.0;
                v.clamp(-1e6, 1e6)
            } else {
                (i % 3) as f64
            }
        })
        .collect();

    let feature_names: Vec<String> = (0..n_features).map(|i| format!("f{i}")).collect();
    let dataset = Dataset::new(features, target, feature_names, "target");

    // Build a small test sample from remaining bytes.
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
            // LinearRegression
            let mut model = LinearRegression::new();
            if model.fit(&dataset).is_ok() {
                let _ = model.predict(&test_sample);
            }
        }
        1 => {
            // Ridge
            let mut model = Ridge::new(1.0);
            if model.fit(&dataset).is_ok() {
                let _ = model.predict(&test_sample);
            }
        }
        2 => {
            // LogisticRegression
            let mut model = LogisticRegression::new()
                .max_iter(10)
                .learning_rate(0.01)
                .solver(Solver::GradientDescent);
            if model.fit(&dataset).is_ok() {
                let _ = model.predict(&test_sample);
                let _ = model.predict_proba(&test_sample);
            }
        }
        3 => {
            // Lasso
            let mut model = LassoRegression::new().alpha(0.1).max_iter(10);
            if model.fit(&dataset).is_ok() {
                let _ = model.predict(&test_sample);
            }
        }
        _ => {
            // ElasticNet
            let mut model = ElasticNet::new().alpha(0.1).l1_ratio(0.5).max_iter(10);
            if model.fit(&dataset).is_ok() {
                let _ = model.predict(&test_sample);
            }
        }
    }
});
