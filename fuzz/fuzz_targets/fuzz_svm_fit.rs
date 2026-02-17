//! Fuzz target: SVM fit + predict paths.
//!
//! Exercises KernelSVC, KernelSVR, LinearSVC, and LinearSVR on
//! fuzz-derived datasets with capped iterations.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_learn::dataset::Dataset;
use scry_learn::svm::{Kernel, KernelSVC, KernelSVR, LinearSVC, LinearSVR};

fuzz_target!(|data: &[u8]| {
    if data.len() < 10 {
        return;
    }

    let mut cursor = 0;

    let n_features = (data[cursor] % 4).max(1) as usize;
    cursor += 1;
    let n_samples = (data[cursor] % 13).max(4) as usize;
    cursor += 1;
    let dispatch = data[cursor] % 5;
    cursor += 1;

    // Build column-major feature matrix.
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(n_features);
    for _ in 0..n_features {
        let mut col = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            if cursor < data.len() {
                let v = (data[cursor] as f64 / 12.8) - 10.0; // [-10, ~10]
                col.push(v);
                cursor += 1;
            } else {
                col.push(0.0);
            }
        }
        features.push(col);
    }

    // Classification target for SVC, regression for SVR.
    let is_regression = dispatch == 1 || dispatch == 4;
    let target: Vec<f64> = (0..n_samples)
        .map(|i| {
            if is_regression {
                if cursor < data.len() {
                    let v = (data[cursor] as f64 / 12.8) - 10.0;
                    cursor += 1;
                    v
                } else {
                    (i as f64) * 0.1
                }
            } else {
                (i % 2) as f64
            }
        })
        .collect();

    let feature_names: Vec<String> = (0..n_features).map(|i| format!("f{i}")).collect();
    let dataset = Dataset::new(features, target, feature_names, "target");

    let test_sample: Vec<Vec<f64>> = vec![(0..n_features)
        .map(|_| {
            if cursor < data.len() {
                let v = (data[cursor] as f64 / 12.8) - 10.0;
                cursor += 1;
                v
            } else {
                0.0
            }
        })
        .collect()];

    match dispatch {
        0 => {
            // KernelSVC with RBF
            let mut model = KernelSVC::new()
                .kernel(Kernel::RBF { gamma: 1.0 })
                .c(1.0)
                .max_iter(20);
            if model.fit(&dataset).is_ok() {
                let _ = model.predict(&test_sample);
                let _ = model.decision_function(&test_sample);
            }
        }
        1 => {
            // KernelSVR
            let mut model = KernelSVR::new()
                .kernel(Kernel::RBF { gamma: 1.0 })
                .c(1.0)
                .epsilon(0.1)
                .max_iter(20);
            if model.fit(&dataset).is_ok() {
                let _ = model.predict(&test_sample);
            }
        }
        2 => {
            // KernelSVC with Polynomial
            let mut model = KernelSVC::new()
                .kernel(Kernel::Polynomial { degree: 2, coef0: 0.0 })
                .c(1.0)
                .max_iter(20);
            if model.fit(&dataset).is_ok() {
                let _ = model.predict(&test_sample);
            }
        }
        3 => {
            // LinearSVC
            let mut model = LinearSVC::new().c(1.0).max_iter(20);
            if model.fit(&dataset).is_ok() {
                let _ = model.predict(&test_sample);
                let _ = model.decision_function(&test_sample);
            }
        }
        _ => {
            // LinearSVR
            let mut model = LinearSVR::new().c(1.0).epsilon(0.1).max_iter(20);
            if model.fit(&dataset).is_ok() {
                let _ = model.predict(&test_sample);
            }
        }
    }
});
