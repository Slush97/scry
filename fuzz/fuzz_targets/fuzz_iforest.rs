//! Fuzz target: Isolation Forest anomaly detection.
//!
//! Exercises IsolationForest fit + predict + predict_labels on fuzz-derived
//! row-major data. Small n_estimators and max_samples for speed.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_learn::anomaly::IsolationForest;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let mut cursor = 0;

    let n_features = (data[cursor] % 4).max(1) as usize;
    cursor += 1;
    let n_samples = (data[cursor] % 13).max(4) as usize;
    cursor += 1;

    // Build row-major feature matrix (IsolationForest takes &[Vec<f64>]).
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let mut row = Vec::with_capacity(n_features);
        for _ in 0..n_features {
            if cursor < data.len() {
                let v = (data[cursor] as f64 / 128.0) - 1.0;
                row.push(v);
                cursor += 1;
            } else {
                row.push(0.0);
            }
        }
        features.push(row);
    }

    let mut model = IsolationForest::new()
        .n_estimators(5)
        .max_samples(n_samples.min(8))
        .seed(42);

    if model.fit(&features).is_ok() {
        // Build test samples.
        let test: Vec<Vec<f64>> = vec![(0..n_features)
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

        let _ = model.predict(&test);
        let _ = model.predict_labels(&test);
    }
});
