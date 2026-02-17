//! Fuzz target: MLP neural network regressor.
//!
//! Feeds random network architectures and inputs through `MLPRegressor`
//! to verify no panics on any combination of architecture + data.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_learn::dataset::Dataset;
use scry_learn::neural::MLPRegressor;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let mut cursor = 0;

    // Parse network architecture from fuzz bytes.
    let n_layers = (data[cursor] % 3).max(1) as usize;
    cursor += 1;

    let mut hidden_layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        if cursor >= data.len() {
            return;
        }
        let size = (data[cursor] % 16).max(1) as usize;
        cursor += 1;
        hidden_layers.push(size);
    }

    if cursor + 2 > data.len() {
        return;
    }
    let n_features = (data[cursor] % 8).max(1) as usize;
    cursor += 1;
    let n_samples = (data[cursor] % 17).max(4) as usize;
    cursor += 1;

    // Build column-major feature matrix.
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(n_features);
    for _ in 0..n_features {
        let mut col = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            if cursor < data.len() {
                col.push((data[cursor] as f64 / 128.0) - 1.0);
                cursor += 1;
            } else {
                col.push(0.0);
            }
        }
        features.push(col);
    }

    // Continuous target for regression.
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

    let mut reg = MLPRegressor::new()
        .hidden_layers(&hidden_layers)
        .max_iter(3)
        .batch_size(n_samples.max(1))
        .learning_rate(0.01)
        .seed(42);

    if reg.fit(&dataset).is_err() {
        return;
    }

    // Build test samples from remaining fuzz bytes.
    let mut test_samples: Vec<Vec<f64>> = Vec::new();
    let test_count = 2.min((data.len().saturating_sub(cursor)) / n_features + 1);
    for _ in 0..test_count {
        let mut sample = Vec::with_capacity(n_features);
        for _ in 0..n_features {
            if cursor < data.len() {
                sample.push((data[cursor] as f64 / 128.0) - 1.0);
                cursor += 1;
            } else {
                sample.push(0.0);
            }
        }
        test_samples.push(sample);
    }

    if !test_samples.is_empty() {
        let _ = reg.predict(&test_samples);
    }
});
