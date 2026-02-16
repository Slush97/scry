//! Fuzz target: MLP neural network forward pass.
//!
//! Feeds random network architectures and inputs through `MLPClassifier`
//! predict path to verify no panics on any combination of architecture + data.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_learn::dataset::Dataset;
use scry_learn::neural::MLPClassifier;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let mut cursor = 0;

    // Parse network architecture from fuzz bytes.
    let n_layers = (data[cursor] % 3).max(1) as usize; // 1-3 hidden layers
    cursor += 1;

    let mut hidden_layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        if cursor >= data.len() {
            return;
        }
        let size = (data[cursor] % 16).max(1) as usize; // 1-16 neurons
        cursor += 1;
        hidden_layers.push(size);
    }

    // Parse dataset dimensions.
    if cursor + 2 > data.len() {
        return;
    }
    let n_features = (data[cursor] % 8).max(1) as usize; // 1-8 features
    cursor += 1;
    let n_samples = (data[cursor] % 20).max(4) as usize; // 4-23 samples
    cursor += 1;

    let n_classes = 2usize; // Binary classification — keep it simple for fuzzing speed.

    // Build column-major feature matrix from remaining fuzz bytes.
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(n_features);
    for _ in 0..n_features {
        let mut col = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            if cursor >= data.len() {
                col.push(0.0);
            } else {
                // Map byte to [-2.0, 2.0] range for reasonable training values.
                col.push((data[cursor] as f64 / 128.0) - 1.0);
                cursor += 1;
            }
        }
        features.push(col);
    }

    // Binary target: assign classes based on fuzz bytes.
    let target: Vec<f64> = (0..n_samples)
        .map(|i| (i % n_classes) as f64)
        .collect();

    let feature_names: Vec<String> = (0..n_features).map(|i| format!("f{i}")).collect();
    let dataset = Dataset::new(features, target, feature_names, "class");

    // Build and train the MLP with very few iterations (speed).
    let mut clf = MLPClassifier::new()
        .hidden_layers(&hidden_layers)
        .max_iter(3)
        .batch_size(n_samples.max(1))
        .learning_rate(0.01)
        .seed(42);

    if clf.fit(&dataset).is_err() {
        return;
    }

    // Build test samples from remaining fuzz bytes.
    let mut test_samples: Vec<Vec<f64>> = Vec::new();
    let test_count = 2.min((data.len() - cursor) / n_features + 1);
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
        // These should not panic regardless of input.
        let _ = clf.predict(&test_samples);
        let _ = clf.predict_proba(&test_samples);
    }
});
