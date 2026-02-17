//! Fuzz target: Naive Bayes classifiers.
//!
//! Exercises GaussianNb, MultinomialNB, and BernoulliNB on fuzz-derived
//! datasets. MultinomialNB gets non-negative count values.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_learn::dataset::Dataset;
use scry_learn::naive_bayes::{BernoulliNB, GaussianNb, MultinomialNB};

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let mut cursor = 0;

    let n_features = (data[cursor] % 4).max(1) as usize;
    cursor += 1;
    let n_samples = (data[cursor] % 13).max(4) as usize;
    cursor += 1;
    let n_classes = (data[cursor] % 3).max(2) as usize;
    cursor += 1;
    let dispatch = data[cursor] % 3;
    cursor += 1;

    // Build column-major feature matrix.
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(n_features);
    for _ in 0..n_features {
        let mut col = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            if cursor < data.len() {
                let v = if dispatch == 1 {
                    // MultinomialNB: non-negative counts.
                    (data[cursor] as f64 / 2.55).clamp(0.0, 100.0)
                } else {
                    (data[cursor] as f64 / 128.0) - 1.0
                };
                col.push(v);
                cursor += 1;
            } else {
                col.push(0.0);
            }
        }
        features.push(col);
    }

    let target: Vec<f64> = (0..n_samples).map(|i| (i % n_classes) as f64).collect();
    let feature_names: Vec<String> = (0..n_features).map(|i| format!("f{i}")).collect();
    let dataset = Dataset::new(features, target, feature_names, "class");

    let test_sample: Vec<Vec<f64>> = vec![(0..n_features)
        .map(|_| {
            if cursor < data.len() {
                let v = if dispatch == 1 {
                    (data[cursor] as f64 / 2.55).clamp(0.0, 100.0)
                } else {
                    (data[cursor] as f64 / 128.0) - 1.0
                };
                cursor += 1;
                v
            } else {
                0.0
            }
        })
        .collect()];

    match dispatch {
        0 => {
            let mut model = GaussianNb::new();
            if model.fit(&dataset).is_ok() {
                let _ = model.predict(&test_sample);
                let _ = model.predict_proba(&test_sample);
            }
        }
        1 => {
            let mut model = MultinomialNB::new().alpha(1.0);
            if model.fit(&dataset).is_ok() {
                let _ = model.predict(&test_sample);
                let _ = model.predict_proba(&test_sample);
            }
        }
        _ => {
            let mut model = BernoulliNB::new().alpha(1.0);
            if model.fit(&dataset).is_ok() {
                let _ = model.predict(&test_sample);
                let _ = model.predict_proba(&test_sample);
            }
        }
    }
});
