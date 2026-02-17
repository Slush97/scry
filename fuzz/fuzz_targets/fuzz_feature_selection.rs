//! Fuzz target: VarianceThreshold + SelectKBest feature selection.
//!
//! Exercises feature selection transformers on fuzz-derived datasets.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_learn::dataset::Dataset;
use scry_learn::feature_selection::{ScoreFn, SelectKBest, VarianceThreshold};
use scry_learn::preprocess::Transformer;

fuzz_target!(|data: &[u8]| {
    if data.len() < 10 {
        return;
    }

    let mut cursor = 0;

    let n_features = (data[cursor] % 5).max(2) as usize;
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

    let target: Vec<f64> = (0..n_samples).map(|i| (i % 2) as f64).collect();
    let feature_names: Vec<String> = (0..n_features).map(|i| format!("f{i}")).collect();
    let mut dataset = Dataset::new(features, target, feature_names, "target");

    match dispatch {
        0 => {
            let threshold = if cursor < data.len() {
                data[cursor] as f64 / 255.0
            } else {
                0.0
            };
            let mut sel = VarianceThreshold::new().threshold(threshold);
            if sel.fit(&dataset).is_ok() {
                let _ = sel.transform(&mut dataset);
            }
        }
        _ => {
            let k = if cursor < data.len() {
                (data[cursor] as usize % n_features).max(1)
            } else {
                1
            };
            let mut sel = SelectKBest::new(ScoreFn::FClassif).k(k);
            if sel.fit(&dataset).is_ok() {
                let _ = sel.transform(&mut dataset);
            }
        }
    }
});
