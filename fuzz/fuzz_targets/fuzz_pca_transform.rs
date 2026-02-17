//! Fuzz target: PCA fit + transform + inverse_transform.
//!
//! Exercises Pca on fuzz-derived datasets with varying n_components.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_learn::dataset::Dataset;
use scry_learn::preprocess::{Pca, Transformer};

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let mut cursor = 0;

    let n_features = (data[cursor] % 6).max(2) as usize;
    cursor += 1;
    let n_samples = (data[cursor] % 13).max(4) as usize;
    cursor += 1;
    let n_components = (data[cursor] as usize % n_features).max(1);
    cursor += 1;

    // Build column-major feature matrix.
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(n_features);
    for _ in 0..n_features {
        let mut col = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            if cursor < data.len() {
                let v = (data[cursor] as f64 / 128.0) - 1.0;
                col.push(v.clamp(-1e6, 1e6));
                cursor += 1;
            } else {
                col.push(0.0);
            }
        }
        features.push(col);
    }

    let target: Vec<f64> = (0..n_samples).map(|i| (i % 2) as f64).collect();
    let feature_names: Vec<String> = (0..n_features).map(|i| format!("f{i}")).collect();
    let dataset = Dataset::new(features, target, feature_names, "target");

    let mut pca = Pca::with_n_components(n_components);
    if pca.fit(&dataset).is_ok() {
        // Transform a clone of the dataset.
        let mut d = dataset.clone();
        if pca.transform(&mut d).is_ok() {
            // Inverse transform back.
            let _ = pca.inverse_transform(&mut d);
        }

        let _ = pca.explained_variance_ratio();
        let _ = pca.explained_variance();
        let _ = pca.components();
    }
});
