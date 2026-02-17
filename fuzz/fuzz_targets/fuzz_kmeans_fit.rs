//! Fuzz target: Clustering fit + predict paths.
//!
//! Exercises KMeans, MiniBatchKMeans, and Dbscan on fuzz-derived datasets.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_learn::cluster::{Dbscan, KMeans, MiniBatchKMeans};
use scry_learn::dataset::Dataset;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let mut cursor = 0;

    let n_features = (data[cursor] % 4).max(1) as usize;
    cursor += 1;
    let n_samples = (data[cursor] % 17).max(4) as usize;
    cursor += 1;
    let dispatch = data[cursor] % 3;
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
            // KMeans
            let k = if cursor < data.len() {
                (data[cursor] % 6).max(1) as usize
            } else {
                2
            };
            let mut model = KMeans::new(k).max_iter(5).seed(42);
            if model.fit(&dataset).is_ok() {
                let _ = model.predict(&test_sample);
                let _ = model.transform(&test_sample);
            }
        }
        1 => {
            // MiniBatchKMeans
            let k = if cursor < data.len() {
                (data[cursor] % 6).max(1) as usize
            } else {
                2
            };
            let mut model = MiniBatchKMeans::new(k)
                .max_iter(5)
                .batch_size(n_samples.max(1))
                .seed(42);
            if model.fit(&dataset).is_ok() {
                let _ = model.predict(&test_sample);
            }
        }
        _ => {
            // Dbscan
            let eps = if cursor < data.len() {
                (data[cursor] as f64 / 50.0).max(0.01)
            } else {
                0.5
            };
            let min_samples = if cursor + 1 < data.len() {
                (data[cursor + 1] % 5).max(1) as usize
            } else {
                2
            };
            let mut model = Dbscan::new(eps, min_samples);
            if model.fit(&dataset).is_ok() {
                let _ = model.predict(&test_sample);
            }
        }
    }
});
