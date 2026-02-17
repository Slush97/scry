//! Fuzz target: Agglomerative (hierarchical) clustering.
//!
//! Exercises AgglomerativeClustering with all four linkage strategies
//! on fuzz-derived datasets.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_learn::cluster::{AgglomerativeClustering, Linkage};
use scry_learn::dataset::Dataset;

fuzz_target!(|data: &[u8]| {
    if data.len() < 10 {
        return;
    }

    let mut cursor = 0;

    let n_features = (data[cursor] % 4).max(1) as usize;
    cursor += 1;
    let n_samples = (data[cursor] % 13).max(4) as usize;
    cursor += 1;
    let n_clusters = (data[cursor] % 4).max(1) as usize;
    cursor += 1;
    let linkage = match data[cursor] % 4 {
        0 => Linkage::Single,
        1 => Linkage::Complete,
        2 => Linkage::Average,
        _ => Linkage::Ward,
    };
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

    let mut model = AgglomerativeClustering::new(n_clusters).linkage(linkage);
    let _ = model.fit(&dataset);
});
