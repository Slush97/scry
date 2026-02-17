//! Fuzz target: remaining preprocessing transformers.
//!
//! Exercises PolynomialFeatures, SimpleImputer, Normalizer, OneHotEncoder,
//! and LabelEncoder on fuzz-derived data.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_learn::dataset::Dataset;
use scry_learn::preprocess::{
    LabelEncoder, Norm, Normalizer, OneHotEncoder, PolynomialFeatures, SimpleImputer, Strategy,
    Transformer,
};

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

    match dispatch {
        0 => {
            // PolynomialFeatures
            let degree = if cursor < data.len() {
                (data[cursor] % 2) + 2 // degree 2-3
            } else {
                2
            };
            let mut dataset = Dataset::new(features, target, feature_names, "target");
            let mut poly = PolynomialFeatures::new().degree(degree as usize);
            if poly.fit(&dataset).is_ok() {
                let _ = poly.transform(&mut dataset);
            }
        }
        1 => {
            // SimpleImputer — inject some NaNs.
            let mut features_with_nan = features;
            for col in &mut features_with_nan {
                for val in col.iter_mut() {
                    if (*val * 100.0) as i64 % 7 == 0 {
                        *val = f64::NAN;
                    }
                }
            }
            let mut dataset =
                Dataset::new(features_with_nan, target, feature_names, "target");
            let mut imputer = SimpleImputer::new().strategy(Strategy::Mean);
            if imputer.fit(&dataset).is_ok() {
                let _ = imputer.transform(&mut dataset);
            }
        }
        2 => {
            // Normalizer
            let norm = if cursor < data.len() {
                match data[cursor] % 3 {
                    0 => Norm::L1,
                    1 => Norm::L2,
                    _ => Norm::Max,
                }
            } else {
                Norm::L2
            };
            let mut dataset = Dataset::new(features, target, feature_names, "target");
            let mut normalizer = Normalizer::new(norm);
            if normalizer.fit(&dataset).is_ok() {
                let _ = normalizer.transform(&mut dataset);
            }
        }
        3 => {
            // OneHotEncoder — use integer-like features.
            let mut int_features: Vec<Vec<f64>> = Vec::with_capacity(n_features);
            for col in &features {
                int_features.push(col.iter().map(|v| (v.abs() * 3.0).floor()).collect());
            }
            let indices: Vec<usize> = (0..n_features).collect();
            let mut dataset =
                Dataset::new(int_features, target, feature_names, "target");
            let mut encoder = OneHotEncoder::new(indices);
            if encoder.fit(&dataset).is_ok() {
                let _ = encoder.transform(&mut dataset);
            }
        }
        _ => {
            // LabelEncoder
            let labels: Vec<String> = (0..n_samples)
                .map(|i| {
                    if cursor < data.len() {
                        let c = cursor;
                        cursor += 1;
                        format!("label_{}", data[c] % 5)
                    } else {
                        format!("label_{}", i % 3)
                    }
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            let mut enc = LabelEncoder::new();
            enc.fit(&label_refs);
            let _ = enc.transform(&label_refs);
        }
    }
});
