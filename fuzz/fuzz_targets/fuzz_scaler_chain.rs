//! Fuzz target: Preprocessing scaler chain.
//!
//! Feeds random data through `StandardScaler`, `MinMaxScaler`, and
//! `RobustScaler` to catch panics on degenerate inputs (zero-variance
//! columns, single-sample datasets, NaN/Inf values).

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_learn::dataset::Dataset;
use scry_learn::preprocess::{MinMaxScaler, RobustScaler, StandardScaler, Transformer};

/// Read an `f64` from fuzz data (little-endian), advancing the cursor.
fn take_f64(data: &[u8], cursor: &mut usize) -> Option<f64> {
    if *cursor + 8 > data.len() {
        return None;
    }
    let v = f64::from_le_bytes([
        data[*cursor],
        data[*cursor + 1],
        data[*cursor + 2],
        data[*cursor + 3],
        data[*cursor + 4],
        data[*cursor + 5],
        data[*cursor + 6],
        data[*cursor + 7],
    ]);
    *cursor += 8;
    Some(v)
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    let mut cursor = 0;

    // Parse dimensions from fuzz bytes.
    let n_samples = (data[cursor] % 50).max(1) as usize;
    cursor += 1;
    let n_features = (data[cursor] % 10).max(1) as usize;
    cursor += 1;

    // Build column-major feature matrix from fuzz bytes.
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(n_features);
    for _ in 0..n_features {
        let mut col = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let v = take_f64(data, &mut cursor).unwrap_or(0.0);
            // Clamp to finite range — we're testing degenerate-but-valid data
            // (zero-variance, single-sample, extreme values), not NaN propagation.
            col.push(if v.is_finite() { v.clamp(-1e18, 1e18) } else { 0.0 });
        }
        features.push(col);
    }

    // Build a simple target vector (not used by scalers, but needed by Dataset).
    let target: Vec<f64> = (0..n_samples).map(|i| (i % 2) as f64).collect();

    let feature_names: Vec<String> = (0..n_features).map(|i| format!("f{i}")).collect();
    let dataset = Dataset::new(features, target, feature_names, "target");

    // Test StandardScaler: fit + transform + inverse_transform.
    {
        let mut scaler = StandardScaler::new();
        let mut d = dataset.clone();
        if scaler.fit(&d).is_ok() {
            let _ = scaler.transform(&mut d);
            let _ = scaler.inverse_transform(&mut d);
        }
    }

    // Test MinMaxScaler: fit + transform + inverse_transform.
    {
        let mut scaler = MinMaxScaler::new();
        let mut d = dataset.clone();
        if scaler.fit(&d).is_ok() {
            let _ = scaler.transform(&mut d);
            let _ = scaler.inverse_transform(&mut d);
        }
    }

    // Test RobustScaler: fit + transform + inverse_transform.
    {
        let mut scaler = RobustScaler::new();
        let mut d = dataset.clone();
        if scaler.fit(&d).is_ok() {
            let _ = scaler.transform(&mut d);
            let _ = scaler.inverse_transform(&mut d);
        }
    }

    // Test fit_transform convenience method on all three.
    {
        let mut d = dataset.clone();
        let mut ss = StandardScaler::new();
        let _ = ss.fit_transform(&mut d);

        let mut mm = MinMaxScaler::new();
        let _ = mm.fit_transform(&mut d);

        let mut rs = RobustScaler::new();
        let _ = rs.fit_transform(&mut d);
    }
});
