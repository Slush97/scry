//! Fuzz target: Memory-mapped dataset parsing.
//!
//! Writes raw fuzz bytes as a `.scry` file and exercises `MmapDataset::open()`
//! and `batch()`. Tests malformed headers, truncated data, bad magic, wrong version.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_learn::mmap::MmapDataset;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    // Write fuzz bytes to a temp file.
    let dir = std::env::temp_dir();
    let id = data.as_ptr() as usize;
    let path = dir.join(format!("fuzz_mmap_{id}.scry"));

    if std::fs::write(&path, data).is_err() {
        return;
    }

    // Try to open the fuzz-generated file.
    if let Ok(dataset) = MmapDataset::open(&path) {
        let _ = dataset.n_samples();
        let _ = dataset.n_features();
        let _ = dataset.feature_names();
        let _ = dataset.target_name();

        // Exercise batch if there are samples.
        let n = dataset.n_samples();
        if n > 0 {
            let end = n.min(4);
            let _ = dataset.batch(0, end);
        }
    }

    // Clean up.
    let _ = std::fs::remove_file(&path);
});
