//! Fuzz target: PipelineEngine transform with random inputs.
//!
//! Splits fuzz bytes into two regions:
//! 1. Pipeline JSON (first N bytes, length prefix)
//! 2. Input row values (remaining bytes, interpreted as f64s)
//!
//! If the JSON parses into a valid `PipelineDef`, we build an engine and
//! attempt to transform the fuzz-derived input row. Errors are fine —
//! we only care about no panics and no UB.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_pipe::engine::PipelineEngine;
use scry_pipe::ir::PipelineDef;

fuzz_target!(|data: &[u8]| {
    // Need at least 2 bytes for the JSON length prefix + some JSON.
    if data.len() < 4 {
        return;
    }

    // First 2 bytes = JSON region length (little-endian u16).
    let json_len = u16::from_le_bytes([data[0], data[1]]) as usize;
    let json_end = 2 + json_len;

    if json_end > data.len() {
        return;
    }

    // Parse the JSON region.
    let Ok(json_str) = std::str::from_utf8(&data[2..json_end]) else {
        return;
    };
    let Ok(def) = PipelineDef::from_json(json_str) else {
        return;
    };

    let engine = PipelineEngine::new(def.clone());
    let n_input = def.input_schema.len();

    if n_input == 0 {
        // Empty schema: just try the empty row.
        let _ = engine.transform_row(&[]);
        return;
    }

    // Build input row from remaining bytes, interpreting as f64s.
    let remaining = &data[json_end..];
    let mut row = Vec::with_capacity(n_input);
    for i in 0..n_input {
        let byte_start = i * 8;
        if byte_start + 8 <= remaining.len() {
            let v = f64::from_le_bytes([
                remaining[byte_start],
                remaining[byte_start + 1],
                remaining[byte_start + 2],
                remaining[byte_start + 3],
                remaining[byte_start + 4],
                remaining[byte_start + 5],
                remaining[byte_start + 6],
                remaining[byte_start + 7],
            ]);
            row.push(if v.is_finite() { v } else { 0.0 });
        } else {
            row.push(0.0);
        }
    }

    // Exercise single-row transform — errors are fine, panics are not.
    let _ = engine.transform_row(&row);

    // Also try batch transform with the single row.
    let _ = engine.transform_batch(&[row]);
});
