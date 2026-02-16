//! Fuzz target: PipelineDef JSON round-trip.
//!
//! Takes arbitrary bytes, attempts to parse them as a JSON `PipelineDef`.
//! If parsing succeeds, re-serializes to JSON, re-parses, and asserts the
//! two `PipelineDef` values are equal. This catches any serde asymmetries
//! where serialize(deserialize(x)) != x.
//!
//! We only care about no-panic, no-UB, and round-trip fidelity.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_pipe::ir::PipelineDef;

fuzz_target!(|data: &[u8]| {
    // Attempt to interpret raw bytes as UTF-8 JSON.
    let Ok(json_str) = std::str::from_utf8(data) else {
        return;
    };

    // Try to parse as PipelineDef.
    let Ok(def1) = PipelineDef::from_json(json_str) else {
        return;
    };

    // Re-serialize to JSON.
    let json2 = def1.to_json().expect("re-serialization must not fail");

    // Re-parse.
    let def2 = PipelineDef::from_json(&json2).expect("re-parse must not fail");

    // Assert structural equality.
    assert_eq!(def1, def2, "round-trip mismatch");
});
