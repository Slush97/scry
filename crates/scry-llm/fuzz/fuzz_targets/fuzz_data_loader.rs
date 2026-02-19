//! Fuzz the DataLoader: arbitrary tokens, variable seq_len and batch_size.
//! Must never panic.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_llm::data::DataLoader;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    let seq_len = (data[0] % 8) as usize + 1; // 1..=8
    let batch_size = (data[1] % 4) as usize + 1; // 1..=4

    // Build token buffer from remaining bytes (as u16 pairs or single bytes)
    let min_tokens = batch_size * (seq_len + 1) + 1;
    let tokens: Vec<u16> = data[2..]
        .iter()
        .map(|&b| u16::from(b) % 50)
        .collect();

    if tokens.len() < min_tokens {
        return;
    }

    let mut loader = DataLoader::from_tokens(tokens, seq_len, batch_size, 42);

    // Call next_batch 3 times — must not panic
    for _ in 0..3 {
        let _ = loader.next_batch();
    }
});
