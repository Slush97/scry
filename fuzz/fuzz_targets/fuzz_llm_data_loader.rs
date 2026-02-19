//! Fuzz target: DataLoader from_tokens + next_batch.
//!
//! Parses token count, seq_len, batch_size, and token values from fuzz bytes.
//! Calls next_batch up to 5 times. Must not panic, OOM, or produce invalid shapes.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_llm::data::DataLoader;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    let mut cursor = 0;

    let n_tokens = (data[cursor] % 46 + 5) as usize; // 5-50
    cursor += 1;
    let seq_len = (data[cursor] % 8 + 1) as usize; // 1-8
    cursor += 1;
    let batch_size = (data[cursor] % 4 + 1) as usize; // 1-4
    cursor += 1;

    // Need at least batch_size * (seq_len + 1) tokens to form one batch
    let min_tokens = batch_size * (seq_len + 1);
    let n_tokens = n_tokens.max(min_tokens);

    let mut tokens = Vec::with_capacity(n_tokens);
    for _ in 0..n_tokens {
        if cursor < data.len() {
            tokens.push(u16::from(data[cursor]));
            cursor += 1;
        } else {
            tokens.push(0);
        }
    }

    let mut loader = DataLoader::from_tokens(tokens, seq_len, batch_size, 42);

    for _ in 0..5 {
        match loader.next_batch() {
            Ok(batch) => {
                assert_eq!(batch.input_ids.len(), batch_size * seq_len);
                assert_eq!(batch.targets.len(), batch_size * seq_len);
                assert_eq!(batch.batch_size, batch_size);
                assert_eq!(batch.seq_len, seq_len);
            }
            Err(_) => break,
        }
    }
});
