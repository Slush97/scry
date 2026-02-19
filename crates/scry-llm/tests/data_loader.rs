//! Tests for the `DataLoader` on synthetic shard data.

use std::path::Path;

use scry_llm::data::DataLoader;

/// Write a shard file (packed u16 LE) to the given directory.
fn write_test_shard(dir: &Path, name: &str, tokens: &[u16]) {
    let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    std::fs::write(dir.join(name), &bytes).unwrap();
}

#[test]
fn correct_batch_shapes() {
    // Synthetic tokens: 0,1,2,...,99
    let tokens: Vec<u16> = (0..100).collect();
    let seq_len = 4;
    let batch_size = 2;
    let mut loader = DataLoader::from_tokens(tokens, seq_len, batch_size, 42);

    let batch = loader.next_batch().unwrap();
    assert_eq!(batch.batch_size, 2);
    assert_eq!(batch.seq_len, 4);
    assert_eq!(batch.input_ids.len(), 8); // 2 * 4
    assert_eq!(batch.targets.len(), 8);
}

#[test]
fn input_target_offset_by_one() {
    // Verify input/target offset relationship regardless of position
    let tokens: Vec<u16> = (0..100).collect();
    let seq_len = 4;
    let batch_size = 1;
    let mut loader = DataLoader::from_tokens(tokens, seq_len, batch_size, 42);

    let batch = loader.next_batch().unwrap();
    // Each target[i] should be input[i] + 1 (since tokens are contiguous integers)
    for i in 0..seq_len {
        assert_eq!(
            batch.targets[i],
            batch.input_ids[i] + 1,
            "target should be input + 1 at index {i}"
        );
    }

    let batch2 = loader.next_batch().unwrap();
    for i in 0..seq_len {
        assert_eq!(
            batch2.targets[i],
            batch2.input_ids[i] + 1,
            "batch2: target should be input + 1 at index {i}"
        );
    }
}

#[test]
fn shard_wrapping() {
    // Only 12 tokens -> can only fit 2 batches of (seq=4, batch=1) before wrapping
    let tokens: Vec<u16> = (0..12).collect();
    let seq_len = 4;
    let batch_size = 1;
    let mut loader = DataLoader::from_tokens(tokens, seq_len, batch_size, 42);

    // Exhaust all positions then force a wrap
    let _ = loader.next_batch().unwrap();
    let _ = loader.next_batch().unwrap();
    // Should wrap around (positions reshuffled)
    let batch3 = loader.next_batch().unwrap();
    assert_eq!(batch3.input_ids.len(), 4);
    assert_eq!(batch3.targets.len(), 4);
    // Verify offset-by-1 property still holds
    for i in 0..seq_len {
        assert_eq!(batch3.targets[i], batch3.input_ids[i] + 1);
    }
}

#[test]
fn multi_batch_sequences_are_valid() {
    let tokens: Vec<u16> = (0..100).collect();
    let seq_len = 3;
    let batch_size = 3;
    let mut loader = DataLoader::from_tokens(tokens, seq_len, batch_size, 42);

    let batch = loader.next_batch().unwrap();
    assert_eq!(batch.input_ids.len(), 9); // 3 * 3
    assert_eq!(batch.targets.len(), 9);

    // Each sequence should have input/target offset by 1
    for b in 0..batch_size {
        for i in 0..seq_len {
            let idx = b * seq_len + i;
            assert_eq!(
                batch.targets[idx],
                batch.input_ids[idx] + 1,
                "seq {b} pos {i}: target should be input + 1"
            );
        }
    }
}

#[test]
fn multi_shard_loading_and_advancement() {
    let dir = std::env::temp_dir().join("scry_test_multi_shard");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // 3 shards with distinct contiguous tokens so we can tell which shard we're in.
    // Shard 0: 0..50, Shard 1: 100..150, Shard 2: 200..250
    let shard0: Vec<u16> = (0..50).collect();
    let shard1: Vec<u16> = (100..150).collect();
    let shard2: Vec<u16> = (200..250).collect();

    write_test_shard(&dir, "train_0000.bin", &shard0);
    write_test_shard(&dir, "train_0001.bin", &shard1);
    write_test_shard(&dir, "train_0002.bin", &shard2);

    let seq_len = 4;
    let batch_size = 1;
    let mut loader = DataLoader::new(&dir, "train", seq_len, batch_size, 42).unwrap();

    // Pull many batches — enough to exhaust the first shard and advance
    let mut seen_ranges = std::collections::HashSet::new();
    for _ in 0..100 {
        let batch = loader.next_batch().unwrap();
        assert_eq!(batch.input_ids.len(), seq_len);
        assert_eq!(batch.targets.len(), seq_len);

        // Determine which value range (shard) this batch came from
        let first = batch.input_ids[0];
        let range = if first < 50 {
            0
        } else if (100..150).contains(&first) {
            1
        } else if (200..250).contains(&first) {
            2
        } else {
            panic!("unexpected token {first} — possible shard boundary corruption");
        };
        seen_ranges.insert(range);

        // Within each sequence, verify contiguity (offset-by-1)
        for i in 0..seq_len {
            assert_eq!(
                batch.targets[i],
                batch.input_ids[i] + 1,
                "target should be input + 1 at pos {i} (input={})",
                batch.input_ids[i]
            );
        }
    }

    // We should have drawn from all 3 shards
    assert!(
        seen_ranges.len() >= 2,
        "Expected batches from multiple shards, only saw ranges: {seen_ranges:?}"
    );
    println!("  multi-shard test: saw shard ranges {seen_ranges:?}");

    // Cleanup
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn shard_with_exact_fit() {
    // Shard has exactly enough tokens for N batches with no remainder
    let seq_len = 4;
    let batch_size = 2;
    let tokens_per_item = seq_len + 1; // 5
    let n_items = batch_size * 3; // 6 items = 30 tokens needed for data, plus 1 for target offset
    let n_tokens = n_items * tokens_per_item;
    let tokens: Vec<u16> = (0..n_tokens as u16).collect();
    let mut loader = DataLoader::from_tokens(tokens, seq_len, batch_size, 42);

    // Should be able to pull at least 3 batches without error
    for i in 0..6 {
        let batch = loader.next_batch().unwrap();
        assert_eq!(batch.input_ids.len(), batch_size * seq_len, "batch {i} input size");
        assert_eq!(batch.targets.len(), batch_size * seq_len, "batch {i} target size");
    }
}
