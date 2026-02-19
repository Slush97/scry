//! Tests for the `DataLoader` on synthetic shard data.

use scry_llm::data::DataLoader;

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
