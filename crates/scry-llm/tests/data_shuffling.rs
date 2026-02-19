//! Tests for intra-shard shuffling in `DataLoader`.

use scry_llm::data::DataLoader;

#[test]
fn different_order_across_epochs() {
    let tokens: Vec<u16> = (0..200).collect();
    let seq_len = 4;
    let batch_size = 2;

    // Epoch 1
    let mut loader1 = DataLoader::from_tokens(tokens.clone(), seq_len, batch_size, 42);
    let mut epoch1_inputs = Vec::new();
    for _ in 0..5 {
        let batch = loader1.next_batch().unwrap();
        epoch1_inputs.extend(batch.input_ids);
    }

    // Force reshuffle (new loader wraps = epoch 2)
    let mut loader2 = DataLoader::from_tokens(tokens, seq_len, batch_size, 43); // different seed
    let mut epoch2_inputs = Vec::new();
    for _ in 0..5 {
        let batch = loader2.next_batch().unwrap();
        epoch2_inputs.extend(batch.input_ids);
    }

    // The order should differ with different seeds
    assert_ne!(
        epoch1_inputs, epoch2_inputs,
        "different seeds should produce different batch orderings"
    );
}

#[test]
fn all_tokens_visited() {
    // With enough batches, all valid starting positions should be covered
    let n_tokens = 50;
    let tokens: Vec<u16> = (0..u16::try_from(n_tokens).unwrap()).collect();
    let seq_len = 4;
    let batch_size = 1;
    let mut loader = DataLoader::from_tokens(tokens, seq_len, batch_size, 42);

    let mut seen_starts = std::collections::HashSet::new();
    // We have n_tokens/(seq_len+1) valid positions ≈ 10
    // Fetch enough batches to see all of them (may need to wrap)
    for _ in 0..20 {
        let batch = loader.next_batch().unwrap();
        // The first input token tells us the starting position
        seen_starts.insert(batch.input_ids[0]);
    }

    // We should have seen at least 2 different starting positions
    assert!(
        seen_starts.len() >= 2,
        "should visit multiple starting positions, got {}",
        seen_starts.len()
    );
}
