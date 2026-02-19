//! Tests for batched gradient checkpointing.

use scry_llm::autograd::backward::backward;
use scry_llm::autograd::ops;
use scry_llm::autograd::GradTape;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::backend::DeviceBackend;
use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};
use scry_llm::nn::Module;

type Cpu = CpuBackend;

fn tiny_config() -> Gpt2Config {
    Gpt2Config {
        vocab_size: 10,
        max_seq_len: 16,
        d_model: 8,
        n_heads: 2,
        n_layers: 4,
        d_ff: 16,
        dropout_rate: 0.0,
    }
}

#[test]
fn checkpointed_matches_standard() {
    let config = tiny_config();
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(config.clone(), &mut rng);

    let batch_size = 2;
    let seq_len = 4;
    let token_ids: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let targets: Vec<usize> = vec![1, 2, 3, 4, 5, 6, 7, 8];

    // Standard forward+backward
    let mut rng1 = fastrand::Rng::with_seed(99);
    let mut tape1 = GradTape::<Cpu>::new();
    let logits1 = model.forward_batch(&token_ids, batch_size, seq_len, &mut rng1, &mut tape1);
    let loss1 = ops::cross_entropy(
        &logits1,
        &targets,
        batch_size * seq_len,
        config.vocab_size,
        Some(&mut tape1),
    );
    let loss1_val = loss1.to_vec()[0];
    let grads1 = backward(&tape1, loss1.id);

    // Checkpointed forward+backward (same RNG seed for same dropout)
    let mut rng2 = fastrand::Rng::with_seed(99);
    let mut tape2 = GradTape::<Cpu>::new();
    let logits2 = model.forward_batch_checkpointed(
        &token_ids,
        batch_size,
        seq_len,
        2, // checkpoint every 2 blocks
        &mut rng2,
        &mut tape2,
    );
    let loss2 = ops::cross_entropy(
        &logits2,
        &targets,
        batch_size * seq_len,
        config.vocab_size,
        Some(&mut tape2),
    );
    let loss2_val = loss2.to_vec()[0];
    let grads2 = model.backward_checkpointed(&tape2, loss2.id);

    // Losses should match exactly (same computation)
    assert!(
        (loss1_val - loss2_val).abs() < 1e-5,
        "losses differ: {loss1_val} vs {loss2_val}"
    );

    // Compare gradients for all parameters
    for param in model.parameters() {
        let g1 = grads1.get(&param.id);
        let g2 = grads2.get(&param.id);
        match (g1, g2) {
            (Some(g1), Some(g2)) => {
                let v1 = Cpu::to_vec(g1);
                let v2 = Cpu::to_vec(g2);
                assert_eq!(v1.len(), v2.len(), "grad length mismatch for param {:?}", param.id);
                let max_diff: f64 = v1
                    .iter()
                    .zip(v2.iter())
                    .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
                    .fold(0.0, f64::max);
                assert!(
                    max_diff < 1e-4,
                    "grad mismatch for param {:?}: max_diff={max_diff}",
                    param.id
                );
            }
            (None, None) => {} // both missing, ok
            _ => panic!("grad presence mismatch for param {:?}", param.id),
        }
    }
}

#[test]
fn checkpoint_every_1_vs_n() {
    let config = tiny_config();
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(config.clone(), &mut rng);

    let batch_size = 2;
    let seq_len = 4;
    let token_ids: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let targets: Vec<usize> = vec![1, 2, 3, 4, 5, 6, 7, 8];

    // checkpoint_every=1 (every block is a checkpoint)
    let mut rng1 = fastrand::Rng::with_seed(99);
    let mut tape1 = GradTape::<Cpu>::new();
    let logits1 = model.forward_batch_checkpointed(
        &token_ids, batch_size, seq_len, 1, &mut rng1, &mut tape1,
    );
    let loss1 = ops::cross_entropy(
        &logits1, &targets, batch_size * seq_len, config.vocab_size, Some(&mut tape1),
    );
    let loss1_val = loss1.to_vec()[0];

    // checkpoint_every=4 (all blocks in one segment)
    let mut rng2 = fastrand::Rng::with_seed(99);
    let mut tape2 = GradTape::<Cpu>::new();
    let logits2 = model.forward_batch_checkpointed(
        &token_ids, batch_size, seq_len, 4, &mut rng2, &mut tape2,
    );
    let loss2 = ops::cross_entropy(
        &logits2, &targets, batch_size * seq_len, config.vocab_size, Some(&mut tape2),
    );
    let loss2_val = loss2.to_vec()[0];

    // Both should produce the same loss (inference path is identical)
    assert!(
        (loss1_val - loss2_val).abs() < 1e-5,
        "checkpoint_every=1 vs 4: losses differ: {loss1_val} vs {loss2_val}"
    );
}
