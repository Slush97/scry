//! Numerical edge cases for the LLM modules.

use scry_llm::autograd::ops;
use scry_llm::autograd::GradTape;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

type Cpu = CpuBackend;

fn tiny_config() -> Gpt2Config {
    Gpt2Config {
        vocab_size: 8,
        max_seq_len: 6,
        d_model: 4,
        n_heads: 2,
        n_layers: 1,
        d_ff: 8,
    }
}

fn tiny_model() -> Gpt2Model<Cpu> {
    let mut rng = fastrand::Rng::with_seed(42);
    Gpt2Model::<Cpu>::new(tiny_config(), &mut rng)
}

#[test]
fn edge_long_sequence() {
    let config = tiny_config();
    let model = tiny_model();
    // Use max_seq_len tokens
    let token_ids: Vec<usize> = (0..config.max_seq_len)
        .map(|i| i % config.vocab_size)
        .collect();
    let logits = model.forward_inference(&token_ids);
    assert_eq!(
        logits.shape.dims(),
        &[config.max_seq_len, config.vocab_size]
    );
    assert!(
        logits.to_vec().iter().all(|v| v.is_finite()),
        "long sequence produced NaN/Inf"
    );
}

#[test]
fn edge_single_token() {
    let model = tiny_model();
    let token_ids = &[3usize];
    let logits = model.forward_inference(token_ids);
    assert_eq!(logits.shape.dims(), &[1, 8]);
    assert!(
        logits.to_vec().iter().all(|v| v.is_finite()),
        "single token produced NaN/Inf"
    );
}

#[test]
fn edge_repeated_tokens() {
    let model = tiny_model();
    let token_ids = &[2usize, 2, 2, 2];
    let logits = model.forward_inference(token_ids);
    assert_eq!(logits.shape.dims(), &[4, 8]);
    assert!(
        logits.to_vec().iter().all(|v| v.is_finite()),
        "repeated tokens produced NaN/Inf"
    );
}

#[test]
fn edge_vocab_boundary() {
    let config = tiny_config();
    let model = tiny_model();
    // Use the last valid token id
    let token_ids = &[config.vocab_size - 1];
    let logits = model.forward_inference(token_ids);
    assert_eq!(logits.shape.dims(), &[1, config.vocab_size]);
    assert!(
        logits.to_vec().iter().all(|v| v.is_finite()),
        "vocab boundary token produced NaN/Inf"
    );
}

#[test]
fn edge_layernorm_zero_input() {
    // LayerNorm with all-zero input should not NaN (eps protects division)
    let input = Tensor::<Cpu>::from_vec(vec![0.0; 8], Shape::new(&[2, 4]));
    let gamma = Tensor::<Cpu>::from_vec(vec![1.0; 4], Shape::new(&[4]));
    let beta = Tensor::<Cpu>::from_vec(vec![0.0; 4], Shape::new(&[4]));

    let out = ops::layernorm(&input, &gamma, &beta, 1e-5, None);
    assert!(
        out.to_vec().iter().all(|v| v.is_finite()),
        "LayerNorm on zeros produced NaN/Inf"
    );
}

#[test]
fn edge_large_logit_cross_entropy() {
    // Logits with values ±100 — loss should still be finite (log-sum-exp stability)
    let mut logits_data = vec![0.0f32; 3 * 5];
    // Row 0: one logit at +100
    logits_data[0] = 100.0;
    // Row 1: one logit at -100
    logits_data[5 + 1] = -100.0;
    // Row 2: two large logits
    logits_data[10 + 2] = 100.0;
    logits_data[10 + 3] = 99.0;

    let logits = Tensor::<Cpu>::from_vec(logits_data, Shape::new(&[3, 5]));
    let targets = vec![0usize, 1, 2];
    let loss = ops::cross_entropy(&logits, &targets, 3, 5, None);
    let loss_val = loss.to_vec()[0];
    assert!(
        loss_val.is_finite(),
        "cross-entropy with large logits produced NaN/Inf: {loss_val}"
    );
    assert!(
        loss_val >= 0.0,
        "cross-entropy should be non-negative, got {loss_val}"
    );
}

#[test]
fn edge_single_token_training_roundtrip() {
    // Full forward + backward with single token — tests indexing edge case
    let model = tiny_model();
    let token_ids = &[1usize];
    let targets = vec![3usize];

    let mut tape = GradTape::<Cpu>::new();
    let logits = model.forward(token_ids, &mut tape);
    let loss = ops::cross_entropy(&logits, &targets, 1, 8, Some(&mut tape));
    let loss_val = loss.to_vec()[0];
    assert!(
        loss_val.is_finite(),
        "single token training loss not finite"
    );

    let grads = scry_llm::autograd::backward::backward(&tape, loss.id);
    for grad_data in grads.values() {
        assert!(
            grad_data.iter().all(|v| v.is_finite()),
            "single token backward produced NaN/Inf gradient"
        );
    }
}
