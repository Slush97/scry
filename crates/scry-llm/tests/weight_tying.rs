//! Verify that GPT-2 weight tying works correctly: the `token_embedding` tensor
//! receives gradient contributions from both the embedding lookup and the LM head.

use scry_llm::autograd::backward::backward;
use scry_llm::autograd::ops;
use scry_llm::autograd::GradTape;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::backend::DeviceBackend;
use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};
use scry_llm::nn::Module;

type Cpu = CpuBackend;

#[test]
fn weight_tying_gradient_receives_both_contributions() {
    let config = Gpt2Config {
        vocab_size: 5,
        max_seq_len: 4,
        d_model: 4,
        n_heads: 2,
        n_layers: 1,
        d_ff: 8,
        dropout_rate: 0.0,
    };
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(config, &mut rng);

    let token_ids = &[0, 2, 4];
    let targets = vec![2usize, 4, 1];

    // Full forward + backward (both embedding and LM head use token_embedding)
    let mut tape = GradTape::<Cpu>::new();
    let logits = model.forward(token_ids, &mut fastrand::Rng::with_seed(99), &mut tape);
    let loss = ops::cross_entropy(&logits, &targets, 3, 5, Some(&mut tape));
    let grads = backward(&tape, loss.id);

    let tok_emb_id = model.embedding.token_embedding.id;
    let full_grad = Cpu::to_vec(grads.get(&tok_emb_id).unwrap());
    let full_grad_norm: f64 = full_grad
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum();

    assert!(
        full_grad_norm > 0.0,
        "token_embedding gradient should be non-zero"
    );
    assert!(
        full_grad.iter().all(|v| v.is_finite()),
        "token_embedding gradient contains NaN/Inf"
    );

    // The token_embedding gradient should have non-zero entries beyond just the
    // rows that were looked up (indices 0, 2, 4), because the LM head (matmul
    // with token_embedding transposed) produces gradients for all vocab rows.
    // Check that rows 1 and 3 (not looked up) have non-zero gradient.
    let d_model = 4;
    let row1_grad: f64 = full_grad[d_model..2 * d_model]
        .iter()
        .map(|v| f64::from(*v).abs())
        .sum();
    let row3_grad: f64 = full_grad[3 * d_model..4 * d_model]
        .iter()
        .map(|v| f64::from(*v).abs())
        .sum();

    assert!(
        row1_grad > 1e-10,
        "Row 1 (not looked up) should have non-zero gradient from LM head, got {row1_grad}"
    );
    assert!(
        row3_grad > 1e-10,
        "Row 3 (not looked up) should have non-zero gradient from LM head, got {row3_grad}"
    );
}

#[test]
fn weight_tying_parameter_count_no_duplicate() {
    let config = Gpt2Config {
        vocab_size: 5,
        max_seq_len: 4,
        d_model: 4,
        n_heads: 2,
        n_layers: 1,
        d_ff: 8,
        dropout_rate: 0.0,
    };
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(config, &mut rng);

    let params = model.parameters();
    let param_ids: Vec<_> = params.iter().map(|p| p.id).collect();

    // The token_embedding should appear exactly once in parameters
    let tok_emb_id = model.embedding.token_embedding.id;
    let count = param_ids.iter().filter(|&&id| id == tok_emb_id).count();
    assert_eq!(count, 1, "token_embedding should appear exactly once in parameters (weight tying means LM head has no separate param)");
}
