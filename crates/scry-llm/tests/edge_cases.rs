//! Numerical edge cases for the LLM modules.

use scry_llm::autograd::ops;
use scry_llm::autograd::GradTape;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};
#[allow(unused_imports)]
use scry_llm::tensor::shape::Shape;
#[allow(unused_imports)]
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
        dropout_rate: 0.0,
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
    let logits = model.forward(token_ids, &mut fastrand::Rng::with_seed(99), &mut tape);
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

#[test]
fn kv_cache_matches_full_inference() {
    let config = tiny_config();
    let model = tiny_model();

    let token_ids: Vec<usize> = vec![0, 3, 1, 5];

    // Full sequence inference
    let full_logits = model.forward_inference(&token_ids);
    let full_logits_vec = full_logits.to_vec();

    // Token-by-token with cache
    let mut cache = model.new_kv_cache();
    let mut last_logits_vec = Vec::new();
    for (pos, &tok) in token_ids.iter().enumerate() {
        let logits = model.forward_with_cache(tok, pos, &mut cache);
        last_logits_vec = logits.to_vec();
    }

    // The last token's logits from cached inference should match
    // the last row of full inference logits
    let vocab = config.vocab_size;
    let last_row_start = (token_ids.len() - 1) * vocab;
    let full_last_row = &full_logits_vec[last_row_start..last_row_start + vocab];

    assert_eq!(last_logits_vec.len(), vocab);
    let mut max_diff: f64 = 0.0;
    for i in 0..vocab {
        let diff = (f64::from(last_logits_vec[i]) - f64::from(full_last_row[i])).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    assert!(
        max_diff < 1e-4,
        "KV-cache logits differ from full inference by {max_diff:.2e} (expected < 1e-4)"
    );
}

#[test]
fn gradient_checkpointing_matches_standard() {
    use scry_llm::autograd::backward::backward;
    use scry_llm::backend::DeviceBackend;
    use scry_llm::nn::Module;

    let config = Gpt2Config {
        vocab_size: 8,
        max_seq_len: 6,
        d_model: 4,
        n_heads: 2,
        n_layers: 4,
        d_ff: 8,
        dropout_rate: 0.0, // deterministic for comparison
    };
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(config.clone(), &mut rng);

    let token_ids = &[0usize, 3, 1, 5];
    let targets = vec![3usize, 1, 5, 2];

    // Standard forward + backward
    let mut tape1 = GradTape::<Cpu>::new();
    let mut rng1 = fastrand::Rng::with_seed(99);
    let logits1 = model.forward(token_ids, &mut rng1, &mut tape1);
    let loss1 = ops::cross_entropy(&logits1, &targets, 4, config.vocab_size, Some(&mut tape1));
    let grads1 = backward(&tape1, loss1.id);

    // Checkpointed forward + backward (checkpoint every 2 blocks)
    let mut tape2 = GradTape::<Cpu>::new();
    let mut rng2 = fastrand::Rng::with_seed(99);
    let logits2 = model.forward_checkpointed(token_ids, 2, &mut rng2, &mut tape2);
    let loss2 = ops::cross_entropy(&logits2, &targets, 4, config.vocab_size, Some(&mut tape2));
    let grads2 = model.backward_checkpointed(&tape2, loss2.id);

    // Compare gradients for all parameters
    let params = model.parameters();
    for param in &params {
        let g1 = grads1.get(&param.id);
        let g2 = grads2.get(&param.id);
        match (g1, g2) {
            (Some(g1), Some(g2)) => {
                let v1 = Cpu::to_vec(g1);
                let v2 = Cpu::to_vec(g2);
                assert_eq!(v1.len(), v2.len());
                let mut max_diff: f64 = 0.0;
                for (a, b) in v1.iter().zip(v2.iter()) {
                    let diff = (f64::from(*a) - f64::from(*b)).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                }
                assert!(
                    max_diff < 1e-5,
                    "checkpointed gradient differs by {max_diff:.2e}"
                );
            }
            (None, None) => {} // both missing is ok
            _ => panic!("gradient present in one but not the other"),
        }
    }

    // Verify tape node count is reduced with checkpointing
    assert!(
        tape2.nodes.len() < tape1.nodes.len(),
        "checkpointed tape should have fewer nodes: {} vs {}",
        tape2.nodes.len(),
        tape1.nodes.len()
    );
}
