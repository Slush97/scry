//! Tests for embedding dropout in GPT-2 forward pass.

use scry_llm::autograd::backward::backward;
use scry_llm::autograd::ops;
use scry_llm::autograd::GradTape;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::backend::DeviceBackend;
use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};

type Cpu = CpuBackend;

fn tiny_config(dropout_rate: f32) -> Gpt2Config {
    Gpt2Config {
        vocab_size: 10,
        max_seq_len: 16,
        d_model: 8,
        n_heads: 2,
        n_layers: 2,
        d_ff: 16,
        dropout_rate,
    }
}

#[test]
fn forward_with_dropout_is_stochastic() {
    let config = tiny_config(0.5);
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(config, &mut rng);

    let token_ids = &[0, 1, 2, 3];

    let mut rng1 = fastrand::Rng::with_seed(100);
    let mut tape1 = GradTape::<Cpu>::new();
    let out1 = model.forward(token_ids, &mut rng1, &mut tape1);

    let mut rng2 = fastrand::Rng::with_seed(200);
    let mut tape2 = GradTape::<Cpu>::new();
    let out2 = model.forward(token_ids, &mut rng2, &mut tape2);

    let v1 = out1.to_vec();
    let v2 = out2.to_vec();

    // With different RNG seeds, dropout should produce different outputs
    let diff: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs()).sum();
    assert!(diff > 1e-6, "dropout outputs should differ with different RNG seeds, diff={diff}");
}

#[test]
fn inference_has_no_dropout() {
    let config = tiny_config(0.5);
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(config, &mut rng);

    let token_ids = &[0, 1, 2, 3];

    // Inference (no tape) should be deterministic regardless of dropout_rate
    let out1 = model.forward_inference(token_ids);
    let out2 = model.forward_inference(token_ids);

    let v1 = out1.to_vec();
    let v2 = out2.to_vec();

    assert_eq!(v1, v2, "inference outputs should be identical");
}

#[test]
fn grad_check_with_zero_dropout() {
    let config = tiny_config(0.0);
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(config.clone(), &mut rng);

    let token_ids = &[0, 1, 2];
    let targets = vec![1usize, 2, 3];

    let mut rng_fwd = fastrand::Rng::with_seed(99);
    let mut tape = GradTape::<Cpu>::new();
    let logits = model.forward(token_ids, &mut rng_fwd, &mut tape);
    let loss = ops::cross_entropy(&logits, &targets, 3, config.vocab_size, Some(&mut tape));
    let loss_val = loss.to_vec()[0];
    assert!(loss_val.is_finite(), "loss should be finite");
    assert!(loss_val > 0.0, "loss should be positive");

    let grads = backward(&tape, loss.id);
    assert!(grads.contains_key(&model.embedding.token_embedding.id));
    let de = Cpu::to_vec(grads.get(&model.embedding.token_embedding.id).unwrap());
    assert!(de.iter().all(|v: &f32| v.is_finite()), "embedding grads should be finite");
}
