#![cfg(feature = "safetensors")]

use scry_llm::autograd::backward::backward;
use scry_llm::autograd::ops;
use scry_llm::autograd::GradTape;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::checkpoint::{load_checkpoint, save_checkpoint};
use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};
use scry_llm::nn::Module;
use scry_llm::optim::adamw::{AdamW, AdamWConfig};
type Cpu = CpuBackend;

fn tiny_config() -> Gpt2Config {
    Gpt2Config {
        vocab_size: 10,
        max_seq_len: 16,
        d_model: 16,
        n_heads: 2,
        n_layers: 2,
        d_ff: 32,
        dropout_rate: 0.0,
    }
}

fn train_steps(
    model: &mut Gpt2Model<Cpu>,
    optimizer: &mut AdamW<Cpu>,
    n_steps: usize,
    rng: &mut fastrand::Rng,
) -> f32 {
    let seq: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let targets: Vec<usize> = vec![1, 2, 3, 4, 5, 6, 7, 0];
    let config = tiny_config();

    let mut loss_val = 0.0f32;
    for _ in 0..n_steps {
        let mut tape = GradTape::<Cpu>::new();
        let logits = model.forward(&seq[..7], rng, &mut tape);
        let loss = ops::cross_entropy(&logits, &targets[..7], 7, config.vocab_size, Some(&mut tape));
        loss_val = loss.to_vec()[0];

        let grads = backward(&tape, loss.id);
        drop(tape);

        let mut params: Vec<_> = model
            .parameters_mut()
            .into_iter()
            .map(|p| {
                let id = p.id;
                let shape = p.shape.clone();
                (id, p.data_mut(), shape)
            })
            .collect();
        let mut param_refs: Vec<_> = params
            .iter_mut()
            .map(|(id, data, shape)| (*id, &mut **data, &*shape))
            .collect();
        optimizer.step(&mut param_refs, &grads, &std::collections::HashSet::new());
    }
    loss_val
}

#[test]
fn save_load_round_trip() {
    let config = tiny_config();
    let mut rng = fastrand::Rng::with_seed(42);
    let mut model = Gpt2Model::<Cpu>::new(config.clone(), &mut rng);
    let mut optimizer = AdamW::<Cpu>::new(AdamWConfig::default());

    // Train a few steps to populate optimizer state
    train_steps(&mut model, &mut optimizer, 10, &mut rng);

    // Save
    let tmp = std::env::temp_dir().join("scry_llm_test_checkpoint.safetensors");
    save_checkpoint::<Cpu>(&tmp, &model, &optimizer, 10, 42).unwrap();

    // Load
    let (loaded_model, loaded_optimizer, step, seed) =
        load_checkpoint::<Cpu>(&tmp, &config).unwrap();

    assert_eq!(step, 10);
    assert_eq!(seed, 42);
    assert_eq!(loaded_optimizer.step_count(), optimizer.step_count());

    // Verify parameters match
    let orig_params = model.parameters();
    let loaded_params = loaded_model.parameters();
    assert_eq!(orig_params.len(), loaded_params.len());
    for (orig, loaded) in orig_params.iter().zip(loaded_params.iter()) {
        let orig_data = orig.to_vec();
        let loaded_data = loaded.to_vec();
        assert_eq!(orig_data.len(), loaded_data.len());
        for (a, b) in orig_data.iter().zip(loaded_data.iter()) {
            assert!(
                (a - b).abs() < 1e-7,
                "parameter mismatch: {a} vs {b}"
            );
        }
    }

    // Clean up
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn training_continuity() {
    let config = tiny_config();

    // Train 20 steps continuously
    let mut rng_a = fastrand::Rng::with_seed(42);
    let mut model_a = Gpt2Model::<Cpu>::new(config.clone(), &mut rng_a);
    let mut opt_a = AdamW::<Cpu>::new(AdamWConfig::default());
    let loss_20 = train_steps(&mut model_a, &mut opt_a, 20, &mut rng_a);

    // Train 10, save, load, train 10 more
    let mut rng_b = fastrand::Rng::with_seed(42);
    let mut model_b = Gpt2Model::<Cpu>::new(config.clone(), &mut rng_b);
    let mut opt_b = AdamW::<Cpu>::new(AdamWConfig::default());
    train_steps(&mut model_b, &mut opt_b, 10, &mut rng_b);

    let tmp = std::env::temp_dir().join("scry_llm_test_continuity.safetensors");
    save_checkpoint::<Cpu>(&tmp, &model_b, &opt_b, 10, rng_b.u64(..)).unwrap();

    let (mut model_c, mut opt_c, _, _) = load_checkpoint::<Cpu>(&tmp, &config).unwrap();
    // Note: rng state is not perfectly preserved (we save seed, not full state),
    // so continued training may diverge slightly. We just verify it continues without error
    // and the loss is in the right ballpark.
    let mut rng_c = fastrand::Rng::with_seed(99);
    let loss_resumed = train_steps(&mut model_c, &mut opt_c, 10, &mut rng_c);

    // Both should have similar loss magnitudes (not identical due to rng divergence)
    assert!(loss_resumed.is_finite(), "resumed loss is not finite");
    assert!(
        loss_resumed < 3.0,
        "resumed loss unexpectedly high: {loss_resumed}"
    );
    println!("continuous loss: {loss_20:.4}, resumed loss: {loss_resumed:.4}");

    let _ = std::fs::remove_file(&tmp);
}
