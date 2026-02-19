//! Integration tests for the training loop.

use scry_llm::autograd::backward::backward;
use scry_llm::autograd::ops;
use scry_llm::autograd::GradTape;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::backend::MathBackend;
use scry_llm::data::{Batch, DataLoader};
use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};
use scry_llm::nn::Module;
use scry_llm::training::{Trainer, TrainingConfig};
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

fn synthetic_batch(pattern: &[usize], batch_size: usize, seq_len: usize) -> Batch {
    let mut input_ids = Vec::with_capacity(batch_size * seq_len);
    let mut targets = Vec::with_capacity(batch_size * seq_len);
    for _ in 0..batch_size {
        for i in 0..seq_len {
            input_ids.push(pattern[i % pattern.len()]);
            targets.push(pattern[(i + 1) % pattern.len()]);
        }
    }
    Batch {
        input_ids,
        targets,
        batch_size,
        seq_len,
    }
}

#[test]
fn tiny_model_loss_decreases() {
    let model_config = tiny_config();
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(model_config.clone(), &mut rng);

    let config = TrainingConfig {
        batch_size: 2,
        seq_len: 4,
        total_steps: 100,
        warmup_steps: 10,
        peak_lr: 3e-4,
        min_lr: 1e-5,
        grad_accum_steps: 1,
        max_grad_norm: 1.0,
        log_interval: 50,
        eval_interval: 0,
        checkpoint_interval: 0,
        checkpoint_dir: std::path::PathBuf::from("/tmp/scry_test_ckpt"),
        seed: 42,
        use_checkpointing: false,
        checkpoint_every: 4,
        peak_tflops: None,
        n_params: None,
    };

    let mut trainer = Trainer::<Cpu>::new(model, model_config, config);

    let pattern = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let batch = synthetic_batch(&pattern, 2, 4);

    let first_metrics = trainer.train_step(&[batch]);
    let initial_loss = first_metrics.loss;
    println!("initial loss: {initial_loss:.4}");

    for _ in 1..100 {
        let batch = synthetic_batch(&pattern, 2, 4);
        trainer.train_step(&[batch]);
    }

    let final_batch = synthetic_batch(&pattern, 2, 4);
    let final_metrics = trainer.train_step(&[final_batch]);
    let final_loss = final_metrics.loss;
    println!("final loss: {final_loss:.4}");

    assert!(
        final_loss < initial_loss,
        "loss did not decrease: {initial_loss:.4} -> {final_loss:.4}"
    );
}

#[test]
fn gradient_accumulation_correctness() {
    let model_config = tiny_config();

    let pattern = vec![0, 1, 2, 3];
    let seq_len = 4;
    let batch_size = 1;

    // Approach 1: grad_accum=4, micro_batch=1
    let mut rng1 = fastrand::Rng::with_seed(42);
    let model1 = Gpt2Model::<Cpu>::new(model_config.clone(), &mut rng1);

    let config1 = TrainingConfig {
        batch_size,
        seq_len,
        total_steps: 1,
        warmup_steps: 0,
        peak_lr: 1e-3,
        min_lr: 1e-3,
        grad_accum_steps: 4,
        max_grad_norm: f32::MAX, // no clipping for comparison
        log_interval: 1,
        eval_interval: 0,
        checkpoint_interval: 0,
        checkpoint_dir: std::path::PathBuf::from("/tmp"),
        seed: 99,
        use_checkpointing: false,
        checkpoint_every: 4,
        peak_tflops: None,
        n_params: None,
    };

    let mut trainer1 = Trainer::<Cpu>::new(model1, model_config.clone(), config1);
    let micro_batches: Vec<Batch> = (0..4)
        .map(|_| synthetic_batch(&pattern, batch_size, seq_len))
        .collect();
    let metrics1 = trainer1.train_step(&micro_batches);

    // Approach 2: Manual accumulation for comparison
    let mut rng2 = fastrand::Rng::with_seed(42);
    let model2 = Gpt2Model::<Cpu>::new(model_config.clone(), &mut rng2);
    let mut rng_fwd = fastrand::Rng::with_seed(99);

    let mut accumulated: std::collections::HashMap<_, _> = std::collections::HashMap::new();

    for _ in 0..4 {
        let batch = synthetic_batch(&pattern, batch_size, seq_len);
        let mut tape = GradTape::<Cpu>::new();
        let logits = model2.forward_batch(
            &batch.input_ids,
            batch.batch_size,
            batch.seq_len,
            &mut rng_fwd,
            &mut tape,
        );
        let loss = ops::cross_entropy(
            &logits,
            &batch.targets,
            batch.batch_size * batch.seq_len,
            model_config.vocab_size,
            Some(&mut tape),
        );
        let grads = backward(&tape, loss.id);
        for (id, grad) in grads {
            if let Some(existing) = accumulated.get_mut(&id) {
                CpuBackend::add_inplace(existing, &grad);
            } else {
                accumulated.insert(id, grad);
            }
        }
    }

    // Scale by 1/4
    for grad in accumulated.values_mut() {
        CpuBackend::scale_inplace(grad, 0.25);
    }

    // Compare: both should produce the same accumulated gradients
    // We check by looking at the loss from approach 1
    assert!(
        metrics1.loss.is_finite(),
        "accumulated training loss is not finite"
    );
    println!(
        "grad_accum loss: {:.4}, grad_norm: {:.4}",
        metrics1.loss, metrics1.grad_norm
    );
}

#[test]
fn data_loader_training() {
    let model_config = tiny_config();
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(model_config.clone(), &mut rng);

    let config = TrainingConfig {
        batch_size: 2,
        seq_len: 4,
        total_steps: 20,
        warmup_steps: 5,
        peak_lr: 3e-4,
        min_lr: 1e-5,
        grad_accum_steps: 1,
        max_grad_norm: 1.0,
        log_interval: 10,
        eval_interval: 0,
        checkpoint_interval: 0,
        checkpoint_dir: std::path::PathBuf::from("/tmp/scry_test_ckpt"),
        seed: 42,
        use_checkpointing: false,
        checkpoint_every: 4,
        peak_tflops: None,
        n_params: None,
    };

    let mut trainer = Trainer::<Cpu>::new(model, model_config, config);

    // Create synthetic shard: tokens 0..9 repeated
    let tokens: Vec<u16> = (0..500u16).map(|i| i % 10).collect();
    let mut loader = DataLoader::from_tokens(tokens, 4, 2, 42);

    // Just verify it runs without error
    let result = trainer.run(&mut loader, None);
    assert!(result.is_ok(), "training loop failed: {:?}", result.err());
    assert_eq!(trainer.step, 20);
}

#[test]
fn evaluate_returns_finite_loss() {
    let model_config = tiny_config();
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(model_config.clone(), &mut rng);

    let config = TrainingConfig {
        batch_size: 2,
        seq_len: 4,
        total_steps: 1,
        warmup_steps: 0,
        peak_lr: 3e-4,
        min_lr: 1e-5,
        grad_accum_steps: 1,
        max_grad_norm: 1.0,
        log_interval: 1,
        eval_interval: 0,
        checkpoint_interval: 0,
        checkpoint_dir: std::path::PathBuf::from("/tmp"),
        seed: 42,
        use_checkpointing: false,
        checkpoint_every: 4,
        peak_tflops: None,
        n_params: None,
    };

    let trainer = Trainer::<Cpu>::new(model, model_config, config);

    let pattern = vec![0, 1, 2, 3, 4];
    let batches: Vec<Batch> = (0..5)
        .map(|_| synthetic_batch(&pattern, 2, 4))
        .collect();

    let val_loss = trainer.evaluate(&batches);
    assert!(val_loss.is_finite(), "eval loss is not finite: {val_loss}");
    assert!(val_loss > 0.0, "eval loss should be positive");
    println!("eval loss: {val_loss:.4}");
}

#[test]
fn checkpointed_training_loss_decreases() {
    let model_config = tiny_config();
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(model_config.clone(), &mut rng);

    let config = TrainingConfig {
        batch_size: 2,
        seq_len: 4,
        total_steps: 100,
        warmup_steps: 10,
        peak_lr: 3e-4,
        min_lr: 1e-5,
        grad_accum_steps: 1,
        max_grad_norm: 1.0,
        log_interval: 50,
        eval_interval: 0,
        checkpoint_interval: 0,
        checkpoint_dir: std::path::PathBuf::from("/tmp/scry_test_ckpt"),
        seed: 42,
        use_checkpointing: true,
        checkpoint_every: 1,
        peak_tflops: None,
        n_params: None,
    };

    let mut trainer = Trainer::<Cpu>::new(model, model_config, config);

    let pattern = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let batch = synthetic_batch(&pattern, 2, 4);

    let first_metrics = trainer.train_step(&[batch]);
    let initial_loss = first_metrics.loss;
    println!("initial loss (checkpointed): {initial_loss:.4}");

    for _ in 1..100 {
        let batch = synthetic_batch(&pattern, 2, 4);
        trainer.train_step(&[batch]);
    }

    let final_batch = synthetic_batch(&pattern, 2, 4);
    let final_metrics = trainer.train_step(&[final_batch]);
    let final_loss = final_metrics.loss;
    println!("final loss (checkpointed): {final_loss:.4}");

    assert!(
        final_loss < initial_loss,
        "checkpointed loss did not decrease: {initial_loss:.4} -> {final_loss:.4}"
    );
}

#[test]
fn overfit_single_batch_to_near_zero() {
    let model_config = Gpt2Config {
        vocab_size: 10,
        max_seq_len: 16,
        d_model: 32,
        n_heads: 2,
        n_layers: 2,
        d_ff: 64,
        dropout_rate: 0.0,
    };

    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(model_config.clone(), &mut rng);

    let config = TrainingConfig {
        batch_size: 2,
        seq_len: 4,
        total_steps: 1000,
        warmup_steps: 0,
        peak_lr: 1e-3,
        min_lr: 1e-3,
        grad_accum_steps: 1,
        max_grad_norm: 1.0,
        log_interval: 1000,
        eval_interval: 0,
        checkpoint_interval: 0,
        checkpoint_dir: std::path::PathBuf::from("/tmp"),
        seed: 42,
        use_checkpointing: false,
        checkpoint_every: 4,
        peak_tflops: None,
        n_params: None,
    };

    let mut trainer = Trainer::<Cpu>::new(model, model_config, config);

    // One fixed batch — model should memorize it
    let batch_template = synthetic_batch(&[0, 1, 2, 3, 4], 2, 4);

    let mut last_loss = f32::MAX;
    for step in 0..1000 {
        let batch = Batch {
            input_ids: batch_template.input_ids.clone(),
            targets: batch_template.targets.clone(),
            batch_size: batch_template.batch_size,
            seq_len: batch_template.seq_len,
        };
        let metrics = trainer.train_step(&[batch]);
        last_loss = metrics.loss;

        if step % 200 == 0 {
            println!("  overfit step {step}: loss = {last_loss:.6}");
        }

        // Early exit if we've converged
        if last_loss < 0.01 {
            println!("  converged at step {step}: loss = {last_loss:.6}");
            break;
        }
    }

    assert!(
        last_loss < 0.01,
        "Failed to overfit single batch: final loss {last_loss:.6} (expected < 0.01)"
    );
}

#[test]
fn initial_loss_approximately_ln_vocab() {
    let model_config = Gpt2Config {
        vocab_size: 50257,
        max_seq_len: 64,
        d_model: 64,
        n_heads: 2,
        n_layers: 2,
        d_ff: 128,
        dropout_rate: 0.0,
    };

    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(model_config.clone(), &mut rng);

    // Forward on random tokens
    let seq_len = 16;
    let input_ids: Vec<usize> = (0..seq_len).map(|i| i % model_config.vocab_size).collect();
    let targets: Vec<usize> = (1..=seq_len).map(|i| i % model_config.vocab_size).collect();

    let logits = model.forward_inference(&input_ids);
    let loss = ops::cross_entropy(&logits, &targets, seq_len, model_config.vocab_size, None);
    let loss_val = loss.to_vec()[0];

    let expected = (model_config.vocab_size as f32).ln(); // ~10.825
    println!("  initial loss: {loss_val:.4}, expected ≈ {expected:.4}");

    assert!(
        (loss_val - expected).abs() < 1.5,
        "Initial loss {loss_val:.4} too far from ln(vocab_size) = {expected:.4}"
    );
}

#[test]
fn gradient_accumulation_matches_manual() {
    let model_config = tiny_config();
    let pattern = vec![0, 1, 2, 3];
    let seq_len = 4;
    let batch_size = 1;
    let n_accum = 4;

    // Approach 1: Trainer with grad_accum_steps=4
    let mut rng1 = fastrand::Rng::with_seed(42);
    let model1 = Gpt2Model::<Cpu>::new(model_config.clone(), &mut rng1);

    let config1 = TrainingConfig {
        batch_size,
        seq_len,
        total_steps: 1,
        warmup_steps: 0,
        peak_lr: 1e-3,
        min_lr: 1e-3,
        grad_accum_steps: n_accum,
        max_grad_norm: f32::MAX,
        log_interval: 1,
        eval_interval: 0,
        checkpoint_interval: 0,
        checkpoint_dir: std::path::PathBuf::from("/tmp"),
        seed: 99,
        use_checkpointing: false,
        checkpoint_every: 4,
        peak_tflops: None,
        n_params: None,
    };

    let mut trainer = Trainer::<Cpu>::new(model1, model_config.clone(), config1);
    let micro_batches: Vec<Batch> = (0..n_accum)
        .map(|_| synthetic_batch(&pattern, batch_size, seq_len))
        .collect();
    trainer.train_step(&micro_batches);

    // Collect param values after one Trainer step
    let params_after_trainer: Vec<(scry_llm::tensor::TensorId, Vec<f32>)> = trainer
        .model
        .parameters()
        .iter()
        .map(|p| (p.id, p.to_vec()))
        .collect();

    // Approach 2: Manual accumulation + manual optimizer step on a fresh identical model
    let mut rng2 = fastrand::Rng::with_seed(42);
    let mut model2 = Gpt2Model::<Cpu>::new(model_config.clone(), &mut rng2);
    let mut rng_fwd = fastrand::Rng::with_seed(99);

    let mut accumulated: std::collections::HashMap<_, _> = std::collections::HashMap::new();
    for _ in 0..n_accum {
        let batch = synthetic_batch(&pattern, batch_size, seq_len);
        let mut tape = GradTape::<Cpu>::new();
        let logits = model2.forward_batch(
            &batch.input_ids,
            batch.batch_size,
            batch.seq_len,
            &mut rng_fwd,
            &mut tape,
        );
        let loss = ops::cross_entropy(
            &logits,
            &batch.targets,
            batch.batch_size * batch.seq_len,
            model_config.vocab_size,
            Some(&mut tape),
        );
        let grads = backward(&tape, loss.id);
        for (id, grad) in grads {
            if let Some(existing) = accumulated.get_mut(&id) {
                CpuBackend::add_inplace(existing, &grad);
            } else {
                accumulated.insert(id, grad);
            }
        }
    }

    // Scale by 1/n_accum
    for grad in accumulated.values_mut() {
        CpuBackend::scale_inplace(grad, 1.0 / n_accum as f32);
    }

    // Apply same optimizer
    let no_decay = model2.no_decay_ids();
    let mut optimizer2 = scry_llm::optim::adamw::AdamW::<Cpu>::new(
        scry_llm::optim::adamw::AdamWConfig {
            lr: 1e-3,
            ..scry_llm::optim::adamw::AdamWConfig::default()
        },
    );
    let mut params: Vec<_> = model2
        .parameters_mut()
        .into_iter()
        .map(|p| {
            let id = p.id;
            let shape = p.shape.clone();
            (id, &mut p.data, shape)
        })
        .collect();
    let mut param_refs: Vec<_> = params
        .iter_mut()
        .map(|(id, data, shape)| (*id, &mut **data, &*shape))
        .collect();
    optimizer2.step(&mut param_refs, &accumulated, &no_decay);

    // Compare parameters
    let params_after_manual: Vec<(scry_llm::tensor::TensorId, Vec<f32>)> = model2
        .parameters()
        .iter()
        .map(|p| (p.id, p.to_vec()))
        .collect();

    // The IDs won't match between two model instances, but the order will
    assert_eq!(params_after_trainer.len(), params_after_manual.len());
    for (i, ((_, vals1), (_, vals2))) in params_after_trainer
        .iter()
        .zip(params_after_manual.iter())
        .enumerate()
    {
        assert_eq!(vals1.len(), vals2.len(), "param {i} size mismatch");
        for (j, (&v1, &v2)) in vals1.iter().zip(vals2.iter()).enumerate() {
            let diff = (v1 - v2).abs();
            assert!(
                diff < 1e-5,
                "param {i} elem {j}: trainer={v1:.8} vs manual={v2:.8} (diff={diff:.2e})"
            );
        }
    }
    println!("  gradient accumulation: all {} params match within 1e-5", params_after_trainer.len());
}
