#![cfg(feature = "bf16")]

//! End-to-end BF16 training convergence test.
//! Verifies that loss decreases over 10 steps in bf16 mode.

use scry_llm::backend::cuda::{init_gpu_bf16, CudaBackend};
use scry_llm::data::Batch;
use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};
use scry_llm::training::{Trainer, TrainingConfig};

fn init() {
    init_gpu_bf16(0);
}

#[test]
fn bf16_training_loss_decreases() {
    init();

    let model_config = Gpt2Config {
        vocab_size: 32,
        max_seq_len: 16,
        d_model: 32,
        n_heads: 2,
        n_layers: 1,
        d_ff: 64,
        dropout_rate: 0.0,
    };

    let training_config = TrainingConfig {
        batch_size: 2,
        seq_len: 8,
        total_steps: 10,
        warmup_steps: 0,
        peak_lr: 1e-3,
        min_lr: 1e-3,
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

    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<CudaBackend>::new(model_config.clone(), &mut rng);
    let mut trainer = Trainer::<CudaBackend>::new(model, model_config.clone(), training_config);

    // Create a simple repeating pattern for the model to learn
    let batch_size = 2;
    let seq_len = 8;
    let mut data_rng = fastrand::Rng::with_seed(123);
    let make_batch = |rng: &mut fastrand::Rng| {
        let mut input_ids = Vec::with_capacity(batch_size * seq_len);
        let mut targets = Vec::with_capacity(batch_size * seq_len);
        for _ in 0..batch_size {
            for _ in 0..seq_len {
                let tok = rng.usize(0..model_config.vocab_size);
                let next = (tok + 1) % model_config.vocab_size;
                input_ids.push(tok);
                targets.push(next);
            }
        }
        Batch {
            input_ids,
            targets,
            batch_size,
            seq_len,
        }
    };

    let mut losses = Vec::new();
    for _ in 0..10 {
        let batch = make_batch(&mut data_rng);
        let metrics = trainer.train_step(&[batch]);
        assert!(
            !metrics.loss.is_nan(),
            "BF16 training produced NaN loss at step {}",
            trainer.step
        );
        assert!(
            !metrics.loss.is_infinite(),
            "BF16 training produced Inf loss at step {}",
            trainer.step
        );
        losses.push(metrics.loss);
    }

    // Loss should decrease: last 3 average should be lower than first 3 average
    let first_3_avg: f32 = losses[..3].iter().sum::<f32>() / 3.0;
    let last_3_avg: f32 = losses[7..].iter().sum::<f32>() / 3.0;

    eprintln!("BF16 training losses: {losses:?}");
    eprintln!("First 3 avg: {first_3_avg:.4}, Last 3 avg: {last_3_avg:.4}");

    assert!(
        last_3_avg < first_3_avg,
        "BF16 training did not converge: first_3_avg={first_3_avg:.4}, last_3_avg={last_3_avg:.4}"
    );
}
