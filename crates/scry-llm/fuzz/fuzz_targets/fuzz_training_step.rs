//! Fuzz a single training step: tiny model, arbitrary token IDs.
//! Must never panic, produce NaN loss, or NaN in params.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::backend::DeviceBackend;
use scry_llm::data::Batch;
use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};
use scry_llm::nn::Module;
use scry_llm::training::{Trainer, TrainingConfig};

type Cpu = CpuBackend;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let vocab_size = 5;
    let seq_len = 3;
    let batch_size = 1;

    // Extract token IDs from fuzz input (clamped to vocab)
    let input_ids: Vec<usize> = data[..seq_len.min(data.len())]
        .iter()
        .map(|&b| (b as usize) % vocab_size)
        .collect();
    if input_ids.len() < seq_len {
        return;
    }
    let targets: Vec<usize> = data[seq_len..2 * seq_len.min(data.len())]
        .iter()
        .map(|&b| (b as usize) % vocab_size)
        .collect();
    if targets.len() < seq_len {
        return;
    }

    let model_config = Gpt2Config {
        vocab_size,
        max_seq_len: 8,
        d_model: 4,
        n_heads: 2,
        n_layers: 1,
        d_ff: 8,
        dropout_rate: 0.0,
    };
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(model_config.clone(), &mut rng);

    let config = TrainingConfig {
        batch_size,
        seq_len,
        total_steps: 10,
        warmup_steps: 2,
        peak_lr: 3e-4,
        min_lr: 1e-5,
        grad_accum_steps: 1,
        max_grad_norm: 1.0,
        log_interval: 100,
        eval_interval: 0,
        checkpoint_interval: 0,
        checkpoint_dir: std::path::PathBuf::from("/tmp"),
        seed: 42,
        use_checkpointing: false,
        checkpoint_every: 4,
    };

    let mut trainer = Trainer::<Cpu>::new(model, model_config, config);

    let batch = Batch {
        input_ids,
        targets,
        batch_size,
        seq_len,
    };

    let metrics = trainer.train_step(&[batch]);
    assert!(metrics.loss.is_finite(), "loss is not finite");

    // Check all params are finite
    for param in trainer.model.parameters() {
        let data = Cpu::to_vec(&param.data);
        assert!(
            data.iter().all(|v| v.is_finite()),
            "NaN/Inf in model params after train_step"
        );
    }
});
