//! Fuzz target: Trainer::train_step with arbitrary batch data.
//!
//! Builds a tiny model and trainer, creates a synthetic batch from fuzz bytes,
//! and runs train_step. Must not panic (NaN in loss is ok — soft failure).

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::data::Batch;
use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};
use scry_llm::training::{Trainer, TrainingConfig};

type Cpu = CpuBackend;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let mut cursor = 0;

    let vocab = (data[cursor] % 5 + 3) as usize; // 3-7
    cursor += 1;
    let d_model_raw = (data[cursor] % 3 + 2) as usize; // 2-4
    cursor += 1;
    let d_model = d_model_raw + (d_model_raw % 2);
    let n_heads = if d_model >= 4 { 2 } else { 1 };
    let seq_len = (data[cursor] % 3 + 1) as usize; // 1-3
    cursor += 1;

    let model_config = Gpt2Config {
        vocab_size: vocab,
        max_seq_len: seq_len + 1,
        d_model,
        n_heads,
        n_layers: 1,
        d_ff: d_model * 2,
        dropout_rate: 0.0,
    };

    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(model_config.clone(), &mut rng);

    let training_config = TrainingConfig {
        batch_size: 1,
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
    };

    let mut trainer = Trainer::<Cpu>::new(model, model_config, training_config);

    // Parse batch token IDs from fuzz bytes
    let mut input_ids = Vec::with_capacity(seq_len);
    let mut targets = Vec::with_capacity(seq_len);
    for _ in 0..seq_len {
        if cursor < data.len() {
            input_ids.push((data[cursor] as usize) % vocab);
            cursor += 1;
        } else {
            input_ids.push(0);
        }
        if cursor < data.len() {
            targets.push((data[cursor] as usize) % vocab);
            cursor += 1;
        } else {
            targets.push(1 % vocab);
        }
    }

    let batch = Batch {
        input_ids,
        targets,
        batch_size: 1,
        seq_len,
    };

    let _metrics = trainer.train_step(&[batch]);
});
