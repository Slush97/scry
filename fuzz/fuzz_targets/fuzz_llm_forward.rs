//! Fuzz target: GPT-2 single-sequence forward + backward.
//!
//! Parses tiny model dims and token IDs from fuzz bytes, runs forward pass,
//! cross-entropy loss, and backward. Must not panic on any input.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_llm::autograd::backward::backward;
use scry_llm::autograd::ops;
use scry_llm::autograd::GradTape;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};

type Cpu = CpuBackend;

fuzz_target!(|data: &[u8]| {
    if data.len() < 6 {
        return;
    }

    let mut cursor = 0;

    let vocab = (data[cursor] % 7 + 2) as usize; // 2-8
    cursor += 1;
    let d_model_raw = (data[cursor] % 7 + 2) as usize; // 2-8
    cursor += 1;
    let d_model = d_model_raw + (d_model_raw % 2); // must be even
    let n_heads = if d_model >= 4 { 2 } else { 1 };
    let seq = (data[cursor] % 4 + 1) as usize; // 1-4
    cursor += 1;

    let config = Gpt2Config {
        vocab_size: vocab,
        max_seq_len: seq + 1,
        d_model,
        n_heads,
        n_layers: 1,
        d_ff: d_model * 2,
        dropout_rate: 0.0,
    };

    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(config.clone(), &mut rng);

    // Parse token IDs from remaining bytes
    let mut token_ids = Vec::with_capacity(seq);
    for _ in 0..seq {
        if cursor >= data.len() {
            token_ids.push(0);
        } else {
            token_ids.push((data[cursor] as usize) % vocab);
            cursor += 1;
        }
    }

    let mut tape = GradTape::<Cpu>::new();
    let logits = model.forward(&token_ids, &mut rng, &mut tape);

    // Targets from remaining bytes
    let targets: Vec<usize> = (0..seq)
        .map(|i| {
            if cursor + i < data.len() {
                (data[cursor + i] as usize) % vocab
            } else {
                (i + 1) % vocab
            }
        })
        .collect();

    let loss = ops::cross_entropy(&logits, &targets, seq, vocab, Some(&mut tape));
    let loss_val = loss.to_vec()[0];

    // NaN/Inf in loss is a soft failure, not a panic
    if loss_val.is_finite() {
        let _grads = backward(&tape, loss.id);
    }
});
