//! Fuzz target: GPT-2 batched forward + backward.
//!
//! Like `fuzz_llm_forward` but tests the batched attention path with
//! variable batch sizes and arbitrary token inputs.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_llm::autograd::backward::backward;
use scry_llm::autograd::ops;
use scry_llm::autograd::GradTape;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};

type Cpu = CpuBackend;

fuzz_target!(|data: &[u8]| {
    if data.len() < 7 {
        return;
    }

    let mut cursor = 0;

    let vocab = (data[cursor] % 7 + 2) as usize; // 2-8
    cursor += 1;
    let d_model_raw = (data[cursor] % 7 + 2) as usize;
    cursor += 1;
    let d_model = d_model_raw + (d_model_raw % 2);
    let n_heads = if d_model >= 4 { 2 } else { 1 };
    let seq = (data[cursor] % 4 + 1) as usize; // 1-4
    cursor += 1;
    let batch_size = (data[cursor] % 3 + 1) as usize; // 1-3
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

    let total_tokens = batch_size * seq;
    let mut token_ids = Vec::with_capacity(total_tokens);
    for _ in 0..total_tokens {
        if cursor >= data.len() {
            token_ids.push(0);
        } else {
            token_ids.push((data[cursor] as usize) % vocab);
            cursor += 1;
        }
    }

    let mut tape = GradTape::<Cpu>::new();
    let logits = model.forward_batch(&token_ids, batch_size, seq, &mut rng, &mut tape);

    let targets: Vec<usize> = (0..total_tokens)
        .map(|i| {
            if cursor + i < data.len() {
                (data[cursor + i] as usize) % vocab
            } else {
                (i + 1) % vocab
            }
        })
        .collect();

    let loss = ops::cross_entropy(&logits, &targets, total_tokens, vocab, Some(&mut tape));
    let loss_val = loss.to_vec()[0];

    if loss_val.is_finite() {
        let _grads = backward(&tape, loss.id);
    }
});
