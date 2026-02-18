//! Training convergence test for a tiny GPT-2 model on synthetic data.
//! Gate: loss decreases to < 0.01 within 500 steps.

use scry_llm::autograd::backward::backward;
use scry_llm::autograd::ops;
use scry_llm::autograd::GradTape;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};
use scry_llm::nn::Module;
use scry_llm::optim::adamw::{AdamW, AdamWConfig};
type Cpu = CpuBackend;

#[test]
fn tiny_gpt2_training_converges() {
    let config = Gpt2Config {
        vocab_size: 10,
        max_seq_len: 16,
        d_model: 32,
        n_heads: 2,
        n_layers: 2,
        d_ff: 64,
    };

    let mut rng = fastrand::Rng::with_seed(42);
    let mut model = Gpt2Model::<Cpu>::new(config.clone(), &mut rng);

    let mut optimizer = AdamW::<Cpu>::new(AdamWConfig {
        lr: 3e-4,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0, // tiny model, no regularization needed
    });

    // Synthetic data: repeated patterns [0,1,2,3,4,5,6,7,8,9] x batch
    let n_sequences = 20;
    let seq_len = 8;
    let mut sequences: Vec<Vec<usize>> = Vec::new();
    for _ in 0..n_sequences {
        let start = rng.usize(0..config.vocab_size);
        let seq: Vec<usize> = (0..seq_len)
            .map(|i| (start + i) % config.vocab_size)
            .collect();
        sequences.push(seq);
    }

    let n_steps = 500;
    let mut initial_loss = 0.0f32;
    let mut final_loss = 0.0f32;

    for step in 0..n_steps {
        let seq_idx = step % n_sequences;
        let input_ids = &sequences[seq_idx][..seq_len - 1];
        let targets: Vec<usize> = sequences[seq_idx][1..seq_len].to_vec();

        let mut tape = GradTape::<Cpu>::new();

        // Forward
        let logits = model.forward(input_ids, &mut tape);

        // Loss
        let batch = input_ids.len();
        let loss = ops::cross_entropy(&logits, &targets, batch, config.vocab_size, Some(&mut tape));

        let loss_val = loss.to_vec()[0];

        if step == 0 {
            initial_loss = loss_val;
            println!("  step {step}: loss = {loss_val:.4}");
        }
        if step == n_steps - 1 {
            final_loss = loss_val;
            println!("  step {step}: loss = {loss_val:.4}");
        }
        if step % 100 == 0 && step > 0 {
            println!("  step {step}: loss = {loss_val:.4}");
        }

        // Backward
        let grads = backward(&tape, loss.id);

        // Optimizer step
        let mut params: Vec<_> = model
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

        optimizer.step(&mut param_refs, &grads);
    }

    println!("  initial_loss = {initial_loss:.4}, final_loss = {final_loss:.4}");
    assert!(
        final_loss < initial_loss,
        "Loss did not decrease: {initial_loss:.4} -> {final_loss:.4}"
    );
    assert!(
        final_loss < 0.5,
        "Final loss {final_loss:.4} too high (expected < 0.5 on synthetic data)"
    );
}
