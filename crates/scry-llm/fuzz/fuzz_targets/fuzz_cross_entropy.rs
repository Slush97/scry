//! Fuzz cross-entropy forward + backward. Loss must be non-negative and finite.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::backend::MathBackend;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }

    let batch = (data[0] % 4) as usize + 1; // 1..=4
    let vocab = (data[1] % 8) as usize + 2; // 2..=9
    let numel = batch * vocab;

    // Build logits — mix of fuzz and deterministic
    let logits: Vec<f32> = (0..numel)
        .map(|i| {
            let idx = 2 + i;
            if idx < data.len() {
                (data[idx] as f32 - 128.0) * 0.1
            } else {
                ((i % 7) as f32 - 3.0) * 0.5
            }
        })
        .collect();

    // Targets: valid class indices
    let targets: Vec<usize> = (0..batch)
        .map(|b| {
            let idx = 2 + numel + b;
            if idx < data.len() {
                data[idx] as usize % vocab
            } else {
                b % vocab
            }
        })
        .collect();

    let loss = CpuBackend::cross_entropy(&logits, &targets, batch, vocab)[0];
    assert!(loss.is_finite(), "cross-entropy loss is not finite: {loss}");
    assert!(loss >= 0.0, "cross-entropy loss is negative: {loss}");

    // Backward
    let d_logits = CpuBackend::cross_entropy_backward(&logits, &targets, batch, vocab, &vec![1.0f32]);
    assert_eq!(d_logits.len(), numel);
    assert!(d_logits.iter().all(|v| v.is_finite()));
});
