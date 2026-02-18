//! Fuzz the attention forward+backward op with random dimensions and weights.
//! Must never panic, produce NaN, or Inf.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_llm::autograd::backward::backward;
use scry_llm::autograd::ops;
use scry_llm::autograd::GradTape;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

type Cpu = CpuBackend;

fuzz_target!(|data: &[u8]| {
    if data.len() < 6 {
        return;
    }

    let seq_len = (data[0] % 4) as usize + 1; // 1..=4
    let d_model = if data[1] % 2 == 0 { 2 } else { 4 };
    let n_heads = if d_model == 2 { 1 } else { (data[2] % 2) as usize + 1 }; // 1 or 2
    let d_head = d_model / n_heads;

    // Build small weights from fuzz bytes, scaled to prevent overflow
    let make_data = |len: usize, offset: usize| -> Vec<f32> {
        (0..len)
            .map(|i| {
                let byte = data.get(offset + i).copied().unwrap_or(128);
                (byte as f32 - 128.0) * 0.001
            })
            .collect()
    };

    let input_data = make_data(seq_len * d_model, 6);
    let qkv_w_data = make_data(d_model * 3 * d_model, 6 + seq_len * d_model);
    let qkv_b_data = vec![0.0f32; 3 * d_model];
    let proj_w_data = make_data(d_model * d_model, 6 + seq_len * d_model + d_model * 3 * d_model);
    let proj_b_data = vec![0.0f32; d_model];

    let mut tape = GradTape::<Cpu>::new();
    let input = Tensor::<Cpu>::from_vec(input_data, Shape::new(&[seq_len, d_model]));
    let qkv_w = Tensor::<Cpu>::from_vec(qkv_w_data, Shape::new(&[d_model, 3 * d_model]));
    let qkv_b = Tensor::<Cpu>::from_vec(qkv_b_data, Shape::new(&[3 * d_model]));
    let proj_w = Tensor::<Cpu>::from_vec(proj_w_data, Shape::new(&[d_model, d_model]));
    let proj_b = Tensor::<Cpu>::from_vec(proj_b_data, Shape::new(&[d_model]));

    let out = ops::attention(
        &input, &qkv_w, &qkv_b, &proj_w, &proj_b,
        n_heads, d_model, d_head,
        Some(&mut tape),
    );

    let out_vec = out.to_vec();
    assert!(out_vec.iter().all(|v| v.is_finite()), "attention output contains NaN/Inf");

    let loss = ops::sum(&out, Some(&mut tape));
    let loss_val = loss.to_vec()[0];
    assert!(loss_val.is_finite(), "attention loss is not finite: {loss_val}");

    let grads = backward(&tape, loss.id);
    for (_, grad_data) in &grads {
        assert!(
            grad_data.iter().all(|v| v.is_finite()),
            "attention gradient contains NaN/Inf"
        );
    }
});
