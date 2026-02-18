//! Fuzz the full autograd pipeline: build a random compute graph from fuzz bytes,
//! run forward + backward. Must never panic, produce NaN, or Inf.

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
    if data.len() < 4 {
        return;
    }

    let m = (data[0] % 4) as usize + 1; // 1..=4
    let k = (data[1] % 4) as usize + 1;
    let n = (data[2] % 4) as usize + 1;
    let op_selector = data[3];

    // Deterministic data scaled to prevent overflow
    let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();

    let mut tape = GradTape::<Cpu>::new();

    let a = Tensor::<Cpu>::from_vec(a_data, Shape::new(&[m, k]));
    let b = Tensor::<Cpu>::from_vec(b_data, Shape::new(&[k, n]));

    // Select chain of ops from fuzz byte
    let loss = match op_selector % 6 {
        0 => {
            // matmul -> sum
            let c = ops::matmul(&a, &b, m, k, n, false, false, Some(&mut tape));
            ops::sum(&c, Some(&mut tape))
        }
        1 => {
            // matmul -> gelu -> sum
            let c = ops::matmul(&a, &b, m, k, n, false, false, Some(&mut tape));
            let g = ops::gelu(&c, Some(&mut tape));
            ops::sum(&g, Some(&mut tape))
        }
        2 => {
            // matmul -> softmax -> sum (but route through matmul for non-trivial grad)
            let c = ops::matmul(&a, &b, m, k, n, false, false, Some(&mut tape));
            let sm = ops::softmax(&c, Some(&mut tape));
            // Need another matmul to get non-trivial gradient through softmax
            let w = Tensor::<Cpu>::from_vec(vec![1.0; n], Shape::new(&[n, 1]));
            let out = ops::matmul(&sm, &w, m, n, 1, false, false, Some(&mut tape));
            ops::sum(&out, Some(&mut tape))
        }
        3 => {
            // layernorm -> matmul -> sum
            let d = k;
            let gamma = Tensor::<Cpu>::from_vec(vec![1.0; d], Shape::new(&[d]));
            let beta = Tensor::<Cpu>::from_vec(vec![0.0; d], Shape::new(&[d]));
            let ln = ops::layernorm(&a, &gamma, &beta, 1e-5, Some(&mut tape));
            let c = ops::matmul(&ln, &b, m, k, n, false, false, Some(&mut tape));
            ops::sum(&c, Some(&mut tape))
        }
        4 => {
            // embedding -> sum
            let vocab = m.max(4);
            let dim = k;
            let weight_data: Vec<f32> = (0..vocab * dim)
                .map(|i| ((i % 9) as f32 - 4.0) * 0.1)
                .collect();
            let weight = Tensor::<Cpu>::from_vec(weight_data, Shape::new(&[vocab, dim]));
            let indices: Vec<usize> = (0..n).map(|i| i % vocab).collect();
            let out = ops::embedding(&weight, &indices, vocab, dim, Some(&mut tape));
            ops::sum(&out, Some(&mut tape))
        }
        _ => {
            // embedding -> attention -> sum
            let d_model = k.max(2); // need at least 2 for 1 head
            let n_heads = 1;
            let d_head = d_model;
            let vocab = m.max(4);
            let seq = n.min(3).max(1);

            let weight_data: Vec<f32> = (0..vocab * d_model)
                .map(|i| ((i % 9) as f32 - 4.0) * 0.01)
                .collect();
            let weight = Tensor::<Cpu>::from_vec(weight_data, Shape::new(&[vocab, d_model]));
            let indices: Vec<usize> = (0..seq).map(|i| i % vocab).collect();
            let emb = ops::embedding(&weight, &indices, vocab, d_model, Some(&mut tape));

            let qkv_w = Tensor::<Cpu>::from_vec(
                vec![0.01; d_model * 3 * d_model],
                Shape::new(&[d_model, 3 * d_model]),
            );
            let qkv_b = Tensor::<Cpu>::from_vec(vec![0.0; 3 * d_model], Shape::new(&[3 * d_model]));
            let proj_w = Tensor::<Cpu>::from_vec(
                vec![0.01; d_model * d_model],
                Shape::new(&[d_model, d_model]),
            );
            let proj_b = Tensor::<Cpu>::from_vec(vec![0.0; d_model], Shape::new(&[d_model]));

            let attn_out = ops::attention(
                &emb, &qkv_w, &qkv_b, &proj_w, &proj_b,
                n_heads, d_model, d_head,
                Some(&mut tape),
            );
            ops::sum(&attn_out, Some(&mut tape))
        }
    };

    let loss_val = loss.to_vec()[0];
    assert!(loss_val.is_finite(), "loss is not finite: {loss_val}");

    let grads = backward(&tape, loss.id);

    // All gradients must be finite
    for (_, grad_data) in &grads {
        assert!(
            grad_data.iter().all(|v| v.is_finite()),
            "gradient contains NaN/Inf"
        );
    }
});
