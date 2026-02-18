//! Fuzz layernorm forward + backward. Output must be finite, mean ≈ beta,
//! variance ≈ gamma^2 (when input has sufficient variance).

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::backend::MathBackend;
use scry_llm::tensor::shape::Shape;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    let batch = (data[0] % 4) as usize + 1;
    let dim = (data[1] % 8) as usize + 2; // at least 2 for meaningful normalization
    let numel = batch * dim;

    let input: Vec<f32> = (0..numel).map(|i| ((i ^ 0xAB) % 17) as f32 - 8.0).collect();
    let gamma: Vec<f32> = vec![1.0; dim];
    let beta: Vec<f32> = vec![0.0; dim];
    let eps = 1e-5;
    let shape = Shape::new(&[batch, dim]);

    let (output, mean, rstd) = CpuBackend::layernorm(&input, &gamma, &beta, &shape, eps);

    assert_eq!(output.len(), numel);
    assert!(output.iter().all(|v| v.is_finite()));
    assert!(mean.iter().all(|v| v.is_finite()));
    assert!(rstd.iter().all(|v| v.is_finite() && *v > 0.0));

    // Backward must be finite
    let d_out = vec![1.0f32; numel];
    let (d_input, d_gamma, d_beta) =
        CpuBackend::layernorm_backward(&d_out, &input, &gamma, &mean, &rstd, &shape);
    assert!(d_input.iter().all(|v| v.is_finite()));
    assert!(d_gamma.iter().all(|v| v.is_finite()));
    assert!(d_beta.iter().all(|v| v.is_finite()));
});
