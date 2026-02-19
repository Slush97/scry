//! Fuzz the AdamW optimizer: random param sizes, learning rates, and gradients.
//! Must never panic, produce NaN, or Inf.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::backend::DeviceBackend;
use scry_llm::optim::adamw::{AdamW, AdamWConfig};
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

type Cpu = CpuBackend;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let param_size = (data[0] % 16) as usize + 1; // 1..=16
    let lr = (data[1] as f32 + 1.0) * 1e-5; // small positive lr
    let beta1 = 0.85 + (data[2] as f32 / 255.0) * 0.14; // 0.85..=0.99
    let beta2 = 0.9 + (data[3] as f32 / 255.0) * 0.099; // 0.9..=0.999
    let wd = (data[4] as f32 / 255.0) * 0.2; // 0..=0.2
    let n_steps = (data[5] % 3) as usize + 1; // 1..=3

    let config = AdamWConfig {
        lr,
        beta1,
        beta2,
        eps: 1e-8,
        weight_decay: wd,
    };

    let shape = Shape::new(&[param_size]);

    // Build param and grad from fuzz bytes
    let param_data: Vec<f32> = (0..param_size)
        .map(|i| {
            let byte = data.get(8 + i).copied().unwrap_or(128);
            (byte as f32 - 128.0) * 0.01
        })
        .collect();
    let grad_data: Vec<f32> = (0..param_size)
        .map(|i| {
            let byte = data.get(8 + param_size + i).copied().unwrap_or(128);
            (byte as f32 - 128.0) * 0.01
        })
        .collect();

    let mut param = Tensor::<Cpu>::from_vec(param_data, shape.clone());
    let mut grad_map = std::collections::HashMap::new();
    grad_map.insert(param.id, Cpu::from_vec(grad_data, &shape));

    let mut optimizer = AdamW::<Cpu>::new(config);

    for _ in 0..n_steps {
        let mut params = vec![(param.id, &mut param.data, &param.shape)];
        optimizer.step(&mut params, &grad_map, &std::collections::HashSet::new());
    }

    let final_data = Cpu::to_vec(&param.data);
    assert!(
        final_data.iter().all(|v| v.is_finite()),
        "AdamW produced NaN/Inf"
    );
});
