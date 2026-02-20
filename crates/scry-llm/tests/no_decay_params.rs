//! Tests for no-decay parameter groups in `AdamW` optimizer.

use std::collections::HashSet;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::backend::DeviceBackend;
use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};
use scry_llm::optim::adamw::{AdamW, AdamWConfig};
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::TensorId;

type Cpu = CpuBackend;

fn tiny_config() -> Gpt2Config {
    Gpt2Config {
        vocab_size: 10,
        max_seq_len: 16,
        d_model: 8,
        n_heads: 2,
        n_layers: 2,
        d_ff: 16,
        dropout_rate: 0.0,
    }
}

#[test]
fn no_decay_ids_correct_set() {
    let config = tiny_config();
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(config.clone(), &mut rng);
    let no_decay = model.no_decay_ids();

    // Per block: ln1.gamma, ln1.beta, qkv_bias, proj_bias, ln2.gamma, ln2.beta, fc1.bias, fc2.bias = 8
    // Plus ln_f.gamma, ln_f.beta = 2
    // Total = 8 * n_layers + 2
    let expected_count = 8 * config.n_layers + 2;
    assert_eq!(no_decay.len(), expected_count, "no_decay set size mismatch");

    // Check specific IDs are present
    for block in &model.blocks {
        assert!(no_decay.contains(&block.ln1.gamma.id));
        assert!(no_decay.contains(&block.ln1.beta.id));
        assert!(no_decay.contains(&block.ln2.gamma.id));
        assert!(no_decay.contains(&block.ln2.beta.id));
        assert!(no_decay.contains(&block.attn.qkv_bias.id));
        assert!(no_decay.contains(&block.attn.proj_bias.id));
        assert!(no_decay.contains(&block.mlp.fc1.bias.id));
        assert!(no_decay.contains(&block.mlp.fc2.bias.id));
    }
    assert!(no_decay.contains(&model.ln_f.gamma.id));
    assert!(no_decay.contains(&model.ln_f.beta.id));

    // Weights should NOT be in no_decay
    assert!(!no_decay.contains(&model.embedding.token_embedding.id));
    assert!(!no_decay.contains(&model.embedding.position_embedding.id));
    for block in &model.blocks {
        assert!(!no_decay.contains(&block.attn.qkv_weight.id));
        assert!(!no_decay.contains(&block.attn.proj_weight.id));
        assert!(!no_decay.contains(&block.mlp.fc1.weight.id));
        assert!(!no_decay.contains(&block.mlp.fc2.weight.id));
    }
}

#[test]
fn adamw_step_respects_no_decay() {
    // Create two identical params, apply AdamW with one in no_decay set
    let shape = Shape::new(&[4]);
    let initial = vec![1.0, 2.0, 3.0, 4.0];
    let grad_data = vec![0.1, 0.1, 0.1, 0.1];

    // Param with decay
    let mut param_decay = scry_llm::tensor::Tensor::<Cpu>::from_vec(initial.clone(), shape.clone());
    let decay_id = param_decay.id;

    // Param without decay
    let mut param_no_decay = scry_llm::tensor::Tensor::<Cpu>::from_vec(initial.clone(), shape.clone());
    let no_decay_id = param_no_decay.id;

    let mut grad_map = std::collections::HashMap::new();
    grad_map.insert(decay_id, Cpu::from_vec(grad_data.clone(), &shape));
    grad_map.insert(no_decay_id, Cpu::from_vec(grad_data, &shape));

    let config = AdamWConfig {
        weight_decay: 0.1,
        ..AdamWConfig::default()
    };

    let no_decay_set: HashSet<TensorId> = [no_decay_id].into();

    let mut optimizer = AdamW::<Cpu>::new(config);
    let decay_shape = param_decay.shape.clone();
    let no_decay_shape = param_no_decay.shape.clone();
    let mut params = vec![
        (decay_id, param_decay.data_mut(), &decay_shape),
        (no_decay_id, param_no_decay.data_mut(), &no_decay_shape),
    ];
    optimizer.step(&mut params, &grad_map, &no_decay_set);
    drop(params);

    let after_decay = Cpu::to_vec(&param_decay.data);
    let after_no_decay = Cpu::to_vec(&param_no_decay.data);

    // Both should have changed from initial
    assert!(after_decay.iter().zip(initial.iter()).any(|(a, b)| (a - b).abs() > 1e-10));
    assert!(after_no_decay.iter().zip(initial.iter()).any(|(a, b)| (a - b).abs() > 1e-10));

    // The decay param should differ from the no-decay param (weight decay effect)
    assert!(
        after_decay.iter().zip(after_no_decay.iter()).any(|(a, b)| (a - b).abs() > 1e-10),
        "decay and no-decay params should differ"
    );
}

#[test]
fn no_decay_count_matches_expected() {
    // Test with different model configs
    for n_layers in [1, 3, 6] {
        let config = Gpt2Config {
            n_layers,
            ..tiny_config()
        };
        let mut rng = fastrand::Rng::with_seed(42);
        let model = Gpt2Model::<Cpu>::new(config, &mut rng);
        let no_decay = model.no_decay_ids();
        let expected = 8 * n_layers + 2;
        assert_eq!(
            no_decay.len(),
            expected,
            "n_layers={n_layers}: expected {expected} no-decay params, got {}",
            no_decay.len()
        );
    }
}
