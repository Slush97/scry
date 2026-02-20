//! Tests designed to run under Miri for undefined behavior detection.
//! Keep dimensions tiny (Miri is ~100x slower than normal execution).
//!
//! Run with: `cargo +nightly miri test -p scry-llm --no-default-features --test miri_safe`

use scry_llm::autograd::backward::backward;
use scry_llm::autograd::ops;
use scry_llm::autograd::GradTape;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::backend::{DeviceBackend, MathBackend};
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

type Cpu = CpuBackend;

// ============================================================
// Tensor basics
// ============================================================

#[test]
fn miri_tensor_zeros_ones() {
    let z = Tensor::<Cpu>::zeros(Shape::new(&[2, 3]));
    assert_eq!(z.to_vec(), vec![0.0; 6]);
    assert_eq!(z.numel(), 6);

    let o = Tensor::<Cpu>::ones(Shape::new(&[3]));
    assert_eq!(o.to_vec(), vec![1.0; 3]);
}

#[test]
fn miri_tensor_from_vec_roundtrip() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::<Cpu>::from_vec(data.clone(), Shape::new(&[2, 3]));
    assert_eq!(t.to_vec(), data);
}

// ============================================================
// Shape operations
// ============================================================

#[test]
fn miri_shape_broadcast() {
    let a = Shape::new(&[2, 1]);
    let b = Shape::new(&[1, 3]);
    let c = Shape::broadcast(&a, &b).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(c.numel(), 6);
}

#[test]
fn miri_shape_strides() {
    let s = Shape::new(&[2, 3, 4]);
    let strides = s.strides();
    assert_eq!(&strides[..], &[12, 4, 1]);
}

#[test]
fn miri_shape_broadcast_strides() {
    let a = Shape::new(&[1, 3]);
    let target = Shape::new(&[2, 3]);
    let bs = a.broadcast_strides(&target);
    assert_eq!(&bs[..], &[0, 1]);
}

// ============================================================
// Backend ops (tiny sizes)
// ============================================================

#[test]
fn miri_matmul_2x2() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let c = CpuBackend::matmul(&a, &b, 2, 2, 2, false, false);
    assert_eq!(c.len(), 4);
    // [1*5+2*7, 1*6+2*8, 3*5+4*7, 3*6+4*8] = [19, 22, 43, 50]
    assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn miri_matmul_transposed() {
    let a = vec![1.0, 3.0, 2.0, 4.0]; // [K=2, M=2] for trans_a
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let c = CpuBackend::matmul(&a, &b, 2, 2, 2, true, false);
    assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn miri_add_broadcast() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![10.0, 20.0, 30.0];
    let a_shape = Shape::new(&[2, 3]);
    let b_shape = Shape::new(&[1, 3]);
    let out_shape = Shape::new(&[2, 3]);
    let c = CpuBackend::add(&a, &b, &a_shape, &b_shape, &out_shape);
    assert_eq!(c, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
fn miri_softmax() {
    let input = vec![1.0, 2.0, 3.0];
    let shape = Shape::new(&[1, 3]);
    let output = CpuBackend::softmax(&input, &shape);
    let sum: f64 = output.iter().map(|&v| f64::from(v)).sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn miri_layernorm() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0, 1.0];
    let beta = vec![0.0, 0.0];
    let shape = Shape::new(&[2, 2]);
    let (out, mean, rstd) = CpuBackend::layernorm(&input, &gamma, &beta, &shape, 1e-5);
    assert_eq!(out.len(), 4);
    assert_eq!(mean.len(), 2);
    assert_eq!(rstd.len(), 2);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn miri_gelu() {
    let input = vec![-1.0, 0.0, 1.0, 2.0];
    let output = CpuBackend::gelu(&input);
    assert_eq!(output.len(), 4);
    assert!((output[1]).abs() < 1e-6); // gelu(0) = 0
}

#[test]
fn miri_cross_entropy() {
    let logits = vec![10.0, -10.0, -10.0, -10.0, 10.0, -10.0];
    let targets = vec![0usize, 1];
    let loss = CpuBackend::cross_entropy(&logits, &targets, 2, 3)[0];
    assert!(loss.is_finite());
    assert!(loss < 0.01); // very confident predictions
}

#[test]
fn miri_embedding() {
    let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [3, 2]
    let indices = vec![0usize, 2, 1];
    let out = CpuBackend::embedding(&weight, &indices, 3, 2);
    assert_eq!(out, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0]);
}

#[test]
fn miri_sum() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let s = CpuBackend::sum(&input)[0];
    assert!((s - 10.0).abs() < 1e-6);
}

// ============================================================
// Autograd tape + backward (tiny graph)
// ============================================================

#[test]
fn miri_autograd_matmul_backward() {
    let mut tape = GradTape::<Cpu>::new();
    let a = Tensor::<Cpu>::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(&[2, 2]));
    let b = Tensor::<Cpu>::from_vec(vec![5.0, 6.0, 7.0, 8.0], Shape::new(&[2, 2]));
    let c = ops::matmul(&a, &b, 2, 2, 2, false, false, Some(&mut tape));
    let loss = ops::sum(&c, Some(&mut tape));

    let grads = backward(&tape, loss.id);
    let da = Cpu::to_vec(grads.get(&a.id).unwrap());
    let db = Cpu::to_vec(grads.get(&b.id).unwrap());
    assert_eq!(da.len(), 4);
    assert_eq!(db.len(), 4);
    assert!(da.iter().all(|v| v.is_finite()));
    assert!(db.iter().all(|v| v.is_finite()));
}

#[test]
fn miri_autograd_chain() {
    // layernorm -> matmul -> gelu -> sum
    let mut tape = GradTape::<Cpu>::new();
    let x = Tensor::<Cpu>::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(&[2, 2]));
    let g = Tensor::<Cpu>::from_vec(vec![1.0, 1.0], Shape::new(&[2]));
    let b = Tensor::<Cpu>::from_vec(vec![0.0, 0.0], Shape::new(&[2]));
    let w = Tensor::<Cpu>::from_vec(vec![0.1, 0.2, 0.3, 0.4], Shape::new(&[2, 2]));

    let ln = ops::layernorm(&x, &g, &b, 1e-5, Some(&mut tape));
    let mm = ops::matmul(&ln, &w, 2, 2, 2, false, false, Some(&mut tape));
    let act = ops::gelu(&mm, Some(&mut tape));
    let loss = ops::sum(&act, Some(&mut tape));

    let loss_val = loss.to_vec()[0];
    assert!(loss_val.is_finite());

    let grads = backward(&tape, loss.id);
    let dx = Cpu::to_vec(grads.get(&x.id).unwrap());
    assert!(dx.iter().all(|v| v.is_finite()));
}

#[test]
fn miri_autograd_embedding_backward() {
    let mut tape = GradTape::<Cpu>::new();
    let weight = Tensor::<Cpu>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(&[3, 2]));
    let indices = vec![0usize, 2, 0]; // duplicate index
    let out = ops::embedding(&weight, &indices, 3, 2, Some(&mut tape));
    let loss = ops::sum(&out, Some(&mut tape));

    let grads = backward(&tape, loss.id);
    let dw = Cpu::to_vec(grads.get(&weight.id).unwrap());
    assert_eq!(dw.len(), 6);
    // Row 0 looked up twice, so gradient = [2.0, 2.0]
    assert!((dw[0] - 2.0).abs() < 1e-6);
    assert!((dw[1] - 2.0).abs() < 1e-6);
    // Row 1 not looked up
    assert!((dw[2]).abs() < 1e-6);
    assert!((dw[3]).abs() < 1e-6);
    // Row 2 looked up once
    assert!((dw[4] - 1.0).abs() < 1e-6);
    assert!((dw[5] - 1.0).abs() < 1e-6);
}

#[test]
fn miri_autograd_cross_entropy_backward() {
    let mut tape = GradTape::<Cpu>::new();
    let logits = Tensor::<Cpu>::from_vec(vec![2.0, -1.0, 0.5, -0.5, 1.0, 0.0], Shape::new(&[2, 3]));
    let targets = vec![0usize, 2];
    let loss = ops::cross_entropy(&logits, &targets, 2, 3, Some(&mut tape));

    let loss_val = loss.to_vec()[0];
    assert!(loss_val.is_finite());
    assert!(loss_val > 0.0);

    let grads = backward(&tape, loss.id);
    let dl = Cpu::to_vec(grads.get(&logits.id).unwrap());
    assert_eq!(dl.len(), 6);
    assert!(dl.iter().all(|v| v.is_finite()));
}

// ============================================================
// Backward ops directly
// ============================================================

#[test]
fn miri_matmul_backward_all_transpose_combos() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // M=2, K=3
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // K=3, N=2
    let d_out = vec![1.0; 4]; // M=2, N=2

    for (ta, tb) in [(false, false), (true, false), (false, true), (true, true)] {
        let (da, db) = CpuBackend::matmul_backward(&d_out, &a, &b, 2, 3, 2, ta, tb);
        assert!(da.iter().all(|v| v.is_finite()));
        assert!(db.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn miri_add_backward_broadcast() {
    let d_out = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a_shape = Shape::new(&[2, 3]);
    let b_shape = Shape::new(&[1, 3]);
    let out_shape = Shape::new(&[2, 3]);
    let (da, db) = CpuBackend::add_backward(&d_out, &a_shape, &b_shape, &out_shape);
    assert_eq!(da.len(), 6);
    assert_eq!(db.len(), 3);
    // db should be sum over broadcast dim: [1+4, 2+5, 3+6] = [5, 7, 9]
    assert!((db[0] - 5.0).abs() < 1e-6);
    assert!((db[1] - 7.0).abs() < 1e-6);
    assert!((db[2] - 9.0).abs() < 1e-6);
}

#[test]
fn miri_softmax_backward() {
    let output = vec![0.25, 0.25, 0.25, 0.25];
    let d_out = vec![1.0, 0.0, 0.0, 0.0];
    let shape = Shape::new(&[1, 4]);
    let d_input = CpuBackend::softmax_backward(&d_out, &output, &shape);
    assert_eq!(d_input.len(), 4);
    assert!(d_input.iter().all(|v| v.is_finite()));
}

#[test]
fn miri_embedding_backward() {
    let d_out = vec![1.0, 2.0, 3.0, 4.0]; // 2 lookups, dim=2
    let indices = vec![1usize, 1]; // both same
    let dw = CpuBackend::embedding_backward(&d_out, &indices, 3, 2);
    assert_eq!(dw.len(), 6);
    // Row 1 gets both gradients accumulated: [1+3, 2+4] = [4, 6]
    assert!((dw[2] - 4.0).abs() < 1e-6);
    assert!((dw[3] - 6.0).abs() < 1e-6);
}

// ============================================================
// Phase 2: Attention, modules, optimizer (dims ≤ 8 for Miri)
// ============================================================

#[test]
fn miri_attention_forward_backward() {
    let d_model = 4;
    let n_heads = 2;
    let d_head = d_model / n_heads;
    let seq = 3;

    let mut tape = GradTape::<Cpu>::new();
    let input = Tensor::<Cpu>::from_vec(vec![0.1; seq * d_model], Shape::new(&[seq, d_model]));
    let qkv_w = Tensor::<Cpu>::from_vec(
        vec![0.02; d_model * 3 * d_model],
        Shape::new(&[d_model, 3 * d_model]),
    );
    let qkv_b = Tensor::<Cpu>::from_vec(vec![0.0; 3 * d_model], Shape::new(&[3 * d_model]));
    let proj_w = Tensor::<Cpu>::from_vec(
        vec![0.02; d_model * d_model],
        Shape::new(&[d_model, d_model]),
    );
    let proj_b = Tensor::<Cpu>::from_vec(vec![0.0; d_model], Shape::new(&[d_model]));

    let out = ops::attention(
        &input,
        &qkv_w,
        &qkv_b,
        &proj_w,
        &proj_b,
        n_heads,
        d_model,
        d_head,
        0.0,
        None,
        Some(&mut tape),
    );
    assert_eq!(out.shape.dims(), &[seq, d_model]);
    assert!(out.to_vec().iter().all(|v| v.is_finite()));

    let loss = ops::sum(&out, Some(&mut tape));
    let grads = backward(&tape, loss.id);
    let di = Cpu::to_vec(grads.get(&input.id).unwrap());
    assert_eq!(di.len(), seq * d_model);
    assert!(di.iter().all(|v| v.is_finite()));
}

#[test]
fn miri_linear_forward_backward() {
    use scry_llm::nn::linear::Linear;

    let mut rng = fastrand::Rng::with_seed(42);
    let linear = Linear::<Cpu>::new(4, 6, &mut rng);
    let mut tape = GradTape::<Cpu>::new();
    let input = Tensor::<Cpu>::from_vec(vec![0.1; 2 * 4], Shape::new(&[2, 4]));
    let out = linear.forward(&input, &mut tape);
    assert_eq!(out.shape.dims(), &[2, 6]);

    let loss = ops::sum(&out, Some(&mut tape));
    let grads = backward(&tape, loss.id);
    let di = Cpu::to_vec(grads.get(&input.id).unwrap());
    assert_eq!(di.len(), 8);
    assert!(di.iter().all(|v| v.is_finite()));
}

#[test]
fn miri_embedding_layer_forward() {
    use scry_llm::nn::embedding::EmbeddingLayer;

    let mut rng = fastrand::Rng::with_seed(42);
    let emb = EmbeddingLayer::<Cpu>::new(5, 8, 4, &mut rng);
    let mut tape = GradTape::<Cpu>::new();
    let token_ids = &[0, 3, 1];
    let out = emb.forward(token_ids, &mut tape);
    assert_eq!(out.shape.dims(), &[3, 4]);
    assert!(out.to_vec().iter().all(|v| v.is_finite()));
}

#[test]
fn miri_mlp_forward_backward() {
    use scry_llm::nn::mlp::Mlp;

    let mut rng = fastrand::Rng::with_seed(42);
    let mlp = Mlp::<Cpu>::new(4, 8, &mut rng);
    let mut tape = GradTape::<Cpu>::new();
    let input = Tensor::<Cpu>::from_vec(vec![0.1; 2 * 4], Shape::new(&[2, 4]));
    let out = mlp.forward(&input, &mut tape);
    assert_eq!(out.shape.dims(), &[2, 4]);

    let loss = ops::sum(&out, Some(&mut tape));
    let grads = backward(&tape, loss.id);
    let di = Cpu::to_vec(grads.get(&input.id).unwrap());
    assert_eq!(di.len(), 8);
    assert!(di.iter().all(|v| v.is_finite()));
}

#[test]
fn miri_transformer_block_forward() {
    use scry_llm::nn::transformer::TransformerBlock;

    let mut rng = fastrand::Rng::with_seed(42);
    let block = TransformerBlock::<Cpu>::new(4, 2, 8, &mut rng);
    let mut tape = GradTape::<Cpu>::new();
    let input = Tensor::<Cpu>::from_vec(vec![0.1; 3 * 4], Shape::new(&[3, 4]));
    let out = block.forward(&input, 0.0, &mut fastrand::Rng::with_seed(99), &mut tape);
    assert_eq!(out.shape.dims(), &[3, 4]);

    let loss = ops::sum(&out, Some(&mut tape));
    let grads = backward(&tape, loss.id);
    let di = Cpu::to_vec(grads.get(&input.id).unwrap());
    assert_eq!(di.len(), 12);
    assert!(di.iter().all(|v| v.is_finite()));
}

#[test]
fn miri_adamw_step() {
    use scry_llm::optim::adamw::{AdamW, AdamWConfig};

    let shape = Shape::new(&[2, 3]);
    let mut param = Tensor::<Cpu>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape.clone());
    let param_before = param.to_vec();

    let mut tape = GradTape::<Cpu>::new();
    let loss_input = Tensor::<Cpu>::from_vec(vec![0.5; 6], Shape::new(&[2, 3]));
    let loss = ops::sum(&loss_input, Some(&mut tape));
    let _grads = backward(&tape, loss.id);

    // Manually create a gradient for the param
    let mut grad_map = std::collections::HashMap::new();
    grad_map.insert(
        param.id,
        Cpu::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &shape),
    );

    let mut optimizer = AdamW::<Cpu>::new(AdamWConfig::default());
    let param_id = param.id;
    let param_shape = param.shape.clone();
    let mut params = vec![(param_id, param.data_mut(), &param_shape)];
    let no_decay = std::collections::HashSet::new();
    optimizer.step(&mut params, &grad_map, &no_decay);
    drop(params);

    let param_after = Cpu::to_vec(&param.data);
    assert!(param_after.iter().all(|v| v.is_finite()));
    // Params must have changed
    assert!(param_before
        .iter()
        .zip(param_after.iter())
        .any(|(a, b)| (a - b).abs() > 1e-10));
}

#[test]
fn miri_gpt2_tiny_forward_backward() {
    use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};

    let config = Gpt2Config {
        vocab_size: 5,
        max_seq_len: 4,
        d_model: 4,
        n_heads: 2,
        n_layers: 1,
        d_ff: 8,
        dropout_rate: 0.0,
    };
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(config, &mut rng);

    let token_ids = &[0, 2, 4];
    let mut tape = GradTape::<Cpu>::new();
    let logits = model.forward(token_ids, &mut rng, &mut tape);
    assert_eq!(logits.shape.dims(), &[3, 5]);
    assert!(logits.to_vec().iter().all(|v| v.is_finite()));

    let targets = vec![2usize, 4, 1];
    let loss = ops::cross_entropy(&logits, &targets, 3, 5, Some(&mut tape));
    let loss_val = loss.to_vec()[0];
    assert!(loss_val.is_finite());
    assert!(loss_val > 0.0);

    let grads = backward(&tape, loss.id);
    // Check that at least the token embedding got a gradient
    assert!(grads.contains_key(&model.embedding.token_embedding.id));
    let de = Cpu::to_vec(grads.get(&model.embedding.token_embedding.id).unwrap());
    assert!(de.iter().all(|v| v.is_finite()));
}

// ============================================================
// Phase 3: Scheduler, DataLoader, Training (dims ≤ 8 for Miri)
// ============================================================

#[test]
fn miri_cosine_scheduler() {
    use scry_llm::optim::scheduler::CosineScheduler;

    let warmup = 10;
    let total = 100;
    let peak_lr = 3e-4;
    let min_lr = 1e-5;
    let sched = CosineScheduler::new(warmup, total, peak_lr, min_lr);

    // Step 0: should be 0 (linear warmup from 0)
    let lr0 = sched.get_lr(0);
    assert!(lr0.is_finite());
    assert!(lr0 >= 0.0);
    assert!(lr0 <= peak_lr);

    // Warmup midpoint
    let lr_mid_warmup = sched.get_lr(warmup / 2);
    assert!(lr_mid_warmup.is_finite());
    assert!(lr_mid_warmup >= min_lr);
    assert!(lr_mid_warmup <= peak_lr);

    // Peak (at warmup boundary)
    let lr_peak = sched.get_lr(warmup);
    assert!(lr_peak.is_finite());
    assert!((lr_peak - peak_lr).abs() < 1e-6);

    // Decay midpoint
    let lr_decay_mid = sched.get_lr((warmup + total) / 2);
    assert!(lr_decay_mid.is_finite());
    assert!(lr_decay_mid >= min_lr);
    assert!(lr_decay_mid <= peak_lr);

    // At total_steps: should be min_lr
    let lr_end = sched.get_lr(total);
    assert!(lr_end.is_finite());
    assert!((lr_end - min_lr).abs() < 1e-6);
}

#[test]
fn miri_data_loader_from_tokens() {
    use scry_llm::data::DataLoader;

    let tokens: Vec<u16> = (0..20).collect();
    let seq_len = 3;
    let batch_size = 2;
    let mut loader = DataLoader::from_tokens(tokens, seq_len, batch_size, 42);

    // First batch
    let b1 = loader.next_batch().unwrap();
    assert_eq!(b1.batch_size, batch_size);
    assert_eq!(b1.seq_len, seq_len);
    assert_eq!(b1.input_ids.len(), batch_size * seq_len);
    assert_eq!(b1.targets.len(), batch_size * seq_len);

    // Verify input/target offset by 1
    // First sequence: tokens[0..3] input, tokens[1..4] target
    assert_eq!(b1.input_ids[0], 0);
    assert_eq!(b1.targets[0], 1);
    assert_eq!(b1.input_ids[1], 1);
    assert_eq!(b1.targets[1], 2);

    // Second batch
    let b2 = loader.next_batch().unwrap();
    assert_eq!(b2.input_ids.len(), batch_size * seq_len);
    assert_eq!(b2.targets.len(), batch_size * seq_len);
}

#[test]
fn miri_data_loader_wrap() {
    use scry_llm::data::DataLoader;

    // Only 8 tokens, seq_len=2, batch_size=1 → need 3 tokens per batch
    // Can fit 2 batches (6 tokens consumed), then must wrap
    let tokens: Vec<u16> = (0..8).collect();
    let mut loader = DataLoader::from_tokens(tokens, 2, 1, 42);

    // Exhaust: ceil(8/3)=2 full batches, then wrap
    for _ in 0..5 {
        let batch = loader.next_batch().unwrap();
        assert_eq!(batch.input_ids.len(), 2);
        assert_eq!(batch.targets.len(), 2);
    }
}

#[test]
fn miri_batched_attention_forward_backward() {
    use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};
    use scry_llm::nn::Module;

    let config = Gpt2Config {
        vocab_size: 5,
        max_seq_len: 8,
        d_model: 4,
        n_heads: 2,
        n_layers: 1,
        d_ff: 8,
        dropout_rate: 0.0,
    };
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(config.clone(), &mut rng);

    let batch_size = 2;
    let seq_len = 3;
    let input_ids: Vec<usize> = vec![0, 1, 2, 3, 4, 0];
    let targets: Vec<usize> = vec![1, 2, 3, 4, 0, 1];

    let mut tape = GradTape::<Cpu>::new();
    let logits = model.forward_batch(&input_ids, batch_size, seq_len, &mut rng, &mut tape);
    assert_eq!(logits.shape.dims(), &[batch_size * seq_len, config.vocab_size]);
    assert!(logits.to_vec().iter().all(|v| v.is_finite()));

    let loss = ops::cross_entropy(
        &logits,
        &targets,
        batch_size * seq_len,
        config.vocab_size,
        Some(&mut tape),
    );
    let loss_val = loss.to_vec()[0];
    assert!(loss_val.is_finite());

    let grads = backward(&tape, loss.id);
    for param in model.parameters() {
        let g = Cpu::to_vec(grads.get(&param.id).unwrap());
        assert!(g.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn miri_gpt2_batched_forward_backward() {
    use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};

    let config = Gpt2Config {
        vocab_size: 5,
        max_seq_len: 8,
        d_model: 4,
        n_heads: 2,
        n_layers: 1,
        d_ff: 8,
        dropout_rate: 0.0,
    };
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(config.clone(), &mut rng);

    let batch_size = 2;
    let seq_len = 3;
    let input_ids: Vec<usize> = vec![0, 2, 4, 1, 3, 0];
    let targets: Vec<usize> = vec![2, 4, 1, 3, 0, 2];

    let mut tape = GradTape::<Cpu>::new();
    let logits = model.forward_batch(&input_ids, batch_size, seq_len, &mut rng, &mut tape);
    let loss = ops::cross_entropy(
        &logits,
        &targets,
        batch_size * seq_len,
        config.vocab_size,
        Some(&mut tape),
    );
    let loss_val = loss.to_vec()[0];
    assert!(loss_val.is_finite());
    assert!(loss_val > 0.0);

    let grads = backward(&tape, loss.id);
    assert!(grads.contains_key(&model.embedding.token_embedding.id));
    let de = Cpu::to_vec(grads.get(&model.embedding.token_embedding.id).unwrap());
    assert!(de.iter().all(|v| v.is_finite()));
}

#[test]
fn miri_training_step() {
    use scry_llm::data::Batch;
    use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};
    use scry_llm::training::{Trainer, TrainingConfig};

    let model_config = Gpt2Config {
        vocab_size: 5,
        max_seq_len: 8,
        d_model: 4,
        n_heads: 2,
        n_layers: 1,
        d_ff: 8,
        dropout_rate: 0.0,
    };
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(model_config.clone(), &mut rng);

    let config = TrainingConfig {
        batch_size: 1,
        seq_len: 3,
        total_steps: 10,
        warmup_steps: 2,
        peak_lr: 3e-4,
        min_lr: 1e-5,
        grad_accum_steps: 1,
        max_grad_norm: 1.0,
        log_interval: 10,
        eval_interval: 0,
        checkpoint_interval: 0,
        checkpoint_dir: std::path::PathBuf::from("/tmp"),
        seed: 42,
        use_checkpointing: false,
        checkpoint_every: 4,
        peak_tflops: None,
        n_params: None,
    };

    let mut trainer = Trainer::<Cpu>::new(model, model_config, config);
    assert_eq!(trainer.step, 0);

    let batch = Batch {
        input_ids: vec![0, 1, 2],
        targets: vec![1, 2, 3],
        batch_size: 1,
        seq_len: 3,
    };

    let metrics = trainer.train_step(&[batch]);
    assert!(metrics.loss.is_finite());
    assert!(metrics.grad_norm.is_finite());
    assert_eq!(trainer.step, 1);
}

#[test]
fn miri_gradient_accumulation() {
    use scry_llm::data::Batch;
    use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};
    use scry_llm::training::{Trainer, TrainingConfig};

    let model_config = Gpt2Config {
        vocab_size: 5,
        max_seq_len: 8,
        d_model: 4,
        n_heads: 2,
        n_layers: 1,
        d_ff: 8,
        dropout_rate: 0.0,
    };
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(model_config.clone(), &mut rng);

    let config = TrainingConfig {
        batch_size: 1,
        seq_len: 3,
        total_steps: 10,
        warmup_steps: 2,
        peak_lr: 3e-4,
        min_lr: 1e-5,
        grad_accum_steps: 2,
        max_grad_norm: 1.0,
        log_interval: 10,
        eval_interval: 0,
        checkpoint_interval: 0,
        checkpoint_dir: std::path::PathBuf::from("/tmp"),
        seed: 42,
        use_checkpointing: false,
        checkpoint_every: 4,
        peak_tflops: None,
        n_params: None,
    };

    let mut trainer = Trainer::<Cpu>::new(model, model_config, config);

    let batch1 = Batch {
        input_ids: vec![0, 1, 2],
        targets: vec![1, 2, 3],
        batch_size: 1,
        seq_len: 3,
    };
    let batch2 = Batch {
        input_ids: vec![2, 3, 4],
        targets: vec![3, 4, 0],
        batch_size: 1,
        seq_len: 3,
    };

    let metrics = trainer.train_step(&[batch1, batch2]);
    assert!(metrics.loss.is_finite());
    assert!(metrics.grad_norm.is_finite());
    assert_eq!(trainer.step, 1);
}

#[test]
fn miri_evaluate() {
    use scry_llm::data::Batch;
    use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};
    use scry_llm::training::{Trainer, TrainingConfig};

    let model_config = Gpt2Config {
        vocab_size: 5,
        max_seq_len: 8,
        d_model: 4,
        n_heads: 2,
        n_layers: 1,
        d_ff: 8,
        dropout_rate: 0.0,
    };
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(model_config.clone(), &mut rng);

    let config = TrainingConfig {
        batch_size: 1,
        seq_len: 3,
        total_steps: 10,
        warmup_steps: 2,
        peak_lr: 3e-4,
        min_lr: 1e-5,
        grad_accum_steps: 1,
        max_grad_norm: 1.0,
        log_interval: 10,
        eval_interval: 0,
        checkpoint_interval: 0,
        checkpoint_dir: std::path::PathBuf::from("/tmp"),
        seed: 42,
        use_checkpointing: false,
        checkpoint_every: 4,
        peak_tflops: None,
        n_params: None,
    };

    let trainer = Trainer::<Cpu>::new(model, model_config, config);

    let batches = vec![
        Batch {
            input_ids: vec![0, 1, 2],
            targets: vec![1, 2, 3],
            batch_size: 1,
            seq_len: 3,
        },
        Batch {
            input_ids: vec![2, 3, 4],
            targets: vec![3, 4, 0],
            batch_size: 1,
            seq_len: 3,
        },
    ];

    let val_loss = trainer.evaluate(&batches);
    assert!(val_loss.is_finite());
    assert!(val_loss > 0.0);
}

// ============================================================
// Phase 4: No-decay, embedding dropout, checkpointing, generate
// ============================================================

#[test]
fn miri_no_decay_ids() {
    use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};

    let config = Gpt2Config {
        vocab_size: 5,
        max_seq_len: 4,
        d_model: 4,
        n_heads: 2,
        n_layers: 2,
        d_ff: 8,
        dropout_rate: 0.0,
    };
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(config, &mut rng);
    let no_decay = model.no_decay_ids();
    // 8 per layer + 2 for ln_f = 18
    assert_eq!(no_decay.len(), 18);
}

#[test]
fn miri_adamw_no_decay_step() {
    use scry_llm::optim::adamw::{AdamW, AdamWConfig};
    let shape = Shape::new(&[3]);
    let mut param = Tensor::<Cpu>::from_vec(vec![1.0, 2.0, 3.0], shape.clone());
    let mut grad_map = std::collections::HashMap::new();
    grad_map.insert(param.id, Cpu::from_vec(vec![0.1, 0.2, 0.3], &shape));

    let mut no_decay = std::collections::HashSet::new();
    no_decay.insert(param.id);

    let mut optimizer = AdamW::<Cpu>::new(AdamWConfig {
        weight_decay: 0.1,
        ..AdamWConfig::default()
    });
    let param_id = param.id;
    let param_shape = param.shape.clone();
    let mut params = vec![(param_id, param.data_mut(), &param_shape)];
    optimizer.step(&mut params, &grad_map, &no_decay);
    drop(params);

    let after = Cpu::to_vec(&param.data);
    assert!(after.iter().all(|v| v.is_finite()));
}

#[test]
fn miri_embedding_dropout_forward() {
    use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};

    let config = Gpt2Config {
        vocab_size: 5,
        max_seq_len: 4,
        d_model: 4,
        n_heads: 2,
        n_layers: 1,
        d_ff: 8,
        dropout_rate: 0.0, // zero dropout for miri (avoids stochastic issues)
    };
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(config, &mut rng);

    let mut rng_fwd = fastrand::Rng::with_seed(99);
    let mut tape = GradTape::<Cpu>::new();
    let logits = model.forward(&[0, 1, 2], &mut rng_fwd, &mut tape);
    assert!(logits.to_vec().iter().all(|v| v.is_finite()));
}

#[test]
fn miri_batched_checkpointed_forward_backward() {
    use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};

    let config = Gpt2Config {
        vocab_size: 5,
        max_seq_len: 8,
        d_model: 4,
        n_heads: 2,
        n_layers: 2,
        d_ff: 8,
        dropout_rate: 0.0,
    };
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(config.clone(), &mut rng);

    let batch_size = 2;
    let seq_len = 3;
    let token_ids = vec![0, 1, 2, 3, 4, 0];
    let targets = vec![1, 2, 3, 4, 0, 1];

    let mut rng_fwd = fastrand::Rng::with_seed(99);
    let mut tape = GradTape::<Cpu>::new();
    let logits = model.forward_batch_checkpointed(
        &token_ids, batch_size, seq_len, 1, &mut rng_fwd, &mut tape,
    );
    let loss = ops::cross_entropy(
        &logits, &targets, batch_size * seq_len, config.vocab_size, Some(&mut tape),
    );
    let loss_val = loss.to_vec()[0];
    assert!(loss_val.is_finite());

    let grads = model.backward_checkpointed(&tape, loss.id);
    assert!(grads.contains_key(&model.embedding.token_embedding.id));
}

#[test]
fn miri_generate_tiny() {
    use scry_llm::generate::{generate, SamplingConfig};
    use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};

    let config = Gpt2Config {
        vocab_size: 5,
        max_seq_len: 16,
        d_model: 4,
        n_heads: 2,
        n_layers: 1,
        d_ff: 8,
        dropout_rate: 0.0,
    };
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(config.clone(), &mut rng);

    let sampling = SamplingConfig {
        temperature: 0.0, // greedy for determinism
        top_k: 0,
        top_p: 1.0,
        max_tokens: 3,
    };
    let mut gen_rng = fastrand::Rng::with_seed(99);
    let tokens = generate(&model, &[0, 1], &sampling, &mut gen_rng);
    assert_eq!(tokens.len(), 3);
    for &t in &tokens {
        assert!(t < config.vocab_size);
    }
}
