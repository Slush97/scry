//! Gradient checks for Phase 2 NN modules: attention, linear, MLP, transformer block.
//! Central difference with `eps=1e-3`, relative error tolerance.

use scry_llm::autograd::backward::backward;
use scry_llm::autograd::ops;
use scry_llm::autograd::GradTape;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::backend::DeviceBackend;
use scry_llm::nn::attention::CausalSelfAttention;
use scry_llm::nn::linear::Linear;
use scry_llm::nn::mlp::Mlp;
use scry_llm::nn::transformer::TransformerBlock;
use scry_llm::nn::Module;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

type Cpu = CpuBackend;

const EPS: f64 = 1e-3;
const TOL: f64 = 5e-3;
// Relaxed tolerance for ops chained through softmax (QKV weight goes through per-head softmax)
const SOFTMAX_CHAIN_TOL: f64 = 5e-2;

fn rand_vec(rng: &mut fastrand::Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

fn sum_f64(data: &[f32]) -> f64 {
    data.iter().map(|&x| f64::from(x)).sum()
}

fn relative_error(analytical: f64, numerical: f64) -> f64 {
    let denom = analytical.abs().max(numerical.abs()).max(1e-8);
    (analytical - numerical).abs() / denom
}

/// Numerically check gradient of a scalar loss w.r.t. an input vector.
/// Skip near-zero gradients where relative error is unreliable.
const ABS_THRESHOLD: f64 = 1e-5;

fn check_grad(
    name: &str,
    input: &[f32],
    f_forward: impl Fn(&[f32]) -> Vec<f32>,
    f_grad: impl Fn(&[f32]) -> Vec<f32>,
    tol: f64,
) {
    let analytical = f_grad(input);
    let mut max_err = 0.0f64;
    let mut worst_idx = 0;
    let mut worst_an = 0.0;
    let mut worst_num = 0.0;

    for i in 0..input.len() {
        let mut plus = input.to_vec();
        let mut minus = input.to_vec();
        plus[i] += EPS as f32;
        minus[i] -= EPS as f32;
        let fp = sum_f64(&f_forward(&plus));
        let fm = sum_f64(&f_forward(&minus));
        let numerical = (fp - fm) / (2.0 * EPS);
        let an = f64::from(analytical[i]);

        // Skip near-zero entries where relative error is meaningless
        if an.abs() < ABS_THRESHOLD && numerical.abs() < ABS_THRESHOLD {
            continue;
        }

        let err = relative_error(an, numerical);
        if err > max_err {
            max_err = err;
            worst_idx = i;
            worst_an = an;
            worst_num = numerical;
        }
    }

    println!(
        "  {name}: max_rel_err={max_err:.2e} (idx={worst_idx}, analytical={worst_an:.6e}, numerical={worst_num:.6e})"
    );
    assert!(
        max_err < tol,
        "{name}: max relative error {max_err:.2e} exceeds tolerance {tol:.2e} at idx {worst_idx}"
    );
}

// ============================================================
// Attention gradient checks
// ============================================================

#[test]
fn grad_check_attention_d_input() {
    let mut rng = fastrand::Rng::with_seed(42);
    let d_model = 8;
    let n_heads = 2;
    let d_head = d_model / n_heads;
    let seq = 4;

    let input_data = rand_vec(&mut rng, seq * d_model);

    // Fixed weights
    let attn = CausalSelfAttention::<Cpu>::new(d_model, n_heads, &mut rng);
    let qkv_w = attn.qkv_weight.to_vec();
    let qkv_b = attn.qkv_bias.to_vec();
    let proj_w = attn.proj_weight.to_vec();
    let proj_b = attn.proj_bias.to_vec();

    check_grad(
        "attention_d_input",
        &input_data,
        |input| {
            let inp = Tensor::<Cpu>::from_vec(input.to_vec(), Shape::new(&[seq, d_model]));
            let qw = Tensor::<Cpu>::from_vec(qkv_w.clone(), Shape::new(&[d_model, 3 * d_model]));
            let qb = Tensor::<Cpu>::from_vec(qkv_b.clone(), Shape::new(&[3 * d_model]));
            let pw = Tensor::<Cpu>::from_vec(proj_w.clone(), Shape::new(&[d_model, d_model]));
            let pb = Tensor::<Cpu>::from_vec(proj_b.clone(), Shape::new(&[d_model]));
            let out = ops::attention(
                &inp, &qw, &qb, &pw, &pb, n_heads, d_model, d_head, 0.0, None, None,
            );
            out.to_vec()
        },
        |input| {
            let mut tape = GradTape::<Cpu>::new();
            let inp = Tensor::<Cpu>::from_vec(input.to_vec(), Shape::new(&[seq, d_model]));
            let qw = Tensor::<Cpu>::from_vec(qkv_w.clone(), Shape::new(&[d_model, 3 * d_model]));
            let qb = Tensor::<Cpu>::from_vec(qkv_b.clone(), Shape::new(&[3 * d_model]));
            let pw = Tensor::<Cpu>::from_vec(proj_w.clone(), Shape::new(&[d_model, d_model]));
            let pb = Tensor::<Cpu>::from_vec(proj_b.clone(), Shape::new(&[d_model]));
            let out = ops::attention(
                &inp,
                &qw,
                &qb,
                &pw,
                &pb,
                n_heads,
                d_model,
                d_head,
                0.0,
                None,
                Some(&mut tape),
            );
            let loss = ops::sum(&out, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&inp.id).unwrap())
        },
        TOL,
    );
}

#[test]
fn grad_check_attention_d_qkv_weight() {
    let mut rng = fastrand::Rng::with_seed(42);
    let d_model = 8;
    let n_heads = 2;
    let d_head = d_model / n_heads;
    let seq = 4;

    let input_data = rand_vec(&mut rng, seq * d_model);
    let attn = CausalSelfAttention::<Cpu>::new(d_model, n_heads, &mut rng);
    let qkv_w = attn.qkv_weight.to_vec();
    let qkv_b = attn.qkv_bias.to_vec();
    let proj_w = attn.proj_weight.to_vec();
    let proj_b = attn.proj_bias.to_vec();

    check_grad(
        "attention_d_qkv_weight",
        &qkv_w,
        |w| {
            let inp = Tensor::<Cpu>::from_vec(input_data.clone(), Shape::new(&[seq, d_model]));
            let qw = Tensor::<Cpu>::from_vec(w.to_vec(), Shape::new(&[d_model, 3 * d_model]));
            let qb = Tensor::<Cpu>::from_vec(qkv_b.clone(), Shape::new(&[3 * d_model]));
            let pw = Tensor::<Cpu>::from_vec(proj_w.clone(), Shape::new(&[d_model, d_model]));
            let pb = Tensor::<Cpu>::from_vec(proj_b.clone(), Shape::new(&[d_model]));
            let out = ops::attention(
                &inp, &qw, &qb, &pw, &pb, n_heads, d_model, d_head, 0.0, None, None,
            );
            out.to_vec()
        },
        |w| {
            let mut tape = GradTape::<Cpu>::new();
            let inp = Tensor::<Cpu>::from_vec(input_data.clone(), Shape::new(&[seq, d_model]));
            let qw = Tensor::<Cpu>::from_vec(w.to_vec(), Shape::new(&[d_model, 3 * d_model]));
            let qb = Tensor::<Cpu>::from_vec(qkv_b.clone(), Shape::new(&[3 * d_model]));
            let pw = Tensor::<Cpu>::from_vec(proj_w.clone(), Shape::new(&[d_model, d_model]));
            let pb = Tensor::<Cpu>::from_vec(proj_b.clone(), Shape::new(&[d_model]));
            let out = ops::attention(
                &inp,
                &qw,
                &qb,
                &pw,
                &pb,
                n_heads,
                d_model,
                d_head,
                0.0,
                None,
                Some(&mut tape),
            );
            let loss = ops::sum(&out, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&qw.id).unwrap())
        },
        SOFTMAX_CHAIN_TOL,
    );
}

#[test]
fn grad_check_attention_d_proj_weight() {
    let mut rng = fastrand::Rng::with_seed(42);
    let d_model = 8;
    let n_heads = 2;
    let d_head = d_model / n_heads;
    let seq = 4;

    let input_data = rand_vec(&mut rng, seq * d_model);
    let attn = CausalSelfAttention::<Cpu>::new(d_model, n_heads, &mut rng);
    let qkv_w = attn.qkv_weight.to_vec();
    let qkv_b = attn.qkv_bias.to_vec();
    let proj_w = attn.proj_weight.to_vec();
    let proj_b = attn.proj_bias.to_vec();

    check_grad(
        "attention_d_proj_weight",
        &proj_w,
        |w| {
            let inp = Tensor::<Cpu>::from_vec(input_data.clone(), Shape::new(&[seq, d_model]));
            let qw = Tensor::<Cpu>::from_vec(qkv_w.clone(), Shape::new(&[d_model, 3 * d_model]));
            let qb = Tensor::<Cpu>::from_vec(qkv_b.clone(), Shape::new(&[3 * d_model]));
            let pw = Tensor::<Cpu>::from_vec(w.to_vec(), Shape::new(&[d_model, d_model]));
            let pb = Tensor::<Cpu>::from_vec(proj_b.clone(), Shape::new(&[d_model]));
            let out = ops::attention(
                &inp, &qw, &qb, &pw, &pb, n_heads, d_model, d_head, 0.0, None, None,
            );
            out.to_vec()
        },
        |w| {
            let mut tape = GradTape::<Cpu>::new();
            let inp = Tensor::<Cpu>::from_vec(input_data.clone(), Shape::new(&[seq, d_model]));
            let qw = Tensor::<Cpu>::from_vec(qkv_w.clone(), Shape::new(&[d_model, 3 * d_model]));
            let qb = Tensor::<Cpu>::from_vec(qkv_b.clone(), Shape::new(&[3 * d_model]));
            let pw = Tensor::<Cpu>::from_vec(w.to_vec(), Shape::new(&[d_model, d_model]));
            let pb = Tensor::<Cpu>::from_vec(proj_b.clone(), Shape::new(&[d_model]));
            let out = ops::attention(
                &inp,
                &qw,
                &qb,
                &pw,
                &pb,
                n_heads,
                d_model,
                d_head,
                0.0,
                None,
                Some(&mut tape),
            );
            let loss = ops::sum(&out, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&pw.id).unwrap())
        },
        TOL,
    );
}

// ============================================================
// Linear gradient check
// ============================================================

#[test]
fn grad_check_linear() {
    let mut rng = fastrand::Rng::with_seed(42);
    let in_f = 6;
    let out_f = 4;
    let seq = 3;

    let input_data = rand_vec(&mut rng, seq * in_f);
    let linear = Linear::<Cpu>::new(in_f, out_f, &mut rng);
    let w_data = linear.weight.to_vec();
    let b_data = linear.bias.to_vec();

    check_grad(
        "linear_d_input",
        &input_data,
        |input| {
            let inp = Tensor::<Cpu>::from_vec(input.to_vec(), Shape::new(&[seq, in_f]));
            let lin = Linear::<Cpu> {
                weight: Tensor::from_vec(w_data.clone(), Shape::new(&[in_f, out_f])),
                bias: Tensor::from_vec(b_data.clone(), Shape::new(&[out_f])),
                in_features: in_f,
                out_features: out_f,
            };
            lin.forward_inference(&inp).to_vec()
        },
        |input| {
            let mut tape = GradTape::<Cpu>::new();
            let inp = Tensor::<Cpu>::from_vec(input.to_vec(), Shape::new(&[seq, in_f]));
            let lin = Linear::<Cpu> {
                weight: Tensor::from_vec(w_data.clone(), Shape::new(&[in_f, out_f])),
                bias: Tensor::from_vec(b_data.clone(), Shape::new(&[out_f])),
                in_features: in_f,
                out_features: out_f,
            };
            let out = lin.forward(&inp, &mut tape);
            let loss = ops::sum(&out, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&inp.id).unwrap())
        },
        TOL,
    );
}

// ============================================================
// MLP gradient check
// ============================================================

#[test]
fn grad_check_mlp() {
    let mut rng = fastrand::Rng::with_seed(42);
    let d_model = 8;
    let d_ff = 16;
    let seq = 3;

    let input_data = rand_vec(&mut rng, seq * d_model);
    let mlp = Mlp::<Cpu>::new(d_model, d_ff, &mut rng);

    // Snapshot weights
    let fc1_w = mlp.fc1.weight.to_vec();
    let fc1_b = mlp.fc1.bias.to_vec();
    let fc2_w = mlp.fc2.weight.to_vec();
    let fc2_b = mlp.fc2.bias.to_vec();

    let make_mlp = || -> Mlp<Cpu> {
        Mlp {
            fc1: Linear {
                weight: Tensor::from_vec(fc1_w.clone(), Shape::new(&[d_model, d_ff])),
                bias: Tensor::from_vec(fc1_b.clone(), Shape::new(&[d_ff])),
                in_features: d_model,
                out_features: d_ff,
            },
            fc2: Linear {
                weight: Tensor::from_vec(fc2_w.clone(), Shape::new(&[d_ff, d_model])),
                bias: Tensor::from_vec(fc2_b.clone(), Shape::new(&[d_model])),
                in_features: d_ff,
                out_features: d_model,
            },
        }
    };

    check_grad(
        "mlp_d_input",
        &input_data,
        |input| {
            let inp = Tensor::<Cpu>::from_vec(input.to_vec(), Shape::new(&[seq, d_model]));
            let m = make_mlp();
            m.forward_inference(&inp).to_vec()
        },
        |input| {
            let mut tape = GradTape::<Cpu>::new();
            let inp = Tensor::<Cpu>::from_vec(input.to_vec(), Shape::new(&[seq, d_model]));
            let m = make_mlp();
            let out = m.forward(&inp, &mut tape);
            let loss = ops::sum(&out, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&inp.id).unwrap())
        },
        TOL,
    );
}

// ============================================================
// Transformer block gradient check
// ============================================================

#[test]
fn grad_check_transformer_block() {
    let mut rng = fastrand::Rng::with_seed(42);
    let d_model = 8;
    let n_heads = 2;
    let d_ff = 16;
    let seq = 4;

    // Use small init to avoid large values
    let input_data: Vec<f32> = (0..seq * d_model)
        .map(|_| (rng.f32() - 0.5) * 0.2)
        .collect();

    let block = TransformerBlock::<Cpu>::new(d_model, n_heads, d_ff, &mut rng);

    // Snapshot all block weights
    let param_data: Vec<Vec<f32>> = block.parameters().iter().map(|p| p.to_vec()).collect();
    let param_shapes: Vec<Shape> = block.parameters().iter().map(|p| p.shape.clone()).collect();

    let make_block = || -> TransformerBlock<Cpu> {
        let mut new_block =
            TransformerBlock::new(d_model, n_heads, d_ff, &mut fastrand::Rng::with_seed(0));
        for (param, (data, shape)) in new_block
            .parameters_mut()
            .iter_mut()
            .zip(param_data.iter().zip(param_shapes.iter()))
        {
            **param = Tensor::from_vec(data.clone(), shape.clone());
        }
        new_block
    };

    check_grad(
        "transformer_block_d_input",
        &input_data,
        |input| {
            let inp = Tensor::<Cpu>::from_vec(input.to_vec(), Shape::new(&[seq, d_model]));
            let b = make_block();
            b.forward_inference(&inp).to_vec()
        },
        |input| {
            let mut tape = GradTape::<Cpu>::new();
            let inp = Tensor::<Cpu>::from_vec(input.to_vec(), Shape::new(&[seq, d_model]));
            let b = make_block();
            let out = b.forward(&inp, 0.0, &mut fastrand::Rng::with_seed(99), &mut tape);
            let loss = ops::sum(&out, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&inp.id).unwrap())
        },
        TOL,
    );
}
