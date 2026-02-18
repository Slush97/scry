//! Numerical gradient checking for every autograd operation.
//! Central difference: `(f(x+eps) - f(x-eps)) / (2*eps)`.
//! Uses `eps=1e-3` for `f32` backends (standard practice — matches `PyTorch`'s default
//! for `float32` gradcheck). Larger `eps` reduces `f32` rounding noise relative to signal.
//! Relative error: `|analytical - numerical| / max(|analytical|, |numerical|, 1e-8)`

use scry_llm::autograd::backward::backward;
use scry_llm::autograd::ops;
use scry_llm::autograd::GradTape;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::backend::{DeviceBackend, MathBackend};
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

type Cpu = CpuBackend;

// eps=1e-3: optimal for f32. Truncation error O(eps^2)≈1e-6, f32 noise≈ULP/(2*eps)≈5e-5.
const EPS: f64 = 1e-3;
const TOL: f64 = 1e-3;
const CHAIN_TOL: f64 = 5e-3;

fn rand_vec(rng: &mut fastrand::Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

fn rand_vec_positive(rng: &mut fastrand::Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.f32() * 0.9 + 0.1).collect()
}

fn relative_error(analytical: f64, numerical: f64) -> f64 {
    let denom = analytical.abs().max(numerical.abs()).max(1e-8);
    (analytical - numerical).abs() / denom
}

/// Sum a `f32` slice in `f64` for precision.
fn sum_f64(data: &[f32]) -> f64 {
    data.iter().map(|&x| f64::from(x)).sum()
}

/// Check gradient of a scalar-valued function w.r.t. input data.
/// `f_forward` takes input data and returns the output tensor data (`f32`).
/// `f_grad` takes input data and returns analytical gradient for that input.
/// The scalar loss is `sum(output)` computed in `f64`.
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

/// Like `check_grad` but the forward function returns a scalar loss (`f64`) directly.
/// Used for `cross_entropy` which doesn't use `sum(output)` as loss.
fn check_grad_scalar(
    name: &str,
    input: &[f32],
    f_forward: impl Fn(&[f32]) -> f64,
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
        let fp = f_forward(&plus);
        let fm = f_forward(&minus);
        let numerical = (fp - fm) / (2.0 * EPS);
        let an = f64::from(analytical[i]);
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
// Individual operation gradient checks
// ============================================================

#[test]
fn grad_check_matmul() {
    let mut rng = fastrand::Rng::with_seed(42);
    let m = 3;
    let k = 4;
    let n = 2;
    let a_data = rand_vec(&mut rng, m * k);
    let b_data = rand_vec(&mut rng, k * n);

    // Check dA
    check_grad(
        "matmul_dA",
        &a_data,
        |a| CpuBackend::matmul(&a.to_vec(), &b_data, m, k, n, false, false),
        |a| {
            let mut tape = GradTape::<Cpu>::new();
            let a_t = Tensor::<Cpu>::from_vec(a.to_vec(), Shape::new(&[m, k]));
            let b_t = Tensor::<Cpu>::from_vec(b_data.clone(), Shape::new(&[k, n]));
            let c = ops::matmul(&a_t, &b_t, m, k, n, false, false, Some(&mut tape));
            let loss = ops::sum(&c, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&a_t.id).unwrap())
        },
        TOL,
    );

    // Check dB
    check_grad(
        "matmul_dB",
        &b_data,
        |b| CpuBackend::matmul(&a_data, &b.to_vec(), m, k, n, false, false),
        |b| {
            let mut tape = GradTape::<Cpu>::new();
            let a_t = Tensor::<Cpu>::from_vec(a_data.clone(), Shape::new(&[m, k]));
            let b_t = Tensor::<Cpu>::from_vec(b.to_vec(), Shape::new(&[k, n]));
            let c = ops::matmul(&a_t, &b_t, m, k, n, false, false, Some(&mut tape));
            let loss = ops::sum(&c, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&b_t.id).unwrap())
        },
        TOL,
    );
}

#[test]
fn grad_check_matmul_transposed() {
    let mut rng = fastrand::Rng::with_seed(42);
    let m = 3;
    let k = 4;
    let n = 2;

    // trans_a=true: A stored as [K, M]
    let a_data = rand_vec(&mut rng, k * m);
    let b_data = rand_vec(&mut rng, k * n);

    check_grad(
        "matmul_transA_dA",
        &a_data,
        |a| CpuBackend::matmul(&a.to_vec(), &b_data, m, k, n, true, false),
        |a| {
            let mut tape = GradTape::<Cpu>::new();
            let a_t = Tensor::<Cpu>::from_vec(a.to_vec(), Shape::new(&[k, m]));
            let b_t = Tensor::<Cpu>::from_vec(b_data.clone(), Shape::new(&[k, n]));
            let c = ops::matmul(&a_t, &b_t, m, k, n, true, false, Some(&mut tape));
            let loss = ops::sum(&c, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&a_t.id).unwrap())
        },
        TOL,
    );

    // trans_b=true: B stored as [N, K]
    let a_data2 = rand_vec(&mut rng, m * k);
    let b_data2 = rand_vec(&mut rng, n * k);

    check_grad(
        "matmul_transB_dB",
        &b_data2,
        |b| CpuBackend::matmul(&a_data2, &b.to_vec(), m, k, n, false, true),
        |b| {
            let mut tape = GradTape::<Cpu>::new();
            let a_t = Tensor::<Cpu>::from_vec(a_data2.clone(), Shape::new(&[m, k]));
            let b_t = Tensor::<Cpu>::from_vec(b.to_vec(), Shape::new(&[n, k]));
            let c = ops::matmul(&a_t, &b_t, m, k, n, false, true, Some(&mut tape));
            let loss = ops::sum(&c, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&b_t.id).unwrap())
        },
        TOL,
    );
}

#[test]
fn grad_check_matmul_both_transposed() {
    let mut rng = fastrand::Rng::with_seed(42);
    let m = 3;
    let k = 4;
    let n = 2;

    // trans_a=true, trans_b=true: A stored as [K, M], B stored as [N, K]
    let a_data = rand_vec(&mut rng, k * m);
    let b_data = rand_vec(&mut rng, n * k);

    check_grad(
        "matmul_bothTrans_dA",
        &a_data,
        |a| CpuBackend::matmul(&a.to_vec(), &b_data, m, k, n, true, true),
        |a| {
            let mut tape = GradTape::<Cpu>::new();
            let a_t = Tensor::<Cpu>::from_vec(a.to_vec(), Shape::new(&[k, m]));
            let b_t = Tensor::<Cpu>::from_vec(b_data.clone(), Shape::new(&[n, k]));
            let c = ops::matmul(&a_t, &b_t, m, k, n, true, true, Some(&mut tape));
            let loss = ops::sum(&c, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&a_t.id).unwrap())
        },
        TOL,
    );

    check_grad(
        "matmul_bothTrans_dB",
        &b_data,
        |b| CpuBackend::matmul(&a_data, &b.to_vec(), m, k, n, true, true),
        |b| {
            let mut tape = GradTape::<Cpu>::new();
            let a_t = Tensor::<Cpu>::from_vec(a_data.clone(), Shape::new(&[k, m]));
            let b_t = Tensor::<Cpu>::from_vec(b.to_vec(), Shape::new(&[n, k]));
            let c = ops::matmul(&a_t, &b_t, m, k, n, true, true, Some(&mut tape));
            let loss = ops::sum(&c, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&b_t.id).unwrap())
        },
        TOL,
    );
}

#[test]
fn grad_check_matmul_skinny() {
    // Edge case: vector-matrix multiply (m=1)
    let mut rng = fastrand::Rng::with_seed(42);
    let m = 1;
    let k = 5;
    let n = 3;
    let a_data = rand_vec(&mut rng, m * k);
    let b_data = rand_vec(&mut rng, k * n);

    check_grad(
        "matmul_skinny_m1_dA",
        &a_data,
        |a| CpuBackend::matmul(&a.to_vec(), &b_data, m, k, n, false, false),
        |a| {
            let mut tape = GradTape::<Cpu>::new();
            let a_t = Tensor::<Cpu>::from_vec(a.to_vec(), Shape::new(&[m, k]));
            let b_t = Tensor::<Cpu>::from_vec(b_data.clone(), Shape::new(&[k, n]));
            let c = ops::matmul(&a_t, &b_t, m, k, n, false, false, Some(&mut tape));
            let loss = ops::sum(&c, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&a_t.id).unwrap())
        },
        TOL,
    );

    // Edge case: matrix-vector multiply (n=1)
    let m2 = 4;
    let k2 = 3;
    let n2 = 1;
    let a_data2 = rand_vec(&mut rng, m2 * k2);
    let b_data2 = rand_vec(&mut rng, k2 * n2);

    check_grad(
        "matmul_skinny_n1_dB",
        &b_data2,
        |b| CpuBackend::matmul(&a_data2, &b.to_vec(), m2, k2, n2, false, false),
        |b| {
            let mut tape = GradTape::<Cpu>::new();
            let a_t = Tensor::<Cpu>::from_vec(a_data2.clone(), Shape::new(&[m2, k2]));
            let b_t = Tensor::<Cpu>::from_vec(b.to_vec(), Shape::new(&[k2, n2]));
            let c = ops::matmul(&a_t, &b_t, m2, k2, n2, false, false, Some(&mut tape));
            let loss = ops::sum(&c, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&b_t.id).unwrap())
        },
        TOL,
    );
}

#[test]
fn grad_check_add() {
    let mut rng = fastrand::Rng::with_seed(42);
    let a_data = rand_vec(&mut rng, 6);
    let b_data = rand_vec(&mut rng, 6);
    let shape = Shape::new(&[2, 3]);

    check_grad(
        "add_dA",
        &a_data,
        |a| CpuBackend::add(&a.to_vec(), &b_data, &shape, &shape, &shape),
        |a| {
            let mut tape = GradTape::<Cpu>::new();
            let a_t = Tensor::<Cpu>::from_vec(a.to_vec(), shape.clone());
            let b_t = Tensor::<Cpu>::from_vec(b_data.clone(), shape.clone());
            let c = ops::add(&a_t, &b_t, Some(&mut tape));
            let loss = ops::sum(&c, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&a_t.id).unwrap())
        },
        TOL,
    );

    check_grad(
        "add_dB",
        &b_data,
        |b| CpuBackend::add(&a_data, &b.to_vec(), &shape, &shape, &shape),
        |b| {
            let mut tape = GradTape::<Cpu>::new();
            let a_t = Tensor::<Cpu>::from_vec(a_data.clone(), shape.clone());
            let b_t = Tensor::<Cpu>::from_vec(b.to_vec(), shape.clone());
            let c = ops::add(&a_t, &b_t, Some(&mut tape));
            let loss = ops::sum(&c, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&b_t.id).unwrap())
        },
        TOL,
    );
}

#[test]
fn grad_check_add_broadcast() {
    let mut rng = fastrand::Rng::with_seed(42);
    let a_data = rand_vec(&mut rng, 6); // [2, 3]
    let b_data = rand_vec(&mut rng, 3); // [1, 3] broadcast to [2, 3]
    let a_shape = Shape::new(&[2, 3]);
    let b_shape = Shape::new(&[1, 3]);
    let out_shape = Shape::new(&[2, 3]);

    check_grad(
        "add_broadcast_dA",
        &a_data,
        |a| CpuBackend::add(&a.to_vec(), &b_data, &a_shape, &b_shape, &out_shape),
        |a| {
            let mut tape = GradTape::<Cpu>::new();
            let a_t = Tensor::<Cpu>::from_vec(a.to_vec(), a_shape.clone());
            let b_t = Tensor::<Cpu>::from_vec(b_data.clone(), b_shape.clone());
            let c = ops::add(&a_t, &b_t, Some(&mut tape));
            let loss = ops::sum(&c, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&a_t.id).unwrap())
        },
        TOL,
    );

    check_grad(
        "add_broadcast_dB",
        &b_data,
        |b| CpuBackend::add(&a_data, &b.to_vec(), &a_shape, &b_shape, &out_shape),
        |b| {
            let mut tape = GradTape::<Cpu>::new();
            let a_t = Tensor::<Cpu>::from_vec(a_data.clone(), a_shape.clone());
            let b_t = Tensor::<Cpu>::from_vec(b.to_vec(), b_shape.clone());
            let c = ops::add(&a_t, &b_t, Some(&mut tape));
            let loss = ops::sum(&c, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&b_t.id).unwrap())
        },
        TOL,
    );
}

#[test]
fn grad_check_softmax() {
    // Test softmax gradient through the autograd tape.
    // sum(softmax(x)) = 1 always, so we use a matmul after softmax to create
    // a non-trivial loss: loss = sum(softmax(x) @ w).
    let mut rng = fastrand::Rng::with_seed(42);
    let input_data = rand_vec(&mut rng, 12); // [3, 4]
    let w_data = rand_vec(&mut rng, 4 * 2); // [4, 2]
    let shape = Shape::new(&[3, 4]);

    check_grad(
        "softmax",
        &input_data,
        |input| {
            let sm = CpuBackend::softmax(&input.to_vec(), &shape);
            CpuBackend::matmul(&sm, &w_data, 3, 4, 2, false, false)
        },
        |input| {
            let mut tape = GradTape::<Cpu>::new();
            let x = Tensor::<Cpu>::from_vec(input.to_vec(), shape.clone());
            let w = Tensor::<Cpu>::from_vec(w_data.clone(), Shape::new(&[4, 2]));
            let sm = ops::softmax(&x, Some(&mut tape));
            let mm = ops::matmul(&sm, &w, 3, 4, 2, false, false, Some(&mut tape));
            let loss = ops::sum(&mm, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&x.id).unwrap())
        },
        CHAIN_TOL,
    );
}

#[test]
fn grad_check_softmax_uniform() {
    // Edge case: all-equal logits. Softmax output = uniform.
    // sum(softmax(x)) = 1 for any x, so gradient should be ~zero.
    let input_data = vec![1.0f32; 8]; // [2, 4], all ones
    let shape = Shape::new(&[2, 4]);

    let mut tape = GradTape::<Cpu>::new();
    let x = Tensor::<Cpu>::from_vec(input_data, shape);
    let sm = ops::softmax(&x, Some(&mut tape));
    let loss = ops::sum(&sm, Some(&mut tape));
    let grads = backward(&tape, loss.id);
    let dx = Cpu::to_vec(grads.get(&x.id).unwrap());
    for (i, &g) in dx.iter().enumerate() {
        assert!(
            g.abs() < 1e-6,
            "softmax_uniform: gradient at idx {i} should be ~0, got {g}"
        );
    }
    println!(
        "  softmax_uniform: max_abs_grad={:.2e}",
        dx.iter().map(|x| x.abs()).fold(0.0f32, f32::max)
    );
}

#[test]
fn grad_check_gelu() {
    let mut rng = fastrand::Rng::with_seed(42);
    let input_data = rand_vec(&mut rng, 8);

    check_grad(
        "gelu",
        &input_data,
        |input| CpuBackend::gelu(&input.to_vec()),
        |input| {
            let mut tape = GradTape::<Cpu>::new();
            let x = Tensor::<Cpu>::from_vec(input.to_vec(), Shape::new(&[2, 4]));
            let y = ops::gelu(&x, Some(&mut tape));
            let loss = ops::sum(&y, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&x.id).unwrap())
        },
        TOL,
    );
}

#[test]
fn grad_check_layernorm() {
    let mut rng = fastrand::Rng::with_seed(42);
    let d = 4;
    let n = 3;
    let input_data = rand_vec(&mut rng, n * d);
    let gamma_data = rand_vec_positive(&mut rng, d);
    let beta_data = rand_vec(&mut rng, d);
    let eps = 1e-5;
    let shape = Shape::new(&[n, d]);

    // Check d_input
    check_grad(
        "layernorm_dinput",
        &input_data,
        |input| {
            let (out, _, _) =
                CpuBackend::layernorm(&input.to_vec(), &gamma_data, &beta_data, &shape, eps);
            out
        },
        |input| {
            let mut tape = GradTape::<Cpu>::new();
            let x = Tensor::<Cpu>::from_vec(input.to_vec(), shape.clone());
            let g = Tensor::<Cpu>::from_vec(gamma_data.clone(), Shape::new(&[d]));
            let b = Tensor::<Cpu>::from_vec(beta_data.clone(), Shape::new(&[d]));
            let y = ops::layernorm(&x, &g, &b, eps, Some(&mut tape));
            let loss = ops::sum(&y, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&x.id).unwrap())
        },
        TOL,
    );

    // Check d_gamma
    check_grad(
        "layernorm_dgamma",
        &gamma_data,
        |gamma| {
            let (out, _, _) =
                CpuBackend::layernorm(&input_data, &gamma.to_vec(), &beta_data, &shape, eps);
            out
        },
        |gamma| {
            let mut tape = GradTape::<Cpu>::new();
            let x = Tensor::<Cpu>::from_vec(input_data.clone(), shape.clone());
            let g = Tensor::<Cpu>::from_vec(gamma.to_vec(), Shape::new(&[d]));
            let b = Tensor::<Cpu>::from_vec(beta_data.clone(), Shape::new(&[d]));
            let y = ops::layernorm(&x, &g, &b, eps, Some(&mut tape));
            let loss = ops::sum(&y, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&g.id).unwrap())
        },
        TOL,
    );

    // Check d_beta
    check_grad(
        "layernorm_dbeta",
        &beta_data,
        |beta| {
            let (out, _, _) =
                CpuBackend::layernorm(&input_data, &gamma_data, &beta.to_vec(), &shape, eps);
            out
        },
        |beta| {
            let mut tape = GradTape::<Cpu>::new();
            let x = Tensor::<Cpu>::from_vec(input_data.clone(), shape.clone());
            let g = Tensor::<Cpu>::from_vec(gamma_data.clone(), Shape::new(&[d]));
            let b = Tensor::<Cpu>::from_vec(beta.to_vec(), Shape::new(&[d]));
            let y = ops::layernorm(&x, &g, &b, eps, Some(&mut tape));
            let loss = ops::sum(&y, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&b.id).unwrap())
        },
        TOL,
    );
}

#[test]
fn grad_check_layernorm_small_var() {
    // Edge case: low-variance input tests eps stability.
    // Use matmul after layernorm for non-trivial loss (sum(LN(x))≈0 always).
    let d = 4;
    let n = 2;
    let o = 3;
    let input_data: Vec<f32> = (0..n * d).map(|i| 5.0 + (i as f32) * 0.1).collect();
    let gamma_data = vec![1.0f32; d];
    let beta_data = vec![0.0f32; d];
    let mut rng = fastrand::Rng::with_seed(77);
    let w_data = rand_vec(&mut rng, d * o);
    let eps = 1e-5;
    let shape = Shape::new(&[n, d]);

    check_grad(
        "layernorm_small_var_dinput",
        &input_data,
        |input| {
            let (out, _, _) =
                CpuBackend::layernorm(&input.to_vec(), &gamma_data, &beta_data, &shape, eps);
            CpuBackend::matmul(&out, &w_data, n, d, o, false, false)
        },
        |input| {
            let mut tape = GradTape::<Cpu>::new();
            let x = Tensor::<Cpu>::from_vec(input.to_vec(), shape.clone());
            let g = Tensor::<Cpu>::from_vec(gamma_data.clone(), Shape::new(&[d]));
            let b = Tensor::<Cpu>::from_vec(beta_data.clone(), Shape::new(&[d]));
            let w = Tensor::<Cpu>::from_vec(w_data.clone(), Shape::new(&[d, o]));
            let y = ops::layernorm(&x, &g, &b, eps, Some(&mut tape));
            let mm = ops::matmul(&y, &w, n, d, o, false, false, Some(&mut tape));
            let loss = ops::sum(&mm, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&x.id).unwrap())
        },
        CHAIN_TOL,
    );
}

#[test]
fn grad_check_cross_entropy() {
    let mut rng = fastrand::Rng::with_seed(42);
    let batch = 3;
    let vocab = 5;
    let logits_data = rand_vec(&mut rng, batch * vocab);
    let targets = vec![2usize, 0, 4];

    // Cross-entropy returns a scalar, compute in f64 for numerical check
    check_grad_scalar(
        "cross_entropy",
        &logits_data,
        |logits| {
            // Recompute cross-entropy in f64 for numerical precision
            let mut total_loss = 0.0f64;
            for b in 0..batch {
                let start = b * vocab;
                let slice = &logits[start..start + vocab];
                let target = targets[b];
                let max_val: f64 = slice
                    .iter()
                    .map(|&x| f64::from(x))
                    .fold(f64::NEG_INFINITY, f64::max);
                let sum_exp: f64 = slice.iter().map(|&x| (f64::from(x) - max_val).exp()).sum();
                let log_sum_exp = max_val + sum_exp.ln();
                let log_prob = f64::from(slice[target]) - log_sum_exp;
                total_loss -= log_prob;
            }
            total_loss / batch as f64
        },
        |logits| {
            let mut tape = GradTape::<Cpu>::new();
            let x = Tensor::<Cpu>::from_vec(logits.to_vec(), Shape::new(&[batch, vocab]));
            let loss = ops::cross_entropy(&x, &targets, batch, vocab, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&x.id).unwrap())
        },
        TOL,
    );
}

#[test]
fn grad_check_embedding() {
    let mut rng = fastrand::Rng::with_seed(42);
    let vocab = 5;
    let dim = 4;
    let weight_data = rand_vec(&mut rng, vocab * dim);
    let indices = vec![1usize, 3, 0, 3]; // note: 3 appears twice

    check_grad(
        "embedding",
        &weight_data,
        |weight| CpuBackend::embedding(&weight.to_vec(), &indices, vocab, dim),
        |weight| {
            let mut tape = GradTape::<Cpu>::new();
            let w = Tensor::<Cpu>::from_vec(weight.to_vec(), Shape::new(&[vocab, dim]));
            let out = ops::embedding(&w, &indices, vocab, dim, Some(&mut tape));
            let loss = ops::sum(&out, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&w.id).unwrap())
        },
        TOL,
    );
}

#[test]
fn grad_check_embedding_duplicates() {
    // Edge case: all-same index — gradient should accumulate to one row.
    let mut rng = fastrand::Rng::with_seed(42);
    let vocab = 4;
    let dim = 3;
    let weight_data = rand_vec(&mut rng, vocab * dim);
    let indices = vec![2usize, 2, 2, 2]; // all same

    check_grad(
        "embedding_duplicates",
        &weight_data,
        |weight| CpuBackend::embedding(&weight.to_vec(), &indices, vocab, dim),
        |weight| {
            let mut tape = GradTape::<Cpu>::new();
            let w = Tensor::<Cpu>::from_vec(weight.to_vec(), Shape::new(&[vocab, dim]));
            let out = ops::embedding(&w, &indices, vocab, dim, Some(&mut tape));
            let loss = ops::sum(&out, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            let dw = Cpu::to_vec(grads.get(&w.id).unwrap());

            // Verify: gradient for row 2 should be 4.0 (4 lookups), others 0.0
            for r in 0..vocab {
                for c in 0..dim {
                    let g = dw[r * dim + c];
                    if r == 2 {
                        assert!(
                            (g - 4.0).abs() < 1e-6,
                            "row {r} col {c}: expected 4.0, got {g}"
                        );
                    } else {
                        assert!(g.abs() < 1e-6, "row {r} col {c}: expected 0.0, got {g}");
                    }
                }
            }

            dw
        },
        TOL,
    );
}

// ============================================================
// Composition gradient checks
// ============================================================

#[test]
fn grad_check_chain_matmul_gelu() {
    let mut rng = fastrand::Rng::with_seed(42);
    let m = 3;
    let k = 4;
    let n = 2;
    let a_data = rand_vec(&mut rng, m * k);
    let b_data = rand_vec(&mut rng, k * n);

    check_grad(
        "chain_matmul_gelu_dA",
        &a_data,
        |a| {
            let c = CpuBackend::matmul(&a.to_vec(), &b_data, m, k, n, false, false);
            CpuBackend::gelu(&c)
        },
        |a| {
            let mut tape = GradTape::<Cpu>::new();
            let a_t = Tensor::<Cpu>::from_vec(a.to_vec(), Shape::new(&[m, k]));
            let b_t = Tensor::<Cpu>::from_vec(b_data.clone(), Shape::new(&[k, n]));
            let c = ops::matmul(&a_t, &b_t, m, k, n, false, false, Some(&mut tape));
            let g = ops::gelu(&c, Some(&mut tape));
            let loss = ops::sum(&g, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&a_t.id).unwrap())
        },
        CHAIN_TOL,
    );

    check_grad(
        "chain_matmul_gelu_dB",
        &b_data,
        |b| {
            let c = CpuBackend::matmul(&a_data, &b.to_vec(), m, k, n, false, false);
            CpuBackend::gelu(&c)
        },
        |b| {
            let mut tape = GradTape::<Cpu>::new();
            let a_t = Tensor::<Cpu>::from_vec(a_data.clone(), Shape::new(&[m, k]));
            let b_t = Tensor::<Cpu>::from_vec(b.to_vec(), Shape::new(&[k, n]));
            let c = ops::matmul(&a_t, &b_t, m, k, n, false, false, Some(&mut tape));
            let g = ops::gelu(&c, Some(&mut tape));
            let loss = ops::sum(&g, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&b_t.id).unwrap())
        },
        CHAIN_TOL,
    );
}

#[test]
fn grad_check_chain_layernorm_matmul_softmax() {
    // Full chain through the tape: layernorm -> matmul -> softmax -> sum
    let mut rng = fastrand::Rng::with_seed(42);
    let n = 2; // batch
    let d = 4; // hidden dim
    let v = 3; // output dim

    let input_data = rand_vec(&mut rng, n * d);
    let gamma_data = rand_vec_positive(&mut rng, d);
    let beta_data = rand_vec(&mut rng, d);
    let w_data = rand_vec(&mut rng, d * v);
    let eps = 1e-5;

    // Use cross_entropy instead of sum(softmax) since sum(softmax)=1 always.
    let targets = vec![1usize, 0]; // one target per batch

    check_grad_scalar(
        "chain_ln_matmul_softmax_dinput",
        &input_data,
        |input| {
            let shape_nd = Shape::new(&[n, d]);
            let (ln, _, _) =
                CpuBackend::layernorm(&input.to_vec(), &gamma_data, &beta_data, &shape_nd, eps);
            let mm = CpuBackend::matmul(&ln, &w_data, n, d, v, false, false);
            // Compute cross-entropy in f64
            let mut total_loss = 0.0f64;
            for b in 0..n {
                let start = b * v;
                let slice = &mm[start..start + v];
                let target = targets[b];
                let max_val: f64 = slice
                    .iter()
                    .map(|&x| f64::from(x))
                    .fold(f64::NEG_INFINITY, f64::max);
                let sum_exp: f64 = slice.iter().map(|&x| (f64::from(x) - max_val).exp()).sum();
                let log_sum_exp = max_val + sum_exp.ln();
                total_loss -= f64::from(slice[target]) - log_sum_exp;
            }
            total_loss / n as f64
        },
        |input| {
            let mut tape = GradTape::<Cpu>::new();
            let x = Tensor::<Cpu>::from_vec(input.to_vec(), Shape::new(&[n, d]));
            let g = Tensor::<Cpu>::from_vec(gamma_data.clone(), Shape::new(&[d]));
            let b = Tensor::<Cpu>::from_vec(beta_data.clone(), Shape::new(&[d]));
            let w = Tensor::<Cpu>::from_vec(w_data.clone(), Shape::new(&[d, v]));
            let ln = ops::layernorm(&x, &g, &b, eps, Some(&mut tape));
            let mm = ops::matmul(&ln, &w, n, d, v, false, false, Some(&mut tape));
            let loss = ops::cross_entropy(&mm, &targets, n, v, Some(&mut tape));
            let grads = backward(&tape, loss.id);
            Cpu::to_vec(grads.get(&x.id).unwrap())
        },
        CHAIN_TOL,
    );
}

// ============================================================
// GPT-2 scale smoke test
// ============================================================

#[test]
fn scale_test_gpt2_dims() {
    // Forward + backward at GPT-2 scale to catch overflow/NaN/Inf.
    // No gradient check (too slow), just verify no panics or NaN.
    let mut rng = fastrand::Rng::with_seed(42);

    let batch = 2;
    let seq = 128;
    let hidden = 768;
    let vocab = 50257;
    let total_tokens = batch * seq; // 256

    // LayerNorm input: [256, 768]
    let input_data: Vec<f32> = (0..total_tokens * hidden)
        .map(|_| (rng.f32() - 0.5) * 0.1) // small init like GPT-2
        .collect();
    let gamma_data = vec![1.0f32; hidden];
    let beta_data = vec![0.0f32; hidden];

    // Weight matrix: [768, 50257]
    let w_data: Vec<f32> = (0..hidden * vocab)
        .map(|_| (rng.f32() - 0.5) * 0.02) // GPT-2 init scale
        .collect();

    let targets: Vec<usize> = (0..total_tokens).map(|_| rng.usize(0..vocab)).collect();

    let eps = 1e-5;

    println!("  GPT-2 scale test: [{batch}x{seq}, {hidden}] -> [{hidden}, {vocab}]");

    let mut tape = GradTape::<Cpu>::new();

    // Forward: layernorm -> matmul -> cross_entropy
    let x = Tensor::<Cpu>::from_vec(input_data, Shape::new(&[total_tokens, hidden]));
    let g = Tensor::<Cpu>::from_vec(gamma_data, Shape::new(&[hidden]));
    let b = Tensor::<Cpu>::from_vec(beta_data, Shape::new(&[hidden]));
    let w = Tensor::<Cpu>::from_vec(w_data, Shape::new(&[hidden, vocab]));

    let ln = ops::layernorm(&x, &g, &b, eps, Some(&mut tape));
    let logits = ops::matmul(
        &ln,
        &w,
        total_tokens,
        hidden,
        vocab,
        false,
        false,
        Some(&mut tape),
    );
    let loss = ops::cross_entropy(&logits, &targets, total_tokens, vocab, Some(&mut tape));

    let loss_val = loss.to_vec()[0];
    println!("  loss = {loss_val:.4}");
    assert!(!loss_val.is_nan(), "GPT-2 scale: loss is NaN");
    assert!(!loss_val.is_infinite(), "GPT-2 scale: loss is Inf");
    assert!(loss_val > 0.0, "GPT-2 scale: loss should be positive");

    // Backward
    let grads = backward(&tape, loss.id);

    // Check gradients exist and are finite
    let dx = Cpu::to_vec(grads.get(&x.id).unwrap());
    assert!(
        dx.iter().all(|v| v.is_finite()),
        "GPT-2 scale: d_input contains NaN/Inf"
    );

    let dw = Cpu::to_vec(grads.get(&w.id).unwrap());
    assert!(
        dw.iter().all(|v| v.is_finite()),
        "GPT-2 scale: d_weight contains NaN/Inf"
    );

    let dg = Cpu::to_vec(grads.get(&g.id).unwrap());
    assert!(
        dg.iter().all(|v| v.is_finite()),
        "GPT-2 scale: d_gamma contains NaN/Inf"
    );

    println!(
        "  dx_max={:.4e}, dw_max={:.4e}, dg_max={:.4e}",
        dx.iter().map(|v| v.abs()).fold(0.0f32, f32::max),
        dw.iter().map(|v| v.abs()).fold(0.0f32, f32::max),
        dg.iter().map(|v| v.abs()).fold(0.0f32, f32::max),
    );
}
