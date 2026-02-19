#![cfg(feature = "cuda")]

//! Correctness tests: compare CudaBackend against CpuBackend for every MathBackend method.
//! All results must match within tolerance (1e-4 for reductions, 1e-3 for softmax/layernorm).

use scry_llm::backend::cpu::CpuBackend;
use scry_llm::backend::cuda::{init_gpu, CudaBackend};
use scry_llm::backend::{DeviceBackend, MathBackend};
use scry_llm::tensor::shape::Shape;

fn init() {
    init_gpu(0);
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (av - bv).abs();
        let scale = av.abs().max(bv.abs()).max(1e-6);
        assert!(
            diff / scale < tol || diff < tol,
            "{label}[{i}]: cpu={av}, gpu={bv}, diff={diff}"
        );
    }
}

fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..n).map(|_| (rng.f32() - 0.5) * 2.0).collect()
}

// ---- Individual op tests ----

#[test]
fn matmul_nn() {
    init();
    let m = 64;
    let k = 48;
    let n = 32;
    let a = rand_vec(m * k, 1);
    let b = rand_vec(k * n, 2);

    let cpu = CpuBackend::matmul(&a, &b, m, k, n, false, false);
    let ga = CudaBackend::from_vec(a, &Shape::new(&[m, k]));
    let gb = CudaBackend::from_vec(b, &Shape::new(&[k, n]));
    let gpu = CudaBackend::to_vec(&CudaBackend::matmul(&ga, &gb, m, k, n, false, false));

    assert_close(&cpu, &gpu, 1e-4, "matmul_nn");
}

#[test]
fn matmul_tn() {
    init();
    let m = 32;
    let k = 64;
    let n = 16;
    let a = rand_vec(k * m, 3); // [K, M] stored
    let b = rand_vec(k * n, 4);

    let cpu = CpuBackend::matmul(&a, &b, m, k, n, true, false);
    let ga = CudaBackend::from_vec(a, &Shape::new(&[k, m]));
    let gb = CudaBackend::from_vec(b, &Shape::new(&[k, n]));
    let gpu = CudaBackend::to_vec(&CudaBackend::matmul(&ga, &gb, m, k, n, true, false));

    assert_close(&cpu, &gpu, 1e-4, "matmul_tn");
}

#[test]
fn matmul_nt() {
    init();
    let m = 32;
    let k = 64;
    let n = 16;
    let a = rand_vec(m * k, 5);
    let b = rand_vec(n * k, 6); // [N, K] stored

    let cpu = CpuBackend::matmul(&a, &b, m, k, n, false, true);
    let ga = CudaBackend::from_vec(a, &Shape::new(&[m, k]));
    let gb = CudaBackend::from_vec(b, &Shape::new(&[n, k]));
    let gpu = CudaBackend::to_vec(&CudaBackend::matmul(&ga, &gb, m, k, n, false, true));

    assert_close(&cpu, &gpu, 1e-4, "matmul_nt");
}

#[test]
fn matmul_tt() {
    init();
    let m = 16;
    let k = 32;
    let n = 24;
    let a = rand_vec(k * m, 7);
    let b = rand_vec(n * k, 8);

    let cpu = CpuBackend::matmul(&a, &b, m, k, n, true, true);
    let ga = CudaBackend::from_vec(a, &Shape::new(&[k, m]));
    let gb = CudaBackend::from_vec(b, &Shape::new(&[n, k]));
    let gpu = CudaBackend::to_vec(&CudaBackend::matmul(&ga, &gb, m, k, n, true, true));

    assert_close(&cpu, &gpu, 1e-4, "matmul_tt");
}

#[test]
fn softmax_forward() {
    init();
    let rows = 8;
    let cols = 64;
    let input = rand_vec(rows * cols, 10);
    let shape = Shape::new(&[rows, cols]);

    let cpu = CpuBackend::softmax(&input, &shape);
    let gi = CudaBackend::from_vec(input, &shape);
    let gpu = CudaBackend::to_vec(&CudaBackend::softmax(&gi, &shape));

    assert_close(&cpu, &gpu, 1e-4, "softmax_fwd");
}

#[test]
fn softmax_backward() {
    init();
    let rows = 8;
    let cols = 64;
    let d_out = rand_vec(rows * cols, 11);
    let output = {
        let raw = rand_vec(rows * cols, 12);
        let shape = Shape::new(&[rows, cols]);
        CpuBackend::softmax(&raw, &shape)
    };
    let shape = Shape::new(&[rows, cols]);

    let cpu = CpuBackend::softmax_backward(&d_out, &output, &shape);
    let gd = CudaBackend::from_vec(d_out, &shape);
    let go = CudaBackend::from_vec(output, &shape);
    let gpu = CudaBackend::to_vec(&CudaBackend::softmax_backward(&gd, &go, &shape));

    assert_close(&cpu, &gpu, 1e-3, "softmax_bwd");
}

#[test]
fn layernorm_forward() {
    init();
    let rows = 16;
    let d = 64;
    let input = rand_vec(rows * d, 20);
    let gamma = rand_vec(d, 21);
    let beta = rand_vec(d, 22);
    let shape = Shape::new(&[rows, d]);

    let (cpu_out, cpu_mean, cpu_rstd) =
        CpuBackend::layernorm(&input, &gamma, &beta, &shape, 1e-5);
    let gi = CudaBackend::from_vec(input, &shape);
    let gg = CudaBackend::from_vec(gamma, &Shape::new(&[d]));
    let gb = CudaBackend::from_vec(beta, &Shape::new(&[d]));
    let (go, gm, gr) = CudaBackend::layernorm(&gi, &gg, &gb, &shape, 1e-5);
    let gpu_out = CudaBackend::to_vec(&go);
    let gpu_mean = CudaBackend::to_vec(&gm);
    let gpu_rstd = CudaBackend::to_vec(&gr);

    assert_close(&cpu_out, &gpu_out, 1e-3, "layernorm_fwd_out");
    assert_close(&cpu_mean, &gpu_mean, 1e-4, "layernorm_fwd_mean");
    assert_close(&cpu_rstd, &gpu_rstd, 1e-3, "layernorm_fwd_rstd");
}

#[test]
fn gelu_forward() {
    init();
    let n = 256;
    let input = rand_vec(n, 30);

    let cpu = CpuBackend::gelu(&input);
    let gi = CudaBackend::from_vec(input, &Shape::new(&[n]));
    let gpu = CudaBackend::to_vec(&CudaBackend::gelu(&gi));

    assert_close(&cpu, &gpu, 1e-4, "gelu_fwd");
}

#[test]
fn gelu_backward() {
    init();
    let n = 256;
    let d_out = rand_vec(n, 31);
    let input = rand_vec(n, 32);

    let cpu = CpuBackend::gelu_backward(&d_out, &input);
    let gd = CudaBackend::from_vec(d_out, &Shape::new(&[n]));
    let gi = CudaBackend::from_vec(input, &Shape::new(&[n]));
    let gpu = CudaBackend::to_vec(&CudaBackend::gelu_backward(&gd, &gi));

    assert_close(&cpu, &gpu, 1e-4, "gelu_bwd");
}

#[test]
fn cross_entropy_forward() {
    init();
    let batch = 4;
    let vocab = 32;
    let logits = rand_vec(batch * vocab, 40);
    let targets: Vec<usize> = vec![0, 5, 10, 31];

    let cpu_loss = CpuBackend::cross_entropy(&logits, &targets, batch, vocab);
    let gl = CudaBackend::from_vec(logits, &Shape::new(&[batch, vocab]));
    let gpu_loss = CudaBackend::cross_entropy(&gl, &targets, batch, vocab);

    let diff = (cpu_loss - gpu_loss).abs();
    assert!(
        diff < 1e-3,
        "cross_entropy_fwd: cpu={cpu_loss}, gpu={gpu_loss}, diff={diff}"
    );
}

#[test]
fn cross_entropy_backward() {
    init();
    let batch = 4;
    let vocab = 32;
    let logits = rand_vec(batch * vocab, 41);
    let targets: Vec<usize> = vec![0, 5, 10, 31];

    let cpu = CpuBackend::cross_entropy_backward(&logits, &targets, batch, vocab);
    let gl = CudaBackend::from_vec(logits, &Shape::new(&[batch, vocab]));
    let gpu = CudaBackend::to_vec(&CudaBackend::cross_entropy_backward(&gl, &targets, batch, vocab));

    assert_close(&cpu, &gpu, 1e-4, "cross_entropy_bwd");
}

#[test]
fn embedding_forward() {
    init();
    let vocab = 16;
    let dim = 32;
    let weight = rand_vec(vocab * dim, 50);
    let indices = vec![0usize, 3, 7, 15, 2];

    let cpu = CpuBackend::embedding(&weight, &indices, vocab, dim);
    let gw = CudaBackend::from_vec(weight, &Shape::new(&[vocab, dim]));
    let gpu = CudaBackend::to_vec(&CudaBackend::embedding(&gw, &indices, vocab, dim));

    assert_close(&cpu, &gpu, 1e-6, "embedding_fwd");
}

#[test]
fn embedding_backward() {
    init();
    let vocab = 16;
    let dim = 32;
    let n_indices = 5;
    let d_out = rand_vec(n_indices * dim, 51);
    let indices = vec![0usize, 3, 7, 15, 2];

    let cpu = CpuBackend::embedding_backward(&d_out, &indices, vocab, dim);
    let gd = CudaBackend::from_vec(d_out, &Shape::new(&[n_indices, dim]));
    let gpu = CudaBackend::to_vec(&CudaBackend::embedding_backward(&gd, &indices, vocab, dim));

    assert_close(&cpu, &gpu, 1e-5, "embedding_bwd");
}

#[test]
fn mul_elementwise() {
    init();
    let n = 128;
    let a = rand_vec(n, 60);
    let b = rand_vec(n, 61);

    let cpu = CpuBackend::mul_elementwise(&a, &b);
    let ga = CudaBackend::from_vec(a, &Shape::new(&[n]));
    let gb = CudaBackend::from_vec(b, &Shape::new(&[n]));
    let gpu = CudaBackend::to_vec(&CudaBackend::mul_elementwise(&ga, &gb));

    assert_close(&cpu, &gpu, 1e-6, "mul_elementwise");
}

#[test]
fn scale_op() {
    init();
    let n = 128;
    let a = rand_vec(n, 62);
    let scalar = 0.123f32;

    let cpu = CpuBackend::scale(&a, scalar);
    let ga = CudaBackend::from_vec(a, &Shape::new(&[n]));
    let gpu = CudaBackend::to_vec(&CudaBackend::scale(&ga, scalar));

    assert_close(&cpu, &gpu, 1e-6, "scale");
}

#[test]
fn add_inplace_op() {
    init();
    let n = 128;
    let mut a_cpu = rand_vec(n, 63);
    let b = rand_vec(n, 64);

    CpuBackend::add_inplace(&mut a_cpu, &b);

    let mut a_gpu = CudaBackend::from_vec(rand_vec(n, 63), &Shape::new(&[n]));
    let b_gpu = CudaBackend::from_vec(b, &Shape::new(&[n]));
    CudaBackend::add_inplace(&mut a_gpu, &b_gpu);
    let gpu = CudaBackend::to_vec(&a_gpu);

    assert_close(&a_cpu, &gpu, 1e-6, "add_inplace");
}

#[test]
fn adamw_step() {
    init();
    let n = 64;
    let mut param_cpu = rand_vec(n, 70);
    let grad = rand_vec(n, 71);
    let mut m_cpu = vec![0.0f32; n];
    let mut v_cpu = vec![0.0f32; n];

    let mut param_gpu = CudaBackend::from_vec(param_cpu.clone(), &Shape::new(&[n]));
    let grad_gpu = CudaBackend::from_vec(grad.clone(), &Shape::new(&[n]));
    let mut m_gpu = CudaBackend::from_vec(m_cpu.clone(), &Shape::new(&[n]));
    let mut v_gpu = CudaBackend::from_vec(v_cpu.clone(), &Shape::new(&[n]));

    CpuBackend::adamw_step(
        &mut param_cpu,
        &grad,
        &mut m_cpu,
        &mut v_cpu,
        1e-3,
        0.9,
        0.999,
        1e-8,
        0.01,
        1,
    );
    CudaBackend::adamw_step(
        &mut param_gpu,
        &grad_gpu,
        &mut m_gpu,
        &mut v_gpu,
        1e-3,
        0.9,
        0.999,
        1e-8,
        0.01,
        1,
    );

    let gpu_param = CudaBackend::to_vec(&param_gpu);
    let gpu_m = CudaBackend::to_vec(&m_gpu);
    let gpu_v = CudaBackend::to_vec(&v_gpu);

    assert_close(&param_cpu, &gpu_param, 1e-4, "adamw_param");
    assert_close(&m_cpu, &gpu_m, 1e-4, "adamw_m");
    assert_close(&v_cpu, &gpu_v, 1e-4, "adamw_v");
}

// ---- Full forward pass comparison ----

#[test]
fn full_forward_pass() {
    init();
    use scry_llm::autograd::GradTape;
    use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};

    let config = Gpt2Config {
        vocab_size: 64,
        max_seq_len: 16,
        d_model: 32,
        n_heads: 4,
        n_layers: 1,
        d_ff: 64,
        dropout_rate: 0.0,
    };

    // Build CPU model
    let mut rng = fastrand::Rng::with_seed(42);
    let cpu_model = Gpt2Model::<CpuBackend>::new(config.clone(), &mut rng);

    // Build GPU model with same weights
    let mut rng2 = fastrand::Rng::with_seed(42);
    let gpu_model = Gpt2Model::<CudaBackend>::new(config.clone(), &mut rng2);

    let tokens: Vec<usize> = (0..8).map(|i| i % 64).collect();

    let mut cpu_tape = GradTape::<CpuBackend>::new();
    let mut cpu_rng = fastrand::Rng::with_seed(99);
    let cpu_logits = cpu_model.forward(&tokens, &mut cpu_rng, &mut cpu_tape);
    let cpu_vec = CpuBackend::to_vec(&cpu_logits.data);

    let mut gpu_tape = GradTape::<CudaBackend>::new();
    let mut gpu_rng = fastrand::Rng::with_seed(99);
    let gpu_logits = gpu_model.forward(&tokens, &mut gpu_rng, &mut gpu_tape);
    let gpu_vec = CudaBackend::to_vec(&gpu_logits.data);

    assert_close(&cpu_vec, &gpu_vec, 1e-2, "full_forward");
}
