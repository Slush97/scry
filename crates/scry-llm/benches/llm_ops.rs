use criterion::{black_box, criterion_group, criterion_main, Criterion};
use scry_llm::autograd::ops;
use scry_llm::autograd::GradTape;
use scry_llm::backend::cpu::CpuBackend;
#[cfg(feature = "cuda")]
use scry_llm::backend::DeviceBackend;
use scry_llm::backend::MathBackend;
use scry_llm::data::{Batch, DataLoader};
use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;
use scry_llm::training::{Trainer, TrainingConfig};

type Cpu = CpuBackend;

fn bench_config() -> Gpt2Config {
    Gpt2Config {
        vocab_size: 256,
        max_seq_len: 64,
        d_model: 64,
        n_heads: 4,
        n_layers: 2,
        d_ff: 128,
        dropout_rate: 0.0,
    }
}

// ============================================================
// Group 1: Forward pass
// ============================================================

fn forward_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_pass");
    let config = bench_config();
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(config.clone(), &mut rng);

    let seq = 32;
    let tokens_single: Vec<usize> = (0..seq).map(|i| i % config.vocab_size).collect();

    // Warmup
    for _ in 0..2 {
        let mut tape = GradTape::<Cpu>::new();
        let _ = black_box(model.forward(&tokens_single, &mut rng, &mut tape));
    }

    group.bench_function("forward_single_seq", |b| {
        b.iter(|| {
            let mut tape = GradTape::<Cpu>::new();
            black_box(model.forward(black_box(&tokens_single), &mut rng, &mut tape))
        });
    });

    let tokens_b2: Vec<usize> = (0..2 * seq).map(|i| i % config.vocab_size).collect();
    // Warmup
    for _ in 0..2 {
        let mut tape = GradTape::<Cpu>::new();
        let _ = black_box(model.forward_batch(&tokens_b2, 2, seq, &mut rng, &mut tape));
    }

    group.bench_function("forward_batch_2", |b| {
        b.iter(|| {
            let mut tape = GradTape::<Cpu>::new();
            black_box(model.forward_batch(black_box(&tokens_b2), 2, seq, &mut rng, &mut tape))
        });
    });

    let tokens_b4: Vec<usize> = (0..4 * seq).map(|i| i % config.vocab_size).collect();
    // Warmup
    for _ in 0..2 {
        let mut tape = GradTape::<Cpu>::new();
        let _ = black_box(model.forward_batch(&tokens_b4, 4, seq, &mut rng, &mut tape));
    }

    group.bench_function("forward_batch_4", |b| {
        b.iter(|| {
            let mut tape = GradTape::<Cpu>::new();
            black_box(model.forward_batch(black_box(&tokens_b4), 4, seq, &mut rng, &mut tape))
        });
    });

    group.finish();
}

// ============================================================
// Group 2: Training step
// ============================================================

fn training_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("training_step");
    let config = bench_config();

    // No accumulation
    {
        let mut rng = fastrand::Rng::with_seed(42);
        let model = Gpt2Model::<Cpu>::new(config.clone(), &mut rng);
        let training_config = TrainingConfig {
            batch_size: 2,
            seq_len: 16,
            total_steps: 10000,
            warmup_steps: 100,
            peak_lr: 3e-4,
            min_lr: 1e-5,
            grad_accum_steps: 1,
            max_grad_norm: 1.0,
            log_interval: 100_000,
            eval_interval: 0,
            checkpoint_interval: 0,
            checkpoint_dir: std::path::PathBuf::from("/tmp"),
            seed: 42,
            use_checkpointing: false,
            checkpoint_every: 4,
            peak_tflops: None,
            n_params: None,
        };
        let mut trainer = Trainer::<Cpu>::new(model, config.clone(), training_config);

        // Warmup
        for _ in 0..2 {
            let batch = make_batch(2, 16, config.vocab_size);
            let _ = black_box(trainer.train_step(&[batch]));
        }

        group.bench_function("train_step_no_accum", |b| {
            b.iter(|| {
                trainer.step = 0;
                let batch = make_batch(2, 16, config.vocab_size);
                black_box(trainer.train_step(black_box(&[batch])))
            });
        });
    }

    // With accumulation (4 micro-batches)
    {
        let mut rng = fastrand::Rng::with_seed(42);
        let model = Gpt2Model::<Cpu>::new(config.clone(), &mut rng);
        let training_config = TrainingConfig {
            batch_size: 2,
            seq_len: 16,
            total_steps: 10000,
            warmup_steps: 100,
            peak_lr: 3e-4,
            min_lr: 1e-5,
            grad_accum_steps: 4,
            max_grad_norm: 1.0,
            log_interval: 100_000,
            eval_interval: 0,
            checkpoint_interval: 0,
            checkpoint_dir: std::path::PathBuf::from("/tmp"),
            seed: 42,
            use_checkpointing: false,
            checkpoint_every: 4,
            peak_tflops: None,
            n_params: None,
        };
        let mut trainer = Trainer::<Cpu>::new(model, config.clone(), training_config);

        // Warmup
        for _ in 0..2 {
            let batches: Vec<_> = (0..4).map(|_| make_batch(2, 16, config.vocab_size)).collect();
            let _ = black_box(trainer.train_step(&batches));
        }

        group.bench_function("train_step_accum_4", |b| {
            b.iter(|| {
                trainer.step = 0;
                let batches: Vec<_> =
                    (0..4).map(|_| make_batch(2, 16, config.vocab_size)).collect();
                black_box(trainer.train_step(black_box(&batches)))
            });
        });
    }

    group.finish();
}

fn make_batch(batch_size: usize, seq_len: usize, vocab: usize) -> Batch {
    let total = batch_size * seq_len;
    Batch {
        input_ids: (0..total).map(|i| i % vocab).collect(),
        targets: (0..total).map(|i| (i + 1) % vocab).collect(),
        batch_size,
        seq_len,
    }
}

// ============================================================
// Group 3: Micro ops
// ============================================================

fn ops_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("ops_matmul");

    // matmul 64x64
    {
        let a = vec![0.01f32; 64 * 64];
        let b = vec![0.01f32; 64 * 64];
        for _ in 0..2 {
            let _ = black_box(CpuBackend::matmul(&a, &b, 64, 64, 64, false, false));
        }
        group.bench_function("matmul_64x64", |bench| {
            bench.iter(|| {
                black_box(CpuBackend::matmul(
                    black_box(&a),
                    black_box(&b),
                    64, 64, 64, false, false,
                ))
            });
        });
    }

    // matmul 768x768 (GPT-2 small scale)
    {
        let a = vec![0.01f32; 768 * 768];
        let b = vec![0.01f32; 768 * 768];
        for _ in 0..2 {
            let _ = black_box(CpuBackend::matmul(&a, &b, 768, 768, 768, false, false));
        }
        group.bench_function("matmul_768x768", |bench| {
            bench.iter(|| {
                black_box(CpuBackend::matmul(
                    black_box(&a),
                    black_box(&b),
                    768, 768, 768, false, false,
                ))
            });
        });
    }

    group.finish();
}

fn ops_micro(c: &mut Criterion) {
    let mut group = c.benchmark_group("ops_micro");

    // layernorm [32, 32]
    {
        let input = vec![0.5f32; 32 * 32];
        let gamma = vec![1.0f32; 32];
        let beta = vec![0.0f32; 32];
        let shape = Shape::new(&[32, 32]);
        for _ in 0..2 {
            let _ = black_box(CpuBackend::layernorm(&input, &gamma, &beta, &shape, 1e-5));
        }
        group.bench_function("layernorm_1024", |bench| {
            bench.iter(|| {
                black_box(CpuBackend::layernorm(
                    black_box(&input),
                    black_box(&gamma),
                    black_box(&beta),
                    black_box(&shape),
                    1e-5,
                ))
            });
        });
    }

    // softmax [4, 32]
    {
        let input = vec![0.1f32; 4 * 32];
        let shape = Shape::new(&[4, 32]);
        for _ in 0..2 {
            let _ = black_box(CpuBackend::softmax(&input, &shape));
        }
        group.bench_function("softmax_128", |bench| {
            bench.iter(|| black_box(CpuBackend::softmax(black_box(&input), black_box(&shape))));
        });
    }

    // cross_entropy [8, 32]
    {
        let logits = vec![0.1f32; 8 * 32];
        let targets: Vec<usize> = (0..8).collect();
        for _ in 0..2 {
            let _ = black_box(CpuBackend::cross_entropy(&logits, &targets, 8, 32)[0]);
        }
        group.bench_function("cross_entropy_256", |bench| {
            bench.iter(|| {
                black_box(CpuBackend::cross_entropy(
                    black_box(&logits),
                    black_box(&targets),
                    8, 32,
                )[0])
            });
        });
    }

    // attention single seq=16, d=32, heads=4
    {
        let d_model = 32;
        let n_heads = 4;
        let d_head = d_model / n_heads;
        let seq = 16;
        let input = Tensor::<Cpu>::from_vec(vec![0.1; seq * d_model], Shape::new(&[seq, d_model]));
        let mut rng_init = fastrand::Rng::with_seed(42);
        let qkv_w = Tensor::<Cpu>::from_vec(
            (0..d_model * 3 * d_model)
                .map(|_| (rng_init.f32() - 0.5) * 0.1)
                .collect(),
            Shape::new(&[d_model, 3 * d_model]),
        );
        let qkv_b = Tensor::<Cpu>::from_vec(vec![0.0; 3 * d_model], Shape::new(&[3 * d_model]));
        let proj_w = Tensor::<Cpu>::from_vec(
            (0..d_model * d_model)
                .map(|_| (rng_init.f32() - 0.5) * 0.1)
                .collect(),
            Shape::new(&[d_model, d_model]),
        );
        let proj_b = Tensor::<Cpu>::from_vec(vec![0.0; d_model], Shape::new(&[d_model]));

        for _ in 0..2 {
            let _ = black_box(ops::attention(
                &input, &qkv_w, &qkv_b, &proj_w, &proj_b, n_heads, d_model, d_head, 0.0, None,
                None,
            ));
        }

        group.bench_function("attention_single_seq16", |bench| {
            bench.iter(|| {
                black_box(ops::attention(
                    black_box(&input),
                    black_box(&qkv_w),
                    black_box(&qkv_b),
                    black_box(&proj_w),
                    black_box(&proj_b),
                    n_heads, d_model, d_head, 0.0, None, None,
                ))
            });
        });
    }

    group.finish();
}

// ============================================================
// Group 4: DataLoader
// ============================================================

fn data_loader(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_loader");

    let tokens: Vec<u16> = (0..1000u16).map(|i| i % 256).collect();
    let mut loader = DataLoader::from_tokens(tokens, 32, 4, 42);

    // Warmup
    for _ in 0..2 {
        let _ = black_box(loader.next_batch());
    }

    group.bench_function("next_batch_1k_tokens", |b| {
        b.iter(|| black_box(loader.next_batch().unwrap()));
    });

    group.finish();
}

// ============================================================
// Group 5: GPU ops (only with cuda feature)
// ============================================================

#[cfg(feature = "cuda")]
fn gpu_ops(c: &mut Criterion) {
    use scry_llm::backend::cuda::{init_gpu, CudaBackend};

    init_gpu(0);
    type Gpu = CudaBackend;

    let mut group = c.benchmark_group("gpu_ops");

    // matmul 768x768 (GPT-2 small scale)
    {
        let a_data = vec![0.01f32; 768 * 768];
        let b_data = vec![0.01f32; 768 * 768];
        let a = Gpu::from_vec(a_data, &Shape::new(&[768, 768]));
        let b = Gpu::from_vec(b_data, &Shape::new(&[768, 768]));
        CudaBackend::synchronize();

        for _ in 0..2 {
            let _ = black_box(Gpu::matmul(&a, &b, 768, 768, 768, false, false));
        }
        CudaBackend::synchronize();

        group.bench_function("matmul_768x768", |bench| {
            bench.iter(|| {
                let r = black_box(Gpu::matmul(
                    black_box(&a),
                    black_box(&b),
                    768, 768, 768, false, false,
                ));
                CudaBackend::synchronize();
                r
            });
        });
    }

    // Full forward pass at GPT-2 small-ish scale
    {
        let config = Gpt2Config {
            vocab_size: 256,
            max_seq_len: 128,
            d_model: 128,
            n_heads: 4,
            n_layers: 2,
            d_ff: 256,
            dropout_rate: 0.0,
        };
        let mut rng = fastrand::Rng::with_seed(42);
        let model = Gpt2Model::<Gpu>::new(config.clone(), &mut rng);
        let tokens: Vec<usize> = (0..64).map(|i| i % config.vocab_size).collect();

        for _ in 0..2 {
            let mut tape = GradTape::<Gpu>::new();
            let _ = black_box(model.forward(&tokens, &mut rng, &mut tape));
            CudaBackend::synchronize();
        }

        group.bench_function("forward_seq64_d128", |b| {
            b.iter(|| {
                let mut tape = GradTape::<Gpu>::new();
                let r = black_box(model.forward(black_box(&tokens), &mut rng, &mut tape));
                CudaBackend::synchronize();
                r
            });
        });
    }

    // Full train step
    {
        let config = Gpt2Config {
            vocab_size: 256,
            max_seq_len: 64,
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            d_ff: 128,
            dropout_rate: 0.0,
        };
        let mut rng = fastrand::Rng::with_seed(42);
        let model = Gpt2Model::<Gpu>::new(config.clone(), &mut rng);
        let training_config = TrainingConfig {
            batch_size: 2,
            seq_len: 16,
            total_steps: 10000,
            warmup_steps: 100,
            peak_lr: 3e-4,
            min_lr: 1e-5,
            grad_accum_steps: 1,
            max_grad_norm: 1.0,
            log_interval: 100_000,
            eval_interval: 0,
            checkpoint_interval: 0,
            checkpoint_dir: std::path::PathBuf::from("/tmp"),
            seed: 42,
            use_checkpointing: false,
            checkpoint_every: 4,
            peak_tflops: None,
            n_params: None,
        };
        let mut trainer = Trainer::<Gpu>::new(model, config.clone(), training_config);

        for _ in 0..2 {
            let batch = make_batch(2, 16, config.vocab_size);
            let _ = black_box(trainer.train_step(&[batch]));
            CudaBackend::synchronize();
        }

        group.bench_function("train_step_gpu", |b| {
            b.iter(|| {
                trainer.step = 0;
                let batch = make_batch(2, 16, config.vocab_size);
                let r = black_box(trainer.train_step(black_box(&[batch])));
                CudaBackend::synchronize();
                r
            });
        });
    }

    group.finish();
}

#[cfg(feature = "cuda")]
criterion_group!(benches, forward_pass, training_step, ops_matmul, ops_micro, data_loader, gpu_ops);
#[cfg(not(feature = "cuda"))]
criterion_group!(benches, forward_pass, training_step, ops_matmul, ops_micro, data_loader);
criterion_main!(benches);
