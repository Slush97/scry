//! Quick MFU benchmark: runs N training steps on synthetic data,
//! reports tok/s and MFU% for batch=4 and batch=32.
//!
//! Usage:
//!   cargo run --example mfu_bench --release --features cuda
//!   cargo run --example mfu_bench --release --features cuda -- --steps 20 --peak-tflops 88
#![allow(clippy::too_many_lines)]

use std::hint::black_box;
use std::time::Instant;

use scry_llm::backend::MathBackend;
use scry_llm::data::Batch;
use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};
use scry_llm::training::{Trainer, TrainingConfig};

fn make_batch(batch_size: usize, seq_len: usize, vocab: usize) -> Batch {
    let total = batch_size * seq_len;
    Batch {
        input_ids: (0..total).map(|i| i % vocab).collect(),
        targets: (0..total).map(|i| (i + 1) % vocab).collect(),
        batch_size,
        seq_len,
    }
}

fn bench_config() -> Gpt2Config {
    Gpt2Config {
        vocab_size: 50257,
        max_seq_len: 1024,
        d_model: 768,
        n_heads: 12,
        n_layers: 6,
        d_ff: 3072,
        dropout_rate: 0.0,
    }
}

fn run_bench<B: MathBackend>(
    label: &str,
    batch_size: usize,
    seq_len: usize,
    n_steps: usize,
    n_warmup: usize,
    peak_tflops: f64,
    sync_fn: fn(),
) {
    let config = bench_config();
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<B>::new(config.clone(), &mut rng);
    let n_params = model.n_params();

    let training_config = TrainingConfig {
        batch_size,
        seq_len,
        total_steps: 100_000,
        warmup_steps: 0,
        peak_lr: 3e-4,
        min_lr: 3e-4,
        grad_accum_steps: 1,
        max_grad_norm: 1.0,
        log_interval: 100_000,
        eval_interval: 0,
        checkpoint_interval: 0,
        checkpoint_dir: std::path::PathBuf::from("/tmp"),
        seed: 42,
        use_checkpointing: false,
        checkpoint_every: 6,
        peak_tflops: Some(peak_tflops),
        n_params: Some(n_params),
    };
    let mut trainer = Trainer::<B>::new(model, config.clone(), training_config);

    eprintln!("[{label}] n_params={n_params}, batch={batch_size}, seq={seq_len}");
    eprintln!("[{label}] warmup {n_warmup} steps...");

    // Warmup
    for _ in 0..n_warmup {
        let batch = make_batch(batch_size, seq_len, config.vocab_size);
        let _ = black_box(trainer.train_step(&[batch]));
    }
    sync_fn();

    eprintln!("[{label}] timing {n_steps} steps...");
    let start = Instant::now();
    let mut total_loss = 0.0f32;

    for _ in 0..n_steps {
        let batch = make_batch(batch_size, seq_len, config.vocab_size);
        let metrics = trainer.train_step(&[batch]);
        total_loss += metrics.loss;
    }
    sync_fn();

    let elapsed = start.elapsed().as_secs_f64();
    let tokens_per_step = batch_size * seq_len;
    let total_tokens = n_steps * tokens_per_step;
    let tok_per_sec = total_tokens as f64 / elapsed;

    let flops_per_token = 6.0 * n_params as f64;
    let achieved_tflops = tok_per_sec * flops_per_token / 1e12;
    let mfu = achieved_tflops / peak_tflops * 100.0;

    eprintln!("────────────────────────────────────────");
    eprintln!("[{label}] {n_steps} steps in {elapsed:.3}s");
    eprintln!("[{label}] avg loss: {:.4}", total_loss / n_steps as f32);
    eprintln!("[{label}] tok/s: {tok_per_sec:.0}");
    eprintln!("[{label}] achieved: {achieved_tflops:.2} TFLOPS");
    eprintln!("[{label}] MFU: {mfu:.1}%");
    eprintln!("────────────────────────────────────────");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut n_steps = 10;
    let mut peak_tflops = 23.1_f64; // RTX 5070 Ti FP32
    let mut seq_len = 128_usize;
    let mut use_bf16 = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => {
                n_steps = args[i + 1].parse().expect("invalid --steps");
                i += 2;
            }
            "--peak-tflops" => {
                peak_tflops = args[i + 1].parse().expect("invalid --peak-tflops");
                i += 2;
            }
            "--seq-len" => {
                seq_len = args[i + 1].parse().expect("invalid --seq-len");
                i += 2;
            }
            "--bf16" => {
                use_bf16 = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!("Usage: mfu_bench [--steps N] [--peak-tflops F] [--seq-len S] [--bf16]");
                eprintln!("  --steps N         training steps per config (default: 10)");
                eprintln!("  --peak-tflops F   GPU peak TFLOPS (default: 23.1 for FP32)");
                eprintln!("  --seq-len S       sequence length (default: 128)");
                eprintln!("  --bf16            use bf16 tensor cores (default: fp32)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }

    // NOTE: our bf16 kernels cast to f32 for compute (no tensor cores yet),
    // so peak remains the FP32 rate. The bf16 win is memory bandwidth only.
    // Once we add wmma/mma.sync intrinsics, switch to 88.0 for bf16.

    let n_warmup = 3;
    let mode = if use_bf16 { "bf16" } else { "fp32" };

    eprintln!("=== MFU Benchmark ({mode}) ===");
    eprintln!("GPT-2 config: d=768, h=12, L=6, d_ff=3072, V=50257");
    eprintln!("peak_tflops={peak_tflops}, seq_len={seq_len}, steps={n_steps}");
    eprintln!();

    #[cfg(feature = "cuda")]
    {
        use scry_llm::backend::cuda::{init_gpu, CudaBackend};

        if use_bf16 {
            #[cfg(feature = "bf16")]
            scry_llm::backend::cuda::init_gpu_bf16(0);
            #[cfg(not(feature = "bf16"))]
            {
                eprintln!("ERROR: --bf16 requires the 'bf16' feature. Recompile with --features cuda,bf16");
                std::process::exit(1);
            }
        } else {
            init_gpu(0);
        }

        fn cuda_sync() {
            scry_llm::backend::cuda::CudaBackend::synchronize();
        }

        run_bench::<CudaBackend>("GPU batch=4", 4, seq_len, n_steps, n_warmup, peak_tflops, cuda_sync);
        eprintln!();
        run_bench::<CudaBackend>("GPU batch=32", 32, seq_len, n_steps, n_warmup, peak_tflops, cuda_sync);
    }

    #[cfg(not(feature = "cuda"))]
    {
        use scry_llm::backend::cpu::CpuBackend;
        eprintln!("(no CUDA — running on CPU)");

        fn noop_sync() {}

        run_bench::<CpuBackend>("CPU batch=4", 4, seq_len, n_steps, n_warmup, peak_tflops, noop_sync);
    }
}
