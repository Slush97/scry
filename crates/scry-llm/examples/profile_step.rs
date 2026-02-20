//! Comprehensive training step profiler.
//!
//! Instruments a single training step with GPU sync barriers to identify
//! exactly where wall-clock time is spent. Also dumps tape operation counts.
//!
//! Usage:
//!   cargo run --example profile_step --release --features cuda
//!   cargo run --example profile_step --release --features cuda -- --batch 32 --seq 128
#![allow(clippy::too_many_lines, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

use scry_llm::autograd::backward::backward;
use scry_llm::autograd::{ops, GradTape, Operation};
use scry_llm::backend::MathBackend;
use scry_llm::data::Batch;
use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};
use scry_llm::nn::Module;
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

fn gpt2_config() -> Gpt2Config {
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

fn op_name(op: &Operation) -> &'static str {
    match op {
        Operation::Matmul => "Matmul",
        Operation::Add => "Add",
        Operation::Softmax => "Softmax",
        Operation::LayerNorm => "LayerNorm",
        Operation::Gelu => "Gelu",
        Operation::CrossEntropy => "CrossEntropy",
        Operation::Embedding => "Embedding",
        Operation::Sum => "Sum",
        Operation::Attention => "Attention",
        Operation::Dropout => "Dropout",
        Operation::Checkpoint => "Checkpoint",
        Operation::AttentionBatched => "AttentionBatched",
        Operation::ResidualAddLayerNorm => "ResidualAddLayerNorm",
        Operation::FusedBiasGelu => "FusedBiasGelu",
        Operation::FusedBiasDropoutResidual => "FusedBiasDropoutRes",
        Operation::FlashAttention => "FlashAttention",
        Operation::CrossEntropyFused => "CrossEntropyFused",
    }
}

fn profile<B: MathBackend>(
    label: &str,
    batch_size: usize,
    seq_len: usize,
    peak_tflops: f64,
    sync_fn: fn(),
) {
    let config = gpt2_config();
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

    // ── Warmup ──
    eprintln!("[{label}] warmup 3 steps...");
    for _ in 0..3 {
        let batch = make_batch(batch_size, seq_len, config.vocab_size);
        let _ = black_box(trainer.train_step(&[batch]));
    }
    sync_fn();

    // ── Full step timing ──
    let batch = make_batch(batch_size, seq_len, config.vocab_size);
    sync_fn();
    let t_full = Instant::now();
    let metrics = trainer.train_step(&[batch]);
    sync_fn();
    let full_ms = t_full.elapsed().as_secs_f64() * 1000.0;

    let tokens = batch_size * seq_len;
    let tok_per_sec = tokens as f64 / (full_ms / 1000.0);
    let flops_per_token = 6.0 * n_params as f64;
    let achieved = tok_per_sec * flops_per_token / 1e12;
    let mfu = achieved / peak_tflops * 100.0;

    eprintln!("\n╔══════════════════════════════════════════════════════════╗");
    eprintln!("║  FULL TRAIN STEP (batch={batch_size}, seq={seq_len})");
    eprintln!("║  step: {full_ms:.2} ms | loss: {:.4} | tok/s: {tok_per_sec:.0} | MFU: {mfu:.1}%",
              metrics.loss);
    eprintln!("╚══════════════════════════════════════════════════════════╝\n");

    // ── Decomposed profiling: forward pass ──
    // We rebuild so we can instrument each component.
    let model = &trainer.model;
    let total = batch_size * seq_len;

    let batch = make_batch(batch_size, seq_len, config.vocab_size);
    let mut tape = GradTape::<B>::new();

    // Embedding
    sync_fn();
    let t0 = Instant::now();
    let mut x = model.embedding.forward_batch(&batch.input_ids, batch_size, seq_len, &mut tape);
    x = ops::dropout(&x, config.dropout_rate, &mut rng, Some(&mut tape));
    sync_fn();
    let emb_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Transformer blocks
    let mut block_ms = Vec::new();
    for (i, block) in model.blocks.iter().enumerate() {
        sync_fn();
        let t0 = Instant::now();
        x = block.forward_batch(&x, batch_size, seq_len, config.dropout_rate, &mut rng, &mut tape);
        sync_fn();
        block_ms.push((i, t0.elapsed().as_secs_f64() * 1000.0));
    }

    // Final LN
    sync_fn();
    let t0 = Instant::now();
    x = model.ln_f.forward(&x, &mut tape);
    sync_fn();
    let ln_f_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // LM head
    sync_fn();
    let t0 = Instant::now();
    let logits = ops::matmul(
        &x, &model.embedding.token_embedding,
        total, config.d_model, config.vocab_size, false, true, Some(&mut tape),
    );
    sync_fn();
    let lm_head_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Cross-entropy
    sync_fn();
    let t0 = Instant::now();
    let loss = ops::cross_entropy(
        &logits, &batch.targets, total, config.vocab_size, Some(&mut tape),
    );
    sync_fn();
    let ce_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let fwd_total = emb_ms + block_ms.iter().map(|(_, t)| t).sum::<f64>() + ln_f_ms + lm_head_ms + ce_ms;

    // Tape stats
    let mut op_counts: HashMap<&str, usize> = HashMap::new();
    for node in &tape.nodes {
        *op_counts.entry(op_name(&node.op)).or_default() += 1;
    }

    // Backward
    sync_fn();
    let t0 = Instant::now();
    let mut grads = backward(&tape, loss.id);
    sync_fn();
    let bwd_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let grad_count = grads.len();

    // Loss D2H transfer (this happens in train_step too)
    sync_fn();
    let t0 = Instant::now();
    let _loss_val = loss.to_vec()[0];
    sync_fn();
    let loss_d2h_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Clip grad norm
    sync_fn();
    let t0 = Instant::now();
    let _grad_norm = scry_llm::optim::clip::clip_grad_norm::<B>(&mut grads, 1.0);
    sync_fn();
    let clip_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Actual optimizer step
    sync_fn();
    let t0 = Instant::now();
    {
        let mut params: Vec<_> = trainer.model
            .parameters_mut()
            .into_iter()
            .map(|p| {
                let id = p.id;
                let shape = p.shape.clone();
                (id, p.data_mut(), shape)
            })
            .collect();
        let mut param_refs: Vec<_> = params
            .iter_mut()
            .map(|(id, data, shape)| (*id, &mut **data, &*shape))
            .collect();
        trainer.optimizer.step(&mut param_refs, &grads, &trainer.no_decay);
    }
    sync_fn();
    let opt_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // ── Report ──
    let decomposed_total = fwd_total + bwd_ms + loss_d2h_ms + clip_ms + opt_ms;

    eprintln!("╔══════════════════════════════════════════════════════════╗");
    eprintln!("║  DECOMPOSED TIMING");
    eprintln!("╠══════════════════════════════════════════════════════════╣");
    eprintln!("║  FORWARD:");
    eprintln!("║    Embedding + dropout     {:>8.2} ms  ({:>5.1}%)", emb_ms, emb_ms / decomposed_total * 100.0);
    for (i, ms) in &block_ms {
        eprintln!("║    Block {:>2}                {:>8.2} ms  ({:>5.1}%)", i, ms, ms / decomposed_total * 100.0);
    }
    eprintln!("║    Final LayerNorm         {:>8.2} ms  ({:>5.1}%)", ln_f_ms, ln_f_ms / decomposed_total * 100.0);
    eprintln!("║    LM Head matmul          {:>8.2} ms  ({:>5.1}%)", lm_head_ms, lm_head_ms / decomposed_total * 100.0);
    eprintln!("║    Cross-entropy           {:>8.2} ms  ({:>5.1}%)", ce_ms, ce_ms / decomposed_total * 100.0);
    eprintln!("║    ─────────────────────────────────────");
    eprintln!("║    Forward subtotal        {:>8.2} ms  ({:>5.1}%)", fwd_total, fwd_total / decomposed_total * 100.0);
    eprintln!("║");
    eprintln!("║  BACKWARD:                 {:>8.2} ms  ({:>5.1}%)", bwd_ms, bwd_ms / decomposed_total * 100.0);
    eprintln!("║  LOSS D2H:                 {:>8.2} ms  ({:>5.1}%)", loss_d2h_ms, loss_d2h_ms / decomposed_total * 100.0);
    eprintln!("║  CLIP GRAD NORM:           {:>8.2} ms  ({:>5.1}%)", clip_ms, clip_ms / decomposed_total * 100.0);
    eprintln!("║  OPTIMIZER ({:>3} params):  {:>8.2} ms  ({:>5.1}%)", grad_count, opt_ms, opt_ms / decomposed_total * 100.0);
    eprintln!("║    ─────────────────────────────────────");
    eprintln!("║    Decomposed total        {:>8.2} ms", decomposed_total);
    eprintln!("║    Full step (measured)     {:>8.2} ms", full_ms);
    eprintln!("║    Overhead / gap           {:>8.2} ms", full_ms - decomposed_total);
    eprintln!("║    Backward / Forward       {:>8.2}x", bwd_ms / fwd_total);
    eprintln!("╠══════════════════════════════════════════════════════════╣");
    eprintln!("║  TAPE OPERATIONS ({} total nodes):", tape.nodes.len());
    let mut sorted_ops: Vec<_> = op_counts.iter().collect();
    sorted_ops.sort_by(|a, b| b.1.cmp(a.1));
    for (name, count) in &sorted_ops {
        eprintln!("║    {:<26} {:>4}", name, count);
    }
    eprintln!("╠══════════════════════════════════════════════════════════╣");

    // Compute theoretical FLOP breakdown
    let d = config.d_model as f64;
    let s = seq_len as f64;
    let b = batch_size as f64;
    let v = config.vocab_size as f64;
    let l = config.n_layers as f64;

    let flops_qkv = 2.0 * b * s * d * 3.0 * d;           // QKV projection
    let flops_attn_scores = 2.0 * b * s * s * d;           // Q@K^T
    let flops_attn_values = 2.0 * b * s * s * d;           // attn@V
    let flops_proj = 2.0 * b * s * d * d;                  // output projection
    let flops_attn_total = flops_qkv + flops_attn_scores + flops_attn_values + flops_proj;

    let flops_fc1 = 2.0 * b * s * d * 4.0 * d;            // MLP fc1
    let flops_fc2 = 2.0 * b * s * 4.0 * d * d;            // MLP fc2
    let flops_mlp_total = flops_fc1 + flops_fc2;

    let flops_lm_head = 2.0 * b * s * d * v;               // LM head

    let flops_per_layer = flops_attn_total + flops_mlp_total;
    let flops_fwd = flops_per_layer * l + flops_lm_head;

    eprintln!("║  FLOP BREAKDOWN (forward only, per step):");
    eprintln!("║    Per-layer attention      {:>8.2} GFLOP ({:.1}%)", flops_attn_total * l / 1e9, flops_attn_total * l / flops_fwd * 100.0);
    eprintln!("║    Per-layer MLP            {:>8.2} GFLOP ({:.1}%)", flops_mlp_total * l / 1e9, flops_mlp_total * l / flops_fwd * 100.0);
    eprintln!("║    LM head                  {:>8.2} GFLOP ({:.1}%)", flops_lm_head / 1e9, flops_lm_head / flops_fwd * 100.0);
    eprintln!("║    Total fwd FLOPs          {:>8.2} GFLOP", flops_fwd / 1e9);

    // Compare time fraction vs FLOP fraction
    let attn_time_frac: f64 = block_ms.iter().map(|(_, t)| t).sum::<f64>() / decomposed_total;
    let lm_head_time_frac = lm_head_ms / decomposed_total;
    eprintln!("║");
    eprintln!("║  TIME vs FLOP EFFICIENCY:");
    eprintln!("║    Blocks wall time         {:>8.2} ms  ({:>5.1}% of total)", block_ms.iter().map(|(_, t)| t).sum::<f64>(), attn_time_frac * 100.0);
    eprintln!("║    LM head wall time        {:>8.2} ms  ({:>5.1}% of total)", lm_head_ms, lm_head_time_frac * 100.0);

    // Effective TFLOPS per component
    let blocks_time_s = block_ms.iter().map(|(_, t)| t).sum::<f64>() / 1000.0;
    let lm_head_time_s = lm_head_ms / 1000.0;
    // Forward blocks do fwd only, backward does ~2x more
    let blocks_fwd_tflops = if blocks_time_s > 0.0 { (flops_per_layer * l) / 1e12 / blocks_time_s } else { 0.0 };
    let lm_head_tflops = if lm_head_time_s > 0.0 { flops_lm_head / 1e12 / lm_head_time_s } else { 0.0 };

    eprintln!("║    Blocks effective TFLOPS   {:>8.2} (of {peak_tflops:.1} peak)", blocks_fwd_tflops);
    eprintln!("║    LM head effective TFLOPS  {:>8.2} (of {peak_tflops:.1} peak)", lm_head_tflops);
    eprintln!("╚══════════════════════════════════════════════════════════╝");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut batch_size = 4_usize;
    let mut seq_len = 128_usize;
    let mut peak_tflops = 23.1_f64;
    let mut use_bf16 = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--batch" => { batch_size = args[i + 1].parse().expect("invalid --batch"); i += 2; }
            "--seq" => { seq_len = args[i + 1].parse().expect("invalid --seq"); i += 2; }
            "--peak-tflops" => { peak_tflops = args[i + 1].parse().expect("invalid --peak-tflops"); i += 2; }
            "--bf16" => { use_bf16 = true; i += 1; }
            "--help" | "-h" => {
                eprintln!("Usage: profile_step [--batch N] [--seq N] [--peak-tflops F] [--bf16]");
                std::process::exit(0);
            }
            other => { eprintln!("Unknown arg: {other}"); std::process::exit(1); }
        }
    }

    // NOTE: our bf16 kernels cast to f32 for compute (no tensor cores yet),
    // so peak remains the FP32 rate. The bf16 win is memory bandwidth only.

    eprintln!("=== Training Step Profiler ===");
    eprintln!("GPT-2: d=768, h=12, L=6, d_ff=3072, V=50257");
    eprintln!("batch={batch_size}, seq={seq_len}, peak={peak_tflops} TFLOPS\n");

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
        fn cuda_sync() { CudaBackend::synchronize(); }

        let label = if use_bf16 { "GPU bf16" } else { "GPU" };
        profile::<CudaBackend>(label, batch_size, seq_len, peak_tflops, cuda_sync);
    }

    #[cfg(not(feature = "cuda"))]
    {
        use scry_llm::backend::cpu::CpuBackend;
        fn noop() {}
        profile::<CpuBackend>("CPU", batch_size, seq_len, peak_tflops, noop);
    }
}
