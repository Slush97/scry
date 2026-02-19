//! GPT-2 training entry point.
//!
//! ```bash
//! cargo run --example train_gpt2 -p scry-llm --no-default-features --release --features safetensors -- \
//!   --data-dir data/shards/ \
//!   --backend cpu \
//!   --total-steps 10000 \
//!   --batch-size 4 \
//!   --seq-len 256 \
//!   --lr 6e-4 \
//!   --warmup-steps 200 \
//!   --checkpoint-dir checkpoints/
//! ```

use std::path::PathBuf;

use scry_llm::data::DataLoader;
use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};
use scry_llm::training::{TrainingConfig, Trainer};

struct Config {
    data_dir: PathBuf,
    backend: String,
    total_steps: usize,
    batch_size: usize,
    seq_len: usize,
    lr: f32,
    min_lr: f32,
    warmup_steps: usize,
    grad_accum_steps: usize,
    max_grad_norm: f32,
    log_interval: usize,
    eval_interval: usize,
    checkpoint_interval: usize,
    checkpoint_dir: PathBuf,
    seed: u64,
    // Model dimensions
    d_model: usize,
    n_heads: usize,
    n_layers: usize,
    d_ff: Option<usize>,
    vocab_size: usize,
    max_seq_len: usize,
    dropout: f32,
    // Init modes
    from_pretrained: Option<PathBuf>,
    resume: Option<PathBuf>,
    // Generation sampling
    sample_interval: usize,
    sample_prompt: String,
    sample_max_tokens: usize,
    sample_temperature: f32,
    // Tokenizer (for decoding samples)
    vocab_path: Option<PathBuf>,
    merges_path: Option<PathBuf>,
    // Gradient checkpointing
    use_checkpointing: bool,
    checkpoint_every: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("data/shards"),
            backend: String::from("cpu"),
            total_steps: 10_000,
            batch_size: 4,
            seq_len: 256,
            lr: 6e-4,
            min_lr: 6e-5,
            warmup_steps: 200,
            grad_accum_steps: 1,
            max_grad_norm: 1.0,
            log_interval: 10,
            eval_interval: 500,
            checkpoint_interval: 1000,
            checkpoint_dir: PathBuf::from("checkpoints"),
            seed: 42,
            d_model: 768,
            n_heads: 12,
            n_layers: 12,
            d_ff: None, // default: 4 * d_model
            vocab_size: 50257,
            max_seq_len: 1024,
            dropout: 0.1,
            from_pretrained: None,
            resume: None,
            sample_interval: 0,
            sample_prompt: String::from("The"),
            sample_max_tokens: 64,
            sample_temperature: 0.8,
            vocab_path: None,
            merges_path: None,
            use_checkpointing: false,
            checkpoint_every: 3,
        }
    }
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = Config::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data-dir" => { cfg.data_dir = PathBuf::from(&args[i + 1]); i += 2; }
            "--backend" => { cfg.backend = args[i + 1].clone(); i += 2; }
            "--total-steps" => { cfg.total_steps = args[i + 1].parse().expect("invalid --total-steps"); i += 2; }
            "--batch-size" => { cfg.batch_size = args[i + 1].parse().expect("invalid --batch-size"); i += 2; }
            "--seq-len" => { cfg.seq_len = args[i + 1].parse().expect("invalid --seq-len"); i += 2; }
            "--lr" => { cfg.lr = args[i + 1].parse().expect("invalid --lr"); i += 2; }
            "--min-lr" => { cfg.min_lr = args[i + 1].parse().expect("invalid --min-lr"); i += 2; }
            "--warmup-steps" => { cfg.warmup_steps = args[i + 1].parse().expect("invalid --warmup-steps"); i += 2; }
            "--grad-accum" => { cfg.grad_accum_steps = args[i + 1].parse().expect("invalid --grad-accum"); i += 2; }
            "--max-grad-norm" => { cfg.max_grad_norm = args[i + 1].parse().expect("invalid --max-grad-norm"); i += 2; }
            "--log-interval" => { cfg.log_interval = args[i + 1].parse().expect("invalid --log-interval"); i += 2; }
            "--eval-interval" => { cfg.eval_interval = args[i + 1].parse().expect("invalid --eval-interval"); i += 2; }
            "--checkpoint-interval" => { cfg.checkpoint_interval = args[i + 1].parse().expect("invalid --checkpoint-interval"); i += 2; }
            "--checkpoint-dir" => { cfg.checkpoint_dir = PathBuf::from(&args[i + 1]); i += 2; }
            "--seed" => { cfg.seed = args[i + 1].parse().expect("invalid --seed"); i += 2; }
            "--d-model" => { cfg.d_model = args[i + 1].parse().expect("invalid --d-model"); i += 2; }
            "--n-heads" => { cfg.n_heads = args[i + 1].parse().expect("invalid --n-heads"); i += 2; }
            "--n-layers" => { cfg.n_layers = args[i + 1].parse().expect("invalid --n-layers"); i += 2; }
            "--d-ff" => { cfg.d_ff = Some(args[i + 1].parse().expect("invalid --d-ff")); i += 2; }
            "--vocab-size" => { cfg.vocab_size = args[i + 1].parse().expect("invalid --vocab-size"); i += 2; }
            "--max-seq-len" => { cfg.max_seq_len = args[i + 1].parse().expect("invalid --max-seq-len"); i += 2; }
            "--dropout" => { cfg.dropout = args[i + 1].parse().expect("invalid --dropout"); i += 2; }
            "--from-pretrained" => { cfg.from_pretrained = Some(PathBuf::from(&args[i + 1])); i += 2; }
            "--resume" => { cfg.resume = Some(PathBuf::from(&args[i + 1])); i += 2; }
            "--sample-interval" => { cfg.sample_interval = args[i + 1].parse().expect("invalid --sample-interval"); i += 2; }
            "--sample-prompt" => { cfg.sample_prompt = args[i + 1].clone(); i += 2; }
            "--sample-max-tokens" => { cfg.sample_max_tokens = args[i + 1].parse().expect("invalid --sample-max-tokens"); i += 2; }
            "--sample-temperature" => { cfg.sample_temperature = args[i + 1].parse().expect("invalid --sample-temperature"); i += 2; }
            "--vocab" => { cfg.vocab_path = Some(PathBuf::from(&args[i + 1])); i += 2; }
            "--merges" => { cfg.merges_path = Some(PathBuf::from(&args[i + 1])); i += 2; }
            "--grad-checkpoint" => { cfg.use_checkpointing = true; i += 1; }
            "--checkpoint-every" => { cfg.checkpoint_every = args[i + 1].parse().expect("invalid --checkpoint-every"); i += 2; }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(1);
            }
        }
    }

    cfg
}

fn print_help() {
    eprintln!(
        "Usage: train_gpt2 [OPTIONS]\n\n\
         Data:\n  \
           --data-dir PATH          shard directory (default: data/shards)\n\n\
         Backend:\n  \
           --backend cpu|cuda       compute backend (default: cpu)\n\n\
         Training:\n  \
           --total-steps N          total training steps (default: 10000)\n  \
           --batch-size N           batch size (default: 4)\n  \
           --seq-len N              sequence length (default: 256)\n  \
           --lr F                   peak learning rate (default: 6e-4)\n  \
           --min-lr F               minimum learning rate (default: 6e-5)\n  \
           --warmup-steps N         warmup steps (default: 200)\n  \
           --grad-accum N           gradient accumulation steps (default: 1)\n  \
           --max-grad-norm F        gradient clipping norm (default: 1.0)\n  \
           --seed N                 random seed (default: 42)\n\n\
         Model:\n  \
           --d-model N              model dimension (default: 768)\n  \
           --n-heads N              attention heads (default: 12)\n  \
           --n-layers N             transformer layers (default: 12)\n  \
           --d-ff N                 feedforward dim (default: 4*d-model)\n  \
           --vocab-size N           vocabulary size (default: 50257)\n  \
           --max-seq-len N          max sequence length (default: 1024)\n  \
           --dropout F              dropout rate (default: 0.1)\n  \
           --from-pretrained PATH   load HF safetensors weights\n  \
           --resume PATH            resume from checkpoint\n  \
           --grad-checkpoint        enable gradient checkpointing\n  \
           --checkpoint-every N     blocks per checkpoint segment (default: 3)\n\n\
         Logging:\n  \
           --log-interval N         log every N steps (default: 10)\n  \
           --eval-interval N        evaluate every N steps (default: 500)\n  \
           --checkpoint-interval N  save checkpoint every N steps (default: 1000)\n  \
           --checkpoint-dir PATH    checkpoint directory (default: checkpoints)\n\n\
         Sampling:\n  \
           --sample-interval N      generate sample every N steps (0=off, default: 0)\n  \
           --sample-prompt TEXT     prompt for generation (default: \"The\")\n  \
           --sample-max-tokens N    max tokens to generate (default: 64)\n  \
           --sample-temperature F   sampling temperature (default: 0.8)\n  \
           --vocab PATH             vocab.json for decoding samples\n  \
           --merges PATH            merges.txt for decoding samples"
    );
}

fn run_training<B: scry_llm::backend::MathBackend>(cfg: &Config) {
    let model_config = Gpt2Config {
        vocab_size: cfg.vocab_size,
        max_seq_len: cfg.max_seq_len,
        d_model: cfg.d_model,
        n_heads: cfg.n_heads,
        n_layers: cfg.n_layers,
        d_ff: cfg.d_ff.unwrap_or(4 * cfg.d_model),
        dropout_rate: cfg.dropout,
    };

    let training_config = TrainingConfig {
        batch_size: cfg.batch_size,
        seq_len: cfg.seq_len,
        total_steps: cfg.total_steps,
        warmup_steps: cfg.warmup_steps,
        peak_lr: cfg.lr,
        min_lr: cfg.min_lr,
        grad_accum_steps: cfg.grad_accum_steps,
        max_grad_norm: cfg.max_grad_norm,
        log_interval: cfg.log_interval,
        eval_interval: cfg.eval_interval,
        checkpoint_interval: cfg.checkpoint_interval,
        checkpoint_dir: cfg.checkpoint_dir.clone(),
        seed: cfg.seed,
        use_checkpointing: cfg.use_checkpointing,
        checkpoint_every: cfg.checkpoint_every,
    };

    // Print config summary
    eprintln!("=== GPT-2 Training ===");
    eprintln!("Backend:       {}", cfg.backend);
    eprintln!("Model:         d_model={}, n_heads={}, n_layers={}, d_ff={}",
        model_config.d_model, model_config.n_heads, model_config.n_layers, model_config.d_ff);
    eprintln!("Vocab:         {}, max_seq_len={}", model_config.vocab_size, model_config.max_seq_len);
    eprintln!("Training:      steps={}, batch={}, seq_len={}, grad_accum={}",
        cfg.total_steps, cfg.batch_size, cfg.seq_len, cfg.grad_accum_steps);
    eprintln!("LR:            peak={:.2e}, min={:.2e}, warmup={}", cfg.lr, cfg.min_lr, cfg.warmup_steps);
    eprintln!("Grad clip:     {:.1}", cfg.max_grad_norm);
    eprintln!("Dropout:       {:.2}", cfg.dropout);
    if cfg.use_checkpointing {
        eprintln!("Checkpointing: every {} blocks", cfg.checkpoint_every);
    }
    eprintln!("Data dir:      {}", cfg.data_dir.display());
    eprintln!("Seed:          {}", cfg.seed);
    eprintln!();

    // Initialize model
    let mut rng = fastrand::Rng::with_seed(cfg.seed);

    let mut trainer = if let Some(ref path) = cfg.resume {
        #[cfg(feature = "safetensors")]
        {
            eprintln!("Resuming from checkpoint: {}", path.display());
            Trainer::<B>::from_checkpoint(path, model_config.clone(), training_config)
                .unwrap_or_else(|e| {
                    eprintln!("error loading checkpoint: {e}");
                    std::process::exit(1);
                })
        }
        #[cfg(not(feature = "safetensors"))]
        {
            let _ = path;
            eprintln!("error: --resume requires the 'safetensors' feature");
            std::process::exit(1);
        }
    } else if let Some(ref path) = cfg.from_pretrained {
        #[cfg(feature = "safetensors")]
        {
            eprintln!("Loading pretrained weights: {}", path.display());
            let model = Gpt2Model::<B>::load_safetensors(model_config.clone(), path)
                .unwrap_or_else(|e| {
                    eprintln!("error loading weights: {e}");
                    std::process::exit(1);
                });
            Trainer::new(model, model_config.clone(), training_config)
        }
        #[cfg(not(feature = "safetensors"))]
        {
            let _ = path;
            eprintln!("error: --from-pretrained requires the 'safetensors' feature");
            std::process::exit(1);
        }
    } else {
        let n_params: usize = {
            // Rough param count for logging
            let d = model_config.d_model;
            let v = model_config.vocab_size;
            let s = model_config.max_seq_len;
            let ff = model_config.d_ff;
            let nl = model_config.n_layers;
            let embed = v * d + s * d;
            let per_block = 4 * d * d + 4 * d   // attn (qkv + proj + biases)
                          + 2 * d                 // ln1
                          + d * ff + ff + ff * d + d  // mlp
                          + 2 * d;                // ln2
            let final_ln = 2 * d;
            embed + nl * per_block + final_ln
        };
        eprintln!("Initializing fresh model (~{:.1}M params)", n_params as f64 / 1e6);
        let model = Gpt2Model::<B>::new(model_config.clone(), &mut rng);
        Trainer::new(model, model_config.clone(), training_config)
    };

    // Load data
    eprintln!("Loading train shards from {}...", cfg.data_dir.display());
    let mut train_loader =
        DataLoader::new(&cfg.data_dir, "train", cfg.seq_len, cfg.batch_size, cfg.seed)
            .unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });

    let mut val_loader =
        DataLoader::new(&cfg.data_dir, "val", cfg.seq_len, cfg.batch_size, cfg.seed + 1).ok();
    if val_loader.is_some() {
        eprintln!("Loaded validation shards");
    } else {
        eprintln!("No validation shards found (val_*.bin), skipping eval");
    }

    // Load tokenizer for generation sampling (optional)
    #[cfg(feature = "tokenizer")]
    let tokenizer = if cfg.sample_interval > 0 {
        match (&cfg.vocab_path, &cfg.merges_path) {
            (Some(v), Some(m)) => {
                match scry_llm::tokenizer::BpeTokenizer::from_files(v, m) {
                    Ok(t) => Some(t),
                    Err(e) => {
                        eprintln!("warning: cannot load tokenizer for sampling: {e}");
                        None
                    }
                }
            }
            _ => {
                eprintln!("warning: --sample-interval requires --vocab and --merges for decoding");
                None
            }
        }
    } else {
        None
    };

    eprintln!();

    // Run training with periodic generation samples
    if cfg.sample_interval > 0 {
        run_training_with_samples::<B>(
            &mut trainer,
            &mut train_loader,
            val_loader.as_mut(),
            cfg,
            #[cfg(feature = "tokenizer")]
            tokenizer.as_ref(),
        );
    } else {
        trainer
            .run(&mut train_loader, val_loader.as_mut())
            .unwrap_or_else(|e| {
                eprintln!("training error: {e}");
                std::process::exit(1);
            });
    }
}

fn run_training_with_samples<B: scry_llm::backend::MathBackend>(
    trainer: &mut Trainer<B>,
    train_loader: &mut DataLoader,
    mut val_loader: Option<&mut DataLoader>,
    cfg: &Config,
    #[cfg(feature = "tokenizer")] tokenizer: Option<&scry_llm::tokenizer::BpeTokenizer>,
) {
    use scry_llm::generate::{generate, SamplingConfig};

    let sampling_config = SamplingConfig {
        temperature: cfg.sample_temperature,
        top_k: 40,
        top_p: 0.95,
        max_tokens: cfg.sample_max_tokens,
    };

    // Encode the prompt
    #[cfg(feature = "tokenizer")]
    let prompt_tokens: Vec<usize> = tokenizer
        .map(|t| t.encode(&cfg.sample_prompt))
        .unwrap_or_else(|| vec![464]); // "The" in GPT-2 vocab

    #[cfg(not(feature = "tokenizer"))]
    let prompt_tokens: Vec<usize> = vec![464]; // "The"

    let start = std::time::Instant::now();

    while trainer.step < trainer.config.total_steps {
        let mut micro_batches = Vec::with_capacity(trainer.config.grad_accum_steps);
        for _ in 0..trainer.config.grad_accum_steps {
            micro_batches.push(train_loader.next_batch().unwrap_or_else(|e| {
                eprintln!("data error: {e}");
                std::process::exit(1);
            }));
        }

        let metrics = trainer.train_step(&micro_batches);

        // Logging
        if trainer.step % trainer.config.log_interval == 0 || trainer.step == 1 {
            let elapsed = start.elapsed().as_secs_f64();
            let tokens_per_step = trainer.config.batch_size
                * trainer.config.seq_len
                * trainer.config.grad_accum_steps;
            let tokens_per_sec = (trainer.step * tokens_per_step) as f64 / elapsed;
            eprintln!(
                "step {:>6} | loss {:.4} | ppl {:>8.2} | grad_norm {:.4} | lr {:.2e} | tok/s {:.0}",
                trainer.step,
                metrics.loss,
                metrics.loss.exp(),
                metrics.grad_norm,
                metrics.lr,
                tokens_per_sec,
            );
        }

        // Evaluation
        if let Some(ref mut val) = val_loader {
            if trainer.config.eval_interval > 0 && trainer.step % trainer.config.eval_interval == 0
            {
                let mut val_batches = Vec::new();
                for _ in 0..10 {
                    if let Ok(batch) = val.next_batch() {
                        val_batches.push(batch);
                    }
                }
                let val_loss = trainer.evaluate(&val_batches);
                eprintln!("  val_loss {:.4} | val_ppl {:.2}", val_loss, val_loss.exp());
            }
        }

        // Generation sample
        if cfg.sample_interval > 0 && trainer.step % cfg.sample_interval == 0 {
            let mut gen_rng = fastrand::Rng::with_seed(cfg.seed.wrapping_add(trainer.step as u64));
            let tokens = generate(&trainer.model, &prompt_tokens, &sampling_config, &mut gen_rng);

            #[cfg(feature = "tokenizer")]
            if let Some(tok) = tokenizer {
                let mut all_tokens = prompt_tokens.clone();
                all_tokens.extend_from_slice(&tokens);
                let text = tok.decode(&all_tokens);
                eprintln!("  sample: {text}");
            } else {
                eprintln!("  sample tokens: {tokens:?}");
            }

            #[cfg(not(feature = "tokenizer"))]
            eprintln!("  sample tokens: {tokens:?}");
        }

        // Checkpointing
        #[cfg(feature = "safetensors")]
        if trainer.config.checkpoint_interval > 0
            && trainer.step % trainer.config.checkpoint_interval == 0
        {
            let path = trainer
                .config
                .checkpoint_dir
                .join(format!("step_{}.safetensors", trainer.step));
            if let Err(e) = std::fs::create_dir_all(&trainer.config.checkpoint_dir) {
                eprintln!("  warning: cannot create checkpoint dir: {e}");
            } else {
                match scry_llm::checkpoint::save_checkpoint(
                    &path,
                    &trainer.model,
                    &trainer.optimizer,
                    trainer.step,
                    trainer.rng.u64(..),
                ) {
                    Ok(()) => eprintln!("  saved checkpoint: {}", path.display()),
                    Err(e) => eprintln!("  warning: checkpoint save failed: {e}"),
                }
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    eprintln!(
        "Training complete: {} steps in {:.1}s",
        trainer.step, elapsed
    );
}

fn main() {
    let cfg = parse_args();

    match cfg.backend.as_str() {
        "cpu" => run_training::<scry_llm::backend::cpu::CpuBackend>(&cfg),
        #[cfg(feature = "cuda")]
        "cuda" => {
            scry_llm::backend::cuda::init_gpu();
            run_training::<scry_llm::backend::cuda::CudaBackend>(&cfg);
        }
        #[cfg(not(feature = "cuda"))]
        "cuda" => {
            eprintln!("error: CUDA backend not available (compile with --features cuda)");
            std::process::exit(1);
        }
        other => {
            eprintln!("error: unknown backend '{other}' (use 'cpu' or 'cuda')");
            std::process::exit(1);
        }
    }
}
