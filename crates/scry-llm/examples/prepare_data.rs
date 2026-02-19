//! Tokenize text files into packed u16 LE binary shards for `DataLoader`.
//!
//! ```bash
//! cargo run --example prepare_data -p scry-llm --no-default-features --features tokenizer -- \
//!   --vocab crates/scry-llm/tests/fixtures/vocab.json \
//!   --merges crates/scry-llm/tests/fixtures/merges.txt \
//!   --input data/corpus/ \
//!   --output data/shards/ \
//!   --shard-size 100000000 \
//!   --val-ratio 0.01
//! ```

use std::path::{Path, PathBuf};

use scry_llm::tokenizer::BpeTokenizer;

const EOT_TOKEN: u16 = 50256;

struct Config {
    vocab_path: PathBuf,
    merges_path: PathBuf,
    input_path: PathBuf,
    output_dir: PathBuf,
    shard_size: usize,
    val_ratio: f64,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut vocab_path = PathBuf::from("vocab.json");
    let mut merges_path = PathBuf::from("merges.txt");
    let mut input_path = PathBuf::from("data/corpus");
    let mut output_dir = PathBuf::from("data/shards");
    let mut shard_size: usize = 100_000_000;
    let mut val_ratio: f64 = 0.01;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--vocab" => {
                vocab_path = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--merges" => {
                merges_path = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--input" => {
                input_path = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--output" => {
                output_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--shard-size" => {
                shard_size = args[i + 1].parse().expect("invalid --shard-size");
                i += 2;
            }
            "--val-ratio" => {
                val_ratio = args[i + 1].parse().expect("invalid --val-ratio");
                i += 2;
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: prepare_data [OPTIONS]\n\n\
                     Options:\n  \
                       --vocab PATH       vocab.json path\n  \
                       --merges PATH      merges.txt path\n  \
                       --input PATH       input text file or directory\n  \
                       --output PATH      output shard directory\n  \
                       --shard-size N     tokens per shard (default: 100000000)\n  \
                       --val-ratio F      fraction of docs for validation (default: 0.01)"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(1);
            }
        }
    }

    Config {
        vocab_path,
        merges_path,
        input_path,
        output_dir,
        shard_size,
        val_ratio,
    }
}

/// Collect `.txt` files from a path (file or directory).
fn collect_txt_files(path: &Path) -> Vec<PathBuf> {
    if path.is_file() {
        return vec![path.to_path_buf()];
    }
    let mut files = Vec::new();
    collect_txt_recursive(path, &mut files);
    files.sort();
    files
}

fn collect_txt_recursive(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("warning: cannot read {}: {e}", dir.display());
            return;
        }
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_txt_recursive(&p, out);
        } else if p.extension().and_then(|e| e.to_str()) == Some("txt") {
            out.push(p);
        }
    }
}

/// Write a shard of tokens as packed u16 LE bytes.
fn write_shard(dir: &Path, prefix: &str, index: usize, tokens: &[u16]) {
    let path = dir.join(format!("{prefix}_{index:04}.bin"));
    let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    std::fs::write(&path, &bytes).unwrap_or_else(|e| {
        panic!("failed to write {}: {e}", path.display());
    });
    eprintln!(
        "  wrote {} ({} tokens, {:.1} MB)",
        path.display(),
        tokens.len(),
        bytes.len() as f64 / 1_048_576.0
    );
}

fn flush_shards(dir: &Path, prefix: &str, tokens: &[u16], shard_size: usize) -> usize {
    if tokens.is_empty() {
        return 0;
    }
    let mut shard_count = 0;
    for chunk in tokens.chunks(shard_size) {
        write_shard(dir, prefix, shard_count, chunk);
        shard_count += 1;
    }
    shard_count
}

fn main() {
    let config = parse_args();

    eprintln!("Loading tokenizer...");
    let tokenizer =
        BpeTokenizer::from_files(&config.vocab_path, &config.merges_path).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1);
        });
    eprintln!("  vocab size: {}", tokenizer.vocab_size());

    let txt_files = collect_txt_files(&config.input_path);
    if txt_files.is_empty() {
        eprintln!("error: no .txt files found in {}", config.input_path.display());
        std::process::exit(1);
    }
    eprintln!("Found {} text file(s)", txt_files.len());

    // Read and tokenize all documents
    let mut documents: Vec<Vec<u16>> = Vec::new();
    for path in &txt_files {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("warning: skipping {}: {e}", path.display());
                continue;
            }
        };
        if text.is_empty() {
            continue;
        }
        let token_ids: Vec<u16> = tokenizer
            .encode(&text)
            .into_iter()
            .map(|id| id as u16)
            .collect();
        documents.push(token_ids);
    }

    eprintln!("Tokenized {} documents", documents.len());

    // Split documents into train/val
    let n_val = ((documents.len() as f64) * config.val_ratio).ceil() as usize;
    let n_val = n_val.max(if documents.len() > 1 && config.val_ratio > 0.0 { 1 } else { 0 });
    let n_train = documents.len() - n_val;

    let train_docs = &documents[..n_train];
    let val_docs = &documents[n_train..];

    // Concatenate with EOT separators
    let concat = |docs: &[Vec<u16>]| -> Vec<u16> {
        let mut tokens = Vec::new();
        for (i, doc) in docs.iter().enumerate() {
            tokens.extend_from_slice(doc);
            if i + 1 < docs.len() {
                tokens.push(EOT_TOKEN);
            }
        }
        tokens
    };

    let train_tokens = concat(train_docs);
    let val_tokens = concat(val_docs);

    // Create output directory
    std::fs::create_dir_all(&config.output_dir).unwrap_or_else(|e| {
        eprintln!("error: cannot create output dir: {e}");
        std::process::exit(1);
    });

    // Write shards
    eprintln!("\nWriting train shards...");
    let train_shards = flush_shards(&config.output_dir, "train", &train_tokens, config.shard_size);

    let mut val_shards = 0;
    if !val_tokens.is_empty() {
        eprintln!("Writing val shards...");
        val_shards = flush_shards(&config.output_dir, "val", &val_tokens, config.shard_size);
    }

    eprintln!("\n--- Summary ---");
    eprintln!("Documents:    {} total ({} train, {} val)", documents.len(), n_train, n_val);
    eprintln!("Train tokens: {}", train_tokens.len());
    eprintln!("Val tokens:   {}", val_tokens.len());
    eprintln!("Train shards: {}", train_shards);
    eprintln!("Val shards:   {}", val_shards);
    eprintln!("Output dir:   {}", config.output_dir.display());
}
