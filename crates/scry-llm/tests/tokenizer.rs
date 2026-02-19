#![cfg(feature = "tokenizer")]

use scry_llm::tokenizer::BpeTokenizer;
use std::path::PathBuf;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load_tokenizer() -> Option<BpeTokenizer> {
    let vocab = fixture_dir().join("vocab.json");
    let merges = fixture_dir().join("merges.txt");
    if !vocab.exists() || !merges.exists() {
        eprintln!(
            "Skipping tokenizer test: download vocab.json and merges.txt from \
             https://huggingface.co/openai-community/gpt2/tree/main into tests/fixtures/"
        );
        return None;
    }
    Some(BpeTokenizer::from_files(&vocab, &merges).expect("failed to load tokenizer"))
}

#[test]
fn known_tokenizations() {
    let Some(tok) = load_tokenizer() else {
        return;
    };

    // "Hello world" -> [15496, 995]
    let ids = tok.encode("Hello world");
    assert_eq!(ids, vec![15496, 995], "Hello world tokenization mismatch: {ids:?}");

    // "Hello" alone
    let ids = tok.encode("Hello");
    assert_eq!(ids, vec![15496], "Hello tokenization mismatch: {ids:?}");
}

#[test]
fn round_trip_basic() {
    let Some(tok) = load_tokenizer() else {
        return;
    };

    let texts = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog.",
        "GPT-2 is a language model.",
        "1234567890",
        "foo bar baz",
    ];

    for text in texts {
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text, "round trip failed for: {text:?}");
    }
}

#[test]
fn round_trip_unicode() {
    let Some(tok) = load_tokenizer() else {
        return;
    };

    let text = "caf\u{00e9} na\u{00ef}ve";
    let ids = tok.encode(text);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, text, "unicode round trip failed");
}

#[test]
fn empty_input() {
    let Some(tok) = load_tokenizer() else {
        return;
    };

    let ids = tok.encode("");
    assert!(ids.is_empty(), "empty input should produce no tokens");

    let decoded = tok.decode(&[]);
    assert!(decoded.is_empty(), "decoding empty tokens should produce empty string");
}

#[test]
fn special_token_endoftext() {
    let Some(tok) = load_tokenizer() else {
        return;
    };

    // <|endoftext|> = 50256
    let ids = tok.encode("<|endoftext|>");
    // The special token is encoded character-by-character in standard GPT-2 BPE
    // (it's not handled as a single special token in the base tokenizer)
    assert!(!ids.is_empty());

    // But decoding 50256 should give back the special token string
    let decoded = tok.decode(&[50256]);
    assert_eq!(decoded, "<|endoftext|>");
}

#[test]
fn vocab_size_is_50257() {
    let Some(tok) = load_tokenizer() else {
        return;
    };

    assert_eq!(tok.vocab_size(), 50257, "GPT-2 vocab size should be 50257");
}

#[test]
fn whitespace_handling() {
    let Some(tok) = load_tokenizer() else {
        return;
    };

    // Space-only
    let ids = tok.encode(" ");
    assert!(!ids.is_empty(), "space should produce tokens");
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, " ");

    // Multiple spaces
    let text = "hello  world";
    let ids = tok.encode(text);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, text);
}
