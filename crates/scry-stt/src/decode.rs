use scry_llm::backend::MathBackend;
use scry_llm::tensor::Tensor;

use crate::model::WhisperModel;
use crate::model::attention::CrossKvCache;
use crate::model::decoder::DecoderKvCache;

/// Configuration for decoding behavior.
#[derive(Clone, Debug)]
pub struct DecodeConfig {
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Temperature for sampling (0 = greedy).
    pub temperature: f32,
    /// End-of-text token ID.
    pub eot_token: usize,
    /// Start-of-transcript token ID.
    pub sot_token: usize,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        Self {
            max_tokens: 224,
            temperature: 0.0, // greedy by default
            eot_token: 50257,
            sot_token: 50258,
        }
    }
}

/// A transcribed segment with text and timing information.
#[derive(Clone, Debug)]
pub struct Segment {
    /// Transcribed text for this segment.
    pub text: String,
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
    /// Token IDs that produced this segment.
    pub tokens: Vec<usize>,
}

/// Greedy decode: generate tokens until EOT or max length.
///
/// Returns the generated token IDs (excluding SOT, including EOT if hit).
pub fn greedy_decode<B: MathBackend>(
    model: &WhisperModel<B>,
    encoder_output: &Tensor<B>,
    config: &DecodeConfig,
) -> Vec<usize> {
    let cross_kv_caches: Vec<CrossKvCache<B>> = model.compute_cross_kv_caches(encoder_output);
    let mut self_kv_cache: DecoderKvCache<B> = model.new_decoder_kv_cache();

    let mut tokens = Vec::with_capacity(config.max_tokens);
    let mut current_token = config.sot_token;

    for pos in 0..config.max_tokens {
        let logits = model.decode_step(
            current_token,
            pos,
            &mut self_kv_cache,
            &cross_kv_caches,
        );

        let logits_vec = logits.to_vec();

        // Greedy: argmax over vocabulary
        let next_token = if config.temperature < 1e-8 {
            argmax(&logits_vec)
        } else {
            sample_with_temperature(&logits_vec, config.temperature)
        };

        if next_token == config.eot_token {
            break;
        }

        tokens.push(next_token);
        current_token = next_token;
    }

    tokens
}

/// Argmax over a slice.
fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i)
}

/// Sample with temperature scaling.
fn sample_with_temperature(logits: &[f32], temperature: f32) -> usize {
    let scaled: Vec<f64> = logits
        .iter()
        .map(|&x| f64::from(x) / f64::from(temperature))
        .collect();

    // Softmax
    let max_val = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut probs: Vec<f64> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }

    // Weighted sample
    let mut rng = fastrand::Rng::new();
    let r = rng.f64();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }
    probs.len() - 1
}
