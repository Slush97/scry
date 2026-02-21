use scry_llm::backend::MathBackend;
use scry_llm::nn::Module;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

/// Cross-attention layer for Whisper's decoder.
///
/// Unlike self-attention where Q, K, V all come from the same input,
/// cross-attention computes Q from the decoder state and K, V from
/// the encoder output. The encoder KV is computed once and cached.
pub struct CrossAttention<B: MathBackend> {
    /// Query projection: `[d_model, d_model]`.
    pub q_weight: Tensor<B>,
    /// Query bias: `[d_model]`.
    pub q_bias: Tensor<B>,
    /// Key projection: `[d_model, d_model]`.
    pub k_weight: Tensor<B>,
    /// Value projection: `[d_model, d_model]`.
    pub v_weight: Tensor<B>,
    /// Value bias: `[d_model]`.
    pub v_bias: Tensor<B>,
    /// Output projection: `[d_model, d_model]`.
    pub out_weight: Tensor<B>,
    /// Output bias: `[d_model]`.
    pub out_bias: Tensor<B>,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Model dimension.
    pub d_model: usize,
    /// Per-head dimension.
    pub d_head: usize,
}

/// Cached encoder key-value projections for cross-attention.
///
/// Computed once from encoder output, reused for every decode step.
pub struct CrossKvCache<B: MathBackend> {
    /// Projected keys: `[audio_len, d_model]`.
    pub k: Tensor<B>,
    /// Projected values: `[audio_len, d_model]`.
    pub v: Tensor<B>,
    /// Audio sequence length.
    pub audio_len: usize,
}

impl<B: MathBackend> CrossAttention<B> {
    /// Create a new cross-attention layer with random initialization.
    pub fn new(d_model: usize, n_heads: usize, rng: &mut fastrand::Rng) -> Self {
        let d_head = d_model / n_heads;
        let std_dev = 0.02;
        let mut rand_vec = |size: usize| -> Vec<f32> {
            (0..size)
                .map(|_| ((rng.f64() * 2.0 - 1.0) * std_dev) as f32)
                .collect()
        };

        Self {
            q_weight: Tensor::from_vec(
                rand_vec(d_model * d_model),
                Shape::new(&[d_model, d_model]),
            ),
            q_bias: Tensor::from_vec(vec![0.0; d_model], Shape::new(&[d_model])),
            k_weight: Tensor::from_vec(
                rand_vec(d_model * d_model),
                Shape::new(&[d_model, d_model]),
            ),
            // Note: Whisper's cross-attn key projection has no bias
            v_weight: Tensor::from_vec(
                rand_vec(d_model * d_model),
                Shape::new(&[d_model, d_model]),
            ),
            v_bias: Tensor::from_vec(vec![0.0; d_model], Shape::new(&[d_model])),
            out_weight: Tensor::from_vec(
                rand_vec(d_model * d_model),
                Shape::new(&[d_model, d_model]),
            ),
            out_bias: Tensor::from_vec(vec![0.0; d_model], Shape::new(&[d_model])),
            n_heads,
            d_model,
            d_head,
        }
    }

    /// Compute cross-attention KV cache from encoder output.
    ///
    /// This should be called once per audio chunk — the returned cache
    /// is reused for all decode steps within that chunk.
    pub fn compute_kv_cache(&self, encoder_output: &Tensor<B>) -> CrossKvCache<B> {
        let audio_len = encoder_output.shape.dims()[0];

        // K = encoder_output @ k_weight  (no bias in Whisper cross-attn keys)
        let k = scry_llm::ops::matmul(
            encoder_output,
            &self.k_weight,
            audio_len,
            self.d_model,
            self.d_model,
            false,
            false,
        );

        // V = encoder_output @ v_weight + v_bias
        let v_raw = scry_llm::ops::matmul(
            encoder_output,
            &self.v_weight,
            audio_len,
            self.d_model,
            self.d_model,
            false,
            false,
        );
        let v = scry_llm::ops::add(&v_raw, &self.v_bias);

        CrossKvCache { k, v, audio_len }
    }

    /// Forward pass: cross-attention with cached encoder KV.
    ///
    /// `decoder_state`: `[seq_len, d_model]` — current decoder hidden state.
    /// `cache`: pre-computed encoder KV from `compute_kv_cache`.
    ///
    /// Returns: `[seq_len, d_model]`.
    pub fn forward(
        &self,
        decoder_state: &Tensor<B>,
        cache: &CrossKvCache<B>,
    ) -> Tensor<B> {
        let seq_len = decoder_state.shape.dims()[0];
        let audio_len = cache.audio_len;
        let d_model = self.d_model;
        let n_heads = self.n_heads;
        let d_head = self.d_head;

        // Q = decoder_state @ q_weight + q_bias  → [seq_len, d_model]
        let q_raw = scry_llm::ops::matmul(
            decoder_state,
            &self.q_weight,
            seq_len,
            d_model,
            d_model,
            false,
            false,
        );
        let q = scry_llm::ops::add(&q_raw, &self.q_bias);
        let q_vec = q.to_vec();
        let k_vec = cache.k.to_vec();
        let v_vec = cache.v.to_vec();

        let scale = 1.0 / (d_head as f64).sqrt();
        let mut head_concat = vec![0.0f32; seq_len * d_model];

        // Multi-head cross-attention
        for h in 0..n_heads {
            for s in 0..seq_len {
                // Extract q_h for this head and position: [d_head]
                let q_offset = s * d_model + h * d_head;

                // Compute attention scores: q_h @ k_h^T → [audio_len]
                let mut scores = vec![0.0f64; audio_len];
                for t in 0..audio_len {
                    let k_offset = t * d_model + h * d_head;
                    let mut dot = 0.0f64;
                    for d in 0..d_head {
                        dot += f64::from(q_vec[q_offset + d]) * f64::from(k_vec[k_offset + d]);
                    }
                    scores[t] = dot * scale;
                }

                // Softmax over scores (no causal mask — attend to all encoder positions)
                let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut exp_sum = 0.0f64;
                for v in &mut scores {
                    *v = (*v - max_score).exp();
                    exp_sum += *v;
                }
                for v in &mut scores {
                    *v /= exp_sum;
                }

                // Weighted sum of values: attn @ v_h → [d_head]
                for d in 0..d_head {
                    let mut acc = 0.0f64;
                    for t in 0..audio_len {
                        let v_offset = t * d_model + h * d_head + d;
                        acc += scores[t] * f64::from(v_vec[v_offset]);
                    }
                    head_concat[s * d_model + h * d_head + d] = acc as f32;
                }
            }
        }

        // Output projection
        let hc = Tensor::<B>::from_vec(head_concat, Shape::new(&[seq_len, d_model]));
        let out_raw = scry_llm::ops::matmul(
            &hc,
            &self.out_weight,
            seq_len,
            d_model,
            d_model,
            false,
            false,
        );
        scry_llm::ops::add(&out_raw, &self.out_bias)
    }
}

impl<B: MathBackend> Module<B> for CrossAttention<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![
            &self.q_weight,
            &self.q_bias,
            &self.k_weight,
            &self.v_weight,
            &self.v_bias,
            &self.out_weight,
            &self.out_bias,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scry_llm::backend::cpu::CpuBackend;

    #[test]
    fn cross_attention_output_shape() {
        let mut rng = fastrand::Rng::with_seed(42);
        let attn = CrossAttention::<CpuBackend>::new(512, 8, &mut rng);

        let encoder_out = Tensor::<CpuBackend>::from_vec(
            vec![0.1f32; 1500 * 512],
            Shape::new(&[1500, 512]),
        );
        let cache = attn.compute_kv_cache(&encoder_out);

        let decoder_state = Tensor::<CpuBackend>::from_vec(
            vec![0.1f32; 1 * 512],
            Shape::new(&[1, 512]),
        );
        let output = attn.forward(&decoder_state, &cache);
        assert_eq!(output.shape.dims(), &[1, 512]);
    }
}
