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
/// Stores both the original projections and pre-reshaped head views
/// to avoid re-transposing 1500×384 data on every decode step.
pub struct CrossKvCache<B: MathBackend> {
    /// Projected keys: `[audio_len, d_model]`.
    pub k: Tensor<B>,
    /// Projected values: `[audio_len, d_model]`.
    pub v: Tensor<B>,
    /// Audio sequence length.
    pub audio_len: usize,
    /// Pre-reshaped keys: `[n_heads * audio_len, d_head]` — avoids per-step reshape.
    pub k_heads: B::Storage,
    /// Pre-reshaped values: `[n_heads * audio_len, d_head]` — avoids per-step reshape.
    pub v_heads: B::Storage,
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
        let n_heads = self.n_heads;
        let d_head = self.d_head;

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

        // V = encoder_output @ v_weight + v_bias (fused)
        let v = scry_llm::ops::matmul_bias(
            encoder_output,
            &self.v_weight,
            &self.v_bias,
            audio_len,
            self.d_model,
            self.d_model,
            false,
            false,
        );

        // Pre-reshape for heads once — avoids re-transposing 1500×384 on every decode step
        let k_heads = B::reshape_for_heads(&k.data, 1, audio_len, n_heads, d_head);
        let v_heads = B::reshape_for_heads(&v.data, 1, audio_len, n_heads, d_head);

        CrossKvCache { k, v, audio_len, k_heads, v_heads }
    }

    /// Forward pass: cross-attention with cached encoder KV.
    ///
    /// `decoder_state`: `[seq_len, d_model]` — current decoder hidden state.
    /// `cache`: pre-computed encoder KV from `compute_kv_cache`.
    ///
    /// Returns: `[seq_len, d_model]`.
    ///
    /// Uses batched matmul across all heads simultaneously.
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

        // Q = decoder_state @ q_weight + q_bias  → [seq_len, d_model] (fused)
        let q = scry_llm::ops::matmul_bias(
            decoder_state,
            &self.q_weight,
            &self.q_bias,
            seq_len,
            d_model,
            d_model,
            false,
            false,
        );

        // Reshape Q [seq_len, d_model] → [n_heads * seq_len, d_head]
        let q_heads = B::reshape_for_heads(&q.data, 1, seq_len, n_heads, d_head);
        // Use pre-reshaped encoder KV (computed once in compute_kv_cache)
        let k_heads = &cache.k_heads;
        let v_heads = &cache.v_heads;

        // Batched scores = Q_heads @ K_heads^T → [n_heads * seq_len, audio_len]
        let scores = B::matmul_strided_batched(
            &q_heads, k_heads, n_heads, seq_len, d_head, audio_len, false, true,
        );

        // Fused scale + softmax — [n_heads * seq_len, audio_len]
        let scale = 1.0 / (d_head as f32).sqrt();
        let attn = B::scaled_softmax(
            &scores,
            scale,
            &Shape::new(&[n_heads * seq_len, audio_len]),
        );

        // Batched out = attn @ V_heads → [n_heads * seq_len, d_head]
        let out_heads = B::matmul_strided_batched(
            &attn, v_heads, n_heads, seq_len, audio_len, d_head, false, false,
        );

        // Reshape [n_heads * seq_len, d_head] → [seq_len, d_model]
        let head_concat = B::reshape_from_heads(&out_heads, 1, seq_len, n_heads, d_head);

        // Output projection (fused matmul + bias)
        let hc = Tensor::<B>::new(head_concat, Shape::new(&[seq_len, d_model]));
        scry_llm::ops::matmul_bias(
            &hc,
            &self.out_weight,
            &self.out_bias,
            seq_len,
            d_model,
            d_model,
            false,
            false,
        )
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
