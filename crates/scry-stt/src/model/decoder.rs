use scry_llm::backend::MathBackend;
use scry_llm::nn::layernorm::LayerNormModule;
use scry_llm::nn::linear::Linear;
use scry_llm::nn::Module;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

use crate::model::attention::{CrossAttention, CrossKvCache};

/// Whisper text decoder.
///
/// Architecture:
///   1. Token embedding + learned positional embedding
///   2. N Transformer decoder blocks (causal self-attention + cross-attention + MLP)
///   3. Final layer norm
///   4. Linear projection to vocabulary logits (tied with token embedding)
///
/// Input: token IDs
/// Output: logits `[seq_len, vocab_size]`
pub struct WhisperDecoder<B: MathBackend> {
    /// Token embedding: `[vocab_size, d_model]`.
    pub token_embedding: Tensor<B>,
    /// Learned positional embedding: `[n_text_ctx, d_model]`.
    pub positional_embedding: Tensor<B>,
    /// Decoder blocks.
    pub blocks: Vec<DecoderBlock<B>>,
    /// Final layer norm.
    pub ln: LayerNormModule<B>,
    /// Model dimension.
    pub d_model: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum text context length.
    pub n_text_ctx: usize,
}

/// Single Transformer decoder block.
pub struct DecoderBlock<B: MathBackend> {
    /// Pre-self-attention layer norm.
    pub attn_ln: LayerNormModule<B>,
    /// Causal self-attention.
    pub self_attn: DecoderSelfAttention<B>,
    /// Pre-cross-attention layer norm.
    pub cross_attn_ln: LayerNormModule<B>,
    /// Cross-attention to encoder output.
    pub cross_attn: CrossAttention<B>,
    /// Pre-MLP layer norm.
    pub mlp_ln: LayerNormModule<B>,
    /// MLP first projection.
    pub mlp_fc1: Linear<B>,
    /// MLP second projection.
    pub mlp_fc2: Linear<B>,
}

/// Decoder causal self-attention with KV cache.
pub struct DecoderSelfAttention<B: MathBackend> {
    /// Combined QKV projection: `[d_model, 3 * d_model]`.
    pub qkv_weight: Tensor<B>,
    /// Combined QKV bias: `[3 * d_model]`.
    pub qkv_bias: Tensor<B>,
    /// Output projection: `[d_model, d_model]`.
    pub out_weight: Tensor<B>,
    /// Output bias: `[d_model]`.
    pub out_bias: Tensor<B>,
    /// Number of heads.
    pub n_heads: usize,
    /// Model dimension.
    pub d_model: usize,
    /// Per-head dimension.
    pub d_head: usize,
}

/// KV cache for decoder self-attention (grows with each decode step).
pub struct DecoderKvCache<B: MathBackend> {
    /// Per-layer KV cache. Each entry stores past K,V for one block.
    pub layers: Vec<DecoderLayerKv<B>>,
}

/// Per-layer KV storage for decoder self-attention.
pub struct DecoderLayerKv<B: MathBackend> {
    /// Cached keys: accumulated `[seq_so_far, d_model]`.
    pub k: Vec<f32>,
    /// Cached values: accumulated `[seq_so_far, d_model]`.
    pub v: Vec<f32>,
    /// Current sequence length in cache.
    pub seq_len: usize,
    /// Model dimension.
    pub d_model: usize,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: MathBackend> DecoderKvCache<B> {
    /// Create an empty KV cache for `n_layers` decoder blocks.
    pub fn new(n_layers: usize, d_model: usize) -> Self {
        let layers = (0..n_layers)
            .map(|_| DecoderLayerKv {
                k: Vec::new(),
                v: Vec::new(),
                seq_len: 0,
                d_model,
                _phantom: std::marker::PhantomData,
            })
            .collect();
        Self { layers }
    }

    /// Reset the cache (for a new audio chunk).
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.k.clear();
            layer.v.clear();
            layer.seq_len = 0;
        }
    }
}

impl<B: MathBackend> WhisperDecoder<B> {
    /// Create a new decoder with random initialization.
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        n_layers: usize,
        n_heads: usize,
        n_text_ctx: usize,
        rng: &mut fastrand::Rng,
    ) -> Self {
        let std_dev = 0.02;
        let mut rand_vec = |size: usize| -> Vec<f32> {
            (0..size)
                .map(|_| ((rng.f64() * 2.0 - 1.0) * std_dev) as f32)
                .collect()
        };

        let token_embedding = Tensor::from_vec(
            rand_vec(vocab_size * d_model),
            Shape::new(&[vocab_size, d_model]),
        );
        let positional_embedding = Tensor::from_vec(
            rand_vec(n_text_ctx * d_model),
            Shape::new(&[n_text_ctx, d_model]),
        );

        let blocks = (0..n_layers)
            .map(|_| DecoderBlock::new(d_model, n_heads, rng))
            .collect();

        let ln = LayerNormModule::new(d_model);

        Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            d_model,
            vocab_size,
            n_text_ctx,
        }
    }

    /// Forward pass for a single decode step with KV cache.
    ///
    /// `token_id`: the current token being decoded.
    /// `position`: the position in the output sequence.
    /// `self_kv_cache`: decoder self-attention KV cache (grows each step).
    /// `cross_kv_caches`: pre-computed encoder KV for each decoder layer.
    ///
    /// Returns logits `[1, vocab_size]`.
    pub fn forward_step(
        &self,
        token_id: usize,
        position: usize,
        self_kv_cache: &mut DecoderKvCache<B>,
        cross_kv_caches: &[CrossKvCache<B>],
    ) -> Tensor<B> {
        // Token embedding + positional embedding
        let tok_emb = scry_llm::ops::embedding(
            &self.token_embedding,
            &[token_id],
            self.vocab_size,
            self.d_model,
        );
        let pos_vec = self.positional_embedding.to_vec();
        let pos_start = position * self.d_model;
        let pos_slice = &pos_vec[pos_start..pos_start + self.d_model];
        let pos = Tensor::<B>::from_vec(pos_slice.to_vec(), Shape::new(&[1, self.d_model]));
        let mut x = scry_llm::ops::add(&tok_emb, &pos);

        // Decoder blocks
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward_step(&x, &mut self_kv_cache.layers[i], &cross_kv_caches[i]);
        }

        // Final layer norm
        x = self.ln.forward(&x);

        // Project to vocab logits: x @ token_embedding^T → [1, vocab_size]
        // (weight tying: logit projection = transpose of token embedding)
        scry_llm::ops::matmul(
            &x,
            &self.token_embedding,
            1,
            self.d_model,
            self.vocab_size,
            false,
            true, // transpose — [d_model, vocab_size]^T = [vocab_size, d_model]^T
        )
    }
}

impl<B: MathBackend> DecoderBlock<B> {
    fn new(d_model: usize, n_heads: usize, rng: &mut fastrand::Rng) -> Self {
        Self {
            attn_ln: LayerNormModule::new(d_model),
            self_attn: DecoderSelfAttention::new(d_model, n_heads, rng),
            cross_attn_ln: LayerNormModule::new(d_model),
            cross_attn: CrossAttention::new(d_model, n_heads, rng),
            mlp_ln: LayerNormModule::new(d_model),
            mlp_fc1: Linear::new(d_model, d_model * 4, rng),
            mlp_fc2: Linear::new(d_model * 4, d_model, rng),
        }
    }

    fn forward_step(
        &self,
        x: &Tensor<B>,
        self_kv: &mut DecoderLayerKv<B>,
        cross_kv: &CrossKvCache<B>,
    ) -> Tensor<B> {
        // Causal self-attention with KV cache + residual
        let attn_in = self.attn_ln.forward(x);
        let attn_out = self.self_attn.forward_with_cache(&attn_in, self_kv);
        let x = scry_llm::ops::add(x, &attn_out);

        // Cross-attention with encoder KV cache + residual
        let cross_in = self.cross_attn_ln.forward(&x);
        let cross_out = self.cross_attn.forward(&cross_in, cross_kv);
        let x = scry_llm::ops::add(&x, &cross_out);

        // MLP with residual
        let mlp_in = self.mlp_ln.forward(&x);
        let h = self.mlp_fc1.forward(&mlp_in);
        let h = scry_llm::ops::gelu(&h);
        let mlp_out = self.mlp_fc2.forward(&h);
        scry_llm::ops::add(&x, &mlp_out)
    }
}

impl<B: MathBackend> DecoderSelfAttention<B> {
    fn new(d_model: usize, n_heads: usize, rng: &mut fastrand::Rng) -> Self {
        let d_head = d_model / n_heads;
        let std_dev = 0.02;
        let mut rand_vec = |size: usize| -> Vec<f32> {
            (0..size)
                .map(|_| ((rng.f64() * 2.0 - 1.0) * std_dev) as f32)
                .collect()
        };

        Self {
            qkv_weight: Tensor::from_vec(
                rand_vec(d_model * 3 * d_model),
                Shape::new(&[d_model, 3 * d_model]),
            ),
            qkv_bias: Tensor::from_vec(vec![0.0; 3 * d_model], Shape::new(&[3 * d_model])),
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

    /// Single-token causal self-attention with KV cache.
    fn forward_with_cache(
        &self,
        input: &Tensor<B>,
        cache: &mut DecoderLayerKv<B>,
    ) -> Tensor<B> {
        let d_model = self.d_model;
        let n_heads = self.n_heads;
        let d_head = self.d_head;

        // QKV = input @ W_qkv + b_qkv → [1, 3 * d_model]
        let qkv_raw = scry_llm::ops::matmul(
            input,
            &self.qkv_weight,
            1,
            d_model,
            3 * d_model,
            false,
            false,
        );
        let qkv = scry_llm::ops::add(&qkv_raw, &Tensor::from_vec(
            self.qkv_bias.to_vec(),
            Shape::new(&[1, 3 * d_model]),
        ));
        let qkv_vec = qkv.to_vec();

        // Extract Q, K, V and append K, V to cache
        let q: Vec<f32> = qkv_vec[..d_model].to_vec();
        let k: Vec<f32> = qkv_vec[d_model..2 * d_model].to_vec();
        let v: Vec<f32> = qkv_vec[2 * d_model..3 * d_model].to_vec();

        cache.k.extend_from_slice(&k);
        cache.v.extend_from_slice(&v);
        cache.seq_len += 1;
        let cached_len = cache.seq_len;

        let scale = 1.0 / (d_head as f64).sqrt();
        let mut head_concat = vec![0.0f32; d_model];

        for h in 0..n_heads {
            // Compute attention scores: q_h @ cached_k_h^T → [cached_len]
            let mut scores = vec![0.0f64; cached_len];
            for t in 0..cached_len {
                let mut dot = 0.0f64;
                for d in 0..d_head {
                    dot += f64::from(q[h * d_head + d])
                        * f64::from(cache.k[t * d_model + h * d_head + d]);
                }
                scores[t] = dot * scale;
            }

            // Softmax (causal: only attend to positions ≤ current)
            // Since we only compute scores for cached positions, all are valid
            let max_s = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut exp_sum = 0.0f64;
            for v in &mut scores {
                *v = (*v - max_s).exp();
                exp_sum += *v;
            }
            for v in &mut scores {
                *v /= exp_sum;
            }

            // Weighted sum of cached values
            for d in 0..d_head {
                let mut acc = 0.0f64;
                for t in 0..cached_len {
                    acc += scores[t] * f64::from(cache.v[t * d_model + h * d_head + d]);
                }
                head_concat[h * d_head + d] = acc as f32;
            }
        }

        // Output projection
        let hc = Tensor::<B>::from_vec(head_concat, Shape::new(&[1, d_model]));
        let out_raw = scry_llm::ops::matmul(
            &hc,
            &self.out_weight,
            1,
            d_model,
            d_model,
            false,
            false,
        );
        scry_llm::ops::add(&out_raw, &self.out_bias)
    }
}

impl<B: MathBackend> Module<B> for DecoderSelfAttention<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![&self.qkv_weight, &self.qkv_bias, &self.out_weight, &self.out_bias]
    }
}

impl<B: MathBackend> Module<B> for DecoderBlock<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        let mut params = self.attn_ln.parameters();
        params.extend(self.self_attn.parameters());
        params.extend(self.cross_attn_ln.parameters());
        params.extend(self.cross_attn.parameters());
        params.extend(self.mlp_ln.parameters());
        params.extend(self.mlp_fc1.parameters());
        params.extend(self.mlp_fc2.parameters());
        params
    }
}

impl<B: MathBackend> Module<B> for WhisperDecoder<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        let mut params = vec![&self.token_embedding, &self.positional_embedding];
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.ln.parameters());
        params
    }
}
