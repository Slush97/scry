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
        let profile = std::env::var("SCRY_DECODE_PROFILE").is_ok();

        // Token embedding + positional embedding
        let tok_emb = scry_llm::ops::embedding(
            &self.token_embedding,
            &[token_id],
            self.vocab_size,
            self.d_model,
        );
        let pos = scry_llm::ops::embedding(
            &self.positional_embedding,
            &[position],
            self.n_text_ctx,
            self.d_model,
        );
        let mut x = scry_llm::ops::add(&tok_emb, &pos);

        // Decoder blocks
        let t_blocks = std::time::Instant::now();
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward_step(&x, &mut self_kv_cache.layers[i], &cross_kv_caches[i]);
        }
        let blocks_ms = t_blocks.elapsed().as_secs_f64() * 1000.0;

        // Final layer norm
        x = self.ln.forward(&x);

        // Project to vocab logits: x @ token_embedding^T → [1, vocab_size]
        let t_logit = std::time::Instant::now();
        let logits = scry_llm::ops::matmul(
            &x,
            &self.token_embedding,
            1,
            self.d_model,
            self.vocab_size,
            false,
            true,
        );
        let logit_ms = t_logit.elapsed().as_secs_f64() * 1000.0;

        if profile {
            eprintln!("    blocks={blocks_ms:.2}ms logit_proj={logit_ms:.2}ms");
        }

        logits
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
    ///
    /// Uses batched matmul across all heads simultaneously.
    fn forward_with_cache(
        &self,
        input: &Tensor<B>,
        cache: &mut DecoderLayerKv<B>,
    ) -> Tensor<B> {
        let d_model = self.d_model;
        let n_heads = self.n_heads;
        let d_head = self.d_head;

        // QKV = input @ W_qkv + b_qkv → [1, 3 * d_model] (fused)
        let qkv = scry_llm::ops::matmul_bias(
            input,
            &self.qkv_weight,
            &self.qkv_bias,
            1,
            d_model,
            3 * d_model,
            false,
            false,
        );
        let qkv_vec = qkv.to_vec();

        // Extract Q, K, V and append K, V to cache
        let q_flat: Vec<f32> = qkv_vec[..d_model].to_vec();
        let k_new: Vec<f32> = qkv_vec[d_model..2 * d_model].to_vec();
        let v_new: Vec<f32> = qkv_vec[2 * d_model..3 * d_model].to_vec();

        cache.k.extend_from_slice(&k_new);
        cache.v.extend_from_slice(&v_new);
        cache.seq_len += 1;
        let cached_len = cache.seq_len;

        // Convert to backend storage and reshape
        // Q [1, d_model] → [n_heads, 1, d_head]  (identity permutation for seq=1)
        let q_stor = B::from_vec(q_flat, &Shape::new(&[1, d_model]));
        let q_heads = B::reshape_for_heads(&q_stor, 1, 1, n_heads, d_head);

        // Cached K [cached_len, d_model] → [n_heads, cached_len, d_head]
        // Use reshape_for_heads_from_host to avoid clone + from_vec on CpuBackend.
        let k_heads = B::reshape_for_heads_from_host(&cache.k, 1, cached_len, n_heads, d_head);
        // Cached V [cached_len, d_model] → [n_heads, cached_len, d_head]
        let v_heads = B::reshape_for_heads_from_host(&cache.v, 1, cached_len, n_heads, d_head);

        // Batched scores = Q_heads @ K_heads^T → [n_heads * 1, cached_len]
        let scores = B::matmul_strided_batched(
            &q_heads, &k_heads, n_heads, 1, d_head, cached_len, false, true,
        );

        // Fused scale + softmax — [n_heads, cached_len]
        let scale = 1.0 / (d_head as f32).sqrt();
        let attn = B::scaled_softmax(
            &scores,
            scale,
            &Shape::new(&[n_heads, cached_len]),
        );

        // Batched out = attn @ V_heads → [n_heads * 1, d_head]
        let out_heads = B::matmul_strided_batched(
            &attn, &v_heads, n_heads, 1, cached_len, d_head, false, false,
        );

        // Reshape [n_heads, d_head] → [1, d_model]  (identity permutation for seq=1)
        let head_concat = B::reshape_from_heads(&out_heads, 1, 1, n_heads, d_head);

        // Output projection (fused matmul + bias)
        let hc = Tensor::<B>::new(head_concat, Shape::new(&[1, d_model]));
        scry_llm::ops::matmul_bias(
            &hc,
            &self.out_weight,
            &self.out_bias,
            1,
            d_model,
            d_model,
            false,
            false,
        )
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
