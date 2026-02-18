use crate::autograd::ops;
use crate::autograd::GradTape;
use crate::backend::MathBackend;
use crate::nn::init;
use crate::nn::kv_cache::LayerKvCache;
use crate::nn::Module;
use crate::tensor::shape::Shape;
use crate::tensor::Tensor;

/// Multi-head causal self-attention.
pub struct CausalSelfAttention<B: MathBackend> {
    pub qkv_weight: Tensor<B>,
    pub qkv_bias: Tensor<B>,
    pub proj_weight: Tensor<B>,
    pub proj_bias: Tensor<B>,
    pub n_heads: usize,
    pub d_model: usize,
    pub d_head: usize,
}

impl<B: MathBackend> CausalSelfAttention<B> {
    pub fn new(d_model: usize, n_heads: usize, rng: &mut fastrand::Rng) -> Self {
        let d_head = d_model / n_heads;
        let qkv_w = init::normal_vec(rng, d_model * 3 * d_model, 0.0, 0.02);
        let qkv_b = vec![0.0f32; 3 * d_model];
        let proj_w = init::normal_vec(rng, d_model * d_model, 0.0, 0.02);
        let proj_b = vec![0.0f32; d_model];
        Self {
            qkv_weight: Tensor::from_vec(qkv_w, Shape::new(&[d_model, 3 * d_model])),
            qkv_bias: Tensor::from_vec(qkv_b, Shape::new(&[3 * d_model])),
            proj_weight: Tensor::from_vec(proj_w, Shape::new(&[d_model, d_model])),
            proj_bias: Tensor::from_vec(proj_b, Shape::new(&[d_model])),
            n_heads,
            d_model,
            d_head,
        }
    }

    pub fn forward(
        &self,
        input: &Tensor<B>,
        dropout_rate: f32,
        rng: Option<&mut fastrand::Rng>,
        tape: &mut GradTape<B>,
    ) -> Tensor<B> {
        ops::attention(
            input,
            &self.qkv_weight,
            &self.qkv_bias,
            &self.proj_weight,
            &self.proj_bias,
            self.n_heads,
            self.d_model,
            self.d_head,
            dropout_rate,
            rng,
            Some(tape),
        )
    }

    /// Single-token forward with KV cache for autoregressive inference.
    ///
    /// `input`: `[1, d_model]` — embedding for a single token.
    /// Computes Q, K, V for this token, appends K/V to cache, attends over full
    /// cached sequence. Returns `[1, d_model]`.
    #[allow(clippy::too_many_lines)]
    pub fn forward_with_cache(&self, input: &Tensor<B>, cache: &mut LayerKvCache<B>) -> Tensor<B> {
        let d_model = self.d_model;
        let n_heads = self.n_heads;
        let d_head = self.d_head;

        // QKV = input @ W_qkv + b_qkv => [1, 3*d_model]
        let qkv_raw = B::matmul(
            &input.data,
            &self.qkv_weight.data,
            1,
            d_model,
            3 * d_model,
            false,
            false,
        );
        let qkv_shape = Shape::new(&[1, 3 * d_model]);
        let bias_shape = Shape::new(&[1, 3 * d_model]);
        let qkv = B::add(
            &qkv_raw,
            &self.qkv_bias.data,
            &qkv_shape,
            &bias_shape,
            &qkv_shape,
        );
        let qkv_vec = B::to_vec(&qkv);

        // Split into per-head Q, K, V for this single token: [1, d_head] each
        let mut new_k: Vec<B::Storage> = Vec::with_capacity(n_heads);
        let mut new_v: Vec<B::Storage> = Vec::with_capacity(n_heads);
        let mut q_heads: Vec<Vec<f32>> = Vec::with_capacity(n_heads);

        for h in 0..n_heads {
            let q_offset = h * d_head;
            let k_offset = d_model + h * d_head;
            let v_offset = 2 * d_model + h * d_head;

            let q_h = qkv_vec[q_offset..q_offset + d_head].to_vec();
            let k_h = qkv_vec[k_offset..k_offset + d_head].to_vec();
            let v_h = qkv_vec[v_offset..v_offset + d_head].to_vec();

            q_heads.push(q_h);
            let head_shape = Shape::new(&[1, d_head]);
            new_k.push(B::from_vec(k_h, &head_shape));
            new_v.push(B::from_vec(v_h, &head_shape));
        }

        // Append to cache
        cache.append(new_k, new_v);
        let cached_len = cache.seq_len; // includes the token we just added

        // Per-head attention: Q_new @ K_cached^T / sqrt(d_head), causal mask, softmax, @ V_cached
        let scale = 1.0 / (d_head as f64).sqrt();
        let mut head_concat = vec![0.0f32; d_model]; // [1, d_model]

        for h in 0..n_heads {
            let q_h = &q_heads[h]; // [d_head]
            let k_cached = &cache.k_per_head[h]; // [cached_len, d_head]
            let v_cached = &cache.v_per_head[h]; // [cached_len, d_head]

            // scores = q @ K^T => [1, cached_len]
            let scores_raw = B::matmul(
                &B::from_vec(q_h.clone(), &Shape::new(&[1, d_head])),
                k_cached,
                1,
                d_head,
                cached_len,
                false,
                true,
            );
            let mut scores = B::to_vec(&scores_raw);
            for t in 0..cached_len {
                scores[t] = (f64::from(scores[t]) * scale) as f32;
            }

            // Causal mask: for token at position (cached_len - 1), mask future positions
            // Since cached_len - 1 is the current position, all cached positions <= current
            // are valid. No masking needed (all cached K are at positions < current or == current).

            // Softmax
            let scores_storage = B::from_vec(scores, &Shape::new(&[1, cached_len]));
            let attn = B::softmax(&scores_storage, &Shape::new(&[1, cached_len]));

            // out_h = attn @ V_cached => [1, d_head]
            let out_h = B::matmul(&attn, v_cached, 1, cached_len, d_head, false, false);
            let out_h_vec = B::to_vec(&out_h);

            for d in 0..d_head {
                head_concat[h * d_head + d] = out_h_vec[d];
            }
        }

        // Output projection: head_concat @ W_proj + b_proj => [1, d_model]
        let hc_storage = B::from_vec(head_concat, &Shape::new(&[1, d_model]));
        let proj_raw = B::matmul(
            &hc_storage,
            &self.proj_weight.data,
            1,
            d_model,
            d_model,
            false,
            false,
        );
        let proj_shape = Shape::new(&[1, d_model]);
        let pbias_shape = Shape::new(&[1, d_model]);
        let output_data = B::add(
            &proj_raw,
            &self.proj_bias.data,
            &proj_shape,
            &pbias_shape,
            &proj_shape,
        );

        Tensor::new(output_data, Shape::new(&[1, d_model]))
    }

    pub fn forward_inference(&self, input: &Tensor<B>) -> Tensor<B> {
        ops::attention(
            input,
            &self.qkv_weight,
            &self.qkv_bias,
            &self.proj_weight,
            &self.proj_bias,
            self.n_heads,
            self.d_model,
            self.d_head,
            0.0,
            None,
            None,
        )
    }
}

impl<B: MathBackend> Module<B> for CausalSelfAttention<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![
            &self.qkv_weight,
            &self.qkv_bias,
            &self.proj_weight,
            &self.proj_bias,
        ]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<B>> {
        vec![
            &mut self.qkv_weight,
            &mut self.qkv_bias,
            &mut self.proj_weight,
            &mut self.proj_bias,
        ]
    }
}
