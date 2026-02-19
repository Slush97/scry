use crate::autograd::ops;
use crate::autograd::GradTape;
use crate::backend::MathBackend;
use crate::nn::attention::CausalSelfAttention;
use crate::nn::kv_cache::LayerKvCache;
use crate::nn::layernorm::LayerNormModule;
use crate::nn::mlp::Mlp;
use crate::nn::Module;
use crate::tensor::Tensor;

/// A single transformer block: LN → attention → residual, LN → MLP → residual.
pub struct TransformerBlock<B: MathBackend> {
    pub ln1: LayerNormModule<B>,
    pub attn: CausalSelfAttention<B>,
    pub ln2: LayerNormModule<B>,
    pub mlp: Mlp<B>,
}

impl<B: MathBackend> TransformerBlock<B> {
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize, rng: &mut fastrand::Rng) -> Self {
        Self {
            ln1: LayerNormModule::new(d_model),
            attn: CausalSelfAttention::new(d_model, n_heads, rng),
            ln2: LayerNormModule::new(d_model),
            mlp: Mlp::new(d_model, d_ff, rng),
        }
    }

    pub fn forward(
        &self,
        input: &Tensor<B>,
        dropout_rate: f32,
        rng: &mut fastrand::Rng,
        tape: &mut GradTape<B>,
    ) -> Tensor<B> {
        // x = input + dropout(attn(ln1(input)))
        let ln1_out = self.ln1.forward(input, tape);
        let attn_out = self.attn.forward(&ln1_out, dropout_rate, Some(rng), tape);
        let attn_out = ops::dropout(&attn_out, dropout_rate, rng, Some(tape));
        let x = ops::add(input, &attn_out, Some(tape));

        // x = x + dropout(mlp(ln2(x)))
        let ln2_out = self.ln2.forward(&x, tape);
        let mlp_out = self.mlp.forward(&ln2_out, tape);
        let mlp_out = ops::dropout(&mlp_out, dropout_rate, rng, Some(tape));
        ops::add(&x, &mlp_out, Some(tape))
    }

    /// Batched forward pass: `input` is `[batch_size * seq_len, d_model]`.
    ///
    /// Most ops work unchanged on the flattened `[B*seq, d_model]` tensor.
    /// Attention loops over batch items because the causal mask is per-sequence.
    pub fn forward_batch(
        &self,
        input: &Tensor<B>,
        batch_size: usize,
        seq_len: usize,
        dropout_rate: f32,
        rng: &mut fastrand::Rng,
        tape: &mut GradTape<B>,
    ) -> Tensor<B> {
        // LN1 works on [B*seq, d_model]
        let ln1_out = self.ln1.forward(input, tape);
        let attn_out = ops::attention_batched(
            &ln1_out,
            &self.attn.qkv_weight,
            &self.attn.qkv_bias,
            &self.attn.proj_weight,
            &self.attn.proj_bias,
            self.attn.n_heads,
            self.attn.d_model,
            self.attn.d_head,
            batch_size,
            seq_len,
            dropout_rate,
            Some(rng),
            Some(tape),
        );
        let attn_out = ops::dropout(&attn_out, dropout_rate, rng, Some(tape));
        let x = ops::add(input, &attn_out, Some(tape));

        let ln2_out = self.ln2.forward(&x, tape);
        let mlp_out = self.mlp.forward(&ln2_out, tape);
        let mlp_out = ops::dropout(&mlp_out, dropout_rate, rng, Some(tape));
        ops::add(&x, &mlp_out, Some(tape))
    }

    /// Single-token forward with KV cache (inference only).
    pub fn forward_with_cache(&self, input: &Tensor<B>, cache: &mut LayerKvCache<B>) -> Tensor<B> {
        let ln1_out = self.ln1.forward_inference(input);
        let attn_out = self.attn.forward_with_cache(&ln1_out, cache);
        let x = ops::add(input, &attn_out, None);

        let ln2_out = self.ln2.forward_inference(&x);
        let mlp_out = self.mlp.forward_inference(&ln2_out);
        ops::add(&x, &mlp_out, None)
    }

    /// Batched inference forward (no tape, no dropout). Used for checkpointing recomputation.
    pub fn forward_batch_inference(
        &self,
        input: &Tensor<B>,
        batch_size: usize,
        seq_len: usize,
    ) -> Tensor<B> {
        let ln1_out = self.ln1.forward_inference(input);
        let attn_out = ops::attention_batched(
            &ln1_out,
            &self.attn.qkv_weight,
            &self.attn.qkv_bias,
            &self.attn.proj_weight,
            &self.attn.proj_bias,
            self.attn.n_heads,
            self.attn.d_model,
            self.attn.d_head,
            batch_size,
            seq_len,
            0.0,
            None,
            None,
        );
        let x = ops::add(input, &attn_out, None);

        let ln2_out = self.ln2.forward_inference(&x);
        let mlp_out = self.mlp.forward_inference(&ln2_out);
        ops::add(&x, &mlp_out, None)
    }

    pub fn forward_inference(&self, input: &Tensor<B>) -> Tensor<B> {
        let ln1_out = self.ln1.forward_inference(input);
        let attn_out = self.attn.forward_inference(&ln1_out);
        let x = ops::add(input, &attn_out, None);

        let ln2_out = self.ln2.forward_inference(&x);
        let mlp_out = self.mlp.forward_inference(&ln2_out);
        ops::add(&x, &mlp_out, None)
    }
}

impl<B: MathBackend> Module<B> for TransformerBlock<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        let mut params = self.ln1.parameters();
        params.extend(self.attn.parameters());
        params.extend(self.ln2.parameters());
        params.extend(self.mlp.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<B>> {
        let mut params = self.ln1.parameters_mut();
        params.extend(self.attn.parameters_mut());
        params.extend(self.ln2.parameters_mut());
        params.extend(self.mlp.parameters_mut());
        params
    }
}
