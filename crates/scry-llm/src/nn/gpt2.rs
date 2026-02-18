use crate::autograd::ops;
use crate::autograd::GradTape;
use crate::backend::MathBackend;
use crate::nn::embedding::EmbeddingLayer;
use crate::nn::layernorm::LayerNormModule;
use crate::nn::transformer::TransformerBlock;
use crate::nn::Module;
use crate::tensor::Tensor;
#[cfg(feature = "safetensors")]
use crate::tensor::shape::Shape;

/// GPT-2 configuration.
#[derive(Clone, Debug)]
pub struct Gpt2Config {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
}

impl Gpt2Config {
    /// GPT-2 small (124M parameters).
    pub fn gpt2_small() -> Self {
        Self {
            vocab_size: 50257,
            max_seq_len: 1024,
            d_model: 768,
            n_heads: 12,
            n_layers: 12,
            d_ff: 3072,
        }
    }
}

/// Full GPT-2 language model.
///
/// LM head is weight-tied with `embedding.token_embedding` (no separate parameter).
pub struct Gpt2Model<B: MathBackend> {
    pub config: Gpt2Config,
    pub embedding: EmbeddingLayer<B>,
    pub blocks: Vec<TransformerBlock<B>>,
    pub ln_f: LayerNormModule<B>,
}

impl<B: MathBackend> Gpt2Model<B> {
    pub fn new(config: Gpt2Config, rng: &mut fastrand::Rng) -> Self {
        let mut blocks = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            let mut block = TransformerBlock::new(config.d_model, config.n_heads, config.d_ff, rng);
            // Apply residual scaling to attn output projection and MLP fc2
            block.attn.proj_weight = {
                let scale = 1.0 / (2.0 * config.n_layers as f64).sqrt();
                let mut data = block.attn.proj_weight.to_vec();
                for v in &mut data {
                    *v = (f64::from(*v) * scale) as f32;
                }
                Tensor::from_vec(data, block.attn.proj_weight.shape.clone())
            };
            block.mlp.fc2.apply_residual_scaling(config.n_layers);
            blocks.push(block);
        }

        Self {
            embedding: EmbeddingLayer::new(
                config.vocab_size,
                config.max_seq_len,
                config.d_model,
                rng,
            ),
            blocks,
            ln_f: LayerNormModule::new(config.d_model),
            config,
        }
    }

    /// Forward pass: `token_ids` → logits `[seq, vocab]`.
    pub fn forward(&self, token_ids: &[usize], tape: &mut GradTape<B>) -> Tensor<B> {
        let mut x = self.embedding.forward(token_ids, tape);

        for block in &self.blocks {
            x = block.forward(&x, tape);
        }

        x = self.ln_f.forward(&x, tape);

        // LM head: x @ token_embedding^T => [seq, vocab]
        let seq = token_ids.len();
        ops::matmul(
            &x,
            &self.embedding.token_embedding,
            seq,
            self.config.d_model,
            self.config.vocab_size,
            false,
            true,
            Some(tape),
        )
    }

    /// Inference forward pass (no tape).
    pub fn forward_inference(&self, token_ids: &[usize]) -> Tensor<B> {
        let mut x = self.embedding.forward_inference(token_ids);

        for block in &self.blocks {
            x = block.forward_inference(&x);
        }

        x = self.ln_f.forward_inference(&x);

        let seq = token_ids.len();
        ops::matmul(
            &x,
            &self.embedding.token_embedding,
            seq,
            self.config.d_model,
            self.config.vocab_size,
            false,
            true,
            None,
        )
    }

    /// Load weights from a `HuggingFace` GPT-2 safetensors file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, parsed, or is missing expected tensors.
    #[cfg(feature = "safetensors")]
    #[allow(clippy::too_many_lines)]
    pub fn load_safetensors(
        config: Gpt2Config,
        path: &std::path::Path,
    ) -> crate::error::Result<Self> {
        use crate::error::ScryLlmError;
        use crate::nn::attention::CausalSelfAttention;
        use crate::nn::linear::Linear;
        use crate::nn::mlp::Mlp;

        let data = std::fs::read(path).map_err(|e| {
            ScryLlmError::WeightLoadError(format!("failed to read {}: {e}", path.display()))
        })?;
        let tensors = safetensors::SafeTensors::deserialize(&data).map_err(|e| {
            ScryLlmError::WeightLoadError(format!("failed to parse safetensors: {e}"))
        })?;

        // HF GPT-2 checkpoints may or may not have a "transformer." prefix.
        // Try with prefix first, fall back to without.
        let has_prefix = tensors.tensor("transformer.wte.weight").is_ok();
        let prefix_str = if has_prefix { "transformer." } else { "" };

        let load = |name: &str| -> crate::error::Result<Vec<f32>> {
            let full_name = format!("{prefix_str}{name}");
            let t = tensors.tensor(&full_name).map_err(|e| {
                ScryLlmError::WeightLoadError(format!("missing tensor '{full_name}': {e}"))
            })?;
            // safetensors stores f32 as little-endian bytes
            let bytes = t.data();
            let floats: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Ok(floats)
        };

        let d = config.d_model;
        let v = config.vocab_size;
        let s = config.max_seq_len;
        let ff = config.d_ff;
        let nl = config.n_layers;

        let wte = load("wte.weight")?;
        let wpe = load("wpe.weight")?;

        let embedding = EmbeddingLayer {
            token_embedding: Tensor::from_vec(wte, Shape::new(&[v, d])),
            position_embedding: Tensor::from_vec(wpe, Shape::new(&[s, d])),
            vocab_size: v,
            max_seq_len: s,
            d_model: d,
        };

        let d_head = d / config.n_heads;
        let mut blocks = Vec::with_capacity(nl);
        for i in 0..nl {
            let prefix = format!("h.{i}");

            let ln1_w = load(&format!("{prefix}.ln_1.weight"))?;
            let ln1_b = load(&format!("{prefix}.ln_1.bias"))?;
            let ln2_w = load(&format!("{prefix}.ln_2.weight"))?;
            let ln2_b = load(&format!("{prefix}.ln_2.bias"))?;

            let attn_qkv_w = load(&format!("{prefix}.attn.c_attn.weight"))?;
            let attn_qkv_b = load(&format!("{prefix}.attn.c_attn.bias"))?;
            let attn_proj_w = load(&format!("{prefix}.attn.c_proj.weight"))?;
            let attn_proj_b = load(&format!("{prefix}.attn.c_proj.bias"))?;

            let mlp_fc_w = load(&format!("{prefix}.mlp.c_fc.weight"))?;
            let mlp_fc_b = load(&format!("{prefix}.mlp.c_fc.bias"))?;
            let mlp_proj_w = load(&format!("{prefix}.mlp.c_proj.weight"))?;
            let mlp_proj_b = load(&format!("{prefix}.mlp.c_proj.bias"))?;

            let block = TransformerBlock {
                ln1: LayerNormModule {
                    gamma: Tensor::from_vec(ln1_w, Shape::new(&[d])),
                    beta: Tensor::from_vec(ln1_b, Shape::new(&[d])),
                    eps: 1e-5,
                },
                attn: CausalSelfAttention {
                    qkv_weight: Tensor::from_vec(attn_qkv_w, Shape::new(&[d, 3 * d])),
                    qkv_bias: Tensor::from_vec(attn_qkv_b, Shape::new(&[3 * d])),
                    proj_weight: Tensor::from_vec(attn_proj_w, Shape::new(&[d, d])),
                    proj_bias: Tensor::from_vec(attn_proj_b, Shape::new(&[d])),
                    n_heads: config.n_heads,
                    d_model: d,
                    d_head,
                },
                ln2: LayerNormModule {
                    gamma: Tensor::from_vec(ln2_w, Shape::new(&[d])),
                    beta: Tensor::from_vec(ln2_b, Shape::new(&[d])),
                    eps: 1e-5,
                },
                mlp: Mlp {
                    fc1: Linear {
                        weight: Tensor::from_vec(mlp_fc_w, Shape::new(&[d, ff])),
                        bias: Tensor::from_vec(mlp_fc_b, Shape::new(&[ff])),
                        in_features: d,
                        out_features: ff,
                    },
                    fc2: Linear {
                        weight: Tensor::from_vec(mlp_proj_w, Shape::new(&[ff, d])),
                        bias: Tensor::from_vec(mlp_proj_b, Shape::new(&[d])),
                        in_features: ff,
                        out_features: d,
                    },
                },
            };
            blocks.push(block);
        }

        let ln_f_w = load("ln_f.weight")?;
        let ln_f_b = load("ln_f.bias")?;
        let ln_f = LayerNormModule {
            gamma: Tensor::from_vec(ln_f_w, Shape::new(&[d])),
            beta: Tensor::from_vec(ln_f_b, Shape::new(&[d])),
            eps: 1e-5,
        };

        Ok(Self {
            config,
            embedding,
            blocks,
            ln_f,
        })
    }
}

impl<B: MathBackend> Module<B> for Gpt2Model<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        let mut params = self.embedding.parameters();
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.ln_f.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<B>> {
        let mut params = self.embedding.parameters_mut();
        for block in &mut self.blocks {
            params.extend(block.parameters_mut());
        }
        params.extend(self.ln_f.parameters_mut());
        params
    }
}
