use std::collections::HashSet;
use std::sync::Arc;

use crate::autograd::ops;
use crate::autograd::GradTape;
use crate::backend::MathBackend;
use crate::nn::embedding::EmbeddingLayer;
use crate::nn::kv_cache::KvCache;
use crate::nn::layernorm::LayerNormModule;
use crate::nn::transformer::TransformerBlock;
use crate::nn::Module;
use crate::tensor::shape::Shape;
use crate::tensor::{Tensor, TensorId};

/// GPT-2 configuration.
#[derive(Clone, Debug)]
pub struct Gpt2Config {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub dropout_rate: f32,
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
            dropout_rate: 0.1,
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

    /// Returns the total number of trainable parameters.
    pub fn n_params(&self) -> usize {
        self.parameters().iter().map(|t| t.numel()).sum()
    }

    /// Returns the set of parameter IDs that should be exempt from weight decay.
    ///
    /// This includes all bias parameters, and all layernorm gamma/beta parameters:
    /// - Per block: `ln1.gamma`, `ln1.beta`, `qkv_bias`, `proj_bias`, `ln2.gamma`, `ln2.beta`,
    ///   `fc1.bias`, `fc2.bias` (8 per block)
    /// - Final layernorm: `ln_f.gamma`, `ln_f.beta` (2)
    pub fn no_decay_ids(&self) -> HashSet<TensorId> {
        let mut ids = HashSet::new();

        for block in &self.blocks {
            // LayerNorm gamma/beta
            ids.insert(block.ln1.gamma.id);
            ids.insert(block.ln1.beta.id);
            ids.insert(block.ln2.gamma.id);
            ids.insert(block.ln2.beta.id);
            // Attention biases
            ids.insert(block.attn.qkv_bias.id);
            ids.insert(block.attn.proj_bias.id);
            // MLP biases
            ids.insert(block.mlp.fc1.bias.id);
            ids.insert(block.mlp.fc2.bias.id);
        }

        // Final layernorm
        ids.insert(self.ln_f.gamma.id);
        ids.insert(self.ln_f.beta.id);

        ids
    }

    /// Forward pass: `token_ids` → logits `[seq, vocab]`.
    pub fn forward(
        &self,
        token_ids: &[usize],
        rng: &mut fastrand::Rng,
        tape: &mut GradTape<B>,
    ) -> Tensor<B> {
        let mut x = self.embedding.forward(token_ids, tape);
        x = ops::dropout(&x, self.config.dropout_rate, rng, Some(tape));

        for block in &self.blocks {
            x = block.forward(&x, self.config.dropout_rate, rng, tape);
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

    /// Batched forward pass: `token_ids` is `[batch_size * seq_len]` flat.
    /// Returns logits `[batch_size * seq_len, vocab]`.
    pub fn forward_batch(
        &self,
        token_ids: &[usize],
        batch_size: usize,
        seq_len: usize,
        rng: &mut fastrand::Rng,
        tape: &mut GradTape<B>,
    ) -> Tensor<B> {
        assert_eq!(
            token_ids.len(),
            batch_size * seq_len,
            "forward_batch: token_ids length mismatch"
        );

        let mut x = self.embedding.forward_batch(token_ids, batch_size, seq_len, tape);
        x = ops::dropout(&x, self.config.dropout_rate, rng, Some(tape));

        for block in &self.blocks {
            x = block.forward_batch(&x, batch_size, seq_len, self.config.dropout_rate, rng, tape);
        }

        x = self.ln_f.forward(&x, tape);

        let total = batch_size * seq_len;
        ops::matmul(
            &x,
            &self.embedding.token_embedding,
            total,
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

    /// Create a new [`KvCache`] sized for this model.
    pub fn new_kv_cache(&self) -> KvCache<B> {
        let d_head = self.config.d_model / self.config.n_heads;
        KvCache::new(self.config.n_layers, self.config.n_heads, d_head)
    }

    /// Single-token forward with KV cache for autoregressive inference.
    ///
    /// `token_id`: the token to process. `position`: its absolute position in the
    /// sequence (for position embedding lookup). Returns logits `[1, vocab]`.
    pub fn forward_with_cache(
        &self,
        token_id: usize,
        position: usize,
        cache: &mut KvCache<B>,
    ) -> Tensor<B> {
        // Embed single token + position
        let tok_emb = ops::embedding(
            &self.embedding.token_embedding,
            &[token_id],
            self.embedding.vocab_size,
            self.embedding.d_model,
            None,
        );
        let pos_emb = ops::embedding(
            &self.embedding.position_embedding,
            &[position],
            self.embedding.max_seq_len,
            self.embedding.d_model,
            None,
        );
        let mut x = ops::add(&tok_emb, &pos_emb, None);

        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward_with_cache(&x, &mut cache.layers[i]);
        }

        x = self.ln_f.forward_inference(&x);

        ops::matmul(
            &x,
            &self.embedding.token_embedding,
            1,
            self.config.d_model,
            self.config.vocab_size,
            false,
            true,
            None,
        )
    }

    /// Forward pass with gradient checkpointing.
    ///
    /// Runs transformer blocks in segments of `checkpoint_every` blocks. For each
    /// segment, the forward is run without recording on the tape — only a
    /// [`SavedData::Checkpoint`] placeholder is recorded. During
    /// [`backward_checkpointed`](Self::backward_checkpointed), each segment is
    /// recomputed to produce the real tape nodes.
    ///
    /// This trades compute for memory: `O(n_layers / checkpoint_every)` boundary
    /// tensors stored instead of the full tape.
    pub fn forward_checkpointed(
        &self,
        token_ids: &[usize],
        checkpoint_every: usize,
        rng: &mut fastrand::Rng,
        tape: &mut GradTape<B>,
    ) -> Tensor<B> {
        use crate::autograd::{Operation, SavedData, TapeNode};

        let mut x = self.embedding.forward(token_ids, tape);
        x = ops::dropout(&x, self.config.dropout_rate, rng, Some(tape));

        let n = self.blocks.len();
        let mut i = 0;
        while i < n {
            let end = (i + checkpoint_every).min(n);

            // Save RNG state for deterministic recomputation
            let rng_seed = rng.u64(..);

            // Save segment input (data + shape + id) before running without tape
            let seg_input_id = x.id;
            let seg_input_data = B::clone_storage(&x.data);
            let seg_input_shape = x.shape.clone();
            for block in &self.blocks[i..end] {
                x = block.forward_inference(&x);
            }

            // Also advance the main RNG to match what would have happened
            // (dropout samples) — we use seg_rng for actual computation but
            // the main rng was already advanced by the u64 call above.

            // Record a checkpoint placeholder on the tape
            tape.record(TapeNode {
                output_id: x.id,
                input_ids: vec![seg_input_id],
                op: Operation::Checkpoint,
                saved: SavedData::Checkpoint {
                    input_data: seg_input_data,
                    input_shape: seg_input_shape,
                    block_start: i,
                    block_end: end,
                    dropout_rate: self.config.dropout_rate,
                    rng_seed,
                    batch_size: None,
                    seq_len: None,
                },
            });

            i = end;
        }

        x = self.ln_f.forward(&x, tape);

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

    /// Batched forward pass with gradient checkpointing.
    ///
    /// Like `forward_checkpointed` but for batched inputs `[batch_size * seq_len]`.
    pub fn forward_batch_checkpointed(
        &self,
        token_ids: &[usize],
        batch_size: usize,
        seq_len: usize,
        checkpoint_every: usize,
        rng: &mut fastrand::Rng,
        tape: &mut GradTape<B>,
    ) -> Tensor<B> {
        use crate::autograd::{Operation, SavedData, TapeNode};

        assert_eq!(
            token_ids.len(),
            batch_size * seq_len,
            "forward_batch_checkpointed: token_ids length mismatch"
        );

        let mut x = self.embedding.forward_batch(token_ids, batch_size, seq_len, tape);
        x = ops::dropout(&x, self.config.dropout_rate, rng, Some(tape));

        let n = self.blocks.len();
        let mut i = 0;
        while i < n {
            let end = (i + checkpoint_every).min(n);

            let rng_seed = rng.u64(..);

            let seg_input_id = x.id;
            let seg_input_data = B::clone_storage(&x.data);
            let seg_input_shape = x.shape.clone();

            // Run segment without tape (inference)
            for block in &self.blocks[i..end] {
                x = block.forward_batch_inference(&x, batch_size, seq_len);
            }

            tape.record(TapeNode {
                output_id: x.id,
                input_ids: vec![seg_input_id],
                op: Operation::Checkpoint,
                saved: SavedData::Checkpoint {
                    input_data: seg_input_data,
                    input_shape: seg_input_shape,
                    block_start: i,
                    block_end: end,
                    dropout_rate: self.config.dropout_rate,
                    rng_seed,
                    batch_size: Some(batch_size),
                    seq_len: Some(seq_len),
                },
            });

            i = end;
        }

        x = self.ln_f.forward(&x, tape);

        let total = batch_size * seq_len;
        ops::matmul(
            &x,
            &self.embedding.token_embedding,
            total,
            self.config.d_model,
            self.config.vocab_size,
            false,
            true,
            Some(tape),
        )
    }

    /// Backward pass with gradient checkpointing.
    ///
    /// For each `Checkpoint` node encountered during backward traversal,
    /// recomputes the segment's forward pass with a local tape, runs backward
    /// on that local tape, and propagates gradients.
    pub fn backward_checkpointed(
        &self,
        tape: &GradTape<B>,
        loss_id: TensorId,
    ) -> crate::autograd::backward::Gradients<B> {
        use crate::autograd::backward::Gradients;
        use crate::autograd::{Operation, SavedData};
        use std::collections::HashMap;

        let mut grads: Gradients<B> = HashMap::new();

        // Seed
        let ones = B::from_vec(vec![1.0], &Shape::new(&[1]));
        grads.insert(loss_id, ones);

        for node in tape.nodes.iter().rev() {
            let d_out = match grads.remove(&node.output_id) {
                Some(g) => g,
                None => continue,
            };

            match (&node.op, &node.saved) {
                (
                    Operation::Checkpoint,
                    SavedData::Checkpoint {
                        input_data,
                        input_shape,
                        block_start,
                        block_end,
                        dropout_rate,
                        rng_seed,
                        batch_size,
                        seq_len,
                    },
                ) => {
                    // Recompute the segment forward with a local tape
                    let mut local_tape = GradTape::<B>::new();
                    let mut seg_rng = fastrand::Rng::with_seed(*rng_seed);

                    // Create tensor with the SAME ID as the original input
                    // so gradients flow correctly to the segment input
                    let seg_input = Tensor {
                        id: node.input_ids[0],
                        data: Arc::new(B::clone_storage(input_data)),
                        shape: input_shape.clone(),
                    };

                    let mut y = seg_input;
                    if let (Some(bs), Some(sl)) = (batch_size, seq_len) {
                        // Batched recomputation
                        for block in &self.blocks[*block_start..*block_end] {
                            y = block.forward_batch(
                                &y,
                                *bs,
                                *sl,
                                *dropout_rate,
                                &mut seg_rng,
                                &mut local_tape,
                            );
                        }
                    } else {
                        // Single-sequence recomputation
                        for block in &self.blocks[*block_start..*block_end] {
                            y = block.forward(&y, *dropout_rate, &mut seg_rng, &mut local_tape);
                        }
                    }

                    // The local tape's last output should correspond to the
                    // checkpoint output. We need to seed the local backward
                    // with d_out for that output.
                    let local_loss_id = y.id;
                    let mut local_grads: Gradients<B> = HashMap::new();
                    local_grads.insert(local_loss_id, d_out);

                    // Run backward on local tape
                    for local_node in local_tape.nodes.iter().rev() {
                        let local_d_out = match local_grads.remove(&local_node.output_id) {
                            Some(g) => g,
                            None => continue,
                        };
                        crate::autograd::backward::backward_node::<B>(
                            local_node,
                            local_d_out,
                            &mut local_grads,
                        );
                    }

                    // Merge local gradients into main gradients
                    for (id, grad) in local_grads {
                        if let Some(existing) = grads.get_mut(&id) {
                            B::add_inplace(existing, &grad);
                        } else {
                            grads.insert(id, grad);
                        }
                    }
                }
                _ => {
                    // Normal backward — delegate to the standard per-node backward
                    crate::autograd::backward::backward_node::<B>(node, d_out, &mut grads);
                }
            }
        }

        grads
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
