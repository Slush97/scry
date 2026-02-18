use crate::backend::MathBackend;
use crate::tensor::shape::Shape;
use crate::tensor::Tensor;

use super::{GradTape, Operation, SavedData, TapeNode};

/// Matrix multiply: `C = op(A) @ op(B)`
/// `A`: `[M, K]` (or `[K, M]` if `trans_a`), `B`: `[K, N]` (or `[N, K]` if `trans_b`)
/// Output: `[M, N]`
pub fn matmul<B: MathBackend>(
    a: &Tensor<B>,
    b: &Tensor<B>,
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
    tape: Option<&mut GradTape<B>>,
) -> Tensor<B> {
    let data = B::matmul(&a.data, &b.data, m, k, n, trans_a, trans_b);
    let out = Tensor::new(data, Shape::new(&[m, n]));

    if let Some(tape) = tape {
        tape.record(TapeNode {
            output_id: out.id,
            input_ids: vec![a.id, b.id],
            op: Operation::Matmul,
            saved: SavedData::Matmul {
                a: B::clone_storage(&a.data),
                b: B::clone_storage(&b.data),
                m,
                k,
                n,
                trans_a,
                trans_b,
            },
        });
    }

    out
}

/// Elementwise add with broadcasting.
///
/// # Panics
///
/// Panics if the shapes of `a` and `b` are not broadcast-compatible.
pub fn add<B: MathBackend>(
    a: &Tensor<B>,
    b: &Tensor<B>,
    tape: Option<&mut GradTape<B>>,
) -> Tensor<B> {
    let out_shape = Shape::broadcast(&a.shape, &b.shape).expect("broadcast failed in add");
    let data = B::add(&a.data, &b.data, &a.shape, &b.shape, &out_shape);
    let out = Tensor::new(data, out_shape.clone());

    if let Some(tape) = tape {
        tape.record(TapeNode {
            output_id: out.id,
            input_ids: vec![a.id, b.id],
            op: Operation::Add,
            saved: SavedData::Add {
                a_shape: a.shape.clone(),
                b_shape: b.shape.clone(),
                out_shape,
            },
        });
    }

    out
}

/// Softmax along the last axis.
pub fn softmax<B: MathBackend>(input: &Tensor<B>, tape: Option<&mut GradTape<B>>) -> Tensor<B> {
    let data = B::softmax(&input.data, &input.shape);
    let out = Tensor::new(data.clone(), input.shape.clone());

    if let Some(tape) = tape {
        tape.record(TapeNode {
            output_id: out.id,
            input_ids: vec![input.id],
            op: Operation::Softmax,
            saved: SavedData::Softmax {
                output: data,
                shape: input.shape.clone(),
            },
        });
    }

    out
}

/// Layer normalization along the last axis.
pub fn layernorm<B: MathBackend>(
    input: &Tensor<B>,
    gamma: &Tensor<B>,
    beta: &Tensor<B>,
    eps: f32,
    tape: Option<&mut GradTape<B>>,
) -> Tensor<B> {
    let (data, mean, rstd) = B::layernorm(&input.data, &gamma.data, &beta.data, &input.shape, eps);
    let out = Tensor::new(data, input.shape.clone());

    if let Some(tape) = tape {
        tape.record(TapeNode {
            output_id: out.id,
            input_ids: vec![input.id],
            op: Operation::LayerNorm,
            saved: SavedData::LayerNorm {
                input: B::clone_storage(&input.data),
                gamma: B::clone_storage(&gamma.data),
                mean,
                rstd,
                shape: input.shape.clone(),
                gamma_id: gamma.id,
                beta_id: beta.id,
            },
        });
    }

    out
}

/// GELU activation (tanh approximation).
pub fn gelu<B: MathBackend>(input: &Tensor<B>, tape: Option<&mut GradTape<B>>) -> Tensor<B> {
    let data = B::gelu(&input.data);
    let out = Tensor::new(data, input.shape.clone());

    if let Some(tape) = tape {
        tape.record(TapeNode {
            output_id: out.id,
            input_ids: vec![input.id],
            op: Operation::Gelu,
            saved: SavedData::Gelu {
                input: B::clone_storage(&input.data),
            },
        });
    }

    out
}

/// Cross-entropy loss from logits.
/// `logits`: `[B, V]`, `targets`: `[B]` as `usize` indices.
/// Returns scalar loss tensor.
pub fn cross_entropy<B: MathBackend>(
    logits: &Tensor<B>,
    targets: &[usize],
    batch: usize,
    vocab: usize,
    tape: Option<&mut GradTape<B>>,
) -> Tensor<B> {
    let loss_val = B::cross_entropy(&logits.data, targets, batch, vocab);
    let data = B::from_vec(vec![loss_val], &Shape::new(&[1]));
    let out = Tensor::new(data, Shape::new(&[1]));

    if let Some(tape) = tape {
        tape.record(TapeNode {
            output_id: out.id,
            input_ids: vec![logits.id],
            op: Operation::CrossEntropy,
            saved: SavedData::CrossEntropy {
                logits: B::clone_storage(&logits.data),
                targets: targets.to_vec(),
                batch,
                vocab,
            },
        });
    }

    out
}

/// Embedding lookup.
/// `weight`: `[V, D]`, `indices`: `[N]` → output: `[N, D]`
pub fn embedding<B: MathBackend>(
    weight: &Tensor<B>,
    indices: &[usize],
    vocab: usize,
    dim: usize,
    tape: Option<&mut GradTape<B>>,
) -> Tensor<B> {
    let data = B::embedding(&weight.data, indices, vocab, dim);
    let out = Tensor::new(data, Shape::new(&[indices.len(), dim]));

    if let Some(tape) = tape {
        tape.record(TapeNode {
            output_id: out.id,
            input_ids: vec![weight.id],
            op: Operation::Embedding,
            saved: SavedData::Embedding {
                indices: indices.to_vec(),
                vocab,
                dim,
                weight_id: weight.id,
            },
        });
    }

    out
}

/// Sum all elements to a scalar.
pub fn sum<B: MathBackend>(input: &Tensor<B>, tape: Option<&mut GradTape<B>>) -> Tensor<B> {
    let val = B::sum(&input.data);
    let data = B::from_vec(vec![val], &Shape::new(&[1]));
    let out = Tensor::new(data, Shape::new(&[1]));

    if let Some(tape) = tape {
        tape.record(TapeNode {
            output_id: out.id,
            input_ids: vec![input.id],
            op: Operation::Sum,
            saved: SavedData::Sum {
                input_shape: input.shape.clone(),
            },
        });
    }

    out
}

/// Multi-head causal self-attention as a single autograd op.
///
/// `input`: `[seq, d_model]`, `qkv_weight`: `[d_model, 3*d_model]`, `qkv_bias`: `[3*d_model]`,
/// `proj_weight`: `[d_model, d_model]`, `proj_bias`: `[d_model]`.
///
/// Returns `[seq, d_model]`.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn attention<B: MathBackend>(
    input: &Tensor<B>,
    qkv_weight: &Tensor<B>,
    qkv_bias: &Tensor<B>,
    proj_weight: &Tensor<B>,
    proj_bias: &Tensor<B>,
    n_heads: usize,
    d_model: usize,
    d_head: usize,
    dropout_rate: f32,
    mut rng: Option<&mut fastrand::Rng>,
    tape: Option<&mut GradTape<B>>,
) -> Tensor<B> {
    let seq_len = input.shape.dims()[0];

    // QKV = input @ W_qkv + b_qkv  => [seq, 3*d_model]
    let qkv_raw = B::matmul(
        &input.data,
        &qkv_weight.data,
        seq_len,
        d_model,
        3 * d_model,
        false,
        false,
    );
    let qkv_shape = Shape::new(&[seq_len, 3 * d_model]);
    let bias_shape = Shape::new(&[1, 3 * d_model]);
    let qkv = B::add(
        &qkv_raw,
        &qkv_bias.data,
        &qkv_shape,
        &bias_shape,
        &qkv_shape,
    );
    let qkv_vec = B::to_vec(&qkv);

    let is_training = tape.is_some();

    // Split into heads and compute attention
    let mut all_attn_weights: Vec<B::Storage> = Vec::with_capacity(n_heads);
    let mut all_q: Vec<B::Storage> = Vec::with_capacity(n_heads);
    let mut all_k: Vec<B::Storage> = Vec::with_capacity(n_heads);
    let mut all_v: Vec<B::Storage> = Vec::with_capacity(n_heads);
    let mut all_attn_dropout_masks: Vec<B::Storage> = Vec::with_capacity(n_heads);
    let mut head_concat = vec![0.0f32; seq_len * d_model];

    let scale = 1.0 / (d_head as f64).sqrt();

    for h in 0..n_heads {
        // Extract Q, K, V for this head: [seq, d_head] each
        let q_offset = h * d_head;
        let k_offset = d_model + h * d_head;
        let v_offset = 2 * d_model + h * d_head;

        let mut q_h = vec![0.0f32; seq_len * d_head];
        let mut k_h = vec![0.0f32; seq_len * d_head];
        let mut v_h = vec![0.0f32; seq_len * d_head];

        for s in 0..seq_len {
            for d in 0..d_head {
                q_h[s * d_head + d] = qkv_vec[s * 3 * d_model + q_offset + d];
                k_h[s * d_head + d] = qkv_vec[s * 3 * d_model + k_offset + d];
                v_h[s * d_head + d] = qkv_vec[s * 3 * d_model + v_offset + d];
            }
        }

        // scores = Q @ K^T / sqrt(d_head) => [seq, seq]
        let scores_raw = B::matmul(
            &B::from_vec(q_h.clone(), &Shape::new(&[seq_len, d_head])),
            &B::from_vec(k_h.clone(), &Shape::new(&[seq_len, d_head])),
            seq_len,
            d_head,
            seq_len,
            false,
            true,
        );
        let mut scores = B::to_vec(&scores_raw);
        for s in 0..seq_len {
            for t in 0..seq_len {
                scores[s * seq_len + t] = (f64::from(scores[s * seq_len + t]) * scale) as f32;
            }
        }

        // Causal mask: upper triangle -> -inf
        for s in 0..seq_len {
            for t in (s + 1)..seq_len {
                scores[s * seq_len + t] = f32::NEG_INFINITY;
            }
        }

        // Softmax over last dim
        let scores_storage = B::from_vec(scores, &Shape::new(&[seq_len, seq_len]));
        let attn = B::softmax(&scores_storage, &Shape::new(&[seq_len, seq_len]));
        let attn_vec = B::to_vec(&attn);

        // Apply dropout to attention weights (inverted dropout)
        let (attn_for_v, dropout_mask_vec) =
            if is_training && dropout_rate > 0.0 && dropout_rate < 1.0 {
                let rng = rng
                    .as_deref_mut()
                    .expect("rng required for attention dropout during training");
                let n = attn_vec.len();
                let scale_inv = 1.0 / (1.0 - dropout_rate);
                let mut mask = vec![0.0f32; n];
                let mut dropped = attn_vec.clone();
                for i in 0..n {
                    if rng.f32() >= dropout_rate {
                        mask[i] = scale_inv;
                        dropped[i] *= scale_inv;
                    } else {
                        mask[i] = 0.0;
                        dropped[i] = 0.0;
                    }
                }
                (dropped, mask)
            } else {
                let n = attn_vec.len();
                (attn_vec.clone(), vec![1.0f32; n])
            };
        let attn_shape = Shape::new(&[seq_len, seq_len]);
        let attn_dropped = B::from_vec(attn_for_v, &attn_shape);

        // out_h = attn @ V => [seq, d_head]
        let out_h = B::matmul(
            &attn_dropped,
            &B::from_vec(v_h.clone(), &Shape::new(&[seq_len, d_head])),
            seq_len,
            seq_len,
            d_head,
            false,
            false,
        );
        let out_h_vec = B::to_vec(&out_h);

        // Scatter into head_concat: [seq, d_model]
        for s in 0..seq_len {
            for d in 0..d_head {
                head_concat[s * d_model + h * d_head + d] = out_h_vec[s * d_head + d];
            }
        }

        let head_shape = Shape::new(&[seq_len, d_head]);
        all_attn_weights.push(B::from_vec(attn_vec, &attn_shape));
        all_q.push(B::from_vec(q_h, &head_shape));
        all_k.push(B::from_vec(k_h, &head_shape));
        all_v.push(B::from_vec(v_h, &head_shape));
        all_attn_dropout_masks.push(B::from_vec(dropout_mask_vec, &attn_shape));
    }

    // Output projection: head_concat @ W_proj + b_proj => [seq, d_model]
    let hc_storage = B::from_vec(head_concat.clone(), &Shape::new(&[seq_len, d_model]));
    let proj_raw = B::matmul(
        &hc_storage,
        &proj_weight.data,
        seq_len,
        d_model,
        d_model,
        false,
        false,
    );
    let proj_shape = Shape::new(&[seq_len, d_model]);
    let pbias_shape = Shape::new(&[1, d_model]);
    let output_data = B::add(
        &proj_raw,
        &proj_bias.data,
        &proj_shape,
        &pbias_shape,
        &proj_shape,
    );

    let out = Tensor::new(output_data, Shape::new(&[seq_len, d_model]));

    if let Some(tape) = tape {
        tape.record(TapeNode {
            output_id: out.id,
            input_ids: vec![input.id],
            op: Operation::Attention,
            saved: SavedData::Attention {
                input: B::clone_storage(&input.data),
                qkv_weight: B::clone_storage(&qkv_weight.data),
                proj_weight: B::clone_storage(&proj_weight.data),
                attn_weights: all_attn_weights,
                q_per_head: all_q,
                k_per_head: all_k,
                v_per_head: all_v,
                attn_dropout_masks: all_attn_dropout_masks,
                head_concat: B::from_vec(head_concat, &Shape::new(&[seq_len, d_model])),
                n_heads,
                d_model,
                d_head,
                seq_len,
                qkv_weight_id: qkv_weight.id,
                qkv_bias_id: qkv_bias.id,
                proj_weight_id: proj_weight.id,
                proj_bias_id: proj_bias.id,
            },
        });
    }

    out
}

/// Inverted dropout: during training, randomly zeros elements with probability `p`
/// and scales remaining by `1/(1-p)`. During inference (`tape=None`), acts as identity.
///
/// # Panics
///
/// Panics if `p` is not in `[0, 1]`.
pub fn dropout<B: MathBackend>(
    input: &Tensor<B>,
    p: f32,
    rng: &mut fastrand::Rng,
    tape: Option<&mut GradTape<B>>,
) -> Tensor<B> {
    assert!(
        (0.0..=1.0).contains(&p),
        "dropout rate must be in [0, 1], got {p}"
    );

    // Inference: identity (no tape)
    if tape.is_none() {
        return Tensor::new(B::clone_storage(&input.data), input.shape.clone());
    }

    // p=0: record identity on tape for gradient flow
    if p == 0.0 {
        let out = Tensor::new(B::clone_storage(&input.data), input.shape.clone());
        if let Some(tape) = tape {
            tape.record(TapeNode {
                output_id: out.id,
                input_ids: vec![input.id],
                op: Operation::Dropout,
                saved: SavedData::Dropout {
                    mask: B::ones(&input.shape),
                },
            });
        }
        return out;
    }

    // p=1: all zeros
    #[allow(clippy::float_cmp)]
    if p == 1.0 {
        let zeros = B::zeros(&input.shape);
        let out = Tensor::new(zeros, input.shape.clone());
        if let Some(tape) = tape {
            tape.record(TapeNode {
                output_id: out.id,
                input_ids: vec![input.id],
                op: Operation::Dropout,
                saved: SavedData::Dropout {
                    mask: B::zeros(&input.shape),
                },
            });
        }
        return out;
    }

    let data = B::to_vec(&input.data);
    let n = data.len();
    let scale = 1.0 / (1.0 - p);
    let mut mask = vec![0.0f32; n];
    let mut output = vec![0.0f32; n];

    for i in 0..n {
        if rng.f32() >= p {
            mask[i] = scale;
            output[i] = data[i] * scale;
        }
    }

    let mask_storage = B::from_vec(mask, &input.shape);
    let out = Tensor::new(B::from_vec(output, &input.shape), input.shape.clone());

    if let Some(tape) = tape {
        tape.record(TapeNode {
            output_id: out.id,
            input_ids: vec![input.id],
            op: Operation::Dropout,
            saved: SavedData::Dropout { mask: mask_storage },
        });
    }

    out
}
