use std::collections::HashMap;

use crate::backend::{DeviceBackend, MathBackend};
use crate::tensor::shape::Shape;
use crate::tensor::TensorId;

#[allow(unused_imports)]
use super::{GradTape, Operation, SavedData};

/// Gradient storage: maps [`TensorId`] to its accumulated gradient.
pub type Gradients<B> = HashMap<TensorId, <B as DeviceBackend>::Storage>;

/// Run backward pass from a scalar loss.
/// `loss_id` must be the output of the last recorded op.
/// Returns gradients for all tensors that require them.
#[allow(clippy::too_many_lines)]
pub fn backward<B: MathBackend>(tape: &GradTape<B>, loss_id: TensorId) -> Gradients<B> {
    let mut grads: Gradients<B> = HashMap::new();

    // Seed: gradient of loss w.r.t. itself is 1.0
    let ones = B::from_vec(vec![1.0], &Shape::new(&[1]));
    grads.insert(loss_id, ones);

    // Reverse iteration through tape
    for node in tape.nodes.iter().rev() {
        let d_out = match grads.get(&node.output_id) {
            Some(g) => B::clone_storage(g),
            None => continue, // no gradient flows to this node
        };

        match (&node.op, &node.saved) {
            (
                Operation::Matmul,
                SavedData::Matmul {
                    a,
                    b,
                    m,
                    k,
                    n,
                    trans_a,
                    trans_b,
                },
            ) => {
                let (d_a, d_b) = B::matmul_backward(&d_out, a, b, *m, *k, *n, *trans_a, *trans_b);
                accumulate_grad::<B>(&mut grads, node.input_ids[0], d_a);
                accumulate_grad::<B>(&mut grads, node.input_ids[1], d_b);
            }
            (
                Operation::Add,
                SavedData::Add {
                    a_shape,
                    b_shape,
                    out_shape,
                },
            ) => {
                let (d_a, d_b) = B::add_backward(&d_out, a_shape, b_shape, out_shape);
                accumulate_grad::<B>(&mut grads, node.input_ids[0], d_a);
                accumulate_grad::<B>(&mut grads, node.input_ids[1], d_b);
            }
            (Operation::Softmax, SavedData::Softmax { output, shape }) => {
                let d_input = B::softmax_backward(&d_out, output, shape);
                accumulate_grad::<B>(&mut grads, node.input_ids[0], d_input);
            }
            (
                Operation::LayerNorm,
                SavedData::LayerNorm {
                    input,
                    gamma,
                    mean,
                    rstd,
                    shape,
                    gamma_id,
                    beta_id,
                },
            ) => {
                let (d_input, d_gamma, d_beta) =
                    B::layernorm_backward(&d_out, input, gamma, mean, rstd, shape);
                accumulate_grad::<B>(&mut grads, node.input_ids[0], d_input);
                accumulate_grad::<B>(&mut grads, *gamma_id, d_gamma);
                accumulate_grad::<B>(&mut grads, *beta_id, d_beta);
            }
            (Operation::Gelu, SavedData::Gelu { input }) => {
                let d_input = B::gelu_backward(&d_out, input);
                accumulate_grad::<B>(&mut grads, node.input_ids[0], d_input);
            }
            (
                Operation::CrossEntropy,
                SavedData::CrossEntropy {
                    logits,
                    targets,
                    batch,
                    vocab,
                },
            ) => {
                let d_logits = B::cross_entropy_backward(logits, targets, *batch, *vocab);
                // Scale by upstream gradient (chain rule)
                let d_out_scalar = B::to_vec(&d_out)[0];
                let d_logits_scaled = B::scale(&d_logits, d_out_scalar);
                accumulate_grad::<B>(&mut grads, node.input_ids[0], d_logits_scaled);
            }
            (
                Operation::Embedding,
                SavedData::Embedding {
                    indices,
                    vocab,
                    dim,
                    weight_id,
                },
            ) => {
                let d_weight = B::embedding_backward(&d_out, indices, *vocab, *dim);
                accumulate_grad::<B>(&mut grads, *weight_id, d_weight);
            }
            (Operation::Sum, SavedData::Sum { input_shape }) => {
                // Gradient of sum is broadcast 1s scaled by upstream
                let d_out_scalar = B::to_vec(&d_out)[0];
                let d_input = B::from_vec(vec![d_out_scalar; input_shape.numel()], input_shape);
                accumulate_grad::<B>(&mut grads, node.input_ids[0], d_input);
            }
            (
                Operation::Attention,
                SavedData::Attention {
                    input,
                    qkv_weight,
                    proj_weight,
                    attn_weights,
                    q_per_head,
                    k_per_head,
                    v_per_head,
                    head_concat,
                    n_heads,
                    d_model,
                    d_head,
                    seq_len,
                    qkv_weight_id,
                    qkv_bias_id,
                    proj_weight_id,
                    proj_bias_id,
                },
            ) => {
                let n_heads = *n_heads;
                let d_model = *d_model;
                let d_head = *d_head;
                let seq_len = *seq_len;

                attention_backward::<B>(
                    &d_out,
                    input,
                    qkv_weight,
                    proj_weight,
                    head_concat,
                    attn_weights,
                    q_per_head,
                    k_per_head,
                    v_per_head,
                    n_heads,
                    d_model,
                    d_head,
                    seq_len,
                    &mut grads,
                    node.input_ids[0],
                    *qkv_weight_id,
                    *qkv_bias_id,
                    *proj_weight_id,
                    *proj_bias_id,
                );
            }
            _ => unreachable!("mismatched Operation/SavedData variants: backward dispatch bug"),
        }
    }

    grads
}

fn accumulate_grad<B: MathBackend>(grads: &mut Gradients<B>, id: TensorId, grad: B::Storage) {
    if let Some(existing) = grads.get_mut(&id) {
        B::add_inplace(existing, &grad);
    } else {
        grads.insert(id, grad);
    }
}

/// Backward pass for the fused multi-head causal self-attention op.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names
)]
fn attention_backward<B: MathBackend>(
    d_out: &B::Storage,
    input: &B::Storage,
    qkv_weight: &B::Storage,
    proj_weight: &B::Storage,
    head_concat: &B::Storage,
    attn_weights: &[Vec<f32>],
    q_per_head: &[Vec<f32>],
    k_per_head: &[Vec<f32>],
    v_per_head: &[Vec<f32>],
    n_heads: usize,
    d_model: usize,
    d_head: usize,
    seq_len: usize,
    grads: &mut Gradients<B>,
    input_id: TensorId,
    qkv_weight_id: TensorId,
    qkv_bias_id: TensorId,
    proj_weight_id: TensorId,
    proj_bias_id: TensorId,
) {
    // d_out is [seq, d_model]
    // output = head_concat @ W_proj + b_proj

    // d_proj_bias = sum over seq of d_out => [d_model]
    // but with broadcasting it's sum over rows
    let d_out_vec = B::to_vec(d_out);
    let mut d_proj_bias = vec![0.0f32; d_model];
    for s in 0..seq_len {
        for d in 0..d_model {
            d_proj_bias[d] += d_out_vec[s * d_model + d];
        }
    }
    accumulate_grad::<B>(
        grads,
        proj_bias_id,
        B::from_vec(d_proj_bias, &Shape::new(&[d_model])),
    );

    // d_head_concat = d_out @ W_proj^T => [seq, d_model]
    let d_head_concat = B::matmul(d_out, proj_weight, seq_len, d_model, d_model, false, true);

    // d_W_proj = head_concat^T @ d_out => [d_model, d_model]
    let d_proj_weight = B::matmul(head_concat, d_out, d_model, seq_len, d_model, true, false);
    accumulate_grad::<B>(grads, proj_weight_id, d_proj_weight);

    let d_hc_vec = B::to_vec(&d_head_concat);
    let scale = 1.0 / (d_head as f64).sqrt();

    // Accumulate d_QKV across heads
    let mut d_qkv = vec![0.0f32; seq_len * 3 * d_model];

    for h in 0..n_heads {
        // Extract d_out_h from d_head_concat for this head
        let mut d_out_h = vec![0.0f32; seq_len * d_head];
        for s in 0..seq_len {
            for d in 0..d_head {
                d_out_h[s * d_head + d] = d_hc_vec[s * d_model + h * d_head + d];
            }
        }

        let attn = &attn_weights[h]; // [seq, seq]
        let q_h = &q_per_head[h]; // [seq, d_head]
        let k_h = &k_per_head[h];
        let v_h = &v_per_head[h];

        // d_attn = d_out_h @ V_h^T => [seq, seq]
        let d_attn = B::matmul(
            &B::from_vec(d_out_h.clone(), &Shape::new(&[seq_len, d_head])),
            &B::from_vec(v_h.clone(), &Shape::new(&[seq_len, d_head])),
            seq_len,
            d_head,
            seq_len,
            false,
            true,
        );

        // d_V_h = attn^T @ d_out_h => [seq, d_head]
        let d_v_h = B::matmul(
            &B::from_vec(attn.clone(), &Shape::new(&[seq_len, seq_len])),
            &B::from_vec(d_out_h, &Shape::new(&[seq_len, d_head])),
            seq_len,
            seq_len,
            d_head,
            true,
            false,
        );
        let d_v_h_vec = B::to_vec(&d_v_h);

        // d_scores = softmax_backward(d_attn, attn) => [seq, seq]
        let d_scores = B::softmax_backward(
            &d_attn,
            &B::from_vec(attn.clone(), &Shape::new(&[seq_len, seq_len])),
            &Shape::new(&[seq_len, seq_len]),
        );
        let mut d_scores_vec = B::to_vec(&d_scores);

        // Apply scale and causal mask to d_scores
        for s in 0..seq_len {
            for t in 0..seq_len {
                if t > s {
                    d_scores_vec[s * seq_len + t] = 0.0;
                } else {
                    d_scores_vec[s * seq_len + t] =
                        (f64::from(d_scores_vec[s * seq_len + t]) * scale) as f32;
                }
            }
        }

        // d_Q_h = d_scores @ K_h => [seq, d_head]
        let d_q_h = B::matmul(
            &B::from_vec(d_scores_vec.clone(), &Shape::new(&[seq_len, seq_len])),
            &B::from_vec(k_h.clone(), &Shape::new(&[seq_len, d_head])),
            seq_len,
            seq_len,
            d_head,
            false,
            false,
        );
        let d_q_h_vec = B::to_vec(&d_q_h);

        // d_K_h = d_scores^T @ Q_h => [seq, d_head]
        let d_k_h = B::matmul(
            &B::from_vec(d_scores_vec, &Shape::new(&[seq_len, seq_len])),
            &B::from_vec(q_h.clone(), &Shape::new(&[seq_len, d_head])),
            seq_len,
            seq_len,
            d_head,
            true,
            false,
        );
        let d_k_h_vec = B::to_vec(&d_k_h);

        // Scatter d_Q, d_K, d_V into d_QKV
        let q_offset = h * d_head;
        let k_offset = d_model + h * d_head;
        let v_offset = 2 * d_model + h * d_head;

        for s in 0..seq_len {
            for d in 0..d_head {
                d_qkv[s * 3 * d_model + q_offset + d] += d_q_h_vec[s * d_head + d];
                d_qkv[s * 3 * d_model + k_offset + d] += d_k_h_vec[s * d_head + d];
                d_qkv[s * 3 * d_model + v_offset + d] += d_v_h_vec[s * d_head + d];
            }
        }
    }

    // d_input = d_QKV @ W_qkv^T => [seq, d_model]
    let d_qkv_storage = B::from_vec(d_qkv.clone(), &Shape::new(&[seq_len, 3 * d_model]));
    let d_input = B::matmul(
        &d_qkv_storage,
        qkv_weight,
        seq_len,
        3 * d_model,
        d_model,
        false,
        true,
    );
    accumulate_grad::<B>(grads, input_id, d_input);

    // d_W_qkv = input^T @ d_QKV => [d_model, 3*d_model]
    let d_qkv_weight = B::matmul(
        input,
        &d_qkv_storage,
        d_model,
        seq_len,
        3 * d_model,
        true,
        false,
    );
    accumulate_grad::<B>(grads, qkv_weight_id, d_qkv_weight);

    // d_qkv_bias = sum over seq of d_QKV => [3*d_model]
    let mut d_qkv_bias = vec![0.0f32; 3 * d_model];
    for s in 0..seq_len {
        for d in 0..(3 * d_model) {
            d_qkv_bias[d] += d_qkv[s * 3 * d_model + d];
        }
    }
    accumulate_grad::<B>(
        grads,
        qkv_bias_id,
        B::from_vec(d_qkv_bias, &Shape::new(&[3 * d_model])),
    );
}
