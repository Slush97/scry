pub mod backward;
pub mod ops;

use crate::backend::DeviceBackend;
use crate::tensor::shape::Shape;
use crate::tensor::TensorId;

/// What data each operation saves for backward.
pub enum SavedData<B: DeviceBackend> {
    Matmul {
        a: B::Storage,
        b: B::Storage,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    },
    Add {
        a_shape: Shape,
        b_shape: Shape,
        out_shape: Shape,
    },
    Softmax {
        output: B::Storage,
        shape: Shape,
    },
    LayerNorm {
        input: B::Storage,
        gamma: B::Storage,
        mean: B::Storage,
        rstd: B::Storage,
        shape: Shape,
        gamma_id: TensorId,
        beta_id: TensorId,
    },
    Gelu {
        input: B::Storage,
    },
    CrossEntropy {
        logits: B::Storage,
        targets: Vec<usize>,
        batch: usize,
        vocab: usize,
    },
    Embedding {
        indices: Vec<usize>,
        vocab: usize,
        dim: usize,
        weight_id: TensorId,
    },
    Sum {
        input_shape: Shape,
    },
    Attention {
        input: B::Storage,
        qkv_weight: B::Storage,
        proj_weight: B::Storage,
        attn_weights: Vec<Vec<f32>>,
        q_per_head: Vec<Vec<f32>>,
        k_per_head: Vec<Vec<f32>>,
        v_per_head: Vec<Vec<f32>>,
        head_concat: B::Storage,
        n_heads: usize,
        d_model: usize,
        d_head: usize,
        seq_len: usize,
        qkv_weight_id: TensorId,
        qkv_bias_id: TensorId,
        proj_weight_id: TensorId,
        proj_bias_id: TensorId,
    },
}

/// Which operation produced a tensor.
pub enum Operation {
    Matmul,
    Add,
    Softmax,
    LayerNorm,
    Gelu,
    CrossEntropy,
    Embedding,
    Sum,
    Attention,
}

/// A node on the autograd tape.
pub struct TapeNode<B: DeviceBackend> {
    pub output_id: TensorId,
    pub input_ids: Vec<TensorId>,
    pub op: Operation,
    pub saved: SavedData<B>,
}

/// Arena-based gradient tape. Records operations for backward pass.
pub struct GradTape<B: DeviceBackend> {
    pub nodes: Vec<TapeNode<B>>,
}

impl<B: DeviceBackend> GradTape<B> {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn record(&mut self, node: TapeNode<B>) {
        self.nodes.push(node);
    }
}

impl<B: DeviceBackend> Default for GradTape<B> {
    fn default() -> Self {
        Self::new()
    }
}
