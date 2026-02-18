pub mod cpu;

use crate::tensor::shape::Shape;

/// Memory management trait — allocate, copy, transfer between host/device.
pub trait DeviceBackend: Sized {
    type Storage: Clone;
    type Stream;

    fn zeros(shape: &Shape) -> Self::Storage;
    fn ones(shape: &Shape) -> Self::Storage;
    fn from_vec(data: Vec<f32>, shape: &Shape) -> Self::Storage;
    fn to_vec(storage: &Self::Storage) -> Vec<f32>;
    fn clone_storage(storage: &Self::Storage) -> Self::Storage;
}

/// Math operations trait — all operations needed for forward and backward passes.
pub trait MathBackend: DeviceBackend {
    // ---- Forward ops ----

    /// General matrix multiply: `C = alpha * op(A) * op(B) + beta * C`
    /// `A` is `[M, K]` (or `[K, M]` if `trans_a`), `B` is `[K, N]` (or `[N, K]` if `trans_b`)
    fn matmul(
        a: &Self::Storage,
        b: &Self::Storage,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> Self::Storage;

    /// Elementwise add with broadcasting.
    fn add(
        a: &Self::Storage,
        b: &Self::Storage,
        a_shape: &Shape,
        b_shape: &Shape,
        out_shape: &Shape,
    ) -> Self::Storage;

    /// Softmax along the last axis.
    fn softmax(input: &Self::Storage, shape: &Shape) -> Self::Storage;

    /// Layer normalization along the last axis.
    /// Returns `(output, mean, rstd)` for backward pass.
    fn layernorm(
        input: &Self::Storage,
        gamma: &Self::Storage,
        beta: &Self::Storage,
        shape: &Shape,
        eps: f32,
    ) -> (Self::Storage, Self::Storage, Self::Storage);

    /// GELU activation (tanh approximation).
    fn gelu(input: &Self::Storage) -> Self::Storage;

    /// Cross-entropy loss from logits.
    /// `logits`: `[B, V]`, `targets`: `[B]` (class indices as `f32`).
    /// Returns scalar loss.
    fn cross_entropy(logits: &Self::Storage, targets: &[usize], batch: usize, vocab: usize) -> f32;

    /// Embedding lookup: gather rows by indices.
    /// `weight`: `[V, D]`, `indices`: `[N]` → output: `[N, D]`
    fn embedding(
        weight: &Self::Storage,
        indices: &[usize],
        vocab: usize,
        dim: usize,
    ) -> Self::Storage;

    /// Sum all elements to scalar.
    fn sum(input: &Self::Storage) -> f32;

    // ---- Backward ops ----

    /// Backward for matmul. Returns `(d_a, d_b)`.
    fn matmul_backward(
        d_out: &Self::Storage,
        a: &Self::Storage,
        b: &Self::Storage,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> (Self::Storage, Self::Storage);

    /// Backward for add with broadcasting. Returns `(d_a, d_b)`.
    fn add_backward(
        d_out: &Self::Storage,
        a_shape: &Shape,
        b_shape: &Shape,
        out_shape: &Shape,
    ) -> (Self::Storage, Self::Storage);

    /// Backward for softmax. Returns `d_input`.
    fn softmax_backward(
        d_out: &Self::Storage,
        output: &Self::Storage,
        shape: &Shape,
    ) -> Self::Storage;

    /// Backward for layernorm. Returns `(d_input, d_gamma, d_beta)`.
    fn layernorm_backward(
        d_out: &Self::Storage,
        input: &Self::Storage,
        gamma: &Self::Storage,
        mean: &Self::Storage,
        rstd: &Self::Storage,
        shape: &Shape,
    ) -> (Self::Storage, Self::Storage, Self::Storage);

    /// Backward for GELU. Returns `d_input`.
    fn gelu_backward(d_out: &Self::Storage, input: &Self::Storage) -> Self::Storage;

    /// Backward for `cross_entropy`. Returns `d_logits` `[B, V]`.
    fn cross_entropy_backward(
        logits: &Self::Storage,
        targets: &[usize],
        batch: usize,
        vocab: usize,
    ) -> Self::Storage;

    /// Backward for embedding. Returns `d_weight` `[V, D]`.
    fn embedding_backward(
        d_out: &Self::Storage,
        indices: &[usize],
        vocab: usize,
        dim: usize,
    ) -> Self::Storage;

    /// Elementwise multiply: `a * b` (same shape, no broadcast).
    fn mul_elementwise(a: &Self::Storage, b: &Self::Storage) -> Self::Storage;

    /// Scale all elements: `a * scalar`.
    fn scale(a: &Self::Storage, scalar: f32) -> Self::Storage;

    /// Elementwise add in place: `a += b`. Shapes must match.
    fn add_inplace(a: &mut Self::Storage, b: &Self::Storage);

    /// L2 norm of all elements: `sqrt(sum(x^2))`.
    fn norm(storage: &Self::Storage) -> f32;

    /// In-place scalar multiply: `a[i] *= scalar` for all i.
    fn scale_inplace(a: &mut Self::Storage, scalar: f32);

    /// Concatenate two row-major matrices along rows (axis 0).
    /// `a`: `[a_rows, cols]`, `b`: `[b_rows, cols]` → `[a_rows + b_rows, cols]`.
    fn concat_rows(
        a: &Self::Storage,
        b: &Self::Storage,
        a_rows: usize,
        b_rows: usize,
        cols: usize,
    ) -> Self::Storage;

    /// `AdamW` optimizer step (fused). Updates `param` in place.
    #[allow(clippy::too_many_arguments)]
    fn adamw_step(
        param: &mut Self::Storage,
        grad: &Self::Storage,
        m: &mut Self::Storage,
        v: &mut Self::Storage,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: u32,
    );
}
