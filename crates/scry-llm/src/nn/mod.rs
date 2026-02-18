pub mod attention;
pub mod embedding;
pub mod gpt2;
pub mod init;
pub mod kv_cache;
pub mod layernorm;
pub mod linear;
pub mod mlp;
pub mod transformer;

use crate::backend::MathBackend;
use crate::tensor::Tensor;

/// Training vs inference mode.
///
/// Currently used conceptually — dropout (step 3) relies on `tape.is_some()` to
/// distinguish training from inference. This enum exists for future ops that
/// need explicit mode control (e.g. batch normalization).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Train,
    Eval,
}

/// Trait for neural network modules.
///
/// Modules have their own `forward()` signatures (different input types).
/// The trait provides only parameter access for the optimizer.
pub trait Module<B: MathBackend> {
    fn parameters(&self) -> Vec<&Tensor<B>>;
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<B>>;
}
