use crate::autograd::ops;
use crate::autograd::GradTape;
use crate::backend::MathBackend;
use crate::nn::linear::Linear;
use crate::nn::Module;
use crate::tensor::Tensor;

/// Feed-forward network: `fc1 -> gelu -> fc2`.
pub struct Mlp<B: MathBackend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
}

impl<B: MathBackend> Mlp<B> {
    pub fn new(d_model: usize, d_ff: usize, rng: &mut fastrand::Rng) -> Self {
        Self {
            fc1: Linear::new(d_model, d_ff, rng),
            fc2: Linear::new(d_ff, d_model, rng),
        }
    }

    pub fn forward(&self, input: &Tensor<B>, tape: &mut GradTape<B>) -> Tensor<B> {
        // Fused: matmul + bias + gelu in one kernel pass (saves one tensor read+write)
        let h = ops::fused_linear_gelu(
            input,
            &self.fc1.weight,
            &self.fc1.bias,
            self.fc1.in_features,
            self.fc1.out_features,
            Some(tape),
        );
        self.fc2.forward(&h, tape)
    }

    /// Forward pass returning the fc1+gelu intermediate, for use with
    /// `fused_linear_dropout_residual` in the transformer block.
    pub fn forward_pre_fc2(&self, input: &Tensor<B>, tape: &mut GradTape<B>) -> Tensor<B> {
        ops::fused_linear_gelu(
            input,
            &self.fc1.weight,
            &self.fc1.bias,
            self.fc1.in_features,
            self.fc1.out_features,
            Some(tape),
        )
    }

    pub fn forward_inference(&self, input: &Tensor<B>) -> Tensor<B> {
        let h = self.fc1.forward_inference(input);
        let h = ops::gelu(&h, None);
        self.fc2.forward_inference(&h)
    }
}

impl<B: MathBackend> Module<B> for Mlp<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<B>> {
        let mut params = self.fc1.parameters_mut();
        params.extend(self.fc2.parameters_mut());
        params
    }
}
