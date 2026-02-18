use crate::autograd::ops;
use crate::autograd::GradTape;
use crate::backend::MathBackend;
use crate::nn::Module;
use crate::tensor::shape::Shape;
use crate::tensor::Tensor;

/// Layer normalization module.
pub struct LayerNormModule<B: MathBackend> {
    pub gamma: Tensor<B>,
    pub beta: Tensor<B>,
    pub eps: f32,
}

impl<B: MathBackend> LayerNormModule<B> {
    pub fn new(d_model: usize) -> Self {
        Self {
            gamma: Tensor::from_vec(vec![1.0; d_model], Shape::new(&[d_model])),
            beta: Tensor::from_vec(vec![0.0; d_model], Shape::new(&[d_model])),
            eps: 1e-5,
        }
    }

    pub fn forward(&self, input: &Tensor<B>, tape: &mut GradTape<B>) -> Tensor<B> {
        ops::layernorm(input, &self.gamma, &self.beta, self.eps, Some(tape))
    }

    pub fn forward_inference(&self, input: &Tensor<B>) -> Tensor<B> {
        ops::layernorm(input, &self.gamma, &self.beta, self.eps, None)
    }
}

impl<B: MathBackend> Module<B> for LayerNormModule<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![&self.gamma, &self.beta]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<B>> {
        vec![&mut self.gamma, &mut self.beta]
    }
}
