use crate::autograd::ops;
use crate::autograd::GradTape;
use crate::backend::MathBackend;
use crate::nn::init;
use crate::nn::Module;
use crate::tensor::shape::Shape;
use crate::tensor::Tensor;

/// Multi-head causal self-attention.
pub struct CausalSelfAttention<B: MathBackend> {
    pub qkv_weight: Tensor<B>,
    pub qkv_bias: Tensor<B>,
    pub proj_weight: Tensor<B>,
    pub proj_bias: Tensor<B>,
    pub n_heads: usize,
    pub d_model: usize,
    pub d_head: usize,
}

impl<B: MathBackend> CausalSelfAttention<B> {
    pub fn new(d_model: usize, n_heads: usize, rng: &mut fastrand::Rng) -> Self {
        let d_head = d_model / n_heads;
        let qkv_w = init::normal_vec(rng, d_model * 3 * d_model, 0.0, 0.02);
        let qkv_b = vec![0.0f32; 3 * d_model];
        let proj_w = init::normal_vec(rng, d_model * d_model, 0.0, 0.02);
        let proj_b = vec![0.0f32; d_model];
        Self {
            qkv_weight: Tensor::from_vec(qkv_w, Shape::new(&[d_model, 3 * d_model])),
            qkv_bias: Tensor::from_vec(qkv_b, Shape::new(&[3 * d_model])),
            proj_weight: Tensor::from_vec(proj_w, Shape::new(&[d_model, d_model])),
            proj_bias: Tensor::from_vec(proj_b, Shape::new(&[d_model])),
            n_heads,
            d_model,
            d_head,
        }
    }

    pub fn forward(&self, input: &Tensor<B>, tape: &mut GradTape<B>) -> Tensor<B> {
        ops::attention(
            input,
            &self.qkv_weight,
            &self.qkv_bias,
            &self.proj_weight,
            &self.proj_bias,
            self.n_heads,
            self.d_model,
            self.d_head,
            Some(tape),
        )
    }

    pub fn forward_inference(&self, input: &Tensor<B>) -> Tensor<B> {
        ops::attention(
            input,
            &self.qkv_weight,
            &self.qkv_bias,
            &self.proj_weight,
            &self.proj_bias,
            self.n_heads,
            self.d_model,
            self.d_head,
            None,
        )
    }
}

impl<B: MathBackend> Module<B> for CausalSelfAttention<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![
            &self.qkv_weight,
            &self.qkv_bias,
            &self.proj_weight,
            &self.proj_bias,
        ]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<B>> {
        vec![
            &mut self.qkv_weight,
            &mut self.qkv_bias,
            &mut self.proj_weight,
            &mut self.proj_bias,
        ]
    }
}
