use crate::backend::MathBackend;
use crate::nn::init;
use crate::nn::Module;
use crate::ops;
use crate::tensor::shape::Shape;
use crate::tensor::Tensor;

/// Linear layer: `output = input @ weight + bias`.
///
/// Weight stored as `[in_features, out_features]` — matches HF `Conv1D` layout.
pub struct Linear<B: MathBackend> {
    pub weight: Tensor<B>,
    pub bias: Tensor<B>,
    pub in_features: usize,
    pub out_features: usize,
}

impl<B: MathBackend> Linear<B> {
    pub fn new(in_features: usize, out_features: usize, rng: &mut fastrand::Rng) -> Self {
        let w_data = init::normal_vec(rng, in_features * out_features, 0.0, 0.02);
        let b_data = vec![0.0f32; out_features];
        Self {
            weight: Tensor::from_vec(w_data, Shape::new(&[in_features, out_features])),
            bias: Tensor::from_vec(b_data, Shape::new(&[out_features])),
            in_features,
            out_features,
        }
    }

    /// Apply residual scaling: multiply weights by `1/sqrt(2*n_layer)`.
    pub fn apply_residual_scaling(&mut self, n_layer: usize) {
        let scale = 1.0 / (2.0 * n_layer as f64).sqrt();
        let mut data = self.weight.to_vec();
        for v in &mut data {
            *v = (f64::from(*v) * scale) as f32;
        }
        self.weight = Tensor::from_vec(data, self.weight.shape.clone());
    }

    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let seq = input.shape.dims()[0];
        let mm = ops::matmul(
            input,
            &self.weight,
            seq,
            self.in_features,
            self.out_features,
            false,
            false,
        );
        ops::add(&mm, &self.bias)
    }
}

impl<B: MathBackend> Module<B> for Linear<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![&self.weight, &self.bias]
    }
}
