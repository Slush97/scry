//! Neural network engine — chains layers for forward/backward propagation.
//!
//! The [`Network`] struct manages a stack of [`DenseLayer`]s and provides
//! a unified forward/backward interface used by both `MLPClassifier` and
//! `MLPRegressor`.

use crate::accel::{self, ComputeBackend};
use crate::neural::activation::Activation;
use crate::neural::layer::{DenseLayer, FastRng};
use crate::neural::optimizer::OptimizerState;

/// GPU dispatch threshold: use backend matmul when batch * max_dim exceeds this.
const GPU_THRESHOLD: usize = 4096;

/// A feedforward neural network.
pub(crate) struct Network {
    pub layers: Vec<DenseLayer>,
    backend: Box<dyn ComputeBackend>,
}

impl Network {
    /// Build a network from layer sizes (including input and output).
    ///
    /// `sizes` example: `[n_features, 100, 50, n_outputs]`
    /// `activation` is applied to all hidden layers; the output layer uses Identity.
    pub fn new(sizes: &[usize], activation: Activation, seed: u64) -> Self {
        assert!(sizes.len() >= 2, "need at least input and output sizes");
        let mut rng = FastRng::new(seed);
        let n_layers = sizes.len() - 1;

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let act = if i < n_layers - 1 {
                activation
            } else {
                // Output layer: Identity (softmax/loss handled separately)
                Activation::Identity
            };
            layers.push(DenseLayer::new(sizes[i], sizes[i + 1], act, &mut rng));
        }

        Self {
            layers,
            backend: accel::auto(),
        }
    }

    /// Forward pass through all layers.
    ///
    /// `input` is row-major `[batch, n_features]`.
    /// Returns the output `[batch, n_outputs]`.
    pub fn forward(&mut self, input: &[f64], batch: usize, training: bool) -> Vec<f64> {
        let mut current = input.to_vec();
        let use_gpu = batch * self.max_dim() >= GPU_THRESHOLD;

        for layer in &mut self.layers {
            current = if use_gpu {
                layer.forward_with_backend(&current, batch, training, self.backend.as_ref())
            } else {
                layer.forward(&current, batch, training)
            };
        }
        current
    }

    /// Backward pass through all layers.
    ///
    /// `grad_output` is `[batch, n_outputs]` — the gradient of loss w.r.t. network output.
    /// `alpha` is the L2 regularization strength.
    ///
    /// Returns weight and bias gradients for all layers.
    pub fn backward(
        &self,
        grad_output: &[f64],
        alpha: f64,
    ) -> Vec<(Vec<f64>, Vec<f64>)> {
        let n = self.layers.len();
        let mut grads = Vec::with_capacity(n);

        let mut current_grad = grad_output.to_vec();

        for i in (0..n).rev() {
            let layer = &self.layers[i];
            let mut dw = vec![0.0; layer.out_size * layer.in_size];
            let mut db = vec![0.0; layer.out_size];

            current_grad = layer.backward(&current_grad, &mut dw, &mut db);

            // L2 regularization on weights only (not biases)
            if alpha > 0.0 {
                for (w_idx, dw_val) in dw.iter_mut().enumerate() {
                    *dw_val += alpha * layer.weights[w_idx];
                }
            }

            grads.push((dw, db));
        }

        // Reverse so grads[i] corresponds to layers[i]
        grads.reverse();
        grads
    }

    /// Apply optimizer step to all layers.
    pub fn apply_gradients(
        &mut self,
        grads: &[(Vec<f64>, Vec<f64>)],
        optimizer: &mut OptimizerState,
    ) {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let (ref dw, ref db) = grads[i];
            let w_idx = i * 2;
            let b_idx = i * 2 + 1;
            optimizer.step(w_idx, &mut layer.weights, dw);
            optimizer.step(b_idx, &mut layer.biases, db);
        }
    }

    /// Build optimizer param group sizes: [w0_size, b0_size, w1_size, b1_size, ...].
    pub fn param_group_sizes(&self) -> Vec<usize> {
        let mut sizes = Vec::with_capacity(self.layers.len() * 2);
        for layer in &self.layers {
            sizes.push(layer.weights.len());
            sizes.push(layer.biases.len());
        }
        sizes
    }

    /// Total number of trainable parameters.
    #[allow(dead_code)]
    pub fn n_params(&self) -> usize {
        self.layers.iter().map(DenseLayer::n_params).sum()
    }

    /// Get the maximum dimension across all layers (for GPU threshold).
    fn max_dim(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.in_size.max(l.out_size))
            .max()
            .unwrap_or(0)
    }

    /// Clone all weights and biases (for early stopping best-weight saving).
    pub fn save_weights(&self) -> Vec<(Vec<f64>, Vec<f64>)> {
        self.layers
            .iter()
            .map(|l| (l.weights.clone(), l.biases.clone()))
            .collect()
    }

    /// Restore weights and biases from a saved snapshot.
    pub fn restore_weights(&mut self, saved: &[(Vec<f64>, Vec<f64>)]) {
        for (layer, (w, b)) in self.layers.iter_mut().zip(saved.iter()) {
            layer.weights.clone_from(w);
            layer.biases.clone_from(b);
        }
    }

    /// Get weights for each layer (for visualization).
    #[allow(dead_code)]
    pub fn layer_weights(&self) -> Vec<&[f64]> {
        self.layers.iter().map(|l| l.weights.as_slice()).collect()
    }

    /// Get layer dimensions: (in_size, out_size) for each layer.
    pub fn layer_dims(&self) -> Vec<(usize, usize)> {
        self.layers.iter().map(|l| (l.in_size, l.out_size)).collect()
    }
}

// ── Loss functions ──

/// Cross-entropy loss with numerically stable log-sum-exp.
///
/// `logits` is `[batch, n_classes]` (raw network output, no softmax).
/// `targets` is `[batch]` with class indices as f64.
///
/// Returns (mean loss, gradient `[batch, n_classes]`).
pub(crate) fn cross_entropy_loss(
    logits: &[f64],
    targets: &[f64],
    batch: usize,
    n_classes: usize,
) -> (f64, Vec<f64>) {
    debug_assert_eq!(logits.len(), batch * n_classes);
    debug_assert_eq!(targets.len(), batch);

    let mut grad = vec![0.0; batch * n_classes];
    let mut total_loss = 0.0;

    for i in 0..batch {
        let row = &logits[i * n_classes..(i + 1) * n_classes];
        let target_class = targets[i] as usize;

        // Log-sum-exp trick for numerical stability
        let max_logit = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = row.iter().map(|&x| (x - max_logit).exp()).sum();
        let log_sum_exp = max_logit + sum_exp.ln();

        // Loss = -log(softmax[target]) = -(logit[target] - log_sum_exp)
        total_loss += log_sum_exp - row[target_class];

        // Gradient of cross-entropy w.r.t. logits = softmax - one_hot
        for j in 0..n_classes {
            let softmax_j = (row[j] - log_sum_exp).exp();
            grad[i * n_classes + j] = softmax_j;
        }
        grad[i * n_classes + target_class] -= 1.0;
    }

    (total_loss / batch as f64, grad)
}

/// Mean squared error loss.
///
/// `predictions` is `[batch, 1]` (or `[batch]` flattened).
/// `targets` is `[batch]`.
///
/// Returns (mean loss, gradient `[batch, 1]`).
pub(crate) fn mse_loss(
    predictions: &[f64],
    targets: &[f64],
    batch: usize,
) -> (f64, Vec<f64>) {
    debug_assert_eq!(predictions.len(), batch);
    debug_assert_eq!(targets.len(), batch);

    let mut total_loss = 0.0;
    let mut grad = vec![0.0; batch];
    let batch_f = batch as f64;

    for i in 0..batch {
        let diff = predictions[i] - targets[i];
        total_loss += diff * diff;
        grad[i] = 2.0 * diff / batch_f;
    }

    (total_loss / batch_f, grad)
}

/// Convert softmax probabilities to class predictions.
pub(crate) fn argmax_predictions(probs: &[f64], batch: usize, n_classes: usize) -> Vec<f64> {
    let mut preds = Vec::with_capacity(batch);
    for i in 0..batch {
        let row = &probs[i * n_classes..(i + 1) * n_classes];
        let (max_idx, _) = row
            .iter()
            .enumerate()
            .fold((0, f64::NEG_INFINITY), |(bi, bv), (idx, &v)| {
                if v > bv { (idx, v) } else { (bi, bv) }
            });
        preds.push(max_idx as f64);
    }
    preds
}

/// Compute softmax probabilities from logits.
pub(crate) fn softmax(logits: &[f64], batch: usize, n_classes: usize) -> Vec<f64> {
    let mut probs = vec![0.0; batch * n_classes];
    for i in 0..batch {
        let row = &logits[i * n_classes..(i + 1) * n_classes];
        let max_val = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = row.iter().map(|&x| (x - max_val).exp()).sum();
        for j in 0..n_classes {
            probs[i * n_classes + j] = (row[j] - max_val).exp() / sum_exp;
        }
    }
    probs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_forward_shape() {
        let mut net = Network::new(&[4, 10, 3], Activation::Relu, 42);
        let input = vec![1.0; 2 * 4]; // batch=2, features=4
        let output = net.forward(&input, 2, false);
        assert_eq!(output.len(), 2 * 3);
    }

    #[test]
    fn cross_entropy_gradient_sums_to_zero() {
        let logits = vec![2.0, 1.0, 0.1, 0.5, 2.5, 0.3];
        let targets = vec![0.0, 2.0];
        let (_, grad) = cross_entropy_loss(&logits, &targets, 2, 3);
        // Each row of gradient should sum to 0 (softmax sums to 1, minus one-hot sums to 1)
        let row0_sum: f64 = grad[0..3].iter().sum();
        let row1_sum: f64 = grad[3..6].iter().sum();
        assert!(row0_sum.abs() < 1e-10);
        assert!(row1_sum.abs() < 1e-10);
    }

    #[test]
    fn mse_loss_basic() {
        let preds = vec![1.0, 2.0, 3.0];
        let targets = vec![1.0, 2.0, 3.0];
        let (loss, _) = mse_loss(&preds, &targets, 3);
        assert!(loss.abs() < 1e-10);
    }

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![2.0, 1.0, 0.5];
        let probs = softmax(&logits, 1, 3);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn network_backward_gradient_check() {
        let mut net = Network::new(&[3, 5, 2], Activation::Tanh, 42);
        let input = vec![0.5, -0.3, 0.8];
        let batch = 1;
        let targets = vec![1.0]; // class 1

        // Forward + loss
        let logits = net.forward(&input, batch, true);
        let (_loss, grad) = cross_entropy_loss(&logits, &targets, batch, 2);

        // Backward
        let layer_grads = net.backward(&grad, 0.0);

        // Numerical gradient check for first layer weights
        let eps = 1e-5;
        let n_weights = net.layers[0].weights.len();
        for idx in 0..n_weights.min(6) {
            let orig = net.layers[0].weights[idx];

            net.layers[0].weights[idx] = orig + eps;
            let logits_p = net.forward(&input, batch, false);
            let (loss_p, _) = cross_entropy_loss(&logits_p, &targets, batch, 2);

            net.layers[0].weights[idx] = orig - eps;
            let logits_m = net.forward(&input, batch, false);
            let (loss_m, _) = cross_entropy_loss(&logits_m, &targets, batch, 2);

            net.layers[0].weights[idx] = orig;

            let numerical = (loss_p - loss_m) / (2.0 * eps);
            let analytic = layer_grads[0].0[idx];

            let diff = (analytic - numerical).abs();
            let denom = analytic.abs().max(numerical.abs()).max(1e-8);
            assert!(
                diff / denom < 1e-3,
                "layer 0 weight {idx}: analytic={analytic:.8}, numerical={numerical:.8}, rel_err={:.6}",
                diff / denom,
            );
        }
    }

    #[test]
    fn save_restore_weights() {
        let mut net = Network::new(&[3, 5, 2], Activation::Relu, 42);
        let saved = net.save_weights();

        // Mutate weights
        net.layers[0].weights[0] = 999.0;
        assert!((net.layers[0].weights[0] - 999.0).abs() < 1e-10);

        // Restore
        net.restore_weights(&saved);
        assert!((net.layers[0].weights[0] - 999.0).abs() > 1e-5);
    }
}
