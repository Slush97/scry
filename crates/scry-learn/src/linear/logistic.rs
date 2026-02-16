//! Logistic regression via L-BFGS (default) or gradient descent.
//!
//! Supports configurable [`Penalty`] regularization: `None`, `L1`, `L2` (default),
//! and `ElasticNet(l1_ratio)`. L1 and ElasticNet use proximal gradient descent
//! (soft-thresholding); L-BFGS only supports `L2` and `None`.

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::weights::{ClassWeight, compute_sample_weights};

use super::lbfgs;

/// Regularization penalty for logistic regression.
///
/// Controls the type of regularization applied during training:
/// - `None` — no regularization
/// - `L1` — Lasso penalty (promotes sparsity via proximal gradient descent)
/// - `L2` — Ridge penalty (default, shrinks coefficients)
/// - `ElasticNet(l1_ratio)` — Mix of L1 and L2; `l1_ratio` ∈ \[0, 1\]
///   where 1.0 = pure L1, 0.0 = pure L2
///
/// # Solver compatibility
///
/// | Penalty | GradientDescent | L-BFGS |
/// |---------|:-:|:-:|
/// | `None` | ✓ | ✓ |
/// | `L1` | ✓ | ✗ (error) |
/// | `L2` | ✓ | ✓ |
/// | `ElasticNet` | ✓ | ✗ (error) |
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Penalty {
    /// No regularization.
    None,
    /// L1 (Lasso) penalty — promotes sparse coefficients.
    L1,
    /// L2 (Ridge) penalty — shrinks all coefficients (default).
    #[default]
    L2,
    /// Elastic Net — mix of L1 and L2. The `f64` is the L1 ratio ∈ \[0, 1\].
    ElasticNet(f64),
}

/// Solver algorithm for logistic regression.
///
/// L-BFGS is the default and recommended solver — it converges in ~10-20
/// iterations vs 200+ for gradient descent, matching scikit-learn's default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Solver {
    /// L-BFGS quasi-Newton optimizer (default). Fast, recommended.
    #[default]
    Lbfgs,
    /// Vanilla batch gradient descent. Slower, kept for backward compatibility.
    GradientDescent,
}

/// Logistic regression for binary/multiclass classification.
///
/// Uses L-BFGS (default) or gradient descent with configurable learning rate,
/// iterations, and L2 regularization.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LogisticRegression {
    learning_rate: f64,
    max_iter: usize,
    alpha: f64,        // regularization strength
    tolerance: f64,
    class_weight: ClassWeight,
    #[cfg_attr(feature = "serde", serde(default))]
    solver: Solver,
    #[cfg_attr(feature = "serde", serde(default))]
    penalty: Penalty,
    weights: Vec<Vec<f64>>, // [n_classes][n_features + 1] (includes bias)
    n_classes: usize,
    fitted: bool,
}

impl LogisticRegression {
    /// Create a new logistic regression model.
    pub fn new() -> Self {
        Self {
            learning_rate: 0.01,
            max_iter: 1000,
            alpha: 1.0,
            tolerance: 1e-6,
            class_weight: ClassWeight::Uniform,
            solver: Solver::default(),
            penalty: Penalty::default(),
            weights: Vec::new(),
            n_classes: 0,
            fitted: false,
        }
    }

    /// Set the learning rate (used by `GradientDescent` solver only).
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set maximum iterations.
    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Set regularization strength (equivalent to `1/C` in scikit-learn).
    ///
    /// The meaning depends on the [`Penalty`]:
    /// - `L2` / `L1` — multiplier on the penalty term
    /// - `ElasticNet` — total regularization strength (split by l1_ratio)
    /// - `None` — ignored
    ///
    /// To match scikit-learn's `LogisticRegression(C=x)`, use `alpha(1.0 / x)`.
    /// The default `alpha = 1.0` corresponds to `C = 1.0`.
    pub fn alpha(mut self, a: f64) -> Self {
        self.alpha = a;
        self
    }

    /// Set the regularization penalty.
    ///
    /// Default is [`Penalty::L2`]. Use [`Penalty::L1`] for sparse feature selection.
    ///
    /// # Errors
    ///
    /// `L1` and `ElasticNet` are **not** supported with the `Lbfgs` solver — calling
    /// `fit()` will return `Err(InvalidParameter)`. Switch to `GradientDescent`.
    pub fn penalty(mut self, p: Penalty) -> Self {
        self.penalty = p;
        self
    }

    /// Set convergence tolerance.
    pub fn tolerance(mut self, t: f64) -> Self {
        self.tolerance = t;
        self
    }

    /// Set class weighting strategy for imbalanced datasets.
    pub fn class_weight(mut self, cw: ClassWeight) -> Self {
        self.class_weight = cw;
        self
    }

    /// Set the solver algorithm.
    ///
    /// Defaults to `Solver::Lbfgs` which is ~10-20× faster than gradient descent.
    pub fn solver(mut self, s: Solver) -> Self {
        self.solver = s;
        self
    }

    /// Train the model using the configured solver.
    ///
    /// Uses consistent softmax for both training and inference (not one-vs-rest sigmoid).
    ///
    /// # Errors
    ///
    /// Returns `InvalidParameter` if `Penalty::L1` or `Penalty::ElasticNet` is used
    /// with the `Lbfgs` solver (L-BFGS requires a differentiable objective).
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        // Validate solver/penalty compatibility.
        if matches!(self.solver, Solver::Lbfgs)
            && matches!(self.penalty, Penalty::L1 | Penalty::ElasticNet(_))
        {
            return Err(ScryLearnError::InvalidParameter(
                "L-BFGS solver does not support L1 or ElasticNet penalties \
                 (non-differentiable). Use Solver::GradientDescent instead."
                    .into(),
            ));
        }
        match self.solver {
            Solver::Lbfgs => self.fit_lbfgs(data),
            Solver::GradientDescent => self.fit_gd(data),
        }
    }

    /// L-BFGS solver: flatten weights, optimize, unflatten.
    ///
    /// Uses vectorized (batch) gradient computation for cache-friendly
    /// column-major access patterns.
    #[allow(clippy::needless_range_loop)]
    fn fit_lbfgs(&mut self, data: &Dataset) -> Result<()> {
        let n = data.n_samples();
        let m = data.n_features();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }

        self.n_classes = data.n_classes();
        let k = self.n_classes;
        let dim = m + 1; // features + bias

        // Compute per-sample weights for class imbalance.
        let sample_weights = compute_sample_weights(&data.target, &self.class_weight);

        // Pre-convert targets to usize.
        let target_class: Vec<usize> = data.target.iter().map(|&t| t as usize).collect();

        let alpha = self.alpha;
        let inv_n = 1.0 / n as f64;

        // Flatten initial weights: [class0_bias, class0_w1, ..., class1_bias, ...]
        let total_params = k * dim;
        let mut params = vec![0.0; total_params];

        let config = lbfgs::LbfgsConfig {
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            history_size: 10,
        };

        // Pre-allocate batch buffers — reused every closure call.
        let mut logits = vec![0.0; n * k]; // row-major: logits[i * k + c]
        let mut max_logit = vec![0.0; n];
        let mut sum_exp = vec![0.0; n];

        lbfgs::minimize(
            &mut params,
            |x, grad| {
                // ── 1. Batch compute logits: logits[i,c] = bias_c + Σ_j w_{c,j} * X_{j,i}
                // Initialize with bias terms.
                for i in 0..n {
                    for c in 0..k {
                        logits[i * k + c] = x[c * dim]; // bias
                    }
                }
                // Accumulate feature contributions column-by-column (cache-friendly).
                for j in 0..m {
                    let feat_col = &data.features[j];
                    for c in 0..k {
                        let w = x[c * dim + j + 1];
                        for i in 0..n {
                            logits[i * k + c] += w * feat_col[i];
                        }
                    }
                }

                // ── 2. Batch softmax + loss computation.
                let mut loss = 0.0;

                // Find max logit per sample (for numerical stability).
                for i in 0..n {
                    let row = &logits[i * k..(i + 1) * k];
                    max_logit[i] = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                }

                // Exponentiate and sum.
                for i in 0..n {
                    let mut se = 0.0;
                    for c in 0..k {
                        let val = (logits[i * k + c] - max_logit[i]).exp();
                        logits[i * k + c] = val; // now stores exp(logit - max)
                        se += val;
                    }
                    sum_exp[i] = se;
                }

                // Cross-entropy loss: sw * (log_sum_exp - logit_tc).
                // logit_tc was overwritten, so reconstruct from max + log(exp_val).
                for i in 0..n {
                    let tc = target_class[i];
                    let log_sum = max_logit[i] + sum_exp[i].ln();
                    let logit_tc_val = max_logit[i] + logits[i * k + tc].ln();
                    loss += sample_weights[i] * (log_sum - logit_tc_val);
                }

                // Normalize to probabilities.
                for i in 0..n {
                    let se = sum_exp[i];
                    for c in 0..k {
                        logits[i * k + c] /= se;
                    }
                }

                // ── 3. Batch gradient: grad_{c,j} = Σ_i sw_i * (prob_{i,c} - 1_{tc==c}) * x_{i,j}
                // Zero gradient.
                for g in grad.iter_mut() {
                    *g = 0.0;
                }

                // Compute error = prob - one_hot, weighted by sample weight.
                // Then accumulate bias gradients.
                // We modify logits in-place to store sw * error.
                for i in 0..n {
                    let tc = target_class[i];
                    let sw = sample_weights[i];
                    for c in 0..k {
                        let y_i = if tc == c { 1.0 } else { 0.0 };
                        let error = sw * (logits[i * k + c] - y_i);
                        logits[i * k + c] = error; // reuse buffer for weighted errors
                        grad[c * dim] += error; // bias gradient
                    }
                }

                // Accumulate feature gradients column-by-column (cache-friendly).
                for j in 0..m {
                    let feat_col = &data.features[j];
                    for c in 0..k {
                        let grad_idx = c * dim + j + 1;
                        let mut acc = 0.0;
                        for i in 0..n {
                            acc += logits[i * k + c] * feat_col[i];
                        }
                        grad[grad_idx] += acc;
                    }
                }

                // ── 4. Average over samples + L2 regularization.
                //      sklearn formula: min_w  C * mean(log_loss) + 0.5 * ||w||²
                //      Equivalently:    min_w  mean(log_loss) + 0.5 * (1/C) * ||w||²
                //      Our `alpha` = 1/C, so we scale both loss and penalty by inv_n
                //      to get: mean(log_loss) + 0.5 * alpha * inv_n * ||w||²
                //      This ensures regularization strength scales with dataset size
                //      (matching sklearn's behavior).
                loss *= inv_n;
                for g in grad.iter_mut() {
                    *g *= inv_n;
                }

                if alpha > 0.0 {
                    let reg_scale = alpha * inv_n;
                    for c in 0..k {
                        let base = c * dim;
                        for j in 1..dim {
                            let w = x[base + j];
                            loss += 0.5 * reg_scale * w * w;
                            grad[base + j] += reg_scale * w;
                        }
                    }
                }

                loss
            },
            &config,
        );

        // Unflatten back to [n_classes][dim].
        self.weights = (0..k)
            .map(|c| params[c * dim..(c + 1) * dim].to_vec())
            .collect();

        self.fitted = true;
        Ok(())
    }

    /// Gradient descent solver (legacy).
    #[allow(clippy::needless_range_loop)]
    fn fit_gd(&mut self, data: &Dataset) -> Result<()> {
        let n = data.n_samples();
        let m = data.n_features();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }

        self.n_classes = data.n_classes();
        let dim = m + 1; // features + bias

        // Initialize weights to zero.
        self.weights = vec![vec![0.0; dim]; self.n_classes];

        // Compute per-sample weights for class imbalance.
        let sample_weights = compute_sample_weights(&data.target, &self.class_weight);

        // softmax gradient descent.
        let mut probs = vec![0.0; self.n_classes];

        for _epoch in 0..self.max_iter {
            let mut max_grad = 0.0_f64;
            let mut gradient = vec![vec![0.0; dim]; self.n_classes];

            for (i, (&sw, &target_val)) in sample_weights.iter().zip(data.target.iter()).enumerate() {
                let target_class = target_val as usize;

                // Compute logits for all classes.
                for (c, prob) in probs.iter_mut().enumerate().take(self.n_classes) {
                    let mut z = self.weights[c][0]; // bias
                    for j in 0..m {
                        z += self.weights[c][j + 1] * data.features[j][i];
                    }
                    *prob = z;
                }

                // Softmax.
                let max_s = probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0;
                for p in &mut probs[..self.n_classes] {
                    *p = (*p - max_s).exp();
                    sum += *p;
                }
                for p in &mut probs[..self.n_classes] {
                    *p /= sum;
                }

                // Gradient: weight_i * (softmax_prob - one_hot_target).
                for (c, (&pc, gc)) in probs.iter().zip(gradient.iter_mut()).enumerate().take(self.n_classes) {
                    let y_i = if target_class == c { 1.0 } else { 0.0 };
                    let error = sw * (pc - y_i);

                    gc[0] += error; // bias
                    for j in 0..m {
                        gc[j + 1] += error * data.features[j][i];
                    }
                }
            }

            // Compute L2 ratio for the penalty.
            let (l1_ratio, l2_ratio) = match &self.penalty {
                Penalty::None => (0.0, 0.0),
                Penalty::L1 => (1.0, 0.0),
                Penalty::L2 => (0.0, 1.0),
                Penalty::ElasticNet(r) => (*r, 1.0 - *r),
            };

            let inv_n = 1.0 / n as f64;

            // Update weights: gradient step + L2 regularization in gradient.
            for (c_grad, c_w) in gradient.iter_mut().zip(self.weights.iter_mut()).take(self.n_classes) {
                for (j, (g, w)) in c_grad.iter_mut().zip(c_w.iter_mut()).enumerate().take(dim) {
                    *g *= inv_n;
                    if j > 0 {
                        // L2 component goes into the gradient (scaled by inv_n like sklearn).
                        *g += self.alpha * inv_n * l2_ratio * *w;
                    }
                    max_grad = max_grad.max(g.abs());
                    *w -= self.learning_rate * *g;
                }
            }

            // Proximal step for L1 component (soft-thresholding).
            // Applied after the gradient update, only to feature weights (skip bias j=0).
            if l1_ratio > 0.0 {
                let threshold = self.learning_rate * self.alpha * inv_n * l1_ratio;
                for c_w in self.weights.iter_mut().take(self.n_classes) {
                    for w in c_w.iter_mut().skip(1) {
                        let sign = w.signum();
                        *w = sign * (*w * sign - threshold).max(0.0);
                    }
                }
            }

            if max_grad < self.tolerance {
                break;
            }
        }

        self.fitted = true;
        Ok(())
    }

    /// Predict class labels.
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        let probas = self.predict_proba(features)?;
        Ok(probas
            .iter()
            .map(|probs| {
                probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0.0, |(idx, _)| idx as f64)
            })
            .collect())
    }

    /// Predict class probabilities.
    pub fn predict_proba(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }

        Ok(features
            .iter()
            .map(|row| {
                let mut scores: Vec<f64> = self
                    .weights
                    .iter()
                    .map(|w| {
                        let mut z = w[0]; // bias
                        for (j, &x) in row.iter().enumerate() {
                            if j + 1 < w.len() {
                                z += w[j + 1] * x;
                            }
                        }
                        z
                    })
                    .collect();

                // Softmax.
                let max_s = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0;
                for s in &mut scores {
                    *s = (*s - max_s).exp();
                    sum += *s;
                }
                for s in &mut scores {
                    *s /= sum;
                }
                scores
            })
            .collect())
    }

    /// Get learned weights (coefficients + bias) for each class.
    pub fn weights(&self) -> &[Vec<f64>] {
        &self.weights
    }
}

impl Default for LogisticRegression {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logistic_linearly_separable() {
        // Class 0: x < 5, Class 1: x >= 5
        let features = vec![(0..20).map(|i| i as f64).collect()];
        let target: Vec<f64> = (0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }).collect();
        let data = Dataset::new(features, target, vec!["x".into()], "class");

        let mut lr = LogisticRegression::new()
            .alpha(0.0)
            .max_iter(200);
        lr.fit(&data).unwrap();

        let matrix = data.feature_matrix();
        let preds = lr.predict(&matrix).unwrap();
        let acc = preds
            .iter()
            .zip(data.target.iter())
            .filter(|(p, t)| (*p - *t).abs() < 1e-6)
            .count() as f64
            / data.n_samples() as f64;

        assert!(
            acc >= 0.85,
            "expected ≥85% accuracy, got {:.1}%",
            acc * 100.0
        );
    }

    #[test]
    fn test_predict_proba_sums_to_one() {
        let features = vec![vec![1.0, 2.0, 3.0]];
        let target = vec![0.0, 1.0, 0.0];
        let data = Dataset::new(features, target, vec!["x".into()], "class");

        let mut lr = LogisticRegression::new().max_iter(100);
        lr.fit(&data).unwrap();

        let probas = lr.predict_proba(&[vec![2.0]]).unwrap();
        let sum: f64 = probas[0].iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gd_solver_still_works() {
        let features = vec![(0..20).map(|i| i as f64).collect()];
        let target: Vec<f64> = (0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }).collect();
        let data = Dataset::new(features, target, vec!["x".into()], "class");

        let mut lr = LogisticRegression::new()
            .solver(Solver::GradientDescent)
            .learning_rate(0.1)
            .max_iter(1000);
        lr.fit(&data).unwrap();

        let matrix = data.feature_matrix();
        let preds = lr.predict(&matrix).unwrap();
        let acc = preds
            .iter()
            .zip(data.target.iter())
            .filter(|(p, t)| (*p - *t).abs() < 1e-6)
            .count() as f64
            / data.n_samples() as f64;

        assert!(
            acc >= 0.85,
            "GD solver: expected ≥85% accuracy, got {:.1}%",
            acc * 100.0
        );
    }

    #[test]
    fn test_lbfgs_is_default() {
        let lr = LogisticRegression::new();
        assert_eq!(lr.solver, Solver::Lbfgs);
    }

    #[test]
    fn test_l1_sparsity() {
        // 4 features: x0 = signal, x1 = signal, x2 = noise, x3 = noise.
        // Signal features have clear class separation; noise features are random.
        let n = 200;
        let mut f0 = Vec::with_capacity(n);
        let mut f1 = Vec::with_capacity(n);
        let mut f2 = Vec::with_capacity(n);
        let mut f3 = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);

        for i in 0..n {
            // Strong signal: class 0 centered at -3, class 1 centered at +3
            let class = if i < n / 2 { 0 } else { 1 };
            let offset = if class == 0 { -3.0 } else { 3.0 };
            f0.push(offset + (i % 7) as f64 * 0.1);
            f1.push(offset * 0.5 + (i % 5) as f64 * 0.05);
            // Noise: same distribution regardless of class
            f2.push((i % 3) as f64 * 0.01);
            f3.push((i % 5) as f64 * 0.01);
            target.push(class as f64);
        }

        let data = Dataset::new(
            vec![f0, f1, f2, f3],
            target,
            vec!["sig0".into(), "sig1".into(), "noise0".into(), "noise1".into()],
            "class",
        );

        let mut lr = LogisticRegression::new()
            .solver(Solver::GradientDescent)
            .penalty(Penalty::L1)
            .alpha(0.1)
            .learning_rate(0.1)
            .max_iter(3000);
        lr.fit(&data).unwrap();

        // Noise coefficients should be driven toward zero.
        let w = &lr.weights()[0];
        let noise_mag = w[3].abs() + w[4].abs(); // indices 3,4 = features 2,3 (skip bias)
        let signal_mag = w[1].abs() + w[2].abs();
        assert!(
            signal_mag > 0.01,
            "L1: signal coefficients should be nonzero, got {signal_mag:.6}"
        );
        assert!(
            noise_mag < signal_mag * 0.3,
            "L1: noise coefficients ({noise_mag:.4}) should be much smaller than signal ({signal_mag:.4})"
        );
    }

    #[test]
    fn test_l2_no_sparsity() {
        let n = 200;
        let mut f0 = Vec::with_capacity(n);
        let mut f1 = Vec::with_capacity(n);
        let mut f2 = Vec::with_capacity(n);
        let mut f3 = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);

        for i in 0..n {
            let x = i as f64 / n as f64;
            f0.push(x);
            f1.push(x * 2.0);
            f2.push(0.5 + (i % 3) as f64 * 0.01);
            f3.push(0.5 - (i % 5) as f64 * 0.01);
            target.push(if x < 0.5 { 0.0 } else { 1.0 });
        }

        let data = Dataset::new(
            vec![f0, f1, f2, f3],
            target,
            vec!["sig0".into(), "sig1".into(), "noise0".into(), "noise1".into()],
            "class",
        );

        let mut lr = LogisticRegression::new()
            .solver(Solver::GradientDescent)
            .penalty(Penalty::L2)
            .alpha(0.01)
            .learning_rate(0.5)
            .max_iter(2000);
        lr.fit(&data).unwrap();

        // L2 should keep ALL coefficients nonzero (no sparsity).
        let w = &lr.weights()[0];
        for (j, &wj) in w.iter().enumerate().skip(1) {
            assert!(
                wj.abs() > 1e-6,
                "L2: coefficient w[{j}] = {wj:.6} should be nonzero"
            );
        }
    }

    #[test]
    fn test_elasticnet_middle_ground() {
        let n = 200;
        let mut f0 = Vec::with_capacity(n);
        let mut f1 = Vec::with_capacity(n);
        let mut f2 = Vec::with_capacity(n);
        let mut f3 = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);

        for i in 0..n {
            let class = if i < n / 2 { 0 } else { 1 };
            let offset = if class == 0 { -3.0 } else { 3.0 };
            f0.push(offset + (i % 7) as f64 * 0.1);
            f1.push(offset * 0.5 + (i % 5) as f64 * 0.05);
            f2.push((i % 3) as f64 * 0.01);
            f3.push((i % 5) as f64 * 0.01);
            target.push(class as f64);
        }

        let data = Dataset::new(
            vec![f0, f1, f2, f3],
            target,
            vec!["sig0".into(), "sig1".into(), "noise0".into(), "noise1".into()],
            "class",
        );

        let mut lr = LogisticRegression::new()
            .solver(Solver::GradientDescent)
            .penalty(Penalty::ElasticNet(0.5))
            .alpha(0.1)
            .learning_rate(0.1)
            .max_iter(3000);
        lr.fit(&data).unwrap();

        // ElasticNet: signal coefficients should remain present.
        let w = &lr.weights()[0];
        let signal_mag = w[1].abs() + w[2].abs();
        assert!(
            signal_mag > 0.01,
            "ElasticNet: signal coefficients should remain nonzero, got {signal_mag:.6}"
        );
    }

    #[test]
    fn test_lbfgs_rejects_l1() {
        let features = vec![vec![1.0, 2.0, 3.0]];
        let target = vec![0.0, 1.0, 0.0];
        let data = Dataset::new(features, target, vec!["x".into()], "class");

        let mut lr = LogisticRegression::new()
            .solver(Solver::Lbfgs)
            .penalty(Penalty::L1)
            .alpha(0.1);
        let result = lr.fit(&data);
        assert!(
            result.is_err(),
            "L-BFGS should reject L1 penalty"
        );

        // Also reject ElasticNet.
        let mut lr2 = LogisticRegression::new()
            .solver(Solver::Lbfgs)
            .penalty(Penalty::ElasticNet(0.5))
            .alpha(0.1);
        let result2 = lr2.fit(&data);
        assert!(
            result2.is_err(),
            "L-BFGS should reject ElasticNet penalty"
        );
    }
}
