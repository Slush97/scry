//! Lasso regression via coordinate descent (L1 regularization).
//!
//! Coordinate descent iteratively optimizes one coefficient at a time,
//! applying the soft-thresholding operator to drive small coefficients
//! exactly to zero — producing sparse models.

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};

/// Lasso regression (L1-regularized linear regression).
///
/// Uses coordinate descent to find the coefficients `β` that minimize:
///
/// ```text
/// (1 / 2n) ‖y − Xβ − β₀‖² + α ‖β‖₁
/// ```
///
/// Higher `alpha` produces sparser models (more coefficients driven to zero).
///
/// # Example
/// ```
/// use scry_learn::dataset::Dataset;
/// use scry_learn::linear::LassoRegression;
///
/// let features = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
/// let target = vec![2.1, 4.0, 5.9, 8.1, 10.0];
/// let data = Dataset::new(features, target, vec!["x".into()], "y");
///
/// let mut lasso = LassoRegression::new().alpha(0.1);
/// lasso.fit(&data).unwrap();
/// let preds = lasso.predict(&[vec![3.0]]).unwrap();
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LassoRegression {
    /// L1 regularization strength.
    alpha: f64,
    /// Maximum coordinate descent iterations.
    max_iter: usize,
    /// Convergence tolerance.
    tol: f64,
    /// Learned coefficients (one per feature).
    coefficients: Vec<f64>,
    /// Learned intercept.
    intercept: f64,
    /// Whether the model has been fitted.
    fitted: bool,
}

impl LassoRegression {
    /// Create a new Lasso with default parameters (alpha=1.0, max_iter=1000).
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            max_iter: 1000,
            tol: 1e-4,
            coefficients: Vec::new(),
            intercept: 0.0,
            fitted: false,
        }
    }

    /// Set the L1 regularization strength.
    pub fn alpha(mut self, a: f64) -> Self {
        self.alpha = a;
        self
    }

    /// Set the maximum number of iterations.
    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Set convergence tolerance.
    pub fn tol(mut self, t: f64) -> Self {
        self.tol = t;
        self
    }

    /// Fit the Lasso model using coordinate descent.
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        let n = data.n_samples();
        let m = data.n_features();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        if self.alpha < 0.0 {
            return Err(ScryLearnError::InvalidParameter(
                "alpha must be >= 0".into(),
            ));
        }

        // Build row-major feature matrix for efficient access.
        let rows = data.feature_matrix();
        let y = &data.target;

        // Initialize coefficients to zero.
        let mut beta = vec![0.0; m];
        let mut intercept = y.iter().sum::<f64>() / n as f64;

        // Precompute feature norms: ‖X_j‖² / n
        let mut col_norm_sq: Vec<f64> = vec![0.0; m];
        for j in 0..m {
            for row in &rows {
                col_norm_sq[j] += row[j] * row[j];
            }
            col_norm_sq[j] /= n as f64;
        }

        let n_f64 = n as f64;

        for _iter in 0..self.max_iter {
            let mut max_change = 0.0_f64;

            // Update intercept: mean of residuals.
            let mut r_sum = 0.0;
            for (i, row) in rows.iter().enumerate() {
                let pred: f64 = row.iter().zip(beta.iter()).map(|(x, b)| x * b).sum::<f64>() + intercept;
                r_sum += y[i] - pred;
            }
            let new_intercept = intercept + r_sum / n_f64;
            max_change = max_change.max((new_intercept - intercept).abs());
            intercept = new_intercept;

            // Coordinate descent over each feature.
            for j in 0..m {
                if col_norm_sq[j] < 1e-15 {
                    continue; // skip constant features
                }

                // Compute partial residual dot product: (1/n) Σ (r_i * x_ij)
                // where r_i = y_i - (Σ_{k≠j} x_ik β_k + β₀)
                let old_beta_j = beta[j];
                let mut rho = 0.0;
                for (i, row) in rows.iter().enumerate() {
                    let pred_without_j: f64 = intercept
                        + row.iter()
                            .zip(beta.iter())
                            .enumerate()
                            .filter(|&(k, _)| k != j)
                            .map(|(_, (x, b))| x * b)
                            .sum::<f64>();
                    rho += row[j] * (y[i] - pred_without_j);
                }
                rho /= n_f64;

                // Soft-thresholding: β_j = S(ρ, α) / ‖X_j‖²/n
                let new_beta_j = soft_threshold(rho, self.alpha) / col_norm_sq[j];
                max_change = max_change.max((new_beta_j - old_beta_j).abs());
                beta[j] = new_beta_j;
            }

            if max_change < self.tol {
                break;
            }
        }

        self.coefficients = beta;
        self.intercept = intercept;
        self.fitted = true;
        Ok(())
    }

    /// Predict target values for new samples.
    ///
    /// `features` is row-major: `features[sample_idx][feature_idx]`.
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        Ok(features
            .iter()
            .map(|row| {
                row.iter()
                    .zip(self.coefficients.iter())
                    .map(|(x, b)| x * b)
                    .sum::<f64>()
                    + self.intercept
            })
            .collect())
    }

    /// Get learned coefficients.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Get learned intercept.
    pub fn intercept(&self) -> f64 {
        self.intercept
    }
}

impl Default for LassoRegression {
    fn default() -> Self {
        Self::new()
    }
}

/// Soft-thresholding operator: S(z, γ) = sign(z) max(|z| - γ, 0).
#[inline]
fn soft_threshold(z: f64, gamma: f64) -> f64 {
    if z > gamma {
        z - gamma
    } else if z < -gamma {
        z + gamma
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lasso_fit_predict() {
        // y = 2x + 1
        let features = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let target = vec![3.0, 5.0, 7.0, 9.0, 11.0];
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        let mut lasso = LassoRegression::new().alpha(0.01).max_iter(5000);
        lasso.fit(&data).unwrap();

        let preds = lasso.predict(&[vec![3.0]]).unwrap();
        assert!(
            (preds[0] - 7.0).abs() < 0.5,
            "expected ~7.0, got {}",
            preds[0]
        );
    }

    #[test]
    fn test_lasso_sparsity() {
        // y = 2*x1 + 3*x3 + 1, x2 and x4 are noise
        let n = 100;
        let mut rng = fastrand::Rng::with_seed(42);
        let mut x1 = Vec::with_capacity(n);
        let mut x2 = Vec::with_capacity(n);
        let mut x3 = Vec::with_capacity(n);
        let mut x4 = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);

        for _ in 0..n {
            let v1 = rng.f64() * 10.0;
            let v2 = rng.f64() * 10.0;
            let v3 = rng.f64() * 10.0;
            let v4 = rng.f64() * 10.0;
            x1.push(v1);
            x2.push(v2);
            x3.push(v3);
            x4.push(v4);
            y.push(2.0 * v1 + 3.0 * v3 + 1.0);
        }

        let data = Dataset::new(
            vec![x1, x2, x3, x4],
            y,
            vec!["x1".into(), "x2".into(), "x3".into(), "x4".into()],
            "y",
        );

        let mut lasso = LassoRegression::new().alpha(0.5).max_iter(5000);
        lasso.fit(&data).unwrap();

        let coefs = lasso.coefficients();
        // x2 and x4 coefficients should be driven to ~0
        assert!(
            coefs[1].abs() < 0.1,
            "x2 coef should be ~0, got {}",
            coefs[1]
        );
        assert!(
            coefs[3].abs() < 0.1,
            "x4 coef should be ~0, got {}",
            coefs[3]
        );
        // x1 and x3 should be significant
        assert!(coefs[0].abs() > 0.5, "x1 coef should be significant");
        assert!(coefs[2].abs() > 0.5, "x3 coef should be significant");
    }

    #[test]
    fn test_lasso_not_fitted() {
        let lasso = LassoRegression::new();
        assert!(lasso.predict(&[vec![1.0]]).is_err());
    }
}
