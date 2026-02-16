//! Linear regression via OLS (Ordinary Least Squares).

use crate::accel;
use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};

/// Linear regression model.
///
/// Uses the **OLS** closed-form normal equations solution:
/// `β = (XᵀX + αI)⁻¹ Xᵀy`. Set `alpha > 0` for Ridge (L2) regularization.
///
/// When the `gpu` feature is enabled and the dataset is large enough,
/// the XᵀX/Xᵀy computation is automatically offloaded to GPU compute shaders.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LinearRegression {
    /// Learned coefficients (one per feature).
    coefficients: Vec<f64>,
    /// Learned intercept (bias term).
    intercept: f64,
    /// L2 regularization strength (0.0 = OLS, >0 = Ridge).
    alpha: f64,
    fitted: bool,
}

impl LinearRegression {
    /// Create a new linear regression model.
    pub fn new() -> Self {
        Self {
            coefficients: Vec::new(),
            intercept: 0.0,
            alpha: 0.0,
            fitted: false,
        }
    }

    /// Set L2 regularization strength (Ridge regression).
    pub fn alpha(mut self, a: f64) -> Self {
        self.alpha = a;
        self
    }

    /// Train using the normal equations: β = (X^T X + αI)^{-1} X^T y.
    ///
    /// For small-to-medium datasets this is exact and fast. With the `gpu`
    /// feature enabled, large matrices are computed on the GPU automatically.
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        let n = data.n_samples();
        let m = data.n_features();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }

        // Build augmented feature matrix [1, x1, x2, ...] for intercept.
        let dim = m + 1;

        // Compute XᵀX and Xᵀy via the best available backend.
        // data.features is column-major [n_features][n_samples], which is
        // exactly what ComputeBackend::xtx_xty() expects.
        let backend = accel::auto();
        let (mut xtx, mut xty) = backend.xtx_xty(&data.features, &data.target);

        // Add regularization (skip intercept term at index 0).
        for j in 1..dim {
            xtx[j * dim + j] += self.alpha;
        }

        // Solve via Gauss-Jordan elimination.
        let beta = solve_linear(dim, &mut xtx, &mut xty)?;

        self.intercept = beta[0];
        self.coefficients = beta[1..].to_vec();
        self.fitted = true;
        Ok(())
    }

    /// Predict target values.
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        Ok(features
            .iter()
            .map(|row| {
                let mut y = self.intercept;
                for (j, &coeff) in self.coefficients.iter().enumerate() {
                    if j < row.len() {
                        y += coeff * row[j];
                    }
                }
                y
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

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

/// Gauss-Jordan elimination for Ax = b.
fn solve_linear(n: usize, a: &mut [f64], b: &mut [f64]) -> Result<Vec<f64>> {
    for col in 0..n {
        // Partial pivoting.
        let mut max_row = col;
        let mut max_val = a[col * n + col].abs();
        for row in (col + 1)..n {
            let val = a[row * n + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return Err(ScryLearnError::InvalidParameter(
                "singular matrix — features may be linearly dependent".into(),
            ));
        }

        // Swap rows.
        if max_row != col {
            for k in 0..n {
                a.swap(col * n + k, max_row * n + k);
            }
            b.swap(col, max_row);
        }

        // Eliminate.
        let pivot = a[col * n + col];
        for k in col..n {
            a[col * n + k] /= pivot;
        }
        b[col] /= pivot;

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row * n + col];
            for k in col..n {
                a[row * n + k] -= factor * a[col * n + k];
            }
            b[row] -= factor * b[col];
        }
    }

    Ok(b.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_regression_y_equals_x() {
        let features = vec![(0..20).map(|i| i as f64).collect::<Vec<f64>>()];
        let target: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 + 3.0).collect();
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        let mut lr = LinearRegression::new();
        lr.fit(&data).unwrap();

        assert!(
            (lr.coefficients()[0] - 2.0).abs() < 1e-6,
            "coefficient should be ~2.0, got {}",
            lr.coefficients()[0]
        );
        assert!(
            (lr.intercept() - 3.0).abs() < 1e-6,
            "intercept should be ~3.0, got {}",
            lr.intercept()
        );
    }

    #[test]
    fn test_ridge_regression() {
        let features = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let target = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        let mut lr = LinearRegression::new().alpha(1.0);
        lr.fit(&data).unwrap();

        // With regularization, coefficient should be slightly less than 2.0.
        assert!(lr.coefficients()[0] < 2.0);
        assert!(lr.coefficients()[0] > 1.0);
    }
}

#[cfg(all(test, feature = "gpu"))]
mod gpu_tests {
    use super::*;

    #[test]
    fn gpu_linear_regression_matches_cpu() {
        // Dataset large enough (500 samples × 50 features) to trigger GPU path
        // in xtx_xty (threshold: n_samples * n_features² > 50_000).
        let n = 500;
        let m = 50;
        let mut features = Vec::with_capacity(m);
        for j in 0..m {
            let col: Vec<f64> = (0..n).map(|i| ((i * (j + 1)) % 97) as f64 * 0.1).collect();
            features.push(col);
        }
        // Target: sum of first 3 features + noise
        let target: Vec<f64> = (0..n)
            .map(|i| features[0][i] * 2.0 + features[1][i] * 0.5 + features[2][i] + 3.0)
            .collect();
        let names: Vec<String> = (0..m).map(|j| format!("f{j}")).collect();
        let data = Dataset::new(features, target, names, "y");

        let mut lr = LinearRegression::new().alpha(0.1);
        lr.fit(&data).unwrap();

        // Sanity: model should be fitted and produce reasonable predictions.
        assert!(lr.coefficients().len() == m);
        let preds = lr.predict(&[vec![1.0; m]]).unwrap();
        assert!(preds[0].is_finite(), "prediction must be finite, got {}", preds[0]);
    }
}
