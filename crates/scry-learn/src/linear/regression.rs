//! Linear regression via OLS (Ordinary Least Squares).

use crate::accel;
use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};

/// Solver strategy for linear regression.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub enum LinRegSolver {
    /// Normal equation: (X^T X + aI)^-1 X^T y. Fast but numerically fragile.
    #[default]
    Normal,
    /// QR decomposition. More robust than Normal, faster than SVD.
    Qr,
    /// SVD (pseudoinverse). Most robust, handles rank-deficient and wide matrices.
    Svd,
    /// Auto: use Normal for well-conditioned problems, fall back to SVD otherwise.
    Auto,
}

/// Linear regression model.
///
/// Uses the **OLS** closed-form normal equations solution by default:
/// `β = (XᵀX + αI)⁻¹ Xᵀy`. Set `alpha > 0` for Ridge (L2) regularization.
///
/// Alternative solvers (QR, SVD) provide better numerical stability for
/// ill-conditioned or rank-deficient problems.
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
    /// Solver strategy.
    #[cfg_attr(feature = "serde", serde(skip))]
    solver: LinRegSolver,
    fitted: bool,
}

impl LinearRegression {
    /// Create a new linear regression model.
    pub fn new() -> Self {
        Self {
            coefficients: Vec::new(),
            intercept: 0.0,
            alpha: 0.0,
            solver: LinRegSolver::Normal,
            fitted: false,
        }
    }

    /// Set L2 regularization strength (Ridge regression).
    pub fn alpha(mut self, a: f64) -> Self {
        self.alpha = a;
        self
    }

    /// Set the solver strategy.
    pub fn solver(mut self, s: LinRegSolver) -> Self {
        self.solver = s;
        self
    }

    /// Train the model on the given dataset.
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        let n = data.n_samples();
        let m = data.n_features();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }

        match &self.solver {
            LinRegSolver::Normal => self.fit_normal(data),
            LinRegSolver::Qr => self.fit_qr(data),
            LinRegSolver::Svd => self.fit_svd(data),
            LinRegSolver::Auto => {
                if m >= n {
                    return self.fit_svd(data);
                }
                match self.fit_normal(data) {
                    Ok(()) => Ok(()),
                    Err(_) => self.fit_svd(data),
                }
            }
        }
    }

    /// Normal equation solver (existing code path).
    fn fit_normal(&mut self, data: &Dataset) -> Result<()> {
        let m = data.n_features();
        let dim = m + 1;

        let backend = accel::auto();
        let (mut xtx, mut xty) = backend.xtx_xty(&data.features, &data.target);

        for j in 1..dim {
            xtx[j * dim + j] += self.alpha;
        }

        let beta = solve_linear(dim, &mut xtx, &mut xty)?;

        self.intercept = beta[0];
        self.coefficients = beta[1..].to_vec();
        self.fitted = true;
        Ok(())
    }

    /// Build augmented column-major feature matrix [1, x1, x2, ...].
    fn build_augmented(data: &Dataset) -> (Vec<f64>, usize, usize) {
        let n = data.n_samples();
        let m = data.n_features();
        let dim = m + 1;
        let mut x = vec![0.0; n * dim];
        for i in 0..n {
            x[i] = 1.0;
        }
        for (j, col) in data.features.iter().enumerate() {
            let offset = (j + 1) * n;
            x[offset..offset + n].copy_from_slice(col);
        }
        (x, n, dim)
    }

    /// Build augmented matrix with Ridge regularization rows appended.
    fn build_regularized(data: &Dataset, alpha: f64) -> (Vec<f64>, Vec<f64>, usize, usize) {
        let n = data.n_samples();
        let m = data.n_features();
        let dim = m + 1;
        let sqrt_a = alpha.sqrt();
        let aug_rows = n + m;
        let mut x_aug = vec![0.0; aug_rows * dim];
        let mut y_aug = vec![0.0; aug_rows];

        for i in 0..n {
            x_aug[i] = 1.0;
        }
        for (j, col) in data.features.iter().enumerate() {
            let offset = (j + 1) * aug_rows;
            x_aug[offset..offset + n].copy_from_slice(col);
        }
        y_aug[..n].copy_from_slice(&data.target);

        for j in 0..m {
            x_aug[(j + 1) * aug_rows + n + j] = sqrt_a;
        }

        (x_aug, y_aug, aug_rows, dim)
    }

    /// QR decomposition solver.
    fn fit_qr(&mut self, data: &Dataset) -> Result<()> {
        if self.alpha > 0.0 {
            let (x_aug, y_aug, aug_rows, dim) = Self::build_regularized(data, self.alpha);
            let beta = super::qr::qr_solve(&x_aug, &y_aug, aug_rows, dim)?;
            self.intercept = beta[0];
            self.coefficients = beta[1..].to_vec();
        } else {
            let (x, n, dim) = Self::build_augmented(data);
            let beta = super::qr::qr_solve(&x, &data.target, n, dim)?;
            self.intercept = beta[0];
            self.coefficients = beta[1..].to_vec();
        }

        self.fitted = true;
        Ok(())
    }

    /// SVD solver.
    fn fit_svd(&mut self, data: &Dataset) -> Result<()> {
        if self.alpha > 0.0 {
            let (x_aug, y_aug, aug_rows, dim) = Self::build_regularized(data, self.alpha);
            let result = super::svd::svd_solve(&x_aug, &y_aug, aug_rows, dim)?;
            self.intercept = result.coefficients[0];
            self.coefficients = result.coefficients[1..].to_vec();
        } else {
            let (x, n, dim) = Self::build_augmented(data);
            let result = super::svd::svd_solve(&x, &data.target, n, dim)?;
            self.intercept = result.coefficients[0];
            self.coefficients = result.coefficients[1..].to_vec();
        }

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

        if max_row != col {
            for k in 0..n {
                a.swap(col * n + k, max_row * n + k);
            }
            b.swap(col, max_row);
        }

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

        assert!(lr.coefficients()[0] < 2.0);
        assert!(lr.coefficients()[0] > 1.0);
    }

    #[test]
    fn test_svd_solver_matches_normal() {
        let features = vec![(0..20).map(|i| i as f64).collect::<Vec<f64>>()];
        let target: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 + 3.0).collect();
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        let mut lr_normal = LinearRegression::new();
        lr_normal.fit(&data).unwrap();

        let mut lr_svd = LinearRegression::new().solver(LinRegSolver::Svd);
        lr_svd.fit(&data).unwrap();

        assert!(
            (lr_normal.coefficients()[0] - lr_svd.coefficients()[0]).abs() < 1e-6,
            "Normal={} vs SVD={}",
            lr_normal.coefficients()[0],
            lr_svd.coefficients()[0]
        );
        assert!(
            (lr_normal.intercept() - lr_svd.intercept()).abs() < 1e-6,
            "Normal intercept={} vs SVD={}",
            lr_normal.intercept(),
            lr_svd.intercept()
        );
    }

    #[test]
    fn test_qr_solver_matches_normal() {
        let features = vec![(0..20).map(|i| i as f64).collect::<Vec<f64>>()];
        let target: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 + 3.0).collect();
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        let mut lr_normal = LinearRegression::new();
        lr_normal.fit(&data).unwrap();

        let mut lr_qr = LinearRegression::new().solver(LinRegSolver::Qr);
        lr_qr.fit(&data).unwrap();

        assert!(
            (lr_normal.coefficients()[0] - lr_qr.coefficients()[0]).abs() < 1e-6,
            "Normal={} vs QR={}",
            lr_normal.coefficients()[0],
            lr_qr.coefficients()[0]
        );
        assert!(
            (lr_normal.intercept() - lr_qr.intercept()).abs() < 1e-6,
            "Normal intercept={} vs QR={}",
            lr_normal.intercept(),
            lr_qr.intercept()
        );
    }

    #[test]
    fn test_svd_handles_ill_conditioned() {
        let n = 5;
        let mut features = vec![vec![0.0; n]; n];
        for j in 0..n {
            for i in 0..n {
                features[j][i] = 1.0 / (i + j + 1) as f64;
            }
        }
        let true_beta = vec![1.0; n];
        let target: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| features[j][i] * true_beta[j]).sum())
            .collect();
        let names: Vec<String> = (0..n).map(|j| format!("f{j}")).collect();
        let data = Dataset::new(features, target, names, "y");

        let mut lr = LinearRegression::new().solver(LinRegSolver::Svd);
        lr.fit(&data).unwrap();

        for (i, &c) in lr.coefficients().iter().enumerate() {
            assert!(
                (c - 1.0).abs() < 0.5,
                "SVD Hilbert coeff[{}] = {}, expected ~1.0",
                i,
                c
            );
        }
    }

    #[test]
    fn test_ridge_with_svd() {
        let features = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let target = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        let mut lr_normal = LinearRegression::new().alpha(1.0);
        lr_normal.fit(&data).unwrap();

        let mut lr_svd = LinearRegression::new().alpha(1.0).solver(LinRegSolver::Svd);
        lr_svd.fit(&data).unwrap();

        assert!(
            (lr_normal.coefficients()[0] - lr_svd.coefficients()[0]).abs() < 0.1,
            "Ridge Normal={} vs SVD={}",
            lr_normal.coefficients()[0],
            lr_svd.coefficients()[0]
        );
    }

    #[test]
    fn test_auto_solver() {
        let features = vec![(0..20).map(|i| i as f64).collect::<Vec<f64>>()];
        let target: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 + 3.0).collect();
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        let mut lr = LinearRegression::new().solver(LinRegSolver::Auto);
        lr.fit(&data).unwrap();

        assert!(
            (lr.coefficients()[0] - 2.0).abs() < 1e-6,
            "Auto solver coefficient should be ~2.0, got {}",
            lr.coefficients()[0]
        );
    }
}

#[cfg(all(test, feature = "gpu"))]
mod gpu_tests {
    use super::*;

    #[test]
    fn gpu_linear_regression_matches_cpu() {
        let n = 500;
        let m = 50;
        let mut features = Vec::with_capacity(m);
        for j in 0..m {
            let col: Vec<f64> = (0..n).map(|i| ((i * (j + 1)) % 97) as f64 * 0.1).collect();
            features.push(col);
        }
        let target: Vec<f64> = (0..n)
            .map(|i| features[0][i] * 2.0 + features[1][i] * 0.5 + features[2][i] + 3.0)
            .collect();
        let names: Vec<String> = (0..m).map(|j| format!("f{j}")).collect();
        let data = Dataset::new(features, target, names, "y");

        let mut lr = LinearRegression::new().alpha(0.1);
        lr.fit(&data).unwrap();

        assert!(lr.coefficients().len() == m);
        let preds = lr.predict(&[vec![1.0; m]]).unwrap();
        assert!(preds[0].is_finite(), "prediction must be finite, got {}", preds[0]);
    }
}
