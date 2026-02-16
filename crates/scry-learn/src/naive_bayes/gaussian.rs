//! Gaussian Naive Bayes classifier.

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::weights::{ClassWeight, compute_sample_weights};

/// Gaussian Naive Bayes — assumes features follow a normal distribution per class.
///
/// Fast to train (single pass), good baseline classifier.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussianNb {
    /// Per-class means: `means[class][feature]`.
    means: Vec<Vec<f64>>,
    /// Per-class variances: `variances[class][feature]`.
    variances: Vec<Vec<f64>>,
    /// Prior probabilities per class.
    priors: Vec<f64>,
    class_weight: ClassWeight,
    /// Additive smoothing: `var_smoothing * max(all_variances)` is added to each
    /// feature variance, matching scikit-learn's behaviour.
    var_smoothing: f64,
    n_classes: usize,
    fitted: bool,
}

impl GaussianNb {
    /// Create a new Gaussian Naive Bayes classifier.
    pub fn new() -> Self {
        Self {
            means: Vec::new(),
            variances: Vec::new(),
            priors: Vec::new(),
            class_weight: ClassWeight::Uniform,
            var_smoothing: 1e-9,
            n_classes: 0,
            fitted: false,
        }
    }

    /// Set class weighting strategy for imbalanced datasets.
    pub fn class_weight(mut self, cw: ClassWeight) -> Self {
        self.class_weight = cw;
        self
    }

    /// Set variance smoothing parameter (default `1e-9`, matching sklearn).
    ///
    /// The smoothing epsilon added to each feature variance is
    /// `var_smoothing × max(all_variances)`, which adapts to the scale
    /// of the data — important for high-dimensional datasets.
    pub fn var_smoothing(mut self, vs: f64) -> Self {
        self.var_smoothing = vs;
        self
    }

    /// Train the model (single-pass computation of per-class mean/variance).
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        let n = data.n_samples();
        let m = data.n_features();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }

        self.n_classes = data.n_classes();
        self.means = vec![vec![0.0; m]; self.n_classes];
        self.variances = vec![vec![0.0; m]; self.n_classes];

        let mat = data.matrix();
        let sample_weights = compute_sample_weights(&data.target, &self.class_weight);
        let mut w_sums = vec![0.0_f64; self.n_classes];

        // Compute weighted means.
        for (i, (&sw, &target_val)) in sample_weights.iter().zip(data.target.iter()).enumerate() {
            let c = target_val as usize;
            if c >= self.n_classes { continue; }
            w_sums[c] += sw;
            for j in 0..m {
                self.means[c][j] += sw * mat.get(i, j);
            }
        }
        for (c_means, &ws) in self.means.iter_mut().zip(w_sums.iter()) {
            if ws > 0.0 {
                for mj in c_means.iter_mut() {
                    *mj /= ws;
                }
            }
        }

        // Compute weighted variances.
        for (i, (&sw, &target_val)) in sample_weights.iter().zip(data.target.iter()).enumerate() {
            let c = target_val as usize;
            if c >= self.n_classes { continue; }
            for j in 0..m {
                let diff = mat.get(i, j) - self.means[c][j];
                self.variances[c][j] += sw * diff * diff;
            }
        }
        for (c_var, &ws) in self.variances.iter_mut().zip(w_sums.iter()) {
            if ws > 0.0 {
                for vj in c_var.iter_mut() {
                    *vj /= ws;
                }
            }
        }

        // Compute max variance across all classes and features (sklearn-style).
        let max_var = self.variances.iter()
            .flat_map(|cv| cv.iter())
            .copied()
            .fold(0.0_f64, f64::max);
        let epsilon = self.var_smoothing * max_var.max(1e-300);

        // Add scaled smoothing to all variances.
        for c_var in &mut self.variances {
            for vj in c_var.iter_mut() {
                *vj += epsilon;
            }
        }

        // Weighted priors.
        let w_total: f64 = w_sums.iter().sum();
        self.priors = w_sums.iter().map(|&ws| ws / w_total).collect();
        self.fitted = true;
        Ok(())
    }

    /// Per-class means: `class_means()[c][j]` is the mean of feature `j` for class `c`.
    pub fn class_means(&self) -> &[Vec<f64>] {
        &self.means
    }

    /// Per-class variances (smoothed): `class_variances()[c][j]` for class `c`, feature `j`.
    pub fn class_variances(&self) -> &[Vec<f64>] {
        &self.variances
    }

    /// Prior probabilities per class.
    pub fn class_priors(&self) -> &[f64] {
        &self.priors
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

    /// Predict log-probabilities and return normalized probabilities.
    pub fn predict_proba(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }

        Ok(features
            .iter()
            .map(|row| {
                let mut log_probs: Vec<f64> = (0..self.n_classes)
                    .map(|c| {
                        let mut log_p = self.priors[c].ln();
                        for (j, &x) in row.iter().enumerate() {
                            if j < self.means[c].len() {
                                let mean = self.means[c][j];
                                let var = self.variances[c][j];
                                // Gaussian log-likelihood.
                                log_p += -0.5 * ((x - mean).powi(2) / var + var.ln() + std::f64::consts::TAU.ln());
                            }
                        }
                        log_p
                    })
                    .collect();

                // Log-sum-exp trick for numerical stability.
                let max_log = log_probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = log_probs.iter().map(|&lp| (lp - max_log).exp()).sum();
                for lp in &mut log_probs {
                    *lp = ((*lp - max_log).exp()) / sum;
                }
                log_probs
            })
            .collect())
    }
}

impl Default for GaussianNb {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_nb_simple() {
        // Class 0: low values, Class 1: high values.
        let features = vec![
            vec![1.0, 1.5, 2.0, 8.0, 8.5, 9.0],
            vec![1.0, 1.5, 2.0, 8.0, 8.5, 9.0],
        ];
        let target = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");

        let mut nb = GaussianNb::new();
        nb.fit(&data).unwrap();

        let preds = nb.predict(&[vec![1.0, 1.0], vec![9.0, 9.0]]).unwrap();
        assert!((preds[0] - 0.0).abs() < 1e-6);
        assert!((preds[1] - 1.0).abs() < 1e-6);
    }
}
