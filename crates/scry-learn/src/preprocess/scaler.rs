//! Feature scaling transformers.

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::preprocess::Transformer;

/// Standardize features by removing the mean and scaling to unit variance.
///
/// Each feature is transformed as: `x' = (x - mean) / std`.
/// Features with zero variance are left unchanged.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StandardScaler {
    means: Vec<f64>,
    stds: Vec<f64>,
    fitted: bool,
}

impl StandardScaler {
    /// Create a new unfitted scaler.
    pub fn new() -> Self {
        Self {
            means: Vec::new(),
            stds: Vec::new(),
            fitted: false,
        }
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for StandardScaler {
    fn fit(&mut self, data: &Dataset) -> Result<()> {
        let n = data.n_samples() as f64;
        if n == 0.0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        let mat = data.matrix();
        self.means = Vec::with_capacity(data.n_features());
        self.stds = Vec::with_capacity(data.n_features());

        for j in 0..data.n_features() {
            let col = mat.col(j);
            let mean = col.iter().sum::<f64>() / n;
            let var = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
            self.means.push(mean);
            self.stds.push(var.sqrt());
        }
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, data: &mut Dataset) -> Result<()> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        for (j, col) in data.features.iter_mut().enumerate() {
            let mean = self.means[j];
            let std = self.stds[j];
            if std > 1e-12 {
                for x in col.iter_mut() {
                    *x = (*x - mean) / std;
                }
            }
        }
        data.sync_matrix();
        Ok(())
    }

    fn inverse_transform(&self, data: &mut Dataset) -> Result<()> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        for (j, col) in data.features.iter_mut().enumerate() {
            let mean = self.means[j];
            let std = self.stds[j];
            for x in col.iter_mut() {
                *x = *x * std + mean;
            }
        }
        data.sync_matrix();
        Ok(())
    }
}

/// Scale features to a [0, 1] range.
///
/// Each feature is transformed as: `x' = (x - min) / (max - min)`.
/// Features with zero range are set to 0.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MinMaxScaler {
    mins: Vec<f64>,
    maxs: Vec<f64>,
    fitted: bool,
}

impl MinMaxScaler {
    /// Create a new unfitted scaler.
    pub fn new() -> Self {
        Self {
            mins: Vec::new(),
            maxs: Vec::new(),
            fitted: false,
        }
    }
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for MinMaxScaler {
    fn fit(&mut self, data: &Dataset) -> Result<()> {
        if data.n_samples() == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        let mat = data.matrix();
        self.mins = Vec::with_capacity(data.n_features());
        self.maxs = Vec::with_capacity(data.n_features());

        for j in 0..data.n_features() {
            let col = mat.col(j);
            let min = col.iter().copied().fold(f64::INFINITY, f64::min);
            let max = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            self.mins.push(min);
            self.maxs.push(max);
        }
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, data: &mut Dataset) -> Result<()> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        for (j, col) in data.features.iter_mut().enumerate() {
            let min = self.mins[j];
            let range = self.maxs[j] - min;
            if range > 1e-12 {
                for x in col.iter_mut() {
                    *x = (*x - min) / range;
                }
            } else {
                for x in col.iter_mut() {
                    *x = 0.0;
                }
            }
        }
        data.sync_matrix();
        Ok(())
    }

    fn inverse_transform(&self, data: &mut Dataset) -> Result<()> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        for (j, col) in data.features.iter_mut().enumerate() {
            let min = self.mins[j];
            let range = self.maxs[j] - min;
            for x in col.iter_mut() {
                *x = *x * range + min;
            }
        }
        data.sync_matrix();
        Ok(())
    }
}

// ── helpers ──────────────────────────────────────────────────────

/// Compute the quantile of a sorted slice using linear interpolation.
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Scale features using the median and inter-quartile range (IQR).
///
/// Each feature is transformed as: `x' = (x - median) / IQR`.
/// Features with zero IQR are left unchanged.
///
/// `RobustScaler` is less sensitive to outliers than [`StandardScaler`]
/// because it uses the median and quartiles rather than mean and std.
///
/// # Example
///
/// ```ignore
/// let mut scaler = RobustScaler::new();
/// scaler.fit_transform(&mut ds)?;
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RobustScaler {
    medians: Vec<f64>,
    iqrs: Vec<f64>,
    fitted: bool,
}

impl RobustScaler {
    /// Create a new unfitted robust scaler.
    pub fn new() -> Self {
        Self {
            medians: Vec::new(),
            iqrs: Vec::new(),
            fitted: false,
        }
    }
}

impl Default for RobustScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for RobustScaler {
    fn fit(&mut self, data: &Dataset) -> Result<()> {
        if data.n_samples() == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        let mat = data.matrix();
        self.medians = Vec::with_capacity(data.n_features());
        self.iqrs = Vec::with_capacity(data.n_features());

        for j in 0..data.n_features() {
            let col = mat.col(j);
            let mut sorted = col.to_vec();
            sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let median = quantile_sorted(&sorted, 0.5);
            let q1 = quantile_sorted(&sorted, 0.25);
            let q3 = quantile_sorted(&sorted, 0.75);
            self.medians.push(median);
            self.iqrs.push(q3 - q1);
        }
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, data: &mut Dataset) -> Result<()> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        for (j, col) in data.features.iter_mut().enumerate() {
            let median = self.medians[j];
            let iqr = self.iqrs[j];
            if iqr > 1e-12 {
                for x in col.iter_mut() {
                    *x = (*x - median) / iqr;
                }
            } else {
                for x in col.iter_mut() {
                    *x -= median;
                }
            }
        }
        data.sync_matrix();
        Ok(())
    }

    fn inverse_transform(&self, data: &mut Dataset) -> Result<()> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        for (j, col) in data.features.iter_mut().enumerate() {
            let median = self.medians[j];
            let iqr = self.iqrs[j];
            for x in col.iter_mut() {
                *x = *x * iqr + median;
            }
        }
        data.sync_matrix();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_scaler_zero_mean() {
        let mut ds = Dataset::new(
            vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]],
            vec![0.0; 5],
            vec!["x".into()],
            "y",
        );
        let mut scaler = StandardScaler::new();
        scaler.fit_transform(&mut ds).unwrap();

        let mean: f64 = ds.features[0].iter().sum::<f64>() / 5.0;
        assert!((mean).abs() < 1e-10, "mean should be ~0, got {mean}");

        let var: f64 =
            ds.features[0].iter().map(|x| x.powi(2)).sum::<f64>() / 5.0;
        assert!(
            (var - 1.0).abs() < 1e-10,
            "variance should be ~1, got {var}"
        );
    }

    #[test]
    fn test_minmax_scaler_range() {
        let mut ds = Dataset::new(
            vec![vec![10.0, 20.0, 30.0]],
            vec![0.0; 3],
            vec!["x".into()],
            "y",
        );
        let mut scaler = MinMaxScaler::new();
        scaler.fit_transform(&mut ds).unwrap();

        assert!((ds.features[0][0]).abs() < 1e-10);
        assert!((ds.features[0][2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_standard_scaler_not_fitted() {
        let scaler = StandardScaler::new();
        let mut ds = Dataset::new(
            vec![vec![1.0]],
            vec![0.0],
            vec!["x".into()],
            "y",
        );
        assert!(scaler.transform(&mut ds).is_err());
    }

    #[test]
    fn test_standard_scaler_roundtrip() {
        let original = vec![2.0, 4.0, 6.0, 8.0];
        let mut ds = Dataset::new(
            vec![original.clone()],
            vec![0.0; 4],
            vec!["x".into()],
            "y",
        );
        let mut scaler = StandardScaler::new();
        scaler.fit_transform(&mut ds).unwrap();
        scaler.inverse_transform(&mut ds).unwrap();

        for (a, b) in ds.features[0].iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_robust_scaler_median_centering() {
        // [1, 2, 3, 4, 5]: median=3, Q1=1.5 (interp), Q3=4.5, IQR=3
        let mut ds = Dataset::new(
            vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]],
            vec![0.0; 5],
            vec!["x".into()],
            "y",
        );
        let mut scaler = RobustScaler::new();
        scaler.fit_transform(&mut ds).unwrap();

        // median value should map to 0
        assert!(
            ds.features[0][2].abs() < 1e-10,
            "median should map to 0, got {}",
            ds.features[0][2]
        );
    }

    #[test]
    fn test_robust_scaler_outlier_tolerance() {
        // Data with an extreme outlier: [1, 2, 3, 4, 1000]
        let data = vec![1.0, 2.0, 3.0, 4.0, 1000.0];

        // StandardScaler: the outlier heavily influences mean/std
        let mut ds_std = Dataset::new(
            vec![data.clone()],
            vec![0.0; 5],
            vec!["x".into()],
            "y",
        );
        let mut std_scaler = StandardScaler::new();
        std_scaler.fit_transform(&mut ds_std).unwrap();

        // RobustScaler: outlier has minimal effect on median/IQR
        let mut ds_rob = Dataset::new(
            vec![data],
            vec![0.0; 5],
            vec!["x".into()],
            "y",
        );
        let mut rob_scaler = RobustScaler::new();
        rob_scaler.fit_transform(&mut ds_rob).unwrap();

        // In StandardScaler, the non-outlier values are squished near 0
        // because std is dominated by the outlier.
        // In RobustScaler, the non-outlier values have reasonable spread.
        let robust_range = ds_rob.features[0][3] - ds_rob.features[0][0];
        let std_range = ds_std.features[0][3] - ds_std.features[0][0];
        assert!(
            robust_range > std_range,
            "RobustScaler should give wider spread to non-outliers: robust={robust_range:.4} vs std={std_range:.4}"
        );
    }

    #[test]
    fn test_robust_scaler_roundtrip() {
        let original = vec![2.0, 4.0, 6.0, 8.0];
        let mut ds = Dataset::new(
            vec![original.clone()],
            vec![0.0; 4],
            vec!["x".into()],
            "y",
        );
        let mut scaler = RobustScaler::new();
        scaler.fit_transform(&mut ds).unwrap();
        scaler.inverse_transform(&mut ds).unwrap();

        for (a, b) in ds.features[0].iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-10, "roundtrip failed: {a} != {b}");
        }
    }
}
