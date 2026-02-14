//! Data types for chart input.
//!
//! Provides [`Series`] — a named sequence of values that can be fed into charts.

/// A named sequence of floating-point data values.
///
/// # Examples
///
/// ```
/// use pixelchart::data::Series;
///
/// let s = Series::new("Temperature", vec![20.0, 22.5, 19.0, 25.0]);
/// assert_eq!(s.label(), "Temperature");
/// assert_eq!(s.len(), 4);
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[must_use]
pub struct Series {
    label: String,
    values: Vec<f64>,
    /// Symmetric error values (±error) for each data point.
    error_y: Option<Vec<f64>>,
}

impl Series {
    /// Create a new series with a label and values.
    ///
    /// Non-finite values (NaN, Infinity) are preserved in storage but
    /// excluded from statistical calculations (min, max, mean, extent).
    pub fn new(label: impl Into<String>, values: Vec<f64>) -> Self {
        Self {
            label: label.into(),
            values,
            error_y: None,
        }
    }

    /// Create an unlabeled series.
    pub fn from_values(values: Vec<f64>) -> Self {
        Self {
            label: String::new(),
            values,
            error_y: None,
        }
    }

    /// Attach symmetric error values (±error) to this series.
    ///
    /// The error vector must have the same length as the data values.
    /// Each error value represents the magnitude of the error bar.
    pub fn with_error(mut self, errors: Vec<f64>) -> Self {
        self.error_y = Some(errors);
        self
    }

    /// Get the error values, if set.
    #[must_use]
    pub fn error_values(&self) -> Option<&[f64]> {
        self.error_y.as_deref()
    }

    /// Return a new Series with all non-finite values (NaN, Infinity) removed.
    pub fn sanitized(&self) -> Self {
        Self {
            label: self.label.clone(),
            values: self
                .values
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .collect(),
            error_y: None, // loses error association on sanitize
        }
    }

    /// Returns only the finite values (filtering NaN and Infinity).
    pub fn finite_values(&self) -> impl Iterator<Item = f64> + '_ {
        self.values.iter().copied().filter(|v| v.is_finite())
    }

    /// Count of non-finite values in the series.
    #[must_use]
    pub fn non_finite_count(&self) -> usize {
        self.values.iter().filter(|v| !v.is_finite()).count()
    }

    /// The series label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// The data values.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Number of data points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Count of finite (non-NaN, non-Infinity) values.
    #[must_use]
    pub fn finite_count(&self) -> usize {
        self.values.iter().filter(|v| v.is_finite()).count()
    }

    /// Whether the series is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The minimum finite value, or `None` if no finite values exist.
    #[must_use]
    pub fn min(&self) -> Option<f64> {
        self.finite_values().reduce(f64::min)
    }

    /// The maximum finite value, or `None` if no finite values exist.
    #[must_use]
    pub fn max(&self) -> Option<f64> {
        self.finite_values().reduce(f64::max)
    }

    /// (min, max) extent of the data.
    #[must_use]
    pub fn extent(&self) -> Option<(f64, f64)> {
        match (self.min(), self.max()) {
            (Some(lo), Some(hi)) => Some((lo, hi)),
            _ => None,
        }
    }

    /// Mean of the finite data values.
    #[must_use]
    pub fn mean(&self) -> Option<f64> {
        let finite: Vec<f64> = self.finite_values().collect();
        if finite.is_empty() {
            None
        } else {
            Some(finite.iter().sum::<f64>() / finite.len() as f64)
        }
    }
}

impl From<Vec<f64>> for Series {
    fn from(values: Vec<f64>) -> Self {
        Self::from_values(values)
    }
}

impl From<&[f64]> for Series {
    fn from(values: &[f64]) -> Self {
        Self::from_values(values.to_vec())
    }
}

/// Convert a slice of f32 values into a Series (widening to f64).
impl From<&[f32]> for Series {
    fn from(values: &[f32]) -> Self {
        Self::from_values(values.iter().map(|&v| f64::from(v)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn series_basic() {
        let s = Series::new("test", vec![1.0, 5.0, 3.0]);
        assert_eq!(s.label(), "test");
        assert_eq!(s.len(), 3);
        assert!(!s.is_empty());
    }

    #[test]
    fn series_extent() {
        let s = Series::new("", vec![3.0, 1.0, 5.0, 2.0]);
        assert_eq!(s.min(), Some(1.0));
        assert_eq!(s.max(), Some(5.0));
        assert_eq!(s.extent(), Some((1.0, 5.0)));
    }

    #[test]
    fn series_mean() {
        let s = Series::new("", vec![2.0, 4.0, 6.0]);
        assert_eq!(s.mean(), Some(4.0));
    }

    #[test]
    fn empty_series() {
        let s = Series::from_values(vec![]);
        assert!(s.is_empty());
        assert_eq!(s.min(), None);
        assert_eq!(s.extent(), None);
        assert_eq!(s.mean(), None);
    }

    #[test]
    fn nan_infinity_filtered() {
        let s = Series::new(
            "dirty",
            vec![1.0, f64::NAN, 3.0, f64::INFINITY, 2.0, f64::NEG_INFINITY],
        );
        assert_eq!(s.len(), 6); // raw length includes non-finite
        assert_eq!(s.non_finite_count(), 3);
        assert_eq!(s.min(), Some(1.0));
        assert_eq!(s.max(), Some(3.0));
        assert_eq!(s.extent(), Some((1.0, 3.0)));
        assert_eq!(s.mean(), Some(2.0));

        let clean = s.sanitized();
        assert_eq!(clean.len(), 3);
        assert_eq!(clean.values(), &[1.0, 3.0, 2.0]);
    }

    #[test]
    fn all_nan_series() {
        let s = Series::new("all_nan", vec![f64::NAN, f64::NAN]);
        assert_eq!(s.min(), None);
        assert_eq!(s.max(), None);
        assert_eq!(s.extent(), None);
        assert_eq!(s.mean(), None);
    }
}
