//! Error types for chart construction and validation.

use thiserror::Error;

/// Errors that can occur during chart construction or validation.
#[derive(Clone, Debug, PartialEq, Error)]
#[non_exhaustive]
pub enum ChartError {
    /// X and Y series have different lengths.
    #[error("X series ({x_len}) and Y series ({y_len}) have different lengths")]
    MismatchedLengths {
        /// Length of the X series.
        x_len: usize,
        /// Length of the Y series.
        y_len: usize,
    },

    /// All data values are non-finite (NaN or Infinity).
    #[error("all data values are NaN or Infinity")]
    AllNonFinite,

    /// No data was provided to the chart.
    #[error("no data provided")]
    EmptyData,

    /// Heatmap rows have inconsistent lengths.
    #[error("heatmap rows have inconsistent lengths")]
    JaggedGrid,

    /// Value range is invalid (min >= max or non-finite bounds).
    #[error("invalid range: min={min}, max={max}")]
    InvalidRange {
        /// The minimum bound that was provided.
        min: f64,
        /// The maximum bound that was provided.
        max: f64,
    },
}
