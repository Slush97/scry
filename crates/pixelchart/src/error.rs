//! Error types for chart construction and validation.

use std::fmt;

/// Errors that can occur during chart construction or validation.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ChartError {
    /// X and Y series have different lengths.
    MismatchedLengths {
        /// Length of the X series.
        x_len: usize,
        /// Length of the Y series.
        y_len: usize,
    },

    /// All data values are non-finite (NaN or Infinity).
    AllNonFinite,

    /// No data was provided to the chart.
    EmptyData,

    /// Heatmap rows have inconsistent lengths.
    JaggedGrid,

    /// Value range is invalid (min >= max or non-finite bounds).
    InvalidRange {
        /// The minimum bound that was provided.
        min: f64,
        /// The maximum bound that was provided.
        max: f64,
    },
}

impl fmt::Display for ChartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChartError::MismatchedLengths { x_len, y_len } => {
                write!(
                    f,
                    "X series ({x_len}) and Y series ({y_len}) have different lengths"
                )
            }
            ChartError::AllNonFinite => {
                write!(f, "all data values are NaN or Infinity")
            }
            ChartError::EmptyData => {
                write!(f, "no data provided")
            }
            ChartError::JaggedGrid => {
                write!(f, "heatmap rows have inconsistent lengths")
            }
            ChartError::InvalidRange { min, max } => {
                write!(f, "invalid range: min={min}, max={max}")
            }
        }
    }
}

impl std::error::Error for ChartError {}
