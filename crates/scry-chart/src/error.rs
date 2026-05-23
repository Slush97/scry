// SPDX-License-Identifier: MIT OR Apache-2.0
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
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[error("invalid range: min={min}, max={max}")]
    InvalidRange {
        /// The minimum bound that was provided.
        min: f64,
        /// The maximum bound that was provided.
        max: f64,
    },

    /// An I/O error occurred (e.g., writing to terminal or file).
    #[error("I/O error: {0}")]
    Io(String),

    /// A rendering or rasterization error occurred.
    #[error("render error: {0}")]
    Render(String),

    /// A configuration or data constraint was violated.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

impl From<std::io::Error> for ChartError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err.to_string())
    }
}

impl From<ChartError> for String {
    fn from(err: ChartError) -> Self {
        err.to_string()
    }
}
