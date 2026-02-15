//! Error types for scry-learn.

/// Errors produced by scry-learn operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ScryLearnError {
    /// The dataset contains no samples.
    #[error("dataset is empty — no samples to train on")]
    EmptyDataset,

    /// A referenced column does not exist.
    #[error("column not found: {0}")]
    InvalidColumn(String),

    /// Feature matrix dimensions do not match what the model expects.
    #[error("shape mismatch: expected {expected} features, got {got}")]
    ShapeMismatch {
        /// Expected number of features.
        expected: usize,
        /// Actual number of features provided.
        got: usize,
    },

    /// `predict()` or `transform()` called before `fit()`.
    #[error("model has not been fitted — call .fit() first")]
    NotFitted,

    /// An invalid hyperparameter was supplied.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// No convergence within the iteration limit.
    #[error("failed to converge after {iterations} iterations (tolerance: {tolerance})")]
    ConvergenceFailure {
        /// Number of iterations attempted.
        iterations: usize,
        /// Convergence tolerance.
        tolerance: f64,
    },

    /// I/O error during file operations.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// CSV parsing error.
    #[error("CSV error: {0}")]
    Csv(String),

    /// Chart rendering error.
    #[error("visualization error: {0}")]
    ChartError(String),
}

/// Convenience type alias.
pub type Result<T> = std::result::Result<T, ScryLearnError>;
