//! Crate-specific error types for scry-pipe.

use thiserror::Error;

/// All errors produced by scry-pipe operations.
#[derive(Debug, Error)]
pub enum PipeError {
    /// Schema validation failure (e.g. wrong input length).
    #[error("schema error: {0}")]
    Schema(String),

    /// A transform step failed on a specific feature.
    #[error("transform error on feature {feature_idx}: {message}")]
    Transform {
        /// Index of the feature that caused the error.
        feature_idx: usize,
        /// Human-readable description of what went wrong.
        message: String,
    },

    /// The pipeline has not been fitted yet.
    #[error("unfitted pipeline: call fit() before transform()")]
    Unfitted,

    /// JSON serialization / deserialization error.
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// Error during code generation.
    #[error("codegen error: {0}")]
    Codegen(String),

    /// Filesystem I/O error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
