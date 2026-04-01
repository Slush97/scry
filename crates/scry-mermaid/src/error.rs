// SPDX-License-Identifier: MIT OR Apache-2.0
//! Error types for the Mermaid parser and renderer.

/// Errors that can occur when parsing or rendering Mermaid diagrams.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum MermaidError {
    /// Failed to parse the Mermaid source.
    #[error("parse error at line {line}: {message}")]
    Parse {
        /// 1-based line number.
        line: usize,
        /// Human-readable description.
        message: String,
    },

    /// Unsupported diagram type.
    #[error("unsupported diagram type: {0}")]
    UnsupportedDiagram(String),

    /// A referenced node was never defined.
    #[error("undefined node: {0}")]
    UndefinedNode(String),

    /// Layout computation failed.
    #[error("layout failed: {0}")]
    Layout(String),

    /// Rendering failed (scry-engine error).
    #[error("render failed: {0}")]
    Render(String),
}
