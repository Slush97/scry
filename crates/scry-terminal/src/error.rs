// SPDX-License-Identifier: MIT OR Apache-2.0
//! Terminal emulator error types.
//!
//! [`TerminalError`] covers the failure modes of the GPU terminal emulator:
//! PTY operations, compositor creation, config parsing, GPU failures,
//! and window management.

/// Errors from the terminal emulator.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TerminalError {
    /// PTY operation failed (spawning, reading, resizing).
    #[error("PTY: {0}")]
    Pty(#[from] Box<dyn std::error::Error + Send + Sync>),

    /// GPU compositor creation or rendering failed.
    #[error("compositor: {0}")]
    Compositor(String),

    /// Configuration file parsing failed.
    #[error("config: {0}")]
    Config(String),

    /// GPU device initialisation failed.
    #[error("GPU: {0}")]
    Gpu(String),

    /// Native window could not be created.
    #[error("window creation failed: {0}")]
    WindowCreation(String),

    /// Thread spawn failed (e.g. resource exhaustion).
    #[error("thread spawn failed: {0}")]
    ThreadSpawn(std::io::Error),

    /// wgpu surface error.
    #[error("surface: {0}")]
    Surface(String),
}
