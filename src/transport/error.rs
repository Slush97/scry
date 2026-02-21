// SPDX-License-Identifier: MIT OR Apache-2.0
//! Transport layer error types.
//!
//! [`TransportError`] covers failure modes for terminal graphics protocol
//! transmission: I/O, PNG encoding, compression, shared memory, and
//! protocol support.

/// Errors from the terminal graphics transport layer.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TransportError {
    /// Underlying I/O error (terminal write, pipe, etc.).
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),

    /// PNG encoding failed.
    #[error("PNG encoding failed: {0}")]
    PngEncoding(String),

    /// Zlib compression failed.
    #[error("zlib compression failed: {0}")]
    ZlibCompression(String),

    /// POSIX shared memory operation failed.
    #[error("shared memory: {0}")]
    SharedMemory(String),

    /// The requested protocol is not supported / not compiled in.
    #[error("protocol not supported: {0}")]
    UnsupportedProtocol(String),
}
