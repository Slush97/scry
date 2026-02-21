// SPDX-License-Identifier: MIT OR Apache-2.0
//! Rasterization error types.
//!
//! [`RasterError`] covers every failure mode in the rasterization pipeline:
//! pixmap allocation, GPU backend failures, and invalid scene parameters.

use crate::gpu::GpuError;

/// Errors from the rasterization pipeline.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RasterError {
    /// Pixmap creation failed (invalid dimensions or OOM).
    #[error("pixmap creation failed: {0}")]
    PixmapCreation(String),

    /// The GPU backend could not be initialised.
    #[error("GPU backend: {0}")]
    Gpu(#[from] GpuError),

    /// A GPU render pass failed and CPU fallback was not possible.
    #[error("GPU render failed: {0}")]
    GpuRenderFailed(String),

    /// Scene dimensions are invalid (zero width/height).
    #[error("invalid scene dimensions: {width}x{height}")]
    InvalidDimensions {
        /// Requested width.
        width: u32,
        /// Requested height.
        height: u32,
    },
}
