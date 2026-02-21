// SPDX-License-Identifier: MIT OR Apache-2.0
//! SDF renderer error types.
//!
//! [`SdfError`] covers failures specific to the signed distance field
//! rendering pipeline: GPU context creation, shader compilation,
//! readback, and pixmap allocation.

use crate::gpu::GpuError;

/// Errors from the SDF ray marching renderer.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SdfError {
    /// GPU context could not be created.
    #[error("GPU context unavailable: {0}")]
    Gpu(#[from] GpuError),

    /// GPU readback failed (device lost, buffer unmap error, etc.).
    #[error("readback failed: {0}")]
    ReadbackFailed(String),

    /// GPU readback exceeded the timeout (potential device-lost deadlock).
    #[error("readback timed out after {0:?}")]
    ReadbackTimeout(std::time::Duration),

    /// SDF compute shader compilation failed.
    #[error("shader compilation failed: {0}")]
    ShaderCompilation(String),

    /// The scene has no renderable objects.
    #[error("scene has no objects")]
    EmptyScene,

    /// `Pixmap::new()` returned `None` for the given dimensions.
    #[error("pixmap creation failed for {width}x{height}")]
    PixmapCreation {
        /// Requested width.
        width: u32,
        /// Requested height.
        height: u32,
    },
}
