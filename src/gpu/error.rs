// SPDX-License-Identifier: MIT OR Apache-2.0
//! GPU subsystem error types.
//!
//! [`GpuError`] covers every failure mode of the GPU device lifecycle:
//! adapter discovery, device creation, timeout, driver panics, and
//! pipeline compilation.  All GPU-facing APIs return `Result<_, GpuError>`
//! instead of `Result<_, String>`, enabling callers to match on specific
//! failure reasons and provide targeted recovery.

/// Errors from GPU device initialisation and pipeline compilation.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GpuError {
    /// No compatible GPU adapter was found (no Vulkan/Metal/DX12 device).
    #[error("no compatible GPU adapter found")]
    NoAdapter,

    /// The wgpu device could not be created from the adapter.
    #[error("device creation failed: {0}")]
    DeviceCreation(String),

    /// GPU initialisation exceeded the configured timeout.
    #[error("GPU initialization timed out after {0:?}")]
    InitTimeout(std::time::Duration),

    /// GPU initialisation panicked inside the driver (e.g. EGL `BadDisplay`).
    #[error("GPU initialization panicked (driver issue)")]
    InitPanicked,

    /// Shader or pipeline compilation failed.
    #[error("pipeline compilation failed: {0}")]
    PipelineCompilation(String),

    /// GPU buffer allocation failed.
    #[error("buffer allocation failed: {0}")]
    BufferAllocation(String),

    /// The global GPU device was not available.
    #[error("GPU not available: no compatible adapter found or init timed out")]
    Unavailable,
}
