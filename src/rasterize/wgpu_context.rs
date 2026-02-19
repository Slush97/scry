// SPDX-License-Identifier: MIT OR Apache-2.0
//! Reusable GPU context for 2D rasterization.
//!
//! [`WgpuContext2D`] holds the expensive-to-create wgpu device, queue, and
//! a reference to the shared compiled render pipelines from the
//! [`PipelineRegistry`](crate::gpu::PipelineRegistry). Create one context
//! and reuse it across many frames via
//! [`WgpuRasterizer::with_context()`](super::wgpu::WgpuRasterizer).
//!
//! # Feature Gate
//!
//! This module is only available when the `gpu` feature is enabled (default).

// Re-export vertex types from the pipeline registry for use by wgpu.rs
pub(super) use crate::gpu::pipeline_registry::{
    create_frame_resources, GpuGradientStop, GradientUniforms, LineVertex, MeshVertex,
    ShapeInstance,
};

// ---------------------------------------------------------------------------
// WgpuContext2D
// ---------------------------------------------------------------------------

/// Reusable GPU context holding the wgpu device, queue, and a reference to
/// shared compiled pipelines.
///
/// Creating a `WgpuContext2D` via [`with_device()`](Self::with_device) is
/// cheap because it borrows already-compiled pipelines from the global
/// [`PipelineRegistry`](crate::gpu::PipelineRegistry).
///
/// # Example
///
/// ```ignore
/// use scry_engine::gpu::GpuDevice;
/// use scry_engine::rasterize::WgpuContext2D;
///
/// let gpu = GpuDevice::global().expect("no GPU");
/// let ctx = WgpuContext2D::with_device(gpu).unwrap();
/// // reuse `ctx` across frames...
/// ```
pub struct WgpuContext2D {
    pub(crate) device: std::sync::Arc<wgpu::Device>,
    pub(crate) queue: std::sync::Arc<wgpu::Queue>,
    /// Borrowed reference to the shared 2D pipelines.
    pub(crate) pipelines: &'static crate::gpu::Pipelines2D,
}

impl WgpuContext2D {
    /// Initialize the GPU context.
    ///
    /// This performs the expensive one-time setup:
    /// - `Instance` → `Adapter` → `Device` + `Queue`
    /// - Compile shape, line, and gradient WGSL shaders
    ///
    /// # Errors
    ///
    /// Returns an error string if no compatible GPU adapter is found or
    /// device creation fails.
    #[deprecated(
        since = "0.8.0",
        note = "Use `GpuDevice::global()` + `WgpuContext2D::with_device()` to share a single GPU device across contexts"
    )]
    pub fn new() -> Result<Self, String> {
        // For the deprecated path, initialize a GpuDevice and use it.
        // This still shares pipelines through the global singleton.
        let gpu = crate::gpu::GpuDevice::global_or_init()?;
        Self::with_device(gpu)
    }

    /// Create a context sharing an existing [`GpuDevice`](crate::gpu::GpuDevice).
    ///
    /// This is nearly instant because it borrows the lazily-compiled
    /// pipelines from the device's [`PipelineRegistry`](crate::gpu::PipelineRegistry).
    ///
    /// # Errors
    ///
    /// Returns an error string if pipeline compilation fails.
    pub fn with_device(gpu: &'static crate::gpu::GpuDevice) -> Result<Self, String> {
        let device = gpu.device_arc();
        let queue = gpu.queue_arc();
        // Lazily compile 2D pipelines (or return cached)
        let pipelines = gpu.pipelines().get_2d(gpu.device());
        Ok(Self {
            device,
            queue,
            pipelines,
        })
    }
}
