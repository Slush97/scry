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
pub use crate::gpu::pipeline_registry::{
    create_frame_resources, GpuGradientStop, GradientUniforms, LineVertex, MeshVertex,
    ShapeInstance,
};

use std::cell::RefCell;

// ---------------------------------------------------------------------------
// Buffer pool for grow-only GPU buffer reuse across frames
// ---------------------------------------------------------------------------

/// Cached GPU buffer slot: `(buffer, capacity_in_bytes)`.
type BufferSlot = Option<(wgpu::Buffer, u64)>;

/// Pool of reusable GPU buffers to avoid per-frame allocations.
///
/// Each slot grows only — if the data exceeds the current capacity, a new
/// buffer is allocated at 2× the required size (amortising allocations over
/// time). Matching the pattern used in `SdfGpuContext`.
#[derive(Default)]
#[allow(unreachable_pub)]
pub struct BufferPool {
    pub(super) shape: BufferSlot,
    pub(super) line: BufferSlot,
    pub(super) mesh: BufferSlot,
}

impl BufferPool {
    /// Ensure a cached buffer slot has enough capacity for `data`.
    /// Returns a clone of the buffer handle after uploading the data.
    ///
    /// If the cached buffer is too small (or absent), it is replaced with
    /// a new buffer sized at `max(required, 2 × old_capacity)`.
    pub(super) fn ensure_and_upload(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        slot: &mut BufferSlot,
        data: &[u8],
        usage: wgpu::BufferUsages,
        label: &str,
    ) -> wgpu::Buffer {
        let needed = data.len() as u64;
        if needed == 0 {
            // Return a tiny placeholder buffer
            return device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 16,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        let needs_realloc = slot.as_ref().is_none_or(|(_, cap)| *cap < needed);
        if needs_realloc {
            let old_cap = slot.as_ref().map_or(0, |(_, c)| *c);
            let new_cap = needed.max(old_cap * 2).max(256);
            *slot = Some((
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: new_cap,
                    usage: usage | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                new_cap,
            ));
        }
        let (buf, _) = slot.as_ref().unwrap();
        queue.write_buffer(buf, 0, data);
        buf.clone()
    }
}

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
    /// Pooled GPU buffers for cross-frame reuse (interior mutability).
    pub(crate) buffer_pool: RefCell<BufferPool>,
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
    pub fn new() -> Result<Self, crate::gpu::GpuError> {
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
    pub fn with_device(gpu: &'static crate::gpu::GpuDevice) -> Result<Self, crate::gpu::GpuError> {
        let device = gpu.device_arc();
        let queue = gpu.queue_arc();
        // Lazily compile 2D pipelines (or return cached)
        let pipelines = gpu.pipelines().get_2d(gpu.device());
        Ok(Self {
            device,
            queue,
            pipelines,
            buffer_pool: RefCell::new(BufferPool::default()),
        })
    }
}
