// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared GPU device context for cross-module device reuse.
//!
//! [`GpuDevice`] wraps an `Arc<wgpu::Device>` + `Arc<wgpu::Queue>` so that
//! the SDF renderer, chart 3D backend, and 2D rasterizer can share a single
//! GPU device initialization (~100ms saved per additional context).
//!
//! # Example
//!
//! ```ignore
//! use scry_engine::gpu::GpuDevice;
//!
//! let gpu = GpuDevice::new()?;
//! // Pass to any GPU context that accepts `with_device()`
//! ```

use std::sync::Arc;

/// A shared GPU device and queue.
///
/// Wraps wgpu's `Device` and `Queue` in `Arc` for cheap cloning and sharing
/// across multiple rendering contexts (2D rasterizer, 3D chart backend,
/// SDF compute renderer).
///
/// Create once with [`GpuDevice::new()`] and pass to context constructors
/// that accept `with_device()`.
#[derive(Clone)]
pub struct GpuDevice {
    /// The wgpu device.
    pub(crate) device: Arc<wgpu::Device>,
    /// The wgpu queue.
    pub(crate) queue: Arc<wgpu::Queue>,
}

impl GpuDevice {
    /// Initialize a new GPU device.
    ///
    /// # Errors
    ///
    /// Returns an error string if no compatible GPU adapter is found.
    pub fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or("wgpu: no compatible GPU adapter found")?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("scry-shared-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| format!("wgpu: device creation failed: {e}"))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }

    /// Wrap an existing device and queue.
    #[must_use]
    pub fn from_existing(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self { device, queue }
    }

    /// Get a reference to the shared device.
    #[must_use]
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get a clone of the `Arc<Device>`.
    #[must_use]
    pub fn device_arc(&self) -> Arc<wgpu::Device> {
        Arc::clone(&self.device)
    }

    /// Get a reference to the shared queue.
    #[must_use]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Get a clone of the `Arc<Queue>`.
    #[must_use]
    pub fn queue_arc(&self) -> Arc<wgpu::Queue> {
        Arc::clone(&self.queue)
    }
}

impl std::fmt::Debug for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDevice")
            .field("device", &"<wgpu::Device>")
            .field("queue", &"<wgpu::Queue>")
            .finish()
    }
}
