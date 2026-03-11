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
//! let gpu = GpuDevice::global_or_init()?;
//! // Pass to any GPU context that accepts `with_device()`
//! ```

use super::error::GpuError;
use super::health::SharedHealthMonitor;
use super::pipeline_registry::PipelineRegistry;
use std::sync::Arc;

/// Diagnostic information about the GPU adapter.
///
/// Returned by [`GpuDevice::info()`] to expose adapter capabilities
/// without requiring callers to depend on `wgpu` types directly.
#[derive(Clone, Debug)]
pub struct GpuInfo {
    /// Human-readable adapter name (e.g. `NVIDIA GeForce RTX 3080`).
    pub adapter_name: String,
    /// Graphics API backend (e.g. "Vulkan", "Metal", "Dx12").
    pub backend: String,
    /// Device type (e.g. `DiscreteGpu`, `IntegratedGpu`, `Cpu`).
    pub device_type: String,
}

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
    /// Adapter diagnostics (populated during init).
    info: GpuInfo,
    /// Lazy pipeline registry — pipelines are compiled on first access.
    registry: PipelineRegistry,
    /// Shared GPU health monitor for coordinated GPU/CPU decisions.
    health: SharedHealthMonitor,
}

impl GpuDevice {
    /// Initialize a new GPU device.
    ///
    /// # Errors
    ///
    /// Returns an error string if no compatible GPU adapter is found.
    pub fn new() -> Result<Self, GpuError> {
        // Try primary native backends first (Vulkan, Metal, DX12).
        // GL/GLES backends can panic with EGL BadDisplay on headless/SSH
        // systems, and since try_new wraps us in catch_unwind, that panic
        // would prevent Vulkan from ever being tried.
        let primary_backends =
            wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12;

        Self::new_with_backends(primary_backends)
            .or_else(|_| Self::new_with_backends(wgpu::Backends::all()))
    }

    fn new_with_backends(backends: wgpu::Backends) -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or(GpuError::NoAdapter)?;

        let adapter_info = adapter.get_info();
        let info = GpuInfo {
            adapter_name: adapter_info.name.clone(),
            backend: format!("{:?}", adapter_info.backend),
            device_type: format!("{:?}", adapter_info.device_type),
        };

        // Request pipeline cache feature if available (Vulkan only) to
        // persist compiled shaders across runs and avoid 90s+ cold starts.
        let mut features = wgpu::Features::empty();
        if adapter.features().contains(wgpu::Features::PIPELINE_CACHE) {
            features |= wgpu::Features::PIPELINE_CACHE;
        }

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("scry-shared-gpu"),
                required_features: features,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| GpuError::DeviceCreation(e.to_string()))?;

        crate::scry_debug!(
            "[scry-gpu] Initialized: {} ({}, {})",
            info.adapter_name,
            info.backend,
            info.device_type,
        );

        let health = super::health::shared_health_monitor();
        {
            let mut h = health.lock().unwrap();
            h.mark_initialized();
        }

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            info,
            registry: PipelineRegistry::new(),
            health,
        })
    }

    /// Create a GPU device compatible with a specific window surface.
    ///
    /// Unlike [`new()`](Self::new), this requests an adapter that is compatible
    /// with the given surface, ensuring the device can present frames to that
    /// window. Uses the same backend priority (Vulkan/Metal/DX12 first),
    /// `PIPELINE_CACHE` support, and health monitoring as [`new()`](Self::new).
    ///
    /// # Errors
    ///
    /// Returns an error if no compatible GPU adapter is found.
    pub fn new_for_surface(
        instance: &wgpu::Instance,
        surface: &wgpu::Surface<'_>,
    ) -> Result<Self, GpuError> {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(surface),
            force_fallback_adapter: false,
        }))
        .ok_or(GpuError::NoAdapter)?;

        let adapter_info = adapter.get_info();
        let info = GpuInfo {
            adapter_name: adapter_info.name.clone(),
            backend: format!("{:?}", adapter_info.backend),
            device_type: format!("{:?}", adapter_info.device_type),
        };

        let mut features = wgpu::Features::empty();
        if adapter.features().contains(wgpu::Features::PIPELINE_CACHE) {
            features |= wgpu::Features::PIPELINE_CACHE;
        }

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("scry-shared-gpu"),
                required_features: features,
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| GpuError::DeviceCreation(e.to_string()))?;

        crate::scry_debug!(
            "[scry-gpu] Initialized (surface): {} ({}, {})",
            info.adapter_name,
            info.backend,
            info.device_type,
        );

        let health = super::health::shared_health_monitor();
        {
            let mut h = health.lock().unwrap();
            h.mark_initialized();
        }

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            info,
            registry: PipelineRegistry::new(),
            health,
        })
    }

    /// Try to initialize a GPU device with a timeout.
    ///
    /// Spawns GPU init on a background thread so that a hung adapter request
    /// (driver issue, display server contention) doesn't block the caller.
    ///
    /// Returns `None` if initialization fails or exceeds the timeout.
    pub fn try_new(timeout: std::time::Duration) -> Option<Self> {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            // Suppress panic output from wgpu backend probes (e.g. EGL
            // BadDisplay on headless/SSH systems). The panic is harmless —
            // Vulkan still works — but the message alarms users.
            let prev_hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(Self::new));
            std::panic::set_hook(prev_hook);
            let _ = tx.send(result.unwrap_or_else(|_| Err(GpuError::InitPanicked)));
        });
        match rx.recv_timeout(timeout) {
            Ok(Ok(gpu)) => Some(gpu),
            _ => None,
        }
    }

    /// Get or initialize a globally shared GPU device (singleton).
    ///
    /// The device is created on first call and cached for the process lifetime.
    /// Returns `None` if no compatible GPU adapter is available.
    ///
    /// This is the **recommended entry point** for GPU contexts. Pass the
    /// returned `GpuDevice` to `WgpuContext2D::with_device()`,
    /// `SdfGpuContext::with_device()`, etc. to avoid redundant ~100ms
    /// initialization per context.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use scry_engine::gpu::GpuDevice;
    ///
    /// if let Some(gpu) = GpuDevice::global() {
    ///     let ctx_2d = WgpuContext2D::with_device(gpu)?;
    ///     let ctx_sdf = SdfGpuContext::with_device(gpu)?;
    ///     // Both share the same underlying device + queue
    /// }
    /// ```
    #[must_use]
    pub fn global() -> Option<&'static Self> {
        use std::sync::OnceLock;
        static GLOBAL_GPU: OnceLock<Option<GpuDevice>> = OnceLock::new();
        GLOBAL_GPU
            .get_or_init(|| {
                // Use an 8-second timeout to avoid blocking forever on broken drivers.
                // Must exceed the outer GpuRenderCtx timeout (10s) to prevent
                // the OnceLock from permanently caching None on slow compilers.
                let result = Self::try_new(std::time::Duration::from_secs(8));
                if result.is_none() {
                    crate::scry_debug!(
                        "[scry-gpu] Global GPU init failed: no adapter found or init timed out"
                    );
                }
                result
            })
            .as_ref()
    }

    /// Get or initialize the global GPU device, returning a `Result`.
    ///
    /// Like [`global()`](Self::global), but returns an error message instead
    /// of `None` when the GPU is unavailable. Useful for call sites that
    /// want to propagate the failure reason.
    ///
    /// # Errors
    ///
    /// Returns an error string if no compatible GPU adapter is found or
    /// initialization timed out.
    pub fn global_or_init() -> Result<&'static Self, GpuError> {
        Self::global().ok_or(GpuError::Unavailable)
    }

    /// Check whether a GPU is available without initialising one.
    ///
    /// This calls [`global()`](Self::global) lazily — the first call may take
    /// up to 3 seconds if the driver is slow.
    #[must_use]
    pub fn is_available() -> bool {
        Self::global().is_some()
    }

    /// Wrap an existing device and queue.
    #[must_use]
    pub fn from_existing(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let health = super::health::shared_health_monitor();
        {
            let mut h = health.lock().unwrap();
            h.mark_initialized();
        }
        Self {
            device,
            queue,
            info: GpuInfo {
                adapter_name: "external".to_string(),
                backend: "unknown".to_string(),
                device_type: "unknown".to_string(),
            },
            registry: PipelineRegistry::new(),
            health,
        }
    }

    /// Get diagnostic information about the GPU adapter.
    ///
    /// Returns the adapter name, backend API, and device type that were
    /// detected during initialization.
    #[must_use]
    pub fn info(&self) -> &GpuInfo {
        &self.info
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

    /// Get the device limits that were used to create this device.
    ///
    /// Useful for surface configuration clamping (e.g. ensuring texture
    /// dimensions don't exceed `max_texture_dimension_2d`).
    #[must_use]
    pub fn limits(&self) -> wgpu::Limits {
        self.device.limits()
    }

    /// Get the lazy pipeline registry.
    ///
    /// Pipelines are compiled on first access per category (2D, SDF).
    /// Subsequent calls return the cached pipelines.
    #[must_use]
    pub fn pipelines(&self) -> &PipelineRegistry {
        &self.registry
    }

    /// Get the shared GPU health monitor.
    ///
    /// Both the SDF pipeline and 2D rasterizer consult this to make
    /// coordinated GPU/CPU decisions.
    #[must_use]
    pub fn health(&self) -> SharedHealthMonitor {
        Arc::clone(&self.health)
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
