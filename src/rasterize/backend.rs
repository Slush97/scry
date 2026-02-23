// SPDX-License-Identifier: MIT OR Apache-2.0
//! Unified rasterization backend trait.
//!
//! [`RasterBackend`] provides a backend-agnostic interface for converting a
//! [`PixelCanvas`] scene into a `tiny_skia::Pixmap`. The two built-in
//! implementors are:
//!
//! - [`CpuBackend`] — wraps the `tiny-skia` based [`Rasterizer`].
//! - [`GpuBackend`] — wraps `WgpuRasterizer` + `WgpuContext2D`
//!   (requires the `gpu` feature).
//!
//! Consumers that were previously hard-coding GPU/CPU selection logic can
//! instead hold a `Box<dyn RasterBackend>` and swap backends freely.

use crate::scene::PixelCanvas;
use crate::PixelCanvasError;
use tiny_skia::Pixmap;

/// Describes which backend produced a rendering result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// CPU rasterization via `tiny-skia`.
    Cpu,
    /// GPU rasterization via `wgpu`.
    Gpu,
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => f.write_str("cpu"),
            Self::Gpu => f.write_str("gpu"),
        }
    }
}

/// A warning about a GPU command falling back to CPU rasterization.
#[derive(Clone, Debug)]
pub struct GpuFallbackWarning {
    /// The type of draw command that fell back.
    pub command_type: &'static str,
    /// Human-readable reason for the fallback.
    pub reason: &'static str,
}

/// Result of a rasterization including the pixmap and metadata.
#[derive(Debug)]
pub struct RasterResult {
    /// The rendered pixel buffer.
    pub pixmap: Pixmap,
    /// Which backend produced this result.
    pub backend: BackendKind,
    /// GPU commands that fell back to CPU (empty for pure-CPU rendering).
    pub gpu_fallbacks: Vec<GpuFallbackWarning>,
}

/// Lightweight metadata from [`RasterBackend::rasterize_into`].
///
/// Unlike [`RasterResult`], this does not contain a pixmap — callers already
/// have the target pixmap and don't need a duplicate.
#[derive(Clone, Debug)]
pub struct RasterMeta {
    /// Which backend produced this result.
    pub backend: BackendKind,
    /// GPU commands that fell back to CPU (empty for pure-CPU rendering).
    pub gpu_fallbacks: Vec<GpuFallbackWarning>,
}

/// Backend-agnostic rasterization interface.
///
/// Implement this trait to add alternative rendering backends (e.g. Vulkan,
/// software-only, wasm). The [`RasterPipeline`](super::RasterPipeline) uses
/// `dyn RasterBackend` internally so consumers don't need to wire GPU/CPU
/// selection themselves.
pub trait RasterBackend: Send {
    /// Rasterize a canvas scene into a new pixmap.
    ///
    /// # Errors
    ///
    /// Returns [`PixelCanvasError`] if the pixmap cannot be allocated or
    /// the backend encounters an unrecoverable error.
    fn rasterize(&self, canvas: &PixelCanvas) -> Result<RasterResult, PixelCanvasError>;

    /// Rasterize into an existing pixmap, avoiding allocation.
    ///
    /// The pixmap is cleared and fully redrawn. Returns lightweight
    /// metadata (backend kind + fallback warnings) without allocating
    /// a duplicate pixmap.
    ///
    /// # Panics
    ///
    /// Implementations should panic if `pixmap` dimensions don't match the
    /// canvas dimensions.
    fn rasterize_into(&self, canvas: &PixelCanvas, pixmap: &mut Pixmap) -> RasterMeta;

    /// Which backend kind this is.
    fn kind(&self) -> BackendKind;

    /// Human-readable backend name (for diagnostics).
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// CpuBackend
// ---------------------------------------------------------------------------

/// CPU rendering backend using `tiny-skia`.
///
/// This is always available and is the default fallback.
#[derive(Clone, Debug, Default)]
pub struct CpuBackend;

impl CpuBackend {
    /// Create a new CPU backend.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl RasterBackend for CpuBackend {
    fn rasterize(&self, canvas: &PixelCanvas) -> Result<RasterResult, PixelCanvasError> {
        let pixmap = super::skia::Rasterizer::rasterize(canvas)?;
        Ok(RasterResult {
            pixmap,
            backend: BackendKind::Cpu,
            gpu_fallbacks: Vec::new(),
        })
    }

    fn rasterize_into(&self, canvas: &PixelCanvas, pixmap: &mut Pixmap) -> RasterMeta {
        super::skia::Rasterizer::rasterize_into(canvas, pixmap);
        RasterMeta {
            backend: BackendKind::Cpu,
            gpu_fallbacks: Vec::new(),
        }
    }

    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn name(&self) -> &'static str {
        "cpu (tiny-skia)"
    }
}

// ---------------------------------------------------------------------------
// GpuBackend
// ---------------------------------------------------------------------------

/// GPU rendering backend using `wgpu`.
///
/// Wraps a reusable [`WgpuContext2D`](super::WgpuContext2D) and produces
/// rendered pixmaps via GPU shaders. Commands that the GPU cannot handle
/// (text, composited groups, gradient paths) fall back to CPU internally;
/// the fallback details are reported in [`RasterResult::gpu_fallbacks`].
#[cfg(feature = "gpu")]
pub struct GpuBackend {
    ctx: super::wgpu_context::WgpuContext2D,
}

#[cfg(feature = "gpu")]
impl GpuBackend {
    /// Create a new GPU backend by initialising a wgpu context.
    ///
    /// Uses [`GpuDevice::global()`](crate::gpu::GpuDevice::global) to share
    /// the GPU device singleton across all contexts.
    ///
    /// # Errors
    ///
    /// Returns an error string if GPU adapter/device creation fails.
    pub fn try_new() -> Result<Self, crate::gpu::GpuError> {
        let gpu = crate::gpu::GpuDevice::global()
            .ok_or(crate::gpu::GpuError::Unavailable)?;
        let ctx = super::wgpu_context::WgpuContext2D::with_device(gpu)?;
        Ok(Self { ctx })
    }

    /// Create a GPU backend from a pre-existing context.
    #[must_use]
    pub fn from_context(ctx: super::wgpu_context::WgpuContext2D) -> Self {
        Self { ctx }
    }

    /// Borrow the underlying GPU context.
    #[must_use]
    pub fn context(&self) -> &super::wgpu_context::WgpuContext2D {
        &self.ctx
    }
}

#[cfg(feature = "gpu")]
impl RasterBackend for GpuBackend {
    fn rasterize(&self, canvas: &PixelCanvas) -> Result<RasterResult, PixelCanvasError> {
        let pixmap =
            super::wgpu::WgpuRasterizer::rasterize_with_context(&self.ctx, canvas)?;
        Ok(RasterResult {
            pixmap,
            backend: BackendKind::Gpu,
            // TODO: wire fallback tracking from WgpuRasterizer once it
            //       accumulates GpuFallbackWarning entries.
            gpu_fallbacks: Vec::new(),
        })
    }

    fn rasterize_into(&self, canvas: &PixelCanvas, pixmap: &mut Pixmap) -> RasterMeta {
        if let Ok(gpu_pixmap) = super::wgpu::WgpuRasterizer::rasterize_with_context(&self.ctx, canvas) {
            pixmap.data_mut().copy_from_slice(gpu_pixmap.data());
            RasterMeta {
                backend: BackendKind::Gpu,
                gpu_fallbacks: Vec::new(),
            }
        } else {
            // GPU failed for this frame — fall back to CPU for this call
            super::skia::Rasterizer::rasterize_into(canvas, pixmap);
            RasterMeta {
                backend: BackendKind::Cpu,
                gpu_fallbacks: Vec::new(),
            }
        }
    }

    fn kind(&self) -> BackendKind {
        BackendKind::Gpu
    }

    fn name(&self) -> &'static str {
        "gpu (wgpu)"
    }
}

// ---------------------------------------------------------------------------
// AutoBackend
// ---------------------------------------------------------------------------

/// Auto-selecting backend: tries GPU first, falls back to CPU.
///
/// This is the recommended backend for most users. It attempts GPU
/// initialisation once and remembers the result. If GPU init fails, it
/// transparently uses CPU rendering for all subsequent calls.
pub struct AutoBackend {
    inner: Box<dyn RasterBackend>,
    health: crate::gpu::SharedHealthMonitor,
}

impl AutoBackend {
    /// Create an auto-selecting backend.
    ///
    /// Attempts GPU initialisation when the `gpu` feature is enabled.
    /// Falls back to CPU silently if GPU is unavailable.
    #[must_use]
    pub fn new() -> Self {
        #[cfg(feature = "gpu")]
        {
            // Try to get the health monitor from the global device
            let health = crate::gpu::GpuDevice::global()
                .map_or_else(
                    crate::gpu::health::shared_cpu_only_monitor,
                    crate::gpu::GpuDevice::health,
                );

            match GpuBackend::try_new() {
                Ok(gpu) => {
                    return Self {
                        inner: Box::new(gpu),
                        health,
                    };
                }
                Err(e) => {
                    crate::scry_debug!("[scry] GPU init failed, using CPU backend: {e}");
                }
            }

            Self {
                inner: Box::new(CpuBackend),
                health,
            }
        }

        #[cfg(not(feature = "gpu"))]
        Self {
            inner: Box::new(CpuBackend),
            health: crate::gpu::health::shared_cpu_only_monitor(),
        }
    }

    /// Which backend was actually selected.
    #[must_use]
    pub fn active_backend(&self) -> BackendKind {
        self.inner.kind()
    }
}

impl Default for AutoBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl RasterBackend for AutoBackend {
    fn rasterize(&self, canvas: &PixelCanvas) -> Result<RasterResult, PixelCanvasError> {
        // Use the health monitor's sticky CPU hold logic instead of
        // checking gpu_suitable() per-frame (prevents backend flipping).
        if self.inner.kind() == BackendKind::Gpu {
            let use_gpu = self.health.lock()
                .is_ok_and(|mut h| h.should_use_gpu_raster(canvas.gpu_suitable()));
            if !use_gpu {
                return CpuBackend.rasterize(canvas);
            }
        }
        self.inner.rasterize(canvas)
    }

    fn rasterize_into(&self, canvas: &PixelCanvas, pixmap: &mut Pixmap) -> RasterMeta {
        // Same sticky guard as rasterize().
        if self.inner.kind() == BackendKind::Gpu {
            let use_gpu = self.health.lock()
                .is_ok_and(|mut h| h.should_use_gpu_raster(canvas.gpu_suitable()));
            if !use_gpu {
                return CpuBackend.rasterize_into(canvas, pixmap);
            }
        }
        self.inner.rasterize_into(canvas, pixmap)
    }

    fn kind(&self) -> BackendKind {
        self.inner.kind()
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::style::Color;

    #[test]
    fn cpu_backend_rasterize() {
        let canvas = PixelCanvas::new(100, 100)
            .background(Color::WHITE)
            .circle(50.0, 50.0, 30.0)
            .fill(Color::RED)
            .done();

        let result = CpuBackend::new().rasterize(&canvas).unwrap();
        assert_eq!(result.backend, BackendKind::Cpu);
        assert_eq!(result.pixmap.width(), 100);
        assert_eq!(result.pixmap.height(), 100);
        assert!(result.gpu_fallbacks.is_empty());
    }

    #[test]
    fn cpu_backend_name() {
        assert_eq!(CpuBackend::new().name(), "cpu (tiny-skia)");
    }

    #[test]
    fn auto_backend_selects() {
        let auto = AutoBackend::new();
        // Should succeed regardless of GPU availability
        let canvas = PixelCanvas::new(50, 50);
        let result = auto.rasterize(&canvas).unwrap();
        assert!(result.pixmap.width() == 50);
    }

    #[test]
    fn backend_kind_display() {
        assert_eq!(format!("{}", BackendKind::Cpu), "cpu");
        assert_eq!(format!("{}", BackendKind::Gpu), "gpu");
    }
}
