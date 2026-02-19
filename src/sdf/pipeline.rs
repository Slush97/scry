// SPDX-License-Identifier: MIT OR Apache-2.0
//! High-level SDF rendering pipeline with automatic GPU/CPU fallback.
//!
//! [`SdfPipeline`] encapsulates the full GPU SDF lifecycle:
//! submit → readback → optional upscale, with CPU fallback when GPU is
//! unavailable.  It manages the double-buffered pipeline (display previous
//! frame while GPU computes the next) so that callers need only a single
//! `render()` call per frame.
//!
//! # Example
//!
//! ```ignore
//! use scry_engine::sdf::pipeline::SdfPipeline;
//! use scry_engine::sdf::{SdfScene, SdfCamera, SdfLight, SdfObject, SdfShape, Material, Vec3};
//! use scry_engine::scene::style::Color;
//!
//! let mut pipeline = SdfPipeline::new();
//! let scene = SdfScene::new()
//!     .object(SdfObject::new(SdfShape::Sphere { radius: 1.0 },
//!                            Material::matte(Color::RED))
//!         .at(Vec3::new(0.0, 1.0, 0.0)))
//!     .light(SdfLight::new(Vec3::new(5.0, 10.0, 5.0), Color::WHITE, 1.0));
//!
//! let result = pipeline.render(&scene, 400, 300, 0.0);
//! assert_eq!(result.width, 400);
//! ```

use crate::scene::command::ImageData;
use crate::sdf::renderer::SdfRenderer;
use crate::sdf::scene::SdfScene;

/// Which renderer produced the SDF frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdfBackend {
    /// CPU ray marching.
    Cpu,
    /// GPU compute shader.
    Gpu,
}

impl std::fmt::Display for SdfBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => f.write_str("cpu"),
            Self::Gpu => f.write_str("gpu"),
        }
    }
}

/// Result of an SDF pipeline render.
pub struct SdfRenderResult {
    /// The rendered RGBA pixel data.
    pub image: ImageData,
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Which backend produced this frame.
    pub backend: SdfBackend,
}

/// High-level SDF rendering pipeline.
///
/// Handles:
/// - GPU context initialization (via `GpuDevice::global()`)
/// - Double-buffered GPU submit → readback pipeline
/// - Optional render-scale + bicubic upscale
/// - Automatic CPU fallback when GPU is unavailable
///
/// # Double-Buffered Pipeline (GPU mode)
///
/// When GPU is active, the pipeline is 1-frame latent:
/// 1. **Submit** current frame's scene to GPU
/// 2. **Return** previous frame's result immediately (zero-copy)
/// 3. **Readback** GPU result next frame
///
/// This overlaps GPU compute with CPU work (terminal draw, event polling).
/// The first frame always uses CPU rendering.
pub struct SdfPipeline {
    /// GPU context (None = not yet tried, Some(None) = tried and failed).
    #[cfg(feature = "sdf-gpu")]
    gpu_ctx: Option<Option<super::gpu_renderer::SdfGpuContext>>,
    /// Whether GPU is currently active.
    gpu_active: bool,
    /// Render scale factor (0.0..=1.0). Below 1.0, renders at reduced
    /// resolution and upscales with bicubic interpolation.
    render_scale: f32,
    /// Previous frame's GPU result (for the double-buffer pipeline).
    prev_image: Option<ImageData>,
    /// Whether a GPU submission is in-flight.
    #[cfg(feature = "sdf-gpu")]
    gpu_submitted: bool,
    /// Dimensions of the in-flight GPU render.
    #[cfg(feature = "sdf-gpu")]
    pending_render_w: u32,
    #[cfg(feature = "sdf-gpu")]
    pending_render_h: u32,
    /// Full output dimensions (before render scale).
    #[cfg(feature = "sdf-gpu")]
    pending_full_w: u32,
    #[cfg(feature = "sdf-gpu")]
    pending_full_h: u32,
    /// Reusable readback buffer.
    #[cfg(feature = "sdf-gpu")]
    readback_buf: Vec<u8>,
    /// Background thread handle for async GPU init.
    #[cfg(feature = "sdf-gpu")]
    gpu_init_handle: Option<std::thread::JoinHandle<Option<super::gpu_renderer::SdfGpuContext>>>,
}

impl SdfPipeline {
    /// Create a new SDF pipeline with auto-detected GPU.
    ///
    /// GPU initialization (adapter detection + shader compilation) runs on a
    /// background thread so construction returns instantly. Early frames use
    /// CPU rendering; once the GPU is ready it switches automatically.
    #[must_use]
    pub fn new() -> Self {
        #[cfg(feature = "sdf-gpu")]
        let gpu_init_handle = Some(std::thread::spawn(|| {
            use crate::gpu::GpuDevice;
            use std::io::Write;
            // Write to a separate file to avoid any sharing issues with the main diag log
            let mut log = std::fs::File::create("/tmp/scry_gpu_init.log").ok();
            macro_rules! glog {
                ($($arg:tt)*) => {
                    if let Some(f) = log.as_mut() { let _ = writeln!(f, $($arg)*); let _ = f.flush(); }
                };
            }
            glog!("[gpu-thread] thread started");
            let t0 = std::time::Instant::now();
            glog!("[gpu-thread] calling GpuDevice::global()...");
            let gpu = match GpuDevice::global() {
                Some(g) => g,
                None => {
                    glog!("[gpu-thread] GpuDevice::global() returned None after {:?}", t0.elapsed());
                    return None;
                }
            };
            glog!("[gpu-thread] device ready in {:?}: {} ({} {})",
                t0.elapsed(), gpu.info().adapter_name, gpu.info().backend, gpu.info().device_type);
            glog!("[gpu-thread] compiling SDF shader...");
            let t1 = std::time::Instant::now();
            match super::gpu_renderer::SdfGpuContext::with_device(gpu) {
                Ok(ctx) => {
                    glog!("[gpu-thread] SDF context ready in {:?} (total {:?})", t1.elapsed(), t0.elapsed());
                    Some(ctx)
                }
                Err(e) => {
                    glog!("[gpu-thread] SDF context FAILED after {:?}: {e}", t1.elapsed());
                    None
                }
            }
        }));

        Self {
            #[cfg(feature = "sdf-gpu")]
            gpu_ctx: None,
            gpu_active: false,
            render_scale: 1.0,
            prev_image: None,
            #[cfg(feature = "sdf-gpu")]
            gpu_submitted: false,
            #[cfg(feature = "sdf-gpu")]
            pending_render_w: 0,
            #[cfg(feature = "sdf-gpu")]
            pending_render_h: 0,
            #[cfg(feature = "sdf-gpu")]
            pending_full_w: 0,
            #[cfg(feature = "sdf-gpu")]
            pending_full_h: 0,
            #[cfg(feature = "sdf-gpu")]
            readback_buf: Vec::new(),
            #[cfg(feature = "sdf-gpu")]
            gpu_init_handle,
        }
    }

    /// Poll the background GPU init thread (non-blocking).
    ///
    /// If the thread has finished, takes the result and sets up the GPU
    /// context. If still running, returns immediately (CPU renders this frame).
    #[cfg(feature = "sdf-gpu")]
    fn ensure_gpu(&mut self) {
        if self.gpu_ctx.is_some() {
            return; // already resolved
        }
        let Some(handle) = self.gpu_init_handle.take() else {
            return;
        };
        if handle.is_finished() {
            match handle.join() {
                Ok(Some(ctx)) => {
                    self.gpu_ctx = Some(Some(ctx));
                    self.gpu_active = true;
                }
                _ => {
                    self.gpu_ctx = Some(None);
                }
            }
        } else {
            // Still compiling — put the handle back, use CPU this frame
            self.gpu_init_handle = Some(handle);
        }
    }

    /// Create a pipeline that always uses CPU rendering.
    #[must_use]
    pub fn cpu_only() -> Self {
        Self {
            #[cfg(feature = "sdf-gpu")]
            gpu_ctx: Some(None),
            gpu_active: false,
            render_scale: 1.0,
            prev_image: None,
            #[cfg(feature = "sdf-gpu")]
            gpu_submitted: false,
            #[cfg(feature = "sdf-gpu")]
            pending_render_w: 0,
            #[cfg(feature = "sdf-gpu")]
            pending_render_h: 0,
            #[cfg(feature = "sdf-gpu")]
            pending_full_w: 0,
            #[cfg(feature = "sdf-gpu")]
            pending_full_h: 0,
            #[cfg(feature = "sdf-gpu")]
            readback_buf: Vec::new(),
            #[cfg(feature = "sdf-gpu")]
            gpu_init_handle: None,
        }
    }

    /// Set the render scale (0.0..=1.0).
    ///
    /// Values below 1.0 render at reduced resolution and upscale with
    /// bicubic interpolation. Useful for interactive preview.
    #[must_use]
    pub fn render_scale(mut self, scale: f32) -> Self {
        self.render_scale = scale.clamp(0.1, 1.0);
        self
    }

    /// Get the current render scale.
    #[must_use]
    pub fn get_render_scale(&self) -> f32 {
        self.render_scale
    }

    /// Set the render scale at runtime.
    pub fn set_render_scale(&mut self, scale: f32) {
        self.render_scale = scale.clamp(0.1, 1.0);
    }

    /// Whether the GPU backend is active.
    #[must_use]
    pub fn is_gpu_active(&self) -> bool {
        self.gpu_active
    }

    /// Human-readable name of the active SDF backend.
    #[must_use]
    pub fn backend_name(&self) -> &'static str {
        #[cfg(feature = "sdf-gpu")]
        if self.gpu_ctx.is_none() {
            return "pending (GPU init deferred)";
        }
        if self.gpu_active {
            "wgpu SDF compute"
        } else {
            "CPU ray march"
        }
    }

    /// Toggle GPU on/off at runtime.
    pub fn set_gpu_active(&mut self, active: bool) {
        #[cfg(feature = "sdf-gpu")]
        {
            // If GPU init is still pending, try to resolve it now
            if active && self.gpu_ctx.is_none() {
                if let Some(handle) = self.gpu_init_handle.take() {
                    // Block briefly to see if GPU is ready
                    match handle.join() {
                        Ok(Some(ctx)) => {
                            self.gpu_ctx = Some(Some(ctx));
                        }
                        _ => {
                            self.gpu_ctx = Some(None);
                        }
                    }
                }
            }
            if active && self.gpu_ctx.as_ref().and_then(|o| o.as_ref()).is_some() {
                self.gpu_active = true;
            } else {
                self.gpu_active = false;
            }
        }
        #[cfg(not(feature = "sdf-gpu"))]
        {
            let _ = active;
            self.gpu_active = false;
        }
    }

    /// Render a scene, returning the result.
    ///
    /// In GPU mode this uses a 1-frame latent pipeline:
    /// - Submits the current scene to GPU
    /// - Returns the previous frame's GPU result (or CPU-renders the first frame)
    /// - The GPU result is available next frame
    ///
    /// In CPU mode, blocking rendering happens immediately.
    pub fn render(
        &mut self,
        scene: &SdfScene,
        width: u32,
        height: u32,
        time: f32,
    ) -> SdfRenderResult {
        // Lazy GPU init on first render call
        #[cfg(feature = "sdf-gpu")]
        self.ensure_gpu();

        if width == 0 || height == 0 {
            return SdfRenderResult {
                image: ImageData::new(1, 1, vec![0; 4]),
                width: 1,
                height: 1,
                backend: SdfBackend::Cpu,
            };
        }

        // Calculate render dimensions (may be downscaled)
        let render_w = if self.render_scale < 1.0 {
            ((width as f32 * self.render_scale) as u32).max(1)
        } else {
            width
        };
        let render_h = if self.render_scale < 1.0 {
            ((height as f32 * self.render_scale) as u32).max(1)
        } else {
            height
        };

        // ── Phase 1: Readback previous GPU frame (if any) ──
        self.readback_pending();

        // ── Phase 2: Submit new GPU work ──
        #[cfg(feature = "sdf-gpu")]
        if self.gpu_active {
            if let Some(Some(ctx)) = self.gpu_ctx.as_mut() {
                if super::gpu_renderer::SdfGpuRenderer::submit(
                    ctx, scene, render_w, render_h, time,
                )
                .is_ok()
                {
                    self.gpu_submitted = true;
                    self.pending_render_w = render_w;
                    self.pending_render_h = render_h;
                    self.pending_full_w = width;
                    self.pending_full_h = height;
                }
            }
        }

        // ── Phase 3: Return previous frame, GPU sync, or CPU fallback ──
        if let Some(image) = self.prev_image.take() {
            SdfRenderResult {
                width: image.width(),
                height: image.height(),
                image,
                backend: SdfBackend::Gpu,
            }
        } else {
            // First frame: use synchronous GPU render if available,
            // otherwise fall back to CPU.
            #[cfg(feature = "sdf-gpu")]
            if self.gpu_submitted {
                // We just submitted GPU work above — do a blocking readback
                // instead of wasting time on a CPU render.
                self.gpu_submitted = false;
                if let Some(Some(ctx)) = self.gpu_ctx.as_mut() {
                    if render_w == width && render_h == height {
                        if super::gpu_renderer::SdfGpuRenderer::readback_into(
                            ctx,
                            render_w,
                            render_h,
                            &mut self.readback_buf,
                        )
                        .is_ok()
                        {
                            let data = std::mem::take(&mut self.readback_buf);
                            let image = ImageData::new(width, height, data);
                            return SdfRenderResult {
                                width,
                                height,
                                image,
                                backend: SdfBackend::Gpu,
                            };
                        }
                    } else if let Ok(pm) =
                        super::gpu_renderer::SdfGpuRenderer::readback(ctx, render_w, render_h)
                    {
                        let upscaled = crate::sdf::upscale::upscale_bicubic(
                            pm.data(),
                            render_w,
                            render_h,
                            width,
                            height,
                        );
                        let image = ImageData::new(width, height, upscaled);
                        return SdfRenderResult {
                            width,
                            height,
                            image,
                            backend: SdfBackend::Gpu,
                        };
                    }
                }
            }
            self.render_cpu(scene, width, height, render_w, render_h, time)
        }
    }

    /// Render synchronously (blocking), without the double-buffer pipeline.
    ///
    /// Uses GPU if available, with immediate readback. Falls back to CPU.
    /// Useful for single-frame rendering (screenshots, exports).
    pub fn render_sync(
        &mut self,
        scene: &SdfScene,
        width: u32,
        height: u32,
        time: f32,
    ) -> SdfRenderResult {
        // Lazy GPU init on first render call
        #[cfg(feature = "sdf-gpu")]
        self.ensure_gpu();

        let render_w = if self.render_scale < 1.0 {
            ((width as f32 * self.render_scale) as u32).max(1)
        } else {
            width
        };
        let render_h = if self.render_scale < 1.0 {
            ((height as f32 * self.render_scale) as u32).max(1)
        } else {
            height
        };

        // Try GPU first (blocking submit + readback)
        #[cfg(feature = "sdf-gpu")]
        if self.gpu_active {
            if let Some(Some(ctx)) = self.gpu_ctx.as_mut() {
                if let Ok(pixmap) = super::gpu_renderer::SdfGpuRenderer::render_to_pixmap(
                    ctx, scene, render_w, render_h, time,
                ) {
                    let image = if render_w != width || render_h != height {
                        let upscaled = crate::sdf::upscale::upscale_bicubic(
                            pixmap.data(),
                            render_w,
                            render_h,
                            width,
                            height,
                        );
                        ImageData::new(width, height, upscaled)
                    } else {
                        ImageData::new(width, height, pixmap.data().to_vec())
                    };
                    return SdfRenderResult {
                        width,
                        height,
                        image,
                        backend: SdfBackend::Gpu,
                    };
                }
            }
        }

        // CPU fallback
        self.render_cpu(scene, width, height, render_w, render_h, time)
    }

    /// Flush: readback any pending GPU frame and store it.
    ///
    /// Call this after your terminal draw/flush to overlap GPU compute
    /// with terminal I/O. The result will be available via `render()` next frame.
    pub fn flush(&mut self) {
        self.readback_pending();
    }

    // ── Internal helpers ──

    fn readback_pending(&mut self) {
        #[cfg(feature = "sdf-gpu")]
        if self.gpu_submitted {
            if let Some(Some(ctx)) = self.gpu_ctx.as_mut() {
                let rw = self.pending_render_w;
                let rh = self.pending_render_h;
                let fw = self.pending_full_w;
                let fh = self.pending_full_h;

                if rw == fw && rh == fh {
                    // No upscale needed
                    if super::gpu_renderer::SdfGpuRenderer::readback_into(
                        ctx,
                        rw,
                        rh,
                        &mut self.readback_buf,
                    )
                    .is_ok()
                    {
                        let data = std::mem::take(&mut self.readback_buf);
                        self.prev_image = Some(ImageData::new(fw, fh, data));
                    }
                } else if let Ok(pm) =
                    super::gpu_renderer::SdfGpuRenderer::readback(ctx, rw, rh)
                {
                    let upscaled = crate::sdf::upscale::upscale_bicubic(
                        pm.data(),
                        rw,
                        rh,
                        fw,
                        fh,
                    );
                    self.prev_image = Some(ImageData::new(fw, fh, upscaled));
                }
            }
            self.gpu_submitted = false;
        }
    }

    fn render_cpu(
        &self,
        scene: &SdfScene,
        width: u32,
        height: u32,
        render_w: u32,
        render_h: u32,
        time: f32,
    ) -> SdfRenderResult {
        match SdfRenderer::render_to_pixmap(scene, render_w, render_h, time) {
            Ok(pixmap) => {
                let image = if render_w != width || render_h != height {
                    let upscaled = crate::sdf::upscale::upscale_bicubic(
                        pixmap.data(),
                        render_w,
                        render_h,
                        width,
                        height,
                    );
                    ImageData::new(width, height, upscaled)
                } else {
                    ImageData::new(width, height, pixmap.data().to_vec())
                };
                SdfRenderResult {
                    width,
                    height,
                    image,
                    backend: SdfBackend::Cpu,
                }
            }
            Err(e) => {
                if crate::scry_debug_enabled() {
                    eprintln!("[scry] CPU SDF render failed: {e}");
                }
                // Return a blank image rather than crashing
                SdfRenderResult {
                    image: ImageData::new(width, height, vec![0; (width * height * 4) as usize]),
                    width,
                    height,
                    backend: SdfBackend::Cpu,
                }
            }
        }
    }
}

impl Drop for SdfPipeline {
    fn drop(&mut self) {
        // Join the background GPU init thread to avoid heap corruption
        // from wgpu resources being freed during process teardown.
        #[cfg(feature = "sdf-gpu")]
        if let Some(handle) = self.gpu_init_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Default for SdfPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SdfPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SdfPipeline")
            .field("gpu_active", &self.gpu_active)
            .field("render_scale", &self.render_scale)
            .field("has_prev_image", &self.prev_image.is_some())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::style::Color;
    use crate::sdf::*;

    fn simple_scene() -> SdfScene {
        SdfScene::new()
            .object(
                SdfObject::new(
                    SdfShape::Sphere { radius: 1.0 },
                    Material::matte(Color::RED),
                )
                .at(Vec3::new(0.0, 1.0, 0.0)),
            )
            .object(SdfObject::new(
                SdfShape::Plane,
                Material::matte(Color::from_rgba8(180, 180, 180, 255)),
            ))
            .light(SdfLight::new(Vec3::new(5.0, 10.0, 5.0), Color::WHITE, 1.0))
            .camera(SdfCamera::new(
                Vec3::new(0.0, 3.0, 6.0),
                Vec3::ZERO,
                45.0,
            ))
    }

    #[test]
    fn cpu_pipeline_renders() {
        let mut pipeline = SdfPipeline::cpu_only();
        let scene = simple_scene();
        let result = pipeline.render(&scene, 100, 75, 0.0);
        assert_eq!(result.width, 100);
        assert_eq!(result.height, 75);
        assert_eq!(result.backend, SdfBackend::Cpu);
        assert_eq!(result.image.data().len(), 100 * 75 * 4);
    }

    #[test]
    fn cpu_pipeline_sync_renders() {
        let mut pipeline = SdfPipeline::cpu_only();
        let scene = simple_scene();
        let result = pipeline.render_sync(&scene, 80, 60, 0.0);
        assert_eq!(result.width, 80);
        assert_eq!(result.height, 60);
        assert_eq!(result.backend, SdfBackend::Cpu);
    }

    #[test]
    fn render_scale_works() {
        let mut pipeline = SdfPipeline::cpu_only().render_scale(0.5);
        let scene = simple_scene();
        let result = pipeline.render_sync(&scene, 200, 150, 0.0);
        // Output should be full size (upscaled)
        assert_eq!(result.width, 200);
        assert_eq!(result.height, 150);
    }

    #[test]
    fn zero_dimensions_handled() {
        let mut pipeline = SdfPipeline::cpu_only();
        let scene = simple_scene();
        let result = pipeline.render(&scene, 0, 0, 0.0);
        assert_eq!(result.width, 1);
        assert_eq!(result.height, 1);
    }

    #[test]
    fn auto_pipeline_selects_backend() {
        let pipeline = SdfPipeline::new();
        // Should work regardless of GPU availability
        assert!(pipeline.render_scale == 1.0);
    }

    #[test]
    fn backend_display() {
        assert_eq!(format!("{}", SdfBackend::Cpu), "cpu");
        assert_eq!(format!("{}", SdfBackend::Gpu), "gpu");
    }
}
