// SPDX-License-Identifier: MIT OR Apache-2.0
//! GPU-accelerated SDF ray marching renderer via wgpu compute shaders.
//!
//! [`SdfGpuRenderer`] provides the same output as the CPU [`SdfRenderer`]
//! but runs the sphere-tracing, shading, and shadow computations on the GPU.
//!
//! # Example
//!
//! ```ignore
//! use scry_engine::sdf::*;
//! use scry_engine::sdf::gpu_renderer::{SdfGpuContext, SdfGpuRenderer};
//! use scry_engine::scene::style::Color;
//!
//! let mut ctx = SdfGpuContext::new().expect("GPU init failed");
//! let scene = SdfScene::new()
//!     .object(SdfObject::new(SdfShape::Sphere { radius: 1.0 },
//!                            Material::mirror(Color::WHITE, 0.8))
//!         .at(Vec3::new(0.0, 1.0, 0.0)))
//!     .light(SdfLight::new(Vec3::new(5.0, 10.0, 5.0), Color::WHITE, 1.0))
//!     .camera(SdfCamera::new(Vec3::new(0.0, 3.0, 6.0), Vec3::ZERO, 45.0));
//!
//! let pixmap = SdfGpuRenderer::render_to_pixmap(&mut ctx, &scene, 640, 360, 0.0).unwrap();
//! ```

use crate::PixelCanvasError;

use super::gpu_flatten::{
    build_glyph_data, build_lights, build_objects, build_uniforms, GpuLight, GpuObject,
};
use super::scene::SdfScene;

use bytemuck::Zeroable;
use tiny_skia::Pixmap;

// ── GPU context ────────────────────────────────────────────────────

/// Reusable GPU context for SDF rendering.
///
/// Creating a context via [`with_device()`](Self::with_device) is cheap
/// because it borrows already-compiled pipelines from the global
/// [`PipelineRegistry`](crate::gpu::PipelineRegistry).
pub struct SdfGpuContext {
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
    /// Borrowed reference to the shared SDF pipelines.
    pipelines: &'static crate::gpu::PipelinesSdf,
    /// Grow-only buffer pool for all GPU buffers.
    pool: crate::gpu::BufferPool,
    /// Cached bind group: `(bind_group, output_size, objects_size, lights_size, glyph_meta_size, glyph_grids_size)`.
    cached_bind_group: Option<(wgpu::BindGroup, u64, u64, u64, u64, u64)>,
    /// Whether a GPU submission is in-flight and readback is pending.
    pending_readback: bool,
    /// Reusable pixmap for readback to avoid per-frame allocation.
    cached_pixmap: Option<Pixmap>,
}

impl SdfGpuContext {
    /// Initialize the GPU compute context for SDF rendering.
    ///
    /// # Errors
    ///
    /// Returns an error string if no compatible GPU adapter is found.
    #[deprecated(
        since = "0.8.0",
        note = "Use `GpuDevice::global()` + `SdfGpuContext::with_device()` to share a single GPU device across contexts"
    )]
    pub fn new() -> Result<Self, crate::gpu::GpuError> {
        // Delegate to the global device for pipeline sharing.
        let gpu = crate::gpu::GpuDevice::global_or_init()?;
        Self::with_device(gpu)
    }

    /// Try to initialize GPU with a timeout. Returns `None` on failure or timeout.
    ///
    /// Spawns GPU init on a background thread so that a hung adapter request
    /// (driver issue, display server contention, etc.) doesn't block the
    /// caller forever.
    #[deprecated(
        since = "0.8.0",
        note = "Use `GpuDevice::global()` + `SdfGpuContext::with_device()` instead — the global device already has built-in timeout"
    )]
    pub fn try_new(timeout: std::time::Duration) -> Option<Self> {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            #[allow(deprecated)]
            let _ = tx.send(Self::new());
        });
        match rx.recv_timeout(timeout) {
            Ok(Ok(ctx)) => Some(ctx),
            Ok(Err(_)) => None,
            Err(_) => None, // timeout — GPU init hung
        }
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
        let device = std::sync::Arc::clone(&gpu.device);
        let queue = std::sync::Arc::clone(&gpu.queue);
        // Lazily compile SDF pipelines (or return cached)
        let pipelines = gpu.pipelines().get_sdf(gpu.device());
        Ok(Self {
            device,
            queue,
            pipelines,
            pool: crate::gpu::BufferPool::new(),
            cached_bind_group: None,
            pending_readback: false,
            cached_pixmap: None,
        })
    }

    /// Access the wgpu device.
    pub(super) fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Access the buffer pool.
    pub(super) fn pool(&self) -> &crate::gpu::BufferPool {
        &self.pool
    }

    /// Mark readback as complete.
    pub(super) fn clear_pending_readback(&mut self) {
        self.pending_readback = false;
    }

    /// Take the cached pixmap (for reuse), replacing it with None.
    pub(super) fn take_cached_pixmap(&mut self) -> Option<Pixmap> {
        self.cached_pixmap.take()
    }

}

// ── GPU renderer ───────────────────────────────────────────────────

/// GPU-accelerated SDF renderer.
///
/// Provides the same output as [`SdfRenderer`](super::SdfRenderer) but
/// runs the ray marching computation on the GPU via a compute shader.
pub struct SdfGpuRenderer;

impl SdfGpuRenderer {
    /// Render the scene to a `Pixmap` using the GPU (blocking).
    ///
    /// This is a convenience wrapper around [`submit`] + [`readback`].
    ///
    /// # Errors
    ///
    /// Returns an error if the pixmap cannot be created or GPU execution fails.
    pub fn render_to_pixmap(
        ctx: &mut SdfGpuContext,
        scene: &SdfScene,
        width: u32,
        height: u32,
        time: f32,
    ) -> Result<Pixmap, PixelCanvasError> {
        Self::submit(ctx, scene, width, height, time)?;
        Self::readback(ctx, width, height)
    }

    /// Submit GPU work for the given scene. Returns immediately after
    /// `queue.submit()` without waiting for GPU completion.
    ///
    /// Call [`readback`] later to retrieve the result. Between `submit`
    /// and `readback` you can do CPU work (terminal draw, event polling)
    /// to overlap with GPU execution.
    ///
    /// # Errors
    ///
    /// Returns an error if buffer allocation fails.
    pub fn submit(
        ctx: &mut SdfGpuContext,
        scene: &SdfScene,
        width: u32,
        height: u32,
        time: f32,
    ) -> Result<(), PixelCanvasError> {
        // Build uniform data
        let uniforms = build_uniforms(scene, width, height, time);

        // Build object array (ensure at least one element for valid buffer)
        let objects = build_objects(scene);
        let objects_data = if objects.is_empty() {
            vec![GpuObject::zeroed()]
        } else {
            objects
        };

        // Build lights array
        let lights = build_lights(scene);
        let lights_data = if lights.is_empty() {
            vec![GpuLight::zeroed()]
        } else {
            lights
        };

        // Reuse or create uniform buffer
        let uniform_bytes = bytemuck::bytes_of(&uniforms);
        let uniform_size = uniform_bytes.len() as u64;
        let (uniform_buf, _) = ctx.pool.get_or_grow(
            crate::gpu::buffer_pool::BufferKey::SdfUniforms,
            uniform_size,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            &ctx.device,
            "sdf-uniforms",
        );
        ctx.queue.write_buffer(uniform_buf, 0, uniform_bytes);

        // Reuse or create objects buffer
        let objects_bytes = bytemuck::cast_slice::<GpuObject, u8>(&objects_data);
        let objects_size = objects_bytes.len() as u64;
        let (objects_buf, objects_reallocated) = ctx.pool.get_or_grow(
            crate::gpu::buffer_pool::BufferKey::SdfObjects,
            objects_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            &ctx.device,
            "sdf-objects",
        );
        ctx.queue.write_buffer(objects_buf, 0, objects_bytes);

        // Reuse or create lights buffer
        let lights_bytes = bytemuck::cast_slice::<GpuLight, u8>(&lights_data);
        let lights_size = lights_bytes.len() as u64;
        let (lights_buf, lights_reallocated) = ctx.pool.get_or_grow(
            crate::gpu::buffer_pool::BufferKey::SdfLights,
            lights_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            &ctx.device,
            "sdf-lights",
        );
        ctx.queue.write_buffer(lights_buf, 0, lights_bytes);

        // Build glyph data for Text3D objects
        let (glyph_meta_bytes, glyph_grids_bytes) = build_glyph_data(scene);
        let glyph_meta_size = glyph_meta_bytes.len() as u64;
        let glyph_grids_size = glyph_grids_bytes.len() as u64;

        let (glyph_meta_buf, glyph_meta_reallocated) = ctx.pool.get_or_grow(
            crate::gpu::buffer_pool::BufferKey::SdfGlyphMeta,
            glyph_meta_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            &ctx.device,
            "sdf-glyph-meta",
        );
        ctx.queue.write_buffer(glyph_meta_buf, 0, &glyph_meta_bytes);

        let (glyph_grids_buf, glyph_grids_reallocated) = ctx.pool.get_or_grow(
            crate::gpu::buffer_pool::BufferKey::SdfGlyphGrids,
            glyph_grids_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            &ctx.device,
            "sdf-glyph-grids",
        );
        ctx.queue.write_buffer(glyph_grids_buf, 0, &glyph_grids_bytes);

        let output_size = (width * height * 4) as u64;

        // Reuse cached output buffer (grow-only)
        let (_output_buf, output_reallocated) = ctx.pool.get_or_grow(
            crate::gpu::buffer_pool::BufferKey::SdfOutput,
            output_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            &ctx.device,
            "sdf-output",
        );
        let (_readback_buf, _) = ctx.pool.get_or_grow(
            crate::gpu::buffer_pool::BufferKey::SdfReadback,
            output_size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            &ctx.device,
            "sdf-readback",
        );

        // Reuse bind group when buffer sizes haven't changed
        let need_new_bind_group = output_reallocated
            || objects_reallocated
            || lights_reallocated
            || glyph_meta_reallocated
            || glyph_grids_reallocated
            || ctx.cached_bind_group.is_none();
        if need_new_bind_group {
            // Re-borrow from pool (needed because previous borrows ended)
            let uniform_buf = ctx.pool.get(crate::gpu::buffer_pool::BufferKey::SdfUniforms).unwrap();
            let objects_buf = ctx.pool.get(crate::gpu::buffer_pool::BufferKey::SdfObjects).unwrap();
            let lights_buf = ctx.pool.get(crate::gpu::buffer_pool::BufferKey::SdfLights).unwrap();
            let output_buf = ctx.pool.get(crate::gpu::buffer_pool::BufferKey::SdfOutput).unwrap();
            let glyph_meta_buf = ctx.pool.get(crate::gpu::buffer_pool::BufferKey::SdfGlyphMeta).unwrap();
            let glyph_grids_buf = ctx.pool.get(crate::gpu::buffer_pool::BufferKey::SdfGlyphGrids).unwrap();

            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sdf-compute-bg"),
                layout: &ctx.pipelines.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: objects_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: lights_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: glyph_meta_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: glyph_grids_buf.as_entire_binding(),
                    },
                ],
            });
            ctx.cached_bind_group = Some((
                bind_group,
                output_size,
                objects_size,
                lights_size,
                glyph_meta_size,
                glyph_grids_size,
            ));
        }

        // Dispatch compute shader
        let output_buf = ctx.pool.get(crate::gpu::buffer_pool::BufferKey::SdfOutput).unwrap();
        let readback_buf = ctx.pool.get(crate::gpu::buffer_pool::BufferKey::SdfReadback).unwrap();

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sdf-compute-encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sdf-compute-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&ctx.pipelines.pipeline);
            pass.set_bind_group(0, &ctx.cached_bind_group.as_ref().unwrap().0, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }

        // Copy output buffer to readback buffer
        encoder.copy_buffer_to_buffer(output_buf, 0, readback_buf, 0, output_size);
        ctx.queue.submit(std::iter::once(encoder.finish()));
        ctx.pending_readback = true;

        Ok(())
    }

    /// Wait for a previously submitted GPU frame and return the result as a `Pixmap`.
    ///
    /// Must be called after [`submit`]. Blocks until the GPU is done,
    /// maps the readback buffer, copies into a `Pixmap`, and unmaps.
    ///
    /// # Errors
    ///
    /// Returns an error if the readback fails or pixmap creation fails.
    pub fn readback(
        ctx: &mut SdfGpuContext,
        width: u32,
        height: u32,
    ) -> Result<Pixmap, PixelCanvasError> {
        // Reuse cached pixmap if dimensions match, otherwise allocate
        let mut pixmap = match ctx.cached_pixmap.take() {
            Some(pm) if pm.width() == width && pm.height() == height => pm,
            _ => Pixmap::new(width, height).ok_or_else(|| {
                PixelCanvasError::PixmapCreation(format!(
                    "failed to create {width}x{height} pixmap"
                ))
            })?,
        };

        let readback_buf = ctx.pool.get(crate::gpu::buffer_pool::BufferKey::SdfReadback).unwrap();

        let readback_slice = readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        readback_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| PixelCanvasError::Rasterization(format!("GPU readback failed: {e}")))?
            .map_err(|e| PixelCanvasError::Rasterization(format!("GPU buffer map failed: {e}")))?;

        {
            let data = readback_slice.get_mapped_range();
            // GPU shader packs as r|(g<<8)|(b<<16)|(255<<24) which is RGBA byte
            // order on little-endian — identical to tiny_skia::Pixmap layout.
            pixmap.data_mut().copy_from_slice(&data);
        }
        readback_buf.unmap();
        ctx.pending_readback = false;

        Ok(pixmap)
    }

    /// Wait for a previously submitted GPU frame and copy the raw RGBA
    /// bytes into the provided buffer, resizing it as needed.
    ///
    /// This avoids the `Pixmap` allocation entirely — useful for the
    /// pipelined path where the caller builds an `ImageData` directly.
    ///
    /// # Errors
    ///
    /// Returns an error if the readback fails.
    pub fn readback_into(
        ctx: &mut SdfGpuContext,
        width: u32,
        height: u32,
        buf: &mut Vec<u8>,
    ) -> Result<(), PixelCanvasError> {
        let expected = (width as usize) * (height as usize) * 4;
        buf.resize(expected, 0);

        let readback_buf = ctx.pool.get(crate::gpu::buffer_pool::BufferKey::SdfReadback).unwrap();

        let readback_slice = readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        readback_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| PixelCanvasError::Rasterization(format!("GPU readback failed: {e}")))?
            .map_err(|e| PixelCanvasError::Rasterization(format!("GPU buffer map failed: {e}")))?;

        {
            let data = readback_slice.get_mapped_range();
            buf.copy_from_slice(&data);
        }
        readback_buf.unmap();
        ctx.pending_readback = false;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::style::Color;
    use crate::sdf::materials::Material;
    use crate::sdf::math::Vec3;
    use crate::sdf::scene::{SdfCamera, SdfLight, SdfObject, SdfShape};

    fn simple_sphere_scene() -> SdfScene {
        SdfScene::new()
            .object(
                SdfObject::new(
                    SdfShape::Sphere { radius: 1.0 },
                    Material::matte(Color::from_rgba8(200, 50, 50, 255)),
                )
                .at(Vec3::new(0.0, 1.0, 0.0)),
            )
            .light(SdfLight::new(Vec3::new(5.0, 10.0, 5.0), Color::WHITE, 1.0))
            .camera(SdfCamera::new(
                Vec3::new(0.0, 3.0, 6.0),
                Vec3::new(0.0, 1.0, 0.0),
                45.0,
            ))
    }

    #[test]
    fn gpu_context_creates_successfully() {
        // This will fail gracefully in CI without a GPU
        if let Ok(ctx) = SdfGpuContext::new() {
            // Just ensure creation doesn't panic and device is usable
            let _ = ctx;
        }
    }

    #[test]
    fn gpu_render_produces_non_black_pixmap() {
        let mut ctx = match SdfGpuContext::new() {
            Ok(c) => c,
            Err(_) => return, // skip if no GPU
        };

        let scene = simple_sphere_scene();
        let pixmap = SdfGpuRenderer::render_to_pixmap(&mut ctx, &scene, 64, 48, 0.0).unwrap();

        let pixels = pixmap.pixels();
        let first = pixels[0];
        let has_variation = pixels.iter().any(|p| *p != first);
        assert!(has_variation, "GPU render produced a uniform image");

        let has_nonblack = pixels
            .iter()
            .any(|p| p.red() > 0 || p.green() > 0 || p.blue() > 0);
        assert!(has_nonblack, "GPU render produced an all-black image");
    }

    #[test]
    fn gpu_buffer_reuse_across_frames() {
        let mut ctx = match SdfGpuContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let scene = simple_sphere_scene();
        let p1 = SdfGpuRenderer::render_to_pixmap(&mut ctx, &scene, 64, 48, 0.0).unwrap();
        let p2 = SdfGpuRenderer::render_to_pixmap(&mut ctx, &scene, 64, 48, 0.0).unwrap();
        assert_eq!(
            p1.data(),
            p2.data(),
            "same scene should produce identical frames"
        );
        // Verify buffers were cached (pool should have entries after first render)
        assert!(ctx.pool.get(crate::gpu::buffer_pool::BufferKey::SdfOutput).is_some());
        assert!(ctx.pool.get(crate::gpu::buffer_pool::BufferKey::SdfReadback).is_some());
    }

    #[cfg(feature = "sdf-text")]
    #[test]
    fn gpu_text3d_renders_non_block() {
        let mut ctx = match SdfGpuContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };

        let font = include_bytes!("../../crates/scry-chart/src/fonts/Inter-Bold.ttf");
        let text_shape = SdfShape::text_3d(font.as_slice(), "AB", 1.0, 0.3)
            .expect("font parse failed");
        let scene = SdfScene::new()
            .object(
                SdfObject::new(text_shape, Material::matte(Color::from_rgba8(200, 200, 200, 255)))
                    .at(Vec3::new(0.0, 1.0, 0.0)),
            )
            .light(SdfLight::new(Vec3::new(5.0, 10.0, 5.0), Color::WHITE, 1.0))
            .camera(SdfCamera::new(
                Vec3::new(0.0, 1.0, 4.0),
                Vec3::new(0.0, 1.0, 0.0),
                45.0,
            ));

        let result = SdfGpuRenderer::render_to_pixmap(&mut ctx, &scene, 64, 48, 0.0);
        let pixmap = result.expect("GPU Text3D render failed");

        let pixels = pixmap.pixels();
        let has_nonblack = pixels
            .iter()
            .any(|p| p.red() > 0 || p.green() > 0 || p.blue() > 0);
        assert!(has_nonblack, "GPU Text3D render produced an all-black image");

        // Check that it's NOT a uniform block — text should have varied silhouette
        let first = pixels[0];
        let has_variation = pixels.iter().any(|p| *p != first);
        assert!(has_variation, "GPU Text3D render produced a uniform image");
    }

    #[cfg(feature = "sdf-text")]
    #[test]
    fn gpu_text3d_matches_cpu() {
        let mut ctx = match SdfGpuContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };

        let font = include_bytes!("../../crates/scry-chart/src/fonts/Inter-Bold.ttf");
        let text_shape = SdfShape::text_3d(font.as_slice(), "A", 1.0, 0.3)
            .expect("font parse failed");
        let scene = SdfScene::new()
            .object(
                SdfObject::new(
                    text_shape,
                    Material::matte(Color::from_rgba8(200, 200, 200, 255)),
                )
                .at(Vec3::new(0.0, 1.0, 0.0)),
            )
            .light(SdfLight::new(Vec3::new(5.0, 10.0, 5.0), Color::WHITE, 1.0))
            .camera(SdfCamera::new(
                Vec3::new(0.0, 1.0, 4.0),
                Vec3::new(0.0, 1.0, 0.0),
                45.0,
            ));

        let gpu_pm =
            SdfGpuRenderer::render_to_pixmap(&mut ctx, &scene, 32, 24, 0.0).unwrap();
        let cpu_pm =
            super::super::SdfRenderer::render_to_pixmap(&scene, 32, 24, 0.0).unwrap();

        // Count how many pixels match roughly (within tolerance)
        let gpu_px = gpu_pm.pixels();
        let cpu_px = cpu_pm.pixels();
        let mut close = 0usize;
        for (g, c) in gpu_px.iter().zip(cpu_px.iter()) {
            let dr = (g.red() as i32 - c.red() as i32).unsigned_abs();
            let dg = (g.green() as i32 - c.green() as i32).unsigned_abs();
            let db = (g.blue() as i32 - c.blue() as i32).unsigned_abs();
            if dr <= 30 && dg <= 30 && db <= 30 {
                close += 1;
            }
        }
        let total = gpu_px.len();
        let pct = close as f64 / total as f64 * 100.0;
        eprintln!("GPU vs CPU text3d similarity: {close}/{total} ({pct:.1}%)");
        assert!(
            pct > 50.0,
            "GPU and CPU Text3D renders differ too much: only {pct:.1}% similar"
        );
    }

    #[test]
    fn gpu_buffer_resize_on_dimension_change() {
        let mut ctx = match SdfGpuContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let scene = simple_sphere_scene();
        let p1 = SdfGpuRenderer::render_to_pixmap(&mut ctx, &scene, 64, 48, 0.0).unwrap();
        assert_eq!(p1.width(), 64);
        // Different dimensions should trigger reallocation but still succeed
        let p2 = SdfGpuRenderer::render_to_pixmap(&mut ctx, &scene, 128, 96, 0.0).unwrap();
        assert_eq!(p2.width(), 128);
        // Pool should have the output buffer after rendering
        assert!(ctx.pool.get(crate::gpu::buffer_pool::BufferKey::SdfOutput).is_some());
    }
}
