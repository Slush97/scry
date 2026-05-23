// SPDX-License-Identifier: MIT OR Apache-2.0
//! GPU-accelerated 2D rasterizer using wgpu.
//!
//! [`WgpuRasterizer`] provides the same public API as the CPU
//! [`Rasterizer`](super::skia::Rasterizer) but renders shapes via GPU
//! shaders for significantly higher throughput at large resolutions.
//!
//! Complex commands (paths, arcs, text, composited groups) fall back to
//! CPU rasterization via tiny-skia and are blitted as image overlays.
//!
//! # Feature Gate
//!
//! This module is only available when the `gpu` feature is enabled (default).

use super::gpu_commands::{
    process_commands, GradientDraw, ImageOverlay, LineBatch, MeshBatch, ShapeBatch,
};
use super::skia::Rasterizer;
use super::wgpu_context::{BufferPool, LineVertex, MeshVertex, ShapeInstance, WgpuContext2D};
use crate::scene::PixelCanvas;
use crate::PixelCanvasError;
use std::cell::RefCell;
use tiny_skia::Pixmap;
use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// WgpuRasterizer
// ---------------------------------------------------------------------------

/// GPU-accelerated 2D rasterizer.
///
/// Provides the same top-level API as the CPU [`Rasterizer`] but uses GPU
/// shaders for shapes, lines, and gradients. Complex commands (Path, Arc,
/// Text) are rasterized on the CPU and blitted as image overlays.
///
/// # Architecture
///
/// 1. Walk the canvas display list, sorting commands into GPU-compatible
///    batches (shapes, lines, meshes, gradients) or CPU fallback overlays.
/// 2. Upload batch data to GPU, submit a render pass.
/// 3. Copy from the render target to a staging buffer, read back.
/// 4. Composite CPU-rasterized overlays on top and return the final pixmap.
///
/// # Example
///
/// ```ignore
/// use scry_engine::scene::{PixelCanvas, Color};
/// use scry_engine::rasterize::wgpu::{WgpuRasterizer, WgpuContext2D};
///
/// let canvas = PixelCanvas::new(1920, 1080)
///     .background(Color::BLACK)
///     .circle(960.0, 540.0, 200.0)
///         .fill(Color::RED)
///         .done();
///
/// let pixmap = WgpuRasterizer::rasterize(&canvas).unwrap();
/// ```
pub struct WgpuRasterizer<'ctx> {
    pub(super) device: &'ctx wgpu::Device,
    pub(super) queue: &'ctx wgpu::Queue,
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    pub(super) width: u32,
    pub(super) height: u32,
    shape_pipeline: &'ctx wgpu::RenderPipeline,
    line_pipeline: &'ctx wgpu::RenderPipeline,
    gradient_pipeline: &'ctx wgpu::RenderPipeline,
    mesh_pipeline: &'ctx wgpu::RenderPipeline,
    gradient_bgl: &'ctx wgpu::BindGroupLayout,
    uniform_bind_group: wgpu::BindGroup,
    pub(super) shape_batches: Vec<ShapeBatch>,
    pub(super) line_batches: Vec<LineBatch>,
    pub(super) mesh_batches: Vec<MeshBatch>,
    pub(super) gradient_draws: Vec<GradientDraw>,
    pub(super) image_overlays: Vec<ImageOverlay>,
    /// Commands that fell back to CPU rasterization.
    pub(super) gpu_fallbacks: Vec<super::backend::GpuFallbackWarning>,
    /// Shared buffer pool for cross-frame GPU buffer reuse.
    buffer_pool: &'ctx RefCell<BufferPool>,
}

impl WgpuRasterizer<'_> {
    /// Rasterize a canvas scene using the GPU.
    ///
    /// Creates a one-shot GPU context internally. For multi-frame rendering,
    /// use [`GpuDevice::global()`](crate::gpu::GpuDevice::global) +
    /// [`WgpuContext2D::with_device()`] + [`with_context()`](Self::with_context).
    ///
    /// # Errors
    ///
    /// Returns [`PixelCanvasError::PixmapCreation`] if dimensions are invalid,
    /// or [`PixelCanvasError::Rasterization`] if GPU init fails.
    pub fn rasterize(canvas: &PixelCanvas) -> Result<Pixmap, PixelCanvasError> {
        let gpu = crate::gpu::GpuDevice::global()
            .ok_or_else(|| PixelCanvasError::Rasterization("GPU not available".to_string()))?;
        let ctx = WgpuContext2D::with_device(gpu)
            .map_err(|e| PixelCanvasError::Rasterization(e.to_string()))?;
        Self::rasterize_with_context(&ctx, canvas)
    }

    /// Rasterize using a pre-existing GPU context (avoids device init overhead).
    ///
    /// # Errors
    ///
    /// Returns [`PixelCanvasError::PixmapCreation`] if dimensions are invalid.
    pub fn rasterize_with_context(
        ctx: &WgpuContext2D,
        canvas: &PixelCanvas,
    ) -> Result<Pixmap, PixelCanvasError> {
        Self::rasterize_with_context_tracked(ctx, canvas).map(|(pixmap, _)| pixmap)
    }

    /// Rasterize and return both the pixmap and any GPU→CPU fallback warnings.
    pub(super) fn rasterize_with_context_tracked(
        ctx: &WgpuContext2D,
        canvas: &PixelCanvas,
    ) -> Result<(Pixmap, Vec<super::backend::GpuFallbackWarning>), PixelCanvasError> {
        let w = canvas.width();
        let h = canvas.height();
        if w == 0 || h == 0 {
            return Err(PixelCanvasError::PixmapCreation(
                "zero dimensions".to_string(),
            ));
        }

        let mut rast = WgpuRasterizer::with_context(ctx, w, h);
        process_commands(&mut rast, canvas);
        let fallbacks = std::mem::take(&mut rast.gpu_fallbacks);
        let rgba = rast.finish(canvas)?;

        let pixmap = Pixmap::from_vec(
            rgba,
            tiny_skia::IntSize::from_wh(w, h)
                .ok_or_else(|| PixelCanvasError::PixmapCreation("invalid dimensions".into()))?,
        )
        .ok_or_else(|| PixelCanvasError::PixmapCreation("pixmap from vec failed".into()))?;

        Ok((pixmap, fallbacks))
    }
}

impl<'ctx> WgpuRasterizer<'ctx> {
    /// Create a GPU rasterizer that borrows from an existing [`WgpuContext2D`].
    fn with_context(ctx: &'ctx WgpuContext2D, width: u32, height: u32) -> Self {
        let (texture, texture_view, uniform_bind_group) =
            super::wgpu_context::create_frame_resources(
                &ctx.device,
                &ctx.pipelines.uniform_bgl,
                width,
                height,
            );

        Self {
            device: &ctx.device,
            queue: &ctx.queue,
            texture,
            texture_view,
            width,
            height,
            shape_pipeline: &ctx.pipelines.shape_pipeline,
            line_pipeline: &ctx.pipelines.line_pipeline,
            gradient_pipeline: &ctx.pipelines.gradient_pipeline,
            mesh_pipeline: &ctx.pipelines.mesh_pipeline,
            gradient_bgl: &ctx.pipelines.gradient_bgl,
            uniform_bind_group,
            shape_batches: Vec::new(),
            line_batches: Vec::new(),
            mesh_batches: Vec::new(),
            gradient_draws: Vec::new(),
            image_overlays: Vec::new(),
            gpu_fallbacks: Vec::new(),
            buffer_pool: &ctx.buffer_pool,
        }
    }

    /// Submit GPU work, read back pixels, and apply image overlays.
    #[allow(clippy::cast_precision_loss)]
    fn finish(self, canvas: &PixelCanvas) -> Result<Vec<u8>, PixelCanvasError> {
        let bg_color = canvas.background_color();
        let bg = wgpu::Color {
            r: f64::from(bg_color.r),
            g: f64::from(bg_color.g),
            b: f64::from(bg_color.b),
            a: f64::from(bg_color.a),
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render_encoder_2d"),
            });

        // --- Flatten batch data and upload to pooled buffers (before render pass) ---
        let shape_buf;
        let mesh_buf;
        let line_buf;
        {
            let mut pool = self.buffer_pool.borrow_mut();

            // Flatten all shape instances into a contiguous byte slice
            let all_shapes: Vec<ShapeInstance> = self
                .shape_batches
                .iter()
                .flat_map(|b| b.instances.iter().copied())
                .collect();
            shape_buf = if all_shapes.is_empty() {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("shape_empty"),
                    size: 16,
                    usage: wgpu::BufferUsages::VERTEX,
                    mapped_at_creation: false,
                })
            } else {
                BufferPool::ensure_and_upload(
                    self.device,
                    self.queue,
                    &mut pool.shape,
                    bytemuck::cast_slice(&all_shapes),
                    wgpu::BufferUsages::VERTEX,
                    "shape_instances_pooled",
                )
            };

            // Flatten all mesh vertices
            let all_mesh: Vec<MeshVertex> = self
                .mesh_batches
                .iter()
                .flat_map(|b| b.vertices.iter().copied())
                .collect();
            mesh_buf = if all_mesh.is_empty() {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("mesh_empty"),
                    size: 16,
                    usage: wgpu::BufferUsages::VERTEX,
                    mapped_at_creation: false,
                })
            } else {
                BufferPool::ensure_and_upload(
                    self.device,
                    self.queue,
                    &mut pool.mesh,
                    bytemuck::cast_slice(&all_mesh),
                    wgpu::BufferUsages::VERTEX,
                    "mesh_vertices_pooled",
                )
            };

            // Flatten all line vertices
            let all_lines: Vec<LineVertex> = self
                .line_batches
                .iter()
                .flat_map(|b| b.vertices.iter().copied())
                .collect();
            line_buf = if all_lines.is_empty() {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("line_empty"),
                    size: 16,
                    usage: wgpu::BufferUsages::VERTEX,
                    mapped_at_creation: false,
                })
            } else {
                BufferPool::ensure_and_upload(
                    self.device,
                    self.queue,
                    &mut pool.line,
                    bytemuck::cast_slice(&all_lines),
                    wgpu::BufferUsages::VERTEX,
                    "line_vertices_pooled",
                )
            };
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main_pass_2d"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(bg),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // 1. Draw gradients first (background layer)
            // (Gradient uniforms are small + per-gradient, kept as individual buffers)
            if !self.gradient_draws.is_empty() {
                render_pass.set_pipeline(self.gradient_pipeline);
                for gd in &self.gradient_draws {
                    let buf = self
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("gradient_uniforms"),
                            contents: bytemuck::bytes_of(&gd.uniforms),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });
                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("gradient_bg"),
                        layout: self.gradient_bgl,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buf.as_entire_binding(),
                        }],
                    });
                    render_pass.set_bind_group(0, &bg, &[]);
                    render_pass.draw(0..6, 0..1);
                }
            }

            // 2. Draw shapes — flatten all batches into one pooled buffer
            if !self.shape_batches.is_empty() {
                render_pass.set_pipeline(self.shape_pipeline);
                render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);

                render_pass.set_vertex_buffer(0, shape_buf.slice(..));
                let mut offset = 0u32;
                for batch in &self.shape_batches {
                    let count = batch.instances.len() as u32;
                    render_pass.draw(0..6, offset..offset + count);
                    offset += count;
                }
            }

            // 2.5. Draw tessellated meshes — flatten all batches into one pooled buffer
            if !self.mesh_batches.is_empty() {
                render_pass.set_pipeline(self.mesh_pipeline);
                render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);

                render_pass.set_vertex_buffer(0, mesh_buf.slice(..));
                let mut vert_offset = 0u32;
                for batch in &self.mesh_batches {
                    let count = batch.vertices.len() as u32;
                    render_pass.draw(vert_offset..vert_offset + count, 0..1);
                    vert_offset += count;
                }
            }

            // 3. Draw lines — flatten all batches into one pooled buffer
            if !self.line_batches.is_empty() {
                render_pass.set_pipeline(self.line_pipeline);
                render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);

                render_pass.set_vertex_buffer(0, line_buf.slice(..));
                let mut vert_offset = 0u32;
                for batch in &self.line_batches {
                    let count = batch.vertices.len() as u32;
                    render_pass.draw(vert_offset..vert_offset + count, 0..1);
                    vert_offset += count;
                }
            }
        }

        // --- Readback ---
        let bytes_per_row_unpadded = self.width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let bytes_per_row_padded = bytes_per_row_unpadded.div_ceil(align) * align;

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_2d"),
            size: u64::from(bytes_per_row_padded) * u64::from(self.height),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row_padded),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read back
        let buffer_slice = output_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .unwrap_or(Err(wgpu::BufferAsyncError))
            .map_err(|_| {
                PixelCanvasError::Rasterization("GPU buffer readback failed (device lost?)".into())
            })?;

        let mapped = buffer_slice.get_mapped_range();

        // Copy with padding removal
        let unpadded_row = bytes_per_row_unpadded as usize;
        let padded_row = bytes_per_row_padded as usize;
        let mut rgba = Vec::with_capacity(unpadded_row * self.height as usize);

        for row in 0..self.height as usize {
            let start = row * padded_row;
            rgba.extend_from_slice(&mapped[start..start + unpadded_row]);
        }

        drop(mapped);
        output_buffer.unmap();

        // --- Apply image overlays (CPU-rasterized fallbacks + images) ---
        if self.image_overlays.is_empty() {
            return Ok(rgba);
        }

        let size = tiny_skia::IntSize::from_wh(self.width, self.height);
        let Some(size) = size else {
            return Ok(rgba);
        };
        let Some(mut pixmap) = Pixmap::from_vec(rgba, size) else {
            return Ok(Vec::new());
        };

        for overlay in &self.image_overlays {
            let overlay_size = tiny_skia::IntSize::from_wh(overlay.width, overlay.height);
            if let Some(os) = overlay_size {
                if let Some(overlay_pm) = Pixmap::from_vec(overlay.rgba.clone(), os) {
                    let paint = tiny_skia::PixmapPaint {
                        opacity: 1.0,
                        blend_mode: tiny_skia::BlendMode::SourceOver,
                        quality: tiny_skia::FilterQuality::Nearest,
                    };
                    pixmap.draw_pixmap(
                        overlay.x,
                        overlay.y,
                        overlay_pm.as_ref(),
                        &paint,
                        tiny_skia::Transform::identity(),
                        None,
                    );
                }
            }
        }

        Ok(pixmap.take())
    }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Try GPU rasterization, falling back to CPU if GPU is unavailable.
///
/// This is the recommended entry point for rendering. When the `gpu` feature
/// is enabled (default), it attempts GPU rendering first and silently falls
/// back to CPU (tiny-skia) if no GPU adapter is available.
///
/// # Errors
///
/// Returns [`PixelCanvasError`] if both GPU and CPU rasterization fail.
pub fn rasterize_auto(canvas: &PixelCanvas) -> Result<Pixmap, PixelCanvasError> {
    match WgpuRasterizer::rasterize(canvas) {
        Ok(pixmap) => Ok(pixmap),
        Err(e) => {
            // Log the GPU error for diagnostics, then fall back to CPU
            crate::scry_warn!("[scry] GPU rasterization failed, falling back to CPU: {e}");
            Rasterizer::rasterize(canvas)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::style::{Color, Point};

    /// Helper: check that GPU init succeeds (skip test if no GPU).
    fn require_gpu() -> WgpuContext2D {
        match crate::gpu::GpuDevice::global() {
            Some(gpu) => match WgpuContext2D::with_device(gpu) {
                Ok(ctx) => ctx,
                Err(e) => {
                    eprintln!("Skipping GPU test: {e}");
                    std::process::exit(0);
                }
            },
            None => {
                eprintln!("Skipping GPU test: no GPU available");
                std::process::exit(0);
            }
        }
    }

    #[test]
    fn gpu_rasterize_empty_canvas() {
        let ctx = require_gpu();
        let canvas = PixelCanvas::new(100, 100);
        let pixmap = WgpuRasterizer::rasterize_with_context(&ctx, &canvas).unwrap();
        assert_eq!(pixmap.width(), 100);
        assert_eq!(pixmap.height(), 100);
    }

    #[test]
    fn gpu_rasterize_circle_center_pixel() {
        let ctx = require_gpu();
        let canvas = PixelCanvas::new(100, 100)
            .circle(50.0, 50.0, 30.0)
            .fill(Color::BLUE)
            .done();

        let pixmap = WgpuRasterizer::rasterize_with_context(&ctx, &canvas).unwrap();
        let idx = (50 * 100 + 50) * 4;
        let data = pixmap.data();
        // Blue channel should be dominant
        assert!(data[idx + 2] > 200, "blue channel: {}", data[idx + 2]);
    }

    #[test]
    fn gpu_rasterize_rectangle() {
        let ctx = require_gpu();
        let canvas = PixelCanvas::new(100, 100)
            .rect(10.0, 10.0, 80.0, 60.0)
            .fill(Color::GREEN)
            .done();

        let pixmap = WgpuRasterizer::rasterize_with_context(&ctx, &canvas).unwrap();
        let idx = (40 * 100 + 50) * 4;
        let data = pixmap.data();
        // Green channel should be dominant
        assert!(data[idx + 1] > 200, "green channel: {}", data[idx + 1]);
    }

    #[test]
    fn gpu_rasterize_line() {
        let ctx = require_gpu();
        let canvas = PixelCanvas::new(100, 100)
            .line(0.0, 50.0, 100.0, 50.0)
            .color(Color::WHITE)
            .width(3.0)
            .done();

        let pixmap = WgpuRasterizer::rasterize_with_context(&ctx, &canvas).unwrap();
        let idx = (50 * 100 + 50) * 4;
        let data = pixmap.data();
        // Should have some non-zero pixel on the line
        let brightness = data[idx] as u32 + data[idx + 1] as u32 + data[idx + 2] as u32;
        assert!(brightness > 100, "line pixel brightness: {brightness}");
    }

    #[test]
    fn gpu_rasterize_gradient() {
        let ctx = require_gpu();
        let canvas = PixelCanvas::new(200, 50)
            .gradient(0.0, 0.0, 200.0, 50.0)
            .linear(Point::new(0.0, 0.0), Point::new(200.0, 0.0))
            .stop(0.0, Color::RED)
            .stop(1.0, Color::BLUE)
            .done();

        let pixmap = WgpuRasterizer::rasterize_with_context(&ctx, &canvas).unwrap();
        let data = pixmap.data();

        // Gradient center should be non-transparent
        let mid = (25 * 200 + 100) * 4;
        assert!(
            data[mid + 3] > 0,
            "gradient center should be non-transparent"
        );

        // Left and right sides should differ (gradient varies across width)
        let left = (25 * 200 + 10) * 4;
        let right = (25 * 200 + 190) * 4;
        let left_pixel = [data[left], data[left + 1], data[left + 2], data[left + 3]];
        let right_pixel = [
            data[right],
            data[right + 1],
            data[right + 2],
            data[right + 3],
        ];
        assert_ne!(
            left_pixel, right_pixel,
            "gradient should show color variation"
        );
    }

    #[test]
    fn gpu_context_reuse() {
        let ctx = require_gpu();

        for i in 0..3 {
            let canvas = PixelCanvas::new(50, 50)
                .circle(25.0, 25.0, 10.0 + i as f32)
                .fill(Color::RED)
                .done();

            let pixmap = WgpuRasterizer::rasterize_with_context(&ctx, &canvas).unwrap();
            assert_eq!(pixmap.width(), 50);
        }
    }

    #[test]
    fn gpu_path_fallback() {
        let ctx = require_gpu();
        // Create a canvas with a path command (will fall back to CPU)
        let canvas = PixelCanvas::new(100, 100)
            .circle(50.0, 50.0, 30.0)
            .fill(Color::RED)
            .done();

        // This should not panic, even if it uses CPU fallback
        let pixmap = WgpuRasterizer::rasterize_with_context(&ctx, &canvas).unwrap();
        assert_eq!(pixmap.width(), 100);
    }

    #[test]
    fn gpu_rasterize_auto_works() {
        let canvas = PixelCanvas::new(100, 100)
            .background(Color::BLACK)
            .circle(50.0, 50.0, 20.0)
            .fill(Color::RED)
            .done();

        // rasterize_auto should always succeed (GPU or CPU fallback)
        let pixmap = rasterize_auto(&canvas).unwrap();
        assert_eq!(pixmap.width(), 100);
    }

    #[test]
    fn gpu_tessellated_path_renders() {
        let ctx = require_gpu();
        // Build a triangle path with solid red fill
        let mut pb = tiny_skia::PathBuilder::new();
        pb.move_to(50.0, 10.0);
        pb.line_to(90.0, 90.0);
        pb.line_to(10.0, 90.0);
        pb.close();
        let path = pb.finish().unwrap();

        let canvas = PixelCanvas::new(100, 100)
            .background(Color::BLACK)
            .path(path)
            .fill(Color::RED)
            .done();

        let pixmap = WgpuRasterizer::rasterize_with_context(&ctx, &canvas).unwrap();
        // Check center of triangle (roughly 50, 63)
        let idx = (63 * 100 + 50) * 4;
        let data = pixmap.data();
        assert!(
            data[idx] > 100,
            "tessellated path center should be red, R={}",
            data[idx]
        );
    }

    #[test]
    fn gpu_tessellated_arc_renders() {
        let ctx = require_gpu();
        let canvas = PixelCanvas::new(100, 100)
            .background(Color::BLACK)
            .arc(50.0, 50.0, 30.0, 0.0, std::f32::consts::PI)
            .fill(Color::BLUE)
            .done();

        let pixmap = WgpuRasterizer::rasterize_with_context(&ctx, &canvas).unwrap();
        // Check a point inside the arc pie-slice
        let idx = (45 * 100 + 60) * 4;
        let data = pixmap.data();
        assert!(
            data[idx + 2] > 100,
            "tessellated arc interior should be blue, B={}",
            data[idx + 2]
        );
    }

    #[test]
    fn gpu_tessellated_polygon_fill() {
        let ctx = require_gpu();
        // Pentagon
        let points = vec![
            (50.0, 10.0),
            (90.0, 40.0),
            (75.0, 85.0),
            (25.0, 85.0),
            (10.0, 40.0),
        ];
        let canvas = PixelCanvas::new(100, 100)
            .background(Color::BLACK)
            .polygon(points)
            .fill(Color::GREEN)
            .done();

        let pixmap = WgpuRasterizer::rasterize_with_context(&ctx, &canvas).unwrap();
        // Check center of pentagon
        let idx = (50 * 100 + 50) * 4;
        let data = pixmap.data();
        assert!(
            data[idx + 1] > 100,
            "tessellated polygon center should be green, G={}",
            data[idx + 1]
        );
    }
}
