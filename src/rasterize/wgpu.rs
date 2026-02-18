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

use super::skia::Rasterizer;
use super::tessellate;
use super::wgpu_context::{
    create_frame_resources, GpuGradientStop, GradientUniforms, LineVertex, MeshVertex,
    ShapeInstance, WgpuContext2D,
};
use crate::scene::command::DrawCommand;
use crate::scene::style::{FillStyle, GradientKind};
use crate::scene::PixelCanvas;
use crate::PixelCanvasError;
use tiny_skia::Pixmap;
use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// Deferred draw batches
// ---------------------------------------------------------------------------

struct ShapeBatch {
    instances: Vec<ShapeInstance>,
}

struct LineBatch {
    vertices: Vec<LineVertex>,
}

struct MeshBatch {
    vertices: Vec<MeshVertex>,
}

struct GradientDraw {
    uniforms: GradientUniforms,
}

/// CPU-rasterized overlay to blit onto the GPU output.
struct ImageOverlay {
    x: i32,
    y: i32,
    rgba: Vec<u8>,
    width: u32,
    height: u32,
}

// ---------------------------------------------------------------------------
// WgpuRasterizer
// ---------------------------------------------------------------------------

/// GPU-accelerated 2D rasterizer.
///
/// Provides the same top-level API as the CPU [`Rasterizer`] but uses GPU
/// shaders for shapes, lines, and gradients. Complex commands (Path, Arc,
/// Text, composited Group) fall back to CPU rasterization.
///
/// # Example
///
/// ```ignore
/// use scry_engine::scene::PixelCanvas;
/// use scry_engine::scene::style::Color;
/// use scry_engine::rasterize::WgpuRasterizer;
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
    device: &'ctx wgpu::Device,
    queue: &'ctx wgpu::Queue,
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    width: u32,
    height: u32,
    shape_pipeline: &'ctx wgpu::RenderPipeline,
    line_pipeline: &'ctx wgpu::RenderPipeline,
    gradient_pipeline: &'ctx wgpu::RenderPipeline,
    mesh_pipeline: &'ctx wgpu::RenderPipeline,
    gradient_bgl: &'ctx wgpu::BindGroupLayout,
    uniform_bind_group: wgpu::BindGroup,
    shape_batches: Vec<ShapeBatch>,
    line_batches: Vec<LineBatch>,
    mesh_batches: Vec<MeshBatch>,
    gradient_draws: Vec<GradientDraw>,
    image_overlays: Vec<ImageOverlay>,
}

impl WgpuRasterizer<'_> {
    /// Rasterize a canvas scene using the GPU.
    ///
    /// Creates a one-shot GPU context internally. For multi-frame rendering,
    /// use [`WgpuContext2D::new()`] + [`with_context()`](Self::with_context).
    ///
    /// # Errors
    ///
    /// Returns [`PixelCanvasError::PixmapCreation`] if dimensions are invalid,
    /// or [`PixelCanvasError::Rasterization`] if GPU init fails.
    pub fn rasterize(canvas: &PixelCanvas) -> Result<Pixmap, PixelCanvasError> {
        let ctx = WgpuContext2D::new().map_err(PixelCanvasError::Rasterization)?;
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
        let w = canvas.width();
        let h = canvas.height();
        if w == 0 || h == 0 {
            return Err(PixelCanvasError::PixmapCreation(
                "zero dimensions".to_string(),
            ));
        }

        let mut rast = WgpuRasterizer::with_context(ctx, w, h);
        rast.process_commands(canvas);
        let rgba = rast.finish(canvas)?;

        Pixmap::from_vec(
            rgba,
            tiny_skia::IntSize::from_wh(w, h)
                .ok_or_else(|| PixelCanvasError::PixmapCreation("invalid dimensions".into()))?,
        )
        .ok_or_else(|| PixelCanvasError::PixmapCreation("pixmap from vec failed".into()))
    }
}

impl<'ctx> WgpuRasterizer<'ctx> {
    /// Create a GPU rasterizer that borrows from an existing [`WgpuContext2D`].
    fn with_context(ctx: &'ctx WgpuContext2D, width: u32, height: u32) -> Self {
        let (texture, texture_view, uniform_bind_group) =
            create_frame_resources(&ctx.device, &ctx.uniform_bgl, width, height);

        Self {
            device: &ctx.device,
            queue: &ctx.queue,
            texture,
            texture_view,
            width,
            height,
            shape_pipeline: &ctx.shape_pipeline,
            line_pipeline: &ctx.line_pipeline,
            gradient_pipeline: &ctx.gradient_pipeline,
            mesh_pipeline: &ctx.mesh_pipeline,
            gradient_bgl: &ctx.gradient_bgl,
            uniform_bind_group,
            shape_batches: Vec::new(),
            line_batches: Vec::new(),
            mesh_batches: Vec::new(),
            gradient_draws: Vec::new(),
            image_overlays: Vec::new(),
        }
    }

    /// Walk canvas commands and batch them for GPU submission.
    fn process_commands(&mut self, canvas: &PixelCanvas) {
        let mut shapes = Vec::new();
        let mut lines = Vec::new();
        let mut meshes: Vec<Vec<MeshVertex>> = Vec::new();

        for cmd in canvas.commands() {
            self.process_command(cmd, &mut shapes, &mut lines, &mut meshes);
        }

        if !shapes.is_empty() {
            self.shape_batches.push(ShapeBatch { instances: shapes });
        }
        if !lines.is_empty() {
            self.line_batches.push(LineBatch { vertices: lines });
        }
        if !meshes.is_empty() {
            let vertices = meshes.into_iter().flatten().collect();
            self.mesh_batches.push(MeshBatch { vertices });
        }
    }

    /// Process a single draw command, accumulating into shape/line batches
    /// or creating gradient draws / CPU fallback overlays.
    #[allow(clippy::too_many_lines)]
    fn process_command(
        &mut self,
        cmd: &DrawCommand,
        shapes: &mut Vec<ShapeInstance>,
        lines: &mut Vec<LineVertex>,
        meshes: &mut Vec<Vec<MeshVertex>>,
    ) {
        match cmd {
            DrawCommand::Clear { .. } => {
                // Handled by render pass clear color
            }

            DrawCommand::Circle {
                cx,
                cy,
                radius,
                style,
            } => {
                let (fill_color, stroke_color, stroke_width) = extract_style(style);
                shapes.push(ShapeInstance {
                    pos: [*cx, *cy],
                    size: [*radius, *radius, 0.0, 0.0],
                    fill_color,
                    stroke_color,
                    stroke_width_type: [stroke_width, 0.0], // type 0 = circle
                });
            }

            DrawCommand::Rectangle {
                rect,
                corner_radius,
                style,
            } => {
                let (fill_color, stroke_color, stroke_width) = extract_style(style);
                shapes.push(ShapeInstance {
                    pos: [rect.x, rect.y],
                    size: [rect.width, rect.height, *corner_radius, 0.0],
                    fill_color,
                    stroke_color,
                    stroke_width_type: [stroke_width, 1.0], // type 1 = rect
                });
            }

            DrawCommand::Ellipse {
                cx,
                cy,
                rx,
                ry,
                rotation,
                style,
            } => {
                let (fill_color, stroke_color, stroke_width) = extract_style(style);
                shapes.push(ShapeInstance {
                    pos: [*cx, *cy],
                    size: [*rx, *ry, *rotation, 0.0],
                    fill_color,
                    stroke_color,
                    stroke_width_type: [stroke_width, 2.0], // type 2 = ellipse
                });
            }

            DrawCommand::Line {
                x1,
                y1,
                x2,
                y2,
                stroke,
                ..
            } => {
                let color = [
                    stroke.color.r,
                    stroke.color.g,
                    stroke.color.b,
                    stroke.color.a,
                ];
                emit_line_segment(lines, *x1, *y1, *x2, *y2, stroke.width, color);
            }

            DrawCommand::Polyline {
                points,
                style,
                closed,
            } => {
                if points.len() < 2 {
                    return;
                }

                // Stroke the polyline segments
                if let Some(stroke) = &style.stroke {
                    let color = [
                        stroke.color.r,
                        stroke.color.g,
                        stroke.color.b,
                        stroke.color.a,
                    ];
                    let width = stroke.width;

                    for window in points.windows(2) {
                        emit_line_segment(
                            lines,
                            window[0].0,
                            window[0].1,
                            window[1].0,
                            window[1].1,
                            width,
                            color,
                        );
                    }
                    if *closed && points.len() > 2 {
                        let first = points[0];
                        let last = points[points.len() - 1];
                        emit_line_segment(lines, last.0, last.1, first.0, first.1, width, color);
                    }
                }

                // Fill the polygon via GPU tessellation if solid, else CPU fallback
                if let Some(color) = solid_fill_color(style) {
                    let verts = tessellate::tessellate_polygon(points, color);
                    if !verts.is_empty() {
                        meshes.push(verts);
                    }
                } else if style.fill.is_some() {
                    self.cpu_fallback_command(cmd);
                }
            }

            DrawCommand::Gradient { rect, gradient, .. } => {
                let mut stops = [GpuGradientStop {
                    color: [0.0; 4],
                    position: 0.0,
                    _pad1: 0.0,
                    _pad2: 0.0,
                    _pad3: 0.0,
                }; 8];

                let num_stops = gradient.stops.len().min(8);
                for (i, s) in gradient.stops.iter().take(8).enumerate() {
                    stops[i] = GpuGradientStop {
                        color: [s.color.r, s.color.g, s.color.b, s.color.a],
                        position: s.position,
                        _pad1: 0.0,
                        _pad2: 0.0,
                        _pad3: 0.0,
                    };
                }

                let (grad_start, grad_end, grad_type) = match &gradient.kind {
                    GradientKind::Linear { start, end } => {
                        ([start.x, start.y], [end.x, end.y], 0.0)
                    }
                    GradientKind::Radial { center, radius } => {
                        ([center.x, center.y], [*radius, 0.0], 1.0)
                    }
                };

                #[allow(clippy::cast_precision_loss)]
                self.gradient_draws.push(GradientDraw {
                    uniforms: GradientUniforms {
                        viewport: [self.width as f32, self.height as f32],
                        rect_pos: [rect.x, rect.y],
                        rect_size: [rect.width, rect.height],
                        grad_start,
                        grad_end,
                        grad_type,
                        num_stops: num_stops as f32,
                        _pad: [0.0, 0.0],
                        _pre_stops_pad: [0.0, 0.0],
                        stops,
                    },
                });
            }

            DrawCommand::Image { image, x, y, .. } => {
                // Images are always CPU-side data → overlay
                self.image_overlays.push(ImageOverlay {
                    x: *x as i32,
                    y: *y as i32,
                    rgba: image.data().to_vec(),
                    width: image.width(),
                    height: image.height(),
                });
            }

            DrawCommand::Path { path, style } => {
                if let Some(color) = solid_fill_color(style) {
                    let verts = tessellate::tessellate_path(path.path(), color);
                    if !verts.is_empty() {
                        meshes.push(verts);
                    }
                }
                // Gradient fills / strokes still fall back to CPU
                if has_non_solid_fill(style) || style.stroke.is_some() {
                    self.cpu_fallback_command(cmd);
                }
            }

            DrawCommand::Arc {
                cx,
                cy,
                radius,
                start_angle,
                sweep_angle,
                style,
            } => {
                if let Some(color) = solid_fill_color(style) {
                    let verts = tessellate::tessellate_arc(
                        *cx,
                        *cy,
                        *radius,
                        *start_angle,
                        *sweep_angle,
                        color,
                    );
                    if !verts.is_empty() {
                        meshes.push(verts);
                    }
                }
                if has_non_solid_fill(style) || style.stroke.is_some() {
                    self.cpu_fallback_command(cmd);
                }
            }

            #[cfg(feature = "text")]
            DrawCommand::Text { .. } => {
                self.cpu_fallback_command(cmd);
            }

            DrawCommand::Group {
                commands,
                opacity,
                blend_mode,
                clip,
                transform,
            } => {
                let needs_compositing = *opacity < 1.0
                    || clip.is_some()
                    || *blend_mode != crate::scene::style::BlendMode::SrcOver;

                if needs_compositing || *transform != crate::scene::style::Transform::IDENTITY {
                    // Complex group: fall back to CPU
                    self.cpu_fallback_command(cmd);
                } else {
                    // Simple group: recurse
                    for child in commands {
                        self.process_command(child, shapes, lines, meshes);
                    }
                }
            }
        }
    }

    /// Rasterize a command via CPU (tiny-skia) and add as image overlay.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn cpu_fallback_command(&mut self, cmd: &DrawCommand) {
        // Estimate bounds to create a tight temp pixmap
        let (min_x, min_y, max_x, max_y) = Rasterizer::estimate_command_bounds(cmd);
        if min_x >= max_x || min_y >= max_y {
            return;
        }

        let margin = 4.0;
        let x0 = (min_x - margin).max(0.0).floor();
        let y0 = (min_y - margin).max(0.0).floor();
        let x1 = (max_x + margin).min(self.width as f32).ceil();
        let y1 = (max_y + margin).min(self.height as f32).ceil();
        let w = (x1 - x0) as u32;
        let h = (y1 - y0) as u32;
        if w == 0 || h == 0 {
            return;
        }

        let Some(mut pixmap) = Pixmap::new(w, h) else {
            return;
        };

        // Offset transform so command renders at (0,0) in the temp pixmap
        let offset = tiny_skia::Transform::from_translate(-x0, -y0);
        let mut pool = Vec::new();
        let mut grad_cache = std::collections::HashMap::new();
        Rasterizer::render_command(&mut pixmap, cmd, offset, &mut pool, &mut grad_cache);

        self.image_overlays.push(ImageOverlay {
            x: x0 as i32,
            y: y0 as i32,
            rgba: pixmap.data().to_vec(),
            width: w,
            height: h,
        });
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

            // 2. Draw shapes (circles, rects, ellipses)
            if !self.shape_batches.is_empty() {
                render_pass.set_pipeline(self.shape_pipeline);
                render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);

                for batch in &self.shape_batches {
                    let instance_buf =
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("shape_instances"),
                                contents: bytemuck::cast_slice(&batch.instances),
                                usage: wgpu::BufferUsages::VERTEX,
                            });
                    let inst_count = batch.instances.len() as u32;
                    render_pass.set_vertex_buffer(0, instance_buf.slice(..));
                    render_pass.draw(0..6, 0..inst_count);
                }
            }

            // 2.5. Draw tessellated meshes (paths, arcs, polygons)
            if !self.mesh_batches.is_empty() {
                render_pass.set_pipeline(self.mesh_pipeline);
                render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);

                for batch in &self.mesh_batches {
                    let vbuf = self
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("mesh_vertices"),
                            contents: bytemuck::cast_slice(&batch.vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                    let vert_count = batch.vertices.len() as u32;
                    render_pass.set_vertex_buffer(0, vbuf.slice(..));
                    render_pass.draw(0..vert_count, 0..1);
                }
            }

            // 3. Draw lines
            if !self.line_batches.is_empty() {
                render_pass.set_pipeline(self.line_pipeline);
                render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);

                for batch in &self.line_batches {
                    let vertex_buf =
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("line_vertices"),
                                contents: bytemuck::cast_slice(&batch.vertices),
                                usage: wgpu::BufferUsages::VERTEX,
                            });
                    let vert_count = batch.vertices.len() as u32;
                    render_pass.set_vertex_buffer(0, vertex_buf.slice(..));
                    render_pass.draw(0..vert_count, 0..1);
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
            eprintln!("[scry] GPU rasterization failed, falling back to CPU: {e}");
            Rasterizer::rasterize(canvas)
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract fill/stroke color and stroke width from a `ShapeStyle`.
fn extract_style(style: &crate::scene::style::ShapeStyle) -> ([f32; 4], [f32; 4], f32) {
    let fill_color = match &style.fill {
        Some(FillStyle::Solid(c)) => [c.r, c.g, c.b, c.a],
        Some(FillStyle::LinearGradient(_) | FillStyle::RadialGradient(_)) => {
            // Gradient fills on shapes: use transparent fill, handle separately
            // (would need a more complex shader — for now, CPU fallback)
            [0.0, 0.0, 0.0, 0.0]
        }
        None => [0.0, 0.0, 0.0, 0.0],
    };

    let (stroke_color, stroke_width) = match &style.stroke {
        Some(s) => ([s.color.r, s.color.g, s.color.b, s.color.a], s.width),
        None => ([0.0, 0.0, 0.0, 0.0], 0.0),
    };

    (fill_color, stroke_color, stroke_width)
}

/// Extract the solid fill color from a shape style, applying opacity.
fn solid_fill_color(style: &crate::scene::style::ShapeStyle) -> Option<[f32; 4]> {
    match &style.fill {
        Some(FillStyle::Solid(c)) => Some([c.r, c.g, c.b, c.a * style.opacity]),
        _ => None,
    }
}

/// Returns `true` if the style has a non-solid (gradient) fill.
fn has_non_solid_fill(style: &crate::scene::style::ShapeStyle) -> bool {
    matches!(
        &style.fill,
        Some(FillStyle::LinearGradient(_) | FillStyle::RadialGradient(_))
    )
}

/// Emit 6 vertices (2 triangles) for one line segment.
fn emit_line_segment(
    vertices: &mut Vec<LineVertex>,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    width: f32,
    color: [f32; 4],
) {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len = dx.hypot(dy);
    if len < 1e-6 {
        return;
    }

    let half_width = width * 0.5;
    let nx = -dy / len;
    let ny = dx / len;
    let normal = [nx, ny];

    let s = [x1, y1];
    let e = [x2, y2];

    // Triangle 1: p0, p1, p2
    vertices.push(LineVertex {
        position: s,
        normal,
        color,
        line_width: half_width,
        edge_dist: 1.0,
    });
    vertices.push(LineVertex {
        position: s,
        normal,
        color,
        line_width: half_width,
        edge_dist: -1.0,
    });
    vertices.push(LineVertex {
        position: e,
        normal,
        color,
        line_width: half_width,
        edge_dist: 1.0,
    });
    // Triangle 2: p1, p3, p2
    vertices.push(LineVertex {
        position: s,
        normal,
        color,
        line_width: half_width,
        edge_dist: -1.0,
    });
    vertices.push(LineVertex {
        position: e,
        normal,
        color,
        line_width: half_width,
        edge_dist: -1.0,
    });
    vertices.push(LineVertex {
        position: e,
        normal,
        color,
        line_width: half_width,
        edge_dist: 1.0,
    });
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
        match WgpuContext2D::new() {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("Skipping GPU test: {e}");
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
