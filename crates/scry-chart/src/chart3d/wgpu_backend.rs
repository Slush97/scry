// SPDX-License-Identifier: MIT OR Apache-2.0
//! GPU-accelerated 3D rasterizer using wgpu.
//!
//! This module provides [`WgpuRasterizer3D`], a GPU-accelerated implementation
//! of the [`Rasterizer3D`](super::Rasterizer3D) trait. It uses wgpu for
//! cross-platform GPU rendering (Vulkan/Metal/DX12) with headless offscreen
//! support — no window or display is required.
//!
//! # Architecture
//!
//! The 3D pipelines live in the engine's [`PipelineRegistry`] and are compiled
//! lazily on first use. `WgpuRasterizer3D` borrows `&'static` references to
//! the shared device, queue, and pipeline objects — identical to how the 2D
//! and SDF contexts work.
//!
//! [`PipelineRegistry`]: scry_engine::gpu::PipelineRegistry
//!
//! # Performance
//!
//! Targets ≥10x throughput over [`SkiaRasterizer3D`](super::SkiaRasterizer3D)
//! for large point counts (50K–100K+). The CPU backend is already sufficient
//! for 10K points at 96fps.
//!
//! # Feature Gate
//!
//! This module is only available when the `gpu` feature is enabled:
//!
//! ```toml
//! scry-chart = { version = "0.7", features = ["gpu"] }
//! ```

use super::projection::ProjectedPoint;
use super::Rasterizer3D;
use scry_engine::gpu::pipelines_3d::{LineVertex3D, PointInstance3D, Uniforms3D};
use scry_engine::style::Color;

// ---------------------------------------------------------------------------
// Deferred draw commands
// ---------------------------------------------------------------------------

/// CPU-rasterized text overlay to blit onto the GPU texture.
struct TextOverlay {
    x: f32,
    y: f32,
    text: String,
    color: Color,
    font_size: f32,
}

// ---------------------------------------------------------------------------
// WgpuRasterizer3D
// ---------------------------------------------------------------------------

/// GPU-accelerated 3D rasterizer using wgpu.
///
/// Implements [`Rasterizer3D`] with GPU-based point and line rendering.
/// All `draw_*` calls record batches; [`finish()`](Self::finish) submits a
/// single render pass and reads back the RGBA pixel data.
///
/// Text rendering is done CPU-side via fontdue (same as
/// [`SkiaRasterizer3D`](super::SkiaRasterizer3D)) and blitted to the final
/// output after GPU readback.
///
/// # Example
///
/// ```ignore
/// use scry_chart::chart3d::{Chart3D, wgpu_backend::WgpuRasterizer3D};
/// use scry_engine::style::Color;
///
/// let rast = WgpuRasterizer3D::new(1920, 1080, Color::BLACK)?;
/// let chart = Chart3D::scatter(&x, &y, &z);
/// let rgba = chart.render_with(rast)?;
/// ```
pub struct WgpuRasterizer3D {
    device: &'static wgpu::Device,
    queue: &'static wgpu::Queue,
    pipelines: &'static scry_engine::gpu::pipelines_3d::Pipelines3D,
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    width: u32,
    height: u32,
    background: Color,
    uniform_bind_group: wgpu::BindGroup,
    point_instances: Vec<PointInstance3D>,
    line_vertices: Vec<LineVertex3D>,
    text_overlays: Vec<TextOverlay>,
}

/// Create per-frame resources (texture, uniform buffer, bind group) on a device.
fn create_frame_resources(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::BindGroup) {
    use wgpu::util::DeviceExt;

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("render_target_3d"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let uniform_data = Uniforms3D {
        viewport: [width as f32, height as f32],
        _pad: [0.0, 0.0],
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("uniforms_3d"),
        contents: bytemuck::bytes_of(&uniform_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("uniform_bg_3d"),
        layout: bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });

    (texture, texture_view, uniform_bind_group)
}

impl WgpuRasterizer3D {
    /// Create a new GPU rasterizer with the given dimensions and background.
    ///
    /// Uses the global shared [`GpuDevice`](scry_engine::gpu::GpuDevice)
    /// singleton — the first call may take ~100ms for device initialization,
    /// but subsequent calls reuse the cached device and pipelines.
    ///
    /// # Errors
    ///
    /// Returns an error string if no compatible GPU adapter is found.
    pub fn new(width: u32, height: u32, background: Color) -> Result<Self, String> {
        let gpu = scry_engine::gpu::GpuDevice::global_or_init()?;
        Ok(Self::with_device(gpu, width, height, background))
    }

    /// Create a GPU rasterizer using an existing [`GpuDevice`](scry_engine::gpu::GpuDevice).
    ///
    /// This is the **fast path** for multi-frame rendering. Pass the same
    /// `GpuDevice` reference to skip all device and pipeline initialization.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use scry_chart::chart3d::wgpu_backend::WgpuRasterizer3D;
    /// use scry_engine::gpu::GpuDevice;
    /// use scry_engine::style::Color;
    ///
    /// let gpu = GpuDevice::global().unwrap();
    /// for frame in 0..60 {
    ///     let rast = WgpuRasterizer3D::with_device(gpu, 1920, 1080, Color::BLACK);
    ///     // ... draw calls ...
    ///     let rgba = rast.finish();
    /// }
    /// ```
    #[must_use]
    pub fn with_device(
        gpu: &'static scry_engine::gpu::GpuDevice,
        width: u32,
        height: u32,
        background: Color,
    ) -> Self {
        let device = gpu.device();
        let queue = gpu.queue();
        let pipelines = gpu.pipelines().get_3d(device);

        let (texture, texture_view, uniform_bind_group) =
            create_frame_resources(device, &pipelines.uniform_bgl, width, height);

        Self {
            device,
            queue,
            pipelines,
            texture,
            texture_view,
            width,
            height,
            background,
            uniform_bind_group,
            point_instances: Vec::new(),
            line_vertices: Vec::new(),
            text_overlays: Vec::new(),
        }
    }
}

use wgpu::util::DeviceExt;

impl Rasterizer3D for WgpuRasterizer3D {
    fn draw_points(&mut self, points: &[ProjectedPoint], colors: &[Color], sizes: &[f32]) {
        self.point_instances.extend(points.iter().map(|pt| {
            let color = colors
                .get(pt.original_index)
                .copied()
                .unwrap_or(Color::WHITE);
            let size = sizes.get(pt.original_index).copied().unwrap_or(3.0);
            PointInstance3D {
                pos_size: [pt.screen_x, pt.screen_y, size, pt.depth],
                color: [color.r, color.g, color.b, color.a],
            }
        }));
    }

    fn draw_line_segments(
        &mut self,
        segments: &[(ProjectedPoint, ProjectedPoint)],
        color: Color,
        width: f32,
    ) {
        if segments.is_empty() {
            return;
        }

        let half_width = width * 0.5;
        let color_arr = [color.r, color.g, color.b, color.a];

        self.line_vertices.reserve(segments.len() * 6);

        for (start, end) in segments {
            let dx = end.screen_x - start.screen_x;
            let dy = end.screen_y - start.screen_y;
            let len = dx.hypot(dy);
            if len < 1e-6 {
                continue;
            }

            let nx = -dy / len;
            let ny = dx / len;
            let normal = [nx, ny];

            let s = [start.screen_x, start.screen_y];
            let e = [end.screen_x, end.screen_y];

            // Each segment → 2 triangles (6 vertices): p0-p1-p2 and p1-p3-p2
            self.line_vertices.push(LineVertex3D {
                position: s,
                normal,
                color: color_arr,
                line_width: half_width,
                edge_dist: 1.0,
            });
            self.line_vertices.push(LineVertex3D {
                position: s,
                normal,
                color: color_arr,
                line_width: half_width,
                edge_dist: -1.0,
            });
            self.line_vertices.push(LineVertex3D {
                position: e,
                normal,
                color: color_arr,
                line_width: half_width,
                edge_dist: 1.0,
            });
            self.line_vertices.push(LineVertex3D {
                position: s,
                normal,
                color: color_arr,
                line_width: half_width,
                edge_dist: -1.0,
            });
            self.line_vertices.push(LineVertex3D {
                position: e,
                normal,
                color: color_arr,
                line_width: half_width,
                edge_dist: -1.0,
            });
            self.line_vertices.push(LineVertex3D {
                position: e,
                normal,
                color: color_arr,
                line_width: half_width,
                edge_dist: 1.0,
            });
        }
    }

    fn draw_text(&mut self, x: f32, y: f32, text: &str, color: Color, font_size: f32) {
        self.text_overlays.push(TextOverlay {
            x,
            y,
            text: text.to_string(),
            color,
            font_size,
        });
    }

    fn finish(self) -> Vec<u8> {
        let bg = wgpu::Color {
            r: f64::from(self.background.r),
            g: f64::from(self.background.g),
            b: f64::from(self.background.b),
            a: f64::from(self.background.a),
        };

        // --- Single GPU buffer per primitive type (merged batches) ---
        let line_buffer = if self.line_vertices.is_empty() {
            None
        } else {
            Some((
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("line_vertices_3d"),
                        contents: bytemuck::cast_slice(&self.line_vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    }),
                self.line_vertices.len() as u32,
            ))
        };

        let point_buffer = if self.point_instances.is_empty() {
            None
        } else {
            Some((
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("point_instances_3d"),
                        contents: bytemuck::cast_slice(&self.point_instances),
                        usage: wgpu::BufferUsages::VERTEX,
                    }),
                self.point_instances.len() as u32,
            ))
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render_encoder_3d"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main_pass_3d"),
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

            // Draw lines first (behind points) — single draw call
            if let Some((ref buf, vert_count)) = line_buffer {
                render_pass.set_pipeline(&self.pipelines.line_pipeline);
                render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                render_pass.set_vertex_buffer(0, buf.slice(..));
                render_pass.draw(0..vert_count, 0..1);
            }

            // Draw points on top — single instanced draw call
            if let Some((ref buf, inst_count)) = point_buffer {
                render_pass.set_pipeline(&self.pipelines.point_pipeline);
                render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                render_pass.set_vertex_buffer(0, buf.slice(..));
                render_pass.draw(0..6, 0..inst_count);
            }
        }

        // --- Readback ---
        let bytes_per_row_unpadded = self.width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let bytes_per_row_padded = bytes_per_row_unpadded.div_ceil(align) * align;

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_3d"),
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
            .unwrap_or(());

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

        // --- CPU text overlay (stamp directly on raw bytes) ---
        if !self.text_overlays.is_empty() {
            stamp_text_raw(&mut rgba, self.width, self.height, &self.text_overlays);
        }

        rgba
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }
}

// ---------------------------------------------------------------------------
// Direct text stamping on raw RGBA bytes (Fix 4)
// ---------------------------------------------------------------------------

/// Stamp text overlays directly onto a raw RGBA byte buffer, avoiding the
/// round-trip through `tiny_skia::Pixmap`.
fn stamp_text_raw(rgba: &mut [u8], width: u32, height: u32, overlays: &[TextOverlay]) {
    for overlay in overlays {
        super::with_font(false, |font| {
            // Pre-rasterize glyphs and measure width
            let mut glyphs: Vec<(fontdue::Metrics, Vec<u8>)> =
                Vec::with_capacity(overlay.text.len());
            let mut total_width = 0.0_f32;

            for ch in overlay.text.chars() {
                let (metrics, bitmap) = font.rasterize(ch, overlay.font_size);
                total_width += metrics.advance_width;
                glyphs.push((metrics, bitmap));
            }

            let line_metrics = font.horizontal_line_metrics(overlay.font_size);
            let ascent = line_metrics.map_or(overlay.font_size * 0.8, |m| m.ascent);

            // Center text at (x, y)
            let x_start = overlay.x - total_width / 2.0;
            let baseline_y = overlay.y + ascent * 0.5;

            let r = (overlay.color.r * 255.0) as u8;
            let g = (overlay.color.g * 255.0) as u8;
            let b = (overlay.color.b * 255.0) as u8;
            let text_alpha = overlay.color.a;

            let mut cursor_x = x_start;

            for (metrics, bitmap) in &glyphs {
                let gx_f = cursor_x + metrics.xmin as f32;
                let gy_f = baseline_y - metrics.height as f32 - metrics.ymin as f32;

                for row in 0..metrics.height {
                    for col in 0..metrics.width {
                        let coverage = bitmap[row * metrics.width + col];
                        if coverage == 0 {
                            continue;
                        }

                        let px = (gx_f + col as f32) as i32;
                        let py = (gy_f + row as f32) as i32;

                        if px < 0 || py < 0 || (px as u32) >= width || (py as u32) >= height {
                            continue;
                        }

                        let idx = ((py as u32) * width + px as u32) as usize * 4;
                        let sa = ((coverage as f32 / 255.0) * text_alpha * 255.0) as u32;
                        let inv = 255 - sa;

                        rgba[idx] = ((r as u32 * sa + rgba[idx] as u32 * inv) / 255) as u8;
                        rgba[idx + 1] = ((g as u32 * sa + rgba[idx + 1] as u32 * inv) / 255) as u8;
                        rgba[idx + 2] = ((b as u32 * sa + rgba[idx + 2] as u32 * inv) / 255) as u8;
                        rgba[idx + 3] = (sa + rgba[idx + 3] as u32 * inv / 255).min(255) as u8;
                    }
                }

                cursor_x += metrics.advance_width;
            }
        });
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgpu_rasterizer_init() {
        let result = WgpuRasterizer3D::new(200, 150, Color::BLACK);
        assert!(
            result.is_ok(),
            "GPU init should succeed: {:?}",
            result.err()
        );
        let rast = result.unwrap();
        assert_eq!(rast.width(), 200);
        assert_eq!(rast.height(), 150);
    }

    #[test]
    fn wgpu_rasterizer_draw_points() {
        let mut rast = WgpuRasterizer3D::new(100, 100, Color::BLACK).unwrap();
        rast.draw_points(
            &[ProjectedPoint {
                screen_x: 50.0,
                screen_y: 50.0,
                depth: 0.5,
                original_index: 0,
            }],
            &[Color::RED],
            &[8.0],
        );
        let data = rast.finish();
        assert_eq!(data.len(), 100 * 100 * 4, "RGBA should be w*h*4");
        // Should have at least one non-black pixel
        let has_color = data.chunks(4).any(|px| px[0] > 0 || px[1] > 0 || px[2] > 0);
        assert!(has_color, "GPU rasterizer should produce visible output");
    }

    #[test]
    fn wgpu_rasterizer_draw_lines() {
        let mut rast = WgpuRasterizer3D::new(100, 100, Color::BLACK).unwrap();
        let start = ProjectedPoint {
            screen_x: 10.0,
            screen_y: 50.0,
            depth: 0.5,
            original_index: 0,
        };
        let end = ProjectedPoint {
            screen_x: 90.0,
            screen_y: 50.0,
            depth: 0.5,
            original_index: 0,
        };
        rast.draw_line_segments(&[(start, end)], Color::GREEN, 2.0);
        let data = rast.finish();
        assert_eq!(data.len(), 100 * 100 * 4);
        let has_color = data.chunks(4).any(|px| px[1] > 0);
        assert!(has_color, "line should produce green pixels");
    }

    #[test]
    fn wgpu_rasterizer_finish_dimensions() {
        let rast = WgpuRasterizer3D::new(320, 240, Color::from_rgba8(15, 15, 25, 255)).unwrap();
        let data = rast.finish();
        assert_eq!(
            data.len(),
            320 * 240 * 4,
            "output must be exactly width * height * 4"
        );
    }

    #[test]
    fn chart3d_render_gpu_produces_rgba() {
        use super::super::Chart3D;

        let chart = Chart3D::scatter(
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[6.0, 7.0, 8.0, 9.0, 10.0],
            &[11.0, 12.0, 13.0, 14.0, 15.0],
        )
        .title("GPU Test");

        let result = chart.render_gpu(200, 150);
        assert!(
            result.is_ok(),
            "GPU render should succeed: {:?}",
            result.err()
        );
        let data = result.unwrap();
        assert_eq!(data.len(), 200 * 150 * 4);
    }

    #[test]
    fn chart3d_gpu_vs_cpu_same_dimensions() {
        use super::super::Chart3D;

        let chart = Chart3D::scatter(&[0.0, 1.0, 2.0], &[3.0, 4.0, 5.0], &[6.0, 7.0, 8.0]);

        let cpu = chart.render(100, 80).unwrap();
        let gpu = chart.render_gpu(100, 80).unwrap();

        assert_eq!(
            cpu.len(),
            gpu.len(),
            "CPU and GPU output must have same byte count"
        );
        assert_eq!(cpu.len(), 100 * 80 * 4);

        // Both should have non-zero pixels (actual content)
        let cpu_has_content = cpu
            .chunks(4)
            .any(|px| px[0] > 20 || px[1] > 20 || px[2] > 20);
        let gpu_has_content = gpu
            .chunks(4)
            .any(|px| px[0] > 20 || px[1] > 20 || px[2] > 20);
        assert!(cpu_has_content, "CPU output should have visible content");
        assert!(gpu_has_content, "GPU output should have visible content");
    }

    #[test]
    fn wgpu_device_reuse() {
        let gpu = scry_engine::gpu::GpuDevice::global().expect("GpuDevice init");

        // Render 3 frames with different data using the same device
        for i in 0..3 {
            let offset = i as f32 * 10.0;
            let mut rast = WgpuRasterizer3D::with_device(gpu, 120, 90, Color::BLACK);
            rast.draw_points(
                &[ProjectedPoint {
                    screen_x: 60.0 + offset,
                    screen_y: 45.0,
                    depth: 0.5,
                    original_index: 0,
                }],
                &[Color::RED],
                &[6.0],
            );
            let data = rast.finish();
            assert_eq!(data.len(), 120 * 90 * 4, "frame {i}: wrong RGBA size");
            let has_color = data.chunks(4).any(|px| px[0] > 0 || px[1] > 0 || px[2] > 0);
            assert!(has_color, "frame {i}: should have visible pixels");
        }
    }

    #[test]
    fn chart3d_render_gpu_with_device() {
        use super::super::Chart3D;

        let gpu = scry_engine::gpu::GpuDevice::global().expect("GpuDevice init");
        let chart = Chart3D::scatter(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0])
            .title("Cached GPU");

        // Render twice with the same device
        for _ in 0..2 {
            let data = chart.render_gpu_with_device(gpu, 160, 120).unwrap();
            assert_eq!(data.len(), 160 * 120 * 4);
            let has_content = data
                .chunks(4)
                .any(|px| px[0] > 20 || px[1] > 20 || px[2] > 20);
            assert!(
                has_content,
                "cached GPU render should produce visible content"
            );
        }
    }
}
