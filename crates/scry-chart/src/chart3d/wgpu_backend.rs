// SPDX-License-Identifier: MIT OR Apache-2.0
//! GPU-accelerated 3D rasterizer using wgpu.
//!
//! This module provides [`WgpuRasterizer3D`], a GPU-accelerated implementation
//! of the [`Rasterizer3D`](super::Rasterizer3D) trait. It uses wgpu for
//! cross-platform GPU rendering (Vulkan/Metal/DX12) with headless offscreen
//! support — no window or display is required.
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
use scry_engine::style::Color;
use std::cell::RefCell;

// ---------------------------------------------------------------------------
// Vertex types (bytemuck-compatible for GPU upload)
// ---------------------------------------------------------------------------

/// Per-instance data for point rendering.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct PointInstance {
    /// (screen_x, screen_y, radius, depth)
    pos_size: [f32; 4],
    /// (r, g, b, a) in [0, 1]
    color: [f32; 4],
}

/// Per-vertex data for line rendering.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct LineVertex {
    /// Screen-space pixel position.
    position: [f32; 2],
    /// Perpendicular normal direction.
    normal: [f32; 2],
    /// RGBA color.
    color: [f32; 4],
    /// Half-width of the line in pixels.
    line_width: f32,
    /// Signed distance from line center (-1 or +1).
    edge_dist: f32,
}

/// Viewport uniform data.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Uniforms {
    viewport: [f32; 2],
    _pad: [f32; 2],
}

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
// WgpuContext — reusable GPU device + pipeline cache
// ---------------------------------------------------------------------------

/// Reusable GPU context holding the wgpu device, queue, and compiled pipelines.
///
/// Creating a `WgpuContext` is expensive (~100ms) because it initializes the
/// GPU adapter, device, and compiles WGSL shaders into render pipelines.
/// Create one context and reuse it across many frames via
/// [`WgpuRasterizer3D::with_context()`].
///
/// # Example
///
/// ```ignore
/// use scry_chart::chart3d::wgpu_backend::WgpuContext;
/// use scry_chart::chart3d::Chart3D;
///
/// let ctx = WgpuContext::new()?;
/// for _ in 0..60 {
///     let rgba = chart.render_gpu_with_context(&ctx, 1920, 1080)?;
/// }
/// ```
/// Cached per-frame GPU resources, reusable when dimensions match.
struct FrameResourceCache {
    width: u32,
    height: u32,
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    uniform_bind_group: wgpu::BindGroup,
    readback_buffer: wgpu::Buffer,
    readback_padded_row: u32,
}

pub struct WgpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    point_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    frame_cache: RefCell<Option<FrameResourceCache>>,
}

impl WgpuContext {
    /// Initialize the GPU context.
    ///
    /// This performs the expensive one-time setup:
    /// - `Instance` → `Adapter` → `Device` + `Queue`
    /// - Compile point and line WGSL shaders into render pipelines
    ///
    /// # Errors
    ///
    /// Returns an error string if no compatible GPU adapter is found or
    /// device creation fails.
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
                label: Some("scry-chart-3d"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| format!("wgpu: device creation failed: {e}"))?;

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("uniform_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // --- Point pipeline ---
        let point_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("point_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/point.wgsl").into(),
            ),
        });

        let point_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("point_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &point_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<PointInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        // pos_size: vec4<f32>
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        // color: vec4<f32>
                        wgpu::VertexAttribute {
                            offset: 16,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &point_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // --- Line pipeline ---
        let line_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("line_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/line.wgsl").into(),
            ),
        });

        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("line_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &line_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<LineVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        // position: vec2<f32>
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        // normal: vec2<f32>
                        wgpu::VertexAttribute {
                            offset: 8,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        // color: vec4<f32>
                        wgpu::VertexAttribute {
                            offset: 16,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        // line_width: f32
                        wgpu::VertexAttribute {
                            offset: 32,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Float32,
                        },
                        // edge_dist: f32
                        wgpu::VertexAttribute {
                            offset: 36,
                            shader_location: 4,
                            format: wgpu::VertexFormat::Float32,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &line_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            point_pipeline,
            line_pipeline,
            bind_group_layout,
            frame_cache: RefCell::new(None),
        })
    }
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
pub struct WgpuRasterizer3D<'ctx> {
    device: DeviceRef<'ctx>,
    queue: QueueRef<'ctx>,
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    width: u32,
    height: u32,
    background: Color,
    point_pipeline: PipelineRef<'ctx>,
    line_pipeline: PipelineRef<'ctx>,
    uniform_bind_group: wgpu::BindGroup,
    point_instances: Vec<PointInstance>,
    line_vertices: Vec<LineVertex>,
    text_overlays: Vec<TextOverlay>,
    /// Cached readback buffer from previous frame (reused if dimensions match).
    cached_readback: Option<wgpu::Buffer>,
    /// Padded row stride for the cached readback buffer.
    cached_readback_padded_row: u32,
    /// Reference back to context for returning resources to cache.
    ctx_ref: Option<&'ctx WgpuContext>,
}

/// Owned or borrowed reference to a wgpu device.
enum DeviceRef<'a> {
    Owned(wgpu::Device),
    Borrowed(&'a wgpu::Device),
}

impl std::ops::Deref for DeviceRef<'_> {
    type Target = wgpu::Device;
    fn deref(&self) -> &wgpu::Device {
        match self {
            Self::Owned(d) => d,
            Self::Borrowed(d) => d,
        }
    }
}

/// Owned or borrowed reference to a wgpu queue.
enum QueueRef<'a> {
    Owned(wgpu::Queue),
    Borrowed(&'a wgpu::Queue),
}

impl std::ops::Deref for QueueRef<'_> {
    type Target = wgpu::Queue;
    fn deref(&self) -> &wgpu::Queue {
        match self {
            Self::Owned(q) => q,
            Self::Borrowed(q) => q,
        }
    }
}

/// Owned or borrowed reference to a wgpu render pipeline.
enum PipelineRef<'a> {
    Owned(wgpu::RenderPipeline),
    Borrowed(&'a wgpu::RenderPipeline),
}

impl std::ops::Deref for PipelineRef<'_> {
    type Target = wgpu::RenderPipeline;
    fn deref(&self) -> &wgpu::RenderPipeline {
        match self {
            Self::Owned(p) => p,
            Self::Borrowed(p) => p,
        }
    }
}

/// Create per-frame resources (texture, uniform buffer, bind group) on a device.
fn create_frame_resources(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::BindGroup) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("render_target"),
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

    let uniform_data = Uniforms {
        viewport: [width as f32, height as f32],
        _pad: [0.0, 0.0],
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("uniforms"),
        contents: bytemuck::bytes_of(&uniform_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("uniform_bg"),
        layout: bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });

    (texture, texture_view, uniform_bind_group)
}

impl WgpuRasterizer3D<'_> {
    /// Create a new GPU rasterizer with the given dimensions and background.
    ///
    /// Initializes a headless wgpu device (no window/surface) and creates
    /// render pipelines for points and lines. This is a convenience method
    /// for one-shot rendering — for multi-frame rendering, use
    /// [`WgpuContext::new()`] + [`with_context()`](Self::with_context).
    ///
    /// # Errors
    ///
    /// Returns an error string if GPU adapter or device creation fails
    /// (e.g. no compatible GPU found).
    pub fn new(width: u32, height: u32, background: Color) -> Result<Self, String> {
        let ctx = WgpuContext::new()?;

        let (texture, texture_view, uniform_bind_group) =
            create_frame_resources(&ctx.device, &ctx.bind_group_layout, width, height);

        Ok(Self {
            device: DeviceRef::Owned(ctx.device),
            queue: QueueRef::Owned(ctx.queue),
            texture,
            texture_view,
            width,
            height,
            background,
            point_pipeline: PipelineRef::Owned(ctx.point_pipeline),
            line_pipeline: PipelineRef::Owned(ctx.line_pipeline),
            uniform_bind_group,
            point_instances: Vec::new(),
            line_vertices: Vec::new(),
            text_overlays: Vec::new(),
            cached_readback: None,
            cached_readback_padded_row: 0,
            ctx_ref: None,
        })
    }
}

impl<'ctx> WgpuRasterizer3D<'ctx> {
    /// Create a GPU rasterizer that borrows from an existing [`WgpuContext`].
    ///
    /// This skips all device and pipeline initialization, creating only the
    /// per-frame resources (texture, uniform buffer). Use this in render loops
    /// where a `WgpuContext` is created once and reused across many frames.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use scry_chart::chart3d::wgpu_backend::{WgpuContext, WgpuRasterizer3D};
    /// use scry_engine::style::Color;
    ///
    /// let ctx = WgpuContext::new()?;
    /// for frame in 0..60 {
    ///     let rast = WgpuRasterizer3D::with_context(&ctx, 1920, 1080, Color::BLACK);
    ///     // ... draw calls ...
    ///     let rgba = rast.finish();
    /// }
    /// ```
    #[must_use]
    pub fn with_context(
        ctx: &'ctx WgpuContext,
        width: u32,
        height: u32,
        background: Color,
    ) -> Self {
        // Try to reuse cached frame resources if dimensions match
        let cached = ctx.frame_cache.borrow_mut().take();
        let (texture, texture_view, uniform_bind_group, cached_readback, cached_padded_row) =
            if let Some(c) = cached {
                if c.width == width && c.height == height {
                    (c.texture, c.texture_view, c.uniform_bind_group,
                     Some(c.readback_buffer), c.readback_padded_row)
                } else {
                    let (t, tv, bg) =
                        create_frame_resources(&ctx.device, &ctx.bind_group_layout, width, height);
                    (t, tv, bg, None, 0)
                }
            } else {
                let (t, tv, bg) =
                    create_frame_resources(&ctx.device, &ctx.bind_group_layout, width, height);
                (t, tv, bg, None, 0)
            };

        Self {
            device: DeviceRef::Borrowed(&ctx.device),
            queue: QueueRef::Borrowed(&ctx.queue),
            texture,
            texture_view,
            width,
            height,
            background,
            point_pipeline: PipelineRef::Borrowed(&ctx.point_pipeline),
            line_pipeline: PipelineRef::Borrowed(&ctx.line_pipeline),
            uniform_bind_group,
            point_instances: Vec::new(),
            line_vertices: Vec::new(),
            text_overlays: Vec::new(),
            cached_readback,
            cached_readback_padded_row: cached_padded_row,
            ctx_ref: Some(ctx),
        }
    }
}

use wgpu::util::DeviceExt;

impl Rasterizer3D for WgpuRasterizer3D<'_> {
    fn draw_points(&mut self, points: &[ProjectedPoint], colors: &[Color], sizes: &[f32]) {
        self.point_instances.extend(points.iter().map(|pt| {
            let color = colors
                .get(pt.original_index)
                .copied()
                .unwrap_or(Color::WHITE);
            let size = sizes
                .get(pt.original_index)
                .copied()
                .unwrap_or(3.0);
            PointInstance {
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
            self.line_vertices.push(LineVertex {
                position: s, normal, color: color_arr, line_width: half_width, edge_dist: 1.0,
            });
            self.line_vertices.push(LineVertex {
                position: s, normal, color: color_arr, line_width: half_width, edge_dist: -1.0,
            });
            self.line_vertices.push(LineVertex {
                position: e, normal, color: color_arr, line_width: half_width, edge_dist: 1.0,
            });
            self.line_vertices.push(LineVertex {
                position: s, normal, color: color_arr, line_width: half_width, edge_dist: -1.0,
            });
            self.line_vertices.push(LineVertex {
                position: e, normal, color: color_arr, line_width: half_width, edge_dist: -1.0,
            });
            self.line_vertices.push(LineVertex {
                position: e, normal, color: color_arr, line_width: half_width, edge_dist: 1.0,
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

    fn finish(mut self) -> Vec<u8> {
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
                        label: Some("line_vertices"),
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
                        label: Some("point_instances"),
                        contents: bytemuck::cast_slice(&self.point_instances),
                        usage: wgpu::BufferUsages::VERTEX,
                    }),
                self.point_instances.len() as u32,
            ))
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render_encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main_pass"),
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
                render_pass.set_pipeline(&self.line_pipeline);
                render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                render_pass.set_vertex_buffer(0, buf.slice(..));
                render_pass.draw(0..vert_count, 0..1);
            }

            // Draw points on top — single instanced draw call
            if let Some((ref buf, inst_count)) = point_buffer {
                render_pass.set_pipeline(&self.point_pipeline);
                render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                render_pass.set_vertex_buffer(0, buf.slice(..));
                render_pass.draw(0..6, 0..inst_count);
            }
        }

        // --- Readback (reuse cached buffer when dimensions match) ---
        let bytes_per_row_unpadded = self.width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let bytes_per_row_padded = bytes_per_row_unpadded.div_ceil(align) * align;

        let output_buffer = if let Some(buf) = self.cached_readback.take() {
            if self.cached_readback_padded_row == bytes_per_row_padded {
                buf // reuse — same dimensions
            } else {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("readback"),
                    size: u64::from(bytes_per_row_padded) * u64::from(self.height),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            }
        } else {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: u64::from(bytes_per_row_padded) * u64::from(self.height),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

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
            stamp_text_raw(
                &mut rgba,
                self.width,
                self.height,
                &self.text_overlays,
            );
        }

        // --- Return resources to context cache for next frame ---
        let width = self.width;
        let height = self.height;
        let texture = self.texture;
        let texture_view = self.texture_view;
        let uniform_bind_group = self.uniform_bind_group;
        let ctx_ref = self.ctx_ref;

        if let Some(ctx) = ctx_ref {
            *ctx.frame_cache.borrow_mut() = Some(FrameResourceCache {
                width,
                height,
                texture,
                texture_view,
                uniform_bind_group,
                readback_buffer: output_buffer,
                readback_padded_row: bytes_per_row_padded,
            });
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
            let mut glyphs: Vec<(fontdue::Metrics, Vec<u8>)> = Vec::with_capacity(overlay.text.len());
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
        assert!(result.is_ok(), "GPU init should succeed: {:?}", result.err());
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
        assert!(result.is_ok(), "GPU render should succeed: {:?}", result.err());
        let data = result.unwrap();
        assert_eq!(data.len(), 200 * 150 * 4);
    }

    #[test]
    fn chart3d_gpu_vs_cpu_same_dimensions() {
        use super::super::Chart3D;

        let chart = Chart3D::scatter(&[0.0, 1.0, 2.0], &[3.0, 4.0, 5.0], &[6.0, 7.0, 8.0]);

        let cpu = chart.render(100, 80).unwrap();
        let gpu = chart.render_gpu(100, 80).unwrap();

        assert_eq!(cpu.len(), gpu.len(), "CPU and GPU output must have same byte count");
        assert_eq!(cpu.len(), 100 * 80 * 4);

        // Both should have non-zero pixels (actual content)
        let cpu_has_content = cpu.chunks(4).any(|px| px[0] > 20 || px[1] > 20 || px[2] > 20);
        let gpu_has_content = gpu.chunks(4).any(|px| px[0] > 20 || px[1] > 20 || px[2] > 20);
        assert!(cpu_has_content, "CPU output should have visible content");
        assert!(gpu_has_content, "GPU output should have visible content");
    }

    #[test]
    fn wgpu_context_reuse() {
        let ctx = WgpuContext::new().expect("WgpuContext init");

        // Render 3 frames with different data using the same context
        for i in 0..3 {
            let offset = i as f32 * 10.0;
            let mut rast = WgpuRasterizer3D::with_context(&ctx, 120, 90, Color::BLACK);
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
    fn chart3d_render_gpu_with_context() {
        use super::super::Chart3D;

        let ctx = WgpuContext::new().expect("WgpuContext init");
        let chart = Chart3D::scatter(
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 9.0],
        )
        .title("Cached GPU");

        // Render twice with the same context
        for _ in 0..2 {
            let data = chart.render_gpu_with_context(&ctx, 160, 120).unwrap();
            assert_eq!(data.len(), 160 * 120 * 4);
            let has_content = data.chunks(4).any(|px| px[0] > 20 || px[1] > 20 || px[2] > 20);
            assert!(has_content, "cached GPU render should produce visible content");
        }
    }
}
