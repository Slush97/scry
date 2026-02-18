// SPDX-License-Identifier: MIT OR Apache-2.0
//! Reusable GPU context for 2D rasterization.
//!
//! [`WgpuContext2D`] holds the expensive-to-create wgpu device, queue, and
//! compiled render pipelines. Create one context and reuse it across many
//! frames via [`WgpuRasterizer::with_context()`](super::wgpu::WgpuRasterizer).
//!
//! # Feature Gate
//!
//! This module is only available when the `gpu` feature is enabled (default).

use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// Vertex types (bytemuck-compatible for GPU upload)
// ---------------------------------------------------------------------------

/// Per-instance data for shape rendering (circles, rectangles, ellipses).
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(super) struct ShapeInstance {
    /// Center or top-left position in pixels.
    pub pos: [f32; 2],
    /// Shape-type-dependent params: (radius/w, radius/h, `corner_radius`/rotation, 0).
    pub size: [f32; 4],
    /// Fill RGBA \[0,1\].
    pub fill_color: [f32; 4],
    /// Stroke RGBA \[0,1\].
    pub stroke_color: [f32; 4],
    /// (`stroke_width`, `shape_type`) — type: 0=circle, 1=rect, 2=ellipse.
    pub stroke_width_type: [f32; 2],
}

/// Per-vertex data for line rendering.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(super) struct LineVertex {
    /// Screen-space pixel position.
    pub position: [f32; 2],
    /// Perpendicular normal direction.
    pub normal: [f32; 2],
    /// RGBA color.
    pub color: [f32; 4],
    /// Half-width of the line in pixels.
    pub line_width: f32,
    /// Signed distance from line center (-1 or +1).
    pub edge_dist: f32,
}

/// Per-vertex data for tessellated mesh rendering (paths, arcs, polygons).
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(super) struct MeshVertex {
    /// Screen-space pixel position.
    pub position: [f32; 2],
    /// RGBA color.
    pub color: [f32; 4],
}

/// Viewport uniform data for shape and line shaders.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(super) struct Uniforms {
    pub viewport: [f32; 2],
    pub _pad: [f32; 2],
}

/// Gradient stop for GPU upload (matches WGSL layout).
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(super) struct GpuGradientStop {
    pub color: [f32; 4],
    pub position: f32,
    pub _pad1: f32,
    pub _pad2: f32,
    pub _pad3: f32,
}

/// Gradient uniform data (matches WGSL layout).
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(super) struct GradientUniforms {
    pub viewport: [f32; 2],   // offset 0,  size 8
    pub rect_pos: [f32; 2],   // offset 8,  size 8
    pub rect_size: [f32; 2],  // offset 16, size 8
    pub grad_start: [f32; 2], // offset 24, size 8
    pub grad_end: [f32; 2],   // offset 32, size 8
    pub grad_type: f32,       // offset 40, size 4
    pub num_stops: f32,       // offset 44, size 4
    pub _pad: [f32; 2],       // offset 48, size 8
    /// WGSL array<GradientStop,8> requires 16-byte alignment.
    /// Without this, the Rust field sits at offset 56 (not aligned to 16).
    /// This padding pushes stops to offset 64.
    pub _pre_stops_pad: [f32; 2], // offset 56, size 8
    pub stops: [GpuGradientStop; 8], // offset 64, size 256 → total 320
}

// ---------------------------------------------------------------------------
// WgpuContext2D
// ---------------------------------------------------------------------------

/// Reusable GPU context holding the wgpu device, queue, and compiled pipelines.
///
/// Creating a `WgpuContext2D` is expensive (~100ms) because it initializes the
/// GPU adapter, device, and compiles WGSL shaders into render pipelines.
/// Create one context and reuse it across many frames via
/// [`WgpuRasterizer::with_context()`](super::wgpu::WgpuRasterizer).
///
/// # Example
///
/// ```ignore
/// use scry_engine::rasterize::WgpuContext2D;
///
/// let ctx = WgpuContext2D::new()?;
/// // reuse `ctx` across frames...
/// ```
pub struct WgpuContext2D {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) shape_pipeline: wgpu::RenderPipeline,
    pub(crate) line_pipeline: wgpu::RenderPipeline,
    pub(crate) gradient_pipeline: wgpu::RenderPipeline,
    pub(crate) mesh_pipeline: wgpu::RenderPipeline,
    pub(crate) uniform_bgl: wgpu::BindGroupLayout,
    pub(crate) gradient_bgl: wgpu::BindGroupLayout,
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
                label: Some("scry-engine-2d"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| format!("wgpu: device creation failed: {e}"))?;

        // --- Bind group layouts ---
        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniform_bgl_2d"),
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

        let gradient_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gradient_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // --- Pipeline layouts ---
        let shape_line_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shape_line_layout"),
            bind_group_layouts: &[&uniform_bgl],
            push_constant_ranges: &[],
        });

        let gradient_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gradient_layout"),
            bind_group_layouts: &[&gradient_bgl],
            push_constant_ranges: &[],
        });

        let blend_state = wgpu::BlendState::ALPHA_BLENDING;
        let color_target = wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            blend: Some(blend_state),
            write_mask: wgpu::ColorWrites::ALL,
        };

        // --- Shape pipeline ---
        let shape_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shape_shader_2d"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shape.wgsl").into()),
        });

        let shape_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shape_pipeline_2d"),
            layout: Some(&shape_line_layout),
            vertex: wgpu::VertexState {
                module: &shape_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<ShapeInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        // pos: vec2<f32>
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        // size: vec4<f32>
                        wgpu::VertexAttribute {
                            offset: 8,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        // fill_color: vec4<f32>
                        wgpu::VertexAttribute {
                            offset: 24,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        // stroke_color: vec4<f32>
                        wgpu::VertexAttribute {
                            offset: 40,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        // stroke_width_type: vec2<f32>
                        wgpu::VertexAttribute {
                            offset: 56,
                            shader_location: 4,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shape_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(color_target.clone())],
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
            label: Some("line_shader_2d"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/line.wgsl").into()),
        });

        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("line_pipeline_2d"),
            layout: Some(&shape_line_layout),
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
                targets: &[Some(color_target.clone())],
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

        // --- Gradient pipeline ---
        let gradient_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gradient_shader_2d"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gradient.wgsl").into()),
        });

        let gradient_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gradient_pipeline_2d"),
            layout: Some(&gradient_layout),
            vertex: wgpu::VertexState {
                module: &gradient_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &gradient_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(color_target.clone())],
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

        // --- Mesh pipeline (tessellated paths/arcs/polygons) ---
        let mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_shader_2d"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mesh.wgsl").into()),
        });

        let mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh_pipeline_2d"),
            layout: Some(&shape_line_layout),
            vertex: wgpu::VertexState {
                module: &mesh_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<MeshVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        // position: vec2<f32>
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        // color: vec4<f32>
                        wgpu::VertexAttribute {
                            offset: 8,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &mesh_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(color_target)],
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
            shape_pipeline,
            line_pipeline,
            gradient_pipeline,
            mesh_pipeline,
            uniform_bgl,
            gradient_bgl,
        })
    }
}

/// Create per-frame resources: texture, texture view, and uniform bind group.
pub(super) fn create_frame_resources(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::BindGroup) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("render_target_2d"),
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
        label: Some("uniforms_2d"),
        contents: bytemuck::bytes_of(&uniform_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("uniform_bg_2d"),
        layout: bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });

    (texture, texture_view, uniform_bind_group)
}
