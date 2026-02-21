// SPDX-License-Identifier: MIT OR Apache-2.0
//! Compiled GPU pipelines for 3D chart rendering (points + lines).
//!
//! [`Pipelines3D`] is compiled lazily by [`PipelineRegistry::get_3d()`] on
//! first access — identical lifetime model to [`Pipelines2D`] and
//! [`PipelinesSdf`].

/// Compiled GPU pipelines for 3D chart rasterization.
///
/// Holds the point (instanced circle) and line (anti-aliased segment) render
/// pipelines plus their shared uniform bind group layout.
/// Created once by [`PipelineRegistry::get_3d()`](super::PipelineRegistry::get_3d).
pub struct Pipelines3D {
    /// Instanced point (circle) rendering pipeline.
    pub point_pipeline: wgpu::RenderPipeline,
    /// Anti-aliased line segment rendering pipeline.
    pub line_pipeline: wgpu::RenderPipeline,
    /// Viewport uniform bind group layout (shared by both pipelines).
    pub uniform_bgl: wgpu::BindGroupLayout,
}

/// Per-instance data for 3D point rendering.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct PointInstance3D {
    /// (`screen_x`, `screen_y`, radius, depth)
    pub pos_size: [f32; 4],
    /// (r, g, b, a) in \[0, 1\]
    pub color: [f32; 4],
}

/// Per-vertex data for 3D line rendering.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct LineVertex3D {
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

/// Viewport uniform data for 3D shaders.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Uniforms3D {
    /// Viewport width and height in pixels.
    pub viewport: [f32; 2],
    /// Padding to align to 16 bytes.
    pub pad: [f32; 2],
}

impl Pipelines3D {
    /// Compile all 3D chart pipelines for the given device.
    pub(crate) fn compile(device: &wgpu::Device) -> Self {
        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniform_bgl_3d"),
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
            label: Some("pipeline_layout_3d"),
            bind_group_layouts: &[&uniform_bgl],
            push_constant_ranges: &[],
        });

        let color_target = wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        };

        // --- Point pipeline ---
        let point_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("point_shader_3d"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders_3d/point.wgsl").into()),
        });

        let point_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("point_pipeline_3d"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &point_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<PointInstance3D>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x4 },
                        wgpu::VertexAttribute { offset: 16, shader_location: 1, format: wgpu::VertexFormat::Float32x4 },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &point_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(color_target.clone())],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // --- Line pipeline ---
        let line_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("line_shader_3d"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders_3d/line.wgsl").into()),
        });

        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("line_pipeline_3d"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &line_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<LineVertex3D>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x2 },
                        wgpu::VertexAttribute { offset: 8, shader_location: 1, format: wgpu::VertexFormat::Float32x2 },
                        wgpu::VertexAttribute { offset: 16, shader_location: 2, format: wgpu::VertexFormat::Float32x4 },
                        wgpu::VertexAttribute { offset: 32, shader_location: 3, format: wgpu::VertexFormat::Float32 },
                        wgpu::VertexAttribute { offset: 36, shader_location: 4, format: wgpu::VertexFormat::Float32 },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &line_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(color_target)],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            point_pipeline,
            line_pipeline,
            uniform_bgl,
        }
    }
}
