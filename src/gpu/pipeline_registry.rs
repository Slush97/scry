// SPDX-License-Identifier: MIT OR Apache-2.0
//! Lazy pipeline registry for GPU shader compilation.
//!
//! [`PipelineRegistry`] holds [`OnceLock`]-backed pipeline sets that are
//! compiled on first access. This ensures each pipeline is compiled exactly
//! once per process, regardless of how many rendering contexts are created.

use std::sync::OnceLock;
use super::pipelines_3d::Pipelines3D;

// ---------------------------------------------------------------------------
// Pipelines2D — 2D rasterization pipelines (shape, line, gradient, mesh)
// ---------------------------------------------------------------------------

/// Compiled GPU pipelines for 2D rasterization.
///
/// Holds the four render pipelines (shape, line, gradient, mesh) and their
/// associated bind group layouts. Created once by [`PipelineRegistry::get_2d()`].
pub struct Pipelines2D {
    /// Instanced shape rendering (circles, rects, ellipses).
    pub shape_pipeline: wgpu::RenderPipeline,
    /// Anti-aliased line rendering.
    pub line_pipeline: wgpu::RenderPipeline,
    /// Full-screen gradient rectangle rendering.
    pub gradient_pipeline: wgpu::RenderPipeline,
    /// Tessellated mesh rendering (paths, arcs, polygons).
    pub mesh_pipeline: wgpu::RenderPipeline,
    /// Viewport uniform bind group layout (shared by shape/line/mesh).
    pub uniform_bgl: wgpu::BindGroupLayout,
    /// Gradient uniform bind group layout.
    pub gradient_bgl: wgpu::BindGroupLayout,
}

/// Per-instance data for shape rendering (circles, rectangles, ellipses).
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ShapeInstance {
    /// Center or top-left position in pixels.
    pub pos: [f32; 2],
    /// Shape-type-dependent params: (radius/w, radius/h, `corner_radius`/rotation, 0).
    pub size: [f32; 4],
    /// Fill RGBA \[0,1\].
    pub fill_color: [f32; 4],
    /// Stroke RGBA \[0,1\].
    pub stroke_color: [f32; 4],
    /// Stroke width in pixels.
    pub stroke_width: f32,
    /// Shape type discriminant: 0=circle, 1=rect, 2=ellipse.
    pub shape_type: u32,
}

/// Per-vertex data for line rendering.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct LineVertex {
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
pub struct MeshVertex {
    /// Screen-space pixel position.
    pub position: [f32; 2],
    /// RGBA color.
    pub color: [f32; 4],
}

/// Viewport uniform data for shape and line shaders.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Uniforms {
    pub viewport: [f32; 2],
    pub _pad: [f32; 2],
}

/// Gradient stop for GPU upload (matches WGSL layout).
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GpuGradientStop {
    pub color: [f32; 4],
    pub position: f32,
    pub _pad1: f32,
    pub _pad2: f32,
    pub _pad3: f32,
}

/// Gradient uniform data (matches WGSL layout).
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GradientUniforms {
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

impl Pipelines2D {
    /// Compile all 2D rasterization pipelines for the given device.
    pub fn compile(device: &wgpu::Device) -> Self {
        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniform_bgl_2d"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    // vec2<f32> viewport size = 8 bytes
                    min_binding_size: std::num::NonZeroU64::new(8),
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
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<GradientUniforms>() as u64
                    ),
                },
                count: None,
            }],
        });

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

        let shape_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shape_shader_2d"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../rasterize/shaders/shape.wgsl").into()),
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
                        wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x2 },
                        wgpu::VertexAttribute { offset: 8, shader_location: 1, format: wgpu::VertexFormat::Float32x4 },
                        wgpu::VertexAttribute { offset: 24, shader_location: 2, format: wgpu::VertexFormat::Float32x4 },
                        wgpu::VertexAttribute { offset: 40, shader_location: 3, format: wgpu::VertexFormat::Float32x4 },
                        wgpu::VertexAttribute { offset: 56, shader_location: 4, format: wgpu::VertexFormat::Float32 },   // stroke_width
                        wgpu::VertexAttribute { offset: 60, shader_location: 5, format: wgpu::VertexFormat::Uint32 },    // shape_type
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shape_shader,
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

        let line_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("line_shader_2d"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../rasterize/shaders/line.wgsl").into()),
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
                targets: &[Some(color_target.clone())],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let gradient_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gradient_shader_2d"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../rasterize/shaders/gradient.wgsl").into()),
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
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_shader_2d"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../rasterize/shaders/mesh.wgsl").into()),
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
                        wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x2 },
                        wgpu::VertexAttribute { offset: 8, shader_location: 1, format: wgpu::VertexFormat::Float32x4 },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &mesh_shader,
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
            shape_pipeline,
            line_pipeline,
            gradient_pipeline,
            mesh_pipeline,
            uniform_bgl,
            gradient_bgl,
        }
    }
}

// ---------------------------------------------------------------------------
// PipelinesSdf — SDF compute pipeline
// ---------------------------------------------------------------------------

/// Compiled GPU pipeline for SDF ray marching.
///
/// Holds the compute pipeline and its bind group layout.
/// Created once by [`PipelineRegistry::get_sdf()`].
pub struct PipelinesSdf {
    /// The SDF compute pipeline.
    pub pipeline: wgpu::ComputePipeline,
    /// Bind group layout for uniforms, objects, lights, output, glyphs.
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl PipelinesSdf {
    /// Path to the persistent pipeline cache file.
    ///
    /// Uses `$XDG_CACHE_HOME/scry/` or `~/.cache/scry/` on Linux/macOS.
    fn cache_path() -> Option<std::path::PathBuf> {
        let dir = if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
            std::path::PathBuf::from(xdg)
        } else {
            let home = std::env::var("HOME").ok()?;
            std::path::PathBuf::from(home).join(".cache")
        };
        Some(dir.join("scry").join("sdf_pipeline.cache"))
    }

    /// Load pipeline cache data from disk, if available.
    ///
    /// # Safety
    ///
    /// `create_pipeline_cache` is unsafe because corrupted cache data could
    /// cause driver-level UB. We mitigate by using `fallback: true` which
    /// creates an empty cache if the data is invalid, and by using atomic
    /// writes when saving.
    #[allow(unsafe_code)]
    fn load_cache(device: &wgpu::Device) -> Option<wgpu::PipelineCache> {
        let path = Self::cache_path()?;
        let data = std::fs::read(&path).ok()?;
        // Integrity check: reject truncated or obviously corrupt cache files.
        // Valid pipeline caches have at least a header; anything under 16 bytes
        // is certainly not a usable cache blob.
        if data.len() < 16 {
            return Some(Self::empty_cache(device));
        }
        // SAFETY: fallback=true ensures an empty cache on invalid data.
        Some(unsafe {
            device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: Some("sdf-pipeline-cache"),
                data: Some(&data),
                fallback: true,
            })
        })
    }

    /// Create an empty pipeline cache (first run or corrupted data).
    #[allow(unsafe_code)]
    fn empty_cache(device: &wgpu::Device) -> wgpu::PipelineCache {
        // SAFETY: data=None creates a fresh empty cache — no driver UB risk.
        unsafe {
            device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: Some("sdf-pipeline-cache"),
                data: None,
                fallback: true,
            })
        }
    }

    /// Save pipeline cache data to disk for next run.
    fn save_cache(cache: &wgpu::PipelineCache) {
        let Some(path) = Self::cache_path() else { return };
        let Some(data) = cache.get_data() else { return };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        // Atomic write: temp file then rename
        let tmp = path.with_extension("tmp");
        if std::fs::write(&tmp, &data).is_ok() {
            let _ = std::fs::rename(&tmp, &path);
        }
    }


    /// Compile the SDF compute pipeline for the given device.
    pub(crate) fn compile(device: &wgpu::Device) -> Self {
        let t0 = std::time::Instant::now();
        crate::scry_debug!("[scry-gpu] Compiling SDF shader module ({} bytes WGSL)...",
            include_str!("../sdf/shaders/sdf_compute.wgsl").len());

        let shader_source = include_str!("../sdf/shaders/sdf_compute.wgsl");
        crate::scry_info!("[scry-gpu] compiling SDF shader ({} lines)…", shader_source.lines().count());
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sdf-compute-shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        crate::scry_info!("[scry-gpu] shader module ready in {:?}", t0.elapsed());

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sdf-compute-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        // Must fit GpuUniforms — 128 bytes (see sdf/gpu_renderer.rs)
                        min_binding_size: std::num::NonZeroU64::new(128),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 4: glyph metadata (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 5: glyph SDF grids (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sdf-compute-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Load persistent pipeline cache (Vulkan only — other backends ignore it)
        let cache_loaded = Self::load_cache(device);
        let cache = cache_loaded.unwrap_or_else(|| Self::empty_cache(device));
        let had_cache = Self::cache_path()
            .as_ref()
            .is_some_and(|p| p.exists());
        crate::scry_debug!("[scry-gpu] Layout created in {:?}, creating compute pipeline (cache={})...",
            t0.elapsed(), if had_cache { "loaded" } else { "cold" });

        crate::scry_info!("[scry-gpu] creating compute pipeline (cache={})…",
            if had_cache { "warm" } else { "cold" });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sdf-compute-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: Some(&cache),
        });
        crate::scry_info!("[scry-gpu] compute pipeline ready in {:?}", t0.elapsed());

        // Persist cache to disk for next run
        Self::save_cache(&cache);

        Self {
            pipeline,
            bind_group_layout,
        }
    }
}

// ---------------------------------------------------------------------------
// PipelineRegistry
// ---------------------------------------------------------------------------

/// Lazy pipeline registry — compiles each pipeline set on first access.
///
/// Owned by [`GpuDevice`](super::GpuDevice) and shared across all rendering
/// contexts. Each pipeline category is independently lazy: the 2D pipelines
/// are only compiled when a `WgpuContext2D` is first created, and the SDF
/// pipeline is only compiled when `SdfGpuContext` is first created.
#[derive(Clone)]
#[allow(clippy::struct_field_names)]
pub struct PipelineRegistry {
    pipelines_2d: std::sync::Arc<OnceLock<Pipelines2D>>,
    pipelines_sdf: std::sync::Arc<OnceLock<PipelinesSdf>>,
    pipelines_3d: std::sync::Arc<OnceLock<Pipelines3D>>,
}

impl PipelineRegistry {
    /// Create a new empty registry.
    pub(super) fn new() -> Self {
        Self {
            pipelines_2d: std::sync::Arc::new(OnceLock::new()),
            pipelines_sdf: std::sync::Arc::new(OnceLock::new()),
            pipelines_3d: std::sync::Arc::new(OnceLock::new()),
        }
    }

    /// Get or compile the 2D rasterization pipelines.
    pub fn get_2d(&self, device: &wgpu::Device) -> &Pipelines2D {
        self.pipelines_2d.get_or_init(|| {
            crate::scry_debug!("[scry-gpu] Compiling 2D pipelines...");
            Pipelines2D::compile(device)
        })
    }

    /// Get or compile the SDF compute pipeline.
    pub fn get_sdf(&self, device: &wgpu::Device) -> &PipelinesSdf {
        self.pipelines_sdf.get_or_init(|| {
            crate::scry_debug!("[scry-gpu] Compiling SDF pipeline...");
            PipelinesSdf::compile(device)
        })
    }

    /// Get or compile the 3D chart pipelines.
    pub fn get_3d(&self, device: &wgpu::Device) -> &Pipelines3D {
        self.pipelines_3d.get_or_init(|| {
            crate::scry_debug!("[scry-gpu] Compiling 3D pipelines...");
            Pipelines3D::compile(device)
        })
    }
}

// ---------------------------------------------------------------------------
// Create per-frame resources (shared helper)
// ---------------------------------------------------------------------------

/// Create per-frame resources: texture, texture view, and uniform bind group.
pub fn create_frame_resources(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::BindGroup) {
    use wgpu::util::DeviceExt;

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
