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

use crate::scene::style::Color;
use crate::PixelCanvasError;

use super::materials::Material;
use super::math::{self, Vec3};
use super::scene::{SdfScene, SdfShape};

use bytemuck::Zeroable;
use tiny_skia::Pixmap;

// ── GPU-uploadable structs ─────────────────────────────────────────

/// Must match the WGSL `Uniforms` struct exactly (std140 layout).
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct GpuUniforms {
    eye: [f32; 3],
    _pad0: f32,
    cam_right: [f32; 3],
    _pad1: f32,
    cam_up: [f32; 3],
    _pad2: f32,
    cam_forward: [f32; 3],
    fov_scale: f32,

    width: u32,
    height: u32,
    aspect: f32,
    time: f32,

    sky_color: [f32; 4],
    ambient: f32,
    max_bounces: u32,
    num_objects: u32,
    num_lights: u32,
    has_water: u32,
    _pad3: [u32; 3],
}

/// Must match the WGSL `GpuObject` struct.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct GpuObject {
    position: [f32; 3],
    shape_type: u32,
    shape_params: [f32; 4],
    blend_a_params: [f32; 4],
    blend_b_params: [f32; 4],
    blend_b_offset: [f32; 3],
    material_type: u32,
    material_params: [f32; 4],
    material_color: [f32; 4],
    bounding_radius: f32,
    rotation_cos_y: f32,
    rotation_sin_y: f32,
    _pad2: f32,
}

/// Must match the WGSL `GpuLight` struct.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct GpuLight {
    position: [f32; 3],
    intensity: f32,
    color: [f32; 4],
}

// Shape type discriminants (must match WGSL constants)
const SHAPE_SPHERE: u32 = 0;
const SHAPE_BOX: u32 = 1;
const SHAPE_PLANE: u32 = 2;
const SHAPE_TORUS: u32 = 3;
const SHAPE_CYLINDER: u32 = 4;
const SHAPE_SMOOTH_BLEND: u32 = 5;
const SHAPE_CAPSULE: u32 = 6;
const SHAPE_ROUNDED_BOX: u32 = 7;
const SHAPE_CONE: u32 = 8;

// Material type discriminants
const MAT_SOLID: u32 = 0;
const MAT_WATER: u32 = 1;
const MAT_FIRE: u32 = 2;
const MAT_CHECKER: u32 = 3;
const MAT_GLASS: u32 = 4;
const MAT_RAINBOW: u32 = 5;

// ── Scene flattening ───────────────────────────────────────────────

fn vec3_to_arr(v: Vec3) -> [f32; 3] {
    [v.x, v.y, v.z]
}

fn color_to_arr(c: Color) -> [f32; 4] {
    [c.r, c.g, c.b, c.a]
}

/// Flatten an `SdfShape` into type discriminant + params.
/// For `SmoothBlend`, also populates `blend_a`/`blend_b` params and `b_offset`.
fn flatten_shape(shape: &SdfShape) -> (u32, [f32; 4], [f32; 4], [f32; 4], [f32; 3]) {
    match shape {
        SdfShape::Sphere { radius } => (
            SHAPE_SPHERE,
            [*radius, 0.0, 0.0, 0.0],
            [0.0; 4],
            [0.0; 4],
            [0.0; 3],
        ),
        SdfShape::Box { half_extents } => (
            SHAPE_BOX,
            [half_extents.x, half_extents.y, half_extents.z, 0.0],
            [0.0; 4],
            [0.0; 4],
            [0.0; 3],
        ),
        SdfShape::Plane => (SHAPE_PLANE, [0.0; 4], [0.0; 4], [0.0; 4], [0.0; 3]),
        SdfShape::Torus { major, minor } => (
            SHAPE_TORUS,
            [*major, *minor, 0.0, 0.0],
            [0.0; 4],
            [0.0; 4],
            [0.0; 3],
        ),
        SdfShape::Cylinder {
            radius,
            half_height,
        } => (
            SHAPE_CYLINDER,
            [*radius, *half_height, 0.0, 0.0],
            [0.0; 4],
            [0.0; 4],
            [0.0; 3],
        ),
        SdfShape::SmoothBlend { a, b, b_offset, k } => {
            let (a_type, a_params, _, _, _) = flatten_shape(a);
            let (b_type, b_params, _, _, _) = flatten_shape(b);
            (
                SHAPE_SMOOTH_BLEND,
                [*k, a_type as f32, b_type as f32, 0.0],
                a_params,
                b_params,
                vec3_to_arr(*b_offset),
            )
        }
        SdfShape::Capsule {
            radius,
            half_height,
        } => (
            SHAPE_CAPSULE,
            [*radius, *half_height, 0.0, 0.0],
            [0.0; 4],
            [0.0; 4],
            [0.0; 3],
        ),
        SdfShape::RoundedBox {
            half_extents,
            radius,
        } => (
            SHAPE_ROUNDED_BOX,
            [half_extents.x, half_extents.y, half_extents.z, *radius],
            [0.0; 4],
            [0.0; 4],
            [0.0; 3],
        ),
        SdfShape::Cone { radius, height } => (
            SHAPE_CONE,
            [*radius, *height, 0.0, 0.0],
            [0.0; 4],
            [0.0; 4],
            [0.0; 3],
        ),
    }
}

/// Flatten a `Material` into type discriminant + params + color.
fn flatten_material(mat: &Material) -> (u32, [f32; 4], [f32; 4]) {
    match mat {
        Material::Solid {
            color,
            reflectivity,
            specular,
        } => (
            MAT_SOLID,
            [*reflectivity, *specular, 0.0, 0.0],
            color_to_arr(*color),
        ),
        Material::Water {
            tint,
            ior,
            amplitude,
            frequency,
        } => (
            MAT_WATER,
            [*ior, *amplitude, *frequency, 0.0],
            color_to_arr(*tint),
        ),
        Material::Fire {
            intensity,
            noise_scale,
            speed,
        } => (
            MAT_FIRE,
            [*intensity, *noise_scale, *speed, 0.0],
            color_to_arr(Color::WHITE),
        ),
        Material::Checkerboard {
            color_a,
            color_b: _,
            scale,
            reflectivity,
            specular,
        } => (
            MAT_CHECKER,
            [*reflectivity, *specular, *scale, 0.0],
            color_to_arr(*color_a),
        ),
        Material::Glass { tint, ior, opacity, dispersion } => (
            MAT_GLASS,
            [*ior, *opacity, *dispersion, 0.0],
            color_to_arr(*tint),
        ),
        Material::Rainbow { saturation, lightness, hue_offset, specular } => (
            MAT_RAINBOW,
            [*saturation, *lightness, *hue_offset, *specular],
            [0.5, 0.5, 0.5, 1.0], // base color unused (computed from angle)
        ),
    }
}

fn build_uniforms(scene: &SdfScene, width: u32, height: u32, time: f32) -> GpuUniforms {
    let (cam_right, cam_up, cam_fwd) =
        math::look_at(scene.camera.eye, scene.camera.target, Vec3::UP);
    let fov_scale = (scene.camera.fov.to_radians() * 0.5).tan();
    let aspect = width as f32 / height as f32;

    GpuUniforms {
        eye: vec3_to_arr(scene.camera.eye),
        _pad0: 0.0,
        cam_right: vec3_to_arr(cam_right),
        _pad1: 0.0,
        cam_up: vec3_to_arr(cam_up),
        _pad2: 0.0,
        cam_forward: vec3_to_arr(cam_fwd),
        fov_scale,
        width,
        height,
        aspect,
        time,
        sky_color: color_to_arr(scene.sky_color),
        ambient: scene.ambient,
        max_bounces: scene.max_bounces,
        num_objects: scene.objects.len() as u32,
        num_lights: scene.lights.len() as u32,
        has_water: u32::from(scene.has_water || scene.has_glass),
        _pad3: [0; 3],
    }
}

fn build_objects(scene: &SdfScene) -> Vec<GpuObject> {
    scene
        .objects
        .iter()
        .map(|obj| {
            let (shape_type, shape_params, blend_a, blend_b, blend_b_off) =
                flatten_shape(&obj.shape);
            let (material_type, material_params, material_color) = flatten_material(&obj.material);

            // For checkerboard materials, pack color_b into blend_a_params
            // (safe because checkerboard is always on Plane, never SmoothBlend).
            let blend_a = if let Material::Checkerboard { color_b, .. } = &obj.material {
                color_to_arr(*color_b)
            } else {
                blend_a
            };

            let (rot_cos, rot_sin) = obj.rotation_y.unwrap_or((1.0, 0.0));

            GpuObject {
                position: vec3_to_arr(obj.position),
                shape_type,
                shape_params,
                blend_a_params: blend_a,
                blend_b_params: blend_b,
                blend_b_offset: blend_b_off,
                material_type,
                material_params,
                material_color,
                bounding_radius: obj.bounding_radius,
                rotation_cos_y: rot_cos,
                rotation_sin_y: rot_sin,
                _pad2: 0.0,
            }
        })
        .collect()
}

fn build_lights(scene: &SdfScene) -> Vec<GpuLight> {
    scene
        .lights
        .iter()
        .map(|light| GpuLight {
            position: vec3_to_arr(light.position),
            intensity: light.intensity,
            color: color_to_arr(light.color),
        })
        .collect()
}

// ── GPU context ────────────────────────────────────────────────────

/// Reusable GPU context for SDF rendering.
///
/// Creating a context is expensive (~100ms) because it initializes the GPU
/// adapter, device, and compiles the compute shader. Create once and reuse.
pub struct SdfGpuContext {
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Cached output storage buffer: `(buffer, byte_size)`.
    cached_output: Option<(wgpu::Buffer, u64)>,
    /// Cached readback staging buffer: `(buffer, byte_size)`.
    cached_readback: Option<(wgpu::Buffer, u64)>,
    /// Cached uniform buffer.
    cached_uniform: Option<wgpu::Buffer>,
    /// Cached objects storage buffer: `(buffer, byte_size)`.
    cached_objects: Option<(wgpu::Buffer, u64)>,
    /// Cached lights storage buffer: `(buffer, byte_size)`.
    cached_lights: Option<(wgpu::Buffer, u64)>,
    /// Cached bind group: `(bind_group, output_size, objects_size, lights_size)`.
    cached_bind_group: Option<(wgpu::BindGroup, u64, u64, u64)>,
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
    pub fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok_or_else(|| "no suitable GPU adapter found".to_string())?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("sdf-gpu-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| format!("GPU device creation failed: {e}"))?;

        Self::build_pipelines(std::sync::Arc::new(device), std::sync::Arc::new(queue))
    }

    /// Create a context sharing an existing [`GpuDevice`](crate::gpu::GpuDevice).
    ///
    /// This skips the ~100ms adapter/device initialization. Only the
    /// shader compilation and pipeline creation are performed.
    ///
    /// # Errors
    ///
    /// Returns an error string if pipeline creation fails.
    pub fn with_device(gpu: &crate::gpu::GpuDevice) -> Result<Self, String> {
        let device = std::sync::Arc::clone(&gpu.device);
        let queue = std::sync::Arc::clone(&gpu.queue);
        Self::build_pipelines(device, queue)
    }

    fn build_pipelines(
        device: std::sync::Arc<wgpu::Device>,
        queue: std::sync::Arc<wgpu::Queue>,
    ) -> Result<Self, String> {
        let shader_source = include_str!("shaders/sdf_compute.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sdf-compute-shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sdf-compute-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sdf-compute-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sdf-compute-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            cached_output: None,
            cached_readback: None,
            cached_uniform: None,
            cached_objects: None,
            cached_lights: None,
            cached_bind_group: None,
            pending_readback: false,
            cached_pixmap: None,
        })
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
        if ctx.cached_uniform.is_none() {
            ctx.cached_uniform = Some(ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sdf-uniforms"),
                size: uniform_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let uniform_buf = ctx.cached_uniform.as_ref().unwrap();
        ctx.queue.write_buffer(uniform_buf, 0, uniform_bytes);

        // Reuse or create objects buffer
        let objects_bytes = bytemuck::cast_slice::<GpuObject, u8>(&objects_data);
        let objects_size = objects_bytes.len() as u64;
        let mut objects_reallocated = false;
        if ctx
            .cached_objects
            .as_ref()
            .is_none_or(|(_, s)| *s < objects_size)
        {
            ctx.cached_objects = Some((
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("sdf-objects"),
                    size: objects_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                objects_size,
            ));
            objects_reallocated = true;
        }
        let objects_buf = &ctx.cached_objects.as_ref().unwrap().0;
        ctx.queue.write_buffer(objects_buf, 0, objects_bytes);

        // Reuse or create lights buffer
        let lights_bytes = bytemuck::cast_slice::<GpuLight, u8>(&lights_data);
        let lights_size = lights_bytes.len() as u64;
        let mut lights_reallocated = false;
        if ctx
            .cached_lights
            .as_ref()
            .is_none_or(|(_, s)| *s < lights_size)
        {
            ctx.cached_lights = Some((
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("sdf-lights"),
                    size: lights_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                lights_size,
            ));
            lights_reallocated = true;
        }
        let lights_buf = &ctx.cached_lights.as_ref().unwrap().0;
        ctx.queue.write_buffer(lights_buf, 0, lights_bytes);

        let output_size = (width * height * 4) as u64;

        // Reuse cached output buffer if size matches, otherwise allocate
        let mut output_reallocated = false;
        if ctx
            .cached_output
            .as_ref()
            .is_none_or(|(_, s)| *s != output_size)
        {
            ctx.cached_output = Some((
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("sdf-output"),
                    size: output_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                output_size,
            ));
            output_reallocated = true;
        }
        if ctx
            .cached_readback
            .as_ref()
            .is_none_or(|(_, s)| *s != output_size)
        {
            ctx.cached_readback = Some((
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("sdf-readback"),
                    size: output_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                output_size,
            ));
        }
        let output_buf = &ctx.cached_output.as_ref().unwrap().0;
        let readback_buf = &ctx.cached_readback.as_ref().unwrap().0;

        // Reuse bind group when buffer sizes haven't changed
        let need_new_bind_group = output_reallocated
            || objects_reallocated
            || lights_reallocated
            || ctx.cached_bind_group.is_none();
        if need_new_bind_group {
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sdf-compute-bg"),
                layout: &ctx.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.cached_uniform.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: (*objects_buf).as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: (*lights_buf).as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: (*output_buf).as_entire_binding(),
                    },
                ],
            });
            ctx.cached_bind_group = Some((bind_group, output_size, objects_size, lights_size));
        }

        // Dispatch compute shader
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
            pass.set_pipeline(&ctx.pipeline);
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

        let readback_buf = &ctx.cached_readback.as_ref().unwrap().0;

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

        let readback_buf = &ctx.cached_readback.as_ref().unwrap().0;

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
    use crate::sdf::scene::{SdfCamera, SdfLight, SdfObject};

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
        // Verify buffers were cached (not None after first render)
        assert!(ctx.cached_output.is_some());
        assert!(ctx.cached_readback.is_some());
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
        // Cached size should reflect the latest allocation
        assert_eq!(ctx.cached_output.as_ref().unwrap().1, (128 * 96 * 4) as u64);
    }
}
