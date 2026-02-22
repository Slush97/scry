// SPDX-License-Identifier: MIT OR Apache-2.0
//! GPU compositor — assembles the terminal frame from components.
//!
//! Owns the wgpu surface, device, and queue. Composes:
//! 1. Background clear
//! 2. Cell background quads (instanced)
//! 3. Text glyphs (via glyphon)
//! 4. Cursor overlay
//!
//! Each frame, only dirty lines are re-shaped; the atlas caches glyphs.

use std::sync::Arc;
use std::time::Instant;

use winit::window::Window;

use crate::config::TerminalConfig;
use crate::error::TerminalError;
use crate::grid::{CellColor, CursorStyle, TerminalGrid};
use crate::selection::Selection;
use crate::text::TextEngine;

/// Per-instance data for a cell background quad.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CellBgInstance {
    /// Top-left position in pixels.
    pos: [f32; 2],
    /// Cell size in pixels.
    size: [f32; 2],
    /// RGBA color (0.0–1.0).
    color: [f32; 4],
}

/// Uniform data for the cell background shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BgUniforms {
    /// Screen size in pixels.
    screen_size: [f32; 2],
}

/// The GPU compositor.
pub struct Compositor {
    /// wgpu surface for presenting frames.
    surface: wgpu::Surface<'static>,
    /// GPU device.
    device: Arc<wgpu::Device>,
    /// GPU command queue.
    queue: Arc<wgpu::Queue>,
    /// Surface configuration.
    surface_config: wgpu::SurfaceConfiguration,

    /// Text rendering engine.
    text_engine: TextEngine,

    /// Cell background render pipeline.
    bg_pipeline: wgpu::RenderPipeline,
    /// Cell background uniform buffer.
    bg_uniform_buffer: wgpu::Buffer,
    /// Cell background uniform bind group.
    bg_bind_group: wgpu::BindGroup,
    /// Cell background instance buffer.
    bg_instance_buffer: wgpu::Buffer,
    /// Max instances the buffer can hold.
    bg_instance_capacity: u32,

    /// Terminal config (for colors).
    config: TerminalConfig,

    /// Screen dimensions.
    width: u32,
    height: u32,

    // ── Cursor blink ─────────────────────────────────────────────
    /// Whether the cursor is currently visible (blink state).
    cursor_blink_visible: bool,
    /// Last time the cursor blink state changed.
    last_blink: Instant,

    // ── Visual bell ──────────────────────────────────────────────
    /// When the visual bell was triggered (None = no bell active).
    bell_start: Option<Instant>,

    /// Content padding in pixels (applied on all four sides).
    padding: f32,

    /// Shared scry-engine GPU device (bridges terminal GPU to engine pipelines).
    engine_gpu: &'static scry_engine::gpu::GpuDevice,

    // ── Graphics overlay (scry-engine) ──────────────────────────
    /// Overlay render pipeline (fullscreen triangle with texture sampling).
    overlay_pipeline: wgpu::RenderPipeline,
    /// Overlay bind group layout.
    overlay_bgl: wgpu::BindGroupLayout,
    /// Overlay sampler (linear filtering).
    overlay_sampler: wgpu::Sampler,
    /// Overlay texture (Rgba8UnormSrgb, matches scry-engine output).
    overlay_texture: Option<wgpu::Texture>,
    /// Overlay bind group (texture + sampler).
    overlay_bind_group: Option<wgpu::BindGroup>,
    /// The current overlay scene (rasterized by scry-engine's CPU backend).
    overlay_scene: Option<scry_engine::scene::PixelCanvas>,
    /// Whether the overlay needs re-rasterization.
    overlay_dirty: bool,
    /// Cached content hash of the last rasterized scene.
    overlay_hash: u64,
}

impl Compositor {
    /// Create a new compositor for the given window.
    ///
    /// # Errors
    ///
    /// Returns an error string if GPU initialization fails.
    pub fn new(window: Arc<Window>, config: &TerminalConfig) -> Result<Self, TerminalError> {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        // Initialize wgpu
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance
            .create_surface(window)
            .map_err(|e| TerminalError::Compositor(format!("failed to create GPU surface: {e}")))?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| TerminalError::Gpu("no suitable GPU adapter found".to_string()))?;

        let (raw_device, raw_queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("scry-terminal"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| TerminalError::Gpu(format!("failed to create GPU device: {e}")))?;

        let device = Arc::new(raw_device);
        let queue = Arc::new(raw_queue);

        // Bridge to scry-engine: wrap our device/queue so engine pipelines
        // (shapes, gradients, SDF) can render into the terminal viewport.
        let engine_gpu = Box::leak(Box::new(scry_engine::gpu::GpuDevice::from_existing(
            Arc::clone(&device),
            Arc::clone(&queue),
        )));

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo, // Vsync
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // Create text engine
        let text_engine = TextEngine::new(&device, &queue, surface_format, config);

        // Create cell background pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cell_bg"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cell_bg.wgsl").into()),
        });

        let bg_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bg_uniforms"),
            size: std::mem::size_of::<BgUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bg_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bg_bind_group_layout"),
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

        let bg_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_bind_group"),
            layout: &bg_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: bg_uniform_buffer.as_entire_binding(),
            }],
        });

        let bg_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bg_pipeline_layout"),
            bind_group_layouts: &[&bg_bind_group_layout],
            push_constant_ranges: &[],
        });

        let bg_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bg_pipeline"),
            layout: Some(&bg_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<CellBgInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 8,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 16,
                            shader_location: 2,
                        },
                    ],
                }],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        // Instance buffer for cell backgrounds
        let initial_capacity = 4096u32;
        let bg_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bg_instances"),
            size: (initial_capacity as usize * std::mem::size_of::<CellBgInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create overlay pipeline (fullscreen triangle with texture)
        let overlay_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/overlay.wgsl").into()),
        });

        let overlay_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("overlay_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let overlay_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("overlay_pipeline_layout"),
                bind_group_layouts: &[&overlay_bgl],
                push_constant_ranges: &[],
            });

        let overlay_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay_pipeline"),
            layout: Some(&overlay_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &overlay_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &overlay_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        let overlay_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("overlay_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Ok(Self {
            surface,
            device,
            queue,
            surface_config,
            text_engine,
            bg_pipeline,
            bg_uniform_buffer,
            bg_bind_group,
            bg_instance_buffer,
            bg_instance_capacity: initial_capacity,
            config: config.clone(),
            width,
            height,
            cursor_blink_visible: true,
            last_blink: Instant::now(),
            bell_start: None,
            padding: config.window.padding,
            engine_gpu,
            overlay_pipeline,
            overlay_bgl,
            overlay_sampler,
            overlay_texture: None,
            overlay_bind_group: None,
            overlay_scene: None,
            overlay_dirty: false,
            overlay_hash: 0,
        })
    }

    /// Cell width in pixels.
    pub fn cell_width(&self) -> f32 {
        self.text_engine.cell_width()
    }

    /// Cell height in pixels.
    pub fn cell_height(&self) -> f32 {
        self.text_engine.cell_height()
    }

    /// Content padding in pixels.
    pub fn padding(&self) -> f32 {
        self.padding
    }

    /// Get the bridged scry-engine GPU device.
    ///
    /// This shares the same underlying wgpu device/queue as the compositor,
    /// enabling scry-engine's rendering pipelines (shapes, gradients, SDF)
    /// to render into the terminal viewport.
    pub fn engine_gpu(&self) -> &'static scry_engine::gpu::GpuDevice {
        self.engine_gpu
    }

    /// Get a clone of the `Arc<Device>`.
    pub fn device_arc(&self) -> Arc<wgpu::Device> {
        Arc::clone(&self.device)
    }

    /// Get a clone of the `Arc<Queue>`.
    pub fn queue_arc(&self) -> Arc<wgpu::Queue> {
        Arc::clone(&self.queue)
    }

    /// Current font size in pixels.
    pub fn font_size(&self) -> f32 {
        self.text_engine.font_size()
    }

    /// Change the font size and return the new `(cell_width, cell_height)`.
    pub fn set_font_size(&mut self, size: f32) -> (f32, f32) {
        self.text_engine.set_font_size(size);
        (self.text_engine.cell_width(), self.text_engine.cell_height())
    }

    /// Resize the surface.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.width = width;
        self.height = height;
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
        self.overlay_dirty = true;
    }

    /// Trigger the visual bell (called when BEL is received).
    pub fn trigger_bell(&mut self) {
        self.bell_start = Some(Instant::now());
    }

    /// Get the next deadline the event loop should wake at for cursor blink.
    ///
    /// Returns `None` if cursor blink is disabled.
    pub fn next_blink_deadline(&self) -> Option<Instant> {
        Some(self.last_blink + std::time::Duration::from_millis(530))
    }

    /// Reset cursor blink visibility (e.g. after user input).
    pub fn reset_blink(&mut self) {
        self.cursor_blink_visible = true;
        self.last_blink = Instant::now();
    }

    /// Set or clear the graphics overlay scene.
    ///
    /// When set, the scene is rasterized (via scry-engine's CPU backend) and
    /// composited between cell backgrounds and text. Pass `None` to remove
    /// the overlay entirely (zero overhead when no overlay is active).
    pub fn set_overlay_scene(&mut self, scene: Option<scry_engine::scene::PixelCanvas>) {
        let new_hash = scene.as_ref().map_or(0, scry_engine::scene::PixelCanvas::content_hash);
        if new_hash != self.overlay_hash {
            self.overlay_dirty = true;
            self.overlay_hash = new_hash;
        }
        if scene.is_none() {
            self.overlay_texture = None;
            self.overlay_bind_group = None;
        }
        self.overlay_scene = scene;
    }

    /// Rasterize the overlay scene to a GPU texture when dirty.
    fn update_overlay(&mut self) {
        if !self.overlay_dirty {
            return;
        }
        self.overlay_dirty = false;

        let Some(scene) = &self.overlay_scene else {
            self.overlay_texture = None;
            self.overlay_bind_group = None;
            return;
        };

        // Rasterize via scry-engine's CPU backend (tiny-skia)
        let Ok(pixmap) = scry_engine::rasterize::Rasterizer::rasterize(scene) else {
            return;
        };

        let tex_width = pixmap.width();
        let tex_height = pixmap.height();

        // Recreate texture if dimensions changed
        let needs_new_texture = self.overlay_texture.as_ref().is_none_or(|t| {
            t.width() != tex_width || t.height() != tex_height
        });

        if needs_new_texture {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("overlay_texture"),
                size: wgpu::Extent3d {
                    width: tex_width,
                    height: tex_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("overlay_bind_group"),
                layout: &self.overlay_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.overlay_sampler),
                    },
                ],
            });

            self.overlay_texture = Some(texture);
            self.overlay_bind_group = Some(bind_group);
        }

        // Upload pixel data
        if let Some(texture) = &self.overlay_texture {
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                pixmap.data(),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * tex_width),
                    rows_per_image: Some(tex_height),
                },
                wgpu::Extent3d {
                    width: tex_width,
                    height: tex_height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    /// Render a complete frame.
    pub fn render_frame(
        &mut self,
        grid: &TerminalGrid,
        selection: Option<&Selection>,
    ) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Background color
        let (bg_r, bg_g, bg_b) = self.config.colors.bg_rgb();
        let clear_color = wgpu::Color {
            r: bg_r as f64 / 255.0,
            g: bg_g as f64 / 255.0,
            b: bg_b as f64 / 255.0,
            a: 1.0,
        };

        // Update cursor blink state
        let now = Instant::now();
        if now.duration_since(self.last_blink).as_millis() >= 530 {
            self.cursor_blink_visible = !self.cursor_blink_visible;
            self.last_blink = now;
        }

        // Update overlay texture if dirty
        self.update_overlay();

        // Build cell background instances
        let instances = self.build_bg_instances(grid);
        let _instance_count = instances.len() as u32;

        // Add selection highlight + cursor instance
        let selection_instances = selection
            .map(|sel| self.build_selection_instances(sel, grid))
            .unwrap_or_default();
        let cursor_instances = self.build_cursor_instance(grid);
        // Visual bell overlay
        let bell_instances = self.build_bell_instance(now);
        let all_instances: Vec<CellBgInstance> = instances
            .into_iter()
            .chain(selection_instances)
            .chain(cursor_instances)
            .chain(bell_instances)
            .collect();
        let total_instances = all_instances.len() as u32;

        // Grow instance buffer if needed
        if total_instances > self.bg_instance_capacity {
            self.bg_instance_capacity = total_instances.next_power_of_two();
            self.bg_instance_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bg_instances"),
                size: (self.bg_instance_capacity as usize * std::mem::size_of::<CellBgInstance>())
                    as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        // Upload uniforms and instances
        let uniforms = BgUniforms {
            screen_size: [self.width as f32, self.height as f32],
        };
        self.queue
            .write_buffer(&self.bg_uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        if !all_instances.is_empty() {
            self.queue.write_buffer(
                &self.bg_instance_buffer,
                0,
                bytemuck::cast_slice(&all_instances),
            );
        }

        // Prepare text
        let _ = self.text_engine.prepare(
            &self.device,
            &self.queue,
            grid,
            &self.config.colors,
            self.width,
            self.height,
            self.padding,
        );

        // Encode render pass
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terminal"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Draw cell backgrounds + cursor
            if total_instances > 0 {
                pass.set_pipeline(&self.bg_pipeline);
                pass.set_bind_group(0, &self.bg_bind_group, &[]);
                pass.set_vertex_buffer(0, self.bg_instance_buffer.slice(..));
                pass.draw(0..6, 0..total_instances);
            }

            // Draw overlay (scry-engine graphics layer)
            if let Some(bg) = &self.overlay_bind_group {
                pass.set_pipeline(&self.overlay_pipeline);
                pass.set_bind_group(0, bg, &[]);
                pass.draw(0..3, 0..1); // Fullscreen triangle
            }

            // Draw text
            let _ = self.text_engine.render(&mut pass);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        // Trim atlas periodically
        self.text_engine.trim();

        Ok(())
    }

    /// Build cell background instances for cells with non-default backgrounds.
    fn build_bg_instances(&self, grid: &TerminalGrid) -> Vec<CellBgInstance> {
        let mut instances = Vec::new();
        let cw = self.text_engine.cell_width();
        let ch = self.text_engine.cell_height();
        let pad = self.padding;

        for row in 0..grid.rows() {
            for col in 0..grid.cols() {
                let cell = grid.viewport_cell(col, row);

                // Determine effective background color
                let bg = if cell.flags.contains(crate::grid::CellFlags::INVERSE) {
                    cell.fg
                } else {
                    cell.bg
                };

                if bg == CellColor::Default {
                    continue; // Default bg — handled by clear color
                }

                let (r, g, b) = bg.resolve(false, &self.config.colors);

                instances.push(CellBgInstance {
                    pos: [pad + col as f32 * cw, pad + row as f32 * ch],
                    size: [cw * cell.width.max(1) as f32, ch],
                    color: [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0],
                });
            }
        }

        instances
    }

    /// Build selection highlight instances.
    fn build_selection_instances(
        &self,
        selection: &Selection,
        grid: &TerminalGrid,
    ) -> Vec<CellBgInstance> {
        if selection.is_empty() {
            return Vec::new();
        }

        let mut instances = Vec::new();
        let cw = self.text_engine.cell_width();
        let ch = self.text_engine.cell_height();
        let pad = self.padding;
        // Semi-transparent blue highlight
        let color = [0.4, 0.6, 1.0, 0.35];

        for row in 0..grid.rows() {
            for col in 0..grid.cols() {
                if selection.contains(col, row as i64) {
                    instances.push(CellBgInstance {
                        pos: [pad + col as f32 * cw, pad + row as f32 * ch],
                        size: [cw, ch],
                        color,
                    });
                }
            }
        }

        instances
    }

    /// Build cursor instance.
    fn build_cursor_instance(&self, grid: &TerminalGrid) -> Vec<CellBgInstance> {
        if !grid.cursor.visible || grid.scroll_offset > 0 {
            return Vec::new();
        }

        // If cursor blink is enabled and cursor is in the "off" phase, hide it
        if grid.cursor.blink && !self.cursor_blink_visible {
            return Vec::new();
        }

        let cw = self.text_engine.cell_width();
        let ch = self.text_engine.cell_height();
        let pad = self.padding;
        let col = grid.cursor.col as f32;
        let row = grid.cursor.row as f32;

        // Cursor color — use config or foreground
        let (r, g, b) = self.config.colors.fg_rgb();
        let color = [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 0.8];

        let (pos, size) = match grid.cursor.style {
            CursorStyle::Block => (
                [pad + col * cw, pad + row * ch],
                [cw, ch],
            ),
            CursorStyle::Bar => (
                [pad + col * cw, pad + row * ch],
                [2.0, ch],
            ),
            CursorStyle::Underline => (
                [pad + col * cw, pad + (row + 1.0) * ch - 2.0],
                [cw, 2.0],
            ),
        };

        vec![CellBgInstance { pos, size, color }]
    }

    /// Build visual bell overlay instance.
    fn build_bell_instance(&mut self, now: Instant) -> Vec<CellBgInstance> {
        let Some(start) = self.bell_start else {
            return Vec::new();
        };

        let elapsed_ms = now.duration_since(start).as_millis() as f32;
        let duration_ms = 100.0;

        if elapsed_ms >= duration_ms {
            self.bell_start = None;
            return Vec::new();
        }

        // Fade from 0.3 to 0 over duration
        let alpha = 0.3 * (1.0 - elapsed_ms / duration_ms);

        vec![CellBgInstance {
            pos: [0.0, 0.0],
            size: [self.width as f32, self.height as f32],
            color: [1.0, 1.0, 1.0, alpha],
        }]
    }
}
