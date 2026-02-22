// SPDX-License-Identifier: MIT OR Apache-2.0
//! Play subcommand — `scry play`.
//!
//! Interactive TUI animation viewer and inline illusion renderer.
//! Includes optical illusion presets that render pixel-perfect patterns
//! inline in the terminal.

use clap::ValueEnum;
use scry_engine::rasterize::Rasterizer;
use scry_engine::scene::style::{BlendMode, Color as C, Rect as PxRect, Transform};
use scry_engine::scene::PixelCanvas;

use crate::display;

// ---------------------------------------------------------------------------
// CLI types
// ---------------------------------------------------------------------------

/// Available animation presets.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum PlayPreset {
    /// Sacred geometry animation
    Geometry,
    /// Wave interference pattern
    Wave,
    /// Fractal zoomer
    Fractal,
    /// Aurora borealis
    Aurora,
    /// Optical illusions gallery (Moiré, Café Wall, Penrose triangle, …)
    Illusion,
    /// Spinning 3D cube with rainbow gradient (SDF ray marched)
    Cube,
    /// Hypnotic toroidal vortex (glass torus + fire core)
    Vortex,
    /// Breathing organic orb (smooth-blended spheres)
    Pulse,
    /// Cosmic orbital system (mirror sphere + orbiting bodies)
    Orbit,
    /// Rainbow chrome torus sliced to reveal trippy swirl
    Torus,
    /// Psychedelic mirror with swirling 3D text
    Mirror,
    /// Mandelbulb fractal in rainbow chrome
    Mandelbulb,
    /// Glass Menger sponge with fire inside
    Menger,
    /// Gyroid minimal surface in rainbow
    Gyroid,
    /// Gradient descent on a 3D loss landscape
    GradientDescent,
    /// Neural network layers pulsing with signal propagation
    NeuralNet,
    /// K-Means clustering with converging centroids
    KMeans,
    /// Volumetric god rays through Menger sponge
    #[value(name = "godrays")]
    GodRays,
    /// Translucent subsurface scattering demo (jade / wax)
    Sss,
    /// Animated SDF shape morphing (sphere ↔ torus)
    Morph,
}

impl std::fmt::Display for PlayPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Geometry => write!(f, "geometry"),
            Self::Wave => write!(f, "wave"),
            Self::Fractal => write!(f, "fractal"),
            Self::Aurora => write!(f, "aurora"),
            Self::Illusion => write!(f, "illusion"),
            Self::Cube => write!(f, "cube"),
            Self::Vortex => write!(f, "vortex"),
            Self::Pulse => write!(f, "pulse"),
            Self::Orbit => write!(f, "orbit"),
            Self::Torus => write!(f, "torus"),
            Self::Mirror => write!(f, "mirror"),
            Self::Mandelbulb => write!(f, "mandelbulb"),
            Self::Menger => write!(f, "menger"),
            Self::Gyroid => write!(f, "gyroid"),
            Self::GradientDescent => write!(f, "gradient-descent"),
            Self::NeuralNet => write!(f, "neural-net"),
            Self::KMeans => write!(f, "kmeans"),
            Self::GodRays => write!(f, "godrays"),
            Self::Sss => write!(f, "sss"),
            Self::Morph => write!(f, "morph"),
        }
    }
}

/// CLI arguments for the play subcommand.
#[derive(Debug, clap::Args)]
pub struct PlayArgs {
    /// Animation preset to play
    #[arg(default_value = "geometry")]
    pub preset: PlayPreset,

    /// Auto-exit after this many seconds (0 = run until 'q')
    #[arg(short, long, default_value = "0")]
    pub duration: u64,

    /// Target frames per second
    #[arg(long, default_value = "30")]
    pub fps: u32,

    /// Output width in pixels (default: 960)
    #[arg(short = 'W', long, default_value = "960")]
    pub width: u32,

    /// Output height in pixels (default: 640)
    #[arg(short = 'H', long, default_value = "640")]
    pub height: u32,

    /// Use low-resolution rendering (200×200)
    #[arg(long)]
    pub low_res: bool,
}

/// Compute the default SDF resolution from the terminal's actual pixel
/// dimensions.  We target about 40% of the terminal's shorter axis so
/// the rendered object is a compact, crisp inline element — not a
/// massive rectangle that fills the whole screen.
pub(crate) fn sdf_default_res() -> u32 {
    let (pw, ph) = crate::display::terminal_pixel_size();
    // 40% of the smaller axis, capped to [200, 300] for quality vs size.
    let target = ((pw.min(ph) as f64) * 0.4) as u32;
    target.clamp(200, 300)
}
/// Low-res SDF resolution.
pub(crate) const SDF_LOW_RES: u32 = 200;

/// Shared parameters for all SDF animation presets.
///
/// Both `scry play` and `scry see` convert their CLI args into this
/// struct before calling the `run_*` functions.
#[derive(Debug, Clone)]
pub(crate) struct SdfRunParams {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub duration: u64,
}

impl PlayArgs {
    pub(crate) fn to_sdf_params(&self) -> SdfRunParams {
        let default = if self.low_res { SDF_LOW_RES } else { sdf_default_res() };
        SdfRunParams {
            width: if self.width != 960 { self.width } else { default },
            height: if self.height != 640 { self.height } else { default },
            fps: self.fps,
            duration: self.duration,
        }
    }
}

/// Wrapper that uses `SdfPipeline` for GPU-accelerated rendering with
/// automatic CPU fallback.
///
/// Previous implementation directly managed `SdfGpuContext` and made
/// its own GPU/CPU decision — now delegates everything to the unified
/// `SdfPipeline` which uses `GpuHealthMonitor` for coordinated decisions.
pub(crate) struct GpuRenderCtx {
    pipeline: scry_engine::sdf::SdfPipeline,
}

impl GpuRenderCtx {
    /// Create a new render context backed by `SdfPipeline`.
    pub fn new() -> Self {
        let pipeline = scry_engine::sdf::SdfPipeline::new();
        Self { pipeline }
    }

    /// Render a scene, using GPU if available, otherwise CPU.
    ///
    /// Uses the double-buffered pipeline: first frame renders on CPU while
    /// GPU shader compiles in the background, then switches to GPU
    /// automatically once ready.
    pub fn render(
        &mut self,
        scene: &scry_engine::sdf::SdfScene,
        width: u32,
        height: u32,
        time: f32,
    ) -> Result<scry_engine::Pixmap, scry_engine::PixelCanvasError> {
        let result = self.pipeline.render(scene, width, height, time);
        let w = result.width;
        let h = result.height;
        let mut pixmap = scry_engine::Pixmap::new(w, h).ok_or_else(|| {
            scry_engine::PixelCanvasError::PixmapCreation(format!(
                "failed to create {w}x{h} pixmap"
            ))
        })?;
        pixmap.data_mut().copy_from_slice(result.image.data());
        Ok(pixmap)
    }

    /// Flush pending GPU work. Call after displaying a frame so GPU
    /// compute overlaps with terminal I/O.
    pub fn flush(&mut self) {
        self.pipeline.flush();
    }
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

pub fn run(args: &PlayArgs) -> Result<(), String> {
    let params = args.to_sdf_params();
    match args.preset {
        PlayPreset::Illusion => run_illusion(&params),
        PlayPreset::Cube => run_cube(&params),
        PlayPreset::Vortex => run_vortex(&params),
        PlayPreset::Pulse => run_pulse(&params),
        PlayPreset::Orbit => run_orbit(&params),
        PlayPreset::Torus => run_torus(&params),
        PlayPreset::Mirror => run_mirror(&params),
        PlayPreset::Mandelbulb => run_mandelbulb(&params),
        PlayPreset::Menger => run_menger(&params),
        PlayPreset::Gyroid => run_gyroid(&params),
        PlayPreset::GradientDescent => run_gradient_descent(&params),
        PlayPreset::NeuralNet => run_neural_net(&params),
        PlayPreset::KMeans => run_kmeans(&params),
        PlayPreset::GodRays => run_godrays(&params),
        PlayPreset::Sss => run_sss(&params),
        PlayPreset::Morph => run_morph(&params),
        preset => {
            eprintln!("scry play: interactive TUI animations");
            eprintln!();
            eprintln!("  Preset: {preset}");
            eprintln!("  FPS:    {}", args.fps);
            if args.duration > 0 {
                eprintln!("  Duration: {}s", args.duration);
            } else {
                eprintln!("  Duration: until 'q' is pressed");
            }
            eprintln!();
            eprintln!("  This preset is coming soon. For now, try:");
            eprintln!("    scry play --preset illusion");
            eprintln!("    scry play --preset cube");
            eprintln!("    scry play --preset vortex");
            eprintln!("    scry play --preset pulse");
            eprintln!("    scry play --preset orbit");
            eprintln!();
            eprintln!("Available presets:");
            eprintln!("  illusion   Optical illusions gallery ★");
            eprintln!("  cube       Spinning 3D rainbow cube ★");
            eprintln!("  vortex    Glass torus + fire core ★");
            eprintln!("  pulse     Breathing organic orb ★");
            eprintln!("  orbit     Cosmic orbital system ★");
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Cube preset — spinning 3D ray-marched cube with rainbow gradient
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
pub(crate) fn run_cube(args: &SdfRunParams) -> Result<(), String> {
    use crossterm::event::{self, Event, KeyEventKind};
    use scry_engine::scene::style::Color;
    use scry_engine::sdf::{
        Material, SdfCamera, SdfLight, SdfObject, SdfScene, SdfShape, Vec3,
    };
    use std::time::{Duration, Instant};

    let w = args.width;
    let h = args.height;

    let fps = args.fps.max(1);
    let frame_dur = Duration::from_secs_f64(1.0 / fps as f64);
    let deadline = if args.duration > 0 {
        Some(Instant::now() + Duration::from_secs(args.duration))
    } else {
        None
    };



    let mut gpu_ctx = GpuRenderCtx::new();
    let mut driver = display::FrameDriver::detect();
    let mut frame_count: u64 = 0;
    let start = Instant::now();

    crossterm::terminal::enable_raw_mode().map_err(|e| e.to_string())?;

    let result = (|| -> Result<(), String> {
        loop {
            let frame_start = Instant::now();
            let t = start.elapsed().as_secs_f32();

            // Non-blocking keypress check
            if event::poll(Duration::ZERO).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
                    if key.kind == KeyEventKind::Press {
                        break;
                    }
                }
            }

            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    break;
                }
            }

            // Build the scene fresh each frame with updated rotation & hue
            let angle_y = t * 0.7;
            let angle_x = t * 0.4;

            let qx = scry_engine::math3d::Quaternion::from_axis_angle(
                Vec3::new(1.0, 0.0, 0.0),
                angle_x,
            );
            let qy = scry_engine::math3d::Quaternion::from_axis_angle(
                Vec3::new(0.0, 1.0, 0.0),
                angle_y,
            );
            let orientation = qy * qx;

            let cube = SdfObject::new(
                SdfShape::Box {
                    half_extents: Vec3::new(0.85, 0.85, 0.85),
                },
                Material::rainbow_animated(t * 0.5),
            )
            .at(Vec3::ZERO)
            .orient(orientation);

            // Orbiting camera — pulled back so the full rotating cube fits
            // comfortably with margin on all sides.
            let cam_angle = t * 0.3;
            let cam_radius = 5.0;
            let cam_y = 2.0 + (t * 0.2).sin() * 0.3;
            let eye = Vec3::new(
                cam_angle.cos() * cam_radius,
                cam_y,
                cam_angle.sin() * cam_radius,
            );

            // Transparent sky — no background, just the cube
            let transparent = Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

            let scene = SdfScene::new()
                .object(cube)
                .light(SdfLight::new(
                    Vec3::new(5.0, 8.0, 5.0),
                    Color::WHITE,
                    1.2,
                ))
                .light(SdfLight::new(
                    Vec3::new(-4.0, 3.0, -2.0),
                    Color::from_rgba8(100, 150, 255, 255),
                    0.6,
                ))
                .camera(SdfCamera::new(eye, Vec3::ZERO, 45.0))
                .sky_color(transparent)
                .ambient(0.08)
                .max_bounces(1);

            let pixmap = gpu_ctx.render(&scene, w, h, t)
                .map_err(|e| format!("SDF render failed: {e}"))?;

            driver.display_frame(&pixmap, frame_count)
                .map_err(|e| format!("display failed: {e}"))?;

            frame_count += 1;

            let elapsed = frame_start.elapsed();
            if elapsed < frame_dur {
                std::thread::sleep(frame_dur - elapsed);
            }
        }
        Ok(())
    })();

    let _ = crossterm::terminal::disable_raw_mode();

    result?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Vortex preset — hypnotic glass torus with fire core
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
pub(crate) fn run_vortex(args: &SdfRunParams) -> Result<(), String> {
    use crossterm::event::{self, Event, KeyEventKind};
    use scry_engine::scene::style::Color;
    use scry_engine::sdf::{
        Material, SdfCamera, SdfLight, SdfObject, SdfScene, SdfShape, Vec3,
    };
    use std::time::{Duration, Instant};

    let w = args.width;
    let h = args.height;

    let fps = args.fps.max(1);
    let frame_dur = Duration::from_secs_f64(1.0 / fps as f64);
    let deadline = if args.duration > 0 {
        Some(Instant::now() + Duration::from_secs(args.duration))
    } else {
        None
    };



    let mut gpu_ctx = GpuRenderCtx::new();
    let mut driver = display::FrameDriver::detect();
    let mut frame_count: u64 = 0;
    let start = Instant::now();

    crossterm::terminal::enable_raw_mode().map_err(|e| e.to_string())?;

    let result = (|| -> Result<(), String> {
        loop {
            let frame_start = Instant::now();
            let t = start.elapsed().as_secs_f32();

            if event::poll(Duration::ZERO).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
                    if key.kind == KeyEventKind::Press {
                        break;
                    }
                }
            }

            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    break;
                }
            }

            // Triple-axis rotation for the torus
            let qx = scry_engine::math3d::Quaternion::from_axis_angle(
                Vec3::new(1.0, 0.0, 0.0),
                t * 0.5,
            );
            let qy = scry_engine::math3d::Quaternion::from_axis_angle(
                Vec3::new(0.0, 1.0, 0.0),
                t * 0.3,
            );
            let qz = scry_engine::math3d::Quaternion::from_axis_angle(
                Vec3::new(0.0, 0.0, 1.0),
                t * 0.2,
            );
            let orientation = qz * qy * qx;

            // Glass torus — refracts the fire light inside
            let torus = SdfObject::new(
                SdfShape::Torus {
                    major: 1.6,
                    minor: 0.45,
                },
                Material::glass(
                    Color::from_rgba8(200, 220, 255, 255),
                    1.45,
                ),
            )
            .at(Vec3::ZERO)
            .orient(orientation);

            // Pulsating fire core inside the torus
            let pulse = 0.4 + (t * 1.5).sin() * 0.15;
            let fire_core = SdfObject::new(
                SdfShape::Sphere { radius: pulse },
                Material::fire(),
            )
            .at(Vec3::ZERO);

            // Camera orbits slowly
            let cam_angle = t * 0.25;
            let cam_r = 5.0;
            let cam_y = 2.0 + (t * 0.15).sin() * 0.5;
            let eye = Vec3::new(
                cam_angle.cos() * cam_r,
                cam_y,
                cam_angle.sin() * cam_r,
            );

            let transparent = Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

            let scene = SdfScene::new()
                .object(torus)
                .object(fire_core)
                .light(SdfLight::new(
                    Vec3::new(4.0, 6.0, 4.0),
                    Color::WHITE,
                    1.4,
                ))
                .light(SdfLight::new(
                    Vec3::new(-3.0, 2.0, -5.0),
                    Color::from_rgba8(255, 140, 60, 255),
                    0.7,
                ))
                .camera(SdfCamera::new(eye, Vec3::ZERO, 45.0))
                .sky_color(transparent)
                .ambient(0.06)
                .max_bounces(2);

            let pixmap = gpu_ctx.render(&scene, w, h, t)
                .map_err(|e| format!("SDF render failed: {e}"))?;

            driver.display_frame(&pixmap, frame_count)
                .map_err(|e| format!("display failed: {e}"))?;

            frame_count += 1;

            let elapsed = frame_start.elapsed();
            if elapsed < frame_dur {
                std::thread::sleep(frame_dur - elapsed);
            }
        }
        Ok(())
    })();

    let _ = crossterm::terminal::disable_raw_mode();

    result?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Pulse preset — breathing organic orb (smooth-blended spheres)
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
pub(crate) fn run_pulse(args: &SdfRunParams) -> Result<(), String> {
    use crossterm::event::{self, Event, KeyEventKind};
    use scry_engine::scene::style::Color;
    use scry_engine::sdf::{
        Material, SdfCamera, SdfLight, SdfObject, SdfScene, SdfShape, Vec3,
    };
    use std::time::{Duration, Instant};

    let w = args.width;
    let h = args.height;

    let fps = args.fps.max(1);
    let frame_dur = Duration::from_secs_f64(1.0 / fps as f64);
    let deadline = if args.duration > 0 {
        Some(Instant::now() + Duration::from_secs(args.duration))
    } else {
        None
    };



    let mut gpu_ctx = GpuRenderCtx::new();
    let mut driver = display::FrameDriver::detect();
    let mut frame_count: u64 = 0;
    let start = Instant::now();

    crossterm::terminal::enable_raw_mode().map_err(|e| e.to_string())?;

    let result = (|| -> Result<(), String> {
        loop {
            let frame_start = Instant::now();
            let t = start.elapsed().as_secs_f32();

            if event::poll(Duration::ZERO).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
                    if key.kind == KeyEventKind::Press {
                        break;
                    }
                }
            }

            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    break;
                }
            }

            // Breathing factor — slow sinusoidal expansion/contraction
            let breath = 1.0 + (t * 0.8).sin() * 0.2;

            // Build an organic blob from smooth-blended spheres
            // Central sphere + 4 satellite blobs at tetrahedron-like offsets
            let spread = 0.7 * breath;
            let offsets = [
                Vec3::new(0.0, spread, 0.0),
                Vec3::new(spread * 0.94, -spread * 0.33, 0.0),
                Vec3::new(-spread * 0.47, -spread * 0.33, spread * 0.82),
                Vec3::new(-spread * 0.47, -spread * 0.33, -spread * 0.82),
            ];

            // Use SmoothBlend to create the organic shape step by step.
            // Start: core sphere blended with first satellite
            let core_r = 0.8 * breath;
            let sat_r = 0.55 * breath;

            let mut shape = SdfShape::SmoothBlend {
                a: std::boxed::Box::new(SdfShape::Sphere { radius: core_r }),
                b: std::boxed::Box::new(SdfShape::Sphere { radius: sat_r }),
                b_offset: offsets[0],
                k: 0.6,
            };

            // Blend in the remaining satellites
            for &off in &offsets[1..] {
                shape = SdfShape::SmoothBlend {
                    a: std::boxed::Box::new(shape),
                    b: std::boxed::Box::new(SdfShape::Sphere { radius: sat_r }),
                    b_offset: off,
                    k: 0.5,
                };
            }

            // Slow tumbling rotation
            let qx = scry_engine::math3d::Quaternion::from_axis_angle(
                Vec3::new(1.0, 0.0, 0.0),
                t * 0.15,
            );
            let qy = scry_engine::math3d::Quaternion::from_axis_angle(
                Vec3::new(0.0, 1.0, 0.0),
                t * 0.25,
            );
            let orientation = qy * qx;

            let blob = SdfObject::new(shape, Material::rainbow_animated(t * 0.3))
                .at(Vec3::ZERO)
                .orient(orientation);

            // Camera with gentle breathing bob
            let cam_y = 2.5 + (t * 0.4).sin() * 0.3;
            let cam_angle = t * 0.12;
            let cam_r = 4.5;
            let eye = Vec3::new(
                cam_angle.cos() * cam_r,
                cam_y,
                cam_angle.sin() * cam_r,
            );

            let transparent = Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

            let scene = SdfScene::new()
                .object(blob)
                .light(SdfLight::new(
                    Vec3::new(5.0, 8.0, 3.0),
                    Color::WHITE,
                    1.3,
                ))
                .light(SdfLight::new(
                    Vec3::new(-4.0, 2.0, -3.0),
                    Color::from_rgba8(180, 100, 255, 255),
                    0.5,
                ))
                .light(SdfLight::new(
                    Vec3::new(1.0, -2.0, 5.0),
                    Color::from_rgba8(100, 255, 180, 255),
                    0.4,
                ))
                .camera(SdfCamera::new(eye, Vec3::ZERO, 50.0))
                .sky_color(transparent)
                .ambient(0.10)
                .max_bounces(1);

            let pixmap = gpu_ctx.render(&scene, w, h, t)
                .map_err(|e| format!("SDF render failed: {e}"))?;

            driver.display_frame(&pixmap, frame_count)
                .map_err(|e| format!("display failed: {e}"))?;

            frame_count += 1;

            let elapsed = frame_start.elapsed();
            if elapsed < frame_dur {
                std::thread::sleep(frame_dur - elapsed);
            }
        }
        Ok(())
    })();

    let _ = crossterm::terminal::disable_raw_mode();

    result?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Orbit preset — mirror sphere with orbiting bodies
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
pub(crate) fn run_orbit(args: &SdfRunParams) -> Result<(), String> {
    use crossterm::event::{self, Event, KeyEventKind};
    use scry_engine::scene::style::Color;
    use scry_engine::sdf::{
        Material, SdfCamera, SdfLight, SdfObject, SdfScene, SdfShape, Vec3,
    };
    use std::time::{Duration, Instant};

    let w = args.width;
    let h = args.height;

    let fps = args.fps.max(1);
    let frame_dur = Duration::from_secs_f64(1.0 / fps as f64);
    let deadline = if args.duration > 0 {
        Some(Instant::now() + Duration::from_secs(args.duration))
    } else {
        None
    };



    let mut gpu_ctx = GpuRenderCtx::new();
    let mut driver = display::FrameDriver::detect();
    let mut frame_count: u64 = 0;
    let start = Instant::now();

    crossterm::terminal::enable_raw_mode().map_err(|e| e.to_string())?;

    let result = (|| -> Result<(), String> {
        loop {
            let frame_start = Instant::now();
            let t = start.elapsed().as_secs_f32();
            let pi = std::f32::consts::PI;

            if event::poll(Duration::ZERO).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
                    if key.kind == KeyEventKind::Press {
                        break;
                    }
                }
            }

            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    break;
                }
            }

            // Central mirror sphere — reflects the orbiters
            let center = SdfObject::new(
                SdfShape::Sphere { radius: 0.8 },
                Material::mirror(Color::from_rgba8(220, 220, 240, 255), 0.85),
            )
            .at(Vec3::ZERO);

            // Orbiter 1: Fire — fast equatorial orbit
            let a1 = t * 1.2;
            let r1 = 2.2;
            let pos1 = Vec3::new(a1.cos() * r1, (a1 * 0.5).sin() * 0.3, a1.sin() * r1);
            let orbiter1 = SdfObject::new(
                SdfShape::Sphere { radius: 0.3 },
                Material::fire(),
            )
            .at(pos1);

            // Orbiter 2: Glass — medium speed, tilted orbital plane
            let a2 = t * 0.7 + pi * 2.0 / 3.0;
            let r2 = 2.5;
            let tilt2 = 0.6_f32; // tilt angle
            let pos2 = Vec3::new(
                a2.cos() * r2,
                a2.sin() * r2 * tilt2.sin(),
                a2.sin() * r2 * tilt2.cos(),
            );
            let orbiter2 = SdfObject::new(
                SdfShape::Sphere { radius: 0.35 },
                Material::glass_dispersive(
                    Color::from_rgba8(180, 240, 255, 255),
                    1.5,
                    0.03,
                ),
            )
            .at(pos2);

            // Orbiter 3: Rainbow torus — slow, polar orbit
            let a3 = t * 0.4 + pi * 4.0 / 3.0;
            let r3 = 2.8;
            let pos3 = Vec3::new(
                a3.sin() * r3 * 0.3,
                a3.cos() * r3,
                a3.sin() * r3 * 0.95,
            );
            let qr3 = scry_engine::math3d::Quaternion::from_axis_angle(
                Vec3::new(1.0, 0.0, 0.0),
                a3,
            );
            let orbiter3 = SdfObject::new(
                SdfShape::Torus {
                    major: 0.25,
                    minor: 0.08,
                },
                Material::rainbow_animated(t * 0.5),
            )
            .at(pos3)
            .orient(qr3);

            // Camera at a slight angle, slowly rotating
            let cam_angle = t * 0.15;
            let cam_r = 6.0;
            let cam_y = 3.0 + (t * 0.1).sin() * 0.5;
            let eye = Vec3::new(
                cam_angle.cos() * cam_r,
                cam_y,
                cam_angle.sin() * cam_r,
            );

            let transparent = Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

            let scene = SdfScene::new()
                .object(center)
                .object(orbiter1)
                .object(orbiter2)
                .object(orbiter3)
                .light(SdfLight::new(
                    Vec3::new(6.0, 8.0, 4.0),
                    Color::WHITE,
                    1.3,
                ))
                .light(SdfLight::new(
                    Vec3::new(-5.0, 3.0, -3.0),
                    Color::from_rgba8(100, 140, 255, 255),
                    0.6,
                ))
                .camera(SdfCamera::new(eye, Vec3::ZERO, 45.0))
                .sky_color(transparent)
                .ambient(0.06)
                .max_bounces(3);

            let pixmap = gpu_ctx.render(&scene, w, h, t)
                .map_err(|e| format!("SDF render failed: {e}"))?;

            driver.display_frame(&pixmap, frame_count)
                .map_err(|e| format!("display failed: {e}"))?;

            frame_count += 1;

            let elapsed = frame_start.elapsed();
            if elapsed < frame_dur {
                std::thread::sleep(frame_dur - elapsed);
            }
        }
        Ok(())
    })();

    let _ = crossterm::terminal::disable_raw_mode();

    result?;

    Ok(())
}



// ---------------------------------------------------------------------------
// Mandelbulb preset — iconic 3D fractal in rainbow chrome
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
pub(crate) fn run_mandelbulb(args: &SdfRunParams) -> Result<(), String> {
    use crossterm::event::{self, Event, KeyEventKind};
    use scry_engine::scene::style::Color;
    use scry_engine::sdf::{
        Material, SdfCamera, SdfLight, SdfObject, SdfScene, SdfShape, Vec3,
    };
    use std::time::{Duration, Instant};

    let w = args.width;
    let h = args.height;

    let fps = args.fps.max(1);
    let frame_dur = Duration::from_secs_f64(1.0 / fps as f64);
    let deadline = if args.duration > 0 {
        Some(Instant::now() + Duration::from_secs(args.duration))
    } else {
        None
    };

    let mut gpu_ctx = GpuRenderCtx::new();
    let mut driver = display::FrameDriver::detect();
    let mut frame_count: u64 = 0;
    let start = Instant::now();

    crossterm::terminal::enable_raw_mode().map_err(|e| e.to_string())?;

    let result = (|| -> Result<(), String> {
        loop {
            let frame_start = Instant::now();
            let t = start.elapsed().as_secs_f32();

            if event::poll(Duration::ZERO).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
                    if key.kind == KeyEventKind::Press {
                        break;
                    }
                }
            }

            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    break;
                }
            }

            // Slow Y-axis rotation
            let qy = scry_engine::math3d::Quaternion::from_axis_angle(
                Vec3::new(0.0, 1.0, 0.0),
                t * 0.3,
            );

            // Breathing power: oscillates subtly around 8
            let power = 8.0 + (t * 0.2).sin() * 0.5;

            let bulb = SdfObject::new(
                SdfShape::Mandelbulb {
                    power,
                    iterations: 10,
                },
                Material::Rainbow {
                    saturation: 0.9,
                    lightness: 0.45,
                    hue_offset: t * 0.3,
                    specular: 96.0,
                },
            )
            .at(Vec3::ZERO)
            .orient(qy);

            // Camera orbits slowly
            let cam_angle = t * 0.15;
            let cam_r = 3.5;
            let cam_y = 1.5 + (t * 0.1).sin() * 0.5;
            let eye = Vec3::new(
                cam_angle.cos() * cam_r,
                cam_y,
                cam_angle.sin() * cam_r,
            );

            let transparent = Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

            let scene = SdfScene::new()
                .object(bulb)
                .light(SdfLight::new(
                    Vec3::new(4.0, 6.0, 4.0),
                    Color::WHITE,
                    1.2,
                ))
                .light(SdfLight::new(
                    Vec3::new(-3.0, 2.0, -4.0),
                    Color::from_rgba8(180, 120, 255, 255),
                    0.6,
                ))
                .light(SdfLight::new(
                    Vec3::new(2.0, -1.0, 5.0),
                    Color::from_rgba8(120, 255, 180, 255),
                    0.4,
                ))
                .camera(SdfCamera::new(eye, Vec3::ZERO, 45.0))
                .sky_color(transparent)
                .ambient(0.06)
                .max_bounces(1);

            let pixmap = gpu_ctx.render(&scene, w, h, t)
                .map_err(|e| format!("SDF render failed: {e}"))?;

            driver.display_frame(&pixmap, frame_count)
                .map_err(|e| format!("display failed: {e}"))?;

            frame_count += 1;

            let elapsed = frame_start.elapsed();
            if elapsed < frame_dur {
                std::thread::sleep(frame_dur - elapsed);
            }
        }
        Ok(())
    })();

    let _ = crossterm::terminal::disable_raw_mode();
    result?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Menger preset — glass Menger sponge with fire inside
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
pub(crate) fn run_menger(args: &SdfRunParams) -> Result<(), String> {
    use crossterm::event::{self, Event, KeyEventKind};
    use scry_engine::scene::style::Color;
    use scry_engine::sdf::{
        Material, SdfCamera, SdfLight, SdfObject, SdfScene, SdfShape, Vec3,
    };
    use std::time::{Duration, Instant};

    let w = args.width;
    let h = args.height;

    let fps = args.fps.max(1);
    let frame_dur = Duration::from_secs_f64(1.0 / fps as f64);
    let deadline = if args.duration > 0 {
        Some(Instant::now() + Duration::from_secs(args.duration))
    } else {
        None
    };

    let mut gpu_ctx = GpuRenderCtx::new();
    let mut driver = display::FrameDriver::detect();
    let mut frame_count: u64 = 0;
    let start = Instant::now();

    crossterm::terminal::enable_raw_mode().map_err(|e| e.to_string())?;

    let result = (|| -> Result<(), String> {
        loop {
            let frame_start = Instant::now();
            let t = start.elapsed().as_secs_f32();

            if event::poll(Duration::ZERO).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
                    if key.kind == KeyEventKind::Press {
                        break;
                    }
                }
            }

            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    break;
                }
            }

            // Slow tumbling rotation
            let qx = scry_engine::math3d::Quaternion::from_axis_angle(
                Vec3::new(1.0, 0.0, 0.0),
                t * 0.15,
            );
            let qy = scry_engine::math3d::Quaternion::from_axis_angle(
                Vec3::new(0.0, 1.0, 0.0),
                t * 0.25,
            );
            let orientation = qy * qx;

            // Glass Menger sponge — light refracts through the fractal holes
            let sponge = SdfObject::new(
                SdfShape::MengerSponge { iterations: 4 },
                Material::glass(
                    Color::from_rgba8(220, 230, 255, 255),
                    1.35,
                ),
            )
            .at(Vec3::ZERO)
            .orient(orientation);

            // Fire sphere glowing inside the sponge
            let fire_r = 0.3 + (t * 1.2).sin() * 0.08;
            let fire_core = SdfObject::new(
                SdfShape::Sphere { radius: fire_r },
                Material::Fire {
                    intensity: 3.0,
                    noise_scale: 4.0,
                    speed: 1.5,
                },
            )
            .at(Vec3::ZERO);

            // Camera
            let cam_angle = t * 0.2;
            let cam_r = 4.0;
            let cam_y = 2.0 + (t * 0.12).sin() * 0.6;
            let eye = Vec3::new(
                cam_angle.cos() * cam_r,
                cam_y,
                cam_angle.sin() * cam_r,
            );

            let transparent = Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

            let scene = SdfScene::new()
                .object(sponge)
                .object(fire_core)
                .light(SdfLight::new(
                    Vec3::new(5.0, 7.0, 4.0),
                    Color::WHITE,
                    1.4,
                ))
                .light(SdfLight::new(
                    Vec3::new(-3.0, 3.0, -5.0),
                    Color::from_rgba8(255, 180, 100, 255),
                    0.5,
                ))
                .camera(SdfCamera::new(eye, Vec3::ZERO, 45.0))
                .sky_color(transparent)
                .ambient(0.08)
                .max_bounces(2);

            let pixmap = gpu_ctx.render(&scene, w, h, t)
                .map_err(|e| format!("SDF render failed: {e}"))?;

            driver.display_frame(&pixmap, frame_count)
                .map_err(|e| format!("display failed: {e}"))?;

            frame_count += 1;

            let elapsed = frame_start.elapsed();
            if elapsed < frame_dur {
                std::thread::sleep(frame_dur - elapsed);
            }
        }
        Ok(())
    })();

    let _ = crossterm::terminal::disable_raw_mode();
    result?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Gyroid preset — triply periodic minimal surface in rainbow
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
pub(crate) fn run_gyroid(args: &SdfRunParams) -> Result<(), String> {
    use crossterm::event::{self, Event, KeyEventKind};
    use scry_engine::scene::style::Color;
    use scry_engine::sdf::{
        Material, SdfCamera, SdfLight, SdfObject, SdfScene, SdfShape, Vec3,
    };
    use std::time::{Duration, Instant};

    let w = args.width;
    let h = args.height;

    let fps = args.fps.max(1);
    let frame_dur = Duration::from_secs_f64(1.0 / fps as f64);
    let deadline = if args.duration > 0 {
        Some(Instant::now() + Duration::from_secs(args.duration))
    } else {
        None
    };

    let mut gpu_ctx = GpuRenderCtx::new();
    let mut driver = display::FrameDriver::detect();
    let mut frame_count: u64 = 0;
    let start = Instant::now();

    crossterm::terminal::enable_raw_mode().map_err(|e| e.to_string())?;

    let result = (|| -> Result<(), String> {
        loop {
            let frame_start = Instant::now();
            let t = start.elapsed().as_secs_f32();

            if event::poll(Duration::ZERO).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
                    if key.kind == KeyEventKind::Press {
                        break;
                    }
                }
            }

            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    break;
                }
            }

            // Slow tumbling rotation
            let qx = scry_engine::math3d::Quaternion::from_axis_angle(
                Vec3::new(1.0, 0.0, 0.0),
                t * 0.2,
            );
            let qy = scry_engine::math3d::Quaternion::from_axis_angle(
                Vec3::new(0.0, 1.0, 0.0),
                t * 0.35,
            );
            let qz = scry_engine::math3d::Quaternion::from_axis_angle(
                Vec3::new(0.0, 0.0, 1.0),
                t * 0.1,
            );
            let orientation = qz * qy * qx;

            // Animated scale for a subtly morphing organic feel
            let scale = 4.0 + (t * 0.15).sin() * 0.5;

            let gyroid_obj = SdfObject::new(
                SdfShape::Gyroid {
                    scale,
                    thickness: 0.25,
                    bound: 1.5,
                },
                Material::Rainbow {
                    saturation: 1.0,
                    lightness: 0.5,
                    hue_offset: t * 0.4,
                    specular: 64.0,
                },
            )
            .at(Vec3::ZERO)
            .orient(orientation);

            // Camera
            let cam_angle = t * 0.18;
            let cam_r = 4.2;
            let cam_y = 2.0 + (t * 0.1).sin() * 0.6;
            let eye = Vec3::new(
                cam_angle.cos() * cam_r,
                cam_y,
                cam_angle.sin() * cam_r,
            );

            let transparent = Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

            let scene = SdfScene::new()
                .object(gyroid_obj)
                .light(SdfLight::new(
                    Vec3::new(4.0, 6.0, 5.0),
                    Color::WHITE,
                    1.3,
                ))
                .light(SdfLight::new(
                    Vec3::new(-4.0, 2.0, -3.0),
                    Color::from_rgba8(200, 150, 255, 255),
                    0.5,
                ))
                .light(SdfLight::new(
                    Vec3::new(3.0, -2.0, 4.0),
                    Color::from_rgba8(150, 255, 200, 255),
                    0.4,
                ))
                .camera(SdfCamera::new(eye, Vec3::ZERO, 45.0))
                .sky_color(transparent)
                .ambient(0.07)
                .max_bounces(1);

            let pixmap = gpu_ctx.render(&scene, w, h, t)
                .map_err(|e| format!("SDF render failed: {e}"))?;

            driver.display_frame(&pixmap, frame_count)
                .map_err(|e| format!("display failed: {e}"))?;

            frame_count += 1;

            let elapsed = frame_start.elapsed();
            if elapsed < frame_dur {
                std::thread::sleep(frame_dur - elapsed);
            }
        }
        Ok(())
    })();

    let _ = crossterm::terminal::disable_raw_mode();
    result?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Torus preset — rainbow chrome torus, sliced to reveal trippy swirl inside
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
pub(crate) fn run_torus(args: &SdfRunParams) -> Result<(), String> {
    use crossterm::event::{self, Event, KeyEventKind};
    use scry_engine::scene::style::Color;
    use scry_engine::sdf::{
        Material, SdfCamera, SdfLight, SdfObject, SdfScene, SdfShape, Vec3,
    };
    use std::time::{Duration, Instant};

    let w = args.width;
    let h = args.height;

    let fps = args.fps.max(1);
    let frame_dur = Duration::from_secs_f64(1.0 / fps as f64);
    let deadline = if args.duration > 0 {
        Some(Instant::now() + Duration::from_secs(args.duration))
    } else {
        None
    };

    let mut gpu_ctx = GpuRenderCtx::new();
    let mut driver = display::FrameDriver::detect();
    let mut frame_count: u64 = 0;
    let start = Instant::now();

    crossterm::terminal::enable_raw_mode().map_err(|e| e.to_string())?;

    let result = (|| -> Result<(), String> {
        loop {
            let frame_start = Instant::now();
            let t = start.elapsed().as_secs_f32();

            if event::poll(Duration::ZERO).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
                    if key.kind == KeyEventKind::Press {
                        break;
                    }
                }
            }

            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    break;
                }
            }

            // Slow rotation
            let qx = scry_engine::math3d::Quaternion::from_axis_angle(
                Vec3::new(1.0, 0.0, 0.0),
                t * 0.3,
            );
            let qy = scry_engine::math3d::Quaternion::from_axis_angle(
                Vec3::new(0.0, 1.0, 0.0),
                t * 0.5,
            );
            let orientation = qy * qx;

            // Animate the slice: oscillates from closed to open and back.
            // sin(t * 0.6) oscillates -1..1; when negative = closed, positive = opening
            let slice_phase = (t * 0.6).sin();
            let slice_open = slice_phase.max(0.0); // 0 = closed, 1 = fully open

            // Build the torus shape, optionally sliced
            let base_torus = SdfShape::Torus {
                major: 1.4,
                minor: 0.5,
            };

            let torus_shape = if slice_open > 0.01 {
                // Slice with a box that slides through the torus
                let cut_offset = Vec3::new(0.0, 0.0, -2.0 + slice_open * 2.5);
                SdfShape::Subtract {
                    a: std::boxed::Box::new(base_torus),
                    b: std::boxed::Box::new(SdfShape::Box {
                        half_extents: Vec3::new(3.0, 3.0, 1.5),
                    }),
                    b_offset: cut_offset,
                }
            } else {
                base_torus
            };

            // Rainbow chrome material — high specular + reflectivity
            let chrome_rainbow = Material::Rainbow {
                saturation: 0.95,
                lightness: 0.5,
                hue_offset: t * 0.4,
                specular: 128.0,
            };

            let torus_obj = SdfObject::new(torus_shape, chrome_rainbow)
                .at(Vec3::ZERO)
                .orient(orientation);

            // Trippy fire swirl inside — visible when sliced open
            let swirl_r = 0.35 + (t * 2.0).sin() * 0.1;
            let swirl = SdfObject::new(
                SdfShape::Sphere { radius: swirl_r },
                Material::Fire {
                    intensity: 2.5,
                    noise_scale: 3.5,
                    speed: 2.0,
                },
            )
            .at(Vec3::ZERO);

            // Camera
            let cam_angle = t * 0.2;
            let cam_r = 4.5;
            let cam_y = 2.0 + (t * 0.15).sin() * 0.8;
            let eye = Vec3::new(
                cam_angle.cos() * cam_r,
                cam_y,
                cam_angle.sin() * cam_r,
            );

            let transparent = Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

            let scene = SdfScene::new()
                .object(torus_obj)
                .object(swirl)
                .light(SdfLight::new(
                    Vec3::new(5.0, 8.0, 4.0),
                    Color::WHITE,
                    1.5,
                ))
                .light(SdfLight::new(
                    Vec3::new(-4.0, 3.0, -5.0),
                    Color::from_rgba8(200, 100, 255, 255),
                    0.6,
                ))
                .light(SdfLight::new(
                    Vec3::new(2.0, -2.0, 4.0),
                    Color::from_rgba8(100, 255, 200, 255),
                    0.4,
                ))
                .camera(SdfCamera::new(eye, Vec3::ZERO, 50.0))
                .sky_color(transparent)
                .ambient(0.08)
                .max_bounces(2);

            let pixmap = gpu_ctx.render(&scene, w, h, t)
                .map_err(|e| format!("SDF render failed: {e}"))?;

            driver.display_frame(&pixmap, frame_count)
                .map_err(|e| format!("display failed: {e}"))?;

            frame_count += 1;

            let elapsed = frame_start.elapsed();
            if elapsed < frame_dur {
                std::thread::sleep(frame_dur - elapsed);
            }
        }
        Ok(())
    })();

    let _ = crossterm::terminal::disable_raw_mode();

    result?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Mirror preset — psychedelic mirror with swirling "esoc" 3D text
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
pub(crate) fn run_mirror(args: &SdfRunParams) -> Result<(), String> {
    use crossterm::event::{self, Event, KeyEventKind};
    use scry_engine::scene::style::Color;
    use scry_engine::sdf::{
        Material, SdfCamera, SdfLight, SdfObject, SdfScene, SdfShape, SdfTextLabel,
        Vec3,
    };
    use std::time::{Duration, Instant};

    let w = args.width;
    let h = args.height;

    let fps = args.fps.max(1);
    let frame_dur = Duration::from_secs_f64(1.0 / fps as f64);
    let deadline = if args.duration > 0 {
        Some(Instant::now() + Duration::from_secs(args.duration))
    } else {
        None
    };

    let mut gpu_ctx = GpuRenderCtx::new();
    let mut driver = display::FrameDriver::detect();
    let mut frame_count: u64 = 0;
    let start = Instant::now();

    crossterm::terminal::enable_raw_mode().map_err(|e| e.to_string())?;

    let result = (|| -> Result<(), String> {
        let pi = std::f32::consts::PI;
        loop {
            let frame_start = Instant::now();
            let t = start.elapsed().as_secs_f32();

            if event::poll(Duration::ZERO).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
                    if key.kind == KeyEventKind::Press {
                        break;
                    }
                }
            }

            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    break;
                }
            }

            // Central mirror sphere — highly reflective
            let mirror_sphere = SdfObject::new(
                SdfShape::Sphere { radius: 1.2 },
                Material::mirror(Color::from_rgba8(240, 240, 250, 255), 0.95),
            )
            .at(Vec3::ZERO);

            // Small rainbow orbiters that the mirror reflects — creates
            // psychedelic swirling reflections on the mirror surface
            let num_orbiters = 5;
            let mut scene = SdfScene::new().object(mirror_sphere);

            for i in 0..num_orbiters {
                let phase = (i as f32 / num_orbiters as f32) * pi * 2.0;
                let speed = 0.8 + i as f32 * 0.15;
                let orbit_r = 2.5 + i as f32 * 0.3;
                let tilt = 0.2 + i as f32 * 0.25;
                let a = t * speed + phase;

                let pos = Vec3::new(
                    a.cos() * orbit_r * tilt.cos(),
                    a.sin() * orbit_r * tilt.sin() + (t * 0.3 + phase).sin() * 0.5,
                    a.sin() * orbit_r * tilt.cos(),
                );

                // Alternate between fire and rainbow for psychedelic reflections
                let mat = if i % 2 == 0 {
                    Material::Rainbow {
                        saturation: 1.0,
                        lightness: 0.55,
                        hue_offset: t * 0.8 + phase,
                        specular: 64.0,
                    }
                } else {
                    Material::Fire {
                        intensity: 2.0,
                        noise_scale: 3.0,
                        speed: 1.5,
                    }
                };

                let orbiter = SdfObject::new(
                    SdfShape::Sphere { radius: 0.2 },
                    mat,
                )
                .at(pos);
                scene = scene.object(orbiter);
            }

            // 6 billboard "esoc" labels spiralling around at different heights
            let num_labels = 6;
            for i in 0..num_labels {
                let phase = (i as f32 / num_labels as f32) * pi * 2.0;
                let speed = 0.5 + i as f32 * 0.1;
                let orbit_r = 2.8 + (t * 0.2 + phase).sin() * 0.5;
                let a = t * speed + phase;

                // Spiral height that oscillates
                let y = (a * 0.4).sin() * 1.5;
                let pos = Vec3::new(
                    a.cos() * orbit_r,
                    y,
                    a.sin() * orbit_r,
                );

                // Psychedelic cycling colors
                let hue = ((t * 40.0 + i as f32 * 60.0) % 360.0) / 360.0;
                let (cr, cg, cb) = hsl_to_rgb(hue * 360.0, 1.0, 0.6);

                let label = SdfTextLabel::new(pos, "esoc")
                    .font_size(48.0)
                    .color(Color::from_rgba8(cr, cg, cb, 255));

                scene = scene.text_label(label);
            }

            // Dynamic colored lights for psychedelic effect
            let l1_hue = (t * 30.0) % 360.0 / 360.0;
            let (lr, lg, lb) = hsl_to_rgb(l1_hue * 360.0, 0.9, 0.6);
            let l2_hue = ((t * 30.0) + 120.0) % 360.0 / 360.0;
            let (lr2, lg2, lb2) = hsl_to_rgb(l2_hue * 360.0, 0.9, 0.6);

            scene = scene
                .light(SdfLight::new(
                    Vec3::new(5.0, 6.0, 5.0),
                    Color::from_rgba8(lr, lg, lb, 255),
                    1.4,
                ))
                .light(SdfLight::new(
                    Vec3::new(-4.0, 3.0, -4.0),
                    Color::from_rgba8(lr2, lg2, lb2, 255),
                    0.8,
                ))
                .light(SdfLight::new(
                    Vec3::new(0.0, -3.0, 5.0),
                    Color::WHITE,
                    0.5,
                ));

            // Camera orbits slowly
            let cam_angle = t * 0.2;
            let cam_r = 5.5;
            let cam_y = 2.5 + (t * 0.12).sin() * 1.0;
            let eye = Vec3::new(
                cam_angle.cos() * cam_r,
                cam_y,
                cam_angle.sin() * cam_r,
            );

            let transparent = Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

            scene = scene
                .camera(SdfCamera::new(eye, Vec3::ZERO, 45.0))
                .sky_color(transparent)
                .ambient(0.05)
                .max_bounces(3);

            let pixmap = gpu_ctx.render(&scene, w, h, t)
                .map_err(|e| format!("SDF render failed: {e}"))?;

            driver.display_frame(&pixmap, frame_count)
                .map_err(|e| format!("display failed: {e}"))?;

            frame_count += 1;

            let elapsed = frame_start.elapsed();
            if elapsed < frame_dur {
                std::thread::sleep(frame_dur - elapsed);
            }
        }
        Ok(())
    })();

    let _ = crossterm::terminal::disable_raw_mode();

    result?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Illusion preset — renders 6 optical illusion panels inline
// ---------------------------------------------------------------------------

fn run_illusion(args: &SdfRunParams) -> Result<(), String> {
    use crossterm::event::{self, Event, KeyEventKind};
    use std::time::{Duration, Instant};

    // Auto-size from terminal dimensions for the illusion grid.
    let (w, h) = {
        let (term_cols, term_rows) =
            crossterm::terminal::size().unwrap_or((120, 40));
        let (cw, ch) = crate::display::detect_cell_size();
        let auto_w = (term_cols as u32) * u32::from(cw);
        let auto_h = (term_rows as u32).saturating_sub(4) * u32::from(ch);
        let default_res = sdf_default_res();
        // Use the params width/height only if they differ from the SDF default
        let w = if args.width != default_res && args.width != SDF_LOW_RES { args.width } else { auto_w.max(320) };
        let h = if args.height != default_res && args.height != SDF_LOW_RES { args.height } else { auto_h.max(200) };
        (w, h)
    };

    let fps = args.fps.max(1);
    let frame_dur = Duration::from_secs_f64(1.0 / fps as f64);
    let deadline = if args.duration > 0 {
        Some(Instant::now() + Duration::from_secs(args.duration))
    } else {
        None
    };



    let mut driver = display::FrameDriver::detect();
    let mut frame: u64 = 0;
    let start = Instant::now();

    // Enable raw mode so we can poll keypresses without blocking
    crossterm::terminal::enable_raw_mode().map_err(|e| e.to_string())?;

    let result = (|| -> Result<(), String> {
        loop {
            let frame_start = Instant::now();
            let t = start.elapsed().as_secs_f32();

            // Check for keypress (non-blocking)
            if event::poll(Duration::ZERO).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
                    if key.kind == KeyEventKind::Press {
                        break;
                    }
                }
            }

            // Check duration limit
            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    break;
                }
            }

            // Build and rasterize
            let canvas = build_illusions(w, h, t);
            let pixmap = Rasterizer::rasterize(&canvas)
                .map_err(|e| format!("rasterization failed: {e}"))?;
            driver.display_frame(&pixmap, frame)
                .map_err(|e| format!("display failed: {e}"))?;

            frame += 1;

            // Sleep for remainder of frame budget
            let elapsed = frame_start.elapsed();
            if elapsed < frame_dur {
                std::thread::sleep(frame_dur - elapsed);
            }
        }
        Ok(())
    })();

    // Always restore terminal state
    let _ = crossterm::terminal::disable_raw_mode();

    result?;

    Ok(())
}

// ─── Grid helper ──────────────────────────────────────────────────────────────

struct Cell {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    cx: f32,
    cy: f32,
}

fn cell(col: usize, row: usize, total_w: f32, total_h: f32, cols: usize, rows: usize) -> Cell {
    let w = total_w / cols as f32;
    let h = total_h / rows as f32;
    let x = col as f32 * w;
    let y = row as f32 * h;
    Cell {
        x,
        y,
        w,
        h,
        cx: x + w / 2.0,
        cy: y + h / 2.0,
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Scene builder — 6-panel optical illusion grid
// ═════════════════════════════════════════════════════════════════════════════

#[allow(
    clippy::suboptimal_flops,
    clippy::items_after_statements,
    clippy::cast_precision_loss,
    clippy::unreadable_literal,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn build_illusions(w: u32, h: u32, t: f32) -> PixelCanvas {
    let wf = w as f32;
    let hf = h as f32;

    let mut canvas = PixelCanvas::new(w, h).background(C::from_rgba8(10, 10, 18, 255));

    // ── (0,0) Moiré Interference ─────────────────────────────────────────
    // Two sets of concentric circles slightly offset — creates shimmering
    // interference patterns.
    {
        let c = cell(0, 0, wf, hf, 3, 2);
        let r_max = c.w.min(c.h) * 0.48;
        let offset = (t * 0.3).sin() * 8.0 + 6.0;

        let rings = (r_max / 4.0) as usize;
        for i in 0..rings {
            let r = (i as f32 + 0.5) * 4.0;
            if r > r_max {
                break;
            }
            canvas = canvas
                .circle(c.cx - offset, c.cy, r)
                .stroke(C::from_rgba8(100, 200, 255, 100), 1.5)
                .done();
        }
        for i in 0..rings {
            let r = (i as f32 + 0.5) * 4.0;
            if r > r_max {
                break;
            }
            canvas = canvas
                .circle(c.cx + offset, c.cy, r)
                .stroke(C::from_rgba8(255, 100, 200, 100), 1.5)
                .done();
        }
    }

    // ── (1,0) Hypnotic Spiral ────────────────────────────────────────────
    // Concentric rings with alternating colors — appears to pulsate.
    {
        let c = cell(1, 0, wf, hf, 3, 2);
        let r_max = c.w.min(c.h) * 0.45;
        let num_rings = 18;

        for i in (0..num_rings).rev() {
            let r = r_max * (i as f32 + 1.0) / num_rings as f32;
            let phase = (i as f32 + t * 2.0) % 3.0;
            let color = match (phase as usize) % 3 {
                0 => C::from_rgba8(20, 20, 35, 255),
                1 => C::from_rgba8(200, 50, 80, 255),
                _ => C::from_rgba8(255, 220, 100, 255),
            };
            canvas = canvas.circle(c.cx, c.cy, r).fill(color).done();
        }
        canvas = canvas.circle(c.cx, c.cy, 4.0).fill(C::WHITE).done();
    }

    // ── (2,0) Overlapping Translucent Circles (RGB blend) ────────────────
    {
        let c = cell(2, 0, wf, hf, 3, 2);
        let r = c.w.min(c.h) * 0.25;
        let spread = r * 0.45;

        canvas = canvas
            .group(Transform::identity())
            .blend_mode(BlendMode::Screen)
            .canvas(|inner| {
                inner
                    .circle(c.cx, c.cy - spread, r)
                    .fill(C::from_rgba8(255, 40, 40, 200))
                    .done()
                    .circle(c.cx - spread * 0.87, c.cy + spread * 0.5, r)
                    .fill(C::from_rgba8(40, 255, 40, 200))
                    .done()
                    .circle(c.cx + spread * 0.87, c.cy + spread * 0.5, r)
                    .fill(C::from_rgba8(40, 40, 255, 200))
                    .done()
            })
            .done();
    }

    // ── (0,1) Café Wall Illusion ──────────────────────────────────────────
    // Offset rows of black and white tiles with grey mortar lines between.
    {
        let c = cell(0, 1, wf, hf, 3, 2);
        let tile_h = (c.h / 10.0).max(6.0);
        let tile_w = tile_h * 1.8;
        let mortar = 2.0;
        let num_rows = ((c.h - mortar) / (tile_h + mortar)) as usize;
        let tiles_per_row = ((c.w + tile_w) / tile_w) as usize + 1;

        canvas = canvas
            .group(Transform::identity())
            .clip_rect(PxRect::new(c.x + 2.0, c.y + 2.0, c.w - 4.0, c.h - 4.0))
            .canvas(|mut inner| {
                inner = inner
                    .rect(c.x, c.y, c.w, c.h)
                    .fill(C::from_rgba8(128, 128, 128, 255))
                    .done();

                for row in 0..num_rows {
                    let y = c.y + mortar + row as f32 * (tile_h + mortar);
                    let offset = match row % 4 {
                        0 => 0.0,
                        1 => tile_w * 0.5,
                        2 => tile_w * 0.25,
                        _ => tile_w * 0.75,
                    };

                    for col in 0..tiles_per_row {
                        let x = c.x - tile_w + col as f32 * tile_w + offset;
                        let is_dark = col % 2 == 0;
                        let color = if is_dark {
                            C::from_rgba8(20, 20, 30, 255)
                        } else {
                            C::from_rgba8(240, 240, 245, 255)
                        };
                        inner = inner.rect(x, y, tile_w - 0.5, tile_h).fill(color).done();
                    }
                }
                inner
            })
            .done();
    }

    // ── (1,1) Rotating Arc Mandala ────────────────────────────────────────
    {
        let c = cell(1, 1, wf, hf, 3, 2);
        let r_max = c.w.min(c.h) * 0.42;
        let num_layers = 5;
        let arcs_per_layer = 8;
        let pi = std::f32::consts::PI;

        for layer in 0..num_layers {
            let r = r_max * (layer as f32 + 1.0) / num_layers as f32;
            let rotation_offset = t * (1.0 + layer as f32 * 0.3) + layer as f32 * 0.4;
            let hue_base = layer as f32 * 50.0 + t * 20.0;

            for arc_i in 0..arcs_per_layer {
                let start = (arc_i as f32 / arcs_per_layer as f32) * pi * 2.0 + rotation_offset;
                let sweep = pi / (arcs_per_layer as f32) * 1.5;

                let hue = (hue_base + arc_i as f32 * 30.0) % 360.0;
                let (cr, cg, cb) = hsl_to_rgb(hue, 0.8, 0.6);

                canvas = canvas
                    .arc(c.cx, c.cy, r, start, sweep)
                    .stroke(C::from_rgba8(cr, cg, cb, 200), 3.0)
                    .done();
            }
        }
    }

    // ── (2,1) Penrose Impossible Triangle ──────────────────────────────────
    {
        let c = cell(2, 1, wf, hf, 3, 2);
        let size = c.w.min(c.h) * 0.38;
        let thickness = size * 0.22;

        let top = (c.cx, c.cy - size * 0.65);
        let bl = (c.cx - size * 0.65, c.cy + size * 0.45);
        let br = (c.cx + size * 0.65, c.cy + size * 0.45);

        let beam_colors = [
            C::from_rgba8(70, 160, 230, 255),
            C::from_rgba8(230, 90, 70, 255),
            C::from_rgba8(80, 200, 120, 255),
        ];

        // Beam 1: top → bottom-left
        let pts1 = vec![
            (top.0, top.1),
            (top.0 - thickness * 0.5, top.1 + thickness * 0.3),
            (bl.0 + thickness * 0.1, bl.1 + thickness * 0.15),
            (bl.0, bl.1),
            (bl.0 + thickness * 0.7, bl.1 - thickness * 0.2),
            (top.0 + thickness * 0.4, top.1 + thickness * 0.6),
        ];
        canvas = canvas
            .polygon(pts1)
            .fill(beam_colors[0])
            .stroke(C::from_rgba8(30, 30, 50, 255), 1.5)
            .done();

        // Beam 2: bottom-left → bottom-right
        let pts2 = vec![
            (bl.0, bl.1),
            (bl.0 + thickness * 0.1, bl.1 + thickness * 0.15),
            (br.0 - thickness * 0.2, br.1 + thickness * 0.15),
            (br.0, br.1),
            (br.0 - thickness * 0.5, br.1 - thickness * 0.4),
            (bl.0 + thickness * 0.7, bl.1 - thickness * 0.2),
        ];
        canvas = canvas
            .polygon(pts2)
            .fill(beam_colors[1])
            .stroke(C::from_rgba8(30, 30, 50, 255), 1.5)
            .done();

        // Beam 3: bottom-right → top
        let pts3 = vec![
            (br.0, br.1),
            (br.0 - thickness * 0.2, br.1 + thickness * 0.15),
            (top.0 + thickness * 0.5, top.1 + thickness * 0.1),
            (top.0, top.1),
            (top.0 + thickness * 0.4, top.1 + thickness * 0.6),
            (br.0 - thickness * 0.5, br.1 - thickness * 0.4),
        ];
        canvas = canvas
            .polygon(pts3)
            .fill(beam_colors[2])
            .stroke(C::from_rgba8(30, 30, 50, 255), 1.5)
            .done();
    }

    canvas
}

// ---------------------------------------------------------------------------
// Gradient Descent preset — 3D loss landscape with descending particle
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
pub(crate) fn run_gradient_descent(args: &SdfRunParams) -> Result<(), String> {
    use crossterm::event::{self, Event, KeyEventKind};
    use scry_engine::scene::style::Color;
    use scry_engine::sdf::{
        Material, SdfCamera, SdfLight, SdfObject, SdfScene, SdfShape, Vec3,
    };
    use std::time::{Duration, Instant};

    let w = args.width;
    let h = args.height;

    let fps = args.fps.max(1);
    let frame_dur = Duration::from_secs_f64(1.0 / fps as f64);
    let deadline = if args.duration > 0 {
        Some(Instant::now() + Duration::from_secs(args.duration))
    } else {
        None
    };

    let mut gpu_ctx = GpuRenderCtx::new();
    let mut driver = display::FrameDriver::detect();
    let mut frame_count: u64 = 0;
    let start = Instant::now();

    // Pre-defined "valley" positions for the loss landscape
    let valleys: [(f32, f32, f32); 7] = [
        (0.0, -0.6, 0.0),    // global minimum (deepest)
        (2.0, -0.3, 1.5),    // local min 1
        (-1.8, -0.25, -1.2), // local min 2
        (1.0, -0.2, -2.0),   // local min 3
        (-2.5, -0.15, 2.0),  // saddle region
        (0.5, -0.1, 2.5),    // shallow basin
        (-1.0, -0.2, 1.8),   // local min 4
    ];

    crossterm::terminal::enable_raw_mode().map_err(|e| e.to_string())?;

    let result = (|| -> Result<(), String> {
        loop {
            let frame_start = Instant::now();
            let t = start.elapsed().as_secs_f32();

            if event::poll(Duration::ZERO).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
                    if key.kind == KeyEventKind::Press {
                        break;
                    }
                }
            }

            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    break;
                }
            }

            let _pi = std::f32::consts::PI;

            // Build the loss landscape from smooth-blended bowl shapes
            // Start with a large base bowl
            let mut terrain_shape = SdfShape::Sphere { radius: 0.5 };

            for (i, &(vx, vy, vz)) in valleys.iter().enumerate() {
                let depth = if i == 0 { 0.7 } else { 0.3 + 0.1 * (i as f32) };
                terrain_shape = SdfShape::SmoothBlend {
                    a: std::boxed::Box::new(terrain_shape),
                    b: std::boxed::Box::new(SdfShape::Sphere { radius: depth }),
                    b_offset: Vec3::new(vx, vy, vz),
                    k: 0.8,
                };
            }

            // Slow rotation for visual interest
            let qy = scry_engine::math3d::Quaternion::from_axis_angle(
                Vec3::new(0.0, 1.0, 0.0),
                t * 0.08,
            );

            let landscape = SdfObject::new(
                terrain_shape,
                Material::rainbow_animated(t * 0.15),
            )
            .at(Vec3::new(0.0, -1.0, 0.0))
            .orient(qy);

            // Particle descent: spiral path converging to global minimum
            // Cycle: 12s descent, 2s hold, then reset
            let cycle = 14.0_f32;
            let phase = (t % cycle) / cycle;
            let smoothstep = phase * phase * (3.0 - 2.0 * phase);

            // Spiral path from outer rim to center
            let descent_radius = 3.0 * (1.0 - smoothstep);
            let descent_angle = t * 2.0;
            let descent_y = -0.3 - smoothstep * 0.5; // drop into the bowl

            let particle_pos = Vec3::new(
                descent_radius * descent_angle.cos(),
                descent_y + (t * 3.0).sin() * 0.05 * (1.0 - smoothstep), // tiny bounce
                descent_radius * descent_angle.sin(),
            );

            let particle = SdfObject::new(
                SdfShape::Sphere { radius: 0.15 + (t * 4.0).sin() * 0.03 },
                Material::fire(),
            )
            .at(particle_pos);

            // Trail: small fading spheres behind the particle
            let trail_count = 5;
            let mut scene = SdfScene::new().object(landscape).object(particle);

            for i in 1..=trail_count {
                let trail_t = t - (i as f32) * 0.15;
                let trail_phase = ((trail_t % cycle).max(0.0)) / cycle;
                let trail_smooth = trail_phase * trail_phase * (3.0 - 2.0 * trail_phase);
                let tr = 3.0 * (1.0 - trail_smooth);
                let ta = trail_t * 2.0;
                let ty = -0.3 - trail_smooth * 0.5;

                let alpha = 1.0 - (i as f32 / (trail_count as f32 + 1.0));
                let trail_sphere = SdfObject::new(
                    SdfShape::Sphere { radius: 0.06 * alpha },
                    Material::Solid {
                        color: Color {
                            r: 1.0,
                            g: 0.4 * alpha,
                            b: 0.1 * alpha,
                            a: 1.0,
                        },
                        reflectivity: 0.0,
                        specular: 8.0,
                    },
                )
                .at(Vec3::new(tr * ta.cos(), ty, tr * ta.sin()));

                scene = scene.object(trail_sphere);
            }

            // Subtle ground reference plane
            let ground = SdfObject::new(
                SdfShape::Plane,
                Material::matte(Color { r: 0.05, g: 0.05, b: 0.08, a: 1.0 }),
            )
            .at(Vec3::new(0.0, -2.0, 0.0));

            // Orbiting camera from above
            let cam_angle = t * 0.2;
            let cam_r = 7.0;
            let cam_y = 4.5 + (t * 0.12).sin() * 0.5;
            let eye = Vec3::new(
                cam_angle.cos() * cam_r,
                cam_y,
                cam_angle.sin() * cam_r,
            );

            let scene = scene
                .object(ground)
                .light(SdfLight::new(
                    Vec3::new(5.0, 10.0, 5.0),
                    Color::WHITE,
                    1.3,
                ))
                .light(SdfLight::new(
                    Vec3::new(-4.0, 6.0, -3.0),
                    Color::from_rgba8(100, 180, 255, 255),
                    0.6,
                ))
                .light(SdfLight::new(
                    Vec3::new(0.0, 2.0, 0.0),
                    Color::from_rgba8(255, 160, 60, 255),
                    0.4,
                ))
                .camera(SdfCamera::new(eye, Vec3::new(0.0, -0.5, 0.0), 50.0))
                .ambient(0.08)
                .max_bounces(1);

            let pixmap = gpu_ctx.render(&scene, w, h, t)
                .map_err(|e| format!("SDF render failed: {e}"))?;

            driver.display_frame(&pixmap, frame_count)
                .map_err(|e| format!("display failed: {e}"))?;

            frame_count += 1;

            let elapsed = frame_start.elapsed();
            if elapsed < frame_dur {
                std::thread::sleep(frame_dur - elapsed);
            }
        }
        Ok(())
    })();

    let _ = crossterm::terminal::disable_raw_mode();

    result?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Neural Net preset — layered network with signal propagation pulse
// Uses SDF ray marcher with a compact network to keep object count low.
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
pub(crate) fn run_neural_net(args: &SdfRunParams) -> Result<(), String> {
    use crossterm::event::{self, Event, KeyEventKind};
    use scry_engine::scene::style::Color;
    use scry_engine::sdf::{
        Material, SdfCamera, SdfLight, SdfObject, SdfScene, SdfShape, Vec3,
    };
    use std::time::{Duration, Instant};

    let w = args.width;
    let h = args.height;

    let fps = args.fps.max(1);
    let frame_dur = Duration::from_secs_f64(1.0 / fps as f64);
    let deadline = if args.duration > 0 {
        Some(Instant::now() + Duration::from_secs(args.duration))
    } else {
        None
    };

    let mut gpu_ctx = GpuRenderCtx::new();
    let mut driver = display::FrameDriver::detect();
    let mut frame_count: u64 = 0;
    let start = Instant::now();

    // Compact network: 10 neurons, ~23 connections = ~33 objects total
    let layer_sizes: [usize; 4] = [2, 4, 3, 1];
    let layer_x: [f32; 4] = [-3.0, -1.0, 1.0, 3.0];
    let neuron_radius = 0.25;

    // Pre-compute neuron positions (y = vertical spread per layer, z = 0)
    let mut neuron_positions: Vec<Vec<Vec3>> = Vec::new();
    for (li, &size) in layer_sizes.iter().enumerate() {
        let mut positions = Vec::new();
        let spread = (size as f32 - 1.0) * 0.8;
        for ni in 0..size {
            let y = if size == 1 {
                0.0
            } else {
                -spread / 2.0 + (ni as f32 / (size as f32 - 1.0)) * spread
            };
            positions.push(Vec3::new(layer_x[li], y, 0.0));
        }
        neuron_positions.push(positions);
    }

    crossterm::terminal::enable_raw_mode().map_err(|e| e.to_string())?;

    let result = (|| -> Result<(), String> {
        loop {
            let frame_start = Instant::now();
            let t = start.elapsed().as_secs_f32();

            if event::poll(Duration::ZERO).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
                    if key.kind == KeyEventKind::Press {
                        break;
                    }
                }
            }

            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    break;
                }
            }

            let pi = std::f32::consts::PI;

            // Signal wavefront: sweeps left-to-right, repeating every 6s
            let cycle = 6.0_f32;
            let wave_x = -4.0 + (t % cycle) / cycle * 10.0;

            let mut scene = SdfScene::new();

            // Draw neurons as spheres
            for layer in &neuron_positions {
                for &pos in layer {
                    let dist_to_wave = (pos.x - wave_x).abs();
                    let activation = (1.0 - dist_to_wave * 0.6).max(0.0);
                    let pulse_r = neuron_radius * (1.0 + activation * 0.3);

                    let mat = if activation > 0.3 {
                        Material::Solid {
                            color: Color {
                                r: 0.3 + activation * 0.7,
                                g: 0.6 + activation * 0.4,
                                b: 1.0,
                                a: 1.0,
                            },
                            reflectivity: 0.3,
                            specular: 64.0,
                        }
                    } else {
                        Material::glass(
                            Color::from_rgba8(140, 180, 255, 255),
                            1.4,
                        )
                    };

                    let neuron = SdfObject::new(
                        SdfShape::Sphere { radius: pulse_r },
                        mat,
                    )
                    .at(pos);

                    scene = scene.object(neuron);
                }
            }

            // Draw connections as thin capsules between adjacent layers
            for li in 0..layer_sizes.len() - 1 {
                let next_li = li + 1;
                for &from_pos in &neuron_positions[li] {
                    for &to_pos in &neuron_positions[next_li] {
                        let mid = Vec3::new(
                            (from_pos.x + to_pos.x) * 0.5,
                            (from_pos.y + to_pos.y) * 0.5,
                            (from_pos.z + to_pos.z) * 0.5,
                        );
                        let dx = to_pos.x - from_pos.x;
                        let dy = to_pos.y - from_pos.y;
                        let length = (dx * dx + dy * dy).sqrt();

                        let conn_wave_dist = (mid.x - wave_x).abs();
                        let conn_activation = (1.0 - conn_wave_dist * 0.5).max(0.0);

                        let conn_hue = (t * 0.2 + li as f32 * 0.25) % 1.0;
                        let brightness = 0.3 + conn_activation * 0.5;

                        let conn_mat = Material::Solid {
                            color: Color {
                                r: brightness * (1.0 - conn_hue),
                                g: brightness * conn_hue.min(1.0 - conn_hue) * 2.0,
                                b: brightness * conn_hue,
                                a: 1.0,
                            },
                            reflectivity: 0.1,
                            specular: 16.0,
                        };

                        let angle = dy.atan2(dx);
                        let conn = SdfObject::new(
                            SdfShape::Capsule {
                                radius: 0.03 + conn_activation * 0.02,
                                half_height: length * 0.45,
                            },
                            conn_mat,
                        )
                        .at(mid)
                        .rotate(Vec3::new(0.0, 0.0, 1.0), angle - pi * 0.5);

                        scene = scene.object(conn);
                    }
                }
            }

            // Orbiting camera
            let cam_angle = t * 0.15;
            let cam_r = 8.0;
            let cam_y = 1.5 + (t * 0.1).sin() * 0.8;
            let eye = Vec3::new(
                cam_angle.cos() * cam_r,
                cam_y,
                cam_angle.sin() * cam_r,
            );

            let transparent = Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

            let scene = scene
                .light(SdfLight::new(
                    Vec3::new(5.0, 8.0, 6.0),
                    Color::WHITE,
                    1.3,
                ))
                .light(SdfLight::new(
                    Vec3::new(-5.0, 4.0, -4.0),
                    Color::from_rgba8(80, 140, 255, 255),
                    0.7,
                ))
                .light(SdfLight::new(
                    Vec3::new(wave_x, 3.0, 2.0),
                    Color::from_rgba8(255, 200, 100, 255),
                    0.5,
                ))
                .camera(SdfCamera::new(eye, Vec3::ZERO, 45.0))
                .sky_color(transparent)
                .ambient(0.08)
                .max_bounces(1);

            let pixmap = gpu_ctx.render(&scene, w, h, t)
                .map_err(|e| format!("SDF render failed: {e}"))?;

            driver.display_frame(&pixmap, frame_count)
                .map_err(|e| format!("display failed: {e}"))?;

            frame_count += 1;

            let elapsed = frame_start.elapsed();
            if elapsed < frame_dur {
                std::thread::sleep(frame_dur - elapsed);
            }
        }
        Ok(())
    })();

    let _ = crossterm::terminal::disable_raw_mode();

    result?;

    Ok(())
}

// ---------------------------------------------------------------------------
// K-Means preset — 3D point cloud with converging centroids
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
pub(crate) fn run_kmeans(args: &SdfRunParams) -> Result<(), String> {
    use crossterm::event::{self, Event, KeyEventKind};
    use scry_engine::scene::style::Color;
    use scry_engine::sdf::{
        Material, SdfCamera, SdfLight, SdfObject, SdfScene, SdfShape, Vec3,
    };
    use std::time::{Duration, Instant};

    let w = args.width;
    let h = args.height;

    let fps = args.fps.max(1);
    let frame_dur = Duration::from_secs_f64(1.0 / fps as f64);
    let deadline = if args.duration > 0 {
        Some(Instant::now() + Duration::from_secs(args.duration))
    } else {
        None
    };

    let mut gpu_ctx = GpuRenderCtx::new();
    let mut driver = display::FrameDriver::detect();
    let mut frame_count: u64 = 0;
    let start = Instant::now();

    // Cluster colors: warm red, teal green, electric blue
    let cluster_colors: [Color; 3] = [
        Color { r: 1.0, g: 0.25, b: 0.2, a: 1.0 },   // red
        Color { r: 0.2, g: 0.9, b: 0.5, a: 1.0 },     // green
        Color { r: 0.2, g: 0.4, b: 1.0, a: 1.0 },     // blue
    ];

    // True cluster centers
    let true_centers: [Vec3; 3] = [
        Vec3::new(-2.0, 0.5, -1.0),
        Vec3::new(1.5, -0.5, 1.5),
        Vec3::new(0.5, 1.5, -2.0),
    ];

    // Initial (scattered) centroid positions
    let init_centroids: [Vec3; 3] = [
        Vec3::new(2.0, 2.0, 2.0),
        Vec3::new(-2.5, -2.0, 0.0),
        Vec3::new(0.0, -1.0, -3.0),
    ];

    // Pre-generate data points around true centers using deterministic offsets
    // 7 points per cluster = 21 total
    let offsets: [(f32, f32, f32); 7] = [
        (0.3, 0.2, -0.1),
        (-0.4, 0.1, 0.3),
        (0.1, -0.3, -0.4),
        (-0.2, -0.1, 0.5),
        (0.5, 0.4, 0.2),
        (-0.3, 0.5, -0.2),
        (0.2, -0.4, 0.1),
    ];

    let mut data_points: Vec<(Vec3, usize)> = Vec::new(); // (position, cluster_idx)
    for (ci, &center) in true_centers.iter().enumerate() {
        for &(ox, oy, oz) in &offsets {
            let jitter = (ci as f32 + 1.0) * 0.15; // slight per-cluster jitter variation
            data_points.push((
                Vec3::new(
                    center.x + ox * (1.0 + jitter),
                    center.y + oy * (1.0 + jitter),
                    center.z + oz * (1.0 + jitter),
                ),
                ci,
            ));
        }
    }

    crossterm::terminal::enable_raw_mode().map_err(|e| e.to_string())?;

    let result = (|| -> Result<(), String> {
        loop {
            let frame_start = Instant::now();
            let t = start.elapsed().as_secs_f32();

            if event::poll(Duration::ZERO).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
                    if key.kind == KeyEventKind::Press {
                        break;
                    }
                }
            }

            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    break;
                }
            }

            // Phase: 10s convergence, 4s hold, then reset
            let cycle = 14.0_f32;
            let raw_phase = (t % cycle) / 10.0; // converge over first 10s
            let phase = raw_phase.min(1.0);
            let eased = phase * phase * (3.0 - 2.0 * phase); // smoothstep

            // Current centroid positions: lerp from init to true center
            let centroids: Vec<Vec3> = (0..3)
                .map(|i| {
                    Vec3::new(
                        init_centroids[i].x + (true_centers[i].x - init_centroids[i].x) * eased,
                        init_centroids[i].y + (true_centers[i].y - init_centroids[i].y) * eased,
                        init_centroids[i].z + (true_centers[i].z - init_centroids[i].z) * eased,
                    )
                })
                .collect();

            let mut scene = SdfScene::new();

            // Add data points — colored by nearest centroid
            for &(pt, _true_cluster) in &data_points {
                // Find nearest centroid
                let mut nearest_ci = 0;
                let mut nearest_dist = f32::INFINITY;
                for (ci, c) in centroids.iter().enumerate() {
                    let d = (pt.x - c.x).powi(2) + (pt.y - c.y).powi(2) + (pt.z - c.z).powi(2);
                    if d < nearest_dist {
                        nearest_dist = d;
                        nearest_ci = ci;
                    }
                }

                let point_color = cluster_colors[nearest_ci];
                let data_sphere = SdfObject::new(
                    SdfShape::Sphere { radius: 0.1 },
                    Material::Solid {
                        color: point_color,
                        reflectivity: 0.15,
                        specular: 32.0,
                    },
                )
                .at(pt);

                scene = scene.object(data_sphere);
            }

            // Add centroid spheres — larger, glass material
            for (ci, &cpos) in centroids.iter().enumerate() {
                let pulse = 1.0 + (t * 2.0 + ci as f32).sin() * 0.08;
                let centroid_sphere = SdfObject::new(
                    SdfShape::Sphere { radius: 0.3 * pulse },
                    Material::glass_dispersive(
                        cluster_colors[ci],
                        1.5,
                        0.02,
                    ),
                )
                .at(cpos);

                scene = scene.object(centroid_sphere);

                // Add a faint connecting "pull" line from centroid toward true center
                // (visible during convergence)
                if phase < 0.98 {
                    let pull_alpha = 1.0 - phase;
                    let pull_sphere = SdfObject::new(
                        SdfShape::Sphere { radius: 0.04 * pull_alpha },
                        Material::Solid {
                            color: Color {
                                r: cluster_colors[ci].r * 0.5,
                                g: cluster_colors[ci].g * 0.5,
                                b: cluster_colors[ci].b * 0.5,
                                a: 1.0,
                            },
                            reflectivity: 0.0,
                            specular: 8.0,
                        },
                    )
                    .at(true_centers[ci]);

                    scene = scene.object(pull_sphere);
                }
            }

            // Slow tumble of the whole scene for 3D effect
            let cam_angle = t * 0.2;
            let cam_r = 7.0;
            let cam_y = 3.0 + (t * 0.15).sin() * 1.0;
            let eye = Vec3::new(
                cam_angle.cos() * cam_r,
                cam_y,
                cam_angle.sin() * cam_r,
            );

            let transparent = Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

            let scene = scene
                .light(SdfLight::new(
                    Vec3::new(6.0, 8.0, 5.0),
                    Color::WHITE,
                    1.2,
                ))
                .light(SdfLight::new(
                    Vec3::new(-5.0, 4.0, -4.0),
                    Color::from_rgba8(120, 160, 255, 255),
                    0.6,
                ))
                .light(SdfLight::new(
                    Vec3::new(0.0, -2.0, 5.0),
                    Color::from_rgba8(255, 180, 120, 255),
                    0.4,
                ))
                .camera(SdfCamera::new(eye, Vec3::ZERO, 50.0))
                .sky_color(transparent)
                .ambient(0.10)
                .max_bounces(2);

            let pixmap = gpu_ctx.render(&scene, w, h, t)
                .map_err(|e| format!("SDF render failed: {e}"))?;

            driver.display_frame(&pixmap, frame_count)
                .map_err(|e| format!("display failed: {e}"))?;

            frame_count += 1;

            let elapsed = frame_start.elapsed();
            if elapsed < frame_dur {
                std::thread::sleep(frame_dur - elapsed);
            }
        }
        Ok(())
    })();

    let _ = crossterm::terminal::disable_raw_mode();

    result?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Text preset — GPU SDF 3D extruded text with custom message
// ---------------------------------------------------------------------------

/// Options for the text preset, collected from CLI flags.
pub(crate) struct TextOptions {
    pub message: String,
    pub material: crate::see::TextMaterial,
    pub animate: bool,
    pub wiggle: bool,
    pub warp: bool,
}

#[allow(clippy::cast_precision_loss)]
pub(crate) fn run_text(args: &SdfRunParams, opts: &TextOptions) -> Result<(), String> {
    use crossterm::event::{self, Event, KeyCode, KeyEventKind};
    use scry_engine::scene::style::Color;
    use scry_engine::sdf::{
        Material, SdfCamera, SdfLight, SdfObject, SdfScene, SdfShape, Vec3,
    };
    use std::time::{Duration, Instant};

    // Uppercase the message for a bolder 3D look.
    let text = opts.message.to_uppercase();

    // Load the bundled Inter-Bold font (same font the chart library uses).
    const FONT: &[u8] = include_bytes!("../../scry-chart/src/fonts/Inter-Bold.ttf");

    // Fixed font size — camera distance adapts to the text's actual width.
    let font_size = 1.4_f32;

    // Build the Text3D shape.
    let text_shape = SdfShape::text_3d(FONT, &text, font_size, font_size * 0.4)
        .ok_or_else(|| format!("Failed to build 3D text for \"{text}\" — font may not support these characters"))?;

    // Extract actual bounding-box width from the layout for camera framing.
    let text_width = match &text_shape {
        SdfShape::Text3D { layout, .. } => layout.total_width,
        _ => unreachable!(),
    };

    let w = args.width;
    let h = args.height;

    // Helper: build the material for this frame
    let build_material = |t: f32| -> Material {
        match opts.material {
            crate::see::TextMaterial::Rainbow => Material::Rainbow {
                saturation: 0.95,
                lightness: 0.55,
                hue_offset: t * 0.5,
                specular: 128.0,
            },
            crate::see::TextMaterial::Chrome => Material::mirror(
                Color::from_rgba8(220, 220, 230, 255),
                0.9,
            ),
            crate::see::TextMaterial::Glass => Material::glass_dispersive(
                Color::from_rgba8(220, 240, 255, 255),
                1.5,
                0.03,
            ),
            crate::see::TextMaterial::Fire => Material::Fire {
                intensity: 2.0,
                noise_scale: 2.0,
                speed: 1.0,
            },
            crate::see::TextMaterial::Matte => Material::Solid {
                color: Color::from_rgba8(240, 235, 230, 255),
                reflectivity: 0.05,
                specular: 64.0,
            },
        }
    };

    // Helper: build a scene for given camera angles and time
    let build_scene = |yaw: f32, pitch: f32, t: f32| -> SdfScene {
        let mat = build_material(t);

        let mut obj = SdfObject::new(text_shape.clone(), mat)
            .at(Vec3::new(0.0, 1.0, 0.0));

        // No base rotation needed — the text SDF faces -Z by default,
        // and we place the camera at -Z so it reads left-to-right.
        let mut q = scry_engine::math3d::Quaternion::from_axis_angle(
            Vec3::new(0.0, 1.0, 0.0), 0.0,
        );

        if yaw.abs() > 0.001 || pitch.abs() > 0.001 {
            let qy = scry_engine::math3d::Quaternion::from_axis_angle(
                Vec3::new(0.0, 1.0, 0.0), yaw,
            );
            let qx = scry_engine::math3d::Quaternion::from_axis_angle(
                Vec3::new(1.0, 0.0, 0.0), pitch,
            );
            q = qy * qx * q;
        }

        // Apply wiggle: gentle Z-axis oscillation composed on top
        if opts.wiggle {
            let wiggle_angle = (t * 2.0).sin() * 0.15;
            let qw = scry_engine::math3d::Quaternion::from_axis_angle(
                Vec3::new(0.0, 0.0, 1.0), wiggle_angle,
            );
            q = qw * q;
        }

        obj = obj.orient(q);

        // Base distance gives consistent letter height; only pull back if
        // the text is too wide to fit the horizontal FOV.
        let base_cam_r = 5.0_f32;
        let half_fov = (50.0_f32 / 2.0).to_radians();
        let min_cam_for_width = (text_width * 0.5) / half_fov.tan() * 1.3;
        let cam_r = base_cam_r.max(min_cam_for_width);
        let cam_y = 2.2;

        let mut scene = SdfScene::new()
            .object(obj)
            // Warm key light
            .light(SdfLight::new(
                Vec3::new(5.0, 8.0, 5.0),
                Color::from_rgba8(255, 240, 220, 255),
                0.9,
            ))
            // Cool fill light
            .light(SdfLight::new(
                Vec3::new(-4.0, 6.0, -2.0),
                Color::from_rgba8(150, 180, 255, 255),
                0.4,
            ))
            // Back accent
            .light(SdfLight::new(
                Vec3::new(0.0, 4.0, -5.0),
                Color::from_rgba8(255, 200, 160, 255),
                0.3,
            ))
            // Camera at +Z so the right vector is +X, matching glyph layout direction
            .camera(SdfCamera::new(
                Vec3::new(0.0, cam_y, cam_r),
                Vec3::new(0.0, 0.8, 0.0),
                50.0,
            ))
            .sky_color(Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 })
            .ambient(0.08)
            .max_bounces(2);

        // Warp: add small distortion spheres around the text that pulse
        if opts.warp {
            let num_warp = 6;
            for i in 0..num_warp {
                let phase = (i as f32 / num_warp as f32) * std::f32::consts::PI * 2.0;
                let warp_r = 0.15 + (t * 1.5 + phase).sin() * 0.08;
                let wx = (phase + t * 0.5).cos() * 2.5;
                let wy = 1.0 + (t * 1.2 + phase).sin() * 0.4;
                let wz = (phase + t * 0.5).sin() * 1.2;
                let (hr, hg, hb) = hsl_to_rgb(
                    ((i as f32 / num_warp as f32) * 360.0 + t * 40.0) % 360.0,
                    0.9, 0.6,
                );
                scene = scene.object(
                    SdfObject::new(
                        SdfShape::Sphere { radius: warp_r },
                        Material::Solid {
                            color: Color::from_rgba8(hr, hg, hb, 255),
                            reflectivity: 0.6,
                            specular: 96.0,
                        },
                    )
                    .at(Vec3::new(wx, wy, wz)),
                );
            }
        }

        scene
    };

    let mut gpu_ctx = GpuRenderCtx::new();

    // ── Static mode (default): render one frame and display inline ──
    if !opts.animate {
        let t = if opts.wiggle || opts.warp { 0.5 } else { 0.0 };
        let scene = build_scene(0.0, 0.0, t);

        let pixmap = gpu_ctx.render(&scene, w, h, t)
            .map_err(|e| format!("SDF render failed: {e}"))?;

        let mut driver = display::FrameDriver::detect();
        driver.display_static(&pixmap)?;

        return Ok(());
    }

    // ── Animate mode: live loop with arrow-key rotation ──
    let fps = args.fps.max(1);
    let frame_dur = Duration::from_secs_f64(1.0 / fps as f64);
    let deadline = if args.duration > 0 {
        Some(Instant::now() + Duration::from_secs(args.duration))
    } else {
        None
    };

    let mut driver = display::FrameDriver::detect();
    let mut frame_count: u64 = 0;
    let start = Instant::now();
    let mut yaw: f32 = 0.0;
    let mut pitch: f32 = 0.0;
    let rot_speed: f32 = 0.08;

    crossterm::terminal::enable_raw_mode().map_err(|e| e.to_string())?;

    let result = (|| -> Result<(), String> {
        loop {
            let frame_start = Instant::now();
            let t = start.elapsed().as_secs_f32();

            // Handle input: arrow keys rotate, q/Esc quits
            if event::poll(Duration::ZERO).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
                    if key.kind == KeyEventKind::Press {
                        match key.code {
                            KeyCode::Left => yaw -= rot_speed,
                            KeyCode::Right => yaw += rot_speed,
                            KeyCode::Up => pitch -= rot_speed,
                            KeyCode::Down => pitch += rot_speed,
                            KeyCode::Char('q') | KeyCode::Esc => break,
                            _ => {}
                        }
                    }
                }
            }

            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    break;
                }
            }

            let scene = build_scene(yaw, pitch, t);

            let pixmap = gpu_ctx.render(&scene, w, h, t)
                .map_err(|e| format!("SDF render failed: {e}"))?;

            driver.display_frame(&pixmap, frame_count)
                .map_err(|e| format!("display failed: {e}"))?;

            frame_count += 1;

            let elapsed = frame_start.elapsed();
            if elapsed < frame_dur {
                std::thread::sleep(frame_dur - elapsed);
            }
        }
        Ok(())
    })();

    let _ = crossterm::terminal::disable_raw_mode();

    result?;

    Ok(())
}


// ─── Utility: HSL to RGB ──────────────────────────────────────────────────────

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::suboptimal_flops
)]
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h2 = h / 60.0;
    let x = c * (1.0 - (h2 % 2.0 - 1.0).abs());
    let (r1, g1, b1) = match h2 as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = l - c / 2.0;
    (
        ((r1 + m) * 255.0) as u8,
        ((g1 + m) * 255.0) as u8,
        ((b1 + m) * 255.0) as u8,
    )
}

// ---------------------------------------------------------------------------
// GodRays preset — volumetric light shafts through Menger sponge
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
pub(crate) fn run_godrays(args: &SdfRunParams) -> Result<(), String> {
    use crossterm::event::{self, Event, KeyEventKind};
    use scry_engine::scene::style::Color;
    use scry_engine::sdf::{
        Material, SdfCamera, SdfLight, SdfObject, SdfScene, SdfShape, Vec3,
    };
    use std::time::{Duration, Instant};

    let w = args.width;
    let h = args.height;

    let fps = args.fps.max(1);
    let frame_dur = Duration::from_secs_f64(1.0 / fps as f64);
    let deadline = if args.duration > 0 {
        Some(Instant::now() + Duration::from_secs(args.duration))
    } else {
        None
    };

    let mut gpu_ctx = GpuRenderCtx::new();
    let mut driver = display::FrameDriver::detect();
    let mut frame_count: u64 = 0;
    let start = Instant::now();

    eprint!("\x1b[?25l"); // hide cursor
    if scry_engine::scry_debug_enabled() {
        eprintln!("godrays: rendering first frame ({w}x{h})… press any key to quit");
    }

    crossterm::terminal::enable_raw_mode().map_err(|e| e.to_string())?;

    let result = (|| -> Result<(), String> {
        loop {
            let frame_start = Instant::now();
            let t = start.elapsed().as_secs_f32();

            // Drain all pending input events — break on any key press
            while event::poll(Duration::ZERO).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
                    if key.kind == KeyEventKind::Press {
                        return Ok(());
                    }
                } else {
                    let _ = event::read(); // consume non-key events
                }
            }

            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    break;
                }
            }

            // Slowly rotating Menger sponge — light shafts stream through the holes
            let qy = scry_engine::math3d::Quaternion::from_axis_angle(
                Vec3::new(0.0, 1.0, 0.0),
                t * 0.15,
            );
            let qx = scry_engine::math3d::Quaternion::from_axis_angle(
                Vec3::new(1.0, 0.0, 0.0),
                t * 0.08,
            );
            let orientation = qy * qx;

            let sponge = SdfObject::new(
                SdfShape::MengerSponge { iterations: 2 },
                Material::matte(Color::from_rgba8(180, 170, 160, 255)),
            )
            .at(Vec3::ZERO)
            .orient(orientation);

            // Bright point light positioned behind the sponge
            let light_angle = t * 0.3;
            let light_pos = Vec3::new(
                light_angle.sin() * 6.0,
                3.0,
                -4.0 + light_angle.cos() * 2.0,
            );

            // Camera orbiting in front
            let cam_angle = t * 0.12;
            let cam_r = 5.0;
            let eye = Vec3::new(
                cam_angle.cos() * cam_r,
                2.0 + (t * 0.2).sin() * 0.5,
                cam_angle.sin() * cam_r,
            );

            let transparent = Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

            let scene = SdfScene::new()
                .object(sponge)
                .light(SdfLight::new(
                    light_pos,
                    Color::from_rgba8(255, 240, 200, 255),
                    2.5,
                ))
                .light(SdfLight::new(
                    Vec3::new(-3.0, 5.0, 3.0),
                    Color::from_rgba8(100, 140, 255, 255),
                    0.4,
                ))
                .camera(SdfCamera::new(eye, Vec3::ZERO, 45.0))
                .sky_color(transparent)
                .ambient(0.04)
                .max_bounces(1)
                .god_rays(0.4, 12);

            let pixmap = gpu_ctx.render(&scene, w, h, t)
                .map_err(|e| format!("SDF render failed: {e}"))?;

            driver.display_frame(&pixmap, frame_count)
                .map_err(|e| format!("display failed: {e}"))?;

            gpu_ctx.flush(); // overlap GPU compute with terminal I/O
            frame_count += 1;

            let elapsed = frame_start.elapsed();
            if elapsed < frame_dur {
                std::thread::sleep(frame_dur - elapsed);
            }
        }
        Ok(())
    })();

    let _ = crossterm::terminal::disable_raw_mode();
    eprint!("\x1b[?25h"); // restore cursor

    result?;
    Ok(())
}

// ---------------------------------------------------------------------------
// SSS preset — translucent subsurface scattering demo (jade + wax)
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
pub(crate) fn run_sss(args: &SdfRunParams) -> Result<(), String> {
    use crossterm::event::{self, Event, KeyEventKind};
    use scry_engine::scene::style::Color;
    use scry_engine::sdf::{
        Material, SdfCamera, SdfLight, SdfObject, SdfScene, SdfShape, Vec3,
    };
    use std::time::{Duration, Instant};

    let w = args.width;
    let h = args.height;

    let fps = args.fps.max(1);
    let frame_dur = Duration::from_secs_f64(1.0 / fps as f64);
    let deadline = if args.duration > 0 {
        Some(Instant::now() + Duration::from_secs(args.duration))
    } else {
        None
    };

    let mut gpu_ctx = GpuRenderCtx::new();
    let mut driver = display::FrameDriver::detect();
    let mut frame_count: u64 = 0;
    let start = Instant::now();

    crossterm::terminal::enable_raw_mode().map_err(|e| e.to_string())?;

    let result = (|| -> Result<(), String> {
        loop {
            let frame_start = Instant::now();
            let t = start.elapsed().as_secs_f32();

            if event::poll(Duration::ZERO).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
                    if key.kind == KeyEventKind::Press {
                        break;
                    }
                }
            }

            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    break;
                }
            }

            // Jade sphere — green surface, warm yellow-green back-illumination
            let jade = SdfObject::new(
                SdfShape::Sphere { radius: 1.0 },
                Material::Subsurface {
                    color: Color::from_rgba8(30, 120, 60, 255),
                    scatter_color: Color::from_rgba8(180, 255, 100, 255),
                    thickness: 0.8,
                    specular: 48.0,
                },
            )
            .at(Vec3::new(-1.5, 1.0, 0.0));

            // Wax capsule — warm amber body, bright orange scatter
            let wax = SdfObject::new(
                SdfShape::Capsule {
                    radius: 0.6,
                    half_height: 0.8,
                },
                Material::Subsurface {
                    color: Color::from_rgba8(220, 180, 100, 255),
                    scatter_color: Color::from_rgba8(255, 120, 40, 255),
                    thickness: 0.6,
                    specular: 24.0,
                },
            )
            .at(Vec3::new(1.5, 1.0, 0.0));

            // Marble torus — cool white with pinkish scatter
            let qr = scry_engine::math3d::Quaternion::from_axis_angle(
                Vec3::new(1.0, 0.0, 0.0),
                t * 0.3,
            );
            let marble = SdfObject::new(
                SdfShape::Torus {
                    major: 0.8,
                    minor: 0.25,
                },
                Material::Subsurface {
                    color: Color::from_rgba8(230, 220, 210, 255),
                    scatter_color: Color::from_rgba8(255, 180, 160, 255),
                    thickness: 0.4,
                    specular: 64.0,
                },
            )
            .at(Vec3::new(0.0, 1.0, -2.0))
            .orient(qr);

            // Dark ground plane
            let ground = SdfObject::new(
                SdfShape::Plane,
                Material::matte(Color::from_rgba8(40, 40, 45, 255)),
            );

            // Strong back-light to show off SSS + fill light
            let cam_angle = t * 0.15;
            let cam_r = 6.0;
            let eye = Vec3::new(
                cam_angle.cos() * cam_r,
                3.0 + (t * 0.2).sin() * 0.3,
                cam_angle.sin() * cam_r,
            );

            let transparent = Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

            let scene = SdfScene::new()
                .object(jade)
                .object(wax)
                .object(marble)
                .object(ground)
                .light(SdfLight::new(
                    Vec3::new(0.0, 4.0, -5.0),
                    Color::from_rgba8(255, 240, 220, 255),
                    2.0,
                ))
                .light(SdfLight::new(
                    Vec3::new(4.0, 6.0, 4.0),
                    Color::WHITE,
                    0.8,
                ))
                .camera(SdfCamera::new(eye, Vec3::new(0.0, 1.0, 0.0), 45.0))
                .sky_color(transparent)
                .ambient(0.06)
                .max_bounces(1);

            let pixmap = gpu_ctx.render(&scene, w, h, t)
                .map_err(|e| format!("SDF render failed: {e}"))?;

            driver.display_frame(&pixmap, frame_count)
                .map_err(|e| format!("display failed: {e}"))?;

            frame_count += 1;

            let elapsed = frame_start.elapsed();
            if elapsed < frame_dur {
                std::thread::sleep(frame_dur - elapsed);
            }
        }
        Ok(())
    })();

    let _ = crossterm::terminal::disable_raw_mode();

    result?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Morph preset — animated sphere ↔ torus shape morphing
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
pub(crate) fn run_morph(args: &SdfRunParams) -> Result<(), String> {
    use crossterm::event::{self, Event, KeyEventKind};
    use scry_engine::scene::style::Color;
    use scry_engine::sdf::{
        Material, SdfCamera, SdfLight, SdfObject, SdfScene, SdfShape, Vec3,
    };
    use std::time::{Duration, Instant};

    let w = args.width;
    let h = args.height;

    let fps = args.fps.max(1);
    let frame_dur = Duration::from_secs_f64(1.0 / fps as f64);
    let deadline = if args.duration > 0 {
        Some(Instant::now() + Duration::from_secs(args.duration))
    } else {
        None
    };

    let mut gpu_ctx = GpuRenderCtx::new();
    let mut driver = display::FrameDriver::detect();
    let mut frame_count: u64 = 0;
    let start = Instant::now();

    crossterm::terminal::enable_raw_mode().map_err(|e| e.to_string())?;

    let result = (|| -> Result<(), String> {
        loop {
            let frame_start = Instant::now();
            let t = start.elapsed().as_secs_f32();

            if event::poll(Duration::ZERO).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
                    if key.kind == KeyEventKind::Press {
                        break;
                    }
                }
            }

            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    break;
                }
            }

            // Morph factor: smooth oscillation between 0 and 1
            let morph_t = ((t * 0.5).sin() + 1.0) * 0.5;

            // Sphere ↔ torus morph with rainbow material
            let morph = SdfObject::new(
                SdfShape::Morph {
                    a: std::boxed::Box::new(SdfShape::Sphere { radius: 1.2 }),
                    b: std::boxed::Box::new(SdfShape::Torus {
                        major: 1.0,
                        minor: 0.4,
                    }),
                    t: morph_t,
                },
                Material::rainbow_animated(t * 0.4),
            )
            .at(Vec3::ZERO)
            .rotate_y(t * 0.3);

            // Camera orbiting
            let cam_angle = t * 0.2;
            let cam_r = 5.0;
            let cam_y = 2.5 + (t * 0.15).sin() * 0.5;
            let eye = Vec3::new(
                cam_angle.cos() * cam_r,
                cam_y,
                cam_angle.sin() * cam_r,
            );

            let transparent = Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

            let scene = SdfScene::new()
                .object(morph)
                .light(SdfLight::new(
                    Vec3::new(5.0, 8.0, 5.0),
                    Color::WHITE,
                    1.3,
                ))
                .light(SdfLight::new(
                    Vec3::new(-4.0, 3.0, -3.0),
                    Color::from_rgba8(100, 150, 255, 255),
                    0.6,
                ))
                .camera(SdfCamera::new(eye, Vec3::ZERO, 45.0))
                .sky_color(transparent)
                .ambient(0.08)
                .max_bounces(1);

            let pixmap = gpu_ctx.render(&scene, w, h, t)
                .map_err(|e| format!("SDF render failed: {e}"))?;

            driver.display_frame(&pixmap, frame_count)
                .map_err(|e| format!("display failed: {e}"))?;

            frame_count += 1;

            let elapsed = frame_start.elapsed();
            if elapsed < frame_dur {
                std::thread::sleep(frame_dur - elapsed);
            }
        }
        Ok(())
    })();

    let _ = crossterm::terminal::disable_raw_mode();

    result?;
    Ok(())
}
