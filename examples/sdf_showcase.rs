//! **SDF Showcase** — interactive ray-marched 3D scenes in the terminal.
//!
//! Four scene presets demonstrating the SDF renderer:
//!   1. **Reflections** — mirror sphere on a checkerboard plane
//!   2. **Water** — animated water plane reflecting a sphere and sky
//!   3. **Fire** — volumetric fire pillar rising from a platform
//!   4. **Blend** — two shapes smoothly merging via `smooth_min`
//!
//! Controls:
//!   `1`–`4`  — switch scene preset
//!   `Space`  — pause/resume animation
//!   `p`      — toggle per-stage profiler
//!   `q`      — quit
//!
//! Run with: `cargo run --example sdf_showcase --features sdf --release`

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::similar_names,
    clippy::needless_range_loop
)]

use std::io::{stdout, Write};
use std::time::{Duration, Instant};

use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEventKind},
    style,
    terminal::{self, disable_raw_mode, enable_raw_mode},
    ExecutableCommand, QueueableCommand,
};

use scry_engine::sdf::profiler::{render_profile_bar, SdfProfileHistory};
use scry_engine::sdf::*;
use scry_engine::style::Color as C;
use scry_engine::transport::backend::{ProtocolBackend, TerminalPosition};
use scry_engine::transport::{self, Picker, ProtocolKind};

// ═══════════════════════════════════════════════════════════════════
// Resolution cap
// ═══════════════════════════════════════════════════════════════════

const MAX_RENDER_W: u32 = 640;
const MAX_RENDER_H: u32 = 360;

// ═══════════════════════════════════════════════════════════════════
// Scene builders
// ═══════════════════════════════════════════════════════════════════

const PRESET_NAMES: [&str; 4] = ["Reflections", "Water", "Fire", "Blend"];

fn orbit_camera(target: Vec3, radius: f32, height: f32, time: f32, fov: f32) -> SdfCamera {
    let angle = time * 0.3;
    let eye = Vec3::new(
        target.x + angle.cos() * radius,
        height,
        target.z + angle.sin() * radius,
    );
    SdfCamera::new(eye, target, fov)
}

/// Orbit camera parameters per preset: (target, radius, height, fov).
const ORBIT_PARAMS: [(Vec3, f32, f32, f32); 4] = [
    (Vec3::new(0.0, 0.6, 0.0), 5.0, 2.5, 50.0), // Reflections
    (Vec3::new(0.0, 0.3, 0.0), 5.0, 2.0, 50.0), // Water
    (Vec3::new(0.0, 1.2, 0.0), 5.5, 2.5, 50.0), // Fire
    (Vec3::new(0.0, 1.0, 0.0), 5.5, 2.5, 50.0), // Blend
];

/// Build the static scene geometry (no camera — set per-frame).
fn build_scenes() -> [SdfScene; 4] {
    let reflections = SdfScene::new()
        .object(
            SdfObject::new(
                SdfShape::Sphere { radius: 1.0 },
                Material::mirror(C::from_rgba8(220, 220, 240, 255), 0.85),
            )
            .at(Vec3::new(0.0, 1.0, 0.0)),
        )
        .object(
            SdfObject::new(
                SdfShape::Sphere { radius: 0.5 },
                Material::Solid {
                    color: C::from_rgba8(200, 50, 50, 255),
                    reflectivity: 0.3,
                    specular: 48.0,
                },
            )
            .at(Vec3::new(-1.8, 0.5, 0.5)),
        )
        .object(SdfObject::new(
            SdfShape::Plane,
            Material::matte(C::from_rgba8(160, 160, 170, 255)),
        ))
        .light(SdfLight::new(Vec3::new(5.0, 8.0, 5.0), C::WHITE, 0.8))
        .light(SdfLight::new(
            Vec3::new(-3.0, 6.0, -2.0),
            C::from_rgba8(150, 180, 255, 255),
            0.4,
        ))
        .sky_color(C::from_rgba8(60, 80, 120, 255));

    let water = SdfScene::new()
        .object(
            SdfObject::new(
                SdfShape::Sphere { radius: 0.8 },
                Material::Solid {
                    color: C::from_rgba8(220, 180, 50, 255),
                    reflectivity: 0.2,
                    specular: 48.0,
                },
            )
            .at(Vec3::new(0.0, 1.5, 0.0)),
        )
        .object(SdfObject::new(SdfShape::Plane, Material::water()))
        .light(SdfLight::new(
            Vec3::new(4.0, 10.0, 4.0),
            C::from_rgba8(255, 240, 220, 255),
            1.0,
        ))
        .sky_color(C::from_rgba8(80, 130, 200, 255))
        .max_bounces(2);

    let fire = SdfScene::new()
        .object(
            SdfObject::new(
                SdfShape::Cylinder {
                    radius: 0.8,
                    half_height: 1.5,
                },
                Material::Fire {
                    intensity: 2.0,
                    noise_scale: 2.0,
                    speed: 1.2,
                },
            )
            .at(Vec3::new(0.0, 1.5, 0.0)),
        )
        .object(
            SdfObject::new(
                SdfShape::Box {
                    half_extents: Vec3::new(1.5, 0.15, 1.5),
                },
                Material::matte(C::from_rgba8(80, 80, 90, 255)),
            )
            .at(Vec3::new(0.0, 0.15, 0.0)),
        )
        .object(SdfObject::new(
            SdfShape::Plane,
            Material::matte(C::from_rgba8(50, 50, 55, 255)),
        ))
        .light(SdfLight::new(
            Vec3::new(3.0, 6.0, 4.0),
            C::from_rgba8(255, 200, 150, 255),
            0.3,
        ))
        .sky_color(C::from_rgba8(10, 8, 15, 255))
        .ambient(0.02);

    let blend = SdfScene::new()
        .object(
            SdfObject::new(
                SdfShape::SmoothBlend {
                    a: Box::new(SdfShape::Sphere { radius: 1.0 }),
                    b: Box::new(SdfShape::Torus {
                        major: 1.2,
                        minor: 0.3,
                    }),
                    b_offset: Vec3::new(0.0, 0.0, 0.0),
                    k: 0.5,
                },
                Material::Solid {
                    color: C::from_rgba8(100, 180, 220, 255),
                    reflectivity: 0.3,
                    specular: 64.0,
                },
            )
            .at(Vec3::new(0.0, 1.3, 0.0)),
        )
        .object(SdfObject::new(
            SdfShape::Plane,
            Material::matte(C::from_rgba8(150, 150, 160, 255)),
        ))
        .light(SdfLight::new(Vec3::new(4.0, 8.0, 5.0), C::WHITE, 0.9))
        .light(SdfLight::new(
            Vec3::new(-3.0, 5.0, -1.0),
            C::from_rgba8(180, 160, 255, 255),
            0.3,
        ))
        .sky_color(C::from_rgba8(50, 60, 90, 255));

    [reflections, water, fire, blend]
}

/// Set the orbit camera on a pre-built scene for the current frame.
fn set_camera(scene: &mut SdfScene, preset: usize, time: f32) {
    let (target, radius, height, fov) = ORBIT_PARAMS[preset];
    scene.camera = orbit_camera(target, radius, height, time, fov);
}

// ═══════════════════════════════════════════════════════════════════
// Main loop — direct transport, no widget/rasterizer overhead
// ═══════════════════════════════════════════════════════════════════

fn main() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    let mut out = stdout();
    out.execute(terminal::EnterAlternateScreen)?;
    out.execute(cursor::Hide)?;

    let picker = Picker::detect();
    let font = picker.font_size();
    let mut backend: Box<dyn ProtocolBackend> = match picker.protocol() {
        ProtocolKind::Kitty => Box::new(transport::kitty::KittyBackend::new(font)),
        _ => Box::new(transport::halfblock::HalfblockBackend::new()),
    };

    let mut scenes = build_scenes();
    let mut preset_idx: usize = 0;
    let mut paused = false;
    let start = Instant::now();
    let mut frozen_time = 0.0_f32;
    let mut last_fps = 0.0_f32;
    let mut handle: Option<transport::backend::ImageHandle> = None;

    // Profiler state
    let mut profiling = false;
    let mut profile_history = SdfProfileHistory::new(32);

    // Render scale (bicubic upscaler)
    const SCALE_STEPS: [f32; 4] = [0.25, 0.5, 0.75, 1.0];
    let mut scale_idx: usize = 1; // default 0.5

    loop {
        let frame_start = Instant::now();

        let elapsed = if paused {
            frozen_time
        } else {
            let e = start.elapsed().as_secs_f32();
            frozen_time = e;
            e
        };

        // Get terminal size in pixels
        let (cols, rows) = terminal::size()?;
        let status_rows = if profiling { 3 } else { 1 };
        let full_w = u32::from(cols) * u32::from(font.width);
        let full_h = u32::from(rows.saturating_sub(status_rows + 1)) * u32::from(font.height);

        if full_w > 0 && full_h > 0 {
            // Cap to max render resolution, then apply user render scale
            let cap_w = full_w.min(MAX_RENDER_W);
            let cap_h = full_h.min(MAX_RENDER_H);
            let render_scale = SCALE_STEPS[scale_idx];

            // Update only the camera on the pre-built scene
            set_camera(&mut scenes[preset_idx], preset_idx, elapsed);

            let pixmap = if profiling {
                let (pm, profile) = SdfRenderer::render_to_pixmap_upscaled_profiled(
                    &scenes[preset_idx],
                    cap_w,
                    cap_h,
                    render_scale,
                    elapsed,
                )?;
                profile_history.push(profile);
                pm
            } else {
                SdfRenderer::render_to_pixmap_upscaled(
                    &scenes[preset_idx],
                    cap_w,
                    cap_h,
                    render_scale,
                    elapsed,
                )?
            };

            // Transmit directly to terminal — no widget/rasterizer overhead
            let pos = TerminalPosition::new(0, 0, cols, rows.saturating_sub(status_rows + 1));

            let new_handle = if let Some(ref old) = handle {
                backend.replace(old, &pixmap, pos, -1)?
            } else {
                backend.transmit(&pixmap, pos, -1)?
            };
            handle = Some(new_handle);
        }

        // Status bar at bottom
        let (cols, rows) = terminal::size()?;
        let frame_ms = frame_start.elapsed().as_secs_f32();
        if frame_ms > 0.0 {
            last_fps = last_fps * 0.8 + (1.0 / frame_ms) * 0.2;
        }

        if profiling {
            let summary = profile_history.summary();
            let total_ms = summary.total_us as f64 / 1000.0;

            // Line 1: scene info + resolution + total time
            let pct = (SCALE_STEPS[scale_idx] * 100.0) as u32;
            out.queue(cursor::MoveTo(0, rows.saturating_sub(3)))?;
            out.queue(style::Print(format!(
                "\x1b[K SDF: {} | {:.0} fps | {}x{} @{}% | {:.1}ms total",
                PRESET_NAMES[preset_idx],
                last_fps,
                MAX_RENDER_W.min(full_w),
                MAX_RENDER_H.min(full_h),
                pct,
                total_ms,
            )))?;

            // Line 2: colored bar chart
            let bar_width = (cols as usize).saturating_sub(4).min(40);
            let bar = render_profile_bar(&summary, bar_width);
            out.queue(cursor::MoveTo(0, rows.saturating_sub(2)))?;
            out.queue(style::Print(format!("\x1b[K {bar}")))?;

            // Line 3: controls
            out.queue(cursor::MoveTo(0, rows.saturating_sub(1)))?;
            out.queue(style::Print(format!(
                "\x1b[K [1-4] scene  [+/-] scale  [space] pause  [p] profile  [q] quit"
            )))?;
        } else {
            let pct = (SCALE_STEPS[scale_idx] * 100.0) as u32;
            out.queue(cursor::MoveTo(0, rows - 1))?;
            out.queue(style::Print(format!(
                "\x1b[K SDF: {} | {:.0} fps | {}% upscale | [1-4] scene [+/-] scale [space] pause [p] profile [q] quit",
                PRESET_NAMES[preset_idx], last_fps, pct,
            )))?;
        }
        out.flush()?;

        // Drain all pending events
        while event::poll(Duration::ZERO)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => {
                            if let Some(ref h) = handle {
                                let _ = backend.remove(h);
                            }
                            out.execute(cursor::Show)?;
                            out.execute(terminal::LeaveAlternateScreen)?;
                            disable_raw_mode()?;
                            return Ok(());
                        }
                        KeyCode::Char('1') => preset_idx = 0,
                        KeyCode::Char('2') => preset_idx = 1,
                        KeyCode::Char('3') => preset_idx = 2,
                        KeyCode::Char('4') => preset_idx = 3,
                        KeyCode::Char(' ') => paused = !paused,
                        KeyCode::Char('p') => {
                            profiling = !profiling;
                            if profiling {
                                profile_history = SdfProfileHistory::new(32);
                            }
                        }
                        KeyCode::Char('+') | KeyCode::Char('=') => {
                            if scale_idx < SCALE_STEPS.len() - 1 {
                                scale_idx += 1;
                            }
                        }
                        KeyCode::Char('-') => {
                            if scale_idx > 0 {
                                scale_idx -= 1;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}
