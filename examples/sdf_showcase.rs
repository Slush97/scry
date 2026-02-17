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

use std::io::{Write, stdout};
use std::time::{Duration, Instant};

use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEventKind},
    style,
    terminal::{self, disable_raw_mode, enable_raw_mode},
    ExecutableCommand, QueueableCommand,
};

use scry_engine::sdf::*;
use scry_engine::style::Color as C;
use scry_engine::transport::backend::{ProtocolBackend, TerminalPosition};
use scry_engine::transport::{self, Picker, ProtocolKind};

// ═══════════════════════════════════════════════════════════════════
// Resolution cap
// ═══════════════════════════════════════════════════════════════════

const MAX_RENDER_W: u32 = 320;
const MAX_RENDER_H: u32 = 180;

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

fn build_reflections_scene(time: f32) -> SdfScene {
    SdfScene::new()
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
        .camera(orbit_camera(
            Vec3::new(0.0, 0.6, 0.0),
            5.0,
            2.5,
            time,
            50.0,
        ))
        .sky_color(C::from_rgba8(60, 80, 120, 255))
}

fn build_water_scene(time: f32) -> SdfScene {
    SdfScene::new()
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
        .camera(orbit_camera(
            Vec3::new(0.0, 0.3, 0.0),
            5.0,
            2.0,
            time,
            50.0,
        ))
        .sky_color(C::from_rgba8(80, 130, 200, 255))
        .max_bounces(2)
}

fn build_fire_scene(time: f32) -> SdfScene {
    SdfScene::new()
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
        .camera(orbit_camera(
            Vec3::new(0.0, 1.2, 0.0),
            5.5,
            2.5,
            time,
            50.0,
        ))
        .sky_color(C::from_rgba8(10, 8, 15, 255))
        .ambient(0.02)
}

fn build_blend_scene(time: f32) -> SdfScene {
    SdfScene::new()
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
        .camera(orbit_camera(
            Vec3::new(0.0, 1.0, 0.0),
            5.5,
            2.5,
            time,
            50.0,
        ))
        .sky_color(C::from_rgba8(50, 60, 90, 255))
}

fn build_scene(preset: usize, time: f32) -> SdfScene {
    match preset {
        0 => build_reflections_scene(time),
        1 => build_water_scene(time),
        2 => build_fire_scene(time),
        3 => build_blend_scene(time),
        _ => build_reflections_scene(time),
    }
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

    let mut preset_idx: usize = 0;
    let mut paused = false;
    let start = Instant::now();
    let mut frozen_time = 0.0_f32;
    let mut last_fps = 0.0_f32;
    let mut handle: Option<transport::backend::ImageHandle> = None;

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
        let full_w = u32::from(cols) * u32::from(font.width);
        let full_h = u32::from(rows.saturating_sub(2)) * u32::from(font.height); // leave room for status

        if full_w > 0 && full_h > 0 {
            // Render at reduced resolution
            let scale = f32::min(
                MAX_RENDER_W as f32 / full_w as f32,
                MAX_RENDER_H as f32 / full_h as f32,
            )
            .min(1.0);
            let render_w = ((full_w as f32 * scale) as u32).max(1);
            let render_h = ((full_h as f32 * scale) as u32).max(1);

            let scene = build_scene(preset_idx, elapsed);
            let pixmap = SdfRenderer::render_to_pixmap(&scene, render_w, render_h, elapsed)?;

            // Transmit directly to terminal — no widget/rasterizer overhead
            let pos = TerminalPosition::new(0, 0, cols, rows.saturating_sub(2));

            let new_handle = if let Some(ref old) = handle {
                backend.replace(old, &pixmap, pos, -1)?
            } else {
                backend.transmit(&pixmap, pos, -1)?
            };
            handle = Some(new_handle);
        }

        // Status bar at bottom
        let (_, rows) = terminal::size()?;
        let frame_ms = frame_start.elapsed().as_secs_f32();
        if frame_ms > 0.0 {
            last_fps = last_fps * 0.8 + (1.0 / frame_ms) * 0.2;
        }
        out.queue(cursor::MoveTo(0, rows - 1))?;
        out.queue(style::Print(format!(
            " SDF: {} | {:.0} fps | [1-4] scene [space] pause [q] quit   ",
            PRESET_NAMES[preset_idx], last_fps,
        )))?;
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
                        _ => {}
                    }
                }
            }
        }
    }
}
