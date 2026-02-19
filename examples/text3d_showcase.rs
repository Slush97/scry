//! **Text3D Showcase** — extruded 3D text with five material presets.
//!
//! Demonstrates `SdfShape::Text3D` (true 3D extruded text from TTF outlines)
//! and quaternion rotation on `SdfObject`:
//!   1. **Glass** — refractive transparent text with colored spheres behind
//!   2. **Chrome** — mirror-finish text reflecting colored spheres
//!   3. **Fire** — volumetric fire text with flanking fire pillars
//!   4. **Rainbow** — spectral animated text rotating over water
//!   5. **Museum** — white marble text tilted on a pedestal (quaternion demo)
//!
//! Controls:
//!   `1`–`5`  — switch scene preset
//!   `Space`  — pause/resume animation
//!   `p`      — toggle per-stage profiler
//!   `+`/`-`  — cycle render scale (25%, 50%, 75%, 100%)
//!   `q`/`Esc` — quit
//!
//! Run with:
//!   Terminal: `cargo run --example text3d_showcase --features sdf-text --release`
//!   Window:   `cargo run --example text3d_showcase --features "sdf-text,window" --release -- --window`

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::wildcard_imports,
    unused_labels
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
// Resolution cap (terminal mode only)
// ═══════════════════════════════════════════════════════════════════

const MAX_RENDER_W: u32 = 640;
const MAX_RENDER_H: u32 = 360;

// ═══════════════════════════════════════════════════════════════════
// Font data
// ═══════════════════════════════════════════════════════════════════

const FONT: &[u8] = include_bytes!("../crates/scry-chart/src/fonts/Inter-Bold.ttf");

// ═══════════════════════════════════════════════════════════════════
// Scene builders
// ═══════════════════════════════════════════════════════════════════

const NUM_PRESETS: usize = 5;
const PRESET_NAMES: [&str; NUM_PRESETS] = ["Glass", "Chrome", "Fire", "Rainbow", "Museum"];

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
const ORBIT_PARAMS: [(Vec3, f32, f32, f32); NUM_PRESETS] = [
    (Vec3::new(0.0, 0.8, 0.0), 6.0, 2.5, 50.0), // Glass
    (Vec3::new(0.0, 0.8, 0.0), 6.0, 2.0, 50.0), // Chrome
    (Vec3::new(0.0, 1.0, 0.0), 6.5, 2.5, 50.0), // Fire
    (Vec3::new(0.0, 0.6, 0.0), 6.0, 2.5, 50.0), // Rainbow
    (Vec3::new(0.0, 1.2, 0.0), 5.5, 3.0, 45.0), // Museum
];

/// Build the three static scenes (Glass, Chrome, Fire) that only need camera updates per frame.
fn build_static_scenes() -> [SdfScene; 3] {
    // 1. Glass — refractive transparent text
    let glass = SdfScene::new()
        .object(
            SdfObject::new(
                SdfShape::text_3d(FONT, "GLASS", 1.2, 0.4).expect("font parse"),
                Material::glass_dispersive(C::from_rgba8(220, 240, 255, 255), 1.5, 0.03),
            )
            .at(Vec3::new(0.0, 1.0, 0.0)),
        )
        // Dark checkerboard floor
        .object(SdfObject::new(
            SdfShape::Plane,
            Material::Checkerboard {
                color_a: C::from_rgba8(60, 60, 65, 255),
                color_b: C::from_rgba8(40, 40, 45, 255),
                scale: 1.0,
                reflectivity: 0.15,
                specular: 32.0,
            },
        ))
        // Red sphere behind-left
        .object(
            SdfObject::new(
                SdfShape::Sphere { radius: 0.6 },
                Material::Solid {
                    color: C::from_rgba8(200, 40, 40, 255),
                    reflectivity: 0.2,
                    specular: 48.0,
                },
            )
            .at(Vec3::new(-2.0, 0.6, -1.5)),
        )
        // Blue sphere behind-right
        .object(
            SdfObject::new(
                SdfShape::Sphere { radius: 0.5 },
                Material::Solid {
                    color: C::from_rgba8(40, 80, 200, 255),
                    reflectivity: 0.2,
                    specular: 48.0,
                },
            )
            .at(Vec3::new(2.0, 0.5, -1.2)),
        )
        // Warm key light
        .light(SdfLight::new(
            Vec3::new(5.0, 8.0, 5.0),
            C::from_rgba8(255, 240, 220, 255),
            0.8,
        ))
        // Cool fill light
        .light(SdfLight::new(
            Vec3::new(-4.0, 6.0, -2.0),
            C::from_rgba8(150, 180, 255, 255),
            0.4,
        ))
        .sky_color(C::from_rgba8(20, 25, 40, 255))
        .max_bounces(4);

    // 2. Chrome — mirror-finish text
    let chrome = SdfScene::new()
        .object(
            SdfObject::new(
                SdfShape::text_3d(FONT, "CHROME", 1.0, 0.5).expect("font parse"),
                Material::mirror(C::from_rgba8(220, 220, 230, 255), 0.9),
            )
            .at(Vec3::new(0.0, 1.0, 0.0)),
        )
        // Dark matte floor
        .object(SdfObject::new(
            SdfShape::Plane,
            Material::matte(C::from_rgba8(50, 50, 55, 255)),
        ))
        // Gold sphere for interesting reflections
        .object(
            SdfObject::new(
                SdfShape::Sphere { radius: 0.5 },
                Material::Solid {
                    color: C::from_rgba8(220, 180, 50, 255),
                    reflectivity: 0.3,
                    specular: 48.0,
                },
            )
            .at(Vec3::new(-2.0, 0.5, 1.0)),
        )
        // Cyan sphere
        .object(
            SdfObject::new(
                SdfShape::Sphere { radius: 0.4 },
                Material::Solid {
                    color: C::from_rgba8(50, 200, 200, 255),
                    reflectivity: 0.3,
                    specular: 48.0,
                },
            )
            .at(Vec3::new(2.5, 0.4, 0.8)),
        )
        // Bright key light
        .light(SdfLight::new(Vec3::new(5.0, 8.0, 5.0), C::WHITE, 0.9))
        // Warm back light
        .light(SdfLight::new(
            Vec3::new(-3.0, 5.0, -4.0),
            C::from_rgba8(255, 220, 180, 255),
            0.4,
        ))
        // Cool side light
        .light(SdfLight::new(
            Vec3::new(4.0, 3.0, -2.0),
            C::from_rgba8(150, 180, 255, 255),
            0.3,
        ))
        .sky_color(C::from_rgba8(30, 35, 50, 255))
        .max_bounces(3);

    // 3. Fire — volumetric fire text
    let fire = SdfScene::new()
        .object(
            SdfObject::new(
                SdfShape::text_3d(FONT, "FIRE", 1.4, 0.6).expect("font parse"),
                Material::Fire {
                    intensity: 2.0,
                    noise_scale: 2.0,
                    speed: 1.0,
                },
            )
            .at(Vec3::new(0.0, 1.2, 0.0)),
        )
        // Stone platform underneath
        .object(
            SdfObject::new(
                SdfShape::Box {
                    half_extents: Vec3::new(2.5, 0.15, 1.0),
                },
                Material::matte(C::from_rgba8(80, 75, 70, 255)),
            )
            .at(Vec3::new(0.0, 0.15, 0.0)),
        )
        // Dark matte floor
        .object(SdfObject::new(
            SdfShape::Plane,
            Material::matte(C::from_rgba8(40, 38, 35, 255)),
        ))
        // Fire pillar left
        .object(
            SdfObject::new(
                SdfShape::Cylinder {
                    radius: 0.3,
                    half_height: 0.8,
                },
                Material::Fire {
                    intensity: 1.5,
                    noise_scale: 2.5,
                    speed: 1.2,
                },
            )
            .at(Vec3::new(-3.0, 0.8, 0.0)),
        )
        // Fire pillar right
        .object(
            SdfObject::new(
                SdfShape::Cylinder {
                    radius: 0.3,
                    half_height: 0.8,
                },
                Material::Fire {
                    intensity: 1.5,
                    noise_scale: 2.5,
                    speed: 1.2,
                },
            )
            .at(Vec3::new(3.0, 0.8, 0.0)),
        )
        // Dim warm overhead light
        .light(SdfLight::new(
            Vec3::new(0.0, 8.0, 3.0),
            C::from_rgba8(255, 180, 120, 255),
            0.3,
        ))
        .sky_color(C::from_rgba8(8, 6, 12, 255))
        .ambient(0.01);

    [glass, chrome, fire]
}

/// Build time-dependent scenes (Rainbow, Museum) — called every frame.
fn build_rainbow(time: f32) -> SdfScene {
    SdfScene::new()
        .object(
            SdfObject::new(
                SdfShape::text_3d(FONT, "SCRY", 1.3, 0.5).expect("font parse"),
                Material::rainbow_animated(time * 0.6),
            )
            .at(Vec3::new(0.0, 1.0, 0.0))
            .rotate_y(time * 0.4),
        )
        // Water floor
        .object(SdfObject::new(SdfShape::Plane, Material::water()))
        // Warm key light
        .light(SdfLight::new(
            Vec3::new(5.0, 8.0, 5.0),
            C::from_rgba8(255, 240, 220, 255),
            0.8,
        ))
        // Cool fill light
        .light(SdfLight::new(
            Vec3::new(-3.0, 6.0, -2.0),
            C::from_rgba8(150, 180, 255, 255),
            0.4,
        ))
        .sky_color(C::from_rgba8(80, 130, 200, 255))
        .max_bounces(2)
}

fn build_museum() -> SdfScene {
    SdfScene::new()
        // Tilted marble text — quaternion rotation around X axis (20° tilt)
        .object(
            SdfObject::new(
                SdfShape::text_3d_with_options(FONT, "TEXT3D", 1.0, 0.3, 0.05, 64)
                    .expect("font parse"),
                Material::Solid {
                    color: C::from_rgba8(240, 235, 230, 255),
                    reflectivity: 0.1,
                    specular: 96.0,
                },
            )
            .at(Vec3::new(0.0, 1.8, 0.0))
            .rotate(Vec3::X, 0.35),
        )
        // Pedestal
        .object(
            SdfObject::new(
                SdfShape::Box {
                    half_extents: Vec3::new(1.8, 0.6, 0.6),
                },
                Material::Solid {
                    color: C::from_rgba8(180, 175, 170, 255),
                    reflectivity: 0.05,
                    specular: 32.0,
                },
            )
            .at(Vec3::new(0.0, 0.6, 0.0)),
        )
        // Subtle checkerboard floor
        .object(SdfObject::new(
            SdfShape::Plane,
            Material::Checkerboard {
                color_a: C::from_rgba8(160, 160, 165, 255),
                color_b: C::from_rgba8(130, 130, 135, 255),
                scale: 1.0,
                reflectivity: 0.25,
                specular: 32.0,
            },
        ))
        // Gallery-style 3-point lighting
        // Key light (warm, from above-right)
        .light(SdfLight::new(
            Vec3::new(4.0, 8.0, 4.0),
            C::from_rgba8(255, 245, 230, 255),
            0.8,
        ))
        // Fill light (cool, from left)
        .light(SdfLight::new(
            Vec3::new(-4.0, 5.0, 2.0),
            C::from_rgba8(180, 200, 240, 255),
            0.3,
        ))
        // Back accent light
        .light(SdfLight::new(
            Vec3::new(0.0, 4.0, -5.0),
            C::from_rgba8(255, 230, 200, 255),
            0.3,
        ))
        .sky_color(C::from_rgba8(50, 55, 65, 255))
        .max_bounces(2)
}

/// Set the orbit camera on a scene for the current frame.
fn set_camera(scene: &mut SdfScene, preset: usize, time: f32) {
    let (target, radius, height, fov) = ORBIT_PARAMS[preset];
    scene.camera = orbit_camera(target, radius, height, time, fov);
}

// ═══════════════════════════════════════════════════════════════════
// Render scale steps
// ═══════════════════════════════════════════════════════════════════

const SCALE_STEPS: [f32; 4] = [0.25, 0.5, 0.75, 1.0];

// ═══════════════════════════════════════════════════════════════════
// Window mode
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "window")]
fn run_window() -> Result<(), Box<dyn std::error::Error>> {
    use scry_engine::sdf::overlay::StatsOverlay;
    use scry_engine::transport::window::{run_loop_continuous, LoopAction};
    use winit::keyboard::KeyCode as WKey;

    let mut static_scenes = build_static_scenes();
    let mut dynamic_scene: SdfScene = build_museum();

    let mut preset_idx: usize = 0;
    let mut paused = false;
    let start = Instant::now();
    let mut frozen_time = 0.0_f32;
    let mut scale_idx: usize = 1;
    let mut show_overlay = true;
    let mut profile_history = SdfProfileHistory::new(32);
    let mut stats_overlay = StatsOverlay::new(120);

    run_loop_continuous(
        960,
        640,
        "Text3D Showcase",
        true,
        move |backend, keys, (w, h)| {
            for key in keys {
                if !key.pressed {
                    continue;
                }
                match key.code {
                    WKey::Escape | WKey::KeyQ => return LoopAction::Exit,
                    WKey::Digit1 => preset_idx = 0,
                    WKey::Digit2 => preset_idx = 1,
                    WKey::Digit3 => preset_idx = 2,
                    WKey::Digit4 => preset_idx = 3,
                    WKey::Digit5 => preset_idx = 4,
                    WKey::Space => paused = !paused,
                    WKey::F3 => show_overlay = !show_overlay,
                    WKey::Equal => {
                        if scale_idx < SCALE_STEPS.len() - 1 {
                            scale_idx += 1;
                        }
                    }
                    WKey::Minus => {
                        if scale_idx > 0 {
                            scale_idx -= 1;
                        }
                    }
                    _ => {}
                }
            }

            let elapsed = if paused {
                frozen_time
            } else {
                let e = start.elapsed().as_secs_f32();
                frozen_time = e;
                e
            };

            if w == 0 || h == 0 {
                return LoopAction::Continue;
            }

            let render_scale = SCALE_STEPS[scale_idx];

            // Get the active scene (rebuild dynamic ones each frame)
            let scene = match preset_idx {
                0..=2 => {
                    set_camera(&mut static_scenes[preset_idx], preset_idx, elapsed);
                    &static_scenes[preset_idx]
                }
                3 => {
                    dynamic_scene = build_rainbow(elapsed);
                    set_camera(&mut dynamic_scene, 3, elapsed);
                    &dynamic_scene
                }
                _ => {
                    dynamic_scene = build_museum();
                    set_camera(&mut dynamic_scene, 4, elapsed);
                    &dynamic_scene
                }
            };

            let (mut pixmap, profile) = match SdfRenderer::render_to_pixmap_upscaled_profiled(
                scene,
                w,
                h,
                render_scale,
                elapsed,
            ) {
                Ok(r) => r,
                Err(_) => return LoopAction::Continue,
            };
            profile_history.push(profile);

            if show_overlay {
                stats_overlay.tick();
                let summary = profile_history.summary();
                let pct = (render_scale * 100.0) as u32;
                stats_overlay.render_overlay(&mut pixmap, &summary, pct, PRESET_NAMES[preset_idx]);
            }

            let _ = backend.blit(&pixmap);
            LoopAction::Continue
        },
    )?;

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════
// Terminal mode — direct transport, no widget/rasterizer overhead
// ═══════════════════════════════════════════════════════════════════

fn run_terminal() -> Result<(), Box<dyn std::error::Error>> {
    // Install panic hook that restores terminal before printing the panic.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = stdout().execute(terminal::LeaveAlternateScreen);
        let _ = stdout().execute(cursor::Show);
        default_hook(info);
    }));

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

    let mut static_scenes = build_static_scenes();

    // Auto-detect GPU with timeout to avoid hanging on broken drivers
    #[cfg(feature = "sdf-gpu")]
    let mut gpu_ctx =
        scry_engine::sdf::SdfGpuContext::try_new(std::time::Duration::from_secs(5));

    let mut preset_idx: usize = 0;
    let mut paused = false;
    let start = Instant::now();
    let mut frozen_time = 0.0_f32;
    let mut last_fps = 0.0_f32;
    let mut handle: Option<transport::backend::ImageHandle> = None;

    // Profiler state
    let mut profiling = false;
    let mut profile_history = SdfProfileHistory::new(32);

    let mut scale_idx: usize = 1; // default 0.5 (bicubic upscale)

    // Dynamic scenes rebuilt per-frame
    let mut dynamic_scene: SdfScene = build_museum();

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
            let cap_w = full_w.min(MAX_RENDER_W);
            let cap_h = full_h.min(MAX_RENDER_H);
            let render_scale = SCALE_STEPS[scale_idx];

            // Get the active scene (rebuild dynamic ones each frame)
            let scene = match preset_idx {
                0..=2 => {
                    set_camera(&mut static_scenes[preset_idx], preset_idx, elapsed);
                    &static_scenes[preset_idx]
                }
                3 => {
                    dynamic_scene = build_rainbow(elapsed);
                    set_camera(&mut dynamic_scene, 3, elapsed);
                    &dynamic_scene
                }
                _ => {
                    dynamic_scene = build_museum();
                    set_camera(&mut dynamic_scene, 4, elapsed);
                    &dynamic_scene
                }
            };

            let pixmap = 'render: {
                // GPU path: render at full resolution
                #[cfg(feature = "sdf-gpu")]
                if let Some(ref mut ctx) = gpu_ctx {
                    let gpu_start = Instant::now();
                    match scry_engine::sdf::SdfGpuRenderer::render_to_pixmap(
                        ctx,
                        scene,
                        cap_w,
                        cap_h,
                        elapsed,
                    ) {
                        Ok(pm) => {
                            if profiling {
                                let gpu_us = gpu_start.elapsed().as_micros() as u64;
                                profile_history.push(scry_engine::sdf::SdfProfile::total_only(
                                    gpu_us, cap_w, cap_h,
                                ));
                            }
                            break 'render pm;
                        }
                        Err(_) => {
                            // GPU render failed, fall through to CPU
                        }
                    }
                }

                // CPU path
                let pm = if profiling {
                    let (pm, profile) = SdfRenderer::render_to_pixmap_upscaled_profiled(
                        scene,
                        cap_w,
                        cap_h,
                        render_scale,
                        elapsed,
                    )?;
                    profile_history.push(profile);
                    pm
                } else {
                    SdfRenderer::render_to_pixmap_upscaled(
                        scene,
                        cap_w,
                        cap_h,
                        render_scale,
                        elapsed,
                    )?
                };
                pm
            };

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

        #[cfg(feature = "sdf-gpu")]
        let gpu_tag = if gpu_ctx.is_some() { "  [GPU]" } else { "" };
        #[cfg(not(feature = "sdf-gpu"))]
        let gpu_tag = "";

        if profiling {
            let summary = profile_history.summary();
            let total_ms = summary.total_us as f64 / 1000.0;

            let pct = (SCALE_STEPS[scale_idx] * 100.0) as u32;
            out.queue(cursor::MoveTo(0, rows.saturating_sub(3)))?;
            out.queue(style::Print(format!(
                "\x1b[K Text3D: {}{} | {:.0} fps | {}x{} @{}% | {:.1}ms total",
                PRESET_NAMES[preset_idx],
                gpu_tag,
                last_fps,
                MAX_RENDER_W.min(full_w),
                MAX_RENDER_H.min(full_h),
                pct,
                total_ms,
            )))?;

            let bar_width = (cols as usize).saturating_sub(4).min(40);
            let bar = render_profile_bar(&summary, bar_width);
            out.queue(cursor::MoveTo(0, rows.saturating_sub(2)))?;
            out.queue(style::Print(format!("\x1b[K {bar}")))?;

            out.queue(cursor::MoveTo(0, rows.saturating_sub(1)))?;
            out.queue(style::Print(
                "\x1b[K [1-5] scene  [+/-] scale  [space] pause  [p] profile  [q] quit"
            ))?;
        } else {
            let pct = (SCALE_STEPS[scale_idx] * 100.0) as u32;
            out.queue(cursor::MoveTo(0, rows - 1))?;
            out.queue(style::Print(format!(
                "\x1b[K Text3D: {}{} | {:.0} fps | {}% upscale | [1-5] scene [+/-] scale [space] pause [p] profile [q] quit",
                PRESET_NAMES[preset_idx], gpu_tag, last_fps, pct,
            )))?;
        }
        out.flush()?;

        // Drain all pending events
        while event::poll(Duration::ZERO)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => {
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
                        KeyCode::Char('5') => preset_idx = 4,
                        KeyCode::Char(' ') => paused = !paused,
                        KeyCode::Char('p') => {
                            profiling = !profiling;
                            if profiling {
                                profile_history = SdfProfileHistory::new(32);
                            }
                        }
                        KeyCode::Char('+' | '=') => {
                            if scale_idx < SCALE_STEPS.len() - 1 {
                                scale_idx += 1;
                            }
                        }
                        KeyCode::Char('-') => {
                            scale_idx = scale_idx.saturating_sub(1);
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let use_window = std::env::args().any(|a| a == "--window");

    if use_window {
        #[cfg(feature = "window")]
        {
            return run_window();
        }
        #[cfg(not(feature = "window"))]
        {
            eprintln!("error: --window requires the `window` feature");
            eprintln!("  cargo run --example text3d_showcase --features \"sdf-text,window\" --release -- --window");
            std::process::exit(1);
        }
    }

    run_terminal()
}
