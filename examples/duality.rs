//! **Duality** — the peaks and valleys of a bipolar mind, rendered in light.
//!
//! Two pyramids fused at their bases form an hourglass of consciousness.
//! They flip perpetually around the X axis — mania's golden glass tip soaring
//! as depression's indigo core descends, then inverting. A mandelbulb fractal
//! heart pulses at the junction. Mirror sentinels orbit in counter-rotating
//! rings. The ground breathes.
//!
//! Controls:
//!   `Space`  — pause / resume
//!   `p`      — toggle profiler
//!   `+`/`-`  — render scale
//!   `q`/`Esc` — quit
//!
//! Run:
//!   Terminal: `cargo run --example duality --features sdf --release`
//!   Window:   `cargo run --example duality --features "sdf,window" --release -- --window`

#![allow(
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::needless_range_loop,
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
// Constants
// ═══════════════════════════════════════════════════════════════════

const MAX_RENDER_W: u32 = 640;
const MAX_RENDER_H: u32 = 360;
const SCALE_STEPS: [f32; 4] = [0.25, 0.5, 0.75, 1.0];

// ═══════════════════════════════════════════════════════════════════
// Color palette — chosen for emotional resonance
// ═══════════════════════════════════════════════════════════════════

/// Mania gold — warm amber glass
fn mania_tint() -> C {
    C::from_rgba8(255, 180, 40, 255)
}

/// Depression indigo — deep subsurface scatter
fn depression_surface() -> C {
    C::from_rgba8(30, 15, 80, 255)
}

/// Depression scatter — violet light bleeding through
fn depression_scatter() -> C {
    C::from_rgba8(90, 40, 180, 255)
}

/// Mandelbulb heart — iridescent rainbow
fn _heart_color() -> C {
    C::from_rgba8(200, 180, 255, 255)
}

/// Void sky
fn sky() -> C {
    C::from_rgba8(5, 3, 12, 255)
}

/// Ground dark
fn ground_a() -> C {
    C::from_rgba8(15, 12, 25, 255)
}

/// Ground accent
fn ground_b() -> C {
    C::from_rgba8(8, 6, 18, 255)
}

// ═══════════════════════════════════════════════════════════════════
// Scene builder — reconstructed each frame for animation
// ═══════════════════════════════════════════════════════════════════

fn build_scene(t: f32) -> SdfScene {
    // ── Core rotation: the eternal flip ──
    // Slow sinusoidal flip around X axis — smooth, meditative
    let flip_angle = t * 0.6;

    // ── Breathing pulse on pyramid heights ──
    let breath = (t * 1.5).sin() * 0.2;
    let upper_h = 1.8 + breath;
    let lower_h = 1.8 - breath;

    // ── The vertical offset so bases touch at origin ──
    // Upper pyramid: tip points up, base at y=0. Rotated around center.
    // Lower pyramid: tip points down (flipped by π around X), base at y=0.
    // Both anchored at y=0, then the whole assembly is rotated by flip_angle.

    // Center of the hourglass form
    let center_y = 2.2;

    // ── Upper pyramid: mania ──
    // Cone tip at y=height, base at y=0. We position it so base is at center_y.
    let scene = SdfScene::new()
        .object(
            SdfObject::new(
                SdfShape::Cone {
                    radius: 1.2,
                    height: upper_h,
                },
                Material::Glass {
                    tint: mania_tint(),
                    ior: 1.45,
                    opacity: 0.85,
                    dispersion: 0.08,
                },
            )
            .at(Vec3::new(0.0, center_y, 0.0))
            .rotate(Vec3::new(1.0, 0.0, 0.0), flip_angle),
        )
        // ── Lower pyramid: depression ──
        // Same cone, but rotated π around X to flip it upside down,
        // then the whole flip_angle is applied on top.
        .object(
            SdfObject::new(
                SdfShape::Cone {
                    radius: 1.2,
                    height: lower_h,
                },
                Material::Subsurface {
                    color: depression_surface(),
                    scatter_color: depression_scatter(),
                    thickness: 0.7,
                    specular: 32.0,
                },
            )
            .at(Vec3::new(0.0, center_y, 0.0))
            .rotate(Vec3::new(1.0, 0.0, 0.0), flip_angle + std::f32::consts::PI),
        )
        // ── Mandelbulb heart at the junction ──
        // Pulses with a morph between low and high power
        .object(
            SdfObject::new(
                SdfShape::Mandelbulb {
                    power: 6.0 + 2.0 * (t * 0.8).sin(),
                    iterations: 12,
                },
                Material::rainbow_animated(t * 0.3),
            )
            .at(Vec3::new(0.0, center_y, 0.0))
            .rotate(Vec3::new(0.0, 1.0, 0.0), t * 0.4),
        )
        // ── Orbiting sentinel mirrors — 3 counter-rotating spheres ──
        .object(sentinel_sphere(t, 0.0, 2.8, center_y, 0.25))
        .object(sentinel_sphere(
            t,
            std::f32::consts::TAU / 3.0,
            2.8,
            center_y,
            0.20,
        ))
        .object(sentinel_sphere(
            t,
            2.0 * std::f32::consts::TAU / 3.0,
            2.8,
            center_y,
            0.22,
        ))
        // ── Counter-orbiting glass shards — gyroid fragments ──
        .object(orbiting_gyroid(t, 0.0, 3.5, center_y))
        .object(orbiting_gyroid(t, std::f32::consts::PI, 3.5, center_y))
        // ── Ground plane — dark mirror ──
        .object(SdfObject::new(
            SdfShape::Plane,
            Material::checkerboard(ground_a(), ground_b()),
        ))
        // ── Lighting: warm vs cool duality ──
        // Warm amber light — orbits with the mania side
        .light(SdfLight::new(
            Vec3::new((t * 0.4).cos() * 6.0, center_y + 4.0, (t * 0.4).sin() * 6.0),
            C::from_rgba8(255, 200, 120, 255),
            0.9,
        ))
        // Cool violet light — counter-orbits for depression
        .light(SdfLight::new(
            Vec3::new(
                (t * 0.4 + std::f32::consts::PI).cos() * 5.0,
                center_y + 2.0,
                (t * 0.4 + std::f32::consts::PI).sin() * 5.0,
            ),
            C::from_rgba8(120, 80, 255, 255),
            0.7,
        ))
        // Dim fill from below — catches the subsurface
        .light(SdfLight::new(
            Vec3::new(0.0, 0.3, 3.0),
            C::from_rgba8(80, 60, 140, 255),
            0.3,
        ))
        // ── Atmosphere ──
        .sky_color(sky())
        .ambient(0.03)
        .max_bounces(3);

    // Set camera with gentle orbit and breathing height
    let cam_angle = t * 0.2;
    let cam_radius = 8.0 + 0.5 * (t * 0.15).sin();
    let cam_height = center_y + 1.5 + 1.0 * (t * 0.25).sin();
    let cam_eye = Vec3::new(
        cam_angle.cos() * cam_radius,
        cam_height,
        cam_angle.sin() * cam_radius,
    );
    let cam_target = Vec3::new(0.0, center_y + 0.3 * (t * 0.3).sin(), 0.0);

    scene.camera(SdfCamera::new(cam_eye, cam_target, 42.0))
}

/// Create a mirror sentinel sphere orbiting at given phase.
fn sentinel_sphere(t: f32, phase: f32, orbit_radius: f32, center_y: f32, radius: f32) -> SdfObject {
    let angle = t * 0.5 + phase;
    let bob = (t * 1.2 + phase * 2.0).sin() * 0.8;
    SdfObject::new(
        SdfShape::Sphere { radius },
        Material::mirror(C::from_rgba8(200, 190, 220, 255), 0.95),
    )
    .at(Vec3::new(
        angle.cos() * orbit_radius,
        center_y + bob,
        angle.sin() * orbit_radius,
    ))
}

/// Create a small gyroid fragment orbiting counter to the sentinels.
fn orbiting_gyroid(t: f32, phase: f32, orbit_radius: f32, center_y: f32) -> SdfObject {
    let angle = -(t * 0.35) + phase; // counter-rotate
    let bob = (t * 0.9 + phase).cos() * 1.0;
    SdfObject::new(
        SdfShape::Gyroid {
            scale: 4.0,
            thickness: 0.08,
            bound: 0.6,
        },
        Material::Glass {
            tint: C::from_rgba8(180, 140, 255, 180),
            ior: 1.3,
            opacity: 0.6,
            dispersion: 0.12,
        },
    )
    .at(Vec3::new(
        angle.cos() * orbit_radius,
        center_y + bob,
        angle.sin() * orbit_radius,
    ))
    .rotate(Vec3::new(1.0, 1.0, 0.0), t * 0.7 + phase)
}

// ═══════════════════════════════════════════════════════════════════
// Window mode
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "window")]
fn run_window() -> Result<(), Box<dyn std::error::Error>> {
    use scry_engine::sdf::overlay::StatsOverlay;
    use scry_engine::transport::window::{run_loop_continuous, LoopAction};
    use winit::keyboard::KeyCode as WKey;

    #[cfg(feature = "sdf-gpu")]
    let mut gpu_ctx = scry_engine::sdf::SdfGpuContext::try_new(std::time::Duration::from_secs(5));

    let mut paused = false;
    let start = Instant::now();
    let mut frozen_time = 0.0_f32;
    let mut scale_idx: usize = 2; // default 75%
    let mut show_overlay = true;
    let mut profile_history = SdfProfileHistory::new(32);
    let mut stats_overlay = StatsOverlay::new(120);

    run_loop_continuous(
        960,
        640,
        "Duality — Bipolar Pyramids",
        true,
        move |backend, keys, (w, h)| {
            for key in keys {
                if !key.pressed {
                    continue;
                }
                match key.code {
                    WKey::Escape | WKey::KeyQ => return LoopAction::Exit,
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

            let scene = build_scene(elapsed);
            let render_scale = SCALE_STEPS[scale_idx];

            #[cfg(feature = "sdf-gpu")]
            if let Some(ref mut ctx) = gpu_ctx {
                let gpu_start = Instant::now();
                let mut pixmap = match scry_engine::sdf::SdfGpuRenderer::render_to_pixmap(
                    ctx, &scene, w, h, elapsed,
                ) {
                    Ok(p) => p,
                    Err(_) => return LoopAction::Continue,
                };
                let gpu_ms = gpu_start.elapsed().as_secs_f64() * 1000.0;

                if show_overlay {
                    stats_overlay.tick();
                    let pct = (render_scale * 100.0) as u32;
                    let gpu_profile =
                        scry_engine::sdf::SdfProfile::total_only((gpu_ms * 1000.0) as u64, w, h);
                    profile_history.push(gpu_profile);
                    let summary = profile_history.summary();
                    stats_overlay.render_overlay(&mut pixmap, &summary, pct, "Duality  [GPU]");
                }

                let _ = backend.blit(&pixmap);
                return LoopAction::Continue;
            }

            let (mut pixmap, profile) = match SdfRenderer::render_to_pixmap_upscaled_profiled(
                &scene,
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
                let pct = (SCALE_STEPS[scale_idx] * 100.0) as u32;
                stats_overlay.render_overlay(&mut pixmap, &summary, pct, "Duality");
            }

            let _ = backend.blit(&pixmap);
            LoopAction::Continue
        },
    )?;

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════
// Terminal mode
// ═══════════════════════════════════════════════════════════════════

fn run_terminal() -> Result<(), Box<dyn std::error::Error>> {
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

    let mut paused = false;
    let start = Instant::now();
    let mut frozen_time = 0.0_f32;
    let mut last_fps = 0.0_f32;
    let mut handle: Option<transport::backend::ImageHandle> = None;

    #[cfg(feature = "sdf-gpu")]
    let mut gpu_ctx = scry_engine::sdf::SdfGpuContext::try_new(std::time::Duration::from_secs(5));

    let mut profiling = false;
    let mut profile_history = SdfProfileHistory::new(32);
    let mut scale_idx: usize = 1; // default 50%

    loop {
        let frame_start = Instant::now();

        let elapsed = if paused {
            frozen_time
        } else {
            let e = start.elapsed().as_secs_f32();
            frozen_time = e;
            e
        };

        let (cols, rows) = terminal::size()?;
        let status_rows = if profiling { 3 } else { 1 };
        let full_w = u32::from(cols) * u32::from(font.width);
        let full_h = u32::from(rows.saturating_sub(status_rows + 1)) * u32::from(font.height);

        if full_w > 0 && full_h > 0 {
            let cap_w = full_w.min(MAX_RENDER_W);
            let cap_h = full_h.min(MAX_RENDER_H);
            let render_scale = SCALE_STEPS[scale_idx];

            let scene = build_scene(elapsed);

            let pixmap = 'render: {
                #[cfg(feature = "sdf-gpu")]
                if let Some(ref mut ctx) = gpu_ctx {
                    let gpu_start = Instant::now();
                    if let Ok(pm) = scry_engine::sdf::SdfGpuRenderer::render_to_pixmap(
                        ctx, &scene, cap_w, cap_h, elapsed,
                    ) {
                        if profiling {
                            let gpu_us = gpu_start.elapsed().as_micros() as u64;
                            profile_history.push(scry_engine::sdf::SdfProfile::total_only(
                                gpu_us, cap_w, cap_h,
                            ));
                        }
                        break 'render pm;
                    }
                }

                if profiling {
                    let (pm, profile) = SdfRenderer::render_to_pixmap_upscaled_profiled(
                        &scene,
                        cap_w,
                        cap_h,
                        render_scale,
                        elapsed,
                    )?;
                    profile_history.push(profile);
                    pm
                } else {
                    SdfRenderer::render_to_pixmap_upscaled(
                        &scene,
                        cap_w,
                        cap_h,
                        render_scale,
                        elapsed,
                    )?
                }
            };

            let pos = TerminalPosition::new(0, 0, cols, rows.saturating_sub(status_rows + 1));

            let new_handle = if let Some(ref old) = handle {
                backend.replace(old, &pixmap, pos, -1)?
            } else {
                backend.transmit(&pixmap, pos, -1)?
            };
            handle = Some(new_handle);
        }

        // ── Status bar ──
        let (cols, rows) = terminal::size()?;
        let frame_ms = frame_start.elapsed().as_secs_f32();
        if frame_ms > 0.0 {
            last_fps = last_fps * 0.8 + (1.0 / frame_ms) * 0.2;
        }

        if profiling {
            let summary = profile_history.summary();
            let total_ms = summary.total_us as f64 / 1000.0;
            let pct = (SCALE_STEPS[scale_idx] * 100.0) as u32;
            #[cfg(feature = "sdf-gpu")]
            let gpu_tag = if gpu_ctx.is_some() { "  [GPU]" } else { "" };
            #[cfg(not(feature = "sdf-gpu"))]
            let gpu_tag = "";

            out.queue(cursor::MoveTo(0, rows.saturating_sub(3)))?;
            out.queue(style::Print(format!(
                "\x1b[K Duality{} | {:.0} fps | {}x{} @{}% | {:.1}ms",
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
                "\x1b[K [+/-] scale  [space] pause  [p] profile  [q] quit",
            ))?;
        } else {
            let pct = (SCALE_STEPS[scale_idx] * 100.0) as u32;
            #[cfg(feature = "sdf-gpu")]
            let gpu_tag = if gpu_ctx.is_some() { "  [GPU]" } else { "" };
            #[cfg(not(feature = "sdf-gpu"))]
            let gpu_tag = "";
            out.queue(cursor::MoveTo(0, rows - 1))?;
            out.queue(style::Print(format!(
                "\x1b[K Duality{} | {:.0} fps | {}% | [+/-] scale [space] pause [p] profile [q] quit",
                gpu_tag, last_fps, pct,
            )))?;
        }
        out.flush()?;

        // ── Input ──
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
            eprintln!(
                "  cargo run --example duality --features \"sdf,window\" --release -- --window"
            );
            std::process::exit(1);
        }
    }

    run_terminal()
}
