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
//!   `g`      — toggle GPU/CPU rendering
//!   `+`/`-`  — cycle render scale (25%, 50%, 75%, 100%)
//!   `q`/`Esc` — quit
//!
//! Run with:
//!   Terminal: `cargo run --example text3d_showcase --features "sdf-text,sdf-gpu,widget" --release`
//!   Window:   `cargo run --example text3d_showcase --features "sdf-text,sdf-gpu,widget,window" --release -- --window`

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::wildcard_imports
)]

use std::io::stdout;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use scry_engine::prelude::{Picker, PixelCanvasState, PixelCanvasWidget};
use scry_engine::scene::command::ImageData;
use scry_engine::scene::style::Color as C;
use scry_engine::scene::PixelCanvas;
use scry_engine::sdf::pipeline::SdfPipeline;
use scry_engine::sdf::*;

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
// Terminal mode — ratatui + PixelCanvasWidget (matches masonic_mirror)
// ═══════════════════════════════════════════════════════════════════

fn build_canvas(w: u32, h: u32, sdf_image: ImageData) -> PixelCanvas {
    if w == 0 || h == 0 {
        return PixelCanvas::new(1, 1);
    }
    let mut canvas = PixelCanvas::new(w, h).background(C::from_rgba8(10, 12, 20, 255));
    canvas = canvas.image(sdf_image, 0.0, 0.0).done();
    canvas
}

fn run_terminal() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let picker = Picker::detect();
    let backend = picker.create_backend();
    let mut px_state = PixelCanvasState::new(backend, picker.font_size());

    let mut sdf_pipeline = SdfPipeline::new();

    let mut static_scenes = build_static_scenes();
    let mut dynamic_scene: SdfScene = build_museum();

    let mut preset_idx: usize = 0;
    let mut paused = false;
    let start = Instant::now();
    let mut frozen_time = 0.0_f32;
    let mut last_frame = Instant::now();
    let mut scale_idx: usize = 2; // default 75%

    loop {
        let now = Instant::now();
        let dt = now.duration_since(last_frame);
        last_frame = now;
        let fps = if dt.as_secs_f32() > 0.0 {
            1.0 / dt.as_secs_f32()
        } else {
            0.0
        };

        let elapsed = if paused {
            frozen_time
        } else {
            let e = start.elapsed().as_secs_f32();
            frozen_time = e;
            e
        };

        // Compute layout
        let term_size = terminal.size()?;
        let term_rect = ratatui::layout::Rect::new(0, 0, term_size.width, term_size.height);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(2)])
            .split(term_rect);

        let area = chunks[0];
        let font = px_state.font_size();
        let w = u32::from(area.width) * u32::from(font.width);
        let h = u32::from(area.height) * u32::from(font.height);

        if w > 0 && h > 0 {
            // Get the active scene
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

            // Render via SdfPipeline (handles GPU/CPU, scale, upscale)
            let sdf_result = sdf_pipeline.render(scene, w, h, elapsed);
            let canvas = build_canvas(w, h, sdf_result.image);

            let render_mode = sdf_pipeline.backend_name();
            let scale_pct = (sdf_pipeline.get_render_scale() * 100.0) as u32;

            terminal.draw(|frame| {
                frame.render_stateful_widget(
                    PixelCanvasWidget::new(canvas).skip_cache().z_index(-1),
                    area,
                    &mut px_state,
                );

                let status_text = format!(
                    " Text3D: {} | {render_mode} {scale_pct}% | {fps:.0}fps | {elapsed:.1}s | [1-5] scene [+/-] scale [space] pause [g] gpu [q] quit",
                    PRESET_NAMES[preset_idx],
                );
                let status = Paragraph::new(status_text).block(Block::default().borders(Borders::TOP));
                frame.render_widget(status, chunks[1]);
            })?;
            px_state.flush()?;
            sdf_pipeline.flush();
        }

        if event::poll(Duration::ZERO)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Char('1') => preset_idx = 0,
                        KeyCode::Char('2') => preset_idx = 1,
                        KeyCode::Char('3') => preset_idx = 2,
                        KeyCode::Char('4') => preset_idx = 3,
                        KeyCode::Char('5') => preset_idx = 4,
                        KeyCode::Char(' ') => paused = !paused,
                        KeyCode::Char('g') => {
                            let currently_gpu = sdf_pipeline.is_gpu_active();
                            sdf_pipeline.set_gpu_active(!currently_gpu);
                        }
                        KeyCode::Char('+' | '=') => {
                            if scale_idx < SCALE_STEPS.len() - 1 {
                                scale_idx += 1;
                                sdf_pipeline.set_render_scale(SCALE_STEPS[scale_idx]);
                            }
                        }
                        KeyCode::Char('-') => {
                            if scale_idx > 0 {
                                scale_idx -= 1;
                                sdf_pipeline.set_render_scale(SCALE_STEPS[scale_idx]);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    px_state.cleanup();
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
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
