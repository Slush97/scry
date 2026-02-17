//! **Aurora Borealis** — northern lights simulation.
//!
//! Full-screen aurora curtains with realistic vertical ray structure,
//! sine-wave displacement, star field background, and ground reflection.
//! A meditative, slow-evolving art piece.
//!
//! Controls:
//!   `1`–`3`  — intensity: Calm / Vivid / Storm
//!   `s`      — toggle stars
//!   `r`      — toggle ground reflection
//!   `Space`  — pause/resume
//!   `q`      — quit
//!
//! Run with: `cargo run --example aurora_borealis --release`

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

use std::f32::consts::TAU;
use std::io::stdout;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use scry_engine::prelude::{Picker, PixelCanvasState, PixelCanvasWidget, ProtocolKind};
use scry_engine::scene::style::{Color as C, Point};
use scry_engine::scene::PixelCanvas;
use scry_engine::transport;

// ═══════════════════════════════════════════════════════════════════
// Deterministic pseudo-random from a seed (for star positions)
// ═══════════════════════════════════════════════════════════════════

fn hash_f(seed: u32) -> f32 {
    // Simple hash to float [0, 1)
    let mut x = seed;
    x = x.wrapping_mul(0x45d9f3b).wrapping_add(0x12345);
    x = ((x >> 16) ^ x).wrapping_mul(0x45d9f3b);
    x = (x >> 16) ^ x;
    (x & 0xFFFF) as f32 / 65536.0
}

// ═══════════════════════════════════════════════════════════════════
// Aurora noise helpers
// ═══════════════════════════════════════════════════════════════════

/// Simple smooth noise for aurora wave shapes.
fn wave_noise(x: f32, t: f32) -> f32 {
    let a = x.mul_add(1.7, t * 0.4).sin() * 0.5;
    let b = x.mul_add(3.1, -(t * 0.3)).sin() * 0.25;
    let c = x.mul_add(5.3, t * 0.7).sin() * 0.125;
    let d = x.mul_add(0.8, -(t * 0.15)).sin() * 0.6;
    a + b + c + d
}

/// Aurora curtain vertical displacement at horizontal position x.
fn curtain_y(x: f32, time: f32, band: usize) -> f32 {
    let phase = band as f32 * 1.3;
    let freq_mod = (band as f32).mul_add(0.2, 1.0);
    wave_noise(x * freq_mod, time + phase) * 0.15
}

/// Aurora brightness at a given position (0..1).
fn aurora_brightness(x_norm: f32, y_norm: f32, time: f32, intensity: f32) -> f32 {
    // Vertical falloff (brighter at top, fading down)
    let vert = y_norm.mul_add(-2.0, 1.0).clamp(0.0, 1.0).powf(0.6);

    // Horizontal variation
    let horiz = 0.5f32.mul_add(x_norm.mul_add(4.0, time * 0.2).sin(), 0.5);
    let horiz2 = 0.7f32.mul_add(x_norm.mul_add(7.0, -(time * 0.15)).sin().abs(), 0.3);

    // Shimmer (fast vertical rays)
    let shimmer = 0.3f32.mul_add(
        (x_norm * 80.0).sin() * time.mul_add(5.0, x_norm * 20.0).cos(),
        0.7,
    );

    vert * horiz * horiz2 * shimmer * intensity
}

// ═══════════════════════════════════════════════════════════════════
// Presets
// ═══════════════════════════════════════════════════════════════════

#[derive(Clone, Copy)]
struct AuroraPreset {
    name: &'static str,
    curtain_count: usize,
    intensity: f32,
    color_shift_speed: f32,
}

const PRESETS: [AuroraPreset; 3] = [
    AuroraPreset {
        name: "Calm",
        curtain_count: 3,
        intensity: 0.6,
        color_shift_speed: 0.1,
    },
    AuroraPreset {
        name: "Vivid",
        curtain_count: 5,
        intensity: 1.0,
        color_shift_speed: 0.2,
    },
    AuroraPreset {
        name: "Storm",
        curtain_count: 7,
        intensity: 1.5,
        color_shift_speed: 0.4,
    },
];

struct AuroraState {
    preset_idx: usize,
    show_stars: bool,
    show_reflection: bool,
    paused: bool,
}

impl AuroraState {
    const fn new() -> Self {
        Self {
            preset_idx: 1, // Start with Vivid
            show_stars: true,
            show_reflection: true,
            paused: false,
        }
    }

    const fn preset(&self) -> &AuroraPreset {
        &PRESETS[self.preset_idx]
    }
}

// ═══════════════════════════════════════════════════════════════════
// Window mode
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "window")]
fn run_window() -> Result<(), Box<dyn std::error::Error>> {
    use scry_engine::rasterize::Rasterizer;
    use scry_engine::transport::window::{run_loop_continuous, LoopAction};
    use winit::keyboard::KeyCode as WKey;

    let mut aurora = AuroraState::new();
    let start = Instant::now();
    let mut frozen_time = 0.0_f32;

    run_loop_continuous(
        960,
        640,
        "Aurora Borealis",
        true,
        move |backend, keys, (w, h)| {
            for key in keys {
                if !key.pressed {
                    continue;
                }
                match key.code {
                    WKey::Escape | WKey::KeyQ => return LoopAction::Exit,
                    WKey::Digit1 => aurora.preset_idx = 0,
                    WKey::Digit2 => aurora.preset_idx = 1,
                    WKey::Digit3 => aurora.preset_idx = 2,
                    WKey::KeyS => aurora.show_stars = !aurora.show_stars,
                    WKey::KeyR => aurora.show_reflection = !aurora.show_reflection,
                    WKey::Space => aurora.paused = !aurora.paused,
                    _ => {}
                }
            }

            let elapsed = if aurora.paused {
                frozen_time
            } else {
                let e = start.elapsed().as_secs_f32();
                frozen_time = e;
                e
            };

            let canvas = build_aurora_scene(w, h, &aurora, elapsed);
            if let Ok(pixmap) = Rasterizer::rasterize(&canvas) {
                let _ = backend.blit(&pixmap);
            }
            LoopAction::Continue
        },
    )?;
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
            std::process::exit(1);
        }
    }

    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let picker = Picker::detect();
    let backend: Box<dyn transport::ProtocolBackend> = match picker.protocol() {
        ProtocolKind::Kitty => Box::new(transport::kitty::KittyBackend::new(picker.font_size())),
        _ => Box::new(transport::halfblock::HalfblockBackend::new()),
    };
    let mut px_state = PixelCanvasState::new(backend, picker.font_size());

    let mut aurora = AuroraState::new();
    let start = Instant::now();
    let mut last_frame = Instant::now();
    let mut frozen_time = 0.0_f32;

    loop {
        let now = Instant::now();
        let _dt = now.duration_since(last_frame).as_secs_f32();
        last_frame = now;

        let elapsed = if aurora.paused {
            frozen_time
        } else {
            let e = now.duration_since(start).as_secs_f32();
            frozen_time = e;
            e
        };

        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(3)])
                .split(frame.area());

            let area = chunks[0];
            let font = px_state.font_size();
            let w = u32::from(area.width) * u32::from(font.width);
            let h = u32::from(area.height) * u32::from(font.height);
            let canvas = build_aurora_scene(w, h, &aurora, elapsed);

            frame.render_stateful_widget(
                PixelCanvasWidget::new(canvas).skip_cache().z_index(-1),
                area,
                &mut px_state,
            );

            let status_text = format!(
                " Aurora {} │ Stars: {} │ Reflection: {} │ [1-3] intensity [s] stars [r] reflect [space] pause [q] quit",
                aurora.preset().name,
                if aurora.show_stars { "ON" } else { "OFF" },
                if aurora.show_reflection { "ON" } else { "OFF" },
            );
            let status = Paragraph::new(status_text)
                .block(Block::default().borders(Borders::TOP));
            frame.render_widget(status, chunks[1]);
        })?;
        px_state.flush()?;

        if event::poll(Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Char('1') => aurora.preset_idx = 0,
                        KeyCode::Char('2') => aurora.preset_idx = 1,
                        KeyCode::Char('3') => aurora.preset_idx = 2,
                        KeyCode::Char('s') => aurora.show_stars = !aurora.show_stars,
                        KeyCode::Char('r') => aurora.show_reflection = !aurora.show_reflection,
                        KeyCode::Char(' ') => aurora.paused = !aurora.paused,
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
// Scene builder
// ═══════════════════════════════════════════════════════════════════

fn build_aurora_scene(w: u32, h: u32, aurora: &AuroraState, time: f32) -> PixelCanvas {
    if w == 0 || h == 0 {
        return PixelCanvas::new(1, 1);
    }

    let wf = w as f32;
    let hf = h as f32;
    let preset = aurora.preset();

    // Sky gradient: deep dark blue at top → near-black at horizon
    let mut canvas = PixelCanvas::new(w, h);

    // Background sky gradient
    canvas = canvas
        .gradient(0.0, 0.0, wf, hf)
        .linear(Point::new(wf / 2.0, 0.0), Point::new(wf / 2.0, hf))
        .stop(0.0, C::from_rgba8(3, 5, 20, 255)) // Deep night sky
        .stop(0.5, C::from_rgba8(5, 8, 25, 255)) // Dark blue
        .stop(0.75, C::from_rgba8(8, 12, 30, 255)) // Horizon
        .stop(1.0, C::from_rgba8(2, 3, 8, 255)) // Ground
        .done();

    // Ground plane (dark lake/snow line at 80% height)
    let ground_y = hf * 0.80;

    // Star field
    if aurora.show_stars {
        canvas = draw_stars(canvas, wf, ground_y, time);
    }

    // Aurora curtains
    canvas = draw_aurora_curtains(canvas, wf, hf, ground_y, time, preset);

    // Ground plane overlay
    canvas = canvas
        .gradient(0.0, ground_y, wf, hf - ground_y)
        .linear(Point::new(wf / 2.0, ground_y), Point::new(wf / 2.0, hf))
        .stop(0.0, C::from_rgba8(5, 8, 15, 220))
        .stop(0.3, C::from_rgba8(3, 5, 10, 240))
        .stop(1.0, C::from_rgba8(1, 2, 5, 255))
        .done();

    // Ground reflection of aurora
    if aurora.show_reflection {
        canvas = draw_reflection(canvas, wf, hf, ground_y, time, preset);
    }

    // Horizon glow (thin bright line where aurora meets ground)
    let glow_intensity = 0.05f32.mul_add((time * 0.5).sin(), 0.1);
    canvas = canvas
        .line(0.0, ground_y, wf, ground_y)
        .color(C::from_rgba8(20, 60, 30, (glow_intensity * 255.0) as u8))
        .width(2.0)
        .done();

    canvas
}

// ═══════════════════════════════════════════════════════════════════
// Star field
// ═══════════════════════════════════════════════════════════════════

fn draw_stars(mut canvas: PixelCanvas, wf: f32, ground_y: f32, time: f32) -> PixelCanvas {
    let star_count = 200;

    for i in 0..star_count {
        let x = hash_f(i * 3 + 7) * wf;
        let y = hash_f(i * 3 + 13) * ground_y * 0.95;
        let brightness_base = hash_f(i * 3 + 19);
        let size = hash_f(i * 3 + 23);

        // Twinkle
        let twinkle_phase = hash_f(i * 3 + 29) * TAU;
        let twinkle_speed = 0.5 + hash_f(i * 3 + 31) * 3.0;
        let twinkle = 0.6f32.mul_add(time.mul_add(twinkle_speed, twinkle_phase).sin().abs(), 0.4);

        let brightness = brightness_base * twinkle;
        let alpha = (brightness * 255.0).min(255.0) as u8;

        // Star color: mostly white, some blue/yellow tints
        let tint = hash_f(i * 3 + 37);
        let star_color = if tint < 0.15 {
            C::from_rgba8(180, 200, 255, alpha) // Blue-white
        } else if tint < 0.25 {
            C::from_rgba8(255, 240, 200, alpha) // Warm yellow
        } else {
            C::from_rgba8(255, 255, 255, alpha)
        };

        let r = 0.5 + size * 1.5;
        canvas = canvas.circle(x, y, r).fill(star_color).done();

        // Bright stars get a cross sparkle
        if brightness_base > 0.85 && twinkle > 0.7 {
            let sparkle_len = r * 3.0;
            let sparkle_color = star_color.with_alpha(brightness * 0.3);
            canvas = canvas
                .line(x - sparkle_len, y, x + sparkle_len, y)
                .color(sparkle_color)
                .width(0.5)
                .done();
            canvas = canvas
                .line(x, y - sparkle_len, x, y + sparkle_len)
                .color(sparkle_color)
                .width(0.5)
                .done();
        }
    }

    canvas
}

// ═══════════════════════════════════════════════════════════════════
// Aurora curtains
// ═══════════════════════════════════════════════════════════════════

fn draw_aurora_curtains(
    mut canvas: PixelCanvas,
    wf: f32,
    _hf: f32,
    ground_y: f32,
    time: f32,
    preset: &AuroraPreset,
) -> PixelCanvas {
    // Each curtain is a series of vertical polygon strips
    let strip_count = 60; // Horizontal resolution
    let strip_width = wf / strip_count as f32;

    for band in 0..preset.curtain_count {
        let band_offset = band as f32 / preset.curtain_count as f32;

        // Curtain vertical position (fraction of sky height)
        let base_y_frac = band_offset.mul_add(0.25, 0.15);
        let curtain_height = ground_y * band_offset.mul_add(-0.08, 0.4);

        for strip in 0..strip_count {
            let x_left = strip as f32 * strip_width;
            let x_right = x_left + strip_width + 1.0; // +1 for overlap
            let x_norm_left = strip as f32 / strip_count as f32;
            let x_norm_right = (strip + 1) as f32 / strip_count as f32;

            // Wave displacement
            let wave_left = curtain_y(x_norm_left, time, band);
            let wave_right = curtain_y(x_norm_right, time, band);

            // Curtain top and bottom y positions
            let top_left = ground_y * (base_y_frac + wave_left);
            let top_right = ground_y * (base_y_frac + wave_right);
            let bot_left = top_left + curtain_height * 0.3f32.mul_add((1.0 + wave_left).abs(), 0.7);
            let bot_right =
                top_right + curtain_height * 0.3f32.mul_add((1.0 + wave_right).abs(), 0.7);

            // Brightness
            let brightness = aurora_brightness(
                x_norm_left,
                base_y_frac,
                (band as f32).mul_add(0.5, time),
                preset.intensity,
            );

            if brightness < 0.01 {
                continue;
            }

            // Aurora colors: green (557.7nm) dominant, pink/violet at edges
            let color_shift = time.mul_add(preset.color_shift_speed, band as f32 * 0.3);
            let green_amount = 0.3f32.mul_add((x_norm_left * 3.0 + color_shift).cos(), 0.6);

            let aurora_color = if green_amount > 0.5 {
                // Green-dominant
                let g_bright = brightness * green_amount;
                C::from_rgba(
                    0.1 * g_bright,
                    0.8 * g_bright,
                    0.3 * g_bright,
                    brightness * 0.25,
                )
            } else {
                // Pink/violet at curtain edges
                let p_bright = brightness * (1.0 - green_amount);
                C::from_rgba(
                    0.6 * p_bright,
                    0.15 * p_bright,
                    0.5 * p_bright,
                    brightness * 0.20,
                )
            };

            // Draw vertical strip as a polygon
            let points = vec![
                (x_left, top_left),
                (x_right, top_right),
                (x_right, bot_right),
                (x_left, bot_left),
            ];

            canvas = canvas.polygon(points).fill(aurora_color).done();

            // Vertical ray structure (thin bright lines within the curtain)
            if strip % 3 == 0 && brightness > 0.15 {
                let ray_x = x_left + strip_width * 0.5;
                let ray_brightness =
                    brightness * 0.5f32.mul_add((ray_x * 0.3 + time * 2.0).sin().abs(), 0.5);
                let ray_color = C::from_rgba(
                    0.15 * ray_brightness,
                    0.9 * ray_brightness,
                    0.4 * ray_brightness,
                    ray_brightness * 0.15,
                );

                canvas = canvas
                    .line(ray_x, top_left, ray_x, bot_left * 0.95)
                    .color(ray_color)
                    .width(1.0)
                    .done();
            }
        }
    }

    canvas
}

// ═══════════════════════════════════════════════════════════════════
// Ground reflection
// ═══════════════════════════════════════════════════════════════════

fn draw_reflection(
    mut canvas: PixelCanvas,
    wf: f32,
    hf: f32,
    ground_y: f32,
    time: f32,
    preset: &AuroraPreset,
) -> PixelCanvas {
    // Simplified reflection: vertical strips of aurora color below ground line
    let strip_count = 40;
    let strip_width = wf / strip_count as f32;
    let reflect_height = (hf - ground_y) * 0.7;

    for strip in 0..strip_count {
        let x_left = strip as f32 * strip_width;
        let x_right = x_left + strip_width + 1.0;
        let x_norm = strip as f32 / strip_count as f32;

        // Sample aurora brightness above this point
        let brightness = aurora_brightness(x_norm, 0.3, time, preset.intensity) * 0.3; // Reflection is dimmer

        if brightness < 0.02 {
            continue;
        }

        // Distortion from "water" surface
        let ripple = (x_norm * 20.0 + time * 1.5).sin() * 3.0;

        // Reflected color (more diffuse, bluer tint from water)
        let r_color = C::from_rgba(
            0.05 * brightness,
            0.5 * brightness,
            0.25 * brightness,
            brightness * 0.15,
        );

        let top_y = ground_y + 3.0 + ripple.abs();
        let bot_y = (top_y + reflect_height * brightness).min(hf);

        let points = vec![
            (x_left, top_y),
            (x_right, top_y + ripple * 0.5),
            (x_right, bot_y),
            (x_left, bot_y - ripple * 0.3),
        ];

        canvas = canvas.polygon(points).fill(r_color).done();
    }

    canvas
}
