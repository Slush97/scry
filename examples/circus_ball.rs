//! **Circus Ball** — Flash-style bouncing ball animation.
//!
//! A bold, saturated circus ball bounces across the screen with exaggerated
//! squash/stretch physics, motion blur trails, and pop-in text.
//!
//! **Features showcased:**
//! - `fill_linear_gradient` / `fill_radial_gradient` on shapes
//! - Per-shape `.transform()` for spinning ball contents
//! - Clip regions (clip ball shape for stripe overlay)
//! - `AnimationSequence` + `Stagger` + `Parallel`
//! - `Spring` physics for squash/stretch
//! - Dash patterns + Line caps
//! - Text rendering with alignment
//! - Blend modes for glow effects
//! - `Easing::Bounce`, `Easing::Elastic`
//! - `star()` helper for decorative star shapes
//!
//! Controls:
//!   `Space` — pause/resume
//!   `q`     — quit
//!
//! Run with: `cargo run --example circus_ball --features "text,widget" --release`

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::items_after_statements
)]

use std::f32::consts::{FRAC_PI_2, TAU};
use std::io::stdout;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use scry_engine::prelude::{
    AnimationSequence, Easing, Picker, PixelCanvasState, PixelCanvasWidget, ProtocolKind,
    SequencePlayer, SpringConfig, TextAlign,
};
use scry_engine::scene::style::{
    BlendMode, Color, DashPattern, GradientDef, GradientKind, GradientStop, LineCap, Point,
    Transform,
};
use scry_engine::scene::PixelCanvas;
use scry_engine::transport;

// ═══════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════

const CYCLE_DURATION: f32 = 8.0;
const BALL_RADIUS: f32 = 60.0;
const TRAIL_COUNT: usize = 5;

// ═══════════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════════

struct CircusState {
    paused: bool,
    sequence: SequencePlayer,
}

impl CircusState {
    fn new() -> Self {
        let seq = build_sequence();
        Self {
            paused: false,
            sequence: SequencePlayer::new(seq),
        }
    }
}

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

fn build_sequence() -> AnimationSequence {
    AnimationSequence::new()
        // Multi-ball staggered entrance (SHOWCASE: Stagger)
        .stagger(ms(300), |p| {
            p.branch(|b| b.tween("ball0_enter", 0.0, 1.0, ms(800), Easing::Bounce))
                .branch(|b| b.tween("ball1_enter", 0.0, 1.0, ms(800), Easing::Bounce))
                .branch(|b| b.tween("ball2_enter", 0.0, 1.0, ms(800), Easing::Bounce))
        })
        // Bounce phase
        .parallel(|p| {
            p.branch(|b| b.tween("bounce_x", 0.0, 1.0, ms(6000), Easing::Linear))
                .branch(|b| {
                    b.tween("bounce_y", 0.0, 1.0, ms(1500), Easing::Bounce)
                        .tween("bounce_y", 1.0, 0.0, ms(1500), Easing::EaseInQuad)
                        .tween("bounce_y", 0.0, 1.0, ms(1500), Easing::Bounce)
                        .tween("bounce_y", 1.0, 0.0, ms(1500), Easing::EaseInQuad)
                })
                .branch(|b| b.spring_to("squash", 0.0, 1.0, SpringConfig::BOUNCY))
        })
        // Text pop
        .tween("wow_pop", 0.0, 1.0, ms(400), Easing::Elastic)
        .wait(ms(500))
        .tween("wow_pop", 1.0, 0.0, ms(200), Easing::EaseInCubic)
}

// ═══════════════════════════════════════════════════════════════════
// Ball rendering
// ═══════════════════════════════════════════════════════════════════

fn draw_ball(
    canvas: PixelCanvas,
    cx: f32,
    cy: f32,
    radius: f32,
    spin: f32,
    squash: f32,
    opacity: f32,
    _w: f32,
    _h: f32,
) -> PixelCanvas {
    if opacity <= 0.0 {
        return canvas;
    }

    // Squash/stretch via ellipse: when squash > 0, compress ry, expand rx
    let squash_factor = 1.0 + squash * 0.15 * (spin * 2.0).sin().abs();
    let rx = radius * squash_factor;
    let ry = radius / squash_factor;

    // Main body: radial gradient (SHOWCASE)
    let body_gradient = GradientDef {
        kind: GradientKind::Radial {
            center: Point::new(cx - radius * 0.3, cy - radius * 0.3),
            radius: radius * 1.5,
        },
        stops: vec![
            GradientStop {
                position: 0.0,
                color: Color::from_rgba8(255, 80, 80, (opacity * 255.0) as u8),
            },
            GradientStop {
                position: 0.5,
                color: Color::from_rgba8(200, 20, 20, (opacity * 255.0) as u8),
            },
            GradientStop {
                position: 1.0,
                color: Color::from_rgba8(100, 0, 0, (opacity * 255.0) as u8),
            },
        ],
    };

    let canvas = canvas
        .ellipse(cx, cy, rx, ry)
        .fill_radial_gradient(body_gradient)
        .opacity(opacity)
        .done();

    // Blue stripes — clipped to ball shape (SHOWCASE: clip region)
    // Build a circular clip path
    let clip_path = {
        let mut pb = tiny_skia::PathBuilder::new();
        // Approximate circle with bezier curves via the path builder
        let num_segs = 64;
        for i in 0..num_segs {
            let angle = (i as f32 / num_segs as f32) * TAU;
            let px = cx + rx * angle.cos();
            let py = cy + ry * angle.sin();
            if i == 0 {
                pb.move_to(px, py);
            } else {
                pb.line_to(px, py);
            }
        }
        pb.close();
        pb.finish()
    };

    let canvas = if let Some(clip) = clip_path {
        canvas
            .group(Transform::rotate_at(spin * 0.5, cx, cy))
            .clip_path(clip)
            .opacity(opacity)
            .canvas(|inner| {
                let stripe_h = radius * 0.25;
                let num_stripes = 5;
                let mut c = inner;
                for i in 0..num_stripes {
                    let sy = cy - radius + (i as f32 * stripe_h * 2.0) + stripe_h * 0.5;
                    c = c
                        .rect(cx - rx, sy, rx * 2.0, stripe_h)
                        .fill(Color::from_rgba(0.0, 0.2, 0.8, opacity * 0.7))
                        .done();
                }
                c
            })
            .done()
    } else {
        canvas
    };

    // Yellow end caps with linear gradient (SHOWCASE)
    let cap_gradient = GradientDef {
        kind: GradientKind::Linear {
            start: Point::new(cx - rx * 0.5, cy),
            end: Point::new(cx + rx * 0.5, cy),
        },
        stops: vec![
            GradientStop {
                position: 0.0,
                color: Color::from_rgba8(255, 220, 50, (opacity * 255.0) as u8),
            },
            GradientStop {
                position: 1.0,
                color: Color::from_rgba8(255, 180, 0, (opacity * 255.0) as u8),
            },
        ],
    };

    // Top cap arc
    let canvas = canvas
        .arc(cx, cy, ry * 0.95, -FRAC_PI_2 - 0.6, 1.2)
        .fill_linear_gradient(cap_gradient.clone())
        .stroke(Color::from_rgba(0.8, 0.7, 0.0, opacity), 1.5)
        .opacity(opacity)
        .done();

    // Bottom cap arc
    let canvas = canvas
        .arc(cx, cy, ry * 0.95, FRAC_PI_2 - 0.6, 1.2)
        .fill_linear_gradient(cap_gradient)
        .stroke(Color::from_rgba(0.8, 0.7, 0.0, opacity), 1.5)
        .opacity(opacity)
        .done();

    // Stars on caps (per-shape transform for rotation — SHOWCASE)
    let star_r = radius * 0.12;
    let canvas = canvas
        .star(cx, cy - ry * 0.7, star_r, star_r * 0.4, 5)
        .fill(Color::from_rgba(1.0, 1.0, 1.0, opacity))
        .transform(Transform::rotate_at(spin, cx, cy - ry * 0.7))
        .done();

    let canvas = canvas
        .star(cx, cy + ry * 0.7, star_r, star_r * 0.4, 5)
        .fill(Color::from_rgba(1.0, 1.0, 1.0, opacity))
        .transform(Transform::rotate_at(-spin, cx, cy + ry * 0.7))
        .done();

    // Highlight
    canvas
        .circle(cx - radius * 0.25, cy - radius * 0.25, radius * 0.15)
        .fill(Color::from_rgba(1.0, 1.0, 1.0, opacity * 0.5))
        .done()
}

fn draw_motion_trail(
    canvas: PixelCanvas,
    positions: &[(f32, f32)],
    radius: f32,
    _w: f32,
    _h: f32,
) -> PixelCanvas {
    let mut c = canvas;
    for (i, &(tx, ty)) in positions.iter().enumerate() {
        let trail_opacity = (i as f32 / positions.len() as f32) * 0.2;
        let trail_r = radius * (0.5 + 0.5 * (i as f32 / positions.len() as f32));
        c = c
            .circle(tx, ty, trail_r)
            .fill(Color::from_rgba(1.0, 0.2, 0.2, trail_opacity))
            .done();
    }
    c
}

fn draw_decorative_border(canvas: PixelCanvas, w: f32, h: f32) -> PixelCanvas {
    // Dash patterns + Line caps (SHOWCASE)
    let dash = DashPattern::new(vec![8.0, 4.0, 2.0, 4.0], 0.0);

    let canvas = canvas
        .line(10.0, 10.0, w - 10.0, 10.0)
        .color(Color::from_rgba(1.0, 0.84, 0.0, 0.5))
        .width(2.0)
        .dash(dash.clone())
        .line_cap(LineCap::Round)
        .done();

    let canvas = canvas
        .line(10.0, h - 10.0, w - 10.0, h - 10.0)
        .color(Color::from_rgba(1.0, 0.84, 0.0, 0.5))
        .width(2.0)
        .dash(dash.clone())
        .line_cap(LineCap::Round)
        .done();

    let canvas = canvas
        .line(10.0, 10.0, 10.0, h - 10.0)
        .color(Color::from_rgba(1.0, 0.84, 0.0, 0.5))
        .width(2.0)
        .dash(dash.clone())
        .line_cap(LineCap::Round)
        .done();

    canvas
        .line(w - 10.0, 10.0, w - 10.0, h - 10.0)
        .color(Color::from_rgba(1.0, 0.84, 0.0, 0.5))
        .width(2.0)
        .dash(dash)
        .line_cap(LineCap::Round)
        .done()
}

fn draw_floor_reflection(canvas: PixelCanvas, w: f32, h: f32, floor_y: f32) -> PixelCanvas {
    // Gradient floor reflection (SHOWCASE)
    canvas
        .gradient(0.0, floor_y, w, h - floor_y)
        .stop(0.0, Color::from_rgba(0.15, 0.15, 0.2, 0.4))
        .stop(1.0, Color::TRANSPARENT)
        .done()
}

// ═══════════════════════════════════════════════════════════════════
// Scene assembly
// ═══════════════════════════════════════════════════════════════════

fn build_scene(w: u32, h: u32, state: &CircusState, time: f32) -> PixelCanvas {
    if w == 0 || h == 0 {
        return PixelCanvas::new(1, 1);
    }

    let wf = w as f32;
    let hf = h as f32;

    let mut canvas = PixelCanvas::new(w, h).background(Color::from_rgba8(10, 10, 15, 255));

    let v = |id: &str| state.sequence.get(id).unwrap_or(0.0);

    let ball0_enter = v("ball0_enter");
    let ball1_enter = v("ball1_enter");
    let ball2_enter = v("ball2_enter");
    let bounce_x = v("bounce_x");
    let bounce_y = v("bounce_y");
    let squash = v("squash");
    let wow_pop = v("wow_pop");

    // Floor line
    let floor_y = hf * 0.78;
    canvas = draw_floor_reflection(canvas, wf, hf, floor_y);

    // Floor line
    canvas = canvas
        .line(0.0, floor_y, wf, floor_y)
        .color(Color::from_rgba(0.3, 0.3, 0.4, 0.6))
        .width(1.0)
        .done();

    // Decorative border
    canvas = draw_decorative_border(canvas, wf, hf);

    // Spin accumulates over time
    let spin = time * 3.0;

    // Ball trajectory: bouncing parabola
    let ball_x = wf * 0.15 + bounce_x * wf * 0.7;
    let bounce_height = hf * 0.5;
    let ball_y = floor_y - BALL_RADIUS - bounce_y * bounce_height;

    // Motion trail
    let mut trail_positions = Vec::with_capacity(TRAIL_COUNT);
    for i in 0..TRAIL_COUNT {
        let trail_t = (time - i as f32 * 0.03).max(0.0);
        let trail_frac = (trail_t % CYCLE_DURATION) / CYCLE_DURATION;
        let trail_x = wf * 0.15 + trail_frac * wf * 0.7;
        let trail_bounce = (trail_t * 2.5).sin().abs();
        let trail_y = floor_y - BALL_RADIUS - trail_bounce * bounce_height;
        trail_positions.push((trail_x, trail_y));
    }
    canvas = draw_motion_trail(canvas, &trail_positions, BALL_RADIUS, wf, hf);

    // Main ball
    canvas = draw_ball(
        canvas,
        ball_x,
        ball_y,
        BALL_RADIUS,
        spin,
        squash,
        ball0_enter,
        wf,
        hf,
    );

    // Secondary balls (smaller, staggered entry — SHOWCASE: Stagger)
    let ball1_x = wf * 0.25 + (time * 1.5).sin() * wf * 0.1;
    let ball1_y = floor_y - 30.0 - ball1_enter * 40.0 * (time * 3.0).sin().abs();
    canvas = draw_ball(
        canvas,
        ball1_x,
        ball1_y,
        BALL_RADIUS * 0.5,
        spin * 1.3,
        0.0,
        ball1_enter * 0.8,
        wf,
        hf,
    );

    let ball2_x = wf * 0.75 + (time * 1.8).cos() * wf * 0.08;
    let ball2_y = floor_y - 25.0 - ball2_enter * 35.0 * (time * 2.5 + 1.0).sin().abs();
    canvas = draw_ball(
        canvas,
        ball2_x,
        ball2_y,
        BALL_RADIUS * 0.4,
        -spin * 0.8,
        0.0,
        ball2_enter * 0.7,
        wf,
        hf,
    );

    // "WOW!" text on bounce (SHOWCASE: text rendering + alignment)
    if wow_pop > 0.01 {
        let text_size = (wf * 0.06).clamp(20.0, 64.0) * wow_pop;
        let text_y = ball_y - BALL_RADIUS - 20.0;

        // Glow behind text (SHOWCASE: blend mode Screen)
        canvas = canvas
            .group(Transform::IDENTITY)
            .blend_mode(BlendMode::Screen)
            .opacity(wow_pop * 0.6)
            .canvas(|inner| {
                inner
                    .circle(ball_x, text_y - text_size * 0.3, text_size * 1.2)
                    .fill(Color::from_rgba(1.0, 1.0, 0.3, 0.2))
                    .done()
            })
            .done();

        canvas = canvas
            .text("WOW!", ball_x, text_y)
            .size(text_size)
            .color(Color::from_rgba(1.0, 1.0, 0.0, wow_pop))
            .align(TextAlign::Center)
            .done();
    }

    // Decorative stars in corners
    let corner_star_r = 15.0;
    let star_spin = time * 1.5;
    for &(sx, sy) in &[
        (30.0, 30.0),
        (wf - 30.0, 30.0),
        (30.0, hf - 30.0),
        (wf - 30.0, hf - 30.0),
    ] {
        canvas = canvas
            .star(sx, sy, corner_star_r, corner_star_r * 0.4, 5)
            .fill(Color::from_rgba(1.0, 0.84, 0.0, 0.4))
            .transform(Transform::rotate_at(star_spin, sx, sy))
            .done();
    }

    canvas
}

// ═══════════════════════════════════════════════════════════════════
// Window mode
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "window")]
fn run_window() -> Result<(), Box<dyn std::error::Error>> {
    use scry_engine::rasterize::Rasterizer;
    use scry_engine::transport::window::{run_loop_continuous, LoopAction};
    use winit::keyboard::KeyCode as WKey;

    let mut state = CircusState::new();
    let start = Instant::now();
    let mut frozen_time = 0.0_f32;
    let mut last_time = 0.0_f32;

    run_loop_continuous(
        960,
        640,
        "Circus Ball",
        true,
        move |backend, keys, (w, h)| {
            for key in keys {
                if !key.pressed {
                    continue;
                }
                match key.code {
                    WKey::Escape | WKey::KeyQ => return LoopAction::Exit,
                    WKey::Space => state.paused = !state.paused,
                    _ => {}
                }
            }

            let elapsed = if state.paused {
                frozen_time
            } else {
                let e = start.elapsed().as_secs_f32();
                frozen_time = e;
                e
            };

            let dt = elapsed - last_time;
            last_time = elapsed;
            if dt > 0.0 {
                state.sequence.advance(Duration::from_secs_f32(dt));
            }

            let canvas = build_scene(w, h, &state, elapsed);
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

    let mut state = CircusState::new();
    let start = Instant::now();
    let mut last_frame = Instant::now();
    let mut frozen_time = 0.0_f32;

    loop {
        let now = Instant::now();
        let dt = now.duration_since(last_frame);
        last_frame = now;

        let elapsed = if state.paused {
            frozen_time
        } else {
            let e = now.duration_since(start).as_secs_f32();
            frozen_time = e;
            e
        };

        if !state.paused {
            state.sequence.advance(dt);
        }

        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(3)])
                .split(frame.area());

            let area = chunks[0];
            let font = px_state.font_size();
            let w = u32::from(area.width) * u32::from(font.width);
            let h = u32::from(area.height) * u32::from(font.height);
            let canvas = build_scene(w, h, &state, elapsed);

            frame.render_stateful_widget(
                PixelCanvasWidget::new(canvas).skip_cache().z_index(-1),
                area,
                &mut px_state,
            );

            let status_text = format!(
                " Circus Ball | {:.1}s | [space] pause [q] quit",
                elapsed % CYCLE_DURATION,
            );
            let status = Paragraph::new(status_text).block(Block::default().borders(Borders::TOP));
            frame.render_widget(status, chunks[1]);
        })?;
        px_state.flush()?;

        if event::poll(Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Char(' ') => state.paused = !state.paused,
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
