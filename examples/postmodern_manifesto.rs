//! Postmodern Manifesto — layered collage built from gradients, transforms,
//! clipping, and blend modes.
//!
//! Controls:
//!   `c`      cycle palette
//!   `Space`  pause/resume
//!   `q`      quit
//!
//! Run with: `cargo run --example postmodern_manifesto --release`

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::similar_names
)]

use std::f32::consts::{FRAC_PI_2, PI, TAU};
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
use scry_engine::scene::style::{
    BlendMode, Color as C, DashPattern, GradientDef, GradientKind, GradientStop, LineCap, LineJoin,
    Point, Rect as PxRect, Transform,
};
use scry_engine::scene::PixelCanvas;
use scry_engine::transport;

#[derive(Clone, Copy)]
struct Palette {
    name: &'static str,
    bg_a: C,
    bg_b: C,
    bg_c: C,
    primary: C,
    secondary: C,
    accent: C,
    ink: C,
}

const PALETTES: [Palette; 3] = [
    Palette {
        name: "Gallery Concrete",
        bg_a: C::from_rgba8(239, 232, 219, 255),
        bg_b: C::from_rgba8(206, 178, 160, 255),
        bg_c: C::from_rgba8(34, 28, 32, 255),
        primary: C::from_rgba8(210, 42, 49, 255),
        secondary: C::from_rgba8(24, 125, 164, 255),
        accent: C::from_rgba8(245, 196, 71, 255),
        ink: C::from_rgba8(18, 16, 16, 255),
    },
    Palette {
        name: "Museum Fluorescent",
        bg_a: C::from_rgba8(17, 21, 35, 255),
        bg_b: C::from_rgba8(63, 40, 92, 255),
        bg_c: C::from_rgba8(6, 8, 13, 255),
        primary: C::from_rgba8(255, 73, 131, 255),
        secondary: C::from_rgba8(30, 241, 220, 255),
        accent: C::from_rgba8(255, 238, 74, 255),
        ink: C::from_rgba8(236, 239, 244, 255),
    },
    Palette {
        name: "Printmaker Acid",
        bg_a: C::from_rgba8(225, 223, 209, 255),
        bg_b: C::from_rgba8(135, 132, 94, 255),
        bg_c: C::from_rgba8(28, 35, 26, 255),
        primary: C::from_rgba8(162, 28, 35, 255),
        secondary: C::from_rgba8(44, 169, 66, 255),
        accent: C::from_rgba8(247, 118, 36, 255),
        ink: C::from_rgba8(23, 24, 20, 255),
    },
];

// ═══════════════════════════════════════════════════════════════════
// Window mode
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "window")]
fn run_window() -> Result<(), Box<dyn std::error::Error>> {
    use scry_engine::rasterize::Rasterizer;
    use scry_engine::transport::window::{run_loop_continuous, LoopAction};
    use winit::keyboard::KeyCode as WKey;

    let start = Instant::now();
    let mut palette_idx = 0usize;
    let mut paused = false;
    let mut frozen_time = 0.0_f32;

    run_loop_continuous(
        960,
        640,
        "Postmodern Manifesto",
        true,
        move |backend, keys, (w, h)| {
            for key in keys {
                if !key.pressed {
                    continue;
                }
                match key.code {
                    WKey::Escape | WKey::KeyQ => return LoopAction::Exit,
                    WKey::KeyC => {
                        palette_idx = (palette_idx + 1) % PALETTES.len();
                    }
                    WKey::Space => paused = !paused,
                    _ => {}
                }
            }

            let elapsed = if paused {
                frozen_time
            } else {
                let t = start.elapsed().as_secs_f32();
                frozen_time = t;
                t
            };
            let palette = PALETTES[palette_idx];

            let canvas = build_postmodern_scene(w, h, elapsed, palette);
            if let Ok(pixmap) = Rasterizer::rasterize(&canvas) {
                let _ = backend.blit(&pixmap);
            }
            LoopAction::Continue
        },
    )?;
    Ok(())
}

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

    let start = Instant::now();
    let mut palette_idx = 0usize;
    let mut paused = false;
    let mut frozen_time = 0.0_f32;
    let mut last_frame = Instant::now();
    let mut fps_smooth = 0.0_f32;

    loop {
        let now = Instant::now();
        let dt = now.duration_since(last_frame).as_secs_f32();
        last_frame = now;
        if dt > 0.0 {
            let fps_inst = 1.0 / dt;
            fps_smooth = if fps_smooth == 0.0 {
                fps_inst
            } else {
                fps_smooth * 0.9 + fps_inst * 0.1
            };
        }

        let elapsed = if paused {
            frozen_time
        } else {
            let t = start.elapsed().as_secs_f32();
            frozen_time = t;
            t
        };
        let palette = PALETTES[palette_idx];

        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(3)])
                .split(frame.area());

            let area = chunks[0];
            let font = px_state.font_size();
            let w = u32::from(area.width) * u32::from(font.width);
            let h = u32::from(area.height) * u32::from(font.height);
            let canvas = build_postmodern_scene(w, h, elapsed, palette);

            frame.render_stateful_widget(
                PixelCanvasWidget::new(canvas).z_index(-1).skip_cache(),
                area,
                &mut px_state,
            );

            let status = Paragraph::new(format!(
                " Postmodern Manifesto | Palette: {} | FPS: {:>5.1} | [c] palette [space] pause [q] quit",
                palette.name, fps_smooth
            ))
            .block(Block::default().borders(Borders::TOP));
            frame.render_widget(status, chunks[1]);
        })?;
        px_state.flush()?;

        if event::poll(Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Char(' ') => paused = !paused,
                        KeyCode::Char('c') => {
                            palette_idx = (palette_idx + 1) % PALETTES.len();
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

fn build_postmodern_scene(w: u32, h: u32, t: f32, palette: Palette) -> PixelCanvas {
    let (w, h, wf, hf) = if w == 0 || h == 0 {
        (1, 1, 1.0, 1.0)
    } else {
        (w, h, w as f32, h as f32)
    };
    let mut canvas = PixelCanvas::new(w, h).background(palette.bg_a);

    let cx = wf * 0.5;
    let cy = hf * 0.5;
    let min_side = wf.min(hf);

    canvas = canvas
        .gradient(0.0, 0.0, wf, hf)
        .linear(Point::new(0.0, 0.0), Point::new(wf, hf * 0.85))
        .stop(0.0, palette.bg_a)
        .stop(0.55, palette.bg_b)
        .stop(1.0, palette.bg_c)
        .done();

    let band_y = hf * (0.22 + 0.03 * (t * 0.4).sin());
    canvas = canvas
        .group(Transform::identity())
        .blend_mode(BlendMode::Overlay)
        .opacity(0.35)
        .clip_rect(PxRect::new(0.0, band_y, wf, hf * 0.35))
        .canvas(|inner| {
            inner
                .gradient(0.0, band_y, wf, hf * 0.35)
                .linear(Point::new(0.0, band_y), Point::new(0.0, hf))
                .stop(0.0, palette.secondary.with_alpha(0.0))
                .stop(0.5, palette.secondary.with_alpha(0.40))
                .stop(1.0, palette.primary.with_alpha(0.0))
                .done()
        })
        .done();

    for i in 0..=14 {
        let x = wf * i as f32 / 14.0;
        let warp = ((i as f32 * 0.7 + t * 0.6).sin()) * 6.0;
        let alpha = if i % 3 == 0 { 0.38 } else { 0.18 };
        let width = if i % 3 == 0 { 2.0 } else { 1.0 };
        let mut line = canvas
            .line(x + warp, 0.0, x - warp * 0.4, hf)
            .stroke(palette.ink.with_alpha(alpha), width)
            .anti_alias(false);
        if i % 2 == 1 {
            line = line.dash(DashPattern::pair(7.0, 11.0));
        }
        canvas = line.done();
    }

    for j in 0..=8 {
        let y = hf * j as f32 / 8.0 + ((j as f32 * 0.8 - t * 0.5).cos()) * 4.0;
        let alpha = if j % 2 == 0 { 0.25 } else { 0.12 };
        let mut line = canvas
            .line(0.0, y, wf, y)
            .stroke(palette.ink.with_alpha(alpha), 1.0)
            .anti_alias(false);
        if j % 2 == 1 {
            line = line.dash(DashPattern::pair(5.0, 13.0));
        }
        canvas = line.done();
    }

    for i in 0..8 {
        let w_strip = min_side * (0.18 + 0.02 * (i as f32 + t).sin().abs());
        let h_strip = min_side * 0.08;
        let x = ((i as f32 * 83.0 + t * 45.0) % (wf + w_strip)) - w_strip * 0.5;
        let y = hf * 0.15 + (i as f32 * 57.0) % (hf * 0.7);
        let angle = (i as f32 * 0.37 + t * 0.22).sin() * 0.7;
        let fill = if i % 2 == 0 {
            palette.primary.with_alpha(0.62)
        } else {
            palette.secondary.with_alpha(0.54)
        };
        let blend = match i % 3 {
            0 => BlendMode::Multiply,
            1 => BlendMode::Screen,
            _ => BlendMode::Overlay,
        };

        canvas = canvas
            .group(Transform::rotate_at(
                angle,
                x + w_strip * 0.5,
                y + h_strip * 0.5,
            ))
            .blend_mode(blend)
            .opacity(0.9)
            .canvas(|inner| {
                inner
                    .rect(x, y, w_strip, h_strip)
                    .fill(fill)
                    .stroke(palette.ink.with_alpha(0.45), 1.0)
                    .corner_radius(3.0)
                    .done()
            })
            .done();
    }

    let orb_gradient = GradientDef {
        kind: GradientKind::Radial {
            center: Point::new(cx + (t * 0.7).sin() * 18.0, cy + (t * 0.9).cos() * 12.0),
            radius: min_side * 0.28,
        },
        stops: vec![
            GradientStop {
                position: 0.0,
                color: palette.accent.with_alpha(0.55),
            },
            GradientStop {
                position: 0.45,
                color: palette.primary.with_alpha(0.30),
            },
            GradientStop {
                position: 1.0,
                color: palette.bg_c.with_alpha(0.0),
            },
        ],
    };
    canvas = canvas
        .circle(cx, cy, min_side * 0.26)
        .fill_radial_gradient(orb_gradient)
        .stroke(palette.ink.with_alpha(0.40), 1.4)
        .done();

    let polygon = regular_polygon(cx, cy, min_side * 0.19, 7, t * 0.32);
    canvas = canvas
        .polygon(polygon)
        .fill(palette.bg_a.with_alpha(0.15))
        .stroke(palette.accent.with_alpha(0.85), 2.2)
        .line_join(LineJoin::Round)
        .done();

    for i in 0..16 {
        let ring = min_side * (0.12 + i as f32 * 0.019);
        let start = t * (0.25 + i as f32 * 0.01) + i as f32 * 0.55;
        let sweep = PI * (0.45 + 0.25 * (t * 0.9 + i as f32).sin().abs());
        let color = palette
            .primary
            .mix(palette.secondary, i as f32 / 16.0)
            .with_alpha(0.35 + 0.02 * (i % 3) as f32);
        canvas = canvas
            .arc(cx, cy, ring, start, sweep)
            .stroke(color, 1.0 + (i % 4) as f32 * 0.4)
            .line_cap(LineCap::Round)
            .done();
    }

    let star_cx = cx + min_side * 0.23 * (t * 0.37).cos();
    let star_cy = cy - min_side * 0.18 * (t * 0.41).sin();
    if let Some(path) = star_clip_path(star_cx, star_cy, min_side * 0.24, min_side * 0.11, 8) {
        let x0 = cx - min_side * 0.42;
        let y0 = cy - min_side * 0.42;
        let x1 = cx + min_side * 0.42;
        let y1 = cy + min_side * 0.42;

        canvas = canvas
            .group(Transform::identity())
            .clip_path(path)
            .blend_mode(BlendMode::Overlay)
            .opacity(0.78)
            .canvas(|mut inner| {
                let step = (min_side * 0.04).max(10.0);
                let mut y = y0;
                while y <= y1 {
                    let mut x = x0;
                    while x <= x1 {
                        let n = ((x * 0.03 + y * 0.06) + t * 1.4).sin() * 0.5 + 0.5;
                        let r = 0.8 + n * 3.5;
                        let color = palette.accent.mix(palette.secondary, n).with_alpha(0.28);
                        inner = inner.circle(x, y, r).fill(color).done();
                        x += step;
                    }
                    y += step;
                }
                inner
            })
            .done();
    }

    let mut wave = Vec::with_capacity(180);
    let base_y = hf * 0.78;
    for i in 0..180 {
        let x = i as f32 / 179.0 * wf;
        let y = base_y
            + (x * 0.025 + t * 2.1).sin() * (min_side * 0.045)
            + (x * 0.011 - t * 1.3).cos() * (min_side * 0.03);
        wave.push((x, y));
    }
    canvas = canvas
        .polyline(wave)
        .stroke(palette.ink.with_alpha(0.88), 2.2)
        .dash(DashPattern::new(vec![12.0, 8.0, 2.0, 8.0], t * 15.0))
        .line_cap(LineCap::Round)
        .line_join(LineJoin::Round)
        .done();

    for i in 0..9 {
        let a = i as f32 / 9.0 * TAU + t * 0.3;
        let r = min_side * (0.28 + 0.02 * (i as f32 * 2.0 + t).sin());
        let x = cx + a.cos() * r;
        let y = cy + a.sin() * r;
        let color = palette
            .secondary
            .mix(palette.accent, (i as f32 / 9.0).fract())
            .with_alpha(0.75);
        canvas = canvas
            .circle(x, y, 4.0 + (t * 1.2 + i as f32).sin().abs() * 2.5)
            .fill(color)
            .done()
            .line(x, y, cx, cy)
            .stroke(palette.ink.with_alpha(0.25), 1.0)
            .dash(DashPattern::pair(3.0, 7.0))
            .done();
    }

    canvas = canvas
        .group(Transform::rotate_at(t * 0.18, cx, cy))
        .blend_mode(BlendMode::Lighten)
        .opacity(0.55)
        .canvas(|inner| {
            inner
                .rect(
                    cx - min_side * 0.34,
                    cy - min_side * 0.02,
                    min_side * 0.68,
                    min_side * 0.04,
                )
                .fill(palette.accent.with_alpha(0.35))
                .done()
                .rect(
                    cx - min_side * 0.02,
                    cy - min_side * 0.34,
                    min_side * 0.04,
                    min_side * 0.68,
                )
                .fill(palette.secondary.with_alpha(0.35))
                .done()
        })
        .done();

    canvas = canvas
        .arc(cx, cy, min_side * 0.31, t * 0.5 - FRAC_PI_2, PI * 1.6)
        .stroke(palette.accent.with_alpha(0.55), 3.0)
        .line_cap(LineCap::Square)
        .done();

    canvas
}

fn regular_polygon(cx: f32, cy: f32, radius: f32, sides: usize, rotation: f32) -> Vec<(f32, f32)> {
    (0..sides)
        .map(|i| {
            let angle = rotation + i as f32 * TAU / sides as f32;
            (cx + angle.cos() * radius, cy + angle.sin() * radius)
        })
        .collect()
}

fn star_clip_path(
    cx: f32,
    cy: f32,
    outer_radius: f32,
    inner_radius: f32,
    spikes: usize,
) -> Option<tiny_skia::Path> {
    let mut pb = tiny_skia::PathBuilder::new();
    for i in 0..(spikes * 2) {
        let radius = if i % 2 == 0 {
            outer_radius
        } else {
            inner_radius
        };
        let angle = -FRAC_PI_2 + i as f32 * TAU / (spikes * 2) as f32;
        let x = cx + angle.cos() * radius;
        let y = cy + angle.sin() * radius;
        if i == 0 {
            pb.move_to(x, y);
        } else {
            pb.line_to(x, y);
        }
    }
    pb.close();
    pb.finish()
}
