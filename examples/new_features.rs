//! New features showcase — demonstrates every feature added in the
//! graphics-primitives expansion.
//!
//! **Features covered:** Arc primitive, Group clipping (rect & path),
//! Group opacity, `BlendMode` (Multiply, Screen, Overlay), Transform
//! helpers (rotate, `scale_xy`, skew, concat).
//!
//! The screen is divided into a 3×2 grid, each cell highlighting one
//! new capability.
//!
//! Run with: `cargo run --example new_features --features widget`

#![allow(
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

use std::io::stdout;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use scry_engine::prelude::{Picker, PixelCanvasState, PixelCanvasWidget, ProtocolKind};
use scry_engine::scene::style::{BlendMode, Color as C, Point, Rect as PxRect, Transform};
use scry_engine::scene::PixelCanvas;
use scry_engine::transport;

#[cfg(feature = "window")]
fn run_window() -> Result<(), Box<dyn std::error::Error>> {
    use scry_engine::rasterize::Rasterizer;
    use scry_engine::transport::window::{run_loop_continuous, LoopAction};
    use winit::keyboard::KeyCode as WKey;

    let start = std::time::Instant::now();

    run_loop_continuous(
        960,
        640,
        "New Features Showcase",
        true,
        move |backend, keys, (w, h)| {
            for key in keys {
                if !key.pressed {
                    continue;
                }
                match key.code {
                    WKey::Escape | WKey::KeyQ => return LoopAction::Exit,
                    _ => {}
                }
            }

            let t = start.elapsed().as_secs_f32();
            let canvas = build_scene(w, h, t);
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
    let mut state = PixelCanvasState::new(backend, picker.font_size());

    let start = std::time::Instant::now();

    loop {
        let t = start.elapsed().as_secs_f32();

        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(3)])
                .split(frame.area());

            let font = state.font_size();
            let w = u32::from(chunks[0].width) * u32::from(font.width);
            let h = u32::from(chunks[0].height) * u32::from(font.height);
            let canvas = build_scene(w, h, t);
            frame.render_stateful_widget(
                PixelCanvasWidget::new(canvas).z_index(-1),
                chunks[0],
                &mut state,
            );

            let status = Paragraph::new(" ★ New Features Showcase  |  'q' quit")
                .block(Block::default().borders(Borders::TOP));
            frame.render_widget(status, chunks[1]);
        })?;
        state.flush()?;

        if event::poll(std::time::Duration::from_millis(33))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press && key.code == KeyCode::Char('q') {
                    break;
                }
            }
        }
    }

    state.cleanup();
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
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

fn cell(col: usize, row: usize, total_w: f32, total_h: f32) -> Cell {
    let w = total_w / 3.0;
    let h = total_h / 2.0;
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

fn build_scene(w: u32, h: u32, t: f32) -> PixelCanvas {
    let wf = w as f32;
    let hf = h as f32;
    let pad = 10.0;

    let mut canvas = PixelCanvas::new(w, h).background(C::from_rgba8(12, 12, 22, 255));

    // ── (0,0) Arc Primitives ─────────────────────────────────────────────
    // Multiple arcs forming a loading spinner and decorative rings.
    {
        let c = cell(0, 0, wf, hf);
        let r_outer = c.w.min(c.h) * 0.38 - pad;
        let pi = std::f32::consts::PI;

        // Outer ring of arcs with gaps
        for i in 0..8 {
            let start = (i as f32 / 8.0) * pi * 2.0;
            let sweep = pi * 0.2; // 36° each with gaps
            let hue = i as f32 * 45.0;
            let (r, g, b) = hsl_to_rgb(hue, 0.9, 0.65);
            canvas = canvas
                .arc(c.cx, c.cy, r_outer, start + t, sweep)
                .stroke(C::from_rgba8(r, g, b, 255), 4.0)
                .done();
        }

        // Spinning inner arc
        let r_inner = r_outer * 0.6;
        canvas = canvas
            .arc(c.cx, c.cy, r_inner, -t * 2.0, pi * 1.5)
            .stroke(C::from_rgba8(255, 255, 255, 180), 3.0)
            .done();

        // Pulsing center circle
        let pulse = (t * 3.0).sin() * 0.5 + 0.5;
        let center_r = r_inner * 0.3 + pulse * 5.0;
        canvas = canvas
            .circle(c.cx, c.cy, center_r)
            .fill(C::from_rgba8(255, 200, 100, (150.0 + pulse * 105.0) as u8))
            .done();

        // Title
        draw_label(&mut canvas, c.x + pad, c.y + pad, "ARC PRIMITIVE");
    }

    // ── (1,0) Group Clipping ─────────────────────────────────────────────
    // Shapes rendered inside a circular clip region with rotating content.
    {
        let c = cell(1, 0, wf, hf);
        let r = c.w.min(c.h) * 0.38 - pad;

        // Draw the clip boundary (visible ring)
        canvas = canvas
            .circle(c.cx, c.cy, r)
            .stroke(C::from_rgba8(100, 200, 255, 255), 2.0)
            .done();

        // Clip region: rectangle
        let clip_rect = PxRect::new(c.cx - r, c.cy - r, r * 2.0, r * 2.0);

        canvas = canvas
            .group(Transform::rotate_at(t * 0.5, c.cx, c.cy))
            .clip_rect(clip_rect)
            .canvas(|mut inner| {
                // Colorful stripes that rotate inside the clip
                let stripe_w = r * 0.4;
                let colors = [
                    C::from_rgba8(255, 70, 70, 200),
                    C::from_rgba8(70, 255, 70, 200),
                    C::from_rgba8(70, 70, 255, 200),
                    C::from_rgba8(255, 255, 70, 200),
                    C::from_rgba8(255, 70, 255, 200),
                    C::from_rgba8(70, 255, 255, 200),
                ];

                for (i, &color) in colors.iter().enumerate() {
                    let sx = c.cx - r * 1.5 + i as f32 * stripe_w;
                    inner = inner
                        .rect(sx, c.cy - r * 1.5, stripe_w - 2.0, r * 3.0)
                        .fill(color)
                        .done();
                }
                inner
            })
            .done();

        draw_label(&mut canvas, c.x + pad, c.y + pad, "CLIP RECT");
    }

    // ── (2,0) Group Opacity ──────────────────────────────────────────────
    // Multiple overlapping shapes where the entire group fades in and out.
    {
        let c = cell(2, 0, wf, hf);
        let r = c.w.min(c.h) * 0.28 - pad;

        // Background grid lines to show opacity effect
        let grid_step = 12.0;
        let mut gx = c.x + pad;
        while gx < c.x + c.w - pad {
            canvas = canvas
                .line(gx, c.y + pad, gx, c.y + c.h - pad)
                .stroke(C::from_rgba8(40, 40, 60, 255), 0.5)
                .done();
            gx += grid_step;
        }
        let mut gy = c.y + pad;
        while gy < c.y + c.h - pad {
            canvas = canvas
                .line(c.x + pad, gy, c.x + c.w - pad, gy)
                .stroke(C::from_rgba8(40, 40, 60, 255), 0.5)
                .done();
            gy += grid_step;
        }

        // Group with animated opacity
        let opacity = ((t * 1.5).sin() * 0.5 + 0.5).clamp(0.1, 1.0);

        canvas = canvas
            .group(Transform::identity())
            .opacity(opacity)
            .canvas(|inner| {
                inner
                    .rect(c.cx - r, c.cy - r, r * 1.6, r * 1.6)
                    .fill(C::from_rgba8(255, 100, 50, 255))
                    .corner_radius(8.0)
                    .done()
                    .circle(c.cx + r * 0.3, c.cy + r * 0.3, r * 0.8)
                    .fill(C::from_rgba8(50, 100, 255, 255))
                    .done()
                    .circle(c.cx - r * 0.2, c.cy - r * 0.2, r * 0.5)
                    .fill(C::from_rgba8(50, 255, 100, 255))
                    .done()
            })
            .done();

        draw_label(&mut canvas, c.x + pad, c.y + pad, "GROUP OPACITY");
    }

    // ── (0,1) Blend Modes ────────────────────────────────────────────────
    // Side-by-side comparison: Multiply vs Screen vs Overlay.
    {
        let c = cell(0, 1, wf, hf);
        let third = (c.w - pad * 4.0) / 3.0;
        let rect_h = c.h - pad * 4.0;

        let modes = [
            (BlendMode::Multiply, "MUL"),
            (BlendMode::Screen, "SCR"),
            (BlendMode::Overlay, "OVR"),
        ];

        for (i, (mode, label)) in modes.iter().enumerate() {
            let x = c.x + pad + i as f32 * (third + pad);
            let y_top = c.y + pad * 2.0;

            // Base shape (destination)
            canvas = canvas
                .gradient(x, y_top, third, rect_h)
                .linear(Point::new(x, y_top), Point::new(x, y_top + rect_h))
                .stop(0.0, C::from_rgba8(200, 50, 50, 255))
                .stop(0.5, C::from_rgba8(50, 200, 50, 255))
                .stop(1.0, C::from_rgba8(50, 50, 200, 255))
                .done();

            // Overlapping circle with blend mode
            let cr = third * 0.4;
            canvas = canvas
                .group(Transform::identity())
                .blend_mode(*mode)
                .canvas(|inner| {
                    inner
                        .circle(x + third / 2.0, y_top + rect_h / 2.0, cr)
                        .fill(C::from_rgba8(220, 220, 220, 255))
                        .done()
                })
                .done();

            // Mode label at bottom
            draw_small_label(&mut canvas, x + 2.0, y_top + rect_h - 8.0, label);
        }

        draw_label(&mut canvas, c.x + pad, c.y + pad, "BLEND MODES");
    }

    // ── (1,1) Transform Playground ───────────────────────────────────────
    // Demonstrates rotate, scale_xy, skew, and concat.
    {
        let c = cell(1, 1, wf, hf);
        let half_size = 20.0;

        // Rotating squares that scale
        for i in 0..12 {
            let angle = (i as f32 / 12.0) * std::f32::consts::PI * 2.0 + t;
            let scale = 0.5 + (t + i as f32 * 0.4).sin() * 0.3;
            let dist = c.w.min(c.h) * 0.28;
            let tx = c.cx + dist * angle.cos();
            let ty = c.cy + dist * angle.sin();

            let transform =
                Transform::rotate_at(angle * 2.0, tx, ty).concat(Transform::scale_xy(scale, scale));

            let hue = (i as f32 * 30.0 + t * 40.0) % 360.0;
            let (r, g, b) = hsl_to_rgb(hue, 0.85, 0.6);

            canvas = canvas
                .group(transform)
                .canvas(|inner| {
                    inner
                        .rect(
                            tx - half_size,
                            ty - half_size,
                            half_size * 2.0,
                            half_size * 2.0,
                        )
                        .fill(C::from_rgba8(r, g, b, 180))
                        .corner_radius(4.0)
                        .stroke(C::from_rgba8(255, 255, 255, 100), 1.0)
                        .done()
                })
                .done();
        }

        // Center skewed diamond
        let skew_amount = (t * 2.0).sin() * 0.3;
        let skew_transform = Transform::skew(skew_amount, 0.0);
        canvas = canvas
            .group(skew_transform)
            .canvas(|inner| {
                inner
                    .rect(c.cx - 15.0, c.cy - 15.0, 30.0, 30.0)
                    .fill(C::from_rgba8(255, 255, 255, 200))
                    .done()
            })
            .done();

        draw_label(&mut canvas, c.x + pad, c.y + pad, "TRANSFORMS");
    }

    // ── (2,1) Combined: Clipped spinning arcs with opacity ───────────────
    // Everything together: arcs inside a clipped, fading group.
    {
        let c = cell(2, 1, wf, hf);
        let r = c.w.min(c.h) * 0.4 - pad;
        let pi = std::f32::consts::PI;

        // Outer decorative ring
        for i in 0..16 {
            let start = (i as f32 / 16.0) * pi * 2.0;
            let sweep = pi * 0.08;
            canvas = canvas
                .arc(c.cx, c.cy, r + 6.0, start + t * 0.5, sweep)
                .stroke(C::from_rgba8(80, 80, 120, 200), 2.0)
                .done();
        }

        // Clipped, semi-transparent, blended group
        let clip_rect = PxRect::new(c.cx - r, c.cy - r, r * 2.0, r * 2.0);
        let opacity = ((t * 0.8).sin() * 0.3 + 0.7).clamp(0.4, 1.0);

        canvas = canvas
            .group(Transform::rotate_at(t, c.cx, c.cy))
            .clip_rect(clip_rect)
            .opacity(opacity)
            .blend_mode(BlendMode::Screen)
            .canvas(|mut inner| {
                // Spiraling arcs
                for i in 0..6 {
                    let layer_r = r * (i as f32 + 1.0) / 6.0;
                    let start = (i as f32 * pi / 3.0) + t * (1.5 + i as f32 * 0.2);
                    let sweep = pi;
                    let hue = (i as f32 * 60.0 + t * 30.0) % 360.0;
                    let (cr, cg, cb) = hsl_to_rgb(hue, 0.9, 0.55);

                    inner = inner
                        .arc(c.cx, c.cy, layer_r, start, sweep)
                        .stroke(C::from_rgba8(cr, cg, cb, 230), 3.5)
                        .done();
                }
                inner
            })
            .done();

        draw_label(&mut canvas, c.x + pad, c.y + pad, "ALL COMBINED");
    }

    canvas
}

// ─── Draw small text labels using tiny pixel rectangles ───────────────────────
// We draw labels as simple pixel blocks since text rendering requires
// the `text` feature and a font file.

fn draw_label(canvas: &mut PixelCanvas, x: f32, y: f32, text: &str) {
    let font: &[(&str, &[u8])] = &PIXEL_FONT;
    let mut cursor = x;
    for ch in text.chars() {
        let ch_upper = ch.to_ascii_uppercase();
        if let Some((_, bits)) = font.iter().find(|(c, _)| *c == ch_upper.to_string()) {
            draw_glyph(canvas, cursor, y, bits, C::from_rgba8(180, 180, 200, 200));
            cursor += 6.0;
        } else {
            cursor += 4.0; // space
        }
    }
}

fn draw_small_label(canvas: &mut PixelCanvas, x: f32, y: f32, text: &str) {
    let font: &[(&str, &[u8])] = &PIXEL_FONT;
    let mut cursor = x;
    for ch in text.chars() {
        let ch_upper = ch.to_ascii_uppercase();
        if let Some((_, bits)) = font.iter().find(|(c, _)| *c == ch_upper.to_string()) {
            draw_glyph(canvas, cursor, y, bits, C::from_rgba8(200, 200, 220, 160));
            cursor += 6.0;
        } else {
            cursor += 4.0;
        }
    }
}

fn draw_glyph(canvas: &mut PixelCanvas, x: f32, y: f32, bits: &[u8], color: C) {
    // Each glyph is a 5-wide × 7-tall bitmap packed into bytes
    // Each byte is one row, LSB = leftmost pixel
    for (row, &byte) in bits.iter().enumerate() {
        for col in 0..5 {
            if byte & (1 << (4 - col)) != 0 {
                canvas.push_command(scry_engine::scene::command::DrawCommand::Rectangle {
                    rect: PxRect::new(x + col as f32, y + row as f32, 1.0, 1.0),
                    corner_radius: 0.0,
                    style: scry_engine::scene::style::ShapeStyle {
                        fill: Some(scry_engine::scene::style::FillStyle::Solid(color)),
                        stroke: None,
                        anti_alias: false,
                    },
                });
            }
        }
    }
}

// Tiny 5×7 pixel font for labels (subset)
const PIXEL_FONT: [(&str, &[u8]); 30] = [
    (
        "A",
        &[
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
    ),
    (
        "B",
        &[
            0b11110, 0b10001, 0b11110, 0b10001, 0b10001, 0b10001, 0b11110,
        ],
    ),
    (
        "C",
        &[
            0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
        ],
    ),
    (
        "D",
        &[
            0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100,
        ],
    ),
    (
        "E",
        &[
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
        ],
    ),
    (
        "F",
        &[
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
    ),
    (
        "G",
        &[
            0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110,
        ],
    ),
    (
        "H",
        &[
            0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
    ),
    (
        "I",
        &[
            0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
    ),
    (
        "K",
        &[
            0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
        ],
    ),
    (
        "L",
        &[
            0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
        ],
    ),
    (
        "M",
        &[
            0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
        ],
    ),
    (
        "N",
        &[
            0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
        ],
    ),
    (
        "O",
        &[
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
    ),
    (
        "P",
        &[
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
    ),
    (
        "R",
        &[
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ],
    ),
    (
        "S",
        &[
            0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
    ),
    (
        "T",
        &[
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
    ),
    (
        "U",
        &[
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
    ),
    (
        "V",
        &[
            0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b01010, 0b00100,
        ],
    ),
    (
        "W",
        &[
            0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001,
        ],
    ),
    (
        "X",
        &[
            0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001,
        ],
    ),
    (
        "Y",
        &[
            0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
    ),
    (
        " ",
        &[
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000,
        ],
    ),
    (
        "0",
        &[
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
    ),
    (
        "1",
        &[
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
    ),
    (
        ":",
        &[
            0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000,
        ],
    ),
    (
        ".",
        &[
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b00100,
        ],
    ),
    (
        "J",
        &[
            0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100,
        ],
    ),
    (
        "Q",
        &[
            0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101,
        ],
    ),
];

// ─── HSL to RGB ───────────────────────────────────────────────────────────────
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
