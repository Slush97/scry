//! Optical illusions gallery — demonstrates the power of the graphics primitives.
//!
//! Renders 6 mesmerising visual illusion patterns using arcs, transforms,
//! groups, clipping, blend modes, and gradients.
//!
//! **Patterns:** Moiré interference, Café Wall, concentric hypnosis rings,
//! rotating spiral, Fraser spiral, and overlapping transparent circles.
//!
//! Run with: `cargo run --example illusions --features widget`

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
use scry_engine::scene::style::{BlendMode, Color as C, Rect as PxRect, Transform};
use scry_engine::scene::PixelCanvas;
use scry_engine::transport;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

            let canvas = build_illusions(chunks[0], &state, t);
            frame.render_stateful_widget(
                PixelCanvasWidget::new(canvas).z_index(-1),
                chunks[0],
                &mut state,
            );

            let status = Paragraph::new(" ✦ Optical Illusions Gallery  |  'q' quit")
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
// Scene builder
// ═════════════════════════════════════════════════════════════════════════════

fn build_illusions(area: Rect, state: &PixelCanvasState, t: f32) -> PixelCanvas {
    let font = state.font_size();
    let w = u32::from(area.width) * u32::from(font.width);
    let h = u32::from(area.height) * u32::from(font.height);
    let wf = w as f32;
    let hf = h as f32;

    let mut canvas = PixelCanvas::new(w, h).background(C::from_rgba8(10, 10, 18, 255));

    // ── (0,0) Moiré Interference ─────────────────────────────────────────
    // Two sets of concentric circles slightly offset — creates shimmering
    // interference patterns.
    {
        let c = cell(0, 0, wf, hf, 3, 2);
        let r_max = c.w.min(c.h) * 0.48;
        let offset = (t * 0.3).sin() * 8.0;

        // First set of rings
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
        // Second set, offset
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
        // Central dot
        canvas = canvas.circle(c.cx, c.cy, 4.0).fill(C::WHITE).done();
    }

    // ── (2,0) Overlapping Translucent Circles (RGB blend) ────────────────
    // Three semi-transparent circles in R, G, B — demonstrates group opacity
    // and blend modes showing additive color mixing.
    {
        let c = cell(2, 0, wf, hf, 3, 2);
        let r = c.w.min(c.h) * 0.25;
        let spread = r * 0.45;

        // Use Screen blend mode for additive-like mixing
        canvas = canvas
            .group(Transform::identity())
            .blend_mode(BlendMode::Screen)
            .canvas(|inner| {
                inner
                    // Red circle (top)
                    .circle(c.cx, c.cy - spread, r)
                    .fill(C::from_rgba8(255, 40, 40, 200))
                    .done()
                    // Green circle (bottom-left)
                    .circle(c.cx - spread * 0.87, c.cy + spread * 0.5, r)
                    .fill(C::from_rgba8(40, 255, 40, 200))
                    .done()
                    // Blue circle (bottom-right)
                    .circle(c.cx + spread * 0.87, c.cy + spread * 0.5, r)
                    .fill(C::from_rgba8(40, 40, 255, 200))
                    .done()
            })
            .done();
    }

    // ── (0,1) Café Wall Illusion ──────────────────────────────────────────
    // Offset rows of black and white tiles with grey mortar lines between.
    // The parallel lines appear to tilt!
    {
        let c = cell(0, 1, wf, hf, 3, 2);
        let tile_h = (c.h / 10.0).max(6.0);
        let tile_w = tile_h * 1.8;
        let mortar = 2.0;
        let num_rows = ((c.h - mortar) / (tile_h + mortar)) as usize;
        let tiles_per_row = ((c.w + tile_w) / tile_w) as usize + 1;

        // Clip to cell bounds
        canvas = canvas
            .group(Transform::identity())
            .clip_rect(PxRect::new(c.x + 2.0, c.y + 2.0, c.w - 4.0, c.h - 4.0))
            .canvas(|mut inner| {
                // Grey background (mortar)
                inner = inner
                    .rect(c.x, c.y, c.w, c.h)
                    .fill(C::from_rgba8(128, 128, 128, 255))
                    .done();

                for row in 0..num_rows {
                    let y = c.y + mortar + row as f32 * (tile_h + mortar);
                    // Each row is offset by a different amount
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
    // Multiple arcs at different angles with varying colors — creates a
    // spinning mandala pattern using our new Arc primitive.
    {
        let c = cell(1, 1, wf, hf, 3, 2);
        let r_max = c.w.min(c.h) * 0.42;
        let num_layers = 5;
        let arcs_per_layer = 8;
        let pi = std::f32::consts::PI;

        for layer in 0..num_layers {
            let r = r_max * (layer as f32 + 1.0) / num_layers as f32;
            let rotation_offset = t * (1.0 + layer as f32 * 0.3);
            let hue_base = layer as f32 * 50.0 + t * 20.0;

            for arc_i in 0..arcs_per_layer {
                let start = (arc_i as f32 / arcs_per_layer as f32) * pi * 2.0 + rotation_offset;
                let sweep = pi / (arcs_per_layer as f32) * 1.5;

                // HSL-like color cycling
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
    // Three interlocking beams that form an impossible triangle.
    {
        let c = cell(2, 1, wf, hf, 3, 2);
        let size = c.w.min(c.h) * 0.38;
        let thickness = size * 0.22;

        // Triangle vertices (pointing up)
        let top = (c.cx, c.cy - size * 0.65);
        let bl = (c.cx - size * 0.65, c.cy + size * 0.45);
        let br = (c.cx + size * 0.65, c.cy + size * 0.45);

        // Three beams, each a parallelogram creating the impossible overlap
        let beam_colors = [
            C::from_rgba8(70, 160, 230, 255),
            C::from_rgba8(230, 90, 70, 255),
            C::from_rgba8(80, 200, 120, 255),
        ];

        // Beam 1: top to bottom-left (outer edge)
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

        // Beam 2: bottom-left to bottom-right
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

        // Beam 3: bottom-right to top
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

// ─── Utility: HSL to RGB ──────────────────────────────────────────────────────
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
