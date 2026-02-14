//! **Hypnotic Tunnels** — infinite recursive polygon tunnel vortex.
//!
//! An infinite tunnel of concentric polygons that rotate, scale, and
//! color-cycle, creating a deep perspective illusion of falling into infinity.
//! Alternating rotation directions create a spiraling vortex effect.
//!
//! Controls:
//!   `1`–`4`  — geometry: Hexagons / Triangles / Squares / Circles
//!   `c`      — cycle color palette (Neon / Vapor / Void / Acid)
//!   `+`/`-`  — tunnel speed
//!   `Space`  — pause/resume
//!   `q`      — quit
//!
//! Run with: `cargo run --example hypnotic_tunnels --release`

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::similar_names
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

use ratatui_pixelcanvas::prelude::{Picker, PixelCanvasState, PixelCanvasWidget, ProtocolKind};
use ratatui_pixelcanvas::scene::style::Color as C;
use ratatui_pixelcanvas::scene::PixelCanvas;
use ratatui_pixelcanvas::transport;

// ═══════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════

const LAYER_COUNT: usize = 50;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Geometry {
    Hexagons,
    Triangles,
    Squares,
    Circles,
}

impl Geometry {
    const fn label(self) -> &'static str {
        match self {
            Self::Hexagons => "Hexagons",
            Self::Triangles => "Triangles",
            Self::Squares => "Squares",
            Self::Circles => "Circles",
        }
    }

    const fn sides(self) -> usize {
        match self {
            Self::Hexagons => 6,
            Self::Triangles => 3,
            Self::Squares => 4,
            Self::Circles => 0, // Special case
        }
    }
}

#[derive(Clone, Copy)]
struct TunnelPalette {
    name: &'static str,
    hue_a: f32,
    hue_b: f32,
    sat: f32,
    light_near: f32,
    light_far: f32,
    bg: (u8, u8, u8),
}

const PALETTES: [TunnelPalette; 4] = [
    // Neon — electric pinks and cyans
    TunnelPalette {
        name: "Neon",
        hue_a: 300.0,
        hue_b: 180.0,
        sat: 1.0,
        light_near: 0.60,
        light_far: 0.05,
        bg: (2, 0, 8),
    },
    // Vapor — pastel pinks, purples, teals
    TunnelPalette {
        name: "Vapor",
        hue_a: 320.0,
        hue_b: 200.0,
        sat: 0.7,
        light_near: 0.65,
        light_far: 0.08,
        bg: (5, 3, 12),
    },
    // Void — monochrome whites fading to black
    TunnelPalette {
        name: "Void",
        hue_a: 0.0,
        hue_b: 0.0,
        sat: 0.0,
        light_near: 0.80,
        light_far: 0.02,
        bg: (0, 0, 0),
    },
    // Acid — greens, yellows, lime
    TunnelPalette {
        name: "Acid",
        hue_a: 120.0,
        hue_b: 60.0,
        sat: 1.0,
        light_near: 0.55,
        light_far: 0.03,
        bg: (1, 3, 0),
    },
];

struct TunnelState {
    geometry: Geometry,
    palette_idx: usize,
    speed: f32,
    paused: bool,
}

impl TunnelState {
    const fn new() -> Self {
        Self {
            geometry: Geometry::Hexagons,
            palette_idx: 0,
            speed: 1.0,
            paused: false,
        }
    }

    const fn palette(&self) -> &TunnelPalette {
        &PALETTES[self.palette_idx]
    }

    const fn next_palette(&mut self) {
        self.palette_idx = (self.palette_idx + 1) % PALETTES.len();
    }
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

fn main() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let picker = Picker::detect();
    let backend: Box<dyn transport::ProtocolBackend> = match picker.protocol() {
        ProtocolKind::Kitty => Box::new(transport::kitty::KittyBackend::new(picker.font_size())),
        _ => Box::new(transport::halfblock::HalfblockBackend::new()),
    };
    let mut px_state = PixelCanvasState::new(backend, picker.font_size());

    let mut tunnel = TunnelState::new();
    let start = Instant::now();
    let mut last_frame = Instant::now();
    let mut frozen_time = 0.0_f32;

    loop {
        let now = Instant::now();
        let _dt = now.duration_since(last_frame).as_secs_f32();
        last_frame = now;

        let elapsed = if tunnel.paused {
            frozen_time
        } else {
            let e = now.duration_since(start).as_secs_f32();
            frozen_time = e;
            e
        };

        let time = elapsed * tunnel.speed;

        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(3)])
                .split(frame.area());

            let area = chunks[0];
            let canvas = build_tunnel_scene(area, &px_state, &tunnel, time);

            frame.render_stateful_widget(
                PixelCanvasWidget::new(canvas).skip_cache().z_index(-1),
                area,
                &mut px_state,
            );

            let status_text = format!(
                " {} │ {} │ Speed: {:.1}x │ [1-4] shape [c] palette [+/-] speed [space] pause [q] quit",
                tunnel.geometry.label(),
                tunnel.palette().name,
                tunnel.speed,
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
                        KeyCode::Char('1') => tunnel.geometry = Geometry::Hexagons,
                        KeyCode::Char('2') => tunnel.geometry = Geometry::Triangles,
                        KeyCode::Char('3') => tunnel.geometry = Geometry::Squares,
                        KeyCode::Char('4') => tunnel.geometry = Geometry::Circles,
                        KeyCode::Char('c') => tunnel.next_palette(),
                        KeyCode::Char('+' | '=') => {
                            tunnel.speed = (tunnel.speed * 1.3).min(5.0);
                        }
                        KeyCode::Char('-') => {
                            tunnel.speed = (tunnel.speed / 1.3).max(0.1);
                        }
                        KeyCode::Char(' ') => tunnel.paused = !tunnel.paused,
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

fn build_tunnel_scene(
    area: Rect,
    px_state: &PixelCanvasState,
    tunnel: &TunnelState,
    time: f32,
) -> PixelCanvas {
    let font = px_state.font_size();
    let w = u32::from(area.width) * u32::from(font.width);
    let h = u32::from(area.height) * u32::from(font.height);
    if w == 0 || h == 0 {
        return PixelCanvas::new(1, 1);
    }

    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let max_radius = (w.max(h) as f32) * 0.7;
    let pal = tunnel.palette();
    let bg = C::from_rgba8(pal.bg.0, pal.bg.1, pal.bg.2, 255);

    let mut canvas = PixelCanvas::new(w, h).background(bg);

    // Draw layers from back (small/far) to front (large/near)
    // Using a looping depth offset so layers appear to stream toward you
    let loop_period = 2.0; // Seconds for one full layer cycle
    let depth_offset = (time / loop_period).fract();

    for i in (0..LAYER_COUNT).rev() {
        let raw_depth = i as f32 / LAYER_COUNT as f32;
        // Add depth_offset so layers continuously stream forward
        let depth = (raw_depth + depth_offset) % 1.0;

        // Perspective: exponential scaling for depth
        let scale = (1.0 - depth).powi(2);
        let radius = max_radius * scale;

        if radius < 2.0 {
            continue; // Too small to see
        }

        // Color: interpolate between hue_a (near) and hue_b (far) with cycling
        let hue_t = depth;
        let hue = time.mul_add(15.0, (pal.hue_b - pal.hue_a).mul_add(hue_t, pal.hue_a));
        let hue = hue % 360.0;
        let light = (pal.light_near - pal.light_far).mul_add(scale, pal.light_far);

        // Alpha fades at edges
        let alpha = scale.clamp(0.0, 1.0) * 0.85;
        let stroke_color = C::from_hsla(hue, pal.sat, light, alpha);

        // Alternating rotation direction per layer
        let rot_sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        let rotation = (time * 0.5).mul_add(rot_sign, i as f32 * 0.08);

        // Subtle fill with very low alpha for depth
        let fill_alpha = alpha * 0.04;
        let fill_color = C::from_hsla((hue + 30.0) % 360.0, pal.sat * 0.8, light * 0.6, fill_alpha);

        // Stroke width decreases with depth
        let stroke_width = 2.0f32.mul_add(scale, 0.5);

        let sides = tunnel.geometry.sides();
        if sides == 0 {
            // Circles
            canvas = canvas
                .circle(cx, cy, radius)
                .fill(fill_color)
                .stroke(stroke_color, stroke_width)
                .done();
        } else {
            // Polygon
            let points: Vec<(f32, f32)> = (0..sides)
                .map(|s| {
                    let angle = s as f32 * TAU / sides as f32 + rotation;
                    (cx + radius * angle.cos(), cy + radius * angle.sin())
                })
                .collect();

            canvas = canvas
                .polygon(points)
                .fill(fill_color)
                .stroke(stroke_color, stroke_width)
                .done();
        }

        // Add cross-lines for extra hypnotic effect on every 3rd layer
        if i % 3 == 0 && radius > 10.0 {
            let cross_count = sides.max(4);
            let cross_alpha = alpha * 0.15;
            let cross_color =
                C::from_hsla((hue + 180.0) % 360.0, pal.sat, light * 0.8, cross_alpha);

            for s in 0..cross_count {
                let angle = s as f32 * TAU / cross_count as f32 + rotation;
                let x2 = (radius * 0.95).mul_add(angle.cos(), cx);
                let y2 = (radius * 0.95).mul_add(angle.sin(), cy);

                canvas = canvas
                    .line(cx, cy, x2, y2)
                    .color(cross_color)
                    .width(0.5 + scale)
                    .done();
            }
        }
    }

    // Central bright point (vanishing point glow)
    let glow_pulse = 0.4f32.mul_add((time * 3.0).sin().abs(), 0.6);
    for ring in (0..4).rev() {
        let gr = (ring as f32).mul_add(12.0, 8.0);
        let ga = glow_pulse * 0.15 / (ring as f32).mul_add(0.5, 1.0);
        let gh = (time * 25.0) % 360.0;
        canvas = canvas
            .circle(cx, cy, gr)
            .fill(C::from_hsla(gh, pal.sat, 0.7, ga))
            .done();
    }

    canvas
}
