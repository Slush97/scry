//! **Fractal Dreams** — animated Mandelbrot / Julia set explorer.
//!
//! Properly optimized fractal renderer with smooth (continuous) coloring,
//! cardioid/period-2 bulb skipping, high escape radius, and multi-band
//! palette cycling. Renders per-pixel via `ImageData` blitting.
//!
//! Controls:
//!   `m` / `j` — switch between Mandelbrot and Julia modes
//!   `1`–`4`  — palette presets (Electric Sheep, Neon Plasma, Ocean Abyss, Solar Flare)
//!   `+` / `-`  — zoom speed
//!   `Space`  — pause/resume animation
//!   `n`      — next zoom target
//!   `q`      — quit
//!
//! Run with: `cargo run --example fractal_dreams --release`

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::suboptimal_flops,
    clippy::similar_names
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

use scry_engine::prelude::{Picker, PixelCanvasState, PixelCanvasWidget, ProtocolKind};
use scry_engine::scene::command::ImageData;
use scry_engine::scene::PixelCanvas;
use scry_engine::style::Color as C;
use scry_engine::transport;

// ═══════════════════════════════════════════════════════════════════
// Fractal mode
// ═══════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, PartialEq, Eq)]
enum FractalMode {
    Mandelbrot,
    Julia,
}

impl FractalMode {
    const fn label(self) -> &'static str {
        match self {
            Self::Mandelbrot => "Mandelbrot",
            Self::Julia => "Julia",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Palette — multi-band with smooth interpolation
// ═══════════════════════════════════════════════════════════════════

/// A control point in a color palette.
#[derive(Clone, Copy)]
struct ColorStop {
    r: f32,
    g: f32,
    b: f32,
}

impl ColorStop {
    const fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
        }
    }
}

struct Palette {
    name: &'static str,
    stops: &'static [ColorStop],
}

/// Electric Sheep — neon purples, electric blues, molten golds
const PALETTE_ELECTRIC_SHEEP: [ColorStop; 6] = [
    ColorStop::new(0.0, 0.0, 0.0), // Black
    ColorStop::new(0.1, 0.0, 0.3), // Deep violet
    ColorStop::new(0.0, 0.4, 0.8), // Electric blue
    ColorStop::new(0.9, 0.7, 0.1), // Molten gold
    ColorStop::new(1.0, 0.3, 0.6), // Hot pink
    ColorStop::new(0.0, 0.0, 0.0), // Back to black (loops)
];

/// Neon Plasma — hot pinks, cyans, electric greens
const PALETTE_NEON_PLASMA: [ColorStop; 6] = [
    ColorStop::new(0.0, 0.0, 0.05),
    ColorStop::new(0.8, 0.0, 0.5), // Hot magenta
    ColorStop::new(0.0, 1.0, 0.8), // Cyan
    ColorStop::new(0.2, 1.0, 0.0), // Electric green
    ColorStop::new(1.0, 0.8, 0.0), // Yellow
    ColorStop::new(0.0, 0.0, 0.05),
];

/// Ocean Abyss — deep indigos, bioluminescent teals
const PALETTE_OCEAN_ABYSS: [ColorStop; 6] = [
    ColorStop::new(0.0, 0.01, 0.03),
    ColorStop::new(0.05, 0.05, 0.3), // Deep indigo
    ColorStop::new(0.0, 0.5, 0.6),   // Bioluminescent teal
    ColorStop::new(0.0, 0.8, 0.5),   // Bright teal
    ColorStop::new(0.3, 0.15, 0.5),  // Purple
    ColorStop::new(0.0, 0.01, 0.03),
];

/// Solar Flare — reds, oranges, whites, deep blacks
const PALETTE_SOLAR_FLARE: [ColorStop; 6] = [
    ColorStop::new(0.0, 0.0, 0.0),
    ColorStop::new(0.5, 0.0, 0.0), // Deep red
    ColorStop::new(1.0, 0.4, 0.0), // Orange
    ColorStop::new(1.0, 1.0, 0.7), // White-yellow
    ColorStop::new(0.8, 0.2, 0.0), // Red-orange
    ColorStop::new(0.0, 0.0, 0.0),
];

const PALETTES: [Palette; 4] = [
    Palette {
        name: "Electric Sheep",
        stops: &PALETTE_ELECTRIC_SHEEP,
    },
    Palette {
        name: "Neon Plasma",
        stops: &PALETTE_NEON_PLASMA,
    },
    Palette {
        name: "Ocean Abyss",
        stops: &PALETTE_OCEAN_ABYSS,
    },
    Palette {
        name: "Solar Flare",
        stops: &PALETTE_SOLAR_FLARE,
    },
];

/// Map a smooth iteration count to an RGBA color using multi-band palette.
fn palette_color(smooth_iter: f64, palette: &Palette, time_offset: f32) -> (u8, u8, u8, u8) {
    let stops = palette.stops;
    let n = stops.len() as f64;

    // Animated cycling: the palette scrolls over time
    let t = (smooth_iter * 0.05 + f64::from(time_offset) * 0.3) % n;
    let t = if t < 0.0 { t + n } else { t };

    let idx = t.floor() as usize % stops.len();
    let frac = (t - t.floor()) as f32;
    let next = (idx + 1) % stops.len();

    let c = stops[idx].lerp(stops[next], frac);

    let r = (c.r * 255.0).clamp(0.0, 255.0) as u8;
    let g = (c.g * 255.0).clamp(0.0, 255.0) as u8;
    let b = (c.b * 255.0).clamp(0.0, 255.0) as u8;
    (r, g, b, 255)
}

// ═══════════════════════════════════════════════════════════════════
// Fractal computation — properly optimized
// ═══════════════════════════════════════════════════════════════════

/// Escape radius squared. Using 256 (bailout = 16) instead of 4 (bailout = 2)
/// so the smooth coloring formula log2(log2(|z|)) works without artifacts.
const BAILOUT_SQUARED: f64 = 256.0;
const LOG2: f64 = std::f64::consts::LN_2;

/// Interesting zoom targets in the Mandelbrot set boundary.
const ZOOM_TARGETS: [(f64, f64); 5] = [
    // Elephant Valley
    (0.281_502, -0.010_005),
    // Seahorse Valley
    (-0.746_300, 0.110_200),
    // Double spiral
    (-0.159_254_75, 1.033_689),
    // Mini Mandelbrot in antenna
    (-1.749_757, 0.000_001),
    // Spiral near main cardioid
    (-0.101_109_6, 0.956_287),
];

struct FractalState {
    mode: FractalMode,
    palette_idx: usize,
    zoom: f64,
    center_x: f64,
    center_y: f64,
    target_x: f64,
    target_y: f64,
    target_idx: usize,
    julia_cr: f64,
    julia_ci: f64,
    zoom_speed: f64,
    paused: bool,
    max_iter: u32,
}

impl FractalState {
    const fn new() -> Self {
        let (tx, ty) = ZOOM_TARGETS[0];
        Self {
            mode: FractalMode::Mandelbrot,
            palette_idx: 0,
            zoom: 1.0,
            center_x: -0.5,
            center_y: 0.0,
            target_x: tx,
            target_y: ty,
            target_idx: 0,
            julia_cr: -0.7,
            julia_ci: 0.27015,
            zoom_speed: 1.0,
            paused: false,
            max_iter: 256,
        }
    }

    fn update(&mut self, dt: f64, elapsed: f64) {
        if self.paused {
            return;
        }

        self.zoom *= 1.0 + 0.3 * self.zoom_speed * dt;

        // Iteration count scales with log(zoom) for detail at deep zooms
        self.max_iter = (256.0 + self.zoom.ln().max(0.0) * 50.0).min(2000.0) as u32;

        // Drift center toward target
        let drift = (2.0 * dt / self.zoom.max(1.0)).min(0.1);
        self.center_x += (self.target_x - self.center_x) * drift;
        self.center_y += (self.target_y - self.center_y) * drift;

        // Reset zoom when too deep
        if self.zoom > 1e10 {
            self.next_target();
        }

        // Orbit Julia c-parameter for morphing
        self.julia_cr = -0.7 + 0.15 * (elapsed * 0.3).cos();
        self.julia_ci = 0.27015 + 0.1 * (elapsed * 0.23).sin();
    }

    fn next_target(&mut self) {
        self.zoom = 1.0;
        self.target_idx = (self.target_idx + 1) % ZOOM_TARGETS.len();
        let (tx, ty) = ZOOM_TARGETS[self.target_idx];
        self.target_x = tx;
        self.target_y = ty;
        self.center_x = tx + 0.5;
        self.center_y = ty + 0.3;
    }
}

// ───────────────────────────────────────────────────────────────────
// Cardioid & period-2 bulb test (skips points known to be inside)
// ───────────────────────────────────────────────────────────────────

/// Returns true if (cr, ci) is inside the main cardioid or the period-2 bulb.
/// These points always iterate to max_iter, so skipping them is a huge win.
#[inline]
fn in_cardioid_or_bulb(cr: f64, ci: f64) -> bool {
    // Main cardioid check
    let ci2 = ci * ci;
    let q = (cr - 0.25) * (cr - 0.25) + ci2;
    if q * (q + (cr - 0.25)) <= 0.25 * ci2 {
        return true;
    }
    // Period-2 bulb check
    if (cr + 1.0) * (cr + 1.0) + ci2 <= 0.0625 {
        return true;
    }
    false
}

// ───────────────────────────────────────────────────────────────────
// Smooth iteration with escape radius 256
// ───────────────────────────────────────────────────────────────────

/// Compute Mandelbrot with smooth (continuous) iteration count.
/// Returns `None` if point is inside the set, or `Some(smooth_iter)`.
#[inline]
fn mandelbrot_smooth(cr: f64, ci: f64, max_iter: u32) -> Option<f64> {
    // Skip known interior points
    if in_cardioid_or_bulb(cr, ci) {
        return None;
    }

    let mut zr = 0.0_f64;
    let mut zi = 0.0_f64;
    let mut zr2 = 0.0_f64;
    let mut zi2 = 0.0_f64;
    let mut i = 0u32;

    while i < max_iter {
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
        zr2 = zr * zr;
        zi2 = zi * zi;
        if zr2 + zi2 > BAILOUT_SQUARED {
            // Smooth coloring: subtract the fractional part of the escape
            // Formula: i + 1 - log2(log2(|z|)) / log2(2)
            let modulus = (zr2 + zi2).sqrt();
            let smooth = f64::from(i) + 1.0 - modulus.ln().ln() / LOG2;
            return Some(smooth);
        }
        i += 1;
    }
    None // Inside the set
}

/// Compute Julia set with smooth iteration count.
#[inline]
fn julia_smooth(mut zr: f64, mut zi: f64, cr: f64, ci: f64, max_iter: u32) -> Option<f64> {
    let mut zr2 = zr * zr;
    let mut zi2 = zi * zi;
    let mut i = 0u32;

    while i < max_iter {
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
        zr2 = zr * zr;
        zi2 = zi * zi;
        if zr2 + zi2 > BAILOUT_SQUARED {
            let modulus = (zr2 + zi2).sqrt();
            let smooth = f64::from(i) + 1.0 - modulus.ln().ln() / LOG2;
            return Some(smooth);
        }
        i += 1;
    }
    None
}

// ═══════════════════════════════════════════════════════════════════
// Render to pixel buffer
// ═══════════════════════════════════════════════════════════════════

fn render_fractal(width: u32, height: u32, state: &FractalState, time: f32) -> Vec<u8> {
    let palette = &PALETTES[state.palette_idx];
    let scale = 3.0 / state.zoom;
    let aspect = f64::from(width) / f64::from(height);

    let mut buf = vec![0u8; (width as usize) * (height as usize) * 4];

    for py in 0..height {
        for px in 0..width {
            let fx = state.center_x + (f64::from(px) / f64::from(width) - 0.5) * scale * aspect;
            let fy = state.center_y + (f64::from(py) / f64::from(height) - 0.5) * scale;

            let smooth = match state.mode {
                FractalMode::Mandelbrot => mandelbrot_smooth(fx, fy, state.max_iter),
                FractalMode::Julia => {
                    julia_smooth(fx, fy, state.julia_cr, state.julia_ci, state.max_iter)
                }
            };

            let (r, g, b, a) = match smooth {
                Some(s) => palette_color(s, palette, time),
                None => (0, 0, 0, 255), // Inside the set — black void
            };

            let idx = ((py as usize) * (width as usize) + (px as usize)) * 4;
            buf[idx] = r;
            buf[idx + 1] = g;
            buf[idx + 2] = b;
            buf[idx + 3] = a;
        }
    }

    buf
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

    let mut fractal = FractalState::new();
    let start = Instant::now();
    let mut last_frame = Instant::now();
    let mut fps = 0.0_f32;

    loop {
        let now = Instant::now();
        let dt = now.duration_since(last_frame).as_secs_f64();
        let elapsed = now.duration_since(start).as_secs_f64();
        last_frame = now;

        let instant_fps = if dt > 0.0 { 1.0 / dt } else { 0.0 };
        fps = fps * 0.9 + instant_fps as f32 * 0.1;

        fractal.update(dt, elapsed);

        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(3)])
                .split(frame.area());

            let area = chunks[0];
            let canvas = build_fractal_scene(area, &px_state, &fractal, elapsed as f32);

            frame.render_stateful_widget(
                PixelCanvasWidget::new(canvas).skip_cache().z_index(-1),
                area,
                &mut px_state,
            );

            let status_text = format!(
                " {} │ {} │ Zoom: {:.2e} │ Iter: {} │ {:.0} fps │ [m/j] mode [1-4] palette [+/-] speed [n] next [space] pause [q] quit",
                fractal.mode.label(),
                PALETTES[fractal.palette_idx].name,
                fractal.zoom,
                fractal.max_iter,
                fps,
            );
            let status = Paragraph::new(status_text)
                .block(Block::default().borders(Borders::TOP));
            frame.render_widget(status, chunks[1]);
        })?;
        px_state.flush()?;

        if event::poll(Duration::from_millis(1))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Char('m') => fractal.mode = FractalMode::Mandelbrot,
                        KeyCode::Char('j') => fractal.mode = FractalMode::Julia,
                        KeyCode::Char('1') => fractal.palette_idx = 0,
                        KeyCode::Char('2') => fractal.palette_idx = 1,
                        KeyCode::Char('3') => fractal.palette_idx = 2,
                        KeyCode::Char('4') => fractal.palette_idx = 3,
                        KeyCode::Char('+' | '=') => {
                            fractal.zoom_speed = (fractal.zoom_speed * 1.5).min(10.0);
                        }
                        KeyCode::Char('-') => {
                            fractal.zoom_speed = (fractal.zoom_speed / 1.5).max(0.1);
                        }
                        KeyCode::Char(' ') => fractal.paused = !fractal.paused,
                        KeyCode::Char('n') => fractal.next_target(),
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

fn build_fractal_scene(
    area: Rect,
    px_state: &PixelCanvasState,
    fractal: &FractalState,
    time: f32,
) -> PixelCanvas {
    let font = px_state.font_size();
    let w = u32::from(area.width) * u32::from(font.width);
    let h = u32::from(area.height) * u32::from(font.height);

    if w == 0 || h == 0 {
        return PixelCanvas::new(1, 1);
    }

    // Render at half resolution for performance
    let render_w = (w / 2).max(1);
    let render_h = (h / 2).max(1);

    let pixels = render_fractal(render_w, render_h, fractal, time);
    let image = ImageData::new(render_w, render_h, pixels);

    PixelCanvas::new(render_w, render_h)
        .background(C::BLACK)
        .image(image, 0.0, 0.0)
        .done()
}
