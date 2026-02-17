//! **Sacred Geometry** — living geometric construction unfolding in real time.
//!
//! Animated construction of sacred geometric patterns — Flower of Life
//! expanding outward ring by ring, morphing into Metatron's Cube, then
//! a Sri Yantra, in an infinite cycle.
//!
//! Controls:
//!   `1` / `2` / `3` — jump to Flower of Life / Metatron's Cube / Sri Yantra
//!   `Space`         — pause/resume
//!   `c`             — toggle color mode (gold / rainbow / monochrome)
//!   `q`             — quit
//!
//! Run with: `cargo run --example sacred_geometry --release`

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::similar_names
)]

use std::f32::consts::{FRAC_PI_3, TAU};
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
use scry_engine::scene::style::Color as C;
use scry_engine::scene::PixelCanvas;
use scry_engine::transport;

// ═══════════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pattern {
    FlowerOfLife,
    MetatronsCube,
    SriYantra,
}

impl Pattern {
    const ALL: [Self; 3] = [Self::FlowerOfLife, Self::MetatronsCube, Self::SriYantra];

    const fn label(self) -> &'static str {
        match self {
            Self::FlowerOfLife => "Flower of Life",
            Self::MetatronsCube => "Metatron's Cube",
            Self::SriYantra => "Sri Yantra",
        }
    }

    const fn duration(self) -> f32 {
        match self {
            Self::FlowerOfLife => 12.0,
            Self::MetatronsCube => 10.0,
            Self::SriYantra => 10.0,
        }
    }
}

#[derive(Clone, Copy)]
enum ColorMode {
    Gold,
    Rainbow,
    Monochrome,
}

impl ColorMode {
    const fn label(self) -> &'static str {
        match self {
            Self::Gold => "Gold",
            Self::Rainbow => "Rainbow",
            Self::Monochrome => "Mono",
        }
    }

    const fn next(self) -> Self {
        match self {
            Self::Gold => Self::Rainbow,
            Self::Rainbow => Self::Monochrome,
            Self::Monochrome => Self::Gold,
        }
    }
}

struct GeoState {
    pattern_idx: usize,
    color_mode: ColorMode,
    paused: bool,
    phase_start: f32,
}

impl GeoState {
    const fn new() -> Self {
        Self {
            pattern_idx: 0,
            color_mode: ColorMode::Gold,
            paused: false,
            phase_start: 0.0,
        }
    }

    const fn current(&self) -> Pattern {
        Pattern::ALL[self.pattern_idx]
    }

    fn advance(&mut self, elapsed: f32) {
        let pattern = self.current();
        let phase_time = elapsed - self.phase_start;
        if phase_time > pattern.duration() {
            self.pattern_idx = (self.pattern_idx + 1) % Pattern::ALL.len();
            self.phase_start = elapsed;
        }
    }

    const fn jump_to(&mut self, idx: usize, elapsed: f32) {
        self.pattern_idx = idx % Pattern::ALL.len();
        self.phase_start = elapsed;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Color helpers
// ═══════════════════════════════════════════════════════════════════

fn geo_color(mode: ColorMode, depth: f32, time: f32) -> C {
    match mode {
        ColorMode::Gold => {
            let hue = (time * 10.0).sin().mul_add(5.0, depth.mul_add(15.0, 40.0));
            let sat = 0.2f32.mul_add(depth.mul_add(3.0, time).sin().abs(), 0.7);
            let light = 0.15f32.mul_add(depth.mul_add(2.0, time * 0.5).cos(), 0.45);
            C::from_hsla(hue, sat, light, 1.0)
        }
        ColorMode::Rainbow => {
            let hue = depth.mul_add(60.0, time * 20.0) % 360.0;
            C::from_hsla(hue, 0.9, 0.55, 1.0)
        }
        ColorMode::Monochrome => {
            let v = 0.4f32.mul_add(depth.mul_add(2.0, time * 0.5).cos().abs(), 0.3);
            C::from_rgba(v, v, v * 1.1, 1.0)
        }
    }
}

fn glow_color(mode: ColorMode, time: f32) -> C {
    match mode {
        ColorMode::Gold => C::from_hsla((time * 8.0).sin().mul_add(10.0, 45.0), 0.9, 0.7, 0.15),
        ColorMode::Rainbow => C::from_hsla((time * 25.0) % 360.0, 0.8, 0.6, 0.12),
        ColorMode::Monochrome => C::from_rgba(0.8, 0.8, 0.85, 0.10),
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

    let mut geo = GeoState::new();
    let start = Instant::now();
    let mut frozen_time = 0.0_f32;

    run_loop_continuous(
        960,
        640,
        "Sacred Geometry",
        true,
        move |backend, keys, (w, h)| {
            for key in keys {
                if !key.pressed {
                    continue;
                }
                match key.code {
                    WKey::Escape | WKey::KeyQ => return LoopAction::Exit,
                    WKey::Digit1 => geo.jump_to(0, frozen_time),
                    WKey::Digit2 => geo.jump_to(1, frozen_time),
                    WKey::Digit3 => geo.jump_to(2, frozen_time),
                    WKey::KeyC => geo.color_mode = geo.color_mode.next(),
                    WKey::Space => geo.paused = !geo.paused,
                    _ => {}
                }
            }

            let elapsed = if geo.paused {
                frozen_time
            } else {
                let e = start.elapsed().as_secs_f32();
                frozen_time = e;
                e
            };

            if !geo.paused {
                geo.advance(elapsed);
            }

            let canvas = build_scene(w, h, &geo, elapsed);
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

    let mut geo = GeoState::new();
    let start = Instant::now();
    let mut last_frame = Instant::now();
    let mut frozen_time = 0.0_f32;

    loop {
        let now = Instant::now();
        let _dt = now.duration_since(last_frame).as_secs_f32();
        last_frame = now;

        let elapsed = if geo.paused {
            frozen_time
        } else {
            let e = now.duration_since(start).as_secs_f32();
            frozen_time = e;
            e
        };

        if !geo.paused {
            geo.advance(elapsed);
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
            let canvas = build_scene(w, h, &geo, elapsed);

            frame.render_stateful_widget(
                PixelCanvasWidget::new(canvas).skip_cache().z_index(-1),
                area,
                &mut px_state,
            );

            let status_text = format!(
                " {} │ {} │ Phase {:.1}s │ [1-3] pattern [c] color [space] pause [q] quit",
                geo.current().label(),
                geo.color_mode.label(),
                elapsed - geo.phase_start,
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
                        KeyCode::Char('1') => geo.jump_to(0, elapsed),
                        KeyCode::Char('2') => geo.jump_to(1, elapsed),
                        KeyCode::Char('3') => geo.jump_to(2, elapsed),
                        KeyCode::Char('c') => geo.color_mode = geo.color_mode.next(),
                        KeyCode::Char(' ') => geo.paused = !geo.paused,
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
// Scene builder — dispatches to pattern builders
// ═══════════════════════════════════════════════════════════════════

fn build_scene(w: u32, h: u32, geo: &GeoState, time: f32) -> PixelCanvas {
    if w == 0 || h == 0 {
        return PixelCanvas::new(1, 1);
    }

    let phase_t = (time - geo.phase_start) / geo.current().duration();
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let radius = (w.min(h) as f32) * 0.4;

    let mut canvas = PixelCanvas::new(w, h).background(C::from_rgba8(5, 3, 12, 255));

    match geo.current() {
        Pattern::FlowerOfLife => {
            canvas = draw_flower_of_life(canvas, cx, cy, radius, phase_t, time, geo.color_mode);
        }
        Pattern::MetatronsCube => {
            canvas = draw_metatrons_cube(canvas, cx, cy, radius, phase_t, time, geo.color_mode);
        }
        Pattern::SriYantra => {
            canvas = draw_sri_yantra(canvas, cx, cy, radius, phase_t, time, geo.color_mode);
        }
    }

    canvas
}

// ═══════════════════════════════════════════════════════════════════
// Pattern 1: Flower of Life
// ═══════════════════════════════════════════════════════════════════

fn draw_flower_of_life(
    mut canvas: PixelCanvas,
    cx: f32,
    cy: f32,
    radius: f32,
    progress: f32, // 0..1 over the pattern's duration
    time: f32,
    mode: ColorMode,
) -> PixelCanvas {
    let r = radius / 4.0; // Radius of each small circle
    let breath = 0.03f32.mul_add((time * 1.5).sin(), 1.0);

    // Ring 0: center circle (appears first)
    // Ring 1: 6 circles at distance r  (60° apart)
    // Ring 2: 6 circles at distance 2r (30° offset from ring 1 midpoints)
    // Ring 3: 12 circles at distance 2r and √3·r
    // etc.

    // Generate circle centers ring by ring
    let mut centers: Vec<(f32, f32, usize)> = Vec::new(); // (x, y, ring)

    // Ring 0
    centers.push((cx, cy, 0));

    // Ring 1: 6 circles at distance r
    for i in 0..6 {
        let angle = i as f32 * FRAC_PI_3;
        let x = (r * breath).mul_add(angle.cos(), cx);
        let y = (r * breath).mul_add(angle.sin(), cy);
        centers.push((x, y, 1));
    }

    // Ring 2: 6 circles at distance 2r
    for i in 0..6 {
        let angle = i as f32 * FRAC_PI_3;
        let x = (2.0 * r * breath).mul_add(angle.cos(), cx);
        let y = (2.0 * r * breath).mul_add(angle.sin(), cy);
        centers.push((x, y, 2));
    }

    // Ring 2 intermediates: 6 circles between ring 1 positions at distance √3·r
    let sqrt3 = 3.0_f32.sqrt();
    for i in 0..6 {
        let angle = (i as f32).mul_add(FRAC_PI_3, FRAC_PI_3 / 2.0);
        let x = (sqrt3 * r * breath).mul_add(angle.cos(), cx);
        let y = (sqrt3 * r * breath).mul_add(angle.sin(), cy);
        centers.push((x, y, 2));
    }

    // Ring 3: 6 at distance 3r
    for i in 0..6 {
        let angle = i as f32 * FRAC_PI_3;
        let x = (3.0 * r * breath).mul_add(angle.cos(), cx);
        let y = (3.0 * r * breath).mul_add(angle.sin(), cy);
        centers.push((x, y, 3));
    }

    // Ring 3 intermediates
    for i in 0..6 {
        let angle = (i as f32).mul_add(FRAC_PI_3, FRAC_PI_3 / 2.0);
        let d = (4.0_f32 + 3.0).sqrt() * r; // sqrt(7) · r for intermediate positions
        let x = (d * breath).mul_add(angle.cos(), cx);
        let y = (d * breath).mul_add(angle.sin(), cy);
        centers.push((x, y, 3));
    }

    // More ring 3 positions
    for i in 0..6 {
        let a1 = i as f32 * FRAC_PI_3;
        let a2 = (i + 1) as f32 * FRAC_PI_3;
        let x = (2.0f32.mul_add(a1.cos(), a2.cos()) * r).mul_add(breath, cx);
        let y = (2.0f32.mul_add(a1.sin(), a2.sin()) * r).mul_add(breath, cy);
        centers.push((x, y, 3));
    }

    // Determine how many circles to show based on progress
    let max_rings = 3;
    let rings_revealed = progress * (max_rings as f32 + 1.0);

    // Background glow behind the whole construction
    let glow = glow_color(mode, time);
    canvas = canvas.circle(cx, cy, radius * 1.2).fill(glow).done();

    // Draw circles ring by ring with fade-in
    for &(x, y, ring) in &centers {
        let ring_progress = (rings_revealed - ring as f32).clamp(0.0, 1.0);
        if ring_progress <= 0.0 {
            continue;
        }

        let alpha = ring_progress;
        let depth = ring as f32 / max_rings as f32;
        let stroke_color = geo_color(mode, depth, time).with_alpha(alpha * 0.9);
        let fill_alpha = alpha * 0.06;
        let fill_color = geo_color(mode, depth + 0.5, time).with_alpha(fill_alpha);

        let scale = ring_progress; // Grows from 0 to 1
        let current_r = r * scale;

        // Subtle glow halo
        if ring_progress > 0.5 {
            let halo_alpha = (ring_progress - 0.5) * 0.1;
            canvas = canvas
                .circle(x, y, current_r * 1.5)
                .fill(geo_color(mode, depth, time).with_alpha(halo_alpha))
                .done();
        }

        canvas = canvas
            .circle(x, y, current_r)
            .fill(fill_color)
            .stroke(stroke_color, 1.5)
            .done();
    }

    // Outer bounding circle
    let outer_alpha = progress.min(0.3) / 0.3;
    canvas = canvas
        .circle(cx, cy, radius * breath)
        .stroke(
            geo_color(mode, 0.0, time).with_alpha(outer_alpha * 0.6),
            2.0,
        )
        .done();

    // Second outer circle
    canvas = canvas
        .circle(cx, cy, radius * breath * 1.05)
        .stroke(
            geo_color(mode, 0.2, time).with_alpha(outer_alpha * 0.3),
            1.0,
        )
        .done();

    canvas
}

// ═══════════════════════════════════════════════════════════════════
// Pattern 2: Metatron's Cube
// ═══════════════════════════════════════════════════════════════════

fn draw_metatrons_cube(
    mut canvas: PixelCanvas,
    cx: f32,
    cy: f32,
    radius: f32,
    progress: f32,
    time: f32,
    mode: ColorMode,
) -> PixelCanvas {
    // Metatron's Cube: 13 circles (1 center + 6 inner + 6 outer) connected
    // by lines through every pair of centers.
    let r_inner = radius * 0.35;
    let r_outer = radius * 0.70;
    let circle_r = radius * 0.1;

    // Slow rotation
    let rot = time * 0.15;

    // Generate 13 node positions
    let mut nodes: Vec<(f32, f32)> = Vec::with_capacity(13);
    nodes.push((cx, cy)); // Center

    for i in 0..6 {
        let angle = (i as f32).mul_add(FRAC_PI_3, rot);
        nodes.push((cx + r_inner * angle.cos(), cy + r_inner * angle.sin()));
    }
    for i in 0..6 {
        let angle = (i as f32).mul_add(FRAC_PI_3, FRAC_PI_3 / 2.0) + rot;
        nodes.push((cx + r_outer * angle.cos(), cy + r_outer * angle.sin()));
    }

    // Phase 1 (0..0.4): Circles appear
    // Phase 2 (0.3..0.8): Lines connect
    // Phase 3 (0.7..1.0): Full glow and pulse

    let circle_reveal = (progress / 0.4).min(1.0);
    let line_reveal = ((progress - 0.3) / 0.5).clamp(0.0, 1.0);
    let glow_phase = ((progress - 0.7) / 0.3).clamp(0.0, 1.0);

    // Background radial glow
    if glow_phase > 0.0 {
        canvas = canvas
            .circle(cx, cy, radius * 1.1)
            .fill(glow_color(mode, time).with_alpha(glow_phase * 0.15))
            .done();
    }

    // Draw connecting lines
    if line_reveal > 0.0 {
        let total_lines = nodes.len() * (nodes.len() - 1) / 2;
        let lines_shown = (line_reveal * total_lines as f32) as usize;
        let mut count = 0;

        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                if count >= lines_shown {
                    break;
                }
                let (x1, y1) = nodes[i];
                let (x2, y2) = nodes[j];

                let depth = count as f32 / total_lines as f32;
                let color = geo_color(mode, depth, time).with_alpha(0.35);

                canvas = canvas.line(x1, y1, x2, y2).color(color).width(0.8).done();

                count += 1;
            }
            if count >= lines_shown {
                break;
            }
        }
    }

    // Draw circles (nodes)
    let circles_shown = (circle_reveal * nodes.len() as f32) as usize;
    for (i, &(x, y)) in nodes.iter().enumerate() {
        if i >= circles_shown {
            break;
        }

        let depth = i as f32 / nodes.len() as f32;
        let stroke_c = geo_color(mode, depth, time);
        let fill_c = geo_color(mode, depth + 0.3, time).with_alpha(0.12);

        // Pulsing radius
        let pulse = 0.08f32.mul_add(time.mul_add(2.0, i as f32 * 0.5).sin(), 1.0);
        let cr = circle_r * pulse;

        // Glow halo
        if glow_phase > 0.0 {
            canvas = canvas
                .circle(x, y, cr * 2.0)
                .fill(stroke_c.with_alpha(glow_phase * 0.08))
                .done();
        }

        canvas = canvas
            .circle(x, y, cr)
            .fill(fill_c)
            .stroke(stroke_c.with_alpha(0.9), 1.5)
            .done();
    }

    // Outer hexagonal frame
    let hex_alpha = circle_reveal * 0.4;
    let hex_points: Vec<(f32, f32)> = (0..6)
        .map(|i| {
            let angle = (i as f32).mul_add(FRAC_PI_3, rot);
            (
                radius.mul_add(angle.cos(), cx),
                radius.mul_add(angle.sin(), cy),
            )
        })
        .collect();

    canvas = canvas
        .polygon(hex_points)
        .stroke(geo_color(mode, 0.5, time).with_alpha(hex_alpha), 1.0)
        .done();

    canvas
}

// ═══════════════════════════════════════════════════════════════════
// Pattern 3: Sri Yantra
// ═══════════════════════════════════════════════════════════════════

fn draw_sri_yantra(
    mut canvas: PixelCanvas,
    cx: f32,
    cy: f32,
    radius: f32,
    progress: f32,
    time: f32,
    mode: ColorMode,
) -> PixelCanvas {
    // Sri Yantra: 9 interlocking triangles (4 upward, 5 downward)
    // arranged around a central point (bindu), enclosed in concentric
    // circles and a square gate (bhupura).

    let breath = 0.015f32.mul_add((time * 0.8).sin(), 1.0);

    // Phase 1 (0..0.3): Outer frame and circles appear
    // Phase 2 (0.2..0.7): Triangles draw in, largest first
    // Phase 3 (0.6..0.9): Inner triangles and bindu
    // Phase 4 (0.8..1.0): Full radiance

    let frame_reveal = (progress / 0.3).min(1.0);
    let tri_reveal = ((progress - 0.2) / 0.5).clamp(0.0, 1.0);
    let inner_reveal = ((progress - 0.6) / 0.3).clamp(0.0, 1.0);
    let radiance = ((progress - 0.8) / 0.2).clamp(0.0, 1.0);

    // Outer bhupura (square gate with T-shaped gates)
    if frame_reveal > 0.0 {
        let sq = radius * 0.95 * breath;
        let gate_color = geo_color(mode, 0.0, time).with_alpha(frame_reveal * 0.6);

        // Square
        canvas = canvas
            .rect(cx - sq, cy - sq, sq * 2.0, sq * 2.0)
            .stroke(gate_color, 2.0)
            .done();

        // Inner square
        let sq2 = sq * 0.92;
        canvas = canvas
            .rect(cx - sq2, cy - sq2, sq2 * 2.0, sq2 * 2.0)
            .stroke(gate_color.with_alpha(frame_reveal * 0.3), 1.0)
            .done();

        // Concentric circles
        for i in 0..3 {
            let cr = radius * (i as f32).mul_add(-0.05, 0.85) * breath;
            let alpha = frame_reveal * (i as f32).mul_add(-0.12, 0.5);
            canvas = canvas
                .circle(cx, cy, cr)
                .stroke(geo_color(mode, i as f32 * 0.2, time).with_alpha(alpha), 1.2)
                .done();
        }
    }

    // Triangles
    if tri_reveal > 0.0 {
        // 4 upward triangles (Shiva) — sizes decrease inward
        let up_sizes = [0.75, 0.55, 0.38, 0.22];
        let up_offsets = [0.0, 0.08, 0.15, 0.22]; // Vertical offset downward from center

        for (i, (&size, &offset)) in up_sizes.iter().zip(up_offsets.iter()).enumerate() {
            let reveal_t = (tri_reveal * (up_sizes.len() as f32 + 1.0) - i as f32).clamp(0.0, 1.0);
            if reveal_t <= 0.0 {
                continue;
            }

            let r = radius * size * breath;
            let oy = radius * offset;
            let depth = i as f32 / up_sizes.len() as f32;
            let color = geo_color(mode, depth, time).with_alpha(reveal_t * 0.8);
            let fill_c = geo_color(mode, depth + 0.1, time).with_alpha(reveal_t * 0.04);

            // Upward triangle: apex at top
            let points = vec![
                (cx, cy - r + oy),                   // Apex
                (cx - r * 0.866, cy + r * 0.5 + oy), // Bottom-left
                (cx + r * 0.866, cy + r * 0.5 + oy), // Bottom-right
            ];

            canvas = canvas
                .polygon(points)
                .fill(fill_c)
                .stroke(color, (1.0 - depth).mul_add(0.8, 1.2))
                .done();
        }

        // 5 downward triangles (Shakti) — sizes decrease inward
        let down_sizes = [0.82, 0.62, 0.45, 0.30, 0.15];
        let down_offsets = [-0.05, 0.03, 0.10, 0.17, 0.24];

        for (i, (&size, &offset)) in down_sizes.iter().zip(down_offsets.iter()).enumerate() {
            let reveal_t =
                (tri_reveal * (down_sizes.len() as f32 + 1.0) - i as f32).clamp(0.0, 1.0);
            if reveal_t <= 0.0 {
                continue;
            }

            let r = radius * size * breath;
            let oy = radius * offset;
            let depth = (i as f32 + 4.0) / 9.0;
            let color = geo_color(mode, depth, time).with_alpha(reveal_t * 0.8);
            let fill_c = geo_color(mode, depth + 0.1, time).with_alpha(reveal_t * 0.04);

            // Downward triangle: apex at bottom
            let points = vec![
                (cx, cy + r - oy),                   // Apex (bottom)
                (cx - r * 0.866, cy - r * 0.5 - oy), // Top-left
                (cx + r * 0.866, cy - r * 0.5 - oy), // Top-right
            ];

            canvas = canvas
                .polygon(points)
                .fill(fill_c)
                .stroke(color, (1.0 - depth).mul_add(0.8, 1.2))
                .done();
        }
    }

    // Bindu (central point) — innermost sacred point
    if inner_reveal > 0.0 {
        let bindu_r = radius * 0.03 * breath;
        let bindu_color = geo_color(mode, 1.0, time);

        // Glow rings
        for i in 0..4 {
            let gr = bindu_r * (i as f32).mul_add(2.0, 3.0);
            let ga = inner_reveal * 0.08 / (i as f32).mul_add(0.5, 1.0);
            canvas = canvas
                .circle(cx, cy, gr)
                .fill(bindu_color.with_alpha(ga))
                .done();
        }

        // The bindu itself
        canvas = canvas
            .circle(cx, cy, bindu_r)
            .fill(bindu_color.with_alpha(inner_reveal))
            .done();
    }

    // Radiance — pulsing light from center when fully revealed
    if radiance > 0.0 {
        let pulse = (time * 3.0).sin().mul_add(0.5, 0.5);
        let rays = 12;
        for i in 0..rays {
            let angle = time.mul_add(0.2, i as f32 * TAU / rays as f32);
            let len = radius * 0.8 * 0.3f32.mul_add(time.mul_add(1.5, i as f32).sin(), 0.7);
            let x2 = cx + len * angle.cos();
            let y2 = cy + len * angle.sin();

            let ray_c = geo_color(mode, i as f32 / rays as f32, time)
                .with_alpha(radiance * 0.12 * 0.5f32.mul_add(pulse, 0.5));

            canvas = canvas.line(cx, cy, x2, y2).color(ray_c).width(1.5).done();
        }
    }

    canvas
}
