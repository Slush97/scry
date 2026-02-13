//! Mind-Bending Optical Illusions — exploits flaws in human visual processing.
//!
//! Five pages of psychologically intense illusions, each targeting a different
//! vulnerability in how your brain constructs reality.
//!
//! | Page | Illusion                | Effect |
//! |------|------------------------|--------|
//! | 1    | Moiré Interference     | Phantom waves emerge from overlapping grids |
//! | 2    | Rotating Snakes        | Static image appears to rotate |
//! | 3    | Lilac Chaser           | Brain hallucinates green dots; real dots vanish |
//! | 4    | Motion-Induced Blindness | Static dots vanish from consciousness |
//! | 5    | Penrose Triangle       | Impossible geometry assembles itself |
//!
//! Run with: `cargo run --example mind_benders --features widget`

#![allow(
    clippy::suboptimal_flops,
    clippy::items_after_statements,
    clippy::cast_precision_loss,
    clippy::unreadable_literal,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::needless_range_loop
)]

use std::io::stdout;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use ratatui_pixelcanvas::prelude::{
    PixelCanvasState, PixelCanvasWidget, Picker, ProtocolKind,
};
use ratatui_pixelcanvas::scene::style::{
    Color as C, Transform,
};
use ratatui_pixelcanvas::scene::PixelCanvas;
use ratatui_pixelcanvas::transport;

// ═════════════════════════════════════════════════════════════════════════════
// Pages
// ═════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, PartialEq, Eq)]
enum Page {
    Moire,
    RotatingSnakes,
    LilacChaser,
    MotionBlindness,
    PenroseTriangle,
}

impl Page {
    const ALL: [Self; 5] = [
        Self::Moire,
        Self::RotatingSnakes,
        Self::LilacChaser,
        Self::MotionBlindness,
        Self::PenroseTriangle,
    ];

    const fn label(self) -> &'static str {
        match self {
            Self::Moire => "Moiré Interference Engine",
            Self::RotatingSnakes => "Rotating Snakes (Kitaoka)",
            Self::LilacChaser => "Lilac Chaser",
            Self::MotionBlindness => "Motion-Induced Blindness",
            Self::PenroseTriangle => "Penrose Impossible Triangle",
        }
    }

    const fn hint(self) -> &'static str {
        match self {
            Self::Moire => "Watch the phantom waves between the grids",
            Self::RotatingSnakes => "Look at the edges — the discs appear to ROTATE (image is static!)",
            Self::LilacChaser => "Stare at the cross. A GREEN dot will appear. Then they ALL vanish.",
            Self::MotionBlindness => "Stare at center. The yellow dots will DISAPPEAR from your mind.",
            Self::PenroseTriangle => "This triangle cannot exist in 3D space",
        }
    }

    fn index(self) -> usize {
        Self::ALL.iter().position(|&p| p == self).unwrap_or(0)
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Helpers
// ═════════════════════════════════════════════════════════════════════════════

fn pixel_size(area: Rect, state: &PixelCanvasState) -> (u32, u32, f32, f32) {
    let font = state.font_size();
    let w = u32::from(area.width) * u32::from(font.width);
    let h = u32::from(area.height) * u32::from(font.height);
    (w, h, w as f32, h as f32)
}

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

// ═════════════════════════════════════════════════════════════════════════════
// Main
// ═════════════════════════════════════════════════════════════════════════════

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

    let mut page = Page::Moire;
    let start = std::time::Instant::now();

    loop {
        let t = start.elapsed().as_secs_f32();

        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(3)])
                .split(frame.area());

            let area = chunks[0];
            let canvas = match page {
                Page::Moire => build_moire(area, &state, t),
                Page::RotatingSnakes => build_rotating_snakes(area, &state, t),
                Page::LilacChaser => build_lilac_chaser(area, &state, t),
                Page::MotionBlindness => build_motion_blindness(area, &state, t),
                Page::PenroseTriangle => build_penrose(area, &state, t),
            };

            // skip_cache: these scenes change every frame, so content
            // hashing would never hit cache — saves ~2ms per frame.
            let widget = PixelCanvasWidget::new(canvas).z_index(-1).skip_cache();
            frame.render_stateful_widget(widget, area, &mut state);

            let idx = page.index();
            let status_text = format!(
                " ◄ ► page {}/{}  │  {}  │  {}  │  'q' quit",
                idx + 1,
                Page::ALL.len(),
                page.label(),
                page.hint(),
            );
            let status = Paragraph::new(status_text)
                .block(Block::default().borders(Borders::TOP));
            frame.render_widget(status, chunks[1]);
        })?;
        state.flush()?;

        if event::poll(std::time::Duration::from_millis(33))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Left => {
                            let idx = page.index();
                            page = Page::ALL[(idx + Page::ALL.len() - 1) % Page::ALL.len()];
                        }
                        KeyCode::Right => {
                            let idx = page.index();
                            page = Page::ALL[(idx + 1) % Page::ALL.len()];
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    state.cleanup();
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}

// ═════════════════════════════════════════════════════════════════════════════
// Page 1: Moiré Interference Engine
// ═════════════════════════════════════════════════════════════════════════════
//
// Two dense radial grids of concentric circles with slightly offset centers.
// One grid slowly rotates. The interference creates phantom rippling waves
// that don't exist in either grid alone — they're fabricated by your visual
// cortex trying to parse a pattern from noise.

fn build_moire(area: Rect, state: &PixelCanvasState, t: f32) -> PixelCanvas {
    let (w, h, wf, hf) = pixel_size(area, state);
    let mut canvas = PixelCanvas::new(w, h).background(C::from_rgba8(5, 5, 12, 255));

    let cx = wf / 2.0;
    let cy = hf / 2.0;
    let r_max = wf.min(hf) * 0.48;
    // Wider spacing = fewer commands, still great moiré effect
    let spacing = 5.0;
    let num_rings = (r_max / spacing) as usize;

    // Oscillating offset for the second grid
    let offset_x = (t * 0.4).sin() * 15.0;
    let offset_y = (t * 0.3).cos() * 10.0;

    // Grid 1: Cyan rings, stationary
    for i in 1..=num_rings {
        let r = i as f32 * spacing;
        canvas = canvas
            .circle(cx, cy, r)
            .stroke(C::from_rgba8(0, 180, 255, 70), 1.5)
            .done();
    }

    // Grid 2: Magenta rings, offset and slowly drifting
    for i in 1..=num_rings {
        let r = i as f32 * spacing;
        canvas = canvas
            .circle(cx + offset_x, cy + offset_y, r)
            .stroke(C::from_rgba8(255, 0, 180, 70), 1.5)
            .done();
    }

    // Radial spokes — pre-rotate endpoints instead of using Group transforms.
    // This eliminates 2 expensive Group commands (temp pixmap alloc each).
    let num_spokes = 60;
    let spoke_len = r_max;
    let rotation1 = t * 0.15;
    let rotation2 = -t * 0.1;

    for i in 0..num_spokes {
        let base_angle = (i as f32 / num_spokes as f32) * std::f32::consts::TAU;

        // Spoke grid 1 (rotated)
        let a1 = base_angle + rotation1;
        let x2 = cx + a1.cos() * spoke_len;
        let y2 = cy + a1.sin() * spoke_len;
        canvas = canvas
            .line(cx, cy, x2, y2)
            .color(C::from_rgba8(180, 180, 255, 35))
            .width(1.0)
            .done();

        // Spoke grid 2 (counter-rotated)
        let a2 = base_angle + rotation2;
        let x2b = cx + a2.cos() * spoke_len;
        let y2b = cy + a2.sin() * spoke_len;
        canvas = canvas
            .line(cx, cy, x2b, y2b)
            .color(C::from_rgba8(255, 200, 100, 30))
            .width(1.0)
            .done();
    }

    canvas
}

// ═════════════════════════════════════════════════════════════════════════════
// Page 2: Rotating Snakes (Kitaoka Illusion)
// ═════════════════════════════════════════════════════════════════════════════
//
// Multiple discs, each containing concentric ring segments with a specific
// color luminance sequence: black → dark blue → white → yellow.
//
// The asymmetric luminance gradient exploits different processing speeds
// of high-contrast vs low-contrast edges in your peripheral vision. The
// rings appear to ROTATE even though the entire image is completely static.
//
// For best effect: don't stare at any single disc — let your eyes wander.

fn build_rotating_snakes(area: Rect, state: &PixelCanvasState, _t: f32) -> PixelCanvas {
    let (w, h, wf, hf) = pixel_size(area, state);
    let mut canvas = PixelCanvas::new(w, h).background(C::from_rgba8(128, 128, 128, 255));

    // The critical color sequence for the Kitaoka illusion:
    // black → dark → white → light (repeating around each ring)
    let colors = [
        C::from_rgba8(10, 10, 10, 255),     // black
        C::from_rgba8(30, 50, 120, 255),     // dark blue
        C::from_rgba8(250, 250, 250, 255),   // white
        C::from_rgba8(240, 210, 60, 255),    // yellow
    ];

    // Grid of discs — cap total discs to keep arc count manageable
    let disc_size = (wf.min(hf) / 4.0).min(140.0);
    let cols = (((wf - disc_size * 0.5) / (disc_size * 1.2)) as usize).max(1).min(6);
    let rows = (((hf - disc_size * 0.5) / (disc_size * 1.2)) as usize).max(1).min(4);

    let total_w = cols as f32 * disc_size * 1.2;
    let total_h = rows as f32 * disc_size * 1.2;
    let start_x = (wf - total_w) / 2.0 + disc_size * 0.6;
    let start_y = (hf - total_h) / 2.0 + disc_size * 0.6;

    let pi = std::f32::consts::PI;

    for row in 0..rows {
        for col in 0..cols {
            let dcx = start_x + col as f32 * disc_size * 1.2;
            let dcy = start_y + row as f32 * disc_size * 1.2;

            // Alternate rotation direction per disc for maximum effect
            let direction: f32 = if (row + col) % 2 == 0 { 1.0 } else { -1.0 };

            // Fewer rings per disc (3 instead of 5) — still effective
            let num_rings = 3;
            let ring_width = disc_size * 0.5 / num_rings as f32;

            for ring in 0..num_rings {
                let r = disc_size * 0.5 - ring as f32 * ring_width;
                if r < 3.0 {
                    break;
                }

                // Fewer segments per ring — 8 is plenty for the illusion
                let segs = 8;
                let seg_sweep = 2.0 * pi / segs as f32;

                // Phase offset per ring to create the spiral illusion
                let phase = ring as f32 * pi / 6.0 * direction;

                for seg in 0..segs {
                    let start_angle = seg as f32 * seg_sweep + phase;
                    let color_idx = seg % 4;
                    let color = colors[color_idx];

                    // Draw arc segment
                    canvas = canvas
                        .arc(dcx, dcy, r, start_angle, seg_sweep * 0.92)
                        .stroke(color, ring_width * 0.85)
                        .done();
                }
            }

            // Center dot
            canvas = canvas
                .circle(dcx, dcy, 3.0)
                .fill(C::from_rgba8(10, 10, 10, 255))
                .done();
        }
    }

    canvas
}

// ═════════════════════════════════════════════════════════════════════════════
// Page 3: Lilac Chaser
// ═════════════════════════════════════════════════════════════════════════════
//
// 12 magenta/lilac dots arranged in a circle. One dot disappears in
// clockwise sequence (rotating gap).
//
// After staring at the central fixation cross for ~3 seconds:
// 1. Your brain fills the gap with a GREEN dot (complementary afterimage)
// 2. Eventually, ALL the magenta dots fade from awareness (Troxler's fading)
//
// These are real, documented neurological phenomena.

fn build_lilac_chaser(area: Rect, state: &PixelCanvasState, t: f32) -> PixelCanvas {
    let (w, h, wf, hf) = pixel_size(area, state);
    let mut canvas = PixelCanvas::new(w, h).background(C::from_rgba8(180, 180, 180, 255));

    let cx = wf / 2.0;
    let cy = hf / 2.0;
    let radius = wf.min(hf) * 0.32;
    let dot_radius = radius * 0.12;
    let num_dots = 12;

    // Determine which dot is currently hidden (cycles every ~0.15s for best effect)
    let cycle_speed = 6.0; // dots per second
    let hidden_idx = (t * cycle_speed) as usize % num_dots;

    let pi = std::f32::consts::PI;

    // Draw the dots
    for i in 0..num_dots {
        let angle = (i as f32 / num_dots as f32) * 2.0 * pi - pi / 2.0;
        let dx = cx + angle.cos() * radius;
        let dy = cy + angle.sin() * radius;

        if i == hidden_idx {
            // This dot is hidden — the gap where your brain will hallucinate green
            continue;
        }

        // Lilac/magenta with soft Gaussian-like falloff via multiple concentric circles
        // Outer glow
        canvas = canvas
            .circle(dx, dy, dot_radius * 1.4)
            .fill(C::from_rgba8(200, 100, 200, 40))
            .done();
        // Mid
        canvas = canvas
            .circle(dx, dy, dot_radius * 1.1)
            .fill(C::from_rgba8(200, 80, 200, 80))
            .done();
        // Core
        canvas = canvas
            .circle(dx, dy, dot_radius)
            .fill(C::from_rgba8(210, 60, 210, 160))
            .done();
        // Inner bright
        canvas = canvas
            .circle(dx, dy, dot_radius * 0.7)
            .fill(C::from_rgba8(220, 80, 220, 200))
            .done();
    }

    // Fixation cross — MUST stare at this for the illusion to work
    let cross_size = 8.0;
    let cross_width = 2.5;
    canvas = canvas
        .line(cx - cross_size, cy, cx + cross_size, cy)
        .color(C::from_rgba8(40, 40, 40, 255))
        .width(cross_width)
        .done();
    canvas = canvas
        .line(cx, cy - cross_size, cx, cy + cross_size)
        .color(C::from_rgba8(40, 40, 40, 255))
        .width(cross_width)
        .done();

    canvas
}

// ═════════════════════════════════════════════════════════════════════════════
// Page 4: Motion-Induced Blindness
// ═════════════════════════════════════════════════════════════════════════════
//
// A slowly rotating grid of small blue crosses with three static bright
// yellow dots overlaid. After staring at the center fixation point for a
// few seconds, the static yellow dots VANISH FROM YOUR CONSCIOUSNESS —
// your visual system literally deletes stable stimuli in favor of the
// dynamic pattern.
//
// This is not a trick. The dots are always being drawn. Your brain just
// stops perceiving them.

fn build_motion_blindness(area: Rect, state: &PixelCanvasState, t: f32) -> PixelCanvas {
    let (w, h, wf, hf) = pixel_size(area, state);
    let mut canvas = PixelCanvas::new(w, h).background(C::from_rgba8(5, 5, 15, 255));

    let cx = wf / 2.0;
    let cy = hf / 2.0;
    let field_size = wf.min(hf) * 0.45;

    // Rotating field of blue crosses
    let cross_arm = 5.0;
    let cross_width = 1.8;
    let grid_spacing = 40.0; // wider spacing = fewer crosses = faster
    let rotation = t * 0.35; // slow, steady rotation

    canvas = canvas
        .group(Transform::rotate_at(rotation, cx, cy))
        .canvas(|mut inner| {
            let cols = (field_size * 2.0 / grid_spacing) as i32;
            let rows = (field_size * 2.0 / grid_spacing) as i32;

            for row in -rows..=rows {
                for col in -cols..=cols {
                    let bx = cx + col as f32 * grid_spacing;
                    let by = cy + row as f32 * grid_spacing;

                    // Distance from center — skip if too far (circular field)
                    let dx = bx - cx;
                    let dy = by - cy;
                    if dx * dx + dy * dy > field_size * field_size {
                        continue;
                    }

                    // Diagonal cross (×)
                    inner = inner
                        .line(bx - cross_arm, by - cross_arm, bx + cross_arm, by + cross_arm)
                        .color(C::from_rgba8(50, 80, 220, 200))
                        .width(cross_width)
                        .done();
                    inner = inner
                        .line(bx + cross_arm, by - cross_arm, bx - cross_arm, by + cross_arm)
                        .color(C::from_rgba8(50, 80, 220, 200))
                        .width(cross_width)
                        .done();
                }
            }
            inner
        })
        .done();

    // Three static yellow dots — these are NOT in the rotating group.
    // They are always visible, always drawn, yet your brain will
    // make them disappear.
    let dot_radius = 7.0;
    let dot_distance = field_size * 0.55;
    let dot_color = C::from_rgba8(255, 230, 0, 255);
    let pi = std::f32::consts::PI;

    // Arrange in equilateral triangle
    let dot_positions = [
        (cx + dot_distance * (pi * 0.5).cos(), cy + dot_distance * (pi * 0.5).sin()),
        (cx + dot_distance * (pi * 7.0 / 6.0).cos(), cy + dot_distance * (pi * 7.0 / 6.0).sin()),
        (cx + dot_distance * (pi * 11.0 / 6.0).cos(), cy + dot_distance * (pi * 11.0 / 6.0).sin()),
    ];

    for &(dx, dy) in &dot_positions {
        canvas = canvas
            .circle(dx, dy, dot_radius)
            .fill(dot_color)
            .done();
    }

    // Center fixation — green dot to stare at
    canvas = canvas
        .circle(cx, cy, 4.0)
        .fill(C::from_rgba8(0, 255, 80, 255))
        .done();

    canvas
}

// ═════════════════════════════════════════════════════════════════════════════
// Page 5: Penrose Impossible Triangle (Animated Assembly)
// ═════════════════════════════════════════════════════════════════════════════
//
// An impossible triangle that assembles itself — three beams slide into
// place and connect in a geometrically impossible configuration. The beams
// use gradient shading to enhance the 3D depth illusion.
//
// Once assembled, the triangle subtly breathes (scale oscillation).

fn build_penrose(area: Rect, state: &PixelCanvasState, t: f32) -> PixelCanvas {
    let (w, h, wf, hf) = pixel_size(area, state);
    let mut canvas = PixelCanvas::new(w, h).background(C::from_rgba8(15, 15, 25, 255));

    let cx = wf / 2.0;
    let cy = hf / 2.0;
    let base_size = wf.min(hf) * 0.32;

    // Assembly animation: beams slide in over 3 seconds, then hold
    let assembly_t = (t * 0.5).min(1.0); // 0→1 over 2 seconds
    let ease = 1.0 - (1.0 - assembly_t).powi(3); // ease-out cubic

    // Breathing once assembled
    let breath = if assembly_t >= 1.0 {
        1.0 + ((t - 2.0) * 1.2).sin() * 0.02
    } else {
        1.0
    };

    let size = base_size * breath;
    let thickness = size * 0.18;

    // Triangle vertices (equilateral, pointing up)
    let top = (cx, cy - size * 0.6);
    let bl = (cx - size * 0.6, cy + size * 0.4);
    let br = (cx + size * 0.6, cy + size * 0.4);

    // Colors for the 3 beams (with light/dark faces for 3D effect)
    let beam_colors_light = [
        C::from_rgba8(80, 170, 240, 255),
        C::from_rgba8(240, 100, 80, 255),
        C::from_rgba8(90, 210, 130, 255),
    ];
    let beam_colors_dark = [
        C::from_rgba8(40, 100, 160, 255),
        C::from_rgba8(160, 50, 40, 255),
        C::from_rgba8(40, 140, 70, 255),
    ];

    // Each beam slides in from an offset direction
    let slide_offsets = [
        (0.0_f32, -200.0_f32),  // top beam slides down from above
        (-200.0, 150.0),        // left beam slides in from lower-left
        (200.0, 150.0),         // right beam slides in from lower-right
    ];

    // Helper to interpolate position during assembly
    let interp = |base: f32, offset: f32| -> f32 {
        base + offset * (1.0 - ease)
    };

    // ── Beam 1: Top → Bottom-Left ──
    {
        let ox = slide_offsets[0].0;
        let oy = slide_offsets[0].1;

        // Outer face (light)
        let pts = vec![
            (interp(top.0, ox), interp(top.1, oy)),
            (interp(top.0 - thickness * 0.5, ox), interp(top.1 + thickness * 0.3, oy)),
            (interp(bl.0 + thickness * 0.15, ox), interp(bl.1 + thickness * 0.15, oy)),
            (interp(bl.0, ox), interp(bl.1, oy)),
            (interp(bl.0 + thickness * 0.7, ox), interp(bl.1 - thickness * 0.3, oy)),
            (interp(top.0 + thickness * 0.35, ox), interp(top.1 + thickness * 0.5, oy)),
        ];
        canvas = canvas
            .polygon(pts)
            .fill(beam_colors_light[0])
            .stroke(C::from_rgba8(20, 20, 35, 200), 1.5)
            .done();

        // Inner face (dark) — the visible interior face
        let inner_pts = vec![
            (interp(top.0 + thickness * 0.35, ox), interp(top.1 + thickness * 0.5, oy)),
            (interp(bl.0 + thickness * 0.7, ox), interp(bl.1 - thickness * 0.3, oy)),
            (interp(bl.0 + thickness * 0.55, ox), interp(bl.1 + thickness * 0.05, oy)),
            (interp(top.0 + thickness * 0.1, ox), interp(top.1 + thickness * 0.8, oy)),
        ];
        canvas = canvas
            .polygon(inner_pts)
            .fill(beam_colors_dark[0])
            .stroke(C::from_rgba8(20, 20, 35, 120), 1.0)
            .done();
    }

    // ── Beam 2: Bottom-Left → Bottom-Right ──
    {
        let ox = slide_offsets[1].0;
        let oy = slide_offsets[1].1;

        let pts = vec![
            (interp(bl.0, ox), interp(bl.1, oy)),
            (interp(bl.0 + thickness * 0.15, ox), interp(bl.1 + thickness * 0.15, oy)),
            (interp(br.0 - thickness * 0.15, ox), interp(br.1 + thickness * 0.15, oy)),
            (interp(br.0, ox), interp(br.1, oy)),
            (interp(br.0 - thickness * 0.5, ox), interp(br.1 - thickness * 0.4, oy)),
            (interp(bl.0 + thickness * 0.7, ox), interp(bl.1 - thickness * 0.3, oy)),
        ];
        canvas = canvas
            .polygon(pts)
            .fill(beam_colors_light[1])
            .stroke(C::from_rgba8(20, 20, 35, 200), 1.5)
            .done();

        // Top face (dark)
        let inner_pts = vec![
            (interp(bl.0 + thickness * 0.7, ox), interp(bl.1 - thickness * 0.3, oy)),
            (interp(br.0 - thickness * 0.5, ox), interp(br.1 - thickness * 0.4, oy)),
            (interp(br.0 - thickness * 0.35, ox), interp(br.1 - thickness * 0.1, oy)),
            (interp(bl.0 + thickness * 0.55, ox), interp(bl.1 + thickness * 0.05, oy)),
        ];
        canvas = canvas
            .polygon(inner_pts)
            .fill(beam_colors_dark[1])
            .stroke(C::from_rgba8(20, 20, 35, 120), 1.0)
            .done();
    }

    // ── Beam 3: Bottom-Right → Top ──
    // This is the "impossible" connection — it overlaps beam 1 in a way
    // that would be physically impossible in 3D.
    {
        let ox = slide_offsets[2].0;
        let oy = slide_offsets[2].1;

        let pts = vec![
            (interp(br.0, ox), interp(br.1, oy)),
            (interp(br.0 - thickness * 0.15, ox), interp(br.1 + thickness * 0.15, oy)),
            (interp(top.0 + thickness * 0.4, ox), interp(top.1 + thickness * 0.1, oy)),
            (interp(top.0, ox), interp(top.1, oy)),
            (interp(top.0 + thickness * 0.35, ox), interp(top.1 + thickness * 0.5, oy)),
            (interp(br.0 - thickness * 0.5, ox), interp(br.1 - thickness * 0.4, oy)),
        ];
        canvas = canvas
            .polygon(pts)
            .fill(beam_colors_light[2])
            .stroke(C::from_rgba8(20, 20, 35, 200), 1.5)
            .done();

        // Inner face (dark)
        let inner_pts = vec![
            (interp(top.0 + thickness * 0.35, ox), interp(top.1 + thickness * 0.5, oy)),
            (interp(top.0 + thickness * 0.1, ox), interp(top.1 + thickness * 0.8, oy)),
            (interp(br.0 - thickness * 0.35, ox), interp(br.1 - thickness * 0.1, oy)),
            (interp(br.0 - thickness * 0.5, ox), interp(br.1 - thickness * 0.4, oy)),
        ];
        canvas = canvas
            .polygon(inner_pts)
            .fill(beam_colors_dark[2])
            .stroke(C::from_rgba8(20, 20, 35, 120), 1.0)
            .done();
    }

    // Ambient glow around the triangle
    if assembly_t >= 1.0 {
        let glow_r = size * 0.9;
        let glow_alpha = ((t * 0.8).sin() * 0.3 + 0.3).max(0.0);
        let a = (glow_alpha * 255.0) as u8;
        canvas = canvas
            .circle(cx, cy, glow_r)
            .stroke(C::from_rgba8(100, 150, 255, a), 2.0)
            .done();
    }

    // Decorative particles orbiting the triangle once assembled
    if assembly_t >= 1.0 {
        let num_particles = 20;
        let orbit_r = size * 0.75;
        for i in 0..num_particles {
            let angle = (i as f32 / num_particles as f32) * std::f32::consts::TAU + t * 0.4;
            let pr = orbit_r + (angle * 3.0 + t).sin() * 15.0;
            let px = cx + angle.cos() * pr;
            let py = cy + angle.sin() * pr;
            let (cr, cg, cb) = hsl_to_rgb((i as f32 * 18.0 + t * 30.0) % 360.0, 0.7, 0.65);
            canvas = canvas
                .circle(px, py, 2.5)
                .fill(C::from_rgba8(cr, cg, cb, 150))
                .done();
        }
    }

    canvas
}
