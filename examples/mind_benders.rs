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
use ratatui_pixelcanvas::scene::style::Color as C;
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
    let mut canvas = PixelCanvas::new(w, h).background(C::from_rgba8(2, 2, 8, 255));

    let cx = wf / 2.0;
    let cy = hf / 2.0;
    let r_max = wf.min(hf) * 0.48;
    let spacing = 4.0; // tight enough for rich interference
    let num_rings = (r_max / spacing) as usize;

    // Grid 1: Stationary cyan concentric rings
    for i in 1..=num_rings {
        let r = i as f32 * spacing;
        canvas = canvas
            .circle(cx, cy, r)
            .stroke(C::from_rgba8(0, 200, 255, 90), 1.2)
            .done();
    }

    // Grid 2: Magenta rings with slow circular orbit around center
    let orbit_r = 25.0;
    let ox = (t * 0.35).sin() * orbit_r;
    let oy = (t * 0.35).cos() * orbit_r;
    for i in 1..=num_rings {
        let r = i as f32 * spacing;
        canvas = canvas
            .circle(cx + ox, cy + oy, r)
            .stroke(C::from_rgba8(255, 0, 200, 80), 1.2)
            .done();
    }

    // Grid 3: White/gold rings orbiting at 120° offset — triple interference
    let ox2 = (t * 0.35 + std::f32::consts::TAU / 3.0).sin() * orbit_r * 0.7;
    let oy2 = (t * 0.35 + std::f32::consts::TAU / 3.0).cos() * orbit_r * 0.7;
    // Only draw every other ring to save commands while adding a third layer
    for i in (1..=num_rings).step_by(2) {
        let r = i as f32 * spacing;
        canvas = canvas
            .circle(cx + ox2, cy + oy2, r)
            .stroke(C::from_rgba8(255, 220, 100, 50), 1.0)
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
    let mut canvas = PixelCanvas::new(w, h).background(C::from_rgba8(3, 3, 12, 255));

    let cx = wf / 2.0;
    let cy = hf / 2.0;
    let field_r = wf.min(hf) * 0.44;

    // ── Rotating blue crosses ──
    // Pre-rotate each cross position to avoid Group (temp pixmap).
    // Research: ~5.5°/s rotation, dense grid of + shaped crosses.
    let cross_arm = 4.5;
    let cross_w = 1.5;
    let spacing = 25.0;
    let rotation = t * 0.096; // ~5.5°/s in radians
    let (sin_r, cos_r) = rotation.sin_cos();

    let half_n = (field_r / spacing) as i32;
    for gy in -half_n..=half_n {
        for gx in -half_n..=half_n {
            let lx = gx as f32 * spacing;
            let ly = gy as f32 * spacing;

            // Skip if outside circular field
            if lx * lx + ly * ly > field_r * field_r {
                continue;
            }

            // Rotate around center
            let rx = cx + lx * cos_r - ly * sin_r;
            let ry = cy + lx * sin_r + ly * cos_r;

            // + shaped cross (orthogonal arms, also rotated)
            // Horizontal arm endpoints (rotated)
            let hx1 = rx + cross_arm * cos_r;
            let hy1 = ry + cross_arm * sin_r;
            let hx2 = rx - cross_arm * cos_r;
            let hy2 = ry - cross_arm * sin_r;
            // Vertical arm endpoints (rotated)
            let vx1 = rx - cross_arm * sin_r;
            let vy1 = ry + cross_arm * cos_r;
            let vx2 = rx + cross_arm * sin_r;
            let vy2 = ry - cross_arm * cos_r;

            canvas = canvas
                .line(hx1, hy1, hx2, hy2)
                .color(C::from_rgba8(40, 70, 200, 220))
                .width(cross_w)
                .done();
            canvas = canvas
                .line(vx1, vy1, vx2, vy2)
                .color(C::from_rgba8(40, 70, 200, 220))
                .width(cross_w)
                .done();
        }
    }

    // ── Three static yellow target dots ──
    // Research: targets at moderate eccentricity, arranged as equilateral triangle
    let dot_r = 6.0;
    let dot_dist = field_r * 0.45;
    let dot_color = C::from_rgba8(255, 255, 0, 255);
    let pi = std::f32::consts::PI;

    let targets = [
        (cx, cy - dot_dist),                                          // top
        (cx - dot_dist * (pi / 3.0).sin(), cy + dot_dist * 0.5),    // bottom-left
        (cx + dot_dist * (pi / 3.0).sin(), cy + dot_dist * 0.5),    // bottom-right
    ];

    for &(tx, ty) in &targets {
        // Bright dot with slight glow
        canvas = canvas
            .circle(tx, ty, dot_r + 3.0)
            .fill(C::from_rgba8(255, 255, 0, 30))
            .done();
        canvas = canvas
            .circle(tx, ty, dot_r)
            .fill(dot_color)
            .done();
    }

    // ── Center fixation point — must stare here ──
    canvas = canvas
        .circle(cx, cy, 5.0)
        .fill(C::from_rgba8(0, 255, 80, 255))
        .done();
    canvas = canvas
        .circle(cx, cy, 2.0)
        .fill(C::from_rgba8(255, 255, 255, 255))
        .done();

    canvas
}

// ═════════════════════════════════════════════════════════════════════════════
// Page 5: Penrose Impossible Triangle
// ═════════════════════════════════════════════════════════════════════════════
//
// The classic impossible object. Three bars connect in a cycle where
// each bar appears to pass OVER the next — an arrangement that cannot
// exist in 3D. The trick is in the drawing order: we draw each beam
// so that its near end covers the far end of the previous beam,
// creating a cyclic depth contradiction.
//
// Slowly rotates to let people study the impossibility from all angles.

fn build_penrose(area: Rect, state: &PixelCanvasState, t: f32) -> PixelCanvas {
    let (w, h, wf, hf) = pixel_size(area, state);
    let mut canvas = PixelCanvas::new(w, h).background(C::from_rgba8(12, 12, 20, 255));

    let cx = wf / 2.0;
    let cy = hf / 2.0;
    let scale = wf.min(hf) * 0.35;

    // Slow rotation so viewers can study the impossibility
    let angle = t * 0.2;

    // The Penrose triangle is built from three rhombus-shaped bars.
    // Each bar has an outer edge, inner edge, and two end-caps.
    //
    // We define the triangle in local coordinates, then rotate.
    // The outer triangle vertices (equilateral, centered):
    let s3 = 3.0_f32.sqrt();
    let outer = [
        (0.0_f32, -1.0_f32),                  // top
        (-s3 / 2.0, 0.5_f32),                 // bottom-left
        (s3 / 2.0, 0.5_f32),                  // bottom-right
    ];

    // Bar thickness as fraction of triangle size
    let th = 0.28;

    // Inner triangle vertices (offset inward)
    let inner = [
        (0.0, -1.0 + th * s3),
        (-s3 / 2.0 + th * 1.5, 0.5),
        (s3 / 2.0 - th * 1.5, 0.5),
    ];

    // Transform helper: scale and rotate a point, then translate to center
    let xform = |p: (f32, f32)| -> (f32, f32) {
        let (sin_a, cos_a) = angle.sin_cos();
        let rx = p.0 * cos_a - p.1 * sin_a;
        let ry = p.0 * sin_a + p.1 * cos_a;
        (cx + rx * scale, cy + ry * scale)
    };

    // Each beam connects outer[i]→outer[j] along the outer edge,
    // with a parallel inner edge from inner[i]→inner[j].
    // The "impossible" trick: beam drawing order and overlap at corners.

    // Color palettes for 3 faces of each beam:
    // face_a = main visible face, face_b = side face, face_c = end cap
    let faces_a = [
        C::from_rgba8(55, 130, 210, 255),   // blue
        C::from_rgba8(210, 75, 55, 255),    // red
        C::from_rgba8(55, 185, 110, 255),   // green
    ];
    let faces_b = [
        C::from_rgba8(35, 85, 150, 255),
        C::from_rgba8(155, 45, 35, 255),
        C::from_rgba8(30, 130, 70, 255),
    ];
    let faces_c = [
        C::from_rgba8(75, 160, 240, 255),
        C::from_rgba8(240, 110, 80, 255),
        C::from_rgba8(80, 215, 140, 255),
    ];

    let edge_color = C::from_rgba8(10, 10, 18, 255);
    let edge_w = 1.8;

    // The three beams, each defined by 4 corners:
    // outer_start, outer_end, inner_end, inner_start
    // We draw them in a specific order to create the cyclic overlap.

    // Beam 0: Top → Bottom-Left (outer[0]→outer[1], inner[0]→inner[1])
    // Beam 1: Bottom-Left → Bottom-Right (outer[1]→outer[2], inner[1]→inner[2])
    // Beam 2: Bottom-Right → Top (outer[2]→outer[0], inner[2]→inner[0])
    //
    // Draw ORDER for impossibility: 0, 1, then 2.
    // Beam 2's "near" end (at top) overlaps beam 0's start point.
    // But beam 0 was already drawn as "in front" going left...
    // That's the contradiction.

    // For a convincing Penrose, each beam needs proper 3D-like shading.
    // We'll draw each beam as a trapezoid (the main face) plus a
    // parallelogram (the side face visible due to perspective).

    // The thickness offset creates the 3D depth effect

    // Beam drawing function
    let draw_beam = |mut cv: PixelCanvas,
                     o_start: (f32, f32), o_end: (f32, f32),
                     i_start: (f32, f32), i_end: (f32, f32),
                     depth_dir: (f32, f32),
                     face_main: C, face_side: C, face_top: C| -> PixelCanvas {
        let os = xform(o_start);
        let oe = xform(o_end);
        let is_ = xform(i_start);
        let ie = xform(i_end);
        let dd = (depth_dir.0 * scale * 0.12, depth_dir.1 * scale * 0.12);

        // Main face (the big visible surface)
        cv = cv
            .polygon(vec![os, oe, ie, is_])
            .fill(face_main)
            .stroke(edge_color, edge_w)
            .done();

        // Side face (depth illusion — a parallelogram offset along depth direction)
        cv = cv
            .polygon(vec![
                os,
                (os.0 + dd.0, os.1 + dd.1),
                (oe.0 + dd.0, oe.1 + dd.1),
                oe,
            ])
            .fill(face_side)
            .stroke(edge_color, edge_w * 0.8)
            .done();

        // Top face connecting to the depth offset
        cv = cv
            .polygon(vec![
                is_,
                (is_.0 + dd.0, is_.1 + dd.1),
                (os.0 + dd.0, os.1 + dd.1),
                os,
            ])
            .fill(face_top)
            .stroke(edge_color, edge_w * 0.8)
            .done();

        cv
    };

    // Beam 0: Top → Bottom-Left
    canvas = draw_beam(
        canvas,
        outer[0], outer[1], inner[0], inner[1],
        (th * 0.5, th * 0.3),
        faces_a[0], faces_b[0], faces_c[0],
    );

    // Beam 1: Bottom-Left → Bottom-Right
    canvas = draw_beam(
        canvas,
        outer[1], outer[2], inner[1], inner[2],
        (0.0, -th * 0.6),
        faces_a[1], faces_b[1], faces_c[1],
    );

    // Beam 2: Bottom-Right → Top
    // THIS is the impossible beam — it connects back to the top,
    // overlaying beam 0's starting area, creating the contradiction.
    canvas = draw_beam(
        canvas,
        outer[2], outer[0], inner[2], inner[0],
        (-th * 0.5, th * 0.3),
        faces_a[2], faces_b[2], faces_c[2],
    );

    // Subtle pulsing glow ring
    let glow_a = ((t * 0.8).sin() * 0.25 + 0.25).max(0.0);
    canvas = canvas
        .circle(cx, cy, scale * 1.15)
        .stroke(C::from_rgba8(100, 140, 255, (glow_a * 255.0) as u8), 1.5)
        .done();

    // Corner accent dots — highlight the three vertices
    for v in &outer {
        let p = xform(*v);
        canvas = canvas
            .circle(p.0, p.1, 4.0)
            .fill(C::from_rgba8(255, 255, 255, 120))
            .done();
    }

    canvas
}
