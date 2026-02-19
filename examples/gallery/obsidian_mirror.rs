//! **Obsidian Mirror** — esoteric scrying animation.
//!
//! A black obsidian scrying mirror where the reflection reveals more than
//! reality. A tracer point draws Metatron's Cube into existence with fading
//! trails; the reflected world uses Oklab complementary colors and gains
//! extra geometry not present in the real world — the mirror sees deeper truth.
//!
//! Controls:
//!   `Space` — pause/resume
//!   `c`     — cycle palette (Obsidian → Ethereal → Void)
//!   `q`     — quit
//!
//! Run with: `cargo run --example obsidian_mirror --release`

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::similar_names,
    clippy::too_many_arguments
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

const CYCLE_DURATION: f32 = 24.0;

#[derive(Clone, Copy)]
enum Palette {
    Obsidian,
    Ethereal,
    Void,
}

impl Palette {
    const fn label(self) -> &'static str {
        match self {
            Self::Obsidian => "Obsidian",
            Self::Ethereal => "Ethereal",
            Self::Void => "Void",
        }
    }

    const fn next(self) -> Self {
        match self {
            Self::Obsidian => Self::Ethereal,
            Self::Ethereal => Self::Void,
            Self::Void => Self::Obsidian,
        }
    }
}

struct MirrorState {
    palette: Palette,
    paused: bool,
}

impl MirrorState {
    const fn new() -> Self {
        Self {
            palette: Palette::Obsidian,
            paused: false,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Color helpers
// ═══════════════════════════════════════════════════════════════════

/// Negate a and b channels in Oklab, preserving lightness.
fn oklab_complement(color: C) -> C {
    let (l, a, b) = color.to_oklab();
    C::from_oklab(l, -a, -b, color.a)
}

fn background_color(palette: Palette) -> C {
    match palette {
        Palette::Obsidian => C::from_rgba8(4, 2, 8, 255),
        Palette::Ethereal => C::from_rgba8(2, 6, 10, 255),
        Palette::Void => C::from_rgba8(1, 1, 1, 255),
    }
}

fn real_world_color(palette: Palette, depth: f32, time: f32) -> C {
    match palette {
        Palette::Obsidian => {
            let hue = (time * 8.0).sin().mul_add(5.0, depth.mul_add(20.0, 42.0));
            let sat = 0.15f32.mul_add(depth.mul_add(2.0, time).sin().abs(), 0.65);
            let light = 0.1f32.mul_add(depth.mul_add(1.5, time * 0.5).cos(), 0.5);
            C::from_hsla(hue, sat, light, 1.0)
        }
        Palette::Ethereal => {
            let hue = depth.mul_add(30.0, time.mul_add(12.0, 170.0)) % 360.0;
            C::from_hsla(hue, 0.6, 0.55, 1.0)
        }
        Palette::Void => {
            let v = 0.3f32.mul_add(depth.mul_add(2.0, time * 0.3).cos().abs(), 0.35);
            C::from_rgba(v, v, v * 1.05, 1.0)
        }
    }
}

fn mirror_world_color(palette: Palette, depth: f32, time: f32) -> C {
    oklab_complement(real_world_color(palette, depth, time))
}

fn tracer_color(palette: Palette, time: f32) -> C {
    match palette {
        Palette::Obsidian => C::from_hsla((time * 15.0).sin().mul_add(10.0, 50.0), 1.0, 0.75, 1.0),
        Palette::Ethereal => C::from_hsla((time * 20.0) % 360.0, 0.9, 0.8, 1.0),
        Palette::Void => C::from_rgba(0.95, 0.95, 1.0, 1.0),
    }
}

fn mirror_surface_color(palette: Palette, t: f32) -> C {
    match palette {
        Palette::Obsidian => {
            let hue = t.mul_add(5.0, 280.0) % 360.0;
            C::from_hsla(hue, 0.4, 0.08, 1.0)
        }
        Palette::Ethereal => {
            let hue = t.mul_add(3.0, 200.0) % 360.0;
            C::from_hsla(hue, 0.3, 0.06, 1.0)
        }
        Palette::Void => C::from_rgba(0.03, 0.03, 0.04, 1.0),
    }
}

// ═══════════════════════════════════════════════════════════════════
// Metatron's Cube geometry
// ═══════════════════════════════════════════════════════════════════

struct MetatronGeometry {
    /// Line segments: (x1, y1, x2, y2, birth_progress)
    segments: Vec<(f32, f32, f32, f32, f32)>,
    /// Circles: (cx, cy, radius, birth_progress)
    circles: Vec<(f32, f32, f32, f32)>,
    /// Current tracer position
    tracer_pos: (f32, f32),
}

fn metatron_geometry(cx: f32, cy: f32, radius: f32, progress: f32) -> MetatronGeometry {
    let r_inner = radius * 0.3;
    let r_outer = radius * 0.6;
    let circle_r = radius * 0.08;

    // 13 nodes: 1 center + 6 inner + 6 outer
    let mut nodes: Vec<(f32, f32)> = Vec::with_capacity(13);
    nodes.push((cx, cy));

    for i in 0..6 {
        let angle = i as f32 * FRAC_PI_3;
        nodes.push((cx + r_inner * angle.cos(), cy + r_inner * angle.sin()));
    }
    for i in 0..6 {
        let angle = (i as f32).mul_add(FRAC_PI_3, FRAC_PI_3 / 2.0);
        nodes.push((cx + r_outer * angle.cos(), cy + r_outer * angle.sin()));
    }

    // Build all connecting segments
    let mut all_segments: Vec<(f32, f32, f32, f32)> = Vec::new();
    for i in 0..nodes.len() {
        for j in (i + 1)..nodes.len() {
            all_segments.push((nodes[i].0, nodes[i].1, nodes[j].0, nodes[j].1));
        }
    }

    let total = all_segments.len();
    let segments_revealed = (progress * total as f32) as usize;

    let segments: Vec<(f32, f32, f32, f32, f32)> = all_segments
        .iter()
        .enumerate()
        .filter(|(i, _)| *i < segments_revealed)
        .map(|(i, &(x1, y1, x2, y2))| {
            let birth = i as f32 / total as f32;
            (x1, y1, x2, y2, birth)
        })
        .collect();

    // Circles appear as the tracer reaches each node
    let circles: Vec<(f32, f32, f32, f32)> = nodes
        .iter()
        .enumerate()
        .filter(|(i, _)| (*i as f32 / nodes.len() as f32) < progress)
        .map(|(i, &(x, y))| {
            let birth = i as f32 / nodes.len() as f32;
            (x, y, circle_r, birth)
        })
        .collect();

    // Tracer follows the current segment being drawn
    let tracer_pos = if segments_revealed < total {
        let seg = &all_segments[segments_revealed];
        let frac = (progress * total as f32).fract();
        let tx = (seg.2 - seg.0).mul_add(frac, seg.0);
        let ty = (seg.3 - seg.1).mul_add(frac, seg.1);
        (tx, ty)
    } else {
        nodes[0]
    };

    MetatronGeometry {
        segments,
        circles,
        tracer_pos,
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

    let mut state = MirrorState::new();
    let start = Instant::now();
    let mut frozen_time = 0.0_f32;

    run_loop_continuous(
        960,
        640,
        "Obsidian Mirror",
        true,
        move |backend, keys, (w, h)| {
            for key in keys {
                if !key.pressed {
                    continue;
                }
                match key.code {
                    WKey::Escape | WKey::KeyQ => return LoopAction::Exit,
                    WKey::KeyC => state.palette = state.palette.next(),
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

    let mut state = MirrorState::new();
    let start = Instant::now();
    let mut last_frame = Instant::now();
    let mut frozen_time = 0.0_f32;

    loop {
        let now = Instant::now();
        let _dt = now.duration_since(last_frame).as_secs_f32();
        last_frame = now;

        let elapsed = if state.paused {
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
            let canvas = build_scene(w, h, &state, elapsed);

            frame.render_stateful_widget(
                PixelCanvasWidget::new(canvas).skip_cache().z_index(-1),
                area,
                &mut px_state,
            );

            let cycle_t = elapsed % CYCLE_DURATION;
            let phase_name = if cycle_t < 3.0 {
                "Materialize"
            } else if cycle_t < 12.0 {
                "Trace"
            } else if cycle_t < 18.0 {
                "Reveal"
            } else {
                "Converge"
            };

            let status_text = format!(
                " {} │ {} │ {:.1}s │ [c] palette [space] pause [q] quit",
                state.palette.label(),
                phase_name,
                cycle_t,
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
                        KeyCode::Char('c') => state.palette = state.palette.next(),
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

// ═══════════════════════════════════════════════════════════════════
// Scene builder — phase dispatch
// ═══════════════════════════════════════════════════════════════════

fn build_scene(w: u32, h: u32, state: &MirrorState, time: f32) -> PixelCanvas {
    if w == 0 || h == 0 {
        return PixelCanvas::new(1, 1);
    }

    let mut canvas = PixelCanvas::new(w, h).background(background_color(state.palette));

    let cycle_t = time % CYCLE_DURATION;
    let wf = w as f32;
    let hf = h as f32;
    let cx = wf / 2.0;
    // Mirror line at 55% of canvas height
    let mirror_y = hf * 0.55;

    // Phase progression
    let materialize = (cycle_t / 3.0).clamp(0.0, 1.0);
    let trace_progress = ((cycle_t - 3.0) / 9.0).clamp(0.0, 1.0);
    let reveal = ((cycle_t - 12.0) / 6.0).clamp(0.0, 1.0);
    let converge = ((cycle_t - 18.0) / 6.0).clamp(0.0, 1.0);

    // Phase 1: Mirror surface (always drawn once materialized)
    if materialize > 0.0 {
        canvas = draw_mirror_surface(
            canvas,
            cx,
            mirror_y,
            wf,
            hf,
            materialize,
            time,
            state.palette,
        );
    }

    // Phase 2: Tracer geometry (real + reflected)
    if trace_progress > 0.0 {
        canvas = draw_tracer_geometry(
            canvas,
            cx,
            mirror_y,
            wf,
            hf,
            trace_progress,
            time,
            state.palette,
            reveal,
        );
    }

    // Phase 3: Mirror extras (reflection-only geometry)
    if reveal > 0.0 {
        canvas = draw_mirror_reveals(canvas, cx, mirror_y, wf, hf, reveal, time, state.palette);
    }

    // Phase 4: Convergence radiance
    if converge > 0.0 {
        canvas = draw_convergence(canvas, cx, mirror_y, wf, hf, converge, time, state.palette);
    }

    canvas
}

// ═══════════════════════════════════════════════════════════════════
// Phase 1: Mirror surface
// ═══════════════════════════════════════════════════════════════════

fn draw_mirror_surface(
    mut canvas: PixelCanvas,
    cx: f32,
    mirror_y: f32,
    w: f32,
    _h: f32,
    materialize: f32,
    time: f32,
    palette: Palette,
) -> PixelCanvas {
    let rx = w * 0.42;
    let ry = rx * 0.12;

    // Obsidian gradient — concentric filled ellipses from outer to inner
    let layers = 8;
    for i in 0..layers {
        let t = i as f32 / layers as f32;
        let scale = 1.0 - t * 0.6;
        let erx = rx * scale;
        let ery = ry * scale;

        let base = mirror_surface_color(palette, time);
        let (l, a, b) = base.to_oklab();
        let lighter = C::from_oklab(
            l + t * 0.04,
            a * (1.0 - t * 0.3),
            b * (1.0 - t * 0.3),
            materialize * 0.7,
        );

        canvas = canvas.ellipse(cx, mirror_y, erx, ery).fill(lighter).done();
    }

    // Animated ripple rings
    let ripple_count = 4;
    for i in 0..ripple_count {
        let phase = (time * 0.8 + i as f32 * 1.5).sin() * 0.5 + 0.5;
        let ripple_scale = 0.5f32.mul_add(phase, 0.6);
        let erx = rx * ripple_scale;
        let ery = ry * ripple_scale;
        let alpha = materialize * 0.15 * (1.0 - phase.abs());

        let ripple_c = real_world_color(palette, phase, time).with_alpha(alpha);
        canvas = canvas
            .ellipse(cx, mirror_y, erx, ery)
            .stroke(ripple_c, 1.0)
            .done();
    }

    // Mirror rim highlight
    let rim = real_world_color(palette, 0.3, time).with_alpha(materialize * 0.4);
    canvas = canvas
        .ellipse(cx, mirror_y, rx * 1.02, ry * 1.02)
        .stroke(rim, 1.5)
        .done();

    canvas
}

// ═══════════════════════════════════════════════════════════════════
// Phase 2: Tracer geometry (real world + reflection)
// ═══════════════════════════════════════════════════════════════════

fn draw_tracer_geometry(
    mut canvas: PixelCanvas,
    cx: f32,
    mirror_y: f32,
    _w: f32,
    h: f32,
    progress: f32,
    time: f32,
    palette: Palette,
    dim_factor: f32,
) -> PixelCanvas {
    // Real world center: above mirror
    let real_cy = mirror_y * 0.45;
    let radius = mirror_y * 0.35;

    let geo = metatron_geometry(cx, real_cy, radius, progress);

    // Real-world dimming when reveal phase is active
    let real_alpha_mult = 1.0 - dim_factor * 0.6;

    // Draw real-world segments with trails
    for &(x1, y1, x2, y2, birth) in &geo.segments {
        let age = progress - birth;
        let alpha = (1.0 - age / 4.0).clamp(0.1, 1.0) * real_alpha_mult;
        let depth = birth;
        let color = real_world_color(palette, depth, time).with_alpha(alpha * 0.6);

        canvas = canvas.line(x1, y1, x2, y2).color(color).width(1.0).done();
    }

    // Draw real-world circles
    for &(x, y, r, birth) in &geo.circles {
        let age = progress - birth;
        let alpha = (1.0 - age / 4.0).clamp(0.2, 1.0) * real_alpha_mult;
        let depth = birth;
        let stroke_c = real_world_color(palette, depth, time).with_alpha(alpha * 0.8);
        let fill_c = real_world_color(palette, depth + 0.3, time).with_alpha(alpha * 0.08);

        canvas = canvas
            .circle(x, y, r)
            .fill(fill_c)
            .stroke(stroke_c, 1.2)
            .done();
    }

    // Tracer tip (real world) — concentric glow
    if progress < 1.0 {
        let (tx, ty) = geo.tracer_pos;
        let tc = tracer_color(palette, time);
        for i in 0..4 {
            let gr = (i as f32).mul_add(3.0, 2.0);
            let ga = 0.7 / (i as f32 + 1.0);
            canvas = canvas.circle(tx, ty, gr).fill(tc.with_alpha(ga)).done();
        }
    }

    // ─── Reflected world (below mirror) ───
    // y-flip: reflected_y = 2 * mirror_y - y
    let reflect_y = |y: f32| -> f32 { 2.0 * mirror_y - y };

    // Only draw reflected geometry that fits within canvas
    for &(x1, y1, x2, y2, birth) in &geo.segments {
        let ry1 = reflect_y(y1);
        let ry2 = reflect_y(y2);
        if ry1 > h && ry2 > h {
            continue;
        }

        let age = progress - birth;
        let alpha = (1.0 - age / 4.0).clamp(0.1, 1.0);
        let depth = birth;
        let color = mirror_world_color(palette, depth, time).with_alpha(alpha * 0.55);

        canvas = canvas.line(x1, ry1, x2, ry2).color(color).width(1.0).done();
    }

    for &(x, y, r, birth) in &geo.circles {
        let ry = reflect_y(y);
        if ry - r > h {
            continue;
        }

        let age = progress - birth;
        let alpha = (1.0 - age / 4.0).clamp(0.2, 1.0);
        let depth = birth;
        let stroke_c = mirror_world_color(palette, depth, time).with_alpha(alpha * 0.7);
        let fill_c = mirror_world_color(palette, depth + 0.3, time).with_alpha(alpha * 0.06);

        canvas = canvas
            .circle(x, ry, r)
            .fill(fill_c)
            .stroke(stroke_c, 1.2)
            .done();
    }

    // Reflected tracer tip
    if progress < 1.0 {
        let (tx, ty) = geo.tracer_pos;
        let rty = reflect_y(ty);
        let tc = oklab_complement(tracer_color(palette, time));
        for i in 0..4 {
            let gr = (i as f32).mul_add(3.0, 2.0);
            let ga = 0.6 / (i as f32 + 1.0);
            canvas = canvas.circle(tx, rty, gr).fill(tc.with_alpha(ga)).done();
        }
    }

    canvas
}

// ═══════════════════════════════════════════════════════════════════
// Phase 3: Mirror reveals — extra reflection-only geometry
// ═══════════════════════════════════════════════════════════════════

fn draw_mirror_reveals(
    mut canvas: PixelCanvas,
    cx: f32,
    mirror_y: f32,
    _w: f32,
    h: f32,
    reveal: f32,
    time: f32,
    palette: Palette,
) -> PixelCanvas {
    let reflect_y = |y: f32| -> f32 { 2.0 * mirror_y - y };

    // These exist only in the reflection zone (below mirror)
    let real_cy = mirror_y * 0.45;
    let radius = mirror_y * 0.35;

    // 6 intermediate circles between inner and outer rings
    let r_mid = radius * 0.45;
    for i in 0..6 {
        let angle = (i as f32).mul_add(FRAC_PI_3, time * 0.1);
        let x = cx + r_mid * angle.cos();
        let y = real_cy + r_mid * angle.sin();
        let ry = reflect_y(y);

        if ry > h {
            continue;
        }

        let circle_r = radius * 0.06;
        let depth = i as f32 / 6.0;
        let alpha = reveal * (0.2f32.mul_add((time * 2.0 + i as f32).sin(), 0.7));
        let color = mirror_world_color(palette, depth + 0.5, time).with_alpha(alpha * 0.7);
        let fill_c = mirror_world_color(palette, depth, time).with_alpha(alpha * 0.12);

        canvas = canvas
            .circle(x, ry, circle_r)
            .fill(fill_c)
            .stroke(color, 1.5)
            .done();
    }

    // Connecting arcs between intermediate circles
    for i in 0..6 {
        let a1 = (i as f32).mul_add(FRAC_PI_3, time * 0.1);
        let a2 = ((i + 1) as f32).mul_add(FRAC_PI_3, time * 0.1);
        let x1 = cx + r_mid * a1.cos();
        let y1 = reflect_y(real_cy + r_mid * a1.sin());
        let x2 = cx + r_mid * a2.cos();
        let y2 = reflect_y(real_cy + r_mid * a2.sin());

        if y1 > h && y2 > h {
            continue;
        }

        let depth = i as f32 / 6.0;
        let alpha = reveal * 0.4;
        let color = mirror_world_color(palette, depth + 0.2, time).with_alpha(alpha);

        canvas = canvas.line(x1, y1, x2, y2).color(color).width(0.8).done();
    }

    // Glowing nodes at intersections
    let node_count = 12;
    for i in 0..node_count {
        let angle = (i as f32).mul_add(TAU / node_count as f32, time * 0.15);
        let dist = radius * if i % 2 == 0 { 0.22 } else { 0.52 };
        let x = cx + dist * angle.cos();
        let y = real_cy + dist * angle.sin();
        let ry = reflect_y(y);

        if ry > h {
            continue;
        }

        let pulse = (time * 3.0 + i as f32 * 0.8).sin().mul_add(0.5, 0.5);
        let glow_r = 2.0f32.mul_add(pulse, 3.0);
        let alpha = reveal * pulse * 0.5;
        let color =
            mirror_world_color(palette, i as f32 / node_count as f32, time).with_alpha(alpha);

        // Outer glow
        canvas = canvas
            .circle(x, ry, glow_r * 2.5)
            .fill(color.with_alpha(alpha * 0.2))
            .done();

        // Core
        canvas = canvas.circle(x, ry, glow_r).fill(color).done();
    }

    canvas
}

// ═══════════════════════════════════════════════════════════════════
// Phase 4: Convergence — radiance rays + pulsing
// ═══════════════════════════════════════════════════════════════════

fn draw_convergence(
    mut canvas: PixelCanvas,
    cx: f32,
    mirror_y: f32,
    w: f32,
    _h: f32,
    converge: f32,
    time: f32,
    palette: Palette,
) -> PixelCanvas {
    let radius = w.min(mirror_y * 2.0) * 0.4;

    // Radiance rays emanating from mirror surface
    let ray_count = 16;
    for i in 0..ray_count {
        let base_angle = i as f32 * TAU / ray_count as f32;
        let angle = base_angle + time * 0.12;
        let pulse = (time * 2.0 + i as f32 * 0.5).sin().mul_add(0.3, 0.7);
        let len = radius * 0.7 * pulse;

        let x2 = cx + len * angle.cos();
        let y2 = mirror_y + len * angle.sin();

        // Alternate between real and mirror colors
        let color = if i % 2 == 0 {
            real_world_color(palette, i as f32 / ray_count as f32, time)
        } else {
            mirror_world_color(palette, i as f32 / ray_count as f32, time)
        };

        let alpha = converge * 0.15 * pulse;
        canvas = canvas
            .line(cx, mirror_y, x2, y2)
            .color(color.with_alpha(alpha))
            .width(1.5)
            .done();
    }

    // Pulsing glow at mirror center
    let pulse = (time * 2.5).sin().mul_add(0.5, 0.5);
    let glow_r = radius * 0.15 * (0.3f32.mul_add(pulse, 0.7));
    let glow_c = real_world_color(palette, 0.5, time).with_alpha(converge * 0.12 * pulse);
    canvas = canvas.circle(cx, mirror_y, glow_r).fill(glow_c).done();

    // Mirror complement glow
    let comp_c = mirror_world_color(palette, 0.5, time).with_alpha(converge * 0.08 * pulse);
    canvas = canvas
        .circle(cx, mirror_y, glow_r * 1.3)
        .fill(comp_c)
        .done();

    // Slow rotating outer ring
    let ring_alpha = converge * 0.25;
    let rot = time * 0.08;
    let ring_points: Vec<(f32, f32)> = (0..6)
        .map(|i| {
            let angle = (i as f32).mul_add(FRAC_PI_3, rot);
            (
                (radius * 0.5).mul_add(angle.cos(), cx),
                (radius * 0.2).mul_add(angle.sin(), mirror_y),
            )
        })
        .collect();
    let ring_c = real_world_color(palette, 0.8, time).with_alpha(ring_alpha);
    canvas = canvas.polygon(ring_points).stroke(ring_c, 1.0).done();

    canvas
}
