//! **Power Test** — all-features animated stress test.
//!
//! Six concurrent animated panels in a 3×2 grid exercising every API surface
//! of `scry-engine` simultaneously. Also serves as a performance
//! benchmark — the FPS counter and timing breakdown are shown in the status
//! bar.
//!
//! **Features exercised:**
//! - All drawing primitives: circle, rect, ellipse, line, polyline, polygon,
//!   arc, path (Bézier), gradient, group
//! - All style options: fill, stroke, corner_radius, rotation, LineCap,
//!   LineJoin, DashPattern, anti_alias, fill_linear_gradient,
//!   fill_radial_gradient
//! - Animation system: every Easing curve, Keyframes<T>, AnimationState,
//!   Lerp (f32, Color, Point)
//! - Advanced compositing: BlendMode (Multiply, Screen, Overlay),
//!   group opacity, clip_rect, nested transforms
//! - Efficiency: entire scene rebuilt every frame to stress the builder,
//!   rasterizer, content hashing, and protocol transport
//!
//! Run with: `cargo run --example powertest --features widget`

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
    clippy::needless_range_loop,
    clippy::doc_markdown
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

use scry_engine::prelude::{
    AnimationState, Easing, Keyframe, Keyframes, Picker, PixelCanvasState, PixelCanvasWidget,
    ProfileHistory, ProfiledRasterizer, ProtocolKind,
};

use scry_engine::scene::command::DrawCommand;
use scry_engine::scene::style::{
    BlendMode, Color as C, DashPattern, FillStyle, GradientDef, GradientKind, GradientStop,
    LineCap, LineJoin, Point, Rect as PxRect, ShapeStyle, StrokeStyle, Transform,
};
use scry_engine::scene::PixelCanvas;
use scry_engine::transport;

// ───────────────────────────────────────────────────────────────────
// Grid helper
// ───────────────────────────────────────────────────────────────────

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

// ───────────────────────────────────────────────────────────────────
// Main
// ───────────────────────────────────────────────────────────────────

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

    let start = Instant::now();
    let mut last_frame = start;
    let mut frame_times: Vec<Duration> = Vec::with_capacity(60);
    let mut fps: f64 = 0.0;
    let mut draw_ms: f64 = 0.0;
    let mut flush_ms: f64 = 0.0;
    #[allow(unused_assignments)]
    let mut cmd_count: usize = 0;
    let mut profiling = false;
    let mut profile_history = ProfileHistory::default();
    let mut last_profile_str = String::new();

    let mut anim = AnimationState::new();
    setup_orchestrator(&mut anim);

    loop {
        let now = Instant::now();
        let dt = now - last_frame;
        last_frame = now;
        let elapsed = now.duration_since(start);
        let t = elapsed.as_secs_f32();

        frame_times.push(dt);
        if frame_times.len() > 60 {
            frame_times.remove(0);
        }
        if !frame_times.is_empty() {
            let total: Duration = frame_times.iter().sum();
            fps = frame_times.len() as f64 / total.as_secs_f64();
        }

        anim.tick(dt);
        if anim.is_idle() {
            setup_orchestrator(&mut anim);
        }

        let draw_start = Instant::now();

        // Build scene outside terminal.draw() so we can profile it
        let term_size = terminal.size()?;
        let term_rect = Rect::new(0, 0, term_size.width, term_size.height);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(3)])
            .split(term_rect);
        let canvas_area = chunks[0];
        let status_area = chunks[1];

        let scene_start = Instant::now();
        let canvas = build_scene(canvas_area, &state, t, &anim);
        let scene_build_us = scene_start.elapsed().as_micros() as u64;
        cmd_count = canvas.command_count();

        // When profiling, rasterize manually with timing, then hand pre-rasterized
        // data to the widget. When not profiling, use the normal widget path.
        if profiling {
            // Profiled rasterization
            let (pixmap_entry, gc) = state
                .cache_mut()
                .get_or_insert_with_grad_cache(canvas.width(), canvas.height());
            if let Some(pixmap) = pixmap_entry {
                let rp = ProfiledRasterizer::rasterize_into_profiled_cached(&canvas, pixmap, gc);
                profile_history.push(rp);
                let smoothed = profile_history.summary();
                last_profile_str = format!(
                    " \u{1F50D} {fps:.0}fps \u{2502} bld={:.1}m rast={smoothed} flsh={flush_ms:.1}m \u{2502} {cmd_count}cmd f{} \u{2502} p:off q:quit",
                    scene_build_us as f64 / 1000.0,
                    smoothed.frame_count,
                );
            }
            // Use a unique frame seq for cache
            use std::sync::atomic::{AtomicU64, Ordering};
            static FRAME_SEQ: AtomicU64 = AtomicU64::new(1);
            state
                .cache_mut()
                .mark_valid(FRAME_SEQ.fetch_add(1, Ordering::Relaxed));
        }

        // Capture status strings for the closure
        let profile_line = last_profile_str.clone();
        let is_profiling = profiling;

        terminal.draw(|frame| {
            if is_profiling {
                // Widget still needs to set up the pending frame for flush().
                // We already rasterized into the cache, so the widget will find
                // a valid pixmap and skip re-rasterization.
                frame.render_stateful_widget(
                    PixelCanvasWidget::new(canvas).z_index(-1).skip_cache(),
                    canvas_area,
                    &mut state,
                );
            } else {
                frame.render_stateful_widget(
                    PixelCanvasWidget::new(canvas).z_index(-1).skip_cache(),
                    canvas_area,
                    &mut state,
                );
            }

            let status_text = if is_profiling {
                profile_line.clone()
            } else {
                format!(
                    " \u{26A1} POWER TEST \u{2502} {fps:.0} fps \u{2502} draw {draw_ms:.1}ms \u{2502} flush {flush_ms:.1}ms \u{2502} {cmd_count} cmds \u{2502} 'p' profile 'q' quit",
                )
            };
            let status = Paragraph::new(status_text)
                .block(Block::default().borders(Borders::TOP));
            frame.render_widget(status, status_area);
        })?;
        let draw_elapsed = draw_start.elapsed();

        let flush_start = Instant::now();
        state.flush()?;
        let flush_elapsed = flush_start.elapsed();

        draw_ms = draw_elapsed.as_secs_f64() * 1000.0;
        flush_ms = flush_elapsed.as_secs_f64() * 1000.0;

        if event::poll(Duration::from_millis(1))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Char('r') => setup_orchestrator(&mut anim),
                        KeyCode::Char('p') => profiling = !profiling,
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

// ═══════════════════════════════════════════════════════════════════
// Scene builder — 6 panels
// ═══════════════════════════════════════════════════════════════════

fn build_scene(
    area: Rect,
    pxstate: &PixelCanvasState,
    t: f32,
    anim: &AnimationState,
) -> PixelCanvas {
    let font = pxstate.font_size();
    let w = u32::from(area.width) * u32::from(font.width);
    let h = u32::from(area.height) * u32::from(font.height);
    let wf = w as f32;
    let hf = h as f32;

    let mut canvas = PixelCanvas::new(w, h).background(C::from_rgba8(8, 8, 16, 255));

    // Panel dividers
    let div = C::from_rgba8(30, 35, 50, 255);
    canvas = canvas
        .line(wf / 3.0, 0.0, wf / 3.0, hf)
        .color(div)
        .width(1.0)
        .done()
        .line(wf * 2.0 / 3.0, 0.0, wf * 2.0 / 3.0, hf)
        .color(div)
        .width(1.0)
        .done()
        .line(0.0, hf / 2.0, wf, hf / 2.0)
        .color(div)
        .width(1.0)
        .done();

    canvas = panel_particle_storm(canvas, wf, hf, t);
    canvas = panel_gradient_aurora(canvas, wf, hf, t);
    canvas = panel_transform_carousel(canvas, wf, hf, t);
    canvas = panel_precision_strokes(canvas, wf, hf, t);
    canvas = panel_clipped_kaleidoscope(canvas, wf, hf, t);
    canvas = panel_orchestrated_scene(canvas, wf, hf, anim);

    canvas
}

// ─── Panel 1: Particle Storm ─────────────────────────────────────

fn panel_particle_storm(mut canvas: PixelCanvas, wf: f32, hf: f32, t: f32) -> PixelCanvas {
    let c = cell(0, 0, wf, hf);
    let pad = 8.0;

    let easings = [
        Easing::Linear,
        Easing::EaseInQuad,
        Easing::EaseOutCubic,
        Easing::EaseInOutQuart,
        Easing::EaseInSine,
        Easing::EaseOutExpo,
        Easing::EaseInOutCirc,
        Easing::BACK,
        Easing::Bounce,
        Easing::Elastic,
        Easing::CSS_EASE,
        Easing::EaseOutQuad,
        Easing::EaseInCubic,
        Easing::EaseInOutSine,
        Easing::EaseOutCirc,
    ];

    let ring_r = (c.w.min(c.h) * 0.36) - pad;
    let n = easings.len();
    let period = 3.0_f32;
    let raw_t = (t % (period * 2.0)) / period;
    let linear_t = if raw_t > 1.0 { 2.0 - raw_t } else { raw_t };

    for (i, easing) in easings.iter().enumerate() {
        let base_angle = std::f32::consts::TAU * i as f32 / n as f32;
        let eased = easing.ease(linear_t);

        let r = ring_r * (0.3 + 0.7 * eased);
        let angle = base_angle + t * 0.5 + eased * std::f32::consts::PI;
        let px = c.cx + r * angle.cos();
        let py = c.cy + r * angle.sin();

        let hue = (360.0 * i as f32 / n as f32 + t * 40.0) % 360.0;
        let color = C::from_hsla(hue, 0.9, 0.6, 0.5 + 0.5 * eased);
        let size = 3.0 + 6.0 * eased;

        canvas = canvas
            .circle(px, py, size * 2.0)
            .fill(color.with_alpha(0.08))
            .done()
            .circle(px, py, size)
            .fill(color)
            .done();
    }

    // Inner pulsing keyframed circle
    let size_kf = Keyframes::new(vec![
        Keyframe {
            position: 0.0,
            value: 8.0_f32,
            easing: Easing::EaseInOutSine,
        },
        Keyframe {
            position: 0.5,
            value: 20.0,
            easing: Easing::EaseInOutSine,
        },
        Keyframe {
            position: 1.0,
            value: 8.0,
            easing: Easing::Linear,
        },
    ]);
    let color_kf = Keyframes::new(vec![
        Keyframe {
            position: 0.0,
            value: C::from_hsl(0.0, 0.9, 0.6),
            easing: Easing::Linear,
        },
        Keyframe {
            position: 0.5,
            value: C::from_hsl(180.0, 0.9, 0.6),
            easing: Easing::Linear,
        },
        Keyframe {
            position: 1.0,
            value: C::from_hsl(360.0, 0.9, 0.6),
            easing: Easing::Linear,
        },
    ]);
    let kf_t = (t % 2.0) / 2.0;
    canvas = canvas
        .circle(c.cx, c.cy, size_kf.value_at(kf_t))
        .fill(color_kf.value_at(kf_t))
        .stroke(C::WHITE.with_alpha(0.5), 1.5)
        .done();

    // Scattered rects with corner_radius
    for i in 0..20 {
        let seed = i as f32 * 7.3;
        let rx = c.x + pad + ((seed * 13.7 + t * 30.0).sin() * 0.5 + 0.5) * (c.w - pad * 3.0);
        let ry = c.y + pad + ((seed * 9.1 + t * 25.0).cos() * 0.5 + 0.5) * (c.h - pad * 3.0);
        let rr = 2.0 + (seed + t).sin().abs() * 4.0;
        let hue = (seed * 50.0 + t * 60.0) % 360.0;
        canvas = canvas
            .rect(rx, ry, rr * 2.0, rr * 2.0)
            .fill(C::from_hsla(hue, 0.7, 0.5, 0.4))
            .corner_radius(rr * 0.4)
            .done();
    }

    draw_label(&mut canvas, c.x + pad, c.y + pad, "PARTICLE STORM");
    canvas
}

// ─── Panel 2: Gradient Aurora ────────────────────────────────────

fn panel_gradient_aurora(mut canvas: PixelCanvas, wf: f32, hf: f32, t: f32) -> PixelCanvas {
    let c = cell(1, 0, wf, hf);
    let pad = 8.0;

    // Stacked linear gradient bands
    let bands = 5;
    let band_h = (c.h - pad * 2.0) / bands as f32;
    for i in 0..bands {
        let y = c.y + pad + i as f32 * band_h;
        let hue_a = (i as f32 * 60.0 + t * 30.0) % 360.0;
        let hue_b = (hue_a + 120.0) % 360.0;
        let ca = C::from_hsl(hue_a, 0.85, 0.5);
        let cb = C::from_hsl(hue_b, 0.85, 0.5);
        let mid = ca.mix(cb, 0.5); // Oklab mix

        canvas = canvas
            .gradient(c.x + pad, y, c.w - pad * 2.0, band_h - 2.0)
            .linear(
                Point::new(c.x + pad, y),
                Point::new(c.x + c.w - pad, y + band_h),
            )
            .stop(0.0, ca)
            .stop(0.5, mid)
            .stop(1.0, cb)
            .done();
    }

    // Radial gradient orb floating over bands
    let orb_r = c.w.min(c.h) * 0.2;
    let pulse = (t * 2.0).sin() * 0.3 + 0.7;
    let orb_x = c.cx + (t * 0.8).sin() * c.w * 0.15;
    let orb_y = c.cy + (t * 1.1).cos() * c.h * 0.1;
    canvas = canvas
        .gradient(orb_x - orb_r, orb_y - orb_r, orb_r * 2.0, orb_r * 2.0)
        .radial(Point::new(orb_x, orb_y), orb_r * pulse)
        .stop(0.0, C::from_rgba8(255, 255, 255, 200))
        .stop(0.4, C::from_hsla((t * 50.0) % 360.0, 0.9, 0.6, 0.8))
        .stop(1.0, C::from_rgba8(0, 0, 0, 0))
        .done();

    // Ellipse with radial gradient fill (fill_radial_gradient API)
    let ell_r = c.w.min(c.h) * 0.12;
    let ell_x = c.x + c.w * 0.2;
    let ell_y = c.y + c.h * 0.8;
    let grad = GradientDef {
        kind: GradientKind::Radial {
            center: Point::new(ell_x, ell_y),
            radius: ell_r,
        },
        stops: vec![
            GradientStop {
                position: 0.0,
                color: C::from_hsla((t * 70.0) % 360.0, 1.0, 0.8, 0.9),
            },
            GradientStop {
                position: 1.0,
                color: C::from_hsla((t * 70.0 + 180.0) % 360.0, 0.8, 0.3, 0.6),
            },
        ],
    };
    canvas = canvas
        .ellipse(ell_x, ell_y, ell_r, ell_r * 0.6)
        .rotation(t * 0.5)
        .fill_radial_gradient(grad)
        .stroke(C::WHITE.with_alpha(0.3), 1.5)
        .done();

    // Rectangle with linear gradient fill
    let rg_w = c.w * 0.25;
    let rg_h = c.h * 0.12;
    let rg_x = c.x + c.w * 0.65;
    let rg_y = c.y + c.h * 0.78;
    let lin_grad = GradientDef {
        kind: GradientKind::Linear {
            start: Point::new(rg_x, rg_y),
            end: Point::new(rg_x + rg_w, rg_y + rg_h),
        },
        stops: vec![
            GradientStop {
                position: 0.0,
                color: C::from_hsl((t * 40.0) % 360.0, 0.9, 0.5),
            },
            GradientStop {
                position: 1.0,
                color: C::from_hsl((t * 40.0 + 120.0) % 360.0, 0.9, 0.5),
            },
        ],
    };
    canvas = canvas
        .rect(rg_x, rg_y, rg_w, rg_h)
        .fill_linear_gradient(lin_grad)
        .corner_radius(6.0)
        .done();

    draw_label(&mut canvas, c.x + pad, c.y + pad, "GRADIENT AURORA");
    canvas
}

// ─── Panel 3: Transform Carousel ────────────────────────────────

fn panel_transform_carousel(mut canvas: PixelCanvas, wf: f32, hf: f32, t: f32) -> PixelCanvas {
    let c = cell(2, 0, wf, hf);
    let pad = 10.0;

    // Outer ring of transformed squares
    let n = 10;
    let orbit_r = c.w.min(c.h) * 0.32 - pad;
    let sq = 16.0;

    for i in 0..n {
        let angle = std::f32::consts::TAU * i as f32 / n as f32 + t * 0.6;
        let sx = c.cx + orbit_r * angle.cos();
        let sy = c.cy + orbit_r * angle.sin();
        let scale = 0.6 + (t * 2.0 + i as f32 * 0.5).sin() * 0.4;

        let transform =
            Transform::rotate_at(angle * 2.5, sx, sy).concat(Transform::scale_xy(scale, scale));

        let hue = (i as f32 * 36.0 + t * 50.0) % 360.0;
        let color = C::from_hsla(hue, 0.85, 0.55, 0.85);

        canvas = canvas
            .group(transform)
            .canvas(move |inner| {
                inner
                    .rect(sx - sq / 2.0, sy - sq / 2.0, sq, sq)
                    .fill(color)
                    .corner_radius(3.0)
                    .stroke(C::WHITE.with_alpha(0.4), 1.0)
                    .done()
            })
            .done();
    }

    // Center: skewed diamond with pulsing group opacity
    let skew_amount = (t * 1.5).sin() * 0.4;
    let opacity = ((t * 1.2).sin() * 0.4 + 0.6).clamp(0.2, 1.0);
    canvas = canvas
        .group(Transform::skew(skew_amount, 0.0))
        .opacity(opacity)
        .canvas(move |inner| {
            inner
                .rect(c.cx - 18.0, c.cy - 18.0, 36.0, 36.0)
                .fill(C::from_rgba8(255, 255, 255, 220))
                .corner_radius(4.0)
                .done()
        })
        .done();

    // Blend mode demo: overlapping circles
    let blend_r = c.w.min(c.h) * 0.15;
    let modes = [BlendMode::Multiply, BlendMode::Screen, BlendMode::Overlay];
    let offsets: [(f32, f32); 3] = [
        (0.0, -blend_r * 0.4),
        (-blend_r * 0.35, blend_r * 0.2),
        (blend_r * 0.35, blend_r * 0.2),
    ];
    let colors = [
        C::from_rgba8(255, 80, 80, 200),
        C::from_rgba8(80, 255, 80, 200),
        C::from_rgba8(80, 80, 255, 200),
    ];

    for i in 0..3 {
        let bx = c.cx + offsets[i].0 + (t * 0.7 + i as f32).sin() * 5.0;
        let by = c.cy + offsets[i].1 + (t * 0.9 + i as f32).cos() * 5.0;
        let mode = modes[i];
        let col = colors[i];
        canvas = canvas
            .group(Transform::identity())
            .blend_mode(mode)
            .canvas(move |inner| inner.circle(bx, by, blend_r * 0.6).fill(col).done())
            .done();
    }

    draw_label(&mut canvas, c.x + pad, c.y + pad, "TRANSFORMS");
    canvas
}

// ─── Panel 4: Precision Strokes ──────────────────────────────────

fn panel_precision_strokes(mut canvas: PixelCanvas, wf: f32, hf: f32, t: f32) -> PixelCanvas {
    let c = cell(0, 1, wf, hf);
    let pad = 10.0;
    let pi = std::f32::consts::PI;

    // Arc fan — different cap styles
    let arc_r_max = c.w.min(c.h) * 0.32 - pad;
    let caps = [LineCap::Butt, LineCap::Round, LineCap::Square];
    for (i, cap) in caps.iter().enumerate() {
        let r = arc_r_max * (0.5 + i as f32 * 0.2);
        let start = t * (1.0 + i as f32 * 0.3);
        let hue = (i as f32 * 90.0 + t * 30.0) % 360.0;
        canvas = canvas
            .arc(c.cx, c.cy - c.h * 0.05, r, start, pi * 0.8)
            .stroke(C::from_hsl(hue, 0.85, 0.6), 3.5)
            .line_cap(*cap)
            .done();
    }

    // Dashed polyline — marching ants effect
    let march_offset = t * 30.0;
    let poly_pts = vec![
        (c.x + pad, c.y + c.h - pad * 4.0),
        (c.x + c.w * 0.25, c.y + c.h * 0.6),
        (c.x + c.w * 0.5, c.y + c.h - pad * 3.0),
        (c.x + c.w * 0.75, c.y + c.h * 0.65),
        (c.x + c.w - pad, c.y + c.h - pad * 4.0),
    ];
    canvas = canvas
        .polyline(poly_pts)
        .stroke(C::from_rgba8(0, 255, 200, 255), 2.5)
        .line_cap(LineCap::Round)
        .dash(DashPattern::new(vec![8.0, 4.0, 2.0, 4.0], march_offset))
        .done();

    // Spinning polygon with bevel join
    let tri_r = c.w.min(c.h) * 0.12;
    let tri_cx = c.x + c.w * 0.18;
    let tri_cy = c.cy;
    let tri_pts: Vec<(f32, f32)> = (0..3)
        .map(|i| {
            let angle = std::f32::consts::TAU * i as f32 / 3.0 + t;
            (tri_cx + tri_r * angle.cos(), tri_cy + tri_r * angle.sin())
        })
        .collect();
    canvas = canvas
        .polygon(tri_pts)
        .fill(C::from_rgba8(255, 99, 71, 180))
        .stroke(C::WHITE, 2.0)
        .line_join(LineJoin::Bevel)
        .done();

    // Aliased shapes (anti_alias=false)
    let aa_x = c.x + c.w * 0.77;
    let aa_y = c.y + c.h * 0.25;
    let aa_r = 10.0;
    canvas = canvas
        .circle(aa_x, aa_y, aa_r)
        .fill(C::from_rgba8(255, 50, 200, 255))
        .anti_alias(false)
        .done()
        .rect(aa_x - aa_r, aa_y + aa_r + 4.0, aa_r * 2.0, aa_r * 2.0)
        .fill(C::from_rgba8(50, 255, 150, 255))
        .anti_alias(false)
        .done();

    // Crossed lines with different caps
    let lx = c.x + c.w * 0.55;
    let ly = c.y + pad * 2.0;
    canvas = canvas
        .line(lx, ly, lx + c.w * 0.15, ly + c.h * 0.15)
        .stroke(C::from_rgba8(255, 215, 0, 255), 3.0)
        .line_cap(LineCap::Round)
        .done()
        .line(lx + c.w * 0.15, ly + c.h * 0.15, lx + c.w * 0.3, ly)
        .stroke(C::from_rgba8(255, 100, 100, 255), 3.0)
        .line_cap(LineCap::Square)
        .done();

    draw_label(&mut canvas, c.x + pad, c.y + pad, "PRECISION STROKES");
    canvas
}

// ─── Panel 5: Clipped Kaleidoscope ───────────────────────────────

fn panel_clipped_kaleidoscope(mut canvas: PixelCanvas, wf: f32, hf: f32, t: f32) -> PixelCanvas {
    let c = cell(1, 1, wf, hf);
    let pad = 10.0;
    let r = c.w.min(c.h) * 0.36 - pad;

    // Left: rect clip with rotating stripes
    let clip_w = c.w * 0.42;
    let clip_h = c.h * 0.7;
    let clip_x = c.x + pad;
    let clip_y = c.y + pad * 3.0;
    let clip_rect = PxRect::new(clip_x, clip_y, clip_w, clip_h);

    canvas = canvas
        .rect(clip_x, clip_y, clip_w, clip_h)
        .stroke(C::from_rgba8(100, 200, 255, 150), 1.5)
        .corner_radius(4.0)
        .done();

    canvas = canvas
        .group(Transform::rotate_at(
            t * 0.4,
            clip_x + clip_w / 2.0,
            clip_y + clip_h / 2.0,
        ))
        .clip_rect(clip_rect)
        .canvas(move |mut inner| {
            let stripe_w = clip_w * 0.3;
            let colors = [
                C::from_rgba8(255, 60, 60, 200),
                C::from_rgba8(60, 255, 60, 200),
                C::from_rgba8(60, 60, 255, 200),
                C::from_rgba8(255, 255, 60, 200),
                C::from_rgba8(255, 60, 255, 200),
            ];
            for (i, &color) in colors.iter().enumerate() {
                let sx = clip_x - clip_w + i as f32 * stripe_w;
                inner = inner
                    .rect(sx, clip_y - clip_h, stripe_w - 2.0, clip_h * 3.0)
                    .fill(color)
                    .done();
            }
            inner
        })
        .done();

    // Right: circle clip with rotating kaleidoscope
    let circle_cx = c.x + c.w * 0.72;
    let circle_cy = c.cy;
    let circle_r = r * 0.7;

    canvas = canvas
        .circle(circle_cx, circle_cy, circle_r)
        .stroke(C::from_rgba8(255, 200, 100, 150), 2.0)
        .done();

    let circle_clip = PxRect::new(
        circle_cx - circle_r,
        circle_cy - circle_r,
        circle_r * 2.0,
        circle_r * 2.0,
    );

    canvas = canvas
        .group(Transform::rotate_at(-t * 0.6, circle_cx, circle_cy))
        .clip_rect(circle_clip)
        .opacity(0.9)
        .canvas(move |mut inner| {
            for i in 0..6 {
                let angle = std::f32::consts::TAU * i as f32 / 6.0;
                let dx = circle_r * 0.5 * angle.cos();
                let dy = circle_r * 0.5 * angle.sin();
                let hue = (i as f32 * 60.0 + t * 45.0) % 360.0;
                inner = inner
                    .rect(circle_cx + dx - 12.0, circle_cy + dy - 12.0, 24.0, 24.0)
                    .fill(C::from_hsla(hue, 0.9, 0.55, 0.8))
                    .corner_radius(4.0)
                    .done();
            }
            inner
                .circle(circle_cx, circle_cy, circle_r * 0.25)
                .fill(C::WHITE.with_alpha(0.7))
                .done()
        })
        .done();

    draw_label(&mut canvas, c.x + pad, c.y + pad, "CLIP REGIONS");
    canvas
}

// ─── Panel 6: Orchestrated Scene ─────────────────────────────────

fn setup_orchestrator(anim: &mut AnimationState) {
    anim.cancel_all();
    anim.start(
        "x",
        0.0_f32,
        1.0_f32,
        Duration::from_millis(2500),
        Easing::EaseInOutCubic,
    );
    anim.start(
        "y",
        0.0_f32,
        1.0_f32,
        Duration::from_millis(3000),
        Easing::Bounce,
    );
    anim.start(
        "scale",
        0.3_f32,
        2.0_f32,
        Duration::from_millis(2000),
        Easing::EaseInOutQuart,
    );
    anim.start(
        "hue",
        0.0_f32,
        360.0_f32,
        Duration::from_secs(4),
        Easing::Linear,
    );
    anim.start(
        "opacity",
        0.2_f32,
        1.0_f32,
        Duration::from_millis(1500),
        Easing::EaseOutExpo,
    );
    anim.start(
        "rotation",
        0.0_f32,
        std::f32::consts::TAU,
        Duration::from_secs(3),
        Easing::EaseInOutSine,
    );
    anim.start(
        "skew",
        -0.3_f32,
        0.3_f32,
        Duration::from_millis(2800),
        Easing::CSS_EASE,
    );
    anim.start(
        "corner",
        0.0_f32,
        20.0_f32,
        Duration::from_millis(2200),
        Easing::Elastic,
    );
}

fn panel_orchestrated_scene(
    mut canvas: PixelCanvas,
    wf: f32,
    hf: f32,
    anim: &AnimationState,
) -> PixelCanvas {
    let c = cell(2, 1, wf, hf);
    let pad = 10.0;

    let x_norm = anim.get::<f32>("x").unwrap_or(0.0);
    let y_norm = anim.get::<f32>("y").unwrap_or(0.0);
    let scale = anim.get::<f32>("scale").unwrap_or(1.0);
    let hue = anim.get::<f32>("hue").unwrap_or(0.0);
    let opacity = anim.get::<f32>("opacity").unwrap_or(1.0);
    let rotation = anim.get::<f32>("rotation").unwrap_or(0.0);
    let skew = anim.get::<f32>("skew").unwrap_or(0.0);
    let corner = anim.get::<f32>("corner").unwrap_or(0.0);

    let margin = 30.0;
    let cx = c.x + margin + (c.w - margin * 2.0) * x_norm;
    let cy = c.y + margin + (c.h - margin * 2.0 - 30.0) * y_norm;
    let sz = 15.0 * scale;
    let color = C::from_hsla(hue, 0.85, 0.55, opacity);

    // Grid
    for i in 0..=8 {
        let frac = i as f32 / 8.0;
        let gx = c.x + margin + (c.w - margin * 2.0) * frac;
        let gy = c.y + margin + (c.h - margin * 2.0 - 30.0) * frac;
        canvas = canvas
            .line(gx, c.y + margin, gx, c.y + c.h - margin - 30.0)
            .color(C::from_rgba8(20, 22, 35, 255))
            .width(0.5)
            .done()
            .line(c.x + margin, gy, c.x + c.w - margin, gy)
            .color(C::from_rgba8(20, 22, 35, 255))
            .width(0.5)
            .done();
    }

    // Glow
    canvas = canvas
        .circle(cx, cy, sz * 2.0)
        .fill(color.with_alpha(0.06))
        .done()
        .circle(cx, cy, sz * 1.4)
        .fill(color.with_alpha(0.12))
        .done();

    // Main animated shape with transform group
    let transform = Transform::rotate_at(rotation, cx, cy).concat(Transform::skew(skew, 0.0));

    canvas = canvas
        .group(transform)
        .canvas(move |inner| {
            inner
                .rect(cx - sz, cy - sz, sz * 2.0, sz * 2.0)
                .fill(color)
                .corner_radius(corner.clamp(0.0, sz))
                .stroke(C::WHITE.with_alpha(0.5), 1.5)
                .done()
        })
        .done();

    // Bézier heart decoration (exercises Path + push_command)
    let heart_x = c.x + c.w - 35.0;
    let heart_y = c.y + pad + 5.0;
    let hs = 10.0;
    let mut pb = tiny_skia::PathBuilder::new();
    pb.move_to(heart_x, heart_y + hs * 0.7);
    pb.cubic_to(
        heart_x - hs,
        heart_y - hs * 0.1,
        heart_x - hs * 0.3,
        heart_y - hs * 0.6,
        heart_x,
        heart_y + hs * 0.1,
    );
    pb.cubic_to(
        heart_x + hs * 0.3,
        heart_y - hs * 0.6,
        heart_x + hs,
        heart_y - hs * 0.1,
        heart_x,
        heart_y + hs * 0.7,
    );
    pb.close();

    if let Some(path) = pb.finish() {
        canvas.push_command(DrawCommand::Path {
            path: scry_engine::scene::command::PathData::new(path),
            style: ShapeStyle {
                fill: Some(FillStyle::Solid(C::from_rgba8(220, 50, 80, 200))),
                stroke: Some(StrokeStyle {
                    color: C::from_rgba8(255, 180, 190, 200),
                    width: 1.0,
                    line_cap: LineCap::Round,
                    line_join: LineJoin::Round,
                    dash: None,
                }),
                anti_alias: true,
            },
        });
    }

    // Progress bars for 8 animation channels
    let values = [
        x_norm,
        y_norm,
        (scale - 0.3) / 1.7,
        hue / 360.0,
        opacity,
        rotation / std::f32::consts::TAU,
        (skew + 0.3) / 0.6,
        corner / 20.0,
    ];
    let bar_y = c.y + c.h - 25.0;
    let bar_total = c.w - margin * 2.0;
    let bar_each = bar_total / 8.0;

    for i in 0..8 {
        let bx = c.x + margin + bar_each * i as f32;
        let bw = bar_each * 0.8;
        let bh = 5.0;
        let fill_frac = values[i].clamp(0.0, 1.0);
        let bar_color = C::from_hsl(360.0 * i as f32 / 8.0, 0.7, 0.5);

        canvas = canvas
            .rect(bx, bar_y, bw, bh)
            .fill(C::from_rgba8(25, 28, 45, 255))
            .corner_radius(2.0)
            .done()
            .rect(bx, bar_y, bw * fill_frac, bh)
            .fill(bar_color)
            .corner_radius(2.0)
            .done();
    }

    draw_label(&mut canvas, c.x + pad, c.y + pad, "ORCHESTRATOR");
    canvas
}

// ─── Pixel font labels ───────────────────────────────────────────

fn draw_label(canvas: &mut PixelCanvas, x: f32, y: f32, text: &str) {
    // Batch all lit pixels into a single compound path to avoid ~700
    // individual DrawCommand::Rectangle calls per label.
    let mut pb = tiny_skia::PathBuilder::new();
    let mut cursor = x;
    for ch in text.chars() {
        let upper = ch.to_ascii_uppercase();
        if let Some((_, bits)) = PIXEL_FONT.iter().find(|(c, _)| *c == upper.to_string()) {
            for (row, &byte) in bits.iter().enumerate() {
                for col in 0..5 {
                    if byte & (1 << (4 - col)) != 0 {
                        if let Some(r) = tiny_skia::Rect::from_xywh(
                            cursor + col as f32,
                            y + row as f32,
                            1.0,
                            1.0,
                        ) {
                            pb.push_rect(r);
                        }
                    }
                }
            }
            cursor += 6.0;
        } else {
            cursor += 4.0;
        }
    }

    if let Some(path) = pb.finish() {
        canvas.push_command(DrawCommand::Path {
            path: scry_engine::scene::command::PathData::new(path),
            style: ShapeStyle {
                fill: Some(FillStyle::Solid(C::from_rgba8(160, 170, 200, 180))),
                stroke: None,
                anti_alias: false,
            },
        });
    }
}

const PIXEL_FONT: [(&str, &[u8]); 33] = [
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
    (" ", &[0, 0, 0, 0, 0, 0, 0]),
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
    (":", &[0, 0b00100, 0b00100, 0, 0b00100, 0b00100, 0]),
    (".", &[0, 0, 0, 0, 0, 0b00100, 0b00100]),
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
    (
        "Z",
        &[
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111,
        ],
    ),
    ("-", &[0, 0, 0, 0b01110, 0, 0, 0]),
    (
        "!",
        &[0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0, 0b00100],
    ),
];
