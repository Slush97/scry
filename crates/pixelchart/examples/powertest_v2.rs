//! **Power Test V2** — ultimate 10-panel animated stress test.
//!
//! Exercises every public API surface of both `ratatui-pixelcanvas` and
//! `pixelchart` in a single animated dashboard.
//!
//! Run with: `cargo run -p pixelchart --example powertest_v2`

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
    clippy::doc_markdown,
    clippy::wildcard_imports,
    clippy::needless_pass_by_value
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

use pixelchart::prelude::*;

use ratatui_pixelcanvas::prelude::{
    AnimationState, Easing, Keyframe, Keyframes, Picker, PixelCanvasState, PixelCanvasWidget,
    ProtocolKind,
};
use ratatui_pixelcanvas::scene::style::{
    BlendMode, Color as C, DashPattern, GradientDef, GradientKind, GradientStop, LineCap, LineJoin,
    Point, Rect as PxRect, Transform,
};
use ratatui_pixelcanvas::scene::PixelCanvas;
use ratatui_pixelcanvas::transport;

// ═══════════════════════════════════════════════════════════════════
// Data helpers
// ═══════════════════════════════════════════════════════════════════

fn hash_u64(mut x: u64) -> u64 {
    x = x.wrapping_mul(0x517cc1b727220a95);
    x ^= x >> 32;
    x
}

#[allow(dead_code)]
fn pseudo_normal(n: usize, mean: f64, std: f64, seed: u64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let mut sum = 0.0;
            for k in 0..6u64 {
                let v = (hash_u64(i as u64 * 7919 + k * 13 + seed) % 10000) as f64 / 10000.0;
                sum += v;
            }
            mean + (sum - 3.0) * std
        })
        .collect()
}

#[allow(dead_code)]
fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| lo + (hi - lo) * i as f64 / (n - 1).max(1) as f64)
        .collect()
}

// ═══════════════════════════════════════════════════════════════════
// Grid cell helper
// ═══════════════════════════════════════════════════════════════════

struct Cell {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    cx: f32,
    cy: f32,
}

fn cell(col: usize, row: usize, total_w: f32, total_h: f32) -> Cell {
    let w = total_w / 2.0;
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

// ═══════════════════════════════════════════════════════════════════
// App State
// ═══════════════════════════════════════════════════════════════════

struct AppState {
    chart_states: Vec<ChartState>,
    anim: AnimationState,
    line_data_a: Vec<f64>,
    line_data_b: Vec<f64>,
    frame_count: u64,
    stacked_mode: bool,
    donut_mode: bool,
    fps: f64,
    draw_ms: f64,
}

impl AppState {
    fn new() -> Self {
        let chart_states = (0..8).map(|_| ChartState::auto()).collect();
        let mut anim = AnimationState::new();
        setup_anims(&mut anim);
        Self {
            chart_states,
            anim,
            line_data_a: Vec::new(),
            line_data_b: Vec::new(),
            frame_count: 0,
            stacked_mode: false,
            donut_mode: false,
            fps: 0.0,
            draw_ms: 0.0,
        }
    }

    fn flush(&mut self) {
        for s in &mut self.chart_states {
            let _ = s.flush();
        }
    }

    fn cleanup(&mut self) {
        for s in &mut self.chart_states {
            s.cleanup();
        }
    }
}

fn setup_anims(anim: &mut AnimationState) {
    anim.cancel_all();
    anim.start(
        "x",
        0.0_f32,
        1.0_f32,
        Duration::from_millis(2500),
        Easing::EaseInOutCubic,
    );
    anim.start(
        "hue",
        0.0_f32,
        360.0_f32,
        Duration::from_secs(5),
        Easing::Linear,
    );
    anim.start(
        "scale",
        0.3_f32,
        2.0_f32,
        Duration::from_millis(2000),
        Easing::Bounce,
    );
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
    let mut app = AppState::new();

    let start = Instant::now();
    let mut last_frame = start;
    let mut frame_times: Vec<Duration> = Vec::with_capacity(60);

    loop {
        let now = Instant::now();
        let dt = now - last_frame;
        last_frame = now;
        let t = now.duration_since(start).as_secs_f32();

        frame_times.push(dt);
        if frame_times.len() > 60 {
            frame_times.remove(0);
        }
        if !frame_times.is_empty() {
            let total: Duration = frame_times.iter().sum();
            app.fps = frame_times.len() as f64 / total.as_secs_f64();
        }

        app.anim.tick(dt);
        if app.anim.is_idle() {
            setup_anims(&mut app.anim);
        }
        app.frame_count += 1;

        // Toggle modes on timer
        if app.frame_count % 300 == 0 {
            app.stacked_mode = !app.stacked_mode;
        }
        if app.frame_count % 450 == 0 {
            app.donut_mode = !app.donut_mode;
        }

        // Update streaming line data
        let new_a = 30.0 * (0.3 * t as f64 + 0.0).sin() + 15.0 * (0.7 * t as f64).sin() + 50.0;
        let new_b = 25.0 * (0.4 * t as f64 + 1.0).sin() + 10.0 * (0.9 * t as f64).cos() + 45.0;
        app.line_data_a.push(new_a);
        app.line_data_b.push(new_b);
        if app.line_data_a.len() > 50 {
            app.line_data_a.remove(0);
        }
        if app.line_data_b.len() > 50 {
            app.line_data_b.remove(0);
        }

        let draw_start = Instant::now();

        terminal.draw(|frame| render(frame, &mut px_state, &mut app, t))?;

        app.draw_ms = draw_start.elapsed().as_secs_f64() * 1000.0;
        px_state.flush()?;
        app.flush();

        if event::poll(Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press && key.code == KeyCode::Char('q') {
                    break;
                }
            }
        }
    }

    px_state.cleanup();
    app.cleanup();
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════
// Render
// ═══════════════════════════════════════════════════════════════════

fn render(frame: &mut ratatui::Frame, px_state: &mut PixelCanvasState, app: &mut AppState, t: f32) {
    let area = frame.area();
    let rows = Layout::vertical([
        Constraint::Percentage(40), // raw canvas panels ①-④
        Constraint::Percentage(20), // charts ⑤-⑥
        Constraint::Percentage(20), // charts ⑦-⑧
        Constraint::Percentage(17), // charts ⑨-⑩
        Constraint::Length(1),      // status
    ])
    .split(area);

    // ── Raw Canvas (panels ①-④) ──
    let canvas = build_raw_scene(rows[0], px_state, t, &app.anim);
    frame.render_stateful_widget(
        PixelCanvasWidget::new(canvas).z_index(-1).skip_cache(),
        rows[0],
        px_state,
    );

    // ── Chart row 1: ⑤ + ⑥ ──
    let r1_cols =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).split(rows[1]);
    let line_chart = build_live_lines(&app.line_data_a, &app.line_data_b);
    frame.render_stateful_widget(
        ChartWidget::new(&line_chart),
        r1_cols[0],
        &mut app.chart_states[0],
    );
    let scatter = build_scatter_matrix(t);
    frame.render_stateful_widget(
        ChartWidget::new(&scatter),
        r1_cols[1],
        &mut app.chart_states[1],
    );

    // ── Chart row 2: ⑦ + ⑧ ──
    let r2_cols =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).split(rows[2]);
    let bar = build_stat_theater(app.stacked_mode, t);
    frame.render_stateful_widget(ChartWidget::new(&bar), r2_cols[0], &mut app.chart_states[2]);
    let heat_pie = build_thermal_pie(app.donut_mode, t);
    frame.render_stateful_widget(
        ChartWidget::new(&heat_pie),
        r2_cols[1],
        &mut app.chart_states[3],
    );

    // ── Chart row 3: ⑨ + ⑩ ──
    let r3_cols =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).split(rows[3]);
    let theme_chart = build_theme_gallery();
    frame.render_stateful_widget(
        ChartWidget::new(&theme_chart),
        r3_cols[0],
        &mut app.chart_states[4],
    );
    let interactive = build_interactive(t);
    frame.render_stateful_widget(
        ChartWidget::new(&interactive),
        r3_cols[1],
        &mut app.chart_states[5],
    );

    // ── Status bar ──
    let status = Paragraph::new(format!(
        " ⚡ POWERTEST V2 │ {fps:.0} fps │ draw {draw_ms:.1}ms │ frame {} │ 'q' quit",
        app.frame_count,
        fps = app.fps,
        draw_ms = app.draw_ms,
    ))
    .block(Block::default().borders(Borders::TOP));
    frame.render_widget(status, rows[4]);
}

// ═══════════════════════════════════════════════════════════════════
// Raw canvas scene (panels ①-④ in 2×2 sub-grid)
// ═══════════════════════════════════════════════════════════════════

fn build_raw_scene(
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

    let mut canvas = PixelCanvas::new(w, h).background(C::from_rgba8(10, 12, 30, 255));

    // Panel dividers
    let div = C::from_rgba8(30, 40, 70, 200);
    canvas = canvas
        .line(wf / 2.0, 0.0, wf / 2.0, hf)
        .color(div)
        .width(1.0)
        .done()
        .line(0.0, hf / 2.0, wf, hf / 2.0)
        .color(div)
        .width(1.0)
        .done();

    canvas = panel_vector_gauntlet(canvas, wf, hf, t);
    canvas = panel_gradient_symphony(canvas, wf, hf, t);
    canvas = panel_transform_engine(canvas, wf, hf, t, anim);
    canvas = panel_animation_orchestrator(canvas, wf, hf, t);

    canvas
}

// ─── Panel ① Vector Gauntlet ─────────────────────────────────────

fn panel_vector_gauntlet(mut canvas: PixelCanvas, wf: f32, hf: f32, t: f32) -> PixelCanvas {
    let c = cell(0, 0, wf, hf);
    let pad = 8.0;
    let pi = std::f32::consts::PI;

    // 3 circles with staggered pulse (glow + crisp)
    for i in 0..3 {
        let phase = i as f32 * 2.0 * pi / 3.0;
        let r = 12.0 + 6.0 * (t * 1.5 + phase).sin();
        let angle = t * 0.3 + phase;
        let cx = c.cx - c.w * 0.15 + (c.w * 0.12) * angle.cos();
        let cy = c.cy - c.h * 0.1 + (c.h * 0.12) * angle.sin();
        let hue = (i as f32 * 120.0 + t * 40.0) % 360.0;
        let col = C::from_hsla(hue, 0.9, 0.6, 0.9);
        // Glow layer
        canvas = canvas
            .circle(cx, cy, r * 1.3)
            .fill(col.with_alpha(0.12))
            .done();
        // Crisp shape
        canvas = canvas
            .circle(cx, cy, r)
            .fill(col)
            .stroke(C::WHITE.with_alpha(0.4), 1.0)
            .done();
    }

    // Rounded rect with oscillating corner radius
    let cr = 10.0 * ((t * 1.2).sin() * 0.5 + 0.5);
    let rh = (t * 30.0) % 360.0;
    canvas = canvas
        .rect(c.x + pad, c.y + c.h * 0.6, c.w * 0.2, c.h * 0.25)
        .fill(C::from_hsla(rh, 0.8, 0.5, 0.8))
        .corner_radius(cr)
        .stroke(C::WHITE.with_alpha(0.3), 1.5)
        .done();

    // Rotated ellipse
    canvas = canvas
        .ellipse(c.x + c.w * 0.7, c.y + c.h * 0.3, 25.0, 12.0)
        .rotation(t * 0.5)
        .fill(C::from_hsla((t * 50.0) % 360.0, 0.85, 0.55, 0.7))
        .stroke(C::from_rgba8(255, 255, 255, 100), 1.5)
        .done();

    // Polyline with wiggling vertices
    let poly_pts: Vec<(f32, f32)> = (0..5)
        .map(|i| {
            let bx = c.x + pad + (c.w - pad * 2.0) * i as f32 / 4.0;
            let by = c.y + c.h * 0.5 + 8.0 * (t * 2.0 + i as f32 * 1.5).sin();
            (bx, by)
        })
        .collect();
    canvas = canvas
        .polyline(poly_pts)
        .stroke(C::from_rgba8(0, 255, 220, 220), 2.0)
        .line_cap(LineCap::Round)
        .dash(DashPattern::new(vec![6.0, 3.0], t * 20.0))
        .done();

    // Pentagon polygon
    let pent_r = 15.0 + 3.0 * (t * 1.8).sin();
    let pent_cx = c.x + c.w * 0.35;
    let pent_cy = c.y + c.h * 0.75;
    let pent: Vec<(f32, f32)> = (0..5)
        .map(|i| {
            let a = std::f32::consts::TAU * i as f32 / 5.0 - pi / 2.0;
            (pent_cx + pent_r * a.cos(), pent_cy + pent_r * a.sin())
        })
        .collect();
    canvas = canvas
        .polygon(pent)
        .fill(C::from_rgba8(255, 100, 80, 160))
        .stroke(C::WHITE, 1.5)
        .line_join(LineJoin::Bevel)
        .done();

    // Arc with oscillating sweep
    let sweep = pi * (0.3 + 2.0 * ((t * 0.8).sin() * 0.5 + 0.5));
    canvas = canvas
        .arc(c.x + c.w * 0.8, c.y + c.h * 0.7, 20.0, t, sweep)
        .stroke(C::from_hsla((t * 60.0) % 360.0, 0.9, 0.6, 0.9), 2.5)
        .line_cap(LineCap::Round)
        .done();

    // Bézier path (S-curve with drifting control points)
    let mut pb = tiny_skia::PathBuilder::new();
    pb.move_to(c.x + pad, c.cy);
    pb.cubic_to(
        c.x + c.w * 0.3,
        c.y + pad + 10.0 * (t * 1.3).sin(),
        c.x + c.w * 0.6,
        c.y + c.h - pad + 10.0 * (t * 1.7).cos(),
        c.x + c.w - pad,
        c.cy,
    );
    if let Some(path) = pb.finish() {
        canvas = canvas
            .path(path)
            .stroke(C::from_rgba8(255, 200, 50, 200), 2.0)
            .line_cap(LineCap::Round)
            .done();
    }

    // Dashed lines at different widths
    for i in 0..3 {
        let ly = c.y + pad * 2.0 + i as f32 * 8.0;
        canvas = canvas
            .line(c.x + pad, ly, c.x + c.w * 0.4, ly)
            .color(C::from_hsla(
                (i as f32 * 90.0 + t * 30.0) % 360.0,
                0.8,
                0.6,
                0.8,
            ))
            .width(1.0 + i as f32 * 1.5)
            .dash(DashPattern::new(vec![5.0, 3.0], t * 25.0))
            .line_cap([LineCap::Butt, LineCap::Round, LineCap::Square][i])
            .done();
    }

    canvas
}

// ─── Panel ② Gradient Symphony ───────────────────────────────────

fn panel_gradient_symphony(mut canvas: PixelCanvas, wf: f32, hf: f32, t: f32) -> PixelCanvas {
    let c = cell(1, 0, wf, hf);
    let pad = 8.0;

    // Horizontal rainbow gradient band
    let band_h = (c.h - pad * 2.0) * 0.15;
    let base_hue = (t * 25.0) % 360.0;
    canvas = canvas
        .gradient(c.x + pad, c.y + pad, c.w - pad * 2.0, band_h)
        .linear(
            Point::new(c.x + pad, c.y + pad),
            Point::new(c.x + c.w - pad, c.y + pad),
        )
        .stop(0.0, C::from_hsl(base_hue, 0.9, 0.5))
        .stop(0.17, C::from_hsl((base_hue + 50.0) % 360.0, 0.9, 0.55))
        .stop(0.33, C::from_hsl((base_hue + 100.0) % 360.0, 0.85, 0.5))
        .stop(0.5, C::from_hsl((base_hue + 150.0) % 360.0, 0.9, 0.5))
        .stop(0.67, C::from_hsl((base_hue + 200.0) % 360.0, 0.85, 0.55))
        .stop(0.83, C::from_hsl((base_hue + 270.0) % 360.0, 0.9, 0.5))
        .stop(1.0, C::from_hsl((base_hue + 330.0) % 360.0, 0.9, 0.5))
        .done();

    // Vertical gradient
    let v_y = c.y + pad + band_h + 4.0;
    canvas = canvas
        .gradient(c.x + pad, v_y, c.w * 0.3, band_h)
        .linear(
            Point::new(c.x + pad, v_y),
            Point::new(c.x + pad, v_y + band_h),
        )
        .stop(0.0, C::from_rgba8(10, 10, 40, 255))
        .stop(1.0, C::from_hsla((t * 35.0) % 360.0, 0.9, 0.7, 0.9))
        .done();

    // Diagonal gradient
    canvas = canvas
        .gradient(c.x + c.w * 0.35, v_y, c.w * 0.3, band_h)
        .linear(
            Point::new(c.x + c.w * 0.35, v_y),
            Point::new(c.x + c.w * 0.65, v_y + band_h),
        )
        .stop(0.0, C::from_hsla((t * 45.0) % 360.0, 1.0, 0.4, 1.0))
        .stop(0.5, C::from_hsla((t * 45.0 + 120.0) % 360.0, 0.9, 0.6, 0.8))
        .stop(
            1.0,
            C::from_hsla((t * 45.0 + 240.0) % 360.0, 0.85, 0.5, 1.0),
        )
        .done();

    // Centered radial gradient "sun"
    let orb_r = c.w.min(c.h) * 0.22;
    let orb_x = c.cx + (t * 0.5).sin() * c.w * 0.1;
    let orb_y = c.y + c.h * 0.65 + (t * 0.7).cos() * c.h * 0.08;
    canvas = canvas
        .gradient(orb_x - orb_r, orb_y - orb_r, orb_r * 2.0, orb_r * 2.0)
        .radial(Point::new(orb_x, orb_y), orb_r)
        .stop(0.0, C::from_rgba8(255, 255, 240, 220))
        .stop(0.3, C::from_hsla((t * 30.0) % 360.0, 0.95, 0.65, 0.8))
        .stop(0.7, C::from_hsla((t * 30.0 + 180.0) % 360.0, 0.8, 0.4, 0.4))
        .stop(1.0, C::TRANSPARENT)
        .done();

    // Shape with fill_linear_gradient
    let gr = GradientDef {
        kind: GradientKind::Linear {
            start: Point::new(c.x + c.w * 0.7, c.y + c.h * 0.5),
            end: Point::new(c.x + c.w - pad, c.y + c.h * 0.7),
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
        .rect(c.x + c.w * 0.7, c.y + c.h * 0.5, c.w * 0.25, c.h * 0.18)
        .fill_linear_gradient(gr)
        .corner_radius(5.0)
        .done();

    // Circle with fill_radial_gradient
    let rg = GradientDef {
        kind: GradientKind::Radial {
            center: Point::new(c.x + c.w * 0.8, c.y + c.h * 0.8),
            radius: 18.0,
        },
        stops: vec![
            GradientStop {
                position: 0.0,
                color: C::from_hsla((t * 60.0) % 360.0, 1.0, 0.8, 0.9),
            },
            GradientStop {
                position: 1.0,
                color: C::from_hsla((t * 60.0 + 180.0) % 360.0, 0.7, 0.3, 0.5),
            },
        ],
    };
    canvas = canvas
        .circle(c.x + c.w * 0.8, c.y + c.h * 0.8, 18.0)
        .fill_radial_gradient(rg)
        .stroke(C::WHITE.with_alpha(0.3), 1.0)
        .done();

    // Oklab color mixing demo bar
    let mix_y = c.y + c.h - pad - 10.0;
    let mix_w = c.w - pad * 2.0;
    let steps = 20;
    let ca = C::from_hsl((t * 20.0) % 360.0, 1.0, 0.5);
    let cb = C::from_hsl((t * 20.0 + 180.0) % 360.0, 1.0, 0.5);
    for i in 0..steps {
        let frac = i as f32 / (steps - 1) as f32;
        let mixed = ca.mix(cb, frac); // Oklab interpolation
        let sx = c.x + pad + mix_w * frac;
        canvas = canvas
            .rect(sx, mix_y, mix_w / steps as f32 + 1.0, 10.0)
            .fill(mixed)
            .done();
    }

    canvas
}

// ─── Panel ③ Transform Engine ────────────────────────────────────

fn panel_transform_engine(
    mut canvas: PixelCanvas,
    wf: f32,
    hf: f32,
    t: f32,
    _anim: &AnimationState,
) -> PixelCanvas {
    let c = cell(0, 1, wf, hf);
    let pad = 10.0;

    // Central star (rotate_at)
    let star_transform = Transform::rotate_at(t * 0.3, c.cx, c.cy);
    canvas = canvas
        .group(star_transform)
        .canvas(move |inner| {
            let r = 20.0;
            inner
                .circle(c.cx, c.cy, r)
                .fill(C::from_hsla((t * 20.0) % 360.0, 0.9, 0.6, 0.8))
                .stroke(C::WHITE.with_alpha(0.5), 1.5)
                .done()
        })
        .done();

    // 3 orbiting planets at different speeds
    for i in 0..3 {
        let speed = [0.8, 0.5, 0.3][i];
        let orbit_r = [60.0_f32, 90.0, 120.0][i].min(c.w * 0.35).min(c.h * 0.35);
        let angle = t * speed + (i as f32) * std::f32::consts::TAU / 3.0;
        let px = c.cx + orbit_r * angle.cos();
        let py = c.cy + orbit_r * angle.sin();
        let hue = (i as f32 * 120.0 + t * 30.0) % 360.0;
        let planet_r = 8.0 + 2.0 * i as f32;

        // Orbit path (faint dotted)
        canvas = canvas
            .circle(c.cx, c.cy, orbit_r)
            .stroke(C::from_rgba8(40, 50, 80, 80), 0.5)
            .done();

        // Planet with clip
        let clip = PxRect::new(
            px - planet_r * 1.5,
            py - planet_r * 1.5,
            planet_r * 3.0,
            planet_r * 3.0,
        );
        let planet_transform = Transform::rotate_at(-angle * 2.0, px, py);
        canvas = canvas
            .group(planet_transform)
            .clip_rect(clip)
            .opacity(0.7 + 0.3 * (t + i as f32).sin())
            .canvas(move |inner| {
                inner
                    .circle(px, py, planet_r)
                    .fill(C::from_hsla(hue, 0.85, 0.55, 0.9))
                    .done()
                    .rect(px - 4.0, py - 4.0, 8.0, 8.0)
                    .fill(C::WHITE.with_alpha(0.4))
                    .corner_radius(2.0)
                    .done()
            })
            .done();
    }

    // Blend mode demo: overlapping circles
    let blend_modes = [BlendMode::Multiply, BlendMode::Screen, BlendMode::Overlay];
    let blend_colors = [
        C::from_rgba8(255, 80, 80, 180),
        C::from_rgba8(80, 255, 80, 180),
        C::from_rgba8(80, 80, 255, 180),
    ];
    let blend_cx = c.x + c.w * 0.8;
    let blend_cy = c.y + c.h * 0.8;
    for i in 0..3 {
        let a = std::f32::consts::TAU * i as f32 / 3.0 + t * 0.4;
        let bx = blend_cx + 10.0 * a.cos();
        let by = blend_cy + 10.0 * a.sin();
        let mode = blend_modes[i];
        let col = blend_colors[i];
        canvas = canvas
            .group(Transform::identity())
            .blend_mode(mode)
            .canvas(move |inner| inner.circle(bx, by, 14.0).fill(col).done())
            .done();
    }

    // Skew demo
    let skew_amount = (t * 1.2).sin() * 0.3;
    canvas = canvas
        .group(Transform::skew(skew_amount, 0.0))
        .opacity(0.6)
        .canvas(move |inner| {
            inner
                .rect(c.x + pad, c.y + c.h - pad - 20.0, 30.0, 15.0)
                .fill(C::from_rgba8(200, 200, 255, 200))
                .corner_radius(3.0)
                .done()
        })
        .done();

    canvas
}

// ─── Panel ④ Animation Orchestrator ──────────────────────────────

fn panel_animation_orchestrator(mut canvas: PixelCanvas, wf: f32, hf: f32, t: f32) -> PixelCanvas {
    let c = cell(1, 1, wf, hf);
    let pad = 6.0;

    let easings: &[(&str, Easing)] = &[
        ("Linear", Easing::Linear),
        ("InQuad", Easing::EaseInQuad),
        ("OutQuad", Easing::EaseOutQuad),
        ("InOutQuad", Easing::EaseInOutQuad),
        ("InCubic", Easing::EaseInCubic),
        ("OutCubic", Easing::EaseOutCubic),
        ("InOutCubic", Easing::EaseInOutCubic),
        ("InQuart", Easing::EaseInQuart),
        ("OutQuart", Easing::EaseOutQuart),
        ("InSine", Easing::EaseInSine),
        ("OutSine", Easing::EaseOutSine),
        ("InOutSine", Easing::EaseInOutSine),
        ("InExpo", Easing::EaseInExpo),
        ("OutExpo", Easing::EaseOutExpo),
        ("InCirc", Easing::EaseInCirc),
        ("OutCirc", Easing::EaseOutCirc),
        ("Spring", Easing::Spring { overshoot: 1.7 }),
        ("Elastic", Easing::Elastic),
        ("Bounce", Easing::Bounce),
        ("CSSEase", Easing::CSS_EASE),
        ("Back", Easing::BACK),
    ];

    let n = easings.len();
    let track_h = ((c.h - pad * 2.0) / n as f32).min(8.0);
    let track_w = c.w - pad * 2.0 - 60.0;
    let period = 3.0_f32;
    let raw_t = (t % (period * 2.0)) / period;
    let linear_t = if raw_t > 1.0 { 2.0 - raw_t } else { raw_t };

    // Vertical playhead
    let playhead_x = c.x + pad + 60.0 + track_w * linear_t;
    canvas = canvas
        .line(
            playhead_x,
            c.y + pad,
            playhead_x,
            c.y + pad + n as f32 * track_h,
        )
        .color(C::from_rgba8(255, 255, 255, 40))
        .width(0.5)
        .done();

    for (i, (_name, easing)) in easings.iter().enumerate() {
        let y = c.y + pad + i as f32 * track_h;
        let eased = easing.ease(linear_t);

        // Track background
        canvas = canvas
            .rect(c.x + pad + 60.0, y, track_w, track_h - 1.0)
            .fill(C::from_rgba8(15, 18, 35, 255))
            .done();

        // Dot with trail (5 ghost positions)
        let dot_x = c.x + pad + 60.0 + track_w * eased;
        let dot_y = y + track_h / 2.0;
        let hue = (i as f32 * 360.0 / n as f32 + t * 20.0) % 360.0;
        let dot_color = C::from_hsla(hue, 0.9, 0.6, 1.0);

        // Trails
        for trail in 1..=4 {
            let prev_t = (linear_t - trail as f32 * 0.02).clamp(0.0, 1.0);
            let prev_eased = easing.ease(prev_t);
            let prev_x = c.x + pad + 60.0 + track_w * prev_eased;
            let alpha = 0.3 - trail as f32 * 0.06;
            canvas = canvas
                .circle(prev_x, dot_y, 2.0)
                .fill(dot_color.with_alpha(alpha.max(0.05)))
                .done();
        }

        // Main dot
        canvas = canvas.circle(dot_x, dot_y, 3.0).fill(dot_color).done();
    }

    // Bouncing ball with keyframes at the bottom
    let ball_kf = Keyframes::new(vec![
        Keyframe {
            position: 0.0,
            value: c.y + c.h - pad - 5.0,
            easing: Easing::EaseInQuad,
        },
        Keyframe {
            position: 0.3,
            value: c.y + c.h - pad - 40.0,
            easing: Easing::EaseOutQuad,
        },
        Keyframe {
            position: 0.5,
            value: c.y + c.h - pad - 5.0,
            easing: Easing::EaseInQuad,
        },
        Keyframe {
            position: 0.7,
            value: c.y + c.h - pad - 25.0,
            easing: Easing::EaseOutQuad,
        },
        Keyframe {
            position: 1.0,
            value: c.y + c.h - pad - 5.0,
            easing: Easing::Linear,
        },
    ]);
    let ball_t = (t % 2.0) / 2.0;
    let ball_y = ball_kf.value_at(ball_t);
    let ball_x = c.x + pad + 30.0 + (c.w - pad * 2.0 - 60.0) * ball_t;
    canvas = canvas
        .circle(ball_x, ball_y, 5.0)
        .fill(C::from_rgba8(255, 180, 50, 230))
        .done();

    canvas
}

// ═══════════════════════════════════════════════════════════════════
// Chart panel builders (⑤–⑩)
// ═══════════════════════════════════════════════════════════════════

fn build_live_lines(data_a: &[f64], data_b: &[f64]) -> Chart {
    if data_a.len() < 2 {
        return Chart::line(&[0.0, 1.0]).title("Initializing...").build();
    }
    Chart::line(data_a)
        .title("⑤ Mission Control Telemetry")
        .x_label("Time")
        .y_label("Signal")
        .add_series(Series::new("Sensor B", data_b.to_vec()))
        .filled()
        .with_points()
        .smooth()
        .line_width(2.0)
        .y_range(0.0, 100.0)
        .h_line(50.0)
        .theme(Theme::ocean())
        .build()
}

fn build_scatter_matrix(t: f32) -> Chart {
    let n = 30;
    let x: Vec<f64> = (0..n)
        .map(|i| {
            30.0 + 10.0 * ((hash_u64(i as u64 * 31 + 42) % 1000) as f64 / 1000.0 - 0.5)
                + 0.5 * (t as f64 + i as f64 * 0.1).sin()
        })
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            70.0 + 10.0 * ((hash_u64(i as u64 * 97 + 13) % 1000) as f64 / 1000.0 - 0.5)
                + 0.5 * (t as f64 + i as f64 * 0.2).cos()
        })
        .collect();

    Chart::scatter(&x, &y)
        .title("⑥ Particle Taxonomy")
        .x_label("x")
        .y_label("y")
        .marker(Marker::Diamond)
        .size(6.0)
        .connected()
        .theme(Theme::dark())
        .build()
}

fn build_stat_theater(stacked: bool, _t: f32) -> Chart {
    let labels: Vec<String> = vec!["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
        .into_iter()
        .map(String::from)
        .collect();
    let vals1 = vec![42.0, 67.0, 53.0, 78.0, 35.0];
    let vals2 = vec![38.0, 45.0, 62.0, 51.0, 70.0];

    let mut builder = Chart::bar(labels, &vals1)
        .title("⑦ Statistical Theater")
        .y_label("Value")
        .add_series(Series::new("Series B", vals2))
        .corner_radius(3.0)
        .gap(0.25)
        .show_values()
        .h_line(50.0)
        .theme(Theme::pastel());

    if stacked {
        builder = builder.stacked();
    }
    builder.build()
}

fn build_thermal_pie(donut: bool, t: f32) -> Chart {
    let labels: Vec<String> = vec!["Chrome", "Firefox", "Safari", "Edge", "Other"]
        .into_iter()
        .map(String::from)
        .collect();
    let vals = vec![65.0, 12.0, 9.0, 8.0, 6.0];

    let mut builder = Chart::pie(labels, &vals)
        .title("⑧ Propulsion Display")
        .start_angle_degrees(t * 15.0);

    if donut {
        builder = builder.donut(0.45).hide_percentages();
    }
    builder.build()
}

fn build_theme_gallery() -> Chart {
    // Simple line chart rendered with ocean theme for the gallery slot
    let data: Vec<f64> = (0..20)
        .map(|i| 3.0 + (i as f64 * 0.5).sin() * 2.0 + (i as f64 * 0.2).cos())
        .collect();

    Chart::line(&data)
        .title("⑨ Theme: Ocean")
        .filled()
        .with_points()
        .theme(Theme::ocean())
        .build()
}

fn build_interactive(_t: f32) -> Chart {
    let n = 40;
    let x: Vec<f64> = (0..n)
        .map(|i| 50.0 + 30.0 * ((hash_u64(i as u64 * 53 + 7) % 1000) as f64 / 1000.0 - 0.5))
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 50.0 + 30.0 * ((hash_u64(i as u64 * 79 + 3) % 1000) as f64 / 1000.0 - 0.5))
        .collect();

    Chart::scatter(&x, &y)
        .title("⑩ Tactical Display")
        .x_label("Bearing")
        .y_label("Range")
        .marker(Marker::Circle)
        .size(5.0)
        .theme(Theme::dark())
        .build()
}
