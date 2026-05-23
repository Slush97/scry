//! Showcase TUI — demonstrates scry-engine's full visual range.
//!
//! A multi-panel interactive demo featuring animations, gradients, springs,
//! Oklab color interpolation, composable vector shapes, and a live chart
//! powered by scry-chart — all rendered at pixel resolution inside a
//! Ratatui layout.
//!
//! Run with: `cargo run --example showcase_tui`

use std::io::stdout;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::widgets::Paragraph;

use scry_engine::prelude::*;
use scry_engine::scene::style::{
    DashPattern, GradientDef, GradientKind, GradientStop, Point, Transform,
};
use scry_engine::scene::PixelCanvas;
use scry_engine::style::Color as C;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = ratatui::Terminal::new(ratatui::backend::CrosstermBackend::new(stdout()))?;

    // Single state for the composited engine panels
    let mut state = PixelCanvasState::auto();
    let font = state.font_size();

    // Separate state for the chart panel (ChartWidget owns its own PixelCanvasState)
    let mut chart_state = scry_chart::prelude::ChartState::auto();

    let start = Instant::now();
    let mut spring = Spring::new(0.0_f32, 1.0_f32, SpringConfig::BOUNCY);
    let mut spring_target_high = true;
    let mut spring_timer = Duration::ZERO;
    let mut chart_history: Vec<f64> = Vec::new();

    loop {
        let now = Instant::now();
        let t = now.duration_since(start).as_secs_f32();
        let dt = Duration::from_millis(16);

        // Advance spring, retarget every 3s
        spring.advance(dt);
        spring_timer += dt;
        if spring_timer > Duration::from_millis(3000) {
            spring_timer = Duration::ZERO;
            spring_target_high = !spring_target_high;
            spring.retarget(if spring_target_high { 1.0 } else { 0.0 });
        }

        // Update chart data — composite signal
        let sample = 35.0
            + 20.0 * (t as f64 * 0.8).sin()
            + 10.0 * (t as f64 * 2.1).sin()
            + 5.0 * (t as f64 * 5.3).cos();
        chart_history.push(sample);
        if chart_history.len() > 80 {
            chart_history.remove(0);
        }

        terminal.draw(|frame| {
            let outer =
                Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).split(frame.area());

            let top_bottom =
                Layout::vertical([Constraint::Percentage(55), Constraint::Percentage(45)])
                    .split(outer[0]);

            let top_row = Layout::horizontal([
                Constraint::Percentage(40),
                Constraint::Percentage(30),
                Constraint::Percentage(30),
            ])
            .split(top_bottom[0]);

            let bottom_row =
                Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .split(top_bottom[1]);

            // ── Engine panels: one canvas covering everything except bottom-right ──
            // The engine canvas spans the full content area. The bottom-right
            // is left transparent — the chart widget renders there separately.
            let engine_area = outer[0];
            let mut canvas = PixelCanvas::from_area(engine_area, font)
                .background(C::from_rgba8(12, 11, 18, 255));

            let to_px = |r: Rect| -> (f32, f32, f32, f32) {
                let x = (r.x - engine_area.x) as f32 * font.width as f32;
                let y = (r.y - engine_area.y) as f32 * font.height as f32;
                let w = r.width as f32 * font.width as f32;
                let h = r.height as f32 * font.height as f32;
                (x, y, w, h)
            };

            // Top row panels
            canvas = draw_orbital(canvas, to_px(top_row[0]), t);
            canvas = draw_colors(canvas, to_px(top_row[1]), t);
            canvas = draw_spring(canvas, to_px(top_row[2]), &spring);

            // Bottom-left: shape zoo
            canvas = draw_shapes(canvas, to_px(bottom_row[0]), t);

            // Panel borders (subtle separators)
            let cw = canvas.width() as f32;
            let border = C::from_rgba8(40, 36, 55, 100);
            let (_, split_y, _, _) = to_px(top_bottom[1]);
            canvas = canvas
                .line(0.0, split_y, cw, split_y)
                .color(border)
                .width(1.0)
                .done();
            for col in &top_row[1..] {
                let (x, _, _, _) = to_px(*col);
                canvas = canvas
                    .line(x, 0.0, x, split_y)
                    .color(border)
                    .width(1.0)
                    .done();
            }
            let (mid_x, _, _, _) = to_px(bottom_row[1]);
            let ch = canvas.height() as f32;
            canvas = canvas
                .line(mid_x, split_y, mid_x, ch)
                .color(border)
                .width(1.0)
                .done();

            frame.render_stateful_widget(
                PixelCanvasWidget::new(canvas).skip_cache(),
                engine_area,
                &mut state,
            );

            // ── Bottom-right: live chart via scry-chart ──
            if chart_history.len() >= 2 {
                let chart = scry_chart::prelude::Charts::line(&chart_history)
                    .title("Signal")
                    .theme(scry_chart::prelude::Theme::dark())
                    .filled()
                    .smooth()
                    .y_range(0.0, 80.0)
                    .build();

                frame.render_stateful_widget(
                    scry_chart::prelude::ChartWidget::new(&chart),
                    bottom_row[1],
                    &mut chart_state,
                );
            }

            // ── Status bar ──
            let status = Paragraph::new(format!(
                " scry-engine + scry-chart showcase | {t:.1}s | {:?} | q to quit",
                state.backend_kind()
            ))
            .style(Style::default().fg(ratatui::style::Color::DarkGray));
            frame.render_widget(status, outer[1]);
        })?;
        state.flush()?;
        chart_state.flush()?;

        if event::poll(Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press && key.code == KeyCode::Char('q') {
                    break;
                }
            }
        }
    }

    state.cleanup();
    chart_state.cleanup();
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Panel drawing functions
// ─────────────────────────────────────────────────────────────────────

/// Panel 1: Orbital animation — rotating bodies on concentric rings.
#[allow(clippy::cast_precision_loss)]
fn draw_orbital(
    mut canvas: PixelCanvas,
    (px, py, pw, ph): (f32, f32, f32, f32),
    t: f32,
) -> PixelCanvas {
    let cx = px + pw / 2.0;
    let cy = py + ph / 2.0;
    let scale = (pw.min(ph) / 300.0).min(1.0);

    // Central pulsing core
    let pulse = 0.5 + 0.5 * (t * 1.2).sin();
    let core_r = (10.0 + 5.0 * pulse) * scale;
    canvas = canvas
        .circle(cx, cy, core_r + 8.0 * scale)
        .fill(C::from_rgba8(100, 60, 200, 40))
        .done()
        .circle(cx, cy, core_r)
        .fill(C::from_rgba8(180, 120, 255, 220))
        .done();

    // Orbit rings
    let orbits = [55.0 * scale, 90.0 * scale, 125.0 * scale];
    for &r in &orbits {
        canvas = canvas
            .circle(cx, cy, r)
            .stroke(C::from_rgba8(55, 45, 75, 50), 0.7)
            .done();
    }

    // Orbiting bodies
    let bodies: &[(f32, f32, f32, C, f32)] = &[
        (
            orbits[0],
            0.6,
            7.0 * scale,
            C::from_rgba8(255, 100, 120, 230),
            0.0,
        ),
        (
            orbits[0],
            0.6,
            4.0 * scale,
            C::from_rgba8(255, 160, 100, 200),
            std::f32::consts::PI,
        ),
        (
            orbits[1],
            0.35,
            9.0 * scale,
            C::from_rgba8(80, 200, 255, 230),
            0.5,
        ),
        (
            orbits[1],
            0.35,
            5.0 * scale,
            C::from_rgba8(120, 255, 180, 200),
            2.5,
        ),
        (
            orbits[2],
            0.2,
            6.0 * scale,
            C::from_rgba8(255, 220, 80, 220),
            1.0,
        ),
        (
            orbits[2],
            0.2,
            3.5 * scale,
            C::from_rgba8(255, 130, 200, 180),
            4.0,
        ),
    ];

    for &(orbit_r, speed, size, color, phase) in bodies {
        let angle = t * speed + phase;
        let bx = cx + orbit_r * angle.cos();
        let by = cy + orbit_r * angle.sin();

        canvas = canvas
            .circle(bx, by, size + 4.0 * scale)
            .fill(color.with_alpha(0.15))
            .done()
            .circle(bx, by, size)
            .fill(color)
            .done();
    }

    // Orbiting star
    let star_angle = t * 0.15;
    let star_x = cx + orbits[2] * star_angle.cos();
    let star_y = cy + orbits[2] * star_angle.sin();
    let star_rotation = Transform::rotate_at(t * 0.8, star_x, star_y);
    canvas = canvas
        .star(star_x, star_y, 13.0 * scale, 5.0 * scale, 5)
        .fill(C::from_rgba8(255, 240, 100, 240))
        .stroke(C::from_rgba8(255, 200, 50, 255), 1.5)
        .transform(star_rotation)
        .anti_alias(true)
        .done();

    canvas
}

/// Panel 2: Animated color spectrum with Oklab mixing.
#[allow(clippy::cast_precision_loss)]
fn draw_colors(
    mut canvas: PixelCanvas,
    (px, py, pw, ph): (f32, f32, f32, f32),
    t: f32,
) -> PixelCanvas {
    let pad = 8.0;
    let hue_offset = (t * 15.0) % 360.0;
    let bar_count = 10;
    let bar_h = (ph - pad * 2.0) / bar_count as f32;

    for i in 0..bar_count {
        let y = py + pad + i as f32 * bar_h;
        let hue1 = (hue_offset + i as f32 * 36.0) % 360.0;
        let hue2 = (hue1 + 120.0) % 360.0;
        let c1 = C::from_hsl(hue1, 0.8, 0.55);
        let c2 = C::from_hsl(hue2, 0.8, 0.55);

        canvas = canvas
            .gradient(px + pad, y, pw - pad * 2.0, bar_h - 2.0)
            .linear(Point::new(px + pad, y), Point::new(px + pw - pad, y))
            .stop(0.0, c1)
            .stop(0.5, c1.mix(c2, 0.5))
            .stop(1.0, c2)
            .done();
    }

    // Sweep line
    let sweep_x = px + pad + (pw - pad * 2.0) * (0.5 + 0.5 * (t * 0.4).sin());
    canvas = canvas
        .line(sweep_x, py + pad, sweep_x, py + ph - pad)
        .color(C::from_rgba8(255, 255, 255, 150))
        .width(1.5)
        .done();

    // Sample circle
    let sample_hue = (hue_offset + 180.0) % 360.0;
    canvas = canvas
        .circle(sweep_x, py + ph / 2.0, 12.0)
        .fill(C::from_hsl(sample_hue, 0.9, 0.6))
        .stroke(C::WHITE, 1.5)
        .done();

    canvas
}

/// Panel 3: Spring physics — centered oscillation with coil and velocity coloring.
#[allow(clippy::cast_precision_loss)]
fn draw_spring(
    mut canvas: PixelCanvas,
    (px, py, pw, ph): (f32, f32, f32, f32),
    spring: &Spring<f32>,
) -> PixelCanvas {
    let cx = px + pw / 2.0;
    let cy = py + ph / 2.0;
    let pad = 16.0;

    // Spring value 0..1 mapped to vertical displacement from center.
    // 0.0 = top position, 1.0 = bottom position, 0.5 = center.
    let val = spring.value();
    let max_travel = (ph / 2.0 - pad - 14.0).max(10.0); // leave room for ball radius
    let displacement = (val - 0.5) * 2.0 * max_travel; // -max_travel..+max_travel
    let ball_y = cy + displacement;

    // Anchor point at center
    let anchor_y = cy;

    // Track (dashed vertical line spanning the travel range)
    let track_top = cy - max_travel - 10.0;
    let track_bot = cy + max_travel + 10.0;
    canvas = canvas
        .line(cx, track_top, cx, track_bot)
        .color(C::from_rgba8(50, 70, 50, 120))
        .width(1.0)
        .dash(DashPattern::pair(4.0, 3.0))
        .done();

    // Center marker
    canvas = canvas
        .line(cx - 14.0, cy, cx + 14.0, cy)
        .color(C::from_rgba8(80, 80, 100, 80))
        .width(1.0)
        .done();

    // Anchor dot
    canvas = canvas
        .circle(cx, anchor_y, 4.0)
        .fill(C::from_rgba8(150, 150, 170, 200))
        .done();

    // Spring coil between anchor and ball
    let coil_segments = 20;
    let coil_amp = 10.0;
    let spring_len = (ball_y - anchor_y).abs();
    let dir = if ball_y >= anchor_y { 1.0 } else { -1.0 };
    if spring_len > 2.0 {
        for i in 0..coil_segments {
            let fa = i as f32 / coil_segments as f32;
            let fb = (i + 1) as f32 / coil_segments as f32;
            let ya = anchor_y + dir * spring_len * fa;
            let yb = anchor_y + dir * spring_len * fb;
            let xa = cx + coil_amp * (fa * std::f32::consts::TAU * 4.0).sin();
            let xb = cx + coil_amp * (fb * std::f32::consts::TAU * 4.0).sin();
            canvas = canvas
                .line(xa, ya, xb, yb)
                .color(C::from_rgba8(100, 200, 120, 170))
                .width(1.5)
                .done();
        }
    }

    // Ball — color shifts green→red with velocity
    let velocity = spring.velocity().abs();
    let ball_color = C::from_rgba8(80, 220, 120, 255)
        .mix(C::from_rgba8(255, 80, 80, 255), (velocity * 0.4).min(1.0));

    canvas = canvas
        .circle(cx, ball_y, 12.0)
        .fill(ball_color.with_alpha(0.2))
        .done()
        .circle(cx, ball_y, 8.0)
        .fill(ball_color)
        .stroke(C::WHITE.with_alpha(0.4), 1.0)
        .done();

    // Velocity indicator bars on sides
    let bar_h = (velocity * 25.0).min(max_travel);
    if bar_h > 1.0 {
        let bar_alpha = (velocity * 180.0).min(255.0) as u8;
        let bar_color = C::from_rgba8(255, 200, 80, bar_alpha);
        canvas = canvas
            .rect(px + 6.0, ball_y - bar_h / 2.0, 4.0, bar_h)
            .fill(bar_color)
            .corner_radius(2.0)
            .done()
            .rect(px + pw - 10.0, ball_y - bar_h / 2.0, 4.0, bar_h)
            .fill(bar_color)
            .corner_radius(2.0)
            .done();
    }

    canvas
}

/// Panel 5: Shape zoo — every primitive with gentle animation.
#[allow(clippy::cast_precision_loss)]
fn draw_shapes(
    mut canvas: PixelCanvas,
    (px, py, pw, ph): (f32, f32, f32, f32),
    t: f32,
) -> PixelCanvas {
    let cols = 3;
    let rows = 2;
    let cell_w = pw / cols as f32;
    let cell_h = ph / rows as f32;
    let pad = 8.0;

    let cell = |col: usize, row: usize| -> (f32, f32) {
        (
            px + cell_w * col as f32 + cell_w / 2.0,
            py + cell_h * row as f32 + cell_h / 2.0,
        )
    };
    let r = (cell_w.min(cell_h) / 2.0 - pad).min(38.0);

    // 1. Rotating rounded rect
    let (cx, cy) = cell(0, 0);
    let rect_rot = Transform::rotate_at(t * 0.4, cx, cy);
    canvas = canvas
        .rect(cx - r * 0.7, cy - r * 0.5, r * 1.4, r * 1.0)
        .fill(C::from_rgba8(100, 140, 255, 200))
        .corner_radius(8.0)
        .stroke(C::from_rgba8(160, 190, 255, 255), 1.5)
        .transform(rect_rot)
        .anti_alias(true)
        .done();

    // 2. Pulsing ellipse
    let (cx, cy) = cell(1, 0);
    let erx = r * (0.8 + 0.15 * (t * 1.5).sin());
    let ery = r * (0.6 + 0.15 * (t * 1.5).cos());
    canvas = canvas
        .ellipse(cx, cy, erx, ery)
        .fill(C::from_hsl((t * 20.0) % 360.0, 0.7, 0.5).with_alpha(0.7))
        .stroke(C::WHITE.with_alpha(0.3), 1.0)
        .anti_alias(true)
        .done();

    // 3. Spinning star
    let (cx, cy) = cell(2, 0);
    let star_rot = Transform::rotate_at(t * 0.6, cx, cy);
    let star_color = C::from_hsl((t * 25.0) % 360.0, 0.85, 0.6);
    canvas = canvas
        .star(cx, cy, r, r * 0.4, 6)
        .fill(star_color.with_alpha(0.8))
        .stroke(star_color.with_lightness(1.4), 1.5)
        .transform(star_rot)
        .anti_alias(true)
        .done();

    // 4. Animated arc sweep
    let (cx, cy) = cell(0, 1);
    let sweep = std::f32::consts::PI * (1.0 + (t * 0.3).sin());
    let start_angle = t * 0.25;
    canvas = canvas
        .arc(cx, cy, r * 0.8, start_angle, sweep)
        .stroke(C::from_rgba8(255, 160, 80, 255), 3.5)
        .anti_alias(true)
        .done()
        .arc(cx, cy, r * 0.5, -start_angle, -sweep * 0.7)
        .stroke(C::from_rgba8(80, 200, 255, 200), 2.5)
        .anti_alias(true)
        .done();

    // 5. Radial gradient sphere
    let (cx, cy) = cell(1, 1);
    canvas = canvas
        .circle(cx, cy, r)
        .fill_radial_gradient(GradientDef {
            kind: GradientKind::Radial {
                center: Point::new(cx - r * 0.2, cy - r * 0.2),
                radius: r * 1.2,
            },
            stops: vec![
                GradientStop {
                    position: 0.0,
                    color: C::from_hsl((t * 25.0) % 360.0, 0.9, 0.75),
                },
                GradientStop {
                    position: 0.5,
                    color: C::from_hsl((t * 25.0 + 120.0) % 360.0, 0.8, 0.4),
                },
                GradientStop {
                    position: 1.0,
                    color: C::from_rgba8(20, 10, 30, 255),
                },
            ],
        })
        .anti_alias(true)
        .done();

    // 6. Morphing dashed polygon
    let (cx, cy) = cell(2, 1);
    let sides = 5;
    let morph = 0.5 + 0.5 * (t * 0.6).sin();
    let inner = r * (0.3 + 0.4 * morph);
    let pts: Vec<(f32, f32)> = (0..sides * 2)
        .map(|i| {
            let angle = std::f32::consts::TAU * i as f32 / (sides * 2) as f32
                - std::f32::consts::FRAC_PI_2
                + t * 0.15;
            let rad = if i % 2 == 0 { r * 0.9 } else { inner };
            (cx + rad * angle.cos(), cy + rad * angle.sin())
        })
        .collect();
    canvas = canvas
        .polygon(pts)
        .stroke(C::from_rgba8(200, 180, 255, 220), 2.0)
        .fill(C::from_rgba8(120, 80, 200, 40))
        .dash(DashPattern::pair(7.0, 4.0))
        .anti_alias(true)
        .done();

    canvas
}
