//! **Mission Control** — space telemetry simulation.
//!
//! The flagship hero example for `scry-engine`. A spacecraft orbits a
//! planet while real-time telemetry, fuel gauges, attitude indicators, and
//! communications waveforms update across six synchronized panels.
//!
//! **API coverage:** This single example exercises *every* drawing primitive and
//! compositing feature — circles, rectangles, lines, ellipses, arcs, polygons,
//! polylines, Bézier paths, gradients (linear & radial), groups (transforms,
//! clipping, opacity, blend modes), `ImageData`, `DashPattern`,
//! `LineCap`/`LineJoin`, and ratatui widget composability.
//!
//! | Key     | Action               |
//! |---------|----------------------|
//! | `q`     | Quit                 |
//! | `Space` | Pause / resume       |
//! | `+`/`=` | Speed up time        |
//! | `-`     | Slow down time       |
//! | `r`     | Reset orbit          |
//!
//! Run with: `cargo run --example mission_control --features widget`

#![allow(
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::too_many_arguments
)]

use std::collections::VecDeque;
use std::io::stdout;
use std::time::Instant;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use scry_engine::prelude::{Picker, PixelCanvasState, PixelCanvasWidget, ProtocolKind};
use scry_engine::scene::command::ImageData;
use scry_engine::scene::style::{
    BlendMode, Color as C, DashPattern, GradientDef, GradientKind, GradientStop, LineCap, LineJoin,
    Point, Rect as PxRect, Transform,
};
use scry_engine::scene::PixelCanvas;
use scry_engine::transport;

// ─────────────────────────────────────────────────────────────────────────────
// Color Palette — NASA-inspired deep space
// ─────────────────────────────────────────────────────────────────────────────

const BG: C = C {
    r: 0.043,
    g: 0.063,
    b: 0.149,
    a: 1.0,
}; // #0B1026
const GRID: C = C {
    r: 0.102,
    g: 0.125,
    b: 0.251,
    a: 1.0,
}; // #1A2040
const CYAN: C = C {
    r: 0.0,
    g: 0.831,
    b: 1.0,
    a: 1.0,
}; // #00D4FF
const AMBER: C = C {
    r: 1.0,
    g: 0.6,
    b: 0.0,
    a: 1.0,
}; // #FF9900
const GREEN: C = C {
    r: 0.0,
    g: 1.0,
    b: 0.533,
    a: 1.0,
}; // #00FF88
const RED_ALERT: C = C {
    r: 1.0,
    g: 0.2,
    b: 0.4,
    a: 1.0,
}; // #FF3366
const TEXT_DIM: C = C {
    r: 0.533,
    g: 0.565,
    b: 0.690,
    a: 1.0,
}; // #8890B0
const TEXT_BRIGHT: C = C {
    r: 0.753,
    g: 0.784,
    b: 0.878,
    a: 1.0,
}; // #C0C8E0

// ─────────────────────────────────────────────────────────────────────────────
// Panel bounds helper
// ─────────────────────────────────────────────────────────────────────────────

struct Panel {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

impl Panel {
    fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }
    fn cx(&self) -> f32 {
        self.x + self.w / 2.0
    }
    fn cy(&self) -> f32 {
        self.y + self.h / 2.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Simulation State
// ─────────────────────────────────────────────────────────────────────────────

struct SimState {
    time: f64,
    paused: bool,
    speed: f64,

    // Orbital parameters
    true_anomaly: f64,
    semi_major: f64,
    eccentricity: f64,
    orbit_tilt: f64,

    // Telemetry ring buffers
    alt_history: VecDeque<f32>,
    vel_history: VecDeque<f32>,

    // Subsystems
    fuel: f32,
    roll: f32,
    pitch: f32,
    power: f32,
    thermal: f32,
    comms_strength: f32,

    // Star field (fractional positions + brightness)
    stars: Vec<(f32, f32, f32)>,
}

impl SimState {
    fn new() -> Self {
        // Deterministic star field
        let mut stars = Vec::with_capacity(40);
        let mut seed: u32 = 42;
        for _ in 0..40 {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let x = (seed >> 16) as f32 / 65535.0;
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let y = (seed >> 16) as f32 / 65535.0;
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let b = 0.3 + (seed >> 16) as f32 / 65535.0 * 0.7;
            stars.push((x, y, b));
        }

        Self {
            time: 0.0,
            paused: false,
            speed: 1.0,
            true_anomaly: 0.0,
            semi_major: 0.35,
            eccentricity: 0.3,
            orbit_tilt: 0.25,
            alt_history: VecDeque::from(vec![200.0; 120]),
            vel_history: VecDeque::from(vec![7.5; 120]),
            fuel: 100.0,
            roll: 0.0,
            pitch: 0.0,
            power: 85.0,
            thermal: 40.0,
            comms_strength: 90.0,
            stars,
        }
    }

    fn update(&mut self, dt: f64) {
        if self.paused {
            return;
        }
        let dt = dt * self.speed;
        self.time += dt;
        let t = self.time;

        // Kepler-like orbit: variable speed (faster near periapsis)
        let r_factor = 1.0 + self.eccentricity * (self.true_anomaly as f32).cos() as f64;
        self.true_anomaly += dt * 0.4 * r_factor;

        // Altitude varies with orbit
        let r = self.semi_major * (1.0 - self.eccentricity.powi(2))
            / (1.0 + self.eccentricity * self.true_anomaly.cos());
        let altitude = (r * 1000.0 - 200.0) as f32;
        self.alt_history.push_back(altitude.clamp(50.0, 500.0));
        if self.alt_history.len() > 120 {
            self.alt_history.pop_front();
        }

        // Velocity (faster at periapsis)
        let vel = (7.8 + 2.0 * (1.0 / r - 0.5 / self.semi_major)) as f32;
        self.vel_history.push_back(vel.clamp(5.0, 12.0));
        if self.vel_history.len() > 120 {
            self.vel_history.pop_front();
        }

        // Fuel depletes slowly
        self.fuel = (self.fuel - dt as f32 * 0.15).max(0.0);

        // Attitude oscillation (damped multi-frequency)
        self.roll = 0.12 * (t * 0.7).sin() as f32
            + 0.06 * (t * 1.9).sin() as f32
            + 0.03 * (t * 4.3).cos() as f32;
        self.pitch = 0.08 * (t * 0.5).cos() as f32 + 0.04 * (t * 2.1).sin() as f32;

        // Power oscillates (solar panel angle)
        self.power =
            (85.0 + 10.0 * (t * 0.3).sin() as f32 + 5.0 * (t * 1.1).cos() as f32).clamp(0.0, 100.0);

        // Thermal increases near periapsis
        self.thermal =
            (40.0 + 25.0 * (self.true_anomaly * 0.5).sin() as f32 + 8.0 * (t * 0.8).sin() as f32)
                .clamp(0.0, 100.0);

        // Comms strength varies with position
        self.comms_strength =
            (90.0 + 8.0 * (t * 0.4).cos() as f32 - 5.0 * (t * 1.7).sin() as f32).clamp(0.0, 100.0);
    }

    fn reset(&mut self) {
        self.true_anomaly = 0.0;
        self.fuel = 100.0;
        self.time = 0.0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

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
    let mut sim = SimState::new();
    let mut last_frame = Instant::now();

    loop {
        let now = Instant::now();
        let dt = now.duration_since(last_frame).as_secs_f64();
        last_frame = now;
        sim.update(dt);

        terminal.draw(|frame| {
            let outer = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(3)])
                .split(frame.area());

            let canvas = build_scene(outer[0], &px_state, &sim);
            frame.render_stateful_widget(
                PixelCanvasWidget::new(canvas).skip_cache(),
                outer[0],
                &mut px_state,
            );

            // ── Status bar (ratatui composability) ──
            let met = format_met(sim.time);
            let speed_str = if sim.paused {
                "PAUSED".to_string()
            } else {
                format!("{:.0}×", sim.speed)
            };
            let status = Paragraph::new(format!(
                " ◈ MISSION CONTROL │ MET {met} │ {speed_str} │ FUEL {:.0}% │ q:quit  space:pause  +/-:speed  r:reset",
                sim.fuel
            ))
            .style(Style::default().fg(Color::Rgb(136, 144, 176)))
            .block(
                Block::default()
                    .borders(Borders::TOP)
                    .border_style(Style::default().fg(Color::Rgb(26, 32, 64))),
            );
            frame.render_widget(status, outer[1]);
        })?;
        px_state.flush()?;

        if event::poll(std::time::Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Char(' ') => sim.paused = !sim.paused,
                        KeyCode::Char('+' | '=') => {
                            sim.speed = (sim.speed * 1.5).min(16.0);
                        }
                        KeyCode::Char('-') => {
                            sim.speed = (sim.speed / 1.5).max(0.25);
                        }
                        KeyCode::Char('r') => sim.reset(),
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

fn format_met(seconds: f64) -> String {
    let total = seconds as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    format!("{h:02}:{m:02}:{s:02}")
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene builder — layout + dispatch
// ─────────────────────────────────────────────────────────────────────────────

fn build_scene(area: Rect, px_state: &PixelCanvasState, sim: &SimState) -> PixelCanvas {
    let font = px_state.font_size();
    let w = u32::from(area.width) * u32::from(font.width);
    let h = u32::from(area.height) * u32::from(font.height);
    let wf = w as f32;
    let hf = h as f32;

    let mut canvas = PixelCanvas::new(w, h).background(BG);

    // Layout: top row 65% height, bottom row 35% height
    let top_h = hf * 0.65;
    let bot_h = hf - top_h;

    // Top: orbital view (60%) | right panels (40%)
    let orb_w = wf * 0.6;
    let right_w = wf - orb_w;

    let p_orbital = Panel::new(0.0, 0.0, orb_w, top_h);
    let p_telemetry = Panel::new(orb_w, 0.0, right_w, top_h * 0.55);
    let p_fuel = Panel::new(orb_w, top_h * 0.55, right_w, top_h * 0.45);

    // Bottom: attitude (33%) | systems (34%) | comms (33%)
    let bot_w = wf / 3.0;
    let p_attitude = Panel::new(0.0, top_h, bot_w, bot_h);
    let p_systems = Panel::new(bot_w, top_h, bot_w, bot_h);
    let p_comms = Panel::new(bot_w * 2.0, top_h, wf - bot_w * 2.0, bot_h);

    // Draw panel dividers
    let div_color = GRID;
    canvas = canvas
        .line(orb_w, 0.0, orb_w, top_h)
        .color(div_color)
        .width(1.0)
        .done()
        .line(0.0, top_h, wf, top_h)
        .color(div_color)
        .width(1.0)
        .done()
        .line(orb_w, top_h * 0.55, wf, top_h * 0.55)
        .color(div_color)
        .width(1.0)
        .done()
        .line(bot_w, top_h, bot_w, hf)
        .color(div_color)
        .width(1.0)
        .done()
        .line(bot_w * 2.0, top_h, bot_w * 2.0, hf)
        .color(div_color)
        .width(1.0)
        .done();

    // Draw each panel
    canvas = draw_orbital_view(canvas, &p_orbital, sim);
    canvas = draw_telemetry(canvas, &p_telemetry, sim);
    canvas = draw_fuel_gauge(canvas, &p_fuel, sim);
    canvas = draw_attitude(canvas, &p_attitude, sim);
    canvas = draw_systems(canvas, &p_systems, sim);
    canvas = draw_comms(canvas, &p_comms, sim);

    canvas
}

// ═════════════════════════════════════════════════════════════════════════════
// Panel 1: Orbital View
// ═════════════════════════════════════════════════════════════════════════════

fn draw_orbital_view(canvas: PixelCanvas, p: &Panel, sim: &SimState) -> PixelCanvas {
    let mut canvas = canvas;
    let pad = 8.0;
    let cx = p.cx();
    let cy = p.cy() + pad;

    // ── Star field ──
    for &(fx, fy, brightness) in &sim.stars {
        let sx = p.x + pad + fx * (p.w - pad * 2.0);
        let sy = p.y + pad + fy * (p.h - pad * 2.0);
        let r = 0.5 + brightness * 1.0;
        let alpha = 0.4 + brightness * 0.6;
        canvas = canvas
            .circle(sx, sy, r)
            .fill(C::from_rgba(1.0, 1.0, brightness * 0.3 + 0.7, alpha))
            .done();
    }

    // ── Planet — radial gradient ──
    let planet_r = p.w.min(p.h) * 0.18;
    canvas = canvas
        .circle(cx, cy, planet_r)
        .fill_radial_gradient(GradientDef {
            kind: GradientKind::Radial {
                center: Point::new(cx - planet_r * 0.3, cy - planet_r * 0.3),
                radius: planet_r * 1.3,
            },
            stops: vec![
                GradientStop {
                    position: 0.0,
                    color: C::from_rgba8(80, 160, 220, 255),
                },
                GradientStop {
                    position: 0.4,
                    color: C::from_rgba8(30, 100, 160, 255),
                },
                GradientStop {
                    position: 0.7,
                    color: C::from_rgba8(15, 60, 100, 255),
                },
                GradientStop {
                    position: 1.0,
                    color: C::from_rgba8(5, 25, 50, 255),
                },
            ],
        })
        .done();

    // ── Atmosphere glow — Screen blend + opacity ──
    canvas = canvas
        .group(Transform::identity())
        .blend_mode(BlendMode::Screen)
        .opacity(0.35)
        .canvas(|inner| {
            inner
                .circle(cx, cy, planet_r + 8.0)
                .fill(C::from_rgba8(60, 180, 255, 255))
                .done()
                .circle(cx, cy, planet_r + 16.0)
                .fill(C::from_rgba8(40, 120, 255, 80))
                .done()
        })
        .done();

    // ── Orbit path — dashed ellipse with tilt ──
    let orbit_rx = p.w * 0.38;
    let orbit_ry = p.h * 0.32;
    let tilt = sim.orbit_tilt as f32;

    canvas = canvas
        .ellipse(cx, cy, orbit_rx, orbit_ry)
        .rotation(tilt)
        .stroke(C::from_rgba8(60, 100, 160, 120), 1.5)
        .dash(DashPattern {
            intervals: vec![8.0, 5.0],
            offset: 0.0,
        })
        .done();

    // ── Spacecraft position on orbit ──
    let nu = sim.true_anomaly as f32;
    let r_orbit = (sim.semi_major * (1.0 - sim.eccentricity.powi(2))
        / (1.0 + sim.eccentricity * sim.true_anomaly.cos())) as f32;
    let orbit_scale = orbit_rx / sim.semi_major as f32;
    let local_x = r_orbit * orbit_scale * nu.cos();
    let local_y = r_orbit * orbit_scale * (orbit_ry / orbit_rx) * nu.sin();

    // Apply orbit tilt
    let ship_x = cx + local_x * tilt.cos() - local_y * tilt.sin();
    let ship_y = cy + local_x * tilt.sin() + local_y * tilt.cos();

    // ── Spacecraft (polygon — triangle) ──
    let heading = nu + tilt + std::f32::consts::FRAC_PI_2;
    let ship_size = 6.0;
    let ship_pts = vec![
        (
            ship_x + ship_size * heading.cos(),
            ship_y + ship_size * heading.sin(),
        ),
        (
            ship_x + ship_size * 0.6 * (heading + 2.4).cos(),
            ship_y + ship_size * 0.6 * (heading + 2.4).sin(),
        ),
        (
            ship_x + ship_size * 0.6 * (heading - 2.4).cos(),
            ship_y + ship_size * 0.6 * (heading - 2.4).sin(),
        ),
    ];
    canvas = canvas.polygon(ship_pts).fill(CYAN).done();

    // ── Thrust exhaust — Bézier path ──
    let exhaust_dir = heading + std::f32::consts::PI;
    let ex1 = ship_x + 8.0 * exhaust_dir.cos();
    let ey1 = ship_y + 8.0 * exhaust_dir.sin();
    let ex2 = ship_x + 22.0 * exhaust_dir.cos() + 4.0 * (sim.time as f32 * 8.0).sin();
    let ey2 = ship_y + 22.0 * exhaust_dir.sin() + 4.0 * (sim.time as f32 * 6.0).cos();

    let mut pb = tiny_skia::PathBuilder::new();
    pb.move_to(ship_x, ship_y);
    pb.quad_to(ex1, ey1, ex2, ey2);
    if let Some(path) = pb.finish() {
        canvas = canvas
            .path(path)
            .stroke(AMBER.with_alpha(0.6), 2.0)
            .line_cap(LineCap::Round)
            .done();
    }

    // ── Orbit prediction arc (ahead of spacecraft) ──
    let pred_points: Vec<(f32, f32)> = (0..30)
        .map(|i| {
            let future_nu = nu + i as f32 * 0.04;
            let fr = (sim.semi_major as f32 * (1.0 - (sim.eccentricity as f32).powi(2)))
                / (1.0 + sim.eccentricity as f32 * future_nu.cos());
            let flx = fr * orbit_scale * future_nu.cos();
            let fly = fr * orbit_scale * (orbit_ry / orbit_rx) * future_nu.sin();
            (
                cx + flx * tilt.cos() - fly * tilt.sin(),
                cy + flx * tilt.sin() + fly * tilt.cos(),
            )
        })
        .collect();

    canvas = canvas
        .polyline(pred_points)
        .stroke(CYAN.with_alpha(0.5), 1.5)
        .dash(DashPattern {
            intervals: vec![3.0, 3.0],
            offset: 0.0,
        })
        .line_cap(LineCap::Round)
        .done();

    // ── Radar sweep arc ──
    let sweep_angle = sim.time as f32 * 0.8;
    canvas = canvas
        .arc(cx, cy, planet_r + 30.0, sweep_angle, 0.4)
        .stroke(GREEN.with_alpha(0.3), 2.0)
        .done();

    // ── Glow dot on spacecraft ──
    let pulse = ((sim.time * 4.0).sin() as f32 * 0.5 + 0.5).clamp(0.0, 1.0);
    canvas = canvas
        .circle(ship_x, ship_y, 4.0 + pulse * 2.0)
        .fill(CYAN.with_alpha(0.2 + pulse * 0.15))
        .done();

    // ── Labels ──
    draw_label(&mut canvas, p.x + pad, p.y + pad, "ORBITAL VIEW", TEXT_DIM);

    canvas
}

// ═════════════════════════════════════════════════════════════════════════════
// Panel 2: Telemetry Strip Charts
// ═════════════════════════════════════════════════════════════════════════════

fn draw_telemetry(canvas: PixelCanvas, p: &Panel, sim: &SimState) -> PixelCanvas {
    let mut canvas = canvas;
    let pad = 8.0;
    let label_h = 12.0;

    // Two charts stacked vertically
    let chart_h = (p.h - pad * 3.0 - label_h) / 2.0;

    // ── Altitude chart (cyan) ──
    let alt_y = p.y + pad + label_h;
    canvas = draw_sparkline(
        canvas,
        p.x + pad,
        alt_y,
        p.w - pad * 2.0,
        chart_h,
        &sim.alt_history,
        50.0,
        500.0,
        CYAN,
    );

    // ── Velocity chart (amber) ──
    let vel_y = alt_y + chart_h + pad;
    canvas = draw_sparkline(
        canvas,
        p.x + pad,
        vel_y,
        p.w - pad * 2.0,
        chart_h,
        &sim.vel_history,
        5.0,
        12.0,
        AMBER,
    );

    // ── Labels ──
    draw_label(&mut canvas, p.x + pad, p.y + pad, "TELEMETRY", TEXT_DIM);

    let alt_val = sim.alt_history.back().copied().unwrap_or(0.0);
    let vel_val = sim.vel_history.back().copied().unwrap_or(0.0);
    draw_label(
        &mut canvas,
        p.x + pad + 80.0,
        p.y + pad,
        &format!("ALT {alt_val:.0}KM  VEL {vel_val:.1}KMS"),
        TEXT_BRIGHT,
    );

    canvas
}

fn draw_sparkline(
    canvas: PixelCanvas,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    data: &VecDeque<f32>,
    min_val: f32,
    max_val: f32,
    color: C,
) -> PixelCanvas {
    let mut canvas = canvas;
    let range = max_val - min_val;

    // Grid lines
    for i in 0..=4 {
        let gy = y + h * (i as f32 / 4.0);
        canvas = canvas
            .line(x, gy, x + w, gy)
            .color(GRID.with_alpha(0.5))
            .width(0.5)
            .done();
    }

    if data.len() < 2 {
        return canvas;
    }

    let n = data.len();
    let points: Vec<(f32, f32)> = data
        .iter()
        .enumerate()
        .map(|(i, &val)| {
            let px = x + w * (i as f32 / (n - 1) as f32);
            let py = y + h * (1.0 - (val - min_val) / range);
            (px, py)
        })
        .collect();

    // Gradient fill under the line
    let mut fill_pts = Vec::with_capacity(points.len() + 2);
    fill_pts.push((x, y + h));
    fill_pts.extend_from_slice(&points);
    fill_pts.push((x + w, y + h));

    canvas = canvas
        .polygon(fill_pts)
        .fill_linear_gradient(GradientDef {
            kind: GradientKind::Linear {
                start: Point::new(x, y),
                end: Point::new(x, y + h),
            },
            stops: vec![
                GradientStop {
                    position: 0.0,
                    color: color.with_alpha(0.35),
                },
                GradientStop {
                    position: 1.0,
                    color: color.with_alpha(0.02),
                },
            ],
        })
        .done();

    // Data line
    canvas = canvas
        .polyline(points.clone())
        .stroke(color, 1.5)
        .line_join(LineJoin::Round)
        .done();

    // Glowing current-value dot
    if let Some(&(lx, ly)) = points.last() {
        canvas = canvas
            .circle(lx, ly, 4.0)
            .fill(color.with_alpha(0.3))
            .done()
            .circle(lx, ly, 2.0)
            .fill(color)
            .done();
    }

    canvas
}

// ═════════════════════════════════════════════════════════════════════════════
// Panel 3: Fuel Gauge
// ═════════════════════════════════════════════════════════════════════════════

fn draw_fuel_gauge(canvas: PixelCanvas, p: &Panel, sim: &SimState) -> PixelCanvas {
    let mut canvas = canvas;
    let pad = 8.0;
    let cx = p.cx();
    let cy = p.cy() + 8.0;
    let radius = (p.w.min(p.h) * 0.36 - pad).max(20.0);

    let start_angle = 135.0_f32.to_radians();
    let full_sweep = 270.0_f32.to_radians();
    let track_color = C::from_rgba8(35, 38, 55, 255);

    // ── Background arc track ──
    let segments = 50;
    for i in 0..segments {
        let a1 = start_angle + full_sweep * (i as f32 / segments as f32);
        let a2 = start_angle + full_sweep * ((i + 1) as f32 / segments as f32);
        canvas = canvas
            .line(
                cx + radius * a1.cos(),
                cy + radius * a1.sin(),
                cx + radius * a2.cos(),
                cy + radius * a2.sin(),
            )
            .color(track_color)
            .width(7.0)
            .line_cap(LineCap::Round)
            .done();
    }

    // ── Value arc — green→yellow→red ──
    let ratio = (sim.fuel / 100.0).clamp(0.0, 1.0);
    let val_sweep = full_sweep * ratio;
    let val_segments = (segments as f32 * ratio).max(1.0) as usize;

    for i in 0..val_segments {
        let frac = i as f32 / val_segments.max(1) as f32;
        let a1 = start_angle + val_sweep * (i as f32 / val_segments as f32);
        let a2 = start_angle + val_sweep * ((i + 1) as f32 / val_segments as f32);

        // Color: green → yellow → red
        let color = if frac < 0.5 {
            GREEN.mix(AMBER, frac * 2.0)
        } else {
            AMBER.mix(RED_ALERT, (frac - 0.5) * 2.0)
        };

        canvas = canvas
            .line(
                cx + radius * a1.cos(),
                cy + radius * a1.sin(),
                cx + radius * a2.cos(),
                cy + radius * a2.sin(),
            )
            .color(color)
            .width(7.0)
            .line_cap(LineCap::Round)
            .done();
    }

    // ── Tick marks ──
    for i in 0..=10 {
        let angle = start_angle + full_sweep * (i as f32 / 10.0);
        let inner = radius + 10.0;
        let outer = if i % 5 == 0 {
            radius + 18.0
        } else {
            radius + 14.0
        };
        canvas = canvas
            .line(
                cx + inner * angle.cos(),
                cy + inner * angle.sin(),
                cx + outer * angle.cos(),
                cy + outer * angle.sin(),
            )
            .color(TEXT_DIM.with_alpha(0.6))
            .width(1.5)
            .line_cap(LineCap::Round)
            .done();
    }

    // ── Center dot ──
    canvas = canvas.circle(cx, cy, 4.0).fill(TEXT_BRIGHT).done();

    // ── Labels ──
    draw_label(&mut canvas, p.x + pad, p.y + pad, "FUEL", TEXT_DIM);
    draw_label(
        &mut canvas,
        cx - 15.0,
        cy - 6.0,
        &format!("{:.0}", sim.fuel),
        if sim.fuel < 20.0 {
            RED_ALERT
        } else {
            TEXT_BRIGHT
        },
    );

    canvas
}

// ═════════════════════════════════════════════════════════════════════════════
// Panel 4: Attitude Indicator (Gyroscope with clipping)
// ═════════════════════════════════════════════════════════════════════════════

fn draw_attitude(canvas: PixelCanvas, p: &Panel, sim: &SimState) -> PixelCanvas {
    let mut canvas = canvas;
    let pad = 8.0;
    let cx = p.cx();
    let cy = p.cy() + 6.0;
    let radius = (p.w.min(p.h) * 0.38 - pad).max(20.0);

    // ── Outer ring ──
    canvas = canvas
        .circle(cx, cy, radius + 2.0)
        .stroke(C::from_rgba8(60, 70, 100, 255), 2.0)
        .done();

    // ── Artificial horizon with circular clip ──
    let clip_path = tiny_skia::PathBuilder::from_circle(cx, cy, radius);
    if let Some(clip) = clip_path {
        let pitch_offset = sim.pitch * radius * 2.0;

        canvas = canvas
            .group(Transform::rotate_at(sim.roll, cx, cy))
            .clip_path(clip)
            .canvas(|inner| {
                // Sky (blue)
                inner
                    .rect(
                        cx - radius,
                        cy - radius * 2.0 + pitch_offset,
                        radius * 2.0,
                        radius * 2.0,
                    )
                    .fill(C::from_rgba8(30, 80, 160, 255))
                    .done()
                    // Ground (brown)
                    .rect(cx - radius, cy + pitch_offset, radius * 2.0, radius * 2.0)
                    .fill(C::from_rgba8(100, 70, 30, 255))
                    .done()
                    // Horizon line
                    .line(
                        cx - radius,
                        cy + pitch_offset,
                        cx + radius,
                        cy + pitch_offset,
                    )
                    .color(C::WHITE)
                    .width(1.5)
                    .done()
                    // Pitch ladder lines
                    .line(
                        cx - radius * 0.3,
                        cy + pitch_offset - 15.0,
                        cx + radius * 0.3,
                        cy + pitch_offset - 15.0,
                    )
                    .color(C::WHITE.with_alpha(0.5))
                    .width(1.0)
                    .done()
                    .line(
                        cx - radius * 0.3,
                        cy + pitch_offset + 15.0,
                        cx + radius * 0.3,
                        cy + pitch_offset + 15.0,
                    )
                    .color(C::WHITE.with_alpha(0.5))
                    .width(1.0)
                    .done()
            })
            .done();
    }

    // ── Fixed aircraft symbol (not rotated) ──
    canvas = canvas
        .line(cx - radius * 0.4, cy, cx - radius * 0.15, cy)
        .color(AMBER)
        .width(2.5)
        .line_cap(LineCap::Round)
        .done()
        .line(cx + radius * 0.15, cy, cx + radius * 0.4, cy)
        .color(AMBER)
        .width(2.5)
        .line_cap(LineCap::Round)
        .done()
        .circle(cx, cy, 3.0)
        .fill(AMBER)
        .done();

    // ── Roll indicator (top arc markers) ──
    for i in 0..7 {
        let angle = -std::f32::consts::FRAC_PI_2 + (i as f32 - 3.0) * 0.3;
        let inner_r = radius - 3.0;
        let outer_r = radius + 1.0;
        canvas = canvas
            .line(
                cx + inner_r * angle.cos(),
                cy + inner_r * angle.sin(),
                cx + outer_r * angle.cos(),
                cy + outer_r * angle.sin(),
            )
            .color(TEXT_DIM.with_alpha(0.8))
            .width(if i == 3 { 2.0 } else { 1.0 })
            .done();
    }

    // ── Label ──
    draw_label(&mut canvas, p.x + pad, p.y + pad, "ATTITUDE", TEXT_DIM);

    canvas
}

// ═════════════════════════════════════════════════════════════════════════════
// Panel 5: Systems Status
// ═════════════════════════════════════════════════════════════════════════════

fn draw_systems(canvas: PixelCanvas, p: &Panel, sim: &SimState) -> PixelCanvas {
    let mut canvas = canvas;
    let pad = 8.0;
    let label_h = 14.0;
    let bar_h = 12.0;
    let spacing = bar_h + 18.0;
    let bar_x = p.x + pad + 60.0;
    let bar_w = p.w - pad * 2.0 - 65.0;

    let systems = [
        ("PWR", sim.power, GREEN),
        (
            "THR",
            sim.thermal,
            if sim.thermal > 70.0 { RED_ALERT } else { AMBER },
        ),
        ("COM", sim.comms_strength, CYAN),
    ];

    for (i, (name, value, color)) in systems.iter().enumerate() {
        let by = p.y + pad + label_h + 8.0 + i as f32 * spacing;
        let ratio = (value / 100.0).clamp(0.0, 1.0);

        // Background track with rounded corners
        canvas = canvas
            .rect(bar_x, by, bar_w, bar_h)
            .fill(C::from_rgba8(25, 28, 45, 255))
            .corner_radius(bar_h / 2.0)
            .done();

        // Value fill with linear gradient
        let fill_w = bar_w * ratio;
        if fill_w > 1.0 {
            canvas = canvas
                .rect(bar_x, by, fill_w, bar_h)
                .fill_linear_gradient(GradientDef {
                    kind: GradientKind::Linear {
                        start: Point::new(bar_x, by),
                        end: Point::new(bar_x + fill_w, by),
                    },
                    stops: vec![
                        GradientStop {
                            position: 0.0,
                            color: color.with_alpha(0.7),
                        },
                        GradientStop {
                            position: 1.0,
                            color: *color,
                        },
                    ],
                })
                .corner_radius(bar_h / 2.0)
                .done();
        }

        // Label
        draw_label(&mut canvas, p.x + pad, by + 2.0, name, TEXT_DIM);

        // Value
        draw_label(
            &mut canvas,
            bar_x + bar_w + 4.0,
            by + 2.0,
            &format!("{value:.0}"),
            TEXT_BRIGHT,
        );
    }

    // ── Warning pulse on thermal ──
    if sim.thermal > 70.0 {
        let pulse = ((sim.time * 3.0).sin() as f32 * 0.5 + 0.5).clamp(0.0, 1.0);
        canvas = canvas
            .group(Transform::identity())
            .opacity(pulse * 0.6)
            .canvas(|inner| {
                let by = p.y + pad + label_h + 8.0 + spacing;
                inner
                    .rect(bar_x - 2.0, by - 2.0, bar_w + 4.0, bar_h + 4.0)
                    .stroke(RED_ALERT, 1.5)
                    .corner_radius(bar_h / 2.0 + 2.0)
                    .done()
            })
            .done();
    }

    // ── Decorative border lines with LineJoin ──
    let border_y = p.y + pad + label_h + 4.0;
    canvas = canvas
        .line(p.x + pad, border_y, p.x + p.w - pad, border_y)
        .color(GRID)
        .width(1.0)
        .line_join(LineJoin::Miter)
        .done();

    draw_label(&mut canvas, p.x + pad, p.y + pad, "SYSTEMS", TEXT_DIM);

    canvas
}

// ═════════════════════════════════════════════════════════════════════════════
// Panel 6: Communications Waveform (+ ImageData noise)
// ═════════════════════════════════════════════════════════════════════════════

fn draw_comms(canvas: PixelCanvas, p: &Panel, sim: &SimState) -> PixelCanvas {
    let mut canvas = canvas;
    let pad = 8.0;
    let label_h = 14.0;
    let wave_x = p.x + pad;
    let wave_y = p.y + pad + label_h + 4.0;
    let wave_w = p.w - pad * 2.0;
    let wave_h = p.h - pad * 2.0 - label_h - 4.0;

    // ── Noise background — ImageData ──
    let noise_w = (wave_w as u32).clamp(4, 256);
    let noise_h = (wave_h as u32).clamp(4, 128);
    let mut noise_data = vec![0u8; (noise_w * noise_h * 4) as usize];
    let mut seed: u32 = (sim.time * 1000.0) as u32;
    for pixel in noise_data.chunks_exact_mut(4) {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        let val = ((seed >> 16) & 0xFF) as u8;
        if val > 240 {
            pixel[0] = 30;
            pixel[1] = 35;
            pixel[2] = 60;
            pixel[3] = 80;
        }
        // else stays transparent (0,0,0,0)
    }
    let noise_img = ImageData::new(noise_w, noise_h, noise_data);
    canvas = canvas.image(noise_img, wave_x, wave_y).opacity(0.6).done();

    // ── Carrier frequency — dashed line ──
    let carrier_y = wave_y + wave_h / 2.0;
    canvas = canvas
        .line(wave_x, carrier_y, wave_x + wave_w, carrier_y)
        .color(GRID.with_alpha(0.6))
        .width(1.0)
        .dash(DashPattern {
            intervals: vec![4.0, 4.0],
            offset: sim.time as f32 * 10.0,
        })
        .done();

    // ── Signal waveform — polyline ──
    let t = sim.time as f32;
    let n = 120;
    let signal: Vec<(f32, f32)> = (0..n)
        .map(|i| {
            let frac = i as f32 / (n - 1) as f32;
            let x = wave_x + frac * wave_w;
            let phase = frac * 16.0 + t * 4.0;
            let amp = wave_h * 0.35 * (sim.comms_strength / 100.0);
            let noise = 3.0 * (phase * 7.3 + t * 2.0).sin() * (1.0 - sim.comms_strength / 100.0);
            let y = carrier_y - (amp * phase.sin() + noise);
            (x, y)
        })
        .collect();

    canvas = canvas
        .polyline(signal)
        .stroke(GREEN, 1.5)
        .line_cap(LineCap::Round)
        .line_join(LineJoin::Round)
        .done();

    // ── Signal strength indicator dots ──
    let dots = 5;
    for i in 0..dots {
        let dot_x = p.x + p.w - pad - 6.0;
        let dot_y = p.y + p.h - pad - (i as f32 * 8.0);
        let active = (i as f32 / dots as f32) < (sim.comms_strength / 100.0);
        let color = if active { GREEN } else { GRID };
        canvas = canvas.circle(dot_x, dot_y, 2.5).fill(color).done();
    }

    draw_label(&mut canvas, p.x + pad, p.y + pad, "COMMS", TEXT_DIM);

    canvas
}

// ═════════════════════════════════════════════════════════════════════════════
// Pixel Font — 5×7 bitmap font for labels
// ═════════════════════════════════════════════════════════════════════════════

fn draw_label(canvas: &mut PixelCanvas, x: f32, y: f32, text: &str, color: C) {
    let mut cursor = x;
    for ch in text.chars() {
        let ch_upper = ch.to_ascii_uppercase();
        if let Some(bits) = glyph_bits(ch_upper) {
            draw_glyph(canvas, cursor, y, bits, color);
            cursor += 6.0;
        } else {
            cursor += 4.0; // space or unknown
        }
    }
}

fn draw_glyph(canvas: &mut PixelCanvas, x: f32, y: f32, bits: &[u8], color: C) {
    for (row, &byte) in bits.iter().enumerate() {
        for col in 0..5 {
            if byte & (1 << (4 - col)) != 0 {
                canvas.push_command(scry_engine::scene::command::DrawCommand::Rectangle {
                    rect: PxRect::new(x + col as f32, y + row as f32, 1.0, 1.0),
                    corner_radius: 0.0,
                    style: scry_engine::scene::style::ShapeStyle {
                        fill: Some(scry_engine::scene::style::FillStyle::Solid(color)),
                        stroke: None,
                        anti_alias: false,
                        ..scry_engine::scene::style::ShapeStyle::default()
                    },
                });
            }
        }
    }
}

fn glyph_bits(ch: char) -> Option<&'static [u8]> {
    match ch {
        'A' => Some(&[
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ]),
        'B' => Some(&[
            0b11110, 0b10001, 0b11110, 0b10001, 0b10001, 0b10001, 0b11110,
        ]),
        'C' => Some(&[
            0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
        ]),
        'D' => Some(&[
            0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100,
        ]),
        'E' => Some(&[
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
        ]),
        'F' => Some(&[
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
        ]),
        'G' => Some(&[
            0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110,
        ]),
        'H' => Some(&[
            0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ]),
        'I' => Some(&[
            0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ]),
        'J' => Some(&[
            0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100,
        ]),
        'K' => Some(&[
            0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
        ]),
        'L' => Some(&[
            0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
        ]),
        'M' => Some(&[
            0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
        ]),
        'N' => Some(&[
            0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
        ]),
        'O' => Some(&[
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ]),
        'P' => Some(&[
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ]),
        'Q' => Some(&[
            0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101,
        ]),
        'R' => Some(&[
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ]),
        'S' => Some(&[
            0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
        ]),
        'T' => Some(&[
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ]),
        'U' => Some(&[
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ]),
        'V' => Some(&[
            0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b01010, 0b00100,
        ]),
        'W' => Some(&[
            0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001,
        ]),
        'X' => Some(&[
            0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001,
        ]),
        'Y' => Some(&[
            0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100,
        ]),
        'Z' => Some(&[
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111,
        ]),
        '0' => Some(&[
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ]),
        '1' => Some(&[
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ]),
        '2' => Some(&[
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
        ]),
        '3' => Some(&[
            0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110,
        ]),
        '4' => Some(&[
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ]),
        '5' => Some(&[
            0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
        ]),
        '6' => Some(&[
            0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ]),
        '7' => Some(&[
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ]),
        '8' => Some(&[
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ]),
        '9' => Some(&[
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
        ]),
        ':' => Some(&[
            0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000,
        ]),
        '.' => Some(&[
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b00100,
        ]),
        '-' => Some(&[
            0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000,
        ]),
        '%' => Some(&[
            0b11001, 0b11010, 0b00100, 0b00100, 0b01011, 0b10011, 0b00000,
        ]),
        ' ' => Some(&[
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000,
        ]),
        _ => None,
    }
}
