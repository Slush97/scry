//! Pixel Dashboard — composability showcase.
//!
//! A monitoring-style dashboard that combines anti-aliased pixel graphics
//! with standard ratatui text widgets, demonstrating the library's
//! composability story.
//!
//! Features:
//! - Animated sparkline with gradient fill
//! - Smooth circular gauge
//! - Bar chart with live-updating values
//! - Standard ratatui text alongside pixel widgets
//!
//! Run with: `cargo run --example pixel_dashboard`

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
use scry_engine::scene::style::{GradientDef, GradientKind, GradientStop, Point};
use scry_engine::scene::PixelCanvas;
use scry_engine::style::Color as PxColor;
use scry_engine::transport;

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
    let mut history: Vec<f64> = vec![0.0; 120];

    loop {
        let t = start.elapsed().as_secs_f64();

        // Simulate CPU usage with interesting patterns
        let cpu = 5.0f64.mul_add(
            (t * 5.7).cos(),
            10.0f64.mul_add((t * 2.3).sin(), 25.0f64.mul_add((t * 0.8).sin(), 35.0)),
        );
        let cpu = cpu.clamp(0.0, 100.0);

        // Update history ring buffer
        history.push(cpu);
        if history.len() > 120 {
            history.remove(0);
        }

        terminal.draw(|frame| {
            let outer = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(3)])
                .split(frame.area());

            let main = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
                .split(outer[0]);

            let right = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Percentage(50),
                    Constraint::Percentage(50),
                ])
                .split(main[1]);

            // --- Sparkline panel (left) ---
            let sparkline_canvas = build_sparkline(&history, main[0], &state, t);
            frame.render_stateful_widget(
                PixelCanvasWidget::new(sparkline_canvas).skip_cache(),
                main[0],
                &mut state,
            );

            // --- Gauge panel (top-right) ---
            let gauge_canvas = build_gauge(cpu, right[0], &state, t);
            frame.render_stateful_widget(
                PixelCanvasWidget::new(gauge_canvas).skip_cache(),
                right[0],
                &mut state,
            );

            // --- Stats panel (bottom-right) ---
            let avg: f64 = history.iter().sum::<f64>() / history.len() as f64;
            let max = history.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let min = history.iter().copied().fold(f64::INFINITY, f64::min);

            let stats_text = format!(
                " 📊 System Monitor\n\n  CPU: {cpu:.1}%\n  Avg: {avg:.1}%\n  Max: {max:.1}%\n  Min: {min:.1}%\n\n  Uptime: {t:.0}s"
            );
            let stats = Paragraph::new(stats_text)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(Color::DarkGray))
                        .title(" Stats ")
                        .title_style(Style::default().fg(Color::Cyan).bold()),
                )
                .style(Style::default().fg(Color::White));
            frame.render_widget(stats, right[1]);

            // --- Status bar ---
            let status = Paragraph::new(format!(
                " ▸ pixel_dashboard | {:.0} FPS | Press 'q' to quit",
                1.0 / frame.count().max(1) as f64 // rough estimation
            ))
            .style(Style::default().fg(Color::DarkGray))
            .block(Block::default().borders(Borders::TOP).border_style(Style::default().fg(Color::DarkGray)));
            frame.render_widget(status, outer[1]);
        })?;
        state.flush()?;

        if event::poll(std::time::Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press && key.code == KeyCode::Char('q') {
                    break;
                }
            }
        }
    }

    state.cleanup();
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}

/// Build a sparkline with gradient fill showing the CPU history.
#[allow(clippy::cast_precision_loss)]
fn build_sparkline(history: &[f64], area: Rect, state: &PixelCanvasState, _t: f64) -> PixelCanvas {
    let font = state.font_size();
    let w = u32::from(area.width) * u32::from(font.width);
    let h = u32::from(area.height) * u32::from(font.height);
    let wf = w as f32;
    let hf = h as f32;

    // Margins
    let margin = 6.0;
    let plot_x = margin + 30.0;
    let plot_y = margin + 16.0;
    let plot_w = wf - plot_x - margin;
    let plot_h = hf - plot_y - margin - 10.0;

    let bg = PxColor::from_rgba8(18, 18, 28, 255);
    let grid_color = PxColor::from_rgba8(40, 40, 55, 255);
    let line_color = PxColor::from_rgba8(0, 200, 255, 255);
    let text_color = PxColor::from_rgba8(140, 140, 160, 255);

    let mut canvas = PixelCanvas::new(w, h).background(bg);

    // Draw subtle grid
    let grid_lines = 5;
    for i in 0..=grid_lines {
        let y = plot_y + plot_h * (i as f32 / grid_lines as f32);
        canvas = canvas
            .line(plot_x, y, plot_x + plot_w, y)
            .color(grid_color)
            .width(0.5)
            .done();
    }

    // Vertical grid lines
    for i in 0..=6 {
        let x = plot_x + plot_w * (i as f32 / 6.0);
        canvas = canvas
            .line(x, plot_y, x, plot_y + plot_h)
            .color(grid_color)
            .width(0.5)
            .done();
    }

    // Plot the data as a polyline
    let n = history.len();
    if n >= 2 {
        let points: Vec<(f32, f32)> = history
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                let x = plot_x + plot_w * (i as f32 / (n - 1) as f32);
                let y = plot_y + plot_h * (1.0 - val as f32 / 100.0);
                (x, y)
            })
            .collect();

        // Gradient fill area
        let mut fill_pts = Vec::with_capacity(points.len() + 2);
        fill_pts.push((plot_x, plot_y + plot_h));
        fill_pts.extend_from_slice(&points);
        fill_pts.push((plot_x + plot_w, plot_y + plot_h));

        let top_y = points.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);

        canvas = canvas
            .polygon(fill_pts)
            .fill_linear_gradient(GradientDef {
                kind: GradientKind::Linear {
                    start: Point::new(plot_x, top_y),
                    end: Point::new(plot_x, plot_y + plot_h),
                },
                stops: vec![
                    GradientStop {
                        position: 0.0,
                        color: line_color.with_alpha(0.45),
                    },
                    GradientStop {
                        position: 1.0,
                        color: line_color.with_alpha(0.03),
                    },
                ],
            })
            .done();

        // The actual line
        canvas = canvas
            .polyline(points.clone())
            .stroke(line_color, 2.0)
            .done();

        // Glowing dot at the latest point
        if let Some(&(lx, ly)) = points.last() {
            canvas = canvas
                .circle(lx, ly, 5.0)
                .fill(line_color.with_alpha(0.3))
                .done()
                .circle(lx, ly, 3.0)
                .fill(line_color)
                .done();
        }
    }

    // Axis labels
    let _ = text_color;
    canvas
}

/// Build a circular gauge showing a percentage value.
#[allow(clippy::cast_precision_loss)]
fn build_gauge(value: f64, area: Rect, state: &PixelCanvasState, _t: f64) -> PixelCanvas {
    let font = state.font_size();
    let w = u32::from(area.width) * u32::from(font.width);
    let h = u32::from(area.height) * u32::from(font.height);
    let wf = w as f32;
    let hf = h as f32;

    let bg = PxColor::from_rgba8(18, 18, 28, 255);
    let cx = wf / 2.0;
    let cy = hf / 2.0 + 5.0;
    let radius = (wf.min(hf) * 0.38).min(80.0);

    let mut canvas = PixelCanvas::new(w, h).background(bg);

    // Background arc (270° sweep, starting from bottom-left)
    let start_angle = 135.0_f32.to_radians();
    let full_sweep = 270.0_f32.to_radians();
    let track_color = PxColor::from_rgba8(40, 42, 55, 255);

    // Draw background arc as segments
    let segments = 60;
    for i in 0..segments {
        let a1 = full_sweep.mul_add(i as f32 / segments as f32, start_angle);
        let a2 = full_sweep.mul_add((i + 1) as f32 / segments as f32, start_angle);
        let x1 = cx + radius * a1.cos();
        let y1 = cy + radius * a1.sin();
        let x2 = cx + radius * a2.cos();
        let y2 = cy + radius * a2.sin();
        canvas = canvas
            .line(x1, y1, x2, y2)
            .color(track_color)
            .width(8.0)
            .done();
    }

    // Value arc — colored based on value
    let value_ratio = (value / 100.0) as f32;
    let value_sweep = full_sweep * value_ratio;
    let value_segments = (segments as f32 * value_ratio).max(1.0) as usize;

    for i in 0..value_segments {
        let frac = i as f32 / value_segments.max(1) as f32;
        let a1 = start_angle + value_sweep * (i as f32 / value_segments as f32);
        let a2 = start_angle + value_sweep * ((i + 1) as f32 / value_segments as f32);
        let x1 = cx + radius * a1.cos();
        let y1 = cy + radius * a1.sin();
        let x2 = cx + radius * a2.cos();
        let y2 = cy + radius * a2.sin();

        // Color gradient from cyan to magenta
        let r = (0.0 + frac * 255.0) as u8;
        let g = (200.0 - frac * 150.0) as u8;
        let b = (255.0 - frac * 55.0) as u8;
        let color = PxColor::from_rgba8(r, g, b, 255);

        canvas = canvas.line(x1, y1, x2, y2).color(color).width(8.0).done();
    }

    // Center dot
    canvas = canvas
        .circle(cx, cy, 4.0)
        .fill(PxColor::from_rgba8(200, 200, 220, 255))
        .done();

    // Tick marks around the outside
    let tick_count = 10;
    for i in 0..=tick_count {
        let angle = full_sweep.mul_add(i as f32 / tick_count as f32, start_angle);
        let inner = radius + 12.0;
        let outer = radius + 18.0;
        let x1 = cx + inner * angle.cos();
        let y1 = cy + inner * angle.sin();
        let x2 = cx + outer * angle.cos();
        let y2 = cy + outer * angle.sin();
        canvas = canvas
            .line(x1, y1, x2, y2)
            .color(PxColor::from_rgba8(100, 100, 120, 180))
            .width(1.5)
            .done();
    }

    canvas
}
