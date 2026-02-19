//! Session 1 Feature Showcase — demonstrates the four new chart features:
//!
//! | Panel | Feature |
//! |-------|---------|
//! | Top-left | **Subtitle & Footer** — title, subtitle below it, footer at bottom |
//! | Top-right | **Custom Margins** — thick padding around the chart |
//! | Bottom-left | **Inverted Y-Axis** — high values at bottom |
//! | Bottom-right | **Dual Y-Axis** — secondary axis on the right |
//!
//! Run with: `cargo run --example chart_features`

#![allow(
    clippy::suboptimal_flops,
    clippy::items_after_statements,
    clippy::cast_precision_loss,
    clippy::unreadable_literal,
    clippy::similar_names,
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

use scry_chart::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let mut state = ChartState::auto();

    loop {
        terminal.draw(|frame| {
            let outer = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(2)])
                .split(frame.area());

            let rows = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(outer[0]);

            let top = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(rows[0]);

            let bottom = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(rows[1]);

            // ── Top-Left: Subtitle & Footer ──────────────────────────────
            let chart1 = build_subtitle_footer();
            frame.render_stateful_widget(ChartWidget::new(&chart1), top[0], &mut state);

            // ── Top-Right: Custom Margins ────────────────────────────────
            let chart2 = build_margins();
            frame.render_stateful_widget(ChartWidget::new(&chart2), top[1], &mut state);

            // ── Bottom-Left: Inverted Y-Axis ─────────────────────────────
            let chart3 = build_inverted();
            frame.render_stateful_widget(ChartWidget::new(&chart3), bottom[0], &mut state);

            // ── Bottom-Right: Dual Y-Axis ────────────────────────────────
            let chart4 = build_dual_y();
            frame.render_stateful_widget(ChartWidget::new(&chart4), bottom[1], &mut state);

            // ── Status bar ───────────────────────────────────────────────
            let status = Paragraph::new(
                " ★ Session 1 Features: Subtitle/Footer · Margins · Inverted Axis · Dual Y-Axis  |  'q' quit",
            )
            .style(Style::default().fg(Color::DarkGray))
            .block(
                Block::default()
                    .borders(Borders::TOP)
                    .border_style(Style::default().fg(Color::DarkGray)),
            );
            frame.render_widget(status, outer[1]);
        })?;
        state.flush()?;

        if event::poll(std::time::Duration::from_millis(100))? {
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

// ─────────────────────────────────────────────────────────────────────────────
// Chart builders
// ─────────────────────────────────────────────────────────────────────────────

/// Demonstrates: `.subtitle()` and `.footer()` text overlays.
fn build_subtitle_footer() -> Chart {
    let y: Vec<f64> = (0..30)
        .map(|i| {
            let x = i as f64 * 0.2;
            (x * 1.5).sin() * 40.0 + 60.0 + (x * 0.3).cos() * 15.0
        })
        .collect();

    Chart::line(&y)
        .title("Monthly Revenue")
        .subtitle("Q3–Q4 2025 · All Regions")
        .footer("Source: internal analytics · Updated Feb 2026")
        .x_label("Month")
        .y_label("Revenue ($K)")
        .build()
}

/// Demonstrates: `.margin(top, right, bottom, left)` for thick padding.
fn build_margins() -> Chart {
    let y: Vec<f64> = (0..20)
        .map(|i| {
            let x = i as f64 * 0.3;
            x.powi(2) * 0.5
        })
        .collect();

    Chart::line(&y)
        .title("Growth Curve")
        .subtitle("with 30px margins all around")
        .x_label("Time")
        .y_label("Value")
        .margin(30.0, 30.0, 30.0, 30.0)
        .build()
}

/// Demonstrates: `.y_inverted()` — high values at the bottom of the chart.
fn build_inverted() -> Chart {
    let y: Vec<f64> = (0..25)
        .map(|i| {
            let x = i as f64 * 0.25;
            100.0 - (x * 2.0).sin().abs() * 50.0 - x * 3.0
        })
        .collect();

    Chart::line(&y)
        .title("Depth Profile")
        .subtitle("Y-axis inverted ↓ deeper = lower")
        .x_label("Station")
        .y_label("Depth (m)")
        .y_inverted()
        .build()
}

/// Demonstrates: dual Y-axis with `.secondary_y_label()` and `.secondary_axis()`.
fn build_dual_y() -> Chart {
    // Series 0: temperature (left Y-axis, range 15–35)
    let temp: Vec<f64> = (0..24)
        .map(|i| {
            let hour = i as f64;
            20.0 + 8.0 * ((hour - 14.0) * std::f64::consts::PI / 12.0).cos()
                + (hour * 0.5).sin() * 2.0
        })
        .collect();

    // Series 1: humidity % (right Y-axis, range 30–90)
    // NOTE: The secondary axis *label* and *gutter* render now, but routing
    // individual series through the secondary scale is a later task —
    // so we define the data here for reference but prefix with `_`.
    let _humidity: Vec<f64> = (0..24)
        .map(|i| {
            let hour = i as f64;
            60.0 - 20.0 * ((hour - 14.0) * std::f64::consts::PI / 12.0).cos()
                + (hour * 0.7).sin() * 5.0
        })
        .collect();

    // Combine into two series
    let x: Vec<f64> = (0..24).map(|i| i as f64).collect();

    Chart::line_xy(&x, &temp)
        .title("Weather Station")
        .subtitle("Temperature + Humidity (dual axis)")
        .x_label("Hour of Day")
        .y_label("Temperature (°C)")
        .secondary_y_label("Humidity (%)")
        .secondary_y_range(30.0, 90.0)
        .build()
}
