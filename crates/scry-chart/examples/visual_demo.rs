//! Visual Demo — a beautiful 6-panel dashboard showcasing scry-chart's capabilities.
//!
//! Displays line, scatter, bar, histogram, box plot, and heatmap charts
//! in a polished 3×2 grid with dark theme.
//!
//! Press 'q' or Esc to quit.
//!
//! ```bash
//! cargo run -p scry-chart --example visual_demo
//! ```

use std::io::stdout;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use scry_chart::prelude::*;

// ─── Data Helpers ────────────────────────────────────────────────────────────

fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| lo + (hi - lo) * i as f64 / (n - 1).max(1) as f64)
        .collect()
}

fn pseudo_normal(n: usize, mean: f64, std: f64, seed: u64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let mut sum = 0.0;
            for k in 0..6u64 {
                let v = ((i as u64 * 2654435761 + k * 7919 + seed) % 10000) as f64 / 10000.0;
                sum += v;
            }
            mean + (sum - 3.0) * std
        })
        .collect()
}

// ─── Chart Builders ──────────────────────────────────────────────────────────

/// Panel 1: Multi-series line chart with area fill
fn build_line() -> Chart {
    let x = linspace(0.0, 12.0, 50);
    let revenue: Vec<f64> = x
        .iter()
        .map(|&v| 3.0 + v * 0.6 + (v * 0.5).sin() * 2.5)
        .collect();
    let costs: Vec<f64> = x
        .iter()
        .map(|&v| 2.0 + v * 0.3 + (v * 0.3).cos() * 1.0)
        .collect();

    Chart::line(&revenue)
        .title("Revenue vs Costs")
        .x_label("Month")
        .y_label("$M")
        .x_values(x)
        .add_series(Series::new("Costs", costs))
        .filled()
        .with_points()
        .h_line(6.0)
        .theme(Theme::dark())
        .build()
}

/// Panel 2: Scatter plot with trend line and annotations
fn build_scatter() -> Chart {
    let x: Vec<f64> = (0..40).map(|i| i as f64 * 0.25).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let noise = ((i as u64 * 2654435761 + 42) % 1000) as f64 / 1000.0 - 0.5;
            v.sqrt() * 2.0 + noise * 1.5
        })
        .collect();

    Chart::scatter(&x, &y)
        .title("Growth Analysis")
        .x_label("Time")
        .y_label("Value")
        .x_range(0.0, 10.0)
        .y_range(-1.0, 8.0)
        .connected()
        .marker(Marker::Circle)
        .trend_line()
        .annotate(8.0, 6.0, "converge")
        .theme(Theme::dark())
        .build()
}

/// Panel 3: Grouped bar chart
fn build_bar() -> Chart {
    let labels: Vec<String> = vec!["Rust", "Go", "Python", "TS"]
        .into_iter()
        .map(String::from)
        .collect();

    let perf = vec![95.0, 78.0, 42.0, 35.0];
    let safety = vec![99.0, 74.0, 58.0, 72.0];

    Chart::bar(labels, &perf)
        .title("Language Scores")
        .y_label("Score")
        .add_series(Series::new("Safety", safety))
        .y_range(0.0, 100.0)
        .corner_radius(3.0)
        .h_line(70.0)
        .theme(Theme::dark())
        .build()
}

/// Panel 4: Histogram with density curve
fn build_histogram() -> Chart {
    use scry_engine::style::Color;

    let data = pseudo_normal(300, 0.0, 1.0, 777);
    let mean = data.iter().sum::<f64>() / data.len() as f64;

    Chart::histogram(&data)
        .title("Normal Distribution")
        .x_label("σ")
        .y_label("Density")
        .bins(20)
        .density()
        .opacity(0.75)
        .v_line_styled(mean, Color::from_rgba8(255, 90, 90, 200))
        .theme(Theme::dark())
        .build()
}

/// Panel 5: Box plot comparison
fn build_boxplot() -> Chart {
    Chart::boxplot(vec![
        ("Control", pseudo_normal(50, 10.0, 2.0, 100)),
        ("Group A", pseudo_normal(50, 14.0, 3.0, 200)),
        ("Group B", pseudo_normal(50, 12.0, 1.8, 300)),
    ])
    .title("Clinical Trial")
    .x_label("Group")
    .y_label("Score")
    .notched()
    .h_line(12.0)
    .theme(Theme::dark())
    .build()
}

/// Panel 6: Correlation heatmap
fn build_heatmap() -> Chart {
    use scry_engine::style::Color;

    let labels: Vec<String> = vec!["A", "B", "C", "D"]
        .into_iter()
        .map(String::from)
        .collect();

    let data = vec![
        vec![1.00, 0.82, 0.15, -0.30],
        vec![0.82, 1.00, 0.55, 0.10],
        vec![0.15, 0.55, 1.00, 0.70],
        vec![-0.30, 0.10, 0.70, 1.00],
    ];

    Chart::heatmap(data)
        .title("Correlation Matrix")
        .row_labels(labels.clone())
        .col_labels(labels)
        .colors(
            Color::from_rgba8(40, 80, 200, 255),
            Color::from_rgba8(220, 60, 60, 255),
        )
        .range(-1.0, 1.0)
        .values(true)
        .cell_radius(3.0)
        .cell_gap(2.0)
        .theme(Theme::dark())
        .build()
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    // We need separate ChartStates for each panel in the grid
    let mut states: Vec<ChartState> = (0..6).map(|_| ChartState::auto()).collect();

    loop {
        terminal.draw(|frame| {
            let outer = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(1)])
                .split(frame.area());

            // 3×2 grid
            let rows = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Percentage(33),
                    Constraint::Percentage(34),
                    Constraint::Percentage(33),
                ])
                .split(outer[0]);

            let top = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(rows[0]);

            let mid = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(rows[1]);

            let bot = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(rows[2]);

            // Build all charts
            let charts = [
                build_line(),
                build_scatter(),
                build_bar(),
                build_histogram(),
                build_boxplot(),
                build_heatmap(),
            ];

            let areas = [top[0], top[1], mid[0], mid[1], bot[0], bot[1]];

            for (i, (chart, area)) in charts.iter().zip(areas.iter()).enumerate() {
                frame.render_stateful_widget(ChartWidget::new(chart), *area, &mut states[i]);
            }

            // Status bar
            let status =
                Paragraph::new(" scry-chart visual demo  │  6 chart types  │  q/Esc to quit")
                    .block(Block::default().borders(Borders::TOP))
                    .style(Style::default().fg(Color::DarkGray));
            frame.render_widget(status, outer[1]);
        })?;

        // Flush all states
        for s in &mut states {
            let _ = s.flush();
        }

        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                if matches!(key.code, KeyCode::Char('q') | KeyCode::Esc) {
                    break;
                }
            }
        }
    }

    for s in &mut states {
        s.cleanup();
    }
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}
