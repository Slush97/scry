//! Robustness test — edge-case charts that stress axis scaling,
//! negative values, extreme ranges, NaN handling, and rendering correctness.
//!
//! Navigate pages: ←/→ or h/l    Quit: q
//!
//! ```bash
//! cargo run -p scry-chart --example robustness_test
//! ```

#![allow(
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::cast_possible_truncation
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
use scry_engine::scene::style::Color;

// ─────────────────────────────────────────────────────────────────────────────
// Pages
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Page {
    NegativeBars,
    MixedSignBars,
    HorizontalNegBars,
    StackedMixed,
    ExtremeScatter,
    NaNScatter,
    TinyRange,
    HugeRange,
    NotchedBoxPlot,
    NaNHeatmap,
    SinglePoint,
    DensityHistogram,
    Dashboard,
}

impl Page {
    const ALL: [Self; 13] = [
        Self::NegativeBars,
        Self::MixedSignBars,
        Self::HorizontalNegBars,
        Self::StackedMixed,
        Self::ExtremeScatter,
        Self::NaNScatter,
        Self::TinyRange,
        Self::HugeRange,
        Self::NotchedBoxPlot,
        Self::NaNHeatmap,
        Self::SinglePoint,
        Self::DensityHistogram,
        Self::Dashboard,
    ];

    fn index(self) -> usize {
        Self::ALL.iter().position(|&p| p == self).unwrap_or(0)
    }

    fn label(self) -> &'static str {
        match self {
            Self::NegativeBars => "1/13  Negative Bars",
            Self::MixedSignBars => "2/13  Mixed-Sign Bars",
            Self::HorizontalNegBars => "3/13  Horizontal Negative Bars",
            Self::StackedMixed => "4/13  Stacked Mixed",
            Self::ExtremeScatter => "5/13  Extreme Scatter (1e6)",
            Self::NaNScatter => "6/13  NaN-contaminated Scatter",
            Self::TinyRange => "7/13  Tiny Range (0.001)",
            Self::HugeRange => "8/13  Huge Range (1M)",
            Self::NotchedBoxPlot => "9/13  Notched BoxPlot",
            Self::NaNHeatmap => "10/13  Heatmap NaN Cells",
            Self::SinglePoint => "11/13  Single Data Point",
            Self::DensityHistogram => "12/13  Density + NaN Histogram",
            Self::Dashboard => "13/13  Stress Dashboard (4-up)",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Chart builders — each tests a specific edge case
// ─────────────────────────────────────────────────────────────────────────────

/// All-negative values — bars should grow downward from zero.
fn build_negative_bars() -> Chart {
    let labels = vec![
        "Jan".into(),
        "Feb".into(),
        "Mar".into(),
        "Apr".into(),
        "May".into(),
    ];
    Chart::bar(labels, &[-15.0, -30.0, -22.0, -8.0, -45.0])
        .title("All-Negative Bars")
        .x_label("Month")
        .y_label("P&L ($K)")
        .h_line_styled(0.0, Color::from_rgba8(255, 255, 0, 200))
        .build()
}

/// Mixed positive and negative — bars grow both directions from zero.
fn build_mixed_sign_bars() -> Chart {
    let labels = vec![
        "A".into(),
        "B".into(),
        "C".into(),
        "D".into(),
        "E".into(),
        "F".into(),
        "G".into(),
        "H".into(),
    ];
    Chart::bar(labels, &[25.0, -10.0, 40.0, -35.0, 15.0, -5.0, 30.0, -20.0])
        .title("Mixed-Sign Bars — Bidirectional")
        .x_label("Category")
        .y_label("Value")
        .h_line_styled(0.0, Color::from_rgba8(255, 200, 0, 200))
        .build()
}

/// Horizontal negative bars — extend left from zero.
fn build_horizontal_neg_bars() -> Chart {
    let labels = vec![
        "Alpha".into(),
        "Beta".into(),
        "Gamma".into(),
        "Delta".into(),
        "Epsilon".into(),
    ];
    Chart::bar(labels, &[-40.0, 20.0, -15.0, 60.0, -30.0])
        .title("Horizontal Mixed-Sign")
        .x_label("Group")
        .y_label("Δ Change")
        .horizontal()
        .build()
}

/// Stacked bars with mixed signs — tricky stacking geometry.
fn build_stacked_mixed() -> Chart {
    let labels = vec!["Q1".into(), "Q2".into(), "Q3".into(), "Q4".into()];
    Chart::bar(labels, &[20.0, -10.0, 30.0, -5.0])
        .title("Stacked Mixed-Sign")
        .x_label("Quarter")
        .y_label("Revenue vs Cost")
        .stacked()
        .corner_radius(3.0)
        .add_series(Series::new("Costs", vec![-15.0, -25.0, -10.0, -30.0]))
        .add_series(Series::new("Taxes", vec![-5.0, 5.0, -8.0, 10.0]))
        .h_line_styled(0.0, Color::from_rgba8(200, 200, 200, 180))
        .build()
}

/// Scatter with extreme values — tests axis label formatting at 1e6 scale.
fn build_extreme_scatter() -> Chart {
    let x: Vec<f64> = (0..20).map(|i| i as f64 * 50_000.0).collect();
    let y: Vec<f64> = x.iter().map(|&v| v * 1.5 + 100_000.0).collect();

    Chart::scatter(&x, &y)
        .title("Extreme Scale (Millions)")
        .x_label("Population")
        .y_label("GDP ($)")
        .connected()
        .marker(Marker::Circle)
        .trend_line()
        .h_line_styled(500_000.0, Color::from_rgba8(255, 100, 100, 180))
        .annotate(500_000.0, 850_000.0, "Midpoint")
        .build()
}

/// Scatter with NaN/Inf injected — should NOT crash, NaN points should be skipped.
fn build_nan_scatter() -> Chart {
    let x = vec![
        1.0,
        2.0,
        f64::NAN,
        4.0,
        5.0,
        f64::INFINITY,
        7.0,
        8.0,
        f64::NEG_INFINITY,
        10.0,
    ];
    let y = vec![
        2.0,
        f64::NAN,
        3.0,
        5.0,
        f64::INFINITY,
        4.0,
        7.0,
        f64::NAN,
        8.0,
        9.0,
    ];

    Chart::scatter(&x, &y)
        .title("NaN/Inf Contaminated — Should Not Crash")
        .x_label("X (has NaN, ±∞)")
        .y_label("Y (has NaN, ±∞)")
        .marker(Marker::Diamond)
        .connected()
        .annotate(5.0, 7.0, "Finite only")
        .build()
}

/// A series where values differ by only 0.001 — tests axis tick granularity.
fn build_tiny_range() -> Chart {
    let y: Vec<f64> = (0..15).map(|i| 1.000 + (i as f64) * 0.001).collect();

    Chart::line(&y)
        .title("Tiny Range: 1.000 → 1.014")
        .x_label("Sample")
        .y_label("Voltage (V)")
        .with_points()
        .h_line_styled(1.007, Color::from_rgba8(100, 255, 100, 200))
        .annotate(7.0, 1.007, "Target")
        .build()
}

/// Large numerical range — tests axis label formatting and tick distribution.
fn build_huge_range() -> Chart {
    let y: Vec<f64> = (0..20).map(|i| (i as f64 * 0.3).exp() * 100.0).collect();

    Chart::line(&y)
        .title("Exponential Growth → 28K+")
        .x_label("Day")
        .y_label("Users")
        .filled()
        .annotate(19.0, y[19], "Peak")
        .build()
}

/// Notched boxplot with varying group sizes — tests confidence interval rendering.
fn build_notched_boxplot() -> Chart {
    Chart::boxplot(vec![
        ("n=5", vec![2.0, 4.0, 5.0, 6.0, 8.0]),
        (
            "n=10",
            vec![1.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 9.0, 15.0],
        ),
        ("n=30", {
            let mut v: Vec<f64> = (0..30)
                .map(|i| 3.0 + (i as f64 * 0.2).sin() * 4.0 + i as f64 * 0.1)
                .collect();
            v.push(20.0); // outlier
            v
        }),
        ("n=100", {
            (0..100)
                .map(|i| {
                    let x = i as f64 / 10.0;
                    5.0 + (x * 0.5).sin() * 3.0 + (i % 7) as f64 * 0.3
                })
                .collect()
        }),
    ])
    .title("Notched BoxPlot — Varying Group Sizes")
    .x_label("Group")
    .y_label("Score")
    .notched()
    .build()
}

/// Heatmap with NaN cells — NaN cells should be skipped (transparent gap).
fn build_nan_heatmap() -> Chart {
    let nan = f64::NAN;
    let labels = vec![
        "Mon".into(),
        "Tue".into(),
        "Wed".into(),
        "Thu".into(),
        "Fri".into(),
    ];

    Chart::heatmap(vec![
        vec![1.0, nan, 3.0, 4.0, 5.0],
        vec![nan, 7.0, 8.0, nan, 10.0],
        vec![11.0, 12.0, nan, 14.0, 15.0],
        vec![16.0, 17.0, 18.0, 19.0, nan],
        vec![nan, nan, 23.0, 24.0, 25.0],
    ])
    .title("Heatmap — NaN Cells (should be transparent)")
    .row_labels(labels.clone())
    .col_labels(labels)
    .values(true)
    .cell_gap(2.0)
    .cell_radius(3.0)
    .colors(
        Color::from_rgba8(30, 80, 180, 255),
        Color::from_rgba8(220, 50, 50, 255),
    )
    .build()
}

/// Single data point — the degenerate case for all chart math.
fn build_single_point() -> Chart {
    Chart::scatter(&[42.0], &[99.0])
        .title("Single Point — Degenerate Case")
        .x_label("X")
        .y_label("Y")
        .marker(Marker::Circle)
        .annotate(42.0, 99.0, "Only point")
        .build()
}

/// Density histogram with NaN values injected — density should be based on finite count.
fn build_density_histogram() -> Chart {
    let mut data: Vec<f64> = (0..200)
        .map(|i| {
            let x = i as f64 / 20.0;
            (x * 0.8).sin() * 4.0 + 5.0 + (i % 5) as f64 * 0.3
        })
        .collect();
    // Inject NaN values
    data[10] = f64::NAN;
    data[30] = f64::NAN;
    data[50] = f64::INFINITY;
    data[70] = f64::NEG_INFINITY;

    let finite_mean = data.iter().filter(|v| v.is_finite()).sum::<f64>()
        / data.iter().filter(|v| v.is_finite()).count() as f64;

    Chart::histogram(&data)
        .title("Density Histogram — NaN/Inf in Data")
        .x_label("Value")
        .y_label("Density")
        .bins(15)
        .density()
        .opacity(0.7)
        .v_line_styled(finite_mean, Color::from_rgba8(255, 255, 0, 200))
        .build()
}

// ─────────────────────────────────────────────────────────────────────────────
// Main loop
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let mut state = ChartState::auto();
    let mut page = Page::NegativeBars;

    loop {
        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(1)])
                .split(frame.area());

            if page == Page::Dashboard {
                // 2×2 stress dashboard
                let rows = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .split(chunks[0]);

                let top = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .split(rows[0]);

                let bot = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .split(rows[1]);

                let charts = [
                    build_mixed_sign_bars(),
                    build_nan_scatter(),
                    build_notched_boxplot(),
                    build_nan_heatmap(),
                ];

                let areas = [top[0], top[1], bot[0], bot[1]];
                for (chart, area) in charts.iter().zip(areas.iter()) {
                    frame.render_stateful_widget(ChartWidget::new(chart), *area, &mut state);
                }
            } else {
                let chart = match page {
                    Page::NegativeBars => build_negative_bars(),
                    Page::MixedSignBars => build_mixed_sign_bars(),
                    Page::HorizontalNegBars => build_horizontal_neg_bars(),
                    Page::StackedMixed => build_stacked_mixed(),
                    Page::ExtremeScatter => build_extreme_scatter(),
                    Page::NaNScatter => build_nan_scatter(),
                    Page::TinyRange => build_tiny_range(),
                    Page::HugeRange => build_huge_range(),
                    Page::NotchedBoxPlot => build_notched_boxplot(),
                    Page::NaNHeatmap => build_nan_heatmap(),
                    Page::SinglePoint => build_single_point(),
                    Page::DensityHistogram => build_density_histogram(),
                    Page::Dashboard => unreachable!(),
                };

                frame.render_stateful_widget(ChartWidget::new(&chart), chunks[0], &mut state);
            }

            let status = Paragraph::new(format!(" {}  |  ←/→ navigate · q quit", page.label(),))
                .block(Block::default().borders(Borders::TOP));
            frame.render_widget(status, chunks[1]);
        })?;
        state.flush()?;

        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Right | KeyCode::Char('l') => {
                        let idx = page.index();
                        page = Page::ALL[(idx + 1) % Page::ALL.len()];
                    }
                    KeyCode::Left | KeyCode::Char('h') => {
                        let idx = page.index();
                        page = Page::ALL[(idx + Page::ALL.len() - 1) % Page::ALL.len()];
                    }
                    _ => {}
                }
            }
        }
    }

    state.cleanup();
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}
