//! Comprehensive pixelchart showcase — all 6 chart types with every builder option.
//!
//! Navigate: ←/→ or h/l to switch pages, T to cycle themes, q to quit.
//!
//! | Page | Chart              | Features shown                                              |
//! |------|--------------------|-------------------------------------------------------------|
//! | 1    | Scatter            | Multi-series, markers, trend line, annotations, ref lines   |
//! | 2    | Line (basic)       | Multi-series, polyline rendering, dashed grids              |
//! | 3    | Line (filled)      | Gradient area fill, data points, reference lines            |
//! | 4    | Bar (vertical)     | Grouped + stacked, stroke outlines, corner radius           |
//! | 5    | Bar (horizontal)   | Stacked horizontal, custom gap, dashed grids                |
//! | 6    | Histogram          | Density, multi-series, rounded bins, v-line for mean        |
//! | 7    | Box plot           | Outliers, notched, median/quartile stats                    |
//! | 8    | Heatmap            | Correlation matrix, custom colors, cell labels              |
//! | 9    | Dashboard          | 4 charts in a 2×2 grid                                     |
//!
//! ```bash
//! cargo run -p pixelchart --example chart_showcase
//! ```

#![allow(
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::too_many_lines
)]

use std::io::stdout;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use pixelchart::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// Pages
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Page {
    Scatter,
    LineBasic,
    LineFilled,
    BarVertical,
    BarHorizontal,
    HistogramDemo,
    BoxPlotDemo,
    HeatmapDemo,
    Dashboard,
}

impl Page {
    const ALL: [Self; 9] = [
        Self::Scatter,
        Self::LineBasic,
        Self::LineFilled,
        Self::BarVertical,
        Self::BarHorizontal,
        Self::HistogramDemo,
        Self::BoxPlotDemo,
        Self::HeatmapDemo,
        Self::Dashboard,
    ];

    fn index(self) -> usize {
        Self::ALL.iter().position(|&p| p == self).unwrap_or(0)
    }

    fn label(self) -> &'static str {
        match self {
            Self::Scatter => "1/9: Scatter Plot",
            Self::LineBasic => "2/9: Line Chart",
            Self::LineFilled => "3/9: Filled Line",
            Self::BarVertical => "4/9: Bar (Vertical)",
            Self::BarHorizontal => "5/9: Bar (Horizontal)",
            Self::HistogramDemo => "6/9: Histogram",
            Self::BoxPlotDemo => "7/9: Box Plot",
            Self::HeatmapDemo => "8/9: Heatmap",
            Self::Dashboard => "9/9: Dashboard",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Theme cycling
// ─────────────────────────────────────────────────────────────────────────────

const THEME_NAMES: [&str; 3] = ["Dark", "Light", "Pastel"];

fn theme_by_index(i: usize) -> Theme {
    match i % 3 {
        0 => Theme::dark(),
        1 => Theme::light(),
        _ => Theme::pastel(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Chart builders
// ─────────────────────────────────────────────────────────────────────────────

fn build_scatter(theme: Theme) -> Chart {
    use ratatui_pixelcanvas::scene::style::Color;

    Chart::scatter(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        &[2.1, 3.8, 1.5, 7.2, 5.5, 4.0, 8.1, 6.3, 9.0, 7.5],
    )
    .title("Multi-Series Scatter Plot")
    .x_label("Experiment #")
    .y_label("Measurement")
    .theme(theme)
    .marker(Marker::Circle)
    .connected()
    .add_series(
        Series::new(
            "Series B",
            vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
        ),
        Series::new(
            "Series B Y",
            vec![5.0, 3.2, 6.1, 2.8, 7.3, 4.5, 3.0, 8.2, 5.8, 9.1],
        ),
    )
    .trend_line()
    .annotate(4.0, 7.2, "Peak A")
    .annotate(8.5, 8.2, "Peak B")
    .h_line_styled(5.0, Color::from_rgba8(255, 200, 0, 140))
    .v_line(5.5)
    .build()
}

fn build_line_basic(theme: Theme) -> Chart {
    Chart::line(&[12.0, 19.0, 3.0, 5.0, 2.0, 14.0, 8.0, 11.0, 17.0, 6.0])
        .title("Multi-Series Line Chart")
        .x_label("Month")
        .y_label("Revenue ($K)")
        .theme(theme)
        .add_series(Series::new(
            "Product B",
            vec![5.0, 12.0, 8.0, 15.0, 11.0, 7.0, 13.0, 9.0, 4.0, 18.0],
        ))
        .add_series(Series::new(
            "Product C",
            vec![8.0, 6.0, 14.0, 10.0, 16.0, 3.0, 11.0, 7.0, 13.0, 9.0],
        ))
        .build()
}

fn build_line_filled(theme: Theme) -> Chart {
    use ratatui_pixelcanvas::scene::style::Color;

    Chart::line(&[4.0, 7.0, 2.0, 9.0, 5.0, 11.0, 8.0, 3.0, 10.0, 6.0])
        .title("Area Fill with Gradient")
        .x_label("Time (s)")
        .y_label("CPU %")
        .theme(theme)
        .filled()
        .with_points()
        .add_series(Series::new(
            "GPU %",
            vec![2.0, 5.0, 8.0, 4.0, 10.0, 7.0, 3.0, 9.0, 6.0, 11.0],
        ))
        .h_line_styled(6.0, Color::from_rgba8(255, 100, 100, 180))
        .annotate(5.0, 11.0, "Spike")
        .build()
}

fn build_bar_vertical(theme: Theme) -> Chart {
    let labels = vec!["Q1".into(), "Q2".into(), "Q3".into(), "Q4".into()];

    Chart::bar(labels, &[42.0, 65.0, 53.0, 78.0])
        .title("Quarterly Revenue — Grouped")
        .x_label("Quarter")
        .y_label("$M")
        .theme(theme)
        .add_series(Series::new("2024", vec![38.0, 55.0, 60.0, 72.0]))
        .corner_radius(4.0)
        .build()
}

fn build_bar_horizontal(theme: Theme) -> Chart {
    let labels = vec![
        "Rust".into(),
        "Go".into(),
        "Python".into(),
        "JS".into(),
        "C++".into(),
    ];

    Chart::bar(labels, &[92.0, 78.0, 85.0, 70.0, 65.0])
        .title("Language Satisfaction — Stacked")
        .x_label("Language")
        .y_label("Score")
        .theme(theme)
        .horizontal()
        .stacked()
        .add_series(Series::new("2nd Poll", vec![88.0, 80.0, 82.0, 68.0, 60.0]))
        .gap(0.3)
        .build()
}

fn build_histogram(theme: Theme) -> Chart {
    use ratatui_pixelcanvas::scene::style::Color;

    // Generate sample data for a normal-ish distribution
    let data: Vec<f64> = (0..100)
        .map(|i| {
            let x = i as f64 / 10.0;
            (x * 0.7).sin() * 3.0 + 5.0 + (i as f64 % 3.0) * 0.5
        })
        .collect();

    let mean = data.iter().sum::<f64>() / data.len() as f64;

    Chart::histogram(&data)
        .title("Value Distribution — Density")
        .x_label("Value")
        .y_label("Density")
        .theme(theme)
        .bins(12)
        .density()
        .add_series(Series::new(
            "Group B",
            (0..80)
                .map(|i| {
                    let x = i as f64 / 8.0;
                    (x * 0.5).cos() * 2.0 + 6.0
                })
                .collect(),
        ))
        .opacity(0.6)
        .v_line_styled(mean, Color::from_rgba8(255, 255, 0, 200))
        .build()
}

fn build_boxplot(theme: Theme) -> Chart {
    Chart::boxplot(vec![
        (
            "Control",
            vec![2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 15.0],
        ),
        (
            "Drug A",
            vec![4.0, 5.0, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0],
        ),
        (
            "Drug B",
            vec![1.0, 5.5, 6.0, 7.0, 7.5, 8.0, 9.0, 9.5, 10.0, 11.0, 18.0],
        ),
        (
            "Drug C",
            vec![3.0, 4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 12.0],
        ),
    ])
    .title("Clinical Trial Results")
    .x_label("Treatment Group")
    .y_label("Response Score")
    .theme(theme)
    .notched()
    .build()
}

fn build_heatmap(theme: Theme) -> Chart {
    use ratatui_pixelcanvas::scene::style::Color;

    let labels = vec!["A".into(), "B".into(), "C".into(), "D".into(), "E".into()];
    Chart::heatmap(vec![
        vec![1.00, 0.85, 0.30, -0.10, -0.45],
        vec![0.85, 1.00, 0.55, 0.20, -0.30],
        vec![0.30, 0.55, 1.00, 0.70, 0.10],
        vec![-0.10, 0.20, 0.70, 1.00, 0.60],
        vec![-0.45, -0.30, 0.10, 0.60, 1.00],
    ])
    .title("Correlation Matrix")
    .theme(theme)
    .row_labels(labels.clone())
    .col_labels(labels)
    .colors(
        Color::from_rgba8(60, 100, 220, 255),
        Color::from_rgba8(220, 60, 60, 255),
    )
    .range(-1.0, 1.0)
    .values(true)
    .cell_radius(3.0)
    .cell_gap(2.0)
    .build()
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let mut state = ChartState::auto();
    let mut page = Page::Scatter;
    let mut theme_idx: usize = 0;

    loop {
        let theme = theme_by_index(theme_idx);

        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(1)])
                .split(frame.area());

            if page == Page::Dashboard {
                // 2×2 dashboard grid
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
                    build_line_basic(theme.clone()),
                    build_bar_vertical(theme.clone()),
                    build_scatter(theme.clone()),
                    build_histogram(theme.clone()),
                ];

                let areas = [top[0], top[1], bot[0], bot[1]];
                for (chart, area) in charts.iter().zip(areas.iter()) {
                    frame.render_stateful_widget(ChartWidget::new(chart), *area, &mut state);
                }
            } else {
                let chart = match page {
                    Page::Scatter => build_scatter(theme.clone()),
                    Page::LineBasic => build_line_basic(theme.clone()),
                    Page::LineFilled => build_line_filled(theme.clone()),
                    Page::BarVertical => build_bar_vertical(theme.clone()),
                    Page::BarHorizontal => build_bar_horizontal(theme.clone()),
                    Page::HistogramDemo => build_histogram(theme.clone()),
                    Page::BoxPlotDemo => build_boxplot(theme.clone()),
                    Page::HeatmapDemo => build_heatmap(theme.clone()),
                    Page::Dashboard => unreachable!(),
                };

                frame.render_stateful_widget(ChartWidget::new(&chart), chunks[0], &mut state);
            }

            let status = Paragraph::new(format!(
                " {}  |  Theme: {}  |  ←/→ navigate · T theme · q quit",
                page.label(),
                THEME_NAMES[theme_idx % 3],
            ))
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
                    KeyCode::Char('t' | 'T') => {
                        theme_idx = (theme_idx + 1) % 3;
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
