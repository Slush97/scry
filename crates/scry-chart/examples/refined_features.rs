//! Refined Features Showcase — demonstrating every improvement.
//!
//! This example highlights **all** the refined and newly-added features from
//! the scry-chart remediation:
//!
//!   Page 1: Smooth, Step & Filled Line Modes
//!   Page 2: Line Width & Trend Lines  
//!   Page 3: Pie ↔ Donut Transform + Start Angle
//!   Page 4: Bar Corner Radius & Horizontal Stacked
//!   Page 5: Annotations & Styled Reference Lines
//!   Page 6: Dynamic Heatmap Labels + Correlation
//!   Page 7: All Five Themes Side-by-Side
//!   Page 8: Histogram h_lines + Density Mode
//!   Page 9: Scatter Marker Gallery + Custom Size
//!
//! Run with: `cargo run -p scry-chart --example refined_features`
//! Navigate:  ← → arrows  |  1-9 number keys  |  q to quit

use std::io::stdout;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use scry_chart::prelude::*;
use scry_engine::style::Color as PxColor;

// ---------------------------------------------------------------------------
// Pages
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Page {
    LineModes,
    LineWidthTrend,
    PieDonut,
    BarRefined,
    AnnotationsRefs,
    HeatmapDynamic,
    ThemeGallery,
    HistogramRefined,
    ScatterGallery,
}

impl Page {
    const ALL: [Page; 9] = [
        Page::LineModes,
        Page::LineWidthTrend,
        Page::PieDonut,
        Page::BarRefined,
        Page::AnnotationsRefs,
        Page::HeatmapDynamic,
        Page::ThemeGallery,
        Page::HistogramRefined,
        Page::ScatterGallery,
    ];

    fn index(self) -> usize {
        Self::ALL.iter().position(|&p| p == self).unwrap_or(0)
    }

    fn title(self) -> &'static str {
        match self {
            Page::LineModes => "1: Smooth, Step & Filled Lines",
            Page::LineWidthTrend => "2: Line Width + Trend Lines",
            Page::PieDonut => "3: Pie ↔ Donut + Start Angle",
            Page::BarRefined => "4: Corner Radius & Horizontal Stacked",
            Page::AnnotationsRefs => "5: Annotations & Styled References",
            Page::HeatmapDynamic => "6: Dynamic Heatmap Labels + Correlation",
            Page::ThemeGallery => "7: Theme Gallery — All Five",
            Page::HistogramRefined => "8: Histogram — h_lines + Density",
            Page::ScatterGallery => "9: Scatter — Marker Gallery",
        }
    }

    fn next(self) -> Page {
        let i = (self.index() + 1) % Self::ALL.len();
        Self::ALL[i]
    }

    fn prev(self) -> Page {
        let i = (self.index() + Self::ALL.len() - 1) % Self::ALL.len();
        Self::ALL[i]
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

struct AppState {
    page: Page,
    states: Vec<ChartState>,
}

impl AppState {
    fn new() -> Self {
        let states = (0..6).map(|_| ChartState::auto()).collect();
        Self {
            page: Page::LineModes,
            states,
        }
    }

    fn flush(&mut self) {
        for s in &mut self.states {
            let _ = s.flush();
        }
    }

    fn cleanup(&mut self) {
        for s in &mut self.states {
            s.cleanup();
        }
    }
}

// ---------------------------------------------------------------------------
// Data generators
// ---------------------------------------------------------------------------

fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| lo + (hi - lo) * i as f64 / (n - 1).max(1) as f64)
        .collect()
}

fn noisy_sin(x: &[f64], freq: f64, amp: f64, seed: u64) -> Vec<f64> {
    x.iter()
        .enumerate()
        .map(|(i, &v)| {
            let noise = ((i as u64 * 2654435761 + seed) % 1000) as f64 / 1000.0 - 0.5;
            (v * freq).sin() * amp + noise * amp * 0.15
        })
        .collect()
}

fn random_normal(n: usize, mean: f64, std: f64, seed: u64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let mut sum = 0.0;
            for k in 0..6 {
                let v = ((i as u64 * 2654435761 + k * 7919 + seed) % 10000) as f64 / 10000.0;
                sum += v;
            }
            mean + (sum - 3.0) * std
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Page 1: Line Modes (smooth, step, filled)
// ---------------------------------------------------------------------------

fn build_line_modes() -> (Chart, Chart, Chart) {
    let x = linspace(0.0, 8.0, 20);
    let y = noisy_sin(&x, 1.0, 3.0, 42);

    let smooth = Chart::line(&y)
        .title("Smooth Interpolation")
        .x_label("Time")
        .y_label("Signal")
        .x_values(x.clone())
        .smooth()
        .with_points()
        .theme(Theme::dark())
        .build();

    let step = Chart::line(&y)
        .title("Step Function")
        .x_values(x.clone())
        .step()
        .theme(Theme::ocean())
        .build();

    let filled = Chart::line(&y)
        .title("Filled Area + Points")
        .x_values(x)
        .filled()
        .with_points()
        .theme(Theme::forest())
        .build();

    (smooth, step, filled)
}

// ---------------------------------------------------------------------------
// Page 2: Line Width & Trend Lines
// ---------------------------------------------------------------------------

fn build_line_width_trend() -> (Chart, Chart) {
    let x = linspace(0.0, 10.0, 40);
    let y_thin = noisy_sin(&x, 0.8, 2.5, 10);
    let y_thick: Vec<f64> = x
        .iter()
        .map(|&v| 1.5 + v * 0.4 + (v * 0.3).cos() * 1.5)
        .collect();

    let widths = Chart::line(&y_thin)
        .title("Line Width: 1px vs 4px")
        .x_values(x.clone())
        .line_width(1.0)
        .add_named_series("Thick (4px)", &y_thick)
        .theme(Theme::dark())
        .build();

    let x_scatter = linspace(0.0, 8.0, 30);
    let y_scatter: Vec<f64> = x_scatter
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let noise = ((i as u64 * 2654435761 + 99) % 1000) as f64 / 500.0 - 1.0;
            v * 1.3 + noise * 2.0
        })
        .collect();

    let trend = Chart::scatter(&x_scatter, &y_scatter)
        .title("Scatter + Trend Line (Linear Regression)")
        .x_label("x")
        .y_label("y")
        .trend_line()
        .size(5.0)
        .theme(Theme::pastel())
        .build();

    (widths, trend)
}

// ---------------------------------------------------------------------------
// Page 3: Pie & Donut
// ---------------------------------------------------------------------------

fn build_pie_donut() -> (Chart, Chart, Chart) {
    let labels = vec![
        "Rust".into(),
        "Python".into(),
        "Go".into(),
        "TypeScript".into(),
        "C++".into(),
    ];
    let values = [35.0, 28.0, 18.0, 12.0, 7.0];

    let pie = Chart::pie(labels.clone(), &values)
        .title("Pie — Default Start")
        .theme(Theme::dark())
        .build();

    let donut = Chart::pie(labels.clone(), &values)
        .title("Donut (0.55 ratio)")
        .donut(0.55)
        .theme(Theme::ocean())
        .build();

    let rotated = Chart::pie(labels, &values)
        .title("Donut — 90° Start + No %")
        .donut(0.4)
        .start_angle_degrees(90.0)
        .hide_percentages()
        .theme(Theme::forest())
        .build();

    (pie, donut, rotated)
}

// ---------------------------------------------------------------------------
// Page 4: Bar Refined
// ---------------------------------------------------------------------------

fn build_bar_refined() -> (Chart, Chart, Chart) {
    let cats: Vec<String> = vec!["Alpha", "Beta", "Gamma", "Delta"]
        .into_iter()
        .map(String::from)
        .collect();
    let v1 = [45.0, 72.0, 58.0, 90.0];
    let v2 = [30.0, 55.0, 42.0, 65.0];

    // Theme-default corner radius (Option<f32> = None → falls back to theme)
    let theme_radius = Chart::bar(cats.clone(), &v1)
        .title("Theme Corner Radius (default)")
        .add_named_series("Series B", &v2)
        .series_labels(&["Revenue", "Costs"])
        .show_values()
        .y_range(0.0, 100.0)
        .theme(Theme::dark())
        .build();

    // Sharp corners (radius = 0)
    let sharp = Chart::bar(cats.clone(), &v1)
        .title("Sharp Corners (r=0)")
        .corner_radius(0.0)
        .horizontal()
        .theme(Theme::pastel())
        .build();

    // Stacked + horizontal combo
    let stacked_h = Chart::bar(cats, &v1)
        .title("Stacked + Horizontal")
        .add_named_series("S2", &v2)
        .stacked()
        .horizontal()
        .corner_radius(6.0)
        .theme(Theme::ocean())
        .build();

    (theme_radius, sharp, stacked_h)
}

// ---------------------------------------------------------------------------
// Page 5: Annotations & Reference Lines
// ---------------------------------------------------------------------------

fn build_annotations() -> Chart {
    let x = linspace(0.0, 10.0, 50);
    let y: Vec<f64> = x
        .iter()
        .map(|&v| 3.0 + (v * 0.8).sin() * 2.5 + v * 0.3)
        .collect();

    // Find the peak
    let peak_idx = y
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let peak_x = x[peak_idx];
    let peak_y = y[peak_idx];

    let mean = y.iter().sum::<f64>() / y.len() as f64;

    Chart::line_xy(&x, &y)
        .title("Annotations & Styled Reference Lines")
        .x_label("Time (s)")
        .y_label("Signal (mV)")
        .smooth()
        .with_points()
        // Reference lines with colors
        .h_line_styled(mean, PxColor::from_rgba8(255, 200, 50, 200))
        .v_line_styled(5.0, PxColor::from_rgba8(100, 200, 255, 180))
        // Annotations
        .annotate(peak_x, peak_y, format!("Peak: {peak_y:.1}"))
        .annotate(0.5, mean + 0.5, format!("Mean: {mean:.1}"))
        .annotate(5.0, 2.0, "Threshold")
        .theme(Theme::dark())
        .build()
}

// ---------------------------------------------------------------------------
// Page 6: Heatmap Dynamic Labels + Correlation
// ---------------------------------------------------------------------------

fn build_heatmap_dynamic() -> (Chart, Chart) {
    // Regular heatmap with very long labels (tests dynamic margin)
    let data = vec![
        vec![85.0, 72.0, 91.0, 68.0],
        vec![77.0, 88.0, 65.0, 94.0],
        vec![92.0, 60.0, 83.0, 71.0],
    ];
    let regular = Chart::heatmap(data)
        .title("Performance Metrics — Dynamic Margins")
        .row_labels(vec![
            "Engineering Team".into(),
            "Product Design".into(),
            "Quality Assurance".into(),
        ])
        .col_labels(vec![
            "Q1 2025".into(),
            "Q2 2025".into(),
            "Q3 2025".into(),
            "Q4 2025".into(),
        ])
        .values(true)
        .cell_radius(4.0)
        .cell_gap(3.0)
        .theme(Theme::dark())
        .build();

    // Correlation matrix with custom color range
    let corr_data = vec![
        vec![1.00, 0.85, -0.42, 0.33, 0.71],
        vec![0.85, 1.00, -0.30, 0.50, 0.62],
        vec![-0.42, -0.30, 1.00, -0.15, -0.38],
        vec![0.33, 0.50, -0.15, 1.00, 0.45],
        vec![0.71, 0.62, -0.38, 0.45, 1.00],
    ];
    let corr_labels = vec![
        "CPU".into(),
        "Memory".into(),
        "Latency".into(),
        "Throughput".into(),
        "Score".into(),
    ];
    let correlation = Heatmap::correlation(corr_data, corr_labels)
        .title("Correlation Matrix — System Metrics")
        .values(true)
        .cell_radius(5.0)
        .cell_gap(2.0)
        .build();

    (regular, correlation)
}

// ---------------------------------------------------------------------------
// Page 7: Theme Gallery
// ---------------------------------------------------------------------------

fn build_theme_gallery() -> Vec<Chart> {
    let data = [12.0, 28.0, 18.0, 35.0, 22.0, 30.0, 15.0, 40.0];

    let themes: Vec<(&str, Theme)> = vec![
        ("Dark", Theme::dark()),
        ("Light", Theme::light()),
        ("Pastel", Theme::pastel()),
        ("Ocean", Theme::ocean()),
        ("Forest", Theme::forest()),
    ];

    themes
        .into_iter()
        .map(|(name, theme)| {
            Chart::line(&data)
                .title(format!("Theme: {name}"))
                .smooth()
                .with_points()
                .filled()
                .theme(theme)
                .build()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Page 8: Histogram Refined
// ---------------------------------------------------------------------------

fn build_histogram_refined() -> (Chart, Chart) {
    let data = random_normal(400, 50.0, 12.0, 777);
    let mean = data.iter().sum::<f64>() / data.len() as f64;

    let count = Chart::histogram(&data)
        .title("Frequency + h_line at Mean")
        .x_label("Value")
        .y_label("Count")
        .bins(25)
        .h_line(mean / 2.0)
        .v_line(mean)
        .no_legend()
        .theme(Theme::dark())
        .build();

    let density = Chart::histogram(&data)
        .title("Density Mode (opacity 0.6)")
        .x_label("Value")
        .y_label("Density")
        .bins(20)
        .density()
        .opacity(0.6)
        .v_line(mean)
        .theme(Theme::ocean())
        .build();

    (count, density)
}

// ---------------------------------------------------------------------------
// Page 9: Scatter Marker Gallery
// ---------------------------------------------------------------------------

fn build_scatter_gallery() -> (Chart, Chart) {
    let x = linspace(0.0, 6.0, 15);
    let y_circle: Vec<f64> = x.iter().map(|&v| (v * 0.8).sin() * 3.0 + 5.0).collect();
    let y_square: Vec<f64> = x.iter().map(|&v| (v * 0.6).cos() * 2.5 + 3.0).collect();
    let y_diamond: Vec<f64> = x.iter().map(|&v| v * 0.5 + 1.0).collect();

    let markers = Chart::scatter(&x, &y_circle)
        .title("Marker Types: Circle, Square, Diamond")
        .x_label("x")
        .y_label("y")
        .marker(Marker::Circle)
        .size(7.0)
        .add_series(
            Series::new("sq_x", x.clone()),
            Series::new("Square", y_square),
        )
        .add_series(
            Series::new("dm_x", x.clone()),
            Series::new("Diamond", y_diamond),
        )
        .connected()
        .theme(Theme::dark())
        .build();

    let x2 = linspace(0.0, 8.0, 20);
    let y_cross: Vec<f64> = x2.iter().map(|&v| (v * 1.2).sin() * 2.0 + 4.0).collect();
    let y_tri: Vec<f64> = x2.iter().map(|&v| (v * 0.9).cos() * 2.5 + 2.0).collect();

    let more = Chart::scatter(&x2, &y_cross)
        .title("Cross & Triangle + Large Size (8px)")
        .marker(Marker::Cross)
        .size(8.0)
        .add_series(
            Series::new("t_x", x2.clone()),
            Series::new("Triangle", y_tri),
        )
        .connected()
        .theme(Theme::pastel())
        .build();

    (markers, more)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let mut app = AppState::new();
    let mut prev_page = app.page;

    loop {
        terminal.draw(|frame| render_page(frame, &mut app))?;
        app.flush();

        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Right | KeyCode::Char('l') => app.page = app.page.next(),
                    KeyCode::Left | KeyCode::Char('h') => app.page = app.page.prev(),
                    KeyCode::Char('1') => app.page = Page::LineModes,
                    KeyCode::Char('2') => app.page = Page::LineWidthTrend,
                    KeyCode::Char('3') => app.page = Page::PieDonut,
                    KeyCode::Char('4') => app.page = Page::BarRefined,
                    KeyCode::Char('5') => app.page = Page::AnnotationsRefs,
                    KeyCode::Char('6') => app.page = Page::HeatmapDynamic,
                    KeyCode::Char('7') => app.page = Page::ThemeGallery,
                    KeyCode::Char('8') => app.page = Page::HistogramRefined,
                    KeyCode::Char('9') => app.page = Page::ScatterGallery,
                    _ => {}
                }

                if app.page != prev_page {
                    app.cleanup();
                    prev_page = app.page;
                    while event::poll(std::time::Duration::from_millis(0))? {
                        let _ = event::read()?;
                    }
                }
            }
        }
    }

    app.cleanup();
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}

fn render_page(frame: &mut Frame, app: &mut AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(3)])
        .split(frame.area());

    let area = chunks[0];

    match app.page {
        Page::LineModes => {
            let (c1, c2, c3) = build_line_modes();
            let cols = Layout::horizontal([
                Constraint::Percentage(33),
                Constraint::Percentage(34),
                Constraint::Percentage(33),
            ])
            .split(area);
            frame.render_stateful_widget(ChartWidget::new(&c1), cols[0], &mut app.states[0]);
            frame.render_stateful_widget(ChartWidget::new(&c2), cols[1], &mut app.states[1]);
            frame.render_stateful_widget(ChartWidget::new(&c3), cols[2], &mut app.states[2]);
        }
        Page::LineWidthTrend => {
            let (c1, c2) = build_line_width_trend();
            let cols = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(area);
            frame.render_stateful_widget(ChartWidget::new(&c1), cols[0], &mut app.states[0]);
            frame.render_stateful_widget(ChartWidget::new(&c2), cols[1], &mut app.states[1]);
        }
        Page::PieDonut => {
            let (c1, c2, c3) = build_pie_donut();
            let cols = Layout::horizontal([
                Constraint::Percentage(33),
                Constraint::Percentage(34),
                Constraint::Percentage(33),
            ])
            .split(area);
            frame.render_stateful_widget(ChartWidget::new(&c1), cols[0], &mut app.states[0]);
            frame.render_stateful_widget(ChartWidget::new(&c2), cols[1], &mut app.states[1]);
            frame.render_stateful_widget(ChartWidget::new(&c3), cols[2], &mut app.states[2]);
        }
        Page::BarRefined => {
            let (c1, c2, c3) = build_bar_refined();
            let cols = Layout::horizontal([
                Constraint::Percentage(33),
                Constraint::Percentage(34),
                Constraint::Percentage(33),
            ])
            .split(area);
            frame.render_stateful_widget(ChartWidget::new(&c1), cols[0], &mut app.states[0]);
            frame.render_stateful_widget(ChartWidget::new(&c2), cols[1], &mut app.states[1]);
            frame.render_stateful_widget(ChartWidget::new(&c3), cols[2], &mut app.states[2]);
        }
        Page::AnnotationsRefs => {
            let chart = build_annotations();
            frame.render_stateful_widget(ChartWidget::new(&chart), area, &mut app.states[0]);
        }
        Page::HeatmapDynamic => {
            let (c1, c2) = build_heatmap_dynamic();
            let cols = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(area);
            frame.render_stateful_widget(ChartWidget::new(&c1), cols[0], &mut app.states[0]);
            frame.render_stateful_widget(ChartWidget::new(&c2), cols[1], &mut app.states[1]);
        }
        Page::ThemeGallery => {
            let charts = build_theme_gallery();
            // 5 themes in a 3-col top + 2-col bottom layout
            let rows = Layout::vertical([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(area);
            let top = Layout::horizontal([
                Constraint::Percentage(33),
                Constraint::Percentage(34),
                Constraint::Percentage(33),
            ])
            .split(rows[0]);
            let bot = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(rows[1]);

            frame.render_stateful_widget(ChartWidget::new(&charts[0]), top[0], &mut app.states[0]);
            frame.render_stateful_widget(ChartWidget::new(&charts[1]), top[1], &mut app.states[1]);
            frame.render_stateful_widget(ChartWidget::new(&charts[2]), top[2], &mut app.states[2]);
            frame.render_stateful_widget(ChartWidget::new(&charts[3]), bot[0], &mut app.states[3]);
            frame.render_stateful_widget(ChartWidget::new(&charts[4]), bot[1], &mut app.states[4]);
        }
        Page::HistogramRefined => {
            let (c1, c2) = build_histogram_refined();
            let cols = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(area);
            frame.render_stateful_widget(ChartWidget::new(&c1), cols[0], &mut app.states[0]);
            frame.render_stateful_widget(ChartWidget::new(&c2), cols[1], &mut app.states[1]);
        }
        Page::ScatterGallery => {
            let (c1, c2) = build_scatter_gallery();
            let cols = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(area);
            frame.render_stateful_widget(ChartWidget::new(&c1), cols[0], &mut app.states[0]);
            frame.render_stateful_widget(ChartWidget::new(&c2), cols[1], &mut app.states[1]);
        }
    }

    // Status bar
    let page_num = app.page.index() + 1;
    let status_text = format!(
        " {} │ ← → navigate │ 1-9 jump │ q quit │ page {}/{}",
        app.page.title(),
        page_num,
        Page::ALL.len()
    );
    let status = Paragraph::new(status_text)
        .block(Block::default().borders(Borders::TOP))
        .style(Style::default().fg(Color::DarkGray));
    frame.render_widget(status, chunks[1]);
}
