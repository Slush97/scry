//! Pixelchart Power Demo — showcases every chart type, feature, and theme.
//!
//! Navigate with ← → or number keys. Press Space to toggle auto-advance.
//! Press 'q' to quit.
//!
//! Run with: `cargo run -p pixelchart --example demo`

use std::io::stdout;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use pixelchart::prelude::*;

// ---------------------------------------------------------------------------
// Pages
// ---------------------------------------------------------------------------

const NUM_PAGES: usize = 12;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
enum Page {
    ScatterConnected = 0,
    ScatterMultiSeries,
    LineAreaFill,
    BarGrouped,
    BarStackedHorizontal,
    HistogramFrequency,
    HistogramDensity,
    BoxPlotStandard,
    BoxPlotNotched,
    HeatmapCorrelation,
    HeatmapCustom,
    Dashboard,
}

impl Page {
    const ALL: [Page; NUM_PAGES] = [
        Page::ScatterConnected,
        Page::ScatterMultiSeries,
        Page::LineAreaFill,
        Page::BarGrouped,
        Page::BarStackedHorizontal,
        Page::HistogramFrequency,
        Page::HistogramDensity,
        Page::BoxPlotStandard,
        Page::BoxPlotNotched,
        Page::HeatmapCorrelation,
        Page::HeatmapCustom,
        Page::Dashboard,
    ];

    fn from_index(i: usize) -> Page {
        Self::ALL[i % NUM_PAGES]
    }
    fn index(self) -> usize {
        self as usize
    }
    fn next(self) -> Page {
        Self::from_index((self.index() + 1) % NUM_PAGES)
    }
    fn prev(self) -> Page {
        Self::from_index((self.index() + NUM_PAGES - 1) % NUM_PAGES)
    }

    fn title(self) -> &'static str {
        match self {
            Page::ScatterConnected => "Scatter — Connected + Annotations",
            Page::ScatterMultiSeries => "Scatter — Multi-Series + Markers",
            Page::LineAreaFill => "Line — Area Fill + Points",
            Page::BarGrouped => "Bar — Grouped Comparison",
            Page::BarStackedHorizontal => "Bar — Stacked + Horizontal",
            Page::HistogramFrequency => "Histogram — Frequency Bins",
            Page::HistogramDensity => "Histogram — Density + Overlay",
            Page::BoxPlotStandard => "Box Plot — Distributions",
            Page::BoxPlotNotched => "Box Plot — Notched Style",
            Page::HeatmapCorrelation => "Heatmap — Correlation Matrix",
            Page::HeatmapCustom => "Heatmap — Custom Colors + Layout",
            Page::Dashboard => "Dashboard — Multi-Panel Layout",
        }
    }

    fn features(self) -> &'static str {
        match self {
            Page::ScatterConnected => "connected · annotate · h_line · dark theme",
            Page::ScatterMultiSeries => "add_series · marker shapes · v_line · trend_line · pastel theme",
            Page::LineAreaFill => "filled · with_points · multi-series · h_line_styled · x_values",
            Page::BarGrouped => "multi-series · y_range · h_line · dark theme",
            Page::BarStackedHorizontal => "stacked · horizontal · corner_radius · gap · pastel theme",
            Page::HistogramFrequency => "bins · v_line · two-series overlay · dark theme",
            Page::HistogramDensity => "density · opacity · v_line_styled · light theme",
            Page::BoxPlotStandard => "4 groups · h_line · outliers · dark theme",
            Page::BoxPlotNotched => "notched · no_outliers · y_range · pastel theme",
            Page::HeatmapCorrelation => "correlation · cell_gap · cell_radius · values",
            Page::HeatmapCustom => "colors · range · values(false) · row/col labels · light theme",
            Page::Dashboard => "4 panels · mixed themes · line + bar + hist + scatter",
        }
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

struct AppState {
    page: Page,
    auto_advance: bool,
    ticks_on_page: u32,
    states: Vec<ChartState>,
}

impl AppState {
    fn new() -> Self {
        let states = (0..4).map(|_| ChartState::auto()).collect();
        Self {
            page: Page::ScatterConnected,
            auto_advance: true,
            ticks_on_page: 0,
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
// Data generators (deterministic, static)
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
            (v * freq).sin() * amp + noise * amp * 0.25
        })
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

// ---------------------------------------------------------------------------
// Chart builders (one per page)
// ---------------------------------------------------------------------------

fn build_scatter_connected() -> Chart {
    let x = linspace(0.0, 10.0, 60);
    let y = noisy_sin(&x, 1.0, 3.0, 42);

    Chart::scatter(&x, &y)
        .title("Signal Analysis — sin(x) + noise")
        .x_label("Time (s)")
        .y_label("Amplitude")
        .connected()
        .marker(Marker::Diamond)
        .h_line(0.0)
        .annotate(std::f64::consts::FRAC_PI_2, 3.0, "peak")
        .annotate(std::f64::consts::PI * 1.5, -3.0, "trough")
        .theme(Theme::dark())
        .build()
}

fn build_scatter_multi() -> Chart {
    let x = linspace(0.0, 8.0, 45);
    let y1 = noisy_sin(&x, 1.0, 2.5, 1);
    let y2: Vec<f64> = x.iter().map(|&v| v.sqrt() * 1.5).collect();
    let y3: Vec<f64> = x.iter().map(|&v| (v * 0.7).cos() * 2.0 + 1.0).collect();

    Chart::scatter(&x, &y1)
        .title("Multi-Series — Diverse Markers")
        .x_label("x")
        .y_label("y")
        .marker(Marker::Circle)
        .add_series(
            Series::new("sqrt", x.clone()),
            Series::new("", y2),
        )
        .add_series(
            Series::new("cos wave", x.clone()),
            Series::new("", y3),
        )
        .connected()
        .v_line(4.0)
        .trend_line()
        .y_range(-4.0, 6.0)
        .theme(Theme::pastel())
        .build()
}

fn build_line_area_fill() -> Chart {
    let x = linspace(0.0, 12.0, 65);
    let revenue: Vec<f64> = x
        .iter()
        .map(|&v| 3.0 + v * 0.7 + (v * 0.5).sin() * 2.0)
        .collect();
    let costs: Vec<f64> = x
        .iter()
        .map(|&v| 2.0 + v * 0.35 + (v * 0.3).cos() * 0.8)
        .collect();
    let profit: Vec<f64> = revenue.iter().zip(&costs).map(|(r, c)| r - c).collect();

    Chart::line(&revenue)
        .title("Revenue vs Costs — Filled Area")
        .x_label("Month")
        .y_label("$M")
        .x_values(x)
        .add_series(Series::new("Costs", costs))
        .add_series(Series::new("Profit", profit))
        .filled()
        .with_points()
        .h_line(5.0)
        .h_line_styled(
            8.0,
            ratatui_pixelcanvas::style::Color::from_rgba8(252, 129, 155, 180),
        )
        .y_range(0.0, 16.0)
        .theme(Theme::dark())
        .build()
}

fn build_bar_grouped() -> Chart {
    let labels: Vec<String> = vec!["Rust", "Python", "Go", "C++", "TypeScript", "Java"]
        .into_iter()
        .map(String::from)
        .collect();

    let perf = vec![96.0, 42.0, 78.0, 93.0, 28.0, 65.0];
    let safety = vec![99.0, 58.0, 74.0, 38.0, 72.0, 55.0];
    let ergonomics = vec![82.0, 96.0, 68.0, 32.0, 90.0, 50.0];

    Chart::bar(labels, &perf)
        .title("Language Comparison — Grouped Bars")
        .y_label("Score")
        .add_series(Series::new("Safety", safety))
        .add_series(Series::new("Ergonomics", ergonomics))
        .y_range(0.0, 100.0)
        .h_line(75.0)
        .theme(Theme::dark())
        .build()
}

fn build_bar_stacked_horizontal() -> Chart {
    let labels: Vec<String> = vec!["Frontend", "Backend", "DevOps", "ML/AI", "Mobile"]
        .into_iter()
        .map(String::from)
        .collect();

    let q1 = vec![180.0, 210.0, 95.0, 150.0, 120.0];
    let q2 = vec![200.0, 190.0, 110.0, 180.0, 140.0];
    let q3 = vec![220.0, 230.0, 130.0, 210.0, 135.0];

    Chart::bar(labels, &q1)
        .title("Engineering Hours — Stacked Horizontal")
        .x_label("Total Hours")
        .add_series(Series::new("Q2", q2))
        .add_series(Series::new("Q3", q3))
        .stacked()
        .horizontal()
        .corner_radius(4.0)
        .gap(0.3)
        .theme(Theme::pastel())
        .build()
}

fn build_histogram_frequency() -> Chart {
    let data1 = pseudo_normal(400, 5.0, 1.5, 100);
    let data2 = pseudo_normal(300, 8.0, 1.2, 200);

    let mean1 = data1.iter().sum::<f64>() / data1.len() as f64;
    let mean2 = data2.iter().sum::<f64>() / data2.len() as f64;

    Chart::histogram(&data1)
        .title("Frequency — Two Normal Distributions")
        .x_label("Value")
        .y_label("Count")
        .bins(30)
        .add_series(Series::new("Group B", data2))
        .opacity(0.7)
        .v_line(mean1)
        .v_line(mean2)
        .theme(Theme::dark())
        .build()
}

fn build_histogram_density() -> Chart {
    let data = pseudo_normal(500, 0.0, 1.0, 333);
    let mean = data.iter().sum::<f64>() / data.len() as f64;

    Chart::histogram(&data)
        .title("Density — Normalized (area = 1)")
        .x_label("Standard Deviations")
        .y_label("Density")
        .bins(25)
        .density()
        .opacity(0.8)
        .v_line_styled(
            mean,
            ratatui_pixelcanvas::style::Color::from_rgba8(255, 90, 90, 220),
        )
        .v_line_styled(
            -1.0,
            ratatui_pixelcanvas::style::Color::from_rgba8(130, 130, 200, 150),
        )
        .v_line_styled(
            1.0,
            ratatui_pixelcanvas::style::Color::from_rgba8(130, 130, 200, 150),
        )
        .theme(Theme::light())
        .build()
}

fn build_boxplot_standard() -> Chart {
    Chart::boxplot(vec![
        ("Control", pseudo_normal(60, 10.0, 2.0, 300)),
        ("Drug A", pseudo_normal(60, 14.0, 3.0, 400)),
        ("Drug B", pseudo_normal(60, 12.0, 1.5, 500)),
        ("Drug C", pseudo_normal(60, 16.0, 4.0, 600)),
    ])
    .title("Clinical Trial — Distribution by Group")
    .x_label("Treatment Group")
    .y_label("Response Score")
    .h_line(12.0)
    .theme(Theme::dark())
    .build()
}

fn build_boxplot_notched() -> Chart {
    Chart::boxplot(vec![
        ("Jan", pseudo_normal(50, 22.0, 3.0, 10)),
        ("Feb", pseudo_normal(50, 20.0, 2.5, 20)),
        ("Mar", pseudo_normal(50, 25.0, 4.0, 30)),
        ("Apr", pseudo_normal(50, 28.0, 3.5, 40)),
        ("May", pseudo_normal(50, 32.0, 5.0, 50)),
        ("Jun", pseudo_normal(50, 35.0, 4.0, 60)),
    ])
    .title("Monthly Temperature — Notched Box Plot")
    .x_label("Month")
    .y_label("Temperature (°C)")
    .notched()
    .no_outliers()
    .y_range(10.0, 50.0)
    .theme(Theme::pastel())
    .build()
}

fn build_heatmap_correlation() -> Chart {
    let labels: Vec<String> = vec!["Height", "Weight", "Age", "BP", "HR", "Chol", "BMI"]
        .into_iter()
        .map(String::from)
        .collect();

    let data = vec![
        vec![ 1.00,  0.82,  0.15,  0.30, -0.10,  0.20,  0.75],
        vec![ 0.82,  1.00,  0.25,  0.45, -0.05,  0.35,  0.90],
        vec![ 0.15,  0.25,  1.00,  0.55,  0.10,  0.60,  0.20],
        vec![ 0.30,  0.45,  0.55,  1.00,  0.40,  0.50,  0.40],
        vec![-0.10, -0.05,  0.10,  0.40,  1.00,  0.15, -0.08],
        vec![ 0.20,  0.35,  0.60,  0.50,  0.15,  1.00,  0.30],
        vec![ 0.75,  0.90,  0.20,  0.40, -0.08,  0.30,  1.00],
    ];

    Heatmap::correlation(data, labels)
        .title("Health Metrics — Correlation Matrix")
        .cell_gap(3.0)
        .cell_radius(3.0)
        .build()
}

fn build_heatmap_custom() -> Chart {
    let rows: Vec<String> = vec!["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        .into_iter()
        .map(String::from)
        .collect();
    let cols: Vec<String> = vec!["6am", "9am", "12pm", "3pm", "6pm", "9pm", "12am"]
        .into_iter()
        .map(String::from)
        .collect();

    #[rustfmt::skip]
    let data = vec![
        vec![  2.0, 15.0, 30.0, 25.0, 40.0, 20.0,  5.0],
        vec![  3.0, 18.0, 35.0, 28.0, 45.0, 22.0,  4.0],
        vec![  1.0, 12.0, 28.0, 22.0, 38.0, 18.0,  6.0],
        vec![  4.0, 20.0, 32.0, 30.0, 42.0, 25.0,  3.0],
        vec![  5.0, 22.0, 38.0, 35.0, 50.0, 30.0,  8.0],
        vec![ 10.0,  8.0, 20.0, 25.0, 35.0, 40.0, 15.0],
        vec![  8.0,  5.0, 15.0, 18.0, 25.0, 30.0, 12.0],
    ];

    Chart::heatmap(data)
        .title("Website Traffic — Users per Hour")
        .row_labels(rows)
        .col_labels(cols)
        .colors(
            ratatui_pixelcanvas::style::Color::from_rgba8(20, 30, 60, 255),
            ratatui_pixelcanvas::style::Color::from_rgba8(80, 255, 120, 255),
        )
        .range(0.0, 55.0)
        .values(false)
        .cell_radius(2.0)
        .cell_gap(2.0)
        .theme(Theme::light())
        .build()
}

fn build_dashboard() -> (Chart, Chart, Chart, Chart) {
    // Line chart
    let x = linspace(0.0, 8.0, 40);
    let line = Chart::line(&noisy_sin(&x, 1.2, 4.0, 10))
        .title("Signal Waveform")
        .x_values(x)
        .filled()
        .with_points()
        .h_line(0.0)
        .theme(Theme::dark())
        .build();

    // Bar chart
    let bar_labels: Vec<String> = vec!["Q1", "Q2", "Q3", "Q4"]
        .into_iter()
        .map(String::from)
        .collect();
    let bar = Chart::bar(bar_labels, &[42.0, 58.0, 35.0, 71.0])
        .title("Quarterly Revenue")
        .y_label("$M")
        .y_range(0.0, 80.0)
        .theme(Theme::pastel())
        .build();

    // Histogram
    let hist = Chart::histogram(&pseudo_normal(200, 0.0, 1.0, 77))
        .title("Noise Distribution")
        .bins(18)
        .theme(Theme::dark())
        .build();

    // Scatter
    let sx: Vec<f64> = (0..35).map(|i| i as f64 * 0.5).collect();
    let sy: Vec<f64> = sx
        .iter()
        .map(|&x| x.sqrt() * 2.0 + (x * 0.8).sin())
        .collect();
    let scatter = Chart::scatter(&sx, &sy)
        .title("Growth Curve")
        .connected()
        .marker(Marker::Triangle)
        .trend_line()
        .theme(Theme::dark())
        .build();

    (line, bar, hist, scatter)
}

// ---------------------------------------------------------------------------
// Main + render loop
// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let mut app = AppState::new();
    let mut prev_page = app.page;

    // Auto-advance timer: ~40 ticks × 100ms = 4 seconds
    let auto_ticks = 40u32;

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
                    KeyCode::Right | KeyCode::Char('l') => {
                        app.page = app.page.next();
                        app.ticks_on_page = 0;
                    }
                    KeyCode::Left | KeyCode::Char('h') => {
                        app.page = app.page.prev();
                        app.ticks_on_page = 0;
                    }
                    KeyCode::Char(' ') => {
                        app.auto_advance = !app.auto_advance;
                        app.ticks_on_page = 0;
                    }
                    KeyCode::Char(c) if c.is_ascii_digit() => {
                        let idx = match c {
                            '0' => 9,  // 0 = page 10
                            '-' => 10, // not reachable but safe
                            '=' => 11,
                            _ => c.to_digit(10).unwrap_or(1) as usize - 1,
                        };
                        if idx < NUM_PAGES {
                            app.page = Page::from_index(idx);
                            app.ticks_on_page = 0;
                        }
                    }
                    _ => {}
                }

                // On page change: clean up old images
                if app.page != prev_page {
                    app.cleanup();
                    prev_page = app.page;
                    while event::poll(std::time::Duration::from_millis(0))? {
                        let _ = event::read()?;
                    }
                }
            }
        }

        // Auto-advance
        app.ticks_on_page += 1;
        if app.auto_advance && app.ticks_on_page >= auto_ticks {
            let new_page = app.page.next();
            if new_page != app.page {
                app.cleanup();
                app.page = new_page;
                prev_page = new_page;
                app.ticks_on_page = 0;
            }
        }
    }

    app.cleanup();
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

fn render_page(frame: &mut Frame, app: &mut AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(3)])
        .split(frame.area());

    let chart_area = chunks[0];

    match app.page {
        Page::ScatterConnected => {
            let chart = build_scatter_connected();
            frame.render_stateful_widget(
                ChartWidget::new(&chart),
                chart_area,
                &mut app.states[0],
            );
        }
        Page::ScatterMultiSeries => {
            let chart = build_scatter_multi();
            frame.render_stateful_widget(
                ChartWidget::new(&chart),
                chart_area,
                &mut app.states[0],
            );
        }
        Page::LineAreaFill => {
            let chart = build_line_area_fill();
            frame.render_stateful_widget(
                ChartWidget::new(&chart),
                chart_area,
                &mut app.states[0],
            );
        }
        Page::BarGrouped => {
            let chart = build_bar_grouped();
            frame.render_stateful_widget(
                ChartWidget::new(&chart),
                chart_area,
                &mut app.states[0],
            );
        }
        Page::BarStackedHorizontal => {
            let chart = build_bar_stacked_horizontal();
            frame.render_stateful_widget(
                ChartWidget::new(&chart),
                chart_area,
                &mut app.states[0],
            );
        }
        Page::HistogramFrequency => {
            let chart = build_histogram_frequency();
            frame.render_stateful_widget(
                ChartWidget::new(&chart),
                chart_area,
                &mut app.states[0],
            );
        }
        Page::HistogramDensity => {
            let chart = build_histogram_density();
            frame.render_stateful_widget(
                ChartWidget::new(&chart),
                chart_area,
                &mut app.states[0],
            );
        }
        Page::BoxPlotStandard => {
            let chart = build_boxplot_standard();
            frame.render_stateful_widget(
                ChartWidget::new(&chart),
                chart_area,
                &mut app.states[0],
            );
        }
        Page::BoxPlotNotched => {
            let chart = build_boxplot_notched();
            frame.render_stateful_widget(
                ChartWidget::new(&chart),
                chart_area,
                &mut app.states[0],
            );
        }
        Page::HeatmapCorrelation => {
            let chart = build_heatmap_correlation();
            frame.render_stateful_widget(
                ChartWidget::new(&chart),
                chart_area,
                &mut app.states[0],
            );
        }
        Page::HeatmapCustom => {
            let chart = build_heatmap_custom();
            frame.render_stateful_widget(
                ChartWidget::new(&chart),
                chart_area,
                &mut app.states[0],
            );
        }
        Page::Dashboard => {
            let (c1, c2, c3, c4) = build_dashboard();
            let rows =
                Layout::vertical([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .split(chart_area);
            let top =
                Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .split(rows[0]);
            let bot =
                Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .split(rows[1]);

            frame.render_stateful_widget(ChartWidget::new(&c1), top[0], &mut app.states[0]);
            frame.render_stateful_widget(ChartWidget::new(&c2), top[1], &mut app.states[1]);
            frame.render_stateful_widget(ChartWidget::new(&c3), bot[0], &mut app.states[2]);
            frame.render_stateful_widget(ChartWidget::new(&c4), bot[1], &mut app.states[3]);
        }
    }

    // Status bar
    let page_num = app.page.index() + 1;
    let auto_indicator = if app.auto_advance { "▶ AUTO" } else { "⏸ MANUAL" };
    let progress = if app.auto_advance {
        let pct = (app.ticks_on_page as f64 / 40.0 * 100.0).min(100.0) as u8;
        format!(" {pct}%")
    } else {
        String::new()
    };

    let status_text = format!(
        " {page_num:>2}/{NUM_PAGES} │ {} │ {} │ {}{progress} │ ← → navigate │ 1-9,0 jump │ Space toggle │ q quit",
        app.page.title(),
        app.page.features(),
        auto_indicator,
    );

    let status = Paragraph::new(status_text)
        .block(Block::default().borders(Borders::TOP))
        .style(Style::default().fg(Color::DarkGray));
    frame.render_widget(status, chunks[1]);
}
