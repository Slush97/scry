//! Comprehensive feature showcase for scry-chart.
//!
//! Cycle through pages with ← → arrows or number keys (1-9, 0, -, =).
//! Press 'q' to quit.
//!
//! Run with: `cargo run -p scry-chart --example showcase`

use std::io::stdout;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use scry_chart::chart::OhlcEntry;
use scry_chart::formatter::{BinarySiFormatter, CurrencyFormatter, EngineeringFormatter};
use scry_chart::prelude::*;

// ---------------------------------------------------------------------------
// Pages
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Page {
    ScatterBasic,
    ScatterAdvanced,
    LineChart,
    BarChart,
    Histogram,
    BoxPlot,
    Heatmap,
    PieChart,
    RadarChart,
    Candlestick,
    MultiPanel,
    Formatters,
}

impl Page {
    const ALL: [Page; 12] = [
        Page::ScatterBasic,
        Page::ScatterAdvanced,
        Page::LineChart,
        Page::BarChart,
        Page::Histogram,
        Page::BoxPlot,
        Page::Heatmap,
        Page::PieChart,
        Page::RadarChart,
        Page::Candlestick,
        Page::MultiPanel,
        Page::Formatters,
    ];

    fn index(self) -> usize {
        Self::ALL.iter().position(|&p| p == self).unwrap_or(0)
    }

    fn title(self) -> &'static str {
        match self {
            Page::ScatterBasic => " 1: Scatter — Basic",
            Page::ScatterAdvanced => " 2: Scatter — Multi-Series + Markers",
            Page::LineChart => " 3: Line — Fill + Reference Lines",
            Page::BarChart => " 4: Bar — Grouped vs Stacked",
            Page::Histogram => " 5: Histogram — Bins + Density",
            Page::BoxPlot => " 6: Box Plot — Distributions",
            Page::Heatmap => " 7: Heatmap — Correlation Matrix",
            Page::PieChart => " 8: Pie — Proportional Slices",
            Page::RadarChart => " 9: Radar — Multi-Axis Comparison",
            Page::Candlestick => "10: Candlestick — OHLC Financial",
            Page::MultiPanel => "11: Multi-Panel Dashboard",
            Page::Formatters => "12: Formatters — Locale + Notation",
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
    // One ChartState per simultaneous chart on screen
    states: Vec<ChartState>,
}

impl AppState {
    fn new() -> Self {
        // Pre-allocate states (max 4 for multi-panel)
        let states = (0..4).map(|_| ChartState::auto()).collect();
        Self {
            page: Page::ScatterBasic,
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
            (v * freq).sin() * amp + noise * amp * 0.3
        })
        .collect()
}

fn random_normal(n: usize, mean: f64, std: f64, seed: u64) -> Vec<f64> {
    // Simple pseudo-normal via central limit theorem (sum of 6 uniforms)
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
// Chart builders per page
// ---------------------------------------------------------------------------

fn build_scatter_basic() -> Chart {
    let x = linspace(0.0, 10.0, 50);
    let y = noisy_sin(&x, 1.0, 3.0, 42);

    Chart::scatter(&x, &y)
        .title("Basic Scatter — sin(x) + noise")
        .x_label("Time (s)")
        .y_label("Amplitude")
        .connected()
        .h_line(0.0)
        .theme(Theme::dark())
        .build()
}

fn build_scatter_advanced() -> (Chart, Chart) {
    let x = linspace(0.0, 8.0, 40);
    let y1 = noisy_sin(&x, 1.0, 2.0, 1);
    let _y2 = noisy_sin(&x, 1.5, 1.5, 2);
    let y3: Vec<f64> = x.iter().map(|&v| v.sqrt() * 1.2).collect();

    let scatter = Chart::scatter(&x, &y1)
        .title("Marker Shapes")
        .x_label("x")
        .y_label("y")
        .marker(Marker::Diamond)
        .add_series(Series::new("sqrt", x.clone()), Series::new("", y3))
        .y_range(-3.0, 5.0)
        .v_line(4.0)
        .theme(Theme::dark())
        .build();

    // Second scatter with different markers — show all marker types
    let x2 = linspace(0.0, 6.0, 25);
    let y_circle: Vec<f64> = x2.iter().map(|&v| v.sin() * 2.0 + 4.0).collect();
    let y_square: Vec<f64> = x2.iter().map(|&v| v.cos() * 1.5 + 1.0).collect();

    let scatter2 = Chart::scatter(&x2, &y_circle)
        .title("Multi-Series + Connected")
        .add_series(Series::new("cos", x2.clone()), Series::new("", y_square))
        .connected()
        .marker(Marker::Circle)
        .theme(Theme::pastel())
        .build();

    (scatter, scatter2)
}

fn build_line_chart() -> Chart {
    let x = linspace(0.0, 12.0, 60);
    let revenue: Vec<f64> = x
        .iter()
        .map(|&v| 3.0 + v * 0.8 + (v * 0.5).sin() * 2.0)
        .collect();
    let costs: Vec<f64> = x.iter().map(|&v| 2.0 + v * 0.4 + (v * 0.3).cos()).collect();
    let profit: Vec<f64> = revenue.iter().zip(&costs).map(|(r, c)| r - c).collect();

    Chart::line(&revenue)
        .title("Revenue vs Costs — Filled Area + Points")
        .x_label("Month")
        .y_label("$M")
        .x_values(x)
        .add_series(Series::new("Costs", costs))
        .add_series(Series::new("Profit", profit))
        .filled()
        .with_points()
        .h_line(5.0)
        .y_range(0.0, 15.0)
        .theme(Theme::dark())
        .build()
}

fn build_bar_charts() -> (Chart, Chart) {
    let labels: Vec<String> = vec!["Rust", "Python", "Go", "C++", "TypeScript"]
        .into_iter()
        .map(String::from)
        .collect();

    let perf = vec![95.0, 45.0, 80.0, 92.0, 30.0];
    let safety = vec![98.0, 60.0, 75.0, 40.0, 70.0];
    let ergonomics = vec![80.0, 95.0, 70.0, 35.0, 88.0];

    // Grouped bars
    let grouped = Chart::bar(labels.clone(), &perf)
        .title("Grouped — Language Comparison")
        .y_label("Score")
        .add_series(Series::new("Safety", safety.clone()))
        .add_series(Series::new("Ergonomics", ergonomics.clone()))
        .y_range(0.0, 100.0)
        .h_line(75.0)
        .theme(Theme::dark())
        .build();

    // Stacked bars
    let stacked = Chart::bar(labels, &perf)
        .title("Stacked — Total Scores")
        .y_label("Combined Score")
        .add_series(Series::new("Safety", safety))
        .add_series(Series::new("Ergonomics", ergonomics))
        .stacked()
        .theme(Theme::pastel())
        .build();

    (grouped, stacked)
}

fn build_histogram() -> (Chart, Chart) {
    let data1 = random_normal(300, 5.0, 1.5, 100);
    let data2 = random_normal(300, 8.0, 1.0, 200);

    let mean1 = data1.iter().sum::<f64>() / data1.len() as f64;
    let mean2 = data2.iter().sum::<f64>() / data2.len() as f64;

    // Count histogram
    let hist1 = Chart::histogram(&data1)
        .title("Frequency — Normal Distribution")
        .x_label("Value")
        .y_label("Count")
        .bins(25)
        .v_line(mean1)
        .theme(Theme::dark())
        .build();

    // Density histogram with overlay
    let hist2 = Chart::histogram(&data2)
        .title("Density — Normalized")
        .x_label("Value")
        .y_label("Density")
        .bins(20)
        .density()
        .opacity(0.7)
        .v_line(mean2)
        .theme(Theme::pastel())
        .build();

    (hist1, hist2)
}

fn build_boxplot() -> Chart {
    let control = random_normal(50, 10.0, 2.0, 300);
    let treatment_a = random_normal(50, 14.0, 3.0, 400);
    let treatment_b = random_normal(50, 12.0, 1.5, 500);
    let treatment_c = random_normal(50, 16.0, 4.0, 600);

    Chart::boxplot(vec![
        ("Control", control),
        ("Drug A", treatment_a),
        ("Drug B", treatment_b),
        ("Drug C", treatment_c),
    ])
    .title("Clinical Trial — Distribution by Group")
    .y_label("Response Score")
    .h_line(12.0)
    .theme(Theme::dark())
    .build()
}

fn build_heatmap() -> Chart {
    // Simulated correlation matrix
    let labels: Vec<String> = vec!["Height", "Weight", "Age", "BP", "HR", "Chol"]
        .into_iter()
        .map(String::from)
        .collect();

    let data = vec![
        vec![1.00, 0.82, 0.15, 0.30, -0.10, 0.20],
        vec![0.82, 1.00, 0.25, 0.45, -0.05, 0.35],
        vec![0.15, 0.25, 1.00, 0.55, 0.10, 0.60],
        vec![0.30, 0.45, 0.55, 1.00, 0.40, 0.50],
        vec![-0.10, -0.05, 0.10, 0.40, 1.00, 0.15],
        vec![0.20, 0.35, 0.60, 0.50, 0.15, 1.00],
    ];

    Heatmap::correlation(data, labels)
        .title("Correlation Matrix — Health Metrics")
        .cell_gap(3.0)
        .cell_radius(3.0)
        .build()
}

fn build_pie() -> (Chart, Chart) {
    let labels: Vec<String> = vec!["Rust", "Python", "Go", "TypeScript", "C++", "Other"]
        .into_iter()
        .map(String::from)
        .collect();
    let values = vec![35.0, 25.0, 15.0, 12.0, 8.0, 5.0];

    let pie = Chart::pie(labels.clone(), &values)
        .title("Language Popularity — Pie")
        .theme(Theme::dark())
        .build();

    let donut = Chart::pie(labels, &values)
        .title("Language Popularity — Donut")
        .donut(0.5)
        .theme(Theme::pastel())
        .build();

    (pie, donut)
}

fn build_radar() -> Chart {
    Chart::radar(vec![
        "Speed", "Power", "Defense", "Magic", "Stamina", "Luck",
    ])
    .add_series("Warrior", &[9.0, 8.0, 7.0, 2.0, 8.0, 4.0])
    .add_series("Mage", &[3.0, 4.0, 3.0, 10.0, 5.0, 6.0])
    .add_series("Rogue", &[8.0, 5.0, 4.0, 3.0, 6.0, 9.0])
    .title("Character Stats — Radar")
    .theme(Theme::ocean())
    .build()
}

fn build_candlestick() -> Chart {
    let data = vec![
        OhlcEntry::new(1.0, 100.0, 110.0, 95.0, 108.0),
        OhlcEntry::new(2.0, 108.0, 115.0, 102.0, 104.0),
        OhlcEntry::new(3.0, 104.0, 112.0, 100.0, 110.0),
        OhlcEntry::new(4.0, 110.0, 118.0, 106.0, 107.0),
        OhlcEntry::new(5.0, 107.0, 120.0, 105.0, 118.0),
        OhlcEntry::new(6.0, 118.0, 125.0, 112.0, 122.0),
        OhlcEntry::new(7.0, 122.0, 130.0, 118.0, 119.0),
        OhlcEntry::new(8.0, 119.0, 128.0, 116.0, 126.0),
        OhlcEntry::new(9.0, 126.0, 135.0, 122.0, 132.0),
        OhlcEntry::new(10.0, 132.0, 138.0, 128.0, 130.0),
    ];

    Chart::candlestick(data)
        .title("AAPL — Daily Candlestick")
        .x_label("Trading Day")
        .y_label("Price ($)")
        .theme(Theme::dark())
        .build()
}

fn build_dashboard() -> (Chart, Chart, Chart, Chart) {
    let x = linspace(0.0, 6.0, 30);
    let line = Chart::line(&noisy_sin(&x, 1.2, 4.0, 10))
        .title("Signal")
        .x_values(x)
        .filled()
        .with_points()
        .theme(Theme::dark())
        .build();

    let bar_labels: Vec<String> = vec!["Q1", "Q2", "Q3", "Q4"]
        .into_iter()
        .map(String::from)
        .collect();
    let bar = Chart::bar(bar_labels, &[42.0, 58.0, 35.0, 71.0])
        .title("Quarterly")
        .theme(Theme::pastel())
        .build();

    let hist = Chart::histogram(&random_normal(200, 0.0, 1.0, 77))
        .title("Noise")
        .bins(15)
        .theme(Theme::dark())
        .build();

    let hm = Chart::heatmap(vec![
        vec![1.0, 0.8, -0.3, 0.5],
        vec![0.8, 1.0, 0.1, -0.2],
        vec![-0.3, 0.1, 1.0, 0.6],
        vec![0.5, -0.2, 0.6, 1.0],
    ])
    .row_labels(vec!["α".into(), "β".into(), "γ".into(), "δ".into()])
    .col_labels(vec!["α".into(), "β".into(), "γ".into(), "δ".into()])
    .title("Correlation")
    .build();

    (line, bar, hist, hm)
}

fn build_formatters() -> (Chart, Chart, Chart, Chart) {
    // Panel 1: European locale (comma decimals, period grouping)
    let x = linspace(0.0, 5.0, 30);
    let revenue: Vec<f64> = x
        .iter()
        .map(|&v| 1200.0 + v * 800.0 + (v * 2.0).sin() * 400.0)
        .collect();
    let locale_chart = Chart::line(&revenue)
        .title("European Locale — Revenue (€)")
        .x_label("Quarter")
        .y_label("Revenue")
        .x_values(x)
        .european_locale()
        .theme(Theme::dark())
        .build();

    // Panel 2: Engineering notation
    let x2 = linspace(0.0, 6.0, 25);
    let power: Vec<f64> = x2.iter().map(|&v| 10_f64.powf(v * 1.5)).collect();
    let eng_chart = Chart::line(&power)
        .title("Engineering — Power Levels")
        .x_label("Stage")
        .y_label("Watts")
        .x_values(x2)
        .y_formatter(EngineeringFormatter::default())
        .theme(Theme::pastel())
        .build();

    // Panel 3: Binary SI (file sizes)
    let x3 = linspace(0.0, 10.0, 20);
    let sizes: Vec<f64> = x3.iter().map(|&v| 1024.0_f64.powf(1.0 + v * 0.3)).collect();
    let binary_chart = Chart::line(&sizes)
        .title("Binary SI — File Sizes")
        .x_label("Build #")
        .y_label("Size")
        .x_values(x3)
        .y_formatter(BinarySiFormatter::default())
        .x_ticks_diagonal()
        .theme(Theme::dark())
        .build();

    // Panel 4: Accounting currency with 30° tick labels
    let labels: Vec<String> = vec!["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        .into_iter()
        .map(String::from)
        .collect();
    let values = vec![
        42_000.0, -15_000.0, 78_000.0, -8_000.0, 120_000.0, -32_000.0,
    ];
    let acct_chart = Chart::bar(labels, &values)
        .title("Accounting — P&L")
        .y_label("USD")
        .y_formatter(CurrencyFormatter {
            accounting: true,
            ..CurrencyFormatter::default()
        })
        .x_tick_angle(30.0)
        .theme(Theme::pastel())
        .build();

    (locale_chart, eng_chart, binary_chart, acct_chart)
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
                    KeyCode::Char('1') => app.page = Page::ScatterBasic,
                    KeyCode::Char('2') => app.page = Page::ScatterAdvanced,
                    KeyCode::Char('3') => app.page = Page::LineChart,
                    KeyCode::Char('4') => app.page = Page::BarChart,
                    KeyCode::Char('5') => app.page = Page::Histogram,
                    KeyCode::Char('6') => app.page = Page::BoxPlot,
                    KeyCode::Char('7') => app.page = Page::Heatmap,
                    KeyCode::Char('8') => app.page = Page::PieChart,
                    KeyCode::Char('9') => app.page = Page::RadarChart,
                    KeyCode::Char('0') => app.page = Page::Candlestick,
                    KeyCode::Char('-') => app.page = Page::MultiPanel,
                    KeyCode::Char('=') => app.page = Page::Formatters,
                    _ => {}
                }

                // On page change: clean up old images and invalidate caches
                if app.page != prev_page {
                    app.cleanup();
                    prev_page = app.page;

                    // Drain any queued key-repeat events
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

    let chart_area = chunks[0];

    match app.page {
        Page::ScatterBasic => {
            let chart = build_scatter_basic();
            frame.render_stateful_widget(ChartWidget::new(&chart), chart_area, &mut app.states[0]);
        }
        Page::ScatterAdvanced => {
            let (left_chart, right_chart) = build_scatter_advanced();
            let cols = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(chart_area);
            frame.render_stateful_widget(
                ChartWidget::new(&left_chart),
                cols[0],
                &mut app.states[0],
            );
            frame.render_stateful_widget(
                ChartWidget::new(&right_chart),
                cols[1],
                &mut app.states[1],
            );
        }
        Page::LineChart => {
            let chart = build_line_chart();
            frame.render_stateful_widget(ChartWidget::new(&chart), chart_area, &mut app.states[0]);
        }
        Page::BarChart => {
            let (grouped, stacked) = build_bar_charts();
            let cols = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(chart_area);
            frame.render_stateful_widget(ChartWidget::new(&grouped), cols[0], &mut app.states[0]);
            frame.render_stateful_widget(ChartWidget::new(&stacked), cols[1], &mut app.states[1]);
        }
        Page::Histogram => {
            let (hist1, hist2) = build_histogram();
            let cols = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(chart_area);
            frame.render_stateful_widget(ChartWidget::new(&hist1), cols[0], &mut app.states[0]);
            frame.render_stateful_widget(ChartWidget::new(&hist2), cols[1], &mut app.states[1]);
        }
        Page::BoxPlot => {
            let chart = build_boxplot();
            frame.render_stateful_widget(ChartWidget::new(&chart), chart_area, &mut app.states[0]);
        }
        Page::Heatmap => {
            let chart = build_heatmap();
            frame.render_stateful_widget(ChartWidget::new(&chart), chart_area, &mut app.states[0]);
        }
        Page::PieChart => {
            let (pie, donut) = build_pie();
            let cols = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(chart_area);
            frame.render_stateful_widget(ChartWidget::new(&pie), cols[0], &mut app.states[0]);
            frame.render_stateful_widget(ChartWidget::new(&donut), cols[1], &mut app.states[1]);
        }
        Page::RadarChart => {
            let chart = build_radar();
            frame.render_stateful_widget(ChartWidget::new(&chart), chart_area, &mut app.states[0]);
        }
        Page::Candlestick => {
            let chart = build_candlestick();
            frame.render_stateful_widget(ChartWidget::new(&chart), chart_area, &mut app.states[0]);
        }
        Page::MultiPanel => {
            let (c1, c2, c3, c4) = build_dashboard();
            let rows = Layout::vertical([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(chart_area);
            let top = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(rows[0]);
            let bot = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(rows[1]);

            frame.render_stateful_widget(ChartWidget::new(&c1), top[0], &mut app.states[0]);
            frame.render_stateful_widget(ChartWidget::new(&c2), top[1], &mut app.states[1]);
            frame.render_stateful_widget(ChartWidget::new(&c3), bot[0], &mut app.states[2]);
            frame.render_stateful_widget(ChartWidget::new(&c4), bot[1], &mut app.states[3]);
        }
        Page::Formatters => {
            let (c1, c2, c3, c4) = build_formatters();
            let rows = Layout::vertical([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(chart_area);
            let top = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(rows[0]);
            let bot = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(rows[1]);

            frame.render_stateful_widget(ChartWidget::new(&c1), top[0], &mut app.states[0]);
            frame.render_stateful_widget(ChartWidget::new(&c2), top[1], &mut app.states[1]);
            frame.render_stateful_widget(ChartWidget::new(&c3), bot[0], &mut app.states[2]);
            frame.render_stateful_widget(ChartWidget::new(&c4), bot[1], &mut app.states[3]);
        }
    }

    // Status bar
    let page_num = app.page.index() + 1;
    let status_text = format!(
        " {} │ ← → navigate │ 1-9,0,-,= jump │ q quit │ page {}/{}",
        app.page.title(),
        page_num,
        Page::ALL.len()
    );
    let status = Paragraph::new(status_text)
        .block(Block::default().borders(Borders::TOP))
        .style(Style::default().fg(Color::DarkGray));
    frame.render_widget(status, chunks[1]);
}
