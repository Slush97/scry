//! Built-in example charts for the `pixelchart example` command.
//!
//! Each function returns a fully-configured `Chart` with realistic sample data
//! so users can see a pixel-perfect chart without writing any JSON.

use pixelchart::chart::Chart;
use pixelchart::theme::Theme;

use crate::spec::ChartType;

/// Build a demo chart for the given type.
pub fn build_example(chart_type: ChartType, theme: Theme) -> Chart {
    match chart_type {
        ChartType::Line => example_line(theme),
        ChartType::Scatter => example_scatter(theme),
        ChartType::Bar => example_bar(theme),
        ChartType::Histogram => example_histogram(theme),
        ChartType::Boxplot => example_boxplot(theme),
        ChartType::Heatmap => example_heatmap(theme),
        ChartType::Pie => example_pie(theme),
    }
}

/// All chart types in display order.
pub fn all_types() -> &'static [ChartType] {
    &[
        ChartType::Line,
        ChartType::Scatter,
        ChartType::Bar,
        ChartType::Histogram,
        ChartType::Boxplot,
        ChartType::Heatmap,
        ChartType::Pie,
    ]
}

// ---------------------------------------------------------------------------
// Individual examples
// ---------------------------------------------------------------------------

fn example_line(theme: Theme) -> Chart {
    // 12-month revenue curve
    let months: Vec<f64> = (1..=12).map(f64::from).collect();
    let revenue = vec![
        12.4, 15.8, 14.2, 18.9, 22.5, 25.1, 23.8, 28.4, 31.2, 29.6, 34.8, 38.1,
    ];
    Chart::line_xy(&months, &revenue)
        .title("Monthly Revenue ($K)")
        .x_label("Month")
        .y_label("Revenue")
        .smooth()
        .filled()
        .with_points()
        .theme(theme)
        .build()
}

fn example_scatter(theme: Theme) -> Chart {
    // Correlated data with some noise
    let n = 50;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 2.0).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            let noise = ((i as f64 * 7.3).sin() * 15.0) + ((i as f64 * 3.1).cos() * 8.0);
            xi * 1.5 + 10.0 + noise
        })
        .collect();
    Chart::scatter(&x, &y)
        .title("Performance vs. Load")
        .x_label("Load (req/s)")
        .y_label("Latency (ms)")
        .trend_line()
        .theme(theme)
        .build()
}

fn example_bar(theme: Theme) -> Chart {
    let labels = vec!["Q1".into(), "Q2".into(), "Q3".into(), "Q4".into()];
    let values = [142.0, 218.0, 195.0, 267.0];
    Chart::bar(labels, &values)
        .title("Quarterly Revenue ($K)")
        .x_label("Quarter")
        .y_label("Revenue")
        .theme(theme)
        .build()
}

fn example_histogram(theme: Theme) -> Chart {
    // Simulated response time distribution (roughly normal)
    let mut values = Vec::with_capacity(200);
    for i in 0..200 {
        // Deterministic pseudo-random normal-ish distribution
        let t = i as f64;
        let v = 50.0
            + 15.0 * (t * 0.37).sin()
            + 10.0 * (t * 0.73).cos()
            + 5.0 * (t * 1.53).sin()
            + 3.0 * (t * 2.91).cos();
        values.push(v);
    }
    Chart::histogram(&values)
        .title("Response Time Distribution")
        .x_label("Latency (ms)")
        .y_label("Frequency")
        .theme(theme)
        .build()
}

fn example_boxplot(theme: Theme) -> Chart {
    // Three service tiers with different latency profiles
    let economy: Vec<f64> = (0..60)
        .map(|i| {
            let t = i as f64;
            120.0 + 40.0 * (t * 0.31).sin() + 20.0 * (t * 0.83).cos()
        })
        .collect();
    let standard: Vec<f64> = (0..60)
        .map(|i| {
            let t = i as f64;
            60.0 + 20.0 * (t * 0.47).sin() + 10.0 * (t * 0.61).cos()
        })
        .collect();
    let premium: Vec<f64> = (0..60)
        .map(|i| {
            let t = i as f64;
            25.0 + 8.0 * (t * 0.53).sin() + 5.0 * (t * 0.97).cos()
        })
        .collect();

    Chart::boxplot(vec![
        ("Economy", economy),
        ("Standard", standard),
        ("Premium", premium),
    ])
    .title("Latency by Service Tier")
    .x_label("Tier")
    .y_label("Latency (ms)")
    .theme(theme)
    .build()
}

fn example_heatmap(theme: Theme) -> Chart {
    // 8×8 activity grid (hour × day-of-week feel)
    let mut grid = Vec::with_capacity(8);
    for row in 0..8 {
        let mut cells = Vec::with_capacity(8);
        for col in 0..8 {
            let r = row as f64;
            let c = col as f64;
            let val = (r * 0.5 + c * 0.3).sin().abs() * 100.0
                + ((r - 4.0).powi(2) + (c - 4.0).powi(2)).sqrt() * 5.0;
            cells.push(val);
        }
        grid.push(cells);
    }
    Chart::heatmap(grid)
        .title("Server Load Heatmap")
        .theme(theme)
        .build()
}

fn example_pie(theme: Theme) -> Chart {
    let labels = vec![
        "Rust".into(),
        "Python".into(),
        "TypeScript".into(),
        "Go".into(),
        "Other".into(),
    ];
    let values = [35.0, 25.0, 20.0, 12.0, 8.0];
    Chart::pie(labels, &values)
        .title("Language Usage Share")
        .theme(theme)
        .build()
}
