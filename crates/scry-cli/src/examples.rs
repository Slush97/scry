// SPDX-License-Identifier: MIT OR Apache-2.0
//! Built-in example charts for the `scry chart example` command.
//!
//! Each function returns a fully-configured `Chart` with realistic sample data
//! so users can see a pixel-perfect chart without writing any JSON.

use scry_chart::chart::{Chart, Charts, OhlcEntry};
use scry_chart::theme::Theme;

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
        ChartType::Radar => example_radar(theme),
        ChartType::Candlestick => example_candlestick(theme),
        ChartType::Bubble => example_bubble(theme),
        ChartType::Violin => example_violin(theme),
        ChartType::Sparkline => example_sparkline(),
        ChartType::Waterfall => example_waterfall(theme),
        ChartType::Funnel => example_funnel(theme),
        ChartType::Gauge => example_gauge(theme),
        ChartType::Lollipop => example_lollipop(theme),
        ChartType::Gantt => {
            // Gantt requires specialized data; show a placeholder message.
            Charts::bar(
                vec!["(Not available as example)".into()],
                &[0.0],
            )
            .title("Gantt — use JSON input")
            .theme(theme)
            .build()
        }
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
        ChartType::Radar,
        ChartType::Candlestick,
        ChartType::Bubble,
        ChartType::Violin,
        ChartType::Sparkline,
        ChartType::Waterfall,
        ChartType::Funnel,
        ChartType::Gauge,
        ChartType::Lollipop,
        ChartType::Gantt,
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
    Charts::line_xy(&months, &revenue)
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
    Charts::scatter(&x, &y)
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
    Charts::bar(labels, &values)
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
    Charts::histogram(&values)
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

    Charts::boxplot(vec![
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
    Charts::heatmap(grid)
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
    Charts::pie(labels, &values)
        .title("Language Usage Share")
        .theme(theme)
        .build()
}

fn example_radar(theme: Theme) -> Chart {
    Charts::radar(vec!["Speed", "Power", "Defense", "Magic", "HP", "Stamina"])
        .add_series("Warrior", &[8.0, 9.0, 7.0, 2.0, 8.0, 6.0])
        .add_series("Mage", &[3.0, 4.0, 3.0, 10.0, 5.0, 4.0])
        .add_series("Rogue", &[9.0, 5.0, 4.0, 5.0, 4.0, 9.0])
        .title("Character Class Comparison")
        .theme(theme)
        .build()
}

fn example_candlestick(theme: Theme) -> Chart {
    // 20-day stock price data
    let data = vec![
        OhlcEntry::new(1.0, 100.0, 105.0, 98.0, 103.0),
        OhlcEntry::new(2.0, 103.0, 108.0, 101.0, 106.0),
        OhlcEntry::new(3.0, 106.0, 110.0, 104.0, 105.0),
        OhlcEntry::new(4.0, 105.0, 107.0, 100.0, 101.0),
        OhlcEntry::new(5.0, 101.0, 104.0, 98.0, 103.0),
        OhlcEntry::new(6.0, 103.0, 109.0, 102.0, 108.0),
        OhlcEntry::new(7.0, 108.0, 115.0, 107.0, 114.0),
        OhlcEntry::new(8.0, 114.0, 116.0, 110.0, 111.0),
        OhlcEntry::new(9.0, 111.0, 113.0, 108.0, 109.0),
        OhlcEntry::new(10.0, 109.0, 112.0, 106.0, 110.0),
        OhlcEntry::new(11.0, 110.0, 118.0, 109.0, 117.0),
        OhlcEntry::new(12.0, 117.0, 120.0, 115.0, 119.0),
        OhlcEntry::new(13.0, 119.0, 121.0, 116.0, 117.0),
        OhlcEntry::new(14.0, 117.0, 119.0, 114.0, 115.0),
        OhlcEntry::new(15.0, 115.0, 118.0, 113.0, 116.0),
        OhlcEntry::new(16.0, 116.0, 122.0, 115.0, 121.0),
        OhlcEntry::new(17.0, 121.0, 125.0, 120.0, 124.0),
        OhlcEntry::new(18.0, 124.0, 126.0, 121.0, 122.0),
        OhlcEntry::new(19.0, 122.0, 124.0, 118.0, 119.0),
        OhlcEntry::new(20.0, 119.0, 123.0, 117.0, 122.0),
    ];
    Charts::candlestick(data)
        .title("ACME Corp (20-Day)")
        .x_label("Day")
        .y_label("Price ($)")
        .theme(theme)
        .build()
}

fn example_bubble(theme: Theme) -> Chart {
    // GDP vs Life Expectancy vs Population (simplified)
    let x = [2.0, 5.0, 8.0, 12.0, 18.0, 25.0, 35.0, 45.0]; // GDP per capita ($K)
    let y = [55.0, 62.0, 68.0, 72.0, 75.0, 78.0, 80.0, 82.0]; // Life expectancy
    let sizes = [200.0, 50.0, 120.0, 80.0, 30.0, 15.0, 60.0, 10.0]; // Population (M)
    Charts::bubble(&x, &y, &sizes)
        .title("GDP vs Life Expectancy")
        .x_label("GDP per Capita ($K)")
        .y_label("Life Expectancy (yrs)")
        .theme(theme)
        .build()
}

fn example_violin(theme: Theme) -> Chart {
    // Response latency by data center region
    let us_east: Vec<f64> = (0..80)
        .map(|i| {
            let t = i as f64;
            45.0 + 20.0 * (t * 0.41).sin() + 12.0 * (t * 0.87).cos()
        })
        .collect();
    let eu_west: Vec<f64> = (0..80)
        .map(|i| {
            let t = i as f64;
            65.0 + 25.0 * (t * 0.53).sin() + 15.0 * (t * 0.71).cos()
        })
        .collect();
    let ap_south: Vec<f64> = (0..80)
        .map(|i| {
            let t = i as f64;
            90.0 + 35.0 * (t * 0.37).sin() + 18.0 * (t * 0.93).cos()
        })
        .collect();

    Charts::violin(vec![
        ("US-East", us_east),
        ("EU-West", eu_west),
        ("AP-South", ap_south),
    ])
    .title("Latency Distribution by Region")
    .x_label("Region")
    .y_label("Latency (ms)")
    .theme(theme)
    .build()
}

fn example_sparkline() -> Chart {
    // CPU usage trend — 30 data points
    let values: Vec<f64> = (0..30)
        .map(|i| {
            let t = i as f64;
            40.0 + 25.0 * (t * 0.3).sin() + 15.0 * (t * 0.7).cos() + 8.0 * (t * 1.1).sin()
        })
        .collect();
    Charts::sparkline(&values).filled().build()
}

fn example_waterfall(theme: Theme) -> Chart {
    let labels = vec![
        "Revenue".into(),
        "COGS".into(),
        "Gross Profit".into(),
        "OpEx".into(),
        "R&D".into(),
        "Tax".into(),
    ];
    let values = [500.0, -200.0, -50.0, -80.0, -45.0, -30.0];
    Charts::waterfall(labels, &values)
        .title("P&L Waterfall ($K)")
        .x_label("Category")
        .y_label("Amount ($K)")
        .show_values()
        .theme(theme)
        .build()
}

fn example_funnel(theme: Theme) -> Chart {
    let labels = vec![
        "Impressions".into(),
        "Clicks".into(),
        "Signups".into(),
        "Trials".into(),
        "Paid".into(),
    ];
    let values = [50000.0, 12000.0, 5000.0, 2000.0, 800.0];
    Charts::funnel(labels, &values)
        .title("Marketing Funnel")
        .theme(theme)
        .build()
}

fn example_gauge(theme: Theme) -> Chart {
    use scry_engine::style::Color;

    Charts::gauge(73.0)
        .range(0.0, 100.0)
        .label("73%")
        .threshold(40.0, Color::from_rgba8(40, 180, 99, 255)) // green
        .threshold(70.0, Color::from_rgba8(241, 196, 15, 255)) // yellow
        .threshold(100.0, Color::from_rgba8(231, 76, 60, 255)) // red
        .title("CPU Usage")
        .theme(theme)
        .build()
}

fn example_lollipop(theme: Theme) -> Chart {
    let labels = vec![
        "Rust".into(),
        "Go".into(),
        "Python".into(),
        "Java".into(),
        "C++".into(),
        "JS".into(),
    ];
    let values = [95.0, 82.0, 78.0, 70.0, 65.0, 60.0];
    Charts::lollipop(labels, &values)
        .title("Developer Satisfaction (%)")
        .x_label("Language")
        .y_label("Score")
        .show_values()
        .theme(theme)
        .build()
}
