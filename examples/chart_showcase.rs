//! # Chart Showcase — every chart type × every theme
//!
//! Renders **all 18 chart types** with **all 6 themes** to PNG files,
//! creating a comprehensive visual catalogue.
//!
//! Output directory: `chart_showcase_output/`
//!
//! Run with:
//! ```sh
//! cargo run --example chart_showcase
//! ```

#![allow(
    clippy::suboptimal_flops,
    clippy::items_after_statements,
    clippy::cast_precision_loss,
    clippy::unreadable_literal,
    clippy::similar_names,
    clippy::needless_range_loop
)]

use scry_chart::prelude::*;
use scry_engine::style::Color;

// ─────────────────────────────────────────────────────────────────────────────
// Theme catalogue
// ─────────────────────────────────────────────────────────────────────────────

fn all_themes() -> Vec<(&'static str, Theme)> {
    vec![
        ("dark", Theme::dark()),
        ("light", Theme::light()),
        ("pastel", Theme::pastel()),
        ("ocean", Theme::ocean()),
        ("forest", Theme::forest()),
        ("colorblind", Theme::colorblind()),
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// Chart builders — one function per chart type
// ─────────────────────────────────────────────────────────────────────────────

fn build_line(theme: Theme) -> Chart {
    let y1: Vec<f64> = (0..40)
        .map(|i| {
            let x = i as f64 * 0.15;
            (x * 1.2).sin() * 30.0 + 50.0
        })
        .collect();
    let y2: Vec<f64> = (0..40)
        .map(|i| {
            let x = i as f64 * 0.15;
            (x * 0.8).cos() * 25.0 + 45.0
        })
        .collect();

    Charts::line(&y1)
        .add_named_series("Series B", &y2)
        .title("Line Chart")
        .subtitle("Two smooth series with points")
        .x_label("Time")
        .y_label("Value")
        .smooth()
        .with_points()
        .theme(theme)
        .legend(|l| {
            l.visible = true;
        })
        .build()
}

fn build_area(theme: Theme) -> Chart {
    let y: Vec<f64> = (0..30)
        .map(|i| {
            let x = i as f64 * 0.2;
            (x * 0.7).sin().abs() * 40.0 + 10.0
        })
        .collect();

    Charts::area(&y)
        .title("Area Chart")
        .subtitle("Filled line with smooth interpolation")
        .x_label("Time")
        .y_label("Volume")
        .theme(theme)
        .build()
}

fn build_scatter(theme: Theme) -> Chart {
    let x: Vec<f64> = (0..50)
        .map(|i| i as f64 * 0.5 + (i as f64 * 0.3).sin() * 3.0)
        .collect();
    let y: Vec<f64> = (0..50)
        .map(|i| i as f64 * 0.4 + (i as f64 * 0.7).cos() * 5.0)
        .collect();

    Charts::scatter(&x, &y)
        .title("Scatter Plot — 50 points")
        .x_label("X Dimension")
        .y_label("Y Dimension")
        .theme(theme)
        .build()
}

fn build_bar(theme: Theme) -> Chart {
    let labels: Vec<String> = vec!["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        .into_iter()
        .map(String::from)
        .collect();

    Charts::bar(labels, &[42.0, 58.0, 35.0, 67.0, 49.0, 73.0])
        .add_named_series("Product B", &[30.0, 45.0, 50.0, 40.0, 60.0, 55.0])
        .title("Bar Chart")
        .subtitle("Grouped bars — two product lines")
        .x_label("Month")
        .y_label("Revenue ($K)")
        .series_labels(&["Product A", "Product B"])
        .legend(|l| {
            l.visible = true;
        })
        .theme(theme)
        .build()
}

fn build_histogram(theme: Theme) -> Chart {
    // Simulated normal distribution
    let data: Vec<f64> = (0..200)
        .map(|i| {
            let t = i as f64 / 200.0;
            let u1 = (t * 7.3 + 0.1).sin().abs().max(0.001);
            let u2 = (t * 13.7 + 0.5).cos().abs().max(0.001);
            let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            z * 15.0 + 50.0
        })
        .collect();

    Charts::histogram(&data)
        .title("Histogram")
        .subtitle("Distribution of 200 samples")
        .x_label("Value")
        .y_label("Frequency")
        .bins(15)
        .theme(theme)
        .build()
}

fn build_boxplot(theme: Theme) -> Chart {
    Charts::boxplot(vec![
        (
            "Control",
            vec![12.0, 15.0, 14.0, 10.0, 13.0, 18.0, 11.0, 16.0, 14.5, 13.2],
        ),
        (
            "Drug A",
            vec![20.0, 22.0, 19.0, 25.0, 21.0, 18.0, 24.0, 23.0, 20.5, 26.0],
        ),
        (
            "Drug B",
            vec![28.0, 30.0, 27.0, 35.0, 32.0, 29.0, 31.0, 33.0, 28.5, 34.0],
        ),
        (
            "Drug C",
            vec![15.0, 18.0, 22.0, 12.0, 20.0, 17.0, 19.0, 16.0, 21.0, 14.0],
        ),
    ])
    .title("Box Plot")
    .subtitle("Treatment group comparisons")
    .x_label("Group")
    .y_label("Response")
    .theme(theme)
    .build()
}

fn build_heatmap(theme: Theme) -> Chart {
    let size = 8;
    let data: Vec<Vec<f64>> = (0..size)
        .map(|r| {
            (0..size)
                .map(|c| {
                    let x = r as f64 / size as f64;
                    let y = c as f64 / size as f64;
                    (x * 3.0).sin() * (y * 3.0).cos() * 50.0 + 50.0
                })
                .collect()
        })
        .collect();

    Charts::heatmap(data)
        .title("Heatmap")
        .subtitle("8×8 sinusoidal pattern")
        .theme(theme)
        .build()
}

fn build_pie(theme: Theme) -> Chart {
    let labels: Vec<String> = vec!["Rust", "Python", "Go", "TypeScript", "C++"]
        .into_iter()
        .map(String::from)
        .collect();

    Charts::pie(labels, &[35.0, 25.0, 15.0, 15.0, 10.0])
        .title("Pie Chart")
        .subtitle("Language usage share")
        .theme(theme)
        .build()
}

fn build_donut(theme: Theme) -> Chart {
    let labels: Vec<String> = vec!["Desktop", "Mobile", "Tablet", "Other"]
        .into_iter()
        .map(String::from)
        .collect();

    Charts::pie(labels, &[45.0, 35.0, 12.0, 8.0])
        .donut(0.55)
        .title("Donut Chart")
        .subtitle("Traffic by device")
        .theme(theme)
        .build()
}

fn build_candlestick(theme: Theme) -> Chart {
    let data: Vec<OhlcEntry> = (0..20)
        .map(|i| {
            let base = 100.0 + (i as f64 * 0.5).sin() * 20.0 + i as f64 * 0.5;
            let open = base + (i as f64 * 1.3).cos() * 3.0;
            let close = base + (i as f64 * 0.9).sin() * 4.0;
            let high = open.max(close) + (i as f64 * 0.7).sin().abs() * 5.0 + 1.0;
            let low = open.min(close) - (i as f64 * 0.4).cos().abs() * 5.0 - 1.0;
            OhlcEntry::new(i as f64, open, high, low, close)
        })
        .collect();

    Charts::candlestick(data)
        .title("Candlestick Chart")
        .subtitle("20-period OHLC data")
        .x_label("Period")
        .y_label("Price ($)")
        .theme(theme)
        .build()
}

fn build_radar(theme: Theme) -> Chart {
    Charts::radar(vec!["Speed", "Power", "Defense", "Magic", "HP", "Stamina"])
        .add_series("Warrior", &[8.0, 9.0, 7.0, 2.0, 8.0, 6.0])
        .add_series("Mage", &[3.0, 4.0, 3.0, 10.0, 5.0, 4.0])
        .add_series("Rogue", &[9.0, 5.0, 4.0, 6.0, 4.0, 9.0])
        .title("Radar Chart")
        .subtitle("Character class comparison")
        .theme(theme)
        .build()
}

fn build_bubble(theme: Theme) -> Chart {
    let x: Vec<f64> = (0..15).map(|i| i as f64 * 3.0 + 5.0).collect();
    let y: Vec<f64> = (0..15)
        .map(|i| (i as f64 * 0.4).sin() * 30.0 + 50.0)
        .collect();
    let sizes: Vec<f64> = (0..15)
        .map(|i| (i as f64 * 0.6).cos().abs() * 40.0 + 5.0)
        .collect();

    Charts::bubble(&x, &y, &sizes)
        .title("Bubble Chart — 15 points")
        .x_label("Market Cap ($B)")
        .y_label("Growth (%)")
        .theme(theme)
        .build()
}

fn build_violin(theme: Theme) -> Chart {
    Charts::violin(vec![
        (
            "Spring",
            vec![
                12.0, 14.0, 15.0, 13.0, 16.0, 11.0, 14.5, 15.5, 13.5, 12.5, 14.0, 15.0, 13.0, 16.0,
                14.5, 12.0, 15.5, 13.5, 14.0, 15.0,
            ],
        ),
        (
            "Summer",
            vec![
                25.0, 28.0, 30.0, 27.0, 32.0, 26.0, 29.0, 31.0, 28.0, 27.5, 26.0, 30.0, 28.5, 29.5,
                31.5, 27.0, 28.0, 30.5, 29.0, 26.5,
            ],
        ),
        (
            "Autumn",
            vec![
                18.0, 16.0, 15.0, 17.0, 14.0, 19.0, 16.5, 15.5, 17.5, 18.5, 16.0, 15.0, 17.0, 14.5,
                18.0, 19.5, 16.5, 15.0, 17.5, 18.0,
            ],
        ),
        (
            "Winter",
            vec![
                2.0, 0.0, -1.0, 3.0, 1.0, -2.0, 2.5, 0.5, 1.5, -0.5, 3.0, 1.0, -1.0, 2.0, 0.0,
                -2.5, 1.5, 3.5, 0.5, -1.5,
            ],
        ),
    ])
    .inner_box()
    .title("Violin Plot — Seasons")
    .x_label("Season")
    .y_label("Temperature (°C)")
    .theme(theme)
    .build()
}

fn build_sparkline(_theme: Theme) -> Chart {
    let values: Vec<f64> = (0..30)
        .map(|i| {
            let x = i as f64 * 0.3;
            (x * 1.5).sin() * 20.0 + (x * 0.4).cos() * 10.0 + 30.0
        })
        .collect();

    Charts::sparkline(&values).filled().build()
}

fn build_waterfall(theme: Theme) -> Chart {
    let labels: Vec<String> = vec![
        "Revenue",
        "COGS",
        "Gross Profit",
        "OpEx",
        "Tax",
        "Net Income",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    Charts::waterfall(labels, &[500.0, -180.0, 0.0, -120.0, -50.0, 0.0])
        .title("Waterfall Chart")
        .subtitle("P&L breakdown")
        .y_label("Amount ($K)")
        .theme(theme)
        .build()
}

fn build_funnel(theme: Theme) -> Chart {
    let labels: Vec<String> = vec!["Visitors", "Signups", "Trials", "Paid", "Enterprise"]
        .into_iter()
        .map(String::from)
        .collect();

    Charts::funnel(labels, &[10000.0, 5200.0, 2100.0, 850.0, 320.0])
        .title("Funnel Chart")
        .subtitle("Conversion pipeline")
        .theme(theme)
        .build()
}

fn build_gauge(theme: Theme) -> Chart {
    Charts::gauge(73.0)
        .range(0.0, 100.0)
        .threshold(40.0, Color::from_rgba8(40, 180, 99, 255))
        .threshold(70.0, Color::from_rgba8(241, 196, 15, 255))
        .threshold(100.0, Color::from_rgba8(231, 76, 60, 255))
        .label("73%")
        .title("Gauge Chart")
        .subtitle("CPU utilization")
        .theme(theme)
        .build()
}

fn build_lollipop(theme: Theme) -> Chart {
    let labels: Vec<String> = vec!["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        .into_iter()
        .map(String::from)
        .collect();

    Charts::lollipop(labels, &[12.0, 19.0, 8.0, 15.0, 22.0, 28.0, 18.0])
        .title("Lollipop Chart")
        .subtitle("Daily activity scores")
        .x_label("Day")
        .y_label("Score")
        .show_values()
        .theme(theme)
        .build()
}

fn build_contour(theme: Theme) -> Chart {
    let size = 30;
    let data: Vec<Vec<f64>> = (0..size)
        .map(|r| {
            (0..size)
                .map(|c| {
                    let x = (c as f64 - size as f64 / 2.0) / 5.0;
                    let y = (r as f64 - size as f64 / 2.0) / 5.0;
                    (-(x * x + y * y) / 2.0).exp() * 100.0
                })
                .collect()
        })
        .collect();

    Charts::contour(data)
        .levels(8)
        .filled()
        .title("Contour Chart")
        .subtitle("2D Gaussian field, 8 levels")
        .theme(theme)
        .build()
}

// ─────────────────────────────────────────────────────────────────────────────
// Chart catalogue
// ─────────────────────────────────────────────────────────────────────────────

type ChartBuilder = fn(Theme) -> Chart;

fn all_charts() -> Vec<(&'static str, ChartBuilder)> {
    vec![
        ("line", build_line),
        ("area", build_area),
        ("scatter", build_scatter),
        ("bar", build_bar),
        ("histogram", build_histogram),
        ("boxplot", build_boxplot),
        ("heatmap", build_heatmap),
        ("pie", build_pie),
        ("donut", build_donut),
        ("candlestick", build_candlestick),
        ("radar", build_radar),
        ("bubble", build_bubble),
        ("violin", build_violin),
        ("sparkline", build_sparkline),
        ("waterfall", build_waterfall),
        ("funnel", build_funnel),
        ("gauge", build_gauge),
        ("lollipop", build_lollipop),
        ("contour", build_contour),
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// Main — render all combinations to PNG
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    let out_dir = std::path::Path::new("chart_showcase_output");
    std::fs::create_dir_all(out_dir).expect("failed to create output directory");

    let themes = all_themes();
    let charts = all_charts();
    let total = themes.len() * charts.len();
    let mut count = 0;

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║          scry-chart  ·  Full Showcase Gallery           ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!(
        "║  {} chart types × {} themes = {} images              ║",
        charts.len(),
        themes.len(),
        total
    );
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    for (theme_name, theme) in &themes {
        let theme_dir = out_dir.join(theme_name);
        std::fs::create_dir_all(&theme_dir).expect("failed to create theme directory");

        println!("  ── Theme: {} ──", theme_name);

        for (chart_name, builder) in &charts {
            count += 1;
            let chart = builder(theme.clone());
            let filename = format!("{}_{}.png", chart_name, theme_name);
            let path = theme_dir.join(&filename);

            match save_png(&chart, 800, 500, &path) {
                Ok(()) => {
                    println!("    [{:3}/{}] ✓  {}", count, total, filename);
                }
                Err(e) => {
                    eprintln!("    [{:3}/{}] ✗  {} — {}", count, total, filename, e);
                }
            }
        }
        println!();
    }

    println!("Done! All images saved to: {}/", out_dir.display());
    println!();

    // ── Summary table ──
    println!("┌─────────────────┬────────────────────────────────────────┐");
    println!("│  Chart Types    │  Themes                                │");
    println!("├─────────────────┼────────────────────────────────────────┤");
    for (i, (chart_name, _)) in charts.iter().enumerate() {
        let theme_str = if i < themes.len() { themes[i].0 } else { "" };
        println!("│  {:14} │  {:37} │", chart_name, theme_str);
    }
    println!("└─────────────────┴────────────────────────────────────────┘");
}
