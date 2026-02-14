//! Feature showcase — demonstrates every new feature added to pixelchart.
//!
//! Creates PNG + SVG outputs for each feature:
//! - Candlestick/OHLC chart
//! - Radar/spider chart
//! - Stacked area chart
//! - Line dash patterns
//! - Data labels (show_values)
//! - Error bars
//! - SVG export
//! - Time scale
//!
//! Usage: cargo run -p pixelchart --example feature_showcase

use pixelchart::data::Series;
use pixelchart::export;
use pixelchart::prelude::*;
use pixelchart::svg_export;
use pixelchart::theme::Theme;
use pixelchart::time_scale::TimeScale;

fn main() {
    let out = std::path::PathBuf::from("feature_showcase_output");
    std::fs::create_dir_all(&out).unwrap();

    let features: Vec<(&str, Chart)> = vec![
        ("01_candlestick", candlestick_chart()),
        ("02_radar", radar_chart()),
        ("03_stacked_area", stacked_area_chart()),
        ("04_dash_patterns", dash_pattern_chart()),
        ("05_data_labels", data_labels_chart()),
        ("06_error_bars", error_bars_chart()),
        ("07_time_scale", time_scale_chart()),
    ];

    println!("╔══════════════════════════════════════════╗");
    println!("║      pixelchart Feature Showcase         ║");
    println!("╚══════════════════════════════════════════╝\n");

    for (name, chart) in &features {
        // PNG export
        let png_path = out.join(format!("{name}.png"));
        export::save_png(chart, 800, 500, &png_path).expect("PNG export failed");
        let png_size = std::fs::metadata(&png_path).unwrap().len();

        // SVG export
        let svg_path = out.join(format!("{name}.svg"));
        svg_export::save_svg(chart, 800, 500, &svg_path).expect("SVG export failed");
        let svg_size = std::fs::metadata(&svg_path).unwrap().len();

        println!("✓ {name}");
        println!("    PNG: {png_path:?} ({png_size} bytes)");
        println!("    SVG: {svg_path:?} ({svg_size} bytes)\n");
    }

    println!(
        "All {count} features rendered to {out:?}/",
        count = features.len()
    );
    println!("Each feature has both PNG and SVG output.\n");
}

// ─── Feature 1: Candlestick / OHLC Chart ────────────────────────────────────

fn candlestick_chart() -> Chart {
    let data = vec![
        OhlcEntry::new(1.0, 100.0, 110.0, 95.0, 108.0), // bullish
        OhlcEntry::new(2.0, 108.0, 115.0, 102.0, 104.0), // bearish
        OhlcEntry::new(3.0, 104.0, 112.0, 100.0, 110.0), // bullish
        OhlcEntry::new(4.0, 110.0, 118.0, 106.0, 107.0), // bearish
        OhlcEntry::new(5.0, 107.0, 120.0, 105.0, 118.0), // bullish
        OhlcEntry::new(6.0, 118.0, 125.0, 112.0, 122.0), // bullish
        OhlcEntry::new(7.0, 122.0, 130.0, 118.0, 119.0), // bearish
        OhlcEntry::new(8.0, 119.0, 128.0, 116.0, 126.0), // bullish
        OhlcEntry::new(9.0, 126.0, 135.0, 122.0, 132.0), // bullish
        OhlcEntry::new(10.0, 132.0, 138.0, 128.0, 130.0), // bearish
    ];

    Chart::candlestick(data)
        .title("AAPL Daily Candlestick")
        .x_label("Trading Day")
        .y_label("Price ($)")
        .theme(Theme::dark())
        .build()
}

// ─── Feature 2: Radar / Spider Chart ─────────────────────────────────────────

fn radar_chart() -> Chart {
    Chart::radar(vec![
        "Speed", "Power", "Defense", "Magic", "Stamina", "Luck",
    ])
    .add_series("Warrior", &[9.0, 8.0, 7.0, 2.0, 8.0, 4.0])
    .add_series("Mage", &[3.0, 4.0, 3.0, 10.0, 5.0, 6.0])
    .add_series("Rogue", &[8.0, 5.0, 4.0, 3.0, 6.0, 9.0])
    .title("Character Stats Comparison")
    .theme(Theme::ocean())
    .build()
}

// ─── Feature 3: Stacked Area Chart ──────────────────────────────────────────

fn stacked_area_chart() -> Chart {
    let frontend = Series::new(
        "Frontend",
        vec![12.0, 15.0, 18.0, 22.0, 25.0, 28.0, 32.0, 35.0],
    );
    let backend = Series::new(
        "Backend",
        vec![8.0, 10.0, 14.0, 16.0, 20.0, 22.0, 24.0, 28.0],
    );
    let devops = Series::new("DevOps", vec![3.0, 4.0, 5.0, 7.0, 8.0, 10.0, 12.0, 14.0]);

    LineChart::new(vec![frontend, backend, devops])
        .stacked()
        .filled()
        .title("Engineering Headcount Growth")
        .x_label("Quarter")
        .y_label("Team Size")
        .theme(Theme::pastel())
        .build()
}

// ─── Feature 4: Line Dash Patterns ──────────────────────────────────────────

fn dash_pattern_chart() -> Chart {
    let actual = Series::new(
        "Actual",
        vec![10.0, 22.0, 18.0, 35.0, 28.0, 42.0, 38.0, 50.0],
    );
    let forecast = Series::new(
        "Forecast",
        vec![12.0, 20.0, 20.0, 32.0, 30.0, 40.0, 42.0, 48.0],
    );
    let baseline = Series::new(
        "Baseline",
        vec![8.0, 16.0, 14.0, 24.0, 22.0, 30.0, 28.0, 36.0],
    );
    let target = Series::new(
        "Target",
        vec![15.0, 25.0, 25.0, 38.0, 35.0, 45.0, 45.0, 55.0],
    );

    LineChart::new(vec![actual, forecast, baseline, target])
        .dash_lines()
        .with_points()
        .title("Revenue: Actual vs Forecast")
        .x_label("Month")
        .y_label("Revenue ($M)")
        .line_width(2.5)
        .theme(Theme::dark())
        .build()
}

// ─── Feature 5: Data Labels (show_values) ───────────────────────────────────

fn data_labels_chart() -> Chart {
    Chart::line(&[42.0, 67.0, 53.0, 89.0, 74.0, 95.0, 82.0])
        .show_values()
        .with_points()
        .title("Weekly KPI Scores")
        .x_label("Day")
        .y_label("Score")
        .theme(Theme::forest())
        .build()
}

// ─── Feature 6: Error Bars ──────────────────────────────────────────────────

fn error_bars_chart() -> Chart {
    let x = Series::from_values(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let y = Series::new(
        "Measurement",
        vec![10.2, 15.1, 12.8, 18.5, 16.3, 22.1, 19.7, 25.4],
    )
    .with_error(vec![1.5, 2.0, 1.2, 2.5, 1.8, 2.2, 1.6, 2.8]);

    ScatterChart::new(x, y)
        .connected()
        .show_values()
        .title("Lab Measurements with Error Bars")
        .x_label("Sample #")
        .y_label("Measurement (μg/mL)")
        .theme(Theme::light())
        .build()
}

// ─── Feature 7: Time Scale ──────────────────────────────────────────────────

fn time_scale_chart() -> Chart {
    // Simulate daily data over 30 days starting from epoch second 1700000000
    // (approx Nov 14, 2023)
    let base_epoch = 1_700_000_000.0;
    let day_secs = 86_400.0;

    let x_times: Vec<f64> = (0..30).map(|d| base_epoch + d as f64 * day_secs).collect();
    let y_values: Vec<f64> = (0..30)
        .map(|d| {
            let t = d as f64 / 30.0 * std::f64::consts::PI * 2.0;
            50.0 + 20.0 * t.sin() + 5.0 * (t * 3.0).cos()
        })
        .collect();

    // Create a time scale for the x-axis
    let _time_scale = TimeScale::new((x_times[0], *x_times.last().unwrap()), (0.0, 800.0));

    Chart::line_xy(&x_times, &y_values)
        .filled()
        .smooth()
        .with_points()
        .title("Sensor Readings — November 2023")
        .x_label("Date")
        .y_label("Temperature (°C)")
        .theme(Theme::ocean())
        .build()
}
