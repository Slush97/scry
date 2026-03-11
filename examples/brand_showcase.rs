//! # Scry Brand Showcase — Prismatic Theme
//!
//! Generates a series of compelling demo images showcasing scry's capabilities
//! with the prismatic brand color scheme. Outputs PNG files to `/tmp/scry_showcase/`.
//!
//! Run with: `cargo run --example brand_showcase --release`

#![allow(
    clippy::suboptimal_flops,
    clippy::items_after_statements,
    clippy::cast_precision_loss,
    clippy::unreadable_literal,
    clippy::similar_names,
    clippy::needless_range_loop
)]

use scry_chart::export::{save_png, save_subplot_png};
use scry_chart::prelude::*;
use scry_chart::subplot::SubplotGrid;
use scry_engine::rasterize::Rasterizer;
use scry_engine::scene::style::Point;
use scry_engine::scene::PixelCanvas;
use scry_engine::style::Color;

// ═══════════════════════════════════════════════════════════════════════════
// Brand Colors — Prismatic Palette
// ═══════════════════════════════════════════════════════════════════════════

mod brand {
    use scry_engine::style::Color;

    pub const BG: Color = Color {
        r: 0.059,
        g: 0.078,
        b: 0.137,
        a: 1.0,
    }; // #0F1423
    pub const SURFACE: Color = Color {
        r: 0.118,
        g: 0.161,
        b: 0.231,
        a: 1.0,
    }; // #1E293B
    pub const RED: Color = Color {
        r: 0.973,
        g: 0.443,
        b: 0.443,
        a: 1.0,
    }; // #F87171
    pub const ORANGE: Color = Color {
        r: 0.984,
        g: 0.573,
        b: 0.235,
        a: 1.0,
    }; // #FB923C
    pub const YELLOW: Color = Color {
        r: 0.980,
        g: 0.800,
        b: 0.082,
        a: 1.0,
    }; // #FACC15
    pub const GREEN: Color = Color {
        r: 0.290,
        g: 0.871,
        b: 0.502,
        a: 1.0,
    }; // #4ADE80
    pub const BLUE: Color = Color {
        r: 0.376,
        g: 0.647,
        b: 0.980,
        a: 1.0,
    }; // #60A5FA
    pub const VIOLET: Color = Color {
        r: 0.655,
        g: 0.545,
        b: 0.980,
        a: 1.0,
    }; // #A78BFA
    pub const WHITE: Color = Color {
        r: 0.95,
        g: 0.95,
        b: 0.97,
        a: 1.0,
    };
    pub const MUTED: Color = Color {
        r: 0.392,
        g: 0.455,
        b: 0.545,
        a: 1.0,
    }; // #647888
}

/// Build a custom prismatic theme for charts.
fn prismatic_theme() -> Theme {
    let mut theme = Theme::dark().with_palette(vec![
        brand::BLUE,
        brand::VIOLET,
        brand::GREEN,
        brand::ORANGE,
        brand::RED,
        brand::YELLOW,
    ]);
    theme.background = brand::BG;
    theme
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. "The Scrying Mirror" — GPU-style orb rendered with scry-engine
// ═══════════════════════════════════════════════════════════════════════════

fn scrying_mirror() -> PixelCanvas {
    let w: u32 = 1200;
    let h: u32 = 800;
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0 + 20.0;

    let mut c = PixelCanvas::new(w, h).background(Color::from_rgba8(10, 14, 28, 255));

    // Subtle grid lines
    for i in 0..20 {
        let y = i as f32 * (h as f32 / 20.0);
        c = c
            .line(0.0, y, w as f32, y)
            .color(Color::from_rgba8(255, 255, 255, 8))
            .width(1.0)
            .done();
    }
    for i in 0..25 {
        let x = i as f32 * (w as f32 / 25.0);
        c = c
            .line(x, 0.0, x, h as f32)
            .color(Color::from_rgba8(255, 255, 255, 8))
            .width(1.0)
            .done();
    }

    // Outer glow rings
    for r in (0..6).rev() {
        let radius = 200.0 + r as f32 * 30.0;
        let alpha = 8 - r;
        c = c
            .circle(cx, cy, radius)
            .fill(Color::from_rgba8(96, 165, 250, alpha as u8))
            .done();
    }

    // Main orb — layered for depth
    c = c
        .circle(cx, cy, 180.0)
        .fill(Color::from_rgba8(15, 23, 42, 240))
        .stroke(Color::from_rgba8(96, 165, 250, 80), 2.0)
        .done();

    // Inner prismatic refraction bands
    let spectrum = [
        (160.0, brand::VIOLET, 18u8),
        (140.0, brand::BLUE, 22),
        (120.0, brand::GREEN, 18),
        (100.0, brand::YELLOW, 15),
        (80.0, brand::ORANGE, 12),
        (60.0, brand::RED, 10),
    ];
    for (radius, color, alpha) in spectrum.iter() {
        let c_alpha = Color {
            a: *alpha as f32 / 255.0,
            ..*color
        };
        c = c.circle(cx, cy, *radius).fill(c_alpha).done();
    }

    // Central bright core
    c = c
        .circle(cx, cy, 30.0)
        .fill(Color::from_rgba8(255, 255, 255, 15))
        .done();
    c = c
        .circle(cx, cy, 15.0)
        .fill(Color::from_rgba8(255, 255, 255, 25))
        .done();

    // Specular highlight
    c = c
        .circle(cx - 50.0, cy - 60.0, 25.0)
        .fill(Color::from_rgba8(255, 255, 255, 20))
        .done();

    // Data visualization rays emanating from orb
    let ray_colors = [
        brand::RED,
        brand::ORANGE,
        brand::YELLOW,
        brand::GREEN,
        brand::BLUE,
        brand::VIOLET,
    ];
    for (i, &color) in ray_colors.iter().enumerate() {
        let angle = std::f32::consts::PI * 2.0 * i as f32 / 6.0 - std::f32::consts::FRAC_PI_2;
        let inner_r = 190.0;
        let outer_r = 320.0;
        let x1 = cx + inner_r * angle.cos();
        let y1 = cy + inner_r * angle.sin();
        let x2 = cx + outer_r * angle.cos();
        let y2 = cy + outer_r * angle.sin();
        let c_alpha = Color { a: 0.25, ..color };
        c = c.line(x1, y1, x2, y2).color(c_alpha).width(2.0).done();

        // Data point at end of ray
        c = c.circle(x2, y2, 5.0).fill(color).done();
    }

    // Floating data particles around the orb
    let particles = [
        (200.0, 150.0, 4.0, brand::BLUE),
        (950.0, 200.0, 3.0, brand::VIOLET),
        (150.0, 600.0, 5.0, brand::GREEN),
        (1000.0, 550.0, 4.0, brand::ORANGE),
        (350.0, 100.0, 3.0, brand::RED),
        (850.0, 650.0, 5.0, brand::YELLOW),
        (100.0, 350.0, 3.0, brand::BLUE),
        (1080.0, 380.0, 4.0, brand::VIOLET),
    ];
    for (px, py, pr, pc) in particles.iter() {
        c = c.circle(*px, *py, *pr).fill(*pc).done();
        // Trail
        c = c
            .line(*px, *py, cx, cy)
            .color(Color { a: 0.04, ..*pc })
            .width(1.0)
            .done();
    }

    // Brand accent bar at top
    c = c
        .gradient(0.0, 0.0, w as f32, 3.0)
        .linear(Point::new(0.0, 0.0), Point::new(w as f32, 0.0))
        .stop(0.0, brand::RED)
        .stop(0.17, brand::ORANGE)
        .stop(0.33, brand::YELLOW)
        .stop(0.5, brand::GREEN)
        .stop(0.67, brand::BLUE)
        .stop(0.83, brand::VIOLET)
        .stop(1.0, brand::RED)
        .done();

    c
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Forecasting Chart — "scry predicts"
// ═══════════════════════════════════════════════════════════════════════════

fn forecasting_chart() -> Chart {
    // Historical data + forecast
    let historical: Vec<f64> = (0..30)
        .map(|i| {
            let x = i as f64 * 0.3;
            (x * 0.8).sin() * 20.0 + x * 2.0 + 40.0 + (x * 2.5).cos() * 5.0
        })
        .collect();

    // Forecast continuation (diverging from historical)
    let forecast: Vec<f64> = (0..40)
        .map(|i| {
            if i < 30 {
                f64::NAN // No forecast for historical period
            } else {
                let x = i as f64 * 0.3;
                (x * 0.8).sin() * 15.0 + x * 2.5 + 35.0
            }
        })
        .collect();

    // Upper bound
    let upper: Vec<f64> = (0..40)
        .map(|i| {
            if i < 30 {
                f64::NAN
            } else {
                let x = i as f64 * 0.3;
                (x * 0.8).sin() * 15.0 + x * 2.5 + 35.0 + (i - 30) as f64 * 2.0 + 5.0
            }
        })
        .collect();

    // Lower bound
    let lower: Vec<f64> = (0..40)
        .map(|i| {
            if i < 30 {
                f64::NAN
            } else {
                let x = i as f64 * 0.3;
                (x * 0.8).sin() * 15.0 + x * 2.5 + 35.0 - (i - 30) as f64 * 2.0 - 5.0
            }
        })
        .collect();

    Charts::line(&historical)
        .add_named_series("Forecast", &forecast)
        .add_named_series("Upper 95%", &upper)
        .add_named_series("Lower 95%", &lower)
        .title("scry::forecast — Time Series Prediction")
        .subtitle("ML-powered forecasting with confidence intervals")
        .x_label("Time Period")
        .y_label("Value")
        .smooth()
        .with_points()
        .theme(prismatic_theme())
        .legend(|l| {
            l.visible = true;
        })
        .build()
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. ML Performance Dashboard (subplot grid)
// ═══════════════════════════════════════════════════════════════════════════

fn ml_dashboard() -> SubplotGrid {
    let theme = prismatic_theme();

    // 1. Model accuracy comparison
    let models: Vec<String> = vec!["DT", "RF", "GBT", "HistGBT", "LogReg", "KNN", "NB", "SVM"]
        .into_iter()
        .map(String::from)
        .collect();
    let scry_acc = [0.935, 0.953, 0.931, 0.965, 0.976, 0.961, 0.942, 0.970];
    let sklearn_acc = [0.910, 0.958, 0.953, 0.967, 0.974, 0.963, 0.939, 0.967];

    let accuracy = Charts::bar(models, &scry_acc)
        .add_named_series("scikit-learn", &sklearn_acc)
        .title("Accuracy: scry vs sklearn")
        .subtitle("Breast Cancer dataset")
        .y_label("Accuracy")
        .series_labels(&["scry", "sklearn"])
        .legend(|l| {
            l.visible = true;
        })
        .theme(theme.clone())
        .build();

    // 2. Training throughput
    let throughput_labels: Vec<String> = vec!["GaussNB", "LinReg", "KNN", "LogReg", "DT", "RF"]
        .into_iter()
        .map(String::from)
        .collect();
    let throughput_vals = [0.185, 0.349, 3.02, 3.94, 5.12, 6.43]; // ms

    let throughput = Charts::bar(throughput_labels, &throughput_vals)
        .title("Training Time (10K samples)")
        .subtitle("Lower is better — milliseconds")
        .y_label("Time (ms)")
        .theme(theme.clone())
        .build();

    // 3. Speedup vs competitors
    let speed_labels: Vec<String> = vec!["DT pred", "RF train", "RF pred", "KNN", "LogReg"]
        .into_iter()
        .map(String::from)
        .collect();
    let speedups = [14.1, 4.7, 7.6, 4.7, 3.8];

    let speedup_chart = Charts::bar(speed_labels, &speedups)
        .title("Speedup vs smartcore")
        .subtitle("Single-threaded, algorithmic advantage")
        .y_label("×  faster")
        .theme(theme.clone())
        .build();

    // 4. Inference latency scatter (p50 vs p95)
    let p50: Vec<f64> = vec![0.02, 0.07, 0.06, 0.13, 0.21]; // µs
    let p95: Vec<f64> = vec![0.03, 0.07, 0.07, 0.14, 0.21];

    let latency = Charts::scatter(&p50, &p95)
        .title("Inference Latency (µs)")
        .x_label("p50 (µs)")
        .y_label("p95 (µs)")
        .theme(theme.clone())
        .build();

    SubplotGrid::new(2, 2)
        .title("scry-learn — ML Performance Dashboard")
        .set(0, 0, accuracy)
        .set(0, 1, throughput)
        .set(1, 0, speedup_chart)
        .set(1, 1, latency)
        .gap(20)
        .background(brand::BG)
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Chart Gallery — showcasing visualization capabilities
// ═══════════════════════════════════════════════════════════════════════════

fn chart_gallery() -> SubplotGrid {
    let theme = prismatic_theme();

    // Line chart
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
    let line = Charts::line(&y1)
        .add_named_series("Series B", &y2)
        .title("Line Chart")
        .smooth()
        .with_points()
        .theme(theme.clone())
        .legend(|l| {
            l.visible = true;
        })
        .build();

    // Candlestick
    let ohlc: Vec<OhlcEntry> = (0..20)
        .map(|i| {
            let base = 100.0 + (i as f64 * 0.5).sin() * 20.0 + i as f64 * 0.5;
            let open = base + (i as f64 * 1.3).cos() * 3.0;
            let close = base + (i as f64 * 0.9).sin() * 4.0;
            let high = open.max(close) + (i as f64 * 0.7).sin().abs() * 5.0 + 1.0;
            let low = open.min(close) - (i as f64 * 0.4).cos().abs() * 5.0 - 1.0;
            OhlcEntry::new(i as f64, open, high, low, close)
        })
        .collect();
    let candle = Charts::candlestick(ohlc)
        .title("Candlestick — OHLC")
        .theme(theme.clone())
        .build();

    // Radar
    let radar = Charts::radar(vec!["Speed", "Power", "Defense", "Magic", "HP", "Stamina"])
        .add_series("Warrior", &[8.0, 9.0, 7.0, 2.0, 8.0, 6.0])
        .add_series("Mage", &[3.0, 4.0, 3.0, 10.0, 5.0, 4.0])
        .title("Radar Chart")
        .theme(theme.clone())
        .build();

    // Heatmap
    let hm_data: Vec<Vec<f64>> = (0..10)
        .map(|r| {
            (0..10)
                .map(|c| {
                    let x = r as f64 / 10.0;
                    let y = c as f64 / 10.0;
                    (x * 3.0).sin() * (y * 3.0).cos() * 50.0 + 50.0
                })
                .collect()
        })
        .collect();
    let heatmap = Charts::heatmap(hm_data)
        .title("Heatmap — Correlation")
        .theme(theme.clone())
        .build();

    // Violin
    let violin = Charts::violin(vec![
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
    ])
    .inner_box()
    .title("Violin Plot")
    .theme(theme.clone())
    .build();

    // Gauge
    let gauge = Charts::gauge(87.0)
        .range(0.0, 100.0)
        .threshold(40.0, brand::GREEN)
        .threshold(75.0, brand::YELLOW)
        .threshold(100.0, brand::RED)
        .label("87%")
        .title("Model Accuracy")
        .theme(theme.clone())
        .build();

    SubplotGrid::new(2, 3)
        .title("scry-chart — Visualization Gallery")
        .set(0, 0, line)
        .set(0, 1, candle)
        .set(0, 2, radar)
        .set(1, 0, heatmap)
        .set(1, 1, violin)
        .set(1, 2, gauge)
        .gap(16)
        .background(brand::BG)
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Architecture diagram — scry-engine rendering
// ═══════════════════════════════════════════════════════════════════════════

fn architecture_visual() -> PixelCanvas {
    let w: u32 = 1200;
    let h: u32 = 600;

    let mut c = PixelCanvas::new(w, h).background(Color::from_rgba8(10, 14, 28, 255));

    // Prismatic top bar
    c = c
        .gradient(0.0, 0.0, w as f32, 3.0)
        .linear(Point::new(0.0, 0.0), Point::new(w as f32, 0.0))
        .stop(0.0, brand::RED)
        .stop(0.17, brand::ORANGE)
        .stop(0.33, brand::YELLOW)
        .stop(0.5, brand::GREEN)
        .stop(0.67, brand::BLUE)
        .stop(0.83, brand::VIOLET)
        .stop(1.0, brand::RED)
        .done();

    // 5 crate boxes
    let boxes = [
        ("scry-engine", 60.0, 60.0, 200.0, 160.0, brand::BLUE),
        ("scry-chart", 300.0, 60.0, 200.0, 160.0, brand::GREEN),
        ("scry-learn", 540.0, 60.0, 200.0, 160.0, brand::VIOLET),
        ("scry-pipe", 780.0, 60.0, 180.0, 160.0, brand::ORANGE),
        ("scry-cli", 1000.0, 60.0, 150.0, 160.0, brand::RED),
    ];

    for (_, x, y, bw, bh, color) in boxes.iter() {
        // Box
        c = c
            .rect(*x, *y, *bw, *bh)
            .fill(Color::from_rgba8(30, 41, 59, 255))
            .corner_radius(12.0)
            .stroke(Color { a: 0.4, ..*color }, 2.0)
            .done();

        // Top accent bar
        c = c
            .rect(*x, *y, *bw, 4.0)
            .fill(*color)
            .corner_radius(2.0)
            .done();

        // Fake text lines
        c = c
            .rect(*x + 16.0, *y + 30.0, *bw * 0.7, 12.0)
            .fill(Color::from_rgba8(255, 255, 255, 180))
            .corner_radius(2.0)
            .done();
        c = c
            .rect(*x + 16.0, *y + 52.0, *bw * 0.5, 8.0)
            .fill(Color::from_rgba8(255, 255, 255, 60))
            .corner_radius(2.0)
            .done();
        c = c
            .rect(*x + 16.0, *y + 66.0, *bw * 0.6, 8.0)
            .fill(Color::from_rgba8(255, 255, 255, 60))
            .corner_radius(2.0)
            .done();

        // LOC badge
        c = c
            .rect(*x + 16.0, *y + *bh - 36.0, 60.0, 22.0)
            .fill(Color { a: 0.15, ..*color })
            .corner_radius(4.0)
            .done();
    }

    // Connection lines between boxes
    let connections = [
        (260.0, 140.0, 300.0, 140.0, brand::BLUE), // engine → chart
        (500.0, 140.0, 540.0, 140.0, brand::GREEN), // chart → learn
        (740.0, 140.0, 780.0, 140.0, brand::VIOLET), // learn → pipe
        (960.0, 140.0, 1000.0, 140.0, brand::ORANGE), // pipe → cli
    ];
    for (x1, y1, x2, y2, color) in connections.iter() {
        c = c
            .line(*x1, *y1, *x2, *y2)
            .color(Color { a: 0.5, ..*color })
            .width(2.0)
            .done();
    }

    // Bottom section — metrics row
    let metrics_y = 280.0;
    let metric_boxes = [
        (60.0, "164K LOC", brand::BLUE),
        (260.0, "31 Models", brand::VIOLET),
        (460.0, "18 Charts", brand::GREEN),
        (660.0, "6 Transports", brand::ORANGE),
        (860.0, "781+ Tests", brand::RED),
        (1020.0, "9 Days", brand::YELLOW),
    ];

    for (x, _, color) in metric_boxes.iter() {
        c = c
            .rect(*x, metrics_y, 160.0, 80.0)
            .fill(Color::from_rgba8(22, 28, 45, 255))
            .corner_radius(10.0)
            .stroke(Color { a: 0.2, ..*color }, 1.0)
            .done();

        // Large number placeholder
        c = c
            .rect(*x + 20.0, metrics_y + 18.0, 80.0, 18.0)
            .fill(Color { a: 0.9, ..*color })
            .corner_radius(3.0)
            .done();

        // Label placeholder
        c = c
            .rect(*x + 20.0, metrics_y + 48.0, 100.0, 10.0)
            .fill(Color::from_rgba8(255, 255, 255, 80))
            .corner_radius(2.0)
            .done();
    }

    // Bottom gradient bar
    let bottom_y = 400.0;

    // Performance comparison bars
    let bar_labels = [
        ("DT Prediction", 14.1, brand::BLUE),
        ("RF Training", 4.7, brand::VIOLET),
        ("RF Prediction", 7.6, brand::GREEN),
        ("KNN", 4.7, brand::ORANGE),
        ("LogReg", 3.8, brand::RED),
    ];

    let max_val = 14.1_f32;
    let bar_width = 160.0_f32;

    for (i, (_, val, color)) in bar_labels.iter().enumerate() {
        let x = 60.0 + i as f32 * 220.0;
        let bar_h = (*val / max_val) * 120.0;

        // Bar
        c = c
            .rect(x, bottom_y + 140.0 - bar_h, bar_width, bar_h)
            .fill(Color { a: 0.8, ..*color })
            .corner_radius(6.0)
            .done();

        // Label
        c = c
            .rect(x + 20.0, bottom_y + 150.0, 60.0, 10.0)
            .fill(Color::from_rgba8(255, 255, 255, 100))
            .corner_radius(2.0)
            .done();
    }

    // Bottom prismatic bar
    c = c
        .gradient(0.0, h as f32 - 3.0, w as f32, 3.0)
        .linear(Point::new(0.0, 0.0), Point::new(w as f32, 0.0))
        .stop(0.0, brand::VIOLET)
        .stop(0.17, brand::BLUE)
        .stop(0.33, brand::GREEN)
        .stop(0.5, brand::YELLOW)
        .stop(0.67, brand::ORANGE)
        .stop(0.83, brand::RED)
        .stop(1.0, brand::VIOLET)
        .done();

    c
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    let out_dir = std::path::Path::new("/tmp/scry_showcase");
    std::fs::create_dir_all(out_dir).expect("failed to create output dir");

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║          scry  ·  Brand Showcase Generator              ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Generating prismatic-themed demo outputs...            ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // 1. Scrying Mirror
    print!("  [1/5] Scrying Mirror... ");
    let mirror = scrying_mirror();
    let pixmap = Rasterizer::rasterize(&mirror).expect("rasterize failed");
    let path = out_dir.join("01_scrying_mirror.png");
    pixmap.save_png(&path).expect("save failed");
    println!("✓  {}", path.display());

    // 2. Forecasting Chart
    print!("  [2/5] Forecasting Chart... ");
    let forecast = forecasting_chart();
    let path = out_dir.join("02_forecast.png");
    save_png(&forecast, 1200, 700, &path).expect("save failed");
    println!("✓  {}", path.display());

    // 3. ML Dashboard
    print!("  [3/5] ML Performance Dashboard... ");
    let dashboard = ml_dashboard();
    let path = out_dir.join("03_ml_dashboard.png");
    save_subplot_png(&dashboard, 1600, 900, &path).expect("save failed");
    println!("✓  {}", path.display());

    // 4. Chart Gallery
    print!("  [4/5] Chart Gallery... ");
    let gallery = chart_gallery();
    let path = out_dir.join("04_chart_gallery.png");
    save_subplot_png(&gallery, 1800, 900, &path).expect("save failed");
    println!("✓  {}", path.display());

    // 5. Architecture Visual
    print!("  [5/5] Architecture Visual... ");
    let arch = architecture_visual();
    let pixmap = Rasterizer::rasterize(&arch).expect("rasterize failed");
    let path = out_dir.join("05_architecture.png");
    pixmap.save_png(&path).expect("save failed");
    println!("✓  {}", path.display());

    println!();
    println!("🎨 All showcase images saved to {}/", out_dir.display());
    println!();
    println!("  01_scrying_mirror.png   — The scrying orb (brand hero image)");
    println!("  02_forecast.png         — ML forecasting with confidence intervals");
    println!("  03_ml_dashboard.png     — Performance dashboard (scry vs sklearn)");
    println!("  04_chart_gallery.png    — 6-chart visualization gallery");
    println!("  05_architecture.png     — Crate architecture overview");
}
