//! Fresh chart examples showcasing Phase 1 foundation fixes.
//!
//! Demonstrates:
//! - Legend auto-swatch: scatter → ●, line → ─, bar → ■
//! - AspectRatio::Equal for square plots
//! - Multi-series legends with correct swatch shapes
//! - All 5 available themes
//!
//! Usage: cargo run -p scry-chart --example fresh_charts

use scry_chart::chart::{Chart, Charts, LineChart};
use scry_chart::config::AspectRatio;
use scry_chart::data::Series;
use scry_chart::export;
use scry_chart::theme::Theme;

fn main() {
    let out = std::path::PathBuf::from("chart_output");
    std::fs::create_dir_all(&out).unwrap();

    let charts: Vec<(&str, Chart)> = vec![
        // --- Legend auto-swatch demos ---
        ("scatter_circle_swatch", scatter_multi()),
        ("line_segment_swatch", line_multi()),
        ("bar_rect_swatch", bar_multi()),
        // --- AspectRatio demos ---
        ("aspect_equal", scatter_equal()),
        ("aspect_2to1", scatter_wide()),
        // --- Theme gallery ---
        ("theme_dark", themed_line(Theme::dark(), "Dark Theme")),
        ("theme_light", themed_line(Theme::light(), "Light Theme")),
        ("theme_pastel", themed_line(Theme::pastel(), "Pastel Theme")),
        ("theme_ocean", themed_line(Theme::ocean(), "Ocean Theme")),
        ("theme_forest", themed_line(Theme::forest(), "Forest Theme")),
        // --- Mixed showcase ---
        ("histogram_overlay", histogram_dual()),
        ("boxplot_groups", boxplot_demo()),
    ];

    for (name, chart) in &charts {
        let path = out.join(format!("{name}.png"));
        export::save_png(chart, 800, 500, &path).expect("export failed");
        println!(
            "✓ {name:>25}.png  ({:>6} bytes)",
            std::fs::metadata(&path).unwrap().len()
        );
    }

    println!("\n🎨 {} charts written to {out:?}/", charts.len());
}

// ---------------------------------------------------------------------------
// Legend auto-swatch: scatter → Circle
// ---------------------------------------------------------------------------
fn scatter_multi() -> Chart {
    Charts::scatter(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2.1, 3.5, 2.8, 5.2, 4.1, 6.3, 5.5, 7.0],
    )
    .add_named_series(
        "Strain B",
        &[1.5, 3.0, 4.5, 6.0, 7.5],
        &[3.0, 4.2, 5.8, 6.1, 8.2],
    )
    .add_named_series("Strain C", &[2.0, 4.0, 6.0, 8.0], &[1.5, 3.8, 4.5, 6.8])
    .title("Scatter: Legend ● Circle Swatches")
    .x_label("Concentration (μg/mL)")
    .y_label("Response (AU)")
    .theme(Theme::dark())
    .build()
}

// ---------------------------------------------------------------------------
// Legend auto-swatch: line → Line segment
// ---------------------------------------------------------------------------
fn line_multi() -> Chart {
    LineChart::new(vec![
        Series::new(
            "Revenue",
            vec![10.0, 25.0, 18.0, 35.0, 28.0, 42.0, 38.0, 50.0],
        ),
        Series::new(
            "Expenses",
            vec![15.0, 20.0, 22.0, 25.0, 30.0, 28.0, 35.0, 32.0],
        ),
        Series::new("Profit", vec![-5.0, 5.0, -4.0, 10.0, -2.0, 14.0, 3.0, 18.0]),
    ])
    .title("Line: Legend ─ Line Swatches")
    .x_label("Quarter")
    .y_label("Amount ($K)")
    .with_points()
    .theme(Theme::ocean())
    .build()
}

// ---------------------------------------------------------------------------
// Legend auto-swatch: bar → Rect (default, but multi-series shows it)
// ---------------------------------------------------------------------------
fn bar_multi() -> Chart {
    Charts::bar(
        vec![
            "Jan".into(),
            "Feb".into(),
            "Mar".into(),
            "Apr".into(),
            "May".into(),
        ],
        &[120.0, 200.0, 150.0, 280.0, 190.0],
    )
    .add_named_series("Region B", &[90.0, 160.0, 180.0, 110.0, 220.0])
    .add_named_series("Region C", &[110.0, 140.0, 200.0, 170.0, 160.0])
    .title("Bar: Legend ■ Rect Swatches")
    .y_label("Sales ($K)")
    .theme(Theme::pastel())
    .build()
}

// ---------------------------------------------------------------------------
// AspectRatio::Equal → square plot area
// ---------------------------------------------------------------------------
fn scatter_equal() -> Chart {
    let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
    let y: Vec<f64> = x.iter().map(|v| (v * 0.7).sin() * 4.0 + 5.0).collect();
    Charts::scatter(&x, &y)
        .title("AspectRatio::Equal (Square Plot)")
        .x_label("X")
        .y_label("Y")
        .aspect_ratio(AspectRatio::Equal)
        .theme(Theme::dark())
        .build()
}

// ---------------------------------------------------------------------------
// AspectRatio::Fixed(2.0) → wide plot area
// ---------------------------------------------------------------------------
fn scatter_wide() -> Chart {
    let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|v| (v * 0.3).cos() * 10.0 + 15.0).collect();
    Charts::scatter(&x, &y)
        .title("AspectRatio::Fixed(2.0) (Wide Plot)")
        .x_label("Time (s)")
        .y_label("Signal (mV)")
        .aspect_ratio(AspectRatio::Fixed(2.0))
        .theme(Theme::light())
        .build()
}

// ---------------------------------------------------------------------------
// Theme gallery helper
// ---------------------------------------------------------------------------
fn themed_line(theme: Theme, label: &str) -> Chart {
    Charts::line(&[3.0, 12.0, 8.0, 22.0, 15.0, 30.0, 25.0, 38.0, 32.0, 45.0])
        .add_named_series(
            "Series B",
            &[8.0, 15.0, 12.0, 28.0, 20.0, 35.0, 30.0, 42.0, 38.0, 50.0],
        )
        .title(label)
        .x_label("Sample")
        .y_label("Measurement")
        .smooth()
        .filled()
        .with_points()
        .theme(theme)
        .build()
}

// ---------------------------------------------------------------------------
// Histogram with overlay series
// ---------------------------------------------------------------------------
fn histogram_dual() -> Chart {
    use std::f64::consts::PI;
    let normal: Vec<f64> = (0..400)
        .map(|i| {
            let t = i as f64 / 400.0 * 6.0 * PI;
            t.sin() * 20.0 + 50.0
        })
        .collect();

    Charts::histogram(&normal)
        .add_series(Series::new(
            "Noise",
            (0..300)
                .map(|i| (i as f64 * 0.137).sin() * 35.0 + 45.0)
                .collect(),
        ))
        .title("Histogram: Dual Distribution")
        .x_label("Amplitude")
        .y_label("Frequency")
        .bins(20)
        .theme(Theme::dark())
        .build()
}

// ---------------------------------------------------------------------------
// Box plot
// ---------------------------------------------------------------------------
fn boxplot_demo() -> Chart {
    Charts::boxplot(vec![
        (
            "Control",
            vec![4.2, 5.1, 4.8, 5.3, 4.9, 5.0, 5.2, 4.7, 5.1, 4.6],
        ),
        (
            "Low Dose",
            vec![5.5, 6.2, 5.8, 6.0, 6.5, 5.7, 6.1, 5.9, 6.3, 5.6],
        ),
        (
            "Mid Dose",
            vec![7.1, 8.3, 7.5, 8.0, 7.8, 8.5, 7.2, 8.1, 7.9, 7.6],
        ),
        (
            "High Dose",
            vec![9.0, 10.5, 9.8, 11.2, 10.0, 9.5, 10.8, 9.2, 10.3, 12.0],
        ),
    ])
    .title("Box Plot: Treatment Response")
    .y_label("Biomarker (ng/mL)")
    .theme(Theme::forest())
    .build()
}
