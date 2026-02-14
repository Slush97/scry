//! Render charts to PNG files using the library export function.
//!
//! Usage: cargo run -p pixelchart --example render_png

use pixelchart::chart::Chart;
use pixelchart::export;
use pixelchart::theme::Theme;

fn main() {
    let out_dir = std::path::PathBuf::from("chart_output");
    std::fs::create_dir_all(&out_dir).unwrap();

    let charts: Vec<(&str, Chart)> = vec![
        ("scatter", scatter_chart()),
        ("line", line_chart()),
        ("bar", bar_chart()),
        ("histogram", histogram_chart()),
        ("line_ocean", line_ocean_chart()),
        ("line_forest", line_forest_chart()),
    ];

    for (name, chart) in &charts {
        let path = out_dir.join(format!("{name}.png"));
        export::save_png(chart, 800, 500, &path).expect("export failed");
        println!(
            "✓ {path:?} ({} bytes)",
            std::fs::metadata(&path).unwrap().len()
        );
    }

    println!("\nAll charts written to {out_dir:?}/");
}

fn scatter_chart() -> Chart {
    Chart::scatter(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2.1, 3.5, 2.8, 5.2, 4.1, 6.3, 5.5, 7.0],
    )
    .title("Scatter: Temperature vs Growth")
    .x_label("Temperature (°C)")
    .y_label("Growth (cm)")
    .theme(Theme::dark())
    .build()
}

fn line_chart() -> Chart {
    Chart::line(&[10.0, 25.0, 18.0, 35.0, 28.0, 42.0, 38.0, 50.0])
        .title("Line: Monthly Revenue")
        .x_label("Month")
        .y_label("Revenue ($K)")
        .add_named_series("Q2", &[15.0, 30.0, 22.0, 40.0, 32.0, 48.0, 44.0, 55.0])
        .filled()
        .with_points()
        .theme(Theme::dark())
        .build()
}

fn bar_chart() -> Chart {
    Chart::bar(
        vec![
            "Mon".into(),
            "Tue".into(),
            "Wed".into(),
            "Thu".into(),
            "Fri".into(),
        ],
        &[120.0, 200.0, 150.0, 280.0, 190.0],
    )
    .title("Bar: Daily Sales")
    .y_label("Sales")
    .add_named_series("Region B", &[90.0, 160.0, 180.0, 110.0, 220.0])
    .theme(Theme::pastel())
    .build()
}

fn histogram_chart() -> Chart {
    use std::f64::consts::PI;
    let data: Vec<f64> = (0..500)
        .map(|i| {
            let t = i as f64 / 500.0 * 4.0 * PI;
            (t.sin() * 30.0 + 50.0) + (i as f64 * 0.1).sin() * 10.0
        })
        .collect();

    Chart::histogram(&data)
        .title("Histogram: Signal Distribution")
        .x_label("Amplitude")
        .y_label("Frequency")
        .bins(25)
        .theme(Theme::light())
        .build()
}

fn line_ocean_chart() -> Chart {
    Chart::line(&[5.0, 12.0, 8.0, 20.0, 15.0, 25.0, 22.0, 30.0])
        .title("Ocean Theme Demo")
        .x_label("Sample")
        .y_label("Value")
        .smooth()
        .filled()
        .theme(Theme::ocean())
        .build()
}

fn line_forest_chart() -> Chart {
    Chart::line(&[3.0, 8.0, 5.0, 15.0, 12.0, 18.0, 14.0, 22.0])
        .title("Forest Theme Demo")
        .x_label("Sample")
        .y_label("Value")
        .smooth()
        .with_points()
        .theme(Theme::forest())
        .build()
}
