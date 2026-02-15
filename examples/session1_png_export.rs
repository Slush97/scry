//! Quick PNG export for Session 1 feature demo.
//!
//! Run with: `cargo run --example session1_png_export`

use scry_chart::prelude::*;

fn main() {
    let dir = "/home/esoc/.gemini/antigravity/brain/28b6aba6-8eeb-4244-9c94-ed56e72100a5";

    // 1. Subtitle & Footer
    let y: Vec<f64> = (0..30)
        .map(|i| {
            let x = i as f64 * 0.2;
            (x * 1.5).sin() * 40.0 + 60.0 + (x * 0.3).cos() * 15.0
        })
        .collect();
    let chart1 = Chart::line(&y)
        .title("Monthly Revenue")
        .subtitle("Q3-Q4 2025 - All Regions")
        .footer("Source: internal analytics - Updated Feb 2026")
        .x_label("Month")
        .y_label("Revenue ($K)")
        .build();
    save_png(&chart1, 800, 500, format!("{dir}/subtitle_footer.png")).unwrap();
    eprintln!("  subtitle_footer.png");

    // 2. Margins
    let y2: Vec<f64> = (0..20)
        .map(|i| {
            let x = i as f64 * 0.3;
            x.powi(2) * 0.5
        })
        .collect();
    let chart2 = Chart::line(&y2)
        .title("Growth Curve")
        .subtitle("with 30px margins all around")
        .x_label("Time")
        .y_label("Value")
        .margin(30.0, 30.0, 30.0, 30.0)
        .build();
    save_png(&chart2, 800, 500, format!("{dir}/margins.png")).unwrap();
    eprintln!("  margins.png");

    // 3. Inverted Y-Axis
    let y3: Vec<f64> = (0..25)
        .map(|i| {
            let x = i as f64 * 0.25;
            100.0 - (x * 2.0).sin().abs() * 50.0 - x * 3.0
        })
        .collect();
    let chart3 = Chart::line(&y3)
        .title("Depth Profile")
        .subtitle("Y-axis inverted - deeper = lower")
        .x_label("Station")
        .y_label("Depth (m)")
        .y_inverted()
        .build();
    save_png(&chart3, 800, 500, format!("{dir}/inverted.png")).unwrap();
    eprintln!("  inverted.png");

    // 4. Dual Y-Axis
    let temp: Vec<f64> = (0..24)
        .map(|i| {
            let hour = i as f64;
            20.0 + 8.0 * ((hour - 14.0) * std::f64::consts::PI / 12.0).cos()
                + (hour * 0.5).sin() * 2.0
        })
        .collect();
    let x: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let chart4 = Chart::line_xy(&x, &temp)
        .title("Weather Station")
        .subtitle("Temperature + Humidity (dual axis)")
        .x_label("Hour of Day")
        .y_label("Temperature (C)")
        .secondary_y_label("Humidity (%)")
        .secondary_y_range(30.0, 90.0)
        .build();
    save_png(&chart4, 800, 500, format!("{dir}/dual_y_axis.png")).unwrap();
    eprintln!("  dual_y_axis.png");

    eprintln!("\nAll 4 PNGs saved!");
}
