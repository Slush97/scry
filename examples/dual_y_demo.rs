//! Dual Y-axis line chart demo — saves a PNG to /tmp/dual_y_axis.png.
//!
//! Left Y:  Monthly Revenue ($K) — filled area with gradient
//! Right Y: Conversion Rate (%) — dashed line with points
//!
//! Run:  cargo run --example dual_y_demo

use scry_chart::chart::LineChart;
use scry_chart::data::{GradientFill, Series, SeriesStyle};
use scry_chart::export::save_png;
use scry_chart::formatter::PercentFormatter;
use scry_chart::theme::Theme;
use scry_engine::style::Color;

fn main() {
    // --- Data ---
    let months: Vec<f64> = (1..=12).map(|m| m as f64).collect();

    let revenue = Series::new(
        "Revenue ($K)",
        vec![
            42.0, 45.5, 51.2, 48.8, 55.3, 62.1, 59.7, 67.4, 72.8, 68.9, 78.2, 85.0,
        ],
    )
    .style(
        SeriesStyle::new()
            .color(Color::from_rgba8(99, 179, 237, 255))
            .line_width(2.8)
            .fill_gradient(GradientFill::TopToBottom),
    );

    let conversion = Series::new(
        "Conversion %",
        vec![3.2, 3.5, 3.1, 3.8, 4.2, 4.0, 4.5, 4.3, 4.8, 5.1, 4.9, 5.4],
    )
    .style(
        SeriesStyle::new()
            .color(Color::from_rgba8(252, 129, 155, 255))
            .line_width(2.5),
    );

    // --- Build chart ---
    let chart = LineChart::new(vec![revenue, conversion])
        .x_values(months)
        .filled()
        .with_points()
        .dash_lines()
        .title("2025 Monthly Performance")
        .subtitle("Revenue vs Conversion Rate")
        .x_label("Month")
        .y_label("Revenue ($K)")
        .y_range(0.0, 100.0)
        .secondary_y_label("Conversion Rate (%)")
        .secondary_y_range(0.0, 8.0)
        .secondary_y_formatter(PercentFormatter {
            decimals: 1,
            is_fraction: false,
        })
        .secondary_axis(&[1])
        .theme(Theme::dark())
        .legend_horizontal()
        .build();

    // --- Export ---
    let out = "/tmp/dual_y_axis.png";
    save_png(&chart, 1000, 600, out).expect("PNG export failed");
    println!("✓ Saved dual Y-axis chart to {out}");
}
