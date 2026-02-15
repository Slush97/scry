//! Subplot grid demo — renders a 2×2 grid of different chart types.

use scry_chart::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let line = Chart::line(&[1.0, 4.0, 2.0, 8.0, 5.0, 7.0, 3.0])
        .title("Revenue Trend")
        .x_label("Month")
        .y_label("Revenue ($k)")
        .theme(Theme::dark())
        .build();

    let scatter = Chart::scatter(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2.0, 4.5, 3.0, 7.0, 5.5, 8.0],
    )
    .title("Price vs Volume")
    .x_label("Price")
    .y_label("Volume")
    .theme(Theme::dark())
    .build();

    let bar = Chart::bar(
        vec![
            "Q1".to_string(),
            "Q2".to_string(),
            "Q3".to_string(),
            "Q4".to_string(),
        ],
        &[120.0, 185.0, 210.0, 175.0],
    )
    .title("Quarterly Sales")
    .y_label("Units")
    .theme(Theme::dark())
    .build();

    let hist = Chart::histogram(&[
        22.0, 25.0, 27.0, 28.0, 30.0, 31.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 40.0, 42.0,
        45.0,
    ])
    .title("Response Times")
    .x_label("ms")
    .y_label("Count")
    .theme(Theme::dark())
    .build();

    let grid = SubplotGrid::new(2, 2)
        .gap(16)
        .title("Dashboard Overview")
        .set(0, 0, line)
        .set(0, 1, scatter)
        .set(1, 0, bar)
        .set(1, 1, hist);

    scry_chart::export::save_subplot_png(&grid, 1600, 1000, "subplot_demo.png")
        .map_err(|e| e.into())
}
