//! Render every chart type to PNG for visual axis audit.
//!
//! Run: cargo run -p pixelchart --example render_all

use pixelchart::export::save_png;
use pixelchart::prelude::*;

fn main() -> Result<(), String> {
    let dir = "/tmp/pixelchart_audit";
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;

    // 1. Simple line chart
    let chart = Chart::line(&[1.0, 4.0, 2.0, 8.0, 5.0, 3.0, 7.0, 6.0, 9.0, 4.5])
        .title("Line Chart — Basic")
        .x_label("Time")
        .y_label("Value")
        .theme(Theme::dark())
        .build();
    save_png(&chart, 800, 500, format!("{dir}/01_line_basic.png"))?;

    // 2. Line chart with fill + points
    let chart = Chart::line(&[10.0, 25.0, 18.0, 35.0, 28.0, 42.0, 30.0])
        .title("Line Chart — Filled + Points")
        .x_label("Month")
        .y_label("Revenue ($K)")
        .filled()
        .with_points()
        .theme(Theme::dark())
        .build();
    save_png(&chart, 800, 500, format!("{dir}/02_line_filled.png"))?;

    // 3. Scatter plot
    let chart = Chart::scatter(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2.0, 4.0, 1.5, 8.0, 5.0, 7.5, 3.0, 9.0],
    )
    .title("Scatter Plot — Basic")
    .x_label("X Axis")
    .y_label("Y Axis")
    .theme(Theme::dark())
    .build();
    save_png(&chart, 800, 500, format!("{dir}/03_scatter_basic.png"))?;

    // 4. Bar chart — vertical
    let chart = Chart::bar(
        vec![
            "Mon".into(),
            "Tue".into(),
            "Wed".into(),
            "Thu".into(),
            "Fri".into(),
        ],
        &[12.0, 19.0, 8.0, 15.0, 22.0],
    )
    .title("Bar Chart — Vertical")
    .y_label("Units Sold")
    .theme(Theme::dark())
    .build();
    save_png(&chart, 800, 500, format!("{dir}/04_bar_vertical.png"))?;

    // 5. Bar chart — horizontal
    let chart = Chart::bar(
        vec![
            "Alpha".into(),
            "Beta".into(),
            "Gamma".into(),
            "Delta".into(),
        ],
        &[30.0, 50.0, 20.0, 45.0],
    )
    .title("Bar Chart — Horizontal")
    .x_label("Score")
    .horizontal()
    .theme(Theme::dark())
    .build();
    save_png(&chart, 800, 500, format!("{dir}/05_bar_horizontal.png"))?;

    // 6. Histogram
    let data: Vec<f64> = (0..200)
        .map(|i| {
            let x = i as f64 / 200.0;
            // Pseudo-normal using sum of uniform
            let mut sum = 0.0;
            for k in 0..6u64 {
                sum += ((i as u64 * 2654435761 + k * 7919) % 10000) as f64 / 10000.0;
            }
            (sum - 3.0) * 2.0 + 10.0
        })
        .collect();
    let chart = Chart::histogram(&data)
        .title("Histogram — Frequency")
        .x_label("Value")
        .y_label("Count")
        .bins(20)
        .theme(Theme::dark())
        .build();
    save_png(&chart, 800, 500, format!("{dir}/06_histogram.png"))?;

    // 7. Box plot
    let g1: Vec<f64> = (0..40)
        .map(|i| 10.0 + (i as f64 * 0.5).sin() * 3.0 + i as f64 * 0.1)
        .collect();
    let g2: Vec<f64> = (0..40)
        .map(|i| 15.0 + (i as f64 * 0.3).cos() * 4.0)
        .collect();
    let g3: Vec<f64> = (0..40)
        .map(|i| 8.0 + (i as f64 * 0.7).sin() * 2.0 + i as f64 * 0.15)
        .collect();
    let chart = Chart::boxplot(vec![("Group A", g1), ("Group B", g2), ("Group C", g3)])
        .title("Box Plot — Distributions")
        .y_label("Score")
        .theme(Theme::dark())
        .build();
    save_png(&chart, 800, 500, format!("{dir}/07_boxplot.png"))?;

    // 8. Heatmap
    let chart = Chart::heatmap(vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![5.0, 4.0, 3.0, 2.0, 1.0],
        vec![2.0, 4.0, 6.0, 4.0, 2.0],
        vec![3.0, 1.0, 5.0, 7.0, 3.0],
    ])
    .title("Heatmap — Basic")
    .row_labels(vec!["R1".into(), "R2".into(), "R3".into(), "R4".into()])
    .col_labels(vec![
        "C1".into(),
        "C2".into(),
        "C3".into(),
        "C4".into(),
        "C5".into(),
    ])
    .theme(Theme::dark())
    .build();
    save_png(&chart, 800, 500, format!("{dir}/08_heatmap.png"))?;

    // 9. Pie chart
    let chart = Chart::pie(
        vec![
            "Rust".into(),
            "Python".into(),
            "Go".into(),
            "TypeScript".into(),
            "Other".into(),
        ],
        &[35.0, 25.0, 15.0, 15.0, 10.0],
    )
    .title("Pie Chart — Language Share")
    .theme(Theme::dark())
    .build();
    save_png(&chart, 800, 500, format!("{dir}/09_pie.png"))?;

    // 10. Small canvas (stress test)
    let chart = Chart::line(&[1.0, 4.0, 2.0, 8.0, 5.0])
        .title("Small Canvas Test")
        .x_label("X")
        .y_label("Y")
        .theme(Theme::dark())
        .build();
    save_png(&chart, 300, 200, format!("{dir}/10_small_canvas.png"))?;

    // 11. Negative data
    let chart = Chart::bar(
        vec!["A".into(), "B".into(), "C".into(), "D".into()],
        &[10.0, -5.0, 15.0, -8.0],
    )
    .title("Bar Chart — Negative Values")
    .y_label("Profit/Loss")
    .theme(Theme::dark())
    .build();
    save_png(&chart, 800, 500, format!("{dir}/11_bar_negative.png"))?;

    // 12. Large values
    let chart = Chart::line(&[150000.0, 250000.0, 180000.0, 350000.0, 280000.0])
        .title("Line Chart — Large Values")
        .x_label("Quarter")
        .y_label("Revenue")
        .theme(Theme::dark())
        .build();
    save_png(&chart, 800, 500, format!("{dir}/12_line_large_values.png"))?;

    // 13. Micro-range values
    let chart = Chart::line(&[0.001, 0.0015, 0.0012, 0.0018, 0.0014])
        .title("Line Chart — Micro Range")
        .x_label("Sample")
        .y_label("PPM")
        .theme(Theme::dark())
        .build();
    save_png(&chart, 800, 500, format!("{dir}/13_line_micro_range.png"))?;

    // 14. Light theme
    let chart = Chart::scatter(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2.0, 4.0, 3.0, 7.0, 5.0])
        .title("Scatter — Light Theme")
        .x_label("X")
        .y_label("Y")
        .theme(Theme::light())
        .build();
    save_png(&chart, 800, 500, format!("{dir}/14_scatter_light.png"))?;

    println!("✓ Rendered 14 charts to {dir}/");
    println!("  View them with: ls {dir}/*.png");
    Ok(())
}
