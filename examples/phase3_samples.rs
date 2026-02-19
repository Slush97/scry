// Phase 3 samples — unified text rendering via DrawCommand::Text
use scry_chart::chart::Charts;
use scry_chart::export::save_png;
use scry_chart::svg_export::save_svg;
use scry_chart::theme::Theme;

fn main() {
    let out = std::path::Path::new("/tmp/scry_phase3");
    std::fs::create_dir_all(out).unwrap();

    // 1. Line chart with title, axis labels, subtitle
    let line = Charts::line(&[2.0, 5.0, 3.0, 8.0, 4.0, 7.0, 6.0, 9.0, 5.0, 11.0])
        .title("Revenue Growth")
        .subtitle("Q1–Q10 FY2025")
        .x_label("Quarter")
        .y_label("Revenue ($M)")
        .theme(Theme::dark())
        .build();
    save_png(&line, 800, 500, out.join("01_line.png")).unwrap();
    save_svg(&line, 800, 500, out.join("01_line.svg")).unwrap();

    // 2. Multi-series line with legend
    let multi = Charts::line(&[1.0, 4.0, 2.0, 6.0, 3.0])
        .add_named_series("Product B", &[3.0, 2.0, 5.0, 4.0, 7.0])
        .add_named_series("Product C", &[2.0, 3.0, 1.0, 5.0, 4.0])
        .title("Product Comparison")
        .x_label("Month")
        .y_label("Sales")
        .theme(Theme::dark())
        .build();
    save_png(&multi, 800, 500, out.join("02_multi_line.png")).unwrap();

    // 3. Bar chart with show_values
    let bar = Charts::bar(
        vec!["Engineering".into(), "Sales".into(), "Marketing".into(), "Support".into(), "HR".into()],
        &[42.0, 38.0, 25.0, 18.0, 12.0],
    )
    .title("Headcount by Department")
    .y_label("Employees")
    .theme(Theme::dark())
    .show_values()
    .build();
    save_png(&bar, 800, 500, out.join("03_bar.png")).unwrap();

    // 4. Horizontal bar
    let hbar = Charts::bar(
        vec!["Rust".into(), "Python".into(), "Go".into(), "TypeScript".into(), "Java".into()],
        &[95.0, 82.0, 78.0, 71.0, 65.0],
    )
    .title("Developer Satisfaction (%)")
    .horizontal()
    .theme(Theme::pastel())
    .show_values()
    .build();
    save_png(&hbar, 800, 500, out.join("04_hbar.png")).unwrap();

    // 5. Scatter with annotation and trend line
    let scatter = Charts::scatter(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2.1, 4.3, 5.8, 8.2, 9.5, 12.1, 13.8, 16.0],
    )
    .title("Correlation Study")
    .x_label("Input Variable")
    .y_label("Response")
    .trend_line()
    .theme(Theme::dark())
    .build();
    save_png(&scatter, 800, 500, out.join("05_scatter.png")).unwrap();

    // 6. Pie chart
    let pie = Charts::pie(
        vec!["Desktop".into(), "Mobile".into(), "Tablet".into(), "Other".into()],
        &[45.0, 35.0, 15.0, 5.0],
    )
    .title("Traffic by Device")
    .theme(Theme::dark())
    .build();
    save_png(&pie, 600, 600, out.join("06_pie.png")).unwrap();

    // 7. Heatmap with cell values
    let heatmap = Charts::heatmap(vec![
        vec![8.0, 3.0, 1.0, 5.0],
        vec![2.0, 7.0, 4.0, 6.0],
        vec![9.0, 1.0, 8.0, 3.0],
        vec![4.0, 6.0, 2.0, 9.0],
    ])
    .title("Correlation Matrix")
    .theme(Theme::dark())
    .build();
    save_png(&heatmap, 600, 500, out.join("07_heatmap.png")).unwrap();

    // 8. Gauge
    let gauge = Charts::gauge(73.5)
        .title("System Health")
        .label("73.5%")
        .theme(Theme::dark())
        .build();
    save_png(&gauge, 500, 400, out.join("08_gauge.png")).unwrap();

    // 9. Histogram
    let data: Vec<f64> = (0..200).map(|i| {
        let x = i as f64 / 20.0;
        (x - 5.0).powi(2) * 0.1 + (i as f64 * 0.1).sin() * 2.0
    }).collect();
    let hist = Charts::histogram(&data)
        .title("Distribution of Values")
        .x_label("Value")
        .y_label("Frequency")
        .theme(Theme::dark())
        .build();
    save_png(&hist, 800, 500, out.join("09_histogram.png")).unwrap();

    // 10. Radar
    let radar = Charts::radar(vec!["Speed", "Power", "Range", "Defense", "Magic"])
        .add_series("Warrior", &[9.0, 8.0, 4.0, 7.0, 2.0])
        .add_series("Mage", &[3.0, 4.0, 6.0, 3.0, 10.0])
        .title("Character Stats")
        .theme(Theme::dark())
        .build();
    save_png(&radar, 600, 600, out.join("10_radar.png")).unwrap();

    // 11. Light theme line with rotated tick labels
    let light = Charts::line(&[100.0, 250.0, 180.0, 320.0, 275.0, 400.0])
        .title("Monthly Revenue")
        .subtitle("All regions combined")
        .x_label("Month")
        .y_label("Revenue ($K)")
        .x_tick_rotation(scry_chart::axis::LabelRotation::Diagonal)
        .theme(Theme::light())
        .build();
    save_png(&light, 800, 500, out.join("11_light_rotated.png")).unwrap();

    // 12. Funnel
    let funnel = Charts::funnel(
        vec!["Visitors".into(), "Signups".into(), "Trials".into(), "Paid".into()],
        &[10000.0, 5200.0, 2100.0, 840.0],
    )
    .title("Conversion Funnel")
    .theme(Theme::dark())
    .build();
    save_png(&funnel, 700, 500, out.join("12_funnel.png")).unwrap();

    println!("Generated 12 PNG samples + 1 SVG in {}", out.display());
    println!("Open with: xdg-open {}/01_line.png", out.display());
}
