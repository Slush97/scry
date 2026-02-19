//! Axis stress test — renders every edge case that previously broke axis formatting.
//!
//! Run: cargo run -p scry-chart --example axis_stress_test
//! Output: /tmp/axis_stress_test/*.png

use scry_chart::chart::Charts;
use scry_chart::export::save_png;
use scry_chart::prelude::*;

fn main() {
    let out = "/tmp/axis_stress_test";
    std::fs::create_dir_all(out).unwrap();

    let mut count = 0;

    // ── GROUP 1: Y-label vs tick label collision edge cases ──

    // Case 1: Micro-range values → wide tick labels like "0.0020"
    let chart = Charts::line(&[0.001, 0.0015, 0.0012, 0.0018, 0.0014])
        .title("Micro Range (0.001–0.002)")
        .x_label("Sample")
        .y_label("PPM")
        .theme(Theme::dark())
        .build();
    for &(w, h) in &[(300, 200), (600, 400), (1200, 800)] {
        count += 1;
        let name = format!("{out}/{count:02}_micro_range_{w}x{h}.png");
        save_png(&chart, w, h, &name).unwrap();
        println!("✓ {name}");
    }

    // Case 2: Large values → tick labels like "250K", "350K"
    let chart = Charts::line(&[150000.0, 250000.0, 180000.0, 350000.0, 280000.0])
        .title("Large Values (150K–350K)")
        .x_label("Quarter")
        .y_label("Revenue")
        .theme(Theme::dark())
        .build();
    for &(w, h) in &[(300, 200), (600, 400), (1200, 800)] {
        count += 1;
        let name = format!("{out}/{count:02}_large_values_{w}x{h}.png");
        save_png(&chart, w, h, &name).unwrap();
        println!("✓ {name}");
    }

    // Case 3: Long Y-axis label "Revenue ($K)" — was the hardest collision
    let chart = Charts::line(&[10.0, 25.0, 18.0, 35.0, 28.0, 42.0, 30.0])
        .title("Revenue Analysis")
        .x_label("Month of Year")
        .y_label("Revenue ($K)")
        .filled()
        .with_points()
        .theme(Theme::dark())
        .build();
    for &(w, h) in &[(300, 200), (600, 400), (1200, 800)] {
        count += 1;
        let name = format!("{out}/{count:02}_long_labels_{w}x{h}.png");
        save_png(&chart, w, h, &name).unwrap();
        println!("✓ {name}");
    }

    // Case 4: Very long Y-label (worst case)
    let chart = Charts::line(&[5.0, 10.0, 8.0, 15.0, 12.0])
        .title("Extremely Long Axis Labels")
        .x_label("Time (Milliseconds)")
        .y_label("Temperature (°C)")
        .theme(Theme::dark())
        .build();
    for &(w, h) in &[(300, 200), (800, 500)] {
        count += 1;
        let name = format!("{out}/{count:02}_very_long_labels_{w}x{h}.png");
        save_png(&chart, w, h, &name).unwrap();
        println!("✓ {name}");
    }

    // ── GROUP 2: Negative values and zero-crossing ──

    // Case 5: Negative bars (tick labels like "-5", "-8")
    let chart = Charts::bar(
        vec!["Q1".into(), "Q2".into(), "Q3".into(), "Q4".into()],
        &[10.0, -5.0, 15.0, -8.0],
    )
    .title("Profit/Loss by Quarter")
    .y_label("Profit/Loss")
    .theme(Theme::dark())
    .build();
    for &(w, h) in &[(300, 200), (600, 400), (1200, 800)] {
        count += 1;
        let name = format!("{out}/{count:02}_negative_bars_{w}x{h}.png");
        save_png(&chart, w, h, &name).unwrap();
        println!("✓ {name}");
    }

    // ── GROUP 3: Narrow range (many decimals) ──

    // Case 6: Range 100.1–100.5 → tick labels like "100.1", "100.3"
    let chart = Charts::line(&[100.1, 100.3, 100.2, 100.5, 100.4])
        .title("Narrow Range Pressure")
        .x_label("Index")
        .y_label("Pressure")
        .theme(Theme::dark())
        .build();
    for &(w, h) in &[(300, 200), (800, 500), (1200, 800)] {
        count += 1;
        let name = format!("{out}/{count:02}_narrow_range_{w}x{h}.png");
        save_png(&chart, w, h, &name).unwrap();
        println!("✓ {name}");
    }

    // ── GROUP 4: Scatter with both axis labels ──

    let chart = Charts::scatter(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2.0, 4.0, 1.5, 8.0, 5.0, 7.5, 3.0, 9.0],
    )
    .title("Scatter Plot")
    .x_label("X Axis")
    .y_label("Y Axis")
    .theme(Theme::dark())
    .build();
    for &(w, h) in &[(300, 200), (600, 400)] {
        count += 1;
        let name = format!("{out}/{count:02}_scatter_{w}x{h}.png");
        save_png(&chart, w, h, &name).unwrap();
        println!("✓ {name}");
    }

    // ── GROUP 5: Histogram ──

    let hist_data: Vec<f64> = (0..200)
        .map(|i| {
            let mut sum = 0.0;
            for k in 0..6u64 {
                sum += ((i as u64 * 2654435761 + k * 7919) % 10000) as f64 / 10000.0;
            }
            (sum - 3.0) * 2.0 + 10.0
        })
        .collect();
    let chart = Charts::histogram(&hist_data)
        .title("Distribution")
        .x_label("Value")
        .y_label("Count")
        .bins(20)
        .theme(Theme::dark())
        .build();
    for &(w, h) in &[(300, 200), (800, 500)] {
        count += 1;
        let name = format!("{out}/{count:02}_histogram_{w}x{h}.png");
        save_png(&chart, w, h, &name).unwrap();
        println!("✓ {name}");
    }

    // ── GROUP 6: Boxplot ──

    let chart = Charts::boxplot(vec![
        ("Low", vec![1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 3.5]),
        ("Medium", vec![5.0, 6.0, 7.0, 8.0, 9.0, 6.5, 7.5]),
        ("High", vec![10.0, 12.0, 14.0, 16.0, 18.0, 13.0, 15.0]),
    ])
    .title("Distribution by Group")
    .y_label("Measurement")
    .theme(Theme::dark())
    .build();
    for &(w, h) in &[(300, 200), (800, 500)] {
        count += 1;
        let name = format!("{out}/{count:02}_boxplot_{w}x{h}.png");
        save_png(&chart, w, h, &name).unwrap();
        println!("✓ {name}");
    }

    // ── GROUP 7: Light theme (tests text on light background) ──

    let chart = Charts::line(&[3.0, 7.0, 5.0, 12.0, 9.0, 15.0])
        .title("Light Theme Test")
        .x_label("Period")
        .y_label("Sales Volume")
        .theme(Theme::light())
        .build();
    count += 1;
    let name = format!("{out}/{count:02}_light_theme_800x500.png");
    save_png(&chart, 800, 500, &name).unwrap();
    println!("✓ {name}");

    // ── GROUP 8: Edge case — no labels at all ──

    let chart = Charts::line(&[1.0, 5.0, 3.0, 8.0, 6.0])
        .theme(Theme::dark())
        .build();
    count += 1;
    let name = format!("{out}/{count:02}_no_labels_800x500.png");
    save_png(&chart, 800, 500, &name).unwrap();
    println!("✓ {name}");

    // ── GROUP 9: Extreme values ──

    let chart = Charts::line(&[0.0000001, 0.0000002, 0.00000015, 0.00000025])
        .title("Sub-Micro Values")
        .y_label("Concentration")
        .theme(Theme::dark())
        .build();
    count += 1;
    let name = format!("{out}/{count:02}_submicro_800x500.png");
    save_png(&chart, 800, 500, &name).unwrap();
    println!("✓ {name}");

    let chart = Charts::line(&[1e9, 2e9, 1.5e9, 3e9, 2.5e9])
        .title("Billions")
        .y_label("GDP")
        .theme(Theme::dark())
        .build();
    count += 1;
    let name = format!("{out}/{count:02}_billions_800x500.png");
    save_png(&chart, 800, 500, &name).unwrap();
    println!("✓ {name}");

    println!("\n✅ Rendered {count} stress test charts to {out}/");
    println!("   View them: ls {out}/*.png");
}
