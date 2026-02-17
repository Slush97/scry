//! Render the four new Tier 2 chart types to PNG files.
//!
//! Usage: cargo run -p scry-chart --example tier2_charts

use scry_chart::chart::Chart;
use scry_chart::export;
use scry_chart::theme::Theme;
use scry_engine::style::Color;

fn main() {
    let out_dir = std::path::PathBuf::from("chart_output");
    std::fs::create_dir_all(&out_dir).unwrap();

    let charts: Vec<(&str, Chart)> = vec![
        ("waterfall", waterfall_chart()),
        ("funnel", funnel_chart()),
        ("gauge", gauge_chart()),
        ("lollipop", lollipop_chart()),
    ];

    for (name, chart) in &charts {
        let path = out_dir.join(format!("{name}.png"));
        export::save_png(chart, 800, 500, &path).expect("export failed");
        println!(
            "✓ {path:?} ({} bytes)",
            std::fs::metadata(&path).unwrap().len()
        );
    }

    println!("\nAll Tier 2 charts written to {out_dir:?}/");
}

fn waterfall_chart() -> Chart {
    Chart::waterfall(
        vec![
            "Revenue".into(),
            "COGS".into(),
            "Gross Profit".into(),
            "OpEx".into(),
            "Tax".into(),
            "Marketing".into(),
        ],
        &[1200.0, -450.0, 200.0, -300.0, -120.0, -80.0],
    )
    .title("P&L Waterfall — FY2025")
    .y_label("USD ($K)")
    .show_values()
    .theme(Theme::dark())
    .build()
}

fn funnel_chart() -> Chart {
    Chart::funnel(
        vec![
            "Website Visitors".into(),
            "Sign-ups".into(),
            "Free Trials".into(),
            "Paid Conversions".into(),
            "Enterprise Upsell".into(),
        ],
        &[50000.0, 18000.0, 7500.0, 3200.0, 800.0],
    )
    .title("SaaS Conversion Funnel")
    .theme(Theme::dark())
    .build()
}

fn gauge_chart() -> Chart {
    Chart::gauge(73.0)
        .range(0.0, 100.0)
        .threshold(40.0, Color::from_rgba8(46, 204, 113, 255)) // green
        .threshold(70.0, Color::from_rgba8(241, 196, 15, 255)) // yellow
        .threshold(90.0, Color::from_rgba8(230, 126, 34, 255)) // orange
        .threshold(100.0, Color::from_rgba8(231, 76, 60, 255)) // red
        .label("73%")
        .title("CPU Utilization")
        .theme(Theme::dark())
        .build()
}

fn lollipop_chart() -> Chart {
    Chart::lollipop(
        vec![
            "Rust".into(),
            "Python".into(),
            "Go".into(),
            "TypeScript".into(),
            "Java".into(),
            "C++".into(),
        ],
        &[92.0, 78.0, 71.0, 65.0, 58.0, 45.0],
    )
    .title("Developer Satisfaction Score")
    .y_label("Score")
    .show_values()
    .theme(Theme::dark())
    .build()
}
