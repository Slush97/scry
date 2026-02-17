//! Generate sample charts at multiple resolutions to demonstrate font scaling.
//!
//! Run: cargo run -p scry-chart --example font_scaling_demo

use scry_chart::chart::Chart;
use scry_chart::export;
use scry_chart::theme::Theme;

fn main() {
    let out_dir = std::path::PathBuf::from("target/font_scaling_demo");
    std::fs::create_dir_all(&out_dir).unwrap();

    let sizes: &[(u32, u32)] = &[
        (200, 150),  // tiny
        (400, 300),  // reference (1x)
        (800, 600),  // 2x area
        (1200, 800), // large
    ];

    let charts: Vec<(&str, Chart)> = vec![
        ("bar", bar_chart()),
        ("line", line_chart()),
        ("pie", pie_chart()),
    ];

    for (name, chart) in &charts {
        for &(w, h) in sizes {
            let path = out_dir.join(format!("{name}_{w}x{h}.png"));
            export::save_png(chart, w, h, &path).expect("export failed");
            println!(
                "✓ {} ({} bytes)",
                path.display(),
                std::fs::metadata(&path).unwrap().len()
            );
        }
    }

    println!("\nAll charts saved to {}/", out_dir.display());
}

fn bar_chart() -> Chart {
    Chart::bar(
        vec!["Q1".into(), "Q2".into(), "Q3".into(), "Q4".into()],
        &[120.0, 340.0, 280.0, 410.0],
    )
    .title("Quarterly Revenue")
    .y_label("USD (K)")
    .show_values()
    .add_named_series("Region B", &[90.0, 260.0, 310.0, 370.0])
    .theme(Theme::dark())
    .build()
}

fn line_chart() -> Chart {
    Chart::line(&[10.0, 25.0, 18.0, 32.0, 45.0, 38.0, 52.0])
        .title("Growth vs Target")
        .x_label("Month")
        .y_label("Revenue ($K)")
        .add_named_series("Target", &[15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0])
        .filled()
        .with_points()
        .theme(Theme::ocean())
        .build()
}

fn pie_chart() -> Chart {
    Chart::pie(
        vec![
            "Product A".into(),
            "Product B".into(),
            "Product C".into(),
            "Product D".into(),
            "Other".into(),
        ],
        &[35.0, 25.0, 20.0, 12.0, 8.0],
    )
    .title("Market Share 2026")
    .theme(Theme::pastel())
    .build()
}
