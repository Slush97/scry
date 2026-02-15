//! Session 3 showcase — generates PNG output for every new chart type.

use scry_chart::chart::Chart;
use scry_chart::data::{Series, SeriesStyle};
use scry_chart::export::save_png;
use scry_chart::theme::Theme;
use scry_engine::style::Color;

fn main() {
    // ── 1. Bubble Chart ────────────────────────────────────────────────
    let bubble = Chart::bubble(
        &[1.0, 2.5, 4.0, 5.5, 7.0, 8.5],
        &[30.0, 55.0, 45.0, 70.0, 40.0, 65.0],
        &[500.0, 1200.0, 300.0, 2000.0, 800.0, 1500.0],
    )
    .add_named_series("Series B",
        &[2.0, 3.5, 5.0, 6.5, 8.0],
        &[50.0, 35.0, 60.0, 25.0, 55.0],
        &[700.0, 400.0, 1800.0, 600.0, 1100.0],
    )
    .title("Market Analysis — Bubble Chart")
    .x_label("Revenue ($M)")
    .y_label("Growth (%)")
    .size_range(4.0, 22.0)
    .opacity(0.65)
    .theme(Theme::dark())
    .build();

    save_png(&bubble, 800, 500, "/tmp/session3_bubble.png").unwrap();
    println!("✓ Bubble chart → /tmp/session3_bubble.png");

    // ── 2. Violin Plot ─────────────────────────────────────────────────
    let violin = Chart::violin(vec![
        ("Control", vec![
            2.1, 2.5, 2.8, 3.0, 3.1, 2.9, 2.7, 3.3, 2.4, 2.6,
            3.2, 2.8, 3.0, 2.5, 2.9, 3.1, 2.7, 2.8, 3.4, 2.6,
        ]),
        ("Drug A", vec![
            4.2, 4.5, 3.8, 5.1, 4.7, 4.3, 4.9, 3.7, 4.6, 5.0,
            4.1, 4.8, 4.4, 5.2, 3.9, 4.6, 4.3, 5.3, 4.0, 4.7,
        ]),
        ("Drug B", vec![
            3.5, 4.0, 3.2, 4.5, 3.8, 3.6, 4.2, 3.3, 3.9, 4.1,
            3.7, 3.4, 4.3, 3.1, 3.8, 4.4, 3.6, 3.5, 4.0, 3.9,
        ]),
    ])
    .inner_box()
    .title("Clinical Trial — Violin Plot")
    .y_label("Response Score")
    .theme(Theme::dark())
    .build();

    save_png(&violin, 800, 500, "/tmp/session3_violin.png").unwrap();
    println!("✓ Violin plot  → /tmp/session3_violin.png");

    // ── 3. Bar Chart with Error Bars ───────────────────────────────────
    let labels: Vec<String> = vec![
        "Q1".into(), "Q2".into(), "Q3".into(), "Q4".into(),
    ];
    let revenue = Series::new("Revenue", vec![120.0, 145.0, 132.0, 158.0])
        .with_error(vec![12.0, 8.0, 15.0, 10.0]);
    let costs = Series::new("Costs", vec![85.0, 92.0, 88.0, 95.0])
        .with_error(vec![7.0, 5.0, 9.0, 6.0]);
    let bar = scry_chart::chart::BarChart::new(labels, vec![revenue, costs])
        .title("Quarterly Results with Error Bars")
        .y_label("Amount ($K)")
        .show_values()
        .theme(Theme::dark())
        .build();

    save_png(&bar, 800, 500, "/tmp/session3_bar_errors.png").unwrap();
    println!("✓ Bar + errors → /tmp/session3_bar_errors.png");

    // ── 4. Sparklines ──────────────────────────────────────────────────
    // Line sparkline
    let spark_line = Chart::sparkline(&[
        3.0, 7.0, 4.0, 8.0, 2.0, 9.0, 5.0, 6.0, 1.0, 8.0, 4.0, 7.0,
    ])
    .filled()
    .color(Color::from_rgba8(100, 200, 255, 255))
    .build();

    save_png(&spark_line, 200, 40, "/tmp/session3_sparkline_line.png").unwrap();
    println!("✓ Sparkline (line)     → /tmp/session3_sparkline_line.png");

    // Bar sparkline
    let spark_bar = Chart::sparkline(&[
        5.0, 8.0, 3.0, 7.0, 9.0, 4.0, 6.0, 2.0, 8.0, 5.0,
    ])
    .bar()
    .color(Color::from_rgba8(130, 230, 130, 255))
    .build();

    save_png(&spark_bar, 200, 40, "/tmp/session3_sparkline_bar.png").unwrap();
    println!("✓ Sparkline (bar)      → /tmp/session3_sparkline_bar.png");

    // Win/Loss sparkline
    let spark_wl = Chart::sparkline(&[
        1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0,
    ])
    .win_loss()
    .color(Color::from_rgba8(255, 170, 80, 255))
    .build();

    save_png(&spark_wl, 200, 40, "/tmp/session3_sparkline_wl.png").unwrap();
    println!("✓ Sparkline (win/loss) → /tmp/session3_sparkline_wl.png");

    println!("\nAll Session 3 charts generated!");
}
