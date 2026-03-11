//! Render one PNG per chart type for the visual walkthrough.
//!
//! Usage: cargo run -p scry-chart --example chart_gallery -- /path/to/output/dir

use scry_chart::chart::OhlcEntry;
use scry_chart::chart::{Chart, Charts};
use scry_chart::export;
use scry_chart::theme::Theme;

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/chart_gallery".to_string());
    let out = std::path::PathBuf::from(&out_dir);
    std::fs::create_dir_all(&out).unwrap();

    let charts: Vec<(&str, Chart)> = vec![
        // ---- Core chart types ----
        (
            "01_line",
            Charts::line(&[10.0, 25.0, 18.0, 35.0, 28.0, 42.0, 38.0, 50.0])
                .title("Line Chart")
                .x_label("Month")
                .y_label("Revenue ($K)")
                .add_named_series("Q2", &[15.0, 30.0, 22.0, 40.0, 32.0, 48.0, 44.0, 55.0])
                .filled()
                .with_points()
                .theme(Theme::dark())
                .build(),
        ),
        (
            "02_area",
            Charts::area(&[3.0, 7.0, 4.0, 9.0, 6.0, 8.0, 5.0, 11.0])
                .title("Area Chart")
                .x_label("Time")
                .y_label("Value")
                .theme(Theme::dark())
                .build(),
        ),
        (
            "03_scatter",
            Charts::scatter(
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                &[2.1, 3.5, 2.8, 5.2, 4.1, 6.3, 5.5, 7.0],
            )
            .title("Scatter Plot")
            .x_label("Temperature (°C)")
            .y_label("Growth (cm)")
            .trend_line()
            .theme(Theme::dark())
            .build(),
        ),
        (
            "04_bar",
            Charts::bar(
                vec![
                    "Mon".into(),
                    "Tue".into(),
                    "Wed".into(),
                    "Thu".into(),
                    "Fri".into(),
                ],
                &[120.0, 200.0, 150.0, 280.0, 190.0],
            )
            .title("Bar Chart")
            .y_label("Sales")
            .add_named_series("Region B", &[90.0, 160.0, 180.0, 110.0, 220.0])
            .theme(Theme::dark())
            .build(),
        ),
        ("05_histogram", {
            let data: Vec<f64> = (0..500)
                .map(|i| {
                    let x = i as f64 / 500.0 * 12.56;
                    x.sin() * 30.0 + 50.0 + (i as f64 * 0.1).sin() * 10.0
                })
                .collect();
            Charts::histogram(&data)
                .bins(25)
                .title("Histogram (Touching Bins)")
                .x_label("Amplitude")
                .y_label("Frequency")
                .theme(Theme::dark())
                .build()
        }),
        (
            "06_boxplot",
            Charts::boxplot(vec![
                (
                    "Group A",
                    (0..50)
                        .map(|i| 10.0 + (i as f64 * 0.5).sin() * 3.0)
                        .collect(),
                ),
                (
                    "Group B",
                    (0..50)
                        .map(|i| 15.0 + (i as f64 * 0.3).cos() * 4.0)
                        .collect(),
                ),
                (
                    "Group C",
                    (0..50)
                        .map(|i| 8.0 + (i as f64 * 0.7).sin() * 2.0 + i as f64 * 0.15)
                        .collect(),
                ),
            ])
            .title("Box Plot")
            .y_label("Score")
            .theme(Theme::dark())
            .build(),
        ),
        (
            "07_heatmap",
            Charts::heatmap(vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0],
                vec![5.0, 4.0, 3.0, 2.0, 1.0],
                vec![2.0, 4.0, 6.0, 4.0, 2.0],
                vec![3.0, 1.0, 5.0, 7.0, 3.0],
            ])
            .title("Heatmap")
            .row_labels(vec!["R1".into(), "R2".into(), "R3".into(), "R4".into()])
            .col_labels(vec![
                "C1".into(),
                "C2".into(),
                "C3".into(),
                "C4".into(),
                "C5".into(),
            ])
            .theme(Theme::dark())
            .build(),
        ),
        (
            "08_pie",
            Charts::pie(
                vec![
                    "Rust".into(),
                    "Python".into(),
                    "Go".into(),
                    "TypeScript".into(),
                    "Other".into(),
                ],
                &[35.0, 25.0, 15.0, 15.0, 10.0],
            )
            .title("Pie Chart")
            .theme(Theme::dark())
            .build(),
        ),
        (
            "09_radar",
            Charts::radar(vec!["Speed", "Power", "Range", "Defense", "Magic"])
                .add_series("Hero", &[0.8, 0.6, 0.9, 0.4, 0.7])
                .add_series("Villain", &[0.5, 0.9, 0.3, 0.8, 0.6])
                .title("Radar Chart")
                .theme(Theme::dark())
                .build(),
        ),
        (
            "10_candlestick",
            Charts::candlestick(vec![
                OhlcEntry::new(1.0, 100.0, 110.0, 95.0, 105.0),
                OhlcEntry::new(2.0, 105.0, 115.0, 100.0, 98.0),
                OhlcEntry::new(3.0, 98.0, 108.0, 92.0, 106.0),
                OhlcEntry::new(4.0, 106.0, 120.0, 104.0, 118.0),
                OhlcEntry::new(5.0, 118.0, 125.0, 112.0, 110.0),
                OhlcEntry::new(6.0, 110.0, 116.0, 105.0, 114.0),
                OhlcEntry::new(7.0, 114.0, 122.0, 108.0, 108.0),
            ])
            .title("Candlestick (Okabe-Ito)")
            .x_label("Day")
            .y_label("Price")
            .theme(Theme::dark())
            .build(),
        ),
        (
            "11_violin",
            Charts::violin(vec![
                (
                    "A",
                    (0..80)
                        .map(|i| 10.0 + (i as f64 * 0.2).sin() * 5.0)
                        .collect(),
                ),
                (
                    "B",
                    (0..80)
                        .map(|i| 15.0 + (i as f64 * 0.15).cos() * 6.0)
                        .collect(),
                ),
            ])
            .title("Violin Plot")
            .theme(Theme::dark())
            .build(),
        ),
        (
            "12_waterfall",
            Charts::waterfall(
                vec!["Revenue".into(), "COGS".into(), "OpEx".into(), "Tax".into()],
                &[100.0, -40.0, -25.0, -10.0],
            )
            .title("Waterfall Chart")
            .show_values()
            .theme(Theme::dark())
            .build(),
        ),
        (
            "13_bubble",
            Charts::bubble(
                &[1.0, 3.0, 5.0, 7.0, 9.0],
                &[2.0, 8.0, 4.0, 6.0, 3.0],
                &[10.0, 30.0, 20.0, 40.0, 15.0],
            )
            .title("Bubble Chart")
            .x_label("X")
            .y_label("Y")
            .theme(Theme::dark())
            .build(),
        ),
        (
            "14_lollipop",
            Charts::lollipop(
                vec!["A".into(), "B".into(), "C".into(), "D".into(), "E".into()],
                &[15.0, 30.0, 22.0, 40.0, 18.0],
            )
            .title("Lollipop Chart")
            .theme(Theme::dark())
            .build(),
        ),
        (
            "15_funnel",
            Charts::funnel(
                vec![
                    "Visitors".into(),
                    "Leads".into(),
                    "Qualified".into(),
                    "Deals".into(),
                ],
                &[1000.0, 600.0, 300.0, 100.0],
            )
            .title("Sales Funnel")
            .theme(Theme::dark())
            .build(),
        ),
        (
            "16_gauge",
            Charts::gauge(73.0)
                .range(0.0, 100.0)
                .title("CPU Usage")
                .label("73%")
                .theme(Theme::dark())
                .build(),
        ),
        (
            "17_contour",
            Charts::contour({
                let n = 20;
                (0..n)
                    .map(|r| {
                        let y = r as f64 / n as f64 * 4.0 - 2.0;
                        (0..n)
                            .map(|c| {
                                let x = c as f64 / n as f64 * 4.0 - 2.0;
                                (-(x * x + y * y)).exp()
                            })
                            .collect()
                    })
                    .collect()
            })
            .levels(8)
            .filled()
            .title("Contour Plot")
            .theme(Theme::dark())
            .build(),
        ),
        (
            "18_sparkline",
            Charts::sparkline(&[3.0, 7.0, 4.0, 9.0, 6.0, 8.0, 5.0, 11.0, 7.0, 13.0]).build(),
        ),
        // ---- Theme comparison (line chart across all themes) ----
        (
            "19_theme_light",
            Charts::line(&[10.0, 25.0, 18.0, 35.0, 28.0, 42.0])
                .title("Light Theme")
                .x_label("X")
                .y_label("Y")
                .theme(Theme::light())
                .build(),
        ),
        (
            "20_theme_pastel",
            Charts::line(&[10.0, 25.0, 18.0, 35.0, 28.0, 42.0])
                .title("Pastel Theme")
                .x_label("X")
                .y_label("Y")
                .theme(Theme::pastel())
                .build(),
        ),
        (
            "21_theme_ocean",
            Charts::line(&[10.0, 25.0, 18.0, 35.0, 28.0, 42.0])
                .title("Ocean Theme")
                .x_label("X")
                .y_label("Y")
                .theme(Theme::ocean())
                .build(),
        ),
        (
            "22_theme_forest",
            Charts::line(&[10.0, 25.0, 18.0, 35.0, 28.0, 42.0])
                .title("Forest Theme")
                .x_label("X")
                .y_label("Y")
                .theme(Theme::forest())
                .build(),
        ),
        (
            "23_theme_colorblind",
            Charts::line(&[10.0, 25.0, 18.0, 35.0, 28.0, 42.0])
                .title("Colorblind Theme")
                .x_label("X")
                .y_label("Y")
                .theme(Theme::colorblind())
                .build(),
        ),
        // ---- Stress tests ----
        (
            "24_bar_many_categories",
            Charts::bar(
                (0..15).map(|i| format!("Cat_{i:02}")).collect(),
                &(0..15).map(|i| (i as f64 + 1.0) * 7.0).collect::<Vec<_>>(),
            )
            .title("15 Categories")
            .y_label("Count")
            .theme(Theme::dark())
            .build(),
        ),
        ("25_histogram_dense", {
            let data: Vec<f64> = (0..2000)
                .map(|i| {
                    let t = i as f64 / 200.0;
                    t.sin() * 40.0 + 60.0 + (t * 3.0).cos() * 15.0
                })
                .collect();
            Charts::histogram(&data)
                .bins(40)
                .title("Dense Histogram (2000 pts, 40 bins)")
                .theme(Theme::dark())
                .build()
        }),
        (
            "26_bar_horizontal",
            Charts::bar(
                vec![
                    "Alpha".into(),
                    "Beta".into(),
                    "Gamma".into(),
                    "Delta".into(),
                ],
                &[30.0, 50.0, 20.0, 45.0],
            )
            .horizontal()
            .title("Horizontal Bar")
            .theme(Theme::dark())
            .build(),
        ),
    ];

    let w = 800;
    let h = 500;

    for (name, chart) in &charts {
        let path = out.join(format!("{name}.png"));
        export::save_png(chart, w, h, &path).expect("export failed");
        let size = std::fs::metadata(&path).unwrap().len();
        println!("✓ {name}.png ({:.1} KB)", size as f64 / 1024.0);
    }

    println!("\n✅ {} charts written to {out_dir}", charts.len());
}
