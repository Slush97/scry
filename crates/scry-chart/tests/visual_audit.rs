//! Comprehensive visual audit — renders every chart type × builder option × edge case to PNG.
//!
//! Run: `cargo run -p scry-chart --example visual_audit`
//! Output: `/tmp/scry_chart_audit/*.png`

#![allow(
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::cast_possible_truncation
)]

use scry_chart::prelude::*;
use scry_engine::style::Color;

const W: u32 = 800;
const H: u32 = 500;
const DIR: &str = "/tmp/scry_chart_audit";

fn main() {
    std::fs::create_dir_all(DIR).expect("create output dir");

    let mut count = 0u32;

    macro_rules! emit {
        ($name:expr, $chart:expr) => {{
            count += 1;
            let path = format!("{DIR}/{count:02}_{}.png", $name);
            match save_png(&$chart, W, H, &path) {
                Ok(()) => println!("  ✓ {path}"),
                Err(e) => eprintln!("  ✗ {path}: {e}"),
            }
        }};
    }

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║         PIXELCHART VISUAL AUDIT — PNG Generator         ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // ═══════════════════════════════════════════════════════════════════
    // SECTION 1: LINE CHARTS
    // ═══════════════════════════════════════════════════════════════════
    println!("▸ Section 1: Line Charts");

    // 1. Basic line
    let y: Vec<f64> = (0..30)
        .map(|i| (i as f64 * 0.3).sin() * 5.0 + 10.0)
        .collect();
    emit!(
        "line_basic",
        Charts::line(&y)
            .title("Basic Line Chart")
            .x_label("Sample Index")
            .y_label("Value")
            .build()
    );

    // 2. Multi-series line
    let x = linspace(0.0, 10.0, 50);
    let y1: Vec<f64> = x.iter().map(|&v| v.sin() * 3.0).collect();
    let y2: Vec<f64> = x.iter().map(|&v| v.cos() * 3.0).collect();
    let y3: Vec<f64> = x.iter().map(|&v| (v * 1.5).sin() * 2.0 + 1.0).collect();
    emit!(
        "line_multi_series",
        Charts::line(&y1)
            .title("Multi-Series Line")
            .x_label("Time")
            .y_label("Amplitude")
            .x_values(x.clone())
            .add_named_series("cos(x)", &y2)
            .add_named_series("sin(1.5x)+1", &y3)
            .build()
    );

    // 3. Smooth (Catmull-Rom)
    let y_smooth: Vec<f64> = (0..15)
        .map(|i| {
            let x = i as f64;
            (x * 0.5).sin() * 4.0 + (x * 0.2).cos() * 2.0
        })
        .collect();
    emit!(
        "line_smooth",
        Charts::line(&y_smooth)
            .title("Smooth Line (Catmull-Rom Spline)")
            .x_label("X")
            .y_label("Y")
            .smooth()
            .with_points()
            .theme(Theme::pastel())
            .build()
    );

    // 4. Step line
    let y_step: Vec<f64> = vec![2.0, 2.0, 5.0, 5.0, 3.0, 8.0, 8.0, 4.0, 4.0, 6.0];
    emit!(
        "line_step",
        Charts::line(&y_step)
            .title("Step Line (Stairstep)")
            .x_label("Stage")
            .y_label("Level")
            .step()
            .with_points()
            .theme(Theme::ocean())
            .build()
    );

    // 5. Filled area
    let y_fill: Vec<f64> = (0..40)
        .map(|i| {
            let x = i as f64 * 0.2;
            x.sin().abs() * 6.0 + 2.0
        })
        .collect();
    emit!(
        "line_filled_area",
        Charts::line(&y_fill)
            .title("Filled Area Chart")
            .x_label("Time")
            .y_label("Load")
            .filled()
            .build()
    );

    // 6. Line with all options
    let x_full = linspace(0.0, 12.0, 60);
    let y_revenue: Vec<f64> = x_full
        .iter()
        .map(|&v| 5.0 + v * 0.8 + (v * 0.5).sin() * 3.0)
        .collect();
    let y_cost: Vec<f64> = x_full
        .iter()
        .map(|&v| 3.0 + v * 0.4 + (v * 0.3).cos() * 1.5)
        .collect();
    emit!(
        "line_full_features",
        Charts::line(&y_revenue)
            .title("Revenue vs Costs — Full Features")
            .x_label("Month")
            .y_label("$M")
            .x_values(x_full)
            .add_named_series("Costs", &y_cost)
            .filled()
            .with_points()
            .line_width(3.0)
            .h_line(8.0)
            .h_line_styled(12.0, Color::from_rgba8(255, 100, 100, 200))
            .y_range(0.0, 20.0)
            .annotate(6.0, 10.0, "Crossover")
            .theme(Theme::dark())
            .build()
    );

    // 7. line_xy (non-uniform spacing)
    let x_nonuniform = vec![0.0, 1.0, 1.5, 4.0, 4.5, 7.0, 10.0];
    let y_nonuniform = vec![1.0, 3.0, 2.0, 8.0, 5.0, 9.0, 4.0];
    emit!(
        "line_xy_nonuniform",
        Charts::line_xy(&x_nonuniform, &y_nonuniform)
            .title("Line XY — Non-Uniform X Spacing")
            .x_label("Position")
            .y_label("Value")
            .with_points()
            .build()
    );

    // ═══════════════════════════════════════════════════════════════════
    // SECTION 2: BAR CHARTS
    // ═══════════════════════════════════════════════════════════════════
    println!("▸ Section 2: Bar Charts");

    let labels5 = || {
        vec![
            "Alpha".into(),
            "Beta".into(),
            "Gamma".into(),
            "Delta".into(),
            "Epsilon".into(),
        ]
    };
    let labels4 = || vec!["Q1".into(), "Q2".into(), "Q3".into(), "Q4".into()];

    // 8. Vertical bars
    emit!(
        "bar_vertical",
        Charts::bar(labels5(), &[40.0, 70.0, 55.0, 90.0, 35.0])
            .title("Vertical Bar Chart")
            .x_label("Category")
            .y_label("Score")
            .build()
    );

    // 9. Horizontal bars
    emit!(
        "bar_horizontal",
        Charts::bar(labels5(), &[40.0, 70.0, 55.0, 90.0, 35.0])
            .title("Horizontal Bar Chart")
            .x_label("Category")
            .y_label("Score")
            .horizontal()
            .theme(Theme::forest())
            .build()
    );

    // 10. Grouped bars (3 series)
    emit!(
        "bar_grouped",
        Charts::bar(
            vec!["Rust".into(), "Go".into(), "Python".into(), "C++".into()],
            &[95.0, 78.0, 42.0, 90.0]
        )
        .title("Language Comparison — Grouped")
        .y_label("Score")
        .add_named_series("Safety", &[99.0, 74.0, 60.0, 35.0])
        .add_named_series("Ergonomics", &[85.0, 70.0, 95.0, 30.0])
        .y_range(0.0, 100.0)
        .h_line(75.0)
        .build()
    );

    // 11. Stacked bars
    emit!(
        "bar_stacked",
        Charts::bar(labels4(), &[100.0, 120.0, 90.0, 150.0])
            .title("Quarterly Revenue — Stacked")
            .y_label("$K")
            .add_named_series("Services", &[60.0, 80.0, 70.0, 100.0])
            .add_named_series("Licensing", &[30.0, 40.0, 35.0, 50.0])
            .stacked()
            .theme(Theme::pastel())
            .build()
    );

    // 12. Mixed-sign bars
    emit!(
        "bar_mixed_sign",
        Charts::bar(
            vec![
                "Jan".into(),
                "Feb".into(),
                "Mar".into(),
                "Apr".into(),
                "May".into(),
                "Jun".into()
            ],
            &[25.0, -10.0, 40.0, -35.0, 15.0, -20.0]
        )
        .title("P&L — Mixed Sign")
        .y_label("$K")
        .h_line_styled(0.0, Color::from_rgba8(255, 255, 0, 200))
        .build()
    );

    // 13. Value labels + rounded bars
    emit!(
        "bar_value_labels",
        Charts::bar(labels4(), &[42.0, 67.0, 31.0, 89.0])
            .title("Sales — Value Labels + Rounded")
            .y_label("Units")
            .show_values()
            .corner_radius(6.0)
            .gap(0.35)
            .theme(Theme::ocean())
            .build()
    );

    // 14. Stacked horizontal
    emit!(
        "bar_stacked_horizontal",
        Charts::bar(labels5(), &[180.0, 210.0, 95.0, 150.0, 120.0])
            .title("Team Hours — Stacked Horizontal")
            .x_label("Hours")
            .add_named_series("Q2", &[200.0, 190.0, 110.0, 180.0, 140.0])
            .add_named_series("Q3", &[220.0, 230.0, 130.0, 170.0, 160.0])
            .stacked()
            .horizontal()
            .corner_radius(4.0)
            .gap(0.25)
            .theme(Theme::light())
            .build()
    );

    // ═══════════════════════════════════════════════════════════════════
    // SECTION 3: SCATTER CHARTS
    // ═══════════════════════════════════════════════════════════════════
    println!("▸ Section 3: Scatter Charts");

    let sx: Vec<f64> = (0..40).map(|i| i as f64 * 0.5).collect();
    let sy: Vec<f64> = sx
        .iter()
        .map(|&v| v.sqrt() * 3.0 + (v * 0.7).sin() * 2.0)
        .collect();

    // 15-19. Each marker shape
    let markers = [
        ("circle", Marker::Circle),
        ("square", Marker::Square),
        ("diamond", Marker::Diamond),
        ("cross", Marker::Cross),
        ("triangle", Marker::Triangle),
    ];
    for (name, marker) in &markers {
        emit!(
            &format!("scatter_{name}"),
            Charts::scatter(&sx, &sy)
                .title(&format!("Scatter — Marker: {}", name.to_uppercase()))
                .x_label("X")
                .y_label("Y")
                .marker(*marker)
                .build()
        );
    }

    // 20. Connected scatter
    emit!(
        "scatter_connected",
        Charts::scatter(&sx, &sy)
            .title("Scatter — Connected Points")
            .x_label("X")
            .y_label("Y")
            .connected()
            .marker(Marker::Circle)
            .build()
    );

    // 21. Multi-series scatter with named series
    let s2y: Vec<f64> = sx.iter().map(|&v| v.ln().max(0.0) * 4.0).collect();
    emit!(
        "scatter_multi_series",
        Charts::scatter(&sx, &sy)
            .title("Scatter — Multi-Series + Trend Line")
            .x_label("X")
            .y_label("Y")
            .add_named_series("log curve", &sx, &s2y)
            .connected()
            .trend_line()
            .build()
    );

    // 22. Size override
    emit!(
        "scatter_large_markers",
        Charts::scatter(&sx[..15], &sy[..15])
            .title("Scatter — Large Markers (size=8)")
            .x_label("X")
            .y_label("Y")
            .marker(Marker::Circle)
            .size(8.0)
            .theme(Theme::forest())
            .build()
    );

    // ═══════════════════════════════════════════════════════════════════
    // SECTION 4: HISTOGRAMS
    // ═══════════════════════════════════════════════════════════════════
    println!("▸ Section 4: Histograms");

    let normal_data = pseudo_normal(400, 50.0, 10.0, 42);

    // 23. Frequency histogram
    emit!(
        "histogram_frequency",
        Charts::histogram(&normal_data)
            .title("Histogram — Frequency Distribution")
            .x_label("Value")
            .y_label("Count")
            .bins(20)
            .build()
    );

    // 24. Density histogram
    emit!(
        "histogram_density",
        Charts::histogram(&normal_data)
            .title("Histogram — Density Normalized")
            .x_label("Value")
            .y_label("Density")
            .bins(25)
            .density()
            .opacity(0.8)
            .v_line(50.0)
            .theme(Theme::light())
            .build()
    );

    // 25. Overlaid histograms
    let norm2 = pseudo_normal(300, 60.0, 8.0, 99);
    emit!(
        "histogram_overlaid",
        Charts::histogram(&normal_data)
            .title("Histogram — Overlaid Distributions")
            .x_label("Value")
            .y_label("Count")
            .bins(20)
            .add_series(Series::new("Group B", norm2))
            .opacity(0.6)
            .v_line_styled(50.0, Color::from_rgba8(255, 100, 100, 200))
            .v_line_styled(60.0, Color::from_rgba8(100, 100, 255, 200))
            .build()
    );

    // 26. Bin count comparison (few bins)
    emit!(
        "histogram_few_bins",
        Charts::histogram(&normal_data)
            .title("Histogram — 8 Bins (Coarse)")
            .x_label("Value")
            .y_label("Count")
            .bins(8)
            .theme(Theme::pastel())
            .build()
    );

    // ═══════════════════════════════════════════════════════════════════
    // SECTION 5: BOX PLOTS
    // ═══════════════════════════════════════════════════════════════════
    println!("▸ Section 5: Box Plots");

    // 27. Standard box plot
    emit!(
        "boxplot_standard",
        Charts::boxplot(vec![
            ("Control", pseudo_normal(80, 10.0, 2.0, 300)),
            ("Drug A", pseudo_normal(80, 14.0, 3.0, 400)),
            ("Drug B", pseudo_normal(80, 12.0, 1.5, 500)),
            ("Drug C", pseudo_normal(80, 18.0, 4.5, 600)),
        ])
        .title("Clinical Trial — Standard Box Plot")
        .x_label("Treatment Group")
        .y_label("Response Score")
        .h_line(12.0)
        .build()
    );

    // 28. Notched box plot
    emit!(
        "boxplot_notched",
        Charts::boxplot(vec![
            ("Mon", pseudo_normal(60, 22.0, 3.0, 10)),
            ("Tue", pseudo_normal(60, 20.0, 2.5, 20)),
            ("Wed", pseudo_normal(60, 25.0, 4.0, 30)),
            ("Thu", pseudo_normal(60, 28.0, 3.5, 40)),
            ("Fri", pseudo_normal(60, 32.0, 5.0, 50)),
        ])
        .title("Daily Temperature — Notched")
        .x_label("Day")
        .y_label("°C")
        .notched()
        .theme(Theme::ocean())
        .build()
    );

    // 29. No outliers + varying sizes
    emit!(
        "boxplot_no_outliers",
        Charts::boxplot(vec![
            ("n=5", vec![2.0, 4.0, 5.0, 6.0, 8.0]),
            ("n=20", pseudo_normal(20, 6.0, 2.0, 100)),
            ("n=100", pseudo_normal(100, 5.0, 3.0, 200)),
        ])
        .title("Box Plot — No Outliers, Varying N")
        .x_label("Group Size")
        .y_label("Value")
        .no_outliers()
        .theme(Theme::pastel())
        .build()
    );

    // ═══════════════════════════════════════════════════════════════════
    // SECTION 6: HEATMAPS
    // ═══════════════════════════════════════════════════════════════════
    println!("▸ Section 6: Heatmaps");

    // 30. Basic heatmap
    let heatmap_data = vec![
        vec![1.0, 3.0, 5.0, 7.0, 9.0],
        vec![2.0, 4.0, 6.0, 8.0, 10.0],
        vec![3.0, 6.0, 9.0, 12.0, 15.0],
        vec![4.0, 8.0, 12.0, 16.0, 20.0],
        vec![5.0, 10.0, 15.0, 20.0, 25.0],
    ];
    emit!(
        "heatmap_basic",
        Charts::heatmap(heatmap_data)
            .title("Basic Heatmap — Multiplication Table")
            .values(true)
            .build()
    );

    // 31. Correlation matrix
    let corr_labels: Vec<String> = vec!["A", "B", "C", "D", "E"]
        .into_iter()
        .map(String::from)
        .collect();
    let corr_data = vec![
        vec![1.00, 0.85, -0.30, 0.60, 0.10],
        vec![0.85, 1.00, -0.20, 0.72, -0.05],
        vec![-0.30, -0.20, 1.00, -0.45, 0.80],
        vec![0.60, 0.72, -0.45, 1.00, 0.15],
        vec![0.10, -0.05, 0.80, 0.15, 1.00],
    ];
    emit!(
        "heatmap_correlation",
        Heatmap::correlation(corr_data, corr_labels)
            .title("Correlation Matrix")
            .cell_gap(3.0)
            .cell_radius(4.0)
            .build()
    );

    // 32. Custom colors
    let activity = vec![
        vec![2.0, 15.0, 30.0, 25.0, 40.0, 20.0, 5.0],
        vec![3.0, 18.0, 35.0, 28.0, 45.0, 22.0, 4.0],
        vec![1.0, 12.0, 28.0, 22.0, 38.0, 18.0, 6.0],
        vec![4.0, 20.0, 32.0, 30.0, 42.0, 25.0, 3.0],
        vec![5.0, 22.0, 38.0, 35.0, 50.0, 30.0, 8.0],
    ];
    emit!(
        "heatmap_custom_colors",
        Charts::heatmap(activity)
            .title("Website Traffic — Custom Palette")
            .row_labels(vec![
                "Mon".into(),
                "Tue".into(),
                "Wed".into(),
                "Thu".into(),
                "Fri".into()
            ])
            .col_labels(vec![
                "6am".into(),
                "9am".into(),
                "12pm".into(),
                "3pm".into(),
                "6pm".into(),
                "9pm".into(),
                "12am".into()
            ])
            .colors(
                Color::from_rgba8(10, 10, 40, 255),
                Color::from_rgba8(0, 255, 140, 255)
            )
            .values(false)
            .cell_radius(3.0)
            .cell_gap(2.0)
            .range(0.0, 55.0)
            .build()
    );

    // 33. Heatmap with NaN cells
    let nan = f64::NAN;
    emit!(
        "heatmap_nan_cells",
        Charts::heatmap(vec![
            vec![1.0, nan, 3.0, 4.0],
            vec![nan, 7.0, 8.0, nan],
            vec![11.0, 12.0, nan, 14.0],
            vec![16.0, 17.0, 18.0, nan],
        ])
        .title("Heatmap — NaN Cells (Transparent Gaps)")
        .values(true)
        .cell_gap(3.0)
        .theme(Theme::dark())
        .build()
    );

    // ═══════════════════════════════════════════════════════════════════
    // SECTION 7: PIE / DONUT CHARTS
    // ═══════════════════════════════════════════════════════════════════
    println!("▸ Section 7: Pie / Donut Charts");

    let pie_labels = || {
        vec![
            "Desktop".into(),
            "Mobile".into(),
            "Tablet".into(),
            "Other".into(),
        ]
    };
    let pie_vals = &[45.0, 35.0, 15.0, 5.0];

    // 34. Basic pie
    emit!(
        "pie_basic",
        Charts::pie(pie_labels(), pie_vals)
            .title("Traffic Sources — Pie")
            .build()
    );

    // 35. Donut chart
    emit!(
        "pie_donut",
        Charts::pie(pie_labels(), pie_vals)
            .title("Traffic Sources — Donut")
            .donut(0.5)
            .build()
    );

    // 36. Thin donut
    emit!(
        "pie_thin_donut",
        Charts::pie(pie_labels(), pie_vals)
            .title("Traffic Sources — Thin Donut (0.7)")
            .donut(0.7)
            .theme(Theme::pastel())
            .build()
    );

    // 37. No percentages + custom angle
    emit!(
        "pie_custom",
        Charts::pie(
            vec![
                "A".into(),
                "B".into(),
                "C".into(),
                "D".into(),
                "E".into(),
                "F".into()
            ],
            &[20.0, 15.0, 30.0, 10.0, 18.0, 7.0]
        )
        .title("Distribution — No Percentages, Rotated")
        .hide_percentages()
        .start_angle_degrees(45.0)
        .theme(Theme::ocean())
        .build()
    );

    // ═══════════════════════════════════════════════════════════════════
    // SECTION 8: EDGE CASES
    // ═══════════════════════════════════════════════════════════════════
    println!("▸ Section 8: Edge Cases");

    // 38. Single data point
    emit!(
        "edge_single_point",
        Charts::scatter(&[42.0], &[99.0])
            .title("Single Data Point")
            .x_label("X")
            .y_label("Y")
            .marker(Marker::Circle)
            .annotate(42.0, 99.0, "Only point")
            .build()
    );

    // 39. All-NaN data (should not crash)
    emit!(
        "edge_all_nan",
        Charts::line(&[f64::NAN, f64::NAN, f64::NAN])
            .title("All-NaN Data (Should Render Empty)")
            .build()
    );

    // 40. Tiny range (values differ by 0.001)
    let tiny: Vec<f64> = (0..15).map(|i| 1.000 + i as f64 * 0.001).collect();
    emit!(
        "edge_tiny_range",
        Charts::line(&tiny)
            .title("Tiny Range: 1.000 → 1.014")
            .x_label("Sample")
            .y_label("Voltage (V)")
            .with_points()
            .h_line_styled(1.007, Color::from_rgba8(100, 255, 100, 200))
            .build()
    );

    // 41. Huge exponential range
    let huge: Vec<f64> = (0..20).map(|i| (i as f64 * 0.3).exp() * 100.0).collect();
    emit!(
        "edge_huge_range",
        Charts::line(&huge)
            .title("Exponential Growth → 28K+")
            .x_label("Day")
            .y_label("Users")
            .filled()
            .build()
    );

    // 42. NaN/Inf-contaminated scatter
    emit!(
        "edge_nan_inf_scatter",
        Charts::scatter(
            &[1.0, 2.0, f64::NAN, 4.0, 5.0, f64::INFINITY, 7.0, 8.0],
            &[2.0, f64::NAN, 3.0, 5.0, f64::INFINITY, 4.0, 7.0, 6.0]
        )
        .title("NaN/Inf Contaminated — Should Render Finite Points")
        .marker(Marker::Diamond)
        .connected()
        .build()
    );

    // 43. Many categories bar (15 categories)
    let many_labels: Vec<String> = (1..=15).map(|i| format!("Cat{i}")).collect();
    let many_vals: Vec<f64> = (1..=15).map(|i| (i as f64 * 7.0 % 23.0) + 5.0).collect();
    emit!(
        "edge_many_categories",
        Charts::bar(many_labels, &many_vals)
            .title("15 Categories — Label Crowding Test")
            .y_label("Value")
            .build()
    );

    // 44. Identical values (flat line / zero-range)
    emit!(
        "edge_identical_values",
        Charts::line(&[5.0, 5.0, 5.0, 5.0, 5.0])
            .title("All Identical Values (y=5)")
            .x_label("X")
            .y_label("Y")
            .with_points()
            .build()
    );

    // 45. All-negative bars
    emit!(
        "edge_negative_bars",
        Charts::bar(
            vec!["A".into(), "B".into(), "C".into(), "D".into(), "E".into()],
            &[-15.0, -30.0, -22.0, -8.0, -45.0]
        )
        .title("All-Negative Bars")
        .y_label("P&L ($K)")
        .h_line_styled(0.0, Color::from_rgba8(255, 255, 0, 200))
        .build()
    );

    // ═══════════════════════════════════════════════════════════════════
    // SECTION 9: FEATURE COMBINATIONS & THEMES
    // ═══════════════════════════════════════════════════════════════════
    println!("▸ Section 9: Features & Themes");

    // 46. All annotations
    let ax = linspace(0.0, 10.0, 30);
    let ay: Vec<f64> = ax.iter().map(|&v| v.sin() * 4.0 + 5.0).collect();
    emit!(
        "feature_annotations",
        Charts::scatter(&ax, &ay)
            .title("Annotations + Reference Lines")
            .x_label("X")
            .y_label("Y")
            .connected()
            .h_line(5.0)
            .h_line_styled(8.0, Color::from_rgba8(255, 150, 150, 200))
            .v_line(5.0)
            .v_line_styled(std::f64::consts::PI, Color::from_rgba8(150, 255, 150, 200))
            .annotate(std::f64::consts::FRAC_PI_2, 9.0, "Peak")
            .annotate(std::f64::consts::PI * 1.5, 1.0, "Trough")
            .build()
    );

    // 47-51. Theme gallery (one scatter each)
    let themes: [(&str, Theme); 5] = [
        ("dark", Theme::dark()),
        ("light", Theme::light()),
        ("pastel", Theme::pastel()),
        ("ocean", Theme::ocean()),
        ("forest", Theme::forest()),
    ];
    for (name, theme) in &themes {
        emit!(
            &format!("theme_{name}"),
            Charts::scatter(&sx[..20], &sy[..20])
                .title(&format!("Theme: {}", name.to_uppercase()))
                .x_label("X")
                .y_label("Y")
                .connected()
                .marker(Marker::Circle)
                .h_line(5.0)
                .theme(theme.clone())
                .build()
        );
    }

    // 52. Trend line (linear regression)
    let trend_x: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let trend_y: Vec<f64> = trend_x
        .iter()
        .map(|&x| 2.0 * x + 5.0 + ((x as u64 * 2654435761 % 100) as f64 / 100.0 - 0.5) * 10.0)
        .collect();
    emit!(
        "feature_trend_line",
        Charts::scatter(&trend_x, &trend_y)
            .title("Scatter + Linear Regression Trend")
            .x_label("X")
            .y_label("Y")
            .marker(Marker::Circle)
            .trend_line()
            .build()
    );

    println!();
    println!("══════════════════════════════════════════════════════════");
    println!("  Generated {count} charts → {DIR}/");
    println!("  Open in any image viewer to review.");
    println!("══════════════════════════════════════════════════════════");
}

// ─────────────────────────────────────────────────────────────────────────────
// Data generators (deterministic)
// ─────────────────────────────────────────────────────────────────────────────

fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| lo + (hi - lo) * i as f64 / (n - 1).max(1) as f64)
        .collect()
}

fn pseudo_normal(n: usize, mean: f64, std: f64, seed: u64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let mut sum = 0.0;
            for k in 0..6u64 {
                let v = ((i as u64 * 2654435761 + k * 7919 + seed) % 10000) as f64 / 10000.0;
                sum += v;
            }
            mean + (sum - 3.0) * std
        })
        .collect()
}
