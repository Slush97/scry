//! Interactive TUI visual audit — browse all 52 charts with arrow keys.
//!
//! Navigate: ← → or h/l     Jump: 1-9     Quit: q
//!
//! ```bash
//! cargo run -p scry-chart --example visual_audit_tui
//! ```

#![allow(
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::cast_possible_truncation
)]

use std::io::stdout;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use scry_chart::prelude::*;
use scry_engine::style::Color;

// ─────────────────────────────────────────────────────────────────────────────
// Page catalogue
// ─────────────────────────────────────────────────────────────────────────────

const NUM_PAGES: usize = 52;

struct PageInfo {
    section: &'static str,
    title: &'static str,
    features: &'static str,
}

const PAGES: [PageInfo; NUM_PAGES] = [
    // Section 1: Line Charts (0-6)
    PageInfo {
        section: "Line",
        title: "Basic Line Chart",
        features: "line()",
    },
    PageInfo {
        section: "Line",
        title: "Multi-Series Line",
        features: "add_named_series · x_values",
    },
    PageInfo {
        section: "Line",
        title: "Smooth Line (Catmull-Rom)",
        features: "smooth · with_points · pastel",
    },
    PageInfo {
        section: "Line",
        title: "Step Line (Stairstep)",
        features: "step · with_points · ocean",
    },
    PageInfo {
        section: "Line",
        title: "Filled Area Chart",
        features: "filled",
    },
    PageInfo {
        section: "Line",
        title: "Revenue vs Costs — Full Features",
        features: "filled · points · line_width · h_line · annotate · y_range",
    },
    PageInfo {
        section: "Line",
        title: "Line XY — Non-Uniform Spacing",
        features: "line_xy · with_points",
    },
    // Section 2: Bar Charts (7-13)
    PageInfo {
        section: "Bar",
        title: "Vertical Bar Chart",
        features: "bar()",
    },
    PageInfo {
        section: "Bar",
        title: "Horizontal Bar Chart",
        features: "horizontal · forest",
    },
    PageInfo {
        section: "Bar",
        title: "Grouped Bars (3 series)",
        features: "add_named_series · y_range · h_line",
    },
    PageInfo {
        section: "Bar",
        title: "Stacked Bars",
        features: "stacked · pastel",
    },
    PageInfo {
        section: "Bar",
        title: "Mixed-Sign Bars",
        features: "h_line_styled · negative values",
    },
    PageInfo {
        section: "Bar",
        title: "Value Labels + Rounded",
        features: "show_values · corner_radius · gap · ocean",
    },
    PageInfo {
        section: "Bar",
        title: "Stacked Horizontal",
        features: "stacked · horizontal · corner_radius · gap · light",
    },
    // Section 3: Scatter Charts (14-21)
    PageInfo {
        section: "Scatter",
        title: "Scatter — Circle Marker",
        features: "Marker::Circle",
    },
    PageInfo {
        section: "Scatter",
        title: "Scatter — Square Marker",
        features: "Marker::Square",
    },
    PageInfo {
        section: "Scatter",
        title: "Scatter — Diamond Marker",
        features: "Marker::Diamond",
    },
    PageInfo {
        section: "Scatter",
        title: "Scatter — Cross Marker",
        features: "Marker::Cross",
    },
    PageInfo {
        section: "Scatter",
        title: "Scatter — Triangle Marker",
        features: "Marker::Triangle",
    },
    PageInfo {
        section: "Scatter",
        title: "Scatter — Connected Points",
        features: "connected · Circle",
    },
    PageInfo {
        section: "Scatter",
        title: "Scatter — Multi-Series + Trend",
        features: "add_named_series · connected · trend_line",
    },
    PageInfo {
        section: "Scatter",
        title: "Scatter — Large Markers",
        features: "size(8) · forest",
    },
    // Section 4: Histograms (22-25)
    PageInfo {
        section: "Hist",
        title: "Histogram — Frequency",
        features: "bins(20)",
    },
    PageInfo {
        section: "Hist",
        title: "Histogram — Density",
        features: "density · opacity · v_line · light",
    },
    PageInfo {
        section: "Hist",
        title: "Histogram — Overlaid",
        features: "add_series · opacity · v_line_styled",
    },
    PageInfo {
        section: "Hist",
        title: "Histogram — 8 Bins (Coarse)",
        features: "bins(8) · pastel",
    },
    // Section 5: Box Plots (26-28)
    PageInfo {
        section: "Box",
        title: "Clinical Trial — Standard",
        features: "boxplot · h_line",
    },
    PageInfo {
        section: "Box",
        title: "Temperature — Notched",
        features: "notched · ocean",
    },
    PageInfo {
        section: "Box",
        title: "No Outliers, Varying N",
        features: "no_outliers · pastel",
    },
    // Section 6: Heatmaps (29-32)
    PageInfo {
        section: "Heat",
        title: "Basic Heatmap",
        features: "values(true)",
    },
    PageInfo {
        section: "Heat",
        title: "Correlation Matrix",
        features: "correlation · cell_gap · cell_radius",
    },
    PageInfo {
        section: "Heat",
        title: "Custom Colors — Traffic",
        features: "colors · range · values(false) · labels",
    },
    PageInfo {
        section: "Heat",
        title: "NaN Cells (Transparent Gaps)",
        features: "NaN handling · cell_gap · dark",
    },
    // Section 7: Pie/Donut (33-36)
    PageInfo {
        section: "Pie",
        title: "Pie — Basic",
        features: "pie()",
    },
    PageInfo {
        section: "Pie",
        title: "Donut (0.5)",
        features: "donut(0.5)",
    },
    PageInfo {
        section: "Pie",
        title: "Thin Donut (0.7)",
        features: "donut(0.7) · pastel",
    },
    PageInfo {
        section: "Pie",
        title: "No Percentages, Rotated",
        features: "hide_percentages · start_angle · ocean",
    },
    // Section 8: Edge Cases (37-44)
    PageInfo {
        section: "Edge",
        title: "Single Data Point",
        features: "degenerate · annotate",
    },
    PageInfo {
        section: "Edge",
        title: "All-NaN Data",
        features: "should render empty",
    },
    PageInfo {
        section: "Edge",
        title: "Tiny Range: 1.000 → 1.014",
        features: "axis tick precision",
    },
    PageInfo {
        section: "Edge",
        title: "Exponential Growth → 28K+",
        features: "huge range · filled",
    },
    PageInfo {
        section: "Edge",
        title: "NaN/Inf Contaminated",
        features: "finite points only",
    },
    PageInfo {
        section: "Edge",
        title: "15 Categories — Label Crowding",
        features: "many categories",
    },
    PageInfo {
        section: "Edge",
        title: "All Identical Values (y=5)",
        features: "zero range",
    },
    PageInfo {
        section: "Edge",
        title: "All-Negative Bars",
        features: "negative · h_line at 0",
    },
    // Section 9: Features & Themes (45-51)
    PageInfo {
        section: "Feat",
        title: "Annotations + Reference Lines",
        features: "annotate · h_line · v_line · styled",
    },
    PageInfo {
        section: "Theme",
        title: "Theme: DARK",
        features: "Theme::dark()",
    },
    PageInfo {
        section: "Theme",
        title: "Theme: LIGHT",
        features: "Theme::light()",
    },
    PageInfo {
        section: "Theme",
        title: "Theme: PASTEL",
        features: "Theme::pastel()",
    },
    PageInfo {
        section: "Theme",
        title: "Theme: OCEAN",
        features: "Theme::ocean()",
    },
    PageInfo {
        section: "Theme",
        title: "Theme: FOREST",
        features: "Theme::forest()",
    },
    PageInfo {
        section: "Feat",
        title: "Scatter + Trend Line",
        features: "trend_line · linear regression",
    },
];

// ─────────────────────────────────────────────────────────────────────────────
// Chart builders — each index corresponds to a page
// ─────────────────────────────────────────────────────────────────────────────

fn build_chart(idx: usize) -> Chart {
    match idx {
        // ── Section 1: Line Charts ──────────────────────────────────
        0 => {
            let y: Vec<f64> = (0..30)
                .map(|i| (i as f64 * 0.3).sin() * 5.0 + 10.0)
                .collect();
            Charts::line(&y)
                .title("Basic Line Chart")
                .x_label("Sample Index")
                .y_label("Value")
                .build()
        }
        1 => {
            let x = linspace(0.0, 10.0, 50);
            let y1: Vec<f64> = x.iter().map(|&v| v.sin() * 3.0).collect();
            let y2: Vec<f64> = x.iter().map(|&v| v.cos() * 3.0).collect();
            let y3: Vec<f64> = x.iter().map(|&v| (v * 1.5).sin() * 2.0 + 1.0).collect();
            Charts::line(&y1)
                .title("Multi-Series Line")
                .x_label("Time")
                .y_label("Amplitude")
                .x_values(x)
                .add_named_series("cos(x)", &y2)
                .add_named_series("sin(1.5x)+1", &y3)
                .build()
        }
        2 => {
            let y: Vec<f64> = (0..15)
                .map(|i| {
                    let x = i as f64;
                    (x * 0.5).sin() * 4.0 + (x * 0.2).cos() * 2.0
                })
                .collect();
            Charts::line(&y)
                .title("Smooth Line (Catmull-Rom Spline)")
                .x_label("X")
                .y_label("Y")
                .smooth()
                .with_points()
                .theme(Theme::pastel())
                .build()
        }
        3 => Charts::line(&[2.0, 2.0, 5.0, 5.0, 3.0, 8.0, 8.0, 4.0, 4.0, 6.0])
            .title("Step Line (Stairstep)")
            .x_label("Stage")
            .y_label("Level")
            .step()
            .with_points()
            .theme(Theme::ocean())
            .build(),
        4 => {
            let y: Vec<f64> = (0..40)
                .map(|i| (i as f64 * 0.2).sin().abs() * 6.0 + 2.0)
                .collect();
            Charts::line(&y)
                .title("Filled Area Chart")
                .x_label("Time")
                .y_label("Load")
                .filled()
                .build()
        }
        5 => {
            let x = linspace(0.0, 12.0, 60);
            let rev: Vec<f64> = x
                .iter()
                .map(|&v| 5.0 + v * 0.8 + (v * 0.5).sin() * 3.0)
                .collect();
            let cost: Vec<f64> = x
                .iter()
                .map(|&v| 3.0 + v * 0.4 + (v * 0.3).cos() * 1.5)
                .collect();
            Charts::line(&rev)
                .title("Revenue vs Costs — Full Features")
                .x_label("Month")
                .y_label("$M")
                .x_values(x)
                .add_named_series("Costs", &cost)
                .filled()
                .with_points()
                .line_width(3.0)
                .h_line(8.0)
                .h_line_styled(12.0, Color::from_rgba8(255, 100, 100, 200))
                .y_range(0.0, 20.0)
                .annotate(6.0, 10.0, "Crossover")
                .theme(Theme::dark())
                .build()
        }
        6 => Charts::line_xy(
            &[0.0, 1.0, 1.5, 4.0, 4.5, 7.0, 10.0],
            &[1.0, 3.0, 2.0, 8.0, 5.0, 9.0, 4.0],
        )
        .title("Line XY — Non-Uniform X Spacing")
        .x_label("Position")
        .y_label("Value")
        .with_points()
        .build(),

        // ── Section 2: Bar Charts ───────────────────────────────────
        7 => Charts::bar(labels5(), &[40.0, 70.0, 55.0, 90.0, 35.0])
            .title("Vertical Bar Chart")
            .x_label("Category")
            .y_label("Score")
            .build(),
        8 => Charts::bar(labels5(), &[40.0, 70.0, 55.0, 90.0, 35.0])
            .title("Horizontal Bar Chart")
            .x_label("Category")
            .y_label("Score")
            .horizontal()
            .theme(Theme::forest())
            .build(),
        9 => Charts::bar(
            vec!["Rust".into(), "Go".into(), "Python".into(), "C++".into()],
            &[95.0, 78.0, 42.0, 90.0],
        )
        .title("Language Comparison — Grouped")
        .y_label("Score")
        .add_named_series("Safety", &[99.0, 74.0, 60.0, 35.0])
        .add_named_series("Ergonomics", &[85.0, 70.0, 95.0, 30.0])
        .y_range(0.0, 100.0)
        .h_line(75.0)
        .build(),
        10 => Charts::bar(labels4(), &[100.0, 120.0, 90.0, 150.0])
            .title("Quarterly Revenue — Stacked")
            .y_label("$K")
            .add_named_series("Services", &[60.0, 80.0, 70.0, 100.0])
            .add_named_series("Licensing", &[30.0, 40.0, 35.0, 50.0])
            .stacked()
            .theme(Theme::pastel())
            .build(),
        11 => Charts::bar(
            vec![
                "Jan".into(),
                "Feb".into(),
                "Mar".into(),
                "Apr".into(),
                "May".into(),
                "Jun".into(),
            ],
            &[25.0, -10.0, 40.0, -35.0, 15.0, -20.0],
        )
        .title("P&L — Mixed Sign")
        .y_label("$K")
        .h_line_styled(0.0, Color::from_rgba8(255, 255, 0, 200))
        .build(),
        12 => Charts::bar(labels4(), &[42.0, 67.0, 31.0, 89.0])
            .title("Sales — Value Labels + Rounded")
            .y_label("Units")
            .show_values()
            .corner_radius(6.0)
            .gap(0.35)
            .theme(Theme::ocean())
            .build(),
        13 => Charts::bar(labels5(), &[180.0, 210.0, 95.0, 150.0, 120.0])
            .title("Team Hours — Stacked Horizontal")
            .x_label("Hours")
            .add_named_series("Q2", &[200.0, 190.0, 110.0, 180.0, 140.0])
            .add_named_series("Q3", &[220.0, 230.0, 130.0, 170.0, 160.0])
            .stacked()
            .horizontal()
            .corner_radius(4.0)
            .gap(0.25)
            .theme(Theme::light())
            .build(),

        // ── Section 3: Scatter Charts ───────────────────────────────
        14..=18 => {
            let (sx, sy) = scatter_data();
            let markers = [
                Marker::Circle,
                Marker::Square,
                Marker::Diamond,
                Marker::Cross,
                Marker::Triangle,
            ];
            let names = ["CIRCLE", "SQUARE", "DIAMOND", "CROSS", "TRIANGLE"];
            let mi = idx - 14;
            Charts::scatter(&sx, &sy)
                .title(&format!("Scatter — Marker: {}", names[mi]))
                .x_label("X")
                .y_label("Y")
                .marker(markers[mi])
                .build()
        }
        19 => {
            let (sx, sy) = scatter_data();
            Charts::scatter(&sx, &sy)
                .title("Scatter — Connected Points")
                .x_label("X")
                .y_label("Y")
                .connected()
                .marker(Marker::Circle)
                .build()
        }
        20 => {
            let (sx, sy) = scatter_data();
            let s2y: Vec<f64> = sx.iter().map(|&v| v.ln().max(0.0) * 4.0).collect();
            Charts::scatter(&sx, &sy)
                .title("Scatter — Multi-Series + Trend Line")
                .x_label("X")
                .y_label("Y")
                .add_named_series("log curve", &sx, &s2y)
                .connected()
                .trend_line()
                .build()
        }
        21 => {
            let (sx, sy) = scatter_data();
            Charts::scatter(&sx[..15], &sy[..15])
                .title("Scatter — Large Markers (size=8)")
                .x_label("X")
                .y_label("Y")
                .marker(Marker::Circle)
                .size(8.0)
                .theme(Theme::forest())
                .build()
        }

        // ── Section 4: Histograms ───────────────────────────────────
        22 => Charts::histogram(&pseudo_normal(400, 50.0, 10.0, 42))
            .title("Histogram — Frequency Distribution")
            .x_label("Value")
            .y_label("Count")
            .bins(20)
            .build(),
        23 => Charts::histogram(&pseudo_normal(400, 50.0, 10.0, 42))
            .title("Histogram — Density Normalized")
            .x_label("Value")
            .y_label("Density")
            .bins(25)
            .density()
            .opacity(0.8)
            .v_line(50.0)
            .theme(Theme::light())
            .build(),
        24 => {
            let d1 = pseudo_normal(400, 50.0, 10.0, 42);
            let d2 = pseudo_normal(300, 60.0, 8.0, 99);
            Charts::histogram(&d1)
                .title("Histogram — Overlaid Distributions")
                .x_label("Value")
                .y_label("Count")
                .bins(20)
                .add_series(Series::new("Group B", d2))
                .opacity(0.6)
                .v_line_styled(50.0, Color::from_rgba8(255, 100, 100, 200))
                .v_line_styled(60.0, Color::from_rgba8(100, 100, 255, 200))
                .build()
        }
        25 => Charts::histogram(&pseudo_normal(400, 50.0, 10.0, 42))
            .title("Histogram — 8 Bins (Coarse)")
            .x_label("Value")
            .y_label("Count")
            .bins(8)
            .theme(Theme::pastel())
            .build(),

        // ── Section 5: Box Plots ────────────────────────────────────
        26 => Charts::boxplot(vec![
            ("Control", pseudo_normal(80, 10.0, 2.0, 300)),
            ("Drug A", pseudo_normal(80, 14.0, 3.0, 400)),
            ("Drug B", pseudo_normal(80, 12.0, 1.5, 500)),
            ("Drug C", pseudo_normal(80, 18.0, 4.5, 600)),
        ])
        .title("Clinical Trial — Standard Box Plot")
        .x_label("Treatment Group")
        .y_label("Response Score")
        .h_line(12.0)
        .build(),
        27 => Charts::boxplot(vec![
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
        .build(),
        28 => Charts::boxplot(vec![
            ("n=5", vec![2.0, 4.0, 5.0, 6.0, 8.0]),
            ("n=20", pseudo_normal(20, 6.0, 2.0, 100)),
            ("n=100", pseudo_normal(100, 5.0, 3.0, 200)),
        ])
        .title("Box Plot — No Outliers, Varying N")
        .x_label("Group Size")
        .y_label("Value")
        .no_outliers()
        .theme(Theme::pastel())
        .build(),

        // ── Section 6: Heatmaps ─────────────────────────────────────
        29 => Charts::heatmap(vec![
            vec![1.0, 3.0, 5.0, 7.0, 9.0],
            vec![2.0, 4.0, 6.0, 8.0, 10.0],
            vec![3.0, 6.0, 9.0, 12.0, 15.0],
            vec![4.0, 8.0, 12.0, 16.0, 20.0],
            vec![5.0, 10.0, 15.0, 20.0, 25.0],
        ])
        .title("Basic Heatmap — Multiplication Table")
        .values(true)
        .build(),
        30 => {
            let labels: Vec<String> = vec!["A", "B", "C", "D", "E"]
                .into_iter()
                .map(String::from)
                .collect();
            Heatmap::correlation(
                vec![
                    vec![1.00, 0.85, -0.30, 0.60, 0.10],
                    vec![0.85, 1.00, -0.20, 0.72, -0.05],
                    vec![-0.30, -0.20, 1.00, -0.45, 0.80],
                    vec![0.60, 0.72, -0.45, 1.00, 0.15],
                    vec![0.10, -0.05, 0.80, 0.15, 1.00],
                ],
                labels,
            )
            .title("Correlation Matrix")
            .cell_gap(3.0)
            .cell_radius(4.0)
            .build()
        }
        31 => Charts::heatmap(vec![
            vec![2.0, 15.0, 30.0, 25.0, 40.0, 20.0, 5.0],
            vec![3.0, 18.0, 35.0, 28.0, 45.0, 22.0, 4.0],
            vec![1.0, 12.0, 28.0, 22.0, 38.0, 18.0, 6.0],
            vec![4.0, 20.0, 32.0, 30.0, 42.0, 25.0, 3.0],
            vec![5.0, 22.0, 38.0, 35.0, 50.0, 30.0, 8.0],
        ])
        .title("Website Traffic — Custom Palette")
        .row_labels(vec![
            "Mon".into(),
            "Tue".into(),
            "Wed".into(),
            "Thu".into(),
            "Fri".into(),
        ])
        .col_labels(vec![
            "6am".into(),
            "9am".into(),
            "12pm".into(),
            "3pm".into(),
            "6pm".into(),
            "9pm".into(),
            "12am".into(),
        ])
        .colors(
            Color::from_rgba8(10, 10, 40, 255),
            Color::from_rgba8(0, 255, 140, 255),
        )
        .values(false)
        .cell_radius(3.0)
        .cell_gap(2.0)
        .range(0.0, 55.0)
        .build(),
        32 => {
            let nan = f64::NAN;
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
        }

        // ── Section 7: Pie/Donut ────────────────────────────────────
        33 => Charts::pie(pie_labels(), &[45.0, 35.0, 15.0, 5.0])
            .title("Traffic Sources — Pie")
            .build(),
        34 => Charts::pie(pie_labels(), &[45.0, 35.0, 15.0, 5.0])
            .title("Traffic Sources — Donut")
            .donut(0.5)
            .build(),
        35 => Charts::pie(pie_labels(), &[45.0, 35.0, 15.0, 5.0])
            .title("Traffic Sources — Thin Donut (0.7)")
            .donut(0.7)
            .theme(Theme::pastel())
            .build(),
        36 => Charts::pie(
            vec![
                "A".into(),
                "B".into(),
                "C".into(),
                "D".into(),
                "E".into(),
                "F".into(),
            ],
            &[20.0, 15.0, 30.0, 10.0, 18.0, 7.0],
        )
        .title("Distribution — No Percentages, Rotated")
        .hide_percentages()
        .start_angle_degrees(45.0)
        .theme(Theme::ocean())
        .build(),

        // ── Section 8: Edge Cases ───────────────────────────────────
        37 => Charts::scatter(&[42.0], &[99.0])
            .title("Single Data Point")
            .x_label("X")
            .y_label("Y")
            .marker(Marker::Circle)
            .annotate(42.0, 99.0, "Only point")
            .build(),
        38 => Charts::line(&[f64::NAN, f64::NAN, f64::NAN])
            .title("All-NaN Data (Should Render Empty)")
            .build(),
        39 => {
            let tiny: Vec<f64> = (0..15).map(|i| 1.000 + i as f64 * 0.001).collect();
            Charts::line(&tiny)
                .title("Tiny Range: 1.000 → 1.014")
                .x_label("Sample")
                .y_label("Voltage (V)")
                .with_points()
                .h_line_styled(1.007, Color::from_rgba8(100, 255, 100, 200))
                .build()
        }
        40 => {
            let huge: Vec<f64> = (0..20).map(|i| (i as f64 * 0.3).exp() * 100.0).collect();
            Charts::line(&huge)
                .title("Exponential Growth → 28K+")
                .x_label("Day")
                .y_label("Users")
                .filled()
                .build()
        }
        41 => Charts::scatter(
            &[1.0, 2.0, f64::NAN, 4.0, 5.0, f64::INFINITY, 7.0, 8.0],
            &[2.0, f64::NAN, 3.0, 5.0, f64::INFINITY, 4.0, 7.0, 6.0],
        )
        .title("NaN/Inf Contaminated — Should Render Finite Points")
        .marker(Marker::Diamond)
        .connected()
        .build(),
        42 => {
            let labels: Vec<String> = (1..=15).map(|i| format!("Cat{i}")).collect();
            let vals: Vec<f64> = (1..=15).map(|i| (i as f64 * 7.0 % 23.0) + 5.0).collect();
            Charts::bar(labels, &vals)
                .title("15 Categories — Label Crowding Test")
                .y_label("Value")
                .build()
        }
        43 => Charts::line(&[5.0, 5.0, 5.0, 5.0, 5.0])
            .title("All Identical Values (y=5)")
            .x_label("X")
            .y_label("Y")
            .with_points()
            .build(),
        44 => Charts::bar(labels5(), &[-15.0, -30.0, -22.0, -8.0, -45.0])
            .title("All-Negative Bars")
            .y_label("P&L ($K)")
            .h_line_styled(0.0, Color::from_rgba8(255, 255, 0, 200))
            .build(),

        // ── Section 9: Features & Themes ────────────────────────────
        45 => {
            let ax = linspace(0.0, 10.0, 30);
            let ay: Vec<f64> = ax.iter().map(|&v| v.sin() * 4.0 + 5.0).collect();
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
        }
        46..=50 => {
            let (sx, sy) = scatter_data();
            let themes = [
                Theme::dark(),
                Theme::light(),
                Theme::pastel(),
                Theme::ocean(),
                Theme::forest(),
            ];
            let names = ["DARK", "LIGHT", "PASTEL", "OCEAN", "FOREST"];
            let ti = idx - 46;
            Charts::scatter(&sx[..20], &sy[..20])
                .title(&format!("Theme: {}", names[ti]))
                .x_label("X")
                .y_label("Y")
                .connected()
                .marker(Marker::Circle)
                .h_line(5.0)
                .theme(themes[ti].clone())
                .build()
        }
        51 => {
            let tx: Vec<f64> = (0..30).map(|i| i as f64).collect();
            let ty: Vec<f64> = tx
                .iter()
                .map(|&x| {
                    2.0 * x + 5.0 + ((x as u64 * 2654435761 % 100) as f64 / 100.0 - 0.5) * 10.0
                })
                .collect();
            Charts::scatter(&tx, &ty)
                .title("Scatter + Linear Regression Trend")
                .x_label("X")
                .y_label("Y")
                .marker(Marker::Circle)
                .trend_line()
                .build()
        }

        _ => unreachable!(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Data helpers
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
                sum += ((i as u64 * 2654435761 + k * 7919 + seed) % 10000) as f64 / 10000.0;
            }
            mean + (sum - 3.0) * std
        })
        .collect()
}

fn scatter_data() -> (Vec<f64>, Vec<f64>) {
    let sx: Vec<f64> = (0..40).map(|i| i as f64 * 0.5).collect();
    let sy: Vec<f64> = sx
        .iter()
        .map(|&v| v.sqrt() * 3.0 + (v * 0.7).sin() * 2.0)
        .collect();
    (sx, sy)
}

fn labels5() -> Vec<String> {
    vec![
        "Alpha".into(),
        "Beta".into(),
        "Gamma".into(),
        "Delta".into(),
        "Epsilon".into(),
    ]
}

fn labels4() -> Vec<String> {
    vec!["Q1".into(), "Q2".into(), "Q3".into(), "Q4".into()]
}

fn pie_labels() -> Vec<String> {
    vec![
        "Desktop".into(),
        "Mobile".into(),
        "Tablet".into(),
        "Other".into(),
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// Main TUI loop
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let mut state = ChartState::auto();
    let mut page: usize = 0;
    let mut prev_page: usize = usize::MAX;

    loop {
        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(2)])
                .split(frame.area());

            let chart = build_chart(page);
            frame.render_stateful_widget(ChartWidget::new(&chart), chunks[0], &mut state);

            let info = &PAGES[page];
            let status = Paragraph::new(format!(
                " {:>2}/{NUM_PAGES} │ [{}] {} │ {} │ ← → navigate · q quit",
                page + 1,
                info.section,
                info.title,
                info.features,
            ))
            .block(Block::default().borders(Borders::TOP))
            .style(Style::default().fg(ratatui::style::Color::DarkGray));
            frame.render_widget(status, chunks[1]);
        })?;

        // Clean up Kitty images on page change
        if page != prev_page {
            state.cleanup();
            prev_page = page;
        }
        state.flush()?;

        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Right | KeyCode::Char('l') => {
                        page = (page + 1) % NUM_PAGES;
                    }
                    KeyCode::Left | KeyCode::Char('h') => {
                        page = (page + NUM_PAGES - 1) % NUM_PAGES;
                    }
                    KeyCode::Home => page = 0,
                    KeyCode::End => page = NUM_PAGES - 1,
                    // Jump by section with number keys
                    KeyCode::Char('1') => page = 0,  // Line
                    KeyCode::Char('2') => page = 7,  // Bar
                    KeyCode::Char('3') => page = 14, // Scatter
                    KeyCode::Char('4') => page = 22, // Histogram
                    KeyCode::Char('5') => page = 26, // BoxPlot
                    KeyCode::Char('6') => page = 29, // Heatmap
                    KeyCode::Char('7') => page = 33, // Pie
                    KeyCode::Char('8') => page = 37, // Edge
                    KeyCode::Char('9') => page = 45, // Features
                    _ => {}
                }
            }
        }
    }

    state.cleanup();
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}
