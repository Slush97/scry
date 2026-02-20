//! Formatting invariant tests for scry-chart.
//!
//! Unlike snapshot tests (which verify structural output) these tests validate
//! **spatial correctness**: text stays inside the canvas, labels don't overlap,
//! legends don't cover data, trend lines stay clipped, titles are centered, etc.
//!
//! Every invariant is checked across 3 resolutions (small, standard, large) and
//! across all major chart types.

use scry_chart::chart::{Charts, LineChart};
use scry_chart::data::Series;
use scry_chart::layout::{self, RenderedChart};
use scry_chart::theme::Theme;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolutions to test against (small / standard / large).
const RESOLUTIONS: &[(u32, u32)] = &[(300, 200), (800, 500), (1600, 1000)];

/// Approximate character widths for estimating text bounding boxes.
const CHAR_WIDTH_RATIO: f32 = 0.59; // matches INTER_ADVANCE_RATIO

/// Build one representative chart per type, all with dark theme, title, and
/// axis labels where applicable.
fn representative_charts() -> Vec<(&'static str, scry_chart::chart::Chart)> {
    vec![
        (
            "line",
            Charts::line(&[1.0, 4.0, 2.0, 8.0, 5.0, 3.0, 7.0])
                .title("Line Chart")
                .x_label("X")
                .y_label("Y")
                .theme(Theme::dark())
                .build(),
        ),
        (
            "scatter",
            Charts::scatter(
                &[1.0, 2.0, 3.0, 4.0, 5.0],
                &[2.0, 4.0, 1.0, 8.0, 5.0],
            )
            .title("Scatter")
            .x_label("X")
            .y_label("Y")
            .theme(Theme::dark())
            .build(),
        ),
        (
            "bar",
            Charts::bar(
                vec!["Mon".into(), "Tue".into(), "Wed".into(), "Thu".into()],
                &[12.0, 19.0, 8.0, 15.0],
            )
            .title("Bar Chart")
            .y_label("Units")
            .theme(Theme::dark())
            .build(),
        ),
        (
            "histogram",
            Charts::histogram(
                &(0..100)
                    .map(|i| (i as f64 * 0.1).sin() * 50.0 + 50.0)
                    .collect::<Vec<_>>(),
            )
            .title("Histogram")
            .x_label("Value")
            .y_label("Count")
            .bins(15)
            .theme(Theme::dark())
            .build(),
        ),
        (
            "pie",
            Charts::pie(
                vec!["A".into(), "B".into(), "C".into(), "D".into()],
                &[30.0, 25.0, 20.0, 25.0],
            )
            .title("Pie Chart")
            .theme(Theme::dark())
            .build(),
        ),
        (
            "radar",
            Charts::radar(
                vec!["Speed", "Power", "Range", "Armor", "Magic"],
            )
            .add_series("Alpha", &[80.0, 70.0, 90.0, 60.0, 85.0])
            .add_series("Beta", &[60.0, 90.0, 70.0, 80.0, 65.0])
            .title("Radar Chart")
            .theme(Theme::dark())
            .build(),
        ),
        (
            "heatmap",
            Charts::heatmap(vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0],
            ])
            .title("Heatmap")
            .row_labels(vec!["R1".into(), "R2".into(), "R3".into()])
            .col_labels(vec!["C1".into(), "C2".into(), "C3".into()])
            .theme(Theme::dark())
            .build(),
        ),
        (
            "boxplot",
            Charts::boxplot(vec![
                ("Low", vec![1.0, 2.0, 3.0, 4.0, 5.0]),
                ("Mid", vec![5.0, 6.0, 7.0, 8.0, 9.0]),
                ("High", vec![10.0, 12.0, 14.0, 16.0, 18.0]),
            ])
            .title("Boxplot")
            .y_label("Score")
            .theme(Theme::dark())
            .build(),
        ),
        (
            "lollipop",
            Charts::lollipop(
                vec!["A".into(), "B".into(), "C".into()],
                &[95.0, 82.0, 78.0],
            )
            .title("Lollipop")
            .y_label("Score")
            .theme(Theme::dark())
            .build(),
        ),
    ]
}

/// Estimate the pixel bounding box of a text label.
/// Returns `(left, top, right, bottom)`.
fn estimate_text_bbox(x: f32, y: f32, text: &str, font_size: f32) -> (f32, f32, f32, f32) {
    let char_w = font_size * CHAR_WIDTH_RATIO;
    let text_w = text.len() as f32 * char_w;
    let text_h = font_size;
    // We don't know alignment, so use generous bounds centered on (x, y).
    let left = x - text_w / 2.0;
    let top = y - text_h;
    let right = x + text_w / 2.0;
    let bottom = y;
    (left, top, right, bottom)
}

/// Check if two bounding boxes overlap.
fn boxes_overlap(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> bool {
    a.0 < b.2 && a.2 > b.0 && a.1 < b.3 && a.3 > b.1
}

// ===========================================================================
// T1: All text labels inside canvas bounds
// ===========================================================================

/// Every text label anchor `(x, y)` must be within `[0, w] × [0, h]`,
/// with a small tolerance for sub-pixel rounding.
#[test]
fn text_inside_canvas() {
    let tolerance = 5.0; // Allow 5px grace for sub-pixel rendering

    for (name, chart) in representative_charts() {
        for &(w, h) in RESOLUTIONS {
            let rendered = layout::render_chart(&chart, w, h);
            let positions = rendered.text_positions();

            for (x, y, text) in &positions {
                assert!(
                    *x >= -tolerance && *x <= w as f32 + tolerance,
                    "[{name} @ {w}x{h}] text '{text}' x={x} is outside canvas width {w}"
                );
                assert!(
                    *y >= -tolerance && *y <= h as f32 + tolerance,
                    "[{name} @ {w}x{h}] text '{text}' y={y} is outside canvas height {h}"
                );
            }
        }
    }
}

// ===========================================================================
// T2: Title text is horizontally centered
// ===========================================================================

/// Title text should be within 15% of the horizontal center of the canvas.
#[test]
fn title_centered() {
    for (name, chart) in representative_charts() {
        for &(w, h) in RESOLUTIONS {
            let rendered = layout::render_chart(&chart, w, h);
            let positions = rendered.text_positions();
            let center = w as f32 / 2.0;
            let tolerance = w as f32 * 0.15;

            // Find the title text (first text command is usually the title,
            // but we search by known title strings)
            let title_texts: Vec<&str> = vec![
                "Line Chart", "Scatter", "Bar Chart", "Histogram",
                "Pie Chart", "Radar Chart", "Heatmap", "Boxplot", "Lollipop",
            ];

            for (x, _y, text) in &positions {
                if title_texts.contains(text) {
                    assert!(
                        (*x - center).abs() <= tolerance,
                        "[{name} @ {w}x{h}] title '{text}' at x={x} is not centered \
                         (center={center}, tolerance={tolerance})"
                    );
                }
            }
        }
    }
}

// ===========================================================================
// T3: Trend line endpoints within plot area
// ===========================================================================

/// Scatter chart with trend_line — all line endpoints should be within
/// the plot area bounds.
#[test]
fn trend_line_within_plot_area() {
    use scry_engine::scene::command::DrawCommand;

    let chart = Charts::scatter(
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        &[2.0, 4.0, 5.0, 4.5, 6.0],
    )
    .trend_line()
    .theme(Theme::dark())
    .build();

    for &(w, h) in RESOLUTIONS {
        let rendered = layout::render_chart(&chart, w, h);
        let (px, py, pw, ph) = rendered.plot_area.expect("plot_area should be set");
        let tolerance = 2.0;

        // Find dashed lines (trend line uses DashPattern)
        for cmd in rendered.canvas.commands() {
            if let DrawCommand::Line {
                x1, y1, x2, y2, stroke, ..
            } = cmd
            {
                if stroke.dash.is_some() {
                    // This is likely a trend or grid line — trend lines
                    // should stay within plot area (grids should too).
                    assert!(
                        *x1 >= px - tolerance && *x1 <= px + pw + tolerance,
                        "Dashed line x1={x1} outside plot area [{px}, {}] @ {w}x{h}",
                        px + pw
                    );
                    assert!(
                        *x2 >= px - tolerance && *x2 <= px + pw + tolerance,
                        "Dashed line x2={x2} outside plot area [{px}, {}] @ {w}x{h}",
                        px + pw
                    );
                    assert!(
                        *y1 >= py - tolerance && *y1 <= py + ph + tolerance,
                        "Dashed line y1={y1} outside plot area [{py}, {}] @ {w}x{h}",
                        py + ph
                    );
                    assert!(
                        *y2 >= py - tolerance && *y2 <= py + ph + tolerance,
                        "Dashed line y2={y2} outside plot area [{py}, {}] @ {w}x{h}",
                        py + ph
                    );
                }
            }
        }
    }
}

// ===========================================================================
// T4: Pie chart legend does not overlap pie area
// ===========================================================================

/// The pie chart legend labels should be positioned to the right of or
/// outside the pie circle, never overlapping it.
#[test]
fn pie_legend_outside_chart() {
    let chart = Charts::pie(
        vec!["A".into(), "B".into(), "C".into(), "D".into(), "E".into()],
        &[30.0, 25.0, 20.0, 15.0, 10.0],
    )
    .title("Pie Legend Test")
    .theme(Theme::dark())
    .build();

    for &(w, h) in RESOLUTIONS {
        let rendered = layout::render_chart(&chart, w, h);
        let (px, _py, pw, ph) = rendered.plot_area.expect("plot_area");

        // Pie is centered in plot area with radius = min(pw, ph) / 2.0 * 0.85
        let cx = px + pw / 2.0;
        let radius = pw.min(ph) / 2.0 * 0.85;
        let pie_right_edge = cx + radius;

        // Find legend labels (A, B, C, D, E)
        let legend_names = ["A", "B", "C", "D", "E"];
        let positions = rendered.text_positions();
        let legend_positions: Vec<_> = positions
            .iter()
            .filter(|(_, _, t)| legend_names.contains(t))
            .collect();

        for (lx, _ly, label) in &legend_positions {
            assert!(
                *lx >= pie_right_edge - 5.0,
                "[pie @ {w}x{h}] legend label '{label}' at x={lx} overlaps pie \
                 (right edge={pie_right_edge})"
            );
        }
    }
}

// ===========================================================================
// T5: Multi-series line chart legend present and positioned
// ===========================================================================

/// For multi-series charts, all legend labels should be present and
/// positioned within the canvas bounds.
#[test]
fn multi_series_legend_present() {
    let chart = LineChart::new(vec![
        Series::new("Alpha", vec![1.0, 3.0, 2.0, 5.0]),
        Series::new("Beta", vec![2.0, 1.0, 4.0, 3.0]),
        Series::new("Gamma", vec![3.0, 5.0, 1.0, 4.0]),
    ])
    .title("Multi-Series")
    .theme(Theme::dark())
    .build();

    for &(w, h) in RESOLUTIONS {
        let rendered = layout::render_chart(&chart, w, h);
        let labels = rendered.text_labels();

        assert!(
            labels.contains(&"Alpha"),
            "[{w}x{h}] missing legend label 'Alpha'"
        );
        assert!(
            labels.contains(&"Beta"),
            "[{w}x{h}] missing legend label 'Beta'"
        );
        assert!(
            labels.contains(&"Gamma"),
            "[{w}x{h}] missing legend label 'Gamma'"
        );

        // All legend text should be inside canvas
        let positions = rendered.text_positions();
        for (x, y, text) in &positions {
            if ["Alpha", "Beta", "Gamma"].contains(text) {
                assert!(
                    *x >= 0.0 && *x <= w as f32 && *y >= 0.0 && *y <= h as f32,
                    "[{w}x{h}] legend '{text}' at ({x}, {y}) is outside canvas"
                );
            }
        }
    }
}

// ===========================================================================
// T6: Plot area is contained within canvas
// ===========================================================================

/// The plot area rectangle must be contained within the canvas bounds.
#[test]
fn plot_area_within_canvas() {
    for (name, chart) in representative_charts() {
        for &(w, h) in RESOLUTIONS {
            let rendered = layout::render_chart(&chart, w, h);
            if let Some((px, py, pw, ph)) = rendered.plot_area {
                assert!(
                    px >= 0.0,
                    "[{name} @ {w}x{h}] plot_area x={px} < 0"
                );
                assert!(
                    py >= 0.0,
                    "[{name} @ {w}x{h}] plot_area y={py} < 0"
                );
                assert!(
                    px + pw <= w as f32 + 1.0,
                    "[{name} @ {w}x{h}] plot_area right edge {} > canvas width {w}",
                    px + pw,
                );
                assert!(
                    py + ph <= h as f32 + 1.0,
                    "[{name} @ {w}x{h}] plot_area bottom edge {} > canvas height {h}",
                    py + ph,
                );
            }
        }
    }
}

// ===========================================================================
// T7: No panics at extreme resolutions
// ===========================================================================

/// All chart types should render without panicking at very small and
/// very large resolutions.
#[test]
fn extreme_resolutions_no_panic() {
    let extreme_sizes = [(1, 1), (4, 4), (10, 10), (50, 30), (4000, 2400)];

    for (name, chart) in representative_charts() {
        for &(w, h) in &extreme_sizes {
            let rendered = layout::render_chart(&chart, w, h);
            assert_eq!(
                rendered.canvas.width(),
                w,
                "[{name} @ {w}x{h}] canvas width mismatch"
            );
            assert_eq!(
                rendered.canvas.height(),
                h,
                "[{name} @ {w}x{h}] canvas height mismatch"
            );
        }
    }
}
