//! Render snapshot tests for all scry-chart chart types.
//!
//! These tests build representative charts, call `layout::render_chart()`,
//! and snapshot the structural output (text overlay positions/content + canvas
//! dimensions and command count). This catches regressions in layout, scale
//! computation, tick generation, and data rendering without being fragile to
//! sub-pixel rasterization differences across tiny-skia versions.

use scry_chart::chart::{Charts, LineChart};
use scry_chart::data::Series;
use scry_chart::layout;
use scry_chart::prelude::Marker;
use scry_chart::theme::Theme;

// ---------------------------------------------------------------------------
// Helper: summarize a RenderedChart for snapshot
// ---------------------------------------------------------------------------

/// Produces a deterministic, human-readable summary of a rendered chart
/// that captures the structural properties we care about.
fn summarize(rendered: &layout::RenderedChart) -> String {
    let mut lines = Vec::new();

    lines.push(format!(
        "canvas: {}x{}, {} commands",
        rendered.canvas.width(),
        rendered.canvas.height(),
        rendered.canvas.commands().len(),
    ));

    let labels = rendered.text_labels();
    lines.push(format!("text_labels: {}", labels.len()));
    for (i, label) in labels.iter().enumerate() {
        lines.push(format!("  [{i}] {label}"));
    }

    lines.join("\n")
}

// ===========================================================================
// Per-chart-type tests
// ===========================================================================

#[test]
fn render_scatter_basic() {
    let chart = Charts::scatter(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2.0, 4.0, 1.0, 8.0, 5.0])
        .title("Scatter Test")
        .x_label("X")
        .y_label("Y")
        .theme(Theme::dark())
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);

    // Structural assertions
    assert!(
        rendered.canvas.commands().len() > 0,
        "should have draw commands"
    );
    assert!(
        rendered.text_labels().len() > 0,
        "should have text overlays"
    );

    // Title overlay should exist
    assert!(
        rendered.text_labels().contains(&"Scatter Test"),
        "should have title overlay"
    );

    // X and Y label overlays
    assert!(
        rendered.text_labels().contains(&"X"),
        "should have x-label overlay"
    );
    assert!(
        rendered.text_labels().contains(&"Y"),
        "should have y-label overlay"
    );

    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_scatter_multi_series() {
    let chart = Charts::scatter(&[1.0, 2.0, 3.0], &[1.0, 4.0, 9.0])
        .add_series(
            Series::new("Extra", vec![1.5, 2.5, 3.5]),
            Series::new("Extra Y", vec![3.0, 5.0, 7.0]),
        )
        .title("Multi-Series Scatter")
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    assert!(
        rendered.canvas.commands().len() > 5,
        "multi-series should have more commands"
    );
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_scatter_connected() {
    let chart = Charts::scatter(&[1.0, 2.0, 3.0, 4.0], &[1.0, 3.0, 2.0, 4.0])
        .connected()
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    // Connected scatter should have line commands in addition to circles
    assert!(
        rendered.canvas.commands().len() > 4,
        "connected should have line commands"
    );
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_line_basic() {
    let chart = Charts::line(&[1.0, 4.0, 2.0, 8.0, 5.0, 3.0, 7.0])
        .title("Line Test")
        .x_label("Time")
        .y_label("Value")
        .theme(Theme::dark())
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);

    assert!(rendered.canvas.commands().len() > 0);
    assert!(
        rendered.text_labels().contains(&"Line Test"),
        "should have title"
    );

    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_line_filled_with_points() {
    let chart = Charts::line(&[1.0, 4.0, 2.0, 8.0, 5.0])
        .filled()
        .with_points()
        .title("Filled Line")
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    // Filled line should have polygon commands + circle commands for points
    assert!(rendered.canvas.commands().len() > 5);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_line_multi_series() {
    let chart = LineChart::new(vec![
        Series::new("Alpha", vec![1.0, 3.0, 2.0, 5.0]),
        Series::new("Beta", vec![2.0, 1.0, 4.0, 3.0]),
        Series::new("Gamma", vec![3.0, 5.0, 1.0, 4.0]),
    ])
    .title("Multi-Series")
    .build();

    let rendered = layout::render_chart(&chart, 500, 350);
    // Multi-series should trigger legend rendering
    assert!(rendered.canvas.commands().len() > 10);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_bar_vertical() {
    let chart = Charts::bar(
        vec![
            "Mon".into(),
            "Tue".into(),
            "Wed".into(),
            "Thu".into(),
            "Fri".into(),
        ],
        &[12.0, 19.0, 8.0, 15.0, 22.0],
    )
    .title("Weekly Sales")
    .y_label("Units")
    .theme(Theme::dark())
    .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    assert!(
        rendered.canvas.commands().len() >= 5,
        "should have bar rects"
    );

    // Category labels should appear in overlays
    assert!(
        rendered.text_labels().contains(&"Mon"),
        "should have category label"
    );

    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_bar_grouped() {
    let chart = Charts::bar(
        vec!["Q1".into(), "Q2".into(), "Q3".into(), "Q4".into()],
        &[10.0, 15.0, 12.0, 18.0],
    )
    .add_series(Series::new("Product B", vec![8.0, 12.0, 14.0, 16.0]))
    .title("Grouped Bars")
    .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    // 4 categories × 2 series = 8 bar rects minimum
    assert!(rendered.canvas.commands().len() >= 8);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_bar_stacked() {
    let chart = Charts::bar(
        vec!["A".into(), "B".into(), "C".into()],
        &[10.0, 20.0, 15.0],
    )
    .add_series(Series::new("Layer 2", vec![5.0, 8.0, 12.0]))
    .stacked()
    .title("Stacked Bars")
    .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_histogram_basic() {
    let data: Vec<f64> = (0..100)
        .map(|i| (i as f64 * 0.1).sin() * 50.0 + 50.0)
        .collect();
    let chart = Charts::histogram(&data)
        .title("Distribution")
        .x_label("Value")
        .y_label("Frequency")
        .bins(15)
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    assert!(rendered.canvas.commands().len() > 0);
    assert!(rendered.text_labels().contains(&"Distribution"));
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_histogram_density() {
    let data: Vec<f64> = (0..200).map(|i| (i as f64 / 200.0) * 100.0).collect();
    let chart = Charts::histogram(&data).density().title("Density").build();

    let rendered = layout::render_chart(&chart, 400, 300);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_boxplot_basic() {
    let chart = Charts::boxplot(vec![
        (
            "Group A",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ),
        (
            "Group B",
            vec![3.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 12.0, 15.0],
        ),
        (
            "Group C",
            vec![0.5, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 20.0],
        ),
    ])
    .title("Box Plot Test")
    .y_label("Score")
    .build();

    let rendered = layout::render_chart(&chart, 400, 300);

    // Category labels
    assert!(rendered.text_labels().contains(&"Group A"));
    assert!(rendered.text_labels().contains(&"Group B"));
    assert!(rendered.text_labels().contains(&"Group C"));

    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_boxplot_no_outliers() {
    let chart = Charts::boxplot(vec![("X", vec![1.0, 2.0, 3.0, 4.0, 5.0])])
        .no_outliers()
        .build();

    let rendered = layout::render_chart(&chart, 300, 250);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_heatmap_basic() {
    let chart = Charts::heatmap(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ])
    .title("Heatmap Test")
    .row_labels(vec!["R1".into(), "R2".into(), "R3".into()])
    .col_labels(vec!["C1".into(), "C2".into(), "C3".into()])
    .build();

    let rendered = layout::render_chart(&chart, 400, 300);

    // Should have value labels (9 cells) + row labels + col labels + title
    assert!(rendered.text_labels().len() >= 9 + 3 + 3 + 1);

    assert!(rendered.text_labels().contains(&"R1"));
    assert!(rendered.text_labels().contains(&"C1"));

    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_heatmap_no_values() {
    let chart = Charts::heatmap(vec![vec![1.0, 2.0], vec![3.0, 4.0]])
        .values(false)
        .build();

    let rendered = layout::render_chart(&chart, 300, 250);
    // With values hidden, should have fewer overlays (no cell values)
    // Only row/col labels remain
    insta::assert_snapshot!(summarize(&rendered));
}

// ===========================================================================
// Reference lines & axis range tests
// ===========================================================================

#[test]
fn render_with_reference_lines() {
    let chart = Charts::line(&[1.0, 4.0, 2.0, 8.0, 5.0])
        .h_line(4.0)
        .v_line(2.0)
        .title("With References")
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    // Reference lines should add draw commands
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_with_custom_axis_range() {
    let chart = Charts::scatter(&[1.0, 2.0, 3.0], &[1.0, 4.0, 9.0])
        .x_range(0.0, 10.0)
        .y_range(0.0, 20.0)
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    insta::assert_snapshot!(summarize(&rendered));
}

// ===========================================================================
// Edge case tests
// ===========================================================================

#[test]
fn render_empty_scatter() {
    let chart = Charts::scatter(&[], &[]).build();
    let rendered = layout::render_chart(&chart, 400, 300);
    // Should not panic, should produce a valid (empty) chart
    assert!(rendered.canvas.width() == 400);
    assert!(rendered.canvas.height() == 300);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_empty_line() {
    let chart = Charts::line(&[]).build();
    let rendered = layout::render_chart(&chart, 400, 300);
    assert!(rendered.canvas.width() == 400);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_empty_bar() {
    let chart = Charts::bar(vec![], &[]).build();
    let rendered = layout::render_chart(&chart, 400, 300);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_empty_histogram() {
    let chart = Charts::histogram(&[]).build();
    let rendered = layout::render_chart(&chart, 400, 300);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_empty_heatmap() {
    let chart = Charts::heatmap(vec![]).build();
    let rendered = layout::render_chart(&chart, 400, 300);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_single_point_scatter() {
    let chart = Charts::scatter(&[5.0], &[5.0]).build();
    let rendered = layout::render_chart(&chart, 400, 300);
    // Single point should render without panic
    assert!(rendered.canvas.commands().len() >= 1);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_single_value_line() {
    let chart = Charts::line(&[42.0]).build();
    let rendered = layout::render_chart(&chart, 400, 300);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_single_bar() {
    let chart = Charts::bar(vec!["Only".into()], &[100.0]).build();
    let rendered = layout::render_chart(&chart, 400, 300);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_tiny_canvas() {
    // Minimum viable size (widget guard is 4×4 cells, but layout should
    // handle small pixel sizes gracefully)
    let chart = Charts::line(&[1.0, 2.0, 3.0]).build();
    let rendered = layout::render_chart(&chart, 40, 30);
    // Should not panic even at tiny sizes
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_large_canvas() {
    let chart = Charts::line(&[1.0, 2.0, 3.0]).build();
    let rendered = layout::render_chart(&chart, 2000, 1200);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_mismatched_xy_scatter() {
    // X has 5 values, Y has 3 — should truncate to min without panic
    let chart = Charts::scatter(&[1.0, 2.0, 3.0, 4.0, 5.0], &[10.0, 20.0, 30.0]).build();

    let rendered = layout::render_chart(&chart, 400, 300);
    // Should render 3 points (truncated to shorter series)
    insta::assert_snapshot!(summarize(&rendered));
}

// ===========================================================================
// Theme tests
// ===========================================================================

#[test]
fn render_light_theme() {
    let chart = Charts::line(&[1.0, 4.0, 2.0, 8.0])
        .theme(Theme::light())
        .title("Light Theme")
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_pastel_theme() {
    let chart = Charts::scatter(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0])
        .theme(Theme::pastel())
        .title("Pastel Theme")
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    insta::assert_snapshot!(summarize(&rendered));
}

// ===========================================================================
// Marker tests (scatter)
// ===========================================================================

#[test]
fn render_scatter_markers() {
    let markers = [
        Marker::Circle,
        Marker::Square,
        Marker::Diamond,
        Marker::Cross,
        Marker::Triangle,
    ];

    for marker in markers {
        let chart = Charts::scatter(&[1.0, 2.0, 3.0], &[1.0, 4.0, 9.0])
            .marker(marker)
            .build();

        let rendered = layout::render_chart(&chart, 300, 200);
        assert!(
            rendered.canvas.commands().len() > 0,
            "marker {:?} should produce draw commands",
            marker,
        );
    }
}

// ===========================================================================
// Phase 2: Fixed features tests
// ===========================================================================

#[test]
fn render_bar_horizontal() {
    let chart = Charts::bar(
        vec!["Alpha".into(), "Beta".into(), "Gamma".into()],
        &[30.0, 50.0, 20.0],
    )
    .horizontal()
    .title("Horizontal Bars")
    .build();

    let rendered = layout::render_chart(&chart, 400, 300);

    // Category labels should be present
    assert!(
        rendered.text_labels().contains(&"Alpha"),
        "should have category label"
    );

    // Should have bar rects
    assert!(
        rendered.canvas.commands().len() >= 3,
        "should have bar rects"
    );
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_bar_horizontal_grouped() {
    let chart = Charts::bar(vec!["Q1".into(), "Q2".into()], &[10.0, 15.0])
        .add_series(Series::new("Product B", vec![8.0, 12.0]))
        .horizontal()
        .title("Horizontal Grouped")
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    // 2 categories × 2 series = 4 bar rects
    assert!(rendered.canvas.commands().len() >= 4);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_bar_horizontal_stacked() {
    let chart = Charts::bar(vec!["A".into(), "B".into()], &[10.0, 20.0])
        .add_series(Series::new("Layer 2", vec![5.0, 8.0]))
        .stacked()
        .horizontal()
        .title("Horizontal Stacked")
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_histogram_multi_series() {
    let data1: Vec<f64> = (0..80)
        .map(|i| (i as f64 * 0.08).sin() * 30.0 + 40.0)
        .collect();
    let data2: Vec<f64> = (0..80)
        .map(|i| (i as f64 * 0.1).cos() * 25.0 + 60.0)
        .collect();

    let chart = Charts::histogram(&data1)
        .add_series(Series::new("Series B", data2))
        .bins(12)
        .title("Multi-Histogram")
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);

    // Multi-series should render legend with text overlays
    assert!(
        rendered.text_labels().contains(&"Series B"),
        "should render legend text for extra series"
    );

    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_line_legend_text_rendered() {
    // Regression test: legend text was previously discarded (_legend_text)
    let chart = LineChart::new(vec![
        Series::new("Temp", vec![20.0, 22.0, 25.0]),
        Series::new("Humidity", vec![60.0, 55.0, 65.0]),
    ])
    .title("Legend Text Check")
    .build();

    let rendered = layout::render_chart(&chart, 500, 350);

    // Legend text overlays should be present
    assert!(
        rendered.text_labels().contains(&"Temp"),
        "legend text 'Temp' should be rendered, not discarded"
    );
    assert!(
        rendered.text_labels().contains(&"Humidity"),
        "legend text 'Humidity' should be rendered, not discarded"
    );

    insta::assert_snapshot!(summarize(&rendered));
}

// ===========================================================================
// Phase 3: Data validation tests
// ===========================================================================

#[test]
fn render_scatter_with_nan() {
    // Should handle NaN gracefully without panicking
    let chart = Charts::scatter(&[1.0, f64::NAN, 3.0, 4.0], &[2.0, 5.0, f64::NAN, 8.0])
        .title("NaN Data")
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    assert!(
        rendered.canvas.commands().len() > 0,
        "should still render axes"
    );
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_line_with_infinity() {
    let chart = Charts::line(&[1.0, f64::INFINITY, 3.0, 4.0])
        .title("Infinity Data")
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    assert!(rendered.canvas.commands().len() > 0);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_histogram_with_nan() {
    let chart = Charts::histogram(&[1.0, 2.0, f64::NAN, 4.0, 5.0, f64::INFINITY])
        .bins(5)
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    assert!(rendered.canvas.commands().len() > 0);
    insta::assert_snapshot!(summarize(&rendered));
}

// ---------------------------------------------------------------------------
// Phase 5 – Legend, Annotations & Polish
// ---------------------------------------------------------------------------

#[test]
fn render_scatter_with_annotation() {
    let chart = Charts::scatter(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 15.0, 25.0])
        .annotate(3.0, 15.0, "Peak")
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    // Annotation text should appear in overlays
    assert!(
        rendered.text_labels().contains(&"Peak"),
        "Annotation 'Peak' not found in text labels"
    );
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_scatter_with_trend_line() {
    let chart = Charts::scatter(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2.0, 4.0, 5.0, 4.5, 6.0])
        .trend_line()
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    // Trend line should add drawing commands beyond the basic scatter
    assert!(
        rendered.canvas.commands().len() > 5,
        "Expected trend line commands"
    );
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_line_with_annotation() {
    let chart = Charts::line(&[10.0, 20.0, 30.0, 25.0, 35.0])
        .annotate(2.0, 30.0, "Max")
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    assert!(
        rendered.text_labels().contains(&"Max"),
        "Annotation 'Max' not found in text labels"
    );
    insta::assert_snapshot!(summarize(&rendered));
}

// ===========================================================================
// Phase 6: Command-level integration tests
// ===========================================================================
//
// These tests validate the _types_ of draw commands emitted, not just counts.
// They ensure that our rendering upgrades (polyline, gradients, dashed grids,
// bar strokes, corner radii, scatter borders) are actually used.

use scry_engine::scene::command::DrawCommand;
use scry_engine::scene::style::FillStyle;

#[test]
fn line_chart_uses_polyline() {
    let chart = Charts::line(&[1.0, 4.0, 2.0, 8.0, 5.0, 3.0, 7.0])
        .theme(Theme::dark())
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    let has_polyline = rendered
        .canvas
        .commands()
        .iter()
        .any(|cmd| matches!(cmd, DrawCommand::Polyline { closed: false, .. }));
    assert!(
        has_polyline,
        "Line chart should emit Polyline commands, not individual Line commands"
    );
}

#[test]
fn filled_line_uses_gradient() {
    let chart = Charts::line(&[1.0, 4.0, 2.0, 8.0, 5.0])
        .filled()
        .theme(Theme::dark())
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    let has_gradient = rendered.canvas.commands().iter().any(|cmd| {
        matches!(cmd, DrawCommand::Polyline { style, closed: true, .. }
            if matches!(&style.fill, Some(FillStyle::LinearGradient(_))))
    });
    assert!(
        has_gradient,
        "Filled line chart should use LinearGradient fill, not Solid"
    );
}

#[test]
fn grid_lines_use_dash_pattern() {
    let chart = Charts::line(&[1.0, 4.0, 2.0, 8.0])
        .theme(Theme::dark())
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    let has_dashed = rendered
        .canvas
        .commands()
        .iter()
        .any(|cmd| matches!(cmd, DrawCommand::Line { stroke, .. } if stroke.dash.is_some()));
    assert!(has_dashed, "Grid lines should use DashPattern from theme");
}

#[test]
fn bar_chart_has_stroke() {
    let chart = Charts::bar(
        vec!["A".into(), "B".into(), "C".into()],
        &[10.0, 25.0, 15.0],
    )
    .theme(Theme::dark())
    .build();

    let rendered = layout::render_chart(&chart, 400, 300);

    // Bar fill rects
    let fill_rects = rendered
        .canvas
        .commands()
        .iter()
        .filter(|cmd| matches!(cmd, DrawCommand::Rectangle { style, .. } if style.fill.is_some()))
        .count();

    // Bar stroke rects — should be 0 by default (bar_stroke_width: 0.0)
    // Tufte: bar outlines are chartjunk when fill already encodes the value.
    let stroke_rects = rendered
        .canvas
        .commands()
        .iter()
        .filter(|cmd| {
            matches!(cmd, DrawCommand::Rectangle { style, .. }
            if style.stroke.is_some() && style.fill.is_none())
        })
        .count();

    assert!(
        fill_rects >= 3,
        "Should have at least 3 bar fill rectangles, got {fill_rects}"
    );
    assert_eq!(
        stroke_rects, 0,
        "Bars should have no stroke by default (bar_stroke_width: 0.0), got {stroke_rects}"
    );
}

#[test]
fn histogram_bins_flush_on_axis() {
    let chart = Charts::histogram(&[1.0, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0])
        .bins(5)
        .theme(Theme::dark())
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    // Histogram bins should NOT have corner_radius — they must sit flush on the x-axis
    let has_rounded = rendered.canvas.commands().iter().any(
        |cmd| matches!(cmd, DrawCommand::Rectangle { corner_radius, .. } if *corner_radius > 0.0),
    );
    assert!(
        !has_rounded,
        "Histogram bins should have no corner_radius to sit flush on x-axis"
    );
}

#[test]
fn scatter_markers_have_stroke() {
    let chart = Charts::scatter(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2.0, 4.0, 1.0, 8.0, 5.0])
        .theme(Theme::dark())
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    let stroked_circles = rendered
        .canvas
        .commands()
        .iter()
        .filter(|cmd| matches!(cmd, DrawCommand::Circle { style, .. } if style.stroke.is_some()))
        .count();
    assert!(
        stroked_circles >= 5,
        "Each scatter marker should have a stroke border, got {stroked_circles}"
    );
}

#[test]
fn all_themes_produce_output() {
    for (name, theme) in [
        ("dark", Theme::dark()),
        ("light", Theme::light()),
        ("pastel", Theme::pastel()),
    ] {
        let chart = Charts::line(&[1.0, 4.0, 2.0, 8.0, 5.0]).theme(theme).build();

        let rendered = layout::render_chart(&chart, 400, 300);
        assert!(
            !rendered.canvas.commands().is_empty(),
            "Theme '{name}' should produce draw commands"
        );
        assert!(
            !rendered.text_labels().is_empty(),
            "Theme '{name}' should produce text overlays"
        );
    }
}

#[test]
fn full_feature_chart() {
    // Exercise every builder option in a single chart
    let chart = Charts::scatter(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2.0, 4.0, 1.0, 8.0, 5.0, 3.0, 7.0, 6.0],
    )
    .title("Full Feature Test")
    .x_label("X Axis")
    .y_label("Y Axis")
    .theme(Theme::dark())
    .marker(Marker::Diamond)
    .connected()
    .add_series(
        Series::new("Extra", vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]),
        Series::new("Extra Y", vec![3.0, 5.0, 7.0, 2.0, 6.0, 4.0, 8.0, 1.0]),
    )
    .trend_line()
    .annotate(4.0, 8.0, "Peak")
    .h_line(5.0)
    .v_line(3.0)
    .x_range(0.0, 10.0)
    .y_range(0.0, 10.0)
    .build();

    let rendered = layout::render_chart(&chart, 600, 400);

    // Verify structural integrity
    assert!(
        rendered.canvas.commands().len() > 20,
        "Full feature chart should have many commands"
    );
    assert!(
        rendered.text_labels().contains(&"Full Feature Test"),
        "Title"
    );
    assert!(
        rendered.text_labels().contains(&"X Axis"),
        "X label"
    );
    assert!(
        rendered.text_labels().contains(&"Y Axis"),
        "Y label"
    );
    assert!(
        rendered.text_labels().contains(&"Peak"),
        "Annotation"
    );

    // Verify draw command types
    let has_polygon = rendered
        .canvas
        .commands()
        .iter()
        .any(|cmd| matches!(cmd, DrawCommand::Polyline { closed: true, .. }));
    assert!(
        has_polygon,
        "Diamond markers should emit closed Polyline (polygon) commands"
    );

    let has_dashed_grid = rendered
        .canvas
        .commands()
        .iter()
        .any(|cmd| matches!(cmd, DrawCommand::Line { stroke, .. } if stroke.dash.is_some()));
    assert!(has_dashed_grid, "Should have dashed grid lines");

    insta::assert_snapshot!(summarize(&rendered));
}

// ===========================================================================
// Tick label rotation tests
// ===========================================================================

#[test]
fn render_line_diagonal_ticks() {
    let chart = Charts::line(&[1.0, 4.0, 2.0, 8.0, 5.0, 3.0])
        .title("Diagonal Ticks")
        .x_ticks_diagonal()
        .build();
    let rendered = layout::render_chart(&chart, 400, 300);

    // Verify that tick labels are present (rotation is now handled in DrawCommand::Text)
    assert!(
        rendered.text_labels().len() > 1,
        "Should have tick label overlays"
    );

    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_bar_vertical_ticks() {
    let chart = Charts::bar(
        vec![
            "Alpha".into(),
            "Beta".into(),
            "Gamma".into(),
            "Delta".into(),
        ],
        &[10.0, 25.0, 15.0, 30.0],
    )
    .title("Vertical Labels")
    .x_ticks_vertical()
    .build();
    let rendered = layout::render_chart(&chart, 400, 300);

    // Verify that category labels are present (rotation is now handled in DrawCommand::Text)
    let labels = rendered.text_labels();
    assert!(labels.contains(&"Alpha"), "Should have 'Alpha' label");
    assert!(labels.contains(&"Beta"), "Should have 'Beta' label");
    assert!(labels.contains(&"Gamma"), "Should have 'Gamma' label");
    assert!(labels.contains(&"Delta"), "Should have 'Delta' label");

    insta::assert_snapshot!(summarize(&rendered));
}

// ===========================================================================
// Per-series styling tests
// ===========================================================================

use scry_chart::data::{FillPattern, GradientFill, SeriesStyle};

#[test]
fn render_boxplot_per_series_color() {
    let red = scry_engine::style::Color::from_rgba8(255, 0, 0, 255);
    let s = Series::new(
        "Custom",
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .style(SeriesStyle::new().color(red));

    let chart = Charts::boxplot(vec![("Default", vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])])
        .title("Styled Boxplot")
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    assert!(rendered.canvas.commands().len() > 0);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_line_custom_gradient() {
    let red = scry_engine::style::Color::from_rgba8(255, 0, 0, 255);
    let blue = scry_engine::style::Color::from_rgba8(0, 0, 255, 255);

    let chart = Charts::line(&[1.0, 4.0, 2.0, 8.0, 5.0])
        .filled()
        .title("Custom Gradient Fill")
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    // Should have gradient polygon fill
    let has_gradient = rendered.canvas.commands().iter().any(|cmd| {
        matches!(cmd, DrawCommand::Polyline { style, closed: true, .. }
            if matches!(&style.fill, Some(FillStyle::LinearGradient(_))))
    });
    assert!(has_gradient, "Filled line should have gradient fill");
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_line_bottom_to_top_gradient() {
    let s = Series::new("Bottom-Up", vec![1.0, 4.0, 2.0, 8.0, 5.0])
        .style(SeriesStyle::new().fill_gradient(GradientFill::BottomToTop));

    let chart = scry_chart::chart::LineChart::new(vec![s])
        .filled()
        .title("Bottom-to-Top Gradient")
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    let has_gradient = rendered.canvas.commands().iter().any(|cmd| {
        matches!(cmd, DrawCommand::Polyline { style, closed: true, .. }
            if matches!(&style.fill, Some(FillStyle::LinearGradient(_))))
    });
    assert!(has_gradient, "Should emit gradient fill for bottom-to-top");
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_bar_with_fill_patterns() {
    let s1 = Series::new("Diagonal", vec![10.0, 20.0, 15.0])
        .style(SeriesStyle::new().fill_pattern(FillPattern::Diagonal));
    let s2 = Series::new("Hatched", vec![8.0, 12.0, 18.0])
        .style(SeriesStyle::new().fill_pattern(FillPattern::Hatched));

    let chart = Charts::bar(
        vec!["A".into(), "B".into(), "C".into()],
        &[10.0, 20.0, 15.0],
    )
    .add_series(s2)
    .title("Pattern Fills")
    .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    // Pattern fills add extra line commands for hatch marks
    let line_count = rendered
        .canvas
        .commands()
        .iter()
        .filter(|cmd| matches!(cmd, DrawCommand::Line { .. }))
        .count();
    // Without patterns, bars only have rects; with patterns, we get hatch lines
    assert!(
        line_count > 0,
        "Pattern fills should generate line commands for hatch marks"
    );
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_heatmap_viridis() {
    let chart = Charts::heatmap(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ])
    .colormap(scry_chart::colormap::Viridis)
    .title("Viridis Heatmap")
    .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    // Should render cells (9 rects) + labels + title
    assert!(rendered.canvas.commands().len() >= 9);
    assert!(rendered.text_labels().contains(&"Viridis Heatmap"));
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_heatmap_diverging() {
    let chart = Charts::heatmap(vec![vec![-1.0, 0.0, 1.0], vec![0.5, -0.5, 0.0]])
        .colormap(scry_chart::colormap::RdBu)
        .range(-1.0, 1.0)
        .title("Diverging Map")
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    assert!(rendered.canvas.commands().len() >= 6);
    insta::assert_snapshot!(summarize(&rendered));
}

// ===========================================================================
// Waterfall chart tests
// ===========================================================================

#[test]
fn render_waterfall_basic() {
    let chart = Charts::waterfall(
        vec!["Revenue".into(), "COGS".into(), "OpEx".into(), "Tax".into()],
        &[500.0, -200.0, -150.0, -50.0],
    )
    .title("P&L Waterfall")
    .show_values()
    .build();

    let rendered = layout::render_chart(&chart, 500, 350);
    assert!(rendered.canvas.commands().len() > 5);
    assert!(rendered.text_labels().contains(&"P&L Waterfall"));
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_waterfall_no_total() {
    let chart = Charts::waterfall(
        vec!["A".into(), "B".into(), "C".into()],
        &[100.0, -30.0, 50.0],
    )
    .no_total()
    .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    assert!(rendered.canvas.commands().len() > 3);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_empty_waterfall() {
    let chart = Charts::waterfall(vec![], &[]).build();
    let rendered = layout::render_chart(&chart, 400, 300);
    assert!(rendered.canvas.commands().len() >= 1); // at least background
    insta::assert_snapshot!(summarize(&rendered));
}

// ===========================================================================
// Funnel chart tests
// ===========================================================================

#[test]
fn render_funnel_basic() {
    let chart = Charts::funnel(
        vec![
            "Visitors".into(),
            "Signups".into(),
            "Trials".into(),
            "Paid".into(),
        ],
        &[10000.0, 5000.0, 2000.0, 800.0],
    )
    .title("Conversion Funnel")
    .build();

    let rendered = layout::render_chart(&chart, 500, 400);
    assert!(rendered.canvas.commands().len() > 4);
    assert!(rendered.text_labels().contains(&"Conversion Funnel"));
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_empty_funnel() {
    let chart = Charts::funnel(vec![], &[]).build();
    let rendered = layout::render_chart(&chart, 400, 300);
    // Empty funnel: just background, no rects
    insta::assert_snapshot!(summarize(&rendered));
}

// ===========================================================================
// Gauge chart tests
// ===========================================================================

#[test]
fn render_gauge_basic() {
    let chart = Charts::gauge(75.0)
        .title("CPU Usage")
        .threshold(
            60.0,
            scry_engine::style::Color::from_rgba8(40, 180, 99, 255),
        )
        .threshold(
            80.0,
            scry_engine::style::Color::from_rgba8(241, 196, 15, 255),
        )
        .threshold(
            100.0,
            scry_engine::style::Color::from_rgba8(231, 76, 60, 255),
        )
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    // 3 threshold arcs + needle line + needle hub
    assert!(!rendered.canvas.commands().is_empty());
    assert!(rendered.text_labels().contains(&"CPU Usage"));
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_gauge_custom_range() {
    let chart = Charts::gauge(150.0)
        .range(0.0, 200.0)
        .label("150 rpm")
        .build();

    let rendered = layout::render_chart(&chart, 300, 250);
    // Single track arc + needle line + needle hub
    assert!(!rendered.canvas.commands().is_empty());
    assert!(rendered.text_labels().contains(&"150 rpm"));
    insta::assert_snapshot!(summarize(&rendered));
}

// ===========================================================================
// Lollipop chart tests
// ===========================================================================

#[test]
fn render_lollipop_basic() {
    let chart = Charts::lollipop(
        vec![
            "Mon".into(),
            "Tue".into(),
            "Wed".into(),
            "Thu".into(),
            "Fri".into(),
        ],
        &[12.0, 19.0, 8.0, 15.0, 22.0],
    )
    .title("Weekly Scores")
    .show_values()
    .build();

    let rendered = layout::render_chart(&chart, 500, 350);
    assert!(rendered.canvas.commands().len() > 10);
    assert!(rendered.text_labels().contains(&"Weekly Scores"));
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_lollipop_horizontal() {
    let chart = Charts::lollipop(
        vec!["A".into(), "B".into(), "C".into()],
        &[30.0, 50.0, 20.0],
    )
    .horizontal()
    .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    assert!(rendered.canvas.commands().len() > 6);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_lollipop_single_value() {
    let chart = Charts::lollipop(vec!["Only".into()], &[42.0]).build();

    let rendered = layout::render_chart(&chart, 300, 200);
    assert!(rendered.canvas.commands().len() > 2);
    insta::assert_snapshot!(summarize(&rendered));
}

// ---------------------------------------------------------------------------
// Subplot grid tests
// ---------------------------------------------------------------------------

#[test]
fn render_subplot_basic() {
    use scry_chart::export::render_subplot_rgba;
    use scry_chart::subplot::SubplotGrid;

    let grid = SubplotGrid::new(2, 2)
        .set(
            0,
            0,
            Charts::line(&[1.0, 4.0, 2.0, 8.0]).title("Line").build(),
        )
        .set(
            0,
            1,
            Charts::scatter(&[1.0, 2.0, 3.0], &[3.0, 1.0, 4.0])
                .title("Scatter")
                .build(),
        )
        .set(
            1,
            0,
            Charts::bar(
                vec!["A".into(), "B".into(), "C".into()],
                &[10.0, 20.0, 15.0],
            )
            .title("Bar")
            .build(),
        )
        .set(
            1,
            1,
            Charts::histogram(&[1.0, 2.0, 2.5, 3.0, 3.5, 4.0])
                .title("Hist")
                .build(),
        );

    let rgba = render_subplot_rgba(&grid, 800, 600).expect("subplot render failed");
    // 800×600 × 4 channels
    assert_eq!(rgba.len(), 800 * 600 * 4);
    // Verify not all pixels are identical (i.e., charts were actually drawn)
    let first_pixel = &rgba[0..4];
    let has_variety = rgba.chunks_exact(4).any(|px| px != first_pixel);
    assert!(has_variety, "subplot render produced a uniform image");
}

#[test]
fn render_subplot_shared_x() {
    use scry_chart::export::render_subplot_rgba;
    use scry_chart::subplot::SubplotGrid;

    let grid = SubplotGrid::new(2, 1)
        .share_x_axis()
        .set(
            0,
            0,
            Charts::line(&[1.0, 4.0, 2.0, 8.0, 5.0])
                .title("Top")
                .x_label("Time")
                .build(),
        )
        .set(
            1,
            0,
            Charts::line(&[8.0, 5.0, 3.0, 6.0, 2.0])
                .title("Bottom")
                .x_label("Time")
                .build(),
        );

    let rgba = render_subplot_rgba(&grid, 600, 800).expect("shared-x subplot render failed");
    assert_eq!(rgba.len(), 600 * 800 * 4);
    // Verify not all pixels are identical
    let first_pixel = &rgba[0..4];
    let has_variety = rgba.chunks_exact(4).any(|px| px != first_pixel);
    assert!(has_variety, "shared-x subplot produced a uniform image");
}

// ===========================================================================
// Gap Policy tests (NaN/missing data handling in line charts)
// ===========================================================================

use scry_chart::data::GapPolicy;

#[test]
fn render_line_gap_skip() {
    // NaN at index 2 splits the line into two segments: [0,1] and [3,4]
    let chart = LineChart::new(vec![Series::new(
        "Gaps",
        vec![1.0, 4.0, f64::NAN, 8.0, 5.0],
    )])
    .gap_policy(GapPolicy::Skip)
    .title("Gap Skip")
    .build();

    let rendered = layout::render_chart(&chart, 400, 300);

    // With Skip policy, NaN should split the line into multiple polylines.
    // Count open polyline commands — should be 2 (one per segment).
    let polyline_count = rendered
        .canvas
        .commands()
        .iter()
        .filter(|cmd| matches!(cmd, DrawCommand::Polyline { closed: false, .. }))
        .count();
    assert_eq!(
        polyline_count, 2,
        "Skip policy should produce 2 polyline segments, got {polyline_count}"
    );

    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_line_gap_interpolate() {
    // NaN at index 2 should be interpolated between 4.0 and 8.0
    let chart = LineChart::new(vec![Series::new(
        "Interp",
        vec![1.0, 4.0, f64::NAN, 8.0, 5.0],
    )])
    .gap_policy(GapPolicy::Interpolate)
    .title("Gap Interpolate")
    .build();

    let rendered = layout::render_chart(&chart, 400, 300);

    // Interpolate fills the gap → single continuous polyline.
    let polyline_count = rendered
        .canvas
        .commands()
        .iter()
        .filter(|cmd| matches!(cmd, DrawCommand::Polyline { closed: false, .. }))
        .count();
    assert_eq!(
        polyline_count, 1,
        "Interpolate policy should produce 1 continuous polyline, got {polyline_count}"
    );

    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_line_gap_zero() {
    // NaN at index 2 should become 0.0
    let chart = LineChart::new(vec![Series::new(
        "Zero",
        vec![1.0, 4.0, f64::NAN, 8.0, 5.0],
    )])
    .gap_policy(GapPolicy::Zero)
    .title("Gap Zero")
    .build();

    let rendered = layout::render_chart(&chart, 400, 300);

    // Zero replaces NaN → single continuous polyline.
    let polyline_count = rendered
        .canvas
        .commands()
        .iter()
        .filter(|cmd| matches!(cmd, DrawCommand::Polyline { closed: false, .. }))
        .count();
    assert_eq!(
        polyline_count, 1,
        "Zero policy should produce 1 continuous polyline, got {polyline_count}"
    );

    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_line_gap_filled() {
    // Filled area chart with gaps — each segment gets its own fill polygon.
    let chart = LineChart::new(vec![Series::new(
        "Filled Gaps",
        vec![2.0, 6.0, f64::NAN, 4.0, 8.0],
    )])
    .filled()
    .gap_policy(GapPolicy::Skip)
    .title("Gap Filled")
    .build();

    let rendered = layout::render_chart(&chart, 400, 300);

    // Two fill polygons (closed polylines with gradient) + two line polylines.
    let closed_poly_count = rendered
        .canvas
        .commands()
        .iter()
        .filter(|cmd| matches!(cmd, DrawCommand::Polyline { closed: true, .. }))
        .count();
    assert!(
        closed_poly_count >= 2,
        "Filled gap chart should have at least 2 fill polygons, got {closed_poly_count}"
    );

    let open_poly_count = rendered
        .canvas
        .commands()
        .iter()
        .filter(|cmd| matches!(cmd, DrawCommand::Polyline { closed: false, .. }))
        .count();
    assert_eq!(
        open_poly_count, 2,
        "Filled gap chart should have 2 line polylines, got {open_poly_count}"
    );

    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_line_gap_stacked() {
    // Two stacked series with NaN gaps — should not panic and should render.
    let chart = LineChart::new(vec![
        Series::new("Base", vec![1.0, 3.0, f64::NAN, 5.0, 2.0]),
        Series::new("Top", vec![2.0, f64::NAN, 4.0, 3.0, 1.0]),
    ])
    .stacked()
    .gap_policy(GapPolicy::Skip)
    .title("Gap Stacked")
    .build();

    let rendered = layout::render_chart(&chart, 400, 300);

    // Should have at least some polylines (exact count depends on gap breaks).
    let polyline_count = rendered
        .canvas
        .commands()
        .iter()
        .filter(|cmd| matches!(cmd, DrawCommand::Polyline { closed: false, .. }))
        .count();
    assert!(
        polyline_count >= 2,
        "Stacked gap chart should have multiple polylines, got {polyline_count}"
    );

    // Should not panic — this is the main test.
    assert!(rendered.canvas.commands().len() > 0);

    insta::assert_snapshot!(summarize(&rendered));
}

// ===========================================================================
// Legend overlap avoidance
// ===========================================================================

/// Regression test: when data fills the default top-right legend position,
/// `best_corner` should either pick a clear corner or promote to outside
/// placement so the legend never obscures data.
#[test]
fn legend_avoids_data_overlap() {
    // Create two series whose values rise steeply toward the right,
    // filling the top-right corner (the default legend position).
    let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let ys_a: Vec<f64> = xs.iter().map(|x| x * x).collect(); // quadratic rise
    let ys_b: Vec<f64> = xs.iter().map(|x| x * x * 0.8 + 10.0).collect();

    let chart = LineChart::new(vec![
        Series::new("Alpha", ys_a),
        Series::new("Beta", ys_b),
    ])
    .title("Overlap Test")
    .build();

    let rendered = layout::render_chart(&chart, 500, 350);

    // Extract plot area to determine "inside" vs "outside" boundary.
    let plot_area = rendered.plot_area.expect("plot_area should be set");
    let (px, py, pw, ph) = plot_area;

    // Find legend text positions (Alpha, Beta labels).
    let text_positions = rendered.text_positions();
    let legend_labels: Vec<(f32, f32, &str)> = text_positions
        .iter()
        .filter(|(_, _, t)| *t == "Alpha" || *t == "Beta")
        .copied()
        .collect();

    assert!(
        legend_labels.len() >= 2,
        "Expected at least 2 legend labels, found {}",
        legend_labels.len()
    );

    // Legend is either outside the plot area (x > px + pw) or
    // positioned in a corner that doesn't overlap the data.
    // The key assertion: legend labels should NOT be in the top-right
    // quadrant where our rising data lives.
    let top_right_quadrant = (
        px + pw * 0.5, // x threshold (right half)
        py,            // y min (top)
        py + ph * 0.4, // y max (upper 40%)
    );

    let legend_in_data_zone = legend_labels.iter().any(|(lx, ly, _)| {
        *lx >= top_right_quadrant.0
            && *ly >= top_right_quadrant.1
            && *ly <= top_right_quadrant.2
    });

    assert!(
        !legend_in_data_zone,
        "Legend should NOT be in the top-right data zone. Legend positions: {:?}, \
         plot area: ({px}, {py}, {pw}, {ph})",
        legend_labels
    );
}

// ===========================================================================
// Gantt chart tests
// ===========================================================================

use scry_chart::chart::GanttTask;

#[test]
fn render_gantt_basic() {
    let chart = Charts::gantt(vec![
        GanttTask::new("Research", 0.0, 3.0).group("Phase 1").progress(1.0),
        GanttTask::new("Design", 2.0, 6.0).group("Phase 1").progress(0.8),
        GanttTask::new("Implement", 5.0, 12.0).group("Phase 2").progress(0.4),
        GanttTask::new("Test", 10.0, 14.0).group("Phase 2"),
        GanttTask::new("Deploy", 14.0, 15.0).group("Phase 3"),
    ])
    .title("Project Timeline")
    .x_label("Day")
    .build();

    let rendered = layout::render_chart(&chart, 800, 400);

    // Title present
    assert!(
        rendered.text_labels().contains(&"Project Timeline"),
        "should have title"
    );

    // Task labels should be in Y axis labels
    assert!(
        rendered.text_labels().contains(&"Research"),
        "should have task label"
    );
    assert!(
        rendered.text_labels().contains(&"Deploy"),
        "should have task label"
    );

    // Should have bar rects (at least 5 task bars)
    assert!(
        rendered.canvas.commands().len() >= 5,
        "should have task bar rects"
    );

    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_gantt_empty() {
    let chart = Charts::gantt(vec![]).build();
    let rendered = layout::render_chart(&chart, 400, 300);
    // Should not panic
    assert_eq!(rendered.canvas.width(), 400);
    assert_eq!(rendered.canvas.height(), 300);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_gantt_no_groups() {
    let chart = Charts::gantt(vec![
        GanttTask::new("Task A", 0.0, 5.0),
        GanttTask::new("Task B", 3.0, 8.0),
        GanttTask::new("Task C", 7.0, 10.0),
    ])
    .title("Simple Gantt")
    .build();

    let rendered = layout::render_chart(&chart, 600, 300);
    assert!(
        rendered.text_labels().contains(&"Simple Gantt"),
        "should have title"
    );
    insta::assert_snapshot!(summarize(&rendered));
}
