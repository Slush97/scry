//! Render snapshot tests for all pixelchart chart types.
//!
//! These tests build representative charts, call `layout::render_chart()`,
//! and snapshot the structural output (text overlay positions/content + canvas
//! dimensions and command count). This catches regressions in layout, scale
//! computation, tick generation, and data rendering without being fragile to
//! sub-pixel rasterization differences across tiny-skia versions.

use pixelchart::chart::{Chart, LineChart};
use pixelchart::data::Series;
use pixelchart::layout;
use pixelchart::prelude::Marker;
use pixelchart::theme::Theme;

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

    lines.push(format!("overlays: {}", rendered.text_overlays.len()));

    for (i, overlay) in rendered.text_overlays.iter().enumerate() {
        lines.push(format!(
            "  [{i}] ({:.0}, {:.0}) {:?} \"{}\"",
            overlay.x_px, overlay.y_px, overlay.align, overlay.text,
        ));
    }

    lines.join("\n")
}

// ===========================================================================
// Per-chart-type tests
// ===========================================================================

#[test]
fn render_scatter_basic() {
    let chart = Chart::scatter(
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        &[2.0, 4.0, 1.0, 8.0, 5.0],
    )
    .title("Scatter Test")
    .x_label("X")
    .y_label("Y")
    .theme(Theme::dark())
    .build();

    let rendered = layout::render_chart(&chart, 400, 300);

    // Structural assertions
    assert!(rendered.canvas.commands().len() > 0, "should have draw commands");
    assert!(rendered.text_overlays.len() > 0, "should have text overlays");

    // Title overlay should exist
    assert!(
        rendered.text_overlays.iter().any(|o| o.text == "Scatter Test"),
        "should have title overlay"
    );

    // X and Y label overlays
    assert!(
        rendered.text_overlays.iter().any(|o| o.text == "X"),
        "should have x-label overlay"
    );
    assert!(
        rendered.text_overlays.iter().any(|o| o.text == "Y"),
        "should have y-label overlay"
    );

    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_scatter_multi_series() {
    let chart = Chart::scatter(
        &[1.0, 2.0, 3.0],
        &[1.0, 4.0, 9.0],
    )
    .add_series(
        Series::new("Extra", vec![1.5, 2.5, 3.5]),
        Series::new("Extra Y", vec![3.0, 5.0, 7.0]),
    )
    .title("Multi-Series Scatter")
    .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    assert!(rendered.canvas.commands().len() > 5, "multi-series should have more commands");
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_scatter_connected() {
    let chart = Chart::scatter(
        &[1.0, 2.0, 3.0, 4.0],
        &[1.0, 3.0, 2.0, 4.0],
    )
    .connected()
    .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    // Connected scatter should have line commands in addition to circles
    assert!(rendered.canvas.commands().len() > 4, "connected should have line commands");
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_line_basic() {
    let chart = Chart::line(&[1.0, 4.0, 2.0, 8.0, 5.0, 3.0, 7.0])
        .title("Line Test")
        .x_label("Time")
        .y_label("Value")
        .theme(Theme::dark())
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);

    assert!(rendered.canvas.commands().len() > 0);
    assert!(
        rendered.text_overlays.iter().any(|o| o.text == "Line Test"),
        "should have title"
    );

    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_line_filled_with_points() {
    let chart = Chart::line(&[1.0, 4.0, 2.0, 8.0, 5.0])
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
    let chart = Chart::bar(
        vec!["Mon".into(), "Tue".into(), "Wed".into(), "Thu".into(), "Fri".into()],
        &[12.0, 19.0, 8.0, 15.0, 22.0],
    )
    .title("Weekly Sales")
    .y_label("Units")
    .theme(Theme::dark())
    .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    assert!(rendered.canvas.commands().len() >= 5, "should have bar rects");

    // Category labels should appear in overlays
    assert!(
        rendered.text_overlays.iter().any(|o| o.text == "Mon"),
        "should have category label"
    );

    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_bar_grouped() {
    let chart = Chart::bar(
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
    let chart = Chart::bar(
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
    let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() * 50.0 + 50.0).collect();
    let chart = Chart::histogram(&data)
        .title("Distribution")
        .x_label("Value")
        .y_label("Frequency")
        .bins(15)
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    assert!(rendered.canvas.commands().len() > 0);
    assert!(
        rendered.text_overlays.iter().any(|o| o.text == "Distribution"),
    );
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_histogram_density() {
    let data: Vec<f64> = (0..200).map(|i| (i as f64 / 200.0) * 100.0).collect();
    let chart = Chart::histogram(&data)
        .density()
        .title("Density")
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_boxplot_basic() {
    let chart = Chart::boxplot(vec![
        ("Group A", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        ("Group B", vec![3.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 12.0, 15.0]),
        ("Group C", vec![0.5, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 20.0]),
    ])
    .title("Box Plot Test")
    .y_label("Score")
    .build();

    let rendered = layout::render_chart(&chart, 400, 300);

    // Category labels
    assert!(rendered.text_overlays.iter().any(|o| o.text == "Group A"));
    assert!(rendered.text_overlays.iter().any(|o| o.text == "Group B"));
    assert!(rendered.text_overlays.iter().any(|o| o.text == "Group C"));

    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_boxplot_no_outliers() {
    let chart = Chart::boxplot(vec![
        ("X", vec![1.0, 2.0, 3.0, 4.0, 5.0]),
    ])
    .no_outliers()
    .build();

    let rendered = layout::render_chart(&chart, 300, 250);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_heatmap_basic() {
    let chart = Chart::heatmap(vec![
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
    assert!(rendered.text_overlays.len() >= 9 + 3 + 3 + 1);

    assert!(rendered.text_overlays.iter().any(|o| o.text == "R1"));
    assert!(rendered.text_overlays.iter().any(|o| o.text == "C1"));

    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_heatmap_no_values() {
    let chart = Chart::heatmap(vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
    ])
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
    let chart = Chart::line(&[1.0, 4.0, 2.0, 8.0, 5.0])
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
    let chart = Chart::scatter(
        &[1.0, 2.0, 3.0],
        &[1.0, 4.0, 9.0],
    )
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
    let chart = Chart::scatter(&[], &[]).build();
    let rendered = layout::render_chart(&chart, 400, 300);
    // Should not panic, should produce a valid (empty) chart
    assert!(rendered.canvas.width() == 400);
    assert!(rendered.canvas.height() == 300);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_empty_line() {
    let chart = Chart::line(&[]).build();
    let rendered = layout::render_chart(&chart, 400, 300);
    assert!(rendered.canvas.width() == 400);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_empty_bar() {
    let chart = Chart::bar(vec![], &[]).build();
    let rendered = layout::render_chart(&chart, 400, 300);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_empty_histogram() {
    let chart = Chart::histogram(&[]).build();
    let rendered = layout::render_chart(&chart, 400, 300);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_empty_heatmap() {
    let chart = Chart::heatmap(vec![]).build();
    let rendered = layout::render_chart(&chart, 400, 300);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_single_point_scatter() {
    let chart = Chart::scatter(&[5.0], &[5.0]).build();
    let rendered = layout::render_chart(&chart, 400, 300);
    // Single point should render without panic
    assert!(rendered.canvas.commands().len() >= 1);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_single_value_line() {
    let chart = Chart::line(&[42.0]).build();
    let rendered = layout::render_chart(&chart, 400, 300);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_single_bar() {
    let chart = Chart::bar(vec!["Only".into()], &[100.0]).build();
    let rendered = layout::render_chart(&chart, 400, 300);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_tiny_canvas() {
    // Minimum viable size (widget guard is 4×4 cells, but layout should
    // handle small pixel sizes gracefully)
    let chart = Chart::line(&[1.0, 2.0, 3.0]).build();
    let rendered = layout::render_chart(&chart, 40, 30);
    // Should not panic even at tiny sizes
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_large_canvas() {
    let chart = Chart::line(&[1.0, 2.0, 3.0]).build();
    let rendered = layout::render_chart(&chart, 2000, 1200);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_mismatched_xy_scatter() {
    // X has 5 values, Y has 3 — should truncate to min without panic
    let chart = Chart::scatter(
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        &[10.0, 20.0, 30.0],
    ).build();

    let rendered = layout::render_chart(&chart, 400, 300);
    // Should render 3 points (truncated to shorter series)
    insta::assert_snapshot!(summarize(&rendered));
}

// ===========================================================================
// Theme tests
// ===========================================================================

#[test]
fn render_light_theme() {
    let chart = Chart::line(&[1.0, 4.0, 2.0, 8.0])
        .theme(Theme::light())
        .title("Light Theme")
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_pastel_theme() {
    let chart = Chart::scatter(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0])
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
        let chart = Chart::scatter(&[1.0, 2.0, 3.0], &[1.0, 4.0, 9.0])
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
    let chart = Chart::bar(
        vec!["Alpha".into(), "Beta".into(), "Gamma".into()],
        &[30.0, 50.0, 20.0],
    )
    .horizontal()
    .title("Horizontal Bars")
    .build();

    let rendered = layout::render_chart(&chart, 400, 300);

    // Category labels should be on the left (Right-aligned)
    let alpha_overlay = rendered.text_overlays.iter().find(|o| o.text == "Alpha");
    assert!(alpha_overlay.is_some(), "should have category label");
    assert_eq!(
        alpha_overlay.unwrap().align,
        layout::TextAlign::Right,
        "horizontal bar category labels should be right-aligned"
    );

    // Should have bar rects
    assert!(rendered.canvas.commands().len() >= 3, "should have bar rects");
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_bar_horizontal_grouped() {
    let chart = Chart::bar(
        vec!["Q1".into(), "Q2".into()],
        &[10.0, 15.0],
    )
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
    let chart = Chart::bar(
        vec!["A".into(), "B".into()],
        &[10.0, 20.0],
    )
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
    let data1: Vec<f64> = (0..80).map(|i| (i as f64 * 0.08).sin() * 30.0 + 40.0).collect();
    let data2: Vec<f64> = (0..80).map(|i| (i as f64 * 0.1).cos() * 25.0 + 60.0).collect();

    let chart = Chart::histogram(&data1)
        .add_series(Series::new("Series B", data2))
        .bins(12)
        .title("Multi-Histogram")
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);

    // Multi-series should render legend with text overlays
    let has_legend_text = rendered.text_overlays.iter().any(|o| o.text == "Series B");
    assert!(has_legend_text, "should render legend text for extra series");

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
    let has_temp = rendered.text_overlays.iter().any(|o| o.text == "Temp");
    let has_humidity = rendered.text_overlays.iter().any(|o| o.text == "Humidity");
    assert!(has_temp, "legend text 'Temp' should be rendered, not discarded");
    assert!(has_humidity, "legend text 'Humidity' should be rendered, not discarded");

    insta::assert_snapshot!(summarize(&rendered));
}

// ===========================================================================
// Phase 3: Data validation tests
// ===========================================================================

#[test]
fn render_scatter_with_nan() {
    // Should handle NaN gracefully without panicking
    let chart = Chart::scatter(
        &[1.0, f64::NAN, 3.0, 4.0],
        &[2.0, 5.0, f64::NAN, 8.0],
    )
    .title("NaN Data")
    .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    assert!(rendered.canvas.commands().len() > 0, "should still render axes");
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_line_with_infinity() {
    let chart = Chart::line(&[1.0, f64::INFINITY, 3.0, 4.0])
        .title("Infinity Data")
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    assert!(rendered.canvas.commands().len() > 0);
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_histogram_with_nan() {
    let chart = Chart::histogram(&[1.0, 2.0, f64::NAN, 4.0, 5.0, f64::INFINITY])
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
    let chart = Chart::scatter(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 15.0, 25.0])
        .annotate(3.0, 15.0, "Peak")
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    // Annotation text should appear in overlays
    let has_annotation = rendered.text_overlays.iter().any(|o| o.text == "Peak");
    assert!(has_annotation, "Annotation 'Peak' not found in overlays");
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_scatter_with_trend_line() {
    let chart = Chart::scatter(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2.0, 4.0, 5.0, 4.5, 6.0])
        .trend_line()
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    // Trend line should add drawing commands beyond the basic scatter
    assert!(rendered.canvas.commands().len() > 5, "Expected trend line commands");
    insta::assert_snapshot!(summarize(&rendered));
}

#[test]
fn render_line_with_annotation() {
    let chart = Chart::line(&[10.0, 20.0, 30.0, 25.0, 35.0])
        .annotate(2.0, 30.0, "Max")
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    let has_annotation = rendered.text_overlays.iter().any(|o| o.text == "Max");
    assert!(has_annotation, "Annotation 'Max' not found in overlays");
    insta::assert_snapshot!(summarize(&rendered));
}
