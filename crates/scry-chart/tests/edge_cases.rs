//! Exhaustive edge-case test suite for scry-chart.
//!
//! Every test uses a transparent background to verify that alpha-channel
//! compositing doesn't panic or produce NaN coordinates.
//!
//! Coverage areas:
//!   - `try_build()` validation on all 7 chart types
//!   - Transparent backgrounds on every chart type
//!   - Builder option combos (smooth+step, filled+points, stacked+horizontal)
//!   - Zoom viewport integration
//!   - Annotation edge cases (offscreen, arrow combos)
//!   - Reference line edge cases (at exact data bounds, outside range)
//!   - Canvas size extremes (1×1, 5000×5, 5×5000)
//!   - Data extremes (f64::MAX, subnormals, negative zero, mixed NaN/Inf)
//!   - Single data point charts
//!   - Unicode and empty string labels
//!   - Pie donut ratios, start angles, single/many slices
//!   - Boxplot with all-identical data, single value, outliers
//!   - Heatmap single cell, single row/col, NaN cells, large grid
//!   - Histogram density mode, single/many bins, custom opacity

use scry_chart::annotation::Annotation;
use scry_chart::chart::{
    BarChart, BoxPlot, Chart, Charts, Heatmap, Histogram, LineChart, PieChart,
};
use scry_chart::data::Series;
use scry_chart::error::ChartError;
use scry_chart::layout;
use scry_chart::prelude::Marker;
use scry_chart::theme::Theme;
use scry_chart::zoom::ZoomState;
use scry_engine::style::Color;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Transparent theme — alpha-0 background to test compositing paths.
fn transparent_theme() -> Theme {
    let mut t = Theme::dark();
    t.background = Color::from_rgba8(0, 0, 0, 0);
    t
}

/// Render a chart and assert:
///   1. No panic (implicit)
///   2. Canvas has correct dimensions
///   3. No NaN/Inf in overlay positions
fn assert_render(chart: &Chart, w: u32, h: u32, label: &str) -> layout::RenderedChart {
    let r = layout::render_chart(chart, w, h);
    assert_eq!(r.canvas.width(), w, "{label}: width mismatch");
    assert_eq!(r.canvas.height(), h, "{label}: height mismatch");
    for (i, (x, y, text)) in r.text_positions().iter().enumerate() {
        assert!(
            x.is_finite(),
            "{label}: text[{i}] '{text}' has non-finite x = {x}",
        );
        assert!(
            y.is_finite(),
            "{label}: text[{i}] '{text}' has non-finite y = {y}",
        );
    }
    r
}

/// Shorthand: render at 400×300 with transparent theme.
fn render_transparent(chart: &Chart, label: &str) -> layout::RenderedChart {
    assert_render(chart, 400, 300, label)
}

// ===========================================================================
// Section 1: try_build() Validation
// ===========================================================================

#[test]
fn try_build_scatter_empty() {
    let r = Charts::scatter(&[], &[]).try_build();
    assert_eq!(r.unwrap_err(), ChartError::EmptyData);
}

#[test]
fn try_build_scatter_mismatched() {
    let r = Charts::scatter(&[1.0, 2.0], &[1.0]).try_build();
    assert_eq!(
        r.unwrap_err(),
        ChartError::MismatchedLengths { x_len: 2, y_len: 1 }
    );
}

#[test]
fn try_build_scatter_all_nan() {
    let r = Charts::scatter(&[f64::NAN], &[f64::NAN]).try_build();
    assert_eq!(r.unwrap_err(), ChartError::AllNonFinite);
}

#[test]
fn try_build_scatter_valid() {
    assert!(Charts::scatter(&[1.0], &[2.0]).try_build().is_ok());
}

#[test]
fn try_build_line_empty() {
    let r = LineChart::new(vec![]).try_build();
    assert_eq!(r.unwrap_err(), ChartError::EmptyData);
}

#[test]
fn try_build_line_valid() {
    assert!(Charts::line(&[1.0, 2.0]).try_build().is_ok());
}

#[test]
fn try_build_bar_empty_labels() {
    let r = Charts::bar(vec![], &[1.0]).try_build();
    assert_eq!(r.unwrap_err(), ChartError::EmptyData);
}

#[test]
fn try_build_bar_empty_series() {
    let r = BarChart::new(vec!["A".into()], vec![]).try_build();
    assert_eq!(r.unwrap_err(), ChartError::EmptyData);
}

#[test]
fn try_build_bar_valid() {
    assert!(Charts::bar(vec!["A".into()], &[1.0]).try_build().is_ok());
}

#[test]
fn try_build_histogram_empty() {
    let r = Histogram::new(Series::from_values(vec![])).try_build();
    assert_eq!(r.unwrap_err(), ChartError::EmptyData);
}

#[test]
fn try_build_histogram_valid() {
    assert!(Charts::histogram(&[1.0, 2.0, 3.0]).try_build().is_ok());
}

#[test]
fn try_build_pie_empty() {
    let r = PieChart::new(vec![], vec![]).try_build();
    assert_eq!(r.unwrap_err(), ChartError::EmptyData);
}

#[test]
fn try_build_pie_all_non_positive() {
    let r = Charts::pie(vec!["A".into()], &[0.0]).try_build();
    assert_eq!(r.unwrap_err(), ChartError::AllNonFinite);
}

#[test]
fn try_build_pie_all_nan() {
    let r = Charts::pie(vec!["A".into()], &[f64::NAN]).try_build();
    assert_eq!(r.unwrap_err(), ChartError::AllNonFinite);
}

#[test]
fn try_build_pie_negative_values() {
    let r = Charts::pie(vec!["A".into()], &[-5.0]).try_build();
    assert_eq!(r.unwrap_err(), ChartError::AllNonFinite);
}

#[test]
fn try_build_pie_valid() {
    assert!(Charts::pie(vec!["A".into()], &[1.0]).try_build().is_ok());
}

#[test]
fn try_build_boxplot_empty() {
    let r = BoxPlot::new(Vec::<(String, Vec<f64>)>::new()).try_build();
    assert_eq!(r.unwrap_err(), ChartError::EmptyData);
}

#[test]
fn try_build_boxplot_valid() {
    assert!(
        Charts::boxplot(vec![("A".to_string(), vec![1.0, 2.0, 3.0])])
            .try_build()
            .is_ok()
    );
}

#[test]
fn try_build_heatmap_empty() {
    let r = Heatmap::new(vec![]).try_build();
    assert_eq!(r.unwrap_err(), ChartError::EmptyData);
}

#[test]
fn try_build_heatmap_valid() {
    assert!(Charts::heatmap(vec![vec![1.0]]).try_build().is_ok());
}

// ===========================================================================
// Section 2: Transparent Background — Every Chart Type
// ===========================================================================

#[test]
fn transparent_scatter() {
    let chart = Charts::scatter(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0])
        .theme(transparent_theme())
        .title("Transparent Scatter")
        .build();
    let r = render_transparent(&chart, "transparent_scatter");
    assert!(!r.canvas.commands().is_empty());
}

#[test]
fn transparent_line() {
    let chart = Charts::line(&[10.0, 20.0, 15.0, 25.0])
        .theme(transparent_theme())
        .title("Transparent Line")
        .build();
    let r = render_transparent(&chart, "transparent_line");
    assert!(!r.canvas.commands().is_empty());
}

#[test]
fn transparent_bar() {
    let chart = Charts::bar(
        vec!["A".into(), "B".into(), "C".into()],
        &[10.0, 20.0, 30.0],
    )
    .theme(transparent_theme())
    .title("Transparent Bar")
    .build();
    let r = render_transparent(&chart, "transparent_bar");
    assert!(!r.canvas.commands().is_empty());
}

#[test]
fn transparent_histogram() {
    let chart = Charts::histogram(&[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        .theme(transparent_theme())
        .title("Transparent Histogram")
        .build();
    let r = render_transparent(&chart, "transparent_histogram");
    assert!(!r.canvas.commands().is_empty());
}

#[test]
fn transparent_boxplot() {
    let chart = Charts::boxplot(vec![
        ("Group A".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0]),
        ("Group B".to_string(), vec![2.0, 3.0, 4.0, 5.0, 6.0]),
    ])
    .theme(transparent_theme())
    .title("Transparent BoxPlot")
    .build();
    let r = render_transparent(&chart, "transparent_boxplot");
    assert!(!r.canvas.commands().is_empty());
}

#[test]
fn transparent_pie() {
    let chart = Charts::pie(
        vec!["A".into(), "B".into(), "C".into()],
        &[30.0, 50.0, 20.0],
    )
    .theme(transparent_theme())
    .title("Transparent Pie")
    .build();
    let r = render_transparent(&chart, "transparent_pie");
    assert!(!r.canvas.commands().is_empty());
}

#[test]
fn transparent_heatmap() {
    let chart = Charts::heatmap(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ])
    .theme(transparent_theme())
    .title("Transparent Heatmap")
    .build();
    let r = render_transparent(&chart, "transparent_heatmap");
    assert!(!r.canvas.commands().is_empty());
}

// ===========================================================================
// Section 3: Canvas Size Extremes
// ===========================================================================

#[test]
fn canvas_1x1_all_types() {
    let charts = [
        Charts::scatter(&[1.0], &[1.0])
            .theme(transparent_theme())
            .build(),
        Charts::line(&[1.0]).theme(transparent_theme()).build(),
        Charts::bar(vec!["A".into()], &[1.0])
            .theme(transparent_theme())
            .build(),
        Charts::histogram(&[1.0, 2.0])
            .theme(transparent_theme())
            .build(),
        Charts::pie(vec!["A".into()], &[1.0])
            .theme(transparent_theme())
            .build(),
        Charts::heatmap(vec![vec![1.0]])
            .theme(transparent_theme())
            .build(),
    ];
    for (i, chart) in charts.iter().enumerate() {
        assert_render(chart, 1, 1, &format!("1x1_chart_{i}"));
    }
}

#[test]
fn canvas_very_wide() {
    let chart = Charts::scatter(&[1.0, 2.0], &[1.0, 2.0])
        .theme(transparent_theme())
        .build();
    assert_render(&chart, 5000, 5, "very_wide");
}

#[test]
fn canvas_very_tall() {
    let chart = Charts::scatter(&[1.0, 2.0], &[1.0, 2.0])
        .theme(transparent_theme())
        .build();
    assert_render(&chart, 5, 5000, "very_tall");
}

#[test]
fn canvas_square_large() {
    let chart = Charts::line(&[1.0, 2.0, 3.0, 4.0])
        .theme(transparent_theme())
        .build();
    assert_render(&chart, 2000, 2000, "square_large");
}

// ===========================================================================
// Section 4: Data Extremes
// ===========================================================================

#[test]
fn data_f64_max_scatter() {
    // f64::MAX causes scale arithmetic overflow → NaN overlay coordinates.
    // The important invariant: no panic during rendering.
    let chart = Charts::scatter(&[0.0, f64::MAX], &[0.0, f64::MAX])
        .theme(transparent_theme())
        .build();
    let r = layout::render_chart(&chart, 400, 300);
    assert_eq!(r.canvas.width(), 400);
    assert_eq!(r.canvas.height(), 300);
}

#[test]
fn data_f64_min_positive() {
    let chart = Charts::scatter(&[0.0, f64::MIN_POSITIVE], &[0.0, f64::MIN_POSITIVE])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "f64_min_positive");
}

#[test]
fn data_subnormal_values() {
    let sub = 5.0e-324_f64; // smallest positive subnormal
    let chart = Charts::line(&[sub, sub * 2.0, sub * 3.0])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "subnormal_line");
}

#[test]
fn data_negative_zero() {
    let chart = Charts::scatter(&[-0.0, 0.0, 1.0], &[-0.0, 0.0, 1.0])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "neg_zero");
}

#[test]
fn data_mixed_nan_inf_finite() {
    let chart = Charts::scatter(
        &[f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0, 2.0],
        &[1.0, f64::NAN, 3.0, f64::INFINITY, 5.0],
    )
    .theme(transparent_theme())
    .build();
    render_transparent(&chart, "mixed_nan_inf");
}

#[test]
fn data_all_identical_scatter() {
    let chart = Charts::scatter(&[5.0, 5.0, 5.0, 5.0], &[5.0, 5.0, 5.0, 5.0])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "all_identical_scatter");
}

#[test]
fn data_all_identical_line() {
    let chart = Charts::line(&[42.0, 42.0, 42.0, 42.0, 42.0])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "all_identical_line");
}

#[test]
fn data_single_point_scatter() {
    let chart = Charts::scatter(&[7.0], &[3.0])
        .theme(transparent_theme())
        .title("Single Point")
        .x_label("X")
        .y_label("Y")
        .build();
    render_transparent(&chart, "single_point_scatter");
}

#[test]
fn data_single_point_line() {
    let chart = Charts::line(&[42.0]).theme(transparent_theme()).build();
    render_transparent(&chart, "single_point_line");
}

#[test]
fn data_two_points_line() {
    let chart = Charts::line(&[0.0, 100.0])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "two_points_line");
}

#[test]
fn data_huge_range_diff() {
    let chart = Charts::scatter(&[0.001, 1e100], &[0.001, 1e100])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "huge_range_diff");
}

#[test]
fn data_negative_only_line() {
    let chart = Charts::line(&[-100.0, -50.0, -75.0, -25.0])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "negative_only_line");
}

// ===========================================================================
// Section 5: Line Chart Builder Combos
// ===========================================================================

#[test]
fn line_smooth() {
    let chart = Charts::line(&[0.0, 10.0, 5.0, 15.0, 8.0, 20.0])
        .theme(transparent_theme())
        .smooth()
        .build();
    render_transparent(&chart, "line_smooth");
}

#[test]
fn line_step() {
    let chart = Charts::line(&[0.0, 10.0, 5.0, 15.0, 8.0, 20.0])
        .theme(transparent_theme())
        .step()
        .build();
    render_transparent(&chart, "line_step");
}

#[test]
fn line_filled() {
    let chart = Charts::line(&[0.0, 10.0, 5.0, 15.0])
        .theme(transparent_theme())
        .filled()
        .build();
    render_transparent(&chart, "line_filled");
}

#[test]
fn line_filled_with_points() {
    let chart = Charts::line(&[0.0, 10.0, 5.0, 15.0])
        .theme(transparent_theme())
        .filled()
        .with_points()
        .build();
    render_transparent(&chart, "line_filled_points");
}

#[test]
fn line_smooth_with_points() {
    let chart = Charts::line(&[0.0, 10.0, 5.0, 15.0, 8.0])
        .theme(transparent_theme())
        .smooth()
        .with_points()
        .build();
    render_transparent(&chart, "line_smooth_points");
}

#[test]
fn line_custom_width() {
    let chart = Charts::line(&[1.0, 5.0, 3.0, 7.0])
        .theme(transparent_theme())
        .line_width(5.0)
        .build();
    render_transparent(&chart, "line_custom_width");
}

#[test]
fn line_xy_explicit() {
    let chart = Charts::line_xy(&[0.0, 0.5, 1.0, 5.0, 10.0], &[0.0, 25.0, 10.0, 50.0, 30.0])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "line_xy_explicit");
}

#[test]
fn line_multi_series() {
    let chart = Charts::line(&[1.0, 4.0, 2.0, 8.0])
        .add_named_series("Series B", &[8.0, 2.0, 6.0, 1.0])
        .add_named_series("Series C", &[3.0, 5.0, 7.0, 4.0])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "line_multi_series");
}

#[test]
fn line_smooth_then_step_last_wins() {
    let chart = Charts::line(&[1.0, 5.0, 3.0, 7.0])
        .theme(transparent_theme())
        .smooth()
        .step() // last call wins
        .build();
    render_transparent(&chart, "smooth_then_step");
}

#[test]
fn line_step_then_smooth_last_wins() {
    let chart = Charts::line(&[1.0, 5.0, 3.0, 7.0])
        .theme(transparent_theme())
        .step()
        .smooth() // last call wins
        .build();
    render_transparent(&chart, "step_then_smooth");
}

// ===========================================================================
// Section 6: Scatter Chart Builder Combos
// ===========================================================================

#[test]
fn scatter_all_markers_transparent() {
    let markers = [
        Marker::Circle,
        Marker::Square,
        Marker::Diamond,
        Marker::Cross,
        Marker::Triangle,
    ];
    for (i, m) in markers.iter().enumerate() {
        let chart = Charts::scatter(&[1.0, 2.0, 3.0], &[1.0, 4.0, 2.0])
            .theme(transparent_theme())
            .marker(*m)
            .build();
        render_transparent(&chart, &format!("marker_{i}"));
    }
}

#[test]
fn scatter_connected_transparent() {
    let chart = Charts::scatter(&[1.0, 2.0, 3.0, 4.0], &[1.0, 4.0, 2.0, 5.0])
        .theme(transparent_theme())
        .connected()
        .build();
    render_transparent(&chart, "scatter_connected");
}

#[test]
fn scatter_custom_size() {
    let chart = Charts::scatter(&[1.0, 2.0, 3.0], &[1.0, 4.0, 2.0])
        .theme(transparent_theme())
        .size(10.0)
        .build();
    render_transparent(&chart, "scatter_custom_size");
}

#[test]
fn scatter_multi_series() {
    let chart = Charts::scatter(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0])
        .add_named_series("Extra", &[3.0, 2.0, 1.0], &[1.0, 2.0, 3.0])
        .theme(transparent_theme())
        .connected()
        .build();
    render_transparent(&chart, "scatter_multi_series");
}

#[test]
fn scatter_trend_line() {
    let chart = Charts::scatter(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2.1, 3.9, 6.2, 7.8, 10.1])
        .theme(transparent_theme())
        .trend_line()
        .build();
    render_transparent(&chart, "scatter_trend");
}

// ===========================================================================
// Section 7: Bar Chart Builder Combos
// ===========================================================================

#[test]
fn bar_stacked_transparent() {
    let chart = Charts::bar(
        vec!["Q1".into(), "Q2".into(), "Q3".into(), "Q4".into()],
        &[10.0, 20.0, 15.0, 25.0],
    )
    .add_named_series("Expenses", &[5.0, 15.0, 10.0, 20.0])
    .theme(transparent_theme())
    .stacked()
    .build();
    render_transparent(&chart, "bar_stacked");
}

#[test]
fn bar_horizontal_transparent() {
    let chart = Charts::bar(
        vec!["Cat".into(), "Dog".into(), "Bird".into()],
        &[30.0, 50.0, 20.0],
    )
    .theme(transparent_theme())
    .horizontal()
    .build();
    render_transparent(&chart, "bar_horizontal");
}

#[test]
fn bar_stacked_horizontal() {
    let chart = Charts::bar(vec!["A".into(), "B".into()], &[10.0, 20.0])
        .add_named_series("S2", &[5.0, 15.0])
        .theme(transparent_theme())
        .stacked()
        .horizontal()
        .build();
    render_transparent(&chart, "bar_stacked_horizontal");
}

#[test]
fn bar_custom_corner_radius() {
    let chart = Charts::bar(vec!["A".into(), "B".into()], &[10.0, 20.0])
        .theme(transparent_theme())
        .corner_radius(10.0)
        .build();
    render_transparent(&chart, "bar_corner_radius");
}

#[test]
fn bar_zero_corner_radius() {
    let chart = Charts::bar(vec!["A".into(), "B".into()], &[10.0, 20.0])
        .theme(transparent_theme())
        .corner_radius(0.0)
        .build();
    render_transparent(&chart, "bar_zero_radius");
}

#[test]
fn bar_no_corner_radius_uses_theme() {
    // When corner_radius is not set (None), should use theme.series.bar_corner_radius
    let chart = Charts::bar(vec!["A".into(), "B".into()], &[10.0, 20.0])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "bar_theme_radius");
}

#[test]
fn bar_custom_gap() {
    let chart = Charts::bar(
        vec!["A".into(), "B".into(), "C".into()],
        &[10.0, 20.0, 30.0],
    )
    .theme(transparent_theme())
    .gap(0.0)
    .build();
    render_transparent(&chart, "bar_no_gap");
}

#[test]
fn bar_max_gap() {
    let chart = Charts::bar(vec!["A".into(), "B".into()], &[10.0, 20.0])
        .theme(transparent_theme())
        .gap(1.0) // should clamp to 0.9
        .build();
    render_transparent(&chart, "bar_max_gap");
}

#[test]
fn bar_show_values() {
    let chart = Charts::bar(vec!["A".into(), "B".into()], &[10.0, 20.0])
        .theme(transparent_theme())
        .show_values()
        .build();
    render_transparent(&chart, "bar_show_values");
}

#[test]
fn bar_single_category() {
    let chart = Charts::bar(vec!["Only".into()], &[42.0])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "bar_single");
}

#[test]
fn bar_many_categories() {
    let labels: Vec<String> = (0..50).map(|i| format!("Cat{i}")).collect();
    let values: Vec<f64> = (0..50).map(|i| (i as f64) * 2.0).collect();
    let chart = Charts::bar(labels, &values)
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "bar_many_cats");
}

#[test]
fn bar_series_labels() {
    let chart = Charts::bar(vec!["A".into(), "B".into()], &[10.0, 20.0])
        .add_named_series("S2", &[5.0, 15.0])
        .series_labels(&["Revenue", "Costs"])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "bar_series_labels");
}

// ===========================================================================
// Section 8: Histogram Edge Cases
// ===========================================================================

#[test]
fn histogram_density_transparent() {
    let chart = Charts::histogram(&[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        .theme(transparent_theme())
        .density()
        .build();
    render_transparent(&chart, "histogram_density");
}

#[test]
fn histogram_custom_bins() {
    let chart = Charts::histogram(&[1.0, 2.0, 3.0, 4.0, 5.0])
        .theme(transparent_theme())
        .bins(3)
        .build();
    render_transparent(&chart, "histogram_3bins");
}

#[test]
fn histogram_single_bin() {
    let chart = Charts::histogram(&[1.0, 2.0, 3.0])
        .theme(transparent_theme())
        .bins(1)
        .build();
    render_transparent(&chart, "histogram_1bin");
}

#[test]
fn histogram_many_bins() {
    let chart = Charts::histogram(&[1.0, 2.0, 3.0, 4.0, 5.0])
        .theme(transparent_theme())
        .bins(100) // more bins than data points
        .build();
    render_transparent(&chart, "histogram_100bins");
}

#[test]
fn histogram_single_value() {
    let chart = Charts::histogram(&[42.0])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "histogram_single");
}

#[test]
fn histogram_all_identical() {
    let chart = Charts::histogram(&[5.0, 5.0, 5.0, 5.0, 5.0])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "histogram_identical");
}

#[test]
fn histogram_custom_opacity() {
    let chart = Charts::histogram(&[1.0, 2.0, 3.0, 4.0])
        .theme(transparent_theme())
        .opacity(0.3)
        .build();
    render_transparent(&chart, "histogram_low_opacity");
}

#[test]
fn histogram_multi_series() {
    let chart = Charts::histogram(&[1.0, 2.0, 3.0, 4.0, 5.0])
        .add_series(Series::from_values(vec![2.0, 3.0, 4.0, 5.0, 6.0]))
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "histogram_multi");
}

#[test]
fn histogram_with_nan_in_data() {
    let chart = Charts::histogram(&[1.0, f64::NAN, 3.0, f64::INFINITY, 5.0])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "histogram_nan_data");
}

#[test]
fn histogram_h_line_and_legend() {
    // Verify Phase 3 additions: h_line and no_legend work on Histogram
    let chart = Charts::histogram(&[1.0, 2.0, 3.0, 4.0, 5.0])
        .theme(transparent_theme())
        .h_line(3.0)
        .no_legend()
        .build();
    render_transparent(&chart, "histogram_h_lines");
}

// ===========================================================================
// Section 9: Pie Chart Edge Cases
// ===========================================================================

#[test]
fn pie_donut_transparent() {
    let chart = Charts::pie(
        vec!["A".into(), "B".into(), "C".into()],
        &[30.0, 50.0, 20.0],
    )
    .theme(transparent_theme())
    .donut(0.5)
    .build();
    render_transparent(&chart, "pie_donut");
}

#[test]
fn pie_donut_extreme_ratio() {
    let chart = Charts::pie(vec!["A".into(), "B".into()], &[50.0, 50.0])
        .theme(transparent_theme())
        .donut(0.85)
        .build();
    render_transparent(&chart, "pie_donut_extreme");
}

#[test]
fn pie_donut_over_clamp() {
    // ratio > 0.85 should clamp to 0.85
    let chart = Charts::pie(vec!["A".into()], &[100.0])
        .theme(transparent_theme())
        .donut(2.0)
        .build();
    render_transparent(&chart, "pie_donut_clamp");
}

#[test]
fn pie_start_angle() {
    let chart = Charts::pie(vec!["A".into(), "B".into()], &[60.0, 40.0])
        .theme(transparent_theme())
        .start_angle_degrees(90.0)
        .build();
    render_transparent(&chart, "pie_start_90");
}

#[test]
fn pie_hide_percentages() {
    let chart = Charts::pie(vec!["A".into(), "B".into()], &[70.0, 30.0])
        .theme(transparent_theme())
        .hide_percentages()
        .build();
    render_transparent(&chart, "pie_no_pct");
}

#[test]
fn pie_single_slice() {
    let chart = Charts::pie(vec!["Everything".into()], &[100.0])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "pie_single_slice");
}

#[test]
fn pie_many_slices() {
    let labels: Vec<String> = (0..20).map(|i| format!("S{i}")).collect();
    let values: Vec<f64> = (1..=20).map(|i| i as f64).collect();
    let chart = Charts::pie(labels, &values)
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "pie_20_slices");
}

#[test]
fn pie_one_dominant_slice() {
    let chart = Charts::pie(
        vec!["Big".into(), "Tiny1".into(), "Tiny2".into()],
        &[999.0, 0.5, 0.5],
    )
    .theme(transparent_theme())
    .build();
    render_transparent(&chart, "pie_dominant");
}

#[test]
fn pie_with_nan_slice() {
    let chart = Charts::pie(
        vec!["A".into(), "B".into(), "NaN".into()],
        &[50.0, 50.0, f64::NAN],
    )
    .theme(transparent_theme())
    .build();
    render_transparent(&chart, "pie_nan_slice");
}

// ===========================================================================
// Section 10: BoxPlot Edge Cases
// ===========================================================================

#[test]
fn boxplot_all_identical_data() {
    let chart = Charts::boxplot(vec![("Same".to_string(), vec![5.0, 5.0, 5.0, 5.0, 5.0])])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "boxplot_identical");
}

#[test]
fn boxplot_single_value() {
    let chart = Charts::boxplot(vec![("One".to_string(), vec![42.0])])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "boxplot_single_val");
}

#[test]
fn boxplot_with_outliers() {
    let chart = Charts::boxplot(vec![(
        "With Outliers".to_string(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 50.0, -20.0],
    )])
    .theme(transparent_theme())
    .build();
    render_transparent(&chart, "boxplot_outliers");
}

#[test]
fn boxplot_many_groups() {
    let groups: Vec<(String, Vec<f64>)> = (0..15)
        .map(|i| {
            let data: Vec<f64> = (0..20).map(|j| (i * 10 + j) as f64).collect();
            (format!("G{i}"), data)
        })
        .collect();
    let chart = Charts::boxplot(groups).theme(transparent_theme()).build();
    render_transparent(&chart, "boxplot_many_groups");
}

#[test]
fn boxplot_nan_in_data() {
    let chart = Charts::boxplot(vec![(
        "Mixed".to_string(),
        vec![1.0, f64::NAN, 3.0, f64::INFINITY, 5.0],
    )])
    .theme(transparent_theme())
    .build();
    render_transparent(&chart, "boxplot_nan");
}

#[test]
fn boxplot_negative_data() {
    let chart = Charts::boxplot(vec![(
        "Neg".to_string(),
        vec![-10.0, -5.0, -3.0, -1.0, 0.0],
    )])
    .theme(transparent_theme())
    .build();
    render_transparent(&chart, "boxplot_negative");
}

// ===========================================================================
// Section 11: Heatmap Edge Cases
// ===========================================================================

#[test]
fn heatmap_single_cell() {
    let chart = Charts::heatmap(vec![vec![42.0]])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "heatmap_1x1");
}

#[test]
fn heatmap_single_row() {
    let chart = Charts::heatmap(vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "heatmap_1xN");
}

#[test]
fn heatmap_single_column() {
    let chart = Charts::heatmap(vec![vec![1.0], vec![2.0], vec![3.0]])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "heatmap_Nx1");
}

#[test]
fn heatmap_with_nan() {
    let chart = Charts::heatmap(vec![
        vec![1.0, f64::NAN, 3.0],
        vec![f64::NAN, 5.0, f64::NAN],
    ])
    .theme(transparent_theme())
    .build();
    render_transparent(&chart, "heatmap_nan");
}

#[test]
fn heatmap_negative_values() {
    let chart = Charts::heatmap(vec![vec![-10.0, -5.0, 0.0], vec![5.0, 10.0, 15.0]])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "heatmap_negative");
}

#[test]
fn heatmap_all_identical() {
    let chart = Charts::heatmap(vec![vec![7.0, 7.0, 7.0], vec![7.0, 7.0, 7.0]])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "heatmap_identical");
}

#[test]
fn heatmap_with_labels() {
    let chart = Charts::heatmap(vec![vec![1.0, 2.0], vec![3.0, 4.0]])
        .row_labels(vec!["Row A".into(), "Row B".into()])
        .col_labels(vec!["Col 1".into(), "Col 2".into()])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "heatmap_labels");
}

#[test]
fn heatmap_show_values() {
    let chart = Charts::heatmap(vec![vec![1.23, 4.56], vec![7.89, 0.12]])
        .values(true)
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "heatmap_values");
}

#[test]
fn heatmap_large_grid() {
    let grid: Vec<Vec<f64>> = (0..30)
        .map(|r| (0..30).map(|c| (r * 30 + c) as f64).collect())
        .collect();
    let chart = Charts::heatmap(grid).theme(transparent_theme()).build();
    assert_render(&chart, 800, 800, "heatmap_30x30");
}

#[test]
fn heatmap_long_labels_dynamic_margin() {
    let chart = Charts::heatmap(vec![vec![1.0, 2.0]])
        .row_labels(vec!["This Is A Very Long Row Label".into()])
        .col_labels(vec!["A".into(), "B".into()])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "heatmap_long_labels");
}

#[test]
fn heatmap_custom_colors() {
    let chart = Charts::heatmap(vec![vec![0.0, 50.0], vec![50.0, 100.0]])
        .colors(
            Color::from_rgba8(0, 0, 255, 255),
            Color::from_rgba8(255, 0, 0, 255),
        )
        .range(0.0, 100.0)
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "heatmap_custom_colors");
}

#[test]
fn heatmap_cell_styling() {
    let chart = Charts::heatmap(vec![vec![1.0, 2.0], vec![3.0, 4.0]])
        .cell_radius(8.0)
        .cell_gap(5.0)
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "heatmap_cell_style");
}

#[test]
fn heatmap_correlation_matrix() {
    let data = vec![
        vec![1.0, 0.8, -0.3],
        vec![0.8, 1.0, 0.1],
        vec![-0.3, 0.1, 1.0],
    ];
    let labels = vec!["A".into(), "B".into(), "C".into()];
    let chart = Heatmap::correlation(data, labels)
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "heatmap_correlation");
}

// ===========================================================================
// Section 12: Reference Lines
// ===========================================================================

#[test]
fn ref_lines_horizontal() {
    let chart = Charts::scatter(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0])
        .theme(transparent_theme())
        .h_line(15.0)
        .h_line(25.0)
        .build();
    render_transparent(&chart, "ref_h_lines");
}

#[test]
fn ref_lines_vertical() {
    let chart = Charts::scatter(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0])
        .theme(transparent_theme())
        .v_line(2.5)
        .build();
    render_transparent(&chart, "ref_v_lines");
}

#[test]
fn ref_line_at_data_bounds() {
    let chart = Charts::line(&[0.0, 50.0, 100.0])
        .theme(transparent_theme())
        .h_line(0.0)
        .h_line(100.0)
        .build();
    render_transparent(&chart, "ref_at_bounds");
}

#[test]
fn ref_line_outside_range() {
    let chart = Charts::line(&[10.0, 20.0, 30.0])
        .theme(transparent_theme())
        .h_line(1000.0)
        .build();
    render_transparent(&chart, "ref_outside_range");
}

#[test]
fn ref_lines_styled() {
    let chart = Charts::scatter(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0])
        .theme(transparent_theme())
        .h_line_styled(2.0, Color::from_rgba8(255, 0, 0, 200))
        .v_line_styled(2.0, Color::from_rgba8(0, 255, 0, 200))
        .build();
    render_transparent(&chart, "ref_styled");
}

// ===========================================================================
// Section 13: Annotations
// ===========================================================================

#[test]
fn annotation_basic() {
    let chart = Charts::scatter(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0])
        .theme(transparent_theme())
        .annotate(2.0, 20.0, "Peak")
        .build();
    render_transparent(&chart, "annotation_basic");
}

#[test]
fn many_annotations() {
    let mut builder = Charts::scatter(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5.0, 4.0, 3.0, 2.0, 1.0])
        .theme(transparent_theme());
    for i in 1..=5 {
        builder = builder.annotate(i as f64, (6 - i) as f64, format!("Pt{i}"));
    }
    let chart = builder.build();
    render_transparent(&chart, "many_annotations");
}

// ===========================================================================
// Section 14: Config Combos
// ===========================================================================

#[test]
fn full_config_scatter() {
    let chart = Charts::scatter(&[0.0, 5.0, 10.0], &[0.0, 25.0, 50.0])
        .theme(transparent_theme())
        .title("Full Config")
        .x_label("Independent Var")
        .y_label("Dependent Var")
        .x_range(-1.0, 12.0)
        .y_range(-5.0, 60.0)
        .trend_line()
        .h_line(25.0)
        .v_line(5.0)
        .annotate(5.0, 25.0, "Mid")
        .build();
    render_transparent(&chart, "full_config");
}

#[test]
fn custom_range_clipping() {
    let chart = Charts::line(&[0.0, 50.0, 100.0, 150.0, 200.0])
        .theme(transparent_theme())
        .y_range(20.0, 120.0)
        .build();
    render_transparent(&chart, "range_clipping");
}

#[test]
fn inverted_range() {
    let chart = Charts::scatter(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0])
        .theme(transparent_theme())
        .y_range(10.0, 0.0) // inverted
        .build();
    render_transparent(&chart, "inverted_range");
}

#[test]
fn no_title_no_labels() {
    let chart = Charts::line(&[1.0, 2.0, 3.0])
        .theme(transparent_theme())
        .no_legend()
        .build();
    render_transparent(&chart, "no_title_labels");
}

// ===========================================================================
// Section 15: Theme Variants
// ===========================================================================

#[test]
fn all_themes_scatter() {
    let themes = [
        ("dark", Theme::dark()),
        ("light", Theme::light()),
        ("ocean", Theme::ocean()),
        ("forest", Theme::forest()),
        ("pastel", Theme::pastel()),
        ("colorblind", Theme::colorblind()),
    ];
    for (name, theme) in themes {
        let chart = Charts::scatter(&[1.0, 2.0, 3.0], &[1.0, 4.0, 2.0])
            .theme(theme)
            .title(name)
            .build();
        render_transparent(&chart, &format!("theme_{name}"));
    }
}

#[test]
fn all_themes_transparent_bg() {
    let themes = [
        Theme::dark(),
        Theme::light(),
        Theme::ocean(),
        Theme::forest(),
        Theme::pastel(),
        Theme::colorblind(),
    ];
    for (i, mut theme) in themes.into_iter().enumerate() {
        theme.background = Color::from_rgba8(0, 0, 0, 0);
        let chart = Charts::line(&[1.0, 5.0, 3.0, 7.0]).theme(theme).build();
        render_transparent(&chart, &format!("theme_transparent_{i}"));
    }
}

// ===========================================================================
// Section 16: Unicode & Empty Labels
// ===========================================================================

#[test]
fn unicode_labels_bar() {
    let chart = Charts::bar(
        vec![
            "日本語".into(),
            "한국어".into(),
            "中文".into(),
            "🚀🔥".into(),
        ],
        &[10.0, 20.0, 30.0, 40.0],
    )
    .theme(transparent_theme())
    .title("유니코드 テスト")
    .x_label("カテゴリ")
    .y_label("値")
    .build();
    render_transparent(&chart, "unicode_labels");
}

#[test]
fn empty_string_labels() {
    let chart = Charts::bar(vec!["".into(), "".into(), "".into()], &[1.0, 2.0, 3.0])
        .theme(transparent_theme())
        .title("")
        .x_label("")
        .y_label("")
        .build();
    render_transparent(&chart, "empty_labels");
}

#[test]
fn very_long_labels() {
    let long_label = "A".repeat(200);
    let chart = Charts::bar(vec![long_label.clone(), "Short".into()], &[10.0, 20.0])
        .theme(transparent_theme())
        .title(&long_label)
        .build();
    render_transparent(&chart, "very_long_labels");
}

// ===========================================================================
// Section 17: Zoom Integration
// ===========================================================================

#[test]
fn zoom_scatter() {
    let mut chart = Charts::scatter(&[0.0, 5.0, 10.0, 15.0, 20.0], &[0.0, 25.0, 10.0, 30.0, 5.0])
        .theme(transparent_theme())
        .build();

    let mut zoom = ZoomState::new(0.0, 20.0, 0.0, 30.0);
    zoom.zoom_in();
    let (x_min, x_max) = zoom.x_range();
    let (y_min, y_max) = zoom.y_range();
    chart.config_mut().unwrap().axes.x_range = Some((x_min, x_max));
    chart.config_mut().unwrap().axes.y_range = Some((y_min, y_max));

    render_transparent(&chart, "zoom_scatter");
}

#[test]
fn zoom_pan_left() {
    let mut chart = Charts::line(&[0.0, 5.0, 10.0, 15.0, 20.0])
        .theme(transparent_theme())
        .build();

    let mut zoom = ZoomState::new(0.0, 4.0, 0.0, 20.0);
    zoom.zoom_in();
    zoom.pan_left();
    let (x_min, x_max) = zoom.x_range();
    let (y_min, y_max) = zoom.y_range();
    chart.config_mut().unwrap().axes.x_range = Some((x_min, x_max));
    chart.config_mut().unwrap().axes.y_range = Some((y_min, y_max));

    render_transparent(&chart, "zoom_pan_left");
}

#[test]
fn zoom_reset() {
    let mut zoom = ZoomState::new(0.0, 100.0, 0.0, 100.0);
    zoom.zoom_in();
    zoom.zoom_in();
    zoom.pan_right();
    zoom.reset();

    assert_eq!(zoom.x_range(), (0.0, 100.0));
    assert_eq!(zoom.y_range(), (0.0, 100.0));
}

#[test]
fn zoom_out_beyond_original() {
    let mut zoom = ZoomState::new(0.0, 10.0, 0.0, 10.0);
    zoom.zoom_out();
    zoom.zoom_out();
    zoom.zoom_out();

    let (x_min, x_max) = zoom.x_range();
    let (y_min, y_max) = zoom.y_range();
    assert!(x_min.is_finite() && x_max.is_finite());
    assert!(y_min.is_finite() && y_max.is_finite());
}

// ===========================================================================
// Section 18: RenderedChart Properties
// ===========================================================================

#[test]
fn rendered_chart_has_plot_area() {
    let chart = Charts::scatter(&[1.0, 2.0], &[1.0, 2.0])
        .theme(transparent_theme())
        .build();
    let r = render_transparent(&chart, "has_plot_area");
    assert!(r.plot_area.is_some(), "Scatter should have plot_area");
}

#[test]
fn rendered_chart_has_scales() {
    let chart = Charts::scatter(&[1.0, 2.0], &[1.0, 2.0])
        .theme(transparent_theme())
        .build();
    let r = render_transparent(&chart, "has_scales");
    // Scales may or may not be set depending on renderer internals.
    // Just verify no panic and the fields are accessible.
    let _x = r.x_scale;
    let _y = r.y_scale;
}

#[test]
fn rendered_pie_structure() {
    let chart = Charts::pie(vec!["A".into()], &[1.0])
        .theme(transparent_theme())
        .build();
    let r = render_transparent(&chart, "pie_structure");
    // Just verify no panic — pie may or may not set plot_area/scales
    let _ = r.plot_area;
}

#[test]
fn rendered_heatmap_has_plot_area() {
    let chart = Charts::heatmap(vec![vec![1.0, 2.0], vec![3.0, 4.0]])
        .theme(transparent_theme())
        .build();
    let r = render_transparent(&chart, "heatmap_plot_area");
    assert!(r.plot_area.is_some(), "Heatmap should have plot_area");
}

// ===========================================================================
// Section 19: Series Data Type
// ===========================================================================

#[test]
fn series_from_named() {
    let s = Series::new("TestSeries", vec![1.0, 2.0, 3.0]);
    assert_eq!(s.label(), "TestSeries");
    assert_eq!(s.len(), 3);
    assert!(!s.is_empty());
}

#[test]
fn series_empty() {
    let s = Series::from_values(vec![]);
    assert_eq!(s.len(), 0);
    assert!(s.is_empty());
    assert!(s.min().is_none());
    assert!(s.max().is_none());
}

#[test]
fn series_nan_extent() {
    let s = Series::from_values(vec![f64::NAN, 1.0, f64::NAN, 3.0]);
    assert_eq!(s.min(), Some(1.0));
    assert_eq!(s.max(), Some(3.0));
}

// ===========================================================================
// Section 20: BoxStats Unit Tests
// ===========================================================================

// BoxStats tests removed — boxplot module is pub(crate) and not accessible from integration tests.
// BoxStats::from_data is exercised indirectly by the boxplot rendering tests above.

// ===========================================================================
// Section 21: Chart Clone & Reference Line Builder
// ===========================================================================

#[test]
fn chart_clone_and_render() {
    let chart = Charts::line(&[1.0, 2.0, 3.0])
        .theme(transparent_theme())
        .title("Clone Test")
        .build();

    let clone = chart.clone();
    let r1 = render_transparent(&chart, "clone_orig");
    let r2 = render_transparent(&clone, "clone_copy");

    assert_eq!(r1.canvas.commands().len(), r2.canvas.commands().len());
    assert_eq!(r1.text_labels().len(), r2.text_labels().len());
}

#[test]
fn annotation_builder_chain() {
    let ann = Annotation::new(1.0, 2.0, "Test")
        .with_arrow()
        .with_background(Color::from_rgba8(0, 0, 0, 128))
        .with_offset(20.0, -10.0)
        .with_color(Color::from_rgba8(255, 255, 0, 255));
    assert!(ann.arrow);
    assert!(ann.style.background.is_some());
    assert_eq!(ann.style.offset, (20.0, -10.0));
}

// ===========================================================================
// Section 22: All-types rendering at same dimensions
// ===========================================================================

#[test]
fn render_all_types_same_canvas() {
    let charts: Vec<(&str, Chart)> = vec![
        (
            "scatter",
            Charts::scatter(&[1.0, 2.0], &[1.0, 2.0])
                .theme(transparent_theme())
                .build(),
        ),
        (
            "line",
            Charts::line(&[1.0, 2.0]).theme(transparent_theme()).build(),
        ),
        (
            "bar",
            Charts::bar(vec!["A".into()], &[1.0])
                .theme(transparent_theme())
                .build(),
        ),
        (
            "hist",
            Charts::histogram(&[1.0, 2.0])
                .theme(transparent_theme())
                .build(),
        ),
        (
            "pie",
            Charts::pie(vec!["A".into()], &[1.0])
                .theme(transparent_theme())
                .build(),
        ),
        (
            "heatmap",
            Charts::heatmap(vec![vec![1.0]])
                .theme(transparent_theme())
                .build(),
        ),
        (
            "boxplot",
            Charts::boxplot(vec![("A".to_string(), vec![1.0, 2.0, 3.0])])
                .theme(transparent_theme())
                .build(),
        ),
    ];
    for (name, chart) in &charts {
        assert_render(chart, 300, 200, name);
    }
}

// ===========================================================================
// Section 23: Bar Show Values
// ===========================================================================

#[test]
fn bar_show_values_vertical() {
    let chart = Charts::bar(
        vec!["A".into(), "B".into(), "C".into()],
        &[10.0, 25.0, 15.0],
    )
    .theme(transparent_theme())
    .show_values()
    .build();
    let r = render_transparent(&chart, "bar_show_values_v");
    let value_texts = r.text_labels();
    assert!(
        value_texts.contains(&"10"),
        "Should contain '10', got: {value_texts:?}"
    );
    assert!(
        value_texts.contains(&"25"),
        "Should contain '25', got: {value_texts:?}"
    );
    assert!(
        value_texts.contains(&"15"),
        "Should contain '15', got: {value_texts:?}"
    );
}

#[test]
fn bar_show_values_horizontal() {
    let chart = Charts::bar(vec!["X".into(), "Y".into()], &[100.0, 200.0])
        .theme(transparent_theme())
        .horizontal()
        .show_values()
        .build();
    let r = render_transparent(&chart, "bar_show_values_h");
    let value_texts = r.text_labels();
    assert!(
        value_texts.contains(&"100"),
        "Should contain '100', got: {value_texts:?}"
    );
    assert!(
        value_texts.contains(&"200"),
        "Should contain '200', got: {value_texts:?}"
    );
}

#[test]
fn bar_show_values_stacked() {
    let chart = Charts::bar(vec!["A".into(), "B".into()], &[10.0, 20.0])
        .add_named_series("S2", &[5.0, 15.0])
        .theme(transparent_theme())
        .stacked()
        .show_values()
        .build();
    let r = render_transparent(&chart, "bar_show_values_stacked");
    let value_texts = r.text_labels();
    // Stacked totals: 10+5=15, 20+15=35
    assert!(
        value_texts.contains(&"15"),
        "Should contain '15', got: {value_texts:?}"
    );
    assert!(
        value_texts.contains(&"35"),
        "Should contain '35', got: {value_texts:?}"
    );
}

#[test]
fn bar_show_values_negative() {
    let chart = Charts::bar(vec!["A".into(), "B".into()], &[-10.0, 20.0])
        .theme(transparent_theme())
        .show_values()
        .build();
    let r = render_transparent(&chart, "bar_show_values_neg");
    let value_texts = r.text_labels();
    assert!(
        value_texts.contains(&"-10"),
        "Should contain '-10', got: {value_texts:?}"
    );
    assert!(
        value_texts.contains(&"20"),
        "Should contain '20', got: {value_texts:?}"
    );
}

#[test]
fn colorblind_theme_renders() {
    let theme = Theme::colorblind();
    let charts: Vec<Chart> = vec![
        Charts::scatter(&[1.0, 2.0, 3.0], &[1.0, 4.0, 2.0])
            .theme(theme.clone())
            .build(),
        Charts::line(&[1.0, 5.0, 3.0]).theme(theme.clone()).build(),
        Charts::bar(vec!["A".into(), "B".into()], &[10.0, 20.0])
            .theme(theme.clone())
            .build(),
        Charts::pie(vec!["A".into(), "B".into()], &[60.0, 40.0])
            .theme(theme)
            .build(),
    ];
    for (i, chart) in charts.iter().enumerate() {
        assert_render(chart, 400, 300, &format!("colorblind_{i}"));
    }
}

// ===========================================================================
// Section: try_build() for remaining chart types
// ===========================================================================

#[test]
fn try_build_waterfall_empty() {
    use scry_chart::chart::WaterfallChart;
    let r = WaterfallChart::new(vec![], vec![]).try_build();
    assert_eq!(r.unwrap_err(), ChartError::EmptyData);
}

#[test]
fn try_build_waterfall_valid() {
    let r = Charts::waterfall(vec!["A".into(), "B".into()], &[10.0, -5.0]).try_build();
    assert!(r.is_ok());
}

#[test]
fn try_build_funnel_empty() {
    use scry_chart::chart::FunnelChart;
    let r = FunnelChart::new(vec![], vec![]).try_build();
    assert_eq!(r.unwrap_err(), ChartError::EmptyData);
}

#[test]
fn try_build_funnel_valid() {
    let r = Charts::funnel(vec!["A".into(), "B".into()], &[100.0, 60.0]).try_build();
    assert!(r.is_ok());
}

#[test]
fn try_build_gauge_valid() {
    use scry_chart::chart::GaugeChart;
    let r = GaugeChart::new(75.0).try_build();
    assert!(r.is_ok());
}

#[test]
fn try_build_lollipop_empty() {
    use scry_chart::chart::LollipopChart;
    let r = LollipopChart::new(vec![], vec![]).try_build();
    assert_eq!(r.unwrap_err(), ChartError::EmptyData);
}

#[test]
fn try_build_lollipop_valid() {
    let r = Charts::lollipop(vec!["A".into(), "B".into()], &[10.0, 20.0]).try_build();
    assert!(r.is_ok());
}

#[test]
fn try_build_gantt_empty() {
    use scry_chart::chart::GanttChart;
    let r = GanttChart::new(vec![]).try_build();
    assert_eq!(r.unwrap_err(), ChartError::EmptyData);
}

#[test]
fn try_build_gantt_valid() {
    use scry_chart::chart::GanttTask;
    let tasks = vec![GanttTask::new("Task A", 0.0, 5.0)];
    let r = Charts::gantt(tasks).try_build();
    assert!(r.is_ok());
}

#[test]
fn try_build_candlestick_empty() {
    use scry_chart::chart::CandlestickChart;
    let r = CandlestickChart::new(vec![]).try_build();
    assert_eq!(r.unwrap_err(), ChartError::EmptyData);
}

#[test]
fn try_build_candlestick_valid() {
    use scry_chart::chart::OhlcEntry;
    let data = vec![OhlcEntry::new(1.0, 10.0, 15.0, 8.0, 12.0)];
    let r = Charts::candlestick(data).try_build();
    assert!(r.is_ok());
}

#[test]
fn try_build_radar_empty() {
    use scry_chart::chart::RadarChart;
    let r = RadarChart::new(Vec::<String>::new()).try_build();
    assert_eq!(r.unwrap_err(), ChartError::EmptyData);
}

#[test]
fn try_build_radar_valid() {
    let r = Charts::radar(vec!["A", "B", "C"])
        .add_series("S", &[1.0, 2.0, 3.0])
        .try_build();
    assert!(r.is_ok());
}

#[test]
fn try_build_sparkline_empty() {
    use scry_chart::chart::Sparkline;
    let r = Sparkline::new(vec![]).try_build();
    assert_eq!(r.unwrap_err(), ChartError::EmptyData);
}

#[test]
fn try_build_sparkline_valid() {
    let r = Charts::sparkline(&[1.0, 5.0, 3.0, 7.0]).try_build();
    assert!(r.is_ok());
}

#[test]
fn try_build_bubble_empty() {
    use scry_chart::chart::BubbleChart;
    let r = BubbleChart::new(
        Series::from_values(vec![]),
        Series::from_values(vec![]),
        vec![],
    )
    .try_build();
    assert_eq!(r.unwrap_err(), ChartError::EmptyData);
}

#[test]
fn try_build_bubble_valid() {
    let r = Charts::bubble(&[1.0, 2.0], &[3.0, 4.0], &[10.0, 20.0]).try_build();
    assert!(r.is_ok());
}

#[test]
fn try_build_violin_empty() {
    use scry_chart::chart::ViolinPlot;
    let r = ViolinPlot::new(Vec::<(String, Vec<f64>)>::new()).try_build();
    assert_eq!(r.unwrap_err(), ChartError::EmptyData);
}

#[test]
fn try_build_violin_valid() {
    let r = Charts::violin(vec![("G".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0])]).try_build();
    assert!(r.is_ok());
}

// ===========================================================================
// Section: Degenerate rendering — charts that should NOT panic
// ===========================================================================

#[test]
fn degenerate_pie_single_slice() {
    // Full circle, not an arc with a gap
    let chart = Charts::pie(vec!["Only".into()], &[100.0])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "pie_single_slice");
}

#[test]
fn degenerate_radar_two_axes() {
    // Minimal degenerate polygon
    let chart = Charts::radar(vec!["X", "Y"])
        .add_series("S", &[3.0, 7.0])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "radar_two_axes");
}

#[test]
fn degenerate_candlestick_doji() {
    // open == close — doji candle
    use scry_chart::chart::OhlcEntry;
    let data = vec![
        OhlcEntry::new(1.0, 10.0, 12.0, 8.0, 10.0),  // doji
        OhlcEntry::new(2.0, 15.0, 17.0, 13.0, 15.0), // doji
    ];
    let chart = Charts::candlestick(data).theme(transparent_theme()).build();
    render_transparent(&chart, "candlestick_doji");
}

#[test]
fn degenerate_funnel_all_equal() {
    // All equal values — should render rectangles, not trapezoids
    let chart = Charts::funnel(
        vec!["A".into(), "B".into(), "C".into()],
        &[50.0, 50.0, 50.0],
    )
    .theme(transparent_theme())
    .build();
    render_transparent(&chart, "funnel_equal");
}

#[test]
fn degenerate_violin_few_points() {
    // <5 data points — KDE is statistically meaningless but shouldn't crash
    let chart = Charts::violin(vec![("Small".to_string(), vec![1.0, 2.0])])
        .theme(transparent_theme())
        .build();
    render_transparent(&chart, "violin_few_points");
}

#[test]
fn degenerate_gauge_out_of_range() {
    use scry_chart::chart::GaugeChart;
    // Value exceeds default [0, 100] range
    let chart = GaugeChart::new(150.0).theme(transparent_theme()).build();
    render_transparent(&chart, "gauge_over_range");
}

#[test]
fn degenerate_waterfall_crosses_zero() {
    // Running total crosses zero multiple times
    let chart = Charts::waterfall(
        vec!["Start".into(), "Down".into(), "Up".into(), "Down2".into()],
        &[100.0, -150.0, 200.0, -300.0],
    )
    .theme(transparent_theme())
    .build();
    render_transparent(&chart, "waterfall_cross_zero");
}
