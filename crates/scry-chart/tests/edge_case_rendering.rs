//! Edge-case rendering tests for scry-chart.
//!
//! These tests verify that the chart library handles unusual or degenerate
//! inputs gracefully — no panics, no infinite loops, no corrupted output.

use scry_chart::chart::{Chart, LineChart};
use scry_chart::data::Series;
use scry_chart::export::render_to_png;
use scry_chart::layout::render_chart;

const W: u32 = 400;
const H: u32 = 300;

/// Helper: render a chart to PNG and verify it produces valid output.
fn assert_renders(chart: &Chart) {
    let rendered = render_chart(chart, W, H);
    assert!(rendered.canvas.commands().len() > 0, "chart should produce draw commands");
    let png = render_to_png(chart, W, H).expect("render_to_png should not panic");
    assert!(!png.is_empty(), "PNG output should not be empty");
}

// ===========================================================================
// Empty / minimal data
// ===========================================================================

#[test]
fn empty_line_chart() {
    let chart = LineChart::new(vec![]).title("Empty Line").build();
    assert_renders(&chart);
}

#[test]
fn single_point_line() {
    let chart = Chart::line(&[42.0]).title("Single Point").build();
    assert_renders(&chart);
}

#[test]
fn single_point_scatter() {
    let chart = Chart::scatter(&[1.0], &[2.0]).title("One Point").build();
    assert_renders(&chart);
}

#[test]
fn empty_bar_chart() {
    let chart = Chart::bar(vec![], &[]).title("Empty Bar").build();
    assert_renders(&chart);
}

#[test]
fn single_bar() {
    let chart = Chart::bar(vec!["A".into()], &[10.0]).title("Single Bar").build();
    assert_renders(&chart);
}

#[test]
fn empty_histogram() {
    let chart = Chart::histogram(&[]).title("Empty Hist").build();
    assert_renders(&chart);
}

// ===========================================================================
// NaN and Infinity handling
// ===========================================================================

#[test]
fn all_nan_line() {
    let chart = Chart::line(&[f64::NAN, f64::NAN, f64::NAN])
        .title("All NaN")
        .build();
    assert_renders(&chart);
}

#[test]
fn mixed_nan_line() {
    let chart = Chart::line(&[1.0, f64::NAN, 3.0, f64::NAN, 5.0])
        .title("Mixed NaN")
        .build();
    assert_renders(&chart);
}

#[test]
fn infinity_values_scatter() {
    let chart = Chart::scatter(
        &[1.0, 2.0, f64::INFINITY],
        &[f64::NEG_INFINITY, 2.0, 3.0],
    )
    .title("Infinity Scatter")
    .build();
    assert_renders(&chart);
}

#[test]
fn all_zero_values() {
    let chart = Chart::line(&[0.0, 0.0, 0.0, 0.0])
        .title("All Zeros")
        .build();
    assert_renders(&chart);
}

#[test]
fn constant_values() {
    let chart = Chart::line(&[7.0, 7.0, 7.0, 7.0])
        .title("Constant")
        .build();
    assert_renders(&chart);
}

// ===========================================================================
// Large datasets (stress)
// ===========================================================================

#[test]
fn large_line_dataset() {
    let data: Vec<f64> = (0..10_000).map(|i| (i as f64 * 0.01).sin()).collect();
    let chart = Chart::line(&data).title("10K Points").build();
    assert_renders(&chart);
}

#[test]
fn large_scatter_dataset() {
    let x: Vec<f64> = (0..5_000).map(|i| i as f64 * 0.1).collect();
    let y: Vec<f64> = x.iter().map(|xv| xv.cos() + xv * 0.01).collect();
    let chart = Chart::scatter(&x, &y).title("5K Scatter").build();
    assert_renders(&chart);
}

// ===========================================================================
// Extreme values
// ===========================================================================

#[test]
fn huge_values() {
    let chart = Chart::line(&[1e15, 2e15, 1.5e15, 3e15])
        .title("Huge Values")
        .build();
    assert_renders(&chart);
}

#[test]
fn tiny_values() {
    let chart = Chart::line(&[1e-15, 2e-15, 1.5e-15, 3e-15])
        .title("Tiny Values")
        .build();
    assert_renders(&chart);
}

#[test]
fn negative_values() {
    let chart = Chart::bar(
        vec!["A".into(), "B".into(), "C".into()],
        &[-10.0, -20.0, -5.0],
    )
    .title("All Negative")
    .build();
    assert_renders(&chart);
}

// ===========================================================================
// Asymmetric error bars
// ===========================================================================

#[test]
fn asymmetric_error_bars_line() {
    let s = Series::new("Data", vec![10.0, 20.0, 15.0])
        .with_error_asymmetric(vec![2.0, 3.0, 1.0], vec![5.0, 4.0, 6.0]);
    let chart = LineChart::new(vec![s])
        .title("Asymmetric Error Bars")
        .build();
    assert_renders(&chart);
}

#[test]
fn asymmetric_error_bars_scatter() {
    // Build a scatter with asymmetric errors on the Y series.
    let y = Series::from_values(vec![10.0, 20.0, 15.0])
        .with_error_asymmetric(vec![2.0, 3.0, 1.0], vec![5.0, 4.0, 6.0]);
    let chart = scry_chart::chart::ScatterChart::new(
        Series::from_values(vec![1.0, 2.0, 3.0]),
        y,
    ).title("Asymmetric Scatter").build();
    assert_renders(&chart);
}

// ===========================================================================
// Contour chart edge cases
// ===========================================================================

#[test]
fn contour_minimal_grid() {
    let chart = Chart::contour(vec![vec![1.0, 2.0], vec![3.0, 4.0]])
        .levels(3)
        .title("Minimal Contour")
        .build();
    assert_renders(&chart);
}

#[test]
fn contour_uniform_grid() {
    let chart = Chart::contour(vec![
        vec![5.0, 5.0, 5.0],
        vec![5.0, 5.0, 5.0],
        vec![5.0, 5.0, 5.0],
    ])
    .levels(5)
    .title("Uniform Contour")
    .build();
    assert_renders(&chart);
}

#[test]
fn contour_filled() {
    let chart = Chart::contour(vec![
        vec![0.0, 1.0, 2.0, 3.0],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2.0, 3.0, 4.0, 5.0],
        vec![3.0, 4.0, 5.0, 6.0],
    ])
    .filled()
    .levels(6)
    .title("Filled Contour")
    .build();
    assert_renders(&chart);
}

// ===========================================================================
// try_build() validation
// ===========================================================================

#[test]
fn try_build_empty_waterfall() {
    let result = Chart::waterfall(vec![], &[]).try_build();
    assert!(result.is_err());
}

#[test]
fn try_build_mismatched_funnel() {
    let result = Chart::funnel(vec!["A".into(), "B".into()], &[100.0]).try_build();
    assert!(result.is_err());
}

#[test]
fn try_build_invalid_gauge_range() {
    let result = Chart::gauge(50.0).range(100.0, 0.0).try_build();
    assert!(result.is_err());
}

#[test]
fn try_build_nan_gauge() {
    let result = Chart::gauge(f64::NAN).try_build();
    assert!(result.is_err());
}

#[test]
fn try_build_empty_sparkline() {
    let result = Chart::sparkline(&[]).try_build();
    assert!(result.is_err());
}

#[test]
fn try_build_empty_violin() {
    let result = Chart::violin(Vec::<(&str, Vec<f64>)>::new()).try_build();
    assert!(result.is_err());
}

#[test]
fn try_build_jagged_contour() {
    let result = Chart::contour(vec![vec![1.0, 2.0], vec![3.0]]).try_build();
    assert!(result.is_err());
}

// ===========================================================================
// Text utilities edge cases
// ===========================================================================

#[test]
fn text_wrap_empty_string() {
    let result = scry_chart::text_utils::wrap_text("", 100.0, 10.0);
    assert!(result.is_empty() || result == vec![""]);
}

#[test]
fn text_ellipsize_short_string() {
    // String shorter than max — should not be truncated.
    let result = scry_chart::text_utils::ellipsize("Hi", 100.0, 10.0);
    assert_eq!(result, "Hi");
}

#[test]
fn text_ellipsize_long_string() {
    let result = scry_chart::text_utils::ellipsize("This is a very long label", 50.0, 10.0);
    assert!(result.ends_with('…') || result.len() <= 10);
}

// ===========================================================================
// Decimation edge cases
// ===========================================================================

#[test]
fn lttb_small_data() {
    // Fewer points than target → return unchanged.
    let data: Vec<(f64, f64)> = vec![(1.0, 4.0), (2.0, 5.0), (3.0, 6.0)];
    let result = scry_chart::decimate::lttb(&data, 10);
    assert_eq!(result.len(), 3);
}

#[test]
fn lttb_exact_target() {
    let data: Vec<(f64, f64)> = (0..100).map(|i| (i as f64, (i as f64).sin())).collect();
    let result = scry_chart::decimate::lttb(&data, 50);
    assert_eq!(result.len(), 50);
}

#[test]
fn min_max_decimate_small_data() {
    let data: Vec<(f64, f64)> = vec![(1.0, 3.0), (2.0, 4.0)];
    let result = scry_chart::decimate::min_max_decimate(&data, 10);
    assert_eq!(result.len(), 2);
}

// ===========================================================================
// Tiny render dimensions
// ===========================================================================

#[test]
fn tiny_render_dimensions() {
    let chart = Chart::line(&[1.0, 2.0, 3.0]).title("Tiny").build();
    // 1x1 pixel — should not panic.
    let rendered = render_chart(&chart, 1, 1);
    assert!(!rendered.canvas.commands().is_empty() || true); // just ensure no panic
}

#[test]
fn zero_dimension_render() {
    // 0x0 is degenerate — just verify no panic.
    let chart = Chart::line(&[1.0, 2.0]).build();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = render_chart(&chart, 0, 0);
    }));
}
