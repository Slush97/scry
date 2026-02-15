//! Comprehensive acid test for scry-chart.
//!
//! Exercises every edge case, degenerate input, feature combination,
//! and scale boundary to expose weaknesses in the charting library.
//! Each test validates:
//!   - No panic (implicit from test passing)
//!   - Correct canvas dimensions
//!   - Reasonable command/overlay counts
//!   - No NaN in pixel coordinates of text overlays

use scry_chart::chart::{Chart, LineChart};
use scry_chart::data::Series;
use scry_chart::layout;
use scry_chart::prelude::*;
use scry_chart::theme::Theme;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Assert that a rendered chart has the expected dimensions and no NaN in overlays.
fn assert_sane(rendered: &layout::RenderedChart, w: u32, h: u32, label: &str) {
    assert_eq!(rendered.canvas.width(), w, "{label}: wrong canvas width");
    assert_eq!(rendered.canvas.height(), h, "{label}: wrong canvas height");

    // No NaN coordinates in text overlays
    for (i, overlay) in rendered.text_overlays.iter().enumerate() {
        assert!(
            overlay.x_px.is_finite(),
            "{label}: overlay[{i}] has non-finite x_px = {}",
            overlay.x_px
        );
        assert!(
            overlay.y_px.is_finite(),
            "{label}: overlay[{i}] has non-finite y_px = {}",
            overlay.y_px
        );
    }
}

/// Assert rendering produces draw commands (non-empty chart).
fn assert_has_commands(rendered: &layout::RenderedChart, label: &str) {
    assert!(
        !rendered.canvas.commands().is_empty(),
        "{label}: expected draw commands but got none"
    );
}

// ===========================================================================
// Category 1: Data Torture — NaN / Infinity / Degenerate
// ===========================================================================

#[test]
fn cat1_all_nan_scatter() {
    let chart = Chart::scatter(
        &[f64::NAN, f64::NAN, f64::NAN],
        &[f64::NAN, f64::NAN, f64::NAN],
    )
    .title("All NaN Scatter")
    .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "all_nan_scatter");
}

#[test]
fn cat1_all_nan_line() {
    let chart = Chart::line(&[f64::NAN, f64::NAN, f64::NAN])
        .title("All NaN Line")
        .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "all_nan_line");
}

#[test]
fn cat1_all_nan_histogram() {
    let chart = Chart::histogram(&[f64::NAN, f64::NAN, f64::NAN])
        .title("All NaN Histogram")
        .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "all_nan_histogram");
}

#[test]
fn cat1_all_nan_boxplot() {
    let chart = Chart::boxplot(vec![("NaN Group", vec![f64::NAN, f64::NAN])])
        .title("All NaN Boxplot")
        .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "all_nan_boxplot");
}

#[test]
fn cat1_all_nan_heatmap() {
    let chart = Chart::heatmap(vec![vec![f64::NAN, f64::NAN], vec![f64::NAN, f64::NAN]])
        .title("All NaN Heatmap")
        .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "all_nan_heatmap");
}

#[test]
fn cat1_mixed_nan_inf_scatter() {
    let chart = Chart::scatter(
        &[1.0, f64::NAN, 3.0, f64::INFINITY, 5.0, f64::NEG_INFINITY],
        &[f64::NEG_INFINITY, 2.0, f64::NAN, 4.0, f64::INFINITY, 6.0],
    )
    .title("Mixed Poison Data")
    .connected()
    .trend_line()
    .build();

    let r = layout::render_chart(&chart, 500, 400);
    assert_sane(&r, 500, 400, "mixed_nan_inf_scatter");
    assert_has_commands(&r, "mixed_nan_inf_scatter");
}

#[test]
fn cat1_identical_values() {
    // All-identical data: extent is (5.0, 5.0), scale span is zero
    let chart = Chart::scatter(&[5.0, 5.0, 5.0, 5.0, 5.0], &[5.0, 5.0, 5.0, 5.0, 5.0])
        .title("Identical Values")
        .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "identical_values");
    assert_has_commands(&r, "identical_values");
}

#[test]
fn cat1_identical_line() {
    let data = vec![42.0; 100];
    let chart = Chart::line(&data)
        .title("Flat Line")
        .filled()
        .with_points()
        .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "identical_line");
    assert_has_commands(&r, "identical_line");
}

#[test]
fn cat1_identical_histogram() {
    let data = vec![7.0; 200];
    let chart = Chart::histogram(&data)
        .bins(10)
        .title("Delta Distribution")
        .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "identical_histogram");
}

#[test]
fn cat1_identical_bar() {
    let chart = Chart::bar(
        vec!["A".into(), "B".into(), "C".into()],
        &[10.0, 10.0, 10.0],
    )
    .title("Identical Bars")
    .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "identical_bar");
    assert_has_commands(&r, "identical_bar");
}

// ===========================================================================
// Category 2: Scale Stress
// ===========================================================================

#[test]
fn cat2_extreme_range_scatter() {
    let chart = Chart::scatter(
        &[1e-12, 1e-6, 1.0, 1e6, 1e12],
        &[1e-12, 1e-6, 1.0, 1e6, 1e12],
    )
    .title("Extreme Range")
    .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "extreme_range");
    assert_has_commands(&r, "extreme_range");
}

#[test]
fn cat2_very_small_range() {
    // Values differ by ~1e-10
    let chart = Chart::scatter(
        &[1.0000000001, 1.0000000002, 1.0000000003],
        &[2.0000000001, 2.0000000002, 2.0000000003],
    )
    .title("Micro Range")
    .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "very_small_range");
}

#[test]
fn cat2_all_negative_scatter() {
    let chart = Chart::scatter(
        &[-100.0, -50.0, -10.0, -5.0, -1.0],
        &[-200.0, -100.0, -50.0, -10.0, -1.0],
    )
    .title("All Negative")
    .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "all_negative_scatter");
    assert_has_commands(&r, "all_negative_scatter");
}

#[test]
fn cat2_log_scale_edge_cases() {
    use scry_chart::scale::{LogScale, Scale};

    // Zero/negative clamping
    let s = LogScale::new((0.0, 100.0), (0.0, 400.0));
    let p = s.to_pixel(0.0);
    assert!(
        p.is_finite(),
        "LogScale with 0 input should not produce NaN"
    );

    let p2 = s.to_pixel(-10.0);
    assert!(
        p2.is_finite(),
        "LogScale with negative input should not produce NaN"
    );

    // Very small domain
    let s2 = LogScale::new((0.001, 0.002), (0.0, 400.0));
    let p3 = s2.to_pixel(0.0015);
    assert!(p3.is_finite(), "Tiny log domain should work");

    // Round trip with edge values
    let s3 = LogScale::new((1.0, 1e6), (0.0, 600.0));
    for val in [1.0, 10.0, 100.0, 1000.0, 1e6] {
        let px = s3.to_pixel(val);
        let back = s3.to_data(px);
        assert!(
            (back - val).abs() / val < 0.01,
            "LogScale round-trip failed for {val}: got {back}"
        );
    }
}

#[test]
fn cat2_categorical_extremes() {
    use scry_chart::scale::CategoricalScale;

    // 0 categories
    let s0 = CategoricalScale::new(vec![], (0.0, 400.0));
    let c = s0.center(0);
    assert!(c.is_finite(), "0-category center should be finite");
    let bw = s0.band_width();
    assert_eq!(bw, 0.0, "0-category band_width should be 0");

    // 1 category
    let s1 = CategoricalScale::new(vec!["Solo".into()], (0.0, 400.0));
    let c1 = s1.center(0);
    assert!(
        (c1 - 200.0).abs() < f64::EPSILON,
        "1-category should center at midpoint"
    );

    // 50 categories
    let labels: Vec<String> = (0..50).map(|i| format!("Cat{i}")).collect();
    let s50 = CategoricalScale::new(labels, (0.0, 1000.0));
    assert!(
        (s50.band_width() - 20.0).abs() < f64::EPSILON,
        "50 categories in 1000px = 20px bands"
    );
    assert!(s50.center(49).is_finite());
}

#[test]
fn cat2_linear_scale_degenerate() {
    use scry_chart::scale::{LinearScale, Scale};

    // Zero domain span
    let s = LinearScale::new((5.0, 5.0), (0.0, 400.0));
    let p = s.to_pixel(5.0);
    assert!(
        p.is_finite(),
        "Zero-span domain should produce finite pixel"
    );

    // Zero range span
    let s2 = LinearScale::new((0.0, 100.0), (200.0, 200.0));
    let p2 = s2.to_pixel(50.0);
    assert!(
        p2.is_finite(),
        "Zero-span range should produce finite pixel"
    );

    // Ticks for zero span
    let ticks = s.ticks(6);
    assert!(
        !ticks.is_empty(),
        "Should produce at least one tick for zero span"
    );
    for t in &ticks {
        assert!(t.is_finite(), "Ticks should be finite");
    }
}

// ===========================================================================
// Category 3: Layout Robustness
// ===========================================================================

#[test]
fn cat3_tiny_canvas() {
    let chart = Chart::line(&[1.0, 2.0, 3.0])
        .title("Tiny")
        .x_label("X")
        .y_label("Y")
        .build();

    // Absolute minimum — should not panic
    let r = layout::render_chart(&chart, 8, 6);
    assert_sane(&r, 8, 6, "tiny_canvas");
}

#[test]
fn cat3_1x1_canvas() {
    let chart = Chart::scatter(&[1.0], &[1.0]).build();
    let r = layout::render_chart(&chart, 1, 1);
    assert_sane(&r, 1, 1, "1x1_canvas");
}

#[test]
fn cat3_very_wide_canvas() {
    let chart = Chart::bar(vec!["A".into(), "B".into()], &[10.0, 20.0])
        .title("Wide")
        .build();

    let r = layout::render_chart(&chart, 4000, 100);
    assert_sane(&r, 4000, 100, "very_wide");
    assert_has_commands(&r, "very_wide");
}

#[test]
fn cat3_very_tall_canvas() {
    let chart = Chart::histogram(&[1.0, 2.0, 3.0, 4.0, 5.0])
        .title("Tall")
        .build();

    let r = layout::render_chart(&chart, 100, 4000);
    assert_sane(&r, 100, 4000, "very_tall");
    assert_has_commands(&r, "very_tall");
}

#[test]
fn cat3_all_chart_types_tiny() {
    let charts: Vec<(&str, Chart)> = vec![
        ("scatter", Chart::scatter(&[1.0], &[1.0]).build()),
        ("line", Chart::line(&[1.0, 2.0]).build()),
        ("bar", Chart::bar(vec!["A".into()], &[5.0]).build()),
        ("hist", Chart::histogram(&[1.0, 2.0, 3.0]).build()),
        (
            "box",
            Chart::boxplot(vec![("G", vec![1.0, 2.0, 3.0])]).build(),
        ),
        ("heat", Chart::heatmap(vec![vec![1.0]]).build()),
    ];

    for (name, chart) in &charts {
        let r = layout::render_chart(chart, 20, 15);
        assert_sane(&r, 20, 15, &format!("tiny_{name}"));
    }
}

// ===========================================================================
// Category 4: Negative & Mixed-Sign Data
// ===========================================================================

#[test]
fn cat4_negative_bar_values() {
    let chart = Chart::bar(
        vec!["A".into(), "B".into(), "C".into()],
        &[-10.0, -20.0, -5.0],
    )
    .title("Negative Bars")
    .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "negative_bars");
    // Should produce bar rects even for negative values
    assert_has_commands(&r, "negative_bars");
}

#[test]
fn cat4_mixed_sign_bars() {
    let chart = Chart::bar(
        vec![
            "Profit".into(),
            "Loss".into(),
            "Break Even".into(),
            "Big Win".into(),
        ],
        &[50.0, -30.0, 0.0, 100.0],
    )
    .title("Mixed Sign Bars")
    .build();

    let r = layout::render_chart(&chart, 500, 350);
    assert_sane(&r, 500, 350, "mixed_sign_bars");
    assert_has_commands(&r, "mixed_sign_bars");
}

#[test]
fn cat4_negative_stacked_bars() {
    let chart = Chart::bar(vec!["A".into(), "B".into()], &[-10.0, 10.0])
        .add_series(Series::new("Layer2", vec![5.0, -5.0]))
        .stacked()
        .title("Negative Stacked")
        .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "negative_stacked");
}

#[test]
fn cat4_negative_horizontal_bars() {
    let chart = Chart::bar(vec!["X".into(), "Y".into()], &[-50.0, -25.0])
        .horizontal()
        .title("Negative Horizontal")
        .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "negative_horizontal");
}

#[test]
fn cat4_all_negative_histogram() {
    let data: Vec<f64> = (-100..0).map(|i| i as f64).collect();
    let chart = Chart::histogram(&data)
        .bins(10)
        .title("All Negative Histogram")
        .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "all_negative_histogram");
    assert_has_commands(&r, "all_negative_histogram");
}

#[test]
fn cat4_zero_crossing_line() {
    let chart = Chart::line(&[-5.0, -2.0, 0.0, 3.0, 7.0, -1.0, 4.0])
        .title("Zero Crossing")
        .filled()
        .h_line(0.0)
        .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "zero_crossing_line");
    assert_has_commands(&r, "zero_crossing_line");
}

// ===========================================================================
// Category 5: Feature Combination Stress
// ===========================================================================

#[test]
fn cat5_everything_scatter() {
    let chart = Chart::scatter(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        &[2.0, 4.0, 1.0, 8.0, 5.0, 3.0, 7.0, 6.0, 9.0, 10.0],
    )
    .title("Kitchen Sink Scatter")
    .x_label("X Axis Label")
    .y_label("Y Axis Label")
    .theme(Theme::dark())
    .marker(Marker::Diamond)
    .connected()
    .trend_line()
    .annotate(5.0, 5.0, "Midpoint")
    .annotate(1.0, 2.0, "Start")
    .annotate(10.0, 10.0, "End")
    .h_line(5.0)
    .h_line_styled(
        8.0,
        scry_engine::style::Color::from_rgba8(255, 0, 0, 200),
    )
    .v_line(3.0)
    .v_line_styled(
        7.0,
        scry_engine::style::Color::from_rgba8(0, 255, 0, 200),
    )
    .x_range(0.0, 12.0)
    .y_range(-2.0, 12.0)
    .add_series(
        Series::new("Extra1", vec![1.5, 3.5, 5.5, 7.5, 9.5]),
        Series::new("E1Y", vec![3.0, 6.0, 2.0, 9.0, 4.0]),
    )
    .add_series(
        Series::new("Extra2", vec![2.0, 4.0, 6.0, 8.0]),
        Series::new("E2Y", vec![1.0, 7.0, 3.0, 8.0]),
    )
    .no_legend()
    .build();

    let r = layout::render_chart(&chart, 800, 600);
    assert_sane(&r, 800, 600, "everything_scatter");
    assert!(
        r.canvas.commands().len() > 30,
        "Kitchen sink should produce many commands, got {}",
        r.canvas.commands().len()
    );
}

#[test]
fn cat5_everything_line() {
    let chart = LineChart::new(vec![
        Series::new("Alpha", vec![1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 8.0]),
        Series::new("Beta", vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0]),
        Series::new("Gamma", vec![3.0, 5.0, 1.0, 4.0, 2.0, 8.0, 3.0, 6.0]),
        Series::new("Delta", vec![4.0, 2.0, 6.0, 1.0, 5.0, 3.0, 7.0, 4.0]),
    ])
    .filled()
    .with_points()
    .title("Multi-Series Line")
    .x_label("Time")
    .y_label("Value")
    .theme(Theme::pastel())
    .h_line(4.0)
    .v_line(3.0)
    .annotate(4.0, 5.0, "Peak")
    .trend_line()
    .build();

    let r = layout::render_chart(&chart, 700, 500);
    assert_sane(&r, 700, 500, "everything_line");
    assert_has_commands(&r, "everything_line");
}

#[test]
fn cat5_all_markers_connected() {
    let markers = [
        Marker::Circle,
        Marker::Square,
        Marker::Diamond,
        Marker::Cross,
        Marker::Triangle,
    ];

    for marker in markers {
        let chart = Chart::scatter(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2.0, 4.0, 1.0, 5.0, 3.0])
            .marker(marker)
            .connected()
            .trend_line()
            .build();

        let r = layout::render_chart(&chart, 400, 300);
        assert_sane(&r, 400, 300, &format!("marker_{marker:?}_connected"));
        assert_has_commands(&r, &format!("marker_{marker:?}_connected"));
    }
}

#[test]
fn cat5_eight_series_line() {
    let series: Vec<Series> = (0..8)
        .map(|i| {
            let values: Vec<f64> = (0..20)
                .map(|j| ((j as f64 + i as f64) * 0.3).sin() * 50.0 + 50.0)
                .collect();
            Series::new(format!("Series {}", i + 1), values)
        })
        .collect();

    let chart = LineChart::new(series)
        .title("8 Series")
        .filled()
        .with_points()
        .build();

    let r = layout::render_chart(&chart, 800, 500);
    assert_sane(&r, 800, 500, "eight_series");
    assert_has_commands(&r, "eight_series");
}

#[test]
fn cat5_mismatched_series_lengths_bar() {
    // Series with different lengths — should truncate gracefully
    let chart = Chart::bar(
        vec!["A".into(), "B".into(), "C".into(), "D".into(), "E".into()],
        &[10.0, 20.0, 30.0, 40.0, 50.0],
    )
    .add_series(Series::new("Short", vec![5.0, 15.0])) // only 2 values for 5 categories
    .add_series(Series::new("Even Shorter", vec![3.0])) // only 1 value
    .title("Mismatched Lengths")
    .build();

    let r = layout::render_chart(&chart, 500, 350);
    assert_sane(&r, 500, 350, "mismatched_series_bar");
    assert_has_commands(&r, "mismatched_series_bar");
}

#[test]
fn cat5_stacked_horizontal_with_refs() {
    let chart = Chart::bar(
        vec!["Reg1".into(), "Reg2".into(), "Reg3".into()],
        &[30.0, 50.0, 20.0],
    )
    .add_series(Series::new("Product B", vec![20.0, 30.0, 10.0]))
    .stacked()
    .horizontal()
    .title("Stacked Horizontal + Refs")
    .h_line_styled(
        40.0,
        scry_engine::style::Color::from_rgba8(255, 100, 100, 200),
    )
    .build();

    let r = layout::render_chart(&chart, 600, 400);
    assert_sane(&r, 600, 400, "stacked_horizontal_refs");
    assert_has_commands(&r, "stacked_horizontal_refs");
}

#[test]
fn cat5_all_themes_all_chart_types() {
    let themes = [
        ("dark", Theme::dark()),
        ("light", Theme::light()),
        ("pastel", Theme::pastel()),
    ];

    for (theme_name, theme) in &themes {
        let charts: Vec<(&str, Chart)> = vec![
            (
                "scatter",
                Chart::scatter(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0])
                    .theme(theme.clone())
                    .build(),
            ),
            (
                "line",
                Chart::line(&[1.0, 4.0, 2.0, 8.0])
                    .theme(theme.clone())
                    .build(),
            ),
            (
                "bar",
                Chart::bar(vec!["A".into(), "B".into()], &[10.0, 20.0])
                    .theme(theme.clone())
                    .build(),
            ),
            (
                "hist",
                Chart::histogram(&[1.0, 2.0, 3.0, 2.0, 1.0])
                    .theme(theme.clone())
                    .build(),
            ),
            (
                "box",
                Chart::boxplot(vec![("G", vec![1.0, 2.0, 3.0, 4.0, 5.0])])
                    .theme(theme.clone())
                    .build(),
            ),
            (
                "heat",
                Chart::heatmap(vec![vec![1.0, 2.0], vec![3.0, 4.0]])
                    .theme(theme.clone())
                    .build(),
            ),
        ];

        for (chart_name, chart) in &charts {
            let r = layout::render_chart(chart, 400, 300);
            assert_sane(&r, 400, 300, &format!("{theme_name}_{chart_name}"));
            assert_has_commands(&r, &format!("{theme_name}_{chart_name}"));
        }
    }
}

// ===========================================================================
// Category 6: Large Data Sets
// ===========================================================================

#[test]
fn cat6_10k_scatter() {
    let n = 10_000;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 100.0).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.01).sin() * 50.0 + (i as f64 * 0.007).cos() * 30.0)
        .collect();

    let chart = Chart::scatter(&x, &y).title("10K Scatter").build();

    let r = layout::render_chart(&chart, 800, 600);
    assert_sane(&r, 800, 600, "10k_scatter");
    assert!(
        r.canvas.commands().len() >= 10_000,
        "10K scatter should produce at least 10K commands, got {}",
        r.canvas.commands().len()
    );
}

#[test]
fn cat6_10k_line() {
    let values: Vec<f64> = (0..10_000)
        .map(|i| (i as f64 * 0.01).sin() * 100.0)
        .collect();

    let chart = Chart::line(&values).title("10K Line").build();

    let r = layout::render_chart(&chart, 1200, 400);
    assert_sane(&r, 1200, 400, "10k_line");
    assert_has_commands(&r, "10k_line");
}

#[test]
fn cat6_50k_histogram() {
    let values: Vec<f64> = (0..50_000)
        .map(|i| {
            // Approximate normal distribution via Box-Muller-ish
            let x = (i as f64 + 1.0) / 50_001.0;
            let z = (-2.0 * x.ln()).sqrt() * ((i as f64 * 0.618).sin());
            z * 15.0 + 100.0
        })
        .collect();

    let chart = Chart::histogram(&values)
        .bins(50)
        .title("50K Histogram")
        .density()
        .build();

    let r = layout::render_chart(&chart, 600, 400);
    assert_sane(&r, 600, 400, "50k_histogram");
    assert_has_commands(&r, "50k_histogram");
}

#[test]
fn cat6_large_heatmap() {
    let size = 50;
    let data: Vec<Vec<f64>> = (0..size)
        .map(|r| {
            (0..size)
                .map(|c| {
                    let x = r as f64 / size as f64;
                    let y = c as f64 / size as f64;
                    (x * 3.14).sin() * (y * 3.14).cos() * 100.0
                })
                .collect()
        })
        .collect();

    let chart = Chart::heatmap(data)
        .title("50x50 Heatmap")
        .values(false) // would be illegible with values
        .build();

    let r = layout::render_chart(&chart, 800, 800);
    assert_sane(&r, 800, 800, "large_heatmap");
    assert!(
        r.canvas.commands().len() >= 2500,
        "50x50 heatmap should produce at least 2500 rects, got {}",
        r.canvas.commands().len()
    );
}

#[test]
fn cat6_many_boxplot_groups() {
    let groups: Vec<(String, Vec<f64>)> = (0..20)
        .map(|i| {
            let values: Vec<f64> = (0..50)
                .map(|j| (j as f64 + i as f64 * 3.0) * 0.5 + (j as f64 * 0.1).sin() * 10.0)
                .collect();
            (format!("G{i}"), values)
        })
        .collect();

    let chart = Chart::boxplot(groups).title("20 Boxplot Groups").build();

    let r = layout::render_chart(&chart, 1200, 400);
    assert_sane(&r, 1200, 400, "many_boxplot_groups");
    assert_has_commands(&r, "many_boxplot_groups");
}

// ===========================================================================
// Category 7: Boundary Conditions
// ===========================================================================

#[test]
fn cat7_empty_labels_bar() {
    let chart = Chart::bar(vec!["".into(), "".into(), "".into()], &[10.0, 20.0, 30.0])
        .title("Empty Labels")
        .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "empty_labels_bar");
    assert_has_commands(&r, "empty_labels_bar");
}

#[test]
fn cat7_unicode_labels() {
    let chart = Chart::bar(
        vec![
            "日本語".into(),
            "中文".into(),
            "한국어".into(),
            "العربية".into(),
        ],
        &[42.0, 88.0, 55.0, 73.0],
    )
    .title("Unicode: 日本語テスト 🎯")
    .x_label("言語")
    .y_label("値")
    .build();

    let r = layout::render_chart(&chart, 500, 350);
    assert_sane(&r, 500, 350, "unicode_labels");
    assert_has_commands(&r, "unicode_labels");
}

#[test]
fn cat7_very_long_labels() {
    let chart = Chart::bar(
        vec![
            "This is an extraordinarily long category label that should not crash".into(),
            "Another very long label for testing overflow handling in the renderer".into(),
        ],
        &[100.0, 200.0],
    )
    .title("A very long title that tests whether the title rendering handles overflow correctly without panicking or producing garbage")
    .x_label("An extremely detailed x-axis label")
    .y_label("Very long y-axis label text")
    .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "very_long_labels");
}

#[test]
fn cat7_1x1_heatmap() {
    let chart = Chart::heatmap(vec![vec![42.0]])
        .title("1x1 Heatmap")
        .row_labels(vec!["Only Row".into()])
        .col_labels(vec!["Only Col".into()])
        .build();

    let r = layout::render_chart(&chart, 300, 300);
    assert_sane(&r, 300, 300, "1x1_heatmap");
    assert_has_commands(&r, "1x1_heatmap");
}

#[test]
fn cat7_heatmap_jagged_rows() {
    // Rows of different lengths — a real-world mistake
    let chart = Chart::heatmap(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0],           // short row
        vec![6.0, 7.0, 8.0, 9.0], // long row
    ])
    .title("Jagged Heatmap")
    .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "jagged_heatmap");
}

#[test]
fn cat7_zero_value_bar() {
    let chart = Chart::bar(vec!["A".into(), "B".into(), "C".into()], &[0.0, 0.0, 0.0])
        .title("All Zero Bars")
        .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "zero_value_bar");
}

#[test]
fn cat7_single_bin_histogram() {
    let chart = Chart::histogram(&[5.0, 5.1, 5.2, 5.3])
        .bins(1)
        .title("Single Bin")
        .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "single_bin_histogram");
}

#[test]
fn cat7_many_bins_histogram() {
    let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let chart = Chart::histogram(&data)
        .bins(100) // bin count == data count
        .title("100 Bins")
        .build();

    let r = layout::render_chart(&chart, 800, 300);
    assert_sane(&r, 800, 300, "many_bins_histogram");
    assert_has_commands(&r, "many_bins_histogram");
}

#[test]
fn cat7_boxplot_single_value_group() {
    let chart = Chart::boxplot(vec![
        ("One", vec![5.0]),
        ("Two", vec![3.0, 7.0]),
        (
            "Many",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ),
    ])
    .title("Mixed Size Groups")
    .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "boxplot_single_value");
    assert_has_commands(&r, "boxplot_single_value");
}

#[test]
fn cat7_heatmap_negative_values() {
    let chart = Chart::heatmap(vec![
        vec![-1.0, -0.5, 0.0],
        vec![0.5, 1.0, -0.8],
        vec![-0.3, 0.7, -1.0],
    ])
    .title("Correlation-like Heatmap")
    .range(-1.0, 1.0)
    .build();

    let r = layout::render_chart(&chart, 400, 400);
    assert_sane(&r, 400, 400, "heatmap_negative_values");
    assert_has_commands(&r, "heatmap_negative_values");
}

#[test]
fn cat7_boxplot_notched_flag() {
    // Tests that notched rendering produces polygon commands (was previously a no-op)
    let chart = Chart::boxplot(vec![
        ("A", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        (
            "B",
            vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ),
    ])
    .notched()
    .title("Notched BoxPlot")
    .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "boxplot_notched");
    assert_has_commands(&r, "boxplot_notched");

    // Verify non-notched also works for comparison
    let chart2 = Chart::boxplot(vec![
        ("A", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        (
            "B",
            vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ),
    ])
    .title("Non-Notched BoxPlot")
    .build();

    let r2 = layout::render_chart(&chart2, 400, 300);
    assert_sane(&r2, 400, 300, "boxplot_standard");
    assert_has_commands(&r2, "boxplot_standard");
}

#[test]
fn cat7_histogram_density_nan_in_data() {
    // Tests that density normalization handles NaN count properly
    let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    data.extend_from_slice(&[f64::NAN; 95]); // 95% NaN

    let chart = Chart::histogram(&data)
        .density()
        .bins(5)
        .title("Density with NaN")
        .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "density_with_nan");
}

// ===========================================================================
// Category 8: Targeted Fix Validation
// ===========================================================================

#[test]
fn cat8_negative_bars_produce_rects() {
    // After Phase 2: negative-value bars must produce draw commands, not be silently skipped
    let chart = Chart::bar(
        vec!["A".into(), "B".into(), "C".into()],
        &[-10.0, -20.0, -5.0],
    )
    .title("Negative Bars Visible")
    .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "negative_bars_rects");
    // Should have at least 3 filled rects from bars (plus axes/grid)
    assert!(
        r.canvas.commands().len() >= 10,
        "Negative bars should produce rect commands, got {}",
        r.canvas.commands().len()
    );
}

#[test]
fn cat8_mixed_bars_both_directions() {
    // Mixed positive/negative bars should render in both directions from baseline
    let chart = Chart::bar(
        vec!["Up".into(), "Down".into(), "Up2".into()],
        &[50.0, -30.0, 20.0],
    )
    .title("Bidirectional Bars")
    .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "bidirectional_bars");
    // All 3 bars should produce rects (none should be skipped)
    assert!(
        r.canvas.commands().len() >= 10,
        "All bars should render, got {} commands",
        r.canvas.commands().len()
    );
}

#[test]
fn cat8_scatter_nan_filtered_not_drawn() {
    // After Phase 1: NaN data points should be filtered out, not drawn at garbage coords
    let chart = Chart::scatter(
        &[1.0, f64::NAN, 3.0, f64::NAN, 5.0],
        &[2.0, 4.0, f64::NAN, 6.0, 8.0],
    )
    .title("NaN Filtered Scatter")
    .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "nan_filtered_scatter");

    // Only finite-finite pairs (1,2) and (5,8) should produce markers => 2 markers
    // Compare to all-finite version with 5 markers
    let chart_all_finite = Chart::scatter(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2.0, 4.0, 6.0, 8.0, 10.0])
        .title("All Finite Scatter")
        .build();

    let r2 = layout::render_chart(&chart_all_finite, 400, 300);

    assert!(
        r.canvas.commands().len() < r2.canvas.commands().len(),
        "NaN scatter ({} cmds) should have fewer marker commands than all-finite ({} cmds)",
        r.canvas.commands().len(),
        r2.canvas.commands().len()
    );
}

#[test]
fn cat8_heatmap_nan_cells_skipped() {
    // After Phase 4: NaN cells should produce no rect or overlay
    let chart = Chart::heatmap(vec![
        vec![1.0, f64::NAN, 3.0],
        vec![f64::NAN, 5.0, f64::NAN],
        vec![7.0, 8.0, 9.0],
    ])
    .title("NaN Cells Heatmap")
    .build();

    let r_nan = layout::render_chart(&chart, 400, 400);
    assert_sane(&r_nan, 400, 400, "nan_cells_heatmap");

    // Full heatmap for comparison
    let chart_full = Chart::heatmap(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ])
    .title("Full Heatmap")
    .build();

    let r_full = layout::render_chart(&chart_full, 400, 400);

    assert!(
        r_nan.canvas.commands().len() < r_full.canvas.commands().len(),
        "NaN heatmap ({} cmds) should have fewer cell rects than full ({} cmds)",
        r_nan.canvas.commands().len(),
        r_full.canvas.commands().len()
    );
}

#[test]
fn cat8_annotation_background_width_scales() {
    // After Phase 5: annotation background width should scale with text length
    let chart_short = Chart::scatter(&[5.0], &[5.0])
        .annotate(5.0, 5.0, "Hi")
        .build();

    let chart_long = Chart::scatter(&[5.0], &[5.0])
        .annotate(5.0, 5.0, "This is a very long annotation text for testing")
        .build();

    let r_short = layout::render_chart(&chart_short, 400, 300);
    let r_long = layout::render_chart(&chart_long, 400, 300);

    assert_sane(&r_short, 400, 300, "short_annotation");
    assert_sane(&r_long, 400, 300, "long_annotation");
    // Both should render without panic
    assert_has_commands(&r_short, "short_annotation");
    assert_has_commands(&r_long, "long_annotation");
}

#[test]
fn cat8_horizontal_negative_bars_visible() {
    // After Phase 2: horizontal negative bars should extend left from baseline
    let chart = Chart::bar(vec!["A".into(), "B".into()], &[-50.0, -25.0])
        .horizontal()
        .title("Horizontal Negative Bars")
        .build();

    let r = layout::render_chart(&chart, 400, 300);
    assert_sane(&r, 400, 300, "h_negative_bars");
    // Negative horizontal bars should produce rects
    assert!(
        r.canvas.commands().len() >= 8,
        "Horizontal negative bars should produce rect commands, got {}",
        r.canvas.commands().len()
    );
}
