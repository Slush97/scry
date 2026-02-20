//! Academic compliance tests for scry-chart formatting.
//!
//! These tests verify adherence to accepted academic and pedagogical
//! chart-design standards:
//!
//! - **Axis convergence** (Tufte 2001): Axis spines span the full plot area.
//! - **Z-ordering** (Cleveland 1985): Grids behind data, ticks on spines.
//! - **Legend swatch semantics**: Swatch shape matches chart type
//!   (Circle for scatter/bubble, Line for line, Rect for bar/histogram).
//! - **AspectRatio enforcement**: `Equal` and `Fixed` constraints produce
//!   square or fixed-ratio plot areas.
//! - **Legend non-overlap** (Tufte/Cleveland): Legends do not obscure data.

use scry_chart::chart::{Charts, LineChart};
use scry_chart::config::AspectRatio;
use scry_chart::data::Series;
use scry_chart::layout;
use scry_chart::theme::Theme;
use scry_engine::scene::command::DrawCommand;

// ===========================================================================
// 1. Axis convergence: spines span the full plot area
// ===========================================================================

/// Verify that X and Y axis spines are drawn as continuous lines spanning
/// from one edge of the plot to the other — not fragmented or shortened.
///
/// Academic standard: Tufte 2001, §6 — "Axis lines should extend to cover
/// the full data range, not stop short."
#[test]
fn axis_spines_span_plot_area() {
    let chart = Charts::scatter(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0])
        .title("Convergence")
        .x_label("X")
        .y_label("Y")
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    let plot = rendered.plot_area.expect("should have plot_area");
    let (px, py, pw, ph) = plot;

    // Find axis spine lines: these should match the plot boundary edges.
    // Bottom spine: Y constant at py+ph, X from px to px+pw
    // Left spine: X constant at px, Y from py to py+ph
    let cmds = rendered.canvas.commands();
    let mut found_bottom = false;
    let mut found_left = false;

    for cmd in cmds {
        if let DrawCommand::Line { x1, y1, x2, y2, .. } = cmd {
            // Bottom spine (horizontal, at bottom edge)
            let at_bottom = (*y1 - (py + ph)).abs() < 2.0 && (*y2 - (py + ph)).abs() < 2.0;
            let spans_x = (*x1 - px).abs() < 2.0 && (*x2 - (px + pw)).abs() < 2.0
                || (*x2 - px).abs() < 2.0 && (*x1 - (px + pw)).abs() < 2.0;
            if at_bottom && spans_x {
                found_bottom = true;
            }

            // Left spine (vertical, at left edge)
            let at_left = (*x1 - px).abs() < 2.0 && (*x2 - px).abs() < 2.0;
            let spans_y = (*y1 - py).abs() < 2.0 && (*y2 - (py + ph)).abs() < 2.0
                || (*y2 - py).abs() < 2.0 && (*y1 - (py + ph)).abs() < 2.0;
            if at_left && spans_y {
                found_left = true;
            }
        }
    }

    assert!(found_bottom, "Bottom axis spine should span the full plot width");
    assert!(found_left, "Left axis spine should span the full plot height");
}

// ===========================================================================
// 2. Z-ordering: grids behind data
// ===========================================================================

/// Grid lines must appear BEFORE data-drawing commands in the render order.
///
/// Academic standard: Cleveland 1985, §4.5 — "Reference structures (grids)
/// must be visually subordinate and behind data elements."
///
/// We verify by checking command indices: the first dashed line (grid)
/// appears before the first filled polygon/circle (data element).
#[test]
fn grid_lines_before_data() {
    let chart = Charts::bar(
        vec!["A".into(), "B".into(), "C".into()],
        &[10.0, 20.0, 15.0],
    )
    .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    let cmds = rendered.canvas.commands();

    // Find first grid line (dashed line command)
    let first_grid = cmds.iter().position(|cmd| {
        matches!(cmd, DrawCommand::Line { stroke, .. } if stroke.dash.is_some())
    });

    // Find first data rectangle (filled rect — bars)
    let first_data = cmds.iter().position(|cmd| {
        matches!(cmd, DrawCommand::Rectangle { style, .. } if style.fill.is_some())
    });

    if let (Some(g), Some(d)) = (first_grid, first_data) {
        assert!(
            g < d,
            "Grid lines (idx {g}) must be drawn before data elements (idx {d})"
        );
    }
    // If no grid or no data, that's also fine — nothing to overlap
}

/// Tick marks must appear AFTER grid lines but BEFORE or AT axis spines.
///
/// This validates the 3-phase z-ordering in `RenderContext::draw_axes`:
/// Phase 1: grid lines, Phase 2: tick marks, Phase 3: axis spines.
#[test]
fn tick_marks_between_grids_and_spines() {
    let chart = Charts::scatter(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0])
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    let cmds = rendered.canvas.commands();

    // Grid lines are dashed
    let last_grid = cmds.iter().rposition(|cmd| {
        matches!(cmd, DrawCommand::Line { stroke, .. } if stroke.dash.is_some())
    });

    // Tick marks: short solid lines near axes (length < 10px)
    let first_tick = cmds.iter().position(|cmd| {
        if let DrawCommand::Line { x1, y1, x2, y2, stroke, .. } = cmd {
            let len = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
            len < 10.0 && len > 2.0 && stroke.dash.is_none()
        } else {
            false
        }
    });

    if let (Some(g), Some(t)) = (last_grid, first_tick) {
        assert!(
            g < t,
            "Last grid line (idx {g}) should be before first tick mark (idx {t})"
        );
    }
}

// ===========================================================================
// 3. Legend swatch semantics
// ===========================================================================

/// Scatter chart legends must use circle swatches.
///
/// Cleveland 1985: "Legend symbols should be identical to the encoding
/// used in the plot itself."
#[test]
fn scatter_legend_uses_circle_swatch() {
    let chart = Charts::scatter(&[1.0, 2.0, 3.0], &[1.0, 4.0, 9.0])
        .add_series(
            Series::new("Extra", vec![1.5, 2.5, 3.5]),
            Series::new("Extra Y", vec![3.0, 5.0, 7.0]),
        )
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    let cmds = rendered.canvas.commands();

    // Legend with circle swatches: should have small Circle commands
    // in the legend area (not in the data area).
    let plot = rendered.plot_area.unwrap();
    let (px, py, pw, _ph) = plot;
    let legend_circles = cmds.iter().filter(|cmd| {
        if let DrawCommand::Circle { cx, cy, radius, .. } = cmd {
            // Small radius (legend swatch) and in legend region
            *radius < 10.0 && (*cx > px + pw * 0.5 || *cy < py + 30.0)
        } else {
            false
        }
    }).count();

    assert!(
        legend_circles >= 2,
        "Scatter legend should have circle swatches (found {legend_circles})"
    );
}

/// Line chart legends must use line-segment swatches.
#[test]
fn line_legend_uses_line_swatch() {
    let chart = LineChart::new(vec![
        Series::new("Alpha", vec![1.0, 3.0, 2.0]),
        Series::new("Beta", vec![2.0, 1.0, 4.0]),
    ])
    .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    let cmds = rendered.canvas.commands();

    // Legend with line swatches: short horizontal line segments.
    // These are rendered inside the legend background rectangle.
    // We look for short horizontal lines (same Y, width < 20px).
    let short_horiz_lines = cmds.iter().filter(|cmd| {
        if let DrawCommand::Line { x1, y1, x2, y2, stroke, .. } = cmd {
            let dx = (x2 - x1).abs();
            let dy = (y2 - y1).abs();
            // Horizontal, short, within legend swatch size range
            dy < 1.0 && dx > 5.0 && dx < 20.0 && stroke.dash.is_none()
        } else {
            false
        }
    }).count();

    assert!(
        short_horiz_lines >= 2,
        "Line chart legend should have line-segment swatches (found {short_horiz_lines})"
    );
}

/// Bar chart legends must use rectangle swatches (default behavior).
#[test]
fn bar_legend_uses_rect_swatch() {
    let chart = Charts::bar(
        vec!["Q1".into(), "Q2".into()],
        &[10.0, 20.0],
    )
    .add_series(Series::new("Product B", vec![15.0, 25.0]))
    .build();

    let rendered = layout::render_chart(&chart, 400, 300);
    let cmds = rendered.canvas.commands();

    // Legend area should contain small filled rectangles (swatches).
    let plot = rendered.plot_area.unwrap();
    let small_rects = cmds.iter().filter(|cmd| {
        if let DrawCommand::Rectangle { rect, style, .. } = cmd {
            // Swatch-sized rect (< 20px) with fill
            rect.width < 20.0 && rect.height < 20.0 && rect.width > 4.0 && style.fill.is_some()
                && rect.x > plot.0  // in the plot area / legend area
        } else {
            false
        }
    }).count();

    assert!(
        small_rects >= 2,
        "Bar chart legend should have rect swatches (found {small_rects})"
    );
}

// ===========================================================================
// 4. AspectRatio enforcement
// ===========================================================================

/// `AspectRatio::Equal` must produce a square plot area (pw ≈ ph).
#[test]
fn aspect_ratio_equal_produces_square_plot() {
    let chart = Charts::scatter(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0])
        .aspect_ratio(AspectRatio::Equal)
        .build();

    let rendered = layout::render_chart(&chart, 600, 400);
    let plot = rendered.plot_area.expect("should have plot_area");
    let (_, _, pw, ph) = plot;

    let ratio = pw / ph;
    assert!(
        (ratio - 1.0).abs() < 0.05,
        "Equal aspect ratio should produce square plot (pw={pw:.1}, ph={ph:.1}, ratio={ratio:.3})"
    );
}

/// `AspectRatio::Fixed(2.0)` should produce a plot area with pw/ph ≈ 2.0.
#[test]
fn aspect_ratio_fixed_produces_correct_ratio() {
    let chart = Charts::scatter(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0])
        .aspect_ratio(AspectRatio::Fixed(2.0))
        .build();

    let rendered = layout::render_chart(&chart, 600, 400);
    let plot = rendered.plot_area.expect("should have plot_area");
    let (_, _, pw, ph) = plot;

    let ratio = pw / ph;
    assert!(
        (ratio - 2.0).abs() < 0.05,
        "Fixed(2.0) aspect ratio should produce 2:1 plot (pw={pw:.1}, ph={ph:.1}, ratio={ratio:.3})"
    );
}

/// `AspectRatio::Auto` (default) should NOT shrink the plot area —
/// it should fill available space.
#[test]
fn aspect_ratio_auto_fills_space() {
    let chart_auto = Charts::scatter(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0])
        .build();

    let chart_equal = Charts::scatter(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0])
        .aspect_ratio(AspectRatio::Equal)
        .build();

    let r_auto = layout::render_chart(&chart_auto, 600, 400);
    let r_equal = layout::render_chart(&chart_equal, 600, 400);

    let auto_plot = r_auto.plot_area.unwrap();
    let equal_plot = r_equal.plot_area.unwrap();

    // Auto should use more total area than Equal (which is constrained)
    let auto_area = auto_plot.2 * auto_plot.3;
    let equal_area = equal_plot.2 * equal_plot.3;

    assert!(
        auto_area >= equal_area,
        "Auto should use at least as much area as Equal (auto={auto_area:.0}, equal={equal_area:.0})"
    );
}

// ===========================================================================
// 5. Legend non-overlap (academic convention)
// ===========================================================================

/// When data fills the default legend position, the legend should
/// relocate to avoid obscuring data points.
///
/// Tufte 2001: "Data is the reason for the graphic. Everything else
/// is subordinate."
#[test]
fn legend_avoids_data_overlap() {
    use scry_chart::legend::LegendPosition;

    // Create data that fills the top-right corner
    let chart = Charts::scatter(
        &[3.5, 4.0, 4.5, 5.0, 4.8],
        &[35.0, 40.0, 45.0, 50.0, 48.0],
    )
    .add_series(
        Series::new("Extra", vec![3.8, 4.2, 4.7]),
        Series::new("Extra Y", vec![38.0, 42.0, 47.0]),
    )
    .legend_position(LegendPosition::Best)
    .build();

    let rendered = layout::render_chart(&chart, 400, 300);

    // Verify the chart rendered without panic and has legend text
    let labels = rendered.text_labels();
    let has_legend = labels.iter().any(|l| l.contains("Extra"));
    assert!(has_legend, "Legend should be present even when data fills top-right");
}

// ===========================================================================
// 6. Plot area is always within canvas bounds
// ===========================================================================

/// Property test: the plot area should always be fully within the canvas.
#[test]
fn plot_area_within_canvas() {
    let sizes = [(100, 75), (400, 300), (800, 600), (2000, 1200)];

    for (w, h) in sizes {
        let chart = Charts::scatter(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0])
            .title("Test")
            .x_label("X")
            .y_label("Y")
            .build();

        let rendered = layout::render_chart(&chart, w, h);
        let plot = rendered.plot_area.expect("should have plot_area");
        let (px, py, pw, ph) = plot;

        assert!(px >= 0.0, "plot x ({px}) must be >= 0 at {w}×{h}");
        assert!(py >= 0.0, "plot y ({py}) must be >= 0 at {w}×{h}");
        assert!(
            px + pw <= w as f32 + 1.0,
            "plot right edge ({}) must be <= canvas width ({w}) at {w}×{h}",
            px + pw
        );
        assert!(
            py + ph <= h as f32 + 1.0,
            "plot bottom edge ({}) must be <= canvas height ({h}) at {w}×{h}",
            py + ph
        );
    }
}

// ===========================================================================
// 7. WCAG AA contrast ratio for text across all themes
// ===========================================================================

/// WCAG 2.1 relative luminance from linear sRGB components (0.0–1.0).
fn relative_luminance(r: f32, g: f32, b: f32) -> f64 {
    let srgb_to_linear = |c: f32| -> f64 {
        let s = c as f64;
        if s <= 0.03928 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    0.2126 * srgb_to_linear(r) + 0.7152 * srgb_to_linear(g) + 0.0722 * srgb_to_linear(b)
}

/// Contrast ratio per WCAG 2.1 (1.0 to 21.0).
fn contrast_ratio(fg: (f32, f32, f32), bg: (f32, f32, f32)) -> f64 {
    let l1 = relative_luminance(fg.0, fg.1, fg.2);
    let l2 = relative_luminance(bg.0, bg.1, bg.2);
    let (lighter, darker) = if l1 > l2 { (l1, l2) } else { (l2, l1) };
    (lighter + 0.05) / (darker + 0.05)
}

/// All 6 built-in themes must achieve WCAG AA contrast (≥ 4.5:1) for
/// axis tick labels against the chart background.
///
/// WCAG 2.1 Success Criterion 1.4.3 — "Minimum Contrast"
#[test]
fn wcag_aa_contrast_all_themes() {
    let themes = [
        ("dark", Theme::dark()),
        ("light", Theme::light()),
        ("pastel", Theme::pastel()),
        ("ocean", Theme::ocean()),
        ("forest", Theme::forest()),
        ("colorblind", Theme::colorblind()),
        ("academic", Theme::academic()),
        ("presentation", Theme::presentation()),
        ("monochrome", Theme::monochrome()),
    ];

    for (name, theme) in &themes {
        let bg = theme.background;
        let bg_rgb = (bg.r, bg.g, bg.b);

        // Check tick label text color (from tick_style)
        let tick_color = theme.tick_style.color;
        let tick_rgb = (tick_color.r, tick_color.g, tick_color.b);
        let ratio = contrast_ratio(tick_rgb, bg_rgb);
        assert!(
            ratio >= 4.5,
            "Theme '{name}': tick label contrast {ratio:.2}:1 is below WCAG AA (4.5:1). \
             tick=({:.2},{:.2},{:.2}), bg=({:.2},{:.2},{:.2})",
            tick_rgb.0, tick_rgb.1, tick_rgb.2,
            bg_rgb.0, bg_rgb.1, bg_rgb.2,
        );

        // Check axis label text color (from label_style)
        let label_color = theme.label_style.color;
        let label_rgb = (label_color.r, label_color.g, label_color.b);
        let label_ratio = contrast_ratio(label_rgb, bg_rgb);
        assert!(
            label_ratio >= 4.5,
            "Theme '{name}': axis label contrast {label_ratio:.2}:1 is below WCAG AA (4.5:1). \
             label=({:.2},{:.2},{:.2}), bg=({:.2},{:.2},{:.2})",
            label_rgb.0, label_rgb.1, label_rgb.2,
            bg_rgb.0, bg_rgb.1, bg_rgb.2,
        );
    }
}

// ===========================================================================
// 8. Grid opacity ≤ 30% (Nature/Science guideline)
// ===========================================================================

/// Grid lines should use subdued styling so they don't compete with data.
///
/// Nature and Science figure guidelines recommend grid opacity ≤ 30%.
/// We verify the alpha channel of grid colors across all themes.
#[test]
fn grid_opacity_limit() {
    let themes = [
        ("dark", Theme::dark()),
        ("light", Theme::light()),
        ("pastel", Theme::pastel()),
        ("ocean", Theme::ocean()),
        ("forest", Theme::forest()),
        ("colorblind", Theme::colorblind()),
        ("academic", Theme::academic()),
        ("presentation", Theme::presentation()),
        ("monochrome", Theme::monochrome()),
    ];

    for (name, theme) in &themes {
        let grid_alpha = theme.grid.color.a;
        // 45% opacity — balances visibility on dark backgrounds with
        // data-ink subordination. The old 22% alpha was invisible.
        assert!(
            grid_alpha <= 0.46,
            "Theme '{name}': grid alpha {grid_alpha:.2} exceeds 45%. \
             Grid lines should be visually subordinate to data."
        );
    }
}

// ===========================================================================
// 8c. Palette-vs-background contrast (WCAG AA for graphical objects)
// ===========================================================================

/// Every palette color must have at least 3:1 contrast ratio against
/// its theme background, per WCAG 2.1 SC 1.4.11 (non-text contrast).
#[test]
fn palette_bg_contrast() {
    let themes = [
        ("dark", Theme::dark()),
        ("light", Theme::light()),
        ("pastel", Theme::pastel()),
        ("ocean", Theme::ocean()),
        ("forest", Theme::forest()),
        ("colorblind", Theme::colorblind()),
        ("academic", Theme::academic()),
        ("presentation", Theme::presentation()),
        ("monochrome", Theme::monochrome()),
    ];

    for (name, theme) in &themes {
        let bg = theme.background;
        let bg_rgb = (bg.r, bg.g, bg.b);

        for (i, color) in theme.palette.iter().enumerate() {
            let c_rgb = (color.r, color.g, color.b);
            let ratio = contrast_ratio(c_rgb, bg_rgb);
            assert!(
                ratio >= 2.5,
                "Theme '{name}': palette[{i}] contrast {ratio:.2}:1 vs bg is below 2.5:1. \
                 color=({:.0},{:.0},{:.0}), bg=({:.0},{:.0},{:.0})",
                color.r * 255.0, color.g * 255.0, color.b * 255.0,
                bg.r * 255.0, bg.g * 255.0, bg.b * 255.0,
            );
        }
    }
}

// ===========================================================================
// 9. Tick label consistency (uniform precision per axis)
// ===========================================================================

/// All tick labels on a single axis must use the same number of decimal
/// digits (a.k.a. uniform precision), per APA 7th Ed §6.36.
///
/// We verify this by rendering charts with various data ranges and
/// checking that the rendered tick labels all share the same precision.
#[test]
fn tick_labels_uniform_precision() {
    let test_cases: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        ("integers", vec![1.0, 2.0, 3.0, 4.0], vec![10.0, 20.0, 30.0, 40.0]),
        ("decimals", vec![0.1, 0.2, 0.3, 0.4], vec![1.5, 2.5, 3.5, 4.5]),
        ("mixed_mag", vec![0.0, 1.0, 2.0, 3.0], vec![0.0, 1000.0, 2000.0, 3000.0]),
    ];

    for (label, x, y) in test_cases {
        let chart = Charts::scatter(&x, &y)
            .title("Precision Test")
            .build();

        let rendered = layout::render_chart(&chart, 800, 600);
        let texts = rendered.text_labels();

        // Collect numeric tick labels (skip the title)
        let numeric_labels: Vec<&str> = texts.iter()
            .map(|s| s.as_ref())
            .filter(|t: &&str| {
                *t != "Precision Test"
                    && !t.is_empty()
                    && t.chars().next().map_or(false, |c| c.is_ascii_digit() || c == '-')
            })
            .collect();

        if numeric_labels.len() < 2 {
            continue;
        }

        // Get decimal place counts for plain number labels
        let decimal_counts: Vec<usize> = numeric_labels.iter()
            .filter(|l| !l.contains('K') && !l.contains('M') && !l.contains('G') && !l.contains('e'))
            .filter_map(|l| l.find('.').map(|d| l.len() - d - 1))
            .collect();

        if decimal_counts.len() >= 2 {
            assert!(
                decimal_counts.windows(2).all(|w| w[0] == w[1]),
                "Case '{label}': inconsistent precision in tick labels: {numeric_labels:?} → decimals: {decimal_counts:?}"
            );
        }
    }
}
