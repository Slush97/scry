//! Axis formatting diagnostic test.
//!
//! This test renders charts across a range of data scenarios and canvas sizes,
//! then programmatically checks that EVERY text overlay (tick labels, axis labels,
//! title) is:
//!   1. Fully within the canvas bounds
//!   2. Not overlapping with adjacent labels on the same axis
//!   3. Y-axis label not colliding with Y-axis tick labels
//!   4. X-axis label not colliding with X-axis tick labels
//!
//! This is the definitive test for axis formatting correctness.
//!
//! Run: cargo test -p scry-chart --test axis_diagnostic -- --nocapture

use scry_chart::chart::{Chart, Charts};
use scry_chart::layout::{self, RenderedChart, TextAlign, TextOverlay};
use scry_chart::theme::Theme;

// ---------------------------------------------------------------------------
// Text measurement (must match export.rs logic exactly)
// ---------------------------------------------------------------------------

static FONT_DATA: &[u8] = include_bytes!("../src/fonts/Inter-Regular.ttf");
static FONT_DATA_BOLD: &[u8] = include_bytes!("../src/fonts/Inter-Bold.ttf");

/// Bounding box in pixel coordinates.
#[derive(Debug, Clone)]
struct TextBBox {
    label: String,
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
    /// Role for debugging: "title", "x_label", "y_label", "x_tick", "y_tick", "other"
    role: String,
}

impl TextBBox {
    fn overlaps(&self, other: &TextBBox) -> bool {
        self.x_min < other.x_max
            && self.x_max > other.x_min
            && self.y_min < other.y_max
            && self.y_max > other.y_min
    }

    fn within_canvas(&self, w: f32, h: f32) -> bool {
        self.x_min >= -2.0 && self.x_max <= w + 2.0 && self.y_min >= -2.0 && self.y_max <= h + 2.0
    }

    #[allow(dead_code)]
    fn width(&self) -> f32 {
        self.x_max - self.x_min
    }
}

fn measure_overlay(overlay: &TextOverlay) -> TextBBox {
    let settings = fontdue::FontSettings::default();
    let font = if overlay.bold {
        fontdue::Font::from_bytes(FONT_DATA_BOLD, settings).unwrap()
    } else {
        fontdue::Font::from_bytes(FONT_DATA, settings).unwrap()
    };

    let size = overlay.font_size;
    let mut total_width = 0.0_f32;
    for ch in overlay.text.chars() {
        let (metrics, _) = font.rasterize(ch, size);
        total_width += metrics.advance_width;
    }

    let line_metrics = font.horizontal_line_metrics(size);
    let ascent = line_metrics.map_or(size * 0.8, |m| m.ascent);
    let descent = line_metrics.map_or(size * 0.2, |m| -m.descent);

    let x_start = match overlay.align {
        TextAlign::Left => overlay.x_px,
        TextAlign::Center => overlay.x_px - total_width / 2.0,
        TextAlign::Right => overlay.x_px - total_width,
        _ => overlay.x_px, // future variants default to left
    };

    let baseline_y = overlay.y_px + ascent * 0.5;

    // For rotated text, compute the axis-aligned bounding box of the
    // rotated rectangle.
    if overlay.rotation_deg.abs() > 0.01 {
        // Counter-clockwise rotation (positive rotation_deg = CCW).
        let rad = (-overlay.rotation_deg).to_radians();
        let cos_a = rad.cos();
        let sin_a = rad.sin();

        // Rotation pivot: visual center of the text
        let cx = match overlay.align {
            TextAlign::Left => overlay.x_px + total_width / 2.0,
            TextAlign::Center => overlay.x_px,
            TextAlign::Right => overlay.x_px - total_width / 2.0,
            _ => overlay.x_px,
        };
        let cy = baseline_y - ascent / 2.0 + (ascent + descent) / 2.0;

        // Four corners of the unrotated text box
        let corners = [
            (x_start, baseline_y - ascent),
            (x_start + total_width, baseline_y - ascent),
            (x_start + total_width, baseline_y + descent),
            (x_start, baseline_y + descent),
        ];

        // Rotate each corner around (cx, cy)
        let rotated: Vec<(f32, f32)> = corners
            .iter()
            .map(|(x, y)| {
                let dx = x - cx;
                let dy = y - cy;
                (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)
            })
            .collect();

        // Compute the AABB of the rotated corners
        let x_min = rotated.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
        let x_max = rotated
            .iter()
            .map(|p| p.0)
            .fold(f32::NEG_INFINITY, f32::max);
        let y_min = rotated.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
        let y_max = rotated
            .iter()
            .map(|p| p.1)
            .fold(f32::NEG_INFINITY, f32::max);

        TextBBox {
            label: overlay.text.clone(),
            x_min,
            x_max,
            y_min,
            y_max,
            role: String::new(),
        }
    } else {
        TextBBox {
            label: overlay.text.clone(),
            x_min: x_start,
            x_max: x_start + total_width,
            y_min: baseline_y - ascent,
            y_max: baseline_y + descent,
            role: String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Classification — figure out what each overlay is
// ---------------------------------------------------------------------------

fn classify_overlays(
    rendered: &RenderedChart,
    config_title: Option<&str>,
    config_x_label: Option<&str>,
    config_y_label: Option<&str>,
) -> Vec<TextBBox> {
    let plot = rendered.plot_area.unwrap();
    let (px, py, _pw, ph) = plot;

    rendered
        .text_overlays_from_canvas()
        .iter()
        .map(|o| {
            let mut bbox = measure_overlay(o);

            // Classify by matching against known labels and position
            if let Some(t) = config_title {
                if o.text == t {
                    bbox.role = "title".to_string();
                    return bbox;
                }
            }
            if let Some(xl) = config_x_label {
                if o.text == xl && o.y_px > py + ph * 0.5 {
                    bbox.role = "x_label".to_string();
                    return bbox;
                }
            }
            if let Some(yl) = config_y_label {
                if o.text == yl && o.x_px < px {
                    bbox.role = "y_label".to_string();
                    return bbox;
                }
            }

            // Y-tick labels: right-aligned, left of plot area, NOT rotated.
            // Rotated X-tick labels also use Right alignment but have rotation_deg != 0
            // and are positioned below the plot midpoint.
            if o.align == TextAlign::Right && o.x_px <= px + 5.0 && o.rotation_deg.abs() < 0.01 {
                bbox.role = "y_tick".to_string();
                return bbox;
            }

            // X-tick labels: center-aligned OR right-aligned with rotation,
            // positioned below the plot area.
            let is_rotated_x = o.align == TextAlign::Right
                && o.rotation_deg.abs() > 0.01
                && o.y_px > py + ph * 0.5;
            if (o.align == TextAlign::Center && o.y_px > py + ph * 0.8) || is_rotated_x {
                bbox.role = "x_tick".to_string();
                return bbox;
            }

            bbox.role = "other".to_string();
            bbox
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Violation checks
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Violation {
    chart_name: String,
    canvas_size: (u32, u32),
    description: String,
}

fn check_chart(
    chart_name: &str,
    chart: &Chart,
    w: u32,
    h: u32,
    title: Option<&str>,
    x_label: Option<&str>,
    y_label: Option<&str>,
) -> Vec<Violation> {
    let rendered = layout::render_chart(chart, w, h);
    let bboxes = classify_overlays(&rendered, title, x_label, y_label);
    let mut violations = Vec::new();
    let canvas_size = (w, h);

    // 1. All labels within canvas bounds
    for bbox in &bboxes {
        if !bbox.within_canvas(w as f32, h as f32) {
            violations.push(Violation {
                chart_name: chart_name.to_string(),
                canvas_size,
                description: format!(
                    "OUT OF BOUNDS: {} '{}' extends to ({:.1}, {:.1})–({:.1}, {:.1}) on {}×{} canvas",
                    bbox.role, bbox.label, bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max, w, h
                ),
            });
        }
    }

    // 2. Y-tick labels don't overlap with Y-axis label
    let y_label_bbox: Vec<&TextBBox> = bboxes.iter().filter(|b| b.role == "y_label").collect();
    let y_tick_bboxes: Vec<&TextBBox> = bboxes.iter().filter(|b| b.role == "y_tick").collect();

    for yl in &y_label_bbox {
        for yt in &y_tick_bboxes {
            if yl.overlaps(yt) {
                violations.push(Violation {
                    chart_name: chart_name.to_string(),
                    canvas_size,
                    description: format!(
                        "Y-LABEL/TICK COLLISION: '{}' ({:.1}–{:.1}) overlaps tick '{}' ({:.1}–{:.1})",
                        yl.label, yl.x_min, yl.x_max, yt.label, yt.x_min, yt.x_max
                    ),
                });
            }
        }
    }

    // 3. X-tick labels don't overlap each other
    let x_tick_bboxes: Vec<&TextBBox> = bboxes.iter().filter(|b| b.role == "x_tick").collect();
    for i in 0..x_tick_bboxes.len() {
        for j in (i + 1)..x_tick_bboxes.len() {
            if x_tick_bboxes[i].overlaps(x_tick_bboxes[j]) {
                violations.push(Violation {
                    chart_name: chart_name.to_string(),
                    canvas_size,
                    description: format!(
                        "X-TICK OVERLAP: '{}' ({:.1}–{:.1}) overlaps '{}' ({:.1}–{:.1})",
                        x_tick_bboxes[i].label,
                        x_tick_bboxes[i].x_min,
                        x_tick_bboxes[i].x_max,
                        x_tick_bboxes[j].label,
                        x_tick_bboxes[j].x_min,
                        x_tick_bboxes[j].x_max,
                    ),
                });
            }
        }
    }

    // 4. Y-tick labels don't overlap each other
    for i in 0..y_tick_bboxes.len() {
        for j in (i + 1)..y_tick_bboxes.len() {
            if y_tick_bboxes[i].overlaps(y_tick_bboxes[j]) {
                violations.push(Violation {
                    chart_name: chart_name.to_string(),
                    canvas_size,
                    description: format!(
                        "Y-TICK OVERLAP: '{}' ({:.1}–{:.1}) overlaps '{}' ({:.1}–{:.1})",
                        y_tick_bboxes[i].label,
                        y_tick_bboxes[i].y_min,
                        y_tick_bboxes[i].y_max,
                        y_tick_bboxes[j].label,
                        y_tick_bboxes[j].y_min,
                        y_tick_bboxes[j].y_max,
                    ),
                });
            }
        }
    }

    // 5. X-axis label doesn't overlap with X-tick labels
    let x_label_bbox: Vec<&TextBBox> = bboxes.iter().filter(|b| b.role == "x_label").collect();
    for xl in &x_label_bbox {
        for xt in &x_tick_bboxes {
            if xl.overlaps(xt) {
                violations.push(Violation {
                    chart_name: chart_name.to_string(),
                    canvas_size,
                    description: format!(
                        "X-LABEL/TICK COLLISION: '{}' overlaps tick '{}'",
                        xl.label, xt.label
                    ),
                });
            }
        }
    }

    // 6. Title doesn't overlap with Y-tick labels or data
    let title_bboxes: Vec<&TextBBox> = bboxes.iter().filter(|b| b.role == "title").collect();
    for t in &title_bboxes {
        for yt in &y_tick_bboxes {
            if t.overlaps(yt) {
                violations.push(Violation {
                    chart_name: chart_name.to_string(),
                    canvas_size,
                    description: format!(
                        "TITLE/TICK COLLISION: '{}' overlaps y-tick '{}'",
                        t.label, yt.label
                    ),
                });
            }
        }
    }

    // 7. Y-tick labels don't extend to the left of the Y-label
    // (i.e. tick text shouldn't reach x < 0 or overlap the y-label)
    for yt in &y_tick_bboxes {
        if yt.x_min < 0.0 {
            violations.push(Violation {
                chart_name: chart_name.to_string(),
                canvas_size,
                description: format!(
                    "Y-TICK CLIPPED: '{}' starts at x={:.1} (left of canvas)",
                    yt.label, yt.x_min
                ),
            });
        }
    }

    // 8. Plot area utilization — flag if plot uses less than 50% of canvas width
    if let Some((px, _py, pw, _ph)) = rendered.plot_area {
        let utilization = pw / w as f32;
        if utilization < 0.50 {
            violations.push(Violation {
                chart_name: chart_name.to_string(),
                canvas_size,
                description: format!(
                    "LOW UTILIZATION: plot uses {:.0}% of canvas width (px={:.0}, pw={:.0}, canvas_w={})",
                    utilization * 100.0, px, pw, w
                ),
            });
        }
    }

    violations
}

// ---------------------------------------------------------------------------
// Test matrices
// ---------------------------------------------------------------------------

/// Chart configurations crossed with canvas sizes
fn test_matrix() -> Vec<(
    String,
    Chart,
    Option<String>,
    Option<String>,
    Option<String>,
)> {
    let mut charts = Vec::new();

    // 1. Normal range line chart
    charts.push((
        "line_normal".to_string(),
        Charts::line(&[1.0, 4.0, 2.0, 8.0, 5.0, 3.0, 7.0])
            .title("Normal Line")
            .x_label("Time")
            .y_label("Value")
            .theme(Theme::dark())
            .build(),
        Some("Normal Line".to_string()),
        Some("Time".to_string()),
        Some("Value".to_string()),
    ));

    // 2. Micro-range values (wide tick labels like "0.0020")
    charts.push((
        "line_micro_range".to_string(),
        Charts::line(&[0.001, 0.0015, 0.0012, 0.0018, 0.0014])
            .title("Micro Range")
            .x_label("Sample")
            .y_label("PPM")
            .theme(Theme::dark())
            .build(),
        Some("Micro Range".to_string()),
        Some("Sample".to_string()),
        Some("PPM".to_string()),
    ));

    // 3. Large values (tick labels like "350K")
    charts.push((
        "line_large_values".to_string(),
        Charts::line(&[150000.0, 250000.0, 180000.0, 350000.0, 280000.0])
            .title("Large Values")
            .x_label("Quarter")
            .y_label("Revenue")
            .theme(Theme::dark())
            .build(),
        Some("Large Values".to_string()),
        Some("Quarter".to_string()),
        Some("Revenue".to_string()),
    ));

    // 4. Negative values
    charts.push((
        "bar_negative".to_string(),
        Charts::bar(
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            &[10.0, -5.0, 15.0, -8.0],
        )
        .title("Negative Bars")
        .y_label("Profit/Loss")
        .theme(Theme::dark())
        .build(),
        Some("Negative Bars".to_string()),
        None,
        Some("Profit/Loss".to_string()),
    ));

    // 5. Scatter with annotations
    charts.push((
        "scatter_basic".to_string(),
        Charts::scatter(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2.0, 4.0, 1.5, 8.0, 5.0, 7.5, 3.0, 9.0],
        )
        .title("Scatter")
        .x_label("X Axis")
        .y_label("Y Axis")
        .theme(Theme::dark())
        .build(),
        Some("Scatter".to_string()),
        Some("X Axis".to_string()),
        Some("Y Axis".to_string()),
    ));

    // 6. Line chart with long axis label
    charts.push((
        "line_long_labels".to_string(),
        Charts::line(&[10.0, 25.0, 18.0, 35.0, 28.0, 42.0, 30.0])
            .title("Revenue Analysis")
            .x_label("Month of Year")
            .y_label("Revenue ($K)")
            .filled()
            .with_points()
            .theme(Theme::dark())
            .build(),
        Some("Revenue Analysis".to_string()),
        Some("Month of Year".to_string()),
        Some("Revenue ($K)".to_string()),
    ));

    // 7. Histogram
    let hist_data: Vec<f64> = (0..200)
        .map(|i| {
            let mut sum = 0.0;
            for k in 0..6u64 {
                sum += ((i as u64 * 2654435761 + k * 7919) % 10000) as f64 / 10000.0;
            }
            (sum - 3.0) * 2.0 + 10.0
        })
        .collect();
    charts.push((
        "histogram".to_string(),
        Charts::histogram(&hist_data)
            .title("Histogram")
            .x_label("Value")
            .y_label("Count")
            .bins(20)
            .theme(Theme::dark())
            .build(),
        Some("Histogram".to_string()),
        Some("Value".to_string()),
        Some("Count".to_string()),
    ));

    // 8. Very narrow range (forces many decimal places)
    charts.push((
        "line_narrow_range".to_string(),
        Charts::line(&[100.1, 100.3, 100.2, 100.5, 100.4])
            .title("Narrow Range")
            .x_label("Index")
            .y_label("Pressure")
            .theme(Theme::dark())
            .build(),
        Some("Narrow Range".to_string()),
        Some("Index".to_string()),
        Some("Pressure".to_string()),
    ));

    // 9. Rotated tick labels (diagonal 45°) — tests rotated X-tick positioning
    charts.push((
        "line_diagonal_ticks".to_string(),
        Charts::line(&[10.0, 25.0, 18.0, 35.0, 28.0, 42.0, 30.0])
            .title("Diagonal Ticks")
            .x_label("Category")
            .y_label("Amount")
            .x_tick_rotation(scry_chart::axis::LabelRotation::Diagonal)
            .theme(Theme::dark())
            .build(),
        Some("Diagonal Ticks".to_string()),
        Some("Category".to_string()),
        Some("Amount".to_string()),
    ));

    // 10. Many data points — triggers auto-skip edge case with 11 ticks
    //     where (total-1) % skip != 0 could create last-label overlap
    let many_pts: Vec<f64> = (0..11).map(|i| (i as f64) * 10.0 + 5.0).collect();
    charts.push((
        "line_11_points".to_string(),
        Charts::line(&many_pts)
            .title("11 Points")
            .x_label("Seq")
            .y_label("Val")
            .theme(Theme::dark())
            .build(),
        Some("11 Points".to_string()),
        Some("Seq".to_string()),
        Some("Val".to_string()),
    ));

    // ── Decimal y-label overlap edge cases ──
    // These force wide decimal tick labels (e.g. "0.0012") that may collide
    // with the y-axis label string.

    // 11. Scatter — micro-fractional y values
    charts.push((
        "scatter_decimal_y".to_string(),
        Charts::scatter(
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[0.001, 0.0025, 0.0018, 0.004, 0.0035],
        )
        .title("Scatter Decimals")
        .x_label("X")
        .y_label("Concentration")
        .theme(Theme::dark())
        .build(),
        Some("Scatter Decimals".to_string()),
        Some("X".to_string()),
        Some("Concentration".to_string()),
    ));

    // 12. Bar — fractional y values in 0.1–0.9 range
    charts.push((
        "bar_decimal_y".to_string(),
        Charts::bar(
            vec!["A".into(), "B".into(), "C".into(), "D".into(), "E".into()],
            &[0.12, 0.45, 0.78, 0.33, 0.91],
        )
        .title("Bar Decimals")
        .y_label("Probability")
        .theme(Theme::dark())
        .build(),
        Some("Bar Decimals".to_string()),
        None,
        Some("Probability".to_string()),
    ));

    // 13. Histogram — tiny fractional data forcing decimal bin edges
    let hist_frac: Vec<f64> = (0..100)
        .map(|i| 0.01 + (i as f64 * 0.0004) + ((i as u64 * 2654435761 % 100) as f64 * 0.00001))
        .collect();
    charts.push((
        "histogram_decimal_y".to_string(),
        Charts::histogram(&hist_frac)
            .title("Histogram Decimals")
            .x_label("Value")
            .y_label("Frequency")
            .bins(15)
            .theme(Theme::dark())
            .build(),
        Some("Histogram Decimals".to_string()),
        Some("Value".to_string()),
        Some("Frequency".to_string()),
    ));

    // 14. Line — large base with tiny fractional offsets (e.g. "1000000.002")
    charts.push((
        "line_large_decimal_y".to_string(),
        Charts::line(&[1000000.001, 1000000.003, 1000000.002, 1000000.005, 1000000.004])
            .title("Large Decimal")
            .x_label("Step")
            .y_label("Measurement")
            .theme(Theme::dark())
            .build(),
        Some("Large Decimal".to_string()),
        Some("Step".to_string()),
        Some("Measurement".to_string()),
    ));

    // 15. Boxplot — fractional values
    charts.push((
        "boxplot_decimal_y".to_string(),
        Charts::boxplot(vec![
            (
                "Group A".to_string(),
                vec![0.0012, 0.0018, 0.0025, 0.0031, 0.0045, 0.0052, 0.0060],
            ),
            (
                "Group B".to_string(),
                vec![0.0008, 0.0015, 0.0022, 0.0028, 0.0034, 0.0041, 0.0055],
            ),
        ])
        .title("Boxplot Decimals")
        .y_label("Rate")
        .theme(Theme::dark())
        .build(),
        Some("Boxplot Decimals".to_string()),
        None,
        Some("Rate".to_string()),
    ));

    charts
}

fn canvas_sizes() -> Vec<(u32, u32)> {
    vec![
        (300, 200),  // Small
        (400, 300),  // Medium-small (test default)
        (600, 400),  // Medium
        (800, 500),  // Standard
        (1200, 800), // Large
    ]
}

// ---------------------------------------------------------------------------
// THE TEST
// ---------------------------------------------------------------------------

#[test]
fn axis_formatting_diagnostic() {
    let charts = test_matrix();
    let sizes = canvas_sizes();
    let mut all_violations = Vec::new();

    for (name, chart, title, x_label, y_label) in &charts {
        for &(w, h) in &sizes {
            let violations = check_chart(
                &format!("{name}@{w}x{h}"),
                chart,
                w,
                h,
                title.as_deref(),
                x_label.as_deref(),
                y_label.as_deref(),
            );
            all_violations.extend(violations);
        }
    }

    // Print all violations
    if !all_violations.is_empty() {
        eprintln!("\n╔══════════════════════════════════════════════════════╗");
        eprintln!("║          AXIS FORMATTING VIOLATIONS FOUND            ║");
        eprintln!("╚══════════════════════════════════════════════════════╝\n");

        // Group by type
        let mut by_type: std::collections::BTreeMap<String, Vec<&Violation>> =
            std::collections::BTreeMap::new();

        for v in &all_violations {
            let vtype = v
                .description
                .split(':')
                .next()
                .unwrap_or("UNKNOWN")
                .to_string();
            by_type.entry(vtype).or_default().push(v);
        }

        for (vtype, violations) in &by_type {
            eprintln!("── {} ({} occurrences) ──", vtype, violations.len());
            for v in violations {
                eprintln!(
                    "  [{} {}×{}] {}",
                    v.chart_name, v.canvas_size.0, v.canvas_size.1, v.description
                );
            }
            eprintln!();
        }

        eprintln!(
            "TOTAL: {} violations across {} chart×size combinations",
            all_violations.len(),
            charts.len() * sizes.len()
        );
    }

    assert!(
        all_violations.is_empty(),
        "{} axis formatting violations found (see output above)",
        all_violations.len()
    );
}
