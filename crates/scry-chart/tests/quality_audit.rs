//! Quality audit tests for scry-chart rendering correctness.
//!
//! These tests programmatically verify contrast, scaling, layout, and
//! regression properties across chart types and themes.

use scry_chart::chart::Chart;
use scry_chart::layout;
use scry_chart::theme::{contrast_text_color, Theme};
use scry_engine::style::Color;

// ===========================================================================
// Helper: WCAG relative luminance
// ===========================================================================

/// sRGB → linear conversion.
fn srgb_linearize(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// WCAG 2.0 relative luminance.
fn relative_luminance(c: Color) -> f32 {
    0.2126 * srgb_linearize(c.r) + 0.7152 * srgb_linearize(c.g) + 0.0722 * srgb_linearize(c.b)
}

/// WCAG contrast ratio between two colors (always ≥ 1.0).
fn contrast_ratio(a: Color, b: Color) -> f32 {
    let la = relative_luminance(a) + 0.05;
    let lb = relative_luminance(b) + 0.05;
    if la > lb {
        la / lb
    } else {
        lb / la
    }
}

// ===========================================================================
// All 6 built-in themes
// ===========================================================================

fn all_themes() -> Vec<(&'static str, Theme)> {
    vec![
        ("dark", Theme::dark()),
        ("light", Theme::light()),
        ("pastel", Theme::pastel()),
        ("ocean", Theme::ocean()),
        ("forest", Theme::forest()),
        ("colorblind", Theme::colorblind()),
    ]
}

// ===========================================================================
// Test 1: Contrast ratio across themes (pie, heatmap, funnel)
// ===========================================================================

#[test]
fn contrast_ratio_across_themes() {
    for (theme_name, theme) in all_themes() {
        // --- Pie chart ---
        let pie = Chart::pie(
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            &[30.0, 25.0, 20.0, 25.0],
        )
        .theme(theme.clone())
        .build();

        let rendered = layout::render_chart(&pie, 400, 300);
        for overlay in &rendered.text_overlays {
            // Percentage labels use contrast_text_color, so they should be
            // black or white — both guarantee high contrast against any bg.
            let is_contrast_color = overlay.color == Color::BLACK || overlay.color == Color::WHITE;
            // Skip non-data overlays (title, axis labels use theme text color)
            if overlay.text.contains('%') {
                assert!(
                    is_contrast_color,
                    "Pie [{theme_name}]: label '{}' has non-contrast color {:?}",
                    overlay.text, overlay.color
                );
            }
        }

        // --- Heatmap ---
        let heatmap = Chart::heatmap(vec![
            vec![1.0, 5.0, 9.0],
            vec![3.0, 6.0, 2.0],
            vec![7.0, 4.0, 8.0],
        ])
        .theme(theme.clone())
        .build();

        let rendered = layout::render_chart(&heatmap, 400, 300);
        for overlay in &rendered.text_overlays {
            // Cell value labels use contrast_text_color and contain a decimal
            // point (e.g. "1.00", "5.00"). Row/col index labels ("0", "1", "2")
            // correctly use theme text color, so we skip them.
            if overlay.text.contains('.') && overlay.text.parse::<f64>().is_ok() {
                let is_contrast_color =
                    overlay.color == Color::BLACK || overlay.color == Color::WHITE;
                assert!(
                    is_contrast_color,
                    "Heatmap [{theme_name}]: cell label '{}' has non-contrast color {:?}",
                    overlay.text, overlay.color
                );
            }
        }

        // --- Funnel ---
        let funnel = Chart::funnel(
            vec![
                "Visitors".into(),
                "Signups".into(),
                "Trials".into(),
                "Paid".into(),
            ],
            &[10000.0, 5000.0, 2000.0, 800.0],
        )
        .theme(theme.clone())
        .build();

        let rendered = layout::render_chart(&funnel, 500, 400);
        for overlay in &rendered.text_overlays {
            // Inside labels should use contrast_text_color (black or white),
            // or white/black with alpha. Check the base RGB channels.
            let base = Color {
                r: overlay.color.r,
                g: overlay.color.g,
                b: overlay.color.b,
                a: 1.0,
            };
            let is_contrast_color = base == Color::BLACK || base == Color::WHITE;
            // Skip non-data overlays (title, subtitle, footer)
            let is_stage_label = ["Visitors", "Signups", "Trials", "Paid"]
                .iter()
                .any(|s| overlay.text.contains(s));
            if is_stage_label {
                assert!(
                    is_contrast_color,
                    "Funnel [{theme_name}]: label '{}' has non-contrast base color {:?}",
                    overlay.text, base
                );
            }
        }
    }
}

// ===========================================================================
// Test 2: Font scaling consistency (3 canvas sizes)
// ===========================================================================

#[test]
fn font_scaling_consistency() {
    let sizes: [(u32, u32); 3] = [(40, 30), (400, 300), (2000, 1200)];

    let chart = Chart::line(&[1.0, 4.0, 2.0, 8.0, 5.0])
        .title("Scaling Test")
        .x_label("X Axis")
        .y_label("Y Axis")
        .theme(Theme::dark())
        .build();

    let mut prev_title_fs: Option<f32> = None;

    for (w, h) in sizes {
        let rendered = layout::render_chart(&chart, w, h);

        // Collect font sizes by role
        let mut title_fs = None;
        let mut label_fs = Vec::new();
        let mut tick_fs = Vec::new();

        for overlay in &rendered.text_overlays {
            // Clamp check: all font sizes within [7, 48]
            assert!(
                overlay.font_size >= 7.0 && overlay.font_size <= 48.0,
                "Font size {:.1} at {}×{} out of clamp range [7, 48] for '{}'",
                overlay.font_size,
                w,
                h,
                overlay.text
            );

            if overlay.text == "Scaling Test" {
                title_fs = Some(overlay.font_size);
            } else if overlay.text == "X Axis" || overlay.text == "Y Axis" {
                label_fs.push(overlay.font_size);
            } else {
                tick_fs.push(overlay.font_size);
            }
        }

        // Hierarchy: title > label > tick
        if let Some(tfs) = title_fs {
            for &lfs in &label_fs {
                assert!(
                    tfs >= lfs,
                    "Title font ({tfs:.1}) should be >= label font ({lfs:.1}) at {w}×{h}"
                );
            }
            for &tks in &tick_fs {
                assert!(
                    tfs >= tks,
                    "Title font ({tfs:.1}) should be >= tick font ({tks:.1}) at {w}×{h}"
                );
            }
        }

        if !label_fs.is_empty() && !tick_fs.is_empty() {
            let max_label = label_fs.iter().cloned().reduce(f32::max).unwrap();
            let min_tick = tick_fs.iter().cloned().reduce(f32::min).unwrap();
            assert!(
                max_label >= min_tick,
                "Label font ({max_label:.1}) should be >= tick font ({min_tick:.1}) at {w}×{h}"
            );
        }

        // Font sizes should scale with canvas size (larger canvas → larger fonts)
        if let (Some(prev), Some(curr)) = (prev_title_fs, title_fs) {
            assert!(
                curr >= prev,
                "Title font should grow with canvas: {prev:.1} at prev size vs {curr:.1} at {w}×{h}"
            );
        }
        prev_title_fs = title_fs;
    }
}

// ===========================================================================
// Test 3: Label non-overlap (bar chart with 20 long labels)
// ===========================================================================

#[test]
fn label_non_overlap_bar_20_categories() {
    let labels: Vec<String> = (0..20)
        .map(|i| format!("Category_{i:02}_LongName"))
        .collect();
    let values: Vec<f64> = (0..20).map(|i| (i as f64 + 1.0) * 5.0).collect();

    let chart = Chart::bar(labels, &values)
        .title("20 Categories")
        .theme(Theme::dark())
        .build();

    // Use a narrow canvas to stress-test label placement
    let rendered = layout::render_chart(&chart, 250, 300);

    // Collect category label overlays sorted by x position.
    // With auto-formatting, some labels may be skipped or truncated.
    let mut cat_overlays: Vec<(f32, &str)> = rendered
        .text_overlays
        .iter()
        .filter(|o| o.text.starts_with("Category_") || o.text.starts_with("Catego"))
        .map(|o| (o.x_px, o.text.as_str()))
        .collect();

    // At least some labels should be visible (first and last always kept).
    assert!(
        cat_overlays.len() >= 2,
        "Should render at least first and last category labels, got {}",
        cat_overlays.len()
    );

    // Sort by x position
    cat_overlays.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // All visible labels should have strictly increasing x positions.
    for pair in cat_overlays.windows(2) {
        let (x1, label1) = pair[0];
        let (x2, label2) = pair[1];
        assert!(
            x2 > x1,
            "Category labels should have strictly increasing x positions: \
             '{label1}' at x={x1:.1} vs '{label2}' at x={x2:.1}"
        );
    }

    // On a wider canvas, all 20 labels should be visible (possibly rotated/staggered).
    let wide_rendered = layout::render_chart(
        &Chart::bar(
            (0..20).map(|i| format!("Cat_{i:02}")).collect(),
            &(0..20).map(|i| (i as f64 + 1.0) * 5.0).collect::<Vec<_>>(),
        )
        .title("20 Short Categories")
        .theme(Theme::dark())
        .build(),
        800,
        300,
    );
    let wide_cat_count = wide_rendered
        .text_overlays
        .iter()
        .filter(|o| o.text.starts_with("Cat_"))
        .count();
    assert_eq!(
        wide_cat_count, 20,
        "Wide canvas with short labels should render all 20 categories, got {wide_cat_count}"
    );
}

// ===========================================================================
// Test 4: Heatmap subtitle regression (Session 1D)
// ===========================================================================

#[test]
fn heatmap_subtitle_present() {
    let chart = Chart::heatmap(vec![vec![1.0, 2.0], vec![3.0, 4.0]])
        .subtitle("Sub")
        .title("HM")
        .build();

    let rendered = layout::render_chart(&chart, 400, 300);

    let has_subtitle = rendered.text_overlays.iter().any(|o| o.text == "Sub");
    assert!(
        has_subtitle,
        "Heatmap with .subtitle(\"Sub\") should have 'Sub' in text overlays. \
         Found: {:?}",
        rendered
            .text_overlays
            .iter()
            .map(|o| &o.text)
            .collect::<Vec<_>>()
    );
}

// ===========================================================================
// Test 5: Proportional offsets (gauge + radar at small vs large canvas)
// ===========================================================================

#[test]
fn proportional_offsets_gauge_radar() {
    // --- Gauge ---
    let gauge = Chart::gauge(75.0).title("Gauge").label("75%").build();

    let small = layout::render_chart(&gauge, 100, 75);
    let large = layout::render_chart(&gauge, 2000, 1200);

    // Find the value label overlay (the "75%" text)
    let small_label = small.text_overlays.iter().find(|o| o.text == "75%");
    let large_label = large.text_overlays.iter().find(|o| o.text == "75%");

    if let (Some(s), Some(l)) = (small_label, large_label) {
        // The y-offset from center should scale — not be a fixed 16px at both sizes
        // Since canvas height differs 16×, the y_px should differ significantly
        assert!(
            (l.y_px - s.y_px).abs() > 5.0,
            "Gauge label y-offset should scale: small={:.1}, large={:.1}",
            s.y_px,
            l.y_px
        );
    }

    // --- Radar ---
    let radar = Chart::radar(vec!["A", "B", "C", "D", "E"])
        .add_series("S1", &[8.0, 6.0, 7.0, 5.0, 9.0])
        .title("Radar")
        .build();

    let small = layout::render_chart(&radar, 100, 75);
    let large = layout::render_chart(&radar, 2000, 1200);

    // Axis labels should be at different radii at different canvas sizes
    let small_labels: Vec<f32> = small
        .text_overlays
        .iter()
        .filter(|o| ["A", "B", "C", "D", "E"].contains(&o.text.as_str()))
        .map(|o| (o.x_px * o.x_px + o.y_px * o.y_px).sqrt())
        .collect();

    let large_labels: Vec<f32> = large
        .text_overlays
        .iter()
        .filter(|o| ["A", "B", "C", "D", "E"].contains(&o.text.as_str()))
        .map(|o| (o.x_px * o.x_px + o.y_px * o.y_px).sqrt())
        .collect();

    if !small_labels.is_empty() && !large_labels.is_empty() {
        let small_avg: f32 = small_labels.iter().sum::<f32>() / small_labels.len() as f32;
        let large_avg: f32 = large_labels.iter().sum::<f32>() / large_labels.len() as f32;

        // Large canvas labels should be at a larger radius
        assert!(
            large_avg > small_avg * 1.5,
            "Radar label radius should scale: small_avg={small_avg:.1}, large_avg={large_avg:.1}"
        );
    }
}
