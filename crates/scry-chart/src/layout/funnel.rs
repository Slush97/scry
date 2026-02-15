//! Funnel chart rendering — trapezoid stages for conversion pipeline.

use crate::chart::funnel::FunnelChart;
use crate::theme::contrast_text_color;

use super::{RenderContext, RenderedChart, TextAlign, TextOverlay};

pub(crate) fn render_funnel(fc: &FunnelChart, w: u32, h: u32) -> RenderedChart {
    let config = &fc.config;
    let theme = &config.theme;
    let tick_fs = super::scaled_font_size(theme.tick_style.font_size, w, h);
    let data_fs = super::scaled_font_size(9.0, w, h);
    let label_fs = super::scaled_font_size(10.0, w, h);

    let mut ctx = RenderContext::new(config, w, h, None);
    let (px, py, pw, ph) = ctx.plot;

    let n = fc.labels.len().min(fc.values.len());
    if n == 0 {
        ctx.add_common_overlays(config);
        return ctx.finish();
    }

    // Find max value for proportional width calculation
    let max_val = fc
        .values
        .iter()
        .take(n)
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .reduce(f64::max)
        .unwrap_or(1.0);

    let first_val = fc.values.first().copied().unwrap_or(1.0).max(f64::EPSILON);

    // Layout: equal-height stages stacked vertically
    let total_gap = fc.gap * (n.saturating_sub(1)) as f32;
    let stage_h = ((ph - total_gap) / n as f32).max(4.0);
    let center_x = px + pw / 2.0;

    // ── Single-hue gradient: base color with decreasing lightness per stage ──
    let base_color = theme.resolve_series_color(0, &crate::data::SeriesStyle::default());
    let stage_colors: Vec<_> = (0..n)
        .map(|i| {
            // Alpha-based gradient: later stages become more transparent,
            // letting the background show through for a natural lightening
            // effect that works with any theme palette.
            let t = if n > 1 { i as f32 / (n - 1) as f32 } else { 0.0 };
            let alpha = 1.0 - t * 0.55; // 1.0 → 0.45
            scry_engine::style::Color {
                r: base_color.r,
                g: base_color.g,
                b: base_color.b,
                a: alpha,
            }
        })
        .collect();

    // ── Precompute widths for trapezoid shaping ──
    let widths: Vec<f32> = (0..n)
        .map(|i| {
            let v = fc.values[i];
            if v.is_finite() && v > 0.0 {
                (pw * (v / max_val) as f32).max(4.0)
            } else {
                4.0
            }
        })
        .collect();

    for i in 0..n {
        let value = fc.values[i];
        if !value.is_finite() || value <= 0.0 {
            continue;
        }

        let color = stage_colors[i];
        let bar_y = py + i as f32 * (stage_h + fc.gap);

        // ── Trapezoid: top edge = this stage's width, bottom edge = next stage's width ──
        let top_w = widths[i];
        let bottom_w = if i + 1 < n { widths[i + 1] } else { top_w * 0.3 }; // last stage tapers

        let top_left = center_x - top_w / 2.0;
        let top_right = center_x + top_w / 2.0;
        let bot_left = center_x - bottom_w / 2.0;
        let bot_right = center_x + bottom_w / 2.0;
        let bot_y = bar_y + stage_h;

        let trapezoid = vec![
            (top_left, bar_y),
            (top_right, bar_y),
            (bot_right, bot_y),
            (bot_left, bot_y),
        ];

        // Fill
        ctx.draw(|c| c.polygon(trapezoid.clone()).fill(color).done());

        // Stroke outline
        let stroke_color = {
            let (r, g, b, _) = ((color.r * 255.0) as u8, (color.g * 255.0) as u8, (color.b * 255.0) as u8, (color.a * 255.0) as u8);
            scry_engine::style::Color::from_rgba8(r, g, b, 255)
        };
        ctx.draw(|c| c.polygon(trapezoid).stroke(stroke_color, 1.0).done());

        // ── Text placement: inside if tall enough, outside if cramped ──
        let text_inside = stage_h >= 24.0;

        if text_inside {
            // Label centered inside
            let label_y = bar_y + stage_h / 2.0 - 6.0;
            ctx.overlays.push(TextOverlay {
                x_px: center_x,
                y_px: label_y,
                text: fc.labels[i].clone(),
                color: contrast_text_color(color),
                align: TextAlign::Center,
                font_size: tick_fs,
                bold: true,
                rotation_deg: 0.0,
            });

            // Detail text (value / percentage / conversion) below label
            let detail = build_detail(fc, i, first_val);
            if !detail.is_empty() {
                ctx.overlays.push(TextOverlay {
                    x_px: center_x,
                    y_px: label_y + 14.0,
                    text: detail,
                    color: contrast_text_color(color).with_alpha(0.7),
                    align: TextAlign::Center,
                    font_size: data_fs,
                    bold: false,
                    rotation_deg: 0.0,
                });
            }
        } else {
            // ── Text overflow guard: move labels outside the bar ──
            let label_x = top_right.max(bot_right) + 8.0;
            let label_y = bar_y + stage_h / 2.0;

            let mut text = fc.labels[i].clone();
            let detail = build_detail(fc, i, first_val);
            if !detail.is_empty() {
                text.push_str(" · ");
                text.push_str(&detail);
            }
            ctx.overlays.push(TextOverlay {
                x_px: label_x,
                y_px: label_y,
                text,
                color: theme.text_color(),
                align: TextAlign::Left,
                font_size: label_fs,
                bold: false,
                rotation_deg: 0.0,
            });
        }
    }

    ctx.add_common_overlays(config);
    ctx.finish()
}

/// Build the detail string with value, top-of-funnel %, and step conversion %.
fn build_detail(fc: &FunnelChart, i: usize, first_val: f64) -> String {
    let value = fc.values[i];
    let mut parts: Vec<String> = Vec::new();

    if fc.show_values {
        parts.push(format_value(value));
    }
    if fc.show_percentages {
        let pct = (value / first_val * 100.0).round();
        parts.push(format!("{pct}%"));
    }
    // Step conversion rate: ratio to previous stage
    if fc.show_percentages && i > 0 {
        let prev = fc.values[i - 1];
        if prev.is_finite() && prev > 0.0 {
            let step_pct = (value / prev * 100.0).round();
            parts.push(format!("→ {step_pct}%"));
        }
    }

    parts.join(" · ")
}

fn format_value(v: f64) -> String {
    if v.abs() >= 1_000_000.0 {
        format!("{:.1}M", v / 1_000_000.0)
    } else if v.abs() >= 1000.0 {
        format!("{:.1}K", v / 1000.0)
    } else if v.fract().abs() < 1e-9 {
        format!("{}", v as i64)
    } else {
        format!("{v:.1}")
    }
}
