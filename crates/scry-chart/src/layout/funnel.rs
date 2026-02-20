// SPDX-License-Identifier: MIT OR Apache-2.0
//! Funnel chart rendering — trapezoid stages for conversion pipeline.

use crate::chart::funnel::FunnelChart;
use crate::theme::{contrast_text_color_composited};

use super::{RenderContext, RenderedChart, TextAlign};

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
            let t = if n > 1 {
                i as f32 / (n - 1) as f32
            } else {
                0.0
            };
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
    let min_width = pw * 0.12; // minimum 12% of plot width to prevent crushing
    let widths: Vec<f32> = (0..n)
        .map(|i| {
            let v = fc.values[i];
            if v.is_finite() && v > 0.0 {
                (pw * (v / max_val) as f32).max(min_width)
            } else {
                min_width
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
        let bottom_w = if i + 1 < n {
            widths[i + 1]
        } else {
            (top_w * 0.3).max(min_width)
        }; // last stage tapers but respects minimum

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
            let (r, g, b, _) = (
                (color.r * 255.0) as u8,
                (color.g * 255.0) as u8,
                (color.b * 255.0) as u8,
                (color.a * 255.0) as u8,
            );
            scry_engine::style::Color::from_rgba8(r, g, b, 255)
        };
        ctx.draw(|c| c.polygon(trapezoid).stroke(stroke_color, 1.0).done());

        // ── Text placement: inside if tall enough, outside if cramped ──
        let detail = build_detail(fc, i, first_val);
        let has_detail = !detail.is_empty();
        // Need more height when detail is present (label + detail + spacing)
        let min_inside_h = if has_detail { 40.0 } else { 28.0 };
        let text_inside = stage_h >= min_inside_h;

        if text_inside {
            // Vertically center the text block (label + optional detail)
            let text_block_h = if has_detail { tick_fs + data_fs * 0.8 + 4.0 } else { tick_fs };
            let block_top = bar_y + (stage_h - text_block_h) / 2.0;

            ctx.add_text(center_x, block_top, &fc.labels[i], contrast_text_color_composited(color, theme.background), TextAlign::Center, tick_fs, true, 0.0);

            if has_detail {
                ctx.add_text(center_x, block_top + tick_fs + 2.0, &detail, contrast_text_color_composited(color, theme.background).with_alpha(0.65), TextAlign::Center, data_fs * 0.85, false, 0.0);
            }
        } else {
            // ── Text overflow guard: move labels outside the bar ──
            let label_x = top_right.max(bot_right) + 8.0;
            let label_y = bar_y + stage_h / 2.0;

            let mut text = fc.labels[i].clone();
            if has_detail {
                text.push_str(" · ");
                text.push_str(&detail);
            }
            ctx.add_text(label_x, label_y, &text, theme.text_color(), TextAlign::Left, label_fs, false, 0.0);
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
