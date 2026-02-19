// SPDX-License-Identifier: MIT OR Apache-2.0
//! Gauge chart rendering — semicircular arc with needle indicator.

use std::f64::consts::PI;

use crate::chart::gauge::GaugeChart;

use super::{RenderContext, RenderedChart, TextAlign, TextOverlay};

pub(crate) fn render_gauge(gc: &GaugeChart, w: u32, h: u32) -> RenderedChart {
    let config = &gc.config;
    let theme = &config.theme;
    let value_fs = super::scaled_font_size(14.0, w, h);
    let data_fs = super::scaled_font_size(9.0, w, h);

    let mut ctx = RenderContext::new(config, w, h, None);
    let (px, py, pw, ph) = ctx.plot;

    let center_x = px + pw / 2.0;
    // Place arc center at ~60% of plot height to leave room for labels below
    let center_y = py + ph * 0.6;
    let radius = (pw / 2.0).min(ph * 0.55) - gc.arc_width;

    if radius < 5.0 {
        ctx.add_common_overlays(config);
        return ctx.finish();
    }

    let range = gc.max - gc.min;
    let range_safe = if range.abs() < f64::EPSILON {
        1.0
    } else {
        range
    };

    // Draw threshold bands (or single track if no thresholds)
    let arc = ArcParams {
        cx: center_x,
        cy: center_y,
        radius,
        width: gc.arc_width,
        segments: 60,
    };

    let track_color = theme.text_color().with_alpha(0.15);

    if gc.thresholds.is_empty() {
        draw_arc_band(&mut ctx, &arc, 0.0, 1.0, track_color);
    } else {
        let mut prev_t = 0.0_f64;
        for &(upper, color) in &gc.thresholds {
            let t = ((upper - gc.min) / range_safe).clamp(0.0, 1.0);
            if t > prev_t {
                draw_arc_band(
                    &mut ctx,
                    &arc,
                    prev_t as f32,
                    t as f32,
                    color.with_alpha(0.6),
                );
                prev_t = t;
            }
        }
        if prev_t < 1.0 {
            draw_arc_band(&mut ctx, &arc, prev_t as f32, 1.0, track_color);
        }
    }

    // Draw value fill arc (from 0 to current value)
    let value_t = ((gc.value - gc.min) / range_safe).clamp(0.0, 1.0);
    if value_t > 0.0 {
        let fill_color = gc
            .needle_color
            .unwrap_or_else(|| theme.palette.first().copied().unwrap_or(theme.text_color()));
        draw_arc_band(&mut ctx, &arc, 0.0, value_t as f32, fill_color.with_alpha(0.7));
    }

    // Draw needle
    let needle_angle = PI - value_t * PI; // π (left) to 0 (right)
    let nx = center_x + (radius * needle_angle.cos() as f32);
    let ny = center_y - (radius * needle_angle.sin() as f32);
    let needle_color = gc.needle_color.unwrap_or_else(|| theme.text_color());

    ctx.draw(|c| {
        c.line(center_x, center_y, nx, ny)
            .color(needle_color)
            .width(2.5)
            .done()
    });

    // Needle hub circle
    ctx.draw(|c| c.circle(center_x, center_y, 4.0).fill(needle_color).done());

    // Value label below center
    let label_text = gc.label.clone().unwrap_or_else(|| {
        if gc.value.fract().abs() < 1e-9 {
            format!("{}", gc.value as i64)
        } else {
            format!("{:.1}", gc.value)
        }
    });
    ctx.overlays.push(TextOverlay {
        x_px: center_x,
        y_px: center_y + radius * 0.15,
        text: label_text,
        color: theme.text_color(),
        align: TextAlign::Center,
        font_size: value_fs,
        bold: true,
        rotation_deg: 0.0,
    });

    // Min/max labels at arc ends
    ctx.overlays.push(TextOverlay {
        x_px: center_x - radius - gc.arc_width / 2.0,
        y_px: center_y + 8.0,
        text: format_compact(gc.min),
        color: theme.text_color().with_alpha(0.6),
        align: TextAlign::Center,
        font_size: data_fs,
        bold: false,
        rotation_deg: 0.0,
    });
    ctx.overlays.push(TextOverlay {
        x_px: center_x + radius + gc.arc_width / 2.0,
        y_px: center_y + 8.0,
        text: format_compact(gc.max),
        color: theme.text_color().with_alpha(0.6),
        align: TextAlign::Center,
        font_size: data_fs,
        bold: false,
        rotation_deg: 0.0,
    });

    ctx.add_common_overlays(config);
    ctx.finish()
}

/// Parameters for drawing arc bands.
struct ArcParams {
    cx: f32,
    cy: f32,
    radius: f32,
    width: f32,
    segments: usize,
}

/// Draw an arc band (thick arc) between two normalized positions [0, 1].
///
/// Uses a filled polygon approximation: outer arc + reversed inner arc.
fn draw_arc_band(
    ctx: &mut RenderContext,
    arc: &ArcParams,
    t_start: f32,
    t_end: f32,
    color: scry_engine::style::Color,
) {
    let r_outer = arc.radius + arc.width / 2.0;
    let r_inner = arc.radius - arc.width / 2.0;
    let seg_count = ((arc.segments as f32 * (t_end - t_start)).ceil() as usize).max(2);

    let mut pts: Vec<(f32, f32)> = Vec::with_capacity(seg_count * 2 + 2);

    // Outer arc (left to right)
    for i in 0..=seg_count {
        let t = t_start + (t_end - t_start) * i as f32 / seg_count as f32;
        let angle = (PI as f32) * (1.0 - t); // π (left) to 0 (right)
        pts.push((
            arc.cx + r_outer * angle.cos(),
            arc.cy - r_outer * angle.sin(),
        ));
    }

    // Inner arc (right to left)
    for i in (0..=seg_count).rev() {
        let t = t_start + (t_end - t_start) * i as f32 / seg_count as f32;
        let angle = (PI as f32) * (1.0 - t);
        pts.push((
            arc.cx + r_inner * angle.cos(),
            arc.cy - r_inner * angle.sin(),
        ));
    }

    ctx.draw(|c| c.polygon(pts).fill(color).done());
}

fn format_compact(v: f64) -> String {
    if v.fract().abs() < 1e-9 {
        format!("{}", v as i64)
    } else {
        format!("{v:.1}")
    }
}
