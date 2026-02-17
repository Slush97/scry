// SPDX-License-Identifier: MIT OR Apache-2.0
//! Lollipop chart rendering — thin stems with dot markers.

use crate::chart::lollipop::LollipopChart;
use crate::scale::{CategoricalScale, LinearScale, Scale};

use super::{resolve_y_extent, RenderContext, RenderedChart, TextAlign, TextOverlay};

pub(crate) fn render_lollipop(lc: &LollipopChart, w: u32, h: u32) -> RenderedChart {
    if lc.horizontal {
        render_lollipop_horizontal(lc, w, h)
    } else {
        render_lollipop_vertical(lc, w, h)
    }
}

fn render_lollipop_vertical(lc: &LollipopChart, w: u32, h: u32) -> RenderedChart {
    let config = &lc.config;
    let theme = &config.theme;
    let data_fs = super::scaled_font_size(9.0, w, h);

    let n = lc.labels.len().min(lc.values.len());
    let y_lo = lc
        .values
        .iter()
        .take(n)
        .copied()
        .filter(|v| v.is_finite())
        .reduce(f64::min)
        .unwrap_or(0.0)
        .min(0.0);
    let y_hi = lc
        .values
        .iter()
        .take(n)
        .copied()
        .filter(|v| v.is_finite())
        .reduce(f64::max)
        .unwrap_or(1.0)
        .max(0.0);
    let y_extent = resolve_y_extent(config, (y_lo, y_hi));

    let mut ctx = RenderContext::new(config, w, h, Some(y_extent));
    let (px, py, pw, ph) = ctx.plot;

    let y_scale = LinearScale::nice_zero(y_extent, ((py + ph) as f64, py as f64));
    let cat_scale = CategoricalScale::new(lc.labels.clone(), (px as f64, (px + pw) as f64));

    // Axes
    let y_ticks = ctx.draw_y_axis(config, &y_scale);
    ctx.add_y_tick_overlays(&y_ticks, theme.text_color());
    ctx.draw_x_axis_line(config);
    ctx.draw_categorical_x_labels(config, &cat_scale, &lc.labels);

    // Reference lines
    let x_dummy = LinearScale::new((0.0, 1.0), (px as f64, (px + pw) as f64));
    ctx.draw_reference_lines(config, &x_dummy, &y_scale);

    let baseline_y = y_scale.to_pixel(0.0) as f32;
    let color = theme.resolve_series_color(0, &crate::data::SeriesStyle::default());

    for i in 0..n {
        let value = lc.values[i];
        if !value.is_finite() {
            continue;
        }
        let cx = cat_scale.center(i) as f32;
        let vy = y_scale.to_pixel(value) as f32;

        // Stem
        ctx.draw(|c| {
            c.line(cx, baseline_y, cx, vy)
                .color(color)
                .width(lc.stem_width)
                .done()
        });

        // Dot
        ctx.draw(|c| c.circle(cx, vy, lc.dot_radius).fill(color).done());

        // Value label
        if lc.show_values {
            let label_y = if value >= 0.0 {
                vy - lc.dot_radius - 4.0
            } else {
                vy + lc.dot_radius + 12.0
            };
            ctx.overlays.push(TextOverlay {
                x_px: cx,
                y_px: label_y,
                text: format_value(value),
                color: theme.text_color(),
                align: TextAlign::Center,
                font_size: data_fs,
                bold: false,
                rotation_deg: 0.0,
            });
        }
    }

    ctx.add_common_overlays(config);
    ctx.finish()
}

fn render_lollipop_horizontal(lc: &LollipopChart, w: u32, h: u32) -> RenderedChart {
    let config = &lc.config;
    let theme = &config.theme;
    let data_fs = super::scaled_font_size(9.0, w, h);
    let tick_fs = super::scaled_font_size(theme.tick_style.font_size, w, h);

    let n = lc.labels.len().min(lc.values.len());
    let mut ctx = RenderContext::new(config, w, h, None);
    let (px, py, pw, ph) = ctx.plot;

    let x_lo = lc
        .values
        .iter()
        .take(n)
        .copied()
        .filter(|v| v.is_finite())
        .reduce(f64::min)
        .unwrap_or(0.0)
        .min(0.0);
    let x_hi = lc
        .values
        .iter()
        .take(n)
        .copied()
        .filter(|v| v.is_finite())
        .reduce(f64::max)
        .unwrap_or(1.0)
        .max(0.0);
    let x_scale = LinearScale::nice_zero((x_lo, x_hi), (px as f64, (px + pw) as f64));
    let cat_scale = CategoricalScale::new(lc.labels.clone(), (py as f64, (py + ph) as f64));

    // X value-axis (bottom)
    ctx.draw_x_value_axis(config, &x_scale);

    // Y axis line (left side)
    let axis_color = theme.axis_color();
    let axis_width = theme.axis_width();
    ctx.draw(|c| {
        c.line(px, py, px, py + ph)
            .color(axis_color)
            .width(axis_width)
            .done()
    });

    let baseline_x = x_scale.to_pixel(0.0) as f32;
    let color = theme.resolve_series_color(0, &crate::data::SeriesStyle::default());

    // Category labels on the left
    for (ci, label) in lc.labels.iter().enumerate().take(n) {
        ctx.overlays.push(TextOverlay {
            x_px: px - super::y_tick_label_offset(w),
            y_px: cat_scale.center(ci) as f32,
            text: label.clone(),
            color: theme.text_color(),
            align: TextAlign::Right,
            font_size: tick_fs,
            bold: false,
            rotation_deg: 0.0,
        });
    }

    for i in 0..n {
        let value = lc.values[i];
        if !value.is_finite() {
            continue;
        }
        let cy = cat_scale.center(i) as f32;
        let vx = x_scale.to_pixel(value) as f32;

        // Stem
        ctx.draw(|c| {
            c.line(baseline_x, cy, vx, cy)
                .color(color)
                .width(lc.stem_width)
                .done()
        });

        // Dot
        ctx.draw(|c| c.circle(vx, cy, lc.dot_radius).fill(color).done());

        // Value label
        if lc.show_values {
            let (label_x, align) = if value >= 0.0 {
                (vx + lc.dot_radius + 4.0, TextAlign::Left)
            } else {
                (vx - lc.dot_radius - 4.0, TextAlign::Right)
            };
            ctx.overlays.push(TextOverlay {
                x_px: label_x,
                y_px: cy,
                text: format_value(value),
                color: theme.text_color(),
                align,
                font_size: data_fs,
                bold: false,
                rotation_deg: 0.0,
            });
        }
    }

    ctx.add_common_overlays(config);
    ctx.finish()
}

fn format_value(v: f64) -> String {
    if v.abs() >= 1000.0 || (v.abs() < 0.01 && v != 0.0) {
        format!("{v:.2e}")
    } else if v.fract().abs() < 1e-9 {
        format!("{}", v as i64)
    } else {
        format!("{v:.1}")
    }
}
