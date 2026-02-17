// SPDX-License-Identifier: MIT OR Apache-2.0
//! Waterfall chart rendering — sequential bars with running totals.

use crate::chart::waterfall::WaterfallChart;
use crate::scale::{CategoricalScale, LinearScale, Scale};
use scry_engine::style::Color;

use super::{resolve_y_extent, RenderContext, RenderedChart, TextAlign, TextOverlay};

pub(crate) fn render_waterfall(wc: &WaterfallChart, w: u32, h: u32) -> RenderedChart {
    let config = &wc.config;
    let theme = &config.theme;
    let data_fs = super::scaled_font_size(9.0, w, h);

    // Build effective labels + values (append total if requested)
    let mut labels = wc.labels.clone();
    let mut values = wc.values.clone();
    if wc.show_total {
        labels.push("Total".to_string());
        let total: f64 = values.iter().copied().filter(|v| v.is_finite()).sum();
        values.push(total);
    }

    let n = labels.len().min(values.len());
    if n == 0 {
        let ctx = RenderContext::new(config, w, h, None);
        return ctx.finish();
    }

    // Compute running cumulative values and Y extent
    let mut cumulative = Vec::with_capacity(n);
    let mut running = 0.0_f64;
    for i in 0..n {
        let is_total = wc.show_total && i == n - 1;
        let v = values[i];
        if is_total {
            // Total bar starts from 0
            cumulative.push((0.0, running));
        } else if v.is_finite() {
            let start = running;
            running += v;
            cumulative.push((start, running));
        } else {
            cumulative.push((running, running));
        }
    }

    let y_lo = cumulative
        .iter()
        .map(|(a, b)| a.min(*b))
        .reduce(f64::min)
        .unwrap_or(0.0)
        .min(0.0);
    let y_hi = cumulative
        .iter()
        .map(|(a, b)| a.max(*b))
        .reduce(f64::max)
        .unwrap_or(1.0)
        .max(0.0);

    let y_extent = resolve_y_extent(config, (y_lo, y_hi));
    let mut ctx = RenderContext::new(config, w, h, Some(y_extent));
    let (px, py, pw, ph) = ctx.plot;

    let y_scale = LinearScale::nice_zero(y_extent, ((py + ph) as f64, py as f64));
    let cat_scale = CategoricalScale::new(labels.clone(), (px as f64, (px + pw) as f64));

    // Axes
    let y_ticks = ctx.draw_y_axis(config, &y_scale);
    ctx.add_y_tick_overlays(&y_ticks, theme.text_color());
    ctx.draw_x_axis_line(config);
    ctx.draw_categorical_x_labels(config, &cat_scale, &labels);

    // Reference lines
    let x_dummy = LinearScale::new((0.0, 1.0), (px as f64, (px + pw) as f64));
    ctx.draw_reference_lines(config, &x_dummy, &y_scale);

    // ── Zero baseline: faint line at y=0 ──
    let zero_py = y_scale.to_pixel(0.0) as f32;
    if zero_py >= py && zero_py <= py + ph {
        let baseline_color = Color::from_rgba8(160, 160, 160, 128);
        ctx.draw(|c| {
            c.line(px, zero_py, px + pw, zero_py)
                .color(baseline_color)
                .width(1.0)
                .done()
        });
    }

    // Colors
    let increase_color = wc.increase_color.unwrap_or_else(|| {
        theme.resolve_series_color(0, &crate::data::SeriesStyle::default())
    });
    let decrease_color = wc.decrease_color.unwrap_or_else(|| {
        theme.resolve_series_color(1, &crate::data::SeriesStyle::default())
    });
    // ── Neutral total bar: gray instead of palette color ──
    let total_color = wc
        .total_color
        .unwrap_or(Color::from_rgba8(160, 160, 160, 255));

    let band = cat_scale.band_width() as f32;
    let bar_width = band * 0.7;

    // Draw bars and connectors
    for i in 0..n {
        let (bottom, top) = cumulative[i];
        let is_total = wc.show_total && i == n - 1;
        let color = if is_total {
            total_color
        } else if values[i] >= 0.0 {
            increase_color
        } else {
            decrease_color
        };

        let center = cat_scale.center(i) as f32;
        let py_top = y_scale.to_pixel(top.max(bottom)) as f32;
        let py_bot = y_scale.to_pixel(top.min(bottom)) as f32;
        let bar_h = (py_bot - py_top).max(0.5);

        ctx.draw(|c| {
            c.rect(center - bar_width / 2.0, py_top, bar_width, bar_h)
                .fill(color)
                .corner_radius(2.0)
                .done()
        });

        // ── Stronger connectors: higher alpha, neutral gray ──
        if wc.show_connectors && i + 1 < n {
            let connect_y = y_scale.to_pixel(cumulative[i].1) as f32;
            let next_center = cat_scale.center(i + 1) as f32;
            let conn_color = Color::from_rgba8(140, 140, 140, 166); // 0.65 alpha, neutral gray
            ctx.draw(|c| {
                c.line(
                    center + bar_width / 2.0,
                    connect_y,
                    next_center - bar_width / 2.0,
                    connect_y,
                )
                .color(conn_color)
                .width(1.0)
                .dash(scry_engine::style::DashPattern::new(vec![3.0, 2.0], 0.0))
                .done()
            });
        }

        // Value labels
        if wc.show_values {
            let val = values[i];
            if val.is_finite() {
                let label = format_value(val);
                let label_y = if val >= 0.0 || is_total {
                    py_top - 4.0
                } else {
                    py_bot + 12.0
                };
                ctx.overlays.push(TextOverlay {
                    x_px: center,
                    y_px: label_y,
                    text: label,
                    color: theme.text_color(),
                    align: TextAlign::Center,
                    font_size: data_fs,
                    bold: false,
                    rotation_deg: 0.0,
                });
            }
        }
    }

    ctx.add_common_overlays(config);
    ctx.finish()
}

/// Compact K/M formatter — replaces scientific notation with human-readable labels.
fn format_value(v: f64) -> String {
    let abs = v.abs();
    if abs >= 1_000_000.0 {
        format!("{:.1}M", v / 1_000_000.0)
    } else if abs >= 1000.0 {
        format!("{:.1}K", v / 1000.0)
    } else if v.fract().abs() < 1e-9 {
        format!("{}", v as i64)
    } else {
        format!("{v:.1}")
    }
}
