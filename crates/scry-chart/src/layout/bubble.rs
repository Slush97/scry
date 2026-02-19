// SPDX-License-Identifier: MIT OR Apache-2.0
//! Bubble chart rendering — scatter with variable-size markers.

use crate::chart::bubble::BubbleChart;
use crate::legend::{self, LegendEntry};
use crate::scale::{LinearScale, Scale};

use super::scatter::draw_marker;
use super::{
    resolve_x_extent, resolve_y_extent, RenderContext, RenderedChart, TextAlign,
};

pub(crate) fn render_bubble(bc: &BubbleChart, w: u32, h: u32) -> RenderedChart {
    let config = &bc.config;
    let theme = &config.theme;
    let data_fs = super::scaled_font_size(9.0, w, h);
    let tick_fs = super::scaled_font_size(theme.tick_style.font_size, w, h);

    // Pre-compute Y extent
    let y_extent = resolve_y_extent(config, bc.y.extent().unwrap_or((0.0, 1.0)));
    let mut ctx = RenderContext::new(config, w, h, Some(y_extent));
    let (px, py, pw, ph) = ctx.plot;

    let x_extent = resolve_x_extent(config, bc.x.extent().unwrap_or((0.0, 1.0)));

    let x_exact = config
        .axes
        .x_range
        .is_some_and(|(a, b): (f64, f64)| a.is_finite() && b.is_finite());
    let y_exact = config
        .axes
        .y_range
        .is_some_and(|(a, b): (f64, f64)| a.is_finite() && b.is_finite());
    let x_scale = if x_exact {
        LinearScale::new(x_extent, (px as f64, (px + pw) as f64))
    } else {
        LinearScale::nice(x_extent, (px as f64, (px + pw) as f64))
    };
    let y_scale = if y_exact {
        LinearScale::new(y_extent, ((py + ph) as f64, py as f64))
    } else {
        LinearScale::nice(y_extent, ((py + ph) as f64, py as f64))
    };

    ctx.draw_axes(config, &x_scale, &y_scale);
    ctx.draw_reference_lines(config, &x_scale, &y_scale);

    // Compute global size extent across all series for consistent mapping
    let (size_min, size_max) = {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &s in &bc.sizes {
            if s.is_finite() {
                lo = lo.min(s);
                hi = hi.max(s);
            }
        }
        for (_, _, sizes) in &bc.extra_series {
            for &s in sizes {
                if s.is_finite() {
                    lo = lo.min(s);
                    hi = hi.max(s);
                }
            }
        }
        if lo > hi {
            (0.0, 1.0)
        } else {
            (lo, hi)
        }
    };

    let min_r = bc.min_radius;
    let max_r = bc.max_radius;
    let size_range = size_max - size_min;

    let map_radius = |size: f64| -> f32 {
        if size_range < f64::EPSILON {
            (min_r + max_r) / 2.0
        } else {
            let t = ((size - size_min) / size_range) as f32;
            min_r + t * (max_r - min_r)
        }
    };

    let marker = bc.marker;
    let opacity = bc.opacity;

    // Draw main series
    let color0 = theme
        .resolve_series_color(0, bc.y.series_style())
        .with_alpha(opacity);
    let n = bc.x.len().min(bc.y.len()).min(bc.sizes.len());
    for i in 0..n {
        let xv = bc.x.values()[i];
        let yv = bc.y.values()[i];
        let sv = bc.sizes[i];
        if !xv.is_finite() || !yv.is_finite() || !sv.is_finite() {
            continue;
        }
        let sx = x_scale.to_pixel(xv) as f32;
        let sy = y_scale.to_pixel(yv) as f32;
        let radius = map_radius(sv);
        ctx.draw(|c| draw_marker(c, sx, sy, radius, color0, marker));
    }

    // Draw extra series
    for (si, (xs, ys, sizes)) in bc.extra_series.iter().enumerate() {
        let color = theme
            .resolve_series_color(si + 1, ys.series_style())
            .with_alpha(opacity);
        let sn = xs.len().min(ys.len()).min(sizes.len());
        for i in 0..sn {
            let xv = xs.values()[i];
            let yv = ys.values()[i];
            let sv = sizes[i];
            if !xv.is_finite() || !yv.is_finite() || !sv.is_finite() {
                continue;
            }
            let sx = x_scale.to_pixel(xv) as f32;
            let sy = y_scale.to_pixel(yv) as f32;
            let radius = map_radius(sv);
            ctx.draw(|c| draw_marker(c, sx, sy, radius, color, marker));
        }
    }

    // Data value labels
    if bc.show_values {
        for i in 0..n {
            let xv = bc.x.values()[i];
            let yv = bc.y.values()[i];
            let sv = bc.sizes[i];
            if !xv.is_finite() || !yv.is_finite() || !sv.is_finite() {
                continue;
            }
            let sx = x_scale.to_pixel(xv) as f32;
            let sy = y_scale.to_pixel(yv) as f32;
            let radius = map_radius(sv);
            let label = format_value(sv);
            ctx.add_text(sx, sy - radius - 4.0, &label, theme.text_color(), TextAlign::Center, data_fs, false, 0.0);
        }
    }

    // Annotations
    if !config.overlays.annotations.is_empty() {
        ctx.draw_annotations(config, &x_scale, &y_scale);
    }

    // Legend for multi-series
    let total_series = 1 + bc.extra_series.len();
    if total_series > 1 && config.show_legend {
        let mut entries = Vec::with_capacity(total_series);
        // Primary series
        let primary_label = if bc.y.label().is_empty() {
            "Series 1".to_string()
        } else {
            bc.y.label().to_string()
        };
        entries.push(LegendEntry {
            label: primary_label,
            color: theme.resolve_series_color(0, bc.y.series_style()),
        });
        // Extra series
        for (si, (_, ys, _)) in bc.extra_series.iter().enumerate() {
            let label = if ys.label().is_empty() {
                format!("Series {}", si + 2)
            } else {
                ys.label().to_string()
            };
            entries.push(LegendEntry {
                label,
                color: theme.resolve_series_color(si + 1, ys.series_style()),
            });
        }

        let plot = ctx.plot;
        let legend_text = ctx.draw_with(|c| {
            legend::draw_positioned_legend(c, &entries, plot, &config.legend, 10.0, 4.0, None)
        });
        for (lx, ly, label) in legend_text {
            ctx.add_text(lx, ly, &label, theme.text_color(), TextAlign::Left, tick_fs, false, 0.0);
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
