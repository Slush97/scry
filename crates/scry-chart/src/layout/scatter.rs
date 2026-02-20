// SPDX-License-Identifier: MIT OR Apache-2.0
//! Scatter plot rendering.

use scry_engine::scene::PixelCanvas;
use scry_engine::style::Color;

use crate::chart::scatter::{Marker, ScatterChart};
use crate::scale::{LinearScale, Scale};

use super::{resolve_x_extent, resolve_y_extent, RenderContext, RenderedChart};

pub(crate) fn render_scatter(sc: &ScatterChart, w: u32, h: u32) -> RenderedChart {
    let config = &sc.config;
    let theme = &config.theme;
    let data_fs = super::scaled_font_size(9.0, w, h);

    // Compute extent across *all* series so extra series aren't clipped.
    let mut x_lo = f64::INFINITY;
    let mut x_hi = f64::NEG_INFINITY;
    let mut y_lo = f64::INFINITY;
    let mut y_hi = f64::NEG_INFINITY;

    // Primary series
    if let Some((lo, hi)) = sc.x.extent() {
        x_lo = x_lo.min(lo);
        x_hi = x_hi.max(hi);
    }
    if let Some((lo, hi)) = sc.y.extent() {
        y_lo = y_lo.min(lo);
        y_hi = y_hi.max(hi);
    }
    // Extra series
    for (xs, ys) in &sc.extra_series {
        if let Some((lo, hi)) = xs.extent() {
            x_lo = x_lo.min(lo);
            x_hi = x_hi.max(hi);
        }
        if let Some((lo, hi)) = ys.extent() {
            y_lo = y_lo.min(lo);
            y_hi = y_hi.max(hi);
        }
    }

    let raw_x = if x_lo.is_finite() && x_hi.is_finite() {
        (x_lo, x_hi)
    } else {
        (0.0, 1.0)
    };
    let raw_y = if y_lo.is_finite() && y_hi.is_finite() {
        (y_lo, y_hi)
    } else {
        (0.0, 1.0)
    };

    // Pre-compute Y extent for measurement-based layout
    let y_extent = resolve_y_extent(config, raw_y);

    let mut ctx = RenderContext::new(config, w, h, Some(y_extent));
    let (px, py, pw, ph) = ctx.plot;

    let x_extent = resolve_x_extent(config, raw_x);

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

    // Draw data points for main series
    let color0 = theme.resolve_series_color(0, sc.y.series_style());
    let radius = theme.point_radius();
    let marker = sc.marker;
    for i in 0..sc.x.len().min(sc.y.len()) {
        let xv = sc.x.values()[i];
        let yv = sc.y.values()[i];
        if !xv.is_finite() || !yv.is_finite() {
            continue;
        }
        let sx = x_scale.to_pixel(xv) as f32;
        let sy = y_scale.to_pixel(yv) as f32;
        ctx.draw(|c| draw_marker(c, sx, sy, radius, color0, marker));
    }

    // Connect with lines if requested
    if sc.connect {
        let n = sc.x.len().min(sc.y.len());
        let line_w = theme.line_width() * 0.7;
        for i in 1..n {
            let xv1 = sc.x.values()[i - 1];
            let yv1 = sc.y.values()[i - 1];
            let xv2 = sc.x.values()[i];
            let yv2 = sc.y.values()[i];
            if !xv1.is_finite() || !yv1.is_finite() || !xv2.is_finite() || !yv2.is_finite() {
                continue;
            }
            let x1 = x_scale.to_pixel(xv1) as f32;
            let y1 = y_scale.to_pixel(yv1) as f32;
            let x2 = x_scale.to_pixel(xv2) as f32;
            let y2 = y_scale.to_pixel(yv2) as f32;
            ctx.draw(|c| c.line(x1, y1, x2, y2).color(color0).width(line_w).done());
        }
    }

    // Extra series
    for (si, (xs, ys)) in sc.extra_series.iter().enumerate() {
        let color = theme.resolve_series_color(si + 1, ys.series_style());
        let n = xs.len().min(ys.len());
        for i in 0..n {
            let xv = xs.values()[i];
            let yv = ys.values()[i];
            if !xv.is_finite() || !yv.is_finite() {
                continue;
            }
            let sx = x_scale.to_pixel(xv) as f32;
            let sy = y_scale.to_pixel(yv) as f32;
            ctx.draw(|c| draw_marker(c, sx, sy, radius, color, marker));
        }
        // Connect extra series with lines when requested
        if sc.connect {
            let line_w = theme.line_width() * 0.7;
            for i in 1..n {
                let xv1 = xs.values()[i - 1];
                let yv1 = ys.values()[i - 1];
                let xv2 = xs.values()[i];
                let yv2 = ys.values()[i];
                if !xv1.is_finite() || !yv1.is_finite() || !xv2.is_finite() || !yv2.is_finite() {
                    continue;
                }
                let x1 = x_scale.to_pixel(xv1) as f32;
                let y1 = y_scale.to_pixel(yv1) as f32;
                let x2 = x_scale.to_pixel(xv2) as f32;
                let y2 = y_scale.to_pixel(yv2) as f32;
                ctx.draw(|c| c.line(x1, y1, x2, y2).color(color).width(line_w).done());
            }
        }
    }

    // Legend for multi-series scatter
    let total_series = 1 + sc.extra_series.len();
    if total_series > 1 && config.show_legend {
        use crate::legend::{self, LegendEntry};
        use super::TextAlign;

        // Collect all marker pixel positions for overlap detection.
        let mut all_points: Vec<(f32, f32)> = Vec::new();
        let n0 = sc.x.len().min(sc.y.len());
        for i in 0..n0 {
            let xv = sc.x.values()[i];
            let yv = sc.y.values()[i];
            if xv.is_finite() && yv.is_finite() {
                all_points.push((x_scale.to_pixel(xv) as f32, y_scale.to_pixel(yv) as f32));
            }
        }
        for (xs, ys) in &sc.extra_series {
            let n = xs.len().min(ys.len());
            for i in 0..n {
                let xv = xs.values()[i];
                let yv = ys.values()[i];
                if xv.is_finite() && yv.is_finite() {
                    all_points.push((x_scale.to_pixel(xv) as f32, y_scale.to_pixel(yv) as f32));
                }
            }
        }

        let mut entries = Vec::with_capacity(total_series);
        // Primary series
        let primary_label = if sc.y.label().is_empty() {
            "Series 1".to_string()
        } else {
            sc.y.label().to_string()
        };
        entries.push(LegendEntry {
            label: primary_label,
            color: theme.resolve_series_color(0, sc.y.series_style()),
        });
        // Extra series
        for (si, (_, ys)) in sc.extra_series.iter().enumerate() {
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
        let legend_fs = super::scaled_font_size(theme.legend.font_size, w, h);
        let mut legend_cfg = config.legend.clone();
        legend_cfg.apply_theme_and_font_size(&theme.legend, legend_fs);
        // Scatter plots use circle swatches (Cleveland 1985 — match marker shape)
        legend_cfg.swatch_shape = crate::legend::SwatchShape::Circle;
        let data_pts = if all_points.is_empty() { None } else { Some(all_points.as_slice()) };
        let legend_text = ctx.draw_with(|c| {
            legend::draw_positioned_legend(c, &entries, plot, &legend_cfg, 10.0, 4.0, data_pts)
        });
        for (lx, ly, label) in legend_text {
            ctx.add_text(lx, ly, &label, theme.text_color(), TextAlign::Left, legend_fs, false, 0.0);
        }
    }

    // Error bars on main series (symmetric or asymmetric)
    let has_sym = sc.y.error_values().is_some();
    let has_asym = sc.y.error_low().is_some() && sc.y.error_high().is_some();
    if has_sym || has_asym {
        let cap_w = radius * 0.8;
        let err_color = color0.with_alpha(0.8);
        let n = sc.x.len().min(sc.y.len());
        let sym_errors = sc.y.error_values();
        let asym_lo = sc.y.error_low();
        let asym_hi = sc.y.error_high();
        for i in 0..n {
            let xv = sc.x.values()[i];
            let yv = sc.y.values()[i];
            if !xv.is_finite() || !yv.is_finite() {
                continue;
            }

            // Determine lower and upper offsets.
            let (lo_off, hi_off) = if let (Some(lo), Some(hi)) = (asym_lo, asym_hi) {
                if i >= lo.len() || i >= hi.len() { continue; }
                let l = lo[i];
                let h = hi[i];
                if !l.is_finite() || !h.is_finite() { continue; }
                (l, h)
            } else if let Some(errs) = sym_errors {
                if i >= errs.len() { continue; }
                let ev = errs[i];
                if !ev.is_finite() || ev <= 0.0 { continue; }
                (ev, ev)
            } else {
                continue;
            };

            let sx = x_scale.to_pixel(xv) as f32;
            let sy_top = y_scale.to_pixel(yv + hi_off) as f32;
            let sy_bot = y_scale.to_pixel(yv - lo_off) as f32;
            // Vertical line
            ctx.draw(|c| {
                c.line(sx, sy_top, sx, sy_bot)
                    .color(err_color)
                    .width(1.5)
                    .done()
            });
            // Top cap
            ctx.draw(|c| {
                c.line(sx - cap_w, sy_top, sx + cap_w, sy_top)
                    .color(err_color)
                    .width(1.5)
                    .done()
            });
            // Bottom cap
            ctx.draw(|c| {
                c.line(sx - cap_w, sy_bot, sx + cap_w, sy_bot)
                    .color(err_color)
                    .width(1.5)
                    .done()
            });
        }
    }

    // Data value labels
    if sc.show_values {
        for i in 0..sc.x.len().min(sc.y.len()) {
            let xv = sc.x.values()[i];
            let yv = sc.y.values()[i];
            if !xv.is_finite() || !yv.is_finite() {
                continue;
            }
            let sx = x_scale.to_pixel(xv) as f32;
            let sy = y_scale.to_pixel(yv) as f32;
            let label = if yv.abs() >= 1000.0 || (yv.abs() < 0.01 && yv != 0.0) {
                format!("{yv:.2e}")
            } else if yv.fract().abs() < 1e-9 {
                format!("{}", yv as i64)
            } else {
                format!("{yv:.1}")
            };
            ctx.add_text(sx, sy - radius - 4.0, &label, theme.text_color(), super::TextAlign::Center, data_fs, false, 0.0);
        }
    }

    // Trend line (linear regression)
    if config.overlays.show_trend {
        ctx.draw_trend_line(
            sc.x.values(),
            sc.y.values(),
            &x_scale,
            &y_scale,
            theme.resolve_series_color(0, sc.y.series_style()),
        );
    }

    // Annotations
    if !config.overlays.annotations.is_empty() {
        ctx.draw_annotations(config, &x_scale, &y_scale);
    }

    ctx.add_common_overlays(config);
    ctx.finish()
}

/// Draw a shaped marker on the canvas with optional border stroke.
pub(crate) fn draw_marker(
    canvas: PixelCanvas,
    x: f32,
    y: f32,
    radius: f32,
    color: Color,
    marker: Marker,
) -> PixelCanvas {
    // Derive a subtle border color: darkened variant at 50% opacity
    let border = color.with_alpha(0.5);

    match marker {
        Marker::Circle => canvas
            .circle(x, y, radius)
            .fill(color)
            .stroke(border, 1.0)
            .done(),
        Marker::Square => {
            let half = radius * 0.85;
            canvas
                .rect(x - half, y - half, half * 2.0, half * 2.0)
                .fill(color)
                .stroke(border, 1.0)
                .done()
        }
        Marker::Diamond => {
            let r = radius * 1.1;
            canvas
                .polygon(vec![(x, y - r), (x + r, y), (x, y + r), (x - r, y)])
                .fill(color)
                .stroke(border, 1.0)
                .done()
        }
        Marker::Cross => {
            let r = radius * 0.8;
            let w = radius * 0.4;
            let c = canvas
                .rect(x - r, y - w / 2.0, r * 2.0, w)
                .fill(color)
                .done();
            c.rect(x - w / 2.0, y - r, w, r * 2.0).fill(color).done()
        }
        Marker::Triangle => {
            let r = radius * 1.1;
            canvas
                .polygon(vec![
                    (x, y - r),
                    (x + r * 0.866, y + r * 0.5),
                    (x - r * 0.866, y + r * 0.5),
                ])
                .fill(color)
                .stroke(border, 1.0)
                .done()
        }
    }
}
