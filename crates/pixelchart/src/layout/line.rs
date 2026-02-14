//! Line chart rendering.

use crate::chart::LineChart;
use crate::legend::{self, LegendEntry};
use crate::scale::{LinearScale, Scale};
use ratatui_pixelcanvas::style::{GradientDef, GradientKind, GradientStop, Point};

use super::{
    resolve_x_extent, resolve_y_extent, RenderContext, RenderedChart, TextAlign, TextOverlay,
};

pub(crate) fn render_line(lc: &LineChart, w: u32, h: u32) -> RenderedChart {
    let config = &lc.config;
    let theme = &config.theme;

    // Pre-compute Y extent for measurement-based layout
    let (y_lo, y_hi) = if lc.stacked {
        // For stacked: y extent is 0..cumulative_max
        let max_len = lc.series.iter().map(|s| s.len()).max().unwrap_or(0);
        let mut cumsum = vec![0.0_f64; max_len];
        for series in &lc.series {
            for (i, &v) in series.values().iter().enumerate().take(max_len) {
                if v.is_finite() {
                    cumsum[i] += v;
                }
            }
        }
        let hi = cumsum.iter().copied().reduce(f64::max).unwrap_or(1.0);
        (0.0, hi)
    } else {
        let lo = lc
            .series
            .iter()
            .filter_map(|s| s.min())
            .reduce(f64::min)
            .unwrap_or(0.0);
        let hi = lc
            .series
            .iter()
            .filter_map(|s| s.max())
            .reduce(f64::max)
            .unwrap_or(1.0);
        (lo, hi)
    };
    let y_extent = resolve_y_extent(config, (y_lo, y_hi));

    let mut ctx = RenderContext::new(config, w, h, Some(y_extent));
    let (px, py, pw, ph) = ctx.plot;

    let max_len = lc.series.iter().map(|s| s.len()).max().unwrap_or(0);
    let x_data: Vec<f64> = lc
        .x_values
        .clone()
        .unwrap_or_else(|| (0..max_len).map(|i| i as f64).collect());
    let data_x_extent = if x_data.is_empty() {
        (0.0, 1.0)
    } else {
        let lo = x_data.iter().copied().reduce(f64::min).unwrap_or(0.0);
        let hi = x_data.iter().copied().reduce(f64::max).unwrap_or(1.0);
        (lo, hi)
    };

    let x_extent = resolve_x_extent(config, data_x_extent);

    // When user explicitly sets x_range/y_range with BOTH finite bounds,
    // use exact bounds (no nice rounding). For partial overrides (e.g. only
    // --y-min), resolve_x/y_extent already merged the user bound with data,
    // so we still use nice rounding to get clean tick marks.
    let x_exact = config
        .x_range
        .map_or(false, |(a, b)| a.is_finite() && b.is_finite());
    let y_exact = config
        .y_range
        .map_or(false, |(a, b)| a.is_finite() && b.is_finite());
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

    // Draw each series
    // For stacked charts, track cumulative y-values per x-index
    let mut cumulative: Vec<f64> = vec![0.0; x_data.len()];
    for (si, series) in lc.series.iter().enumerate() {
        let color = theme.series_color(si);
        let n = series.len().min(x_data.len());

        // Collect pixel coordinates for this series
        let points: Vec<(f32, f32)> = (0..n)
            .filter_map(|i| {
                let xv = x_data[i];
                let yv = series.values()[i];
                if !xv.is_finite() || !yv.is_finite() {
                    return None;
                }
                let effective_y = if lc.stacked { cumulative[i] + yv } else { yv };
                let sx = x_scale.to_pixel(xv) as f32;
                let sy = y_scale.to_pixel(effective_y) as f32;
                Some((sx, sy))
            })
            .collect();

        // Fill area under curve with vertical gradient
        if lc.fill_area && points.len() >= 2 {
            if lc.stacked && si > 0 {
                // Stacked: fill between this series and the previous cumulative baseline
                let prev_points: Vec<(f32, f32)> = (0..n)
                    .filter_map(|i| {
                        let xv = x_data[i];
                        if !xv.is_finite() || !series.values()[i].is_finite() {
                            return None;
                        }
                        let sx = x_scale.to_pixel(xv) as f32;
                        let sy = y_scale.to_pixel(cumulative[i]) as f32;
                        Some((sx, sy))
                    })
                    .collect();

                // Build polygon: top line forward + bottom line reversed
                let mut path_pts: Vec<(f32, f32)> = Vec::with_capacity(points.len() * 2);
                path_pts.extend_from_slice(&points);
                for &pt in prev_points.iter().rev() {
                    path_pts.push(pt);
                }

                let opacity = theme.fill_opacity();
                ctx.draw(|c| c.polygon(path_pts).fill(color.with_alpha(opacity)).done());
            } else {
                // Non-stacked fill or first stacked series: fill to baseline
                let baseline_y = y_scale.to_pixel(y_scale.domain().0) as f32;
                let mut path_points: Vec<(f32, f32)> = Vec::with_capacity(n + 2);
                path_points.push((points[0].0, baseline_y));
                path_points.extend_from_slice(&points);
                path_points.push((points.last().unwrap().0, baseline_y));

                let top_y = points
                    .iter()
                    .map(|(_, y)| *y)
                    .reduce(f32::min)
                    .unwrap_or(baseline_y);
                let opacity = theme.fill_opacity();
                let start_x = points[0].0;

                ctx.draw(|c| {
                    c.polygon(path_points)
                        .fill_linear_gradient(GradientDef {
                            kind: GradientKind::Linear {
                                start: Point::new(start_x, top_y),
                                end: Point::new(start_x, baseline_y),
                            },
                            stops: vec![
                                GradientStop {
                                    position: 0.0,
                                    color: color.with_alpha(opacity * 1.4),
                                },
                                GradientStop {
                                    position: 1.0,
                                    color: color.with_alpha(opacity * 0.08),
                                },
                            ],
                        })
                        .done()
                });
            }
        }

        // Lines — single polyline instead of N individual segments
        if points.len() >= 2 {
            let line_width = lc.line_width.unwrap_or_else(|| theme.line_width());
            if lc.dash_lines {
                if let Some(dash) = theme.series_dash(si) {
                    ctx.draw(|c| {
                        c.polyline(points.clone())
                            .stroke(color, line_width)
                            .dash(dash)
                            .done()
                    });
                } else {
                    ctx.draw(|c| c.polyline(points.clone()).stroke(color, line_width).done());
                }
            } else {
                ctx.draw(|c| c.polyline(points.clone()).stroke(color, line_width).done());
            }
        }

        // Data point markers
        if lc.show_points {
            let point_r = theme.point_radius() * 0.7;
            for &(sx, sy) in &points {
                ctx.draw(|c| c.circle(sx, sy, point_r).fill(color).done());
            }
        }

        // Error bars
        if let Some(errors) = series.error_values() {
            let cap_w = theme.point_radius() * 0.6;
            let err_color = color.with_alpha(0.8);
            for i in 0..n.min(errors.len()) {
                let xv = x_data[i];
                let yv = series.values()[i];
                let ev = errors[i];
                if !xv.is_finite() || !yv.is_finite() || !ev.is_finite() || ev <= 0.0 {
                    continue;
                }
                let effective_y = if lc.stacked { cumulative[i] + yv } else { yv };
                let sx = x_scale.to_pixel(xv) as f32;
                let sy_top = y_scale.to_pixel(effective_y + ev) as f32;
                let sy_bot = y_scale.to_pixel(effective_y - ev) as f32;
                ctx.draw(|c| {
                    c.line(sx, sy_top, sx, sy_bot)
                        .color(err_color)
                        .width(1.5)
                        .done()
                });
                ctx.draw(|c| {
                    c.line(sx - cap_w, sy_top, sx + cap_w, sy_top)
                        .color(err_color)
                        .width(1.5)
                        .done()
                });
                ctx.draw(|c| {
                    c.line(sx - cap_w, sy_bot, sx + cap_w, sy_bot)
                        .color(err_color)
                        .width(1.5)
                        .done()
                });
            }
        }

        // Data value labels
        if lc.show_values {
            let offset = theme.point_radius() * 0.7 + 4.0;
            for (idx, &(sx, sy)) in points.iter().enumerate() {
                if idx >= n {
                    break;
                }
                let yv = if lc.stacked {
                    cumulative[idx] + series.values()[idx]
                } else {
                    series.values()[idx]
                };
                if !yv.is_finite() {
                    continue;
                }
                let label = if yv.abs() >= 1000.0 || (yv.abs() < 0.01 && yv != 0.0) {
                    format!("{yv:.2e}")
                } else if yv.fract().abs() < 1e-9 {
                    format!("{}", yv as i64)
                } else {
                    format!("{yv:.1}")
                };
                ctx.overlays.push(super::TextOverlay {
                    x_px: sx,
                    y_px: sy - offset,
                    text: label,
                    color: theme.text_color(),
                    align: super::TextAlign::Center,
                    font_size: 9.0,
                    bold: false,
                    rotation_deg: 0.0,
                });
            }
        }

        // Update cumulative tracker for stacked mode
        if lc.stacked {
            for (i, &v) in series.values().iter().enumerate().take(n) {
                if v.is_finite() {
                    cumulative[i] += v;
                }
            }
        }
    }

    // Legend
    if lc.series.len() > 1 && config.show_legend {
        let entries: Vec<LegendEntry> = lc
            .series
            .iter()
            .enumerate()
            .map(|(i, s)| LegendEntry {
                label: if s.label().is_empty() {
                    format!("Series {}", i + 1)
                } else {
                    s.label().to_string()
                },
                color: theme.series_color(i),
            })
            .collect();

        let plot = ctx.plot;
        let legend_text = ctx.draw_with(|c| {
            legend::draw_positioned_legend(c, &entries, plot, &config.legend, 10.0, 4.0, None)
        });

        // Add legend text overlays
        for (lx, ly, label) in legend_text {
            ctx.overlays.push(TextOverlay {
                x_px: lx,
                y_px: ly,
                text: label,
                color: theme.text_color(),
                align: TextAlign::Left,
                font_size: 11.0,
                bold: false,
                rotation_deg: 0.0,
            });
        }
    }

    // Annotations
    if !config.annotations.is_empty() {
        ctx.draw_annotations(config, &x_scale, &y_scale);
    }

    ctx.add_common_overlays(config);
    ctx.finish()
}
