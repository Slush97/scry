// SPDX-License-Identifier: MIT OR Apache-2.0
//! Line chart rendering.

use crate::chart::LineChart;
use crate::data::GapPolicy;
use crate::legend::{self, LegendEntry};
use crate::scale::{LinearScale, Scale};
use scry_engine::style::{GradientDef, GradientKind, GradientStop, Point};

use super::{
    resolve_x_extent, resolve_y_extent, RenderContext, RenderedChart, TextAlign, TextOverlay,
};

/// A contiguous segment of pixel-space points with their original data indices.
struct Segment {
    /// Pixel coordinates for this contiguous run of finite values.
    points: Vec<(f32, f32)>,
    /// Original data indices corresponding to each point in `points`.
    indices: Vec<usize>,
}

/// Preprocess y-values according to the gap policy, then map to pixel-space
/// segments. Returns one or more `Segment`s depending on the policy.
///
/// - `Zero`: replaces NaN with 0.0 → single segment
/// - `Interpolate`: linearly fills NaN gaps → single segment
/// - `Skip`: splits at NaN boundaries → multiple segments
#[allow(clippy::too_many_arguments)]
fn build_segments(
    values: &[f64],
    x_data: &[f64],
    n: usize,
    gap_policy: GapPolicy,
    stacked: bool,
    cumulative: &[f64],
    x_scale: &LinearScale,
    y_scale: &LinearScale,
) -> Vec<Segment> {
    // First, produce effective y-values based on the gap policy.
    let effective: Vec<Option<f64>> = match gap_policy {
        GapPolicy::Zero => (0..n)
            .map(|i| {
                let xv = x_data[i];
                if !xv.is_finite() {
                    return None;
                }
                let yv = if values[i].is_finite() {
                    values[i]
                } else {
                    0.0
                };
                let eff = if stacked { cumulative[i] + yv } else { yv };
                Some(eff)
            })
            .collect(),
        GapPolicy::Interpolate => {
            // Build the raw effective values, keeping None for non-finite.
            let raw: Vec<Option<f64>> = (0..n)
                .map(|i| {
                    let xv = x_data[i];
                    let yv = values[i];
                    if !xv.is_finite() || !yv.is_finite() {
                        None
                    } else {
                        let eff = if stacked { cumulative[i] + yv } else { yv };
                        Some(eff)
                    }
                })
                .collect();
            // Linearly interpolate across None gaps.
            let mut result = raw.clone();
            let mut i = 0;
            while i < n {
                if result[i].is_none() {
                    // Find the previous finite value.
                    let prev = if i > 0 {
                        (0..i).rev().find_map(|j| result[j].map(|v| (j, v)))
                    } else {
                        None
                    };
                    // Find the next finite value.
                    let next = ((i + 1)..n).find_map(|j| result[j].map(|v| (j, v)));

                    match (prev, next) {
                        (Some((pj, pv)), Some((nj, nv))) => {
                            // Interpolate between prev and next.
                            for k in (pj + 1)..nj {
                                if result[k].is_none() && x_data[k].is_finite() {
                                    let t = (k - pj) as f64 / (nj - pj) as f64;
                                    result[k] = Some(pv + t * (nv - pv));
                                }
                            }
                            i = nj + 1;
                        }
                        _ => {
                            // No neighbors — can't interpolate, leave as gap.
                            i += 1;
                        }
                    }
                } else {
                    i += 1;
                }
            }
            result
        }
        GapPolicy::Skip => (0..n)
            .map(|i| {
                let xv = x_data[i];
                let yv = values[i];
                if !xv.is_finite() || !yv.is_finite() {
                    None
                } else {
                    let eff = if stacked { cumulative[i] + yv } else { yv };
                    Some(eff)
                }
            })
            .collect(),
    };

    // Convert to pixel-space and split into segments at None boundaries.
    let mut segments = Vec::new();
    let mut current_points = Vec::new();
    let mut current_indices = Vec::new();

    for (i, val) in effective.iter().enumerate().take(n) {
        if let Some(eff_y) = val {
            let sx = x_scale.to_pixel(x_data[i]) as f32;
            let sy = y_scale.to_pixel(*eff_y) as f32;
            current_points.push((sx, sy));
            current_indices.push(i);
        } else if !current_points.is_empty() {
            // End current segment at gap.
            segments.push(Segment {
                points: std::mem::take(&mut current_points),
                indices: std::mem::take(&mut current_indices),
            });
        }
    }
    // Final segment.
    if !current_points.is_empty() {
        segments.push(Segment {
            points: current_points,
            indices: current_indices,
        });
    }

    segments
}


pub(crate) fn render_line(lc: &LineChart, w: u32, h: u32) -> RenderedChart {
    let config = &lc.config;
    let theme = &config.theme;
    let data_fs = super::scaled_font_size(9.0, w, h);
    let tick_fs = super::scaled_font_size(theme.tick_style.font_size, w, h);

    // Classify series into primary (left Y) and secondary (right Y)
    let sec_indices = &lc.config.secondary_series_indices;
    let has_secondary = !sec_indices.is_empty()
        && config.secondary_y_range.is_some();

    // Pre-compute Y extent for primary series only
    let (y_lo, y_hi) = if lc.stacked {
        let max_len = lc.series.iter().map(|s| s.len()).max().unwrap_or(0);
        let mut cumsum = vec![0.0_f64; max_len];
        for (si, series) in lc.series.iter().enumerate() {
            if sec_indices.contains(&si) {
                continue; // skip secondary series for primary extent
            }
            for (i, &v) in series.values().iter().enumerate().take(max_len) {
                if v.is_finite() {
                    cumsum[i] += v;
                }
            }
        }
        let hi = cumsum.iter().copied().reduce(f64::max).unwrap_or(1.0);
        (0.0, hi)
    } else {
        let primary_series = lc.series.iter().enumerate()
            .filter(|(i, _)| !sec_indices.contains(i));
        let lo = primary_series.clone()
            .filter_map(|(_, s)| s.min())
            .reduce(f64::min)
            .unwrap_or(0.0);
        let hi = primary_series
            .filter_map(|(_, s)| s.max())
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

    let x_exact = config
        .x_range
        .is_some_and(|(a, b)| a.is_finite() && b.is_finite());
    let y_exact = config
        .y_range
        .is_some_and(|(a, b)| a.is_finite() && b.is_finite());
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

    // Build secondary Y scale if configured
    let secondary_y_scale = if has_secondary {
        let (sy_lo, sy_hi) = config.secondary_y_range.unwrap();
        let sy_scale = LinearScale::nice(
            (sy_lo, sy_hi),
            ((py + ph) as f64, py as f64),
        );

        // Draw secondary (right) Y axis ticks
        let sy_cfg = super::axis_config_from_theme_secondary(config);
        let plot = ctx.plot;
        let sy_ticks = ctx.draw_with(|c| {
            crate::axis::draw_axis(c, plot, &sy_scale, &sy_cfg)
        });

        // Add right-side tick overlays
        let y_off = super::y_tick_label_offset(w);
        for (y_pos, label) in &sy_ticks {
            ctx.overlays.push(super::TextOverlay {
                x_px: px + pw + y_off,
                y_px: *y_pos,
                text: label.clone(),
                color: theme.foreground,
                align: super::TextAlign::Left,
                font_size: tick_fs,
                bold: false,
                rotation_deg: 0.0,
            });
        }

        Some(sy_scale)
    } else {
        None
    };

    // Draw each series
    // For stacked charts, track cumulative y-values per x-index
    let mut cumulative: Vec<f64> = vec![0.0; x_data.len()];
    for (si, series) in lc.series.iter().enumerate() {
        // Choose which Y scale this series should use
        let is_secondary = has_secondary && sec_indices.contains(&si);
        let active_y_scale = if is_secondary {
            secondary_y_scale.as_ref().unwrap()
        } else {
            &y_scale
        };

        let sty = series.series_style();
        let color = theme.resolve_series_color(si, sty);
        let n = series.len().min(x_data.len());

        // Build segments according to gap policy
        let segments = build_segments(
            series.values(),
            &x_data,
            n,
            lc.gap_policy,
            lc.stacked,
            &cumulative,
            &x_scale,
            active_y_scale,
        );

        // Render each contiguous segment independently
        for seg in &segments {
            let points = &seg.points;

            // Fill area under curve with vertical gradient
            if lc.fill_area && points.len() >= 2 {
                if lc.stacked && si > 0 {
                    // Stacked: fill between this series and the previous cumulative baseline
                    let prev_points: Vec<(f32, f32)> = seg
                        .indices
                        .iter()
                        .map(|&i| {
                            let sx = x_scale.to_pixel(x_data[i]) as f32;
                            let sy = active_y_scale.to_pixel(cumulative[i]) as f32;
                            (sx, sy)
                        })
                        .collect();

                    // Build polygon: top line forward + bottom line reversed
                    let mut path_pts: Vec<(f32, f32)> =
                        Vec::with_capacity(points.len() * 2);
                    path_pts.extend_from_slice(points);
                    for &pt in prev_points.iter().rev() {
                        path_pts.push(pt);
                    }

                    let opacity = theme.resolve_fill_opacity(sty);
                    ctx.draw(|c| {
                        c.polygon(path_pts).fill(color.with_alpha(opacity)).done()
                    });
                } else {
                    // Non-stacked fill or first stacked series: fill to baseline
                    let baseline_y =
                        active_y_scale.to_pixel(active_y_scale.domain().0) as f32;
                    let mut path_points: Vec<(f32, f32)> =
                        Vec::with_capacity(points.len() + 2);
                    path_points.push((points[0].0, baseline_y));
                    path_points.extend_from_slice(points);
                    path_points.push((points.last().unwrap().0, baseline_y));

                    let top_y = points
                        .iter()
                        .map(|(_, y)| *y)
                        .reduce(f32::min)
                        .unwrap_or(baseline_y);
                    let opacity = theme.resolve_fill_opacity(sty);
                    let start_x = points[0].0;

                    // Build gradient based on per-series override or default
                    let gradient = match sty.fill_gradient.as_ref() {
                        Some(crate::data::GradientFill::BottomToTop) => GradientDef {
                            kind: GradientKind::Linear {
                                start: Point::new(start_x, baseline_y),
                                end: Point::new(start_x, top_y),
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
                        },
                        Some(crate::data::GradientFill::Custom(stops))
                            if stops.len() >= 2 =>
                        {
                            GradientDef {
                                kind: GradientKind::Linear {
                                    start: Point::new(start_x, top_y),
                                    end: Point::new(start_x, baseline_y),
                                },
                                stops: stops
                                    .iter()
                                    .map(|&(pos, c)| GradientStop {
                                        position: pos,
                                        color: c,
                                    })
                                    .collect(),
                            }
                        }
                        // TopToBottom or default (including Custom with < 2 stops)
                        _ => GradientDef {
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
                        },
                    };

                    ctx.draw(|c| {
                        c.polygon(path_points)
                            .fill_linear_gradient(gradient)
                            .done()
                    });
                }
            }

            // Lines — one polyline per segment
            if points.len() >= 2 {
                let line_width = theme.resolve_line_width(sty, lc.line_width);
                if let Some(dash) = theme.resolve_series_dash(si, sty, lc.dash_lines) {
                    ctx.draw(|c| {
                        c.polyline(points.clone())
                            .stroke(color, line_width)
                            .dash(dash)
                            .done()
                    });
                } else {
                    ctx.draw(|c| {
                        c.polyline(points.clone())
                            .stroke(color, line_width)
                            .done()
                    });
                }
            }

            // Data point markers
            if lc.show_points {
                let point_r = theme.point_radius() * 0.7;
                for &(sx, sy) in points {
                    ctx.draw(|c| c.circle(sx, sy, point_r).fill(color).done());
                }
            }

            // Data value labels
            if lc.show_values {
                let offset = theme.point_radius() * 0.7 + 4.0;
                for (pt_idx, &(sx, sy)) in points.iter().enumerate() {
                    let data_idx = seg.indices[pt_idx];
                    let yv = if lc.stacked {
                        cumulative[data_idx] + series.values()[data_idx]
                    } else {
                        series.values()[data_idx]
                    };
                    if !yv.is_finite() {
                        continue;
                    }
                    let label =
                        if yv.abs() >= 1000.0 || (yv.abs() < 0.01 && yv != 0.0) {
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
                        font_size: data_fs,
                        bold: false,
                        rotation_deg: 0.0,
                    });
                }
            }
        }

        // Error bars — use original data indices directly (independent of segments)
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
                let sy_top = active_y_scale.to_pixel(effective_y + ev) as f32;
                let sy_bot = active_y_scale.to_pixel(effective_y - ev) as f32;
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
                color: theme.resolve_series_color(i, s.series_style()),
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
                font_size: tick_fs,
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
