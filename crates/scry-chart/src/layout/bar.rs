// SPDX-License-Identifier: MIT OR Apache-2.0
//! Bar chart rendering.

use crate::chart::BarChart;
use crate::data::FillPattern;
use crate::legend::{self, LegendEntry};
use crate::scale::{CategoricalScale, LinearScale, Scale};

use super::{resolve_y_extent, RenderContext, RenderedChart, TextAlign, TextOverlay};

pub(crate) fn render_bar(bc: &BarChart, w: u32, h: u32) -> RenderedChart {
    if bc.horizontal {
        render_bar_horizontal(bc, w, h)
    } else {
        render_bar_vertical(bc, w, h)
    }
}

fn render_bar_vertical(bc: &BarChart, w: u32, h: u32) -> RenderedChart {
    let config = &bc.config;
    let theme = &config.theme;
    let data_fs = super::scaled_font_size(9.0, w, h);

    // Pre-compute Y extent for measurement-based layout
    let (y_lo, y_hi) = compute_value_extent(bc);
    let y_extent = resolve_y_extent(config, (y_lo, y_hi));

    let mut ctx = RenderContext::new(config, w, h, Some(y_extent));
    let (px, py, pw, ph) = ctx.plot;

    let y_scale = LinearScale::nice_zero(y_extent, ((py + ph) as f64, py as f64));

    let cat_scale = CategoricalScale::new(bc.labels.clone(), (px as f64, (px + pw) as f64));

    // Y axis
    let y_ticks = ctx.draw_y_axis(config, &y_scale);
    ctx.add_y_tick_overlays(&y_ticks, theme.text_color());

    // X axis line
    ctx.draw_x_axis_line(config);

    // Reference lines
    let x_dummy = LinearScale::new((0.0, 1.0), (px as f64, (px + pw) as f64));
    ctx.draw_reference_lines(config, &x_dummy, &y_scale);

    // ── Emphasized zero-line for ± data ──
    if y_extent.0 < 0.0 && y_extent.1 > 0.0 {
        let zero_y = y_scale.to_pixel(0.0) as f32;
        let zero_color = theme.axis_color().with_alpha(0.9);
        ctx.draw(|c| {
            c.line(px, zero_y, px + pw, zero_y)
                .color(zero_color)
                .width(theme.axis_width() * 2.0)
                .done()
        });
    }

    // Draw bars
    let n_series = bc.series.len();
    let band = cat_scale.band_width() as f32;
    let inner_band = band * (1.0 - bc.bar_gap);
    let corner_r = bc.corner_radius.unwrap_or(theme.series.bar_corner_radius);
    let stroke_w = theme.bar_stroke_width();

    for (ci, _label) in bc.labels.iter().enumerate() {
        let center = cat_scale.center(ci) as f32;

        if bc.stacked {
            let bar_width = inner_band;
            let bar_x = center - bar_width / 2.0;
            let mut cumulative = 0.0;

            for (si, series) in bc.series.iter().enumerate() {
                if ci >= series.len() {
                    continue;
                }
                let value = series.values()[ci];
                if !value.is_finite() {
                    continue;
                }
                let bottom_y = y_scale.to_pixel(cumulative) as f32;
                let top_y = y_scale.to_pixel(cumulative + value) as f32;
                let (rect_y, bar_h) = if top_y <= bottom_y {
                    (top_y, bottom_y - top_y)
                } else {
                    (bottom_y, top_y - bottom_y)
                };

                if bar_h > 0.5 {
                    let color = theme.resolve_series_color(si, series.series_style());
                    let cr = if si == n_series - 1 { corner_r } else { 0.0 };
                    ctx.draw(|c| {
                        c.rect(bar_x, rect_y, bar_width, bar_h)
                            .fill(color)
                            .corner_radius(cr)
                            .done()
                    });
                    if stroke_w > 0.0 {
                        let sc = color.with_alpha(0.5);
                        ctx.draw(|c| {
                            c.rect(bar_x, rect_y, bar_width, bar_h)
                                .stroke(sc, stroke_w)
                                .corner_radius(cr)
                                .done()
                        });
                    }
                    draw_fill_pattern(
                        &mut ctx,
                        series.series_style().fill_pattern.as_ref(),
                        color,
                        bar_x,
                        rect_y,
                        bar_width,
                        bar_h,
                    );
                }
                cumulative += value;
            }
        } else {
            let bar_width = if n_series > 0 {
                inner_band / n_series as f32
            } else {
                inner_band
            };
            let group_left = center - inner_band / 2.0;

            for (si, series) in bc.series.iter().enumerate() {
                if ci >= series.len() {
                    continue;
                }
                let value = series.values()[ci];
                if !value.is_finite() {
                    continue;
                }
                let baseline_y = y_scale.to_pixel(0.0) as f32;
                let top_y = y_scale.to_pixel(value) as f32;
                let bar_x = group_left + si as f32 * bar_width;
                let (rect_y, bar_h) = if value >= 0.0 {
                    (top_y, baseline_y - top_y)
                } else {
                    (baseline_y, top_y - baseline_y)
                };

                if bar_h > 0.5 {
                    let color = theme.resolve_series_color(si, series.series_style());
                    ctx.draw(|c| {
                        c.rect(bar_x, rect_y, bar_width, bar_h)
                            .fill(color)
                            .corner_radius(corner_r)
                            .done()
                    });
                    if stroke_w > 0.0 {
                        let sc = color.with_alpha(0.5);
                        ctx.draw(|c| {
                            c.rect(bar_x, rect_y, bar_width, bar_h)
                                .stroke(sc, stroke_w)
                                .corner_radius(corner_r)
                                .done()
                        });
                    }
                    draw_fill_pattern(
                        &mut ctx,
                        series.series_style().fill_pattern.as_ref(),
                        color,
                        bar_x,
                        rect_y,
                        bar_width,
                        bar_h,
                    );
                }
            }
        }
    }

    // Error bars on vertical bars (non-stacked only)
    if !bc.stacked {
        let bar_width = if n_series > 0 {
            inner_band / n_series as f32
        } else {
            inner_band
        };
        for (ci, _label) in bc.labels.iter().enumerate() {
            let center = cat_scale.center(ci) as f32;
            let group_left = center - inner_band / 2.0;
            for (si, series) in bc.series.iter().enumerate() {
                if ci >= series.len() {
                    continue;
                }
                let errors = match series.error_values() {
                    Some(e) if ci < e.len() => e[ci],
                    _ => continue,
                };
                let value = series.values()[ci];
                if !value.is_finite() || !errors.is_finite() || errors <= 0.0 {
                    continue;
                }
                let bar_center_x = group_left + si as f32 * bar_width + bar_width / 2.0;
                let cap_w = bar_width * 0.25;
                let color = theme.resolve_series_color(si, series.series_style());
                let err_color = color.with_alpha(0.8);
                let sy_top = y_scale.to_pixel(value + errors) as f32;
                let sy_bot = y_scale.to_pixel(value - errors) as f32;
                ctx.draw(|c| {
                    c.line(bar_center_x, sy_top, bar_center_x, sy_bot)
                        .color(err_color)
                        .width(1.5)
                        .done()
                });
                ctx.draw(|c| {
                    c.line(bar_center_x - cap_w, sy_top, bar_center_x + cap_w, sy_top)
                        .color(err_color)
                        .width(1.5)
                        .done()
                });
                ctx.draw(|c| {
                    c.line(bar_center_x - cap_w, sy_bot, bar_center_x + cap_w, sy_bot)
                        .color(err_color)
                        .width(1.5)
                        .done()
                });
            }
        }
    }

    // Value labels above bars
    if bc.show_values {
        let val_start = ctx.overlays.len();
        let text_color = theme.text_color();
        for (ci, _label) in bc.labels.iter().enumerate() {
            let center = cat_scale.center(ci) as f32;

            if bc.stacked {
                // Show cumulative total above the full stack
                let total: f64 = bc
                    .series
                    .iter()
                    .filter_map(|s| {
                        if ci < s.len() {
                            let v = s.values()[ci];
                            if v.is_finite() {
                                Some(v)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .sum();
                if total.is_finite() {
                    let top_y = y_scale.to_pixel(total) as f32;
                    ctx.add_text(
                        center,
                        top_y - 4.0,
                        &format_value(total),
                        text_color,
                        TextAlign::Center,
                        data_fs,
                        false,
                        0.0,
                    );
                }
            } else {
                let bar_width = if n_series > 0 {
                    inner_band / n_series as f32
                } else {
                    inner_band
                };
                let group_left = center - inner_band / 2.0;

                for (si, series) in bc.series.iter().enumerate() {
                    if ci >= series.len() {
                        continue;
                    }
                    let value = series.values()[ci];
                    if !value.is_finite() {
                        continue;
                    }
                    let top_y = y_scale.to_pixel(value) as f32;
                    let label_x = group_left + si as f32 * bar_width + bar_width / 2.0;
                    // Offset above error bar cap if error bars are present
                    let err_offset = match series.error_values() {
                        Some(e) if ci < e.len() && e[ci].is_finite() && e[ci] > 0.0 => {
                            let cap_y = y_scale.to_pixel(value + e[ci]) as f32;
                            (top_y - cap_y).abs() + 4.0
                        }
                        _ => 0.0,
                    };
                    let label_y = if value >= 0.0 {
                        top_y - 4.0 - err_offset
                    } else {
                        top_y + 12.0 + err_offset
                    };
                    ctx.add_text(
                        label_x,
                        label_y,
                        &format_value(value),
                        text_color,
                        TextAlign::Center,
                        data_fs,
                        false,
                        0.0,
                    );
                }
            }
        }
        // Cull overlapping value labels, keeping min/max/endpoints.
        cull_overlapping_value_labels(&mut ctx.overlays, val_start, data_fs, false);
    }

    // Category label overlays
    ctx.draw_categorical_x_labels(config, &cat_scale, &bc.labels);

    // Legend for multi-series
    if bc.series.len() > 1 && config.show_legend {
        // Collect bar rectangle corners for overlap detection.
        // Using corners instead of centroids ensures the legend never
        // overlaps any bar, even at the edges.
        let mut all_points: Vec<(f32, f32)> = Vec::new();
        let baseline_px = y_scale.to_pixel(0.0) as f32;
        for (ci, _label) in bc.labels.iter().enumerate() {
            let center = cat_scale.center(ci) as f32;
            if bc.stacked {
                let total: f64 = bc
                    .series
                    .iter()
                    .filter_map(|s| {
                        if ci < s.len() {
                            let v = s.values()[ci];
                            if v.is_finite() {
                                Some(v)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .sum();
                if total.is_finite() {
                    let bar_w = inner_band;
                    let bar_x = center - bar_w / 2.0;
                    let top_y = y_scale.to_pixel(total) as f32;
                    all_points.push((bar_x, top_y));
                    all_points.push((bar_x + bar_w, top_y));
                    all_points.push((bar_x, baseline_px));
                    all_points.push((bar_x + bar_w, baseline_px));
                }
            } else {
                let bw = if n_series > 0 {
                    inner_band / n_series as f32
                } else {
                    inner_band
                };
                let gl = center - inner_band / 2.0;
                for (si, series) in bc.series.iter().enumerate() {
                    if ci < series.len() {
                        let v = series.values()[ci];
                        if v.is_finite() {
                            let bar_x = gl + si as f32 * bw;
                            let top_y = y_scale.to_pixel(v) as f32;
                            all_points.push((bar_x, top_y));
                            all_points.push((bar_x + bw, top_y));
                            all_points.push((bar_x, baseline_px));
                            all_points.push((bar_x + bw, baseline_px));
                        }
                    }
                }
            }
        }

        let entries: Vec<LegendEntry> = bc
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
        let legend_fs = super::scaled_font_size(theme.legend.font_size, w, h);
        let mut legend_cfg = config.legend.clone();
        legend_cfg.apply_theme_and_font_size(&theme.legend, legend_fs);
        let data_pts = if all_points.is_empty() {
            None
        } else {
            Some(all_points.as_slice())
        };
        let legend_text = ctx.draw_with(|c| {
            legend::draw_positioned_legend(c, &entries, plot, &legend_cfg, 10.0, 4.0, data_pts)
        });

        for (lx, ly, label) in legend_text {
            ctx.add_text(
                lx,
                ly,
                &label,
                theme.text_color(),
                TextAlign::Left,
                legend_fs,
                false,
                0.0,
            );
        }
    }

    ctx.add_common_overlays(config);
    ctx.finish()
}

fn render_bar_horizontal(bc: &BarChart, w: u32, h: u32) -> RenderedChart {
    let config = &bc.config;
    let theme = &config.theme;
    let data_fs = super::scaled_font_size(9.0, w, h);
    let tick_fs = super::scaled_font_size(theme.tick_style.font_size, w, h);
    let mut ctx = RenderContext::new(config, w, h, None);
    let (px, py, pw, ph) = ctx.plot;

    let (x_lo, x_hi) = compute_value_extent(bc);
    let x_extent = (x_lo, x_hi);
    let x_scale = LinearScale::nice_zero(x_extent, (px as f64, (px + pw) as f64));

    // Categorical axis is on the left (Y), value axis on bottom (X)
    let cat_scale = CategoricalScale::new(bc.labels.clone(), (py as f64, (py + ph) as f64));

    // X value-axis (bottom) — shared infrastructure with ticks, gridlines, labels
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

    // Draw bars (growing rightward from left axis)
    let n_series = bc.series.len();
    let band = cat_scale.band_width() as f32;
    let inner_band = band * (1.0 - bc.bar_gap);
    let baseline_x = x_scale.to_pixel(0.0) as f32;
    let corner_r = bc.corner_radius.unwrap_or(theme.series.bar_corner_radius);
    let stroke_w = theme.bar_stroke_width();

    for (ci, _label) in bc.labels.iter().enumerate() {
        let center = cat_scale.center(ci) as f32;

        if bc.stacked {
            let bar_height = inner_band;
            let bar_y = center - bar_height / 2.0;
            let mut cumulative = 0.0;

            for (si, series) in bc.series.iter().enumerate() {
                if ci >= series.len() {
                    continue;
                }
                let value = series.values()[ci];
                if !value.is_finite() {
                    continue;
                }
                let left_x = x_scale.to_pixel(cumulative) as f32;
                let right_x = x_scale.to_pixel(cumulative + value) as f32;
                let (rect_x, bar_w) = if right_x >= left_x {
                    (left_x, right_x - left_x)
                } else {
                    (right_x, left_x - right_x)
                };

                if bar_w > 0.5 {
                    let color = theme.resolve_series_color(si, series.series_style());
                    let cr = if si == n_series - 1 { corner_r } else { 0.0 };
                    ctx.draw(|c| {
                        c.rect(rect_x, bar_y, bar_w, bar_height)
                            .fill(color)
                            .corner_radius(cr)
                            .done()
                    });
                    if stroke_w > 0.0 {
                        let sc = color.with_alpha(0.5);
                        ctx.draw(|c| {
                            c.rect(rect_x, bar_y, bar_w, bar_height)
                                .stroke(sc, stroke_w)
                                .corner_radius(cr)
                                .done()
                        });
                    }
                    draw_fill_pattern(
                        &mut ctx,
                        series.series_style().fill_pattern.as_ref(),
                        color,
                        rect_x,
                        bar_y,
                        bar_w,
                        bar_height,
                    );
                }
                cumulative += value;
            }
        } else {
            let bar_height = if n_series > 0 {
                inner_band / n_series as f32
            } else {
                inner_band
            };
            let group_top = center - inner_band / 2.0;

            for (si, series) in bc.series.iter().enumerate() {
                if ci >= series.len() {
                    continue;
                }
                let value = series.values()[ci];
                if !value.is_finite() {
                    continue;
                }
                let right_x = x_scale.to_pixel(value) as f32;
                let bar_y = group_top + si as f32 * bar_height;
                let (rect_x, bar_w) = if value >= 0.0 {
                    (baseline_x, right_x - baseline_x)
                } else {
                    (right_x, baseline_x - right_x)
                };

                if bar_w > 0.5 {
                    let color = theme.resolve_series_color(si, series.series_style());
                    ctx.draw(|c| {
                        c.rect(rect_x, bar_y, bar_w, bar_height)
                            .fill(color)
                            .corner_radius(corner_r)
                            .done()
                    });
                    if stroke_w > 0.0 {
                        let sc = color.with_alpha(0.5);
                        ctx.draw(|c| {
                            c.rect(rect_x, bar_y, bar_w, bar_height)
                                .stroke(sc, stroke_w)
                                .corner_radius(corner_r)
                                .done()
                        });
                    }
                    draw_fill_pattern(
                        &mut ctx,
                        series.series_style().fill_pattern.as_ref(),
                        color,
                        rect_x,
                        bar_y,
                        bar_w,
                        bar_height,
                    );
                }
            }
        }
    }

    // Value labels beside bars
    if bc.show_values {
        let val_start = ctx.overlays.len();
        let text_color = theme.text_color();
        for (ci, _label) in bc.labels.iter().enumerate() {
            let center = cat_scale.center(ci) as f32;

            if bc.stacked {
                // Show cumulative total to the right of the full stack
                let total: f64 = bc
                    .series
                    .iter()
                    .filter_map(|s| {
                        if ci < s.len() {
                            let v = s.values()[ci];
                            if v.is_finite() {
                                Some(v)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .sum();
                if total.is_finite() {
                    let right_x = x_scale.to_pixel(total) as f32;
                    ctx.add_text(
                        right_x + 4.0,
                        center,
                        &format_value(total),
                        text_color,
                        TextAlign::Left,
                        data_fs,
                        false,
                        0.0,
                    );
                }
            } else {
                let bar_height = if n_series > 0 {
                    inner_band / n_series as f32
                } else {
                    inner_band
                };
                let group_top = center - inner_band / 2.0;

                for (si, series) in bc.series.iter().enumerate() {
                    if ci >= series.len() {
                        continue;
                    }
                    let value = series.values()[ci];
                    if !value.is_finite() {
                        continue;
                    }
                    let right_x = x_scale.to_pixel(value) as f32;
                    let label_y = group_top + si as f32 * bar_height + bar_height / 2.0;
                    let (label_x, align) = if value >= 0.0 {
                        (right_x + 4.0, TextAlign::Left)
                    } else {
                        (right_x - 4.0, TextAlign::Right)
                    };
                    ctx.add_text(
                        label_x,
                        label_y,
                        &format_value(value),
                        text_color,
                        align,
                        data_fs,
                        false,
                        0.0,
                    );
                }
            }
        }
        // Cull overlapping value labels, keeping min/max/endpoints.
        cull_overlapping_value_labels(&mut ctx.overlays, val_start, data_fs, true);
    }

    // Category label overlays (on the left side)
    for (ci, label) in bc.labels.iter().enumerate() {
        ctx.add_text(
            px - super::y_tick_label_offset(w),
            cat_scale.center(ci) as f32,
            label,
            theme.text_color(),
            TextAlign::Right,
            tick_fs,
            false,
            0.0,
        );
    }

    // Legend for multi-series
    if bc.series.len() > 1 && config.show_legend {
        // Collect bar rectangle corners for overlap detection.
        let mut all_points: Vec<(f32, f32)> = Vec::new();
        for (ci, _label) in bc.labels.iter().enumerate() {
            let center = cat_scale.center(ci) as f32;
            if bc.stacked {
                let total: f64 = bc
                    .series
                    .iter()
                    .filter_map(|s| {
                        if ci < s.len() {
                            let v = s.values()[ci];
                            if v.is_finite() {
                                Some(v)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .sum();
                if total.is_finite() {
                    let bar_h = inner_band;
                    let bar_y = center - bar_h / 2.0;
                    let right_x = x_scale.to_pixel(total) as f32;
                    all_points.push((baseline_x, bar_y));
                    all_points.push((right_x, bar_y));
                    all_points.push((baseline_x, bar_y + bar_h));
                    all_points.push((right_x, bar_y + bar_h));
                }
            } else {
                let bh = if n_series > 0 {
                    inner_band / n_series as f32
                } else {
                    inner_band
                };
                let gt = center - inner_band / 2.0;
                for (si, series) in bc.series.iter().enumerate() {
                    if ci < series.len() {
                        let v = series.values()[ci];
                        if v.is_finite() {
                            let bar_y = gt + si as f32 * bh;
                            let right_x = x_scale.to_pixel(v) as f32;
                            all_points.push((baseline_x, bar_y));
                            all_points.push((right_x, bar_y));
                            all_points.push((baseline_x, bar_y + bh));
                            all_points.push((right_x, bar_y + bh));
                        }
                    }
                }
            }
        }

        let entries: Vec<LegendEntry> = bc
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
        let legend_fs = super::scaled_font_size(theme.legend.font_size, w, h);
        let mut legend_cfg = config.legend.clone();
        legend_cfg.apply_theme_and_font_size(&theme.legend, legend_fs);
        let data_pts = if all_points.is_empty() {
            None
        } else {
            Some(all_points.as_slice())
        };
        let legend_text = ctx.draw_with(|c| {
            legend::draw_positioned_legend(c, &entries, plot, &legend_cfg, 10.0, 4.0, data_pts)
        });

        for (lx, ly, label) in legend_text {
            ctx.add_text(
                lx,
                ly,
                &label,
                theme.text_color(),
                TextAlign::Left,
                legend_fs,
                false,
                0.0,
            );
        }
    }

    ctx.add_common_overlays(config);
    ctx.finish()
}

/// Compute the (min, max) extent across all series/categories.
/// Always includes 0.0 so the baseline is visible.
fn compute_value_extent(bc: &BarChart) -> (f64, f64) {
    if bc.stacked {
        let sums: Vec<f64> = (0..bc.labels.len())
            .map(|ci| {
                bc.series
                    .iter()
                    .map(|s| {
                        if ci < s.len() {
                            let v = s.values()[ci];
                            if v.is_finite() {
                                v
                            } else {
                                0.0
                            }
                        } else {
                            0.0
                        }
                    })
                    .sum::<f64>()
            })
            .collect();
        let lo = sums
            .iter()
            .copied()
            .reduce(f64::min)
            .unwrap_or(0.0)
            .min(0.0);
        let hi = sums
            .iter()
            .copied()
            .reduce(f64::max)
            .unwrap_or(1.0)
            .max(0.0);
        (lo, hi)
    } else {
        let lo = bc
            .series
            .iter()
            .filter_map(|s| s.min())
            .reduce(f64::min)
            .unwrap_or(0.0)
            .min(0.0);
        let hi = bc
            .series
            .iter()
            .filter_map(|s| s.max())
            .reduce(f64::max)
            .unwrap_or(1.0)
            .max(0.0);
        (lo, hi)
    }
}

/// Format a numeric value adaptively for data labels (bar, histogram, etc.).
///
/// Uses human-readable SI suffixes for large values instead of scientific
/// notation, keeping labels compact and glance-able:
/// - ≥ 1 billion  → `1.2B`
/// - ≥ 1 million  → `3.4M`
/// - ≥ 1 thousand → `1.5k`
/// - Very small   → scientific notation (only for |v| < 0.01)
/// - Integer      → no decimal point
/// - Otherwise    → one decimal place
pub(crate) fn format_value(value: f64) -> String {
    let abs = value.abs();
    if abs >= 1_000_000_000.0 {
        format!("{:.1}B", value / 1_000_000_000.0)
    } else if abs >= 1_000_000.0 {
        format!("{:.1}M", value / 1_000_000.0)
    } else if abs >= 10_000.0 {
        format!("{:.1}k", value / 1_000.0)
    } else if abs >= 1_000.0 {
        // 1000–9999: show as integer (compact enough without suffix)
        format!("{}", value as i64)
    } else if abs < 0.01 && value != 0.0 {
        format!("{value:.2e}")
    } else if value.fract().abs() < 1e-9 {
        format!("{}", value as i64)
    } else {
        format!("{value:.1}")
    }
}

/// Remove overlapping value labels, prioritising min, max, and endpoints.
///
/// Walks the overlays added after `start_idx` and marks labels for removal
/// if they would visually collide with a higher-priority neighbour.
/// Priority: endpoints (first/last) > extremes (min/max value) > interior.
pub(crate) fn cull_overlapping_value_labels(
    overlays: &mut Vec<TextOverlay>,
    start_idx: usize,
    font_size: f32,
    horizontal: bool,
) {
    let n = overlays.len();
    if start_idx >= n || n - start_idx < 2 {
        return;
    }

    let slice = &overlays[start_idx..];
    let count = slice.len();

    // Minimum pixel distance between centres to avoid overlap.
    let min_dist = if horizontal {
        font_size * 3.0
    } else {
        font_size * 1.4
    };

    // Build priority flags: endpoints + extremes are protected.
    let mut is_protected = vec![false; count];
    is_protected[0] = true;
    is_protected[count - 1] = true;

    // Find min/max value labels by parsing their text (best-effort).
    if count > 2 {
        let (mut min_i, mut max_i) = (0, 0);
        let (mut min_y, mut max_y) = (f32::MAX, f32::MIN);
        for (i, ov) in slice.iter().enumerate() {
            let coord = if horizontal { ov.x_px } else { -ov.y_px };
            if coord < min_y {
                min_y = coord;
                min_i = i;
            }
            if coord > max_y {
                max_y = coord;
                max_i = i;
            }
        }
        is_protected[min_i] = true;
        is_protected[max_i] = true;
    }

    // Sort indices by position along the primary axis, then greedily cull.
    let mut indices: Vec<usize> = (0..count).collect();
    if horizontal {
        indices.sort_by(|a, b| slice[*a].x_px.partial_cmp(&slice[*b].x_px).unwrap());
    } else {
        indices.sort_by(|a, b| slice[*a].y_px.partial_cmp(&slice[*b].y_px).unwrap());
    }

    let mut remove = vec![false; count];
    let mut last_kept_pos: Option<f32> = None;

    for &idx in &indices {
        if remove[idx] {
            continue;
        }
        let pos = if horizontal {
            slice[idx].x_px
        } else {
            slice[idx].y_px
        };
        if let Some(prev) = last_kept_pos {
            if (pos - prev).abs() < min_dist && !is_protected[idx] {
                remove[idx] = true;
                continue;
            }
            // Protected overlaps a previous one — keep this, remove the
            // previous if it isn't also protected (backtrack).
        }
        last_kept_pos = Some(pos);
    }

    // Actually remove, from back to front to preserve indices.
    let mut to_remove: Vec<usize> = remove
        .iter()
        .enumerate()
        .filter(|(_, r)| **r)
        .map(|(i, _)| start_idx + i)
        .collect();
    to_remove.sort_unstable_by(|a, b| b.cmp(a));
    for idx in to_remove {
        overlays.remove(idx);
    }
}

/// Draw a fill pattern overlay on top of a bar rectangle.
///
/// This renders geometric hatch marks (lines or dots) inside the bar bounds,
/// making multi-series charts accessible without relying on color alone.
fn draw_fill_pattern(
    ctx: &mut RenderContext,
    pattern: Option<&FillPattern>,
    color: scry_engine::style::Color,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
) {
    let pattern = match pattern {
        Some(p) if *p != FillPattern::Solid => p,
        _ => return,
    };

    // Use darkened, semi-transparent version of the bar color for hatch lines
    let hatch_color = color.with_lightness(0.6).with_alpha(0.7);
    let line_w = 1.0_f32;

    match pattern {
        FillPattern::Hatched => {
            // Horizontal lines at ~4px spacing
            let spacing = 4.0_f32;
            let mut ly = y + spacing;
            while ly < y + h {
                ctx.draw(|c| {
                    c.line(x, ly, x + w, ly)
                        .color(hatch_color)
                        .width(line_w)
                        .done()
                });
                ly += spacing;
            }
        }
        FillPattern::CrossHatched => {
            // Horizontal + vertical lines
            let spacing = 5.0_f32;
            let mut ly = y + spacing;
            while ly < y + h {
                ctx.draw(|c| {
                    c.line(x, ly, x + w, ly)
                        .color(hatch_color)
                        .width(line_w)
                        .done()
                });
                ly += spacing;
            }
            let mut lx = x + spacing;
            while lx < x + w {
                ctx.draw(|c| {
                    c.line(lx, y, lx, y + h)
                        .color(hatch_color)
                        .width(line_w)
                        .done()
                });
                lx += spacing;
            }
        }
        FillPattern::Dotted => {
            // Grid of small circles
            let spacing = 5.0_f32;
            let dot_r = 1.2_f32;
            let mut dy = y + spacing;
            while dy < y + h {
                let mut dx = x + spacing;
                while dx < x + w {
                    ctx.draw(|c| c.circle(dx, dy, dot_r).fill(hatch_color).done());
                    dx += spacing;
                }
                dy += spacing;
            }
        }
        FillPattern::Diagonal => {
            // 45° diagonal lines at ~6px spacing
            let spacing = 6.0_f32;
            let total = w + h;
            let mut offset = spacing;
            while offset < total {
                // Line from bottom-left to top-right within bar bounds
                let x1 = x + (offset - h).max(0.0);
                let y1 = (y + h - (offset - (x1 - x)).max(0.0)).max(y);
                let x2 = (x + offset).min(x + w);
                let y2 = (y + h - (offset - (x2 - x))).max(y);
                if x1 < x + w && y2 <= y + h {
                    ctx.draw(|c| {
                        c.line(x1, y1, x2, y2)
                            .color(hatch_color)
                            .width(line_w)
                            .done()
                    });
                }
                offset += spacing;
            }
        }
        FillPattern::Solid => {} // handled above
    }
}
