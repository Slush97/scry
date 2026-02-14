//! Bar chart rendering.

use crate::chart::BarChart;
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
                    let color = theme.series_color(si);
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
                    let color = theme.series_color(si);
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
                }
            }
        }
    }

    // Category label overlays
    ctx.draw_categorical_x_labels(config, &cat_scale, &bc.labels);

    ctx.add_common_overlays(config);
    ctx.finish()
}

fn render_bar_horizontal(bc: &BarChart, w: u32, h: u32) -> RenderedChart {
    let config = &bc.config;
    let theme = &config.theme;
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
                    let color = theme.series_color(si);
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
                    let color = theme.series_color(si);
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
                }
            }
        }
    }

    // Category label overlays (on the left side)
    for (ci, label) in bc.labels.iter().enumerate() {
        ctx.overlays.push(TextOverlay {
            x_px: px - super::y_tick_label_offset(w),
            y_px: cat_scale.center(ci) as f32,
            text: label.clone(),
            color: theme.text_color(),
            align: TextAlign::Right,
            font_size: 11.0,
            bold: false,
            rotation_deg: 0.0,
        });
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
