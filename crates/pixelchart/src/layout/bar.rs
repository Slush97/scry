//! Bar chart rendering.

use crate::chart::BarChart;
use crate::scale::{CategoricalScale, LinearScale, Scale};

use super::{resolve_y_extent, take_canvas, RenderContext, RenderedChart, TextAlign, TextOverlay};

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
    let mut ctx = RenderContext::new(config, w, h);
    let (px, py, pw, ph) = ctx.plot;

    let y_max = compute_value_max(bc);
    let y_extent = resolve_y_extent(config, (0.0, y_max));
    let y_scale = LinearScale::nice(y_extent, ((py + ph) as f64, py as f64));

    let cat_scale = CategoricalScale::new(bc.labels.clone(), (px as f64, (px + pw) as f64));

    // Y axis
    let y_ticks = ctx.draw_y_axis(config, &y_scale);
    ctx.add_y_tick_overlays(&y_ticks, theme.text_color);

    // X axis line
    ctx.draw_x_axis_line(config);

    // Reference lines
    let x_dummy = LinearScale::new((0.0, 1.0), (px as f64, (px + pw) as f64));
    ctx.draw_reference_lines(config, &x_dummy, &y_scale);

    // Draw bars
    let n_series = bc.series.len();
    let band = cat_scale.band_width() as f32;
    let inner_band = band * (1.0 - bc.bar_gap);
    let baseline_y = y_scale.to_pixel(0.0) as f32;

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
                let bottom_y = y_scale.to_pixel(cumulative) as f32;
                let top_y = y_scale.to_pixel(cumulative + value) as f32;
                let bar_h = bottom_y - top_y;

                if bar_h > 0.0 {
                    let color = theme.series_color(si);
                    ctx.canvas = take_canvas(&mut ctx)
                        .rect(bar_x, top_y, bar_width, bar_h)
                        .fill(color)
                        .corner_radius(if si == n_series - 1 { bc.corner_radius } else { 0.0 })
                        .done();
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
                let top_y = y_scale.to_pixel(value) as f32;
                let bar_x = group_left + si as f32 * bar_width;
                let bar_h = baseline_y - top_y;

                if bar_h > 0.0 {
                    let color = theme.series_color(si);
                    ctx.canvas = take_canvas(&mut ctx)
                        .rect(bar_x, top_y, bar_width, bar_h)
                        .fill(color)
                        .corner_radius(bc.corner_radius)
                        .done();
                }
            }
        }
    }

    // Category label overlays
    for (ci, label) in bc.labels.iter().enumerate() {
        ctx.overlays.push(TextOverlay {
            x_px: cat_scale.center(ci) as f32,
            y_px: py + ph + 8.0,
            text: label.clone(),
            color: theme.text_color,
            align: TextAlign::Center,
        });
    }

    ctx.add_common_overlays(config);
    ctx.finish()
}

fn render_bar_horizontal(bc: &BarChart, w: u32, h: u32) -> RenderedChart {
    let config = &bc.config;
    let theme = &config.theme;
    let mut ctx = RenderContext::new(config, w, h);
    let (px, py, pw, ph) = ctx.plot;

    let x_max = compute_value_max(bc);
    let x_extent = (0.0, x_max);
    let x_scale = LinearScale::nice(x_extent, (px as f64, (px + pw) as f64));

    // Categorical axis is on the left (Y), value axis on bottom (X)
    let cat_scale = CategoricalScale::new(bc.labels.clone(), (py as f64, (py + ph) as f64));

    // Draw X axis (value axis on bottom)
    ctx.draw_x_axis_line(config);

    // X axis tick labels
    let ticks = x_scale.ticks(6);
    for t in &ticks {
        let x_pos = x_scale.to_pixel(*t) as f32;
        ctx.overlays.push(TextOverlay {
            x_px: x_pos,
            y_px: py + ph + 8.0,
            text: x_scale.format_tick(*t),
            color: theme.text_color,
            align: TextAlign::Center,
        });

        // Grid line
        if theme.show_grid {
            ctx.canvas = take_canvas(&mut ctx)
                .line(x_pos, py, x_pos, py + ph)
                .color(theme.grid_color)
                .width(theme.grid_width)
                .done();
        }
    }

    // Y axis line (left side)
    ctx.canvas = take_canvas(&mut ctx)
        .line(px, py, px, py + ph)
        .color(theme.axis_color)
        .width(theme.axis_width)
        .done();

    // Draw bars (growing rightward from left axis)
    let n_series = bc.series.len();
    let band = cat_scale.band_width() as f32;
    let inner_band = band * (1.0 - bc.bar_gap);
    let baseline_x = x_scale.to_pixel(0.0) as f32;

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
                let left_x = x_scale.to_pixel(cumulative) as f32;
                let right_x = x_scale.to_pixel(cumulative + value) as f32;
                let bar_w = right_x - left_x;

                if bar_w > 0.0 {
                    let color = theme.series_color(si);
                    ctx.canvas = take_canvas(&mut ctx)
                        .rect(left_x, bar_y, bar_w, bar_height)
                        .fill(color)
                        .corner_radius(if si == n_series - 1 { bc.corner_radius } else { 0.0 })
                        .done();
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
                let right_x = x_scale.to_pixel(value) as f32;
                let bar_y = group_top + si as f32 * bar_height;
                let bar_w = right_x - baseline_x;

                if bar_w > 0.0 {
                    let color = theme.series_color(si);
                    ctx.canvas = take_canvas(&mut ctx)
                        .rect(baseline_x, bar_y, bar_w, bar_height)
                        .fill(color)
                        .corner_radius(bc.corner_radius)
                        .done();
                }
            }
        }
    }

    // Category label overlays (on the left side)
    for (ci, label) in bc.labels.iter().enumerate() {
        ctx.overlays.push(TextOverlay {
            x_px: px - 8.0,
            y_px: cat_scale.center(ci) as f32,
            text: label.clone(),
            color: theme.text_color,
            align: TextAlign::Right,
        });
    }

    ctx.add_common_overlays(config);
    ctx.finish()
}

/// Compute the maximum value across all series/categories.
fn compute_value_max(bc: &BarChart) -> f64 {
    if bc.stacked {
        (0..bc.labels.len())
            .map(|ci| {
                bc.series
                    .iter()
                    .map(|s| if ci < s.len() { s.values()[ci] } else { 0.0 })
                    .sum::<f64>()
            })
            .reduce(f64::max)
            .unwrap_or(1.0)
    } else {
        bc.series
            .iter()
            .filter_map(|s| s.max())
            .reduce(f64::max)
            .unwrap_or(1.0)
    }
}
