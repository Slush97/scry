//! Line chart rendering.

use crate::chart::LineChart;
use crate::legend::{self, LegendEntry};
use crate::scale::{LinearScale, Scale};

use super::{resolve_x_extent, resolve_y_extent, take_canvas, RenderContext, RenderedChart, TextAlign, TextOverlay};

pub(crate) fn render_line(lc: &LineChart, w: u32, h: u32) -> RenderedChart {
    let config = &lc.config;
    let theme = &config.theme;
    let mut ctx = RenderContext::new(config, w, h);
    let (px, py, pw, ph) = ctx.plot;

    let max_len = lc.series.iter().map(|s| s.len()).max().unwrap_or(0);
    let x_data: Vec<f64> = lc.x_values.clone().unwrap_or_else(|| {
        (0..max_len).map(|i| i as f64).collect()
    });
    let data_x_extent = if x_data.is_empty() {
        (0.0, 1.0)
    } else {
        let lo = x_data.iter().copied().reduce(f64::min).unwrap_or(0.0);
        let hi = x_data.iter().copied().reduce(f64::max).unwrap_or(1.0);
        (lo, hi)
    };

    let y_lo = lc.series.iter().filter_map(|s| s.min()).reduce(f64::min).unwrap_or(0.0);
    let y_hi = lc.series.iter().filter_map(|s| s.max()).reduce(f64::max).unwrap_or(1.0);

    let x_extent = resolve_x_extent(config, data_x_extent);
    let y_extent = resolve_y_extent(config, (y_lo, y_hi));

    let x_scale = LinearScale::nice(x_extent, (px as f64, (px + pw) as f64));
    let y_scale = LinearScale::nice(y_extent, ((py + ph) as f64, py as f64));

    ctx.draw_axes(config, &x_scale, &y_scale);
    ctx.draw_reference_lines(config, &x_scale, &y_scale);

    // Draw each series
    for (si, series) in lc.series.iter().enumerate() {
        let color = theme.series_color(si);
        let n = series.len().min(x_data.len());

        // Fill area under curve
        if lc.fill_area && n >= 2 {
            let baseline_y = y_scale.to_pixel(y_scale.domain().0) as f32;
            let mut path_points: Vec<(f32, f32)> = Vec::with_capacity(n + 2);
            path_points.push((x_scale.to_pixel(x_data[0]) as f32, baseline_y));
            for i in 0..n {
                let sx = x_scale.to_pixel(x_data[i]) as f32;
                let sy = y_scale.to_pixel(series.values()[i]) as f32;
                path_points.push((sx, sy));
            }
            path_points.push((x_scale.to_pixel(x_data[n - 1]) as f32, baseline_y));

            ctx.canvas = take_canvas(&mut ctx)
                .polygon(path_points)
                .fill(color.with_alpha(0.15))
                .done();
        }

        // Lines
        for i in 1..n {
            let x1 = x_scale.to_pixel(x_data[i - 1]) as f32;
            let y1 = y_scale.to_pixel(series.values()[i - 1]) as f32;
            let x2 = x_scale.to_pixel(x_data[i]) as f32;
            let y2 = y_scale.to_pixel(series.values()[i]) as f32;
            ctx.canvas = take_canvas(&mut ctx)
                .line(x1, y1, x2, y2)
                .color(color)
                .width(theme.line_width)
                .done();
        }

        // Data point markers
        if lc.show_points {
            for i in 0..n {
                let sx = x_scale.to_pixel(x_data[i]) as f32;
                let sy = y_scale.to_pixel(series.values()[i]) as f32;
                ctx.canvas = take_canvas(&mut ctx)
                    .circle(sx, sy, theme.point_radius * 0.7)
                    .fill(color)
                    .done();
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

        let (canvas, legend_text) = legend::draw_legend_swatches(
            take_canvas(&mut ctx),
            &entries,
            px + pw - 80.0,
            py + 8.0,
            10.0,
            4.0,
        );
        ctx.canvas = canvas;

        // Add legend text overlays
        for (lx, ly, label) in legend_text {
            ctx.overlays.push(TextOverlay {
                x_px: lx,
                y_px: ly,
                text: label,
                color: theme.text_color,
                align: TextAlign::Left,
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
