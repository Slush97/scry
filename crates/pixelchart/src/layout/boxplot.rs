//! Box plot rendering.

use crate::chart::boxplot::{BoxPlot, BoxStats};
use crate::scale::{CategoricalScale, LinearScale, Scale};

use super::{resolve_y_extent, take_canvas, RenderContext, RenderedChart, TextAlign, TextOverlay};

pub(crate) fn render_boxplot(bp: &BoxPlot, w: u32, h: u32) -> RenderedChart {
    let config = &bp.config;
    let theme = &config.theme;
    let mut ctx = RenderContext::new(config, w, h);
    let (px, py, pw, ph) = ctx.plot;

    // Compute stats for each group
    let stats: Vec<Option<BoxStats>> = bp
        .groups
        .iter()
        .map(|g| BoxStats::from_data(g.data.values()))
        .collect();

    // Y extent from all groups
    let data_y_lo = stats
        .iter()
        .filter_map(|s| s.as_ref().map(|s| s.min))
        .reduce(f64::min)
        .unwrap_or(0.0);
    let data_y_hi = stats
        .iter()
        .filter_map(|s| s.as_ref().map(|s| s.max))
        .reduce(f64::max)
        .unwrap_or(1.0);
    let y_extent = resolve_y_extent(config, (data_y_lo, data_y_hi));
    let y_scale = LinearScale::nice(y_extent, ((py + ph) as f64, py as f64));

    let labels: Vec<String> = bp.groups.iter().map(|g| g.label.clone()).collect();
    let cat_scale = CategoricalScale::new(labels.clone(), (px as f64, (px + pw) as f64));

    // Y axis
    let y_ticks = ctx.draw_y_axis(config, &y_scale);
    ctx.add_y_tick_overlays(&y_ticks, theme.text_color);

    // X axis line
    ctx.draw_x_axis_line(config);

    // Reference lines
    let x_dummy = LinearScale::new((0.0, 1.0), (px as f64, (px + pw) as f64));
    ctx.draw_reference_lines(config, &x_dummy, &y_scale);

    // Draw each box
    let band = cat_scale.band_width() as f32;
    let box_w = band * bp.box_width;

    for (gi, stat_opt) in stats.iter().enumerate() {
        let Some(stat) = stat_opt else {
            continue;
        };

        let center_x = cat_scale.center(gi) as f32;
        let box_left = center_x - box_w / 2.0;
        let color = theme.series_color(gi);

        // Map data values to pixel positions
        let y_q1 = y_scale.to_pixel(stat.q1) as f32;
        let y_q3 = y_scale.to_pixel(stat.q3) as f32;
        let y_med = y_scale.to_pixel(stat.median) as f32;
        let y_wlo = y_scale.to_pixel(stat.whisker_lo) as f32;
        let y_whi = y_scale.to_pixel(stat.whisker_hi) as f32;

        // IQR box (filled with translucent color)
        let box_top = y_q3.min(y_q1);
        let box_h = (y_q1 - y_q3).abs();
        ctx.canvas = take_canvas(&mut ctx)
            .rect(box_left, box_top, box_w, box_h)
            .fill(color.with_alpha(0.3))
            .corner_radius(2.0)
            .done();

        // IQR box outline
        ctx.canvas = take_canvas(&mut ctx)
            .rect(box_left, box_top, box_w, box_h)
            .stroke(color, 1.5)
            .corner_radius(2.0)
            .done();

        // Median line (thicker)
        ctx.canvas = take_canvas(&mut ctx)
            .line(box_left, y_med, box_left + box_w, y_med)
            .color(color)
            .width(2.5)
            .done();

        // Whisker lines (vertical)
        ctx.canvas = take_canvas(&mut ctx)
            .line(center_x, y_q3, center_x, y_whi)
            .color(color)
            .width(1.0)
            .done();
        ctx.canvas = take_canvas(&mut ctx)
            .line(center_x, y_q1, center_x, y_wlo)
            .color(color)
            .width(1.0)
            .done();

        // Whisker caps (horizontal)
        let cap_w = box_w * 0.3;
        ctx.canvas = take_canvas(&mut ctx)
            .line(center_x - cap_w, y_whi, center_x + cap_w, y_whi)
            .color(color)
            .width(1.5)
            .done();
        ctx.canvas = take_canvas(&mut ctx)
            .line(center_x - cap_w, y_wlo, center_x + cap_w, y_wlo)
            .color(color)
            .width(1.5)
            .done();

        // Outlier points
        if bp.show_outliers {
            for &val in &stat.outliers {
                let oy = y_scale.to_pixel(val) as f32;
                ctx.canvas = take_canvas(&mut ctx)
                    .circle(center_x, oy, theme.point_radius * 0.6)
                    .fill(color.with_alpha(0.7))
                    .done();
            }
        }
    }

    // Category label overlays
    for (ci, label) in labels.iter().enumerate() {
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
