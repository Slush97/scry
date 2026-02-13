//! Scatter plot rendering.

use ratatui_pixelcanvas::scene::PixelCanvas;
use ratatui_pixelcanvas::style::Color;

use crate::chart::scatter::{Marker, ScatterChart};
use crate::scale::{LinearScale, Scale};

use super::{resolve_x_extent, resolve_y_extent, take_canvas, RenderContext, RenderedChart};

pub(crate) fn render_scatter(sc: &ScatterChart, w: u32, h: u32) -> RenderedChart {
    let config = &sc.config;
    let theme = &config.theme;
    let mut ctx = RenderContext::new(config, w, h);
    let (px, py, pw, ph) = ctx.plot;

    let x_extent = resolve_x_extent(config, sc.x.extent().unwrap_or((0.0, 1.0)));
    let y_extent = resolve_y_extent(config, sc.y.extent().unwrap_or((0.0, 1.0)));

    let x_scale = LinearScale::nice(x_extent, (px as f64, (px + pw) as f64));
    let y_scale = LinearScale::nice(y_extent, ((py + ph) as f64, py as f64));

    ctx.draw_axes(config, &x_scale, &y_scale);
    ctx.draw_reference_lines(config, &x_scale, &y_scale);

    // Draw data points for main series
    let color0 = theme.series_color(0);
    for i in 0..sc.x.len().min(sc.y.len()) {
        let sx = x_scale.to_pixel(sc.x.values()[i]) as f32;
        let sy = y_scale.to_pixel(sc.y.values()[i]) as f32;
        ctx.canvas = draw_marker(take_canvas(&mut ctx), sx, sy, theme.point_radius, color0, sc.marker);
    }

    // Connect with lines if requested
    if sc.connect {
        let n = sc.x.len().min(sc.y.len());
        for i in 1..n {
            let x1 = x_scale.to_pixel(sc.x.values()[i - 1]) as f32;
            let y1 = y_scale.to_pixel(sc.y.values()[i - 1]) as f32;
            let x2 = x_scale.to_pixel(sc.x.values()[i]) as f32;
            let y2 = y_scale.to_pixel(sc.y.values()[i]) as f32;
            ctx.canvas = take_canvas(&mut ctx)
                .line(x1, y1, x2, y2)
                .color(color0)
                .width(theme.line_width * 0.7)
                .done();
        }
    }

    // Extra series
    for (si, (xs, ys)) in sc.extra_series.iter().enumerate() {
        let color = theme.series_color(si + 1);
        for i in 0..xs.len().min(ys.len()) {
            let sx = x_scale.to_pixel(xs.values()[i]) as f32;
            let sy = y_scale.to_pixel(ys.values()[i]) as f32;
            ctx.canvas = draw_marker(take_canvas(&mut ctx), sx, sy, theme.point_radius, color, sc.marker);
        }
    }

    // Trend line (linear regression)
    if config.show_trend {
        ctx.draw_trend_line(sc.x.values(), sc.y.values(), &x_scale, &y_scale, theme.series_color(0));
    }

    // Annotations
    if !config.annotations.is_empty() {
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
                .polygon(vec![
                    (x, y - r),
                    (x + r, y),
                    (x, y + r),
                    (x - r, y),
                ])
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
            c.rect(x - w / 2.0, y - r, w, r * 2.0)
                .fill(color)
                .done()
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
