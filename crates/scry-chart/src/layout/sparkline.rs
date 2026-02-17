// SPDX-License-Identifier: MIT OR Apache-2.0
//! Sparkline rendering — chrome-free inline charts.
//!
//! No axes, no title, no margins. The data fills the entire canvas.

use scry_engine::scene::PixelCanvas;

use crate::chart::sparkline::{Sparkline, SparklineKind};

use super::RenderedChart;

pub(crate) fn render_sparkline(sp: &Sparkline, w: u32, h: u32) -> RenderedChart {
    let theme = &sp.config.theme;
    let color = sp.color.unwrap_or_else(|| theme.resolve_series_color(0, &crate::data::SeriesStyle::default()));

    let mut canvas = PixelCanvas::new(w, h).background(theme.background);

    let finite_vals: Vec<f64> = sp.values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite_vals.is_empty() {
        return RenderedChart {
            canvas,
            text_overlays: Vec::new(),
            plot_area: Some((0.0, 0.0, w as f32, h as f32)),
            x_scale: None,
            y_scale: None,
            series_points: Vec::new(),
        };
    }

    let lo = finite_vals.iter().copied().reduce(f64::min).unwrap();
    let hi = finite_vals.iter().copied().reduce(f64::max).unwrap();
    let range = if (hi - lo).abs() < f64::EPSILON { 1.0 } else { hi - lo };
    let n = sp.values.len();

    let pad = 1.0_f32; // 1px padding
    let wf = w as f32 - 2.0 * pad;
    let hf = h as f32 - 2.0 * pad;

    match sp.kind {
        SparklineKind::Line => {
            // Build points
            let mut points: Vec<(f32, f32)> = Vec::with_capacity(n);
            for (i, &v) in sp.values.iter().enumerate() {
                if !v.is_finite() { continue; }
                let px = pad + (i as f32 / (n - 1).max(1) as f32) * wf;
                let py = pad + (1.0 - ((v - lo) / range) as f32) * hf;
                points.push((px, py));
            }

            // Fill area
            if sp.fill && points.len() >= 2 {
                let fill_color = color.with_alpha(0.2);
                let mut fill_pts = points.clone();
                fill_pts.push((points.last().unwrap().0, pad + hf));
                fill_pts.push((points[0].0, pad + hf));
                canvas = canvas.polygon(fill_pts).fill(fill_color).done();
            }

            // Polyline
            if points.len() >= 2 {
                canvas = canvas.polyline(points).stroke(color, sp.line_width).done();
            }
        }
        SparklineKind::Bar => {
            let bar_w = (wf / n as f32).max(1.0);
            let gap = (bar_w * 0.15).max(0.5);
            let bar_inner = bar_w - gap;
            for (i, &v) in sp.values.iter().enumerate() {
                if !v.is_finite() { continue; }
                let t = ((v - lo) / range) as f32;
                let bar_h = (t * hf).max(1.0);
                let bx = pad + i as f32 * bar_w;
                let by = pad + hf - bar_h;
                canvas = canvas.rect(bx, by, bar_inner, bar_h).fill(color).done();
            }
        }
        SparklineKind::WinLoss => {
            let bar_w = (wf / n as f32).max(1.0);
            let gap = (bar_w * 0.15).max(0.5);
            let bar_inner = bar_w - gap;
            let center_y = pad + hf / 2.0;
            let half_h = hf / 2.0 * 0.85;
            let win_color = color;
            // Use a contrasting color for losses — second palette color or desaturated red
            let loss_color = if theme.palette.len() >= 2 {
                theme.palette[1].with_alpha(0.85)
            } else {
                scry_engine::style::Color::from_rgba8(200, 80, 80, 220)
            };
            for (i, &v) in sp.values.iter().enumerate() {
                if !v.is_finite() { continue; }
                let bx = pad + i as f32 * bar_w;
                if v >= 0.0 {
                    canvas = canvas.rect(bx, center_y - half_h, bar_inner, half_h)
                        .fill(win_color).done();
                } else {
                    canvas = canvas.rect(bx, center_y, bar_inner, half_h)
                        .fill(loss_color).done();
                }
            }
        }
    }

    RenderedChart {
        canvas,
        text_overlays: Vec::new(),
        plot_area: Some((0.0, 0.0, w as f32, h as f32)),
        x_scale: None,
        y_scale: None,
        series_points: Vec::new(),
    }
}
