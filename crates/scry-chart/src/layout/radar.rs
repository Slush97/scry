// SPDX-License-Identifier: MIT OR Apache-2.0
//! Radar / spider chart rendering.

use crate::chart::radar::RadarChart;
use crate::legend::{self, LegendEntry};

use super::{RenderContext, RenderedChart, TextAlign, TextOverlay};

pub(crate) fn render_radar(rc: &RadarChart, w: u32, h: u32) -> RenderedChart {
    let config = &rc.config;
    let theme = &config.theme;
    let label_fs = super::scaled_font_size(theme.label_style.font_size, w, h);
    let tick_fs = super::scaled_font_size(theme.tick_style.font_size, w, h);

    let mut ctx = RenderContext::new(config, w, h, None);
    let (px, py, pw, ph) = ctx.plot;

    let n_axes = rc.axes.len();
    if n_axes < 3 || rc.series.is_empty() {
        ctx.add_common_overlays(config);
        return ctx.finish();
    }

    let cx = px + pw / 2.0;
    let cy = py + ph / 2.0;
    let radius = pw.min(ph) / 2.0 * 0.75;

    // Find global max value across all series for normalization
    let max_val = rc
        .series
        .iter()
        .flat_map(|(_, vals)| vals.iter().copied())
        .filter(|v| v.is_finite())
        .fold(1.0_f64, f64::max);

    let angle_step = std::f32::consts::TAU / n_axes as f32;
    let start_angle = -std::f32::consts::FRAC_PI_2; // 12 o'clock

    // Draw concentric rings (grid)
    let n_rings = 4;
    let ring_color = theme.grid_color();
    for r in 1..=n_rings {
        let ring_r = radius * r as f32 / n_rings as f32;
        let ring_pts: Vec<(f32, f32)> = (0..n_axes)
            .map(|i| {
                let angle = start_angle + i as f32 * angle_step;
                (cx + ring_r * angle.cos(), cy + ring_r * angle.sin())
            })
            .collect();
        ctx.draw(|c| c.polygon(ring_pts).stroke(ring_color, 0.5).done());
    }

    // Draw spokes (axes)
    let spoke_color = theme.axis_color();
    for i in 0..n_axes {
        let angle = start_angle + i as f32 * angle_step;
        let ex = cx + radius * angle.cos();
        let ey = cy + radius * angle.sin();
        ctx.draw(|c| c.line(cx, cy, ex, ey).color(spoke_color).width(1.0).done());

        // Axis label
        let label_r = radius + super::scaled_font_size(theme.tick_style.font_size, w, h) * 1.2;
        let lx = cx + label_r * angle.cos();
        let ly = cy + label_r * angle.sin();
        let align = if angle.cos().abs() < 0.1 {
            TextAlign::Center
        } else if angle.cos() > 0.0 {
            TextAlign::Left
        } else {
            TextAlign::Right
        };
        let label = if i < rc.axes.len() {
            rc.axes[i].clone()
        } else {
            String::new()
        };
        ctx.overlays.push(TextOverlay {
            x_px: lx,
            y_px: ly,
            text: label,
            color: theme.text_color(),
            align,
            font_size: label_fs,
            bold: false,
            rotation_deg: 0.0,
        });
    }

    // Draw each series polygon
    for (si, (_, values)) in rc.series.iter().enumerate() {
        let color = theme.series_color(si);

        let pts: Vec<(f32, f32)> = (0..n_axes)
            .map(|i| {
                let val = if i < values.len() && values[i].is_finite() {
                    values[i]
                } else {
                    0.0
                };
                let norm = (val / max_val).clamp(0.0, 1.0) as f32;
                let angle = start_angle + i as f32 * angle_step;
                let r = radius * norm;
                (cx + r * angle.cos(), cy + r * angle.sin())
            })
            .collect();

        // Fill polygon
        if rc.fill {
            let fill_color = color.with_alpha(theme.fill_opacity() * 0.7);
            ctx.draw(|c| c.polygon(pts.clone()).fill(fill_color).done());
        }

        // Outline
        let line_width = theme.line_width();
        // Draw as closed polyline by appending first point
        let mut outline = pts.clone();
        outline.push(pts[0]);
        ctx.draw(|c| c.polyline(outline).stroke(color, line_width).done());

        // Point markers
        if rc.show_points {
            let pr = theme.point_radius() * 0.6;
            for &(px, py) in &pts {
                ctx.draw(|c| c.circle(px, py, pr).fill(color).done());
            }
        }
    }

    // Legend
    if rc.series.len() > 1 && config.show_legend {
        let entries: Vec<LegendEntry> = rc
            .series
            .iter()
            .enumerate()
            .map(|(i, (label, _))| LegendEntry {
                label: if label.is_empty() {
                    format!("Series {}", i + 1)
                } else {
                    label.clone()
                },
                color: theme.series_color(i),
            })
            .collect();

        let plot = ctx.plot;
        let legend_text = ctx.draw_with(|c| {
            legend::draw_positioned_legend(c, &entries, plot, &config.legend, 10.0, 4.0, None)
        });

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

    ctx.add_common_overlays(config);
    ctx.finish()
}
