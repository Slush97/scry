// SPDX-License-Identifier: MIT OR Apache-2.0
//! Pie / donut chart rendering.

use crate::chart::pie::PieChart;
use crate::legend::{self, LegendEntry};
use crate::theme::contrast_text_color;

use super::{RenderContext, RenderedChart, TextAlign};

pub(crate) fn render_pie(pc: &PieChart, w: u32, h: u32) -> RenderedChart {
    // Degenerate canvas — nothing meaningful can be drawn.
    if w < 4 || h < 4 {
        let canvas = scry_engine::scene::PixelCanvas::new(w, h);
        return RenderedChart {
            canvas,
            text_overlays: Vec::new(),
            plot_area: None,
            x_scale: None,
            y_scale: None,
            series_points: Vec::new(),
        };
    }
    // Clone config and force OutsideRight legend *before* computing the plot
    // area so `compute_plot_area` reserves right-side space for the legend.
    let config = {
        let mut cfg = pc.config.clone();
        if cfg.show_legend {
            cfg.legend.position = crate::legend::LegendPosition::OutsideRight;
        }
        cfg
    };
    let config = &config;
    let theme = &config.theme;
    let tick_fs = super::scaled_font_size(theme.tick_style.font_size, w, h);
    let mut ctx = RenderContext::new(config, w, h, None);
    let (px, py, pw, ph) = ctx.plot;

    // Center and radius
    let cx = px + pw / 2.0;
    let cy = py + ph / 2.0;
    let max_radius = (pw.min(ph) / 2.0) * 0.85;
    let outer_r = max_radius;
    let inner_r = outer_r * pc.donut_ratio;

    // Normalize values to angles
    let total: f64 = pc
        .values
        .iter()
        .filter(|v| v.is_finite() && **v > 0.0)
        .sum();
    if total <= 0.0 {
        ctx.add_common_overlays(config);
        return ctx.finish();
    }

    let mut current_angle = pc.start_angle;
    let segments = 64; // segments per arc for smooth circles
                       // Collect external labels for post-loop collision avoidance:
                       // (inner_x, inner_y, outer_x, outer_y, label_text, alignment)
    let mut external_labels: Vec<(f32, f32, f32, f32, String, TextAlign)> = Vec::new();

    for (i, (&value, _label)) in pc.values.iter().zip(pc.labels.iter()).enumerate() {
        if !value.is_finite() || value <= 0.0 {
            continue;
        }

        let fraction = value / total;
        let sweep = (fraction * std::f64::consts::TAU) as f32;
        let color = theme.series_color(i);

        // Build polygon: outer arc → inner arc (reversed) for donut, or center for pie
        let mut points: Vec<(f32, f32)> = Vec::with_capacity(segments * 2 + 2);

        // Outer arc
        for s in 0..=segments {
            let angle = current_angle + (s as f32 / segments as f32) * sweep;
            points.push((cx + outer_r * angle.cos(), cy + outer_r * angle.sin()));
        }

        if inner_r > 1.0 {
            // Donut: inner arc in reverse
            for s in (0..=segments).rev() {
                let angle = current_angle + (s as f32 / segments as f32) * sweep;
                points.push((cx + inner_r * angle.cos(), cy + inner_r * angle.sin()));
            }
        } else {
            // Pie: close to center
            points.push((cx, cy));
        }

        ctx.draw(|c| c.polygon(points).fill(color).done());

        // Draw slice border as a single polyline for visual separation
        let border_pts: Vec<(f32, f32)> = (0..=segments)
            .map(|s| {
                let angle = current_angle + (s as f32 / segments as f32) * sweep;
                (cx + outer_r * angle.cos(), cy + outer_r * angle.sin())
            })
            .collect();
        let bg_color = theme.background.with_alpha(0.3);
        ctx.draw(|c| c.polyline(border_pts).stroke(bg_color, 1.5).done());

        // Draw radial divider lines
        let start_x = cx + outer_r * current_angle.cos();
        let start_y = cy + outer_r * current_angle.sin();
        let bg = theme.background;
        if inner_r > 1.0 {
            let inner_start_x = cx + inner_r * current_angle.cos();
            let inner_start_y = cy + inner_r * current_angle.sin();
            ctx.draw(|c| {
                c.line(inner_start_x, inner_start_y, start_x, start_y)
                    .color(bg)
                    .width(2.0)
                    .done()
            });
        } else {
            ctx.draw(|c| c.line(cx, cy, start_x, start_y).color(bg).width(2.0).done());
        }

        // Percentage label at mid-angle — collect external labels for collision resolution
        if pc.show_percentages && fraction > 0.02 {
            let mid_angle = current_angle + sweep / 2.0;

            // Compute arc width at label radius to see if text fits inside
            let label_r_inside = if inner_r > 1.0 {
                (outer_r + inner_r) / 2.0
            } else {
                outer_r * 0.65
            };
            let arc_width = sweep * label_r_inside;
            let label_text = format!("{:.0}%", fraction * 100.0);
            let label_px_w = tick_fs * label_text.len() as f32 * 0.6;

            if arc_width > label_px_w + 4.0 {
                // Fits inside the slice
                let lx = cx + label_r_inside * mid_angle.cos();
                let ly = cy + label_r_inside * mid_angle.sin();
                ctx.add_text(
                    lx,
                    ly,
                    &label_text,
                    contrast_text_color(color),
                    TextAlign::Center,
                    tick_fs,
                    true,
                    0.0,
                );
            } else {
                // Too small — collect for external placement with collision avoidance
                let inner_pt_r = outer_r + 4.0;
                let outer_pt_r = outer_r + 18.0 + tick_fs;
                let ix = cx + inner_pt_r * mid_angle.cos();
                let iy = cy + inner_pt_r * mid_angle.sin();
                let ox = cx + outer_pt_r * mid_angle.cos();
                let oy = cy + outer_pt_r * mid_angle.sin();
                let align = if mid_angle.cos() >= 0.0 {
                    TextAlign::Left
                } else {
                    TextAlign::Right
                };
                external_labels.push((ix, iy, ox, oy, label_text, align));
            }
        }

        current_angle += sweep;
    }

    // ── External label collision avoidance ──
    // Sort by Y position, then push apart any labels that would overlap
    {
        let label_h = tick_fs * 1.3; // approximate label height with padding
        external_labels.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));

        // Push overlapping labels apart vertically
        for pass in 0..3 {
            let _ = pass;
            for i in 1..external_labels.len() {
                let prev_y = external_labels[i - 1].3;
                let cur_y = external_labels[i].3;
                if (cur_y - prev_y).abs() < label_h {
                    let nudge = (label_h - (cur_y - prev_y).abs()) / 2.0 + 1.0;
                    external_labels[i - 1].3 -= nudge;
                    external_labels[i].3 += nudge;
                    // Also adjust inner point for leader line continuity
                    external_labels[i - 1].1 -= nudge * 0.3;
                    external_labels[i].1 += nudge * 0.3;
                }
            }
        }

        // Draw external labels with leader lines
        let leader_color = theme.text_color().with_alpha(0.4);
        for (ix, iy, ox, oy, label_text, align) in &external_labels {
            ctx.draw(|c| {
                c.line(*ix, *iy, *ox, *oy)
                    .color(leader_color)
                    .width(0.8)
                    .done()
            });
            ctx.add_text(
                *ox,
                *oy,
                label_text,
                theme.foreground,
                *align,
                tick_fs * 0.9,
                false,
                0.0,
            );
        }
    }

    // Legend — pie charts always use OutsideRight to avoid overlapping the circle.
    if config.show_legend {
        let entries: Vec<LegendEntry> = pc
            .labels
            .iter()
            .enumerate()
            .map(|(i, label)| LegendEntry {
                label: label.clone(),
                color: theme.series_color(i),
            })
            .collect();

        let plot = ctx.plot;
        let legend_fs = super::scaled_font_size(theme.legend.font_size, w, h);
        let mut legend_cfg = config.legend.clone();
        legend_cfg.apply_theme_and_font_size(&theme.legend, legend_fs);
        let legend_text = ctx.draw_with(|c| {
            legend::draw_positioned_legend(c, &entries, plot, &legend_cfg, 10.0, 4.0, None)
        });

        for (lx, ly, label) in legend_text {
            ctx.add_text(
                lx,
                ly,
                &label,
                theme.foreground,
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
