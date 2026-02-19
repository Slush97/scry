// SPDX-License-Identifier: MIT OR Apache-2.0
//! Heatmap rendering.
//!
//! Uses `RenderContext` for canvas/overlay management and `add_common_overlays()`
//! so that subtitles, footers, and axis labels work consistently.

use crate::chart::heatmap::{self, Heatmap};
use crate::theme::contrast_text_color;

use super::{
    proportional_margin, proportional_title_height, scaled_font_size, RenderContext, RenderedChart,
    TextAlign, TextOverlay, INTER_ADVANCE_RATIO,
};

pub(crate) fn render_heatmap(hm: &Heatmap, w: u32, h: u32) -> RenderedChart {
    let config = &hm.config;
    let theme = &config.theme;
    let data_fs = scaled_font_size(9.0, w, h);
    let tick_fs = scaled_font_size(theme.tick_style.font_size, w, h);

    // Use RenderContext — pass None for y_extent since heatmaps don't have
    // numeric Y axes. We'll override the plot area with heatmap-specific layout.
    let mut ctx = RenderContext::new(config, w, h, None);

    // Heatmap-specific layout: row labels on the left, col labels on top.
    let margin = proportional_margin(w, h);
    let title_h = if config.titles.title.is_some() {
        proportional_title_height(h)
    } else {
        0.0
    };
    // Subtitle adds extra space below title
    let subtitle_h = if config.titles.subtitle.is_some() {
        title_h * 0.5
    } else {
        0.0
    };

    let col_label_h = tick_fs * 1.6;
    let row_label_w = {
        let char_w = tick_fs * INTER_ADVANCE_RATIO;
        let max_chars = hm
            .row_labels
            .iter()
            .map(|l| l.chars().count())
            .max()
            .unwrap_or(0);
        ((max_chars as f32) * char_w + tick_fs)
            .min(w as f32 * 0.3)
            .max(20.0)
    };

    let grid_x = margin + row_label_w;
    let grid_y = margin + title_h + subtitle_h + col_label_h;
    // Reserve space on the right for the color legend bar
    let legend_bar_w = 16.0;
    let legend_label_w = tick_fs * 4.0; // space for min/max labels
    let legend_gap = 12.0;
    let legend_total_w = legend_gap + legend_bar_w + 4.0 + legend_label_w;
    let grid_w = (w as f32 - grid_x - margin - legend_total_w).max(1.0);
    let grid_h = (h as f32 - grid_y - margin).max(1.0);

    // Override the plot area so add_common_overlays positions elements correctly
    ctx.plot = (grid_x, grid_y, grid_w, grid_h);

    let n_rows = hm.data.len();
    let n_cols = hm.data.first().map_or(0, |r| r.len());

    if n_rows == 0 || n_cols == 0 {
        ctx.add_common_overlays(config);
        return ctx.finish();
    }

    let cell_w = (grid_w - hm.cell_gap * (n_cols as f32 - 1.0).max(0.0)) / n_cols as f32;
    let cell_h = (grid_h - hm.cell_gap * (n_rows as f32 - 1.0).max(0.0)) / n_rows as f32;

    let (v_lo, v_hi) = hm.value_range.unwrap_or_else(|| hm.data_extent());
    let v_span = if (v_hi - v_lo).abs() < f64::EPSILON {
        1.0
    } else {
        v_hi - v_lo
    };

    // Draw cells
    for (ri, row) in hm.data.iter().enumerate() {
        let cy = grid_y + ri as f32 * (cell_h + hm.cell_gap);

        for (ci, &val) in row.iter().enumerate() {
            if !val.is_finite() {
                continue;
            }
            let cx = grid_x + ci as f32 * (cell_w + hm.cell_gap);
            let t = (val - v_lo) / v_span;
            let cell_color = hm.colormap.as_ref().map_or_else(
                || heatmap::lerp_color(hm.color_lo, hm.color_hi, t),
                |cmap| cmap.color_at(t as f32),
            );

            let corner = hm.cell_radius;
            ctx.draw(|c| {
                c.rect(cx, cy, cell_w, cell_h)
                    .fill(cell_color)
                    .corner_radius(corner)
                    .done()
            });

            // Value label in cell center — contrast-aware text color
            if hm.show_values {
                let text = if val.abs() < 10.0 {
                    format!("{val:.2}")
                } else {
                    format!("{val:.0}")
                };
                ctx.overlays.push(TextOverlay {
                    x_px: cx + cell_w / 2.0,
                    y_px: cy + cell_h / 2.0,
                    text,
                    color: contrast_text_color(cell_color),
                    align: TextAlign::Center,
                    font_size: data_fs,
                    bold: false,
                    rotation_deg: 0.0,
                });
            }
        }
    }

    // ── Color legend bar (vertical gradient on right side) ──
    let bar_x = grid_x + grid_w + legend_gap;
    let bar_y = grid_y;
    let bar_h = grid_h;
    let n_steps = 50;
    let step_h = bar_h / n_steps as f32;
    for s in 0..n_steps {
        // Draw from top (high) to bottom (low)
        let t = 1.0 - s as f64 / (n_steps - 1) as f64;
        let c = hm.colormap.as_ref().map_or_else(
            || heatmap::lerp_color(hm.color_lo, hm.color_hi, t),
            |cmap| cmap.color_at(t as f32),
        );
        let sy = bar_y + s as f32 * step_h;
        ctx.draw(|canv| canv.rect(bar_x, sy, legend_bar_w, step_h + 0.5).fill(c).done());
    }
    // Outline around the bar
    let bar_outline = theme.text_color().with_alpha(0.3);
    ctx.draw(|canv| {
        canv.rect(bar_x, bar_y, legend_bar_w, bar_h)
            .stroke(bar_outline, 0.5)
            .done()
    });
    // Min/max value labels
    let fmt_hi = if v_hi.abs() < 10.0 {
        format!("{v_hi:.1}")
    } else {
        format!("{}", v_hi as i64)
    };
    let fmt_lo = if v_lo.abs() < 10.0 {
        format!("{v_lo:.1}")
    } else {
        format!("{}", v_lo as i64)
    };
    ctx.overlays.push(TextOverlay {
        x_px: bar_x + legend_bar_w + 4.0,
        y_px: bar_y + 2.0,
        text: fmt_hi,
        color: theme.text_color(),
        align: TextAlign::Left,
        font_size: data_fs,
        bold: false,
        rotation_deg: 0.0,
    });
    ctx.overlays.push(TextOverlay {
        x_px: bar_x + legend_bar_w + 4.0,
        y_px: bar_y + bar_h - 2.0,
        text: fmt_lo,
        color: theme.text_color(),
        align: TextAlign::Left,
        font_size: data_fs,
        bold: false,
        rotation_deg: 0.0,
    });

    // Row labels
    for (ri, label) in hm.row_labels.iter().enumerate() {
        let cy = grid_y + ri as f32 * (cell_h + hm.cell_gap) + cell_h / 2.0;
        ctx.overlays.push(TextOverlay {
            x_px: grid_x - 6.0,
            y_px: cy,
            text: label.clone(),
            color: theme.text_color(),
            align: TextAlign::Right,
            font_size: tick_fs,
            bold: false,
            rotation_deg: 0.0,
        });
    }

    // Column labels — positioned above the grid with adequate clearance
    for (ci, label) in hm.col_labels.iter().enumerate() {
        let cx = grid_x + ci as f32 * (cell_w + hm.cell_gap) + cell_w / 2.0;
        ctx.overlays.push(TextOverlay {
            x_px: cx,
            y_px: grid_y - col_label_h * 0.45,
            text: label.clone(),
            color: theme.text_color(),
            align: TextAlign::Center,
            font_size: tick_fs,
            bold: false,
            rotation_deg: 0.0,
        });
    }

    // Title, subtitle, footer, axis labels — handled by RenderContext
    ctx.add_common_overlays(config);
    ctx.finish()
}
