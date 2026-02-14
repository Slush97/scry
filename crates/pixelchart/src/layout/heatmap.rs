//! Heatmap rendering.

use ratatui_pixelcanvas::scene::PixelCanvas;
use ratatui_pixelcanvas::style::Color;

use crate::chart::heatmap::{self, Heatmap};

use super::{
    proportional_margin, proportional_title_height, RenderedChart, TextAlign, TextOverlay,
};

pub(crate) fn render_heatmap(hm: &Heatmap, w: u32, h: u32) -> RenderedChart {
    let config = &hm.config;
    let theme = &config.theme;

    // Heatmap has a special layout: more left margin for row labels
    let margin = proportional_margin(w, h);
    let title_h = if config.title.is_some() {
        proportional_title_height(h)
    } else {
        0.0
    };
    let col_label_h = 20.0_f32;
    let row_label_w = {
        let max_chars = hm
            .row_labels
            .iter()
            .map(|l| l.chars().count())
            .max()
            .unwrap_or(0);
        ((max_chars as f32) * 7.0 + 12.0)
            .min(w as f32 * 0.3)
            .max(20.0)
    };

    let grid_x = margin + row_label_w;
    let grid_y = margin + title_h + col_label_h;
    let grid_w = (w as f32 - grid_x - margin).max(1.0);
    let grid_h = (h as f32 - grid_y - margin).max(1.0);

    let n_rows = hm.data.len();
    let n_cols = hm.data.first().map_or(0, |r| r.len());

    if n_rows == 0 || n_cols == 0 {
        return RenderedChart {
            canvas: PixelCanvas::new(w, h).background(theme.background),
            text_overlays: Vec::new(),
            plot_area: None,
            x_scale: None,
            y_scale: None,
            series_points: Vec::new(),
        };
    }

    let cell_w = (grid_w - hm.cell_gap * (n_cols as f32 - 1.0).max(0.0)) / n_cols as f32;
    let cell_h = (grid_h - hm.cell_gap * (n_rows as f32 - 1.0).max(0.0)) / n_rows as f32;

    let (v_lo, v_hi) = hm.value_range.unwrap_or_else(|| hm.data_extent());
    let v_span = if (v_hi - v_lo).abs() < f64::EPSILON {
        1.0
    } else {
        v_hi - v_lo
    };

    let mut canvas = PixelCanvas::new(w, h).background(theme.background);
    let mut overlays: Vec<TextOverlay> = Vec::new();

    // Draw cells
    for (ri, row) in hm.data.iter().enumerate() {
        let cy = grid_y + ri as f32 * (cell_h + hm.cell_gap);

        for (ci, &val) in row.iter().enumerate() {
            if !val.is_finite() {
                continue;
            }
            let cx = grid_x + ci as f32 * (cell_w + hm.cell_gap);
            let t = (val - v_lo) / v_span;
            let cell_color = heatmap::lerp_color(hm.color_lo, hm.color_hi, t);

            canvas = canvas
                .rect(cx, cy, cell_w, cell_h)
                .fill(cell_color)
                .corner_radius(hm.cell_radius)
                .done();

            // Value label in cell center
            if hm.show_values {
                let text = if val.abs() < 10.0 {
                    format!("{val:.2}")
                } else {
                    format!("{val:.0}")
                };
                overlays.push(TextOverlay {
                    x_px: cx + cell_w / 2.0,
                    y_px: cy + cell_h / 2.0,
                    text,
                    color: if t > 0.5 {
                        Color::from_rgba8(255, 255, 255, 220)
                    } else {
                        Color::from_rgba8(200, 200, 200, 220)
                    },
                    align: TextAlign::Center,
                    font_size: 10.0,
                    bold: false,
                    rotation_deg: 0.0,
                });
            }
        }
    }

    // Row labels
    for (ri, label) in hm.row_labels.iter().enumerate() {
        let cy = grid_y + ri as f32 * (cell_h + hm.cell_gap) + cell_h / 2.0;
        overlays.push(TextOverlay {
            x_px: grid_x - 6.0,
            y_px: cy,
            text: label.clone(),
            color: theme.text_color(),
            align: TextAlign::Right,
            font_size: 11.0,
            bold: false,
            rotation_deg: 0.0,
        });
    }

    // Column labels
    for (ci, label) in hm.col_labels.iter().enumerate() {
        let cx = grid_x + ci as f32 * (cell_w + hm.cell_gap) + cell_w / 2.0;
        overlays.push(TextOverlay {
            x_px: cx,
            y_px: grid_y - 6.0,
            text: label.clone(),
            color: theme.text_color(),
            align: TextAlign::Center,
            font_size: 11.0,
            bold: false,
            rotation_deg: 0.0,
        });
    }

    // Title
    if let Some(ref title) = config.title {
        overlays.push(TextOverlay {
            x_px: grid_x + grid_w / 2.0,
            y_px: margin / 2.0,
            text: title.clone(),
            color: theme.title_style.color,
            align: TextAlign::Center,
            font_size: 18.0,
            bold: true,
            rotation_deg: 0.0,
        });
    }

    RenderedChart {
        canvas,
        text_overlays: overlays,
        plot_area: Some((grid_x, grid_y, grid_w, grid_h)),
        x_scale: None,
        y_scale: None,
        series_points: Vec::new(),
    }
}
