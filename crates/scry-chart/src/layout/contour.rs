// SPDX-License-Identifier: MIT OR Apache-2.0
//! Contour chart layout — iso-level line rendering from 2D scalar fields.

use crate::chart::contour::ContourChart;
use crate::scale::{LinearScale, Scale};

use super::{RenderContext, RenderedChart};

/// Render a contour chart.
pub(crate) fn render_contour(cc: &ContourChart, w: u32, h: u32) -> RenderedChart {
    let config = &cc.config;

    let rows = cc.data.len();
    let cols = cc.data.first().map_or(0, |r| r.len());

    // Grid axes: x = 0..cols-1, y = 0..rows-1
    let x_extent = (0.0, (cols.saturating_sub(1)) as f64);
    let y_extent = (0.0, (rows.saturating_sub(1)) as f64);

    let mut ctx = RenderContext::new(config, w, h, Some(y_extent));
    let (px, py, pw, ph) = ctx.plot;

    let x_scale = LinearScale::nice(x_extent, (px as f64, (px + pw) as f64));
    let y_scale = LinearScale::nice(y_extent, ((py + ph) as f64, py as f64));

    ctx.draw_axes(config, &x_scale, &y_scale);

    // Compute value range
    let (v_min, v_max) = cc.data_extent().unwrap_or((0.0, 1.0));
    let v_range = (v_max - v_min).max(1e-12);

    // Generate iso-levels
    let levels: Vec<f64> = (0..cc.levels)
        .map(|i| v_min + (i as f64 + 0.5) * v_range / cc.levels as f64)
        .collect();

    // Marching squares for each level
    for (li, &level) in levels.iter().enumerate() {
        let t = li as f32 / (levels.len().max(1) - 1).max(1) as f32;
        let color = if let Some(ref cm) = cc.colormap {
            cm.color_at(t)
        } else {
            lerp_color(cc.color_lo, cc.color_hi, t)
        };

        // If filled, draw smooth bilinear-interpolated sub-cells.
        if cc.filled {
            let next_level = levels.get(li + 1).copied().unwrap_or(v_max + 1.0);
            // Subdivide each data cell into SUB×SUB sub-cells for smooth fill.
            const SUB: usize = 4;
            let sub_f = SUB as f64;
            for r in 0..rows.saturating_sub(1) {
                for c in 0..cols.saturating_sub(1) {
                    let v00 = cc.data[r][c];
                    let v10 = cc.data[r][c + 1];
                    let v01 = cc.data[r + 1][c];
                    let v11 = cc.data[r + 1][c + 1];

                    for sr in 0..SUB {
                        for sc in 0..SUB {
                            // Centre of this sub-cell in [0,1]² within the data cell
                            let u = (sc as f64 + 0.5) / sub_f;
                            let v = (sr as f64 + 0.5) / sub_f;
                            let val = bilinear(v00, v10, v01, v11, u, v);

                            if val >= level && val < next_level {
                                // Map value to a continuous color via the full range
                                let ct = ((val - v_min) / v_range).clamp(0.0, 1.0) as f32;
                                let sub_color = if let Some(ref cm) = cc.colormap {
                                    cm.color_at(ct)
                                } else {
                                    lerp_color(cc.color_lo, cc.color_hi, ct)
                                };
                                let fill_color = sub_color.with_alpha(0.55);

                                // Sub-cell pixel coordinates
                                let x0 = x_scale.to_pixel(c as f64 + sc as f64 / sub_f) as f32;
                                let x1 = x_scale.to_pixel(c as f64 + (sc + 1) as f64 / sub_f) as f32;
                                let y0 = y_scale.to_pixel(r as f64 + sr as f64 / sub_f) as f32;
                                let y1 = y_scale.to_pixel(r as f64 + (sr + 1) as f64 / sub_f) as f32;
                                let rx = x0.min(x1);
                                let ry = y0.min(y1);
                                // +0.5 to avoid hairline gaps between sub-cells
                                let rw = (x1 - x0).abs() + 0.5;
                                let rh = (y1 - y0).abs() + 0.5;
                                ctx.draw(|cv| cv.rect(rx, ry, rw, rh).fill(fill_color).done());
                            }
                        }
                    }
                }
            }
        }

        // Marching squares: extract line segments at this iso-level.
        let segments = march_squares(&cc.data, rows, cols, level);
        let line_width = if cc.filled { 0.8 } else { 1.5 };
        for (r0, c0, r1, c1) in &segments {
            let x0 = x_scale.to_pixel(*c0) as f32;
            let y0 = y_scale.to_pixel(*r0) as f32;
            let x1 = x_scale.to_pixel(*c1) as f32;
            let y1 = y_scale.to_pixel(*r1) as f32;
            ctx.draw(|cv| cv.line(x0, y0, x1, y1).color(color).width(line_width).done());
        }
    }

    ctx.add_common_overlays(config);
    ctx.finish()
}

/// Linearly interpolate between two colors.
fn lerp_color(a: scry_engine::style::Color, b: scry_engine::style::Color, t: f32) -> scry_engine::style::Color {
    let t = t.clamp(0.0, 1.0);
    scry_engine::style::Color::from_rgba(
        a.r + (b.r - a.r) * t,
        a.g + (b.g - a.g) * t,
        a.b + (b.b - a.b) * t,
        a.a + (b.a - a.a) * t,
    )
}

/// Bilinear interpolation of four corner values.
///
/// `v00` = top-left, `v10` = top-right, `v01` = bottom-left, `v11` = bottom-right.
/// `u` and `v` are in [0, 1].
#[inline]
fn bilinear(v00: f64, v10: f64, v01: f64, v11: f64, u: f64, v: f64) -> f64 {
    let top = v00 + (v10 - v00) * u;
    let bot = v01 + (v11 - v01) * u;
    top + (bot - top) * v
}

/// Marching squares: compute iso-level line segments from a 2D grid.
///
/// Returns segments as `(row0, col0, row1, col1)` in fractional grid coords.
fn march_squares(
    data: &[Vec<f64>],
    rows: usize,
    cols: usize,
    level: f64,
) -> Vec<(f64, f64, f64, f64)> {
    let mut segments = Vec::new();
    if rows < 2 || cols < 2 {
        return segments;
    }

    for r in 0..rows - 1 {
        for c in 0..cols - 1 {
            let v00 = data[r][c];
            let v10 = data[r][c + 1];
            let v01 = data[r + 1][c];
            let v11 = data[r + 1][c + 1];

            // Classify corners: 1 = above level, 0 = below
            let case = ((v00 >= level) as u8)
                | (((v10 >= level) as u8) << 1)
                | (((v11 >= level) as u8) << 2)
                | (((v01 >= level) as u8) << 3);

            if case == 0 || case == 15 {
                continue; // fully inside or outside
            }

            // Interpolation helpers: position along edge where level crosses.
            let interp = |a: f64, b: f64| -> f64 {
                if (b - a).abs() < 1e-12 {
                    0.5
                } else {
                    (level - a) / (b - a)
                }
            };

            // Edge midpoints (fractional coords):
            // top:    (r, c+t)    between v00 and v10
            // right:  (r+t, c+1)  between v10 and v11
            // bottom: (r+1, c+t)  between v01 and v11
            // left:   (r+t, c)    between v00 and v01
            let top = (r as f64, c as f64 + interp(v00, v10));
            let right = (r as f64 + interp(v10, v11), (c + 1) as f64);
            let bottom = ((r + 1) as f64, c as f64 + interp(v01, v11));
            let left = (r as f64 + interp(v00, v01), c as f64);

            // Emit segments based on the 16 marching-squares cases.
            match case {
                1 | 14 => segments.push((top.0, top.1, left.0, left.1)),
                2 | 13 => segments.push((top.0, top.1, right.0, right.1)),
                3 | 12 => segments.push((left.0, left.1, right.0, right.1)),
                4 | 11 => segments.push((right.0, right.1, bottom.0, bottom.1)),
                5 => {
                    // Saddle: two segments
                    segments.push((top.0, top.1, right.0, right.1));
                    segments.push((left.0, left.1, bottom.0, bottom.1));
                }
                6 | 9 => segments.push((top.0, top.1, bottom.0, bottom.1)),
                7 | 8 => segments.push((left.0, left.1, bottom.0, bottom.1)),
                10 => {
                    // Saddle: two segments
                    segments.push((top.0, top.1, left.0, left.1));
                    segments.push((right.0, right.1, bottom.0, bottom.1));
                }
                _ => {}
            }
        }
    }

    segments
}
