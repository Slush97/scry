// SPDX-License-Identifier: MIT OR Apache-2.0
//! Violin plot rendering — mirrored KDE curves with optional inner box-and-whisker.

use crate::chart::violin::ViolinPlot;
use crate::scale::{CategoricalScale, LinearScale, Scale};

use super::{RenderContext, RenderedChart};

pub(crate) fn render_violin(vp: &ViolinPlot, w: u32, h: u32) -> RenderedChart {
    let config = &vp.config;
    let theme = &config.theme;
    let labels: Vec<String> = vp.groups.iter().map(|(l, _)| l.clone()).collect();

    // Pre-compute global y extent across all groups
    let (global_min, global_max) = {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for (_, vals) in &vp.groups {
            for &v in vals {
                if v.is_finite() {
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
        }
        if lo > hi {
            (0.0, 1.0)
        } else {
            (lo - (hi - lo) * 0.1, hi + (hi - lo) * 0.1)
        }
    };

    let y_extent = (global_min, global_max);
    let mut ctx = RenderContext::new(config, w, h, Some(y_extent));
    let (px, py, pw, ph) = ctx.plot;

    let cat_scale = CategoricalScale::new(labels.clone(), (px as f64, (px + pw) as f64));
    let y_scale = LinearScale::nice(y_extent, ((py + ph) as f64, py as f64));

    // Draw Y axis
    let y_ticks = ctx.draw_y_axis(config, &y_scale);
    ctx.add_y_tick_overlays(&y_ticks, theme.text_color());
    ctx.draw_x_axis_line(config);
    ctx.draw_categorical_x_labels(config, &cat_scale, &labels);

    // Compute max KDE density across all groups for normalization
    let kde_points = 50;
    let mut all_kdes: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut max_density: f64 = 0.0;

    for (_, vals) in &vp.groups {
        let mut sorted: Vec<f64> = vals.iter().copied().filter(|v| v.is_finite()).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if sorted.is_empty() {
            all_kdes.push(Vec::new());
            continue;
        }
        let bw = vp.bandwidth.unwrap_or_else(|| silverman_bandwidth(&sorted));
        let kde = compute_kde(&sorted, bw, global_min, global_max, kde_points);
        for &(_, d) in &kde {
            max_density = max_density.max(d);
        }
        all_kdes.push(kde);
    }

    if max_density < f64::EPSILON {
        max_density = 1.0;
    }

    // Draw each violin
    let band_width = cat_scale.band_width();
    let half_max = band_width * 0.4; // Max half-width of violin in pixels

    for (gi, kde) in all_kdes.iter().enumerate() {
        if kde.is_empty() {
            continue;
        }
        let center_x = cat_scale.center(gi) as f32;
        let color = theme.resolve_series_color(gi, &crate::data::SeriesStyle::default());

        // Build mirrored polygon: right side (top to bottom), then left side (bottom to top)
        let mut points: Vec<(f32, f32)> = Vec::with_capacity(kde.len() * 2 + 2);

        // Right side
        for &(y_val, density) in kde {
            let py_val = y_scale.to_pixel(y_val) as f32;
            let dx = (density / max_density * half_max) as f32;
            points.push((center_x + dx, py_val));
        }

        // Left side (reversed)
        for &(y_val, density) in kde.iter().rev() {
            let py_val = y_scale.to_pixel(y_val) as f32;
            let dx = (density / max_density * half_max) as f32;
            points.push((center_x - dx, py_val));
        }

        // Draw filled polygon
        let fill_color = color.with_alpha(0.4);
        let stroke_color = color;
        let pts = points.clone();
        ctx.draw(|c| {
            c.polygon(pts)
                .fill(fill_color)
                .stroke(stroke_color, 1.5)
                .done()
        });

        // Inner box-and-whisker
        if vp.show_inner_box {
            let mut sorted: Vec<f64> = vp.groups[gi]
                .1
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if sorted.len() >= 2 {
                let q1 = percentile(&sorted, 25.0);
                let median = percentile(&sorted, 50.0);
                let q3 = percentile(&sorted, 75.0);
                let iqr = q3 - q1;
                let whisker_lo = sorted
                    .iter()
                    .copied()
                    .find(|&v| v >= q1 - 1.5 * iqr)
                    .unwrap_or(q1);
                let whisker_hi = sorted
                    .iter()
                    .rev()
                    .copied()
                    .find(|&v| v <= q3 + 1.5 * iqr)
                    .unwrap_or(q3);

                let box_half_w = (half_max * 0.15).max(2.0) as f32;
                let pq1 = y_scale.to_pixel(q1) as f32;
                let pq3 = y_scale.to_pixel(q3) as f32;
                let pmed = y_scale.to_pixel(median) as f32;
                let pwlo = y_scale.to_pixel(whisker_lo) as f32;
                let pwhi = y_scale.to_pixel(whisker_hi) as f32;
                let box_color = theme.foreground.with_alpha(0.9);

                // Box Q1–Q3
                let bw = box_half_w;
                ctx.draw(|c| {
                    c.rect(center_x - bw, pq3, bw * 2.0, pq1 - pq3)
                        .fill(box_color.with_alpha(0.2))
                        .stroke(box_color, 1.0)
                        .done()
                });

                // Median line
                ctx.draw(|c| {
                    c.line(center_x - bw, pmed, center_x + bw, pmed)
                        .color(box_color)
                        .width(2.0)
                        .done()
                });

                // Whiskers
                ctx.draw(|c| {
                    c.line(center_x, pq3, center_x, pwhi)
                        .color(box_color)
                        .width(1.0)
                        .done()
                });
                ctx.draw(|c| {
                    c.line(center_x, pq1, center_x, pwlo)
                        .color(box_color)
                        .width(1.0)
                        .done()
                });
            }
        }
    }

    ctx.add_common_overlays(config);
    ctx.finish()
}

// ---------------------------------------------------------------------------
// KDE helpers
// ---------------------------------------------------------------------------

/// Silverman's rule-of-thumb bandwidth.
fn silverman_bandwidth(sorted: &[f64]) -> f64 {
    let n = sorted.len() as f64;
    if n < 2.0 {
        return 1.0;
    }
    let mean = sorted.iter().sum::<f64>() / n;
    let variance = sorted.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();
    let iqr = percentile(sorted, 75.0) - percentile(sorted, 25.0);
    let spread = std_dev.min(iqr / 1.34);
    let s = if spread > 0.0 {
        spread
    } else {
        std_dev.max(1.0)
    };
    0.9 * s * n.powf(-0.2)
}

/// Compute KDE using Gaussian kernel.
fn compute_kde(
    sorted: &[f64],
    bandwidth: f64,
    lo: f64,
    hi: f64,
    num_points: usize,
) -> Vec<(f64, f64)> {
    let n = sorted.len() as f64;
    let bw = bandwidth.max(f64::EPSILON);
    let step = (hi - lo) / (num_points - 1).max(1) as f64;

    (0..num_points)
        .map(|i| {
            let x = lo + i as f64 * step;
            let density: f64 = sorted
                .iter()
                .map(|&xi| {
                    let z = (x - xi) / bw;
                    (-0.5 * z * z).exp() / (bw * (2.0 * std::f64::consts::PI).sqrt())
                })
                .sum::<f64>()
                / n;
            (x, density)
        })
        .collect()
}

/// Linear interpolation percentile on a sorted slice.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let rank = (p / 100.0) * (n - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let frac = rank - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi.min(n - 1)] * frac
}
