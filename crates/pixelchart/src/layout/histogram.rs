//! Histogram rendering.

use crate::chart::histogram::Histogram;
use crate::legend::{self, LegendEntry};
use crate::scale::{LinearScale, Scale};

use super::{resolve_x_extent, resolve_y_extent, take_canvas, RenderContext, RenderedChart, TextAlign, TextOverlay};

pub(crate) fn render_histogram(hc: &Histogram, w: u32, h: u32) -> RenderedChart {
    let config = &hc.config;
    let theme = &config.theme;
    let mut ctx = RenderContext::new(config, w, h);
    let (px, py, pw, ph) = ctx.plot;

    // Compute shared x extent across all series
    let primary_extent = hc.data.extent().unwrap_or((0.0, 1.0));
    let x_extent = if hc.extra.is_empty() {
        resolve_x_extent(config, primary_extent)
    } else {
        // Expand extent to cover all series
        let mut lo = primary_extent.0;
        let mut hi = primary_extent.1;
        for extra in &hc.extra {
            if let Some((elo, ehi)) = extra.extent() {
                lo = lo.min(elo);
                hi = hi.max(ehi);
            }
        }
        resolve_x_extent(config, (lo, hi))
    };

    let n_bins = hc.bins.unwrap_or_else(|| Histogram::auto_bins(hc.data.len()));

    // Compute bins for all series
    let primary_bins = compute_bins(hc.data.values(), x_extent, n_bins, hc.density);
    let extra_bins: Vec<Vec<Bin>> = hc
        .extra
        .iter()
        .map(|s| compute_bins(s.values(), x_extent, n_bins, hc.density))
        .collect();

    // Y axis max across all series
    let mut y_max = primary_bins.iter().map(|b| b.count).reduce(f64::max).unwrap_or(1.0);
    for bins in &extra_bins {
        if let Some(m) = bins.iter().map(|b| b.count).reduce(f64::max) {
            y_max = y_max.max(m);
        }
    }

    let x_scale = LinearScale::nice(x_extent, (px as f64, (px + pw) as f64));
    let y_extent = resolve_y_extent(config, (0.0, y_max));
    let y_scale = LinearScale::nice(y_extent, ((py + ph) as f64, py as f64));

    ctx.draw_axes(config, &x_scale, &y_scale);
    ctx.draw_reference_lines(config, &x_scale, &y_scale);

    let baseline_y = y_scale.to_pixel(0.0) as f32;

    // Draw primary histogram
    let n_total_series = 1 + hc.extra.len();
    let primary_opacity = if n_total_series > 1 { 0.6 } else { hc.opacity };
    draw_bins_on_ctx(&mut ctx, &primary_bins, &x_scale, &y_scale, baseline_y, theme.series_color(0), primary_opacity);

    // Draw extra series with translucent overlay
    for (si, bins) in extra_bins.iter().enumerate() {
        let color = theme.series_color(si + 1);
        draw_bins_on_ctx(&mut ctx, bins, &x_scale, &y_scale, baseline_y, color, 0.5);
    }

    // Legend for multi-series
    if n_total_series > 1 {
        let mut entries = vec![
            LegendEntry {
                label: if hc.data.label().is_empty() {
                    "Series 1".to_string()
                } else {
                    hc.data.label().to_string()
                },
                color: theme.series_color(0),
            }
        ];
        for (si, extra) in hc.extra.iter().enumerate() {
            entries.push(LegendEntry {
                label: if extra.label().is_empty() {
                    format!("Series {}", si + 2)
                } else {
                    extra.label().to_string()
                },
                color: theme.series_color(si + 1),
            });
        }

        let (canvas, legend_text) = legend::draw_legend_swatches(
            take_canvas(&mut ctx),
            &entries,
            px + pw - 80.0,
            py + 8.0,
            10.0,
            4.0,
        );
        ctx.canvas = canvas;

        // Add legend text overlays (was previously discarded!)
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

    ctx.add_common_overlays(config);
    ctx.finish()
}

/// Draw histogram bins onto a RenderContext.
fn draw_bins_on_ctx(
    ctx: &mut RenderContext,
    bins: &[Bin],
    x_scale: &LinearScale,
    y_scale: &LinearScale,
    baseline_y: f32,
    color: ratatui_pixelcanvas::style::Color,
    opacity: f32,
) {
    let fill_color = color.with_alpha(opacity);

    for bin in bins {
        let x1 = x_scale.to_pixel(bin.lo) as f32;
        let x2 = x_scale.to_pixel(bin.hi) as f32;
        let top = y_scale.to_pixel(bin.count) as f32;
        let bw = (x2 - x1).max(1.0);
        let bh = baseline_y - top;

        if bh > 0.0 {
            ctx.canvas = take_canvas(ctx)
                .rect(x1, top, bw, bh)
                .fill(fill_color)
                .done();

            ctx.canvas = take_canvas(ctx)
                .rect(x1, top, bw, bh)
                .stroke(color, 1.0)
                .done();
        }
    }
}

// ---------------------------------------------------------------------------
// Binning logic
// ---------------------------------------------------------------------------

pub(crate) struct Bin {
    pub lo: f64,
    pub hi: f64,
    pub count: f64,
}

pub(crate) fn compute_bins(data: &[f64], extent: (f64, f64), n_bins: usize, density: bool) -> Vec<Bin> {
    let (lo, hi) = extent;
    let span = hi - lo;
    if span.abs() < f64::EPSILON || n_bins == 0 {
        return vec![];
    }

    let bin_width = span / n_bins as f64;
    let mut counts = vec![0usize; n_bins];

    for &v in data {
        if v < lo || v > hi {
            continue;
        }
        let idx = ((v - lo) / bin_width) as usize;
        let idx = idx.min(n_bins - 1);
        counts[idx] += 1;
    }

    let n = data.len() as f64;
    counts
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let bin_lo = lo + i as f64 * bin_width;
            let bin_hi = bin_lo + bin_width;
            let count = if density {
                c as f64 / (n * bin_width)
            } else {
                c as f64
            };
            Bin {
                lo: bin_lo,
                hi: bin_hi,
                count,
            }
        })
        .collect()
}
