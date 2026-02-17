// SPDX-License-Identifier: MIT OR Apache-2.0
//! Candlestick chart rendering.
//!
//! Each OHLC entry is drawn as:
//! - A thin vertical line (wick) from low to high
//! - A filled rectangle (body) from open to close
//! - Colored green for bullish, red for bearish

use crate::chart::candlestick::CandlestickChart;
use crate::scale::{LinearScale, Scale};

use super::{resolve_y_extent, RenderContext, RenderedChart};

pub(crate) fn render_candlestick(cc: &CandlestickChart, w: u32, h: u32) -> RenderedChart {
    let config = &cc.config;

    if cc.data.is_empty() {
        let mut ctx = RenderContext::new(config, w, h, None);
        ctx.add_common_overlays(config);
        return ctx.finish();
    }

    // Compute X domain
    let x_lo = cc.data.iter().map(|e| e.x).reduce(f64::min).unwrap_or(0.0);
    let x_hi = cc.data.iter().map(|e| e.x).reduce(f64::max).unwrap_or(1.0);
    // Add half-candle padding so edge candles aren't clipped
    let x_span = if (x_hi - x_lo).abs() < f64::EPSILON {
        1.0
    } else {
        x_hi - x_lo
    };
    let x_pad = x_span * 0.02;

    // Compute Y domain (min of all lows, max of all highs)
    let y_lo = cc
        .data
        .iter()
        .map(|e| e.low)
        .reduce(f64::min)
        .unwrap_or(0.0);
    let y_hi = cc
        .data
        .iter()
        .map(|e| e.high)
        .reduce(f64::max)
        .unwrap_or(1.0);
    let y_extent = resolve_y_extent(config, (y_lo, y_hi));

    let mut ctx = RenderContext::new(config, w, h, Some(y_extent));
    let (px, py, pw, ph) = ctx.plot;

    // Override x domain if user set x_range
    let x_domain = config.x_range.unwrap_or((x_lo - x_pad, x_hi + x_pad));
    let x_scale = LinearScale::nice(x_domain, (px as f64, (px + pw) as f64));
    let y_scale = LinearScale::nice(y_extent, ((py + ph) as f64, py as f64));

    // Draw axes
    let y_ticks = ctx.draw_y_axis(config, &y_scale);
    ctx.add_y_tick_overlays(&y_ticks, config.theme.text_color());
    ctx.draw_x_value_axis(config, &x_scale);
    ctx.draw_reference_lines(config, &x_scale, &y_scale);

    // Compute candle body width
    let n = cc.data.len();
    let body_width = if n > 1 {
        let min_gap = cc
            .data
            .windows(2)
            .map(|pair| (pair[1].x - pair[0].x).abs())
            .reduce(f64::min)
            .unwrap_or(1.0);
        let gap_px = (x_scale.to_pixel(x_lo + min_gap) - x_scale.to_pixel(x_lo)).abs() as f32;
        (gap_px * cc.body_width_frac).max(2.0)
    } else {
        (pw * cc.body_width_frac * 0.3).max(4.0)
    };

    // Draw candles
    for entry in &cc.data {
        let cx = x_scale.to_pixel(entry.x) as f32;
        let wick_top = y_scale.to_pixel(entry.high) as f32;
        let wick_bottom = y_scale.to_pixel(entry.low) as f32;
        let body_top = y_scale.to_pixel(entry.open.max(entry.close)) as f32;
        let body_bottom = y_scale.to_pixel(entry.open.min(entry.close)) as f32;

        let color = if entry.is_bullish() {
            cc.up_color
        } else {
            cc.down_color
        };

        let wick_w = cc.wick_width;
        let half_body = body_width / 2.0;

        // Wick (thin vertical line from low to high)
        ctx.draw(|c| {
            c.line(cx, wick_top, cx, wick_bottom)
                .color(color)
                .width(wick_w)
                .done()
        });

        // Body (filled rectangle from open to close)
        let body_h = (body_bottom - body_top).max(1.0);
        ctx.draw(|c| {
            c.rect(cx - half_body, body_top, body_width, body_h)
                .fill(color)
                .done()
        });

        // Body outline for bearish candles (visual distinction)
        if !entry.is_bullish() {
            let outline = color.with_alpha(0.7);
            ctx.draw(|c| {
                c.rect(cx - half_body, body_top, body_width, body_h)
                    .stroke(outline, 1.0)
                    .done()
            });
        }
    }

    ctx.add_common_overlays(config);
    ctx.finish()
}
