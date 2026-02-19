// SPDX-License-Identifier: MIT OR Apache-2.0
//! Candlestick / OHLC chart type.

use crate::chart::config_builder::{
    chart_config_annotations, chart_config_axis_labels, chart_config_core, chart_config_formatters,
    chart_config_grid, chart_config_h_lines, chart_config_invert, chart_config_legend,
    chart_config_locale, chart_config_margin, chart_config_ranges, chart_config_semantic_zoom,
    chart_config_subtitle_footer, chart_config_tick_rotation, chart_config_tick_steps,
    chart_config_v_lines,
};
use crate::chart::{Chart, ChartConfig};
use crate::spec::ChartSpec;
use scry_engine::style::Color;

/// A single OHLC (Open-High-Low-Close) data point.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OhlcEntry {
    /// X-axis value (timestamp or index).
    pub x: f64,
    /// Opening price.
    pub open: f64,
    /// Highest price.
    pub high: f64,
    /// Lowest price.
    pub low: f64,
    /// Closing price.
    pub close: f64,
}

impl OhlcEntry {
    /// Create a new OHLC entry.
    #[must_use]
    pub fn new(x: f64, open: f64, high: f64, low: f64, close: f64) -> Self {
        Self {
            x,
            open,
            high,
            low,
            close,
        }
    }

    /// Whether this candle is bullish (close ≥ open).
    #[must_use]
    pub fn is_bullish(&self) -> bool {
        self.close >= self.open
    }
}

/// A candlestick chart for financial OHLC data.
///
/// Each candle has a wick (thin line from low to high) and a body
/// (filled rectangle from open to close), colored green for bullish
/// and red for bearish candles.
#[derive(Clone, Debug)]
#[must_use]
pub struct CandlestickChart {
    /// OHLC data entries.
    pub(crate) data: Vec<OhlcEntry>,
    /// Shared chart configuration.
    pub(crate) config: ChartConfig,
    /// Color for bullish (up) candles. Default: green.
    pub(crate) up_color: Color,
    /// Color for bearish (down) candles. Default: red.
    pub(crate) down_color: Color,
    /// Wick width in pixels.
    pub(crate) wick_width: f32,
    /// Body width as fraction of available space (0.0–1.0).
    pub(crate) body_width_frac: f32,
}

impl CandlestickChart {
    /// Create a new candlestick chart from OHLC data.
    pub fn new(data: Vec<OhlcEntry>) -> Self {
        Self {
            data,
            config: ChartConfig::default(),
            up_color: Color::from_rgba8(0, 114, 178, 255),   // Okabe-Ito blue (bullish)
            down_color: Color::from_rgba8(213, 94, 0, 255),  // Okabe-Ito vermillion (bearish)
            wick_width: 1.5,
            body_width_frac: 0.7,
        }
    }

    // --- Generated common methods ---
    chart_config_core!();
    chart_config_axis_labels!();
    chart_config_ranges!(xy);
    chart_config_h_lines!();
    chart_config_v_lines!();
    chart_config_legend!();
    chart_config_annotations!();
    chart_config_grid!();
    chart_config_tick_rotation!();
    chart_config_formatters!();
    chart_config_locale!();
    chart_config_tick_steps!();
    chart_config_subtitle_footer!();
    chart_config_margin!();
    chart_config_invert!();
    chart_config_semantic_zoom!();

    /// Set the color for bullish (up) candles.
    pub fn up_color(mut self, color: Color) -> Self {
        self.up_color = color;
        self
    }

    /// Set the color for bearish (down) candles.
    pub fn down_color(mut self, color: Color) -> Self {
        self.down_color = color;
        self
    }

    /// Set the wick line width in pixels.
    pub fn wick_width(mut self, width: f32) -> Self {
        self.wick_width = width;
        self
    }

    /// Set the body width as a fraction of available space (0.0–1.0).
    pub fn body_width(mut self, frac: f32) -> Self {
        self.body_width_frac = frac.clamp(0.1, 1.0);
        self
    }

    /// Validate inputs and build into a Chart enum variant.
    ///
    /// Returns [`ChartError`](crate::error::ChartError) if data is empty.
    pub fn try_build(self) -> Result<Chart, crate::error::ChartError> {
        if self.data.is_empty() {
            return Err(crate::error::ChartError::EmptyData);
        }
        Ok(self.build())
    }

    /// Build into a Chart.
    pub fn build(self) -> Chart {
        Box::new(self) as Chart
    }
}

impl ChartSpec for CandlestickChart {
    fn render(&self, w: u32, h: u32) -> crate::layout::RenderedChart {
        crate::layout::candlestick::render_candlestick(self, w, h)
    }
    fn render_with_viewport(&self, w: u32, h: u32, vp: Option<(f64, f64, f64, f64)>) -> crate::layout::RenderedChart {
        if let Some((x0, x1, y0, y1)) = vp {
            let mut c = self.clone();
            c.config.axes.x_range = Some((x0, x1));
            c.config.axes.y_range = Some((y0, y1));
            c.render(w, h)
        } else {
            self.render(w, h)
        }
    }
    fn config(&self) -> Option<&ChartConfig> { Some(&self.config) }
    fn config_mut(&mut self) -> Option<&mut ChartConfig> { Some(&mut self.config) }
    fn data_extent(&self) -> Option<(f64, f64, f64, f64)> {
        if self.data.is_empty() { return None; }
        let x_min = self.data.iter().map(|e| e.x).fold(f64::INFINITY, f64::min);
        let x_max = self.data.iter().map(|e| e.x).fold(f64::NEG_INFINITY, f64::max);
        let y_min = self.data.iter().map(|e| e.low).fold(f64::INFINITY, f64::min);
        let y_max = self.data.iter().map(|e| e.high).fold(f64::NEG_INFINITY, f64::max);
        Some((x_min, x_max, y_min, y_max))
    }
    fn clone_boxed(&self) -> Box<dyn ChartSpec> { Box::new(self.clone()) }
}
