// SPDX-License-Identifier: MIT OR Apache-2.0
//! Gauge chart type — KPI speedometer visualization.
//!
//! Renders a semicircular arc with colored threshold bands and a needle
//! pointing to the current value.

use crate::chart::config_builder::{
    chart_config_core, chart_config_margin, chart_config_subtitle_footer,
};
use crate::chart::{Chart, ChartConfig};
use crate::spec::ChartSpec;
use scry_engine::style::Color;

/// A gauge chart — semicircular arc with a needle indicator.
///
/// # Examples
///
/// ```
/// use scry_chart::chart::Chart;
/// use scry_engine::style::Color;
///
/// let chart = Charts::gauge(75.0)
///     .range(0.0, 100.0)
///     .threshold(60.0, Color::from_rgba8(40, 180, 99, 255))
///     .threshold(80.0, Color::from_rgba8(241, 196, 15, 255))
///     .threshold(100.0, Color::from_rgba8(231, 76, 60, 255))
///     .title("CPU Usage")
///     .build();
/// ```
#[derive(Clone, Debug)]
#[must_use]
pub struct GaugeChart {
    /// Current value.
    pub(crate) value: f64,
    /// Gauge minimum (default: 0.0).
    pub(crate) min: f64,
    /// Gauge maximum (default: 100.0).
    pub(crate) max: f64,
    /// Shared config.
    pub(crate) config: ChartConfig,
    /// Center label (e.g. "75%").
    pub(crate) label: Option<String>,
    /// Color threshold bands: (upper_bound, color).
    pub(crate) thresholds: Vec<(f64, Color)>,
    /// Needle color override.
    pub(crate) needle_color: Option<Color>,
    /// Arc track thickness (default: 12.0).
    pub(crate) arc_width: f32,
}

impl GaugeChart {
    /// Create a new gauge chart with the given value.
    pub fn new(value: f64) -> Self {
        Self {
            value,
            min: 0.0,
            max: 100.0,
            config: ChartConfig::default(),
            label: None,
            thresholds: Vec::new(),
            needle_color: None,
            arc_width: 12.0,
        }
    }

    // --- Generated common methods ---
    chart_config_core!();
    chart_config_subtitle_footer!();
    chart_config_margin!();

    /// Set the gauge range.
    pub fn range(mut self, min: f64, max: f64) -> Self {
        self.min = min;
        self.max = max;
        self
    }

    /// Set the center label text (displayed below the needle).
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Add a color threshold band.
    ///
    /// Thresholds are drawn in order from `min` to each `upper_bound`.
    /// The final threshold should equal `max`.
    pub fn threshold(mut self, upper_bound: f64, color: Color) -> Self {
        self.thresholds.push((upper_bound, color));
        self
    }

    /// Set the needle color.
    pub fn needle_color(mut self, color: Color) -> Self {
        self.needle_color = Some(color);
        self
    }

    /// Set the arc track thickness in pixels.
    pub fn arc_width(mut self, width: f32) -> Self {
        self.arc_width = width.max(2.0);
        self
    }

    /// Build into a Chart.
    pub fn build(self) -> Chart {
        Box::new(self) as Chart
    }

    /// Build with validation.
    pub fn try_build(self) -> Result<Chart, crate::error::ChartError> {
        if !self.value.is_finite() {
            return Err(crate::error::ChartError::AllNonFinite);
        }
        if self.min >= self.max || !self.min.is_finite() || !self.max.is_finite() {
            return Err(crate::error::ChartError::InvalidRange {
                min: self.min,
                max: self.max,
            });
        }
        Ok(self.build())
    }
}

impl ChartSpec for GaugeChart {
    fn render(&self, w: u32, h: u32) -> crate::layout::RenderedChart {
        crate::layout::gauge::render_gauge(self, w, h)
    }
    fn config(&self) -> Option<&ChartConfig> { Some(&self.config) }
    fn config_mut(&mut self) -> Option<&mut ChartConfig> { Some(&mut self.config) }
    fn clone_boxed(&self) -> Box<dyn ChartSpec> { Box::new(self.clone()) }
}
