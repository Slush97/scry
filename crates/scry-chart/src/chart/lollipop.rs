// SPDX-License-Identifier: MIT OR Apache-2.0
//! Lollipop chart type — bar chart variant with thin stems and dot markers.
//!
//! Each category gets a thin vertical (or horizontal) line from the baseline
//! to the value, topped with a filled circle.

use crate::chart::config_builder::{
    chart_config_axis_labels, chart_config_core, chart_config_formatters, chart_config_grid,
    chart_config_h_lines, chart_config_legend, chart_config_locale, chart_config_margin,
    chart_config_subtitle_footer, chart_config_tick_rotation, chart_config_tick_steps,
};
use crate::chart::{Chart, ChartConfig};

/// A lollipop chart — thin stems topped with circular markers.
///
/// # Examples
///
/// ```
/// use scry_chart::chart::Chart;
///
/// let chart = Chart::lollipop(
///     vec!["Mon".into(), "Tue".into(), "Wed".into(), "Thu".into(), "Fri".into()],
///     &[12.0, 19.0, 8.0, 15.0, 22.0],
/// )
/// .title("Weekly Scores")
/// .build();
/// ```
#[derive(Clone, Debug)]
#[must_use]
pub struct LollipopChart {
    /// Category labels.
    pub(crate) labels: Vec<String>,
    /// Data values.
    pub(crate) values: Vec<f64>,
    /// Shared config.
    pub(crate) config: ChartConfig,
    /// Horizontal orientation (default: false = vertical).
    pub(crate) horizontal: bool,
    /// Dot radius in pixels (default: 5.0).
    pub(crate) dot_radius: f32,
    /// Stem line width (default: 2.0).
    pub(crate) stem_width: f32,
    /// Whether to show numeric value labels.
    pub(crate) show_values: bool,
}

impl LollipopChart {
    /// Create a new lollipop chart.
    pub fn new(labels: Vec<String>, values: Vec<f64>) -> Self {
        Self {
            labels,
            values,
            config: ChartConfig::default(),
            horizontal: false,
            dot_radius: 5.0,
            stem_width: 2.0,
            show_values: false,
        }
    }

    // --- Generated common methods ---
    chart_config_core!();
    chart_config_axis_labels!();
    chart_config_h_lines!();
    chart_config_legend!();
    chart_config_grid!();
    chart_config_tick_rotation!();
    chart_config_formatters!();
    chart_config_locale!();
    chart_config_tick_steps!();
    chart_config_subtitle_footer!();
    chart_config_margin!();

    /// Use horizontal orientation (stems grow rightward).
    pub fn horizontal(mut self) -> Self {
        self.horizontal = true;
        self
    }

    /// Set the dot radius in pixels.
    pub fn dot_radius(mut self, radius: f32) -> Self {
        self.dot_radius = radius.max(1.0);
        self
    }

    /// Set the stem line width.
    pub fn stem_width(mut self, width: f32) -> Self {
        self.stem_width = width.max(0.5);
        self
    }

    /// Show numeric value labels above each dot.
    pub fn show_values(mut self) -> Self {
        self.show_values = true;
        self
    }

    /// Build into a Chart enum variant.
    pub fn build(self) -> Chart {
        Chart::Lollipop(self)
    }

    /// Build with validation.
    pub fn try_build(self) -> Result<Chart, crate::error::ChartError> {
        if self.labels.is_empty() || self.values.is_empty() {
            return Err(crate::error::ChartError::EmptyData);
        }
        if self.labels.len() != self.values.len() {
            return Err(crate::error::ChartError::InvalidConfig(
                format!("labels ({}) and values ({}) have different lengths", self.labels.len(), self.values.len()),
            ));
        }
        Ok(Chart::Lollipop(self))
    }
}
