// SPDX-License-Identifier: MIT OR Apache-2.0
//! Waterfall chart type — financial P&L visualization.
//!
//! Sequential bars show how values add/subtract from a running total,
//! with optional connector lines between bar tops.

use crate::chart::config_builder::{
    chart_config_axis_labels, chart_config_core, chart_config_formatters, chart_config_grid,
    chart_config_h_lines, chart_config_legend, chart_config_locale, chart_config_margin,
    chart_config_subtitle_footer, chart_config_tick_rotation, chart_config_tick_steps,
};
use crate::chart::{Chart, ChartConfig};
use scry_engine::style::Color;

/// A waterfall chart — sequential bars showing running cumulative totals.
///
/// Positive values render as increase bars, negative as decrease bars.
/// An optional total bar is appended at the end.
///
/// # Examples
///
/// ```
/// use scry_chart::chart::Chart;
///
/// let chart = Chart::waterfall(
///     vec!["Revenue".into(), "COGS".into(), "OpEx".into(), "Tax".into()],
///     &[500.0, -200.0, -150.0, -50.0],
/// )
/// .title("P&L Waterfall")
/// .build();
/// ```
#[derive(Clone, Debug)]
#[must_use]
pub struct WaterfallChart {
    /// Category labels.
    pub(crate) labels: Vec<String>,
    /// Values (positive = increase, negative = decrease).
    pub(crate) values: Vec<f64>,
    /// Shared config.
    pub(crate) config: ChartConfig,
    /// Draw connector lines between bar tops (default: true).
    pub(crate) show_connectors: bool,
    /// Append a "Total" bar (default: true).
    pub(crate) show_total: bool,
    /// Color for increase bars.
    pub(crate) increase_color: Option<Color>,
    /// Color for decrease bars.
    pub(crate) decrease_color: Option<Color>,
    /// Color for total bar.
    pub(crate) total_color: Option<Color>,
    /// Whether to show numeric value labels above bars.
    pub(crate) show_values: bool,
}

impl WaterfallChart {
    /// Create a new waterfall chart.
    pub fn new(labels: Vec<String>, values: Vec<f64>) -> Self {
        Self {
            labels,
            values,
            config: ChartConfig::default(),
            show_connectors: true,
            show_total: true,
            increase_color: None,
            decrease_color: None,
            total_color: None,
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

    /// Disable connector lines between bars.
    pub fn no_connectors(mut self) -> Self {
        self.show_connectors = false;
        self
    }

    /// Disable the auto-appended total bar.
    pub fn no_total(mut self) -> Self {
        self.show_total = false;
        self
    }

    /// Set the color for increase (positive) bars.
    pub fn increase_color(mut self, color: Color) -> Self {
        self.increase_color = Some(color);
        self
    }

    /// Set the color for decrease (negative) bars.
    pub fn decrease_color(mut self, color: Color) -> Self {
        self.decrease_color = Some(color);
        self
    }

    /// Set the color for the total bar.
    pub fn total_color(mut self, color: Color) -> Self {
        self.total_color = Some(color);
        self
    }

    /// Show numeric value labels above each bar.
    pub fn show_values(mut self) -> Self {
        self.show_values = true;
        self
    }

    /// Build into a Chart enum variant.
    pub fn build(self) -> Chart {
        Chart::Waterfall(self)
    }
}
