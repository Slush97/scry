// SPDX-License-Identifier: MIT OR Apache-2.0
//! Bar chart type.

use crate::chart::config_builder::{
    chart_config_axis_labels, chart_config_core, chart_config_formatters, chart_config_grid,
    chart_config_h_lines, chart_config_invert, chart_config_legend, chart_config_locale,
    chart_config_margin, chart_config_ranges, chart_config_subtitle_footer,
    chart_config_tick_rotation, chart_config_tick_steps,
};
use crate::chart::{Chart, ChartConfig};
use crate::data::Series;
use crate::spec::ChartSpec;

/// A bar chart — categorical data shown as vertical or horizontal bars.
#[derive(Clone, Debug)]
#[must_use]
pub struct BarChart {
    /// Category labels.
    pub(crate) labels: Vec<String>,
    /// Data series (each becomes a group of bars per category).
    pub(crate) series: Vec<Series>,
    /// Shared config.
    pub(crate) config: ChartConfig,
    /// Bar orientation.
    pub(crate) horizontal: bool,
    /// Corner radius for bars (overrides theme if set).
    pub(crate) corner_radius: Option<f32>,
    /// Gap between bars as fraction of bar width (0.0 – 1.0).
    pub(crate) bar_gap: f32,
    /// Whether to stack bars instead of grouping.
    pub(crate) stacked: bool,
    /// Whether to show numeric value labels above/beside bars.
    pub(crate) show_values: bool,
}

impl BarChart {
    /// Create a new bar chart.
    pub fn new(labels: Vec<String>, series: Vec<Series>) -> Self {
        Self {
            labels,
            series,
            config: ChartConfig::default(),
            horizontal: false,
            corner_radius: None,
            bar_gap: 0.2,
            stacked: false,
            show_values: false,
        }
    }

    // --- Generated common methods ---
    chart_config_core!();
    chart_config_axis_labels!();
    chart_config_ranges!(y);
    chart_config_h_lines!();
    chart_config_legend!();
    chart_config_grid!();
    chart_config_tick_rotation!();
    chart_config_formatters!();
    chart_config_locale!();
    chart_config_tick_steps!();
    chart_config_subtitle_footer!();
    chart_config_margin!();
    chart_config_invert!();

    /// Add another data series for grouped bars.
    pub fn add_series(mut self, s: Series) -> Self {
        self.series.push(s);
        self
    }

    /// Add a named data series from raw values.
    ///
    /// This is a convenience method that creates a [`Series`] with a label.
    /// The label is used in legend rendering.
    pub fn add_named_series(mut self, label: impl Into<String>, values: &[f64]) -> Self {
        self.series.push(Series::new(label, values.to_vec()));
        self
    }

    /// Set labels for all existing series at once.
    ///
    /// If more labels than series are provided, extras are ignored.
    /// If fewer labels than series, remaining series keep their current labels.
    pub fn series_labels(mut self, labels: &[&str]) -> Self {
        for (i, label) in labels.iter().enumerate() {
            if i < self.series.len() {
                let vals = self.series[i].values().to_vec();
                self.series[i] = Series::new(*label, vals);
            }
        }
        self
    }

    /// Use horizontal bars instead of vertical.
    pub fn horizontal(mut self) -> Self {
        self.horizontal = true;
        self
    }

    /// Stack bars instead of grouping them side by side.
    pub fn stacked(mut self) -> Self {
        self.stacked = true;
        self
    }

    /// Set corner radius for bar rounding (overrides theme default).
    pub fn corner_radius(mut self, r: f32) -> Self {
        self.corner_radius = Some(r);
        self
    }

    /// Set gap between bars (0.0 = no gap, 1.0 = no bars).
    pub fn gap(mut self, gap: f32) -> Self {
        self.bar_gap = gap.clamp(0.0, 0.9);
        self
    }

    /// Show numeric value labels above each bar.
    pub fn show_values(mut self) -> Self {
        self.show_values = true;
        self
    }

    /// Validate inputs and build into a Chart enum variant.
    ///
    /// Returns [`ChartError`](crate::error::ChartError) if labels are empty or no series are provided.
    pub fn try_build(self) -> Result<Chart, crate::error::ChartError> {
        if self.labels.is_empty() {
            return Err(crate::error::ChartError::EmptyData);
        }
        if self.series.is_empty() {
            return Err(crate::error::ChartError::EmptyData);
        }
        Ok(self.build())
    }

    /// Build into a Chart.
    pub fn build(self) -> Chart {
        Box::new(self) as Chart
    }
}

impl ChartSpec for BarChart {
    fn render(&self, w: u32, h: u32) -> crate::layout::RenderedChart {
        crate::layout::bar::render_bar(self, w, h)
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
        let ys: Vec<f64> = self.series.iter().flat_map(|s| s.values().iter().copied()).collect();
        if ys.is_empty() { return None; }
        let n = self.labels.len();
        let y_min = ys.iter().copied().fold(0.0_f64, f64::min);
        let y_max = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Some((0.0, (n.saturating_sub(1)) as f64, y_min, y_max))
    }
    fn clone_boxed(&self) -> Box<dyn ChartSpec> { Box::new(self.clone()) }
}
