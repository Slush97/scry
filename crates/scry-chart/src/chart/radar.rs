//! Radar / spider chart type.

use crate::chart::config_builder::{
    chart_config_core, chart_config_margin, chart_config_subtitle_footer,
};
use crate::chart::{Chart, ChartConfig};

/// A radar (spider) chart — multi-axis polygon comparison.
///
/// Each axis represents a dimension, and data series are plotted as
/// polygons connecting values on each axis.
#[derive(Clone, Debug)]
#[must_use]
pub struct RadarChart {
    /// Axis labels (one per spoke).
    pub(crate) axes: Vec<String>,
    /// Data series: each is (label, values) where values.len() == axes.len().
    pub(crate) series: Vec<(String, Vec<f64>)>,
    /// Shared config.
    pub(crate) config: ChartConfig,
    /// Whether to fill the polygon area.
    pub(crate) fill: bool,
    /// Whether to show data point markers.
    pub(crate) show_points: bool,
}

impl RadarChart {
    /// Create a new radar chart with axis labels.
    ///
    /// # Example
    /// ```ignore
    /// use scry_chart::chart::RadarChart;
    ///
    /// let chart = RadarChart::new(vec!["Speed", "Power", "Defense", "Magic", "HP"])
    ///     .add_series("Warrior", &[8.0, 9.0, 7.0, 2.0, 8.0])
    ///     .add_series("Mage", &[3.0, 4.0, 3.0, 10.0, 5.0]);
    /// ```
    pub fn new(axes: Vec<impl Into<String>>) -> Self {
        Self {
            axes: axes.into_iter().map(Into::into).collect(),
            series: Vec::new(),
            config: ChartConfig::default(),
            fill: true,
            show_points: true,
        }
    }

    chart_config_core!();
    chart_config_subtitle_footer!();
    chart_config_margin!();

    /// Add a named data series.
    ///
    /// Values should match the number of axes. Extra values are ignored;
    /// missing values default to 0.
    pub fn add_series(mut self, label: impl Into<String>, values: &[f64]) -> Self {
        self.series.push((label.into(), values.to_vec()));
        self
    }

    /// Disable polygon fill (outline only).
    pub fn no_fill(mut self) -> Self {
        self.fill = false;
        self
    }

    /// Hide data point markers on vertices.
    pub fn hide_points(mut self) -> Self {
        self.show_points = false;
        self
    }

    /// Validate inputs and build into a Chart enum variant.
    ///
    /// Returns [`ChartError`](crate::error::ChartError) if no series or axes are provided.
    pub fn try_build(self) -> Result<Chart, crate::error::ChartError> {
        if self.axes.is_empty() || self.series.is_empty() {
            return Err(crate::error::ChartError::EmptyData);
        }
        Ok(self.build())
    }

    /// Build into a Chart enum variant.
    pub fn build(self) -> Chart {
        Chart::Radar(self)
    }
}
