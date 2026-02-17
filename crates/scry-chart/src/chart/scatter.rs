// SPDX-License-Identifier: MIT OR Apache-2.0
//! Scatter plot chart type.

use crate::chart::config_builder::{
    chart_config_annotations, chart_config_axis_labels, chart_config_core, chart_config_formatters,
    chart_config_grid, chart_config_h_lines, chart_config_legend, chart_config_locale,
    chart_config_ranges, chart_config_semantic_zoom, chart_config_tick_rotation,
    chart_config_tick_steps, chart_config_v_lines,
};
use crate::chart::{Chart, ChartConfig};
use crate::data::Series;

/// A scatter plot — individual data points plotted on x/y axes.
#[derive(Clone, Debug)]
#[must_use]
pub struct ScatterChart {
    /// X-axis data.
    pub(crate) x: Series,
    /// Y-axis data.
    pub(crate) y: Series,
    /// Additional y series for multi-series scatter.
    pub(crate) extra_series: Vec<(Series, Series)>,
    /// Shared config (title, labels, theme).
    pub(crate) config: ChartConfig,
    /// Whether to connect points with lines.
    pub(crate) connect: bool,
    /// Marker shape.
    pub(crate) marker: Marker,
    /// Override marker radius (uses theme default if `None`).
    pub(crate) marker_size: Option<f32>,
    /// Whether to show data value labels on points.
    pub(crate) show_values: bool,
}

/// Shape of data point markers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Marker {
    /// Filled circle (default).
    Circle,
    /// Filled square.
    Square,
    /// Diamond shape.
    Diamond,
    /// Plus/cross.
    Cross,
    /// Triangle pointing up.
    Triangle,
}

impl ScatterChart {
    /// Create a new scatter chart.
    pub fn new(x: Series, y: Series) -> Self {
        Self {
            x,
            y,
            extra_series: Vec::new(),
            config: ChartConfig::default(),
            connect: false,
            marker: Marker::Circle,
            marker_size: None,
            show_values: false,
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
    chart_config_semantic_zoom!();

    /// Add an additional data series.
    pub fn add_series(mut self, x: Series, y: Series) -> Self {
        self.extra_series.push((x, y));
        self
    }

    /// Add a named data series from raw x/y arrays.
    ///
    /// Creates [`Series`] with labels; the y-series label is used in legend rendering.
    pub fn add_named_series(mut self, label: impl Into<String>, x: &[f64], y: &[f64]) -> Self {
        let lbl: String = label.into();
        self.extra_series.push((
            Series::new(format!("{lbl}_x"), x.to_vec()),
            Series::new(lbl, y.to_vec()),
        ));
        self
    }

    /// Connect points with lines.
    pub fn connected(mut self) -> Self {
        self.connect = true;
        self
    }

    /// Set the marker shape.
    pub fn marker(mut self, marker: Marker) -> Self {
        self.marker = marker;
        self
    }

    /// Set the marker radius in pixels.
    ///
    /// If not set, uses the theme's default `series.point_radius`.
    pub fn size(mut self, radius: f32) -> Self {
        self.marker_size = Some(radius);
        self
    }

    /// Show data value labels on each point.
    pub fn show_values(mut self) -> Self {
        self.show_values = true;
        self
    }

    /// Validate inputs and build into a Chart enum variant.
    ///
    /// Returns [`ChartError`](crate::error::ChartError) if data is empty, all non-finite, or x/y lengths mismatch.
    pub fn try_build(self) -> Result<Chart, crate::error::ChartError> {
        if self.x.is_empty() && self.extra_series.is_empty() {
            return Err(crate::error::ChartError::EmptyData);
        }
        if self.x.len() != self.y.len() {
            return Err(crate::error::ChartError::MismatchedLengths {
                x_len: self.x.len(),
                y_len: self.y.len(),
            });
        }
        if self
            .x
            .values()
            .iter()
            .chain(self.y.values().iter())
            .all(|v| !v.is_finite())
        {
            return Err(crate::error::ChartError::AllNonFinite);
        }
        Ok(self.build())
    }

    /// Build into a Chart enum variant.
    pub fn build(self) -> Chart {
        Chart::Scatter(self)
    }
}
