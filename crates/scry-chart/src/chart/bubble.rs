// SPDX-License-Identifier: MIT OR Apache-2.0
//! Bubble chart type — scatter plot with a size dimension.
//!
//! Each data point is plotted at (x, y) with a circle whose radius
//! is proportional to a third `size` value.

use crate::chart::config_builder::{
    chart_config_annotations, chart_config_axis_labels, chart_config_core, chart_config_formatters,
    chart_config_grid, chart_config_h_lines, chart_config_legend, chart_config_locale,
    chart_config_ranges, chart_config_semantic_zoom, chart_config_tick_rotation,
    chart_config_tick_steps, chart_config_v_lines,
};
use crate::chart::{Chart, ChartConfig};
use crate::chart::scatter::Marker;
use crate::data::Series;

/// A bubble chart — scatter plot where each point has a variable size.
///
/// The size dimension is mapped to circle radius, linearly interpolated
/// between `min_radius` and `max_radius`.
///
/// # Examples
///
/// ```
/// use scry_chart::chart::Chart;
/// use scry_chart::data::Series;
///
/// let chart = Chart::bubble(
///     &[1.0, 2.0, 3.0],
///     &[10.0, 20.0, 15.0],
///     &[5.0, 20.0, 10.0],
/// )
/// .title("Market Analysis")
/// .x_label("Revenue ($M)")
/// .y_label("Growth (%)")
/// .build();
/// ```
#[derive(Clone, Debug)]
#[must_use]
pub struct BubbleChart {
    /// X-axis data.
    pub(crate) x: Series,
    /// Y-axis data.
    pub(crate) y: Series,
    /// Size values (mapped to bubble radius).
    pub(crate) sizes: Vec<f64>,
    /// Additional series: (x, y, sizes).
    pub(crate) extra_series: Vec<(Series, Series, Vec<f64>)>,
    /// Shared config.
    pub(crate) config: ChartConfig,
    /// Marker shape (default: Circle).
    pub(crate) marker: Marker,
    /// Minimum bubble radius in pixels.
    pub(crate) min_radius: f32,
    /// Maximum bubble radius in pixels.
    pub(crate) max_radius: f32,
    /// Whether to show data value labels on points.
    pub(crate) show_values: bool,
    /// Bubble opacity (0.0–1.0).
    pub(crate) opacity: f32,
}

impl BubbleChart {
    /// Create a new bubble chart.
    pub fn new(x: Series, y: Series, sizes: Vec<f64>) -> Self {
        Self {
            x,
            y,
            sizes,
            extra_series: Vec::new(),
            config: ChartConfig::default(),
            marker: Marker::Circle,
            min_radius: 3.0,
            max_radius: 25.0,
            show_values: false,
            opacity: 0.7,
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
    pub fn add_series(mut self, x: Series, y: Series, sizes: Vec<f64>) -> Self {
        self.extra_series.push((x, y, sizes));
        self
    }

    /// Add a named data series from raw arrays.
    pub fn add_named_series(
        mut self,
        label: impl Into<String>,
        x: &[f64],
        y: &[f64],
        sizes: &[f64],
    ) -> Self {
        let lbl: String = label.into();
        self.extra_series.push((
            Series::new(format!("{lbl}_x"), x.to_vec()),
            Series::new(lbl, y.to_vec()),
            sizes.to_vec(),
        ));
        self
    }

    /// Set the marker shape.
    pub fn marker(mut self, marker: Marker) -> Self {
        self.marker = marker;
        self
    }

    /// Set the bubble size range (min and max radius in pixels).
    ///
    /// The smallest data value maps to `min_r`, the largest to `max_r`.
    pub fn size_range(mut self, min_r: f32, max_r: f32) -> Self {
        self.min_radius = min_r.max(1.0);
        self.max_radius = max_r.max(min_r + 1.0);
        self
    }

    /// Set bubble opacity (0.0–1.0).
    pub fn opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// Show data value labels on each bubble.
    pub fn show_values(mut self) -> Self {
        self.show_values = true;
        self
    }

    /// Validate inputs and build into a Chart enum variant.
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
        Ok(self.build())
    }

    /// Build into a Chart enum variant.
    pub fn build(self) -> Chart {
        Chart::Bubble(self)
    }
}
