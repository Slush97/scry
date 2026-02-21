// SPDX-License-Identifier: MIT OR Apache-2.0
//! Line chart type.

use crate::chart::config_builder::{
    chart_config_annotations, chart_config_axis_labels, chart_config_core, chart_config_formatters,
    chart_config_grid, chart_config_h_lines, chart_config_invert, chart_config_legend,
    chart_config_locale, chart_config_margin, chart_config_ranges, chart_config_secondary_y,
    chart_config_semantic_zoom, chart_config_subtitle_footer, chart_config_tick_rotation,
    chart_config_tick_steps, chart_config_v_lines,
};
use crate::chart::{Chart, ChartConfig};
use crate::data::{GapPolicy, Series};
use crate::spec::ChartSpec;

/// A line chart — one or more data series plotted as continuous lines.
#[derive(Clone, Debug)]
#[must_use]
#[allow(clippy::struct_excessive_bools)]
pub struct LineChart {
    /// Data series (each becomes a separate line).
    pub(crate) series: Vec<Series>,
    /// Optional explicit x values (shared across all series).
    pub(crate) x_values: Option<Vec<f64>>,
    /// Shared config.
    pub(crate) config: ChartConfig,
    /// Whether to fill the area under the line.
    pub(crate) fill_area: bool,
    /// Whether to draw data points on the line.
    pub(crate) show_points: bool,
    /// Whether to use Catmull-Rom spline interpolation.
    pub(crate) smooth: bool,
    /// Whether to render as a step (stairstep) line.
    pub(crate) step: bool,
    /// Whether to stack series (cumulative y-values).
    pub(crate) stacked: bool,
    /// Whether to apply distinct dash patterns per series.
    pub(crate) dash_lines: bool,
    /// Whether to show data value labels on points.
    pub(crate) show_values: bool,
    /// Override line width (uses theme default if `None`).
    pub(crate) line_width: Option<f32>,
    /// How to handle NaN (missing) values in data series.
    pub(crate) gap_policy: GapPolicy,
}

impl LineChart {
    /// Create a new line chart from one or more y-value series.
    pub fn new(series: Vec<Series>) -> Self {
        Self {
            series,
            x_values: None,
            config: ChartConfig::default(),
            fill_area: false,
            show_points: false,
            smooth: false,
            step: false,
            stacked: false,
            dash_lines: false,
            show_values: false,
            line_width: None,
            gap_policy: GapPolicy::Skip,
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
    chart_config_secondary_y!();
    chart_config_subtitle_footer!();
    chart_config_semantic_zoom!();
    chart_config_margin!();
    chart_config_invert!();

    /// Set explicit x values (otherwise 0, 1, 2, …).
    pub fn x_values(mut self, x: Vec<f64>) -> Self {
        self.x_values = Some(x);
        self
    }

    /// Add another data series.
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

    /// Fill the area under each line with a translucent version of its color.
    pub fn filled(mut self) -> Self {
        self.fill_area = true;
        self
    }

    /// Show data point markers on the lines.
    pub fn with_points(mut self) -> Self {
        self.show_points = true;
        self
    }

    /// Enable Catmull-Rom spline interpolation for smooth curves.
    ///
    /// This produces aesthetically pleasing curves that pass through each
    /// data point. Mutually exclusive with [`step()`](Self::step) — the last
    /// one set wins.
    pub fn smooth(mut self) -> Self {
        self.smooth = true;
        self.step = false;
        self
    }

    /// Render as a step (stairstep) line.
    ///
    /// Each segment transitions horizontally first, then vertically.
    /// Mutually exclusive with [`smooth()`](Self::smooth) — the last one set
    /// wins.
    pub fn step(mut self) -> Self {
        self.step = true;
        self.smooth = false;
        self
    }

    /// Override the line width for this chart.
    ///
    /// If not set, uses the theme's default `series.line_width`.
    pub fn line_width(mut self, width: f32) -> Self {
        self.line_width = Some(width);
        self
    }

    /// Stack series — each series' y-values accumulate on top of previous.
    ///
    /// Best combined with [`.filled()`](Self::filled) for stacked area charts.
    pub fn stacked(mut self) -> Self {
        self.stacked = true;
        self
    }

    /// Apply distinct dash patterns to each series for accessibility.
    ///
    /// Series 0 is solid, subsequent series cycle through dashed, dotted,
    /// dash-dot, and long-dash patterns.
    pub fn dash_lines(mut self) -> Self {
        self.dash_lines = true;
        self
    }

    /// Set the gap handling policy for NaN (missing) values in data series.
    ///
    /// - [`GapPolicy::Skip`] breaks the line at NaN gaps (default).
    /// - [`GapPolicy::Interpolate`] linearly interpolates across NaN gaps.
    /// - [`GapPolicy::Zero`] replaces NaN with 0.0.
    pub fn gap_policy(mut self, policy: GapPolicy) -> Self {
        self.gap_policy = policy;
        self
    }

    /// Show data value labels above each data point.
    pub fn show_values(mut self) -> Self {
        self.show_values = true;
        self
    }

    /// Validate inputs and build into a Chart enum variant.
    ///
    /// Returns [`ChartError`](crate::error::ChartError) if no series are provided.
    pub fn try_build(self) -> Result<Chart, crate::error::ChartError> {
        if self.series.is_empty() {
            return Err(crate::error::ChartError::EmptyData);
        }
        Ok(self.build())
    }

    /// Build into a Chart.
    #[must_use]
    pub fn build(self) -> Chart {
        Box::new(self) as Chart
    }
}

impl ChartSpec for LineChart {
    fn render(&self, w: u32, h: u32) -> crate::layout::RenderedChart {
        crate::layout::line::render_line(self, w, h)
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
        let x_max = self.x_values.as_ref().map_or_else(
            || (ys.len().saturating_sub(1)) as f64,
            |xv| xv.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        );
        let x_min = self.x_values.as_ref().map_or(0.0, |xv| xv.iter().copied().fold(f64::INFINITY, f64::min));
        let y_min = ys.iter().copied().fold(f64::INFINITY, f64::min);
        let y_max = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Some((x_min, x_max, y_min, y_max))
    }
    fn clone_boxed(&self) -> Box<dyn ChartSpec> { Box::new(self.clone()) }
}
