// SPDX-License-Identifier: MIT OR Apache-2.0
//! Violin plot chart type.
//!
//! Shows data distribution as mirrored KDE (kernel density estimation)
//! curves, optionally with an inner box-and-whisker overlay.

use crate::chart::config_builder::{
    chart_config_axis_labels, chart_config_core, chart_config_formatters, chart_config_grid,
    chart_config_legend, chart_config_locale, chart_config_tick_rotation,
};
use crate::chart::{Chart, ChartConfig};
use crate::spec::ChartSpec;

/// A violin plot — one or more groups rendered as mirrored KDE curves.
///
/// # Examples
///
/// ```
/// use scry_chart::chart::Charts;
///
/// let chart = Charts::violin(vec![
///     ("Control", vec![2.0, 3.1, 2.8, 3.5, 2.9]),
///     ("Treatment", vec![4.0, 4.5, 3.9, 5.1, 4.2]),
/// ])
/// .inner_box()
/// .title("Distribution Comparison")
/// .build();
/// ```
#[derive(Clone, Debug)]
#[must_use]
pub struct ViolinPlot {
    /// Labeled data groups.
    pub(crate) groups: Vec<(String, Vec<f64>)>,
    /// Shared config.
    pub(crate) config: ChartConfig,
    /// Whether to draw an inner box-and-whisker.
    pub(crate) show_inner_box: bool,
    /// KDE bandwidth (None = auto via Silverman's rule).
    pub(crate) bandwidth: Option<f64>,
    /// Whether to draw horizontally.
    pub(crate) horizontal: bool,
}

impl ViolinPlot {
    /// Create a new violin plot from labeled data groups.
    pub fn new(groups: Vec<(impl Into<String>, Vec<f64>)>) -> Self {
        Self {
            groups: groups.into_iter().map(|(l, v)| (l.into(), v)).collect(),
            config: ChartConfig::default(),
            show_inner_box: false,
            bandwidth: None,
            horizontal: false,
        }
    }

    // --- Generated common methods ---
    chart_config_core!();
    chart_config_axis_labels!();
    chart_config_legend!();
    chart_config_grid!();
    chart_config_tick_rotation!();
    chart_config_formatters!();
    chart_config_locale!();

    /// Show an inner box-and-whisker overlay on each violin.
    pub fn inner_box(mut self) -> Self {
        self.show_inner_box = true;
        self
    }

    /// Set the KDE bandwidth manually.
    ///
    /// If not set, uses Silverman's rule of thumb.
    pub fn bandwidth(mut self, bw: f64) -> Self {
        self.bandwidth = Some(bw);
        self
    }

    /// Render horizontally (groups on Y axis, density on X axis).
    pub fn horizontal(mut self) -> Self {
        self.horizontal = true;
        self
    }

    /// Build into a Chart.
    pub fn build(self) -> Chart {
        Box::new(self) as Chart
    }

    /// Build with validation.
    pub fn try_build(self) -> Result<Chart, crate::error::ChartError> {
        if self.groups.is_empty() {
            return Err(crate::error::ChartError::EmptyData);
        }
        if self.groups.iter().all(|(_, v)| v.is_empty()) {
            return Err(crate::error::ChartError::EmptyData);
        }
        Ok(self.build())
    }
}

impl ChartSpec for ViolinPlot {
    fn render(&self, w: u32, h: u32) -> crate::layout::RenderedChart {
        crate::layout::violin::render_violin(self, w, h)
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
    fn data_extent(&self) -> Option<(f64, f64, f64, f64)> { None }
    fn clone_boxed(&self) -> Box<dyn ChartSpec> { Box::new(self.clone()) }
}
