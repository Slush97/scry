// SPDX-License-Identifier: MIT OR Apache-2.0
//! Funnel chart type — conversion pipeline visualization.
//!
//! Stages are rendered as centered horizontal bars of decreasing width,
//! stacked vertically from top to bottom.

use crate::chart::config_builder::{
    chart_config_core, chart_config_legend, chart_config_locale, chart_config_margin,
    chart_config_subtitle_footer,
};
use crate::chart::{Chart, ChartConfig};
use crate::spec::ChartSpec;

/// A funnel chart — conversion pipeline with stages of decreasing width.
///
/// # Examples
///
/// ```
/// use scry_chart::chart::Charts;
///
/// let chart = Charts::funnel(
///     vec!["Visitors".into(), "Signups".into(), "Trials".into(), "Paid".into()],
///     &[10000.0, 5000.0, 2000.0, 800.0],
/// )
/// .title("Conversion Funnel")
/// .build();
/// ```
#[derive(Clone, Debug)]
#[must_use]
pub struct FunnelChart {
    /// Stage labels.
    pub(crate) labels: Vec<String>,
    /// Stage values.
    pub(crate) values: Vec<f64>,
    /// Shared config.
    pub(crate) config: ChartConfig,
    /// Show percentage of first stage (default: true).
    pub(crate) show_percentages: bool,
    /// Show absolute values (default: true).
    pub(crate) show_values: bool,
    /// Gap between stages in pixels (default: 2.0).
    pub(crate) gap: f32,
}

impl FunnelChart {
    /// Create a new funnel chart.
    pub fn new(labels: Vec<String>, values: Vec<f64>) -> Self {
        Self {
            labels,
            values,
            config: ChartConfig::default(),
            show_percentages: true,
            show_values: true,
            gap: 4.0,
        }
    }

    // --- Generated common methods ---
    chart_config_core!();
    chart_config_legend!();
    chart_config_locale!();
    chart_config_subtitle_footer!();
    chart_config_margin!();

    /// Hide percentage labels.
    pub fn no_percentages(mut self) -> Self {
        self.show_percentages = false;
        self
    }

    /// Show absolute value labels beside each stage.
    pub fn show_values(mut self) -> Self {
        self.show_values = true;
        self
    }

    /// Set the gap between stages (pixels).
    pub fn gap(mut self, gap: f32) -> Self {
        self.gap = gap.max(0.0);
        self
    }

    /// Build into a Chart.
    pub fn build(self) -> Chart {
        Box::new(self) as Chart
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
        Ok(self.build())
    }
}

impl ChartSpec for FunnelChart {
    fn render(&self, w: u32, h: u32) -> crate::layout::RenderedChart {
        crate::layout::funnel::render_funnel(self, w, h)
    }
    fn config(&self) -> Option<&ChartConfig> { Some(&self.config) }
    fn config_mut(&mut self) -> Option<&mut ChartConfig> { Some(&mut self.config) }
    fn clone_boxed(&self) -> Box<dyn ChartSpec> { Box::new(self.clone()) }
}
