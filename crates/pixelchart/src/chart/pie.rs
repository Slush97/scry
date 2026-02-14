//! Pie / donut chart type.

use crate::chart::config_builder::chart_config_core;
use crate::chart::{Chart, ChartConfig};

/// A pie or donut chart — proportional data shown as arc segments.
#[derive(Clone, Debug)]
#[must_use]
pub struct PieChart {
    /// Slice labels.
    pub(crate) labels: Vec<String>,
    /// Slice values (will be normalized to percentages).
    pub(crate) values: Vec<f64>,
    /// Shared config.
    pub(crate) config: ChartConfig,
    /// Inner radius ratio (0.0 = pie, 0.3–0.6 = donut).
    pub(crate) donut_ratio: f32,
    /// Whether to show percentage labels on slices.
    pub(crate) show_percentages: bool,
    /// Starting angle in radians (default: -π/2, i.e. 12 o'clock).
    pub(crate) start_angle: f32,
}

impl PieChart {
    /// Create a new pie chart from labels and values.
    pub fn new(labels: Vec<String>, values: Vec<f64>) -> Self {
        Self {
            labels,
            values,
            config: ChartConfig::default(),
            donut_ratio: 0.0,
            show_percentages: true,
            start_angle: -std::f32::consts::FRAC_PI_2,
        }
    }

    chart_config_core!();

    /// Convert to a donut chart with the given inner-to-outer radius ratio.
    /// Values between 0.3 and 0.7 work best.
    pub fn donut(mut self, ratio: f32) -> Self {
        self.donut_ratio = ratio.clamp(0.0, 0.85);
        self
    }

    /// Hide percentage labels on slices.
    pub fn hide_percentages(mut self) -> Self {
        self.show_percentages = false;
        self
    }

    /// Set the starting angle (in degrees, 0 = right, 90 = top).
    pub fn start_angle_degrees(mut self, degrees: f32) -> Self {
        self.start_angle = degrees.to_radians() - std::f32::consts::FRAC_PI_2;
        self
    }

    /// Validate inputs and build into a Chart enum variant.
    ///
    /// Returns [`ChartError`](crate::error::ChartError) if no values are provided.
    pub fn try_build(self) -> Result<Chart, crate::error::ChartError> {
        if self.values.is_empty() {
            return Err(crate::error::ChartError::EmptyData);
        }
        if self.values.iter().all(|v| !v.is_finite() || *v <= 0.0) {
            return Err(crate::error::ChartError::AllNonFinite);
        }
        Ok(self.build())
    }

    /// Build into a Chart enum variant.
    pub fn build(self) -> Chart {
        Chart::Pie(self)
    }
}
