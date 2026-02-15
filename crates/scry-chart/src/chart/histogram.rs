//! Histogram chart type.

use crate::chart::config_builder::{
    chart_config_axis_labels, chart_config_core, chart_config_formatters, chart_config_grid,
    chart_config_h_lines, chart_config_invert, chart_config_legend, chart_config_locale,
    chart_config_margin, chart_config_ranges, chart_config_subtitle_footer,
    chart_config_tick_rotation, chart_config_tick_steps, chart_config_v_lines,
};
use crate::chart::{Chart, ChartConfig};
use crate::data::Series;

/// A histogram — distribution of values shown as binned bars.
#[derive(Clone, Debug)]
#[must_use]
pub struct Histogram {
    /// Raw data to bin.
    pub(crate) data: Series,
    /// Additional data series for overlaid histograms.
    pub(crate) extra: Vec<Series>,
    /// Shared config.
    pub(crate) config: ChartConfig,
    /// Number of bins (auto-selected if None).
    pub(crate) bins: Option<usize>,
    /// Whether to normalize to density (area = 1).
    pub(crate) density: bool,
    /// Bar opacity (0.0–1.0).
    pub(crate) opacity: f32,
}

impl Histogram {
    /// Create a new histogram from raw data.
    pub fn new(data: Series) -> Self {
        Self {
            data,
            extra: Vec::new(),
            config: ChartConfig::default(),
            bins: None,
            density: false,
            opacity: 0.8,
        }
    }

    // --- Generated common methods ---
    chart_config_core!();
    chart_config_axis_labels!();
    chart_config_ranges!(xy);
    chart_config_h_lines!();
    chart_config_v_lines!();
    chart_config_legend!();
    chart_config_grid!();
    chart_config_tick_rotation!();
    chart_config_formatters!();
    chart_config_locale!();
    chart_config_tick_steps!();
    chart_config_subtitle_footer!();
    chart_config_margin!();
    chart_config_invert!();

    /// Set the number of bins (default: auto via Sturges' rule).
    pub fn bins(mut self, n: usize) -> Self {
        self.bins = Some(n.max(1));
        self
    }

    /// Normalize to density (total area = 1).
    pub fn density(mut self) -> Self {
        self.density = true;
        self
    }

    /// Set bar opacity.
    pub fn opacity(mut self, a: f32) -> Self {
        self.opacity = a.clamp(0.0, 1.0);
        self
    }

    /// Add another data series for overlaid histograms.
    pub fn add_series(mut self, s: Series) -> Self {
        self.extra.push(s);
        self
    }

    /// Validate inputs and build into a Chart enum variant.
    ///
    /// Returns [`ChartError`](crate::error::ChartError) if data is empty.
    pub fn try_build(self) -> Result<Chart, crate::error::ChartError> {
        if self.data.is_empty() {
            return Err(crate::error::ChartError::EmptyData);
        }
        Ok(self.build())
    }

    /// Build into a Chart enum variant.
    pub fn build(self) -> Chart {
        Chart::Histogram(self)
    }

    /// Auto-select number of bins using Sturges' rule.
    #[must_use]
    pub fn auto_bins(n: usize) -> usize {
        let bins = (1.0 + (n as f64).log2()).ceil() as usize;
        bins.max(5).min(50)
    }
}
