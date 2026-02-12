//! Histogram chart type.

use crate::chart::{Chart, ChartConfig, ReferenceLine};
use crate::data::Series;
use crate::theme::Theme;
use ratatui_pixelcanvas::style::Color;

/// A histogram — distribution of values shown as binned bars.
#[derive(Clone, Debug)]
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

    /// Set the chart title.
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.config.title = Some(title.into());
        self
    }

    /// Set the x-axis label.
    pub fn x_label(mut self, label: impl Into<String>) -> Self {
        self.config.x_label = Some(label.into());
        self
    }

    /// Set the y-axis label.
    pub fn y_label(mut self, label: impl Into<String>) -> Self {
        self.config.y_label = Some(label.into());
        self
    }

    /// Set the visual theme.
    pub fn theme(mut self, theme: Theme) -> Self {
        self.config.theme = theme;
        self
    }

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

    /// Override the x-axis range.
    pub fn x_range(mut self, min: f64, max: f64) -> Self {
        self.config.x_range = Some((min, max));
        self
    }

    /// Override the y-axis range.
    pub fn y_range(mut self, min: f64, max: f64) -> Self {
        self.config.y_range = Some((min, max));
        self
    }

    /// Add a vertical reference line (e.g., for the mean).
    pub fn v_line(mut self, value: f64) -> Self {
        self.config.v_lines.push(ReferenceLine::new(value));
        self
    }

    /// Add a vertical reference line with color.
    pub fn v_line_styled(mut self, value: f64, color: Color) -> Self {
        self.config.v_lines.push(ReferenceLine::new(value).color(color));
        self
    }

    /// Build into a Chart enum variant.
    pub fn build(self) -> Chart {
        Chart::Histogram(self)
    }

    /// Auto-select number of bins using Sturges' rule.
    pub fn auto_bins(n: usize) -> usize {
        let bins = (1.0 + (n as f64).log2()).ceil() as usize;
        bins.max(5).min(50)
    }
}
