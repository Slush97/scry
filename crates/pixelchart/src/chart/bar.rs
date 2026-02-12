//! Bar chart type.

use crate::chart::{Chart, ChartConfig, ReferenceLine};
use crate::data::Series;
use crate::theme::Theme;
use ratatui_pixelcanvas::style::Color;

/// A bar chart — categorical data shown as vertical or horizontal bars.
#[derive(Clone, Debug)]
pub struct BarChart {
    /// Category labels.
    pub(crate) labels: Vec<String>,
    /// Data series (each becomes a group of bars per category).
    pub(crate) series: Vec<Series>,
    /// Shared config.
    pub(crate) config: ChartConfig,
    /// Bar orientation.
    pub(crate) horizontal: bool,
    /// Corner radius for bars.
    pub(crate) corner_radius: f32,
    /// Gap between bars as fraction of bar width (0.0 – 1.0).
    pub(crate) bar_gap: f32,
    /// Whether to stack bars instead of grouping.
    pub(crate) stacked: bool,
}

impl BarChart {
    /// Create a new bar chart.
    pub fn new(labels: Vec<String>, series: Vec<Series>) -> Self {
        Self {
            labels,
            series,
            config: ChartConfig::default(),
            horizontal: false,
            corner_radius: 3.0,
            bar_gap: 0.2,
            stacked: false,
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

    /// Add another data series for grouped bars.
    pub fn add_series(mut self, s: Series) -> Self {
        self.series.push(s);
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

    /// Set corner radius for bar rounding.
    pub fn corner_radius(mut self, r: f32) -> Self {
        self.corner_radius = r;
        self
    }

    /// Set gap between bars (0.0 = no gap, 1.0 = no bars).
    pub fn gap(mut self, gap: f32) -> Self {
        self.bar_gap = gap.clamp(0.0, 0.9);
        self
    }

    /// Override the y-axis range.
    pub fn y_range(mut self, min: f64, max: f64) -> Self {
        self.config.y_range = Some((min, max));
        self
    }

    /// Add a horizontal reference line.
    pub fn h_line(mut self, value: f64) -> Self {
        self.config.h_lines.push(ReferenceLine::new(value));
        self
    }

    /// Add a horizontal reference line with color.
    pub fn h_line_styled(mut self, value: f64, color: Color) -> Self {
        self.config.h_lines.push(ReferenceLine::new(value).color(color));
        self
    }

    /// Hide the legend.
    pub fn no_legend(mut self) -> Self {
        self.config.show_legend = false;
        self
    }

    /// Build into a Chart enum variant.
    pub fn build(self) -> Chart {
        Chart::Bar(self)
    }
}
