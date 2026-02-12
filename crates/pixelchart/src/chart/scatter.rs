//! Scatter plot chart type.

use crate::annotation::Annotation;
use crate::chart::{Chart, ChartConfig, ReferenceLine};
use crate::data::Series;
use crate::theme::Theme;
use ratatui_pixelcanvas::style::Color;

/// A scatter plot — individual data points plotted on x/y axes.
#[derive(Clone, Debug)]
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
}

/// Shape of data point markers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

    /// Add an additional data series.
    pub fn add_series(mut self, x: Series, y: Series) -> Self {
        self.extra_series.push((x, y));
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

    /// Add a vertical reference line.
    pub fn v_line(mut self, value: f64) -> Self {
        self.config.v_lines.push(ReferenceLine::new(value));
        self
    }

    /// Add a vertical reference line with color.
    pub fn v_line_styled(mut self, value: f64, color: Color) -> Self {
        self.config.v_lines.push(ReferenceLine::new(value).color(color));
        self
    }

    /// Hide the legend.
    pub fn no_legend(mut self) -> Self {
        self.config.show_legend = false;
        self
    }

    /// Add an annotation at the given data coordinates.
    pub fn annotate(mut self, x: f64, y: f64, text: impl Into<String>) -> Self {
        self.config.annotations.push(Annotation::new(x, y, text));
        self
    }

    /// Show a linear regression trend line.
    pub fn trend_line(mut self) -> Self {
        self.config.show_trend = true;
        self
    }

    /// Build into a Chart enum variant.
    pub fn build(self) -> Chart {
        Chart::Scatter(self)
    }
}
