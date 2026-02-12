//! Line chart type.

use crate::annotation::Annotation;
use crate::chart::{Chart, ChartConfig, ReferenceLine};
use crate::data::Series;
use crate::theme::Theme;
use ratatui_pixelcanvas::style::Color;

/// A line chart — one or more data series plotted as continuous lines.
#[derive(Clone, Debug)]
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
        }
    }

    /// Set explicit x values (otherwise 0, 1, 2, …).
    pub fn x_values(mut self, x: Vec<f64>) -> Self {
        self.x_values = Some(x);
        self
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

    /// Add another data series.
    pub fn add_series(mut self, s: Series) -> Self {
        self.series.push(s);
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
        Chart::Line(self)
    }
}
