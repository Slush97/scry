//! Chart type definitions and builders.
//!
//! Each chart type has a builder struct for fluent configuration.

pub(crate) mod bar;
pub(crate) mod boxplot;
pub(crate) mod heatmap;
pub(crate) mod histogram;
pub(crate) mod line;
pub(crate) mod scatter;

pub use bar::BarChart;
pub use boxplot::BoxPlot;
pub use heatmap::Heatmap;
pub use histogram::Histogram;
pub use line::LineChart;
pub use scatter::ScatterChart;

use crate::annotation::Annotation;
use crate::data::Series;
use crate::theme::Theme;
use ratatui_pixelcanvas::style::Color;

// ---------------------------------------------------------------------------
// Chart enum
// ---------------------------------------------------------------------------

/// A fully-configured chart ready for rendering.
#[derive(Clone, Debug)]
pub enum Chart {
    /// Scatter plot.
    Scatter(ScatterChart),
    /// Line chart.
    Line(LineChart),
    /// Bar chart.
    Bar(BarChart),
    /// Histogram.
    Histogram(Histogram),
    /// Box plot.
    BoxPlot(BoxPlot),
    /// Heatmap.
    Heatmap(Heatmap),
}

impl Chart {
    // --- One-liner constructors ---

    /// Create a scatter chart from x and y data.
    pub fn scatter(x: &[f64], y: &[f64]) -> ScatterChart {
        ScatterChart::new(
            Series::from_values(x.to_vec()),
            Series::from_values(y.to_vec()),
        )
    }

    /// Create a line chart from y values (x = 0, 1, 2, …).
    pub fn line(y: &[f64]) -> LineChart {
        LineChart::new(vec![Series::from_values(y.to_vec())])
    }

    /// Create a bar chart from labels and values.
    pub fn bar(labels: Vec<String>, values: &[f64]) -> BarChart {
        BarChart::new(labels, vec![Series::from_values(values.to_vec())])
    }

    /// Create a histogram from raw data.
    pub fn histogram(values: &[f64]) -> Histogram {
        Histogram::new(Series::from_values(values.to_vec()))
    }

    /// Create a box plot from labeled data groups.
    pub fn boxplot(groups: Vec<(impl Into<String>, Vec<f64>)>) -> BoxPlot {
        BoxPlot::new(groups)
    }

    /// Create a heatmap from a 2D grid.
    pub fn heatmap(data: Vec<Vec<f64>>) -> Heatmap {
        Heatmap::new(data)
    }
}

// ---------------------------------------------------------------------------
// Common chart config
// ---------------------------------------------------------------------------

/// Configuration shared across all chart types.
#[derive(Clone, Debug)]
pub struct ChartConfig {
    /// Chart title.
    pub title: Option<String>,
    /// X-axis label.
    pub x_label: Option<String>,
    /// Y-axis label.
    pub y_label: Option<String>,
    /// Visual theme.
    pub theme: Theme,
    /// Whether to display the legend.
    pub show_legend: bool,
    /// Manual x-axis domain override (min, max).
    pub x_range: Option<(f64, f64)>,
    /// Manual y-axis domain override (min, max).
    pub y_range: Option<(f64, f64)>,
    /// Horizontal reference lines.
    pub h_lines: Vec<ReferenceLine>,
    /// Vertical reference lines.
    pub v_lines: Vec<ReferenceLine>,
    /// Data-coordinate annotations.
    pub annotations: Vec<Annotation>,
    /// Whether to show a trend line (linear regression).
    pub show_trend: bool,
}

/// A horizontal or vertical reference line.
#[derive(Clone, Debug)]
pub struct ReferenceLine {
    /// The data-space position of the line.
    pub value: f64,
    /// Line color.
    pub color: Color,
    /// Optional label.
    pub label: Option<String>,
    /// Line width.
    pub width: f32,
    /// Dash pattern (solid if empty).
    pub dashed: bool,
}

impl ReferenceLine {
    /// Create a new reference line at the given value.
    pub fn new(value: f64) -> Self {
        Self {
            value,
            color: Color::from_rgba8(255, 255, 255, 140),
            label: None,
            width: 1.0,
            dashed: true,
        }
    }

    /// Set the line color.
    pub fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Set an optional label.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

impl Default for ChartConfig {
    fn default() -> Self {
        Self {
            title: None,
            x_label: None,
            y_label: None,
            theme: Theme::default(),
            show_legend: true,
            x_range: None,
            y_range: None,
            h_lines: Vec::new(),
            v_lines: Vec::new(),
            annotations: Vec::new(),
            show_trend: false,
        }
    }
}
