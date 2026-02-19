// SPDX-License-Identifier: MIT OR Apache-2.0
//! Chart type definitions and builders.
//!
//! Each chart type has a builder struct for fluent configuration.
//! [`Chart`] is a type alias for `Box<dyn ChartSpec>`.

pub(crate) mod bar;
pub(crate) mod boxplot;
pub(crate) mod bubble;
pub(crate) mod candlestick;
pub(crate) mod config_builder;
pub(crate) mod contour;
pub(crate) mod funnel;
pub(crate) mod gauge;
pub(crate) mod heatmap;
pub(crate) mod histogram;
pub(crate) mod line;
pub(crate) mod lollipop;
pub(crate) mod pie;
pub(crate) mod radar;
pub(crate) mod scatter;
pub(crate) mod sparkline;
pub(crate) mod violin;
pub(crate) mod waterfall;

pub use bar::BarChart;
pub use boxplot::BoxPlot;
pub use bubble::BubbleChart;
pub use candlestick::{CandlestickChart, OhlcEntry};
pub use contour::ContourChart;
pub use funnel::FunnelChart;
pub use gauge::GaugeChart;
pub use heatmap::Heatmap;
pub use histogram::Histogram;
pub use line::LineChart;
pub use lollipop::LollipopChart;
pub use pie::PieChart;
pub use radar::RadarChart;
pub use scatter::ScatterChart;
pub use sparkline::{Sparkline, SparklineKind};
pub use violin::ViolinPlot;
pub use waterfall::WaterfallChart;

use crate::config::{
    AxisRangeConfig, DataLabelConfig, ExportConfig, OverlayConfig, SecondaryAxisConfig, TickConfig,
    TitleConfig,
};
use crate::legend::LegendConfig;
use crate::margin::Margin;
use crate::spec::ChartSpec;
use crate::theme::Theme;
use scry_engine::style::Color;

// ---------------------------------------------------------------------------
// Chart type alias
// ---------------------------------------------------------------------------

/// A fully-configured chart ready for rendering.
///
/// This is a type alias for `Box<dyn ChartSpec>`. All built-in chart types
/// implement [`ChartSpec`] and their `build()` methods return a `Chart`.
pub type Chart = Box<dyn ChartSpec>;

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

/// Namespace for one-liner chart constructors.
///
/// Each method returns a chart builder (e.g. `LineChart`, `ScatterChart`).
/// Call `.build()` on the builder to get a `Chart`.
pub struct Charts;

impl Charts {
    /// Create a scatter chart from x and y data.
    pub fn scatter(x: &[f64], y: &[f64]) -> ScatterChart {
        use crate::data::Series;
        ScatterChart::new(
            Series::from_values(x.to_vec()),
            Series::from_values(y.to_vec()),
        )
    }

    /// Create a line chart from y values (x = 0, 1, 2, …).
    pub fn line(y: &[f64]) -> LineChart {
        use crate::data::Series;
        LineChart::new(vec![Series::from_values(y.to_vec())])
    }

    /// Create a line chart from explicit x and y values.
    pub fn line_xy(x: &[f64], y: &[f64]) -> LineChart {
        use crate::data::Series;
        LineChart::new(vec![Series::from_values(y.to_vec())]).x_values(x.to_vec())
    }

    /// Create a bar chart from labels and values.
    pub fn bar(labels: Vec<String>, values: &[f64]) -> BarChart {
        use crate::data::Series;
        BarChart::new(labels, vec![Series::from_values(values.to_vec())])
    }

    /// Create a histogram from raw data.
    pub fn histogram(values: &[f64]) -> Histogram {
        use crate::data::Series;
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

    /// Create a pie chart from labels and values.
    pub fn pie(labels: Vec<String>, values: &[f64]) -> PieChart {
        PieChart::new(labels, values.to_vec())
    }

    /// Create a candlestick chart from OHLC data.
    pub fn candlestick(data: Vec<OhlcEntry>) -> CandlestickChart {
        CandlestickChart::new(data)
    }

    /// Create a radar chart with the given axis labels.
    pub fn radar(axes: Vec<impl Into<String>>) -> RadarChart {
        RadarChart::new(axes)
    }

    /// Create a bubble chart from x, y, and size data.
    pub fn bubble(x: &[f64], y: &[f64], sizes: &[f64]) -> BubbleChart {
        use crate::data::Series;
        BubbleChart::new(
            Series::from_values(x.to_vec()),
            Series::from_values(y.to_vec()),
            sizes.to_vec(),
        )
    }

    /// Create a violin plot from labeled data groups.
    pub fn violin(groups: Vec<(impl Into<String>, Vec<f64>)>) -> ViolinPlot {
        ViolinPlot::new(groups)
    }

    /// Create a line sparkline from data values.
    pub fn sparkline(values: &[f64]) -> Sparkline {
        Sparkline::new(values.to_vec())
    }

    /// Create a waterfall chart from labels and incremental values.
    pub fn waterfall(labels: Vec<String>, values: &[f64]) -> WaterfallChart {
        WaterfallChart::new(labels, values.to_vec())
    }

    /// Create a funnel chart from labels and stage values.
    pub fn funnel(labels: Vec<String>, values: &[f64]) -> FunnelChart {
        FunnelChart::new(labels, values.to_vec())
    }

    /// Create a gauge chart from a single value.
    pub fn gauge(value: f64) -> GaugeChart {
        GaugeChart::new(value)
    }

    /// Create a lollipop chart from labels and values.
    pub fn lollipop(labels: Vec<String>, values: &[f64]) -> LollipopChart {
        LollipopChart::new(labels, values.to_vec())
    }

    /// Create a contour chart from a 2D grid of values.
    pub fn contour(data: Vec<Vec<f64>>) -> ContourChart {
        ContourChart::new(data)
    }

    /// Create an area chart (filled + smooth line chart).
    pub fn area(y: &[f64]) -> LineChart {
        use crate::data::Series;
        LineChart::new(vec![Series::from_values(y.to_vec())])
            .filled()
            .smooth()
    }
}

// ---------------------------------------------------------------------------
// Common chart config (decomposed into sub-structs)
// ---------------------------------------------------------------------------

/// Configuration shared across all chart types.
///
/// Fields are organized into sub-structs by concern. For backward
/// compatibility, the top-level convenience accessors remain available.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct ChartConfig {
    /// Title, subtitle, footer, and axis label text.
    pub titles: TitleConfig,
    /// Visual theme.
    pub theme: Theme,
    /// Whether to display the legend.
    pub show_legend: bool,
    /// Legend configuration (position, title, orientation, etc.).
    pub legend: LegendConfig,
    /// Axis range overrides and axis direction.
    pub axes: AxisRangeConfig,
    /// Tick label formatting and stepping.
    pub ticks: TickConfig,
    /// Reference lines, annotations, and trend line.
    pub overlays: OverlayConfig,
    /// Secondary (dual) Y-axis.
    pub secondary: SecondaryAxisConfig,
    /// Export settings (DPI).
    pub export: ExportConfig,
    /// Data value labels on bars, points, etc.
    pub data_labels: DataLabelConfig,
    /// Extra margins / padding around the plot area (pixels).
    pub margin: Option<Margin>,
}

impl Clone for ChartConfig {
    fn clone(&self) -> Self {
        Self {
            titles: self.titles.clone(),
            theme: self.theme.clone(),
            show_legend: self.show_legend,
            legend: self.legend.clone(),
            axes: self.axes.clone(),
            ticks: self.ticks.clone(),
            overlays: self.overlays.clone(),
            secondary: self.secondary.clone(),
            export: self.export.clone(),
            data_labels: self.data_labels.clone(),
            margin: self.margin.clone(),
        }
    }
}

impl std::fmt::Debug for ChartConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChartConfig")
            .field("titles", &self.titles)
            .field("theme", &"..")
            .field("show_legend", &self.show_legend)
            .field("axes", &self.axes)
            .field("ticks", &self.ticks)
            .field("overlays", &self.overlays)
            .field("secondary", &self.secondary)
            .field("export", &self.export)
            .field("data_labels", &self.data_labels)
            .field("margin", &self.margin)
            .finish()
    }
}

impl Default for ChartConfig {
    fn default() -> Self {
        Self {
            titles: TitleConfig::default(),
            theme: Theme::default(),
            show_legend: true,
            legend: LegendConfig::default(),
            axes: AxisRangeConfig::default(),
            ticks: TickConfig::default(),
            overlays: OverlayConfig::default(),
            secondary: SecondaryAxisConfig::default(),
            export: ExportConfig::default(),
            data_labels: DataLabelConfig::default(),
            margin: None,
        }
    }
}

/// A horizontal or vertical reference line.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[must_use]
#[non_exhaustive]
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
