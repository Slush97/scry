//! Chart type definitions and builders.
//!
//! Each chart type has a builder struct for fluent configuration.

pub(crate) mod bar;
pub(crate) mod boxplot;
pub(crate) mod bubble;
pub(crate) mod candlestick;
pub(crate) mod config_builder;
pub mod extent;
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

use std::sync::Arc;

use crate::annotation::Annotation;
use crate::axis::LabelRotation;
use crate::data::Series;
use crate::formatter::{LocaleConfig, TickFormatter};
use crate::legend::LegendConfig;
use crate::margin::Margin;
use crate::theme::Theme;
use scry_engine::style::Color;

// ---------------------------------------------------------------------------
// Chart enum
// ---------------------------------------------------------------------------

/// A fully-configured chart ready for rendering.
#[derive(Clone, Debug)]
#[must_use]
#[non_exhaustive]
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
    /// Pie / donut chart.
    Pie(PieChart),
    /// Candlestick / OHLC chart.
    Candlestick(CandlestickChart),
    /// Radar / spider chart.
    Radar(RadarChart),
    /// Bubble chart (scatter + size dimension).
    Bubble(BubbleChart),
    /// Violin plot (mirrored KDE curves).
    Violin(ViolinPlot),
    /// Sparkline (minimal inline chart, no chrome).
    Sparkline(Sparkline),
    /// Waterfall chart (financial P&L — running cumulative bars).
    Waterfall(WaterfallChart),
    /// Funnel chart (conversion pipeline).
    Funnel(FunnelChart),
    /// Gauge chart (KPI speedometer arc).
    Gauge(GaugeChart),
    /// Lollipop chart (dot plot — thin stem + circle marker).
    Lollipop(LollipopChart),
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

    /// Create a line chart from explicit x and y values.
    ///
    /// Unlike [`line()`](Self::line), this allows non-uniform x spacing.
    pub fn line_xy(x: &[f64], y: &[f64]) -> LineChart {
        LineChart::new(vec![Series::from_values(y.to_vec())]).x_values(x.to_vec())
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

    /// Get a mutable reference to the chart's shared configuration.
    ///
    /// Useful for injecting zoom viewport ranges before rendering.
    pub fn config_mut(&mut self) -> &mut ChartConfig {
        match self {
            Self::Scatter(c) => &mut c.config,
            Self::Line(c) => &mut c.config,
            Self::Bar(c) => &mut c.config,
            Self::Histogram(c) => &mut c.config,
            Self::BoxPlot(c) => &mut c.config,
            Self::Heatmap(c) => &mut c.config,
            Self::Pie(c) => &mut c.config,
            Self::Candlestick(c) => &mut c.config,
            Self::Radar(c) => &mut c.config,
            Self::Bubble(c) => &mut c.config,
            Self::Violin(c) => &mut c.config,
            Self::Sparkline(c) => &mut c.config,
            Self::Waterfall(c) => &mut c.config,
            Self::Funnel(c) => &mut c.config,
            Self::Gauge(c) => &mut c.config,
            Self::Lollipop(c) => &mut c.config,
        }
    }
}

// ---------------------------------------------------------------------------
// Common chart config
// ---------------------------------------------------------------------------

/// Configuration shared across all chart types.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(clippy::struct_excessive_bools)]
#[non_exhaustive]
pub struct ChartConfig {
    /// Chart title.
    pub title: Option<String>,
    /// Chart subtitle (rendered below the title in smaller text).
    pub subtitle: Option<String>,
    /// Chart footer (rendered at the bottom edge).
    pub footer: Option<String>,
    /// X-axis label.
    pub x_label: Option<String>,
    /// Y-axis label.
    pub y_label: Option<String>,
    /// Visual theme.
    pub theme: Theme,
    /// Whether to display the legend.
    pub show_legend: bool,
    /// Legend configuration (position, title, orientation, etc.).
    pub legend: LegendConfig,
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
    /// Rotation for X-axis tick labels.
    pub x_tick_rotation: LabelRotation,
    /// Custom tick formatter for the X axis.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub x_tick_formatter: Option<Arc<dyn TickFormatter>>,
    /// Custom tick formatter for the Y axis.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub y_tick_formatter: Option<Arc<dyn TickFormatter>>,
    /// Fixed tick step for the X axis (overrides adaptive generation).
    pub x_tick_step: Option<f64>,
    /// Fixed tick step for the Y axis (overrides adaptive generation).
    pub y_tick_step: Option<f64>,
    /// Locale configuration for number formatting.
    /// When set, the layout engine wraps the default formatter
    /// with locale-aware post-processing.
    pub locale: Option<LocaleConfig>,
    /// Extra margins / padding around the plot area (pixels).
    pub margin: Option<Margin>,
    /// Whether to invert (reverse) the X axis.
    pub x_inverted: bool,
    /// Whether to invert (reverse) the Y axis.
    pub y_inverted: bool,
    /// Secondary Y-axis label (rendered on the right side).
    pub secondary_y_label: Option<String>,
    /// Manual secondary Y-axis domain override.
    pub secondary_y_range: Option<(f64, f64)>,
    /// Custom tick formatter for the secondary Y axis.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub secondary_y_formatter: Option<Arc<dyn TickFormatter>>,
    /// Series indices to plot against the secondary Y axis.
    pub secondary_series_indices: Vec<usize>,
    /// Export DPI (dots per inch). Default: 144.
    ///
    /// The export functions scale the output pixel dimensions by `dpi / 144`.
    /// Set to 288 for 2× (Retina) resolution, 72 for lower-res, etc.
    pub dpi: u32,
}

impl Clone for ChartConfig {
    fn clone(&self) -> Self {
        Self {
            title: self.title.clone(),
            subtitle: self.subtitle.clone(),
            footer: self.footer.clone(),
            x_label: self.x_label.clone(),
            y_label: self.y_label.clone(),
            theme: self.theme.clone(),
            show_legend: self.show_legend,
            legend: self.legend.clone(),
            x_range: self.x_range,
            y_range: self.y_range,
            h_lines: self.h_lines.clone(),
            v_lines: self.v_lines.clone(),
            annotations: self.annotations.clone(),
            show_trend: self.show_trend,
            x_tick_rotation: self.x_tick_rotation,
            x_tick_formatter: self.x_tick_formatter.clone(),
            y_tick_formatter: self.y_tick_formatter.clone(),
            x_tick_step: self.x_tick_step,
            y_tick_step: self.y_tick_step,
            locale: self.locale.clone(),
            margin: self.margin.clone(),
            x_inverted: self.x_inverted,
            y_inverted: self.y_inverted,
            secondary_y_label: self.secondary_y_label.clone(),
            secondary_y_range: self.secondary_y_range,
            secondary_y_formatter: self.secondary_y_formatter.clone(),
            secondary_series_indices: self.secondary_series_indices.clone(),
            dpi: self.dpi,
        }
    }
}

impl std::fmt::Debug for ChartConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChartConfig")
            .field("title", &self.title)
            .field("subtitle", &self.subtitle)
            .field("footer", &self.footer)
            .field("theme", &"..")
            .field("show_legend", &self.show_legend)
            .field("x_tick_rotation", &self.x_tick_rotation)
            .field(
                "x_tick_formatter",
                &self.x_tick_formatter.as_ref().map(|_| ".."),
            )
            .field(
                "y_tick_formatter",
                &self.y_tick_formatter.as_ref().map(|_| ".."),
            )
            .field("x_tick_step", &self.x_tick_step)
            .field("y_tick_step", &self.y_tick_step)
            .field("locale", &self.locale)
            .field("margin", &self.margin)
            .field("x_inverted", &self.x_inverted)
            .field("y_inverted", &self.y_inverted)
            .field("secondary_y_label", &self.secondary_y_label)
            .field("dpi", &self.dpi)
            .finish()
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

impl Default for ChartConfig {
    fn default() -> Self {
        Self {
            title: None,
            subtitle: None,
            footer: None,
            x_label: None,
            y_label: None,
            theme: Theme::default(),
            show_legend: true,
            legend: LegendConfig::default(),
            x_range: None,
            y_range: None,
            h_lines: Vec::new(),
            v_lines: Vec::new(),
            annotations: Vec::new(),
            show_trend: false,
            x_tick_rotation: LabelRotation::Horizontal,
            x_tick_formatter: None,
            y_tick_formatter: None,
            x_tick_step: None,
            y_tick_step: None,
            locale: None,
            margin: None,
            x_inverted: false,
            y_inverted: false,
            secondary_y_label: None,
            secondary_y_range: None,
            secondary_y_formatter: None,
            secondary_series_indices: Vec::new(),
            dpi: 144,
        }
    }
}
