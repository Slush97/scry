// SPDX-License-Identifier: MIT OR Apache-2.0
//! JSON chart specification types.
//!
//! These structs define the JSON schema that AI models and users produce
//! to describe a chart. The spec is converted to a `scry_chart::Chart`
//! for rendering.

use scry_chart::chart::{
    BarChart, BoxPlot, BubbleChart, CandlestickChart, Chart, FunnelChart, GaugeChart, Heatmap,
    Histogram, LineChart, LollipopChart, OhlcEntry, PieChart, RadarChart, ScatterChart, Sparkline,
    ViolinPlot, WaterfallChart,
};
use scry_chart::data::Series;
use scry_chart::theme::Theme;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Top-level spec
// ---------------------------------------------------------------------------

/// A complete chart specification in JSON.
///
/// Example:
/// ```json
/// {
///   "type": "line",
///   "data": {"y": [1, 4, 2, 8, 5]},
///   "title": "My Chart",
///   "theme": "dark"
/// }
/// ```
#[derive(Debug, Deserialize)]
pub struct ChartSpec {
    /// Chart type
    #[serde(rename = "type")]
    pub chart_type: ChartType,

    /// Chart data — shape depends on chart_type
    pub data: ChartData,

    /// Optional chart title
    pub title: Option<String>,

    /// Optional x-axis label
    pub x_label: Option<String>,

    /// Optional y-axis label
    pub y_label: Option<String>,

    /// Theme name (default: "dark")
    pub theme: Option<String>,

    /// Image width in pixels (default: 800)
    pub width: Option<u32>,

    /// Image height in pixels (default: 500)
    pub height: Option<u32>,

    /// Manual X-axis domain override (min, max)
    #[serde(default)]
    pub x_range: Option<(f64, f64)>,

    /// Manual Y-axis domain override (min, max)
    #[serde(default)]
    pub y_range: Option<(f64, f64)>,

    /// Number of histogram bins
    #[serde(default)]
    pub bins: Option<usize>,

    /// Whether to use a transparent background
    #[serde(default)]
    pub transparent_bg: bool,
}

/// Supported chart types.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum ChartType {
    Line,
    Scatter,
    Bar,
    Histogram,
    #[serde(alias = "box", alias = "box_plot")]
    Boxplot,
    Heatmap,
    Pie,
    Radar,
    Candlestick,
    Bubble,
    Violin,
    Sparkline,
    Waterfall,
    Funnel,
    Gauge,
    Lollipop,
}

impl std::fmt::Display for ChartType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Line => write!(f, "line"),
            Self::Scatter => write!(f, "scatter"),
            Self::Bar => write!(f, "bar"),
            Self::Histogram => write!(f, "histogram"),
            Self::Boxplot => write!(f, "boxplot"),
            Self::Heatmap => write!(f, "heatmap"),
            Self::Pie => write!(f, "pie"),
            Self::Radar => write!(f, "radar"),
            Self::Candlestick => write!(f, "candlestick"),
            Self::Bubble => write!(f, "bubble"),
            Self::Violin => write!(f, "violin"),
            Self::Sparkline => write!(f, "sparkline"),
            Self::Waterfall => write!(f, "waterfall"),
            Self::Funnel => write!(f, "funnel"),
            Self::Gauge => write!(f, "gauge"),
            Self::Lollipop => write!(f, "lollipop"),
        }
    }
}

// ---------------------------------------------------------------------------
// Chart data variants
// ---------------------------------------------------------------------------

/// Union of all chart data shapes. Serde will pick fields that are present.
#[derive(Debug, Default, Deserialize)]
pub struct ChartData {
    // --- Shared ---
    /// Y values (line, scatter)
    pub y: Option<Vec<f64>>,
    /// X values (line_xy, scatter)
    pub x: Option<Vec<f64>>,

    // --- Multi-series ---
    /// Named series for multi-line charts
    pub series: Option<Vec<SeriesSpec>>,

    // --- Bar / Pie / Waterfall / Funnel / Lollipop ---
    /// Category labels
    pub labels: Option<Vec<String>>,
    /// Values corresponding to labels
    pub values: Option<Vec<f64>>,

    // --- Heatmap ---
    /// 2D grid of values
    pub grid: Option<Vec<Vec<f64>>>,

    // --- BoxPlot / Violin ---
    /// Named groups with raw values
    pub groups: Option<Vec<GroupSpec>>,

    // --- Line options ---
    /// Fill area under line
    pub filled: Option<bool>,
    /// Show data points
    pub points: Option<bool>,
    /// Use smooth interpolation
    pub smooth: Option<bool>,
    /// Use step interpolation
    pub step: Option<bool>,

    // --- Candlestick ---
    /// OHLC data entries
    pub ohlc: Option<Vec<OhlcSpec>>,

    // --- Radar ---
    /// Axis labels for radar chart
    pub axes: Option<Vec<String>>,
    /// Named radar series (each with label + values)
    pub radar_series: Option<Vec<RadarSeriesSpec>>,

    // --- Bubble ---
    /// Size values for bubble chart
    pub sizes: Option<Vec<f64>>,

    // --- Gauge ---
    /// Single value for gauge chart
    pub value: Option<f64>,
    /// Gauge minimum
    pub min: Option<f64>,
    /// Gauge maximum
    pub max: Option<f64>,
}

/// A named data series.
#[derive(Debug, Deserialize)]
pub struct SeriesSpec {
    pub label: String,
    pub values: Vec<f64>,
}

/// A named group of raw values (for boxplot/violin).
#[derive(Debug, Deserialize)]
pub struct GroupSpec {
    pub label: String,
    pub values: Vec<f64>,
}

/// An OHLC (Open-High-Low-Close) data point for candlestick charts.
#[derive(Debug, Deserialize)]
pub struct OhlcSpec {
    /// X-axis value (index or timestamp). When absent, the array index is used.
    pub x: Option<f64>,
    /// Opening price
    pub open: f64,
    /// Highest price
    pub high: f64,
    /// Lowest price
    pub low: f64,
    /// Closing price
    pub close: f64,
}

/// A named radar chart series.
#[derive(Debug, Deserialize)]
pub struct RadarSeriesSpec {
    pub label: String,
    pub values: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Conversion to scry_chart::Chart
// ---------------------------------------------------------------------------

impl ChartSpec {
    /// Convert this spec into a `scry_chart::Chart`.
    pub fn into_chart(self) -> Result<Chart, String> {
        let mut theme = resolve_theme(self.theme.as_deref());
        if self.transparent_bg {
            theme.background = theme.background.with_alpha(0.0);
        }

        let chart = match self.chart_type {
            ChartType::Line => self.build_line(theme)?,
            ChartType::Scatter => self.build_scatter(theme)?,
            ChartType::Bar => self.build_bar(theme)?,
            ChartType::Histogram => self.build_histogram(theme)?,
            ChartType::Boxplot => self.build_boxplot(theme)?,
            ChartType::Heatmap => self.build_heatmap(theme)?,
            ChartType::Pie => self.build_pie(theme)?,
            ChartType::Radar => self.build_radar(theme)?,
            ChartType::Candlestick => self.build_candlestick(theme)?,
            ChartType::Bubble => self.build_bubble(theme)?,
            ChartType::Violin => self.build_violin(theme)?,
            ChartType::Sparkline => self.build_sparkline(theme)?,
            ChartType::Waterfall => self.build_waterfall(theme)?,
            ChartType::Funnel => self.build_funnel(theme)?,
            ChartType::Gauge => self.build_gauge(theme)?,
            ChartType::Lollipop => self.build_lollipop(theme)?,
        };

        Ok(chart)
    }

    fn build_line(&self, theme: Theme) -> Result<Chart, String> {
        let d = &self.data;

        let mut chart = if let Some(ref series) = d.series {
            // Multi-series
            let series_vec: Vec<Series> = series
                .iter()
                .map(|s| Series::new(&s.label, s.values.clone()))
                .collect();
            LineChart::new(series_vec)
        } else if let Some(ref y) = d.y {
            LineChart::new(vec![Series::from_values(y.clone())])
        } else {
            return Err("line chart requires 'y' or 'series' data".into());
        };

        // Apply x values if present
        if let Some(ref x) = d.x {
            chart = chart.x_values(x.clone());
        }

        // Line-specific options
        if d.filled.unwrap_or(false) {
            chart = chart.filled();
        }
        if d.points.unwrap_or(false) {
            chart = chart.with_points();
        }
        if d.smooth.unwrap_or(false) {
            chart = chart.smooth();
        }
        if d.step.unwrap_or(false) {
            chart = chart.step();
        }

        // Common config
        chart = chart.theme(theme);
        if let Some(ref t) = self.title {
            chart = chart.title(t);
        }
        if let Some(ref l) = self.x_label {
            chart = chart.x_label(l);
        }
        if let Some(ref l) = self.y_label {
            chart = chart.y_label(l);
        }
        if let Some((min, max)) = self.x_range {
            chart = chart.x_range(min, max);
        }
        if let Some((min, max)) = self.y_range {
            chart = chart.y_range(min, max);
        }

        Ok(chart.build())
    }

    fn build_scatter(&self, theme: Theme) -> Result<Chart, String> {
        let d = &self.data;
        let x = d.x.as_ref().ok_or("scatter chart requires 'x' data")?;
        let y = d.y.as_ref().ok_or("scatter chart requires 'y' data")?;

        let mut chart = ScatterChart::new(
            Series::from_values(x.clone()),
            Series::from_values(y.clone()),
        );

        chart = chart.theme(theme);
        if let Some(ref t) = self.title {
            chart = chart.title(t);
        }
        if let Some(ref l) = self.x_label {
            chart = chart.x_label(l);
        }
        if let Some(ref l) = self.y_label {
            chart = chart.y_label(l);
        }
        if let Some((min, max)) = self.x_range {
            chart = chart.x_range(min, max);
        }
        if let Some((min, max)) = self.y_range {
            chart = chart.y_range(min, max);
        }

        Ok(chart.build())
    }

    fn build_bar(&self, theme: Theme) -> Result<Chart, String> {
        let d = &self.data;
        let labels = d.labels.as_ref().ok_or("bar chart requires 'labels'")?;

        // Support multi-series bar charts: prefer data.series over data.values
        let series_vec: Vec<Series> = if let Some(ref series) = d.series {
            series
                .iter()
                .map(|s| Series::new(&s.label, s.values.clone()))
                .collect()
        } else {
            let values = d
                .values
                .as_ref()
                .ok_or("bar chart requires 'values' or 'series'")?;
            vec![Series::from_values(values.clone())]
        };

        let mut chart = BarChart::new(labels.clone(), series_vec);

        chart = chart.theme(theme);
        if let Some(ref t) = self.title {
            chart = chart.title(t);
        }
        if let Some(ref l) = self.x_label {
            chart = chart.x_label(l);
        }
        if let Some(ref l) = self.y_label {
            chart = chart.y_label(l);
        }

        if let Some((min, max)) = self.y_range {
            chart = chart.y_range(min, max);
        }

        Ok(chart.build())
    }

    fn build_histogram(&self, theme: Theme) -> Result<Chart, String> {
        let d = &self.data;
        let values = d
            .values
            .as_ref()
            .or(d.y.as_ref())
            .ok_or("histogram requires 'values' or 'y' data")?;

        let mut chart = Histogram::new(Series::from_values(values.clone()));

        if let Some(n) = self.bins {
            chart = chart.bins(n);
        }

        chart = chart.theme(theme);
        if let Some(ref t) = self.title {
            chart = chart.title(t);
        }
        if let Some(ref l) = self.x_label {
            chart = chart.x_label(l);
        }
        if let Some(ref l) = self.y_label {
            chart = chart.y_label(l);
        }
        if let Some((min, max)) = self.x_range {
            chart = chart.x_range(min, max);
        }
        if let Some((min, max)) = self.y_range {
            chart = chart.y_range(min, max);
        }

        Ok(chart.build())
    }

    fn build_boxplot(&self, theme: Theme) -> Result<Chart, String> {
        let d = &self.data;
        let groups = d.groups.as_ref().ok_or("boxplot requires 'groups' data")?;

        let group_data: Vec<(String, Vec<f64>)> = groups
            .iter()
            .map(|g| (g.label.clone(), g.values.clone()))
            .collect();

        let mut chart = BoxPlot::new(group_data);

        chart = chart.theme(theme);
        if let Some(ref t) = self.title {
            chart = chart.title(t);
        }
        if let Some(ref l) = self.x_label {
            chart = chart.x_label(l);
        }
        if let Some(ref l) = self.y_label {
            chart = chart.y_label(l);
        }

        if let Some((min, max)) = self.y_range {
            chart = chart.y_range(min, max);
        }

        Ok(chart.build())
    }

    fn build_heatmap(&self, theme: Theme) -> Result<Chart, String> {
        let d = &self.data;
        let grid = d.grid.as_ref().ok_or("heatmap requires 'grid' data")?;

        let mut chart = Heatmap::new(grid.clone());

        chart = chart.theme(theme);
        if let Some(ref t) = self.title {
            chart = chart.title(t);
        }

        Ok(chart.build())
    }

    fn build_pie(&self, theme: Theme) -> Result<Chart, String> {
        let d = &self.data;
        let labels = d.labels.as_ref().ok_or("pie chart requires 'labels'")?;
        let values = d.values.as_ref().ok_or("pie chart requires 'values'")?;

        let mut chart = PieChart::new(labels.clone(), values.clone());

        chart = chart.theme(theme);
        if let Some(ref t) = self.title {
            chart = chart.title(t);
        }

        Ok(chart.build())
    }

    fn build_radar(&self, theme: Theme) -> Result<Chart, String> {
        let d = &self.data;
        let axes = d.axes.as_ref().ok_or("radar chart requires 'axes' data")?;
        let radar_series = d
            .radar_series
            .as_ref()
            .ok_or("radar chart requires 'radar_series' data")?;

        let mut chart = RadarChart::new(axes.clone());

        for s in radar_series {
            chart = chart.add_series(&s.label, &s.values);
        }

        chart = chart.theme(theme);
        if let Some(ref t) = self.title {
            chart = chart.title(t);
        }

        Ok(chart.build())
    }

    fn build_candlestick(&self, theme: Theme) -> Result<Chart, String> {
        let d = &self.data;
        let ohlc = d
            .ohlc
            .as_ref()
            .ok_or("candlestick chart requires 'ohlc' data")?;

        let entries: Vec<OhlcEntry> = ohlc
            .iter()
            .enumerate()
            .map(|(i, o)| {
                let x = o.x.unwrap_or(i as f64);
                OhlcEntry::new(x, o.open, o.high, o.low, o.close)
            })
            .collect();

        let mut chart = CandlestickChart::new(entries);

        chart = chart.theme(theme);
        if let Some(ref t) = self.title {
            chart = chart.title(t);
        }
        if let Some(ref l) = self.x_label {
            chart = chart.x_label(l);
        }
        if let Some(ref l) = self.y_label {
            chart = chart.y_label(l);
        }

        Ok(chart.build())
    }

    fn build_bubble(&self, theme: Theme) -> Result<Chart, String> {
        let d = &self.data;
        let x = d.x.as_ref().ok_or("bubble chart requires 'x' data")?;
        let y = d.y.as_ref().ok_or("bubble chart requires 'y' data")?;
        let sizes = d
            .sizes
            .as_ref()
            .ok_or("bubble chart requires 'sizes' data")?;

        let mut chart = BubbleChart::new(
            Series::from_values(x.clone()),
            Series::from_values(y.clone()),
            sizes.clone(),
        );

        chart = chart.theme(theme);
        if let Some(ref t) = self.title {
            chart = chart.title(t);
        }
        if let Some(ref l) = self.x_label {
            chart = chart.x_label(l);
        }
        if let Some(ref l) = self.y_label {
            chart = chart.y_label(l);
        }
        if let Some((min, max)) = self.x_range {
            chart = chart.x_range(min, max);
        }
        if let Some((min, max)) = self.y_range {
            chart = chart.y_range(min, max);
        }

        Ok(chart.build())
    }

    fn build_violin(&self, theme: Theme) -> Result<Chart, String> {
        let d = &self.data;
        let groups = d
            .groups
            .as_ref()
            .ok_or("violin plot requires 'groups' data")?;

        let group_data: Vec<(String, Vec<f64>)> = groups
            .iter()
            .map(|g| (g.label.clone(), g.values.clone()))
            .collect();

        let mut chart = ViolinPlot::new(group_data);

        chart = chart.theme(theme);
        if let Some(ref t) = self.title {
            chart = chart.title(t);
        }
        if let Some(ref l) = self.x_label {
            chart = chart.x_label(l);
        }
        if let Some(ref l) = self.y_label {
            chart = chart.y_label(l);
        }

        Ok(chart.build())
    }

    fn build_sparkline(&self, theme: Theme) -> Result<Chart, String> {
        let d = &self.data;
        let values = d
            .values
            .as_ref()
            .or(d.y.as_ref())
            .ok_or("sparkline requires 'values' or 'y' data")?;

        let _ = theme; // sparklines are chrome-free, theme is unused
        let chart = Sparkline::new(values.clone());

        Ok(chart.build())
    }

    fn build_waterfall(&self, theme: Theme) -> Result<Chart, String> {
        let d = &self.data;
        let labels = d
            .labels
            .as_ref()
            .ok_or("waterfall chart requires 'labels'")?;
        let values = d
            .values
            .as_ref()
            .ok_or("waterfall chart requires 'values'")?;

        let mut chart = WaterfallChart::new(labels.clone(), values.clone());

        chart = chart.theme(theme);
        if let Some(ref t) = self.title {
            chart = chart.title(t);
        }
        if let Some(ref l) = self.x_label {
            chart = chart.x_label(l);
        }
        if let Some(ref l) = self.y_label {
            chart = chart.y_label(l);
        }

        Ok(chart.build())
    }

    fn build_funnel(&self, theme: Theme) -> Result<Chart, String> {
        let d = &self.data;
        let labels = d.labels.as_ref().ok_or("funnel chart requires 'labels'")?;
        let values = d.values.as_ref().ok_or("funnel chart requires 'values'")?;

        let mut chart = FunnelChart::new(labels.clone(), values.clone());

        chart = chart.theme(theme);
        if let Some(ref t) = self.title {
            chart = chart.title(t);
        }

        Ok(chart.build())
    }

    fn build_gauge(&self, theme: Theme) -> Result<Chart, String> {
        let d = &self.data;
        let value = d.value.ok_or("gauge chart requires 'value' data")?;

        let mut chart = GaugeChart::new(value);

        if let Some(min) = d.min {
            if let Some(max) = d.max {
                chart = chart.range(min, max);
            }
        }

        chart = chart.theme(theme);
        if let Some(ref t) = self.title {
            chart = chart.title(t);
        }

        Ok(chart.build())
    }

    fn build_lollipop(&self, theme: Theme) -> Result<Chart, String> {
        let d = &self.data;
        let labels = d
            .labels
            .as_ref()
            .ok_or("lollipop chart requires 'labels'")?;
        let values = d
            .values
            .as_ref()
            .ok_or("lollipop chart requires 'values'")?;

        let mut chart = LollipopChart::new(labels.clone(), values.clone());

        chart = chart.theme(theme);
        if let Some(ref t) = self.title {
            chart = chart.title(t);
        }
        if let Some(ref l) = self.x_label {
            chart = chart.x_label(l);
        }
        if let Some(ref l) = self.y_label {
            chart = chart.y_label(l);
        }

        Ok(chart.build())
    }
}

// ---------------------------------------------------------------------------
// Theme resolver
// ---------------------------------------------------------------------------

/// Resolve a theme name to a `Theme` instance.
pub fn resolve_theme(name: Option<&str>) -> Theme {
    match name {
        Some("light") => Theme::light(),
        Some("pastel") => Theme::pastel(),
        Some("ocean") => Theme::ocean(),
        Some("forest") => Theme::forest(),
        Some("colorblind") => Theme::colorblind(),
        _ => Theme::dark(),
    }
}
