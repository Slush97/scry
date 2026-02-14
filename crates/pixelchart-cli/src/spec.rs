//! JSON chart specification types.
//!
//! These structs define the JSON schema that AI models and users produce
//! to describe a chart. The spec is converted to a `pixelchart::Chart`
//! for rendering.

use pixelchart::chart::{
    BarChart, BoxPlot, Chart, Heatmap, Histogram, LineChart, PieChart, ScatterChart,
};
use pixelchart::data::Series;
use pixelchart::theme::Theme;
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
    /// Chart type: "line", "scatter", "bar", "histogram", "boxplot", "heatmap", "pie"
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

    /// Theme: "dark" or "light" (default: "dark")
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
        }
    }
}

// ---------------------------------------------------------------------------
// Chart data variants
// ---------------------------------------------------------------------------

/// Union of all chart data shapes. Serde will pick fields that are present.
#[derive(Debug, Deserialize)]
pub struct ChartData {
    // --- Shared ---
    /// Y values (line, scatter)
    pub y: Option<Vec<f64>>,
    /// X values (line_xy, scatter)
    pub x: Option<Vec<f64>>,

    // --- Multi-series ---
    /// Named series for multi-line charts
    pub series: Option<Vec<SeriesSpec>>,

    // --- Bar / Pie ---
    /// Category labels
    pub labels: Option<Vec<String>>,
    /// Values corresponding to labels
    pub values: Option<Vec<f64>>,

    // --- Heatmap ---
    /// 2D grid of values
    pub grid: Option<Vec<Vec<f64>>>,

    // --- BoxPlot ---
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
}

/// A named data series.
#[derive(Debug, Deserialize)]
pub struct SeriesSpec {
    pub label: String,
    pub values: Vec<f64>,
}

/// A named group of raw values (for boxplot).
#[derive(Debug, Deserialize)]
pub struct GroupSpec {
    pub label: String,
    pub values: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Conversion to pixelchart::Chart
// ---------------------------------------------------------------------------

impl ChartSpec {
    /// Convert this spec into a `pixelchart::Chart`.
    pub fn into_chart(self) -> Result<Chart, String> {
        let mut theme = match self.theme.as_deref() {
            Some("light") => Theme::light(),
            _ => Theme::dark(),
        };
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
        let values = d.values.as_ref().ok_or("bar chart requires 'values'")?;

        let mut chart = BarChart::new(labels.clone(), vec![Series::from_values(values.clone())]);

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
}
