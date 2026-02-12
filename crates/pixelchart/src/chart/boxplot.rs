//! Box plot chart type.

use crate::chart::{Chart, ChartConfig, ReferenceLine};
use crate::data::Series;
use crate::theme::Theme;
use ratatui_pixelcanvas::style::Color;

/// A box plot — shows distribution statistics (median, quartiles, whiskers).
#[derive(Clone, Debug)]
pub struct BoxPlot {
    /// Data series (each becomes a separate box).
    pub(crate) groups: Vec<BoxGroup>,
    /// Shared config.
    pub(crate) config: ChartConfig,
    /// Whether to show individual outlier points.
    pub(crate) show_outliers: bool,
    /// Whether to show a notch for median confidence interval.
    pub(crate) notched: bool,
    /// Box width as fraction of available band (0.0 – 1.0).
    pub(crate) box_width: f32,
}

/// A single group in a box plot.
#[derive(Clone, Debug)]
pub struct BoxGroup {
    /// Label for this group.
    pub label: String,
    /// Raw data values.
    pub data: Series,
}

/// Pre-computed statistics for a box.
#[derive(Clone, Debug)]
pub struct BoxStats {
    pub min: f64,
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
    pub max: f64,
    pub whisker_lo: f64,
    pub whisker_hi: f64,
    pub outliers: Vec<f64>,
}

impl BoxStats {
    /// Compute box plot statistics from sorted data.
    pub fn from_data(values: &[f64]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }

        let mut sorted: Vec<f64> = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let q1 = percentile(&sorted, 25.0);
        let median = percentile(&sorted, 50.0);
        let q3 = percentile(&sorted, 75.0);
        let iqr = q3 - q1;

        let whisker_lo = sorted
            .iter()
            .copied()
            .find(|&v| v >= q1 - 1.5 * iqr)
            .unwrap_or(sorted[0]);
        let whisker_hi = sorted
            .iter()
            .rev()
            .copied()
            .find(|&v| v <= q3 + 1.5 * iqr)
            .unwrap_or(sorted[n - 1]);

        let outliers: Vec<f64> = sorted
            .iter()
            .copied()
            .filter(|&v| v < whisker_lo || v > whisker_hi)
            .collect();

        Some(Self {
            min: sorted[0],
            q1,
            median,
            q3,
            max: sorted[n - 1],
            whisker_lo,
            whisker_hi,
            outliers,
        })
    }
}

/// Linear interpolation percentile.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let rank = (p / 100.0) * (n - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let frac = rank - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi.min(n - 1)] * frac
}

impl BoxPlot {
    /// Create a new box plot from labeled data groups.
    pub fn new(groups: Vec<(impl Into<String>, Vec<f64>)>) -> Self {
        let groups = groups
            .into_iter()
            .map(|(label, values)| BoxGroup {
                label: label.into(),
                data: Series::from_values(values),
            })
            .collect();

        Self {
            groups,
            config: ChartConfig::default(),
            show_outliers: true,
            notched: false,
            box_width: 0.6,
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

    /// Hide outlier points.
    pub fn no_outliers(mut self) -> Self {
        self.show_outliers = false;
        self
    }

    /// Use notched box plot style.
    pub fn notched(mut self) -> Self {
        self.notched = true;
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

    /// Build into a Chart enum variant.
    pub fn build(self) -> Chart {
        Chart::BoxPlot(self)
    }
}
