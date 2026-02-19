// SPDX-License-Identifier: MIT OR Apache-2.0
//! Box plot chart type.

use crate::chart::config_builder::{
    chart_config_axis_labels, chart_config_core, chart_config_formatters, chart_config_grid,
    chart_config_h_lines, chart_config_invert, chart_config_locale, chart_config_margin,
    chart_config_ranges, chart_config_subtitle_footer, chart_config_tick_rotation,
    chart_config_tick_steps,
};
use crate::chart::{Chart, ChartConfig};
use crate::data::Series;
use crate::spec::ChartSpec;

/// A box plot — shows distribution statistics (median, quartiles, whiskers).
#[derive(Clone, Debug)]
#[must_use]
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

        let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        if sorted.is_empty() {
            return None;
        }
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

    // --- Generated common methods ---
    chart_config_core!();
    chart_config_axis_labels!();
    chart_config_ranges!(y);
    chart_config_h_lines!();
    chart_config_grid!();
    chart_config_tick_rotation!();
    chart_config_formatters!();
    chart_config_locale!();
    chart_config_tick_steps!();
    chart_config_subtitle_footer!();
    chart_config_margin!();
    chart_config_invert!();

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

    /// Validate inputs and build into a Chart enum variant.
    ///
    /// Returns [`ChartError`](crate::error::ChartError) if no groups are provided.
    pub fn try_build(self) -> Result<Chart, crate::error::ChartError> {
        if self.groups.is_empty() {
            return Err(crate::error::ChartError::EmptyData);
        }
        Ok(self.build())
    }

    /// Build into a Chart.
    pub fn build(self) -> Chart {
        Box::new(self) as Chart
    }
}

impl ChartSpec for BoxPlot {
    fn render(&self, w: u32, h: u32) -> crate::layout::RenderedChart {
        crate::layout::boxplot::render_boxplot(self, w, h)
    }
    fn render_with_viewport(&self, w: u32, h: u32, vp: Option<(f64, f64, f64, f64)>) -> crate::layout::RenderedChart {
        if let Some((x0, x1, y0, y1)) = vp {
            let mut c = self.clone();
            c.config.axes.x_range = Some((x0, x1));
            c.config.axes.y_range = Some((y0, y1));
            c.render(w, h)
        } else {
            self.render(w, h)
        }
    }
    fn config(&self) -> Option<&ChartConfig> { Some(&self.config) }
    fn config_mut(&mut self) -> Option<&mut ChartConfig> { Some(&mut self.config) }
    fn data_extent(&self) -> Option<(f64, f64, f64, f64)> { None }
    fn clone_boxed(&self) -> Box<dyn ChartSpec> { Box::new(self.clone()) }
}
