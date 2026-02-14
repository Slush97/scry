//! Heatmap chart type.

use crate::chart::config_builder::chart_config_core;
use crate::chart::{Chart, ChartConfig};
use ratatui_pixelcanvas::style::Color;

/// A heatmap — 2D grid of values mapped to colors.
///
/// Great for correlation matrices, confusion matrices, or any 2D data.
#[derive(Clone, Debug)]
#[must_use]
pub struct Heatmap {
    /// 2D data grid (row-major: `data[row][col]`).
    pub(crate) data: Vec<Vec<f64>>,
    /// Row labels (one per row).
    pub(crate) row_labels: Vec<String>,
    /// Column labels (one per column).
    pub(crate) col_labels: Vec<String>,
    /// Shared config.
    pub(crate) config: ChartConfig,
    /// Color for the minimum value.
    pub(crate) color_lo: Color,
    /// Color for the maximum value.
    pub(crate) color_hi: Color,
    /// Whether to show values in cells.
    pub(crate) show_values: bool,
    /// Manual value range override.
    pub(crate) value_range: Option<(f64, f64)>,
    /// Corner radius for cells.
    pub(crate) cell_radius: f32,
    /// Gap between cells in pixels.
    pub(crate) cell_gap: f32,
}

impl Heatmap {
    /// Create a new heatmap from a 2D data grid.
    pub fn new(data: Vec<Vec<f64>>) -> Self {
        let n_rows = data.len();
        let n_cols = data.first().map_or(0, |r| r.len());

        Self {
            data,
            row_labels: (0..n_rows).map(|i| i.to_string()).collect(),
            col_labels: (0..n_cols).map(|i| i.to_string()).collect(),
            config: ChartConfig::default(),
            color_lo: Color::from_rgba8(15, 20, 50, 255), // dark blue
            color_hi: Color::from_rgba8(255, 90, 80, 255), // warm red
            show_values: true,
            value_range: None,
            cell_radius: 2.0,
            cell_gap: 2.0,
        }
    }

    /// Create a heatmap for a correlation matrix (values -1 to 1).
    pub fn correlation(data: Vec<Vec<f64>>, labels: Vec<String>) -> Self {
        let mut hm = Self::new(data);
        hm.row_labels.clone_from(&labels);
        hm.col_labels = labels;
        hm.color_lo = Color::from_rgba8(60, 100, 220, 255); // blue = negative
        hm.color_hi = Color::from_rgba8(220, 60, 60, 255); // red = positive
        hm.value_range = Some((-1.0, 1.0));
        hm
    }

    // --- Generated common methods ---
    chart_config_core!();

    /// Set row labels.
    pub fn row_labels(mut self, labels: Vec<String>) -> Self {
        self.row_labels = labels;
        self
    }

    /// Set column labels.
    pub fn col_labels(mut self, labels: Vec<String>) -> Self {
        self.col_labels = labels;
        self
    }

    /// Set the color gradient endpoints.
    pub fn colors(mut self, lo: Color, hi: Color) -> Self {
        self.color_lo = lo;
        self.color_hi = hi;
        self
    }

    /// Show/hide values in cells.
    pub fn values(mut self, show: bool) -> Self {
        self.show_values = show;
        self
    }

    /// Set explicit value range for color mapping.
    pub fn range(mut self, min: f64, max: f64) -> Self {
        self.value_range = Some((min, max));
        self
    }

    /// Set corner radius for cells.
    pub fn cell_radius(mut self, r: f32) -> Self {
        self.cell_radius = r;
        self
    }

    /// Set gap between cells.
    pub fn cell_gap(mut self, gap: f32) -> Self {
        self.cell_gap = gap;
        self
    }

    /// Validate inputs and build into a Chart enum variant.
    ///
    /// Returns [`ChartError`](crate::error::ChartError) if no data rows are provided or rows have
    /// inconsistent lengths.
    pub fn try_build(self) -> Result<Chart, crate::error::ChartError> {
        if self.data.is_empty() {
            return Err(crate::error::ChartError::EmptyData);
        }
        let expected_cols = self.data[0].len();
        if self.data.iter().any(|row| row.len() != expected_cols) {
            return Err(crate::error::ChartError::JaggedGrid);
        }
        Ok(self.build())
    }

    /// Build into a Chart enum variant.
    pub fn build(self) -> Chart {
        Chart::Heatmap(self)
    }

    /// Get the data extent across all cells.
    #[must_use]
    pub fn data_extent(&self) -> (f64, f64) {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for row in &self.data {
            for &v in row {
                if v.is_finite() {
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
        }
        if lo > hi {
            (0.0, 1.0)
        } else {
            (lo, hi)
        }
    }
}

/// Linearly interpolate between two colors.
pub fn lerp_color(a: Color, b: Color, t: f64) -> Color {
    let t = t.clamp(0.0, 1.0) as f32;
    Color {
        r: a.r + (b.r - a.r) * t,
        g: a.g + (b.g - a.g) * t,
        b: a.b + (b.b - a.b) * t,
        a: a.a + (b.a - a.a) * t,
    }
}
