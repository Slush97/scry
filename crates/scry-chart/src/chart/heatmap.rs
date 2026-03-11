// SPDX-License-Identifier: MIT OR Apache-2.0
//! Heatmap chart type.

use crate::chart::config_builder::{chart_config_margin, chart_config_subtitle_footer};
use crate::chart::{Chart, ChartConfig};
use crate::colormap::Colormap;
use crate::spec::ChartSpec;
use scry_engine::style::Color;
use std::sync::Arc;

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
    /// Optional colormap (overrides color_lo/color_hi interpolation).
    pub(crate) colormap: Option<Arc<dyn Colormap>>,
    /// Whether the user explicitly set colors via `.colors()`.
    pub(crate) colors_explicit: bool,
}

impl Heatmap {
    /// Create a new heatmap from a 2D data grid.
    pub fn new(data: Vec<Vec<f64>>) -> Self {
        let n_rows = data.len();
        let n_cols = data.first().map_or(0, |r| r.len());

        let config = ChartConfig::default();
        let (color_lo, color_hi) = derive_heatmap_colors(&config.theme);
        Self {
            data,
            row_labels: (0..n_rows).map(|i| i.to_string()).collect(),
            col_labels: (0..n_cols).map(|i| i.to_string()).collect(),
            config,
            color_lo,
            color_hi,
            show_values: true,
            value_range: None,
            cell_radius: 2.0,
            cell_gap: 2.0,
            colormap: None,
            colors_explicit: false,
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

    // --- Generated common methods (except theme, which we override) ---
    chart_config_subtitle_footer!();
    chart_config_margin!();

    /// Set the chart title.
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.config.titles.title = Some(title.into());
        self
    }

    /// Set the visual theme.
    ///
    /// Re-derives the heatmap color gradient from the new theme unless
    /// the user has explicitly called `.colors()`.
    pub fn theme(mut self, theme: crate::theme::Theme) -> Self {
        self.config.theme = theme;
        if !self.colors_explicit && self.colormap.is_none() {
            let (lo, hi) = derive_heatmap_colors(&self.config.theme);
            self.color_lo = lo;
            self.color_hi = hi;
        }
        self
    }

    /// Set the export DPI (dots per inch).
    pub fn dpi(mut self, dpi: u32) -> Self {
        self.config.export.dpi = dpi.max(36);
        self
    }

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
    ///
    /// Marks colors as explicitly set, so `.theme()` won't override them.
    pub fn colors(mut self, lo: Color, hi: Color) -> Self {
        self.color_lo = lo;
        self.color_hi = hi;
        self.colors_explicit = true;
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

    /// Set a colormap for value→color mapping.
    ///
    /// When set, overrides the `colors()` lo/hi endpoints with a
    /// perceptually uniform palette.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use scry_chart::colormap::Viridis;
    /// let hm = Charts::heatmap(data).colormap(Viridis).build();
    /// ```
    pub fn colormap(mut self, cmap: impl Colormap + 'static) -> Self {
        self.colormap = Some(Arc::new(cmap));
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

    /// Build into a Chart.
    #[must_use]
    pub fn build(self) -> Chart {
        Box::new(self) as Chart
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

impl ChartSpec for Heatmap {
    fn render(&self, w: u32, h: u32) -> crate::layout::RenderedChart {
        crate::layout::heatmap::render_heatmap(self, w, h)
    }
    fn render_with_viewport(
        &self,
        w: u32,
        h: u32,
        vp: Option<(f64, f64, f64, f64)>,
    ) -> crate::layout::RenderedChart {
        if let Some((x0, x1, y0, y1)) = vp {
            let mut c = self.clone();
            c.config.axes.x_range = Some((x0, x1));
            c.config.axes.y_range = Some((y0, y1));
            c.render(w, h)
        } else {
            self.render(w, h)
        }
    }
    fn config(&self) -> Option<&ChartConfig> {
        Some(&self.config)
    }
    fn config_mut(&mut self) -> Option<&mut ChartConfig> {
        Some(&mut self.config)
    }
    fn data_extent(&self) -> Option<(f64, f64, f64, f64)> {
        None
    }
    fn clone_boxed(&self) -> Box<dyn ChartSpec> {
        Box::new(self.clone())
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

/// Derive heatmap gradient colors from a theme.
///
/// Low = theme background (tinted slightly), High = first palette color.
/// This ensures each theme produces a visually distinct heatmap gradient.
pub(crate) fn derive_heatmap_colors(theme: &crate::theme::Theme) -> (Color, Color) {
    let bg = theme.background;
    let color_lo = Color::from_rgba(
        (bg.r * 0.6 + 0.1).clamp(0.0, 1.0),
        (bg.g * 0.6 + 0.1).clamp(0.0, 1.0),
        (bg.b * 0.6 + 0.15).clamp(0.0, 1.0),
        1.0,
    );
    let color_hi = theme.series_color(0);
    (color_lo, color_hi)
}
