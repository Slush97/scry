// SPDX-License-Identifier: MIT OR Apache-2.0
//! Contour chart type.
//!
//! Renders iso-level contour lines (or filled regions) from a 2D scalar field.

use crate::chart::config_builder::{
    chart_config_core, chart_config_margin, chart_config_subtitle_footer,
};
use crate::chart::{Chart, ChartConfig};
use crate::colormap::Colormap;
use scry_engine::style::Color;
use std::sync::Arc;

/// A contour chart — iso-level lines extracted from a 2D scalar field.
///
/// Build via [`Chart::contour`].
#[derive(Clone, Debug)]
#[must_use]
pub struct ContourChart {
    /// 2D data grid (row-major: `data[row][col]`).
    pub(crate) data: Vec<Vec<f64>>,
    /// Shared config.
    pub(crate) config: ChartConfig,
    /// Number of iso-levels to extract.
    pub(crate) levels: usize,
    /// Whether to fill between contour levels (area mode).
    pub(crate) filled: bool,
    /// Color for the minimum level.
    pub(crate) color_lo: Color,
    /// Color for the maximum level.
    pub(crate) color_hi: Color,
    /// Optional colormap (overrides lo/hi interpolation).
    pub(crate) colormap: Option<Arc<dyn Colormap>>,
}

impl ContourChart {
    /// Create a new contour chart from a 2D data grid.
    pub fn new(data: Vec<Vec<f64>>) -> Self {
        Self {
            data,
            config: ChartConfig::default(),
            levels: 10,
            filled: false,
            color_lo: Color::from_rgba8(30, 100, 200, 255),
            color_hi: Color::from_rgba8(230, 60, 60, 255),
            colormap: None,
        }
    }

    /// Set the number of iso-levels.
    pub fn levels(mut self, n: usize) -> Self {
        self.levels = n.max(2);
        self
    }

    /// Enable filled contour regions.
    pub fn filled(mut self) -> Self {
        self.filled = true;
        self
    }

    /// Set the color for the minimum value.
    pub fn color_lo(mut self, color: Color) -> Self {
        self.color_lo = color;
        self
    }

    /// Set the color for the maximum value.
    pub fn color_hi(mut self, color: Color) -> Self {
        self.color_hi = color;
        self
    }

    /// Use a named colormap.
    pub fn colormap(mut self, cm: impl Colormap + 'static) -> Self {
        self.colormap = Some(Arc::new(cm));
        self
    }

    /// Build as a [`Chart`].
    pub fn build(self) -> Chart {
        Chart::Contour(self)
    }

    /// Build with validation.
    pub fn try_build(self) -> Result<Chart, crate::error::ChartError> {
        if self.data.is_empty() {
            return Err(crate::error::ChartError::EmptyData);
        }
        let cols = self.data[0].len();
        if cols == 0 {
            return Err(crate::error::ChartError::EmptyData);
        }
        if self.data.iter().any(|r| r.len() != cols) {
            return Err(crate::error::ChartError::JaggedGrid);
        }
        Ok(Chart::Contour(self))
    }

    /// Get the (min, max) of all finite values in the grid.
    pub fn data_extent(&self) -> Option<(f64, f64)> {
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
        if lo.is_finite() && hi.is_finite() {
            Some((lo, hi))
        } else {
            None
        }
    }

    // --- Builder macros ---
    chart_config_core!();
    chart_config_subtitle_footer!();
    chart_config_margin!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contour_basic_build() {
        let chart = ContourChart::new(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ])
        .levels(5)
        .filled()
        .title("Contour Test")
        .build();
        assert!(matches!(chart, Chart::Contour(_)));
    }

    #[test]
    fn contour_try_build_empty() {
        let result = ContourChart::new(vec![]).try_build();
        assert!(result.is_err());
    }

    #[test]
    fn contour_try_build_jagged() {
        let result = ContourChart::new(vec![vec![1.0, 2.0], vec![3.0]]).try_build();
        assert!(result.is_err());
    }

    #[test]
    fn contour_data_extent() {
        let cc = ContourChart::new(vec![vec![1.0, 5.0], vec![3.0, 2.0]]);
        assert_eq!(cc.data_extent(), Some((1.0, 5.0)));
    }
}
