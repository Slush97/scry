// SPDX-License-Identifier: MIT OR Apache-2.0
//! The [`ChartSpec`] trait — the polymorphic interface for all chart types.
//!
//! Replaces the monolithic `Chart` enum with a trait-object–based design.
//! Every built-in chart type (`LineChart`, `ScatterChart`, …) implements
//! `ChartSpec`, and `Chart` is now a type alias for `Box<dyn ChartSpec>`.

use crate::chart::ChartConfig;
use crate::layout::RenderedChart;

/// A fully-configured chart that can render itself at a given pixel size.
///
/// All built-in chart types implement this trait. Third-party chart types
/// can implement it too, replacing the old `ChartType` escape-hatch.
pub trait ChartSpec: Send + Sync {
    /// Render the chart into a [`RenderedChart`] at the given pixel dimensions.
    fn render(&self, width: u32, height: u32) -> RenderedChart;

    /// Render with an optional viewport override for zoom/pan.
    ///
    /// When `viewport` is `Some((x_min, x_max, y_min, y_max))`, those ranges
    /// override the chart's own axis ranges. The default implementation
    /// ignores the viewport and delegates to [`render`](Self::render).
    fn render_with_viewport(
        &self,
        width: u32,
        height: u32,
        viewport: Option<(f64, f64, f64, f64)>,
    ) -> RenderedChart {
        let _ = viewport;
        self.render(width, height)
    }

    /// Immutable access to the chart's shared configuration.
    fn config(&self) -> Option<&ChartConfig> {
        None
    }

    /// Mutable access to the chart's shared configuration.
    fn config_mut(&mut self) -> Option<&mut ChartConfig> {
        None
    }

    /// Compute the data extent as `(x_min, x_max, y_min, y_max)`.
    ///
    /// Returns `None` for chart types without conventional XY axes.
    fn data_extent(&self) -> Option<(f64, f64, f64, f64)> {
        None
    }

    /// Export DPI (convenience accessor).
    fn dpi(&self) -> u32 {
        self.config().map_or(144, |c| c.export.dpi)
    }

    /// Clone into a boxed trait object.
    fn clone_boxed(&self) -> Box<dyn ChartSpec>;
}

impl Clone for Box<dyn ChartSpec> {
    fn clone(&self) -> Self {
        self.clone_boxed()
    }
}

impl std::fmt::Debug for dyn ChartSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("dyn ChartSpec").finish()
    }
}
