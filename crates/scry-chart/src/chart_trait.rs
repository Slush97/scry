// SPDX-License-Identifier: MIT OR Apache-2.0
//! Trait for user-defined custom chart types.
//!
//! Implement [`ChartType`] to create chart types that render through the
//! same pipeline as built-in charts — including PNG, SVG, and PDF export,
//! widget rendering, and subplot composition.
//!
//! # Example
//!
//! ```ignore
//! use scry_chart::chart_trait::ChartType;
//! use scry_chart::layout::{RenderedChart, TextOverlay};
//! use scry_engine::scene::PixelCanvas;
//!
//! struct MyCustomChart {
//!     data: Vec<f64>,
//! }
//!
//! impl ChartType for MyCustomChart {
//!     fn render(&self, width: u32, height: u32) -> RenderedChart {
//!         let canvas = PixelCanvas::new(width, height);
//!         // ... draw custom visuals ...
//!         RenderedChart {
//!             canvas,
//!             text_overlays: vec![],
//!             plot_area: None,
//!             x_scale: None,
//!             y_scale: None,
//!             series_points: vec![],
//!         }
//!     }
//! }
//!
//! let chart = Chart::custom(MyCustomChart { data: vec![1.0, 2.0] });
//! save_png(&chart, 800, 500, "custom.png")?;
//! ```

use crate::layout::RenderedChart;

/// Trait for implementing custom chart types.
///
/// Downstream crates can implement this trait to define their own chart
/// visualizations without modifying the `Chart` enum. Custom charts participate
/// in the full rendering pipeline: they can be exported to PNG, SVG, and PDF,
/// rendered in the terminal widget, and composed into subplot grids.
pub trait ChartType: Send + Sync {
    /// Render the chart into a `RenderedChart` at the given pixel dimensions.
    ///
    /// The returned `RenderedChart` should contain:
    /// - A `PixelCanvas` with all vector drawing commands
    /// - `TextOverlay`s for labels, titles, and tick values
    /// - Optionally, `plot_area`, scale info, and series points for interactivity
    fn render(&self, width: u32, height: u32) -> RenderedChart;
}

// Allow Box<dyn ChartType> to be cloned via a manual clone that re-renders.
// This is needed because Chart derives Clone, and Custom variant carries the trait object.
impl Clone for Box<dyn ChartType> {
    fn clone(&self) -> Self {
        // We can't generically clone trait objects, but we also don't need to —
        // the Chart enum's Clone is only used for viewport injection in
        // render_chart_with_viewport, where it will immediately re-render.
        // We use a stub that produces an empty chart on clone.
        Box::new(StubChartType)
    }
}

impl std::fmt::Debug for dyn ChartType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("dyn ChartType").finish()
    }
}

/// Stub custom chart used as a placeholder when cloning `Box<dyn ChartType>`.
#[derive(Clone, Debug)]
struct StubChartType;

impl ChartType for StubChartType {
    fn render(&self, width: u32, height: u32) -> RenderedChart {
        RenderedChart {
            canvas: scry_engine::scene::PixelCanvas::new(width, height),
            text_overlays: vec![],
            plot_area: None,
            x_scale: None,
            y_scale: None,
            series_points: vec![],
        }
    }
}
