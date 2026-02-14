//! Ratatui `StatefulWidget` integration for `PixelCanvas`.
//!
//! This module bridges the drawing API with Ratatui's rendering model.
//! The [`PixelCanvasWidget`] implements `StatefulWidget`, and
//! [`PixelCanvasState`] manages the image lifecycle across frames.

pub(crate) mod widget_impl;

#[cfg(feature = "svg")]
mod svg_widget;

pub use widget_impl::{PixelCanvasState, PixelCanvasWidget};

#[cfg(feature = "svg")]
pub use svg_widget::SvgWidget;
