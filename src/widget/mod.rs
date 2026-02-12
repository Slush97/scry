//! Ratatui `StatefulWidget` integration for `PixelCanvas`.
//!
//! This module bridges the drawing API with Ratatui's rendering model.
//! The [`PixelCanvasWidget`] implements `StatefulWidget`, and
//! [`PixelCanvasState`] manages the image lifecycle across frames.

mod widget_impl;

pub use widget_impl::{PixelCanvasState, PixelCanvasWidget};
