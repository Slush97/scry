//! Ratatui `StatefulWidget` integration for `PixelCanvas`.
//!
//! This module bridges the drawing API with Ratatui's rendering model.
//! The [`PixelCanvasWidget`] implements `StatefulWidget`, and
//! [`PixelCanvasState`] manages the image lifecycle across frames.
//!
//! # Render → Flush Lifecycle
//!
//! Pixel graphics use a two-phase rendering model:
//!
//! 1. **Render** — `frame.render_stateful_widget(widget, area, &mut state)` rasterizes
//!    the scene and prepares protocol data, but does not transmit it yet.
//! 2. **Flush** — `state.flush()` transmits the image **after** `terminal.draw()`
//!    returns, ensuring Kitty escape sequences are written after ratatui's buffer diff.
//!
//! # Performance Modes
//!
//! - **Default** — content-hash caching skips re-render when the scene is unchanged.
//! - [`skip_cache()`](PixelCanvasWidget::skip_cache) — skip hashing for fully-animated
//!   scenes that change every frame.
//! - [`incremental()`](PixelCanvasWidget::incremental) — transmit only changed 64×64
//!   tiles for partially-animated scenes (Kitty backend only).


pub(crate) mod widget_impl;

#[cfg(feature = "svg")]
mod svg_widget;

pub use widget_impl::{PixelCanvasState, PixelCanvasWidget};

#[cfg(feature = "svg")]
pub use svg_widget::SvgWidget;
