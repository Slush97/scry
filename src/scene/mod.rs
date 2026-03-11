// SPDX-License-Identifier: MIT OR Apache-2.0
//! Scene construction and drawing API.
//!
//! This module provides the core drawing primitives for `scry-engine`.
//! It is **independent of any terminal protocol or Ratatui** — you can use it
//! standalone to build scenes and rasterize them to pixel buffers.
//!
//! # Architecture
//!
//! - [`PixelCanvas`] is the fluent builder that collects [`DrawCommand`]s
//! - [`DrawCommand`] is the display list entry (circle, rect, line, path, etc.)
//! - Style types ([`Color`], [`FillStyle`], [`StrokeStyle`], etc.) describe
//!   how shapes are rendered
//!
//! # Example
//!
//! ```
//! use scry_engine::scene::PixelCanvas;
//! use scry_engine::scene::style::Color;
//!
//! let canvas = PixelCanvas::new(200, 200)
//!     .background(Color::from_rgba8(20, 20, 30, 255))
//!     .circle(100.0, 100.0, 60.0)
//!         .fill(Color::from_rgba8(70, 130, 180, 255))
//!         .stroke(Color::WHITE, 2.0)
//!         .done();
//!
//! assert_eq!(canvas.commands().len(), 1);
//! ```

#[allow(
    clippy::suboptimal_flops,
    clippy::imprecise_flops,
    clippy::many_single_char_names
)]
pub mod animation;
pub mod builder;
pub mod command;
pub mod style;
pub mod validate;

#[cfg(feature = "input")]
pub mod hit;
#[cfg(feature = "input")]
pub mod input;

// Re-export the main types at the module level for convenience.
#[cfg(feature = "text")]
pub use builder::TextBuilder;
pub use builder::{
    GradientBuilder, GroupBuilder, ImageBuilder, LineBuilder, PixelCanvas, ShapeBuilder,
};
pub use command::{DrawCommand, ImageData, PathData};
#[cfg(feature = "text")]
pub use command::{FontData, TextAlign, TextMetrics, TextStyle};
pub use style::{
    BlendMode, ClipRegion, Color, DashPattern, FillRule, FillStyle, GradientDef, GradientKind,
    GradientStop, LineCap, LineJoin, Point, Rect, ShapeStyle, StrokeStyle, Transform,
};
