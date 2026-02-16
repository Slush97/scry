//! # scry-engine
//!
//! A vector graphics engine for the terminal. Builds anti-aliased scenes with
//! `tiny-skia`, then ships pixels to the screen via Kitty, Sixel, iTerm2, or
//! Unicode halfblocks — whatever the terminal supports.
//!
//! ## Layers
//!
//! Three layers, each usable on its own:
//!
//! 1. **Drawing** ([`scene`]) — builder API for shapes, paths, gradients, text,
//!    and [animations](`scene::animation`). No I/O, no terminal dependency.
//!
//! 2. **Transport** ([`transport`]) — protocol backends that write pixel data to
//!    stdout. Kitty (zlib / SHM), Sixel (median-cut 256-color), iTerm2 (inline
//!    PNG), halfblock fallback. [`transport::Picker`] auto-detects at runtime.
//!
//! 3. **Widget** ([`widget`]) — drops into [Ratatui](https://ratatui.rs) as a
//!    `StatefulWidget`. Content-hash caching, dirty-tile diffing, two-phase
//!    render-then-flush lifecycle.
//!
//! Optional: [`svg`] parses and renders SVG via `resvg`, with a
//! [line-drawing animation](`svg::line_drawing`) module.
//!
//! ## Quick start — Ratatui widget
//!
//! ```no_run
//! use scry_engine::prelude::*;
//!
//! let canvas = PixelCanvas::new(200, 200)
//!     .circle(100.0, 100.0, 50.0)
//!         .fill(Color::RED)
//!         .done();
//!
//! # let area = ratatui::layout::Rect::default();
//! # let mut pixel_canvas_state = todo!();
//! # let frame: &mut ratatui::Frame = todo!();
//! frame.render_stateful_widget(
//!     PixelCanvasWidget::new(canvas),
//!     area,
//!     &mut pixel_canvas_state,
//! );
//! ```
//!
//! ## Quick start — standalone (no Ratatui)
//!
//! ```
//! use scry_engine::scene::{PixelCanvas, Color};
//! use scry_engine::rasterize::Rasterizer;
//!
//! let canvas = PixelCanvas::new(100, 100)
//!     .background(Color::BLACK)
//!     .circle(50.0, 50.0, 30.0)
//!         .fill(Color::from_rgba8(70, 130, 180, 255))
//!         .done();
//!
//! let pixmap = Rasterizer::rasterize(&canvas).unwrap();
//! assert_eq!(pixmap.width(), 100);
//! ```
//!
//! ## Feature flags
//!
//! | Flag | Default | What it enables |
//! |------|---------|-----------------|
//! | `kitty` | ✅ | Kitty graphics protocol |
//! | `sixel` | ❌ | Sixel protocol (DEC terminals, foot, mlterm) |
//! | `iterm2` | ❌ | iTerm2 / `WezTerm` inline images |
//! | `widget` | ✅ | Ratatui `StatefulWidget` |
//! | `text` | ❌ | Glyph rasterization via fontdue |
//! | `shm` | ❌ | Zero-copy Kitty via POSIX shared memory |
//! | `svg` | ❌ | SVG rendering via resvg |
//!
//! **Typical combos:**
//! - `kitty,widget,shm,text` — local Kitty terminal, max quality
//! - `kitty,sixel,iterm2,widget` — broad compatibility, auto-detect
//! - default features off — headless rasterization for PNG export
//!
//! ## MSRV
//!
//! 1.83.0


// Strict lints for production quality
#![warn(missing_docs)]
#![warn(unreachable_pub)]
#![deny(unsafe_code)]

pub mod rasterize;
pub mod scene;
pub mod transport;

#[cfg(feature = "widget")]
pub mod widget;

#[cfg(feature = "svg")]
pub mod svg;

#[cfg(feature = "wasm")]
pub mod wasm;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur in `scry-engine`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PixelCanvasError {
    /// Failed to create a pixel buffer.
    #[error("failed to create pixmap: {0}")]
    PixmapCreation(String),

    /// Failed to rasterize the scene.
    #[error("rasterization failed: {0}")]
    Rasterization(String),

    /// Failed to transmit image data to the terminal.
    #[error("protocol transmission failed: {0}")]
    Transmission(#[from] std::io::Error),

    /// No graphics protocol is available.
    #[error("terminal does not support any graphics protocol")]
    NoProtocolAvailable,

    /// Font size could not be detected.
    #[error("font size detection failed — please provide manually")]
    FontSizeUnknown,
}

// ---------------------------------------------------------------------------
// Prelude
// ---------------------------------------------------------------------------

/// Convenience re-exports for common usage.
///
/// ```
/// use scry_engine::prelude::*;
/// ```
pub mod prelude {
    pub use crate::rasterize::{ProfileHistory, ProfiledRasterizer, RasterCache, Rasterizer};
    pub use crate::scene::animation::{
        AnimationState, Easing, Keyframe, Keyframes, Lerp, Transition,
    };
    pub use crate::scene::style::Color;
    pub use crate::scene::PixelCanvas;
    pub use crate::transport::{FontSize, Picker, ProtocolKind};
    pub use crate::PixelCanvasError;

    #[cfg(feature = "widget")]
    pub use crate::widget::{PixelCanvasState, PixelCanvasWidget};

    #[cfg(all(feature = "widget", feature = "svg"))]
    pub use crate::widget::SvgWidget;

    #[cfg(feature = "svg")]
    pub use crate::svg::{SvgError, SvgImage};

    #[cfg(feature = "svg")]
    pub use crate::svg::line_drawing::{
        DrawMode, PenPressure, PenTip, SvgLineDrawing, SvgPathSegment, Trail,
    };
}

// ---------------------------------------------------------------------------
// Re-exports at crate root
// ---------------------------------------------------------------------------

// Re-export the style module at crate root for convenience
pub use scene::style;
