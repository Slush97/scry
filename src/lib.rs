//! # ratatui-pixelcanvas
//!
//! Pixel-perfect vector graphics for [Ratatui](https://ratatui.rs) via Kitty,
//! Sixel, and Unicode fallbacks.
//!
//! ## Architecture
//!
//! The library is organized into three independent layers:
//!
//! - **Layer 1 — Drawing API** ([`scene`]): Fluent builder for constructing
//!   vector scenes (circles, lines, paths, gradients). No dependency on
//!   Ratatui or terminal protocols.
//!
//! - **Layer 2 — Transport** ([`transport`]): Protocol backends for
//!   transmitting pixel data to the terminal (Kitty, Sixel, Halfblock).
//!
//! - **Layer 3 — Widget** ([`widget`]): Ratatui `StatefulWidget` integration
//!   that coordinates rasterization and transmission.
//!
//! ## Quick Start
//!
//! ```no_run
//! use ratatui_pixelcanvas::prelude::*;
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
//! ## Feature Flags
//!
//! | Flag      | Default | Description                                      |
//! |-----------|---------|--------------------------------------------------|
//! | `kitty`   | ✅      | Kitty graphics protocol backend                  |
//! | `sixel`   | ❌      | Sixel graphics protocol backend                  |
//! | `iterm2`  | ❌      | iTerm2 inline image protocol backend              |
//! | `widget`  | ✅      | Ratatui `StatefulWidget` integration              |
//! | `text`    | ❌      | Text rendering via `fontdue`                      |
//! | `shm`     | ❌      | Zero-copy Kitty transmission via POSIX shared mem |
//! | `svg`     | ❌      | SVG rendering via `resvg`                         |

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

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur in `ratatui-pixelcanvas`.
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
/// use ratatui_pixelcanvas::prelude::*;
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
