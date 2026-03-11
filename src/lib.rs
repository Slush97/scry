// SPDX-License-Identifier: MIT OR Apache-2.0
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
//! | `gpu` | ✅ | GPU-accelerated rasterization via wgpu |
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
//!
//! ## Thread Safety
//!
//! | Type | `Send` | `Sync` | Notes |
//! |------|--------|--------|-------|
//! | [`PixelCanvas`](scene::PixelCanvas) | ✅ | ✅ | Immutable display list, safe to share |
//! | [`RasterPipeline`](rasterize::RasterPipeline) | ✅ | ❌ | Holds `Box<dyn RasterBackend>` |
//! | [`IncrementalRenderer`](render::IncrementalRenderer) | ✅ | ❌ | Mutable protocol state |
//! | [`GpuDevice`](gpu::GpuDevice) | ✅ | ✅ | `Arc`-wrapped internals |
//! | [`SdfPipeline`](sdf::pipeline::SdfPipeline) | ✅ | ❌ | Mutable GPU/readback state |
// Strict lints for production quality
#![warn(missing_docs)]
#![warn(unreachable_pub)]
#![deny(unsafe_code)]

pub mod camera3d;
pub mod diagnostics;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod math3d;
pub mod rasterize;
pub mod render;
pub mod scene;
pub mod transport;

#[cfg(feature = "widget")]
pub mod widget;

#[cfg(feature = "svg")]
pub mod svg;

#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(feature = "sdf")]
pub mod sdf;

// ---------------------------------------------------------------------------
// Debug logging
// ---------------------------------------------------------------------------

/// Returns `true` when the `SCRY_DEBUG` environment variable is set.
///
/// Checked once on first call and cached via `OnceLock`.
#[must_use]
pub fn scry_debug_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("SCRY_DEBUG").is_ok())
}

/// Log a warning-level diagnostic message.
///
/// When the `logging` feature is enabled, emits a `tracing::warn!` event.
/// Otherwise, falls back to `eprintln!` when `SCRY_DEBUG` is set.
#[macro_export]
#[doc(hidden)]
macro_rules! scry_warn {
    ($($arg:tt)*) => {
        {
            #[cfg(feature = "logging")]
            ::tracing::warn!($($arg)*);
            #[cfg(not(feature = "logging"))]
            if $crate::scry_debug_enabled() {
                eprintln!($($arg)*);
            }
        }
    };
}

/// Log an info-level diagnostic message.
///
/// When the `logging` feature is enabled, emits a `tracing::info!` event.
/// Otherwise, falls back to `eprintln!` when `SCRY_DEBUG` is set.
#[macro_export]
#[doc(hidden)]
macro_rules! scry_info {
    ($($arg:tt)*) => {
        {
            #[cfg(feature = "logging")]
            ::tracing::info!($($arg)*);
            #[cfg(not(feature = "logging"))]
            if $crate::scry_debug_enabled() {
                eprintln!($($arg)*);
            }
        }
    };
}

/// Log an error-level diagnostic message.
///
/// When the `logging` feature is enabled, emits a `tracing::error!` event.
/// Otherwise, always prints to stderr (errors should never be silenced).
#[macro_export]
#[doc(hidden)]
macro_rules! scry_error {
    ($($arg:tt)*) => {
        {
            #[cfg(feature = "logging")]
            ::tracing::error!($($arg)*);
            #[cfg(not(feature = "logging"))]
            eprintln!($($arg)*);
        }
    };
}

/// Log a debug-level diagnostic message.
///
/// When the `logging` feature is enabled, emits a `tracing::debug!` event.
/// Otherwise, falls back to `eprintln!` when `SCRY_DEBUG` is set.
///
/// Use this for internal implementation details (GPU init, pipeline selection,
/// backend fallback decisions) that are useful during development but too
/// noisy for normal operation.
#[macro_export]
#[doc(hidden)]
macro_rules! scry_debug {
    ($($arg:tt)*) => {
        {
            #[cfg(feature = "logging")]
            ::tracing::debug!($($arg)*);
            #[cfg(not(feature = "logging"))]
            if $crate::scry_debug_enabled() {
                eprintln!($($arg)*);
            }
        }
    };
}

/// Log a trace-level diagnostic message.
///
/// When the `logging` feature is enabled, emits a `tracing::trace!` event.
/// Without the `logging` feature, this is a **silent no-op** — trace messages
/// are too granular for `eprintln!` output.
///
/// Use this for per-pixel, per-command, or per-iteration diagnostics that
/// would overwhelm any non-structured output.
#[macro_export]
#[doc(hidden)]
macro_rules! scry_trace {
    ($($arg:tt)*) => {
        {
            #[cfg(feature = "logging")]
            ::tracing::trace!($($arg)*);
            // Silent without the logging feature — trace is too noisy for eprintln.
        }
    };
}

/// Enter a tracing span for pipeline stage instrumentation.
///
/// When the `logging` feature is enabled, creates a `tracing::info_span!` and
/// returns its `Entered` guard. When disabled, returns a no-op `()` guard.
///
/// # Usage
///
/// ```ignore
/// let _span = scry_span!("rasterize", width = 200, height = 200);
/// // ... work happens inside the span ...
/// // span is exited when `_span` is dropped
/// ```
#[macro_export]
#[doc(hidden)]
macro_rules! scry_span {
    ($name:expr $(, $($field:tt)*)?) => {
        {
            #[cfg(feature = "logging")]
            {
                let _span = ::tracing::info_span!($name $(, $($field)*)?);
                _span.entered()
            }
            #[cfg(not(feature = "logging"))]
            { () }
        }
    };
}

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
    ///
    /// Note: this variant wraps a raw `io::Error`. For transport-layer
    /// errors with more context, see [`Transport`](Self::Transport).
    #[error("protocol transmission failed: {0}")]
    Transmission(std::io::Error),

    /// No graphics protocol is available.
    #[error("terminal does not support any graphics protocol")]
    NoProtocolAvailable,

    /// Font size could not be detected.
    #[error("font size detection failed — please provide manually")]
    FontSizeUnknown,

    /// Terminal probing failed.
    #[error("terminal probe failed: {0}")]
    ProbeFailed(String),

    /// Rasterization subsystem error (pixmap allocation, GPU backend, etc.).
    #[error(transparent)]
    #[cfg(feature = "gpu")]
    Raster(#[from] crate::rasterize::error::RasterError),

    /// Transport layer error (I/O, PNG encoding, compression, SHM, etc.).
    #[error(transparent)]
    TransportLayer(#[from] crate::transport::error::TransportError),

    /// GPU subsystem error (adapter, device, pipeline, buffer).
    #[error(transparent)]
    #[cfg(feature = "gpu")]
    Gpu(#[from] crate::gpu::error::GpuError),

    /// SDF renderer error (readback, timeout, shader compilation).
    #[error(transparent)]
    #[cfg(feature = "sdf")]
    Sdf(#[from] crate::sdf::error::SdfError),
}

// Manual `From<io::Error>` since we removed `#[from]` on `Transmission`
// to avoid conflicting with `TransportError::Io(#[from] io::Error)`.
impl From<std::io::Error> for PixelCanvasError {
    fn from(err: std::io::Error) -> Self {
        Self::Transmission(err)
    }
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
    #[cfg(feature = "gpu")]
    pub use crate::rasterize::{rasterize_auto, GpuBackend, WgpuContext2D, WgpuRasterizer};
    pub use crate::rasterize::{
        AutoBackend, BackendKind, CpuBackend, RasterBackend, RasterPipeline, RasterResult,
    };
    pub use crate::rasterize::{ProfileHistory, ProfiledRasterizer, RasterCache, Rasterizer};
    pub use crate::render::IncrementalRenderer;
    pub use crate::scene::animation::{
        preset, AnimationSequence, AnimationState, Easing, Keyframe, Keyframes, Lerp,
        SequencePlayer, Spring, SpringConfig, Transition,
    };
    pub use crate::scene::style::Color;
    pub use crate::scene::PixelCanvas;
    #[cfg(feature = "sdf")]
    pub use crate::sdf::{SdfBackend, SdfPipeline, SdfRenderResult};
    pub use crate::transport::{FontSize, Picker, ProtocolKind};
    pub use crate::PixelCanvasError;

    #[cfg(feature = "input")]
    pub use crate::scene::hit::{HitResult, HitTag, HitTestConfig, HitTester};
    #[cfg(feature = "input")]
    pub use crate::scene::input::{InputHandler, Interaction, MouseButton};

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

    #[cfg(feature = "text")]
    pub use crate::rasterize::skia::text::{default_font, measure_text, render_text_to_image};
    #[cfg(feature = "text")]
    pub use crate::scene::{FontData, TextAlign, TextMetrics, TextStyle};

    #[cfg(feature = "sdf")]
    pub use crate::scene::command::SdfSceneRef;
    #[cfg(feature = "sdf")]
    pub use crate::sdf::{
        Material, SdfCamera, SdfLight, SdfObject, SdfRenderer, SdfScene, SdfShape, Vec3,
    };

    #[cfg(feature = "sdf-gpu")]
    pub use crate::sdf::{SdfGpuContext, SdfGpuRenderer};
}

// ---------------------------------------------------------------------------
// Re-exports at crate root
// ---------------------------------------------------------------------------

// Re-export the style module at crate root for convenience
pub use scene::style;

/// Re-export `tiny_skia::Pixmap` so downstream crates don't need a direct dependency.
pub use tiny_skia::Pixmap;
