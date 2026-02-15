//! Rasterization of scenes into pixel buffers.
//!
//! This module provides [`Rasterizer`] which translates a scene display list
//! into a `tiny_skia::Pixmap`, [`RasterCache`] for skipping redundant
//! rasterizations when the scene hasn't changed, and [`ProfiledRasterizer`]
//! for per-command-type timing instrumentation.
//!
//! # Example
//!
//! ```
//! use scry_engine::scene::{PixelCanvas, Color};
//! use scry_engine::rasterize::Rasterizer;
//!
//! let canvas = PixelCanvas::new(200, 200)
//!     .circle(100.0, 100.0, 60.0)
//!         .fill(Color::RED)
//!         .done();
//!
//! let pixmap = Rasterizer::rasterize(&canvas).unwrap();
//! assert_eq!(pixmap.width(), 200);
//! ```
//!
//! For animation loops, use [`Rasterizer::rasterize_into()`] with a reusable
//! pixmap and [`RasterCache`] for content-hash caching.


pub mod batch;
pub mod cache;
pub mod profiler;
pub mod skia;

pub use cache::{DirtyTile, RasterCache, TILE_SIZE};
pub use profiler::{
    CommandTiming, CommandType, PipelineProfile, ProfileHistory, ProfiledRasterizer, RasterProfile,
    SmoothedProfile, TransportProfile,
};
pub use skia::Rasterizer;
