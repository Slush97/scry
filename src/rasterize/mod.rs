//! Rasterization of scenes into pixel buffers.
//!
//! This module provides [`Rasterizer`] which translates a scene display list
//! into a `tiny_skia::Pixmap`, [`RasterCache`] for skipping redundant
//! rasterizations when the scene hasn't changed, and [`ProfiledRasterizer`]
//! for per-command-type timing instrumentation.

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
