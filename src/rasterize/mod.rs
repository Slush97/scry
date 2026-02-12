//! Rasterization of scenes into pixel buffers.
//!
//! This module provides [`Rasterizer`] which translates a scene display list
//! into a `tiny_skia::Pixmap`, and [`RasterCache`] for skipping redundant
//! rasterizations when the scene hasn't changed.

pub mod cache;
pub mod skia;

pub use cache::RasterCache;
pub use skia::Rasterizer;
