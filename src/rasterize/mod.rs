// SPDX-License-Identifier: MIT OR Apache-2.0
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

pub mod backend;
pub mod batch;
pub mod cache;
pub mod error;
pub mod pipeline;
pub mod profiler;
pub mod skia;

#[cfg(feature = "gpu")]
mod gpu_commands;
#[cfg(feature = "gpu")]
mod tessellate;
#[cfg(feature = "gpu")]
pub mod wgpu;
#[cfg(feature = "gpu")]
mod wgpu_context;

pub use backend::{
    AutoBackend, BackendKind, CpuBackend, GpuFallbackWarning, RasterBackend, RasterMeta,
    RasterResult,
};
pub use cache::{DirtyTile, RasterCache, TILE_SIZE};
pub use profiler::{
    CommandTiming, CommandType, PipelineProfile, ProfileHistory, ProfiledRasterizer, RasterProfile,
    SmoothedProfile, TransportProfile,
};
pub use pipeline::RasterPipeline;
pub use skia::{Rasterizer, RenderWarning};

#[cfg(feature = "gpu")]
pub use backend::GpuBackend;
#[cfg(feature = "gpu")]
pub use self::wgpu::rasterize_auto;
#[cfg(feature = "gpu")]
pub use self::wgpu::WgpuRasterizer;
#[cfg(feature = "gpu")]
pub use wgpu_context::WgpuContext2D;
