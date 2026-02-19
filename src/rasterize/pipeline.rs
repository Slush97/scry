// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared rasterization pipeline used by both [`IncrementalRenderer`] and
//! [`PixelCanvasState`].
//!
//! Encapsulates backend selection (GPU/CPU), content-hash caching, and the
//! `skip_cache` sentinel logic that was previously duplicated across both
//! renderers.
//!
//! ## Architecture
//!
//! The pipeline holds an [`AutoBackend`] which tries GPU first and falls back
//! to CPU transparently. Callers never need to handle backend selection
//! themselves.

use crate::rasterize::backend::{AutoBackend, BackendKind, RasterBackend};
use crate::rasterize::RasterCache;
use crate::scene::PixelCanvas;

/// Shared rasterization pipeline: backend + cache + rasterize.
///
/// Both `IncrementalRenderer` and `PixelCanvasState` delegate their
/// rasterization logic here, ensuring a single location for backend
/// selection and cache management.
pub struct RasterPipeline {
    /// Raster cache (content-hash → pixmap).
    pub cache: RasterCache,
    /// The rendering backend (auto-selects GPU vs CPU).
    backend: Box<dyn RasterBackend>,
}

impl RasterPipeline {
    /// Create a new pipeline with an empty cache and auto-selected backend.
    pub fn new() -> Self {
        Self {
            cache: RasterCache::new(),
            backend: Box::new(AutoBackend::new()),
        }
    }

    /// Create a pipeline with a specific backend.
    pub fn with_backend(backend: Box<dyn RasterBackend>) -> Self {
        Self {
            cache: RasterCache::new(),
            backend,
        }
    }

    /// Which backend is currently active.
    #[must_use]
    pub fn active_backend(&self) -> BackendKind {
        self.backend.kind()
    }

    /// Human-readable name of the active backend.
    #[must_use]
    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    /// Returns `true` if the GPU backend was successfully initialized.
    #[must_use]
    pub fn is_gpu(&self) -> bool {
        self.backend.kind() == BackendKind::Gpu
    }

    /// Rasterize a canvas into a new pixmap (no caching).
    ///
    /// This is the simplest entry point for one-shot rendering. For
    /// multi-frame rendering with caching, use [`rasterize_into_cache`].
    ///
    /// Uses the auto-selected backend (GPU when available, CPU fallback).
    ///
    /// # Errors
    ///
    /// Returns [`PixelCanvasError`] if the pixmap cannot be allocated.
    pub fn rasterize(
        &self,
        canvas: &PixelCanvas,
    ) -> Result<tiny_skia::Pixmap, crate::PixelCanvasError> {
        self.backend.rasterize(canvas).map(|r| r.pixmap)
    }

    /// Rasterize `canvas` into the cache's reusable pixmap.
    ///
    /// Returns the content-hash to use for cache lookups (may be a unique
    /// sentinel when `skip_cache` is true).
    ///
    /// Returns `None` if the pixmap could not be allocated.
    pub fn rasterize_into_cache(
        &mut self,
        canvas: &PixelCanvas,
        skip_cache: bool,
    ) -> Option<u64> {
        let content_hash = if skip_cache { 0 } else { canvas.content_hash() };

        // Cache hit — return early
        if !skip_cache && self.cache.is_valid(content_hash) {
            return Some(content_hash);
        }

        // Get or allocate the pixmap
        let (pixmap_opt, gc) = self
            .cache
            .get_or_insert_with_grad_cache(canvas.width(), canvas.height());
        let pixmap = pixmap_opt?;

        // Use the backend to rasterize
        match self.backend.rasterize(canvas) {
            Ok(result) => {
                // Copy GPU/CPU result into the cache pixmap
                if result.pixmap.width() == pixmap.width()
                    && result.pixmap.height() == pixmap.height()
                {
                    pixmap.data_mut().copy_from_slice(result.pixmap.data());
                } else {
                    // Dimension mismatch (shouldn't happen), fall back to CPU
                    crate::rasterize::Rasterizer::rasterize_into_cached(canvas, pixmap, gc);
                }
            }
            Err(_e) => {
                // Backend error — CPU fallback (always works)
                if crate::scry_debug_enabled() {
                    eprintln!("[scry] rasterize failed, using CPU fallback: {_e}");
                }
                crate::rasterize::Rasterizer::rasterize_into_cached(canvas, pixmap, gc);
            }
        }

        let store_hash = Self::compute_store_hash(skip_cache, content_hash);
        self.cache.mark_valid(store_hash);

        Some(store_hash)
    }

    fn compute_store_hash(skip_cache: bool, content_hash: u64) -> u64 {
        if skip_cache {
            use std::sync::atomic::{AtomicU64, Ordering};
            static FRAME_SEQ: AtomicU64 = AtomicU64::new(1);
            FRAME_SEQ.fetch_add(1, Ordering::Relaxed)
        } else {
            content_hash
        }
    }
}

impl Default for RasterPipeline {
    fn default() -> Self {
        Self::new()
    }
}
