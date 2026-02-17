// SPDX-License-Identifier: MIT OR Apache-2.0
//! Content-hash caching for rasterized scenes.
//!
//! The [`RasterCache`] stores the last rasterized result and its content hash,
//! allowing the rendering pipeline to skip rasterization when the scene has
//! not changed between frames.
//!
//! ## Dirty-Tile Tracking
//!
//! For incremental transmission, the cache can compare the current pixmap
//! against the previous frame's pixel data to identify [`TILE_SIZE`]×[`TILE_SIZE`]
//! tiles that changed. Only dirty tiles need to be re-transmitted.

use tiny_skia::Pixmap;

use crate::rasterize::skia::GradientCache;

/// Size of each tile for dirty tracking, in pixels.
pub const TILE_SIZE: usize = 64;

/// FNV-1a offset basis for 64-bit hashing.
/// Used instead of `DefaultHasher` (SipHash-2-4) for faster tile comparison
/// — ~3–5× faster on sequential byte data with no adversarial-input concern.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a prime for 64-bit hashing.
const FNV_PRIME: u64 = 0x0100_0000_01b3;

/// A tile region that has changed between frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirtyTile {
    /// X offset in pixels.
    pub x: usize,
    /// Y offset in pixels.
    pub y: usize,
    /// Width of this tile in pixels (may be smaller at edges).
    pub width: usize,
    /// Height of this tile in pixels (may be smaller at edges).
    pub height: usize,
}

/// Content-addressable cache for a single rasterized scene.
///
/// Stores the content hash and pixmap from the last rasterization. When the
/// scene's content hash matches the cached value, rasterization is skipped
/// entirely.
///
/// Also supports dirty-tile detection for incremental updates.
#[derive(Debug)]
pub struct RasterCache {
    /// Content hash of the last rasterized scene.
    hash: Option<u64>,
    /// The rasterized pixel buffer.
    pixmap: Option<Pixmap>,
    /// Dimensions of the last stored/validated pixmap.
    cached_dims: Option<(u32, u32)>,
    /// Per-tile hashes of the previous frame for dirty detection.
    prev_tile_hashes: Vec<u64>,
    /// Width in tiles of the grid.
    tiles_x: usize,
    /// Height in tiles of the grid.
    tiles_y: usize,
    /// Persistent gradient cache shared across frames.
    grad_cache: GradientCache,
}

impl RasterCache {
    /// Create a new, empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            hash: None,
            pixmap: None,
            cached_dims: None,
            prev_tile_hashes: Vec::new(),
            tiles_x: 0,
            tiles_y: 0,
            grad_cache: GradientCache::new(),
        }
    }

    /// Check whether the cache contains a valid entry for the given content hash.
    ///
    /// Also validates that the cached pixmap dimensions match the stored dimensions
    /// to prevent stale entries after dimension changes.
    #[must_use]
    pub fn is_valid(&self, content_hash: u64) -> bool {
        self.hash == Some(content_hash)
            && self.pixmap.as_ref().map_or(false, |pm| {
                self.cached_dims == Some((pm.width(), pm.height()))
            })
    }

    /// Get the cached pixmap, if the content hash matches.
    #[must_use]
    pub fn get(&self, content_hash: u64) -> Option<&Pixmap> {
        if self.is_valid(content_hash) {
            self.pixmap.as_ref()
        } else {
            None
        }
    }

    /// Get a mutable reference to the cached pixmap, if the content hash matches.
    ///
    /// This enables in-place re-rasterization via [`Rasterizer::rasterize_into`](super::Rasterizer::rasterize_into)
    /// without allocating a new pixmap.
    #[must_use]
    pub fn get_mut(&mut self, content_hash: u64) -> Option<&mut Pixmap> {
        if self.is_valid(content_hash) {
            self.pixmap.as_mut()
        } else {
            None
        }
    }

    /// Get the cached pixmap (regardless of hash), or create one with the given dimensions.
    ///
    /// This is useful for animation loops where you always want a pixmap of
    /// a certain size but don't want to allocate every frame.
    pub fn get_or_insert(&mut self, width: u32, height: u32) -> Option<&mut Pixmap> {
        // If we have a pixmap of the right size, reuse it
        if let Some(ref pm) = self.pixmap {
            if pm.width() == width && pm.height() == height {
                return self.pixmap.as_mut();
            }
        }
        // Allocate a new one
        if let Some(pm) = Pixmap::new(width, height) {
            self.pixmap = Some(pm);
            self.hash = None;
            self.pixmap.as_mut()
        } else {
            None
        }
    }

    /// Store a rasterized result in the cache.
    pub fn store(&mut self, content_hash: u64, pixmap: Pixmap) {
        self.cached_dims = Some((pixmap.width(), pixmap.height()));
        self.hash = Some(content_hash);
        self.pixmap = Some(pixmap);
    }

    /// Mark the cache as valid for the given content hash without replacing the pixmap.
    ///
    /// Used after [`Rasterizer::rasterize_into`](super::Rasterizer::rasterize_into) writes directly into the
    /// pixmap returned by [`get_or_insert`](Self::get_or_insert).
    pub fn mark_valid(&mut self, content_hash: u64) {
        self.cached_dims = self.pixmap.as_ref().map(|pm| (pm.width(), pm.height()));
        self.hash = Some(content_hash);
    }

    /// Invalidate the cache.
    pub fn clear(&mut self) {
        self.hash = None;
        self.pixmap = None;
        self.cached_dims = None;
        self.prev_tile_hashes.clear();
        self.tiles_x = 0;
        self.tiles_y = 0;
        self.grad_cache.clear();
    }

    /// Get a mutable reference to the persistent gradient cache.
    ///
    /// This cache persists across frames so that identical gradients
    /// are rendered once and blitted thereafter (~10× faster).
    pub const fn grad_cache_mut(&mut self) -> &mut GradientCache {
        &mut self.grad_cache
    }

    /// Get the cached pixmap (or create one) AND the gradient cache in one borrow.
    ///
    /// Avoids the double-borrow problem when callers need both simultaneously.
    pub fn get_or_insert_with_grad_cache(
        &mut self,
        width: u32,
        height: u32,
    ) -> (Option<&mut Pixmap>, &mut GradientCache) {
        // If we have a pixmap of the right size, reuse it
        if let Some(ref pm) = self.pixmap {
            if pm.width() != width || pm.height() != height {
                // Size mismatch — allocate a new one
                if let Some(pm) = Pixmap::new(width, height) {
                    self.pixmap = Some(pm);
                    self.hash = None;
                }
            }
        } else {
            // No pixmap — allocate
            if let Some(pm) = Pixmap::new(width, height) {
                self.pixmap = Some(pm);
                self.hash = None;
            }
        }
        (self.pixmap.as_mut(), &mut self.grad_cache)
    }

    /// Compute which tiles have changed since the last call.
    ///
    /// Divides the pixmap into [`TILE_SIZE`]×[`TILE_SIZE`] tiles and compares
    /// per-tile hashes against the previous frame. Returns only the tiles
    /// whose pixel data has changed.
    ///
    /// After this call, the internal tile hashes are updated for the next
    /// frame comparison.
    ///
    /// Returns an empty vec if the entire frame is unchanged.
    pub fn compute_dirty_tiles(&mut self, pixmap: &Pixmap) -> Vec<DirtyTile> {
        self.compute_dirty_tiles_from_data(
            pixmap.data(),
            pixmap.width() as usize,
            pixmap.height() as usize,
        )
    }

    /// Compute dirty tiles using the internally cached pixmap.
    ///
    /// This avoids the self-referential borrow problem that occurs when
    /// calling [`compute_dirty_tiles`](Self::compute_dirty_tiles) with a
    /// pixmap obtained from [`get()`](Self::get) on the same cache.
    ///
    /// Returns `None` if no pixmap is currently cached.
    pub fn compute_dirty_tiles_cached(&mut self) -> Option<Vec<DirtyTile>> {
        let pixmap = self.pixmap.as_ref()?;
        let data = pixmap.data();
        let w = pixmap.width() as usize;
        let h = pixmap.height() as usize;
        // Use free function to avoid self-referential borrow (data borrows
        // self.pixmap immutably while we need &mut self.prev_tile_hashes).
        Some(compute_dirty_from_data(
            data,
            w,
            h,
            &mut self.prev_tile_hashes,
            &mut self.tiles_x,
            &mut self.tiles_y,
        ))
    }

    /// Compute dirty tiles from raw RGBA pixel data.
    ///
    /// This variant avoids the self-referential borrow problem when the
    /// pixmap lives inside the same `RasterCache` (e.g. in `flush()`).
    /// Pass `pixmap.data()`, `width`, and `height` separately.
    pub fn compute_dirty_tiles_from_data(
        &mut self,
        data: &[u8],
        w: usize,
        h: usize,
    ) -> Vec<DirtyTile> {
        compute_dirty_from_data(
            data,
            w,
            h,
            &mut self.prev_tile_hashes,
            &mut self.tiles_x,
            &mut self.tiles_y,
        )
    }
}

/// Compute dirty tiles from raw pixel data, comparing per-tile hashes.
///
/// This is a free function (not a method) so that the caller can borrow
/// pixel data and tile-hash state from different struct fields simultaneously,
/// avoiding the self-referential borrow problem in `compute_dirty_tiles_cached`.
fn compute_dirty_from_data(
    data: &[u8],
    w: usize,
    h: usize,
    prev_tile_hashes: &mut Vec<u64>,
    tiles_x: &mut usize,
    tiles_y: &mut usize,
) -> Vec<DirtyTile> {
    let tx = w.div_ceil(TILE_SIZE);
    let ty = h.div_ceil(TILE_SIZE);
    let total = tx * ty;

    // Take the old hashes out for comparison, then reuse the vec
    // for computing the current frame's hashes (avoids allocation).
    let old_hashes = std::mem::take(prev_tile_hashes);
    let mut current_hashes = Vec::with_capacity(total.max(old_hashes.len()));

    for ty_idx in 0..ty {
        for tx_idx in 0..tx {
            let x0 = tx_idx * TILE_SIZE;
            let y0 = ty_idx * TILE_SIZE;
            let tw = TILE_SIZE.min(w - x0);
            let th = TILE_SIZE.min(h - y0);

            let mut hash = FNV_OFFSET;
            for row in y0..(y0 + th) {
                let start = (row * w + x0) * 4;
                let end = start + tw * 4;
                for &byte in &data[start..end] {
                    hash ^= u64::from(byte);
                    hash = hash.wrapping_mul(FNV_PRIME);
                }
            }
            current_hashes.push(hash);
        }
    }

    // Compare against previous
    let mut dirty = Vec::new();

    if *tiles_x != tx || *tiles_y != ty || old_hashes.len() != total {
        // Dimensions changed — everything is dirty
        for ty_idx in 0..ty {
            for tx_idx in 0..tx {
                dirty.push(DirtyTile {
                    x: tx_idx * TILE_SIZE,
                    y: ty_idx * TILE_SIZE,
                    width: TILE_SIZE.min(w - tx_idx * TILE_SIZE),
                    height: TILE_SIZE.min(h - ty_idx * TILE_SIZE),
                });
            }
        }
    } else {
        for (i, (&prev, &curr)) in old_hashes.iter().zip(current_hashes.iter()).enumerate() {
            if prev != curr {
                let col_idx = i % tx;
                let row_idx = i / tx;
                dirty.push(DirtyTile {
                    x: col_idx * TILE_SIZE,
                    y: row_idx * TILE_SIZE,
                    width: TILE_SIZE.min(w - col_idx * TILE_SIZE),
                    height: TILE_SIZE.min(h - row_idx * TILE_SIZE),
                });
            }
        }
    }

    // Store for next frame
    *prev_tile_hashes = current_hashes;
    *tiles_x = tx;
    *tiles_y = ty;

    dirty
}

impl Default for RasterCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_cache_is_not_valid() {
        let cache = RasterCache::new();
        assert!(!cache.is_valid(42));
        assert!(cache.get(42).is_none());
    }

    #[test]
    fn cache_hit_with_matching_hash() {
        let mut cache = RasterCache::new();
        let pixmap = Pixmap::new(10, 10).unwrap();
        cache.store(42, pixmap);

        assert!(cache.is_valid(42));
        assert!(cache.get(42).is_some());
    }

    #[test]
    fn cache_miss_with_different_hash() {
        let mut cache = RasterCache::new();
        let pixmap = Pixmap::new(10, 10).unwrap();
        cache.store(42, pixmap);

        assert!(!cache.is_valid(99));
        assert!(cache.get(99).is_none());
    }

    #[test]
    fn clear_invalidates_cache() {
        let mut cache = RasterCache::new();
        let pixmap = Pixmap::new(10, 10).unwrap();
        cache.store(42, pixmap);
        cache.clear();

        assert!(!cache.is_valid(42));
    }

    #[test]
    fn dirty_tiles_first_frame_all_dirty() {
        let mut cache = RasterCache::new();
        let pm = Pixmap::new(128, 128).unwrap();
        let dirty = cache.compute_dirty_tiles(&pm);
        // 128/64 = 2×2 = 4 tiles, all dirty on first frame
        assert_eq!(dirty.len(), 4);
    }

    #[test]
    fn dirty_tiles_identical_frame_none_dirty() {
        let mut cache = RasterCache::new();
        let pm = Pixmap::new(128, 128).unwrap();
        cache.compute_dirty_tiles(&pm); // first frame
        let dirty = cache.compute_dirty_tiles(&pm); // same pixmap
        assert!(
            dirty.is_empty(),
            "identical frame should have 0 dirty tiles"
        );
    }

    #[test]
    fn dirty_tiles_detects_single_changed_tile() {
        let mut cache = RasterCache::new();
        let mut pm = Pixmap::new(128, 128).unwrap();
        cache.compute_dirty_tiles(&pm); // baseline

        // Modify one pixel in the top-left tile
        pm.data_mut()[0] = 255;
        let dirty = cache.compute_dirty_tiles(&pm);
        assert_eq!(dirty.len(), 1);
        assert_eq!(dirty[0].x, 0);
        assert_eq!(dirty[0].y, 0);
    }

    #[test]
    fn dirty_tiles_edge_tile_size() {
        let mut cache = RasterCache::new();
        // 100×100 → 2×2 tiles: (64, 64, 36, 36) are edge tiles
        let pm = Pixmap::new(100, 100).unwrap();
        let dirty = cache.compute_dirty_tiles(&pm);
        assert_eq!(dirty.len(), 4);

        // Bottom-right tile should have size 36×36
        let br = dirty.iter().find(|t| t.x == 64 && t.y == 64).unwrap();
        assert_eq!(br.width, 36);
        assert_eq!(br.height, 36);
    }
}
