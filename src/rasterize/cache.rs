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

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use tiny_skia::Pixmap;

/// Size of each tile for dirty tracking, in pixels.
pub const TILE_SIZE: usize = 64;

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
    /// Per-tile hashes of the previous frame for dirty detection.
    prev_tile_hashes: Vec<u64>,
    /// Width in tiles of the grid.
    tiles_x: usize,
    /// Height in tiles of the grid.
    tiles_y: usize,
}

impl RasterCache {
    /// Create a new, empty cache.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            hash: None,
            pixmap: None,
            prev_tile_hashes: Vec::new(),
            tiles_x: 0,
            tiles_y: 0,
        }
    }

    /// Check whether the cache contains a valid entry for the given content hash.
    #[must_use]
    pub fn is_valid(&self, content_hash: u64) -> bool {
        self.hash == Some(content_hash) && self.pixmap.is_some()
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
    /// This enables in-place re-rasterization via [`Rasterizer::rasterize_into`]
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
        self.hash = Some(content_hash);
        self.pixmap = Some(pixmap);
    }

    /// Invalidate the cache.
    pub fn clear(&mut self) {
        self.hash = None;
        self.pixmap = None;
        self.prev_tile_hashes.clear();
        self.tiles_x = 0;
        self.tiles_y = 0;
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
        let w = pixmap.width() as usize;
        let h = pixmap.height() as usize;
        let tx = w.div_ceil(TILE_SIZE);
        let ty = h.div_ceil(TILE_SIZE);
        let total = tx * ty;

        // Compute current tile hashes
        let mut current_hashes = Vec::with_capacity(total);
        let data = pixmap.data();

        for ty_idx in 0..ty {
            for tx_idx in 0..tx {
                let x0 = tx_idx * TILE_SIZE;
                let y0 = ty_idx * TILE_SIZE;
                let tw = TILE_SIZE.min(w - x0);
                let th = TILE_SIZE.min(h - y0);

                let mut hasher = DefaultHasher::new();
                for row in y0..(y0 + th) {
                    let start = (row * w + x0) * 4;
                    let end = start + tw * 4;
                    data[start..end].hash(&mut hasher);
                }
                current_hashes.push(hasher.finish());
            }
        }

        // Compare against previous
        let mut dirty = Vec::new();

        if self.tiles_x != tx || self.tiles_y != ty || self.prev_tile_hashes.len() != total {
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
            for (i, (&prev, &curr)) in self
                .prev_tile_hashes
                .iter()
                .zip(current_hashes.iter())
                .enumerate()
            {
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
        self.prev_tile_hashes = current_hashes;
        self.tiles_x = tx;
        self.tiles_y = ty;

        dirty
    }
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
        assert!(dirty.is_empty(), "identical frame should have 0 dirty tiles");
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
