//! Content-hash caching for rasterized scenes.
//!
//! The [`RasterCache`] stores the last rasterized result and its content hash,
//! allowing the rendering pipeline to skip rasterization when the scene has
//! not changed between frames.

use tiny_skia::Pixmap;

/// Content-addressable cache for a single rasterized scene.
///
/// Stores the content hash and pixmap from the last rasterization. When the
/// scene's content hash matches the cached value, rasterization is skipped
/// entirely.
#[derive(Debug)]
pub struct RasterCache {
    /// Content hash of the last rasterized scene.
    hash: Option<u64>,
    /// The rasterized pixel buffer.
    pixmap: Option<Pixmap>,
}

impl RasterCache {
    /// Create a new, empty cache.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            hash: None,
            pixmap: None,
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
}
