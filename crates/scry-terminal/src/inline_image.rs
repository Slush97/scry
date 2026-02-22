// SPDX-License-Identifier: MIT OR Apache-2.0
//! Inline image rendering — decodes images and produces `RegionOverlay`s.
//!
//! Supports format detection, aspect-ratio preservation, and conversion
//! to scry-engine scenes for compositing in the terminal viewport.

use crate::compositor::OverlayRegion;
use crate::error::TerminalError;

/// Maximum image dimension (width or height) in pixels.
const MAX_IMAGE_DIM: u32 = 16_384;

/// Maximum total pixels (prevents decode bombs like 100000×100000 PNGs).
const MAX_IMAGE_PIXELS: u64 = 64_000_000;

/// Decoded image ready for overlay rendering.
#[derive(Debug)]
pub struct InlineImage {
    /// RGBA pixel data.
    rgba: Vec<u8>,
    /// Image width in pixels.
    width: u32,
    /// Image height in pixels.
    height: u32,
}

impl InlineImage {
    /// Decode an image from raw bytes (format auto-detected).
    ///
    /// Supports PNG, JPEG, GIF, and WebP.
    ///
    /// # Errors
    ///
    /// Returns an error if the image format is unrecognized or decoding fails.
    pub fn decode(data: &[u8]) -> Result<Self, TerminalError> {
        let img = image::load_from_memory(data)
            .map_err(|e| TerminalError::Compositor(format!("image decode failed: {e}")))?;
        let rgba_img = img.to_rgba8();
        let (width, height) = rgba_img.dimensions();
        if width > MAX_IMAGE_DIM
            || height > MAX_IMAGE_DIM
            || (width as u64) * (height as u64) > MAX_IMAGE_PIXELS
        {
            return Err(TerminalError::Compositor(format!(
                "image too large: {width}×{height} (max {MAX_IMAGE_DIM}×{MAX_IMAGE_DIM}, \
                 {MAX_IMAGE_PIXELS} pixels)"
            )));
        }
        Ok(Self {
            rgba: rgba_img.into_raw(),
            width,
            height,
        })
    }

    /// Decode an image from raw RGBA pixel data (already decoded).
    pub fn from_rgba(width: u32, height: u32, rgba: Vec<u8>) -> Result<Self, TerminalError> {
        if width > MAX_IMAGE_DIM
            || height > MAX_IMAGE_DIM
            || (width as u64) * (height as u64) > MAX_IMAGE_PIXELS
        {
            return Err(TerminalError::Compositor(format!(
                "image too large: {width}×{height} (max {MAX_IMAGE_DIM}×{MAX_IMAGE_DIM}, \
                 {MAX_IMAGE_PIXELS} pixels)"
            )));
        }
        let expected = (width as usize) * (height as usize) * 4;
        if rgba.len() != expected {
            return Err(TerminalError::Compositor(format!(
                "RGBA data length {} != expected {} ({}×{}×4)",
                rgba.len(),
                expected,
                width,
                height,
            )));
        }
        Ok(Self {
            rgba,
            width,
            height,
        })
    }

    /// Image width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Image height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Compute the cell region for this image, preserving aspect ratio.
    ///
    /// Given the available grid area and cell dimensions, calculates the
    /// largest region that fits within `max_cols × max_rows` while
    /// maintaining the original image aspect ratio.
    ///
    /// # Arguments
    ///
    /// * `cursor_col` — column where the image should start
    /// * `cursor_row` — row where the image should start
    /// * `max_cols` — maximum width in cells (typically grid cols - cursor_col)
    /// * `max_rows` — maximum height in cells (typically grid rows - cursor_row)
    /// * `cell_width` — cell width in pixels
    /// * `cell_height` — cell height in pixels
    pub fn compute_region(
        &self,
        cursor_col: u16,
        cursor_row: u16,
        max_cols: u16,
        max_rows: u16,
        cell_width: f32,
        cell_height: f32,
    ) -> OverlayRegion {
        if self.width == 0 || self.height == 0 || max_cols == 0 || max_rows == 0 {
            return OverlayRegion::new(cursor_col, cursor_row, 1, 1);
        }

        let img_aspect = self.width as f32 / self.height as f32;

        // Available space in pixels
        let avail_px_w = max_cols as f32 * cell_width;
        let avail_px_h = max_rows as f32 * cell_height;
        let avail_aspect = avail_px_w / avail_px_h;

        // Fit within available space preserving aspect ratio
        let (fit_px_w, fit_px_h) = if img_aspect > avail_aspect {
            // Image is wider than available space → constrain by width
            (avail_px_w, avail_px_w / img_aspect)
        } else {
            // Image is taller → constrain by height
            (avail_px_h * img_aspect, avail_px_h)
        };

        // Convert back to cell dimensions (round up to avoid clipping)
        let cols = ((fit_px_w / cell_width).ceil() as u16).max(1).min(max_cols);
        let rows = ((fit_px_h / cell_height).ceil() as u16).max(1).min(max_rows);

        OverlayRegion::new(cursor_col, cursor_row, cols, rows)
    }

    /// Create a scry-engine `PixelCanvas` scene for this image.
    ///
    /// The scene draws the image at (0, 0) filling the given pixel
    /// dimensions. The compositor will position it correctly via the
    /// region overlay system.
    pub fn to_scene(&self, pixel_width: u32, pixel_height: u32) -> scry_engine::scene::PixelCanvas {
        use scry_engine::prelude::*;

        let image_data =
            scry_engine::scene::command::ImageData::new(self.width, self.height, self.rgba.clone());

        PixelCanvas::new(pixel_width, pixel_height)
            .image(image_data, 0.0, 0.0)
            .opacity(1.0)
            .done()
    }

    /// Convenience: decode an image and produce a region + scene in one step.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails.
    pub fn decode_to_overlay(
        data: &[u8],
        cursor_col: u16,
        cursor_row: u16,
        max_cols: u16,
        max_rows: u16,
        cell_width: f32,
        cell_height: f32,
    ) -> Result<(OverlayRegion, scry_engine::scene::PixelCanvas), TerminalError> {
        let img = Self::decode(data)?;
        let region = img.compute_region(
            cursor_col,
            cursor_row,
            max_cols,
            max_rows,
            cell_width,
            cell_height,
        );
        let pixel_w = (region.width as f32 * cell_width) as u32;
        let pixel_h = (region.height as f32 * cell_height) as u32;
        let scene = img.to_scene(pixel_w.max(1), pixel_h.max(1));
        Ok((region, scene))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_region_landscape_image() {
        let img = InlineImage {
            rgba: vec![0u8; 800 * 400 * 4],
            width: 800,
            height: 400,
        };

        // 80 cols × 24 rows, 8px × 16px cells
        let region = img.compute_region(0, 0, 80, 24, 8.0, 16.0);
        // 800/400 = 2.0 aspect ratio
        // Available: 640px × 384px → aspect = 1.67
        // Image wider → constrain by width: 640px wide, 320px tall
        // Cells: 640/8 = 80 cols, 320/16 = 20 rows
        assert_eq!(region.col, 0);
        assert_eq!(region.row, 0);
        assert_eq!(region.width, 80);
        assert_eq!(region.height, 20);
    }

    #[test]
    fn compute_region_portrait_image() {
        let img = InlineImage {
            rgba: vec![0u8; 400 * 800 * 4],
            width: 400,
            height: 800,
        };

        // 80 cols × 24 rows, 8px × 16px cells
        let region = img.compute_region(0, 0, 80, 24, 8.0, 16.0);
        // 400/800 = 0.5 aspect ratio
        // Available: 640px × 384px → aspect = 1.67
        // Image taller → constrain by height: 192px wide, 384px tall
        // Cells: 192/8 = 24 cols, 384/16 = 24 rows
        assert_eq!(region.col, 0);
        assert_eq!(region.row, 0);
        assert_eq!(region.width, 24);
        assert_eq!(region.height, 24);
    }

    #[test]
    fn compute_region_with_cursor_offset() {
        let img = InlineImage {
            rgba: vec![0u8; 100 * 100 * 4],
            width: 100,
            height: 100,
        };

        let region = img.compute_region(10, 5, 40, 10, 8.0, 16.0);
        assert_eq!(region.col, 10);
        assert_eq!(region.row, 5);
        // 100/100 = 1.0 aspect ratio
        // Available: 320px × 160px → aspect = 2.0
        // Square image → constrain by height: 160px × 160px
        // Cells: 160/8 = 20 cols, 160/16 = 10 rows
        assert_eq!(region.width, 20);
        assert_eq!(region.height, 10);
    }

    #[test]
    fn compute_region_zero_dimensions() {
        let img = InlineImage {
            rgba: Vec::new(),
            width: 0,
            height: 0,
        };

        let region = img.compute_region(0, 0, 80, 24, 8.0, 16.0);
        assert_eq!(region.width, 1);
        assert_eq!(region.height, 1);
    }

    #[test]
    fn from_rgba_validates_size() {
        let result = InlineImage::from_rgba(10, 10, vec![0u8; 100]);
        assert!(result.is_err());

        let result = InlineImage::from_rgba(10, 10, vec![0u8; 400]);
        assert!(result.is_ok());
    }

    #[test]
    fn from_rgba_rejects_oversized_dimensions() {
        // Exceeds MAX_IMAGE_DIM (16384)
        let result = InlineImage::from_rgba(20_000, 100, vec![]);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("too large"), "expected 'too large' error, got: {err}");
    }

    #[test]
    fn from_rgba_rejects_excessive_pixels() {
        // 10000 × 10000 = 100M pixels > MAX_IMAGE_PIXELS (64M)
        let result = InlineImage::from_rgba(10_000, 10_000, vec![]);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("too large"), "expected 'too large' error, got: {err}");
    }
}
