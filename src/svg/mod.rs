//! SVG rendering support via [`resvg`].
//!
//! This module provides [`SvgImage`], which parses SVG content and renders
//! it to a `tiny_skia::Pixmap` that can be displayed through the existing
//! pixel transport layer (Kitty, Sixel, Halfblock).
//!
//! # Example
//!
//! ```no_run
//! use scry_engine::svg::SvgImage;
//!
//! // Parse an SVG string
//! let svg = SvgImage::from_str(r#"
//!     <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
//!         <circle cx="50" cy="50" r="40" fill="red"/>
//!     </svg>
//! "#).unwrap();
//!
//! // Render to a pixmap at specific dimensions
//! let pixmap = svg.render(400, 400).unwrap();
//! ```

pub mod line_drawing;

use std::path::Path;
use tiny_skia::Pixmap;

/// A parsed SVG image ready for rasterization.
///
/// Wraps a `resvg::usvg::Tree` and provides convenient methods for rendering
/// to pixel buffers at arbitrary dimensions with correct aspect ratio
/// preservation.
pub struct SvgImage {
    tree: resvg::usvg::Tree,
}

impl SvgImage {
    /// Parse an SVG from a string.
    ///
    /// # Errors
    ///
    /// Returns an error if the SVG content is invalid or cannot be parsed.
    pub fn from_str(svg_content: &str) -> Result<Self, SvgError> {
        let opts = resvg::usvg::Options::default();
        let tree = resvg::usvg::Tree::from_str(svg_content, &opts)
            .map_err(|e| SvgError::Parse(e.to_string()))?;
        Ok(Self { tree })
    }

    /// Load and parse an SVG from a file path.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or the SVG is invalid.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, SvgError> {
        let content =
            std::fs::read_to_string(path.as_ref()).map_err(|e| SvgError::Io(e.to_string()))?;
        Self::from_str(&content)
    }

    /// The intrinsic width of the SVG (from its viewBox or width attribute).
    pub fn width(&self) -> f32 {
        self.tree.size().width()
    }

    /// The intrinsic height of the SVG (from its viewBox or height attribute).
    pub fn height(&self) -> f32 {
        self.tree.size().height()
    }

    /// The aspect ratio (width / height) of the SVG.
    pub fn aspect_ratio(&self) -> f32 {
        self.width() / self.height()
    }

    /// Access the underlying parsed `usvg::Tree`.
    ///
    /// This is useful for advanced operations like extracting individual
    /// path elements for animation (see [`line_drawing::SvgLineDrawing`]).
    pub const fn tree(&self) -> &resvg::usvg::Tree {
        &self.tree
    }

    /// Render the SVG to a pixmap at the given dimensions.
    ///
    /// The SVG is scaled uniformly to fit within the target dimensions
    /// while preserving aspect ratio (letterboxed if needed).
    ///
    /// # Errors
    ///
    /// Returns an error if the pixmap cannot be created (zero dimensions).
    pub fn render(&self, width: u32, height: u32) -> Result<Pixmap, SvgError> {
        if width == 0 || height == 0 {
            return Err(SvgError::InvalidDimensions);
        }

        let mut pixmap = Pixmap::new(width, height).ok_or(SvgError::InvalidDimensions)?;

        // Compute uniform scale to fit the SVG into the target area
        let scale_x = width as f32 / self.width();
        let scale_y = height as f32 / self.height();
        let scale = scale_x.min(scale_y);

        // Center the SVG within the target area
        let rendered_w = self.width() * scale;
        let rendered_h = self.height() * scale;
        let offset_x = (width as f32 - rendered_w) / 2.0;
        let offset_y = (height as f32 - rendered_h) / 2.0;

        let transform =
            tiny_skia::Transform::from_scale(scale, scale).post_translate(offset_x, offset_y);

        resvg::render(&self.tree, transform, &mut pixmap.as_mut());

        Ok(pixmap)
    }

    /// Render the SVG to a pixmap, filling the entire target area.
    ///
    /// The SVG is scaled non-uniformly to fill the exact dimensions.
    /// Aspect ratio is **not** preserved.
    ///
    /// # Errors
    ///
    /// Returns an error if the pixmap cannot be created.
    pub fn render_fill(&self, width: u32, height: u32) -> Result<Pixmap, SvgError> {
        if width == 0 || height == 0 {
            return Err(SvgError::InvalidDimensions);
        }

        let mut pixmap = Pixmap::new(width, height).ok_or(SvgError::InvalidDimensions)?;

        let scale_x = width as f32 / self.width();
        let scale_y = height as f32 / self.height();
        let transform = tiny_skia::Transform::from_scale(scale_x, scale_y);

        resvg::render(&self.tree, transform, &mut pixmap.as_mut());

        Ok(pixmap)
    }

    /// Render the SVG into an existing pixmap, avoiding allocation.
    ///
    /// Scales to fit the pixmap dimensions while preserving aspect ratio.
    pub fn render_into(&self, pixmap: &mut Pixmap) {
        pixmap.fill(tiny_skia::Color::TRANSPARENT);

        let width = pixmap.width();
        let height = pixmap.height();

        let scale_x = width as f32 / self.width();
        let scale_y = height as f32 / self.height();
        let scale = scale_x.min(scale_y);

        let rendered_w = self.width() * scale;
        let rendered_h = self.height() * scale;
        let offset_x = (width as f32 - rendered_w) / 2.0;
        let offset_y = (height as f32 - rendered_h) / 2.0;

        let transform =
            tiny_skia::Transform::from_scale(scale, scale).post_translate(offset_x, offset_y);

        resvg::render(&self.tree, transform, &mut pixmap.as_mut());
    }
}

impl std::fmt::Debug for SvgImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SvgImage")
            .field("width", &self.width())
            .field("height", &self.height())
            .finish()
    }
}

/// Errors that can occur during SVG parsing and rendering.
#[derive(Debug, thiserror::Error)]
pub enum SvgError {
    /// Failed to parse the SVG content.
    #[error("SVG parse error: {0}")]
    Parse(String),

    /// Failed to read the SVG file.
    #[error("I/O error: {0}")]
    Io(String),

    /// Invalid render dimensions (zero width or height).
    #[error("invalid dimensions: width and height must be non-zero")]
    InvalidDimensions,
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE_SVG: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="40" fill="red"/>
    </svg>"#;

    #[test]
    fn parse_simple_svg() {
        let svg = SvgImage::from_str(SIMPLE_SVG).unwrap();
        assert!((svg.width() - 100.0).abs() < 0.01);
        assert!((svg.height() - 100.0).abs() < 0.01);
    }

    #[test]
    fn render_produces_non_empty_pixmap() {
        let svg = SvgImage::from_str(SIMPLE_SVG).unwrap();
        let pixmap = svg.render(200, 200).unwrap();
        assert_eq!(pixmap.width(), 200);
        assert_eq!(pixmap.height(), 200);
        // Should have non-transparent pixels
        let has_content = pixmap.data().iter().any(|&b| b != 0);
        assert!(
            has_content,
            "rendered pixmap should contain non-transparent pixels"
        );
    }

    #[test]
    fn render_zero_dimensions_errors() {
        let svg = SvgImage::from_str(SIMPLE_SVG).unwrap();
        assert!(svg.render(0, 100).is_err());
        assert!(svg.render(100, 0).is_err());
    }

    #[test]
    fn render_into_reuses_pixmap() {
        let svg = SvgImage::from_str(SIMPLE_SVG).unwrap();
        let mut pixmap = Pixmap::new(150, 150).unwrap();
        svg.render_into(&mut pixmap);
        let has_content = pixmap.data().iter().any(|&b| b != 0);
        assert!(has_content);
    }

    #[test]
    fn invalid_svg_returns_error() {
        assert!(SvgImage::from_str("not svg at all").is_err());
    }
}
