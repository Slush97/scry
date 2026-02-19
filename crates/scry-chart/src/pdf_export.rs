// SPDX-License-Identifier: MIT OR Apache-2.0
//! PDF export for charts.
//!
//! Renders a chart to a single-page PDF document by first rasterizing to RGBA
//! pixels (reusing the existing PNG export pipeline), then embedding the result
//! as a deflate-compressed image XObject in a PDF page.
//!
//! # Feature flag
//!
//! This module is available only when the `pdf` feature is enabled:
//!
//! ```toml
//! scry-chart = { version = "0.7", features = ["pdf"] }
//! ```
//!
//! # Examples
//!
//! ```ignore
//! use scry_chart::prelude::*;
//!
//! let chart = Charts::line(&[1.0, 4.0, 2.0, 8.0])
//!     .title("Demo")
//!     .build();
//! scry_chart::pdf_export::save_pdf(&chart, 800, 500, "chart.pdf")?;
//! ```

use crate::chart::{Chart, Charts};
use crate::export::render_to_rgba;
use crate::subplot::SubplotGrid;
use pdf_writer::{Content, Finish, Name, Pdf, Ref, Rect};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Render a chart to a PDF byte buffer.
///
/// Returns a complete, single-page PDF document as `Vec<u8>`.
///
/// # Arguments
///
/// * `chart` – The chart to render
/// * `width` – Image width in pixels (also used as PDF page width in points)
/// * `height` – Image height in pixels (also used as PDF page height in points)
///
/// # Errors
///
/// Returns an error string if rendering or PDF construction fails.
pub fn render_to_pdf(chart: &Chart, width: u32, height: u32) -> Result<Vec<u8>, String> {
    let rgba = render_to_rgba(chart, width, height)?;
    build_pdf(&rgba, width, height)
}

/// Render a chart and save directly to a PDF file.
///
/// # Errors
///
/// Returns an error if rendering or file I/O fails.
pub fn save_pdf(
    chart: &Chart,
    width: u32,
    height: u32,
    path: impl AsRef<std::path::Path>,
) -> Result<(), String> {
    let data = render_to_pdf(chart, width, height)?;
    std::fs::write(path.as_ref(), data)
        .map_err(|e| format!("failed to write {}: {e}", path.as_ref().display()))
}

/// Render a subplot grid to a PDF byte buffer.
///
/// # Errors
///
/// Returns an error if rendering or PDF construction fails.
pub fn render_subplot_to_pdf(
    grid: &SubplotGrid,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    let rgba = crate::export::render_subplot_rgba(grid, width, height)?;
    build_pdf(&rgba, width, height)
}

/// Render a subplot grid and save directly to a PDF file.
///
/// # Errors
///
/// Returns an error if rendering or file I/O fails.
pub fn save_subplot_pdf(
    grid: &SubplotGrid,
    width: u32,
    height: u32,
    path: impl AsRef<std::path::Path>,
) -> Result<(), String> {
    let data = render_subplot_to_pdf(grid, width, height)?;
    std::fs::write(path.as_ref(), data)
        .map_err(|e| format!("failed to write {}: {e}", path.as_ref().display()))
}

// ---------------------------------------------------------------------------
// Internal: PDF construction
// ---------------------------------------------------------------------------

/// Strip alpha channel from RGBA data → RGB.
fn rgba_to_rgb(rgba: &[u8]) -> Vec<u8> {
    let pixel_count = rgba.len() / 4;
    let mut rgb = Vec::with_capacity(pixel_count * 3);
    for pixel in rgba.chunks_exact(4) {
        rgb.push(pixel[0]);
        rgb.push(pixel[1]);
        rgb.push(pixel[2]);
    }
    rgb
}

/// Deflate-compress a byte buffer.
fn deflate(data: &[u8]) -> Vec<u8> {
    miniz_oxide::deflate::compress_to_vec_zlib(data, 6)
}

/// Build a single-page PDF with an embedded raster image.
fn build_pdf(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let rgb = rgba_to_rgb(rgba);
    let compressed = deflate(&rgb);

    // Ref allocation: catalog=1, page_tree=2, page=3, contents=4, image=5
    let catalog_ref = Ref::new(1);
    let page_tree_ref = Ref::new(2);
    let page_ref = Ref::new(3);
    let content_ref = Ref::new(4);
    let image_ref = Ref::new(5);

    let mut writer = Pdf::new();

    // Catalog
    writer.catalog(catalog_ref).pages(page_tree_ref);

    // Page tree
    writer
        .pages(page_tree_ref)
        .kids([page_ref])
        .count(1);

    // Page — width×height points, references image as resource
    let mut page = writer.page(page_ref);
    page.media_box(Rect::new(0.0, 0.0, width as f32, height as f32));
    page.parent(page_tree_ref);
    page.contents(content_ref);

    // Resources: XObject dictionary mapping /Img to the image ref
    page.resources()
        .x_objects()
        .pair(Name(b"Img"), image_ref);
    page.finish();

    // Content stream: draw the image scaled to the full page
    let mut content = Content::new();
    content.save_state();
    // Transform matrix: scale image to page dimensions
    // The cm operator takes [a b c d e f] → maps [0,1]×[0,1] unit square
    // to [0,width]×[0,height].
    content.transform([width as f32, 0.0, 0.0, height as f32, 0.0, 0.0]);
    content.x_object(Name(b"Img"));
    content.restore_state();
    let content_bytes = content.finish();
    writer.stream(content_ref, &content_bytes);

    // Image XObject
    let mut image = writer.image_xobject(image_ref, &compressed);
    image.filter(pdf_writer::Filter::FlateDecode);
    image.width(width as i32);
    image.height(height as i32);
    image.color_space().device_rgb();
    image.bits_per_component(8);
    image.finish();

    Ok(writer.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_line_to_pdf() {
        let chart = Charts::line(&[1.0, 4.0, 2.0, 8.0, 5.0])
            .title("PDF Export Test")
            .build();
        let pdf_bytes = render_to_pdf(&chart, 400, 300).expect("PDF render should succeed");
        // Verify it's a valid PDF (starts with %PDF)
        assert!(
            pdf_bytes.starts_with(b"%PDF"),
            "output should be a valid PDF"
        );
        assert!(pdf_bytes.len() > 1000, "PDF should have meaningful content");
    }

    #[test]
    fn render_bar_to_pdf() {
        let chart = Charts::bar(
            vec!["A".into(), "B".into(), "C".into()],
            &[10.0, 25.0, 15.0],
        )
        .title("Bar PDF")
        .build();
        let pdf_bytes = render_to_pdf(&chart, 600, 400).expect("PDF render should succeed");
        assert!(pdf_bytes.starts_with(b"%PDF"));
    }

    #[test]
    fn save_and_read_pdf() {
        let chart = Charts::scatter(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0])
            .title("File Save Test")
            .build();
        let tmp_path = std::env::temp_dir().join("scry_test_chart.pdf");
        save_pdf(&chart, 400, 300, &tmp_path).expect("save_pdf should succeed");
        let data = std::fs::read(&tmp_path).expect("should read written PDF");
        assert!(data.starts_with(b"%PDF"));
        std::fs::remove_file(tmp_path).ok();
    }

    #[test]
    fn rgba_to_rgb_conversion() {
        let rgba = vec![255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 0];
        let rgb = rgba_to_rgb(&rgba);
        assert_eq!(rgb, vec![255, 0, 0, 0, 255, 0, 0, 0, 255]);
    }
}
