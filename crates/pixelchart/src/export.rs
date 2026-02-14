//! PNG image export for charts.
//!
//! Provides [`render_to_png`] and [`render_to_rgba`] functions for converting
//! any [`Chart`] into a standalone image — useful for docs, CI artifacts,
//! headless testing, and embedding in web pages.
//!
//! Text overlays (titles, axis labels, tick labels) are burned into the image
//! using `fontdue` for scalable, anti-aliased glyph rasterization with
//! size hierarchy (title > labels > ticks).

use crate::chart::Chart;
use crate::layout::{self, RenderedChart, TextAlign, TextOverlay};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Render a chart to a PNG byte buffer.
///
/// # Arguments
///
/// * `chart` – The chart to render
/// * `width` – Image width in pixels
/// * `height` – Image height in pixels
///
/// # Errors
///
/// Returns an error string if pixmap creation or PNG encoding fails.
///
/// # Example
///
/// ```ignore
/// let chart = Chart::line(&[1.0, 4.0, 2.0, 8.0]).title("Demo").build();
/// let png_bytes = pixelchart::export::render_to_png(&chart, 800, 500)?;
/// std::fs::write("chart.png", png_bytes)?;
/// ```
pub fn render_to_png(chart: &Chart, width: u32, height: u32) -> Result<Vec<u8>, String> {
    let rgba = render_to_rgba(chart, width, height)?;
    let pixmap =
        tiny_skia::Pixmap::from_vec(rgba, tiny_skia::IntSize::from_wh(width, height).unwrap())
            .ok_or("failed to create pixmap from RGBA data")?;
    pixmap
        .encode_png()
        .map_err(|e| format!("PNG encoding failed: {e}"))
}

/// Render a chart to raw RGBA pixel data.
///
/// Returns a `Vec<u8>` of length `width * height * 4` in RGBA order.
pub fn render_to_rgba(chart: &Chart, width: u32, height: u32) -> Result<Vec<u8>, String> {
    let rendered = layout::render_chart(chart, width, height);
    let mut pixmap = ratatui_pixelcanvas::rasterize::Rasterizer::rasterize(&rendered.canvas)
        .map_err(|e| format!("rasterization failed: {e}"))?;

    // Burn text overlays onto the pixmap using fontdue
    stamp_text_overlays(&mut pixmap, &rendered.text_overlays);

    Ok(pixmap.data().to_vec())
}

/// Render a chart and save directly to a PNG file.
///
/// Convenience wrapper around [`render_to_png`].
///
/// # Errors
///
/// Returns an error if rendering or file I/O fails.
pub fn save_png(
    chart: &Chart,
    width: u32,
    height: u32,
    path: impl AsRef<std::path::Path>,
) -> Result<(), String> {
    let data = render_to_png(chart, width, height)?;
    std::fs::write(path.as_ref(), data)
        .map_err(|e| format!("failed to write {}: {e}", path.as_ref().display()))
}

/// Render a chart, returning the PNG bytes and the `RenderedChart` metadata
/// (plot area, scales, series points) for further analysis.
pub fn render_to_png_with_metadata(
    chart: &Chart,
    width: u32,
    height: u32,
) -> Result<(Vec<u8>, RenderedChart), String> {
    let rendered = layout::render_chart(chart, width, height);
    let mut pixmap = ratatui_pixelcanvas::rasterize::Rasterizer::rasterize(&rendered.canvas)
        .map_err(|e| format!("rasterization failed: {e}"))?;

    stamp_text_overlays(&mut pixmap, &rendered.text_overlays);

    let png = pixmap
        .encode_png()
        .map_err(|e| format!("PNG encoding failed: {e}"))?;
    Ok((png, rendered))
}

// ---------------------------------------------------------------------------
// Fontdue-powered text stamping
// ---------------------------------------------------------------------------

/// Embedded font data: Liberation Sans (SIL OFL 1.1 licensed, freely redistributable).
/// Falls back to fontdue's built-in if not available.
static FONT_DATA: &[u8] = include_bytes!("fonts/Inter-Regular.ttf");
static FONT_DATA_BOLD: &[u8] = include_bytes!("fonts/Inter-Bold.ttf");

/// Cache for fontdue Font objects (one regular, one bold).
struct FontCache {
    regular: fontdue::Font,
    bold: fontdue::Font,
}

impl FontCache {
    fn new() -> Self {
        let settings = fontdue::FontSettings::default();
        let regular = fontdue::Font::from_bytes(FONT_DATA, settings)
            .unwrap_or_else(|e| panic!("Failed to parse regular font: {e}"));
        let bold = fontdue::Font::from_bytes(FONT_DATA_BOLD, settings)
            .unwrap_or_else(|e| panic!("Failed to parse bold font: {e}"));
        Self { regular, bold }
    }

    fn font(&self, bold: bool) -> &fontdue::Font {
        if bold {
            &self.bold
        } else {
            &self.regular
        }
    }
}

/// Thread-local font cache to avoid re-parsing font data on every render.
fn with_font_cache<R>(f: impl FnOnce(&FontCache) -> R) -> R {
    use std::cell::RefCell;
    thread_local! {
        static CACHE: RefCell<Option<FontCache>> = const { RefCell::new(None) };
    }
    CACHE.with(|cell| {
        let mut opt = cell.borrow_mut();
        if opt.is_none() {
            *opt = Some(FontCache::new());
        }
        f(opt.as_ref().unwrap())
    })
}

/// Stamp text overlays onto a pixmap using fontdue for anti-aliased glyph rendering.
fn stamp_text_overlays(pixmap: &mut tiny_skia::Pixmap, overlays: &[TextOverlay]) {
    let pw = pixmap.width();
    let ph = pixmap.height();

    with_font_cache(|cache| {
        for overlay in overlays {
            let font = cache.font(overlay.bold);
            let size = overlay.font_size;
            let text = &overlay.text;
            let color = overlay.color;

            // Pre-rasterize all glyphs and measure total width
            let mut glyphs: Vec<(fontdue::Metrics, Vec<u8>)> = Vec::with_capacity(text.len());
            let mut total_width = 0.0_f32;

            for ch in text.chars() {
                let (metrics, bitmap) = font.rasterize(ch, size);
                total_width += metrics.advance_width;
                glyphs.push((metrics, bitmap));
            }

            // Compute baseline position
            // fontdue metrics: ymin is the descent (negative for glyphs that hang below baseline)
            let line_metrics = font.horizontal_line_metrics(size);
            let ascent = line_metrics.map_or(size * 0.8, |m| m.ascent);

            let x_start = match overlay.align {
                TextAlign::Left => overlay.x_px,
                TextAlign::Center => overlay.x_px - total_width / 2.0,
                TextAlign::Right => overlay.x_px - total_width,
            };

            let baseline_y = overlay.y_px + ascent * 0.5;

            let r = (color.r * 255.0) as u8;
            let g = (color.g * 255.0) as u8;
            let b = (color.b * 255.0) as u8;
            let text_alpha = color.a;

            let has_rotation = overlay.rotation_deg.abs() > 0.01;
            let (sin_a, cos_a) = if has_rotation {
                let rad = overlay.rotation_deg.to_radians();
                (rad.sin(), rad.cos())
            } else {
                (0.0, 1.0)
            };

            // Anchor point for rotation — the overlay's position
            let anchor_x = overlay.x_px;
            let anchor_y = overlay.y_px;

            let mut cursor_x = x_start;

            for (metrics, bitmap) in &glyphs {
                // Glyph origin in un-rotated space
                let gx_f = cursor_x + metrics.xmin as f32;
                let gy_f = baseline_y - metrics.height as f32 - metrics.ymin as f32;

                let data = pixmap.data_mut();

                for row in 0..metrics.height {
                    for col in 0..metrics.width {
                        let coverage = bitmap[row * metrics.width + col];
                        if coverage == 0 {
                            continue;
                        }

                        // Position of this pixel in the un-rotated frame
                        let src_x = gx_f + col as f32;
                        let src_y = gy_f + row as f32;

                        // Apply rotation around anchor if needed
                        let (px, py) = if has_rotation {
                            let dx = src_x - anchor_x;
                            let dy = src_y - anchor_y;
                            let rx = dx * cos_a - dy * sin_a + anchor_x;
                            let ry = dx * sin_a + dy * cos_a + anchor_y;
                            (rx as i32, ry as i32)
                        } else {
                            (src_x as i32, src_y as i32)
                        };

                        if px < 0 || py < 0 || (px as u32) >= pw || (py as u32) >= ph {
                            continue;
                        }

                        let idx = ((py as u32) * pw + px as u32) as usize * 4;
                        let sa = ((coverage as f32 / 255.0) * text_alpha * 255.0) as u32;
                        let inv = 255 - sa;

                        data[idx] = ((r as u32 * sa + data[idx] as u32 * inv) / 255) as u8;
                        data[idx + 1] = ((g as u32 * sa + data[idx + 1] as u32 * inv) / 255) as u8;
                        data[idx + 2] = ((b as u32 * sa + data[idx + 2] as u32 * inv) / 255) as u8;
                        data[idx + 3] = (sa + data[idx + 3] as u32 * inv / 255).min(255) as u8;
                    }
                }

                cursor_x += metrics.advance_width;
            }
        }
    });
}
