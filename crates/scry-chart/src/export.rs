// SPDX-License-Identifier: MIT OR Apache-2.0
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
// DPI helper
// ---------------------------------------------------------------------------

/// Default export DPI.
const BASE_DPI: u32 = 144;

/// Extract the DPI from a chart's config.
fn chart_dpi(chart: &Chart) -> u32 {
    chart.dpi()
}

/// Scale pixel dimensions by the chart's DPI relative to the base DPI (144).
fn dpi_scale(width: u32, height: u32, dpi: u32) -> (u32, u32) {
    if dpi == BASE_DPI || dpi == 0 {
        return (width, height);
    }
    let scale = dpi as f64 / BASE_DPI as f64;
    let w = ((width as f64) * scale).round() as u32;
    let h = ((height as f64) * scale).round() as u32;
    (w.max(1), h.max(1))
}

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
/// let chart = Charts::line(&[1.0, 4.0, 2.0, 8.0]).title("Demo").build();
/// let png_bytes = scry_chart::export::render_to_png(&chart, 800, 500)?;
/// std::fs::write("chart.png", png_bytes)?;
/// ```
pub fn render_to_png(chart: &Chart, width: u32, height: u32) -> Result<Vec<u8>, String> {
    let dpi = chart_dpi(chart);
    let (w, h) = dpi_scale(width, height, dpi);
    let rgba = render_to_rgba_raw(chart, w, h)?;
    let pixmap = tiny_skia::Pixmap::from_vec(rgba, tiny_skia::IntSize::from_wh(w, h).unwrap())
        .ok_or("failed to create pixmap from RGBA data")?;
    pixmap
        .encode_png()
        .map_err(|e| format!("PNG encoding failed: {e}"))
}

/// Render a chart to raw RGBA pixel data.
///
/// Returns a `Vec<u8>` of length `width * height * 4` in RGBA order.
/// Dimensions are scaled by the chart's DPI setting.
pub fn render_to_rgba(chart: &Chart, width: u32, height: u32) -> Result<Vec<u8>, String> {
    let dpi = chart_dpi(chart);
    let (w, h) = dpi_scale(width, height, dpi);
    render_to_rgba_raw(chart, w, h)
}

/// Internal render — operates on pre-scaled dimensions.
///
/// Uses [`RasterPipeline`] which auto-selects the GPU backend when
/// available and falls back to CPU (tiny-skia) otherwise.
fn render_to_rgba_raw(chart: &Chart, width: u32, height: u32) -> Result<Vec<u8>, String> {
    let rendered = layout::render_chart(chart, width, height);
    let mut pixmap = scry_engine::rasterize::RasterPipeline::new()
        .rasterize(&rendered.canvas)
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
    let dpi = chart_dpi(chart);
    let (w, h) = dpi_scale(width, height, dpi);
    let rendered = layout::render_chart(chart, w, h);
    let mut pixmap = scry_engine::rasterize::RasterPipeline::new()
        .rasterize(&rendered.canvas)
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
                // Negate the angle so positive rotation_deg = counter-clockwise,
                // matching the SVG convention and standard math convention.
                // For Y-axis labels (90°), this produces text reading bottom-to-top.
                let rad = (-overlay.rotation_deg).to_radians();
                (rad.sin(), rad.cos())
            } else {
                (0.0, 1.0)
            };

            // Anchor point for rotation: use the visual center of the text
            // so the rotation pivots around the text's midpoint, producing
            // consistent results regardless of text length.
            let anchor_x = match overlay.align {
                TextAlign::Left => overlay.x_px + total_width / 2.0,
                TextAlign::Center => overlay.x_px,
                TextAlign::Right => overlay.x_px - total_width / 2.0,
            };
            let anchor_y = overlay.y_px + ascent * 0.25;

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

// ---------------------------------------------------------------------------
// Subplot / multi-panel export
// ---------------------------------------------------------------------------

use crate::subplot::SubplotGrid;

/// Render a subplot grid to a PNG byte buffer.
///
/// Each cell is rendered independently at its calculated sub-dimensions,
/// then composited onto a single output pixmap.
///
/// # Arguments
///
/// * `grid` – The subplot grid
/// * `width` – Total output width in pixels
/// * `height` – Total output height in pixels
///
/// # Errors
///
/// Returns an error if any cell rendering or PNG encoding fails.
pub fn render_subplot_to_png(
    grid: &SubplotGrid,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    let rgba = render_subplot_to_rgba_raw(grid, width, height)?;
    let pixmap =
        tiny_skia::Pixmap::from_vec(rgba, tiny_skia::IntSize::from_wh(width, height).unwrap())
            .ok_or("failed to create pixmap from RGBA data")?;
    pixmap
        .encode_png()
        .map_err(|e| format!("PNG encoding failed: {e}"))
}

/// Render a subplot grid and save directly to a PNG file.
///
/// # Errors
///
/// Returns an error if rendering or file I/O fails.
pub fn save_subplot_png(
    grid: &SubplotGrid,
    width: u32,
    height: u32,
    path: impl AsRef<std::path::Path>,
) -> Result<(), String> {
    let data = render_subplot_to_png(grid, width, height)?;
    std::fs::write(path.as_ref(), data)
        .map_err(|e| format!("failed to write {}: {e}", path.as_ref().display()))
}

/// Render a subplot grid to raw RGBA pixel data.
///
/// # Errors
///
/// Returns an error if rendering fails.
pub fn render_subplot_rgba(grid: &SubplotGrid, width: u32, height: u32) -> Result<Vec<u8>, String> {
    render_subplot_to_rgba_raw(grid, width, height)
}

/// Internal: render a subplot grid to raw RGBA pixel data.
fn render_subplot_to_rgba_raw(
    grid: &SubplotGrid,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    let rows = grid.rows as u32;
    let cols = grid.cols as u32;
    let gap = grid.gap;

    // Reserve space for an optional title at the top
    let title_h = if grid.title.is_some() { 32_u32 } else { 0 };

    // Calculate cell dimensions
    let total_h_gap = gap * (cols.saturating_sub(1));
    let total_v_gap = gap * (rows.saturating_sub(1));
    let cell_w = (width.saturating_sub(total_h_gap)) / cols;
    let cell_h = (height.saturating_sub(title_h).saturating_sub(total_v_gap)) / rows;

    if cell_w == 0 || cell_h == 0 {
        return Err("Subplot grid dimensions too small for the given rows/cols/gap".into());
    }

    // Create master RGBA buffer filled with background color
    let bg = grid.background;
    let bg_r = (bg.r * 255.0) as u8;
    let bg_g = (bg.g * 255.0) as u8;
    let bg_b = (bg.b * 255.0) as u8;
    let bg_a = (bg.a * 255.0) as u8;

    let stride = width as usize * 4;
    let mut master = vec![0u8; stride * height as usize];

    // Fill with background
    for pixel in master.chunks_exact_mut(4) {
        pixel[0] = bg_r;
        pixel[1] = bg_g;
        pixel[2] = bg_b;
        pixel[3] = bg_a;
    }

    // --- Shared axis domain unification ---
    // When axes are shared, we compute the union of all cell data extents
    // and inject the unified domain into each cell's config. Non-primary
    // cells have their axis labels suppressed via NullFormatter.
    let share_x = grid.shared_axes.shares_x();
    let share_y = grid.shared_axes.shares_y();

    // Clone the grid temporarily to mutate cell configs
    let mut local_cells: Vec<Option<Chart>> = grid.cells.clone();

    if share_x || share_y {
        // Compute global extent across all populated cells
        let mut gx_min = f64::INFINITY;
        let mut gx_max = f64::NEG_INFINITY;
        let mut gy_min = f64::INFINITY;
        let mut gy_max = f64::NEG_INFINITY;

        for cell in local_cells.iter().flatten() {
            if let Some((xn, xx, yn, yx)) = cell.data_extent() {
                if xn < gx_min {
                    gx_min = xn;
                }
                if xx > gx_max {
                    gx_max = xx;
                }
                if yn < gy_min {
                    gy_min = yn;
                }
                if yx > gy_max {
                    gy_max = yx;
                }
            }
        }

        // Apply unified domains and suppress labels on non-primary cells
        let null_fmt: std::sync::Arc<dyn crate::formatter::TickFormatter> =
            std::sync::Arc::new(crate::formatter::NullFormatter);

        for (i, cell_opt) in local_cells.iter_mut().enumerate() {
            if let Some(chart) = cell_opt {
                let row = i / cols as usize;
                let col = i % cols as usize;

                if let Some(cfg) = chart.config_mut() {
                    if share_x && gx_min.is_finite() && gx_max.is_finite() {
                        cfg.axes.x_range = Some((gx_min, gx_max));
                        // Only bottom row shows X labels
                        if row + 1 < rows as usize {
                            cfg.titles.x_label = None;
                            cfg.ticks.x_tick_formatter = Some(null_fmt.clone());
                        }
                    }

                    if share_y && gy_min.is_finite() && gy_max.is_finite() {
                        cfg.axes.y_range = Some((gy_min, gy_max));
                        // Only leftmost column shows Y labels
                        if col > 0 {
                            cfg.titles.y_label = None;
                            cfg.ticks.y_tick_formatter = Some(null_fmt.clone());
                        }
                    }
                }
            }
        }
    }

    // Render each cell and blit onto master
    for (i, chart_opt) in local_cells.iter().enumerate() {
        let row = i / cols as usize;
        let col = i % cols as usize;
        let x_off = col as u32 * (cell_w + gap);
        let y_off = title_h + row as u32 * (cell_h + gap);

        if let Some(chart) = chart_opt {
            let cell_rgba = render_to_rgba_raw(chart, cell_w, cell_h)?;

            // Blit cell RGBA onto master at (x_off, y_off)
            let cell_stride = cell_w as usize * 4;
            for cy in 0..cell_h as usize {
                let dst_y = y_off as usize + cy;
                if dst_y >= height as usize {
                    break;
                }
                let src_start = cy * cell_stride;
                let dst_start = dst_y * stride + x_off as usize * 4;
                let copy_len = cell_stride.min(stride - x_off as usize * 4);
                master[dst_start..dst_start + copy_len]
                    .copy_from_slice(&cell_rgba[src_start..src_start + copy_len]);
            }
        }
        // Empty cells keep the background fill
    }

    // Stamp grid title if present
    if let Some(ref title) = grid.title {
        let mut title_pixmap = tiny_skia::Pixmap::from_vec(
            master,
            tiny_skia::IntSize::from_wh(width, height).unwrap(),
        )
        .ok_or("failed to create master pixmap")?;

        let overlay = TextOverlay {
            x_px: width as f32 / 2.0,
            y_px: 4.0,
            text: title.clone(),
            color: scry_engine::style::Color::from_rgba8(230, 230, 240, 255),
            align: TextAlign::Center,
            font_size: 18.0,
            bold: true,
            rotation_deg: 0.0,
        };
        stamp_text_overlays(&mut title_pixmap, &[overlay]);
        master = title_pixmap.take();
    }

    Ok(master)
}
