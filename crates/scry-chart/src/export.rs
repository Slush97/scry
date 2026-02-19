// SPDX-License-Identifier: MIT OR Apache-2.0
//! PNG image export for charts.
//!
//! Provides [`render_to_png`] and [`render_to_rgba`] functions for converting
//! any [`Chart`] into a standalone image — useful for docs, CI artifacts,
//! headless testing, and embedding in web pages.
//!
//! Text rendering is delegated to the engine's `DrawCommand::Text` pipeline.

use crate::chart::Chart;
use crate::layout::{self, RenderedChart};

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
    let pixmap = scry_engine::rasterize::RasterPipeline::new()
        .rasterize(&rendered.canvas)
        .map_err(|e| format!("rasterization failed: {e}"))?;

    // Text is now rasterized by the engine via DrawCommand::Text commands
    // in the canvas — no separate stamp_text_overlays step needed.

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
    let pixmap = scry_engine::rasterize::RasterPipeline::new()
        .rasterize(&rendered.canvas)
        .map_err(|e| format!("rasterization failed: {e}"))?;

    let png = pixmap
        .encode_png()
        .map_err(|e| format!("PNG encoding failed: {e}"))?;
    Ok((png, rendered))
}

// ---------------------------------------------------------------------------
// Subplot / multi-panel export
// ---------------------------------------------------------------------------

/// Embedded bold font data for subplot grid titles.
static FONT_DATA_BOLD: &[u8] = include_bytes!("fonts/Inter-Bold.ttf");

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
        use scry_engine::scene::command::{FontData, TextAlign as EngineTextAlign};

        let bold_fd = FontData::new(FONT_DATA_BOLD.to_vec());
        let font_size = 18.0_f32;
        let color = scry_engine::style::Color::from_rgba8(230, 230, 240, 255);

        // Measure text to compute baseline y (top padding + ascent)
        let metrics =
            scry_engine::rasterize::skia::text::measure_text(title, Some(&bold_fd), font_size);
        let baseline_y = 4.0 + metrics.ascent;

        // Render title text into a small canvas via the engine
        let title_canvas = scry_engine::scene::PixelCanvas::new(width, title_h)
            .text(title, width as f32 / 2.0, baseline_y)
            .size(font_size)
            .color(color)
            .font(bold_fd)
            .align(EngineTextAlign::Center)
            .done();

        let title_pixmap = scry_engine::rasterize::skia::Rasterizer::rasterize(&title_canvas)
            .map_err(|e| format!("title rasterization failed: {e}"))?;

        // Alpha-composite title pixels onto master buffer
        let title_data = title_pixmap.data();
        for ty in 0..title_h as usize {
            for tx in 0..width as usize {
                let src_idx = (ty * width as usize + tx) * 4;
                let dst_idx = (ty * width as usize + tx) * 4;
                let sa = title_data[src_idx + 3] as u32;
                if sa == 0 {
                    continue;
                }
                let inv = 255 - sa;
                master[dst_idx] =
                    ((title_data[src_idx] as u32 * sa + master[dst_idx] as u32 * inv) / 255) as u8;
                master[dst_idx + 1] = ((title_data[src_idx + 1] as u32 * sa
                    + master[dst_idx + 1] as u32 * inv)
                    / 255) as u8;
                master[dst_idx + 2] = ((title_data[src_idx + 2] as u32 * sa
                    + master[dst_idx + 2] as u32 * inv)
                    / 255) as u8;
                master[dst_idx + 3] =
                    (sa + master[dst_idx + 3] as u32 * inv / 255).min(255) as u8;
            }
        }
    }

    Ok(master)
}
