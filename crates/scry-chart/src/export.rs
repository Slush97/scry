// SPDX-License-Identifier: MIT OR Apache-2.0
//! PNG image export for charts.
//!
//! Provides [`render_to_png`] and [`render_to_rgba`] functions for converting
//! any [`Chart`] into a standalone image — useful for docs, CI artifacts,
//! headless testing, and embedding in web pages.
//!
//! Text rendering is delegated to the engine's `DrawCommand::Text` pipeline.

use crate::chart::Chart;
use crate::error::ChartError;
use crate::layout::{self, RenderedChart};

// ---------------------------------------------------------------------------
// DPI helper
// ---------------------------------------------------------------------------

/// Default export DPI.
const BASE_DPI: u32 = 144;

/// Height reserved for a subplot grid title row, in pixels.
const SUBPLOT_TITLE_HEIGHT: u32 = 32;

/// Font size for subplot grid titles, in pixels.
const SUBPLOT_TITLE_FONT_SIZE: f32 = 18.0;

/// Top padding above the subplot grid title text, in pixels.
const SUBPLOT_TITLE_TOP_PAD: f32 = 4.0;

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
/// Returns [`ChartError::Render`] if pixmap creation or PNG encoding fails.
///
/// # Example
///
/// ```ignore
/// let chart = Charts::line(&[1.0, 4.0, 2.0, 8.0]).title("Demo").build();
/// let png_bytes = scry_chart::export::render_to_png(&chart, 800, 500)?;
/// std::fs::write("chart.png", png_bytes)?;
/// ```
pub fn render_to_png(chart: &Chart, width: u32, height: u32) -> Result<Vec<u8>, ChartError> {
    let dpi = chart_dpi(chart);
    let (w, h) = dpi_scale(width, height, dpi);
    let rgba = render_to_rgba_raw(chart, w, h)?;
    let pixmap = tiny_skia::Pixmap::from_vec(rgba, tiny_skia::IntSize::from_wh(w, h).unwrap())
        .ok_or_else(|| ChartError::Render("failed to create pixmap from RGBA data".into()))?;
    pixmap
        .encode_png()
        .map_err(|e| ChartError::Render(format!("PNG encoding failed: {e}")))
}

/// Render a chart to raw RGBA pixel data.
///
/// Returns a `Vec<u8>` of length `width * height * 4` in RGBA order.
/// Dimensions are scaled by the chart's DPI setting.
pub fn render_to_rgba(chart: &Chart, width: u32, height: u32) -> Result<Vec<u8>, ChartError> {
    let dpi = chart_dpi(chart);
    let (w, h) = dpi_scale(width, height, dpi);
    render_to_rgba_raw(chart, w, h)
}

/// Internal render — operates on pre-scaled dimensions.
///
/// Uses [`RasterPipeline`] which auto-selects the GPU backend when
/// available and falls back to CPU (tiny-skia) otherwise.
fn render_to_rgba_raw(chart: &Chart, width: u32, height: u32) -> Result<Vec<u8>, ChartError> {
    let rendered = layout::render_chart(chart, width, height);
    let pixmap = scry_engine::rasterize::RasterPipeline::new()
        .rasterize(&rendered.canvas)
        .map_err(|e| ChartError::Render(format!("rasterization failed: {e}")))?;

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
/// Returns [`ChartError::Render`] if rendering fails, or [`ChartError::Io`]
/// if writing the file fails.
pub fn save_png(
    chart: &Chart,
    width: u32,
    height: u32,
    path: impl AsRef<std::path::Path>,
) -> Result<(), ChartError> {
    let data = render_to_png(chart, width, height)?;
    std::fs::write(path.as_ref(), data)?;
    Ok(())
}

/// Render a chart, returning the PNG bytes and the `RenderedChart` metadata
/// (plot area, scales, series points) for further analysis.
pub fn render_to_png_with_metadata(
    chart: &Chart,
    width: u32,
    height: u32,
) -> Result<(Vec<u8>, RenderedChart), ChartError> {
    let dpi = chart_dpi(chart);
    let (w, h) = dpi_scale(width, height, dpi);
    let rendered = layout::render_chart(chart, w, h);
    let pixmap = scry_engine::rasterize::RasterPipeline::new()
        .rasterize(&rendered.canvas)
        .map_err(|e| ChartError::Render(format!("rasterization failed: {e}")))?;

    let png = pixmap
        .encode_png()
        .map_err(|e| ChartError::Render(format!("PNG encoding failed: {e}")))?;
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
) -> Result<Vec<u8>, ChartError> {
    let rgba = render_subplot_to_rgba_raw(grid, width, height)?;
    let pixmap =
        tiny_skia::Pixmap::from_vec(rgba, tiny_skia::IntSize::from_wh(width, height).unwrap())
            .ok_or_else(|| ChartError::Render("failed to create pixmap from RGBA data".into()))?;
    pixmap
        .encode_png()
        .map_err(|e| ChartError::Render(format!("PNG encoding failed: {e}")))
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
) -> Result<(), ChartError> {
    let data = render_subplot_to_png(grid, width, height)?;
    std::fs::write(path.as_ref(), data)?;
    Ok(())
}

/// Render a subplot grid to raw RGBA pixel data.
///
/// # Errors
///
/// Returns an error if rendering fails.
pub fn render_subplot_rgba(
    grid: &SubplotGrid,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, ChartError> {
    render_subplot_to_rgba_raw(grid, width, height)
}

/// Internal: render a subplot grid to raw RGBA pixel data.
fn render_subplot_to_rgba_raw(
    grid: &SubplotGrid,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, ChartError> {
    let rows = grid.rows as u32;
    let cols = grid.cols as u32;
    let gap = grid.gap;

    // Reserve space for an optional title at the top
    let title_h = if grid.title.is_some() {
        SUBPLOT_TITLE_HEIGHT
    } else {
        0
    };

    // Calculate cell dimensions
    let total_h_gap = gap * (cols.saturating_sub(1));
    let total_v_gap = gap * (rows.saturating_sub(1));
    let cell_w = (width.saturating_sub(total_h_gap)) / cols;
    let cell_h = (height.saturating_sub(title_h).saturating_sub(total_v_gap)) / rows;

    if cell_w == 0 || cell_h == 0 {
        return Err(ChartError::InvalidConfig(
            "Subplot grid dimensions too small for the given rows/cols/gap".into(),
        ));
    }

    // Create the master pixmap
    let mut master = tiny_skia::Pixmap::new(width, height)
        .ok_or_else(|| ChartError::Render("failed to create master pixmap".into()))?;

    // Fill with background color
    let bg = grid.background;
    let bg_color =
        tiny_skia::Color::from_rgba(bg.r, bg.g, bg.b, bg.a).unwrap_or(tiny_skia::Color::BLACK);
    master.fill(bg_color);

    // --- Shared axis domain unification ---
    // When axes are shared, we compute the union of all cell data extents
    // and inject the unified domain into each cell's config. Non-primary
    // cells have their axis labels suppressed via NullFormatter.
    let share_x = grid.shared_axes.shares_x();
    let share_y = grid.shared_axes.shares_y();

    // Only clone cells when we need to mutate configs for shared axes;
    // otherwise borrow directly to avoid unnecessary allocations.
    let owned_cells: Vec<Option<Chart>>;
    let cells: &[Option<Chart>] = if share_x || share_y {
        // Compute global extent across all populated cells
        let mut gx_min = f64::INFINITY;
        let mut gx_max = f64::NEG_INFINITY;
        let mut gy_min = f64::INFINITY;
        let mut gy_max = f64::NEG_INFINITY;

        for cell in grid.cells.iter().flatten() {
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

        // Clone cells so we can mutate their configs
        let mut local_cells: Vec<Option<Chart>> = grid.cells.clone();

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

        owned_cells = local_cells;
        &owned_cells
    } else {
        &grid.cells
    };

    // Render each cell and composite onto master via tiny-skia
    for (i, chart_opt) in cells.iter().enumerate() {
        let row = i / cols as usize;
        let col = i % cols as usize;
        let x_off = col as u32 * (cell_w + gap);
        let y_off = title_h + row as u32 * (cell_h + gap);

        if let Some(chart) = chart_opt {
            let cell_rgba = render_to_rgba_raw(chart, cell_w, cell_h)?;

            if let Some(cell_pixmap) = tiny_skia::PixmapRef::from_bytes(&cell_rgba, cell_w, cell_h)
            {
                #[allow(clippy::cast_possible_wrap)]
                master.draw_pixmap(
                    x_off as i32,
                    y_off as i32,
                    cell_pixmap,
                    &tiny_skia::PixmapPaint::default(),
                    tiny_skia::Transform::identity(),
                    None,
                );
            }
        }
        // Empty cells keep the background fill
    }

    // Stamp grid title if present
    if let Some(ref title) = grid.title {
        use scry_engine::scene::command::{FontData, TextAlign as EngineTextAlign};

        let bold_fd = FontData::new(FONT_DATA_BOLD.to_vec());
        let color = scry_engine::style::Color::from_rgba8(230, 230, 240, 255);

        // Measure text to compute baseline y (top padding + ascent)
        let metrics = scry_engine::rasterize::skia::text::measure_text(
            title,
            Some(&bold_fd),
            SUBPLOT_TITLE_FONT_SIZE,
        );
        let baseline_y = SUBPLOT_TITLE_TOP_PAD + metrics.ascent;

        // Render title text into a small canvas via the engine
        let title_canvas = scry_engine::scene::PixelCanvas::new(width, title_h)
            .text(title, width as f32 / 2.0, baseline_y)
            .size(SUBPLOT_TITLE_FONT_SIZE)
            .color(color)
            .font(bold_fd)
            .align(EngineTextAlign::Center)
            .done();

        let title_pixmap = scry_engine::rasterize::skia::Rasterizer::rasterize(&title_canvas)
            .map_err(|e| ChartError::Render(format!("title rasterization failed: {e}")))?;

        // Composite title onto master using tiny-skia's proper alpha blending
        master.draw_pixmap(
            0,
            0,
            title_pixmap.as_ref(),
            &tiny_skia::PixmapPaint::default(),
            tiny_skia::Transform::identity(),
            None,
        );
    }

    Ok(master.data().to_vec())
}
