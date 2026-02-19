// SPDX-License-Identifier: MIT OR Apache-2.0
//! Standalone incremental renderer — no Ratatui dependency.
//!
//! [`IncrementalRenderer`] provides the same rasterize-cache-dirty-tile-transmit
//! pipeline as the Ratatui widget path, but without requiring a `ratatui::Frame`
//! or `Terminal`.
//!
//! # Example
//!
//! ```no_run
//! use scry_engine::scene::{PixelCanvas, Color};
//! use scry_engine::transport::Picker;
//! use scry_engine::render::IncrementalRenderer;
//!
//! let picker = Picker::detect();
//! let mut renderer = IncrementalRenderer::from_picker(&picker);
//!
//! loop {
//!     let canvas = PixelCanvas::new(200, 200)
//!         .circle(100.0, 100.0, 50.0)
//!             .fill(Color::RED)
//!             .done();
//!     renderer.render_canvas(&canvas).unwrap();
//!     # break;
//! }
//! ```

use std::io::Write;

use crate::rasterize::RasterPipeline;
use crate::scene::PixelCanvas;
use crate::transport::backend::{
    FontSize, ImageHandle, ProtocolBackend, ProtocolKind, TerminalPosition,
};
use crate::transport::halfblock::{HalfblockBackend, HalfblockCell};
use crate::transport::Picker;
use crate::PixelCanvasError;

/// Standalone incremental renderer for pixel canvases.
///
/// Manages a protocol backend, raster cache, and dirty-tile tracking to
/// minimize bandwidth when rendering animated scenes directly to stdout.
///
/// For Kitty/Sixel/iTerm2 backends, only changed tiles are re-transmitted.
/// For the Halfblock backend, ANSI escape sequences are written to stdout.
/// **Important:** This renderer writes halfblock output directly to stdout and
/// must **not** be used inside `ratatui::Terminal::draw()`. For Ratatui
/// integration, use [`PixelCanvasWidget`](crate::widget::PixelCanvasWidget)
/// instead.
pub struct IncrementalRenderer {
    backend: Box<dyn ProtocolBackend>,
    pipeline: RasterPipeline,
    current_handle: Option<ImageHandle>,
    font_size: FontSize,
    /// Whether to skip content-hash caching (for fully-animated scenes).
    skip_cache: bool,
    /// Z-index for Kitty protocol layering.
    z_index: i32,
    /// Terminal position (column, row) for image placement.
    position: TerminalPosition,
    /// Last position at which the image was transmitted, used to detect
    /// when re-transmission is needed even though content hasn't changed.
    last_position: Option<TerminalPosition>,
    /// Reusable flat buffer for halfblock cell rendering.
    halfblock_buf: Vec<HalfblockCell>,
}

impl IncrementalRenderer {
    /// Create a renderer from a [`Picker`].
    ///
    /// Uses the picker's detected protocol and font size.
    pub fn from_picker(picker: &Picker) -> Self {
        let backend = picker.create_backend();
        Self {
            backend,
            pipeline: RasterPipeline::new(),
            current_handle: None,
            font_size: picker.font_size(),
            skip_cache: false,
            z_index: -1,
            position: TerminalPosition::new(0, 0, 0, 0),
            last_position: None,
            halfblock_buf: Vec::new(),
        }
    }

    /// Create a renderer with an explicit backend and font size.
    pub fn new(backend: Box<dyn ProtocolBackend>, font_size: FontSize) -> Self {
        Self {
            backend,
            pipeline: RasterPipeline::new(),
            current_handle: None,
            font_size,
            skip_cache: false,
            z_index: -1,
            position: TerminalPosition::new(0, 0, 0, 0),
            last_position: None,
            halfblock_buf: Vec::new(),
        }
    }

    /// Set the terminal position for image placement (in character cells).
    #[must_use]
    pub const fn position(mut self, col: u16, row: u16) -> Self {
        self.position.col = col;
        self.position.row = row;
        self
    }

    /// Set the z-index for Kitty protocol layering.
    #[must_use]
    pub const fn z_index(mut self, z: i32) -> Self {
        self.z_index = z;
        self
    }

    /// Skip content-hash caching for fully-animated scenes.
    #[must_use]
    pub const fn skip_cache(mut self, skip: bool) -> Self {
        self.skip_cache = skip;
        self
    }

    /// The font size used for pixel-to-cell conversion.
    #[must_use]
    pub const fn font_size(&self) -> FontSize {
        self.font_size
    }

    /// The protocol kind in use.
    pub fn protocol(&self) -> ProtocolKind {
        self.backend.protocol_kind()
    }

    /// Force the next `render_canvas` call to re-transmit, even if the
    /// content hash hasn't changed. Useful after terminal resize (`SIGWINCH`)
    /// or when the widget has been repositioned.
    pub fn invalidate(&mut self) {
        self.last_position = None;
        self.pipeline.cache.clear();
        self.current_handle = None;
    }

    /// Render a canvas to the terminal, sending only dirty tiles.
    ///
    /// This is the main entry point. Each call:
    /// 1. Computes the content hash (unless `skip_cache` is set)
    /// 2. Rasterizes the scene if the hash changed
    /// 3. Computes dirty tiles vs. the previous frame
    /// 4. Transmits only changed tiles to the terminal
    ///
    /// # Errors
    ///
    /// Returns a [`PixelCanvasError`] if rasterization or transmission fails.
    pub fn render_canvas(&mut self, canvas: &PixelCanvas) -> Result<(), PixelCanvasError> {
        let fw = self.font_size.width.max(1);
        let fh = self.font_size.height.max(1);
        let width_cells = (canvas.width() as u16).div_ceil(fw);
        let height_cells = (canvas.height() as u16).div_ceil(fh);
        self.position.width_cells = width_cells;
        self.position.height_cells = height_cells;

        let is_halfblock = self.backend.protocol_kind() == ProtocolKind::Halfblock;

        // Check cache — if valid and position unchanged, skip rasterization
        let position_changed = self.last_position != Some(self.position);
        let content_hash = if self.skip_cache { 0 } else { canvas.content_hash() };
        if !self.skip_cache
            && self.pipeline.cache.is_valid(content_hash)
            && !position_changed
        {
            if is_halfblock {
                let cached = self.pipeline.cache.get(content_hash).cloned();
                if let Some(ref pixmap) = cached {
                    self.render_halfblock_stdout(pixmap)?;
                }
            }
            return Ok(());
        }

        // Rasterize (GPU → CPU fallback handled inside pipeline)
        let store_hash = self
            .pipeline
            .rasterize_into_cache(canvas, self.skip_cache)
            .ok_or_else(|| {
                PixelCanvasError::PixmapCreation("failed to allocate pixmap".to_string())
            })?;

        if is_halfblock {
            let cached = self.pipeline.cache.get(store_hash).cloned();
            if let Some(ref pixmap) = cached {
                self.render_halfblock_stdout(pixmap)?;
            }
            return Ok(());
        }

        // Protocol path: dirty-tile incremental transmission
        let dirty_tiles = self
            .pipeline
            .cache
            .compute_dirty_tiles_cached()
            .unwrap_or_default();

        if dirty_tiles.is_empty() && self.current_handle.is_some() {
            return Ok(());
        }

        if let Some(ref old_handle) = self.current_handle {
            if !dirty_tiles.is_empty() {
                let pixmap = self
                    .pipeline
                    .cache
                    .get(store_hash)
                    .expect("cache validated above");
                let handle = self.backend.transmit_tiles(
                    old_handle,
                    pixmap,
                    self.position,
                    self.z_index,
                    &dirty_tiles,
                )?;
                self.current_handle = Some(handle);
            }
        } else {
            let pixmap = self
                .pipeline
                .cache
                .get(store_hash)
                .expect("cache validated above");
            let handle = self.backend.transmit(pixmap, self.position, self.z_index)?;
            self.current_handle = Some(handle);
        }

        self.last_position = Some(self.position);
        Ok(())
    }

    /// Clean up the current image and reset the cache.
    pub fn cleanup(&mut self) {
        if let Some(handle) = self.current_handle.take() {
            let _ = self.backend.remove(&handle);
        }
        self.pipeline.cache.clear();
    }

    /// Render halfblock content to stdout using ANSI escape sequences.
    fn render_halfblock_stdout(
        &mut self,
        pixmap: &tiny_skia::Pixmap,
    ) -> Result<(), PixelCanvasError> {
        let (cell_rows, cell_cols) =
            HalfblockBackend::render_to_cells_flat(pixmap, &mut self.halfblock_buf);

        if cell_cols == 0 || cell_rows == 0 {
            return Ok(());
        }

        let mut stdout = std::io::stdout().lock();

        // Move cursor to position
        write!(
            stdout,
            "\x1b[{};{}H",
            self.position.row + 1,
            self.position.col + 1
        )?;

        for row in 0..cell_rows {
            if row > 0 {
                // Move to next line at correct column
                write!(
                    stdout,
                    "\x1b[{};{}H",
                    self.position.row as usize + row + 1,
                    self.position.col + 1
                )?;
            }
            for col in 0..cell_cols {
                let cell = &self.halfblock_buf[row * cell_cols + col];
                write!(
                    stdout,
                    "\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m{}",
                    cell.fg.0, cell.fg.1, cell.fg.2, cell.bg.0, cell.bg.1, cell.bg.2, cell.char,
                )?;
            }
        }
        // Reset colors
        write!(stdout, "\x1b[0m")?;
        stdout.flush()?;

        Ok(())
    }
}

impl Drop for IncrementalRenderer {
    fn drop(&mut self) {
        self.cleanup();
    }
}

impl std::fmt::Debug for IncrementalRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IncrementalRenderer")
            .field("backend", &self.backend)
            .field("current_handle", &self.current_handle)
            .field("font_size", &self.font_size)
            .field("skip_cache", &self.skip_cache)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::style::Color;
    use crate::transport::backend::FontSize;

    #[test]
    fn renderer_from_picker() {
        let picker = Picker::new(ProtocolKind::Halfblock, FontSize::default());
        let renderer = IncrementalRenderer::from_picker(&picker);
        assert_eq!(renderer.protocol(), ProtocolKind::Halfblock);
        assert_eq!(renderer.font_size(), FontSize::default());
    }

    #[cfg(feature = "kitty")]
    #[test]
    fn kitty_renderer_sends_data() {
        use crate::transport::kitty::KittyBackend;

        let backend = KittyBackend::with_writer(Vec::new(), FontSize::default());
        let mut renderer = IncrementalRenderer::new(Box::new(backend), FontSize::default());

        let canvas = PixelCanvas::new(64, 64)
            .circle(32.0, 32.0, 16.0)
            .fill(Color::RED)
            .done();

        let result = renderer.render_canvas(&canvas);
        assert!(result.is_ok());
    }

    #[test]
    fn halfblock_renderer_renders() {
        let backend = HalfblockBackend::new();
        let mut renderer = IncrementalRenderer::new(Box::new(backend), FontSize::default());

        let canvas = PixelCanvas::new(16, 16)
            .rect(0.0, 0.0, 16.0, 16.0)
            .fill(Color::BLUE)
            .done();

        // Halfblock writes to stdout, but at least it shouldn't error
        // in a test environment (stdout is available).
        let result = renderer.render_canvas(&canvas);
        assert!(result.is_ok());
    }

    #[test]
    fn builder_methods() {
        let picker = Picker::new(ProtocolKind::Halfblock, FontSize::default());
        let renderer = IncrementalRenderer::from_picker(&picker)
            .position(5, 10)
            .z_index(0)
            .skip_cache(true);

        assert_eq!(renderer.z_index, 0);
        assert!(renderer.skip_cache);
        assert_eq!(renderer.position.col, 5);
        assert_eq!(renderer.position.row, 10);
    }

    #[cfg(feature = "kitty")]
    #[test]
    fn incremental_skips_unchanged_frame() {
        use crate::transport::kitty::KittyBackend;

        let backend = KittyBackend::with_writer(Vec::new(), FontSize::default());
        let mut renderer = IncrementalRenderer::new(Box::new(backend), FontSize::default());

        let canvas = PixelCanvas::new(64, 64)
            .circle(32.0, 32.0, 16.0)
            .fill(Color::RED)
            .done();

        // First render: transmits full frame
        renderer.render_canvas(&canvas).unwrap();
        assert!(renderer.current_handle.is_some());

        // Second render with same canvas: should be a cache hit (no transmission)
        renderer.render_canvas(&canvas).unwrap();
    }
}
