//! `StatefulWidget` implementation for pixel canvas rendering.
//!
//! This module provides [`PixelCanvasWidget`] and [`PixelCanvasState`] which
//! coordinate scene rasterization and protocol transmission within Ratatui's
//! render cycle.
//!
//! ## Deferred Transmission
//!
//! For protocol backends (Kitty, Sixel), image data is **not** sent to the
//! terminal during `render()`. Instead, the rasterized pixmap is stored as a
//! pending frame. Call [`PixelCanvasState::flush()`] **after**
//! `terminal.draw()` to transmit the image — this ensures the Kitty escape
//! sequences are written after ratatui has flushed its buffer diff, avoiding
//! visible flicker from interleaved cursor movements.



use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::widgets::StatefulWidget;

use crate::rasterize::{RasterCache, Rasterizer};
use crate::scene::PixelCanvas;
use crate::transport::backend::{FontSize, ImageHandle, ProtocolBackend, ProtocolKind, TerminalPosition};
use crate::transport::halfblock::{HalfblockBackend, HalfblockCell};
use crate::PixelCanvasError;

// ---------------------------------------------------------------------------
// PixelCanvasWidget
// ---------------------------------------------------------------------------

/// A Ratatui widget that renders pixel graphics via a graphics protocol.
///
/// This is a thin wrapper that carries the scene to be rendered.
/// All persistent state lives in [`PixelCanvasState`].
///
/// # Example
///
/// ```ignore
/// use ratatui_pixelcanvas::scene::{PixelCanvas, Color};
/// use ratatui_pixelcanvas::widget::{PixelCanvasWidget, PixelCanvasState};
///
/// let canvas = PixelCanvas::new(200, 200)
///     .circle(100.0, 100.0, 50.0)
///     .fill(Color::RED)
///     .done();
///
/// frame.render_stateful_widget(
///     PixelCanvasWidget::new(canvas),
///     area,
///     &mut state,
/// );
/// // After terminal.draw() returns:
/// state.flush().unwrap();
/// ```
pub struct PixelCanvasWidget {
    canvas: PixelCanvas,
    z_index: i32,
    skip_cache: bool,
}

impl PixelCanvasWidget {
    /// Create a new widget from a pixel canvas scene.
    #[must_use]
    pub const fn new(canvas: PixelCanvas) -> Self {
        Self {
            canvas,
            z_index: -1, // Default: render behind text
            skip_cache: false,
        }
    }

    /// Set the z-index for Kitty protocol layering.
    ///
    /// - Negative values place the image behind text (default: -1)
    /// - Positive values place the image in front of text
    #[must_use]
    pub const fn z_index(mut self, z: i32) -> Self {
        self.z_index = z;
        self
    }

    /// Skip content-hash caching for this frame.
    ///
    /// Useful for fully-animated scenes where the scene changes every frame
    /// and the content hash would never hit cache. Saves the cost of hashing
    /// all draw commands (~2ms for 1000+ commands).
    #[must_use]
    pub const fn skip_cache(mut self) -> Self {
        self.skip_cache = true;
        self
    }
}

// ---------------------------------------------------------------------------
// Pending frame — deferred image data awaiting transmission
// ---------------------------------------------------------------------------

/// Metadata for a rasterized frame awaiting protocol transmission.
///
/// The actual pixmap data lives in the [`RasterCache`] to avoid per-frame
/// allocations. This struct holds only the transmission metadata needed by
/// [`PixelCanvasState::flush()`].
struct PendingFrame {
    position: TerminalPosition,
    z_index: i32,
    content_hash: u64,
}

// ---------------------------------------------------------------------------
// PixelCanvasState
// ---------------------------------------------------------------------------

/// Persistent state for [`PixelCanvasWidget`] across render frames.
///
/// This manages:
/// - The protocol backend (Kitty, Sixel, or Halfblock)
/// - The raster cache (skip re-rendering unchanged scenes)
/// - The current image handle (for cleanup on re-render or drop)
/// - Font size for pixel ↔ cell coordinate conversion
/// - Deferred frame pending transmission (protocol backends only)
pub struct PixelCanvasState {
    backend: Box<dyn ProtocolBackend>,
    cache: RasterCache,
    current_handle: Option<ImageHandle>,
    font_size: FontSize,
    /// Transmission metadata for a frame rasterized during `render()`.
    /// The pixmap itself lives in `cache`.
    pending: Option<PendingFrame>,
    /// Reusable flat buffer for halfblock cell rendering (avoids per-frame Vec<Vec<>> allocation).
    halfblock_buf: Vec<HalfblockCell>,
}

impl PixelCanvasState {
    /// Create a new state with the given protocol backend and font size.
    pub fn new(backend: Box<dyn ProtocolBackend>, font_size: FontSize) -> Self {
        Self {
            backend,
            cache: RasterCache::new(),
            current_handle: None,
            font_size,
            pending: None,
            halfblock_buf: Vec::new(),
        }
    }

    /// The font size used for pixel ↔ cell conversion.
    #[must_use]
    pub const fn font_size(&self) -> FontSize {
        self.font_size
    }

    /// Transmit the pending image to the terminal.
    ///
    /// Call this **after** `terminal.draw()` returns so that Kitty escape
    /// sequences are written after ratatui has flushed its buffer diff. This
    /// prevents interleaved cursor movements from causing visible flicker.
    ///
    /// For halfblock backends this is a no-op (rendering happens inline in
    /// the ratatui buffer).
    ///
    /// # Errors
    ///
    /// Returns a [`PixelCanvasError`] if the protocol transmission fails.
    pub fn flush(&mut self) -> Result<(), PixelCanvasError> {
        let Some(frame) = self.pending.take() else {
            return Ok(());
        };

        // The pixmap was rasterized into the cache during render().
        let Some(pixmap) = self.cache.get(frame.content_hash) else {
            return Ok(());
        };

        let result = if let Some(ref old_handle) = self.current_handle {
            self.backend.replace(old_handle, pixmap, frame.position, frame.z_index)
        } else {
            self.backend.transmit(pixmap, frame.position, frame.z_index)
        };

        match result {
            Ok(handle) => {
                self.current_handle = Some(handle);
            }
            Err(e) => {
                self.current_handle = None;
                self.cache.clear();
                return Err(e);
            }
        }

        Ok(())
    }

    /// Mutable access to the raster cache.
    ///
    /// Useful for manual profiled rasterization where you need to
    /// rasterize into the cache's pixmap directly.
    pub fn cache_mut(&mut self) -> &mut RasterCache {
        &mut self.cache
    }

    /// Manually clean up the current image.
    ///
    /// This is called automatically on drop, but can be called explicitly
    /// if you need to clear the image before the state is dropped.
    pub fn cleanup(&mut self) {
        self.pending = None;
        if let Some(handle) = self.current_handle.take() {
            let _ = self.backend.remove(&handle);
        }
        self.cache.clear();
    }
}

impl Drop for PixelCanvasState {
    fn drop(&mut self) {
        self.cleanup();
    }
}

impl std::fmt::Debug for PixelCanvasState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PixelCanvasState")
            .field("backend", &self.backend)
            .field("current_handle", &self.current_handle)
            .field("font_size", &self.font_size)
            .field("has_pending", &self.pending.is_some())
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// StatefulWidget implementation
// ---------------------------------------------------------------------------

impl StatefulWidget for PixelCanvasWidget {
    type State = PixelCanvasState;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::State) {
        if area.width == 0 || area.height == 0 {
            return;
        }

        let canvas = self.canvas;
        let is_halfblock = state.backend.protocol_kind() == ProtocolKind::Halfblock;

        // Content hash check — skip hashing entirely for animated scenes
        let content_hash = if self.skip_cache {
            // Use a sentinel that never matches, so we always rasterize
            // but still have a key to store/retrieve the pixmap in cache.
            0
        } else {
            canvas.content_hash()
        };

        if !self.skip_cache && state.cache.is_valid(content_hash) {
            if is_halfblock {
                if let Some(pixmap) = state.cache.get(content_hash) {
                    render_halfblock_to_buffer(buf, area, pixmap, state.font_size, &mut state.halfblock_buf);
                }
            } else if state.current_handle.is_some() {
                fill_area_with_spaces(buf, area);
            }
            return;
        }

        // Rasterize scene into cache's reusable pixmap (avoids per-frame allocation)
        let (pixmap_opt, gc) = state.cache.get_or_insert_with_grad_cache(canvas.width(), canvas.height());
        let Some(pixmap) = pixmap_opt else {
            return;
        };
        Rasterizer::rasterize_into_cached(&canvas, pixmap, gc);
        // Use a unique hash per frame when skip_cache is on, so flush() can find it.
        let store_hash = if self.skip_cache {
            // Monotonic counter to ensure cache is always "valid" for flush()
            use std::sync::atomic::{AtomicU64, Ordering};
            static FRAME_SEQ: AtomicU64 = AtomicU64::new(1);
            FRAME_SEQ.fetch_add(1, Ordering::Relaxed)
        } else {
            content_hash
        };
        state.cache.mark_valid(store_hash);

        if is_halfblock {
            // Halfblock path: render directly into the ratatui Buffer.
            if let Some(pixmap) = state.cache.get(store_hash) {
                render_halfblock_to_buffer(buf, area, pixmap, state.font_size, &mut state.halfblock_buf);
            }
        } else {
            // Protocol path (Kitty/Sixel): defer transmission until flush()
            let position = TerminalPosition::new(area.x, area.y, area.width, area.height);

            state.pending = Some(PendingFrame {
                position,
                z_index: self.z_index,
                content_hash: store_hash,
            });

            fill_area_with_spaces(buf, area);
        }
    }
}

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------

/// Fill a rectangular area of the buffer with space characters.
///
/// This prevents Ratatui from writing text over the image area.
fn fill_area_with_spaces(buf: &mut Buffer, area: Rect) {
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            if let Some(cell) = buf.cell_mut((x, y)) {
                cell.set_char(' ');
            }
        }
    }
}

/// Render a pixmap into the ratatui `Buffer` using halfblock characters.
///
/// Each terminal cell displays two pixel rows using the Unicode upper-half-block
/// character '▀' — foreground color represents the top pixel, background
/// represents the bottom pixel.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn render_halfblock_to_buffer(
    buf: &mut Buffer,
    area: Rect,
    pixmap: &tiny_skia::Pixmap,
    font_size: FontSize,
    hb_buf: &mut Vec<HalfblockCell>,
) {
    let (cell_rows, cell_cols) = HalfblockBackend::render_to_cells_flat(pixmap, hb_buf);

    if cell_cols == 0 || cell_rows == 0 {
        return;
    }

    let fw = font_size.width.max(1) as usize;
    let fh = font_size.height.max(1) as usize;

    // Each terminal cell covers fw × fh pixels.
    // Halfblock cells are 1px wide × 2px tall, so we sample accordingly.
    for row in 0..area.height as usize {
        for col in 0..area.width as usize {
            // Map terminal cell to the center pixel of the halfblock grid
            let px = col * fw + fw / 2;
            // Each halfblock row represents 2 pixel rows; terminal row maps
            // through font height
            let py = row * fh / 2;

            if py >= cell_rows || px >= cell_cols {
                continue;
            }

            let cell_data = &hb_buf[py * cell_cols + px];

            let buf_x = area.x + col as u16;
            let buf_y = area.y + row as u16;

            if let Some(cell) = buf.cell_mut((buf_x, buf_y)) {
                cell.set_char(cell_data.char);
                cell.set_fg(ratatui::style::Color::Rgb(
                    cell_data.fg.0,
                    cell_data.fg.1,
                    cell_data.fg.2,
                ));
                cell.set_bg(ratatui::style::Color::Rgb(
                    cell_data.bg.0,
                    cell_data.bg.1,
                    cell_data.bg.2,
                ));
            }
        }
    }
}
