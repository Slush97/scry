// SPDX-License-Identifier: MIT OR Apache-2.0
//! Unicode halfblock fallback backend.
//!
//! When no graphics protocol is available, this backend renders images using
//! Unicode upper-half-block characters (▀, U+2580) with foreground and
//! background colors. Each character cell represents 1×2 pixels — the
//! foreground color covers the top pixel and the background covers the bottom.
//!
//! This is the lowest-quality fallback but works in every terminal that
//! supports 24-bit color.

use std::io::Write;

use tiny_skia::Pixmap;

use crate::transport::backend::{ImageHandle, ProtocolBackend, ProtocolKind, TerminalPosition};
use crate::PixelCanvasError;

// ---------------------------------------------------------------------------
// HalfblockBackend
// ---------------------------------------------------------------------------

/// Fallback backend using Unicode halfblock characters.
///
/// Renders images using ANSI 24-bit color escape sequences with Unicode
/// upper-half-block characters (▀, U+2580). Each character cell represents
/// 1×2 pixels — the foreground color covers the top pixel and the
/// background covers the bottom.
///
/// When used via the Ratatui widget layer, call
/// [`render_to_cells`](HalfblockBackend::render_to_cells) directly.
/// When used standalone (e.g. `scry see text`), the [`ProtocolBackend`]
/// `transmit()` / `replace()` methods write directly to a configurable
/// writer (stdout by default).
#[derive(Debug)]
pub struct HalfblockBackend<W: Write + std::fmt::Debug = std::io::Stdout> {
    next_id: u32,
    writer: W,
    /// Row of the last transmitted image (for `replace()` cursor repositioning).
    last_row: u16,
    /// Number of halfblock rows in the last transmitted image.
    last_rows: u16,
    /// Reusable cell buffer to avoid per-frame allocation.
    cell_buf: Vec<HalfblockCell>,
    /// Reusable string buffer for building ANSI output.
    line_buf: String,
}

impl HalfblockBackend<std::io::Stdout> {
    /// Create a new halfblock backend that writes to stdout.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_id: 1,
            writer: std::io::stdout(),
            last_row: 0,
            last_rows: 0,
            cell_buf: Vec::new(),
            line_buf: String::new(),
        }
    }
}

impl<W: Write + std::fmt::Debug> HalfblockBackend<W> {
    /// Create a new halfblock backend with a custom writer (useful for testing).
    pub fn with_writer(writer: W) -> Self {
        Self {
            next_id: 1,
            writer,
            last_row: 0,
            last_rows: 0,
            cell_buf: Vec::new(),
            line_buf: String::new(),
        }
    }

    /// Consume the backend and return the underlying writer.
    ///
    /// Useful for inspecting written output in tests.
    pub fn into_writer(self) -> W {
        self.writer
    }

    /// Write downsampled halfblock output to the writer as ANSI escape sequences.
    ///
    /// The pixmap is mapped onto `position.width_cells` columns ×
    /// `position.height_cells` halfblock rows.  Each halfblock character
    /// represents one column and two pixel rows, so we sample the source
    /// pixmap proportionally to fit the target cell dimensions.
    fn write_cells(
        &mut self,
        pixmap: &Pixmap,
        position: TerminalPosition,
    ) -> Result<(), PixelCanvasError> {
        use std::fmt::Write as FmtWrite;

        let src_w = pixmap.width() as usize;
        let src_h = pixmap.height() as usize;
        if src_w == 0 || src_h == 0 {
            return Ok(());
        }

        let data = pixmap.data();

        // Target dimensions in terminal cells.
        let out_cols = (position.width_cells as usize).max(1);
        let out_rows = (position.height_cells as usize).max(1);

        // ANSI positions are 1-indexed.
        let start_row = position.row + 1;
        let start_col = position.col + 1;

        for row in 0..out_rows {
            self.line_buf.clear();
            let ansi_row = start_row + row as u16;
            let _ = write!(self.line_buf, "\x1b[{ansi_row};{start_col}H");

            // Each halfblock row represents two pixel rows.
            // Map top/bottom pixel rows from the source via proportional sampling.
            let top_y = (row * 2) * src_h / (out_rows * 2);
            let bot_y = (row * 2 + 1) * src_h / (out_rows * 2);

            for col in 0..out_cols {
                // Map terminal column to source pixel column.
                let px = col * src_w / out_cols;

                let top = sample_pixel(data, src_w, px, top_y.min(src_h - 1));
                let bot = if bot_y < src_h {
                    sample_pixel(data, src_w, px, bot_y)
                } else {
                    (0, 0, 0)
                };

                let (fr, fg, fb) = top;
                let (br, bg, bb) = bot;
                let _ = write!(
                    self.line_buf,
                    "\x1b[38;2;{fr};{fg};{fb}m\x1b[48;2;{br};{bg};{bb}m▀"
                );
            }
            let _ = write!(self.line_buf, "\x1b[0m");

            self.writer
                .write_all(self.line_buf.as_bytes())
                .map_err(PixelCanvasError::Transmission)?;
        }

        self.writer
            .flush()
            .map_err(PixelCanvasError::Transmission)?;

        self.last_row = position.row;
        self.last_rows = out_rows as u16;

        Ok(())
    }
}

/// Extract RGB from a pixel at (x, y) in RGBA data, compositing against
/// a black background for semi-transparent pixels.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn pixel_at(data: &[u8], width: usize, x: usize, y: usize) -> (u8, u8, u8) {
    let idx = (y * width + x) * 4;
    if idx + 3 < data.len() {
        let (red, green, blue, alpha) = (data[idx], data[idx + 1], data[idx + 2], data[idx + 3]);
        // Fast-path: skip float math for fully-opaque pixels (~95%+ of cases)
        if alpha == 255 {
            (red, green, blue)
        } else if alpha == 0 {
            (0, 0, 0)
        } else {
            let alpha_f = f32::from(alpha) / 255.0;
            (
                (f32::from(red) * alpha_f) as u8,
                (f32::from(green) * alpha_f) as u8,
                (f32::from(blue) * alpha_f) as u8,
            )
        }
    } else {
        (0, 0, 0)
    }
}

/// Sample a pixel for downsampled halfblock output (same as [`pixel_at`]).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
#[inline]
fn sample_pixel(data: &[u8], width: usize, x: usize, y: usize) -> (u8, u8, u8) {
    pixel_at(data, width, x, y)
}

// ---------------------------------------------------------------------------
// Module-level rendering functions
// ---------------------------------------------------------------------------

/// Fill a flat buffer with halfblock cells from a pixmap.
///
/// This is the core rendering routine shared by both the widget layer
/// and the `ProtocolBackend` implementation.
pub fn fill_cells_flat(pixmap: &Pixmap, buf: &mut Vec<HalfblockCell>) {
    let w = pixmap.width() as usize;
    let h = pixmap.height() as usize;
    let rows = h.div_ceil(2);
    let total = rows * w;
    let data = pixmap.data();

    // Resize without shrinking — reuses existing allocation.
    buf.resize(
        total,
        HalfblockCell {
            char: '▀',
            fg: (0, 0, 0),
            bg: (0, 0, 0),
        },
    );

    for row in 0..rows {
        for col in 0..w {
            let top_y = row * 2;
            let bot_y = top_y + 1;

            let top = pixel_at(data, w, col, top_y);
            let bot = if bot_y < h {
                pixel_at(data, w, col, bot_y)
            } else {
                (0, 0, 0)
            };

            buf[row * w + col] = HalfblockCell {
                char: '▀',
                fg: top,
                bg: bot,
            };
        }
    }
}

/// Render a pixmap into a grid of halfblock cell data.
///
/// Returns a 2D grid where each entry contains the character and the
/// (top, bottom) pixel colors for that cell position.
///
/// The grid dimensions are `(pixmap.width(), pixmap.height() / 2)`.
///
/// **Tip**: For animation loops, use [`render_to_cells_flat`]
/// with a reusable buffer to avoid per-frame allocation.
#[must_use]
pub fn render_to_cells(pixmap: &Pixmap) -> Vec<Vec<HalfblockCell>> {
    let w = pixmap.width() as usize;
    let h = pixmap.height() as usize;
    let rows = h.div_ceil(2);
    let mut flat = Vec::with_capacity(rows * w);
    fill_cells_flat(pixmap, &mut flat);
    flat.chunks(w).map(<[HalfblockCell]>::to_vec).collect()
}

/// Render a pixmap into a pre-allocated flat buffer (row-major order).
///
/// The buffer is resized (never shrunk) to fit `rows × width` cells.
/// Reuse the same `Vec` across frames to avoid per-frame allocation —
/// this is **significantly faster** for animation loops.
///
/// To index into the flat buffer: `buf[row * width + col]`.
///
/// Returns `(rows, cols)` — the logical grid dimensions.
pub fn render_to_cells_flat(pixmap: &Pixmap, buf: &mut Vec<HalfblockCell>) -> (usize, usize) {
    fill_cells_flat(pixmap, buf);
    let w = pixmap.width() as usize;
    let h = pixmap.height() as usize;
    (h.div_ceil(2), w)
}

impl Default for HalfblockBackend<std::io::Stdout> {
    fn default() -> Self {
        Self::new()
    }
}

/// A single character cell in the halfblock rendering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HalfblockCell {
    /// The character to display (always '▀').
    pub char: char,
    /// Foreground color (top pixel) as (R, G, B).
    pub fg: (u8, u8, u8),
    /// Background color (bottom pixel) as (R, G, B).
    pub bg: (u8, u8, u8),
}

impl<W: Write + std::fmt::Debug + Send> ProtocolBackend for HalfblockBackend<W> {
    fn transmit(
        &mut self,
        pixmap: &Pixmap,
        position: TerminalPosition,
        _z_index: i32,
    ) -> Result<ImageHandle, PixelCanvasError> {
        self.write_cells(pixmap, position)?;

        let id = self.next_id;
        self.next_id += 1;
        Ok(ImageHandle {
            id,
            protocol: ProtocolKind::Halfblock,
        })
    }

    fn replace(
        &mut self,
        handle: &ImageHandle,
        pixmap: &Pixmap,
        position: TerminalPosition,
        _z_index: i32,
    ) -> Result<ImageHandle, PixelCanvasError> {
        // Overwrite the same region by repositioning the cursor.
        self.write_cells(pixmap, position)?;

        Ok(ImageHandle {
            id: handle.id(),
            protocol: ProtocolKind::Halfblock,
        })
    }

    fn remove(&mut self, _handle: &ImageHandle) -> Result<(), PixelCanvasError> {
        // No-op: halfblock rendering is ephemeral.
        Ok(())
    }

    fn clear_all(&mut self) -> Result<(), PixelCanvasError> {
        // No-op.
        Ok(())
    }

    fn supports_alpha(&self) -> bool {
        false
    }

    fn protocol_kind(&self) -> ProtocolKind {
        ProtocolKind::Halfblock
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a test pixmap with known RGBA pixel values.
    fn make_pixmap(width: u32, height: u32, data: &[(u8, u8, u8, u8)]) -> Pixmap {
        let mut pm = Pixmap::new(width, height).unwrap();
        let buf = pm.data_mut();
        for (i, &(r, g, b, a)) in data.iter().enumerate() {
            buf[i * 4] = r;
            buf[i * 4 + 1] = g;
            buf[i * 4 + 2] = b;
            buf[i * 4 + 3] = a;
        }
        pm
    }

    #[test]
    fn render_2x2_solid_pixels() {
        // 2×2 pixmap: top row = red, bottom row = blue
        let pm = make_pixmap(
            2,
            2,
            &[
                (255, 0, 0, 255),
                (255, 0, 0, 255), // row 0: red
                (0, 0, 255, 255),
                (0, 0, 255, 255), // row 1: blue
            ],
        );
        let cells = render_to_cells(&pm);

        // Should produce 1 row of 2 cells (2 pixel rows → 1 halfblock row)
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].len(), 2);

        // fg = top pixel (red), bg = bottom pixel (blue)
        assert_eq!(cells[0][0].fg, (255, 0, 0));
        assert_eq!(cells[0][0].bg, (0, 0, 255));
        assert_eq!(cells[0][0].char, '▀');
    }

    #[test]
    fn render_odd_height_pads_bottom() {
        // 1×3 pixmap: 3 rows → 2 halfblock rows, bottom of 2nd is black
        let pm = make_pixmap(
            1,
            3,
            &[
                (255, 0, 0, 255), // row 0
                (0, 255, 0, 255), // row 1
                (0, 0, 255, 255), // row 2
            ],
        );
        let cells = render_to_cells(&pm);

        assert_eq!(cells.len(), 2); // ceil(3/2) = 2 rows
        assert_eq!(cells[0][0].fg, (255, 0, 0));
        assert_eq!(cells[0][0].bg, (0, 255, 0));
        assert_eq!(cells[1][0].fg, (0, 0, 255));
        assert_eq!(cells[1][0].bg, (0, 0, 0)); // padded
    }

    #[test]
    fn alpha_compositing_against_black() {
        // 50% alpha white → should composite to ~127
        let pm = make_pixmap(
            1,
            2,
            &[
                (255, 255, 255, 128), // ~50% alpha white
                (255, 0, 0, 0),       // fully transparent red → black
            ],
        );
        let cells = render_to_cells(&pm);

        // Top pixel: 255 * (128/255) ≈ 128
        assert!((i16::from(cells[0][0].fg.0) - 128).unsigned_abs() <= 1);
        // Bottom pixel: fully transparent → (0, 0, 0)
        assert_eq!(cells[0][0].bg, (0, 0, 0));
    }

    #[test]
    fn protocol_kind_is_halfblock() {
        let backend = HalfblockBackend::new();
        assert_eq!(backend.protocol_kind(), ProtocolKind::Halfblock);
        assert!(!backend.supports_alpha());
    }

    #[test]
    fn transmit_returns_dummy_handle() {
        let mut backend = HalfblockBackend::new();
        let pm = Pixmap::new(2, 2).unwrap();
        let pos = TerminalPosition::new(0, 0, 10, 10);
        let handle = backend.transmit(&pm, pos, 0).unwrap();
        assert_eq!(handle.protocol(), ProtocolKind::Halfblock);

        // Subsequent IDs increment
        let handle2 = backend.transmit(&pm, pos, 0).unwrap();
        assert_ne!(handle.id(), handle2.id());
    }

    #[test]
    fn remove_and_clear_are_noop() {
        let mut backend = HalfblockBackend::new();
        let pm = Pixmap::new(2, 2).unwrap();
        let pos = TerminalPosition::new(0, 0, 10, 10);
        let handle = backend.transmit(&pm, pos, 0).unwrap();

        assert!(backend.remove(&handle).is_ok());
        assert!(backend.clear_all().is_ok());
    }

    #[test]
    fn flat_rendering_matches_2d() {
        let pm = make_pixmap(
            2,
            2,
            &[
                (255, 0, 0, 255),
                (0, 255, 0, 255),
                (0, 0, 255, 255),
                (255, 255, 0, 255),
            ],
        );
        let cells_2d = render_to_cells(&pm);
        let mut flat = Vec::new();
        let (rows, cols) = render_to_cells_flat(&pm, &mut flat);
        assert_eq!(rows, 1);
        assert_eq!(cols, 2);
        assert_eq!(flat.len(), rows * cols);
        // Contents should match
        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(flat[r * cols + c], cells_2d[r][c]);
            }
        }
    }

    #[test]
    fn flat_rendering_reuses_buffer() {
        let pm = make_pixmap(
            2,
            2,
            &[
                (255, 0, 0, 255),
                (0, 255, 0, 255),
                (0, 0, 255, 255),
                (255, 255, 0, 255),
            ],
        );
        let mut buf = Vec::new();
        render_to_cells_flat(&pm, &mut buf);
        let ptr1 = buf.as_ptr();
        // Render again — should reuse the same allocation
        render_to_cells_flat(&pm, &mut buf);
        let ptr2 = buf.as_ptr();
        assert_eq!(ptr1, ptr2, "flat buffer should reuse allocation");
    }
}
