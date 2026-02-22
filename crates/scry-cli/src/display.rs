// SPDX-License-Identifier: MIT OR Apache-2.0
//! Protocol-agnostic display driver for the CLI.
//!
//! Replaces the hand-rolled Kitty-only `inline.rs` with a thin wrapper around
//! scry-engine's [`ProtocolBackend`] trait.  This gives the CLI automatic
//! support for Kitty, iTerm2, Sixel, and Halfblock from a single code path.

use std::io::{self, Write};

use scry_engine::transport::backend::{FontSize, ProtocolBackend, ProtocolKind, TerminalPosition};
use scry_engine::transport::Picker;
use scry_engine::Pixmap;

// ---------------------------------------------------------------------------
// FrameDriver
// ---------------------------------------------------------------------------

/// Protocol-agnostic display driver.
///
/// Wraps a `Box<dyn ProtocolBackend>` and manages cursor positioning for
/// both one-shot image display and multi-frame animation loops.
pub struct FrameDriver {
    backend: Box<dyn ProtocolBackend>,
    font_size: FontSize,
    /// Current image handle (for `replace` on subsequent frames).
    handle: Option<scry_engine::transport::ImageHandle>,
    /// Row where the animation was anchored on frame 0 (0-indexed).
    anchor_row: u16,
}

impl FrameDriver {
    /// Auto-detect the best protocol and create a driver.
    ///
    /// Delegates to [`Picker::detect()`] which runs XTVERSION, DA1, Kitty
    /// graphics queries, and env-var fallbacks.  Results are cached globally.
    pub fn detect() -> Self {
        let picker = Picker::detect();
        let font_size = picker.font_size();
        let backend = picker.create_backend();
        Self {
            backend,
            font_size,
            handle: None,
            anchor_row: 0,
        }
    }

    /// The detected protocol kind.
    pub fn protocol(&self) -> ProtocolKind {
        self.backend.protocol_kind()
    }

    /// Whether the detected protocol supports pixel-perfect inline images.
    pub fn supports_inline(&self) -> bool {
        self.backend.protocol_kind() != ProtocolKind::Halfblock
    }

    /// Display a single image from PNG bytes (non-animated).
    ///
    /// Used by `scry chart`, `scry render`, etc.
    /// Decodes PNG to [`Pixmap`], then transmits at the current cursor row.
    pub fn display_png(&mut self, png_data: &[u8]) -> Result<(), String> {
        let pixmap = Pixmap::decode_png(png_data)
            .map_err(|e| format!("failed to decode PNG: {e}"))?;
        self.display_static(&pixmap)
    }

    /// Display a single [`Pixmap`] (non-animated).
    ///
    /// Places the image at the current cursor position by querying the
    /// cursor row and passing that as `TerminalPosition`.
    pub fn display_static(&mut self, pixmap: &Pixmap) -> Result<(), String> {
        let (_, cur_row) = crossterm::cursor::position().unwrap_or((0, 0));
        let (cols, rows) = self.pixmap_cell_dims(pixmap);

        // Position at current cursor row, column 0.
        let position = TerminalPosition::new(0, cur_row, cols, rows);

        self.backend
            .transmit(pixmap, position, 0)
            .map_err(|e| format!("display failed: {e}"))?;

        // Newline after the image so the prompt appears below.
        let mut stdout = io::stdout().lock();
        writeln!(stdout).ok();
        stdout.flush().ok();

        Ok(())
    }

    /// Display one animation frame.
    ///
    /// On frame 0 the terminal is scrolled to reserve enough rows and
    /// the cursor position is anchored.  Subsequent frames atomically
    /// replace the previous image via [`ProtocolBackend::replace`].
    ///
    /// All cursor management and synchronized updates are handled by the
    /// `ProtocolBackend` — we just compute the correct `TerminalPosition`.
    pub fn display_frame(&mut self, pixmap: &Pixmap, frame: u64) -> Result<(), String> {
        // Poll for terminal events (click-to-pause, scroll visibility).
        // This updates internal pause/visibility state on NativeBackend;
        // other backends are no-ops.
        self.backend.poll_events();

        // If the current overlay is paused (click-to-toggle), skip this frame.
        if let Some(ref h) = self.handle {
            if self.backend.is_overlay_paused(h.id()) {
                return Ok(());
            }
        }

        let (term_cols, term_rows) = crossterm::terminal::size().unwrap_or((120, 40));
        let cw = self.font_size.width.max(1);
        let ch = self.font_size.height.max(1);

        // Fit image to terminal, maintaining aspect ratio.
        let (cols_to_use, img_rows) =
            self.fit_to_terminal(pixmap, term_cols, term_rows, cw, ch);

        if frame == 0 {
            let mut stdout = io::stdout().lock();
            let (_, cur_row) = crossterm::cursor::position().unwrap_or((0, 0));

            // Reserve vertical space by printing newlines.
            for _ in 0..img_rows {
                writeln!(stdout).map_err(|e| e.to_string())?;
            }
            stdout.flush().map_err(|e| e.to_string())?;

            // Compute anchor, accounting for possible scroll.
            let space_below = term_rows.saturating_sub(cur_row + 1);
            self.anchor_row = if space_below >= img_rows {
                cur_row
            } else {
                cur_row
                    .saturating_sub(img_rows.saturating_sub(space_below))
                    .max(0)
            };
        }

        // Center horizontally: offset by ~25% of terminal width.
        let col = term_cols.saturating_sub(cols_to_use) / 4;

        // The backend handles cursor positioning, synchronized updates,
        // and escape sequence formatting — we just provide the position.
        let position = TerminalPosition::new(col, self.anchor_row, cols_to_use, img_rows);

        if let Some(ref prev_handle) = self.handle {
            let new_handle = self.backend
                .replace(prev_handle, pixmap, position, 0)
                .map_err(|e| format!("frame replace failed: {e}"))?;
            self.handle = Some(new_handle);
        } else {
            let new_handle = self.backend
                .transmit(pixmap, position, 0)
                .map_err(|e| format!("frame transmit failed: {e}"))?;
            self.handle = Some(new_handle);
        }

        // Move cursor below the image so status text doesn't overlap.
        let below_row = self.anchor_row + img_rows + 1;
        let mut stdout = io::stdout().lock();
        write!(stdout, "\x1b[{below_row};1H").map_err(|e| e.to_string())?;
        stdout.flush().map_err(|e| e.to_string())?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Compute the terminal cell dimensions for a pixmap.
    fn pixmap_cell_dims(&self, pixmap: &Pixmap) -> (u16, u16) {
        let cw = self.font_size.width.max(1) as u32;
        let ch = self.font_size.height.max(1) as u32;
        let cols = pixmap.width().div_ceil(cw) as u16;
        let rows = pixmap.height().div_ceil(ch) as u16;
        (cols.max(1), rows.max(1))
    }

    /// Fit a pixmap into the terminal, preserving aspect ratio.
    ///
    /// Returns `(columns, rows)` in terminal cell units.
    fn fit_to_terminal(
        &self,
        pixmap: &Pixmap,
        term_cols: u16,
        term_rows: u16,
        cw: u16,
        ch: u16,
    ) -> (u16, u16) {
        let max_rows = term_rows.saturating_sub(2).max(1);
        let aspect = pixmap.height() as f64 / pixmap.width().max(1) as f64;

        // Native columns at 1:1, capped at 90% of terminal width.
        let native_cols = (pixmap.width() as f64 / cw as f64).ceil() as u16;
        let max_cols = ((term_cols as f64) * 0.9).round() as u16;
        let mut cols = native_cols.min(max_cols).max(20).min(term_cols);

        // Compute rows from the fitted width.
        let display_px_w = cols as f64 * cw as f64;
        let display_px_h = display_px_w * aspect;
        let mut rows = (display_px_h / ch as f64).ceil() as u16;

        // If it overflows vertically, shrink.
        if rows > max_rows {
            rows = max_rows;
            let fitted_px_h = rows as f64 * ch as f64;
            let fitted_px_w = fitted_px_h / aspect;
            cols = (fitted_px_w / cw as f64).floor() as u16;
            cols = cols.max(20).min(term_cols);
        }

        (cols, rows.max(1))
    }
}

// ---------------------------------------------------------------------------
// Module-level convenience functions
// ---------------------------------------------------------------------------

/// Compute the terminal's visible pixel dimensions using auto-detected font size.
pub fn terminal_pixel_size() -> (u32, u32) {
    let picker = Picker::detect();
    let font = picker.font_size();
    let (cols, rows) = crossterm::terminal::size().unwrap_or((120, 40));
    let cw = font.width.max(1);
    let ch = font.height.max(1);
    (u32::from(cols) * u32::from(cw), u32::from(rows) * u32::from(ch))
}

/// Detect the terminal cell size in pixels `(width, height)`.
pub fn detect_cell_size() -> (u16, u16) {
    let picker = Picker::detect();
    let font = picker.font_size();
    (font.width.max(1), font.height.max(1))
}
