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
    /// True when running inside scry-terminal (Native IPC path).
    is_native: bool,
    /// When true, `cleanup()` will NOT delete the image.
    /// Set by `display_static()` so one-shot images survive process exit.
    persist: bool,
}

impl FrameDriver {
    /// Auto-detect the best protocol and create a driver.
    ///
    /// Delegates to [`Picker::detect()`] which runs XTVERSION, DA1, Kitty
    /// graphics queries, and env-var fallbacks.  Results are cached globally.
    pub fn detect() -> Self {
        let picker = Picker::detect();
        let mut font_size = picker.font_size();
        let is_native = picker.protocol() == ProtocolKind::Native;
        let mut backend = picker.create_backend();

        // When running inside scry-terminal, query the actual font metrics
        // from the terminal rather than relying on TIOCGWINSZ which may
        // report inaccurate cell dimensions in the PTY.
        if is_native {
            if let Ok(info) = backend.query_info() {
                if info.font_w > 0 && info.font_h > 0 {
                    font_size = FontSize::new(info.font_w, info.font_h);
                }
            }
        }

        Self {
            backend,
            font_size,
            handle: None,
            anchor_row: 0,
            is_native,
            persist: false,
        }
    }

    /// Whether this driver is using the native IPC path (inside scry-terminal).
    pub fn is_native(&self) -> bool {
        self.is_native
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
        let pixmap =
            Pixmap::decode_png(png_data).map_err(|e| format!("failed to decode PNG: {e}"))?;
        self.display_static(&pixmap)
    }

    /// Display a single [`Pixmap`] (non-animated).
    ///
    /// Fits the image to the terminal (preserving aspect ratio), centers
    /// it horizontally, reserves vertical space, and marks the image as
    /// persistent so it survives process exit.
    pub fn display_static(&mut self, pixmap: &Pixmap) -> Result<(), String> {
        // Mark as persistent so cleanup() won't delete the image.
        self.persist = true;

        let (term_cols, term_rows) = crossterm::terminal::size().unwrap_or((120, 40));
        let cw = self.font_size.width.max(1);
        let ch = self.font_size.height.max(1);

        // Fit to terminal, preserving aspect ratio.
        let (cols, rows) = self.fit_to_terminal(pixmap, term_cols, term_rows, cw, ch);

        if self.is_native {
            // Native path: use persistent transmit so the overlay survives
            // after the CLI process exits.  Center horizontally.
            let native_cols = self.native_terminal_cols();
            let col = native_cols.saturating_sub(cols) / 2;
            let position = TerminalPosition::new(col, 0, cols, rows);

            let handle = self
                .backend
                .transmit_persistent(pixmap, position, 0)
                .map_err(|e| format!("display failed: {e}"))?;
            self.handle = Some(handle);
            return Ok(());
        }

        let (_, cur_row) = crossterm::cursor::position().unwrap_or((0, 0));

        // Reserve vertical space for the image.
        {
            let mut stdout = io::stdout().lock();
            for _ in 0..rows {
                writeln!(stdout).ok();
            }
            stdout.flush().ok();
        }

        // Compute anchor, accounting for possible scroll.
        let space_below = term_rows.saturating_sub(cur_row + 1);
        let anchor_row = if space_below >= rows {
            cur_row
        } else {
            cur_row.saturating_sub(rows.saturating_sub(space_below))
        };

        // Center horizontally.
        let col = term_cols.saturating_sub(cols) / 2;
        let position = TerminalPosition::new(col, anchor_row, cols, rows);

        let handle = self
            .backend
            .transmit(pixmap, position, 0)
            .map_err(|e| format!("display failed: {e}"))?;
        self.handle = Some(handle);

        // Move cursor below the image so the prompt appears underneath.
        let below_row = anchor_row + rows + 1;
        let mut stdout = io::stdout().lock();
        write!(stdout, "\x1b[{below_row};1H").ok();
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

        if self.is_native {
            return self.display_frame_native(pixmap);
        }

        let (term_cols, term_rows) = crossterm::terminal::size().unwrap_or((120, 40));
        let cw = self.font_size.width.max(1);
        let ch = self.font_size.height.max(1);

        // Fit image to terminal, maintaining aspect ratio.
        let (cols_to_use, img_rows) = self.fit_to_terminal(pixmap, term_cols, term_rows, cw, ch);

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
        let col = term_cols.saturating_sub(cols_to_use) / 2;

        // The backend handles cursor positioning, synchronized updates,
        // and escape sequence formatting — we just provide the position.
        let position = TerminalPosition::new(col, self.anchor_row, cols_to_use, img_rows);

        if let Some(ref prev_handle) = self.handle {
            let new_handle = self
                .backend
                .replace(prev_handle, pixmap, position, 0)
                .map_err(|e| format!("frame replace failed: {e}"))?;
            self.handle = Some(new_handle);
        } else {
            let new_handle = self
                .backend
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

    /// Native IPC render path — skip all crossterm terminal management.
    ///
    /// The compositor handles placement, so we use viewport-relative
    /// coordinates and let `NativeBackend` transmit RGBA pixels over memfd IPC.
    /// Overlay is centered horizontally within the terminal viewport.
    fn display_frame_native(&mut self, pixmap: &Pixmap) -> Result<(), String> {
        let (cols, rows) = self.pixmap_cell_dims(pixmap);

        // Center horizontally within the terminal viewport.
        let term_cols = self.native_terminal_cols();
        let col = term_cols.saturating_sub(cols) / 2;
        let position = TerminalPosition::new(col, 0, cols, rows);

        if let Some(ref prev_handle) = self.handle {
            let new_handle = self
                .backend
                .replace(prev_handle, pixmap, position, 0)
                .map_err(|e| format!("frame replace failed: {e}"))?;
            self.handle = Some(new_handle);
        } else {
            let new_handle = self
                .backend
                .transmit(pixmap, position, 0)
                .map_err(|e| format!("frame transmit failed: {e}"))?;
            self.handle = Some(new_handle);
        }

        Ok(())
    }

    /// Remove the active overlay from the backend.
    ///
    /// When `persist` is true (set by `display_static`), the image handle
    /// is dropped without sending a delete command — the Kitty protocol
    /// image survives naturally and the terminal reclaims it on scroll
    /// or screen clear.
    pub fn cleanup(&mut self) {
        if self.persist {
            // Drop the handle without deleting — image stays on screen.
            self.handle.take();
            return;
        }
        if let Some(ref handle) = self.handle.take() {
            let _ = self.backend.remove(handle);
        }
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

    /// Get the terminal column count for native mode (from cached font size).
    fn native_terminal_cols(&self) -> u16 {
        // Query grid cols from the terminal via the font size and a
        // reasonable terminal width estimate.  In practice the NativeBackend
        // already queried this via `QueryInfo` at detect() time, but we
        // don't cache it separately.  Use TIOCGWINSZ as a fallback for cols.
        crossterm::terminal::size().map_or(80, |(cols, _)| cols)
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
    (
        u32::from(cols) * u32::from(cw),
        u32::from(rows) * u32::from(ch),
    )
}

/// Detect the terminal cell size in pixels `(width, height)`.
pub fn detect_cell_size() -> (u16, u16) {
    let picker = Picker::detect();
    let font = picker.font_size();
    (font.width.max(1), font.height.max(1))
}
