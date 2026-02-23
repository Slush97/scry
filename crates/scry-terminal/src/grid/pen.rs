// SPDX-License-Identifier: MIT OR Apache-2.0
//! Pen reset, tab stop management, and bell.

use super::types::{CellColor, CellFlags};
use super::TerminalGrid;

impl TerminalGrid {
    // ── SGR (Select Graphic Rendition) ─────────────────────────────

    /// Reset all pen attributes to default.
    pub fn reset_pen(&mut self) {
        self.pen_fg = CellColor::Default;
        self.pen_bg = CellColor::Default;
        self.pen_flags = CellFlags::empty();
        self.pen_underline_color = None;
    }

    // ── Tab stop management ────────────────────────────────────────

    /// Set a tab stop at the current cursor column (HTS / ESC H).
    pub fn set_tab_stop(&mut self) {
        let col = self.cursor.col as usize;
        if col < self.tab_stops.len() {
            self.tab_stops[col] = true;
        }
    }

    /// Clear the tab stop at the current cursor column (TBC mode 0).
    pub fn clear_tab_stop(&mut self) {
        let col = self.cursor.col as usize;
        if col < self.tab_stops.len() {
            self.tab_stops[col] = false;
        }
    }

    /// Clear all tab stops (TBC mode 3).
    pub fn clear_all_tab_stops(&mut self) {
        for stop in &mut self.tab_stops {
            *stop = false;
        }
    }

    // ── Bell ────────────────────────────────────────────────────────

    /// Atomically read and clear the visual bell flag.
    ///
    /// Returns `true` if a BEL was pending (and clears it), `false` otherwise.
    /// Callers should check this once per frame and trigger the visual flash
    /// only when it returns `true`.
    pub fn take_bell(&mut self) -> bool {
        let pending = self.bell_pending;
        self.bell_pending = false;
        pending
    }
}
