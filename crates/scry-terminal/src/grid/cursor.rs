// SPDX-License-Identifier: MIT OR Apache-2.0
//! Cursor movement operations.

use super::TerminalGrid;

impl TerminalGrid {
    // ── Cursor movement ────────────────────────────────────────────

    /// Move cursor up by `n` lines (clamped to scroll region top).
    pub fn move_up(&mut self, n: u16) {
        let top = if self.origin_mode { self.scroll_top } else { 0 };
        self.cursor.row = self.cursor.row.saturating_sub(n).max(top);
        self.cursor.pending_wrap = false;
    }

    /// Move cursor down by `n` lines (clamped to scroll region bottom).
    pub fn move_down(&mut self, n: u16) {
        let bottom = if self.origin_mode {
            self.scroll_bottom
        } else {
            self.rows - 1
        };
        self.cursor.row = (self.cursor.row + n).min(bottom);
        self.cursor.pending_wrap = false;
    }

    /// Move cursor forward (right) by `n` columns.
    pub fn move_forward(&mut self, n: u16) {
        self.cursor.col = (self.cursor.col + n).min(self.cols - 1);
        self.cursor.pending_wrap = false;
    }

    /// Move cursor backward (left) by `n` columns.
    pub fn move_backward(&mut self, n: u16) {
        self.cursor.col = self.cursor.col.saturating_sub(n);
        self.cursor.pending_wrap = false;
    }

    /// Move cursor to absolute position (1-indexed input, stored 0-indexed).
    pub fn move_to(&mut self, row: u16, col: u16) {
        let row_offset = if self.origin_mode { self.scroll_top } else { 0 };
        let max_row = if self.origin_mode {
            self.scroll_bottom
        } else {
            self.rows - 1
        };
        self.cursor.row = (row.saturating_sub(1) + row_offset).min(max_row);
        self.cursor.col = col.saturating_sub(1).min(self.cols - 1);
        self.cursor.pending_wrap = false;
    }

    /// Move cursor to column (1-indexed).
    pub fn move_to_col(&mut self, col: u16) {
        self.cursor.col = col.saturating_sub(1).min(self.cols - 1);
        self.cursor.pending_wrap = false;
    }

    /// Carriage return — move cursor to column 0.
    pub fn carriage_return(&mut self) {
        self.cursor.col = 0;
        self.cursor.pending_wrap = false;
    }

    /// Line feed — move cursor down, scroll if at bottom of scroll region.
    pub fn line_feed(&mut self) {
        if self.cursor.row == self.scroll_bottom {
            self.scroll_up(1);
        } else {
            self.cursor.row = (self.cursor.row + 1).min(self.rows - 1);
        }
        self.cursor.pending_wrap = false;
    }

    /// Reverse index — move cursor up, scroll down if at top of scroll region.
    pub fn reverse_index(&mut self) {
        if self.cursor.row == self.scroll_top {
            self.scroll_down(1);
        } else {
            self.cursor.row = self.cursor.row.saturating_sub(1);
        }
        self.cursor.pending_wrap = false;
    }

    /// Tab — advance to next tab stop.
    pub fn tab(&mut self) {
        let start = self.cursor.col as usize + 1;
        for i in start..self.cols as usize {
            if self.tab_stops.get(i).copied().unwrap_or(false) {
                self.cursor.col = i as u16;
                self.cursor.pending_wrap = false;
                return;
            }
        }
        // No tab stop found — move to last column
        self.cursor.col = self.cols - 1;
        self.cursor.pending_wrap = false;
    }

    /// Backspace — move cursor left by 1 (does not erase).
    ///
    /// If a wrap is pending (cursor just reached the right margin),
    /// BS cancels the wrap without moving. At column 0, BS is a no-op.
    pub fn backspace(&mut self) {
        if self.cursor.pending_wrap {
            // Cancel the pending wrap; cursor stays at cols-1.
            self.cursor.pending_wrap = false;
        } else if self.cursor.col > 0 {
            self.cursor.col -= 1;
        }
        // At col 0 with no pending wrap: no-op (cannot backspace further).
    }
}
