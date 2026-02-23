// SPDX-License-Identifier: MIT OR Apache-2.0
//! Insert/delete operations: lines and characters.

use super::TerminalGrid;

impl TerminalGrid {
    // ── Insert/Delete operations ───────────────────────────────────

    /// Insert `n` blank lines at the cursor row, pushing lines down.
    /// Lines that scroll off the bottom of the scroll region are lost.
    pub fn insert_lines(&mut self, n: u16) {
        if self.cursor.row < self.scroll_top || self.cursor.row > self.scroll_bottom {
            return;
        }
        let saved_top = self.scroll_top;
        self.scroll_top = self.cursor.row;
        self.scroll_down(n);
        self.scroll_top = saved_top;
        self.cursor.col = 0;
    }

    /// Delete `n` lines at the cursor row, pulling lines up.
    /// Blank lines appear at the bottom of the scroll region.
    pub fn delete_lines(&mut self, n: u16) {
        if self.cursor.row < self.scroll_top || self.cursor.row > self.scroll_bottom {
            return;
        }
        let saved_top = self.scroll_top;
        self.scroll_top = self.cursor.row;
        self.scroll_up(n);
        self.scroll_top = saved_top;
        self.cursor.col = 0;
    }

    /// Insert `n` blank characters at the cursor, shifting existing chars right.
    pub fn insert_chars(&mut self, n: u16) {
        let row = self.cursor.row;
        let col = self.cursor.col;
        let cols = self.cols;

        // Shift right — work from the right edge to avoid overwriting src cells.
        // Use checked arithmetic to prevent u16 underflow in debug builds.
        for c in (col..cols).rev() {
            let src = c.checked_sub(n);
            if let Some(src) = src {
                if src >= col {
                    let src_idx = self.idx(src, row);
                    let dst_idx = self.idx(c, row);
                    self.cells[dst_idx] = self.cells[src_idx].clone();
                }
            }
        }

        // Clear inserted positions
        for c in col..(col.saturating_add(n)).min(cols) {
            self.cell_mut(c, row).clear();
        }
        self.mark_dirty(row);
    }

    /// Delete `n` characters at the cursor, shifting remaining chars left.
    pub fn delete_chars(&mut self, n: u16) {
        let row = self.cursor.row;
        let col = self.cursor.col;
        let cols = self.cols;

        // Shift left
        for c in col..cols {
            if c + n < cols {
                let src_idx = self.idx(c + n, row);
                let dst_idx = self.idx(c, row);
                self.cells[dst_idx] = self.cells[src_idx].clone();
            } else {
                self.cell_mut(c, row).clear();
            }
        }
        self.mark_dirty(row);
    }

    /// Erase `n` characters at cursor (replace with blanks, don't shift).
    pub fn erase_chars(&mut self, n: u16) {
        let row = self.cursor.row;
        for c in self.cursor.col..(self.cursor.col + n).min(self.cols) {
            self.cell_mut(c, row).clear();
        }
        self.mark_dirty(row);
    }
}
