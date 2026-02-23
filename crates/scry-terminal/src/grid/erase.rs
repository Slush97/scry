// SPDX-License-Identifier: MIT OR Apache-2.0
//! Erase operations: erase in display, erase in line, clear line.

use super::TerminalGrid;

impl TerminalGrid {
    // ── Erase operations ───────────────────────────────────────────

    /// Erase in display (ED).
    /// - mode 0: cursor to end
    /// - mode 1: start to cursor
    /// - mode 2: entire display
    /// - mode 3: entire display + scrollback
    pub fn erase_in_display(&mut self, mode: u16) {
        match mode {
            0 => {
                // Cursor to end of line, then all lines below
                self.erase_in_line(0);
                for row in (self.cursor.row + 1)..self.rows {
                    self.clear_line(row);
                }
            }
            1 => {
                // All lines above, then start of line to cursor
                for row in 0..self.cursor.row {
                    self.clear_line(row);
                }
                self.erase_in_line(1);
            }
            2 => {
                // Entire display
                for row in 0..self.rows {
                    self.clear_line(row);
                }
            }
            3 => {
                // Entire display + scrollback
                for row in 0..self.rows {
                    self.clear_line(row);
                }
                self.scrollback.clear();
                self.scroll_offset = 0;
            }
            _ => {}
        }
    }

    /// Erase in line (EL).
    /// - mode 0: cursor to end
    /// - mode 1: start to cursor
    /// - mode 2: entire line
    pub fn erase_in_line(&mut self, mode: u16) {
        let row = self.cursor.row;
        match mode {
            0 => {
                for col in self.cursor.col..self.cols {
                    self.cell_mut(col, row).clear();
                }
            }
            1 => {
                for col in 0..=self.cursor.col {
                    self.cell_mut(col, row).clear();
                }
            }
            2 => {
                self.clear_line(row);
            }
            _ => {}
        }
        self.mark_dirty(row);
    }

    /// Clear an entire line.
    pub(crate) fn clear_line(&mut self, row: u16) {
        for col in 0..self.cols {
            self.cell_mut(col, row).clear();
        }
        self.mark_dirty(row);
    }
}
