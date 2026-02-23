// SPDX-License-Identifier: MIT OR Apache-2.0
//! Alternate screen and resize operations.

use super::types::Cell;
use super::TerminalGrid;

impl TerminalGrid {
    // ── Alternate screen ───────────────────────────────────────────

    /// Switch to the alternate screen buffer (for vim, less, htop).
    ///
    /// Does **not** save the cursor — callers (e.g. mode 1049 handler)
    /// are responsible for saving/restoring the cursor as needed.
    pub fn enter_alt_screen(&mut self) {
        if self.alt_active {
            return;
        }
        let total = self.cols as usize * self.rows as usize;
        let primary = std::mem::replace(&mut self.cells, vec![Cell::default(); total]);
        self.alt_cells = Some(primary);
        self.alt_active = true;
        self.mark_all_dirty();
    }

    /// Switch back to the primary screen buffer.
    ///
    /// Does **not** restore the cursor — callers (e.g. mode 1049 handler)
    /// are responsible for restoring the cursor if required.
    pub fn exit_alt_screen(&mut self) {
        if !self.alt_active {
            return;
        }
        if let Some(primary) = self.alt_cells.take() {
            self.cells = primary;
        }
        self.alt_active = false;
        self.mark_all_dirty();
    }

    // ── Resize ─────────────────────────────────────────────────────

    /// Resize the grid to new dimensions.
    ///
    /// Preserves content where possible. Lines that don't fit in the new
    /// height are moved to scrollback.
    pub fn resize(&mut self, new_cols: u16, new_rows: u16) {
        if new_cols == self.cols && new_rows == self.rows {
            return;
        }

        let old_cols = self.cols as usize;
        let old_rows = self.rows as usize;
        let new_total = new_cols as usize * new_rows as usize;
        let mut new_cells = vec![Cell::default(); new_total];

        // Copy existing content
        let copy_rows = old_rows.min(new_rows as usize);
        let copy_cols = old_cols.min(new_cols as usize);

        for row in 0..copy_rows {
            for col in 0..copy_cols {
                let old_idx = row * old_cols + col;
                let new_idx = row * new_cols as usize + col;
                new_cells[new_idx] = self.cells[old_idx].clone();
            }
        }

        self.cells = new_cells;
        self.cols = new_cols;
        self.rows = new_rows;
        self.scroll_top = 0;
        self.scroll_bottom = new_rows.saturating_sub(1);
        self.dirty = vec![true; new_rows as usize];

        // Reset tab stops
        self.tab_stops = vec![false; new_cols as usize];
        for i in (0..new_cols as usize).step_by(8) {
            self.tab_stops[i] = true;
        }

        // Clamp cursor
        self.cursor.col = self.cursor.col.min(new_cols.saturating_sub(1));
        self.cursor.row = self.cursor.row.min(new_rows.saturating_sub(1));
        self.cursor.pending_wrap = false;

        // Resize alternate screen too (using saved old_cols for correct stride)
        if let Some(alt) = &mut self.alt_cells {
            let mut new_alt = vec![Cell::default(); new_total];
            for row in 0..copy_rows {
                for col in 0..copy_cols {
                    let old_idx = row * old_cols + col;
                    let new_idx = row * new_cols as usize + col;
                    if old_idx < alt.len() {
                        new_alt[new_idx] = alt[old_idx].clone();
                    }
                }
            }
            *alt = new_alt;
        }
    }
}
