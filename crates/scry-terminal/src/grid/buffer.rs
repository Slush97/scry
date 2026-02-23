// SPDX-License-Identifier: MIT OR Apache-2.0
//! Grid accessors, dirty tracking, and viewport scrolling.

use super::types::Cell;
use super::TerminalGrid;

impl TerminalGrid {
    // ── Accessors ──────────────────────────────────────────────────

    /// Number of columns.
    pub fn cols(&self) -> u16 {
        self.cols
    }

    /// Number of rows.
    pub fn rows(&self) -> u16 {
        self.rows
    }

    /// Whether the alternate screen is active.
    pub fn is_alt_active(&self) -> bool {
        self.alt_active
    }

    /// Get a reference to a cell at (col, row).
    pub fn cell(&self, col: u16, row: u16) -> &Cell {
        &self.cells[self.idx(col, row)]
    }

    /// Get a mutable reference to a cell at (col, row).
    pub fn cell_mut(&mut self, col: u16, row: u16) -> &mut Cell {
        let idx = self.idx(col, row);
        &mut self.cells[idx]
    }

    /// Get a cell from the scrollback buffer.
    ///
    /// `row` uses negative indexing: −1 = most recent scrollback line,
    /// −2 = the one before that, etc.
    pub fn scrollback_cell(&self, col: u16, row: i64) -> Option<&Cell> {
        if row >= 0 || col >= self.cols {
            return None;
        }
        let sb_len = self.scrollback.len() as i64;
        // row = -1 → index = sb_len - 1 (most recent)
        let idx = sb_len + row;
        if idx < 0 {
            return None;
        }
        let line = self.scrollback.get(idx as usize)?;
        line.get(col as usize)
    }

    /// Number of lines in the scrollback buffer.
    pub fn scrollback_len(&self) -> usize {
        self.scrollback.len()
    }

    /// Get the cell at `(col, row)` in the current viewport.
    ///
    /// When `scroll_offset == 0` this returns the live screen cell.
    /// When scrolled back, rows map into scrollback + live buffer:
    /// - Top viewport rows read from the scrollback
    /// - Bottom viewport rows read from the live screen
    pub fn viewport_cell(&self, col: u16, row: u16) -> &Cell {
        if self.scroll_offset == 0 || self.alt_active {
            return self.cell(col, row);
        }

        let sb_len = self.scrollback.len();
        let offset = self.scroll_offset.min(sb_len);
        let absolute_row = row as usize;

        if absolute_row < offset {
            // Reading from scrollback
            let sb_idx = sb_len - offset + absolute_row;
            if let Some(line) = self.scrollback.get(sb_idx) {
                if let Some(cell) = line.get(col as usize) {
                    return cell;
                }
            }
            // Fall back to default cell (scrollback line shorter than cols)
            static DEFAULT_CELL: std::sync::LazyLock<Cell> =
                std::sync::LazyLock::new(Cell::default);
            &DEFAULT_CELL
        } else {
            // Reading from live buffer
            let live_row = (absolute_row - offset) as u16;
            if live_row < self.rows {
                self.cell(col, live_row)
            } else {
                static DEFAULT_CELL: std::sync::LazyLock<Cell> =
                    std::sync::LazyLock::new(Cell::default);
                &DEFAULT_CELL
            }
        }
    }

    /// Linear index for (col, row).
    pub(crate) fn idx(&self, col: u16, row: u16) -> usize {
        row as usize * self.cols as usize + col as usize
    }

    /// Check if a line is dirty.
    pub fn is_dirty(&self, row: u16) -> bool {
        self.dirty.get(row as usize).copied().unwrap_or(false)
    }

    /// Mark a line as dirty.
    pub fn mark_dirty(&mut self, row: u16) {
        if let Some(d) = self.dirty.get_mut(row as usize) {
            *d = true;
        }
    }

    /// Mark all lines as dirty (full redraw needed).
    pub fn mark_all_dirty(&mut self) {
        for d in &mut self.dirty {
            *d = true;
        }
    }

    /// Clear all dirty flags.
    pub fn clear_dirty(&mut self) {
        for d in &mut self.dirty {
            *d = false;
        }
    }

    /// Iterator over dirty line indices.
    pub fn dirty_lines(&self) -> impl Iterator<Item = u16> + '_ {
        self.dirty
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if d { Some(i as u16) } else { None })
    }

    /// Whether any lines are dirty.
    pub fn has_dirty(&self) -> bool {
        self.dirty.iter().any(|&d| d)
    }
}
