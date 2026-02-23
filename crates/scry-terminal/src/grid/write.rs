// SPDX-License-Identifier: MIT OR Apache-2.0
//! Character insertion (put_char, put_grapheme).

use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use super::types::{Cell, GraphemeStorage};
use super::TerminalGrid;

impl TerminalGrid {
    // ── Character insertion ────────────────────────────────────────

    /// Insert a character at the cursor position and advance.
    pub fn put_char(&mut self, c: char) {
        let char_width = UnicodeWidthChar::width(c).unwrap_or(1) as u16;

        // Handle pending wrap
        if self.cursor.pending_wrap {
            self.cursor.pending_wrap = false;
            self.cursor.col = 0;
            if self.cursor.row == self.scroll_bottom {
                self.scroll_up(1);
            } else {
                self.cursor.row = (self.cursor.row + 1).min(self.rows - 1);
            }
        }

        // If wide char doesn't fit, handle edge: only wrap if auto_wrap is enabled.
        if char_width == 2 && self.cursor.col + 1 >= self.cols {
            if self.auto_wrap {
                // Wide char at last column — pad with space and wrap
                let col = self.cursor.col;
                let row = self.cursor.row;
                self.cell_mut(col, row).clear();
                self.cursor.col = 0;
                if self.cursor.row == self.scroll_bottom {
                    self.scroll_up(1);
                } else {
                    self.cursor.row = (self.cursor.row + 1).min(self.rows - 1);
                }
            } else {
                // auto_wrap off: clip the wide char to a narrow cell
                let col = self.cursor.col;
                let row = self.cursor.row;
                // Write it as a single-width char to avoid overflowing the line.
                let pen_fg = self.pen_fg;
                let pen_bg = self.pen_bg;
                let pen_flags = self.pen_flags;
                let idx = self.idx(col, row);
                let cell = &mut self.cells[idx];
                cell.grapheme = GraphemeStorage::from_char(c);
                cell.fg = pen_fg;
                cell.bg = pen_bg;
                cell.flags = pen_flags;
                cell.width = 1;
                self.mark_dirty(row);
                return; // cursor does not advance past end of line
            }
        }

        let col = self.cursor.col;
        let row = self.cursor.row;

        // Write the cell
        {
            let idx = self.idx(col, row);

            // Clean up stale wide-char cells before overwriting
            let old_width = self.cells[idx].width;
            if old_width == 2 && col + 1 < self.cols {
                let cont_idx = self.idx(col + 1, row);
                self.cells[cont_idx].clear();
            }
            if old_width == 0 && col > 0 {
                let lead_idx = self.idx(col - 1, row);
                self.cells[lead_idx].clear();
            }

            let pen_fg = self.pen_fg;
            let pen_bg = self.pen_bg;
            let pen_flags = self.pen_flags;
            let cell = &mut self.cells[idx];
            cell.grapheme = GraphemeStorage::from_char(c);
            cell.fg = pen_fg;
            cell.bg = pen_bg;
            cell.flags = pen_flags;
            cell.width = char_width as u8;
        }

        // For wide chars, set continuation cell
        if char_width == 2 && col + 1 < self.cols {
            let idx = self.idx(col + 1, row);
            let pen_fg = self.pen_fg;
            let pen_bg = self.pen_bg;
            let cell = &mut self.cells[idx];
            *cell = Cell::continuation();
            cell.fg = pen_fg;
            cell.bg = pen_bg;
        }

        self.mark_dirty(row);

        // Advance cursor
        let new_col = col + char_width;
        if new_col >= self.cols {
            if self.auto_wrap {
                self.cursor.pending_wrap = true;
                self.cursor.col = self.cols - 1;
            }
            // else: cursor stays at last column
        } else {
            self.cursor.col = new_col;
        }
    }

    /// Insert a grapheme cluster at the cursor position and advance.
    ///
    /// For single-codepoint graphemes, this delegates to `put_char`.
    /// For multi-codepoint clusters (emoji, combining chars), it uses
    /// `GraphemeStorage::Cluster` and `UnicodeWidthStr` for width.
    pub fn put_grapheme(&mut self, s: &str) {
        // Fast path: single char
        let mut chars = s.chars();
        if let Some(first) = chars.next() {
            if chars.next().is_none() {
                return self.put_char(first);
            }
        }

        // Multi-codepoint grapheme cluster
        let char_width = UnicodeWidthStr::width(s).max(1) as u16;

        // Handle pending wrap
        if self.cursor.pending_wrap {
            self.cursor.pending_wrap = false;
            self.cursor.col = 0;
            if self.cursor.row == self.scroll_bottom {
                self.scroll_up(1);
            } else {
                self.cursor.row = (self.cursor.row + 1).min(self.rows - 1);
            }
        }

        // If wide grapheme doesn't fit, wrap or pad
        if char_width == 2 && self.cursor.col + 1 >= self.cols {
            let col = self.cursor.col;
            let row = self.cursor.row;
            self.cell_mut(col, row).clear();
            self.cursor.col = 0;
            if self.cursor.row == self.scroll_bottom {
                self.scroll_up(1);
            } else {
                self.cursor.row = (self.cursor.row + 1).min(self.rows - 1);
            }
        }

        let col = self.cursor.col;
        let row = self.cursor.row;

        {
            let idx = self.idx(col, row);

            // Clean up stale wide-char cells before overwriting
            let old_width = self.cells[idx].width;
            if old_width == 2 && col + 1 < self.cols {
                let cont_idx = self.idx(col + 1, row);
                self.cells[cont_idx].clear();
            }
            if old_width == 0 && col > 0 {
                let lead_idx = self.idx(col - 1, row);
                self.cells[lead_idx].clear();
            }

            let pen_fg = self.pen_fg;
            let pen_bg = self.pen_bg;
            let pen_flags = self.pen_flags;
            let cell = &mut self.cells[idx];
            cell.grapheme = GraphemeStorage::from_str(s);
            cell.fg = pen_fg;
            cell.bg = pen_bg;
            cell.flags = pen_flags;
            cell.width = char_width as u8;
        }

        // Place continuation cells for wide graphemes
        if char_width == 2 && col + 1 < self.cols {
            let idx = self.idx(col + 1, row);
            let cell = &mut self.cells[idx];
            *cell = Cell::continuation();
        }

        self.mark_dirty(row);

        // Advance cursor
        let new_col = col + char_width;
        if new_col >= self.cols {
            if self.auto_wrap {
                self.cursor.pending_wrap = true;
                self.cursor.col = self.cols - 1;
            }
        } else {
            self.cursor.col = new_col;
        }
    }
}
