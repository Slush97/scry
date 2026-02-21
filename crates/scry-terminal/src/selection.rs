// SPDX-License-Identifier: MIT OR Apache-2.0
//! Text selection — tracks mouse-driven selection and extracts selected text.
//!
//! Selection anchors use `i64` rows: positive values are live screen rows,
//! negative values index into the scrollback buffer (−1 = most recent
//! scrollback line). This allows selections to span across the
//! scrollback/live boundary.

use crate::grid::TerminalGrid;

/// A position in the terminal (column + absolute row).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectionAnchor {
    /// Column (0-indexed).
    pub col: u16,
    /// Row: 0..rows-1 for the live screen, negative for scrollback.
    pub row: i64,
}

impl SelectionAnchor {
    /// Create a new anchor.
    pub fn new(col: u16, row: i64) -> Self {
        Self { col, row }
    }
}

impl PartialOrd for SelectionAnchor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SelectionAnchor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.row.cmp(&other.row).then(self.col.cmp(&other.col))
    }
}

/// Active text selection state.
#[derive(Clone, Debug)]
pub struct Selection {
    /// Starting anchor (where the mouse was pressed).
    start: SelectionAnchor,
    /// Current end anchor (where the mouse is / was released).
    end: SelectionAnchor,
    /// Whether a selection drag is in progress.
    pub active: bool,
}

impl Default for Selection {
    fn default() -> Self {
        Self {
            start: SelectionAnchor::new(0, 0),
            end: SelectionAnchor::new(0, 0),
            active: false,
        }
    }
}

impl Selection {
    /// Start a new selection at the given anchor.
    pub fn begin(&mut self, anchor: SelectionAnchor) {
        self.start = anchor;
        self.end = anchor;
        self.active = true;
    }

    /// Update the end anchor during a drag.
    pub fn update(&mut self, anchor: SelectionAnchor) {
        self.end = anchor;
    }

    /// Finalize the selection (mouse release).
    pub fn finalize(&mut self) {
        self.active = false;
    }

    /// Clear the selection entirely.
    pub fn clear(&mut self) {
        self.active = false;
        self.start = SelectionAnchor::new(0, 0);
        self.end = SelectionAnchor::new(0, 0);
    }

    /// Whether there is a non-empty selection.
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// Return `(start, end)` in reading order (top-left before bottom-right).
    pub fn ordered(&self) -> (SelectionAnchor, SelectionAnchor) {
        if self.start <= self.end {
            (self.start, self.end)
        } else {
            (self.end, self.start)
        }
    }

    /// Test whether a cell at `(col, row)` is within the selection.
    ///
    /// `row` is the absolute row (same coordinate space as `SelectionAnchor::row`).
    pub fn contains(&self, col: u16, row: i64) -> bool {
        if self.is_empty() {
            return false;
        }

        let (start, end) = self.ordered();
        let pos = SelectionAnchor::new(col, row);

        if start.row == end.row {
            // Single-line selection
            row == start.row && col >= start.col && col <= end.col
        } else if row == start.row {
            // First line: from start.col to end of line
            col >= start.col
        } else if row == end.row {
            // Last line: from beginning to end.col
            col <= end.col
        } else {
            // Middle lines: fully selected
            pos.row > start.row && pos.row < end.row
        }
    }

    /// Extract the selected text from the grid.
    ///
    /// Lines are joined by `\n`. Trailing whitespace on each line is trimmed.
    pub fn selected_text(&self, grid: &TerminalGrid) -> String {
        if self.is_empty() {
            return String::new();
        }

        let (start, end) = self.ordered();
        let cols = grid.cols();
        let mut result = String::new();

        for row in start.row..=end.row {
            let col_start = if row == start.row { start.col } else { 0 };
            let col_end = if row == end.row {
                end.col
            } else {
                cols.saturating_sub(1)
            };

            let mut line = String::new();
            for col in col_start..=col_end {
                // Use viewport_cell if available, otherwise skip scrollback lines
                // that we can't access via the current API.
                if row < 0 {
                    // Scrollback line — use scrollback_cell
                    if let Some(cell) = grid.scrollback_cell(col, row) {
                        if cell.width > 0 {
                            cell.write_grapheme(&mut line);
                        }
                    }
                } else if (row as u16) < grid.rows() {
                    let cell = grid.cell(col, row as u16);
                    if cell.width > 0 {
                        cell.write_grapheme(&mut line);
                    }
                }
            }

            // Trim trailing whitespace on each line
            let trimmed = line.trim_end();
            result.push_str(trimmed);

            if row < end.row {
                result.push('\n');
            }
        }

        result
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::TerminalGrid;

    #[test]
    fn selection_ordering() {
        let mut sel = Selection::default();
        sel.begin(SelectionAnchor::new(5, 3));
        sel.update(SelectionAnchor::new(2, 1));

        let (start, end) = sel.ordered();
        assert_eq!(start, SelectionAnchor::new(2, 1));
        assert_eq!(end, SelectionAnchor::new(5, 3));
    }

    #[test]
    fn selection_contains_single_line() {
        let mut sel = Selection::default();
        sel.begin(SelectionAnchor::new(2, 0));
        sel.update(SelectionAnchor::new(5, 0));

        assert!(!sel.contains(1, 0));
        assert!(sel.contains(2, 0));
        assert!(sel.contains(3, 0));
        assert!(sel.contains(5, 0));
        assert!(!sel.contains(6, 0));
        assert!(!sel.contains(3, 1));
    }

    #[test]
    fn selection_contains_multi_line() {
        let mut sel = Selection::default();
        sel.begin(SelectionAnchor::new(3, 1));
        sel.update(SelectionAnchor::new(5, 3));

        // Row 1: col 3+ selected
        assert!(!sel.contains(2, 1));
        assert!(sel.contains(3, 1));
        assert!(sel.contains(10, 1));

        // Row 2: fully selected
        assert!(sel.contains(0, 2));
        assert!(sel.contains(50, 2));

        // Row 3: col 0..5 selected
        assert!(sel.contains(0, 3));
        assert!(sel.contains(5, 3));
        assert!(!sel.contains(6, 3));
    }

    #[test]
    fn selection_empty() {
        let sel = Selection::default();
        assert!(sel.is_empty());
        assert!(!sel.contains(0, 0));
    }

    #[test]
    fn selected_text_single_line() {
        let mut grid = TerminalGrid::new(10, 3, 0);
        // Write "Hello" at row 0
        for (i, c) in "Hello".chars().enumerate() {
            grid.cursor.col = i as u16;
            grid.cursor.row = 0;
            grid.put_char(c);
        }

        let mut sel = Selection::default();
        sel.begin(SelectionAnchor::new(0, 0));
        sel.update(SelectionAnchor::new(4, 0));

        assert_eq!(sel.selected_text(&grid), "Hello");
    }

    #[test]
    fn selected_text_multi_line() {
        let mut grid = TerminalGrid::new(10, 3, 0);
        // Row 0: "AAAA"
        for i in 0..4 {
            grid.cursor.col = i;
            grid.cursor.row = 0;
            grid.put_char('A');
        }
        // Row 1: "BBBB"
        for i in 0..4 {
            grid.cursor.col = i;
            grid.cursor.row = 1;
            grid.put_char('B');
        }

        let mut sel = Selection::default();
        sel.begin(SelectionAnchor::new(2, 0));
        sel.update(SelectionAnchor::new(1, 1));

        let text = sel.selected_text(&grid);
        assert_eq!(text, "AA\nBB");
    }

    #[test]
    fn clear_selection() {
        let mut sel = Selection::default();
        sel.begin(SelectionAnchor::new(0, 0));
        sel.update(SelectionAnchor::new(5, 2));
        assert!(!sel.is_empty());

        sel.clear();
        assert!(sel.is_empty());
        assert!(!sel.active);
    }
}
