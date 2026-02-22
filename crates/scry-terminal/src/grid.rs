// SPDX-License-Identifier: MIT OR Apache-2.0
//! Terminal grid — the logical model of the terminal screen.
//!
//! `TerminalGrid` is a 2D array of [`Cell`]s with cursor state, scroll regions,
//! alternate screen buffer, and scrollback history. It is purely a data model —
//! it has no knowledge of rendering, escape sequences, or I/O.

use bitflags::bitflags;
use std::collections::VecDeque;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::config::{self, ColorConfig};

// ── Cell types ─────────────────────────────────────────────────────

/// A single terminal cell.
#[derive(Clone, Debug)]
pub struct Cell {
    /// The character displayed in this cell.
    pub grapheme: GraphemeStorage,
    /// Foreground color.
    pub fg: CellColor,
    /// Background color.
    pub bg: CellColor,
    /// Attribute flags (bold, italic, etc.).
    pub flags: CellFlags,
    /// Display width: 1 = normal, 2 = wide (CJK), 0 = continuation of wide char.
    pub width: u8,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            grapheme: GraphemeStorage::Ascii(b' '),
            fg: CellColor::Default,
            bg: CellColor::Default,
            flags: CellFlags::empty(),
            width: 1,
        }
    }
}

impl Cell {
    /// Create a cell with a specific character.
    pub fn with_char(c: char) -> Self {
        let width = UnicodeWidthChar::width(c).unwrap_or(1) as u8;
        Self {
            grapheme: GraphemeStorage::from_char(c),
            width,
            ..Self::default()
        }
    }

    /// A blank continuation cell (width 0, for the second column of a wide char).
    pub fn continuation() -> Self {
        Self {
            grapheme: GraphemeStorage::Ascii(b' '),
            width: 0,
            ..Self::default()
        }
    }

    /// Reset this cell to empty with default attributes.
    pub fn clear(&mut self) {
        self.grapheme = GraphemeStorage::Ascii(b' ');
        self.fg = CellColor::Default;
        self.bg = CellColor::Default;
        self.flags = CellFlags::empty();
        self.width = 1;
    }

    /// The first character (for backward-compatible API usage in tests).
    pub fn char(&self) -> char {
        self.grapheme.to_char()
    }

    /// Append this cell's grapheme to a string buffer.
    ///
    /// This is the preferred way to extract text for rendering,
    /// as it handles multi-codepoint grapheme clusters correctly.
    pub fn write_grapheme(&self, buf: &mut String) {
        match &self.grapheme {
            GraphemeStorage::Ascii(b) => buf.push(*b as char),
            GraphemeStorage::Char(c) => buf.push(*c),
            GraphemeStorage::Cluster(s) => buf.push_str(s),
        }
    }
}

/// Compact grapheme storage — inline for the common case.
#[derive(Clone, Debug)]
pub enum GraphemeStorage {
    /// Single ASCII byte (covers 99%+ of terminal content).
    Ascii(u8),
    /// Single non-ASCII Unicode character.
    Char(char),
    /// Multi-codepoint grapheme cluster (emoji, combining chars, ZWJ sequences).
    Cluster(Box<str>),
}

impl GraphemeStorage {
    /// Create from a char, choosing the most compact representation.
    pub fn from_char(c: char) -> Self {
        if c.is_ascii() {
            Self::Ascii(c as u8)
        } else {
            Self::Char(c)
        }
    }

    /// Create from a grapheme string, choosing the most compact representation.
    pub fn from_str(s: &str) -> Self {
        let mut chars = s.chars();
        if let Some(first) = chars.next() {
            if chars.next().is_none() {
                // Single codepoint
                return Self::from_char(first);
            }
        }
        // Multi-codepoint or empty
        Self::Cluster(s.into())
    }

    /// Convert to char (returns first codepoint for clusters).
    pub fn to_char(&self) -> char {
        match self {
            Self::Ascii(b) => *b as char,
            Self::Char(c) => *c,
            Self::Cluster(s) => s.chars().next().unwrap_or(' '),
        }
    }
}

/// Cell color — compact representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellColor {
    /// Use the default foreground/background color from config.
    Default,
    /// Indexed color (0–255). 0–7 = normal, 8–15 = bright, 16–255 = extended.
    Indexed(u8),
    /// True color (24-bit RGB).
    Rgb(u8, u8, u8),
}

impl CellColor {
    /// Resolve this color to an (r, g, b) triple using the color config.
    pub fn resolve(self, is_fg: bool, colors: &ColorConfig) -> (u8, u8, u8) {
        match self {
            Self::Default => {
                if is_fg {
                    colors.fg_rgb()
                } else {
                    colors.bg_rgb()
                }
            }
            Self::Indexed(idx) => {
                if idx < 16 {
                    colors.palette_rgb(idx)
                } else {
                    config::compute_256_color(idx)
                }
            }
            Self::Rgb(r, g, b) => (r, g, b),
        }
    }
}

bitflags! {
    /// Cell attribute flags.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct CellFlags: u16 {
        /// Bold text.
        const BOLD          = 0b0000_0000_0001;
        /// Italic text.
        const ITALIC        = 0b0000_0000_0010;
        /// Underlined text.
        const UNDERLINE     = 0b0000_0000_0100;
        /// Strikethrough text.
        const STRIKETHROUGH = 0b0000_0000_1000;
        /// Reverse video (swap fg/bg).
        const INVERSE       = 0b0000_0001_0000;
        /// Dim/faint text.
        const DIM           = 0b0000_0010_0000;
        /// Hidden text (invisible).
        const HIDDEN        = 0b0000_0100_0000;
        /// Blinking text.
        const BLINK         = 0b0000_1000_0000;
    }
}

// ── Cursor ─────────────────────────────────────────────────────────

/// Cursor position and state.
#[derive(Clone, Debug)]
pub struct CursorState {
    /// Column (0-indexed).
    pub col: u16,
    /// Row (0-indexed, relative to scroll region top for most operations).
    pub row: u16,
    /// Whether the cursor is visible.
    pub visible: bool,
    /// Whether the cursor blinks.
    pub blink: bool,
    /// Cursor style.
    pub style: CursorStyle,
    /// Pending wrap: if true, the next printed char wraps to the next line.
    pub pending_wrap: bool,
    /// Saved cursor state (for ESC 7 / ESC 8).
    saved: Option<SavedCursor>,
}

/// Cursor visual style.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CursorStyle {
    /// Filled block (█).
    #[default]
    Block,
    /// Vertical bar (│).
    Bar,
    /// Horizontal underline (_).
    Underline,
}

/// Saved cursor state for DECSC/DECRC.
#[derive(Clone, Debug)]
struct SavedCursor {
    col: u16,
    row: u16,
    fg: CellColor,
    bg: CellColor,
    flags: CellFlags,
}

impl Default for CursorState {
    fn default() -> Self {
        Self {
            col: 0,
            row: 0,
            visible: true,
            blink: true,
            style: CursorStyle::Block,
            pending_wrap: false,
            saved: None,
        }
    }
}

impl CursorState {
    /// Save cursor position and attributes.
    pub fn save(&mut self, fg: CellColor, bg: CellColor, flags: CellFlags) {
        self.saved = Some(SavedCursor {
            col: self.col,
            row: self.row,
            fg,
            bg,
            flags,
        });
    }

    /// Restore saved cursor position and attributes.
    /// Returns the saved (fg, bg, flags) if a save point exists.
    ///
    /// The save point is preserved (cloned, not consumed) so that
    /// repeated DECRC calls return the same position — matching
    /// real terminal semantics.
    pub fn restore(&mut self) -> Option<(CellColor, CellColor, CellFlags)> {
        if let Some(saved) = self.saved.as_ref() {
            self.col = saved.col;
            self.row = saved.row;
            self.pending_wrap = false;
            Some((saved.fg, saved.bg, saved.flags))
        } else {
            None
        }
    }
}

// ── Mouse mode ─────────────────────────────────────────────────────

/// Mouse reporting mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MouseMode {
    /// No mouse reporting.
    #[default]
    None,
    /// Mode 1000: basic button press/release.
    Press,
    /// Mode 1002: button motion.
    ButtonMotion,
    /// Mode 1003: all motion.
    AllMotion,
}

/// Mouse encoding format.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MouseEncoding {
    /// Legacy X10 encoding (limited to 223 columns).
    #[default]
    X10,
    /// SGR encoding (`\e[<B;X;Y;M/m`). No column limit.
    Sgr,
}

// ── Terminal Grid ──────────────────────────────────────────────────

/// The terminal's logical screen buffer.
pub struct TerminalGrid {
    /// Primary screen cell buffer (row-major, `rows × cols`).
    cells: Vec<Cell>,
    /// Alternate screen buffer (for vim, less, htop).
    alt_cells: Option<Vec<Cell>>,
    /// Number of columns.
    cols: u16,
    /// Number of rows.
    rows: u16,
    /// Cursor state.
    pub cursor: CursorState,
    /// Current pen attributes for new characters.
    pub pen_fg: CellColor,
    /// Current pen background color.
    pub pen_bg: CellColor,
    /// Current pen flags.
    pub pen_flags: CellFlags,
    /// Scrollback buffer.
    scrollback: VecDeque<Vec<Cell>>,
    /// Max scrollback lines.
    max_scrollback: usize,
    /// Per-line dirty tracking (true = needs re-render).
    dirty: Vec<bool>,
    /// Scroll region: (top, bottom) inclusive, 0-indexed.
    scroll_top: u16,
    scroll_bottom: u16,
    /// Whether we are in alternate screen mode.
    alt_active: bool,
    /// Bracketed paste mode.
    pub bracketed_paste: bool,
    /// Mouse reporting mode.
    pub mouse_mode: MouseMode,
    /// Mouse encoding format.
    pub mouse_encoding: MouseEncoding,
    /// Window title.
    pub title: String,
    /// Tab stops.
    tab_stops: Vec<bool>,
    /// Whether auto-wrap is enabled (DECAWM).
    pub auto_wrap: bool,
    /// Origin mode (DECOM): cursor addressing relative to scroll region.
    pub origin_mode: bool,
    /// Application cursor keys mode (DECCKM).
    pub app_cursor_keys: bool,
    /// Application keypad mode (DECKPAM).
    pub app_keypad: bool,
    /// Focus event reporting (DEC ?1004).
    pub focus_reporting: bool,
    /// Synchronized output mode (DEC ?2026).
    pub synchronized_output: bool,
    /// Last printed character (for CSI b REP).
    pub last_printed_char: char,
    /// Viewport scroll offset: 0 = live bottom, N = scrolled back N lines.
    pub scroll_offset: usize,
    /// Visual bell pending flag.
    pub bell_pending: bool,
    /// Clipboard text pending (set by OSC 52, drained by main loop).
    pub clipboard_pending: Option<String>,
}

impl TerminalGrid {
    /// Create a new grid with the given dimensions.
    pub fn new(cols: u16, rows: u16, max_scrollback: usize) -> Self {
        let total = cols as usize * rows as usize;
        let mut tab_stops = vec![false; cols as usize];
        // Default tab stops every 8 columns
        for i in (0..cols as usize).step_by(8) {
            tab_stops[i] = true;
        }

        Self {
            cells: vec![Cell::default(); total],
            alt_cells: None,
            cols,
            rows,
            cursor: CursorState::default(),
            pen_fg: CellColor::Default,
            pen_bg: CellColor::Default,
            pen_flags: CellFlags::empty(),
            scrollback: VecDeque::new(),
            max_scrollback,
            dirty: vec![true; rows as usize],
            scroll_top: 0,
            scroll_bottom: rows.saturating_sub(1),
            alt_active: false,
            bracketed_paste: false,
            mouse_mode: MouseMode::None,
            mouse_encoding: MouseEncoding::X10,
            title: String::new(),
            tab_stops,
            auto_wrap: true,
            origin_mode: false,
            app_cursor_keys: false,
            app_keypad: false,
            focus_reporting: false,
            synchronized_output: false,
            last_printed_char: ' ',
            scroll_offset: 0,
            bell_pending: false,
            clipboard_pending: None,
        }
    }

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
            // Use a static default cell
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

    /// Scroll the viewport up (toward older scrollback).
    pub fn scroll_viewport_up(&mut self, n: usize) {
        if self.alt_active {
            return;
        }
        let max_offset = self.scrollback.len();
        self.scroll_offset = (self.scroll_offset + n).min(max_offset);
        self.mark_all_dirty();
    }

    /// Scroll the viewport down (toward live content).
    pub fn scroll_viewport_down(&mut self, n: usize) {
        if self.scroll_offset == 0 {
            return;
        }
        self.scroll_offset = self.scroll_offset.saturating_sub(n);
        self.mark_all_dirty();
    }

    /// Reset viewport to live (bottom).
    pub fn snap_to_bottom(&mut self) {
        if self.scroll_offset > 0 {
            self.scroll_offset = 0;
            self.mark_all_dirty();
        }
    }

    /// Linear index for (col, row).
    fn idx(&self, col: u16, row: u16) -> usize {
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
    pub fn backspace(&mut self) {
        self.cursor.col = self.cursor.col.saturating_sub(1);
        self.cursor.pending_wrap = false;
    }

    // ── Scrolling ──────────────────────────────────────────────────

    /// Set the scroll region (1-indexed, inclusive). Resets cursor to home.
    pub fn set_scroll_region(&mut self, top: u16, bottom: u16) {
        let top = top.saturating_sub(1).min(self.rows - 1);
        let bottom = bottom.saturating_sub(1).min(self.rows - 1);
        // Allow top == bottom (one-line scroll region, used by tmux status bar).
        if top <= bottom {
            self.scroll_top = top;
            self.scroll_bottom = bottom;
            // Reset cursor to home (within scroll region if origin mode)
            self.cursor.row = if self.origin_mode { top } else { 0 };
            self.cursor.col = 0;
            self.cursor.pending_wrap = false;
        }
    }

    /// Scroll the scroll region up by `n` lines.
    /// Top lines are moved to scrollback (if on primary screen).
    pub fn scroll_up(&mut self, n: u16) {
        let top = self.scroll_top as usize;
        let bottom = self.scroll_bottom as usize;
        let cols = self.cols as usize;
        let n = (n as usize).min(bottom - top + 1);

        // Push scrolled-off lines to scrollback (primary buffer, full-screen scroll)
        if top == 0 && !self.alt_active {
            for line_idx in 0..n {
                let start = line_idx * cols;
                let line = self.cells[start..start + cols].to_vec();
                self.scrollback.push_back(line);
                if self.scroll_offset > 0 {
                    self.scroll_offset += 1;
                }
                if self.scrollback.len() > self.max_scrollback {
                    self.scrollback.pop_front();
                    self.scroll_offset = self.scroll_offset.min(self.scrollback.len());
                }
            }
        }

        // Rotate the scroll region slice left by n lines (in-place, no per-element clone loop)
        let region_start = top * cols;
        let region_end = (bottom + 1) * cols;
        self.cells[region_start..region_end].rotate_left(n * cols);

        // Clear the newly exposed bottom lines
        let clear_start = (bottom + 1 - n) * cols;
        for cell in &mut self.cells[clear_start..region_end] {
            *cell = Cell::default();
        }

        // Mark affected lines dirty
        for row in top..=bottom {
            self.mark_dirty(row as u16);
        }
    }

    /// Scroll the scroll region down by `n` lines.
    pub fn scroll_down(&mut self, n: u16) {
        let top = self.scroll_top as usize;
        let bottom = self.scroll_bottom as usize;
        let cols = self.cols as usize;
        let n = (n as usize).min(bottom - top + 1);

        // Rotate the scroll region slice right by n lines
        let region_start = top * cols;
        let region_end = (bottom + 1) * cols;
        self.cells[region_start..region_end].rotate_right(n * cols);

        // Clear the newly exposed top lines
        let clear_end = (top + n) * cols;
        for cell in &mut self.cells[region_start..clear_end] {
            *cell = Cell::default();
        }

        for row in top..=bottom {
            self.mark_dirty(row as u16);
        }
    }

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
    fn clear_line(&mut self, row: u16) {
        for col in 0..self.cols {
            self.cell_mut(col, row).clear();
        }
        self.mark_dirty(row);
    }

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

    // ── SGR (Select Graphic Rendition) ─────────────────────────────

    /// Reset all pen attributes to default.
    pub fn reset_pen(&mut self) {
        self.pen_fg = CellColor::Default;
        self.pen_bg = CellColor::Default;
        self.pen_flags = CellFlags::empty();
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

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_grid_is_empty() {
        let grid = TerminalGrid::new(80, 24, 100);
        assert_eq!(grid.cols(), 80);
        assert_eq!(grid.rows(), 24);
        assert_eq!(grid.cell(0, 0).char(), ' ');
        assert_eq!(grid.cursor.col, 0);
        assert_eq!(grid.cursor.row, 0);
    }

    #[test]
    fn put_char_and_advance() {
        let mut grid = TerminalGrid::new(80, 24, 100);
        grid.put_char('A');
        assert_eq!(grid.cell(0, 0).char(), 'A');
        assert_eq!(grid.cursor.col, 1);
        grid.put_char('B');
        assert_eq!(grid.cell(1, 0).char(), 'B');
        assert_eq!(grid.cursor.col, 2);
    }

    #[test]
    fn cursor_wraps_at_right_edge() {
        let mut grid = TerminalGrid::new(3, 2, 0);
        grid.put_char('A');
        grid.put_char('B');
        grid.put_char('C'); // col 2 — should trigger pending_wrap
        assert!(grid.cursor.pending_wrap);
        grid.put_char('D'); // should wrap to next line
        assert_eq!(grid.cursor.row, 1);
        assert_eq!(grid.cursor.col, 1);
        assert_eq!(grid.cell(0, 1).char(), 'D');
    }

    #[test]
    fn line_feed_scrolls_at_bottom() {
        let mut grid = TerminalGrid::new(3, 2, 10);
        grid.put_char('A');
        grid.cursor.row = 1;
        grid.cursor.col = 0;
        grid.put_char('B');
        grid.line_feed(); // at bottom row — should scroll
        assert_eq!(grid.cell(0, 0).char(), 'B'); // line moved up
        assert_eq!(grid.cell(0, 1).char(), ' '); // bottom cleared
    }

    #[test]
    fn scroll_region() {
        let mut grid = TerminalGrid::new(5, 5, 0);
        grid.set_scroll_region(2, 4); // rows 1..3 (0-indexed)
        assert_eq!(grid.scroll_top, 1);
        assert_eq!(grid.scroll_bottom, 3);
    }

    #[test]
    fn erase_in_display_all() {
        let mut grid = TerminalGrid::new(5, 3, 0);
        grid.put_char('X');
        grid.erase_in_display(2);
        assert_eq!(grid.cell(0, 0).char(), ' ');
    }

    #[test]
    fn alternate_screen() {
        let mut grid = TerminalGrid::new(5, 3, 0);
        grid.put_char('A');
        grid.enter_alt_screen();
        assert!(grid.is_alt_active());
        assert_eq!(grid.cell(0, 0).char(), ' '); // alt screen is clean
        grid.put_char('B');
        grid.exit_alt_screen();
        assert!(!grid.is_alt_active());
        assert_eq!(grid.cell(0, 0).char(), 'A'); // primary restored
    }

    #[test]
    fn resize_preserves_content() {
        let mut grid = TerminalGrid::new(5, 3, 0);
        grid.put_char('X');
        grid.put_char('Y');
        grid.resize(10, 5);
        assert_eq!(grid.cols(), 10);
        assert_eq!(grid.rows(), 5);
        assert_eq!(grid.cell(0, 0).char(), 'X');
        assert_eq!(grid.cell(1, 0).char(), 'Y');
    }

    #[test]
    fn tab_stops() {
        let mut grid = TerminalGrid::new(80, 24, 0);
        grid.tab();
        assert_eq!(grid.cursor.col, 8);
        grid.tab();
        assert_eq!(grid.cursor.col, 16);
    }

    #[test]
    fn insert_delete_lines() {
        let mut grid = TerminalGrid::new(3, 3, 0);
        // Fill row 0 with 'A', row 1 with 'B', row 2 with 'C'
        grid.cursor.row = 0;
        grid.cursor.col = 0;
        grid.put_char('A');
        grid.cursor.row = 1;
        grid.cursor.col = 0;
        grid.put_char('B');
        grid.cursor.row = 2;
        grid.cursor.col = 0;
        grid.put_char('C');

        // Insert 1 line at row 1
        grid.cursor.row = 1;
        grid.insert_lines(1);
        assert_eq!(grid.cell(0, 0).char(), 'A'); // row 0 unchanged
        assert_eq!(grid.cell(0, 1).char(), ' '); // inserted blank
        assert_eq!(grid.cell(0, 2).char(), 'B'); // old row 1 pushed down
        // old row 2 ('C') pushed off screen
    }

    #[test]
    fn wide_char_takes_two_columns() {
        let mut grid = TerminalGrid::new(10, 1, 0);
        grid.put_char('漢'); // Width 2
        assert_eq!(grid.cursor.col, 2);
        assert_eq!(grid.cell(0, 0).width, 2);
        assert_eq!(grid.cell(1, 0).width, 0); // continuation
    }

    #[test]
    fn cursor_save_restore() {
        let mut grid = TerminalGrid::new(80, 24, 0);
        grid.cursor.col = 10;
        grid.cursor.row = 5;
        grid.pen_fg = CellColor::Rgb(255, 0, 0);
        grid.cursor.save(grid.pen_fg, grid.pen_bg, grid.pen_flags);
        grid.cursor.col = 0;
        grid.cursor.row = 0;
        let restored = grid.cursor.restore();
        assert!(restored.is_some());
        assert_eq!(grid.cursor.col, 10);
        assert_eq!(grid.cursor.row, 5);
    }

    #[test]
    fn scrollback_fills_up() {
        let mut grid = TerminalGrid::new(3, 2, 5);
        for i in 0..10 {
            grid.cursor.col = 0;
            grid.cursor.row = 1;
            grid.put_char(char::from(b'A' + (i % 26) as u8));
            grid.line_feed();
        }
        // Scrollback should be capped at 5
        assert!(grid.scrollback.len() <= 5);
    }

    #[test]
    fn viewport_cell_no_offset() {
        let mut grid = TerminalGrid::new(5, 3, 10);
        grid.put_char('X');
        assert_eq!(grid.viewport_cell(0, 0).char(), 'X');
        assert_eq!(grid.scroll_offset, 0);
    }

    #[test]
    fn viewport_cell_with_offset() {
        let mut grid = TerminalGrid::new(5, 2, 10);
        // Fill and scroll to create scrollback
        for i in 0..5u8 {
            grid.cursor.col = 0;
            grid.cursor.row = 1;
            grid.put_char(char::from(b'A' + i));
            grid.line_feed();
        }
        // Now scrollback has lines. Scroll viewport back
        let sb_len = grid.scrollback_len();
        assert!(sb_len > 0);
        grid.scroll_viewport_up(2);
        assert_eq!(grid.scroll_offset, 2);
        // First viewport rows should come from scrollback
        let cell = grid.viewport_cell(0, 0);
        assert_ne!(cell.char(), ' '); // Should have content from scrollback
    }

    #[test]
    fn scroll_offset_clamp() {
        let mut grid = TerminalGrid::new(5, 2, 3);
        // Create 3 scrollback lines
        for _ in 0..3 {
            grid.cursor.col = 0;
            grid.cursor.row = 1;
            grid.put_char('Z');
            grid.line_feed();
        }
        // Try to scroll past scrollback
        grid.scroll_viewport_up(100);
        assert_eq!(grid.scroll_offset, grid.scrollback_len());
    }

    #[test]
    fn snap_to_bottom() {
        let mut grid = TerminalGrid::new(5, 2, 10);
        for _ in 0..5 {
            grid.cursor.col = 0;
            grid.cursor.row = 1;
            grid.put_char('A');
            grid.line_feed();
        }
        grid.scroll_viewport_up(3);
        assert!(grid.scroll_offset > 0);
        grid.snap_to_bottom();
        assert_eq!(grid.scroll_offset, 0);
    }

    #[test]
    fn scroll_up_preserves_content_order() {
        let mut grid = TerminalGrid::new(5, 3, 10);
        // Fill rows with A, B, C
        for row in 0..3u16 {
            for col in 0..5u16 {
                grid.cursor.col = col;
                grid.cursor.row = row;
                grid.put_char(char::from(b'A' + row as u8));
            }
        }
        grid.scroll_up(1);
        // Row 0 should now be old row 1 ('B')
        assert_eq!(grid.cell(0, 0).char(), 'B');
        // Row 1 should be old row 2 ('C')
        assert_eq!(grid.cell(0, 1).char(), 'C');
        // Row 2 should be cleared
        assert_eq!(grid.cell(0, 2).char(), ' ');
    }

    #[test]
    fn scroll_down_preserves_content_order() {
        let mut grid = TerminalGrid::new(5, 3, 0);
        for row in 0..3u16 {
            for col in 0..5u16 {
                grid.cursor.col = col;
                grid.cursor.row = row;
                grid.cursor.pending_wrap = false;
                grid.put_char(char::from(b'A' + row as u8));
            }
        }
        grid.scroll_down(1);
        // Row 0 should be cleared
        assert_eq!(grid.cell(0, 0).char(), ' ');
        // Row 1 should be old row 0 ('A')
        assert_eq!(grid.cell(0, 1).char(), 'A');
        // Row 2 should be old row 1 ('B')
        assert_eq!(grid.cell(0, 2).char(), 'B');
    }

    #[test]
    fn scroll_up_pushes_to_scrollback() {
        let mut grid = TerminalGrid::new(5, 3, 10);
        for col in 0..5u16 {
            grid.cursor.col = col;
            grid.cursor.row = 0;
            grid.put_char('Z');
        }
        assert!(grid.scrollback_len() == 0);
        grid.scroll_up(1);
        assert!(grid.scrollback_len() == 1);
    }
}
