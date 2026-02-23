// SPDX-License-Identifier: MIT OR Apache-2.0
//! Terminal grid — the logical model of the terminal screen.
//!
//! `TerminalGrid` is a 2D array of [`Cell`]s with cursor state, scroll regions,
//! alternate screen buffer, and scrollback history. It is purely a data model —
//! it has no knowledge of rendering, escape sequences, or I/O.

mod buffer;
mod cursor;
mod edit;
mod erase;
mod pen;
mod screen;
mod scroll;
mod types;
mod write;

#[cfg(test)]
mod tests;

pub use types::*;

use std::collections::VecDeque;

// ── Terminal Grid ──────────────────────────────────────────────────

/// The terminal's logical screen buffer.
pub struct TerminalGrid {
    /// Primary screen cell buffer (row-major, `rows × cols`).
    pub(crate) cells: Vec<Cell>,
    /// Alternate screen buffer (for vim, less, htop).
    pub(crate) alt_cells: Option<Vec<Cell>>,
    /// Number of columns.
    pub(crate) cols: u16,
    /// Number of rows.
    pub(crate) rows: u16,
    /// Cursor state.
    pub cursor: CursorState,
    /// Current pen attributes for new characters.
    pub pen_fg: CellColor,
    /// Current pen background color.
    pub pen_bg: CellColor,
    /// Current pen flags.
    pub pen_flags: CellFlags,
    /// Pen underline color (SGR 58;2;r;g;b). `None` = use fg.
    pub pen_underline_color: Option<(u8, u8, u8)>,
    /// Scrollback buffer.
    pub(crate) scrollback: VecDeque<Vec<Cell>>,
    /// Max scrollback lines.
    pub(crate) max_scrollback: usize,
    /// Per-line dirty tracking (true = needs re-render).
    pub(crate) dirty: Vec<bool>,
    /// Scroll region: (top, bottom) inclusive, 0-indexed.
    pub(crate) scroll_top: u16,
    pub(crate) scroll_bottom: u16,
    /// Whether we are in alternate screen mode.
    pub(crate) alt_active: bool,
    /// Bracketed paste mode.
    pub bracketed_paste: bool,
    /// Mouse reporting mode.
    pub mouse_mode: MouseMode,
    /// Mouse encoding format.
    pub mouse_encoding: MouseEncoding,
    /// Window title.
    pub title: String,
    /// Tab stops.
    pub(crate) tab_stops: Vec<bool>,
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
            pen_underline_color: None,
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
}
