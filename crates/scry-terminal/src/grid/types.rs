// SPDX-License-Identifier: MIT OR Apache-2.0
//! Cell types, color, flags, cursor state, and mouse modes.

use bitflags::bitflags;
use unicode_width::UnicodeWidthChar;

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
    /// Optional underline color (SGR 58). `None` = use fg color.
    pub underline_color: Option<(u8, u8, u8)>,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            grapheme: GraphemeStorage::Ascii(b' '),
            fg: CellColor::Default,
            bg: CellColor::Default,
            flags: CellFlags::empty(),
            width: 1,
            underline_color: None,
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
        self.underline_color = None;
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
        /// Underlined text (single line — default underline style).
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
        /// Double underline (SGR 4:2 / SGR 21).
        const UNDERLINE_DOUBLE = 0b0001_0000_0000;
        /// Curly underline (SGR 4:3).
        const UNDERLINE_CURLY  = 0b0010_0000_0000;
        /// Dotted underline (SGR 4:4).
        const UNDERLINE_DOTTED = 0b0100_0000_0000;
        /// Dashed underline (SGR 4:5).
        const UNDERLINE_DASHED = 0b1000_0000_0000;
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
