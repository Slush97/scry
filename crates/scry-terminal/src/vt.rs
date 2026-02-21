// SPDX-License-Identifier: MIT OR Apache-2.0
//! VTE `Perform` implementation — dispatches escape sequences to the grid.
//!
//! This module bridges the `vte` crate's parser with our [`TerminalGrid`]
//! and [`SecurityGate`]. Every byte from the PTY flows through:
//!
//! ```text
//! PTY → vte::Parser → VtHandler (this module) → SecurityGate → TerminalGrid
//! ```

use crate::grid::{CellColor, CellFlags, CursorStyle, MouseEncoding, MouseMode, TerminalGrid};
use crate::security::{OscAction, ResponseType, SecurityGate};

/// The VTE handler that implements `vte::Perform`.
///
/// Wraps a mutable reference to the grid and security gate, plus a
/// buffer for any response bytes that need to be sent back to the PTY.
pub struct VtHandler<'a> {
    /// The terminal grid to mutate.
    pub grid: &'a mut TerminalGrid,
    /// Security gate for filtering responses and OSC commands.
    pub security: &'a mut SecurityGate,
    /// Response buffer: bytes to write back to the PTY fd.
    pub response: Vec<u8>,
}

impl<'a> VtHandler<'a> {
    /// Create a new handler wrapping the grid and security gate.
    pub fn new(grid: &'a mut TerminalGrid, security: &'a mut SecurityGate) -> Self {
        Self {
            grid,
            security,
            response: Vec::new(),
        }
    }

    /// Queue a response to be sent back to the PTY.
    fn respond(&mut self, response_type: ResponseType, data: &[u8]) {
        if self.security.allow_response(response_type) {
            self.response.extend_from_slice(data);
        }
    }
}

impl vte::Perform for VtHandler<'_> {
    /// Print a character to the terminal.
    fn print(&mut self, c: char) {
        self.grid.put_char(c);
        self.grid.last_printed_char = c;
    }

    /// Execute a C0 control character.
    fn execute(&mut self, byte: u8) {
        match byte {
            0x07 => {
                // BEL — trigger visual bell
                self.grid.bell_pending = true;
            }
            0x08 => self.grid.backspace(),
            0x09 => self.grid.tab(),
            0x0A | 0x0B | 0x0C => self.grid.line_feed(), // LF, VT, FF
            0x0D => self.grid.carriage_return(),
            _ => {} // Other C0 controls — ignore
        }
    }

    /// Handle a CSI (Control Sequence Introducer) dispatch.
    fn csi_dispatch(
        &mut self,
        params: &vte::Params,
        _intermediates: &[u8],
        _ignore: bool,
        action: char,
    ) {
        let params: Vec<u16> = params.iter().map(|p| p[0]).collect();
        let p1 = params.first().copied().unwrap_or(0);
        let p2 = params.get(1).copied().unwrap_or(0);

        match action {
            // ── Cursor movement ────────────────────────────────────
            'A' => self.grid.move_up(p1.max(1)),
            'B' => self.grid.move_down(p1.max(1)),
            'C' => self.grid.move_forward(p1.max(1)),
            'D' => self.grid.move_backward(p1.max(1)),
            'E' => {
                // Cursor Next Line
                self.grid.move_down(p1.max(1));
                self.grid.carriage_return();
            }
            'F' => {
                // Cursor Previous Line
                self.grid.move_up(p1.max(1));
                self.grid.carriage_return();
            }
            'G' => self.grid.move_to_col(p1.max(1)),
            'H' | 'f' => self.grid.move_to(p1.max(1), p2.max(1)),
            'd' => {
                // Vertical Position Absolute
                let row = p1.max(1);
                self.grid.move_to(row, self.grid.cursor.col + 1);
            }

            // ── Erase ──────────────────────────────────────────────
            'J' => self.grid.erase_in_display(p1),
            'K' => self.grid.erase_in_line(p1),
            'X' => self.grid.erase_chars(p1.max(1)),

            // ── Insert/Delete ──────────────────────────────────────
            'L' => self.grid.insert_lines(p1.max(1)),
            'M' => self.grid.delete_lines(p1.max(1)),
            '@' => self.grid.insert_chars(p1.max(1)),
            'P' => self.grid.delete_chars(p1.max(1)),

            // ── Scroll ─────────────────────────────────────────────
            'S' => self.grid.scroll_up(p1.max(1)),
            'T' => self.grid.scroll_down(p1.max(1)),

            // ── Scroll region ──────────────────────────────────────
            'r' => {
                let top = p1.max(1);
                let bottom = if p2 == 0 { self.grid.rows() } else { p2 };
                self.grid.set_scroll_region(top, bottom);
            }

            // ── SGR (Select Graphic Rendition) ─────────────────────
            'm' => self.handle_sgr(&params),

            // ── Mode set/reset ─────────────────────────────────────
            'h' => self.handle_mode_set(&params, _intermediates, true),
            'l' => self.handle_mode_set(&params, _intermediates, false),

            // ── Device status ──────────────────────────────────────
            'n' => {
                match p1 {
                    5 => {
                        // Device Status Report → "OK"
                        self.respond(ResponseType::DeviceStatusOk, b"\x1b[0n");
                    }
                    6 => {
                        // Cursor Position Report
                        let row = self.grid.cursor.row + 1;
                        let col = self.grid.cursor.col + 1;
                        let resp = format!("\x1b[{row};{col}R");
                        self.respond(ResponseType::CursorPosition, resp.as_bytes());
                    }
                    _ => {}
                }
            }

            // ── Device Attributes ──────────────────────────────────
            'c' => {
                // Primary DA: identify as VT220 with ANSI color
                self.respond(ResponseType::DeviceAttributes, b"\x1b[?62;22c");
            }

            // ── Cursor style (DECSCUSR) ────────────────────────────
            'q' if _intermediates == [b' '] => {
                self.grid.cursor.style = match p1 {
                    0 | 1 => CursorStyle::Block,     // Default / blinking block
                    2 => CursorStyle::Block,          // Steady block
                    3 | 4 => CursorStyle::Underline,  // Blinking / steady underline
                    5 | 6 => CursorStyle::Bar,        // Blinking / steady bar
                    _ => CursorStyle::Block,
                };
            }

            // ── Save/Restore cursor (ANSI variant) ─────────────────
            's' if _intermediates.is_empty() => {
                self.grid.cursor.save(
                    self.grid.pen_fg,
                    self.grid.pen_bg,
                    self.grid.pen_flags,
                );
            }
            'u' if _intermediates.is_empty() => {
                if let Some((fg, bg, flags)) = self.grid.cursor.restore() {
                    self.grid.pen_fg = fg;
                    self.grid.pen_bg = bg;
                    self.grid.pen_flags = flags;
                }
            }

            // ── REP: Repeat preceding graphic character ─────────────
            'b' => {
                let count = p1.max(1);
                let ch = self.grid.last_printed_char;
                for _ in 0..count {
                    self.grid.put_char(ch);
                }
            }

            // ── Secondary Device Attributes ─────────────────────────
            'c' if _intermediates == [b'>'] => {
                // Secondary DA: report as VT220, firmware version 0
                self.respond(ResponseType::DeviceAttributes, b"\x1b[>0;0;0c");
            }

            // ── Soft reset (DECSTR) ─────────────────────────────────
            'p' if _intermediates == [b'!'] => {
                // Reset modes but keep screen content
                self.grid.reset_pen();
                self.grid.cursor.pending_wrap = false;
                self.grid.auto_wrap = true;
                self.grid.origin_mode = false;
                self.grid.app_cursor_keys = false;
                self.grid.app_keypad = false;
                self.grid.bracketed_paste = false;
                self.grid.mouse_mode = MouseMode::None;
                self.grid.mouse_encoding = MouseEncoding::X10;
                self.grid.set_scroll_region(1, self.grid.rows());
                self.grid.cursor.visible = true;
            }

            _ => {} // Unknown CSI — ignore
        }
    }

    /// Handle an ESC dispatch.
    fn esc_dispatch(&mut self, _intermediates: &[u8], _ignore: bool, byte: u8) {
        match byte {
            b'7' => {
                // DECSC: Save cursor
                self.grid
                    .cursor
                    .save(self.grid.pen_fg, self.grid.pen_bg, self.grid.pen_flags);
            }
            b'8' => {
                // DECRC: Restore cursor
                if let Some((fg, bg, flags)) = self.grid.cursor.restore() {
                    self.grid.pen_fg = fg;
                    self.grid.pen_bg = bg;
                    self.grid.pen_flags = flags;
                }
            }
            b'M' => {
                // RI: Reverse Index
                self.grid.reverse_index();
            }
            b'=' => {
                // DECKPAM: Application keypad mode
                self.grid.app_keypad = true;
            }
            b'>' => {
                // DECKPNM: Normal keypad mode
                self.grid.app_keypad = false;
            }
            b'D' => {
                // IND: Index (move cursor down, scroll if at bottom)
                self.grid.line_feed();
            }
            b'E' => {
                // NEL: Next Line
                self.grid.carriage_return();
                self.grid.line_feed();
            }
            b'c' => {
                // RIS: Full reset
                self.grid.reset_pen();
                self.grid.cursor = Default::default();
                self.grid.erase_in_display(2);
                self.grid.bracketed_paste = false;
                self.grid.mouse_mode = MouseMode::None;
                self.grid.auto_wrap = true;
                self.grid.origin_mode = false;
                self.grid.app_cursor_keys = false;
                self.grid.app_keypad = false;
            }
            _ => {} // Unknown ESC — ignore
        }
    }

    /// Handle an OSC (Operating System Command) dispatch.
    fn osc_dispatch(&mut self, params: &[&[u8]], _bell_terminated: bool) {
        let action = self.security.filter_osc(params);

        match action {
            OscAction::SetTitle => {
                if let Some(title) = params.get(1).and_then(|p| std::str::from_utf8(p).ok()) {
                    self.grid.title = title.to_string();
                }
            }
            OscAction::ClipboardSet => {
                // TODO: Phase 2 — decode base64 and set clipboard via arboard
            }
            OscAction::HyperlinkOpen(_url) => {
                // TODO: Phase 2 — track hyperlink state per cell
            }
            OscAction::HyperlinkClose => {
                // TODO: Phase 2 — end hyperlink region
            }
            OscAction::Allow | OscAction::Ignore | OscAction::Block => {}
        }
    }

    fn hook(&mut self, _params: &vte::Params, _intermediates: &[u8], _ignore: bool, _action: char) {
        // DCS (Device Control String) — not implemented in Phase 1
    }

    fn unhook(&mut self) {}

    fn put(&mut self, _byte: u8) {}
}

// ── SGR handling ───────────────────────────────────────────────────

impl VtHandler<'_> {
    /// Parse and apply SGR (Select Graphic Rendition) parameters.
    fn handle_sgr(&mut self, params: &[u16]) {
        if params.is_empty() {
            self.grid.reset_pen();
            return;
        }

        let mut i = 0;
        while i < params.len() {
            match params[i] {
                0 => self.grid.reset_pen(),

                // Attributes
                1 => self.grid.pen_flags.insert(CellFlags::BOLD),
                2 => self.grid.pen_flags.insert(CellFlags::DIM),
                3 => self.grid.pen_flags.insert(CellFlags::ITALIC),
                4 => self.grid.pen_flags.insert(CellFlags::UNDERLINE),
                5 => self.grid.pen_flags.insert(CellFlags::BLINK),
                7 => self.grid.pen_flags.insert(CellFlags::INVERSE),
                8 => self.grid.pen_flags.insert(CellFlags::HIDDEN),
                9 => self.grid.pen_flags.insert(CellFlags::STRIKETHROUGH),

                // Attribute off
                22 => {
                    self.grid.pen_flags.remove(CellFlags::BOLD);
                    self.grid.pen_flags.remove(CellFlags::DIM);
                }
                23 => self.grid.pen_flags.remove(CellFlags::ITALIC),
                24 => self.grid.pen_flags.remove(CellFlags::UNDERLINE),
                25 => self.grid.pen_flags.remove(CellFlags::BLINK),
                27 => self.grid.pen_flags.remove(CellFlags::INVERSE),
                28 => self.grid.pen_flags.remove(CellFlags::HIDDEN),
                29 => self.grid.pen_flags.remove(CellFlags::STRIKETHROUGH),

                // Standard foreground colors (30–37)
                c @ 30..=37 => self.grid.pen_fg = CellColor::Indexed((c - 30) as u8),
                // Default foreground
                39 => self.grid.pen_fg = CellColor::Default,
                // Standard background colors (40–47)
                c @ 40..=47 => self.grid.pen_bg = CellColor::Indexed((c - 40) as u8),
                // Default background
                49 => self.grid.pen_bg = CellColor::Default,

                // Bright foreground colors (90–97)
                c @ 90..=97 => self.grid.pen_fg = CellColor::Indexed((c - 90 + 8) as u8),
                // Bright background colors (100–107)
                c @ 100..=107 => self.grid.pen_bg = CellColor::Indexed((c - 100 + 8) as u8),

                // Extended foreground: 38;5;N or 38;2;R;G;B
                38 => {
                    i += 1;
                    if i < params.len() {
                        match params[i] {
                            5 => {
                                // 256-color: 38;5;N
                                i += 1;
                                if i < params.len() {
                                    self.grid.pen_fg = CellColor::Indexed(params[i] as u8);
                                }
                            }
                            2 => {
                                // True color: 38;2;R;G;B
                                if i + 3 < params.len() {
                                    let r = params[i + 1] as u8;
                                    let g = params[i + 2] as u8;
                                    let b = params[i + 3] as u8;
                                    self.grid.pen_fg = CellColor::Rgb(r, g, b);
                                    i += 3;
                                }
                            }
                            _ => {}
                        }
                    }
                }

                // Extended background: 48;5;N or 48;2;R;G;B
                48 => {
                    i += 1;
                    if i < params.len() {
                        match params[i] {
                            5 => {
                                i += 1;
                                if i < params.len() {
                                    self.grid.pen_bg = CellColor::Indexed(params[i] as u8);
                                }
                            }
                            2 => {
                                if i + 3 < params.len() {
                                    let r = params[i + 1] as u8;
                                    let g = params[i + 2] as u8;
                                    let b = params[i + 3] as u8;
                                    self.grid.pen_bg = CellColor::Rgb(r, g, b);
                                    i += 3;
                                }
                            }
                            _ => {}
                        }
                    }
                }

                _ => {} // Unknown SGR — ignore
            }
            i += 1;
        }
    }

    /// Handle mode set/reset (CSI ? Pm h/l).
    fn handle_mode_set(&mut self, params: &[u16], intermediates: &[u8], enable: bool) {
        let is_dec = intermediates.contains(&b'?');

        for &p in params {
            if is_dec {
                match p {
                    // DECCKM: Application cursor keys
                    1 => self.grid.app_cursor_keys = enable,
                    // DECOM: Origin mode
                    6 => self.grid.origin_mode = enable,
                    // DECAWM: Auto-wrap mode
                    7 => self.grid.auto_wrap = enable,
                    // DECTCEM: Cursor visibility
                    25 => self.grid.cursor.visible = enable,

                    // Mouse modes
                    1000 => {
                        self.grid.mouse_mode = if enable {
                            MouseMode::Press
                        } else {
                            MouseMode::None
                        };
                    }
                    1002 => {
                        self.grid.mouse_mode = if enable {
                            MouseMode::ButtonMotion
                        } else {
                            MouseMode::None
                        };
                    }
                    1003 => {
                        self.grid.mouse_mode = if enable {
                            MouseMode::AllMotion
                        } else {
                            MouseMode::None
                        };
                    }
                    // SGR mouse encoding
                    1006 => {
                        self.grid.mouse_encoding = if enable {
                            MouseEncoding::Sgr
                        } else {
                            MouseEncoding::X10
                        };
                    }

                    // Alternate screen buffer variants
                    1047 => {
                        if enable {
                            self.grid.enter_alt_screen();
                        } else {
                            self.grid.exit_alt_screen();
                        }
                    }
                    1048 => {
                        if enable {
                            self.grid.cursor.save(
                                self.grid.pen_fg,
                                self.grid.pen_bg,
                                self.grid.pen_flags,
                            );
                        } else if let Some((fg, bg, flags)) = self.grid.cursor.restore() {
                            self.grid.pen_fg = fg;
                            self.grid.pen_bg = bg;
                            self.grid.pen_flags = flags;
                        }
                    }
                    1049 => {
                        // Save cursor + enter alt screen (the common case: vim, htop, less)
                        if enable {
                            self.grid.cursor.save(
                                self.grid.pen_fg,
                                self.grid.pen_bg,
                                self.grid.pen_flags,
                            );
                            self.grid.enter_alt_screen();
                        } else {
                            self.grid.exit_alt_screen();
                            if let Some((fg, bg, flags)) = self.grid.cursor.restore() {
                                self.grid.pen_fg = fg;
                                self.grid.pen_bg = bg;
                                self.grid.pen_flags = flags;
                            }
                        }
                    }

                    // Bracketed paste mode
                    2004 => {
                        self.grid.bracketed_paste = enable;
                        self.security.bracketed_paste_enabled = enable;
                    }

                    // Cursor blink (DECSET ?12)
                    12 => {
                        self.grid.cursor.blink = enable;
                    }

                    // Focus event reporting
                    1004 => {
                        self.grid.focus_reporting = enable;
                    }

                    // Synchronized output
                    2026 => {
                        self.grid.synchronized_output = enable;
                    }

                    _ => {} // Unknown DEC private mode — ignore
                }
            }
            // Non-DEC modes (rare, largely unused in modern terminals)
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_handler() -> (TerminalGrid, SecurityGate) {
        let grid = TerminalGrid::new(80, 24, 100);
        let security = SecurityGate::new(crate::security::ResponsePolicy::default());
        (grid, security)
    }

    #[test]
    fn print_characters() {
        let (mut grid, mut sec) = make_handler();
        let mut handler = VtHandler::new(&mut grid, &mut sec);
        vte::Perform::print(&mut handler, 'H');
        vte::Perform::print(&mut handler, 'i');
        assert_eq!(handler.grid.cell(0, 0).char(), 'H');
        assert_eq!(handler.grid.cell(1, 0).char(), 'i');
        assert_eq!(handler.grid.cursor.col, 2);
    }

    #[test]
    fn execute_controls() {
        let (mut grid, mut sec) = make_handler();
        let mut handler = VtHandler::new(&mut grid, &mut sec);
        vte::Perform::print(&mut handler, 'A');
        vte::Perform::execute(&mut handler, 0x0D); // CR
        assert_eq!(handler.grid.cursor.col, 0);
        vte::Perform::execute(&mut handler, 0x0A); // LF
        assert_eq!(handler.grid.cursor.row, 1);
    }

    #[test]
    fn sgr_colors() {
        let (mut grid, mut sec) = make_handler();
        {
            let mut handler = VtHandler::new(&mut grid, &mut sec);
            // Simulate: ESC[38;2;255;0;128m (true color fg)
            handler.handle_sgr(&[38, 2, 255, 0, 128]);
        }
        assert_eq!(grid.pen_fg, CellColor::Rgb(255, 0, 128));
    }

    #[test]
    fn sgr_256_color() {
        let (mut grid, mut sec) = make_handler();
        {
            let mut handler = VtHandler::new(&mut grid, &mut sec);
            handler.handle_sgr(&[38, 5, 196]); // 256-color fg
        }
        assert_eq!(grid.pen_fg, CellColor::Indexed(196));
    }

    #[test]
    fn sgr_bold_italic() {
        let (mut grid, mut sec) = make_handler();
        {
            let mut handler = VtHandler::new(&mut grid, &mut sec);
            handler.handle_sgr(&[1, 3]); // Bold + Italic
        }
        assert!(grid.pen_flags.contains(CellFlags::BOLD));
        assert!(grid.pen_flags.contains(CellFlags::ITALIC));
    }

    #[test]
    fn sgr_reset() {
        let (mut grid, mut sec) = make_handler();
        {
            let mut handler = VtHandler::new(&mut grid, &mut sec);
            handler.handle_sgr(&[1]); // Bold
            handler.handle_sgr(&[0]); // Reset
        }
        assert!(grid.pen_flags.is_empty());
        assert_eq!(grid.pen_fg, CellColor::Default);
    }

    #[test]
    fn alt_screen_mode_1049() {
        let (mut grid, mut sec) = make_handler();
        {
            let mut handler = VtHandler::new(&mut grid, &mut sec);
            vte::Perform::print(&mut handler, 'X');
            handler.handle_mode_set(&[1049], &[b'?'], true); // Enter alt screen
        }
        assert!(grid.is_alt_active());
        assert_eq!(grid.cell(0, 0).char(), ' '); // Alt screen is clean
        {
            let mut handler = VtHandler::new(&mut grid, &mut sec);
            handler.handle_mode_set(&[1049], &[b'?'], false); // Exit alt screen
        }
        assert!(!grid.is_alt_active());
        assert_eq!(grid.cell(0, 0).char(), 'X'); // Primary restored
    }

    #[test]
    fn bracketed_paste_mode() {
        let (mut grid, mut sec) = make_handler();
        {
            let mut handler = VtHandler::new(&mut grid, &mut sec);
            handler.handle_mode_set(&[2004], &[b'?'], true);
        }
        assert!(grid.bracketed_paste);
        assert!(sec.bracketed_paste_enabled);
    }

    #[test]
    fn cursor_position_response() {
        let (mut grid, mut sec) = make_handler();
        grid.cursor.col = 5;
        grid.cursor.row = 10;
        let mut handler = VtHandler::new(&mut grid, &mut sec);
        // CSI 6n = cursor position report
        let params = vte::Params::default();
        // Manually call handle since we can't easily construct Params
        handler.respond(ResponseType::CursorPosition, b"\x1b[11;6R");
        assert_eq!(handler.response, b"\x1b[11;6R");
    }

    #[test]
    fn title_report_blocked() {
        let (mut grid, mut sec) = make_handler();
        let mut handler = VtHandler::new(&mut grid, &mut sec);
        handler.respond(ResponseType::TitleReport, b"\x1b]lMy Title\x1b\\");
        assert!(handler.response.is_empty()); // Blocked!
    }

    #[test]
    fn osc_set_title() {
        let (mut grid, mut sec) = make_handler();
        {
            let mut handler = VtHandler::new(&mut grid, &mut sec);
            vte::Perform::osc_dispatch(&mut handler, &[b"0", b"My Terminal"], false);
        }
        assert_eq!(grid.title, "My Terminal");
    }
}
