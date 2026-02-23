// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tests for the terminal grid.

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

#[test]
fn backspace_at_col_zero_is_noop() {
    let mut grid = TerminalGrid::new(80, 24, 0);
    assert_eq!(grid.cursor.col, 0);
    assert_eq!(grid.cursor.row, 0);
    grid.backspace();
    assert_eq!(grid.cursor.col, 0);
    assert_eq!(grid.cursor.row, 0);
    // Repeated backspace still no-op
    grid.backspace();
    grid.backspace();
    assert_eq!(grid.cursor.col, 0);
}

#[test]
fn backspace_cancels_pending_wrap() {
    let mut grid = TerminalGrid::new(3, 2, 0);
    grid.put_char('A');
    grid.put_char('B');
    grid.put_char('C'); // triggers pending_wrap, cursor at col 2
    assert!(grid.cursor.pending_wrap);
    assert_eq!(grid.cursor.col, 2);
    grid.backspace();
    assert!(!grid.cursor.pending_wrap);
    // Cursor stays at cols-1 (wrap cancelled, no movement)
    assert_eq!(grid.cursor.col, 2);
}

#[test]
fn backspace_from_mid_line() {
    let mut grid = TerminalGrid::new(80, 24, 0);
    grid.put_char('A');
    grid.put_char('B');
    grid.put_char('C');
    assert_eq!(grid.cursor.col, 3);
    grid.backspace();
    assert_eq!(grid.cursor.col, 2);
    grid.backspace();
    assert_eq!(grid.cursor.col, 1);
    grid.backspace();
    assert_eq!(grid.cursor.col, 0);
    grid.backspace(); // should be no-op
    assert_eq!(grid.cursor.col, 0);
}
