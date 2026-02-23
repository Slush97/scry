// SPDX-License-Identifier: MIT OR Apache-2.0
//! Scroll operations: scroll region, scroll up/down, viewport scrolling.

use super::types::Cell;
use super::TerminalGrid;

impl TerminalGrid {
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
}
