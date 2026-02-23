// SPDX-License-Identifier: MIT OR Apache-2.0
//! Window event handler methods for `TerminalState`.

use winit::event::{ElementState, MouseButton, MouseScrollDelta};
use winit::keyboard::Key;

use scry_terminal::grid::MouseMode;
use scry_terminal::input;
use scry_terminal::platform::TerminalSize;
use scry_terminal::selection::SelectionAnchor;

use super::{pty_write, zoom_to, TerminalState};

impl TerminalState {
    pub(crate) fn handle_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.compositor.resize(new_size.width, new_size.height);

            let term_size = TerminalSize::from_window(
                new_size.width,
                new_size.height,
                self.compositor.cell_width(),
                self.compositor.cell_height(),
                self.padding,
            );

            self.grid.resize(term_size.cols, term_size.rows);

            let _ = self.pty.resize(
                term_size.cols,
                term_size.rows,
                term_size.pixel_width,
                term_size.pixel_height,
            );

            // Update IPC terminal info for QueryInfo responses
            if let Some(info) = &self.ipc_info {
                if let Ok(mut info) = info.write() {
                    info.cols = term_size.cols;
                    info.rows = term_size.rows;
                }
            }

            self.scheduler.request_redraw();
        }
    }

    /// Handle keyboard input. Returns `true` if the event was consumed
    /// (caller should not process further).
    pub(crate) fn handle_keyboard(&mut self, event: &winit::event::KeyEvent) -> bool {
        // Intercept Shift+PageUp/Down for viewport scrolling
        if event.state == ElementState::Pressed && self.modifiers.shift_key() {
            match &event.logical_key {
                Key::Named(winit::keyboard::NamedKey::PageUp) => {
                    let half = (self.grid.rows() / 2).max(1) as usize;
                    self.grid.scroll_viewport_up(half);
                    self.scheduler.request_redraw();
                    return true;
                }
                Key::Named(winit::keyboard::NamedKey::PageDown) => {
                    let half = (self.grid.rows() / 2).max(1) as usize;
                    self.grid.scroll_viewport_down(half);
                    self.scheduler.request_redraw();
                    return true;
                }
                _ => {}
            }
        }

        // Intercept Ctrl+Shift+C (copy) and Ctrl+Shift+V (paste)
        if event.state == ElementState::Pressed
            && self.modifiers.control_key()
            && self.modifiers.shift_key()
        {
            match &event.logical_key {
                Key::Character(ch) if ch.eq_ignore_ascii_case("c") => {
                    // Copy selection to clipboard
                    if !self.selection.is_empty() {
                        let text = self.selection.selected_text(&self.grid);
                        if let Some(clip) = &mut self.clipboard {
                            let _ = clip.set_text(&text);
                        }
                    }
                    return true;
                }
                Key::Character(ch) if ch.eq_ignore_ascii_case("v") => {
                    // Paste from clipboard
                    if let Some(clip) = &mut self.clipboard {
                        if let Ok(text) = clip.get_text() {
                            let bytes =
                                input::encode_paste(&text, self.grid.bracketed_paste);
                            pty_write(&mut self.pty, &bytes);
                        }
                    }
                    return true;
                }
                _ => {}
            }
        }

        // Zoom: Ctrl+= / Ctrl+- / Ctrl+0
        if event.state == ElementState::Pressed && self.modifiers.control_key() {
            match &event.logical_key {
                Key::Character(ch) if ch.as_str() == "=" || ch.as_str() == "+" => {
                    let new_size = self.compositor.font_size() + 1.0;
                    zoom_to(self, new_size);
                    return true;
                }
                Key::Character(ch) if ch.as_str() == "-" => {
                    let new_size = (self.compositor.font_size() - 1.0).max(8.0);
                    zoom_to(self, new_size);
                    return true;
                }
                Key::Character(ch) if ch.as_str() == "0" => {
                    zoom_to(self, self.original_font_size);
                    return true;
                }
                _ => {}
            }
        }

        if let Some(bytes) = input::encode_key(
            &event.logical_key,
            event.physical_key,
            event.state,
            self.modifiers,
            self.grid.app_cursor_keys,
            self.grid.app_keypad,
        ) {
            // User is typing — snap to bottom
            self.grid.snap_to_bottom();
            pty_write(&mut self.pty, &bytes);
        }

        false
    }

    pub(crate) fn handle_mouse_input(
        &mut self,
        button_state: ElementState,
        button: MouseButton,
    ) {
        // Selection tracking: left button starts/ends selection
        // when mouse reporting is NOT active
        if button == MouseButton::Left && self.grid.mouse_mode == MouseMode::None {
            if button_state == ElementState::Pressed {
                let anchor = SelectionAnchor::new(self.mouse_cell.0, self.mouse_cell.1 as i64);
                self.selection.begin(anchor);
                self.scheduler.request_redraw();
            } else {
                self.selection.finalize();
            }
        }

        if button_state == ElementState::Pressed {
            self.mouse_button = Some(button);
        } else {
            self.mouse_button = None;
        }

        if let Some(bytes) = input::encode_mouse_button(
            button,
            button_state,
            self.mouse_cell.0,
            self.mouse_cell.1,
            self.grid.mouse_mode,
            self.grid.mouse_encoding,
        ) {
            pty_write(&mut self.pty, &bytes);
        }
    }

    pub(crate) fn handle_cursor_moved(&mut self, position: winit::dpi::PhysicalPosition<f64>) {
        let col =
            ((position.x as f32 - self.padding) / self.compositor.cell_width()).max(0.0) as u16;
        let row =
            ((position.y as f32 - self.padding) / self.compositor.cell_height()).max(0.0) as u16;
        let col = col.min(self.grid.cols().saturating_sub(1));
        let row = row.min(self.grid.rows().saturating_sub(1));

        if (col, row) != self.mouse_cell {
            self.mouse_cell = (col, row);

            // Update selection if dragging
            if self.selection.active && self.mouse_button == Some(MouseButton::Left) {
                let anchor = SelectionAnchor::new(col, row as i64);
                self.selection.update(anchor);
                self.scheduler.request_redraw();
            }

            if let Some(bytes) = input::encode_mouse_motion(
                col,
                row,
                self.mouse_button,
                self.grid.mouse_mode,
                self.grid.mouse_encoding,
            ) {
                pty_write(&mut self.pty, &bytes);
            }
        }
    }

    pub(crate) fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta) {
        let up = match delta {
            MouseScrollDelta::LineDelta(_, y) => y > 0.0,
            MouseScrollDelta::PixelDelta(pos) => pos.y > 0.0,
        };

        // When mouse reporting is off, scroll the viewport
        if self.grid.mouse_mode == MouseMode::None {
            let scroll_lines = 3;
            if up {
                self.grid.scroll_viewport_up(scroll_lines);
            } else {
                self.grid.scroll_viewport_down(scroll_lines);
            }
            self.scheduler.request_redraw();
        } else if let Some(bytes) = input::encode_mouse_scroll(
            up,
            self.mouse_cell.0,
            self.mouse_cell.1,
            self.grid.mouse_mode,
            self.grid.mouse_encoding,
        ) {
            pty_write(&mut self.pty, &bytes);
        }
    }

    pub(crate) fn handle_scale_factor_changed(&mut self) {
        let size = self.window.inner_size();
        if size.width > 0 && size.height > 0 {
            self.compositor.resize(size.width, size.height);

            let term_size = TerminalSize::from_window(
                size.width,
                size.height,
                self.compositor.cell_width(),
                self.compositor.cell_height(),
                self.padding,
            );

            self.grid.resize(term_size.cols, term_size.rows);

            let _ = self.pty.resize(
                term_size.cols,
                term_size.rows,
                term_size.pixel_width,
                term_size.pixel_height,
            );

            self.scheduler.request_redraw();
        }
    }

    pub(crate) fn handle_focused(&mut self, focused: bool) {
        if self.grid.focus_reporting {
            let seq = if focused { b"\x1b[I" } else { b"\x1b[O" };
            pty_write(&mut self.pty, seq);
        }
    }
}
