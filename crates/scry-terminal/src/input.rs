// SPDX-License-Identifier: MIT OR Apache-2.0
//! Input handling — translates winit events to terminal escape sequences.
//!
//! Converts keyboard events, mouse events, and paste operations into
//! the byte sequences expected by programs running in the terminal.

use winit::event::{ElementState, MouseButton};
use winit::keyboard::{Key, KeyCode, ModifiersState, NamedKey, PhysicalKey};

use crate::grid::{MouseEncoding, MouseMode};

// ── Keyboard encoding ──────────────────────────────────────────────

/// Encode a keyboard event into bytes to send to the PTY.
///
/// Returns `None` if the key event should not produce output.
pub fn encode_key(
    key: &Key,
    physical: PhysicalKey,
    state: ElementState,
    modifiers: ModifiersState,
    app_cursor: bool,
    _app_keypad: bool,
) -> Option<Vec<u8>> {
    // Only handle key press (not release)
    if state != ElementState::Pressed {
        return None;
    }

    // Check for Ctrl combinations first
    if modifiers.control_key() {
        if let Some(bytes) = encode_ctrl(key, &physical) {
            return Some(bytes);
        }
    }

    match key {
        Key::Named(named) => encode_named_key(named, modifiers, app_cursor),
        Key::Character(text) => {
            if modifiers.alt_key() {
                // Alt+char: send ESC prefix
                let mut bytes = vec![0x1b];
                bytes.extend_from_slice(text.as_bytes());
                Some(bytes)
            } else {
                Some(text.as_bytes().to_vec())
            }
        }
        Key::Unidentified(_) | Key::Dead(_) => None,
    }
}

/// Encode Ctrl+key combinations.
fn encode_ctrl(_key: &Key, physical: &PhysicalKey) -> Option<Vec<u8>> {
    // Try to get the character from the physical key for Ctrl combinations
    let code = match physical {
        PhysicalKey::Code(code) => code,
        _ => return None,
    };

    let ctrl_byte = match code {
        KeyCode::KeyA => Some(0x01),
        KeyCode::KeyB => Some(0x02),
        KeyCode::KeyC => Some(0x03), // SIGINT
        KeyCode::KeyD => Some(0x04), // EOF
        KeyCode::KeyE => Some(0x05),
        KeyCode::KeyF => Some(0x06),
        KeyCode::KeyG => Some(0x07), // BEL
        KeyCode::KeyH => Some(0x08), // Backspace
        KeyCode::KeyI => Some(0x09), // Tab
        KeyCode::KeyJ => Some(0x0A), // LF
        KeyCode::KeyK => Some(0x0B),
        KeyCode::KeyL => Some(0x0C), // Form feed (clear)
        KeyCode::KeyM => Some(0x0D), // CR
        KeyCode::KeyN => Some(0x0E),
        KeyCode::KeyO => Some(0x0F),
        KeyCode::KeyP => Some(0x10),
        KeyCode::KeyQ => Some(0x11), // XON
        KeyCode::KeyR => Some(0x12),
        KeyCode::KeyS => Some(0x13), // XOFF
        KeyCode::KeyT => Some(0x14),
        KeyCode::KeyU => Some(0x15),
        KeyCode::KeyV => Some(0x16),
        KeyCode::KeyW => Some(0x17),
        KeyCode::KeyX => Some(0x18),
        KeyCode::KeyY => Some(0x19),
        KeyCode::KeyZ => Some(0x1A), // SIGTSTP
        KeyCode::BracketLeft => Some(0x1B),  // ESC
        KeyCode::Backslash => Some(0x1C),
        KeyCode::BracketRight => Some(0x1D),
        KeyCode::Digit6 => Some(0x1E),
        KeyCode::Minus => Some(0x1F),
        _ => None,
    };

    ctrl_byte.map(|b| vec![b])
}

/// Encode a named (special) key.
fn encode_named_key(
    key: &NamedKey,
    modifiers: ModifiersState,
    app_cursor: bool,
) -> Option<Vec<u8>> {
    // Modifier encoding for CSI sequences
    let modifier_suffix = modifier_param(modifiers);

    match key {
        NamedKey::Enter => Some(vec![0x0D]),
        NamedKey::Tab => {
            if modifiers.shift_key() {
                Some(b"\x1b[Z".to_vec()) // Backtab
            } else {
                Some(vec![0x09])
            }
        }
        NamedKey::Backspace => Some(vec![0x7F]),
        NamedKey::Escape => Some(vec![0x1B]),
        NamedKey::Space => Some(vec![0x20]),

        // Arrow keys
        NamedKey::ArrowUp => Some(arrow_key(b'A', modifiers, app_cursor, &modifier_suffix)),
        NamedKey::ArrowDown => Some(arrow_key(b'B', modifiers, app_cursor, &modifier_suffix)),
        NamedKey::ArrowRight => Some(arrow_key(b'C', modifiers, app_cursor, &modifier_suffix)),
        NamedKey::ArrowLeft => Some(arrow_key(b'D', modifiers, app_cursor, &modifier_suffix)),

        // Navigation
        NamedKey::Home => Some(csi_tilde_key(1, &modifier_suffix)),
        NamedKey::Insert => Some(csi_tilde_key(2, &modifier_suffix)),
        NamedKey::Delete => Some(csi_tilde_key(3, &modifier_suffix)),
        NamedKey::End => Some(csi_tilde_key(4, &modifier_suffix)),
        NamedKey::PageUp => Some(csi_tilde_key(5, &modifier_suffix)),
        NamedKey::PageDown => Some(csi_tilde_key(6, &modifier_suffix)),

        // Function keys
        NamedKey::F1 => Some(csi_tilde_key(11, &modifier_suffix)),
        NamedKey::F2 => Some(csi_tilde_key(12, &modifier_suffix)),
        NamedKey::F3 => Some(csi_tilde_key(13, &modifier_suffix)),
        NamedKey::F4 => Some(csi_tilde_key(14, &modifier_suffix)),
        NamedKey::F5 => Some(csi_tilde_key(15, &modifier_suffix)),
        NamedKey::F6 => Some(csi_tilde_key(17, &modifier_suffix)),
        NamedKey::F7 => Some(csi_tilde_key(18, &modifier_suffix)),
        NamedKey::F8 => Some(csi_tilde_key(19, &modifier_suffix)),
        NamedKey::F9 => Some(csi_tilde_key(20, &modifier_suffix)),
        NamedKey::F10 => Some(csi_tilde_key(21, &modifier_suffix)),
        NamedKey::F11 => Some(csi_tilde_key(23, &modifier_suffix)),
        NamedKey::F12 => Some(csi_tilde_key(24, &modifier_suffix)),

        _ => None,
    }
}

/// Encode an arrow key (may use application mode).
fn arrow_key(code: u8, _modifiers: ModifiersState, app_cursor: bool, modifier_suffix: &str) -> Vec<u8> {
    if !modifier_suffix.is_empty() {
        // With modifiers: always CSI format
        format!("\x1b[1;{modifier_suffix}{}", code as char).into_bytes()
    } else if app_cursor {
        // Application cursor mode: ESC O A
        vec![0x1b, b'O', code]
    } else {
        // Normal mode: ESC [ A
        vec![0x1b, b'[', code]
    }
}

/// Encode a CSI ~ key (Home, Insert, Delete, End, PgUp, PgDn, Fn keys).
fn csi_tilde_key(number: u8, modifier_suffix: &str) -> Vec<u8> {
    if modifier_suffix.is_empty() {
        format!("\x1b[{number}~").into_bytes()
    } else {
        format!("\x1b[{number};{modifier_suffix}~").into_bytes()
    }
}

/// Compute the modifier parameter for CSI sequences.
///
/// Returns "" for no modifiers, or the parameter string (e.g., "2" for Shift).
fn modifier_param(modifiers: ModifiersState) -> String {
    let mut code: u8 = 1; // Base
    if modifiers.shift_key() {
        code += 1;
    }
    if modifiers.alt_key() {
        code += 2;
    }
    if modifiers.control_key() {
        code += 4;
    }
    if code == 1 {
        String::new()
    } else {
        code.to_string()
    }
}

// ── Mouse encoding ─────────────────────────────────────────────────

/// Encode a mouse button event for sending to the PTY.
///
/// Returns `None` if mouse reporting is disabled or not applicable.
pub fn encode_mouse_button(
    button: MouseButton,
    state: ElementState,
    col: u16,
    row: u16,
    mode: MouseMode,
    encoding: MouseEncoding,
) -> Option<Vec<u8>> {
    if mode == MouseMode::None {
        return None;
    }

    let button_code = match button {
        MouseButton::Left => 0,
        MouseButton::Middle => 1,
        MouseButton::Right => 2,
        _ => return None,
    };

    match encoding {
        MouseEncoding::Sgr => {
            let suffix = if state == ElementState::Pressed {
                'M'
            } else {
                'm'
            };
            Some(format!("\x1b[<{button_code};{};{}{suffix}", col + 1, row + 1).into_bytes())
        }
        MouseEncoding::X10 => {
            if state == ElementState::Released {
                // X10 release: button 3
                let b = b' ' + 3;
                let x = b' ' + (col as u8).min(222) + 1;
                let y = b' ' + (row as u8).min(222) + 1;
                Some(vec![0x1b, b'[', b'M', b, x, y])
            } else {
                let b = b' ' + button_code;
                let x = b' ' + (col as u8).min(222) + 1;
                let y = b' ' + (row as u8).min(222) + 1;
                Some(vec![0x1b, b'[', b'M', b, x, y])
            }
        }
    }
}

/// Encode a mouse motion event.
pub fn encode_mouse_motion(
    col: u16,
    row: u16,
    button_held: Option<MouseButton>,
    mode: MouseMode,
    encoding: MouseEncoding,
) -> Option<Vec<u8>> {
    match mode {
        MouseMode::None | MouseMode::Press => return None,
        MouseMode::ButtonMotion => {
            if button_held.is_none() {
                return None; // Only report motion while button is held
            }
        }
        MouseMode::AllMotion => {} // Report all motion
    }

    let button_code = match button_held {
        Some(MouseButton::Left) => 32,    // 0 + 32 (motion flag)
        Some(MouseButton::Middle) => 33,
        Some(MouseButton::Right) => 34,
        None => 35,                        // No button
        _ => return None,
    };

    match encoding {
        MouseEncoding::Sgr => {
            Some(format!("\x1b[<{button_code};{};{}M", col + 1, row + 1).into_bytes())
        }
        MouseEncoding::X10 => {
            let b = b' ' + button_code;
            let x = b' ' + (col as u8).min(222) + 1;
            let y = b' ' + (row as u8).min(222) + 1;
            Some(vec![0x1b, b'[', b'M', b, x, y])
        }
    }
}

/// Encode a mouse scroll event.
pub fn encode_mouse_scroll(
    up: bool,
    col: u16,
    row: u16,
    mode: MouseMode,
    encoding: MouseEncoding,
) -> Option<Vec<u8>> {
    if mode == MouseMode::None {
        return None;
    }

    let button_code = if up { 64 } else { 65 };

    match encoding {
        MouseEncoding::Sgr => {
            Some(format!("\x1b[<{button_code};{};{}M", col + 1, row + 1).into_bytes())
        }
        MouseEncoding::X10 => {
            let b = b' ' + button_code;
            let x = b' ' + (col as u8).min(222) + 1;
            let y = b' ' + (row as u8).min(222) + 1;
            Some(vec![0x1b, b'[', b'M', b, x, y])
        }
    }
}

// ── Paste handling ─────────────────────────────────────────────────

/// Encode pasted text, wrapping in bracketed paste markers if enabled.
pub fn encode_paste(text: &str, bracketed_paste: bool) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(text.len() + 12);
    if bracketed_paste {
        bytes.extend_from_slice(b"\x1b[200~");
    }
    bytes.extend_from_slice(text.as_bytes());
    if bracketed_paste {
        bytes.extend_from_slice(b"\x1b[201~");
    }
    bytes
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctrl_c() {
        let result = encode_ctrl(
            &Key::Character("c".into()),
            &PhysicalKey::Code(KeyCode::KeyC),
        );
        assert_eq!(result, Some(vec![0x03]));
    }

    #[test]
    fn ctrl_d() {
        let result = encode_ctrl(
            &Key::Character("d".into()),
            &PhysicalKey::Code(KeyCode::KeyD),
        );
        assert_eq!(result, Some(vec![0x04]));
    }

    #[test]
    fn ctrl_z() {
        let result = encode_ctrl(
            &Key::Character("z".into()),
            &PhysicalKey::Code(KeyCode::KeyZ),
        );
        assert_eq!(result, Some(vec![0x1A]));
    }

    #[test]
    fn arrow_keys_normal() {
        let no_mods = ModifiersState::empty();
        let result = encode_named_key(&NamedKey::ArrowUp, no_mods, false);
        assert_eq!(result, Some(b"\x1b[A".to_vec()));
    }

    #[test]
    fn arrow_keys_app_mode() {
        let no_mods = ModifiersState::empty();
        let result = encode_named_key(&NamedKey::ArrowUp, no_mods, true);
        assert_eq!(result, Some(b"\x1bOA".to_vec()));
    }

    #[test]
    fn function_keys() {
        let no_mods = ModifiersState::empty();
        let result = encode_named_key(&NamedKey::F1, no_mods, false);
        assert_eq!(result, Some(b"\x1b[11~".to_vec()));
    }

    #[test]
    fn paste_bracketed() {
        let result = encode_paste("hello", true);
        assert_eq!(result, b"\x1b[200~hello\x1b[201~");
    }

    #[test]
    fn paste_unbracketed() {
        let result = encode_paste("hello", false);
        assert_eq!(result, b"hello");
    }

    #[test]
    fn mouse_sgr_click() {
        let result = encode_mouse_button(
            MouseButton::Left,
            ElementState::Pressed,
            5,
            10,
            MouseMode::Press,
            MouseEncoding::Sgr,
        );
        assert_eq!(result, Some(b"\x1b[<0;6;11M".to_vec()));
    }

    #[test]
    fn mouse_sgr_release() {
        let result = encode_mouse_button(
            MouseButton::Left,
            ElementState::Released,
            5,
            10,
            MouseMode::Press,
            MouseEncoding::Sgr,
        );
        assert_eq!(result, Some(b"\x1b[<0;6;11m".to_vec()));
    }

    #[test]
    fn mouse_disabled() {
        let result = encode_mouse_button(
            MouseButton::Left,
            ElementState::Pressed,
            0,
            0,
            MouseMode::None,
            MouseEncoding::Sgr,
        );
        assert_eq!(result, None);
    }

    #[test]
    fn scroll_encoding() {
        let result = encode_mouse_scroll(true, 10, 5, MouseMode::Press, MouseEncoding::Sgr);
        assert_eq!(result, Some(b"\x1b[<64;11;6M".to_vec()));
    }
}
