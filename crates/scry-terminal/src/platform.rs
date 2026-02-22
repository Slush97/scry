// SPDX-License-Identifier: MIT OR Apache-2.0
//! Platform-specific abstractions.
//!
//! Handles differences between Unix and Windows for signal handling,
//! default shell detection, and environment setup.

// ── Default shell ──────────────────────────────────────────────────

/// Detect the user's default shell.
///
/// On Unix, reads `$SHELL`. On Windows, returns `cmd.exe`.
/// Falls back to `/bin/sh` (Unix) or `cmd.exe` (Windows) if detection fails.
pub fn default_shell() -> String {
    #[cfg(unix)]
    {
        std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string())
    }
    #[cfg(windows)]
    {
        std::env::var("COMSPEC").unwrap_or_else(|_| "cmd.exe".to_string())
    }
}

// ── TERM environment ───────────────────────────────────────────────

/// The `TERM` value to set for child processes.
///
/// Phase 1: identify as `xterm-256color` for maximum compatibility.
/// Phase 2+: ship a custom `scry-terminal` terminfo entry.
pub const TERM_VALUE: &str = "xterm-256color";

/// The `COLORTERM` value for true-color support detection.
pub const COLORTERM_VALUE: &str = "truecolor";

/// Set up environment variables for the child shell.
///
/// Called before spawning the PTY child process.
pub fn setup_child_env(cmd: &mut portable_pty::CommandBuilder) {
    cmd.env("TERM", TERM_VALUE);
    cmd.env("COLORTERM", COLORTERM_VALUE);
    // Clear TERM_PROGRAM so shells don't think they're inside another terminal
    cmd.env("TERM_PROGRAM", "scry-terminal");
    cmd.env("TERM_PROGRAM_VERSION", env!("CARGO_PKG_VERSION"));
}

// ── Signal handling (Unix) ─────────────────────────────────────────

/// Install platform-specific signal handlers.
///
/// On Unix, we do NOT install signal handlers for SIGWINCH or SIGCHLD
/// because `winit` handles window resize events and `portable-pty`
/// handles child process exit detection. This function exists as a
/// hook for any future platform-specific setup.
pub fn install_signal_handlers() {
    #[cfg(unix)]
    {
        // SIGPIPE should be ignored (same as most terminal emulators).
        // This prevents crashes when writing to a closed PTY pipe.
        // SAFETY: SIG_IGN is a valid signal handler constant.
        #[allow(unsafe_code)]
        unsafe {
            libc::signal(libc::SIGPIPE, libc::SIG_IGN);
        }
    }
}

// Note on SIGPIPE: We allow this single unsafe block because ignoring
// SIGPIPE is a standard Unix daemon/terminal practice. The `libc` crate
// is already a transitive dependency via `portable-pty`. If we want to
// eliminate this, we can use the `signal-hook` crate in the future.

// ── Terminal size ──────────────────────────────────────────────────

/// Terminal dimensions in both cells and pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TerminalSize {
    /// Number of character columns.
    pub cols: u16,
    /// Number of character rows.
    pub rows: u16,
    /// Window width in pixels.
    pub pixel_width: u16,
    /// Window height in pixels.
    pub pixel_height: u16,
}

impl TerminalSize {
    /// Compute cell dimensions from window size, cell metrics, and padding.
    pub fn from_window(
        pixel_width: u32,
        pixel_height: u32,
        cell_width: f32,
        cell_height: f32,
        padding: f32,
    ) -> Self {
        let usable_w = (pixel_width as f32 - 2.0 * padding).max(cell_width);
        let usable_h = (pixel_height as f32 - 2.0 * padding).max(cell_height);
        let cols = (usable_w / cell_width).floor().max(1.0) as u16;
        let rows = (usable_h / cell_height).floor().max(1.0) as u16;
        Self {
            cols,
            rows,
            pixel_width: pixel_width.min(u16::MAX as u32) as u16,
            pixel_height: pixel_height.min(u16::MAX as u32) as u16,
        }
    }
}

impl From<TerminalSize> for portable_pty::PtySize {
    fn from(ts: TerminalSize) -> Self {
        Self {
            rows: ts.rows,
            cols: ts.cols,
            pixel_width: ts.pixel_width,
            pixel_height: ts.pixel_height,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_shell_returns_something() {
        let shell = default_shell();
        assert!(!shell.is_empty());
    }

    #[test]
    fn terminal_size_from_window() {
        let size = TerminalSize::from_window(800, 600, 8.0, 16.0, 0.0);
        assert_eq!(size.cols, 100);
        assert_eq!(size.rows, 37);
        assert_eq!(size.pixel_width, 800);
        assert_eq!(size.pixel_height, 600);
    }

    #[test]
    fn terminal_size_with_padding() {
        // 800 - 2*6 = 788 usable, 788/8 = 98.5 → 98 cols
        // 600 - 2*6 = 588 usable, 588/16 = 36.75 → 36 rows
        let size = TerminalSize::from_window(800, 600, 8.0, 16.0, 6.0);
        assert_eq!(size.cols, 98);
        assert_eq!(size.rows, 36);
    }

    #[test]
    fn terminal_size_minimum_1x1() {
        let size = TerminalSize::from_window(1, 1, 100.0, 100.0, 0.0);
        assert_eq!(size.cols, 1);
        assert_eq!(size.rows, 1);
    }

    #[test]
    fn terminal_size_to_pty_size() {
        let ts = TerminalSize {
            cols: 80,
            rows: 24,
            pixel_width: 640,
            pixel_height: 384,
        };
        let ps: portable_pty::PtySize = ts.into();
        assert_eq!(ps.rows, 24);
        assert_eq!(ps.cols, 80);
    }
}
