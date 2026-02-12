//! Auto-detection of the best available graphics protocol.
//!
//! The [`Picker`] queries the terminal environment to determine which graphics
//! protocol to use, and provides the font size needed for pixel ↔ cell
//! coordinate conversion.

use crate::transport::backend::{FontSize, ProtocolKind};

// ---------------------------------------------------------------------------
// Picker
// ---------------------------------------------------------------------------

/// Detects the best available graphics protocol and terminal font size.
///
/// # Detection Strategy
///
/// 1. Check `TERM` / `TERM_PROGRAM` for known Kitty-compatible terminals
/// 2. Check for Sixel support via environment or DA2 response (future)
/// 3. Fall back to halfblock rendering
///
/// # Example
///
/// ```no_run
/// use ratatui_pixelcanvas::transport::Picker;
///
/// let picker = Picker::detect();
/// println!("Protocol: {:?}", picker.protocol());
/// println!("Font size: {:?}", picker.font_size());
/// ```
#[derive(Clone, Debug)]
pub struct Picker {
    protocol: ProtocolKind,
    font_size: FontSize,
}

impl Picker {
    /// Auto-detect the best available protocol and font size.
    ///
    /// This queries environment variables and terminal capabilities.
    #[must_use]
    pub fn detect() -> Self {
        let protocol = Self::detect_protocol();
        let font_size = Self::detect_font_size();
        Self {
            protocol,
            font_size,
        }
    }

    /// Create a picker with explicit values (useful for testing).
    #[must_use]
    pub const fn new(protocol: ProtocolKind, font_size: FontSize) -> Self {
        Self {
            protocol,
            font_size,
        }
    }

    /// The detected (or configured) graphics protocol.
    #[must_use]
    pub const fn protocol(&self) -> ProtocolKind {
        self.protocol
    }

    /// The detected (or configured) font size.
    #[must_use]
    pub const fn font_size(&self) -> FontSize {
        self.font_size
    }

    /// Detect which protocol the terminal supports.
    fn detect_protocol() -> ProtocolKind {
        // Check for Kitty
        if Self::is_kitty_compatible() {
            return ProtocolKind::Kitty;
        }

        // Future: check for Sixel via DA2 or TERM capabilities

        // Default fallback
        ProtocolKind::Halfblock
    }

    /// Check if the terminal is Kitty-compatible.
    fn is_kitty_compatible() -> bool {
        // TERM_PROGRAM is set by Kitty and some other terminals
        if let Ok(prog) = std::env::var("TERM_PROGRAM") {
            let prog_lower = prog.to_lowercase();
            if prog_lower.contains("kitty")
                || prog_lower.contains("wezterm")
                || prog_lower.contains("ghostty")
            {
                return true;
            }
        }

        // TERM=xterm-kitty is set by Kitty
        if let Ok(term) = std::env::var("TERM") {
            if term.contains("kitty") {
                return true;
            }
        }

        // KITTY_WINDOW_ID is set inside Kitty
        if std::env::var("KITTY_WINDOW_ID").is_ok() {
            return true;
        }

        false
    }

    /// Detect font size using the `TIOCGWINSZ` ioctl.
    fn detect_font_size() -> FontSize {
        #[cfg(unix)]
        {
            Self::detect_font_size_unix().unwrap_or_default()
        }

        #[cfg(not(unix))]
        {
            FontSize::default()
        }
    }

    /// Unix-specific font size detection via ioctl.
    #[cfg(unix)]
    fn detect_font_size_unix() -> Option<FontSize> {
        use std::mem::MaybeUninit;

        // TIOCGWINSZ returns: rows, cols, xpixel, ypixel
        #[repr(C)]
        #[allow(clippy::struct_field_names)]
        struct Winsize {
            ws_row: u16,
            ws_col: u16,
            ws_xpixel: u16,
            ws_ypixel: u16,
        }

        let mut ws = MaybeUninit::<Winsize>::uninit();

        // SAFETY: TIOCGWINSZ is a well-defined ioctl that reads terminal
        // window size. It writes to the provided buffer and does not have
        // side effects.
        #[allow(unsafe_code)]
        let result = unsafe { libc_ioctl(1, TIOCGWINSZ_VAL, ws.as_mut_ptr()) };

        if result != 0 {
            return None;
        }

        // SAFETY: ioctl succeeded, so the struct is fully initialized.
        #[allow(unsafe_code)]
        let ws = unsafe { ws.assume_init() };

        if ws.ws_col == 0 || ws.ws_row == 0 || ws.ws_xpixel == 0 || ws.ws_ypixel == 0 {
            return None;
        }

        Some(FontSize::new(
            ws.ws_xpixel / ws.ws_col,
            ws.ws_ypixel / ws.ws_row,
        ))
    }
}

// Platform-specific ioctl bindings.
#[cfg(all(unix, target_os = "linux"))]
const TIOCGWINSZ_VAL: std::ffi::c_ulong = 0x5413;

#[cfg(all(unix, target_os = "macos"))]
const TIOCGWINSZ_VAL: std::ffi::c_ulong = 0x4008_7468;

#[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
const TIOCGWINSZ_VAL: std::ffi::c_ulong = 0x5413; // Best-effort: Linux value

#[cfg(unix)]
extern "C" {
    #[link_name = "ioctl"]
    fn libc_ioctl(fd: std::ffi::c_int, request: std::ffi::c_ulong, ...) -> std::ffi::c_int;
}

