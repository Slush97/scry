// SPDX-License-Identifier: MIT OR Apache-2.0
//! Auto-detection of the best available graphics protocol.
//!
//! The [`Picker`] queries the terminal environment to determine which graphics
//! protocol to use, and provides the font size needed for pixel ↔ cell
//! coordinate conversion.

use std::sync::OnceLock;

use crate::transport::backend::{FontSize, ProtocolKind};
use crate::transport::capabilities::{ProbeConfig, TerminalCapabilities};
use crate::transport::probe;

/// Globally cached terminal capabilities, computed once on first access.
static CACHED_CAPABILITIES: OnceLock<TerminalCapabilities> = OnceLock::new();

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
/// use scry_engine::transport::Picker;
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
    /// This runs the full detection pipeline: manual override via `SCRY_PROTOCOL`,
    /// active terminal probing (XTVERSION, Kitty graphics query, DA1 Sixel), and
    /// env-var fallback. Results are cached globally — subsequent calls are free.
    ///
    /// Set `SCRY_PROBE_TIMEOUT_MS` to adjust the per-query timeout (default: 150 ms).
    #[must_use]
    pub fn detect() -> Self {
        let caps = Self::capabilities();
        let font_size = Self::detect_font_size();
        Self {
            protocol: caps.protocol,
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

    /// Create a protocol backend matching the detected protocol.
    ///
    /// This is a convenience factory that eliminates the need for manual
    /// `match` on [`protocol()`](Self::protocol) when constructing backends.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use scry_engine::transport::Picker;
    ///
    /// let picker = Picker::detect();
    /// let backend = picker.create_backend();
    /// ```
    pub fn create_backend(&self) -> Box<dyn crate::transport::backend::ProtocolBackend> {
        match self.protocol {
            #[cfg(feature = "kitty")]
            ProtocolKind::Kitty => {
                let kb = crate::transport::kitty::KittyBackend::new(self.font_size);
                // When the `shm` feature is compiled in, use shared memory
                // as the default transport for Kitty — zero-copy, no base64,
                // no pipe I/O overhead.
                #[cfg(feature = "shm")]
                let kb = kb.format(crate::transport::kitty::TransmitFormat::SharedMemory);
                Box::new(kb)
            }
            #[cfg(feature = "iterm2")]
            ProtocolKind::Iterm2 => Box::new(crate::transport::iterm2::Iterm2Backend::new()),
            #[cfg(feature = "sixel")]
            ProtocolKind::Sixel => Box::new(crate::transport::sixel::SixelBackend::new()),
            // Window backend requires caller-managed event loop — cannot be
            // auto-created from Picker. Use `WindowBackend::new()` directly.
            _ => Box::new(crate::transport::halfblock::HalfblockBackend::new()),
        }
    }

    /// Auto-detect with an explicit probe configuration.
    ///
    /// Unlike [`detect()`](Self::detect), this does NOT use the global cache —
    /// it always runs the full pipeline with the given config.
    #[must_use]
    pub fn detect_with_config(config: &ProbeConfig) -> Self {
        let caps = probe::probe_capabilities(config);
        let font_size = Self::detect_font_size();
        Self {
            protocol: caps.protocol,
            font_size,
        }
    }

    /// Get the globally cached terminal capabilities.
    ///
    /// On first call, runs the full detection pipeline (including active
    /// terminal probing if stdout is a TTY). The result is cached in a
    /// `OnceLock` — subsequent calls return immediately.
    ///
    /// Set `SCRY_PROBE_TIMEOUT_MS` to adjust the per-query timeout
    /// (default: 150 ms).
    pub fn capabilities() -> &'static TerminalCapabilities {
        CACHED_CAPABILITIES.get_or_init(|| {
            let mut config = ProbeConfig::default();
            if let Ok(val) = std::env::var("SCRY_PROBE_TIMEOUT_MS") {
                if let Ok(ms) = val.parse::<u64>() {
                    config.timeout_ms = ms;
                }
            }
            probe::probe_capabilities(&config)
        })
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
