// SPDX-License-Identifier: MIT OR Apache-2.0
//! Active terminal probing via escape sequences.
//!
//! This module sends escape sequences to the terminal and parses the responses
//! to determine graphics capabilities. All I/O is confined to this module.
//!
//! The response parsers are pure functions and can be tested without a TTY.

use crate::transport::backend::ProtocolKind;
use crate::transport::capabilities::{
    DetectionMethod, Multiplexer, ProbeConfig, TerminalCapabilities,
};

// ---------------------------------------------------------------------------
// Multiplexer detection
// ---------------------------------------------------------------------------

/// Detect whether a terminal multiplexer is wrapping the session.
#[must_use]
pub fn detect_multiplexer() -> Multiplexer {
    if std::env::var("TMUX").is_ok() {
        Multiplexer::Tmux
    } else if std::env::var("STY").is_ok() {
        Multiplexer::Screen
    } else if std::env::var("ZELLIJ").is_ok() {
        Multiplexer::Zellij
    } else {
        Multiplexer::None
    }
}

// ---------------------------------------------------------------------------
// Manual override via SCRY_PROTOCOL
// ---------------------------------------------------------------------------

/// Check for a manual protocol override via `SCRY_PROTOCOL` env var.
///
/// Accepted values: `kitty`, `sixel`, `iterm2`, `halfblock`.
#[must_use]
pub fn check_manual_override() -> Option<ProtocolKind> {
    let val = std::env::var("SCRY_PROTOCOL").ok()?;
    match val.to_lowercase().as_str() {
        "kitty" => Some(ProtocolKind::Kitty),
        "sixel" => Some(ProtocolKind::Sixel),
        "iterm2" => Some(ProtocolKind::Iterm2),
        "halfblock" => Some(ProtocolKind::Halfblock),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Response parsers (pure functions, testable without TTY)
// ---------------------------------------------------------------------------

/// Parse an XTVERSION response.
///
/// Expected format: `\x1bP>|<terminal_name> <version>\x1b\\`
/// Returns `(terminal_name, version)` if parseable.
#[must_use]
pub fn parse_xtversion(response: &[u8]) -> Option<(String, String)> {
    // Look for DCS >| ... ST
    let response_str = std::str::from_utf8(response).ok()?;

    // Find the content between ">|" and either ESC\ or BEL
    let start = response_str.find(">|")?;
    let content = &response_str[start + 2..];

    let end = content
        .find("\x1b\\")
        .or_else(|| content.find('\x07'))
        .unwrap_or(content.len());

    let info = content[..end].trim();
    if info.is_empty() {
        return None;
    }

    // Split on first space: "kitty 0.35.1" → ("kitty", "0.35.1")
    if let Some(space) = info.find(' ') {
        Some((info[..space].to_string(), info[space + 1..].to_string()))
    } else {
        Some((info.to_string(), String::new()))
    }
}

/// Parse a DA1 (Primary Device Attributes) response.
///
/// Expected format: `\x1b[?<attrs>c` where attrs are semicolon-separated.
/// Returns the list of attribute numbers.
#[must_use]
pub fn parse_da1(response: &[u8]) -> Vec<u32> {
    let response_str = match std::str::from_utf8(response) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };

    // Find CSI ? ... c pattern
    let start = match response_str.find("\x1b[?") {
        Some(i) => i + 3,
        None => return Vec::new(),
    };

    let end = match response_str[start..].find('c') {
        Some(i) => start + i,
        None => return Vec::new(),
    };

    response_str[start..end]
        .split(';')
        .filter_map(|s| s.trim().parse::<u32>().ok())
        .collect()
}

/// Check whether DA1 attributes indicate Sixel support (attribute 4).
#[must_use]
pub fn da1_has_sixel(attrs: &[u32]) -> bool {
    attrs.contains(&4)
}

/// Parse a DA2 (Secondary Device Attributes) response.
///
/// Expected format: `\x1b[><type>;<version>;<something>c`
/// Returns `(terminal_type, firmware_version)`.
#[must_use]
pub fn parse_da2(response: &[u8]) -> Option<(u32, u32)> {
    let response_str = std::str::from_utf8(response).ok()?;

    let start = response_str.find("\x1b[>")?;
    let content = &response_str[start + 3..];
    let end = content.find('c')?;

    let parts: Vec<&str> = content[..end].split(';').collect();
    if parts.len() >= 2 {
        let term_type = parts[0].parse::<u32>().ok()?;
        let version = parts[1].parse::<u32>().ok()?;
        Some((term_type, version))
    } else {
        None
    }
}

/// Parse a Kitty graphics query response.
///
/// Expected: `\x1b_Gi=31;OK\x1b\\` for success.
#[must_use]
pub fn parse_kitty_graphics_response(response: &[u8]) -> bool {
    let response_str = match std::str::from_utf8(response) {
        Ok(s) => s,
        Err(_) => return false,
    };
    response_str.contains("OK")
}

/// Infer protocol from XTVERSION terminal name.
#[must_use]
pub fn protocol_from_terminal_name(name: &str) -> Option<ProtocolKind> {
    let lower = name.to_lowercase();
    if lower.contains("kitty") || lower.contains("ghostty") || lower.contains("wezterm") {
        Some(ProtocolKind::Kitty)
    } else if lower.contains("iterm2") || lower.contains("mintty") {
        Some(ProtocolKind::Iterm2)
    } else if lower.contains("foot") || lower.contains("mlterm") || lower.contains("contour") {
        Some(ProtocolKind::Sixel)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Active probing (requires TTY)
// ---------------------------------------------------------------------------

/// Query the terminal with an escape sequence and read the response.
///
/// Puts the terminal into raw mode, writes the query, polls for a response
/// with a timeout, then restores the terminal.
///
/// Returns the raw response bytes, or an empty vec on timeout/error.
#[cfg(unix)]
pub fn query_terminal(query: &[u8], timeout_ms: u64) -> Vec<u8> {
    use std::io::{Read, Write};
    use std::os::unix::io::AsRawFd;

    let stdin = std::io::stdin();
    let fd = stdin.as_raw_fd();

    // Save terminal state and switch to raw mode
    let old_termios = match get_termios(fd) {
        Some(t) => t,
        None => return Vec::new(),
    };

    let mut raw = old_termios;
    // cfmakeraw equivalent — using raw numeric constants from POSIX
    // These are standard across Linux/macOS/BSD.
    raw.c_iflag &= !(0o000_001   // IGNBRK
        | 0o000_002               // BRKINT
        | 0o000_010               // PARMRK (Linux)
        | 0o000_040               // ISTRIP
        | 0o000_100               // INLCR
        | 0o000_200               // IGNCR
        | 0o000_400               // ICRNL
        | 0o002_000); // IXON
    raw.c_oflag &= !(0o000_001); // OPOST
    raw.c_lflag &= !(0o000_010   // ECHO
        | 0o000_100               // ECHONL
        | 0o000_002               // ICANON
        | 0o000_001               // ISIG
        | 0o100_000); // IEXTEN (Linux; macOS may differ but safe to clear)
    raw.c_cflag &= !(0o000_060   // CSIZE
        | 0o000_400); // PARENB
    raw.c_cflag |= 0o000_060; // CS8
                              // VMIN=0, VTIME=0: non-blocking reads
    raw.c_cc[VMIN_IDX] = 0;
    raw.c_cc[VTIME_IDX] = 0;

    // SAFETY: tcsetattr is a well-defined POSIX call.
    #[allow(unsafe_code)]
    if unsafe {
        tcsetattr(fd, 0 /* TCSANOW */, &raw)
    } != 0
    {
        return Vec::new();
    }

    // RAII guard to restore terminal
    struct TermRestore {
        fd: std::ffi::c_int,
        termios: Termios,
    }
    impl Drop for TermRestore {
        fn drop(&mut self) {
            #[allow(unsafe_code)]
            unsafe {
                tcsetattr(self.fd, 0, &self.termios);
            }
        }
    }
    let _guard = TermRestore {
        fd,
        termios: old_termios,
    };

    // Write query
    let mut stdout = std::io::stdout().lock();
    if stdout.write_all(query).is_err() || stdout.flush().is_err() {
        return Vec::new();
    }

    // Poll for response
    let mut response = vec![0u8; 256];
    let mut total = 0;

    let mut pfd = PollFd {
        fd,
        events: 0x001, // POLLIN
        revents: 0,
    };

    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);

    loop {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        if remaining.is_zero() {
            break;
        }

        // SAFETY: poll is a well-defined POSIX call.
        #[allow(unsafe_code)]
        let ready = unsafe { poll(&mut pfd, 1, remaining.as_millis() as std::ffi::c_int) };

        if ready <= 0 {
            break;
        }

        let mut stdin_lock = stdin.lock();
        match stdin_lock.read(&mut response[total..]) {
            Ok(0) => break,
            Ok(n) => {
                total += n;
                // Check if we got a complete response (ends with ESC\ or c or BEL)
                if total > 0 {
                    let last = response[total - 1];
                    if last == b'\\' || last == b'c' || last == b'\x07' {
                        break;
                    }
                }
                if total >= response.len() {
                    break;
                }
            }
            Err(_) => break,
        }
    }

    response.truncate(total);
    response
}

// ---------------------------------------------------------------------------
// Raw FFI bindings (avoids libc dependency)
// ---------------------------------------------------------------------------

/// Minimal termios struct matching the kernel ABI.
#[cfg(unix)]
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(clippy::struct_field_names)]
struct Termios {
    c_iflag: u32,
    c_oflag: u32,
    c_cflag: u32,
    c_lflag: u32,
    // Linux: c_line (u8) + c_cc[32]; macOS: c_cc[20] with different layout.
    // We use an opaque tail large enough for both.
    c_line: u8,
    c_cc: [u8; 32],
    // Padding to cover macOS's larger struct (cfr. __darwin_termios)
    _pad: [u8; 64],
}

/// VMIN index in c_cc (Linux).
#[cfg(all(unix, target_os = "linux"))]
const VMIN_IDX: usize = 6;
/// VTIME index in c_cc (Linux).
#[cfg(all(unix, target_os = "linux"))]
const VTIME_IDX: usize = 5;

/// VMIN index in c_cc (macOS).
#[cfg(all(unix, target_os = "macos"))]
const VMIN_IDX: usize = 16;
/// VTIME index in c_cc (macOS).
#[cfg(all(unix, target_os = "macos"))]
const VTIME_IDX: usize = 17;

/// Fallback for other Unix platforms.
#[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
const VMIN_IDX: usize = 6;
#[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
const VTIME_IDX: usize = 5;

/// Minimal pollfd struct.
#[cfg(unix)]
#[repr(C)]
struct PollFd {
    fd: std::ffi::c_int,
    events: std::ffi::c_short,
    revents: std::ffi::c_short,
}

#[cfg(unix)]
extern "C" {
    fn tcgetattr(fd: std::ffi::c_int, termios: *mut Termios) -> std::ffi::c_int;
    fn tcsetattr(
        fd: std::ffi::c_int,
        action: std::ffi::c_int,
        termios: *const Termios,
    ) -> std::ffi::c_int;
    fn poll(fds: *mut PollFd, nfds: u64, timeout: std::ffi::c_int) -> std::ffi::c_int;
    fn isatty(fd: std::ffi::c_int) -> std::ffi::c_int;
}

/// Get terminal attributes.
#[cfg(unix)]
fn get_termios(fd: std::ffi::c_int) -> Option<Termios> {
    // SAFETY: tcgetattr is a well-defined POSIX call that initializes the struct.
    #[allow(unsafe_code)]
    let mut termios = unsafe { std::mem::zeroed::<Termios>() };
    #[allow(unsafe_code)]
    if unsafe { tcgetattr(fd, &mut termios) } == 0 {
        Some(termios)
    } else {
        None
    }
}

/// Stub for non-Unix platforms.
#[cfg(not(unix))]
pub fn query_terminal(_query: &[u8], _timeout_ms: u64) -> Vec<u8> {
    Vec::new()
}

// ---------------------------------------------------------------------------
// Probe pipeline
// ---------------------------------------------------------------------------

/// Run the full detection pipeline and return terminal capabilities.
///
/// This is the main entry point for protocol detection. It:
/// 1. Checks `SCRY_PROTOCOL` env var override
/// 2. Detects multiplexer
/// 3. Checks if stdout is a TTY
/// 4. If TTY and active probing enabled: sends escape sequences
/// 5. Falls back to env-var heuristics
pub fn probe_capabilities(config: &ProbeConfig) -> TerminalCapabilities {
    let mut caps = TerminalCapabilities::default();

    // 1. Manual override
    if let Some(protocol) = check_manual_override() {
        caps.protocol = protocol;
        caps.detection_method = DetectionMethod::ManualOverride;
        return caps;
    }

    // 2. Multiplexer detection
    caps.multiplexer = detect_multiplexer();

    // 3. Check if stdout is a TTY
    #[cfg(unix)]
    {
        // SAFETY: isatty is a well-defined POSIX call.
        #[allow(unsafe_code)]
        let is_tty = unsafe { isatty(1) } == 1;
        caps.is_tty = is_tty;
    }
    #[cfg(not(unix))]
    {
        caps.is_tty = false;
    }

    // 4. Active probing (TTY only)
    if caps.is_tty && config.active_probe {
        // XTVERSION query
        let response = query_terminal(b"\x1b[>0q", config.timeout_ms);
        if let Some((name, version)) = parse_xtversion(&response) {
            caps.terminal_name = Some(name.clone());
            caps.terminal_version = Some(version);

            if let Some(protocol) = protocol_from_terminal_name(&name) {
                caps.protocol = protocol;
                caps.detection_method = DetectionMethod::ActiveProbe;

                // If Kitty, verify with graphics query
                if protocol == ProtocolKind::Kitty {
                    let gfx_response = query_terminal(
                        b"\x1b_Gi=31,s=1,v=1,a=q,t=d,f=24;AAAA\x1b\\",
                        config.timeout_ms,
                    );
                    caps.kitty.graphics_supported = parse_kitty_graphics_response(&gfx_response);
                }

                return caps;
            }
        }

        // Kitty graphics query (even if XTVERSION didn't identify)
        let gfx_response = query_terminal(
            b"\x1b_Gi=31,s=1,v=1,a=q,t=d,f=24;AAAA\x1b\\",
            config.timeout_ms,
        );
        if parse_kitty_graphics_response(&gfx_response) {
            caps.kitty.graphics_supported = true;
            caps.protocol = ProtocolKind::Kitty;
            caps.detection_method = DetectionMethod::ActiveProbe;
            return caps;
        }

        // DA1 query for Sixel
        let da1_response = query_terminal(b"\x1b[c", config.timeout_ms);
        let attrs = parse_da1(&da1_response);
        if da1_has_sixel(&attrs) {
            caps.sixel.da1_advertised = true;
            caps.protocol = ProtocolKind::Sixel;
            caps.detection_method = DetectionMethod::ActiveProbe;
            return caps;
        }

        // DA2 query for terminal type identification
        let da2_response = query_terminal(b"\x1b[>c", config.timeout_ms);
        if let Some((_term_type, _version)) = parse_da2(&da2_response) {
            // Could use term_type to infer protocol, but for now we
            // fall through to env-var heuristics.
        }
    }

    // 5. Env-var fallback (existing heuristics from Picker)
    caps.detection_method = DetectionMethod::EnvVar;
    caps.protocol = detect_protocol_from_env();

    if caps.protocol == ProtocolKind::Halfblock {
        caps.detection_method = DetectionMethod::Fallback;
    }

    caps
}

/// Detect protocol using environment variables only.
fn detect_protocol_from_env() -> ProtocolKind {
    // Check for Kitty
    #[cfg(feature = "kitty")]
    {
        if let Ok(prog) = std::env::var("TERM_PROGRAM") {
            let prog_lower = prog.to_lowercase();
            if prog_lower.contains("kitty")
                || prog_lower.contains("wezterm")
                || prog_lower.contains("ghostty")
            {
                return ProtocolKind::Kitty;
            }
        }
        if let Ok(term) = std::env::var("TERM") {
            if term.contains("kitty") {
                return ProtocolKind::Kitty;
            }
        }
        if std::env::var("KITTY_WINDOW_ID").is_ok() {
            return ProtocolKind::Kitty;
        }
    }

    // Check for iTerm2
    #[cfg(feature = "iterm2")]
    {
        if let Ok(prog) = std::env::var("TERM_PROGRAM") {
            let prog_lower = prog.to_lowercase();
            if prog_lower.contains("iterm2") || prog_lower.contains("mintty") {
                return ProtocolKind::Iterm2;
            }
        }
        if let Ok(lc) = std::env::var("LC_TERMINAL") {
            if lc.to_lowercase().contains("iterm2") {
                return ProtocolKind::Iterm2;
            }
        }
    }

    // Check for Sixel
    #[cfg(feature = "sixel")]
    {
        if let Ok(prog) = std::env::var("TERM_PROGRAM") {
            let prog_lower = prog.to_lowercase();
            if prog_lower.contains("foot")
                || prog_lower.contains("mlterm")
                || prog_lower.contains("contour")
                || prog_lower.contains("yaft")
            {
                return ProtocolKind::Sixel;
            }
        }
        if let Ok(term) = std::env::var("TERM") {
            let term_lower = term.to_lowercase();
            if term_lower.contains("foot")
                || term_lower.contains("mlterm")
                || term_lower.contains("yaft")
            {
                return ProtocolKind::Sixel;
            }
        }
    }

    ProtocolKind::Halfblock
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_xtversion_kitty() {
        let response = b"\x1bP>|kitty 0.35.1\x1b\\";
        let result = parse_xtversion(response);
        assert_eq!(result, Some(("kitty".to_string(), "0.35.1".to_string())));
    }

    #[test]
    fn parse_xtversion_wezterm() {
        let response = b"\x1bP>|WezTerm 20240203\x1b\\";
        let result = parse_xtversion(response);
        assert_eq!(
            result,
            Some(("WezTerm".to_string(), "20240203".to_string()))
        );
    }

    #[test]
    fn parse_xtversion_empty() {
        let response = b"\x1bP>|\x1b\\";
        assert!(parse_xtversion(response).is_none());
    }

    #[test]
    fn parse_xtversion_no_dcs() {
        let response = b"garbage";
        assert!(parse_xtversion(response).is_none());
    }

    #[test]
    fn parse_da1_sixel() {
        // VT340-style DA1 with attribute 4 (Sixel)
        let response = b"\x1b[?62;4;6;22c";
        let attrs = parse_da1(response);
        assert_eq!(attrs, vec![62, 4, 6, 22]);
        assert!(da1_has_sixel(&attrs));
    }

    #[test]
    fn parse_da1_no_sixel() {
        let response = b"\x1b[?62;6;22c";
        let attrs = parse_da1(response);
        assert!(!da1_has_sixel(&attrs));
    }

    #[test]
    fn parse_da1_empty() {
        assert!(parse_da1(b"").is_empty());
        assert!(parse_da1(b"garbage").is_empty());
    }

    #[test]
    fn parse_da2_basic() {
        let response = b"\x1b[>1;4000;0c";
        let result = parse_da2(response);
        assert_eq!(result, Some((1, 4000)));
    }

    #[test]
    fn parse_da2_invalid() {
        assert!(parse_da2(b"").is_none());
        assert!(parse_da2(b"garbage").is_none());
    }

    #[test]
    fn kitty_graphics_response_ok() {
        assert!(parse_kitty_graphics_response(b"\x1b_Gi=31;OK\x1b\\"));
    }

    #[test]
    fn kitty_graphics_response_fail() {
        assert!(!parse_kitty_graphics_response(b"\x1b_Gi=31;ENOENT\x1b\\"));
    }

    #[test]
    fn protocol_from_name() {
        assert_eq!(
            protocol_from_terminal_name("kitty"),
            Some(ProtocolKind::Kitty)
        );
        assert_eq!(
            protocol_from_terminal_name("Ghostty"),
            Some(ProtocolKind::Kitty)
        );
        assert_eq!(
            protocol_from_terminal_name("WezTerm"),
            Some(ProtocolKind::Kitty)
        );
        assert_eq!(
            protocol_from_terminal_name("iTerm2"),
            Some(ProtocolKind::Iterm2)
        );
        assert_eq!(
            protocol_from_terminal_name("foot"),
            Some(ProtocolKind::Sixel)
        );
        assert_eq!(protocol_from_terminal_name("unknown"), None);
    }

    #[test]
    fn detect_multiplexer_none() {
        // In CI/test, we likely don't have TMUX/STY/ZELLIJ set
        // This is more of a smoke test
        let mux = detect_multiplexer();
        // Just verify it doesn't panic and returns a valid variant
        let _ = format!("{mux:?}");
    }
}
