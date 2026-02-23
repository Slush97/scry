// SPDX-License-Identifier: MIT OR Apache-2.0
//! Security gate — sanitizes VTE actions before they reach the grid.
//!
//! Every escape sequence from the PTY passes through this gate before
//! mutating terminal state. This prevents known attack vectors like
//! title echoback RCE, paste injection, hyperlink spoofing, and
//! resource exhaustion.

// ── Response policy ────────────────────────────────────────────────

/// Controls which terminal response sequences are sent back to the PTY.
///
/// Some programs (vim, tmux) query terminal capabilities by sending
/// escape sequences and reading the response. However, reflecting
/// arbitrary data back to the PTY is a known attack vector
/// (CVE-2003-0063, CVE-2022-45872, CVE-2023-39150).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ResponsePolicy {
    /// Allow standard responses except title reporting.
    ///
    /// This is the recommended default — it allows device attribute
    /// queries (needed by vim/tmux) while blocking the dangerous
    /// title report sequence.
    #[default]
    StandardMinusTitle,

    /// Allow only Device Attributes (DA) responses.
    ///
    /// Most restrictive useful policy. May break some programs that
    /// query cursor position or terminal status.
    MinimalRequired,

    /// Block all responses to the PTY.
    ///
    /// Maximum security. Will break vim's terminal detection and
    /// tmux's feature negotiation.
    None,

    /// Block everything including DA. No bytes ever go back to the PTY.
    ///
    /// Use when running untrusted commands and you want zero
    /// information leakage.
    Paranoid,
}

// ── Security gate ──────────────────────────────────────────────────

/// The security gate that filters escape sequences and responses.
pub struct SecurityGate {
    /// Response policy.
    policy: ResponsePolicy,
    /// Whether bracketed paste is enabled by the shell.
    pub bracketed_paste_enabled: bool,
    /// Response rate limiter: bytes sent in current window.
    rate_bytes: u64,
    /// Start of the current rate-limit window.
    rate_window_start: std::time::Instant,
    /// Maximum response bytes per second.
    rate_limit: u32,
}

impl SecurityGate {
    /// Create a new security gate with the given response policy.
    pub fn new(policy: ResponsePolicy) -> Self {
        Self {
            policy,
            bracketed_paste_enabled: false,
            rate_bytes: 0,
            rate_window_start: std::time::Instant::now(),
            rate_limit: 4096,
        }
    }

    /// Create a security gate with a custom response rate limit.
    pub fn with_rate_limit(policy: ResponsePolicy, rate_limit: u32) -> Self {
        Self {
            policy,
            bracketed_paste_enabled: false,
            rate_bytes: 0,
            rate_window_start: std::time::Instant::now(),
            rate_limit,
        }
    }

    // ── Response filtering ─────────────────────────────────────────

    /// Check if a response type is allowed by the current policy.
    pub fn allow_response(&mut self, response: ResponseType) -> bool {
        match self.policy {
            ResponsePolicy::None | ResponsePolicy::Paranoid => false,
            ResponsePolicy::MinimalRequired => matches!(
                response,
                ResponseType::DeviceAttributes | ResponseType::DeviceStatusOk
            ),
            ResponsePolicy::StandardMinusTitle => !matches!(response, ResponseType::TitleReport),
        }
    }

    /// Check if a response of the given byte length is within rate limits.
    ///
    /// Call this before writing response bytes to the PTY. Returns `false`
    /// if the rate would be exceeded, preventing exfiltration attacks.
    pub fn check_rate(&mut self, byte_count: usize) -> bool {
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.rate_window_start);
        if elapsed.as_secs() >= 1 {
            // Reset window
            self.rate_bytes = 0;
            self.rate_window_start = now;
        }
        let new_total = self.rate_bytes + byte_count as u64;
        if new_total > u64::from(self.rate_limit) {
            false
        } else {
            self.rate_bytes = new_total;
            true
        }
    }

    // ── OSC filtering ──────────────────────────────────────────────

    /// Filter an OSC (Operating System Command) sequence.
    ///
    /// Returns the action to take: allow, block, or allow with constraints.
    pub fn filter_osc(&self, params: &[&[u8]]) -> OscAction {
        let cmd = params.first().and_then(|p| {
            std::str::from_utf8(p)
                .ok()
                .and_then(|s| s.parse::<u32>().ok())
        });

        match cmd {
            // Window title (set) — allowed
            Some(0 | 2) => OscAction::SetTitle,

            // Icon name — allowed but we ignore it
            Some(1) => OscAction::Ignore,

            // Clipboard (OSC 52)
            Some(52) => self.filter_clipboard(params),

            // Hyperlinks (OSC 8)
            Some(8) => self.filter_hyperlink(params),

            // Foreground color query (OSC 10)
            Some(10) => {
                if params.get(1).map_or(false, |p| p == b"?") {
                    OscAction::QueryFg
                } else {
                    OscAction::Ignore // set not yet implemented
                }
            }

            // Background color query (OSC 11)
            Some(11) => {
                if params.get(1).map_or(false, |p| p == b"?") {
                    OscAction::QueryBg
                } else {
                    OscAction::Ignore // set not yet implemented
                }
            }

            // Other color queries (OSC 12–19) — no security risk
            Some(12..=19) => OscAction::Ignore,

            // Unknown — block by default
            _ => OscAction::Ignore,
        }
    }

    /// Filter clipboard OSC 52 commands.
    fn filter_clipboard(&self, params: &[&[u8]]) -> OscAction {
        /// Maximum clipboard payload size (1 MiB base64 ≈ 768 KiB decoded).
        const MAX_CLIPBOARD_B64: usize = 1024 * 1024;

        // OSC 52 format: 52;target;data
        // SET (data is base64) — allowed, the program is pushing to clipboard
        // QUERY (data is "?") — blocked, could leak clipboard contents to PTY
        if let Some(data) = params.get(2) {
            if data == b"?" {
                OscAction::Block // Query → blocked
            } else if data.len() > MAX_CLIPBOARD_B64 {
                OscAction::Block // Oversized clipboard payload
            } else {
                OscAction::ClipboardSet // Set → allowed
            }
        } else {
            OscAction::Block
        }
    }

    /// Filter hyperlink OSC 8 commands.
    fn filter_hyperlink(&self, params: &[&[u8]]) -> OscAction {
        // OSC 8 format: 8;params;uri
        let uri = params.get(2).and_then(|p| std::str::from_utf8(p).ok());

        match uri {
            None | Some("") => OscAction::HyperlinkClose, // Close hyperlink — always OK
            Some(url) => {
                if validate_url_scheme(url) {
                    OscAction::HyperlinkOpen(url.to_string())
                } else {
                    OscAction::Block // Dangerous scheme
                }
            }
        }
    }
}

// ── URL validation ─────────────────────────────────────────────────

/// Validate that a URL uses an allowed scheme.
///
/// Blocks dangerous schemes like `file://`, `javascript:`, `data:`.
fn validate_url_scheme(url: &str) -> bool {
    let lower = url.to_ascii_lowercase();
    lower.starts_with("https://")
        || lower.starts_with("http://")
        || lower.starts_with("mailto:")
        || lower.starts_with("ssh://")
        || lower.starts_with("git://")
}

// ── Action types ───────────────────────────────────────────────────

/// Types of terminal responses that might be sent to the PTY.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponseType {
    /// Device Attributes (DA): `\e[c` → `\e[?1;2c`
    DeviceAttributes,
    /// Device Status OK: `\e[5n` → `\e[0n`
    DeviceStatusOk,
    /// Cursor Position Report: `\e[6n` → `\e[row;colR`
    CursorPosition,
    /// Title Report: `\e[21t` → title string (DANGEROUS)
    TitleReport,
}

/// Action resulting from OSC filtering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OscAction {
    /// Allow the OSC through unchanged.
    Allow,
    /// Set the window title.
    SetTitle,
    /// Ignore (no side effects).
    Ignore,
    /// Block (potentially dangerous).
    Block,
    /// Set clipboard contents.
    ClipboardSet,
    /// Open a hyperlink.
    HyperlinkOpen(String),
    /// Close the current hyperlink.
    HyperlinkClose,
    /// Query foreground color (OSC 10 ?).
    QueryFg,
    /// Query background color (OSC 11 ?).
    QueryBg,
}

/// Sanitize an OSC string parameter by stripping embedded escape sequences.
///
/// Malicious PTY output can embed ESC, CSI, OSC, or DCS starters inside
/// OSC parameters (e.g., inside a title string). This function removes
/// control characters and escape sequences to prevent injection.
pub fn sanitize_osc_string(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_control() || *c == ' ')
        .take(4096)
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_blocks_title_report() {
        let mut gate = SecurityGate::new(ResponsePolicy::default());
        assert!(gate.allow_response(ResponseType::DeviceAttributes));
        assert!(gate.allow_response(ResponseType::CursorPosition));
        assert!(!gate.allow_response(ResponseType::TitleReport));
    }

    #[test]
    fn minimal_policy_blocks_cursor_report() {
        let mut gate = SecurityGate::new(ResponsePolicy::MinimalRequired);
        assert!(gate.allow_response(ResponseType::DeviceAttributes));
        assert!(!gate.allow_response(ResponseType::CursorPosition));
        assert!(!gate.allow_response(ResponseType::TitleReport));
    }

    #[test]
    fn none_policy_blocks_everything() {
        let mut gate = SecurityGate::new(ResponsePolicy::None);
        assert!(!gate.allow_response(ResponseType::DeviceAttributes));
        assert!(!gate.allow_response(ResponseType::CursorPosition));
    }

    #[test]
    fn url_validation() {
        assert!(validate_url_scheme("https://example.com"));
        assert!(validate_url_scheme("http://example.com"));
        assert!(validate_url_scheme("mailto:user@example.com"));
        assert!(validate_url_scheme("ssh://host"));
        assert!(!validate_url_scheme("file:///etc/passwd"));
        assert!(!validate_url_scheme("javascript:alert(1)"));
        assert!(!validate_url_scheme("data:text/html,<h1>hi</h1>"));
        assert!(!validate_url_scheme("ftp://files.example.com"));
    }

    #[test]
    fn osc_title_allowed() {
        let gate = SecurityGate::new(ResponsePolicy::default());
        let result = gate.filter_osc(&[b"0", b"My Title"]);
        assert_eq!(result, OscAction::SetTitle);
    }

    #[test]
    fn osc_clipboard_query_blocked() {
        let gate = SecurityGate::new(ResponsePolicy::default());
        let result = gate.filter_osc(&[b"52", b"c", b"?"]);
        assert_eq!(result, OscAction::Block);
    }

    #[test]
    fn osc_clipboard_set_allowed() {
        let gate = SecurityGate::new(ResponsePolicy::default());
        let result = gate.filter_osc(&[b"52", b"c", b"SGVsbG8="]);
        assert_eq!(result, OscAction::ClipboardSet);
    }

    #[test]
    fn osc_hyperlink_safe_url() {
        let gate = SecurityGate::new(ResponsePolicy::default());
        let result = gate.filter_osc(&[b"8", b"", b"https://example.com"]);
        assert_eq!(
            result,
            OscAction::HyperlinkOpen("https://example.com".to_string())
        );
    }

    #[test]
    fn osc_hyperlink_dangerous_url() {
        let gate = SecurityGate::new(ResponsePolicy::default());
        let result = gate.filter_osc(&[b"8", b"", b"file:///etc/passwd"]);
        assert_eq!(result, OscAction::Block);
    }

    #[test]
    fn osc_hyperlink_close() {
        let gate = SecurityGate::new(ResponsePolicy::default());
        let result = gate.filter_osc(&[b"8", b"", b""]);
        assert_eq!(result, OscAction::HyperlinkClose);
    }

    #[test]
    fn osc_clipboard_set_oversized_blocked() {
        let gate = SecurityGate::new(ResponsePolicy::default());
        // 2 MiB of base64 — should be blocked
        let big = vec![b'A'; 2 * 1024 * 1024];
        let result = gate.filter_osc(&[b"52", b"c", &big]);
        assert_eq!(result, OscAction::Block);
    }

    #[test]
    fn osc_clipboard_set_small_allowed() {
        let gate = SecurityGate::new(ResponsePolicy::default());
        // 100 bytes — well under the limit
        let small = vec![b'A'; 100];
        let result = gate.filter_osc(&[b"52", b"c", &small]);
        assert_eq!(result, OscAction::ClipboardSet);
    }

    #[test]
    fn paranoid_policy_blocks_all() {
        let mut gate = SecurityGate::new(ResponsePolicy::Paranoid);
        assert!(!gate.allow_response(ResponseType::DeviceAttributes));
        assert!(!gate.allow_response(ResponseType::CursorPosition));
        assert!(!gate.allow_response(ResponseType::TitleReport));
        assert!(!gate.allow_response(ResponseType::DeviceStatusOk));
    }

    #[test]
    fn response_rate_limiter() {
        let mut gate = SecurityGate::with_rate_limit(ResponsePolicy::default(), 100);
        // First 100 bytes should be allowed
        assert!(gate.check_rate(50));
        assert!(gate.check_rate(50));
        // Next byte should be rejected
        assert!(!gate.check_rate(1));
    }

    #[test]
    fn sanitize_osc_string_strips_escapes() {
        let dirty = "hello\x1b[31mworld\x07";
        let clean = sanitize_osc_string(dirty);
        // ESC and BEL are stripped; "[31m" is printable ASCII and stays
        assert_eq!(clean, "hello[31mworld");
        assert!(!clean.contains('\x1b'));
        assert!(!clean.contains('\x07'));
    }

    #[test]
    fn sanitize_osc_string_preserves_spaces() {
        let s = "My Terminal Title";
        assert_eq!(sanitize_osc_string(s), s);
    }

    #[test]
    fn sanitize_osc_string_truncates_long() {
        let long = "A".repeat(10_000);
        assert_eq!(sanitize_osc_string(&long).len(), 4096);
    }

    #[test]
    fn osc_query_fg() {
        let gate = SecurityGate::new(ResponsePolicy::default());
        let result = gate.filter_osc(&[b"10", b"?"]);
        assert_eq!(result, OscAction::QueryFg);
    }

    #[test]
    fn osc_query_bg() {
        let gate = SecurityGate::new(ResponsePolicy::default());
        let result = gate.filter_osc(&[b"11", b"?"]);
        assert_eq!(result, OscAction::QueryBg);
    }
}
