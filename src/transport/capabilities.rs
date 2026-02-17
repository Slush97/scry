// SPDX-License-Identifier: MIT OR Apache-2.0
//! Terminal capability data types.
//!
//! These types describe what a terminal can do, as detected by the probing
//! pipeline in [`probe`](super::probe). They are pure data — no I/O, no
//! side effects.

use crate::transport::backend::ProtocolKind;

// ---------------------------------------------------------------------------
// Detection method
// ---------------------------------------------------------------------------

/// How a capability was determined.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DetectionMethod {
    /// Explicitly set via `SCRY_PROTOCOL` env var.
    ManualOverride,
    /// Detected via XTVERSION, DA1, DA2, or graphics query escape sequences.
    ActiveProbe,
    /// Inferred from environment variables (TERM, TERM_PROGRAM, etc.).
    EnvVar,
    /// Default fallback (halfblock).
    Fallback,
}

// ---------------------------------------------------------------------------
// Multiplexer
// ---------------------------------------------------------------------------

/// Terminal multiplexer detected around the session.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Multiplexer {
    /// tmux
    Tmux,
    /// GNU Screen
    Screen,
    /// Zellij
    Zellij,
    /// No multiplexer detected.
    None,
}

// ---------------------------------------------------------------------------
// Kitty features
// ---------------------------------------------------------------------------

/// Detailed Kitty graphics protocol capabilities.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct KittyFeatures {
    /// Whether the terminal responded OK to a Kitty graphics query.
    pub graphics_supported: bool,
    /// Whether shared memory transmission is supported.
    pub shm_supported: bool,
    /// Whether Unicode placement is supported (Kitty 0.28+).
    pub unicode_placement: bool,
}

// ---------------------------------------------------------------------------
// Sixel features
// ---------------------------------------------------------------------------

/// Detailed Sixel protocol capabilities.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct SixelFeatures {
    /// Maximum color register count (0 = unknown).
    pub max_colors: u32,
    /// Whether the terminal advertised Sixel via DA1 attribute 4.
    pub da1_advertised: bool,
}

// ---------------------------------------------------------------------------
// ProbeConfig
// ---------------------------------------------------------------------------

/// Configuration for the terminal probing pipeline.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ProbeConfig {
    /// Whether to perform active escape-sequence probing.
    ///
    /// Set to `false` for env-var-only detection (faster, but less accurate).
    pub active_probe: bool,
    /// Timeout per query in milliseconds.
    pub timeout_ms: u64,
    /// Whether to attempt tmux passthrough for probing.
    pub tmux_passthrough: bool,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            active_probe: true,
            timeout_ms: 150,
            tmux_passthrough: true,
        }
    }
}

impl ProbeConfig {
    /// Create a config that only uses environment variables (no TTY probing).
    #[must_use]
    pub fn env_only() -> Self {
        Self {
            active_probe: false,
            timeout_ms: 0,
            tmux_passthrough: false,
        }
    }
}

// ---------------------------------------------------------------------------
// TerminalCapabilities
// ---------------------------------------------------------------------------

/// Complete terminal capability profile.
///
/// Contains everything known about the terminal's graphics support,
/// detected via the probing pipeline and cached globally.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TerminalCapabilities {
    /// The recommended graphics protocol.
    pub protocol: ProtocolKind,
    /// How the protocol was determined.
    pub detection_method: DetectionMethod,
    /// Multiplexer wrapping the session, if any.
    pub multiplexer: Multiplexer,
    /// Terminal name as reported by XTVERSION (if available).
    pub terminal_name: Option<String>,
    /// Terminal version as reported by XTVERSION (if available).
    pub terminal_version: Option<String>,
    /// Kitty-specific capabilities (only meaningful if protocol == Kitty).
    pub kitty: KittyFeatures,
    /// Sixel-specific capabilities (only meaningful if protocol == Sixel).
    pub sixel: SixelFeatures,
    /// Whether stdout is a TTY.
    pub is_tty: bool,
}

impl Default for TerminalCapabilities {
    fn default() -> Self {
        Self {
            protocol: ProtocolKind::Halfblock,
            detection_method: DetectionMethod::Fallback,
            multiplexer: Multiplexer::None,
            terminal_name: None,
            terminal_version: None,
            kitty: KittyFeatures::default(),
            sixel: SixelFeatures::default(),
            is_tty: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_capabilities() {
        let caps = TerminalCapabilities::default();
        assert_eq!(caps.protocol, ProtocolKind::Halfblock);
        assert_eq!(caps.detection_method, DetectionMethod::Fallback);
        assert_eq!(caps.multiplexer, Multiplexer::None);
        assert!(!caps.is_tty);
    }

    #[test]
    fn probe_config_env_only() {
        let config = ProbeConfig::env_only();
        assert!(!config.active_probe);
    }

    #[test]
    fn kitty_features_default() {
        let f = KittyFeatures::default();
        assert!(!f.graphics_supported);
        assert!(!f.shm_supported);
    }
}
