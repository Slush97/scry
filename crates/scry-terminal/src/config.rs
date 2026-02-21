// SPDX-License-Identifier: MIT OR Apache-2.0
//! Terminal configuration loaded from TOML.
//!
//! Reads `~/.config/scry/terminal.toml` (or `$XDG_CONFIG_HOME/scry/terminal.toml`)
//! and falls back to sensible defaults for every field.

use serde::Deserialize;
use std::path::PathBuf;

// ── Configuration structs ──────────────────────────────────────────

/// Top-level terminal configuration.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct TerminalConfig {
    /// Font settings.
    pub font: FontConfig,
    /// Color scheme.
    pub colors: ColorConfig,
    /// Scrollback settings.
    pub scrollback: ScrollbackConfig,
    /// Cursor settings.
    pub cursor: CursorConfig,
    /// Shell override (default: `$SHELL` or `/bin/sh`).
    pub shell: Option<String>,
    /// Initial window dimensions in cells.
    pub window: WindowConfig,
}

/// Font configuration.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct FontConfig {
    /// Font family name (used for system font lookup).
    pub family: String,
    /// Optional path to a TTF/OTF file (overrides family lookup).
    pub path: Option<PathBuf>,
    /// Font size in pixels.
    pub size: f32,
}

/// Color scheme.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct ColorConfig {
    /// Foreground color as hex string (e.g., "#c0caf5").
    pub foreground: String,
    /// Background color as hex string (e.g., "#1a1b26").
    pub background: String,
    /// The 16 ANSI colors (black, red, green, yellow, blue, magenta, cyan, white × normal + bright).
    pub palette: [String; 16],
}

/// Scrollback buffer settings.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct ScrollbackConfig {
    /// Maximum lines to keep in scrollback.
    pub lines: usize,
}

/// Cursor appearance.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct CursorConfig {
    /// Cursor style.
    pub style: CursorStyleConfig,
    /// Whether the cursor blinks.
    pub blink: bool,
    /// Cursor color as hex string (default: same as foreground).
    pub color: Option<String>,
}

/// Cursor shape.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CursorStyleConfig {
    /// Filled block cursor (█).
    #[default]
    Block,
    /// Vertical bar cursor (│).
    Bar,
    /// Horizontal underline cursor (_).
    Underline,
}

/// Initial window dimensions.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct WindowConfig {
    /// Initial columns.
    pub columns: u16,
    /// Initial rows.
    pub rows: u16,
}

// ── Defaults ───────────────────────────────────────────────────────

impl Default for TerminalConfig {
    fn default() -> Self {
        Self {
            font: FontConfig::default(),
            colors: ColorConfig::default(),
            scrollback: ScrollbackConfig::default(),
            cursor: CursorConfig::default(),
            shell: None,
            window: WindowConfig::default(),
        }
    }
}

impl Default for FontConfig {
    fn default() -> Self {
        Self {
            family: "monospace".to_string(),
            path: None,
            size: 14.0,
        }
    }
}

impl Default for ColorConfig {
    fn default() -> Self {
        // Tokyo Night inspired dark theme
        Self {
            foreground: "#c0caf5".to_string(),
            background: "#1a1b26".to_string(),
            palette: [
                "#15161e".to_string(), // black
                "#f7768e".to_string(), // red
                "#9ece6a".to_string(), // green
                "#e0af68".to_string(), // yellow
                "#7aa2f7".to_string(), // blue
                "#bb9af7".to_string(), // magenta
                "#7dcfff".to_string(), // cyan
                "#a9b1d6".to_string(), // white
                "#414868".to_string(), // bright black
                "#f7768e".to_string(), // bright red
                "#9ece6a".to_string(), // bright green
                "#e0af68".to_string(), // bright yellow
                "#7aa2f7".to_string(), // bright blue
                "#bb9af7".to_string(), // bright magenta
                "#7dcfff".to_string(), // bright cyan
                "#c0caf5".to_string(), // bright white
            ],
        }
    }
}

impl Default for ScrollbackConfig {
    fn default() -> Self {
        Self { lines: 5000 }
    }
}

impl Default for CursorConfig {
    fn default() -> Self {
        Self {
            style: CursorStyleConfig::Block,
            blink: true,
            color: None,
        }
    }
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            columns: 80,
            rows: 24,
        }
    }
}

// ── Color parsing ──────────────────────────────────────────────────

/// Parse a hex color string like "#c0caf5" into (r, g, b).
pub fn parse_hex_color(hex: &str) -> Option<(u8, u8, u8)> {
    let hex = hex.strip_prefix('#').unwrap_or(hex);
    if hex.len() != 6 {
        return None;
    }
    let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
    let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
    let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
    Some((r, g, b))
}

impl ColorConfig {
    /// Parse the foreground color.
    pub fn fg_rgb(&self) -> (u8, u8, u8) {
        parse_hex_color(&self.foreground).unwrap_or((192, 202, 245))
    }

    /// Parse the background color.
    pub fn bg_rgb(&self) -> (u8, u8, u8) {
        parse_hex_color(&self.background).unwrap_or((26, 27, 38))
    }

    /// Get an ANSI palette color by index (0–15).
    pub fn palette_rgb(&self, index: u8) -> (u8, u8, u8) {
        let idx = index as usize;
        if idx < 16 {
            parse_hex_color(&self.palette[idx]).unwrap_or((128, 128, 128))
        } else {
            // Extended 256-color: compute from the 6×6×6 cube + grayscale ramp
            compute_256_color(index)
        }
    }
}

/// Compute RGB for extended 256-color indices (16–255).
pub fn compute_256_color(index: u8) -> (u8, u8, u8) {
    let idx = index as u16;
    if idx < 16 {
        // Shouldn't reach here, but fallback
        (128, 128, 128)
    } else if idx < 232 {
        // 6×6×6 color cube
        let idx = idx - 16;
        let r = (idx / 36) as u8;
        let g = ((idx % 36) / 6) as u8;
        let b = (idx % 6) as u8;
        let to_val = |c: u8| if c == 0 { 0 } else { 55 + 40 * c };
        (to_val(r), to_val(g), to_val(b))
    } else {
        // Grayscale ramp: 232–255 → 8, 18, ..., 238
        let shade = 8 + 10 * (idx - 232) as u8;
        (shade, shade, shade)
    }
}

// ── Loading ────────────────────────────────────────────────────────

impl TerminalConfig {
    /// Load configuration from the default config path.
    ///
    /// Falls back to defaults if the file doesn't exist or can't be parsed.
    pub fn load() -> Self {
        if let Some(path) = Self::config_path() {
            Self::load_from(&path).unwrap_or_default()
        } else {
            Self::default()
        }
    }

    /// Load configuration from a specific file path.
    pub fn load_from(path: &std::path::Path) -> Option<Self> {
        let contents = std::fs::read_to_string(path).ok()?;
        toml::from_str(&contents).ok()
    }

    /// Default config file path: `$XDG_CONFIG_HOME/scry/terminal.toml`
    /// or `~/.config/scry/terminal.toml`.
    pub fn config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("scry").join("terminal.toml"))
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let config = TerminalConfig::default();
        assert_eq!(config.font.size, 14.0);
        assert_eq!(config.window.columns, 80);
        assert_eq!(config.window.rows, 24);
        assert_eq!(config.scrollback.lines, 5000);
    }

    #[test]
    fn parse_hex_colors() {
        assert_eq!(parse_hex_color("#c0caf5"), Some((192, 202, 245)));
        assert_eq!(parse_hex_color("#000000"), Some((0, 0, 0)));
        assert_eq!(parse_hex_color("#ffffff"), Some((255, 255, 255)));
        assert_eq!(parse_hex_color("1a1b26"), Some((26, 27, 38)));
        assert_eq!(parse_hex_color("invalid"), None);
        assert_eq!(parse_hex_color("#fff"), None);
    }

    #[test]
    fn color_config_defaults() {
        let colors = ColorConfig::default();
        assert_eq!(colors.fg_rgb(), (192, 202, 245));
        assert_eq!(colors.bg_rgb(), (26, 27, 38));
    }

    #[test]
    fn extended_256_colors() {
        // Index 16 = first cube color (0,0,0) = (0,0,0)
        assert_eq!(compute_256_color(16), (0, 0, 0));
        // Index 196 = red-ish (5,0,0) = (255,0,0)
        assert_eq!(compute_256_color(196), (255, 0, 0));
        // Index 232 = first grayscale = 8
        assert_eq!(compute_256_color(232), (8, 8, 8));
        // Index 255 = last grayscale = 238
        assert_eq!(compute_256_color(255), (238, 238, 238));
    }

    #[test]
    fn deserialize_minimal_toml() {
        let toml_str = r#"
            [font]
            size = 16.0
            
            [window]
            columns = 120
            rows = 40
        "#;
        let config: TerminalConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.font.size, 16.0);
        assert_eq!(config.window.columns, 120);
        assert_eq!(config.window.rows, 40);
        // Defaults should still apply for unspecified fields
        assert_eq!(config.scrollback.lines, 5000);
    }

    #[test]
    fn cursor_style_deserialize() {
        let toml_str = r#"
            [cursor]
            style = "bar"
            blink = false
        "#;
        let config: TerminalConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.cursor.style, CursorStyleConfig::Bar);
        assert!(!config.cursor.blink);
    }
}
