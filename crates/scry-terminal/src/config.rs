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
    /// Bell (BEL character) behaviour.
    pub bell: BellConfig,
    /// Performance tuning.
    pub performance: PerformanceConfig,
    /// Security policy.
    pub security: SecurityConfig,
    /// Key bindings (scaffold — not yet dispatched at runtime).
    pub keybinds: KeybindsConfig,
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
    /// Bold font family override (empty = derive from `family`).
    pub bold_family: String,
    /// Italic font family override (empty = derive from `family`).
    pub italic_family: String,
    /// Line height multiplier (1.0 = tight, 1.2 = comfortable).
    pub line_height: f32,
    /// Enable OpenType ligature substitution.
    pub ligatures: bool,
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
    /// Half-period of the blink cycle in milliseconds.
    pub blink_rate_ms: u32,
    /// Pixel thickness for bar and underline styles.
    pub thickness: f32,
    /// Fill opacity (0.0–1.0).
    pub opacity: f32,
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
    /// Padding around terminal content in pixels.
    pub padding: f32,
    /// Background opacity (0.0 = fully transparent, 1.0 = fully opaque).
    /// Compositor support required for values below 1.0.
    pub opacity: f32,
    /// Subtle background gradient effect.
    pub background_gradient: bool,
    /// Scrollbar visibility: "none", "thin", or "overlay".
    pub scrollbar: String,
    /// Window title template. `{title}` is replaced with the app-set title.
    pub title_format: String,
    /// Ask before closing if a child process is running.
    pub confirm_close: bool,
}

/// Bell (BEL character) configuration.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct BellConfig {
    /// Master on/off switch. When false, BEL (0x07) is silently ignored.
    pub enabled: bool,
    /// Flash the screen briefly on BEL.
    pub visual: bool,
    /// Duration of the visual flash in milliseconds.
    pub flash_duration_ms: u32,
    /// Emit a system audio beep on BEL.
    pub audio: bool,
}

/// Performance tuning.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct PerformanceConfig {
    /// Maximum target framerate. 0 = vsync-limited.
    pub fps_cap: u32,
    /// Keep the render loop spinning even when idle.
    pub render_on_idle: bool,
}

/// Security policy.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct SecurityConfig {
    /// Response policy: "default", "minimal", "none", or "paranoid".
    pub response_policy: String,
    /// Maximum clipboard content size in bytes (OSC 52).
    pub max_clipboard_bytes: usize,
    /// Maximum response bytes per second (anti-exfiltration).
    pub response_rate_limit: u32,
}

/// Key binding overrides (scaffold — not yet dispatched).
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct KeybindsConfig {
    /// Copy binding (default: Ctrl+Shift+C).
    pub copy: String,
    /// Paste binding (default: Ctrl+Shift+V).
    pub paste: String,
    /// Zoom in binding.
    pub zoom_in: String,
    /// Zoom out binding.
    pub zoom_out: String,
    /// Zoom reset binding.
    pub zoom_reset: String,
    /// Scroll up binding.
    pub scroll_up: String,
    /// Scroll down binding.
    pub scroll_down: String,
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
            bell: BellConfig::default(),
            performance: PerformanceConfig::default(),
            security: SecurityConfig::default(),
            keybinds: KeybindsConfig::default(),
        }
    }
}

impl Default for FontConfig {
    fn default() -> Self {
        Self {
            family: "monospace".to_string(),
            path: None,
            size: 14.0,
            bold_family: String::new(),
            italic_family: String::new(),
            line_height: 1.0,
            ligatures: false,
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
            blink_rate_ms: 530,
            thickness: 2.0,
            opacity: 1.0,
        }
    }
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            columns: 80,
            rows: 24,
            padding: 20.0,
            opacity: 1.0,
            background_gradient: false,
            scrollbar: "none".to_string(),
            title_format: "{title}".to_string(),
            confirm_close: false,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            fps_cap: 60,
            render_on_idle: false,
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            response_policy: "default".to_string(),
            max_clipboard_bytes: 1_048_576,
            response_rate_limit: 4096,
        }
    }
}

impl Default for KeybindsConfig {
    fn default() -> Self {
        Self {
            copy: String::new(),
            paste: String::new(),
            zoom_in: String::new(),
            zoom_out: String::new(),
            zoom_reset: String::new(),
            scroll_up: String::new(),
            scroll_down: String::new(),
        }
    }
}

impl Default for BellConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            visual: false,
            flash_duration_ms: 120,
            audio: false,
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
        let mut config = if let Some(path) = Self::config_path() {
            Self::load_from(&path).unwrap_or_default()
        } else {
            Self::default()
        };
        config.validate();
        config
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

    /// Persist a new font size to the config file.
    ///
    /// Reads the existing TOML (or creates a minimal one), updates `font.size`,
    /// and writes it back. Preserves the rest of the file contents.
    pub fn save_font_size(size: f32) {
        let Some(path) = Self::config_path() else {
            return;
        };
        // Read existing content or start fresh
        let content = std::fs::read_to_string(&path).unwrap_or_default();
        let mut doc: toml::Table = toml::from_str(&content).unwrap_or_default();

        // Ensure [font] table exists and set size
        let font = doc
            .entry("font")
            .or_insert_with(|| toml::Value::Table(toml::Table::new()));
        if let toml::Value::Table(t) = font {
            t.insert(
                "size".to_string(),
                toml::Value::Float(f64::from(size)),
            );
        }

        let output = toml::to_string_pretty(&doc).unwrap_or_default();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        // Atomic write: write to temp file, then rename (prevents corruption on crash)
        let tmp_path = path.with_extension("toml.tmp");
        if std::fs::write(&tmp_path, &output).is_ok() {
            let _ = std::fs::rename(&tmp_path, &path);
        }
    }

    /// Clamp configuration values to safe ranges.
    ///
    /// Prevents panics, GPU glitches, or OOM from extreme values in
    /// user-provided `terminal.toml`.
    pub fn validate(&mut self) {
        self.font.size = self.font.size.clamp(6.0, 72.0);
        self.font.line_height = self.font.line_height.clamp(0.5, 3.0);
        self.scrollback.lines = self.scrollback.lines.clamp(0, 100_000);
        self.window.columns = self.window.columns.clamp(10, 500);
        self.window.rows = self.window.rows.clamp(2, 200);
        self.window.padding = self.window.padding.clamp(0.0, 64.0);
        self.window.opacity = self.window.opacity.clamp(0.0, 1.0);
        self.cursor.blink_rate_ms = self.cursor.blink_rate_ms.clamp(100, 5000);
        self.cursor.thickness = self.cursor.thickness.clamp(1.0, 8.0);
        self.cursor.opacity = self.cursor.opacity.clamp(0.0, 1.0);
        self.performance.fps_cap = self.performance.fps_cap.min(240);
        self.security.max_clipboard_bytes = self.security.max_clipboard_bytes.clamp(0, 16_777_216);
        self.security.response_rate_limit = self.security.response_rate_limit.clamp(256, 65_536);
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

    #[test]
    fn opacity_config() {
        // Default is fully opaque
        let config = TerminalConfig::default();
        assert!((config.window.opacity - 1.0).abs() < f32::EPSILON);

        // Deserialize custom opacity
        let toml_str = r#"
            [window]
            opacity = 0.9
        "#;
        let config: TerminalConfig = toml::from_str(toml_str).unwrap();
        assert!((config.window.opacity - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn bell_config_defaults() {
        let config = TerminalConfig::default();
        assert!(config.bell.enabled);
        assert!(!config.bell.visual);
        assert!(!config.bell.audio);
        assert_eq!(config.bell.flash_duration_ms, 120);
    }

    #[test]
    fn bell_config_deserialize() {
        let toml_str = r#"
            [bell]
            enabled = true
            visual = true
            flash_duration_ms = 200
        "#;
        let config: TerminalConfig = toml::from_str(toml_str).unwrap();
        assert!(config.bell.visual);
        assert_eq!(config.bell.flash_duration_ms, 200);
    }

    #[test]
    fn config_validation_clamps_extremes() {
        let mut config = TerminalConfig::default();
        config.font.size = -5.0;
        config.scrollback.lines = 999_999_999;
        config.window.columns = 0;
        config.window.rows = 0;
        config.window.padding = 200.0;
        config.window.opacity = 3.5;

        config.validate();

        assert_eq!(config.font.size, 6.0);
        assert_eq!(config.scrollback.lines, 100_000);
        assert_eq!(config.window.columns, 10);
        assert_eq!(config.window.rows, 2);
        assert_eq!(config.window.padding, 64.0);
        assert!((config.window.opacity - 1.0).abs() < f32::EPSILON);
    }
}
