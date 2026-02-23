// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fetch scene renderer — builds the `PixelCanvas` for each frame.
//!
//! Layout: logo on the left (square aspect), system info text on the right.
//! Text is rasterized via scry-engine's fontdue-based text rendering.

use scry_engine::scene::PixelCanvas;
use scry_engine::style::Color as C;

use crate::config::{FetchConfig, FetchTheme};

use super::logo::{self, LogoSource};
use super::sysinfo::SysInfo;

// ── Built-in field metadata ─────────────────────────────────────

struct FieldMeta {
    icon: &'static str,
    label: &'static str,
    color: (u8, u8, u8),
}

fn builtin_field(id: &str) -> FieldMeta {
    match id {
        "os"       => FieldMeta { icon: "󰀧", label: "OS",       color: (168, 213, 186) },
        "kernel"   => FieldMeta { icon: "",  label: "Kernel",   color: (158, 193, 255) },
        "uptime"   => FieldMeta { icon: "󰔟", label: "Uptime",   color: (244, 192, 149) },
        "shell"    => FieldMeta { icon: "",  label: "Shell",    color: (243, 231, 179) },
        "terminal" => FieldMeta { icon: "",  label: "Terminal", color: (203, 182, 255) },
        "de_wm"    => FieldMeta { icon: "󰖲", label: "DE / WM",  color: (242, 181, 212) },
        "packages" => FieldMeta { icon: "󰏗", label: "Packages", color: (168, 213, 186) },
        "memory"   => FieldMeta { icon: "",  label: "Memory",   color: (158, 193, 255) },
        "cpu"      => FieldMeta { icon: "󰘚", label: "CPU",      color: (203, 182, 255) },
        _          => FieldMeta { icon: "●",  label: "?",        color: (232, 226, 215) },
    }
}

// ── Renderer ────────────────────────────────────────────────────

/// The fetch scene renderer. Holds collected sysinfo and resolved logo.
pub(crate) struct FetchRenderer {
    info: SysInfo,
    logo: LogoSource,
    fields: Vec<String>,
    theme: FetchTheme,
}

impl FetchRenderer {
    /// Create a new renderer from config and pre-collected sysinfo.
    pub(super) fn new(config: &FetchConfig, info: SysInfo) -> Self {
        let logo = LogoSource::resolve(&config.logo);
        Self {
            info,
            logo,
            fields: config.fields.clone(),
            theme: config.theme.clone(),
        }
    }

    /// Compute the ideal banner height in pixels given cell dimensions.
    pub(super) fn banner_height(&self, cell_height: f32, padding: f32) -> u32 {
        // Header: user@host + separator = 2 lines
        // Fields: one per field
        // Bottom padding: 1 line
        let text_lines = 2 + self.fields.len() + 1;
        let text_height = text_lines as f32 * cell_height;
        (padding + text_height + padding) as u32
    }

    /// Build the composite `PixelCanvas` for one frame.
    ///
    /// The logo is rendered as a separate canvas (since there's no composite
    /// method). Instead, we render the logo directly into the main canvas
    /// by rasterizing it separately and then rendering text on top.
    ///
    /// `screen_width` is the full terminal width in pixels.
    /// `t` is animation time in seconds.
    pub(super) fn render(&self, screen_width: u32, banner_height: u32, t: f32) -> PixelCanvas {
        if screen_width == 0 || banner_height == 0 {
            return PixelCanvas::new(1, 1);
        }

        // Since we can't composite two canvases, render the logo as the base
        // canvas at full banner dimensions, then overlay text on it.
        // The logo renders into the left portion, text goes on the right.
        let logo_w = banner_height.min(screen_width / 3);

        // Start with the logo rendered at full banner size
        // (it will only draw in the left `logo_w` region)
        let mut canvas = logo::render_logo(&self.logo, screen_width, banner_height, t);

        // Render sysinfo text on the right side
        let text_x = logo_w as f32 + 16.0;
        let line_h = (banner_height as f32 - 32.0) / (2.0 + self.fields.len() as f32 + 1.0);
        let font_size = (line_h * 0.7).clamp(12.0, 24.0);
        let mut y = 16.0 + line_h * 0.5;

        // user@host header
        let (user_r, user_g, user_b) = parse_hex_rgb(&self.theme.title_user);
        let (at_r, at_g, at_b) = parse_hex_rgb(&self.theme.title_at);
        let (host_r, host_g, host_b) = parse_hex_rgb(&self.theme.title_host);

        let parts: Vec<&str> = self.info.user_at_host.splitn(2, '@').collect();
        if parts.len() == 2 {
            canvas = canvas
                .text(parts[0], text_x, y)
                .size(font_size)
                .color(C::from_rgb8(user_r, user_g, user_b))
                .done();
            let user_advance = parts[0].len() as f32 * font_size * 0.55;
            canvas = canvas
                .text("@", text_x + user_advance, y)
                .size(font_size)
                .color(C::from_rgb8(at_r, at_g, at_b))
                .done();
            let at_advance = font_size * 0.55;
            canvas = canvas
                .text(parts[1], text_x + user_advance + at_advance, y)
                .size(font_size)
                .color(C::from_rgb8(host_r, host_g, host_b))
                .done();
        } else {
            canvas = canvas
                .text(&self.info.user_at_host, text_x, y)
                .size(font_size)
                .color(C::from_rgb8(user_r, user_g, user_b))
                .done();
        }
        y += line_h;

        // Separator line
        let (sep_r, sep_g, sep_b) = parse_hex_rgb(&self.theme.separator);
        let sep_len = self.info.user_at_host.len() as f32 * font_size * 0.55;
        canvas = canvas
            .line(text_x, y, text_x + sep_len, y)
            .color(C::from_rgb8(sep_r, sep_g, sep_b).with_alpha(0.6))
            .width(1.0)
            .done();
        y += line_h;

        // Info fields with staggered fade-in
        let (val_r, val_g, val_b) = parse_hex_rgb(&self.theme.value);
        let (icon_r, icon_g, icon_b) = parse_hex_rgb(&self.theme.icon);
        let label_size = font_size * 0.9;

        for (i, field_id) in self.fields.iter().enumerate() {
            let alpha = ((t - i as f32 * 0.08) / 0.2).clamp(0.0, 1.0);
            if alpha <= 0.0 {
                y += line_h;
                continue;
            }

            let meta = builtin_field(field_id);
            let value = self.info.field_value(field_id);
            let (lr, lg, lb) = meta.color;

            // Bar indicator
            canvas = canvas
                .line(text_x, y - line_h * 0.15, text_x, y + line_h * 0.35)
                .color(C::from_rgb8(sep_r, sep_g, sep_b).with_alpha(0.4 * alpha))
                .width(2.0)
                .done();

            // Icon
            canvas = canvas
                .text(meta.icon, text_x + 8.0, y)
                .size(label_size)
                .color(C::from_rgb8(icon_r, icon_g, icon_b).with_alpha(alpha))
                .done();

            // Label
            let label_x = text_x + 8.0 + label_size * 2.0;
            canvas = canvas
                .text(meta.label, label_x, y)
                .size(label_size)
                .color(C::from_rgb8(lr, lg, lb).with_alpha(alpha))
                .done();

            // Value
            let value_x = label_x + label_size * 6.0;
            canvas = canvas
                .text(value, value_x, y)
                .size(label_size)
                .color(C::from_rgb8(val_r, val_g, val_b).with_alpha(alpha))
                .done();

            y += line_h;
        }

        canvas
    }
}

/// Parse a `#RRGGBB` hex string into (r, g, b). Returns (128, 128, 128) on error.
fn parse_hex_rgb(s: &str) -> (u8, u8, u8) {
    crate::config::parse_hex_color(s).unwrap_or((128, 128, 128))
}
