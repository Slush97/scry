// SPDX-License-Identifier: MIT OR Apache-2.0
//! Pixel-based stats overlay for the SDF renderer.
//!
//! Provides an embedded 8×8 bitmap font and a `StatsOverlay` that renders
//! FPS counters, per-stage profiler bars, and a frame-time sparkline graph
//! directly onto a `Pixmap` — no terminal escape sequences needed.

use std::collections::VecDeque;
use std::time::Instant;

use tiny_skia::Pixmap;

use super::profiler::{SdfStage, SmoothedSdfProfile};

// ---------------------------------------------------------------------------
// Embedded 8×8 bitmap font (ASCII 0x20–0x7E)
// ---------------------------------------------------------------------------

/// 8×8 monospace bitmap font covering printable ASCII (space through tilde).
/// Each glyph is 8 bytes, one per row, MSB = leftmost pixel.
const FONT_8X8: [[u8; 8]; 95] = [
    // 0x20 ' '
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    // 0x21 '!'
    [0x18, 0x18, 0x18, 0x18, 0x18, 0x00, 0x18, 0x00],
    // 0x22 '"'
    [0x6C, 0x6C, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00],
    // 0x23 '#'
    [0x24, 0x7E, 0x24, 0x24, 0x7E, 0x24, 0x00, 0x00],
    // 0x24 '$'
    [0x18, 0x3E, 0x58, 0x3C, 0x1A, 0x7C, 0x18, 0x00],
    // 0x25 '%'
    [0x62, 0x64, 0x08, 0x10, 0x26, 0x46, 0x00, 0x00],
    // 0x26 '&'
    [0x30, 0x48, 0x30, 0x56, 0x48, 0x34, 0x00, 0x00],
    // 0x27 '''
    [0x18, 0x18, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00],
    // 0x28 '('
    [0x08, 0x10, 0x20, 0x20, 0x20, 0x10, 0x08, 0x00],
    // 0x29 ')'
    [0x20, 0x10, 0x08, 0x08, 0x08, 0x10, 0x20, 0x00],
    // 0x2A '*'
    [0x00, 0x24, 0x18, 0x7E, 0x18, 0x24, 0x00, 0x00],
    // 0x2B '+'
    [0x00, 0x18, 0x18, 0x7E, 0x18, 0x18, 0x00, 0x00],
    // 0x2C ','
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x10],
    // 0x2D '-'
    [0x00, 0x00, 0x00, 0x7E, 0x00, 0x00, 0x00, 0x00],
    // 0x2E '.'
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00],
    // 0x2F '/'
    [0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x00, 0x00],
    // 0x30 '0'
    [0x3C, 0x46, 0x4A, 0x52, 0x62, 0x3C, 0x00, 0x00],
    // 0x31 '1'
    [0x18, 0x38, 0x18, 0x18, 0x18, 0x3C, 0x00, 0x00],
    // 0x32 '2'
    [0x3C, 0x42, 0x04, 0x18, 0x20, 0x7E, 0x00, 0x00],
    // 0x33 '3'
    [0x3C, 0x42, 0x0C, 0x02, 0x42, 0x3C, 0x00, 0x00],
    // 0x34 '4'
    [0x0C, 0x14, 0x24, 0x44, 0x7E, 0x04, 0x00, 0x00],
    // 0x35 '5'
    [0x7E, 0x40, 0x7C, 0x02, 0x42, 0x3C, 0x00, 0x00],
    // 0x36 '6'
    [0x1C, 0x20, 0x7C, 0x42, 0x42, 0x3C, 0x00, 0x00],
    // 0x37 '7'
    [0x7E, 0x02, 0x04, 0x08, 0x10, 0x10, 0x00, 0x00],
    // 0x38 '8'
    [0x3C, 0x42, 0x3C, 0x42, 0x42, 0x3C, 0x00, 0x00],
    // 0x39 '9'
    [0x3C, 0x42, 0x42, 0x3E, 0x04, 0x38, 0x00, 0x00],
    // 0x3A ':'
    [0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x00, 0x00],
    // 0x3B ';'
    [0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x10, 0x00],
    // 0x3C '<'
    [0x04, 0x08, 0x10, 0x20, 0x10, 0x08, 0x04, 0x00],
    // 0x3D '='
    [0x00, 0x00, 0x7E, 0x00, 0x7E, 0x00, 0x00, 0x00],
    // 0x3E '>'
    [0x20, 0x10, 0x08, 0x04, 0x08, 0x10, 0x20, 0x00],
    // 0x3F '?'
    [0x3C, 0x42, 0x04, 0x08, 0x00, 0x08, 0x00, 0x00],
    // 0x40 '@'
    [0x3C, 0x42, 0x5E, 0x56, 0x5E, 0x40, 0x3C, 0x00],
    // 0x41 'A'
    [0x18, 0x24, 0x42, 0x7E, 0x42, 0x42, 0x00, 0x00],
    // 0x42 'B'
    [0x7C, 0x42, 0x7C, 0x42, 0x42, 0x7C, 0x00, 0x00],
    // 0x43 'C'
    [0x3C, 0x42, 0x40, 0x40, 0x42, 0x3C, 0x00, 0x00],
    // 0x44 'D'
    [0x78, 0x44, 0x42, 0x42, 0x44, 0x78, 0x00, 0x00],
    // 0x45 'E'
    [0x7E, 0x40, 0x7C, 0x40, 0x40, 0x7E, 0x00, 0x00],
    // 0x46 'F'
    [0x7E, 0x40, 0x7C, 0x40, 0x40, 0x40, 0x00, 0x00],
    // 0x47 'G'
    [0x3C, 0x42, 0x40, 0x4E, 0x42, 0x3C, 0x00, 0x00],
    // 0x48 'H'
    [0x42, 0x42, 0x7E, 0x42, 0x42, 0x42, 0x00, 0x00],
    // 0x49 'I'
    [0x3C, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00, 0x00],
    // 0x4A 'J'
    [0x1E, 0x04, 0x04, 0x04, 0x44, 0x38, 0x00, 0x00],
    // 0x4B 'K'
    [0x44, 0x48, 0x70, 0x48, 0x44, 0x42, 0x00, 0x00],
    // 0x4C 'L'
    [0x40, 0x40, 0x40, 0x40, 0x40, 0x7E, 0x00, 0x00],
    // 0x4D 'M'
    [0x42, 0x66, 0x5A, 0x42, 0x42, 0x42, 0x00, 0x00],
    // 0x4E 'N'
    [0x42, 0x62, 0x52, 0x4A, 0x46, 0x42, 0x00, 0x00],
    // 0x4F 'O'
    [0x3C, 0x42, 0x42, 0x42, 0x42, 0x3C, 0x00, 0x00],
    // 0x50 'P'
    [0x7C, 0x42, 0x42, 0x7C, 0x40, 0x40, 0x00, 0x00],
    // 0x51 'Q'
    [0x3C, 0x42, 0x42, 0x4A, 0x44, 0x3A, 0x00, 0x00],
    // 0x52 'R'
    [0x7C, 0x42, 0x42, 0x7C, 0x44, 0x42, 0x00, 0x00],
    // 0x53 'S'
    [0x3C, 0x40, 0x3C, 0x02, 0x42, 0x3C, 0x00, 0x00],
    // 0x54 'T'
    [0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00, 0x00],
    // 0x55 'U'
    [0x42, 0x42, 0x42, 0x42, 0x42, 0x3C, 0x00, 0x00],
    // 0x56 'V'
    [0x42, 0x42, 0x42, 0x24, 0x24, 0x18, 0x00, 0x00],
    // 0x57 'W'
    [0x42, 0x42, 0x42, 0x5A, 0x66, 0x42, 0x00, 0x00],
    // 0x58 'X'
    [0x42, 0x24, 0x18, 0x18, 0x24, 0x42, 0x00, 0x00],
    // 0x59 'Y'
    [0x42, 0x42, 0x24, 0x18, 0x18, 0x18, 0x00, 0x00],
    // 0x5A 'Z'
    [0x7E, 0x04, 0x08, 0x10, 0x20, 0x7E, 0x00, 0x00],
    // 0x5B '['
    [0x3C, 0x20, 0x20, 0x20, 0x20, 0x3C, 0x00, 0x00],
    // 0x5C '\'
    [0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x00, 0x00],
    // 0x5D ']'
    [0x3C, 0x04, 0x04, 0x04, 0x04, 0x3C, 0x00, 0x00],
    // 0x5E '^'
    [0x10, 0x28, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00],
    // 0x5F '_'
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7E, 0x00],
    // 0x60 '`'
    [0x20, 0x10, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00],
    // 0x61 'a'
    [0x00, 0x00, 0x3C, 0x02, 0x3E, 0x42, 0x3E, 0x00],
    // 0x62 'b'
    [0x40, 0x40, 0x7C, 0x42, 0x42, 0x42, 0x7C, 0x00],
    // 0x63 'c'
    [0x00, 0x00, 0x3C, 0x40, 0x40, 0x40, 0x3C, 0x00],
    // 0x64 'd'
    [0x02, 0x02, 0x3E, 0x42, 0x42, 0x42, 0x3E, 0x00],
    // 0x65 'e'
    [0x00, 0x00, 0x3C, 0x42, 0x7E, 0x40, 0x3C, 0x00],
    // 0x66 'f'
    [0x0C, 0x10, 0x3C, 0x10, 0x10, 0x10, 0x10, 0x00],
    // 0x67 'g'
    [0x00, 0x00, 0x3E, 0x42, 0x42, 0x3E, 0x02, 0x3C],
    // 0x68 'h'
    [0x40, 0x40, 0x7C, 0x42, 0x42, 0x42, 0x42, 0x00],
    // 0x69 'i'
    [0x18, 0x00, 0x38, 0x18, 0x18, 0x18, 0x3C, 0x00],
    // 0x6A 'j'
    [0x08, 0x00, 0x18, 0x08, 0x08, 0x08, 0x48, 0x30],
    // 0x6B 'k'
    [0x40, 0x40, 0x44, 0x48, 0x70, 0x48, 0x44, 0x00],
    // 0x6C 'l'
    [0x38, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00],
    // 0x6D 'm'
    [0x00, 0x00, 0x64, 0x5A, 0x42, 0x42, 0x42, 0x00],
    // 0x6E 'n'
    [0x00, 0x00, 0x7C, 0x42, 0x42, 0x42, 0x42, 0x00],
    // 0x6F 'o'
    [0x00, 0x00, 0x3C, 0x42, 0x42, 0x42, 0x3C, 0x00],
    // 0x70 'p'
    [0x00, 0x00, 0x7C, 0x42, 0x42, 0x7C, 0x40, 0x40],
    // 0x71 'q'
    [0x00, 0x00, 0x3E, 0x42, 0x42, 0x3E, 0x02, 0x02],
    // 0x72 'r'
    [0x00, 0x00, 0x5C, 0x62, 0x40, 0x40, 0x40, 0x00],
    // 0x73 's'
    [0x00, 0x00, 0x3E, 0x40, 0x3C, 0x02, 0x7C, 0x00],
    // 0x74 't'
    [0x10, 0x10, 0x3C, 0x10, 0x10, 0x10, 0x0C, 0x00],
    // 0x75 'u'
    [0x00, 0x00, 0x42, 0x42, 0x42, 0x42, 0x3E, 0x00],
    // 0x76 'v'
    [0x00, 0x00, 0x42, 0x42, 0x24, 0x24, 0x18, 0x00],
    // 0x77 'w'
    [0x00, 0x00, 0x42, 0x42, 0x42, 0x5A, 0x24, 0x00],
    // 0x78 'x'
    [0x00, 0x00, 0x42, 0x24, 0x18, 0x24, 0x42, 0x00],
    // 0x79 'y'
    [0x00, 0x00, 0x42, 0x42, 0x42, 0x3E, 0x02, 0x3C],
    // 0x7A 'z'
    [0x00, 0x00, 0x7E, 0x04, 0x18, 0x20, 0x7E, 0x00],
    // 0x7B '{'
    [0x0C, 0x10, 0x10, 0x20, 0x10, 0x10, 0x0C, 0x00],
    // 0x7C '|'
    [0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x00],
    // 0x7D '}'
    [0x30, 0x08, 0x08, 0x04, 0x08, 0x08, 0x30, 0x00],
    // 0x7E '~'
    [0x00, 0x00, 0x32, 0x4C, 0x00, 0x00, 0x00, 0x00],
];

/// Draw a single character onto a pixmap at (px, py) in the given RGBA color.
fn draw_char(pixmap: &mut Pixmap, px: i32, py: i32, ch: char, r: u8, g: u8, b: u8, a: u8) {
    let idx = ch as u32;
    if !(0x20..=0x7E).contains(&idx) {
        return;
    }
    let glyph = &FONT_8X8[(idx - 0x20) as usize];
    let pw = pixmap.width() as i32;
    let ph = pixmap.height() as i32;
    let data = pixmap.data_mut();

    for row in 0..8 {
        let bits = glyph[row as usize];
        let y = py + row;
        if y < 0 || y >= ph {
            continue;
        }
        for col in 0..8 {
            if bits & (0x80 >> col) != 0 {
                let x = px + col;
                if x >= 0 && x < pw {
                    let offset = ((y * pw + x) * 4) as usize;
                    if a == 255 {
                        data[offset] = r;
                        data[offset + 1] = g;
                        data[offset + 2] = b;
                        data[offset + 3] = 255;
                    } else {
                        // Alpha blend
                        let sa = a as u32;
                        let da = 255 - sa;
                        data[offset] = ((r as u32 * sa + data[offset] as u32 * da) / 255) as u8;
                        data[offset + 1] =
                            ((g as u32 * sa + data[offset + 1] as u32 * da) / 255) as u8;
                        data[offset + 2] =
                            ((b as u32 * sa + data[offset + 2] as u32 * da) / 255) as u8;
                        data[offset + 3] = 255;
                    }
                }
            }
        }
    }
}

/// Draw a text string onto a pixmap. Returns the width drawn in pixels.
pub fn draw_text(
    pixmap: &mut Pixmap,
    x: i32,
    y: i32,
    text: &str,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
) -> i32 {
    let mut cx = x;
    for ch in text.chars() {
        draw_char(pixmap, cx, y, ch, r, g, b, a);
        cx += 8;
    }
    cx - x
}

/// Draw a filled rectangle onto a pixmap with alpha blending.
pub fn draw_rect(pixmap: &mut Pixmap, x: i32, y: i32, w: i32, h: i32, r: u8, g: u8, b: u8, a: u8) {
    let pw = pixmap.width() as i32;
    let ph = pixmap.height() as i32;
    let data = pixmap.data_mut();

    let x0 = x.max(0);
    let y0 = y.max(0);
    let x1 = (x + w).min(pw);
    let y1 = (y + h).min(ph);

    for py in y0..y1 {
        for px in x0..x1 {
            let offset = ((py * pw + px) * 4) as usize;
            if a == 255 {
                data[offset] = r;
                data[offset + 1] = g;
                data[offset + 2] = b;
                data[offset + 3] = 255;
            } else {
                let sa = a as u32;
                let da = 255 - sa;
                data[offset] = ((r as u32 * sa + data[offset] as u32 * da) / 255) as u8;
                data[offset + 1] = ((g as u32 * sa + data[offset + 1] as u32 * da) / 255) as u8;
                data[offset + 2] = ((b as u32 * sa + data[offset + 2] as u32 * da) / 255) as u8;
                data[offset + 3] = 255;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stage colors (RGB equivalents of the ANSI colors in profiler.rs)
// ---------------------------------------------------------------------------

/// RGB color for each SDF stage, matching the ANSI colors in `SdfStage::ansi_color`.
const fn stage_rgb(stage: SdfStage) -> (u8, u8, u8) {
    match stage {
        SdfStage::March => (80, 120, 220),      // blue
        SdfStage::Shadow => (200, 200, 200),    // white/gray
        SdfStage::Normal => (220, 180, 50),     // orange/yellow
        SdfStage::Shading => (240, 240, 80),    // bright yellow
        SdfStage::Reflection => (180, 80, 220), // purple
        SdfStage::Fire => (220, 60, 60),        // red
    }
}

// ---------------------------------------------------------------------------
// StatsOverlay
// ---------------------------------------------------------------------------

/// Frame time ring buffer and FPS smoother for the stats overlay.
pub struct StatsOverlay {
    frame_times: VecDeque<f32>,
    fps_smooth: f32,
    last_frame: Instant,
    capacity: usize,
}

impl StatsOverlay {
    /// Create a new stats overlay with the given history capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            frame_times: VecDeque::with_capacity(capacity),
            fps_smooth: 0.0,
            last_frame: Instant::now(),
            capacity,
        }
    }

    /// Record a new frame. Call once per frame before `render_overlay`.
    pub fn tick(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;

        if self.frame_times.len() >= self.capacity {
            self.frame_times.pop_front();
        }
        self.frame_times.push_back(dt);

        if dt > 0.0 {
            self.fps_smooth = self.fps_smooth * 0.9 + (1.0 / dt) * 0.1;
        }
    }

    /// Render the stats overlay onto a pixmap.
    ///
    /// Draws a semi-transparent panel in the top-left corner with:
    /// - FPS + frame time
    /// - Resolution and render scale
    /// - Per-stage colored bar
    /// - Frame time sparkline
    pub fn render_overlay(
        &self,
        pixmap: &mut Pixmap,
        profile: &SmoothedSdfProfile,
        render_scale_pct: u32,
        scene_name: &str,
    ) {
        let pad = 6;
        let line_h = 10;
        let panel_w = 300;
        let bar_h = 12;
        let sparkline_h = 30;
        let panel_h = pad + line_h * 3 + bar_h + pad + sparkline_h + pad;

        // Semi-transparent background
        draw_rect(pixmap, pad, pad, panel_w, panel_h, 0, 0, 0, 180);

        let text_x = pad + 4;
        let mut y = pad + 2;
        let white = (230, 230, 230, 255);

        // Line 1: scene + FPS
        let frame_ms = if self.fps_smooth > 0.0 {
            1000.0 / self.fps_smooth
        } else {
            0.0
        };
        let line1 = format!(
            "{} | {:.0} fps | {:.1}ms",
            scene_name, self.fps_smooth, frame_ms
        );
        draw_text(
            pixmap, text_x, y, &line1, white.0, white.1, white.2, white.3,
        );
        y += line_h;

        // Line 2: resolution + scale + total render time
        let total_ms = profile.total_us as f64 / 1000.0;
        let line2 = format!(
            "{}x{} @{}% | {:.1}ms render",
            pixmap.width(),
            pixmap.height(),
            render_scale_pct,
            total_ms
        );
        draw_text(
            pixmap, text_x, y, &line2, white.0, white.1, white.2, white.3,
        );
        y += line_h;

        // Line 3: per-stage labels
        let active: Vec<(SdfStage, u64)> = SdfStage::ALL
            .iter()
            .filter(|s| profile.stage_us[s.index()] > 0)
            .map(|s| (*s, profile.stage_us[s.index()]))
            .collect();

        let mut lx = text_x;
        for (stage, us) in &active {
            let (sr, sg, sb) = stage_rgb(*stage);
            let ms = *us as f64 / 1000.0;
            let label = format!("{} {:.1} ", stage.label(), ms);
            draw_text(pixmap, lx, y, &label, sr, sg, sb, 255);
            lx += label.len() as i32 * 8;
        }
        y += line_h;

        // Per-stage colored bar
        let stage_total: u64 = profile.stage_us.iter().sum();
        let bar_w = panel_w - 8;
        if stage_total > 0 {
            let mut bx = text_x;
            for (i, (stage, us)) in active.iter().enumerate() {
                let frac = *us as f64 / stage_total as f64;
                let seg_w = if i == active.len() - 1 {
                    (text_x + bar_w) - bx
                } else {
                    (frac * bar_w as f64).round() as i32
                };
                if seg_w > 0 {
                    let (sr, sg, sb) = stage_rgb(*stage);
                    draw_rect(pixmap, bx, y, seg_w, bar_h, sr, sg, sb, 220);
                    bx += seg_w;
                }
            }
        }
        y += bar_h + pad;

        // Frame time sparkline
        if self.frame_times.len() >= 2 {
            let sparkline_w = panel_w - 8;
            let n = self.frame_times.len();

            // Find min/max for scaling
            let mut min_t = f32::MAX;
            let mut max_t = f32::MIN;
            for &t in &self.frame_times {
                if t < min_t {
                    min_t = t;
                }
                if t > max_t {
                    max_t = t;
                }
            }
            let range = (max_t - min_t).max(0.001);

            // Draw sparkline background
            draw_rect(pixmap, text_x, y, sparkline_w, sparkline_h, 20, 20, 30, 200);

            // Draw 16ms target line (60fps)
            let target_y = y + sparkline_h
                - ((0.016 - min_t) / range * sparkline_h as f32)
                    .clamp(0.0, sparkline_h as f32 - 1.0) as i32;
            for dx in 0..sparkline_w {
                if dx % 4 < 2 {
                    let offset = ((target_y * pixmap.width() as i32 + text_x + dx) * 4) as usize;
                    let data = pixmap.data_mut();
                    if offset + 3 < data.len() {
                        data[offset] = 80;
                        data[offset + 1] = 180;
                        data[offset + 2] = 80;
                        data[offset + 3] = 255;
                    }
                }
            }

            // Draw bars
            let bar_spacing = if n > sparkline_w as usize {
                1
            } else {
                (sparkline_w as usize / n).max(1)
            };
            let skip = n.saturating_sub(sparkline_w as usize);

            for (i, &t) in self.frame_times.iter().skip(skip).enumerate() {
                let frac = ((t - min_t) / range).clamp(0.0, 1.0);
                let h = (frac * (sparkline_h - 2) as f32) as i32 + 1;
                let bx = text_x + (i * bar_spacing) as i32;
                if bx >= text_x + sparkline_w {
                    break;
                }

                // Color: green < 16ms, yellow < 33ms, red > 33ms
                let (cr, cg, cb) = if t < 0.016 {
                    (80, 200, 80)
                } else if t < 0.033 {
                    (220, 200, 60)
                } else {
                    (220, 60, 60)
                };

                let col_w = (bar_spacing as i32 - 1).max(1);
                draw_rect(pixmap, bx, y + sparkline_h - h, col_w, h, cr, cg, cb, 200);
            }
        }
    }
}
