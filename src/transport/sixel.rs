// SPDX-License-Identifier: MIT OR Apache-2.0
//! Sixel graphics protocol backend.
//!
//! Implements the [DEC Sixel](https://en.wikipedia.org/wiki/Sixel) protocol for
//! transmitting pixel-perfect images to supported terminal emulators such as
//! xterm, foot, mlterm, `WezTerm`, and others.
//!
//! # How Sixel Works
//!
//! Sixel encodes 6 vertical pixels per character row using printable ASCII.
//! Each column of 6 pixels is encoded as a single character whose value is
//! `0x3F` (63, `?`) plus a 6-bit bitmap. Colors are assigned to palette
//! registers (0–255), and each row of sixels is emitted per-color.
//!
//! # Color Quantization
//!
//! Since Sixel supports a maximum of 256 palette colors, input RGBA images
//! must be quantized. This implementation uses a median-cut algorithm for
//! fast, quality quantization.
//!
//! # Feature Gate
//!
//! This module is available when the `sixel` feature is enabled.

use std::io::{self, Write};

use tiny_skia::Pixmap;

use crate::transport::backend::{ImageHandle, ProtocolBackend, ProtocolKind, TerminalPosition};
use crate::PixelCanvasError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// DCS (Device Control String) introducer for Sixel.
const DCS_START: &[u8] = b"\x1bP";
/// ST (String Terminator).
const DCS_END: &[u8] = b"\x1b\\";
/// Maximum palette colors in Sixel.
const MAX_COLORS: usize = 256;

// ---------------------------------------------------------------------------
// SixelBackend
// ---------------------------------------------------------------------------

/// Sixel graphics protocol backend.
///
/// Encodes pixel data as Sixel escape sequences written to stdout (or a
/// custom writer for testing).
#[derive(Debug)]
pub struct SixelBackend<W: Write + std::fmt::Debug = io::Stdout> {
    writer: W,
    next_id: u32,
    /// Reusable encode buffer to avoid per-frame allocation.
    encode_buf: Vec<u8>,
}

impl SixelBackend<io::Stdout> {
    /// Create a new Sixel backend writing to stdout.
    #[must_use]
    pub fn new() -> Self {
        Self {
            writer: io::stdout(),
            next_id: 1,
            encode_buf: Vec::with_capacity(64 * 1024),
        }
    }
}

impl Default for SixelBackend<io::Stdout> {
    fn default() -> Self {
        Self::new()
    }
}

impl<W: Write + std::fmt::Debug> SixelBackend<W> {
    /// Create a backend with a custom writer (useful for testing).
    pub fn with_writer(writer: W) -> Self {
        Self {
            writer,
            next_id: 1,
            encode_buf: Vec::with_capacity(64 * 1024),
        }
    }

    /// Consume the backend and return the writer.
    pub fn into_writer(self) -> W {
        self.writer
    }

    /// Encode a pixmap into Sixel format and write it.
    fn write_sixel(
        &mut self,
        pixmap: &Pixmap,
        position: TerminalPosition,
    ) -> Result<(), PixelCanvasError> {
        let w = pixmap.width() as usize;
        let h = pixmap.height() as usize;
        if w == 0 || h == 0 {
            return Ok(());
        }

        let data = pixmap.data();

        // Step 1: Quantize colors
        let (palette, indexed) = median_cut_quantize(data, w, h);

        // Step 2: Build Sixel data into reusable buffer
        self.encode_buf.clear();

        // Move cursor to position
        if position.col > 0 || position.row > 0 {
            write!(
                self.encode_buf,
                "\x1b[{};{}H",
                position.row + 1,
                position.col + 1
            )
            .expect("Vec<u8> write cannot fail");
        }

        // DCS introducer: P<p1>;<p2>;<p3>;q
        // p1=0: pixel aspect ratio 2:1 (default)
        // p2=1: transparent background
        // p3=0: horizontal grid size
        self.encode_buf.extend_from_slice(DCS_START);
        self.encode_buf.extend_from_slice(b"0;1;0q");

        // Set raster attributes: "pixel-width;pixel-height
        write!(self.encode_buf, "\"1;1;{w};{h}").expect("Vec<u8> write cannot fail");

        // Define palette registers
        for (i, &(r, g, b)) in palette.iter().enumerate() {
            // #register;2;R;G;B — RGB values are 0–100 percentage
            let rp = u32::from(r) * 100 / 255;
            let gp = u32::from(g) * 100 / 255;
            let bp = u32::from(b) * 100 / 255;
            write!(self.encode_buf, "#{i};2;{rp};{gp};{bp}").expect("Vec<u8> write cannot fail");
        }

        // Encode sixel rows (6 pixel rows per sixel row)
        let sixel_rows = h.div_ceil(6);
        // Reusable buffer for sixel line data (avoids per-color-per-row allocation)
        let mut sixel_line: Vec<u8> = Vec::with_capacity(w);

        for sr in 0..sixel_rows {
            let y_base = sr * 6;

            // For each color in palette, emit the sixel data for this row
            for (color_idx, _) in palette.iter().enumerate() {
                // Build the sixel string for this color into the reusable buffer
                let mut has_pixels = false;
                sixel_line.clear();

                for x in 0..w {
                    let mut bits: u8 = 0;
                    for bit in 0..6 {
                        let y = y_base + bit;
                        if y < h && indexed[y * w + x] == u8::try_from(color_idx).unwrap_or(0) {
                            bits |= 1 << bit;
                        }
                    }
                    if bits != 0 {
                        has_pixels = true;
                    }
                    sixel_line.push(0x3F + bits); // Sixel character
                }

                if !has_pixels {
                    continue; // Skip colors not present in this sixel row
                }

                // Apply run-length encoding
                write!(self.encode_buf, "#{color_idx}").expect("Vec<u8> write cannot fail");
                rle_encode(&sixel_line, &mut self.encode_buf);

                // '$' = carriage return (stay on same sixel row)
                self.encode_buf.push(b'$');
            }

            // '-' = newline (advance to next sixel row)
            if sr < sixel_rows - 1 {
                self.encode_buf.push(b'-');
            }
        }

        // String Terminator
        self.encode_buf.extend_from_slice(DCS_END);

        // Write everything at once
        self.writer.write_all(&self.encode_buf)?;
        self.writer.flush()?;

        Ok(())
    }
}

impl<W: Write + std::fmt::Debug + Send> ProtocolBackend for SixelBackend<W> {
    fn transmit(
        &mut self,
        pixmap: &Pixmap,
        position: TerminalPosition,
        _z_index: i32,
    ) -> Result<ImageHandle, PixelCanvasError> {
        self.write_sixel(pixmap, position)?;
        let id = self.next_id;
        self.next_id += 1;
        Ok(ImageHandle {
            id,
            protocol: ProtocolKind::Sixel,
        })
    }

    fn remove(&mut self, _handle: &ImageHandle) -> Result<(), PixelCanvasError> {
        // Sixel doesn't support persistent image removal.
        // Images are replaced by re-drawing at the same position.
        Ok(())
    }

    fn clear_all(&mut self) -> Result<(), PixelCanvasError> {
        // No-op: Sixel images are not retained.
        Ok(())
    }

    fn supports_alpha(&self) -> bool {
        // Sixel supports "transparent" via the background parameter,
        // but not true per-pixel alpha blending.
        false
    }

    fn protocol_kind(&self) -> ProtocolKind {
        ProtocolKind::Sixel
    }
}

// ---------------------------------------------------------------------------
// Run-Length Encoding
// ---------------------------------------------------------------------------

/// Compress consecutive identical sixel characters using RLE.
///
/// Sixel RLE: `!<count><char>` repeats `<char>` `<count>` times.
fn rle_encode(data: &[u8], out: &mut Vec<u8>) {
    if data.is_empty() {
        return;
    }

    // Trim trailing '?' (0x3F = all-zero bits) — they add no visual data
    let len = data.iter().rposition(|&b| b != 0x3F).map_or(0, |i| i + 1);

    if len == 0 {
        return;
    }

    let mut i = 0;
    while i < len {
        let ch = data[i];
        let mut count = 1usize;
        while i + count < len && data[i + count] == ch {
            count += 1;
        }
        if count >= 4 {
            // Use RLE: !<count><char>
            let _ = write!(out, "!{count}");
            out.push(ch);
        } else {
            for _ in 0..count {
                out.push(ch);
            }
        }
        i += count;
    }
}

// ---------------------------------------------------------------------------
// Median-Cut Color Quantization
// ---------------------------------------------------------------------------

/// A color bucket for the median-cut algorithm.
#[derive(Clone, Debug)]
struct ColorBucket {
    /// Pixel indices in this bucket.
    pixels: Vec<(u8, u8, u8)>,
}

impl ColorBucket {
    /// Find the color channel with the widest range.
    fn widest_channel(&self) -> usize {
        let (mut r_min, mut g_min, mut b_min) = (255u8, 255u8, 255u8);
        let (mut r_max, mut g_max, mut b_max) = (0u8, 0u8, 0u8);
        for &(r, g, b) in &self.pixels {
            r_min = r_min.min(r);
            r_max = r_max.max(r);
            g_min = g_min.min(g);
            g_max = g_max.max(g);
            b_min = b_min.min(b);
            b_max = b_max.max(b);
        }
        let r_range = r_max - r_min;
        let g_range = g_max - g_min;
        let b_range = b_max - b_min;
        if r_range >= g_range && r_range >= b_range {
            0
        } else if g_range >= b_range {
            1
        } else {
            2
        }
    }

    /// Split this bucket at the median of the widest channel.
    fn split(mut self) -> (Self, Self) {
        let ch = self.widest_channel();
        self.pixels.sort_unstable_by_key(|&(r, g, b)| match ch {
            0 => r,
            1 => g,
            _ => b,
        });
        let mid = self.pixels.len() / 2;
        let hi = self.pixels.split_off(mid);
        (self, Self { pixels: hi })
    }

    /// Average color of all pixels in this bucket.
    fn average(&self) -> (u8, u8, u8) {
        if self.pixels.is_empty() {
            return (0, 0, 0);
        }
        let (mut sr, mut sg, mut sb) = (0u64, 0u64, 0u64);
        for &(r, g, b) in &self.pixels {
            sr += u64::from(r);
            sg += u64::from(g);
            sb += u64::from(b);
        }
        let n = self.pixels.len() as u64;
        #[allow(clippy::cast_possible_truncation)]
        ((sr / n) as u8, (sg / n) as u8, (sb / n) as u8)
    }
}

/// Quantize RGBA pixels to at most 256 palette colors using median-cut.
///
/// Returns `(palette, indexed)` where `palette` is the color table and
/// `indexed` is a flat array of palette indices (row-major, same layout as
/// the input RGBA data).
#[allow(clippy::many_single_char_names)]
fn median_cut_quantize(data: &[u8], w: usize, h: usize) -> (Vec<(u8, u8, u8)>, Vec<u8>) {
    let total = w * h;

    // Extract RGB pixels, compositing alpha against black
    let mut pixels: Vec<(u8, u8, u8)> = Vec::with_capacity(total);
    for i in 0..total {
        let idx = i * 4;
        let (r, g, b, a) = (data[idx], data[idx + 1], data[idx + 2], data[idx + 3]);
        if a == 255 {
            pixels.push((r, g, b));
        } else if a == 0 {
            pixels.push((0, 0, 0));
        } else {
            let af = f32::from(a) / 255.0;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let composited = (
                (f32::from(r) * af) as u8,
                (f32::from(g) * af) as u8,
                (f32::from(b) * af) as u8,
            );
            pixels.push(composited);
        }
    }

    // Build initial bucket — move pixels, don't clone
    let initial = ColorBucket { pixels };
    let mut buckets = vec![initial];

    // Iteratively split the largest bucket until we have enough colors
    let target = MAX_COLORS.min(total);
    while buckets.len() < target {
        // Find bucket with most pixels
        let (max_idx, _) = buckets
            .iter()
            .enumerate()
            .max_by_key(|(_, b)| b.pixels.len())
            .unwrap();

        let bucket = buckets.swap_remove(max_idx);
        if bucket.pixels.len() <= 1 {
            buckets.push(bucket);
            break;
        }

        let (lo, hi) = bucket.split();
        if !lo.pixels.is_empty() {
            buckets.push(lo);
        }
        if !hi.pixels.is_empty() {
            buckets.push(hi);
        }
    }

    // Build palette
    let palette: Vec<(u8, u8, u8)> = buckets.iter().map(ColorBucket::average).collect();

    // Build a 32×32×32 RGB lookup table (5 bits per channel = 32 KB)
    // for O(1) nearest-color lookup instead of O(n × palette_size).
    let mut lut = vec![0u8; 32 * 32 * 32];
    for ri in 0u8..32 {
        for gi in 0u8..32 {
            for bi in 0u8..32 {
                // Map 5-bit back to 8-bit (center of the bin)
                let r = (u16::from(ri) * 255 / 31) as u8;
                let g = (u16::from(gi) * 255 / 31) as u8;
                let b = (u16::from(bi) * 255 / 31) as u8;
                lut[(ri as usize) * 32 * 32 + (gi as usize) * 32 + bi as usize] =
                    nearest_palette_index(&palette, r, g, b);
            }
        }
    }

    // Re-extract pixels from the original data for indexed mapping
    // (the original pixels vec was moved into buckets and mutated by sorting)
    let indexed: Vec<u8> = (0..total)
        .map(|i| {
            let idx = i * 4;
            let (r, g, b, a) = (data[idx], data[idx + 1], data[idx + 2], data[idx + 3]);
            let (r, g, b) = if a == 255 {
                (r, g, b)
            } else if a == 0 {
                (0, 0, 0)
            } else {
                let af = f32::from(a) / 255.0;
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                {
                    (
                        (f32::from(r) * af) as u8,
                        (f32::from(g) * af) as u8,
                        (f32::from(b) * af) as u8,
                    )
                }
            };
            // 5-bit quantized lookup
            let ri = (r >> 3) as usize;
            let gi = (g >> 3) as usize;
            let bi = (b >> 3) as usize;
            lut[ri * 32 * 32 + gi * 32 + bi]
        })
        .collect();

    (palette, indexed)
}

/// Find the nearest palette entry by squared Euclidean distance in RGB.
/// Used only during LUT construction (called 32³ = 32768 times, not per-pixel).
#[allow(clippy::cast_possible_truncation)]
fn nearest_palette_index(palette: &[(u8, u8, u8)], r: u8, g: u8, b: u8) -> u8 {
    let mut best = 0u8;
    let mut best_dist = u32::MAX;
    for (i, &(pr, pg, pb)) in palette.iter().enumerate() {
        let dr = i32::from(r) - i32::from(pr);
        let dg = i32::from(g) - i32::from(pg);
        let db = i32::from(b) - i32::from(pb);
        #[allow(clippy::cast_sign_loss)]
        let dist = (dr * dr + dg * dg + db * db) as u32;
        if dist < best_dist {
            best_dist = dist;
            best = i as u8;
            if dist == 0 {
                break;
            }
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a pixmap with known RGBA data.
    fn make_pixmap(width: u32, height: u32, data: &[(u8, u8, u8, u8)]) -> Pixmap {
        let mut pm = Pixmap::new(width, height).unwrap();
        let buf = pm.data_mut();
        for (i, &(r, g, b, a)) in data.iter().enumerate() {
            buf[i * 4] = r;
            buf[i * 4 + 1] = g;
            buf[i * 4 + 2] = b;
            buf[i * 4 + 3] = a;
        }
        pm
    }

    #[test]
    fn sixel_basic_output_structure() {
        // 2×2 pixmap, solid red
        let pm = make_pixmap(
            2,
            2,
            &[
                (255, 0, 0, 255),
                (255, 0, 0, 255),
                (255, 0, 0, 255),
                (255, 0, 0, 255),
            ],
        );

        let writer = io::Cursor::new(Vec::new());
        let mut backend = SixelBackend::with_writer(writer);
        let pos = TerminalPosition::new(0, 0, 2, 1);
        let handle = backend.transmit(&pm, pos, 0).unwrap();

        let output = backend.into_writer().into_inner();
        let output_str = String::from_utf8_lossy(&output);

        // Should contain DCS start
        assert!(output_str.contains("\x1bP"), "should contain DCS start");
        // Should contain ST end
        assert!(output_str.contains("\x1b\\"), "should contain ST end");
        // Should contain a color register definition (#N;2;R;G;B)
        assert!(output_str.contains(";2;"), "should contain color register");
        // Handle should be Sixel protocol
        assert_eq!(handle.protocol(), ProtocolKind::Sixel);
    }

    #[test]
    fn sixel_color_registers() {
        // 2×2 pixmap with 2 distinct colors
        let pm = make_pixmap(
            2,
            2,
            &[
                (255, 0, 0, 255),
                (0, 255, 0, 255),
                (255, 0, 0, 255),
                (0, 255, 0, 255),
            ],
        );

        let writer = io::Cursor::new(Vec::new());
        let mut backend = SixelBackend::with_writer(writer);
        let pos = TerminalPosition::new(0, 0, 2, 1);
        backend.transmit(&pm, pos, 0).unwrap();

        let output = backend.into_writer().into_inner();
        let output_str = String::from_utf8_lossy(&output);

        // Should have at least 2 color register definitions
        let register_count = output_str.matches(";2;").count();
        assert!(
            register_count >= 2,
            "need at least 2 color registers, got {register_count}"
        );
    }

    #[test]
    fn sixel_protocol_kind() {
        let backend = SixelBackend::with_writer(io::Cursor::new(Vec::new()));
        assert_eq!(backend.protocol_kind(), ProtocolKind::Sixel);
        assert!(!backend.supports_alpha());
    }

    #[test]
    fn sixel_remove_is_noop() {
        let mut backend = SixelBackend::with_writer(io::Cursor::new(Vec::new()));
        let pm = make_pixmap(
            2,
            2,
            &[
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
            ],
        );
        let pos = TerminalPosition::new(0, 0, 2, 1);
        let handle = backend.transmit(&pm, pos, 0).unwrap();
        assert!(backend.remove(&handle).is_ok());
        assert!(backend.clear_all().is_ok());
    }

    #[test]
    fn rle_compresses_runs() {
        let data = vec![0x40, 0x40, 0x40, 0x40, 0x40, 0x41]; // 5×'@' + 'A'
        let mut out = Vec::new();
        rle_encode(&data, &mut out);
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("!5@"), "RLE should compress run of 5: got {s}");
        assert!(s.contains('A'), "should contain trailing A");
    }

    #[test]
    fn rle_no_compression_for_short_runs() {
        let data = vec![0x40, 0x41, 0x42];
        let mut out = Vec::new();
        rle_encode(&data, &mut out);
        assert_eq!(&out, &[0x40, 0x41, 0x42]);
    }

    #[test]
    fn median_cut_reduces_to_max_colors() {
        // Create a pixmap with more than 256 distinct colors
        let mut data = vec![(0u8, 0u8, 0u8, 255u8); 512];
        for (i, px) in data.iter_mut().enumerate() {
            let r = (i * 7 % 256) as u8;
            let g = (i * 13 % 256) as u8;
            let b = (i * 23 % 256) as u8;
            *px = (r, g, b, 255);
        }
        let pm = make_pixmap(512, 1, &data);
        let (palette, indexed) = median_cut_quantize(pm.data(), 512, 1);
        assert!(
            palette.len() <= MAX_COLORS,
            "palette should be ≤256, got {}",
            palette.len()
        );
        assert_eq!(indexed.len(), 512);
    }

    #[test]
    fn full_pipeline_sixel() {
        use crate::rasterize::Rasterizer;
        use crate::scene::{Color, PixelCanvas};

        let canvas = PixelCanvas::new(50, 50)
            .background(Color::BLUE)
            .circle(25.0, 25.0, 15.0)
            .fill(Color::RED)
            .done();
        let pixmap = Rasterizer::rasterize(&canvas).unwrap();

        let writer = io::Cursor::new(Vec::new());
        let mut backend = SixelBackend::with_writer(writer);
        let pos = TerminalPosition::new(0, 0, 5, 5);
        let handle = backend.transmit(&pixmap, pos, 0).unwrap();
        assert_eq!(handle.protocol(), ProtocolKind::Sixel);

        let output = backend.into_writer().into_inner();
        assert!(!output.is_empty(), "sixel output should not be empty");
        assert!(output.starts_with(b"\x1bP"), "should start with DCS");
    }
}
