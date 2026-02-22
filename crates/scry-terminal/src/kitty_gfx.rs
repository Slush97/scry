// SPDX-License-Identifier: MIT OR Apache-2.0
//! Kitty graphics protocol parser.
//!
//! Implements the core subset of the [Kitty graphics protocol][spec] needed
//! for inline image display (`kitty icat`, `timg`, `chafa`, etc.).
//!
//! ## Supported subset
//!
//! - **Action:** `a=T` (transmit + display), `a=t` (transmit), `a=p` (display)
//! - **Transmission:** `t=d` (direct, base64-encoded data)
//! - **Format:** `f=24` (RGB), `f=32` (RGBA), `f=100` (PNG)
//! - **Chunking:** `m=0` (final chunk) / `m=1` (more data follows)
//! - **Compression:** `o=z` (zlib-compressed payload)
//!
//! ## Wire format
//!
//! ```text
//! ESC _ G <key>=<value>,<key>=<value>,...;<base64 payload> ESC \
//! ```
//!
//! The `ESC _` (APC) introducer and `ESC \` (ST) terminator are stripped
//! by the caller before handing the payload to [`KittyGfxState::feed`].
//!
//! [spec]: https://sw.kovidgoyal.net/kitty/graphics-protocol/

use crate::inline_image::InlineImage;
use data_encoding::BASE64;

/// Maximum APC payload size (64 MB). Prevents OOM from a malicious
/// PTY program sending an unterminated `ESC _` sequence.
const MAX_APC_PAYLOAD: usize = 64 * 1024 * 1024;

/// Maximum accumulated base64 buffer for Kitty graphics (96 MB).
/// At a 4:3 base64 expansion ratio this allows ~72 MB decoded images.
const MAX_B64_BUFFER: usize = 96 * 1024 * 1024;

/// Parsed action from the Kitty graphics protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GfxAction {
    /// Transmit image data (buffer it).
    Transmit,
    /// Transmit and display immediately.
    TransmitAndDisplay,
    /// Display a previously transmitted image.
    Display,
}

/// Image data format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GfxFormat {
    /// 24-bit RGB (3 bytes per pixel).
    Rgb,
    /// 32-bit RGBA (4 bytes per pixel).
    Rgba,
    /// PNG-compressed image.
    Png,
}

/// Parsed parameters from `key=value` pairs.
#[derive(Clone, Debug)]
struct GfxParams {
    action: GfxAction,
    format: GfxFormat,
    /// Image width in pixels (for raw RGB/RGBA).
    width: u32,
    /// Image height in pixels (for raw RGB/RGBA).
    height: u32,
    /// Whether the payload is zlib-compressed before base64.
    compressed: bool,
    /// `m=1` means more chunks follow; `m=0` means final.
    more_chunks: bool,
    /// Image ID (optional, for later display).
    image_id: u32,
    /// Number of rows to occupy (0 = auto).
    rows: u16,
    /// Number of columns to occupy (0 = auto).
    cols: u16,
}

impl Default for GfxParams {
    fn default() -> Self {
        Self {
            action: GfxAction::TransmitAndDisplay,
            format: GfxFormat::Rgba,
            width: 0,
            height: 0,
            compressed: false,
            more_chunks: false,
            image_id: 0,
            rows: 0,
            cols: 0,
        }
    }
}

/// State machine for accumulating chunked Kitty graphics data.
pub struct KittyGfxState {
    /// Accumulated raw base64 data across chunks (decoded only when complete).
    b64_buffer: Vec<u8>,
    /// Parameters from the first chunk.
    params: Option<GfxParams>,
    /// Whether we're mid-sequence (accumulating chunks).
    in_progress: bool,
}

/// Result of feeding data to the Kitty graphics state machine.
pub enum KittyGfxResult {
    /// No complete image yet (accumulating chunks or not a graphics command).
    Pending,
    /// A complete image is ready for display.
    Image(InlineImage),
    /// Parse or decode error (non-fatal — just skip this image).
    Error(String),
}

impl KittyGfxState {
    /// Create a new Kitty graphics state machine.
    pub fn new() -> Self {
        Self {
            b64_buffer: Vec::new(),
            params: None,
            in_progress: false,
        }
    }

    /// Feed an APC payload (everything between `ESC_G` and `ESC\`).
    ///
    /// The leading `G` has already been stripped. The input is the raw bytes
    /// of the `key=val,...;base64data` portion.
    ///
    /// Returns `KittyGfxResult::Image` when a complete image is ready.
    pub fn feed(&mut self, payload: &[u8]) -> KittyGfxResult {
        // Split on ';' — left side is key=value pairs, right side is base64 data
        let (params_str, data_b64) = match payload.iter().position(|&b| b == b';') {
            Some(pos) => (&payload[..pos], &payload[pos + 1..]),
            None => (payload, &[][..]),
        };

        // Parse key=value pairs
        let params_text = match std::str::from_utf8(params_str) {
            Ok(s) => s,
            Err(_) => return KittyGfxResult::Error("invalid UTF-8 in params".into()),
        };

        let parsed = parse_params(params_text);

        if !self.in_progress {
            // First chunk — store params
            self.params = Some(parsed.clone());
            self.b64_buffer.clear();
            self.in_progress = true;
        }

        // Append raw base64 data (don't decode yet — chunks may split mid-encoding)
        if !data_b64.is_empty() {
            // Strip whitespace but keep as base64
            for &b in data_b64 {
                if !b.is_ascii_whitespace() {
                    self.b64_buffer.push(b);
                }
            }
        }

        // Guard against unbounded buffer growth (OOM protection)
        if self.b64_buffer.len() > MAX_B64_BUFFER {
            self.reset();
            return KittyGfxResult::Error(format!(
                "Kitty graphics payload exceeds {} MB limit",
                MAX_B64_BUFFER / (1024 * 1024)
            ));
        }

        // Update more_chunks from the latest chunk's params
        if let Some(ref mut p) = self.params {
            p.more_chunks = parsed.more_chunks;
        }

        // If the latest chunk says more data follows, keep accumulating
        if parsed.more_chunks {
            return KittyGfxResult::Pending;
        }

        // Final chunk — decode the complete image
        self.in_progress = false;
        let Some(params) = self.params.take() else {
            self.reset();
            return KittyGfxResult::Error("no params for image".into());
        };
        let raw_b64 = std::mem::take(&mut self.b64_buffer);

        // Now decode the complete base64 payload
        let raw_data = match BASE64.decode(&raw_b64) {
            Ok(decoded) => decoded,
            Err(e) => return KittyGfxResult::Error(format!("base64 decode error: {e}")),
        };

        // Decompress if needed
        let pixel_data = if params.compressed {
            match decompress_zlib(&raw_data) {
                Ok(d) => d,
                Err(e) => return KittyGfxResult::Error(format!("zlib decompress error: {e}")),
            }
        } else {
            raw_data
        };

        // Build the image based on format
        match params.format {
            GfxFormat::Png => {
                match InlineImage::decode(&pixel_data) {
                    Ok(img) => KittyGfxResult::Image(img),
                    Err(e) => KittyGfxResult::Error(format!("PNG decode: {e}")),
                }
            }
            GfxFormat::Rgba => {
                if params.width == 0 || params.height == 0 {
                    return KittyGfxResult::Error("RGBA format requires s= and v= params".into());
                }
                match InlineImage::from_rgba(params.width, params.height, pixel_data) {
                    Ok(img) => KittyGfxResult::Image(img),
                    Err(e) => KittyGfxResult::Error(format!("RGBA: {e}")),
                }
            }
            GfxFormat::Rgb => {
                if params.width == 0 || params.height == 0 {
                    return KittyGfxResult::Error("RGB format requires s= and v= params".into());
                }
                // Convert RGB to RGBA
                let expected = (params.width as usize) * (params.height as usize) * 3;
                if pixel_data.len() != expected {
                    return KittyGfxResult::Error(format!(
                        "RGB data length {} != expected {expected}",
                        pixel_data.len()
                    ));
                }
                let mut rgba = Vec::with_capacity(pixel_data.len() / 3 * 4);
                for chunk in pixel_data.chunks_exact(3) {
                    rgba.extend_from_slice(chunk);
                    rgba.push(255); // Alpha = opaque
                }
                match InlineImage::from_rgba(params.width, params.height, rgba) {
                    Ok(img) => KittyGfxResult::Image(img),
                    Err(e) => KittyGfxResult::Error(format!("RGB→RGBA: {e}")),
                }
            }
        }
    }

    /// Reset the state machine (discard any partial data).
    pub fn reset(&mut self) {
        self.b64_buffer.clear();
        self.params = None;
        self.in_progress = false;
    }
}

/// Parse `key=value,key=value,...` parameter string.
fn parse_params(s: &str) -> GfxParams {
    let mut params = GfxParams::default();

    for pair in s.split(',') {
        let Some((key, val)) = pair.split_once('=') else {
            continue;
        };
        match key {
            "a" => {
                params.action = match val {
                    "t" => GfxAction::Transmit,
                    "T" | "d" => GfxAction::TransmitAndDisplay,
                    "p" => GfxAction::Display,
                    _ => GfxAction::TransmitAndDisplay,
                };
            }
            "f" => {
                params.format = match val {
                    "24" => GfxFormat::Rgb,
                    "32" => GfxFormat::Rgba,
                    "100" => GfxFormat::Png,
                    _ => GfxFormat::Rgba,
                };
            }
            "s" => {
                params.width = val.parse().unwrap_or(0);
            }
            "v" => {
                params.height = val.parse().unwrap_or(0);
            }
            "o" => {
                params.compressed = val == "z";
            }
            "m" => {
                params.more_chunks = val == "1";
            }
            "i" => {
                params.image_id = val.parse().unwrap_or(0);
            }
            "r" => {
                params.rows = val.parse().unwrap_or(0);
            }
            "c" => {
                params.cols = val.parse().unwrap_or(0);
            }
            _ => {} // Ignore unknown keys
        }
    }

    params
}

/// Decompress zlib-compressed data.
fn decompress_zlib(data: &[u8]) -> Result<Vec<u8>, String> {
    use flate2::read::ZlibDecoder;
    use std::io::Read;

    let mut decoder = ZlibDecoder::new(data);
    let mut output = Vec::new();
    decoder
        .read_to_end(&mut output)
        .map_err(|e| format!("zlib: {e}"))?;
    Ok(output)
}

/// APC (Application Program Command) accumulator.
///
/// Detects and buffers `ESC _ G...payload... ESC \` sequences from the raw
/// PTY byte stream. Non-APC data is passed through unmodified.
///
/// # Usage
///
/// ```ignore
/// let mut apc = ApcAccumulator::new();
/// for byte in pty_data {
///     match apc.feed(byte) {
///         ApcFeed::PassThrough(b) => vte_parser.advance(b),
///         ApcFeed::Consumed => {},       // byte absorbed into APC buffer
///         ApcFeed::Complete(payload) => { // full APC received
///             if payload.starts_with(b"G") {
///                 let result = kitty_gfx.feed(&payload[1..]);
///             }
///         }
///     }
/// }
/// ```
pub struct ApcAccumulator {
    state: ApcState,
    buffer: Vec<u8>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ApcState {
    /// Normal — pass bytes through.
    Normal,
    /// Saw `ESC` — waiting for `_` (APC) or something else.
    EscSeen,
    /// Inside APC — accumulating payload bytes.
    InApc,
    /// Inside APC, saw `ESC` — waiting for `\` (ST) to terminate.
    ApcEscSeen,
}

/// Result of feeding a byte to the APC accumulator.
pub enum ApcFeed {
    /// Byte should be passed to the VTE parser normally.
    PassThrough(u8),
    /// Byte was consumed by the APC accumulator (don't pass to VTE).
    Consumed,
    /// A complete APC payload is ready.
    Complete(Vec<u8>),
}

impl ApcAccumulator {
    /// Create a new APC accumulator.
    pub fn new() -> Self {
        Self {
            state: ApcState::Normal,
            buffer: Vec::with_capacity(4096),
        }
    }

    // NOTE: The single-byte `feed()` API was removed because it had an
    // acknowledged byte-loss bug (dropped a byte when ESC was followed by a
    // non-APC character). Use `feed_slice()` instead — it handles this case
    // correctly by re-processing the non-APC byte inline.

    /// Feed a slice of bytes, returning non-APC bytes and any completed APC payloads.
    ///
    /// This is the preferred API — it handles the edge case of ESC followed by
    /// a non-APC byte correctly by re-feeding.
    pub fn feed_slice<'a>(
        &mut self,
        data: &'a [u8],
        passthrough: &mut Vec<u8>,
        completions: &mut Vec<Vec<u8>>,
    ) {
        let mut i = 0;
        while i < data.len() {
            let byte = data[i];
            i += 1;

            match self.state {
                ApcState::Normal => {
                    if byte == 0x1B {
                        self.state = ApcState::EscSeen;
                    } else {
                        passthrough.push(byte);
                    }
                }
                ApcState::EscSeen => {
                    if byte == b'_' {
                        self.state = ApcState::InApc;
                        self.buffer.clear();
                    } else {
                        // Not APC — pass ESC through and re-process this byte
                        self.state = ApcState::Normal;
                        passthrough.push(0x1B);
                        // Re-process current byte (don't advance i, but we already did)
                        // So process it inline:
                        if byte == 0x1B {
                            self.state = ApcState::EscSeen;
                        } else {
                            passthrough.push(byte);
                        }
                    }
                }
                ApcState::InApc => {
                    if byte == 0x1B {
                        self.state = ApcState::ApcEscSeen;
                    } else if self.buffer.len() < MAX_APC_PAYLOAD {
                        self.buffer.push(byte);
                    } else {
                        // Payload too large — discard and reset
                        self.state = ApcState::Normal;
                        self.buffer.clear();
                    }
                }
                ApcState::ApcEscSeen => {
                    if byte == b'\\' {
                        self.state = ApcState::Normal;
                        completions.push(std::mem::take(&mut self.buffer));
                    } else {
                        self.buffer.push(0x1B);
                        self.buffer.push(byte);
                        self.state = ApcState::InApc;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic_params() {
        let p = parse_params("a=T,f=100,m=0");
        assert_eq!(p.action, GfxAction::TransmitAndDisplay);
        assert_eq!(p.format, GfxFormat::Png);
        assert!(!p.more_chunks);
    }

    #[test]
    fn parse_rgb_params() {
        let p = parse_params("a=T,f=24,s=100,v=50");
        assert_eq!(p.format, GfxFormat::Rgb);
        assert_eq!(p.width, 100);
        assert_eq!(p.height, 50);
    }

    #[test]
    fn parse_compressed() {
        let p = parse_params("f=32,s=10,v=10,o=z,m=1");
        assert_eq!(p.format, GfxFormat::Rgba);
        assert!(p.compressed);
        assert!(p.more_chunks);
    }

    #[test]
    fn apc_accumulator_basic() {
        let mut apc = ApcAccumulator::new();
        let mut passthrough = Vec::new();
        let mut completions = Vec::new();

        // Normal bytes pass through
        apc.feed_slice(b"hello", &mut passthrough, &mut completions);
        assert_eq!(passthrough, b"hello");
        assert!(completions.is_empty());
    }

    #[test]
    fn apc_accumulator_captures_apc() {
        let mut apc = ApcAccumulator::new();
        let mut passthrough = Vec::new();
        let mut completions = Vec::new();

        // ESC _ G payload ESC \    (APC with Kitty graphics)
        let data = b"\x1b_Ga=T,f=100;iVBOR\x1b\\rest";
        apc.feed_slice(data, &mut passthrough, &mut completions);

        assert_eq!(completions.len(), 1);
        assert_eq!(completions[0], b"Ga=T,f=100;iVBOR");
        assert_eq!(passthrough, b"rest");
    }

    #[test]
    fn apc_accumulator_esc_non_apc() {
        let mut apc = ApcAccumulator::new();
        let mut passthrough = Vec::new();
        let mut completions = Vec::new();

        // ESC [ = CSI (not APC) — should pass through
        let data = b"\x1b[31m";
        apc.feed_slice(data, &mut passthrough, &mut completions);

        assert!(completions.is_empty());
        // ESC should be passed through followed by the rest
        assert_eq!(passthrough, b"\x1b[31m");
    }

    #[test]
    fn kitty_gfx_single_chunk_png() {
        let mut state = KittyGfxState::new();

        // Create a minimal 1×1 red PNG
        let png_data = create_test_png(1, 1, &[255, 0, 0, 255]);
        let b64 = BASE64.encode(&png_data);
        let payload = format!("a=T,f=100;{b64}");

        match state.feed(payload.as_bytes()) {
            KittyGfxResult::Image(img) => {
                assert_eq!(img.width(), 1);
                assert_eq!(img.height(), 1);
            }
            KittyGfxResult::Error(e) => panic!("unexpected error: {e}"),
            KittyGfxResult::Pending => panic!("expected image, got pending"),
        }
    }

    #[test]
    fn kitty_gfx_chunked() {
        let mut state = KittyGfxState::new();

        let png_data = create_test_png(2, 2, &[0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255]);
        let b64 = BASE64.encode(&png_data);
        let mid = b64.len() / 2;

        // First chunk: m=1 (more follows)
        let chunk1 = format!("a=T,f=100,m=1;{}", &b64[..mid]);
        assert!(matches!(state.feed(chunk1.as_bytes()), KittyGfxResult::Pending));

        // Second chunk: m=0 (final)
        let chunk2 = format!("m=0;{}", &b64[mid..]);
        match state.feed(chunk2.as_bytes()) {
            KittyGfxResult::Image(img) => {
                assert_eq!(img.width(), 2);
                assert_eq!(img.height(), 2);
            }
            KittyGfxResult::Error(e) => panic!("unexpected error: {e}"),
            KittyGfxResult::Pending => panic!("expected image, got pending"),
        }
    }

    #[test]
    fn kitty_gfx_raw_rgba() {
        let mut state = KittyGfxState::new();

        // 2×2 RGBA image (red, green, blue, white)
        let rgba = vec![
            255, 0, 0, 255,   // red
            0, 255, 0, 255,   // green
            0, 0, 255, 255,   // blue
            255, 255, 255, 255, // white
        ];
        let b64 = BASE64.encode(&rgba);
        let payload = format!("a=T,f=32,s=2,v=2;{b64}");

        match state.feed(payload.as_bytes()) {
            KittyGfxResult::Image(img) => {
                assert_eq!(img.width(), 2);
                assert_eq!(img.height(), 2);
            }
            KittyGfxResult::Error(e) => panic!("unexpected error: {e}"),
            KittyGfxResult::Pending => panic!("expected image, got pending"),
        }
    }

    /// Create a minimal PNG for testing.
    fn create_test_png(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
        use image::{ImageBuffer, RgbaImage};
        let img: RgbaImage = ImageBuffer::from_raw(width, height, rgba.to_vec())
            .expect("valid test image dimensions");
        let mut buf = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut buf);
        img.write_to(&mut cursor, image::ImageFormat::Png)
            .expect("PNG encoding should not fail");
        buf
    }

    #[test]
    fn apc_buffer_capped_at_limit() {
        let mut apc = ApcAccumulator::new();
        let mut passthrough = Vec::new();
        let mut completions = Vec::new();

        // Start APC: ESC _
        apc.feed_slice(b"\x1b_", &mut passthrough, &mut completions);
        assert!(completions.is_empty());

        // Feed slightly more than MAX_APC_PAYLOAD bytes without terminating
        let big_chunk = vec![b'X'; MAX_APC_PAYLOAD + 1024];
        apc.feed_slice(&big_chunk, &mut passthrough, &mut completions);

        // After exceeding the limit, the accumulator should have reset
        // to Normal state and the buffer should be cleared.
        assert!(completions.is_empty(), "oversized APC should not complete");
    }

    #[test]
    fn kitty_b64_buffer_capped() {
        let mut state = KittyGfxState::new();

        // Send a first chunk claiming more follows (m=1) with a huge base64 payload
        let header = "a=T,f=100,m=1;";
        let big_b64 = "A".repeat(MAX_B64_BUFFER + 1024);
        let payload = format!("{header}{big_b64}");

        match state.feed(payload.as_bytes()) {
            KittyGfxResult::Error(e) => {
                assert!(e.contains("limit"), "expected limit error, got: {e}");
            }
            KittyGfxResult::Pending => panic!("should have hit buffer limit, not pending"),
            KittyGfxResult::Image(_) => panic!("should not produce image from oversized buffer"),
        }
    }
}
