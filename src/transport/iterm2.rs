//! iTerm2 inline image protocol backend.
//!
//! Implements the [iTerm2 Inline Images Protocol](https://iterm2.com/documentation-images.html)
//! for transmitting pixel-perfect images to iTerm2, `WezTerm`, Mintty, and other
//! compatible terminal emulators.
//!
//! # How It Works
//!
//! The protocol uses an OSC (Operating System Command) escape sequence:
//! ```text
//! ESC ] 1337 ; File=<params> : <base64_data> BEL
//! ```
//!
//! Images are encoded as base64 PNG data and inlined directly into the
//! terminal output stream.
//!
//! # Feature Gate
//!
//! This module is available when the `iterm2` feature is enabled.

use std::io::{self, Write};

use base64::Engine;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use tiny_skia::Pixmap;

use crate::transport::backend::{ImageHandle, ProtocolBackend, ProtocolKind, TerminalPosition};
use crate::PixelCanvasError;

// ---------------------------------------------------------------------------
// CRC32 lookup table (must be before functions that use it)
// ---------------------------------------------------------------------------

/// Pre-computed CRC32 lookup table (IEEE polynomial).
const CRC_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut n = 0usize;
    while n < 256 {
        #[allow(clippy::cast_possible_truncation)]
        let mut c = n as u32;
        let mut k = 0;
        while k < 8 {
            if c & 1 != 0 {
                c = 0xEDB8_8320 ^ (c >> 1);
            } else {
                c >>= 1;
            }
            k += 1;
        }
        table[n] = c;
        n += 1;
    }
    table
};

// ---------------------------------------------------------------------------
// Iterm2Backend
// ---------------------------------------------------------------------------

/// iTerm2 inline image protocol backend.
///
/// Encodes pixel data as base64 PNG and transmits via OSC 1337 escape sequences.
#[derive(Debug)]
pub struct Iterm2Backend<W: Write + std::fmt::Debug = io::Stdout> {
    writer: W,
    next_id: u32,
    /// Reusable PNG encoding buffer.
    png_buf: Vec<u8>,
    /// Reusable base64 encoding buffer.
    b64_buf: Vec<u8>,
}

impl Iterm2Backend<io::Stdout> {
    /// Create a new iTerm2 backend writing to stdout.
    #[must_use]
    pub fn new() -> Self {
        Self {
            writer: io::stdout(),
            next_id: 1,
            png_buf: Vec::with_capacity(64 * 1024),
            b64_buf: Vec::with_capacity(128 * 1024),
        }
    }
}

impl Default for Iterm2Backend<io::Stdout> {
    fn default() -> Self {
        Self::new()
    }
}

impl<W: Write + std::fmt::Debug> Iterm2Backend<W> {
    /// Create a backend with a custom writer (useful for testing).
    pub fn with_writer(writer: W) -> Self {
        Self {
            writer,
            next_id: 1,
            png_buf: Vec::with_capacity(64 * 1024),
            b64_buf: Vec::with_capacity(128 * 1024),
        }
    }

    /// Consume the backend and return the writer.
    pub fn into_writer(self) -> W {
        self.writer
    }

    /// Encode a pixmap as an iTerm2 inline image and write it.
    fn write_inline_image(
        &mut self,
        pixmap: &Pixmap,
        position: TerminalPosition,
    ) -> Result<(), PixelCanvasError> {
        let pw = pixmap.width();
        let ph = pixmap.height();
        if pw == 0 || ph == 0 {
            return Ok(());
        }

        // Encode as PNG
        self.png_buf.clear();
        encode_pixmap_to_png(pixmap, &mut self.png_buf);

        let png_size = self.png_buf.len();

        // Base64 encode
        let b64_len = base64::encoded_len(png_size, true).unwrap_or(png_size * 2);
        self.b64_buf.clear();
        self.b64_buf.resize(b64_len, 0);
        let written = base64::engine::general_purpose::STANDARD
            .encode_slice(&self.png_buf, &mut self.b64_buf)
            .unwrap_or(0);
        self.b64_buf.truncate(written);

        // Move cursor to position
        if position.col > 0 || position.row > 0 {
            write!(
                self.writer,
                "\x1b[{};{}H",
                position.row + 1,
                position.col + 1
            )?;
        }

        // Write OSC 1337 header
        // File=inline=1;width=<cells>;height=<cells>;size=<bytes>;preserveAspectRatio=0:
        write!(
            self.writer,
            "\x1b]1337;File=inline=1;width={cells_w};height={cells_h};size={png_size};preserveAspectRatio=0:",
            cells_w = position.width_cells,
            cells_h = position.height_cells,
        )?;

        // Write base64 data
        self.writer.write_all(&self.b64_buf)?;

        // Write BEL terminator
        self.writer.write_all(b"\x07")?;
        self.writer.flush()?;

        Ok(())
    }
}

impl<W: Write + std::fmt::Debug + Send> ProtocolBackend for Iterm2Backend<W> {
    fn transmit(
        &mut self,
        pixmap: &Pixmap,
        position: TerminalPosition,
        _z_index: i32,
    ) -> Result<ImageHandle, PixelCanvasError> {
        self.write_inline_image(pixmap, position)?;
        let id = self.next_id;
        self.next_id += 1;
        Ok(ImageHandle {
            id,
            protocol: ProtocolKind::Iterm2,
        })
    }

    fn remove(&mut self, _handle: &ImageHandle) -> Result<(), PixelCanvasError> {
        // iTerm2 inline images are not retained — they become part of
        // the terminal's scrollback. Removal is a no-op.
        Ok(())
    }

    fn clear_all(&mut self) -> Result<(), PixelCanvasError> {
        Ok(())
    }

    fn supports_alpha(&self) -> bool {
        // PNG supports alpha, and iTerm2 renders it correctly.
        true
    }

    fn protocol_kind(&self) -> ProtocolKind {
        ProtocolKind::Iterm2
    }
}

// ---------------------------------------------------------------------------
// Minimal PNG encoder (no external crate dependency)
// ---------------------------------------------------------------------------

/// Encode a tiny-skia `Pixmap` as PNG into `out`.
///
/// Uses a minimal PNG encoder with zlib-compressed IDAT to avoid pulling in
/// a full PNG crate. Handles un-premultiplying alpha from tiny-skia's
/// premultiplied format.
#[allow(clippy::many_single_char_names)]
fn encode_pixmap_to_png(pixmap: &Pixmap, out: &mut Vec<u8>) {
    let width = pixmap.width();
    let height = pixmap.height();
    let data = pixmap.data(); // RGBA premultiplied

    // PNG signature
    out.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);

    // IHDR
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(6); // color type: RGBA
    ihdr.push(0); // compression
    ihdr.push(0); // filter
    ihdr.push(0); // interlace
    write_png_chunk(out, *b"IHDR", &ihdr);

    // IDAT — use zlib-compressed data for smaller output
    // Prepare raw image data with filter byte per row
    let row_bytes = width as usize * 4;
    let raw_len = height as usize * (1 + row_bytes);
    let mut raw = Vec::with_capacity(raw_len);

    for row in 0..height as usize {
        raw.push(0); // filter: None
        // Convert from premultiplied alpha to straight alpha
        for col in 0..width as usize {
            let idx = (row * width as usize + col) * 4;
            let (red, grn, blu, alpha) = (data[idx], data[idx + 1], data[idx + 2], data[idx + 3]);
            if alpha == 0 {
                raw.extend_from_slice(&[0, 0, 0, 0]);
            } else if alpha == 255 {
                raw.extend_from_slice(&[red, grn, blu, alpha]);
            } else {
                // Un-premultiply
                let af = 255.0 / f32::from(alpha);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let ur = (f32::from(red) * af).min(255.0) as u8;
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let ug = (f32::from(grn) * af).min(255.0) as u8;
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let ub = (f32::from(blu) * af).min(255.0) as u8;
                raw.extend_from_slice(&[ur, ug, ub, alpha]);
            }
        }
    }

    // Compress with flate2
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
    encoder.write_all(&raw).expect("zlib encode");
    let compressed = encoder.finish().expect("zlib finish");
    write_png_chunk(out, *b"IDAT", &compressed);

    // IEND
    write_png_chunk(out, *b"IEND", &[]);
}

/// Write a PNG chunk: length + type + data + CRC.
fn write_png_chunk(out: &mut Vec<u8>, chunk_type: [u8; 4], data: &[u8]) {
    #[allow(clippy::cast_possible_truncation)]
    let len = data.len() as u32;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(&chunk_type);
    out.extend_from_slice(data);

    // CRC32 over type + data
    let crc = crc32(&chunk_type, data);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// Compute CRC32 for PNG (type + data).
fn crc32(chunk_type: &[u8], data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in chunk_type.iter().chain(data.iter()) {
        let index = ((crc ^ u32::from(byte)) & 0xFF) as usize;
        crc = CRC_TABLE[index] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::many_single_char_names)]
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
    fn iterm2_output_structure() {
        let pm = make_pixmap(2, 2, &[
            (255, 0, 0, 255), (0, 255, 0, 255),
            (0, 0, 255, 255), (255, 255, 0, 255),
        ]);

        let writer = io::Cursor::new(Vec::new());
        let mut backend = Iterm2Backend::with_writer(writer);
        let pos = TerminalPosition::new(0, 0, 2, 1);
        let handle = backend.transmit(&pm, pos, 0).unwrap();

        let output = backend.into_writer().into_inner();
        let output_str = String::from_utf8_lossy(&output);

        assert!(output_str.contains("\x1b]1337;"), "should contain OSC 1337");
        assert!(output_str.contains("inline=1"), "should set inline=1");
        assert!(output_str.contains("\x07"), "should end with BEL");
        assert_eq!(handle.protocol(), ProtocolKind::Iterm2);
    }

    #[test]
    fn iterm2_protocol_kind() {
        let backend = Iterm2Backend::with_writer(io::Cursor::new(Vec::new()));
        assert_eq!(backend.protocol_kind(), ProtocolKind::Iterm2);
        assert!(backend.supports_alpha());
    }

    #[test]
    fn iterm2_remove_is_noop() {
        let mut backend = Iterm2Backend::with_writer(io::Cursor::new(Vec::new()));
        let pm = make_pixmap(2, 2, &[
            (0, 0, 0, 255), (0, 0, 0, 255),
            (0, 0, 0, 255), (0, 0, 0, 255),
        ]);
        let pos = TerminalPosition::new(0, 0, 2, 1);
        let handle = backend.transmit(&pm, pos, 0).unwrap();
        assert!(backend.remove(&handle).is_ok());
        assert!(backend.clear_all().is_ok());
    }

    #[test]
    fn png_encoder_produces_valid_signature() {
        let pm = Pixmap::new(4, 4).unwrap();
        let mut buf = Vec::new();
        encode_pixmap_to_png(&pm, &mut buf);
        // PNG signature: 137 80 78 71 13 10 26 10
        assert_eq!(&buf[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }

    #[test]
    fn full_pipeline_iterm2() {
        use crate::rasterize::Rasterizer;
        use crate::scene::{Color, PixelCanvas};

        let canvas = PixelCanvas::new(30, 30)
            .background(Color::GREEN)
            .rect(5.0, 5.0, 20.0, 20.0)
            .fill(Color::BLUE)
            .done();
        let pixmap = Rasterizer::rasterize(&canvas).unwrap();

        let writer = io::Cursor::new(Vec::new());
        let mut backend = Iterm2Backend::with_writer(writer);
        let pos = TerminalPosition::new(1, 1, 4, 3);
        let handle = backend.transmit(&pixmap, pos, 0).unwrap();
        assert_eq!(handle.protocol(), ProtocolKind::Iterm2);

        let output = backend.into_writer().into_inner();
        assert!(!output.is_empty());
    }
}
