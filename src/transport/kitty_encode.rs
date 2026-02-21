// SPDX-License-Identifier: MIT OR Apache-2.0
//! Low-level Kitty graphics protocol encoding: compression, base64,
//! chunked escape-sequence construction, and writing.

use std::io::Write;

use flate2::write::ZlibEncoder;
use flate2::Compression;
use tiny_skia::Pixmap;

use super::backend::TerminalPosition;
use crate::PixelCanvasError;

/// Maximum bytes to send in a single Kitty protocol chunk.
/// The protocol spec suggests 4096 as a guideline, but modern terminals
/// (kitty, `WezTerm`, Ghostty) handle much larger chunks efficiently.
/// 64 KB reduces escape-sequence framing overhead by ~16×.
pub(super) const CHUNK_SIZE: usize = 65_536;

/// Encode a pixmap as PNG bytes.
pub(super) fn encode_png(pixmap: &Pixmap) -> Result<Vec<u8>, PixelCanvasError> {
    pixmap
        .encode_png()
        .map_err(|e| PixelCanvasError::Rasterization(e.to_string()))
}

/// Zlib-compress raw pixel data into `compress_buf`, then base64-encode
/// the result into `encode_buf`.
pub(super) fn compress_and_encode(
    raw_data: &[u8],
    compress_buf: &mut Vec<u8>,
    encode_buf: &mut String,
) -> Result<(), PixelCanvasError> {
    use base64::Engine;

    compress_buf.clear();
    {
        let mut encoder = ZlibEncoder::new(&mut *compress_buf, Compression::fast());
        encoder.write_all(raw_data).map_err(|e| {
            PixelCanvasError::Rasterization(format!("zlib compress failed: {e}"))
        })?;
        encoder
            .finish()
            .map_err(|e| PixelCanvasError::Rasterization(format!("zlib finish failed: {e}")))?;
    }

    encode_buf.clear();
    base64::engine::general_purpose::STANDARD
        .encode_string(compress_buf, encode_buf);

    Ok(())
}

/// Build chunked Kitty escape sequences into `send_buf` from already-encoded
/// base64 data in `encode_buf`, then write+flush via `writer`.
///
/// `first_chunk_params` is the parameter string for the first chunk
/// (e.g., `"a=T,q=2,f=32,o=z,s=640,v=480,i=1,p=1,z=-1"`).
pub(super) fn send_encoded<W: Write>(
    writer: &mut W,
    encode_buf: &str,
    send_buf: &mut Vec<u8>,
    first_chunk_params: &str,
    position: TerminalPosition,
) -> Result<(), PixelCanvasError> {
    let total = encode_buf.len();
    let n_chunks = total.div_ceil(CHUNK_SIZE).max(1);

    send_buf.clear();

    // Begin synchronized update
    write!(send_buf, "\x1b[?2026h").map_err(PixelCanvasError::Transmission)?;
    write!(
        send_buf,
        "\x1b[{};{}H",
        position.row + 1,
        position.col + 1
    )
    .map_err(PixelCanvasError::Transmission)?;

    for i in 0..n_chunks {
        let start = i * CHUNK_SIZE;
        let end = (start + CHUNK_SIZE).min(total);
        let chunk = &encode_buf[start..end];
        let more = i32::from(i != n_chunks - 1);

        if i == 0 {
            write!(
                send_buf,
                "\x1b_G{first_chunk_params},c={},r={},m={more};{chunk}\x1b\\",
                position.width_cells, position.height_cells,
            )
            .map_err(PixelCanvasError::Transmission)?;
        } else {
            write!(send_buf, "\x1b_Gm={more};{chunk}\x1b\\")
                .map_err(PixelCanvasError::Transmission)?;
        }
    }

    // End synchronized update
    write!(send_buf, "\x1b[?2026l").map_err(PixelCanvasError::Transmission)?;

    writer
        .write_all(send_buf)
        .map_err(PixelCanvasError::Transmission)?;
    writer
        .flush()
        .map_err(PixelCanvasError::Transmission)?;

    Ok(())
}

/// Send a Kitty graphics command with PNG payload.
pub(super) fn send_chunked<W: Write>(
    writer: &mut W,
    encode_buf: &mut String,
    send_buf: &mut Vec<u8>,
    image_id: u32,
    placement_id: u32,
    png_data: &[u8],
    position: TerminalPosition,
    z_index: i32,
) -> Result<(), PixelCanvasError> {
    use base64::Engine;
    encode_buf.clear();
    base64::engine::general_purpose::STANDARD.encode_string(png_data, encode_buf);

    let params = format!("a=T,q=2,f=100,i={image_id},p={placement_id},z={z_index}");
    send_encoded(writer, encode_buf, send_buf, &params, position)
}
