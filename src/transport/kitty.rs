// SPDX-License-Identifier: MIT OR Apache-2.0
//! Kitty graphics protocol backend.
//!
//! Implements the [Kitty graphics protocol](https://sw.kovidgoyal.net/kitty/graphics-protocol/)
//! for transmitting pixel-perfect images to supported terminal emulators.
//!
//! The protocol works by base64-encoding image data and sending it via
//! escape sequences. Images are assigned unique IDs and can be placed at
//! specific positions, layered with z-index, and cleaned up individually.
//!
//! # Transmission Formats
//!
//! | Format | Wire size | Encode cost | Best for |
//! |---------|-----------|-------------|----------|
//! | `ZlibRgba` | ~500 KB | ~2 ms | **Animation** (default) |
//! | `RawRgba` | ~5.5 MB | ~0 ms | Debugging |
//! | `Png` | ~50 KB | ~30 ms | Static images |

use std::io::Write;

use flate2::write::ZlibEncoder;
use flate2::Compression;
use std::sync::atomic::{AtomicU32, Ordering};

use tiny_skia::Pixmap;

use crate::transport::backend::{
    FontSize, ImageHandle, ProtocolBackend, ProtocolKind, TerminalPosition,
};
use crate::PixelCanvasError;

/// Global image ID counter. Kitty image IDs must be unique across the
/// terminal session, so we use an atomic counter.
static NEXT_IMAGE_ID: AtomicU32 = AtomicU32::new(1);

use super::kitty_encode;

/// Maximum bytes to send in a single Kitty protocol chunk.
/// The protocol spec suggests 4096 as a guideline, but modern terminals
/// (kitty, `WezTerm`, Ghostty) handle much larger chunks efficiently.
/// 64 KB reduces escape-sequence framing overhead by ~16×.
const CHUNK_SIZE: usize = 65_536;

/// How to encode pixel data for Kitty protocol transmission.
///
/// The default is [`ZlibRgba`](TransmitFormat::ZlibRgba), which provides
/// the best balance of encode speed and wire size for animation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TransmitFormat {
    /// Zlib-compressed 32-bit RGBA pixels (`f=32,o=z`). Compresses raw
    /// pixels with deflate level 1 before base64 encoding, giving ~5–10×
    /// smaller payloads than `RawRgba` at a cost of ~2 ms per frame.
    /// This is the **recommended format for animation**.
    #[default]
    ZlibRgba,
    /// Raw 32-bit RGBA pixels (`f=32`). No compression overhead, but
    /// very large on the wire (~5 MB for a 1280×800 image). Useful for
    /// debugging or when the terminal does not support `o=z`.
    RawRgba,
    /// PNG-encoded pixels (`f=100`). 10–20× more compact on the wire, but
    /// requires a PNG encoding step per frame which can be slow for large
    /// images (~30 ms+). Best for static or infrequently updated content.
    Png,
    /// POSIX shared memory (`t=s,f=32`). Zero-copy: pixel data is written
    /// to a shared memory object and the terminal reads it directly.
    /// No base64, no pipe I/O. Fastest possible path for local terminals.
    /// Requires the `shm` feature.
    #[cfg(feature = "shm")]
    SharedMemory,
}

// ---------------------------------------------------------------------------
// KittyBackend
// ---------------------------------------------------------------------------

/// Backend for the Kitty graphics protocol.
///
/// Transmits images via stdout escape sequences. Each image is assigned a
/// unique ID that can be used for later removal.
///
/// # Note
///
/// This backend writes directly to a configurable writer (defaulting to
/// stdout). In testing, you can provide a `Vec<u8>` buffer instead.
#[derive(Debug)]
pub struct KittyBackend<W: Write + std::fmt::Debug = std::io::Stdout> {
    writer: W,
    /// IDs of all images currently managed by this backend.
    active_ids: Vec<u32>,
    /// Font size for pixel ↔ cell conversion.
    font_size: FontSize,
    /// Transmission format: raw RGBA, zlib, PNG, or shm.
    format: TransmitFormat,
    /// Reusable base64 encoding buffer to avoid per-frame allocation.
    encode_buf: String,
    /// Reusable zlib compression buffer to avoid per-frame allocation.
    compress_buf: Vec<u8>,
    /// Reusable write buffer: all escape sequences are assembled here
    /// before a single `write_all` + `flush`, reducing syscall count
    /// from ~124 to 1–2 per frame.
    send_buf: Vec<u8>,
    /// Fixed placement ID for flicker-free in-place replacement.
    /// The Kitty protocol replaces a placement atomically when the
    /// same (`image_id`, `placement_id`) pair is reused.
    placement_id: u32,
    /// Reusable buffer for extracting tile pixel data (avoids per-tile allocation).
    tile_buf: Vec<u8>,
    /// Per-tile Kitty image IDs for incremental updates.
    /// Indexed by `tile_row * tiles_per_row + tile_col`.
    tile_ids: Vec<u32>,
    /// Double-buffered shared memory for zero-copy transmission.
    /// We alternate between two SHM objects so the terminal can read
    /// from one while we write the next frame to the other.
    #[cfg(feature = "shm")]
    shm_bufs: [Option<crate::transport::shm::ShmBuffer>; 2],
    #[cfg(feature = "shm")]
    shm_idx: usize,
}

impl KittyBackend<std::io::Stdout> {
    /// Create a new Kitty backend that writes to stdout.
    #[must_use]
    pub fn new(font_size: FontSize) -> Self {
        Self {
            writer: std::io::stdout(),
            active_ids: Vec::new(),
            font_size,
            format: TransmitFormat::default(),
            encode_buf: String::new(),
            compress_buf: Vec::new(),
            send_buf: Vec::with_capacity(128 * 1024),
            placement_id: 1,
            tile_buf: Vec::new(),
            tile_ids: Vec::new(),
            #[cfg(feature = "shm")]
            shm_bufs: [None, None],
            #[cfg(feature = "shm")]
            shm_idx: 0,
        }
    }
}

impl<W: Write + std::fmt::Debug> KittyBackend<W> {
    /// Create a new Kitty backend with a custom writer.
    ///
    /// This is useful for testing (pass a `Vec<u8>`) or for writing to
    /// a file descriptor other than stdout.
    #[must_use]
    pub fn with_writer(writer: W, font_size: FontSize) -> Self {
        Self {
            writer,
            active_ids: Vec::new(),
            font_size,
            format: TransmitFormat::default(),
            encode_buf: String::new(),
            compress_buf: Vec::new(),
            send_buf: Vec::with_capacity(128 * 1024),
            placement_id: 1,
            tile_buf: Vec::new(),
            tile_ids: Vec::new(),
            #[cfg(feature = "shm")]
            shm_bufs: [None, None],
            #[cfg(feature = "shm")]
            shm_idx: 0,
        }
    }

    /// Set the transmission format.
    #[must_use]
    pub const fn format(mut self, format: TransmitFormat) -> Self {
        self.format = format;
        self
    }

    /// The font size used for pixel ↔ cell conversion.
    #[must_use]
    pub const fn font_size(&self) -> FontSize {
        self.font_size
    }

    /// Consume the backend and return the underlying writer.
    ///
    /// Useful for inspecting written output in tests.
    #[must_use]
    pub fn into_writer(self) -> W {
        self.writer
    }

    /// Allocate a new unique image ID, skipping ID 0 which is reserved
    /// by the Kitty protocol as "unspecified".
    fn next_id() -> u32 {
        loop {
            let id = NEXT_IMAGE_ID.fetch_add(1, Ordering::Relaxed);
            if id != 0 {
                return id;
            }
        }
    }

    /// Encode a pixmap as PNG bytes.
    fn encode_png(pixmap: &Pixmap) -> Result<Vec<u8>, PixelCanvasError> {
        kitty_encode::encode_png(pixmap)
    }

    /// Build chunked Kitty escape sequences into `send_buf` from already-encoded
    /// base64 data in `encode_buf`, then write+flush.
    ///
    /// `first_chunk_params` is the parameter string for the first chunk
    /// (e.g., `"a=T,q=2,f=32,o=z,s=640,v=480,i=1,p=1,z=-1"`).
    fn send_encoded(
        &mut self,
        first_chunk_params: &str,
        position: TerminalPosition,
    ) -> Result<(), PixelCanvasError> {
        kitty_encode::send_encoded(
            &mut self.writer,
            &self.encode_buf,
            &mut self.send_buf,
            first_chunk_params,
            position,
        )
    }

    /// Zlib-compress raw pixel data into `compress_buf`, then base64-encode
    /// the result into `encode_buf`.
    fn compress_and_encode(&mut self, raw_data: &[u8]) -> Result<(), PixelCanvasError> {
        kitty_encode::compress_and_encode(raw_data, &mut self.compress_buf, &mut self.encode_buf)
    }

    /// Send a Kitty graphics command with PNG payload.
    fn send_chunked(
        &mut self,
        image_id: u32,
        png_data: &[u8],
        position: TerminalPosition,
        z_index: i32,
    ) -> Result<(), PixelCanvasError> {
        kitty_encode::send_chunked(
            &mut self.writer,
            &mut self.encode_buf,
            &mut self.send_buf,
            image_id,
            self.placement_id,
            png_data,
            position,
            z_index,
        )
    }

    /// Send a delete command for a specific image ID.
    fn send_delete(&mut self, image_id: u32) -> Result<(), PixelCanvasError> {
        write!(self.writer, "\x1b_Ga=d,d=I,q=2,i={image_id};\x1b\\")
            .map_err(PixelCanvasError::Transmission)?;
        self.writer
            .flush()
            .map_err(PixelCanvasError::Transmission)?;
        Ok(())
    }

    /// Send raw RGBA pixel data (f=32) — much faster than PNG for animation.
    fn send_raw_rgba(
        &mut self,
        image_id: u32,
        pixmap: &Pixmap,
        position: TerminalPosition,
        z_index: i32,
    ) -> Result<(), PixelCanvasError> {
        use base64::Engine;

        let pixel_width = pixmap.width();
        let pixel_height = pixmap.height();
        let placement_id = self.placement_id;

        self.encode_buf.clear();
        base64::engine::general_purpose::STANDARD
            .encode_string(pixmap.data(), &mut self.encode_buf);

        let params = format!("a=T,q=2,f=32,s={pixel_width},v={pixel_height},i={image_id},p={placement_id},z={z_index}");
        self.send_encoded(&params, position)
    }

    /// Send zlib-compressed RGBA pixel data (`f=32,o=z`).
    ///
    /// Compresses with deflate level 1 (fast), then base64-encodes and sends
    /// via chunked escape sequences. ~5–10× compression on typical graphics.
    fn send_zlib_rgba(
        &mut self,
        image_id: u32,
        pixmap: &Pixmap,
        position: TerminalPosition,
        z_index: i32,
    ) -> Result<(), PixelCanvasError> {
        let pixel_width = pixmap.width();
        let pixel_height = pixmap.height();
        let placement_id = self.placement_id;

        self.compress_and_encode(pixmap.data())?;

        let params = format!("a=T,q=2,f=32,o=z,s={pixel_width},v={pixel_height},i={image_id},p={placement_id},z={z_index}");
        self.send_encoded(&params, position)
    }

    /// Send pixel data via POSIX shared memory (`t=s,f=32`).
    ///
    /// Writes raw RGBA directly into a shared memory object, then sends
    /// a tiny escape sequence telling the terminal to read from it.
    /// No base64 encoding, no pipe I/O — just a ~200 byte control sequence.
    #[cfg(feature = "shm")]
    fn send_shm(
        &mut self,
        image_id: u32,
        pixmap: &Pixmap,
        position: TerminalPosition,
        z_index: i32,
    ) -> Result<(), PixelCanvasError> {
        let pixel_width = pixmap.width();
        let pixel_height = pixmap.height();
        let raw_data = pixmap.data();
        let data_size = raw_data.len();
        let placement_id = self.placement_id;

        // Kitty unlinks the SHM object after reading, so we must create a
        // fresh object every frame. Alternate names to avoid racing the
        // terminal's unlink of the previous frame's object.
        let idx = self.shm_idx;
        self.shm_idx ^= 1;
        let shm_name = format!("scry-{}-{idx}", std::process::id());

        // Drop the old buffer (just munmaps — the terminal already unlinked the name).
        self.shm_bufs[idx] = None;

        let buf = crate::transport::shm::ShmBuffer::new(&shm_name, data_size)
            .map_err(|e| PixelCanvasError::Rasterization(format!("shm_open failed: {e}")))?
            .consumer_unlinks();

        // Write pixels into shared memory
        buf.write(raw_data)
            .map_err(|e| PixelCanvasError::Rasterization(format!("shm write failed: {e}")))?;
        let name = buf.name();

        // Base64-encode the SHM name — ALL Kitty protocol payloads are
        // base64-encoded, including shared memory object names.
        use base64::Engine;
        self.encode_buf.clear();
        base64::engine::general_purpose::STANDARD
            .encode_string(name.as_bytes(), &mut self.encode_buf);

        // Store the buffer so the mmap stays alive until the terminal reads it.
        // Don't unlink on drop — Kitty handles that.
        self.shm_bufs[idx] = Some(buf);

        // Batch all escapes into send_buf, then write_all + flush once
        // (matches the pattern used by all other send methods).
        self.send_buf.clear();

        // Begin synchronized update
        write!(self.send_buf, "\x1b[?2026h").map_err(PixelCanvasError::Transmission)?;

        // Move cursor to target position
        write!(
            self.send_buf,
            "\x1b[{};{}H",
            position.row + 1,
            position.col + 1
        )
        .map_err(PixelCanvasError::Transmission)?;

        // Single escape: t=s (shared memory), f=32 (raw RGBA),
        // s=width, v=height, S=byte_count — payload is the base64-encoded
        // shm object name. S= is REQUIRED for t=s.
        write!(
            self.send_buf,
            "\x1b_Ga=T,q=2,t=s,f=32,s={pixel_width},v={pixel_height},S={data_size},i={image_id},p={placement_id},z={z_index},c={},r={};{}\x1b\\",
            position.width_cells,
            position.height_cells,
            self.encode_buf,
        )
        .map_err(PixelCanvasError::Transmission)?;

        // End synchronized update
        write!(self.send_buf, "\x1b[?2026l").map_err(PixelCanvasError::Transmission)?;

        self.writer
            .write_all(&self.send_buf)
            .map_err(PixelCanvasError::Transmission)?;
        self.writer
            .flush()
            .map_err(PixelCanvasError::Transmission)?;

        Ok(())
    }
    /// Send zlib-compressed RGBA with per-stage profiling.
    ///
    /// Returns a `TransportProfile` breaking down compress, encode, and I/O time.
    #[allow(dead_code, clippy::cast_precision_loss)]
    pub(crate) fn send_zlib_rgba_profiled(
        &mut self,
        image_id: u32,
        pixmap: &Pixmap,
        position: TerminalPosition,
        z_index: i32,
    ) -> Result<crate::rasterize::TransportProfile, PixelCanvasError> {
        use base64::Engine;
        use std::time::Instant;

        let total_start = Instant::now();
        let mut tp = crate::rasterize::TransportProfile::default();

        let pixel_width = pixmap.width();
        let pixel_height = pixmap.height();
        let raw_data = pixmap.data();
        let placement_id = self.placement_id;

        tp.raw_bytes = raw_data.len();

        // 1. Zlib compression
        let compress_start = Instant::now();
        self.compress_buf.clear();
        {
            let mut encoder = ZlibEncoder::new(&mut self.compress_buf, Compression::fast());
            encoder.write_all(raw_data).map_err(|e| {
                PixelCanvasError::Rasterization(format!("zlib compress failed: {e}"))
            })?;
            encoder
                .finish()
                .map_err(|e| PixelCanvasError::Rasterization(format!("zlib finish failed: {e}")))?;
        }
        tp.compress_us = compress_start.elapsed().as_micros() as u64;

        // 2. Base64 encoding
        let encode_start = Instant::now();
        self.encode_buf.clear();
        base64::engine::general_purpose::STANDARD
            .encode_string(&self.compress_buf, &mut self.encode_buf);
        tp.encode_us = encode_start.elapsed().as_micros() as u64;

        tp.wire_bytes = self.encode_buf.len();

        // 3. Build + write + flush
        let io_start = Instant::now();
        let params = format!("a=T,q=2,f=32,o=z,s={pixel_width},v={pixel_height},i={image_id},p={placement_id},z={z_index}");
        self.send_encoded(&params, position)?;
        tp.io_us = io_start.elapsed().as_micros() as u64;
        tp.total_us = total_start.elapsed().as_micros() as u64;

        Ok(tp)
    }

    /// Send only the changed tiles via zlib-compressed RGBA.
    ///
    /// For each dirty tile:
    /// 1. Extract tile pixels from the full pixmap into `tile_buf`
    /// 2. Zlib-compress + base64-encode
    /// 3. Emit a Kitty escape sequence with the tile's pixel offset
    ///
    /// All tiles are wrapped in a single synchronized update for atomic display.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn send_tiles_zlib_rgba(
        &mut self,
        pixmap: &Pixmap,
        position: TerminalPosition,
        z_index: i32,
        dirty_tiles: &[crate::rasterize::DirtyTile],
    ) -> Result<(), PixelCanvasError> {
        use base64::Engine;

        let pixmap_w = pixmap.width() as usize;
        let data = pixmap.data();
        let font_w = self.font_size.width.max(1);
        let font_h = self.font_size.height.max(1);

        // Ensure tile_ids is large enough for the tile grid.
        // We compute the grid dimensions from the pixmap size.
        let tiles_x = pixmap_w.div_ceil(crate::rasterize::TILE_SIZE);
        let tiles_y = (pixmap.height() as usize).div_ceil(crate::rasterize::TILE_SIZE);
        let total_tiles = tiles_x * tiles_y;
        if self.tile_ids.len() != total_tiles {
            // Grid changed: delete old tile images and reset
            let ids_to_delete: Vec<u32> = self
                .tile_ids
                .iter()
                .copied()
                .filter(|&tid| tid != 0)
                .collect();
            for tid in ids_to_delete {
                let _ = self.send_delete(tid);
            }
            self.tile_ids.clear();
            self.tile_ids.resize(total_tiles, 0);
        }

        self.send_buf.clear();

        // Begin synchronized update
        write!(self.send_buf, "\x1b[?2026h").map_err(PixelCanvasError::Transmission)?;

        for tile in dirty_tiles {
            let tw = tile.width;
            let th = tile.height;
            if tw == 0 || th == 0 {
                continue;
            }

            // Extract tile pixels into tile_buf
            let tile_bytes = tw * th * 4;
            self.tile_buf.clear();
            self.tile_buf.reserve(tile_bytes);
            for row in tile.y..(tile.y + th) {
                let start = (row * pixmap_w + tile.x) * 4;
                let end = start + tw * 4;
                self.tile_buf.extend_from_slice(&data[start..end]);
            }

            // Zlib compress + base64 encode
            {
                self.compress_buf.clear();
                let mut encoder = ZlibEncoder::new(&mut self.compress_buf, Compression::fast());
                encoder.write_all(&self.tile_buf).map_err(|e| {
                    PixelCanvasError::Rasterization(format!("tile zlib failed: {e}"))
                })?;
                encoder.finish().map_err(|e| {
                    PixelCanvasError::Rasterization(format!("tile zlib finish: {e}"))
                })?;
            }
            self.encode_buf.clear();
            base64::engine::general_purpose::STANDARD
                .encode_string(&self.compress_buf, &mut self.encode_buf);

            // Tile grid index
            let tile_col = tile.x / crate::rasterize::TILE_SIZE;
            let tile_row = tile.y / crate::rasterize::TILE_SIZE;
            let tile_idx = tile_row * tiles_x + tile_col;

            // Get or allocate a Kitty image ID for this tile
            let image_id = if self.tile_ids[tile_idx] != 0 {
                self.tile_ids[tile_idx]
            } else {
                let id = Self::next_id();
                self.tile_ids[tile_idx] = id;
                self.active_ids.push(id);
                id
            };

            // Compute terminal cell position for this tile.
            // The tile's pixel offset within the canvas is (tile.x, tile.y).
            // Convert to cell offset from the widget's base position.
            let cell_col_offset = (tile.x as u16) / font_w;
            let cell_row_offset = (tile.y as u16) / font_h;
            let tile_cell_col = position.col + cell_col_offset;
            let tile_cell_row = position.row + cell_row_offset;
            let tile_w_cells = (tw as u16).div_ceil(font_w);
            let tile_h_cells = (th as u16).div_ceil(font_h);

            // Cursor position
            write!(
                self.send_buf,
                "\x1b[{};{}H",
                tile_cell_row + 1,
                tile_cell_col + 1
            )
            .map_err(PixelCanvasError::Transmission)?;

            // Chunked transmission for this tile
            let total = self.encode_buf.len();
            let n_chunks = total.div_ceil(CHUNK_SIZE).max(1);
            let placement_id = self.placement_id;

            for i in 0..n_chunks {
                let start = i * CHUNK_SIZE;
                let end = (start + CHUNK_SIZE).min(total);
                let chunk = &self.encode_buf[start..end];
                let more = i32::from(i != n_chunks - 1);

                if i == 0 {
                    write!(
                        self.send_buf,
                        "\x1b_Ga=T,q=2,f=32,o=z,s={tw},v={th},i={image_id},p={placement_id},z={z_index},c={tile_w_cells},r={tile_h_cells},m={more};{chunk}\x1b\\",
                    ).map_err(PixelCanvasError::Transmission)?;
                } else {
                    write!(self.send_buf, "\x1b_Gm={more};{chunk}\x1b\\")
                        .map_err(PixelCanvasError::Transmission)?;
                }
            }
        }

        // End synchronized update
        write!(self.send_buf, "\x1b[?2026l").map_err(PixelCanvasError::Transmission)?;

        self.writer
            .write_all(&self.send_buf)
            .map_err(PixelCanvasError::Transmission)?;
        self.writer
            .flush()
            .map_err(PixelCanvasError::Transmission)?;

        Ok(())
    }
}

impl<W: Write + std::fmt::Debug + Send> ProtocolBackend for KittyBackend<W> {
    fn transmit(
        &mut self,
        pixmap: &Pixmap,
        position: TerminalPosition,
        z_index: i32,
    ) -> Result<ImageHandle, PixelCanvasError> {
        let image_id = Self::next_id();

        match self.format {
            TransmitFormat::ZlibRgba => {
                self.send_zlib_rgba(image_id, pixmap, position, z_index)?;
            }
            TransmitFormat::RawRgba => {
                self.send_raw_rgba(image_id, pixmap, position, z_index)?;
            }
            TransmitFormat::Png => {
                let png_data = Self::encode_png(pixmap)?;
                self.send_chunked(image_id, &png_data, position, z_index)?;
            }
            #[cfg(feature = "shm")]
            TransmitFormat::SharedMemory => {
                self.send_shm(image_id, pixmap, position, z_index)?;
            }
        }
        self.active_ids.push(image_id);

        Ok(ImageHandle {
            id: image_id,
            protocol: ProtocolKind::Kitty,
        })
    }

    fn remove(&mut self, handle: &ImageHandle) -> Result<(), PixelCanvasError> {
        self.send_delete(handle.id)?;
        self.active_ids.retain(|&id| id != handle.id);
        Ok(())
    }

    fn clear_all(&mut self) -> Result<(), PixelCanvasError> {
        // Delete all images managed by this backend
        let ids: Vec<u32> = self.active_ids.drain(..).collect();
        for id in ids {
            self.send_delete(id)?;
        }
        Ok(())
    }

    fn replace(
        &mut self,
        handle: &ImageHandle,
        pixmap: &Pixmap,
        position: TerminalPosition,
        z_index: i32,
    ) -> Result<ImageHandle, PixelCanvasError> {
        // Reuse the existing image ID — Kitty atomically replaces the pixel
        // data for an ID when you transmit with a=T and the same i=.
        let image_id = handle.id;

        match self.format {
            TransmitFormat::ZlibRgba => {
                self.send_zlib_rgba(image_id, pixmap, position, z_index)?;
            }
            TransmitFormat::RawRgba => {
                self.send_raw_rgba(image_id, pixmap, position, z_index)?;
            }
            TransmitFormat::Png => {
                let png_data = Self::encode_png(pixmap)?;
                self.send_chunked(image_id, &png_data, position, z_index)?;
            }
            #[cfg(feature = "shm")]
            TransmitFormat::SharedMemory => {
                self.send_shm(image_id, pixmap, position, z_index)?;
            }
        }

        Ok(ImageHandle {
            id: image_id,
            protocol: ProtocolKind::Kitty,
        })
    }

    fn transmit_tiles(
        &mut self,
        handle: &ImageHandle,
        pixmap: &Pixmap,
        position: TerminalPosition,
        z_index: i32,
        dirty_tiles: &[crate::rasterize::DirtyTile],
    ) -> Result<ImageHandle, PixelCanvasError> {
        if dirty_tiles.is_empty() {
            // Nothing changed — return existing handle
            return Ok(ImageHandle {
                id: handle.id,
                protocol: ProtocolKind::Kitty,
            });
        }

        self.send_tiles_zlib_rgba(pixmap, position, z_index, dirty_tiles)?;

        Ok(ImageHandle {
            id: handle.id,
            protocol: ProtocolKind::Kitty,
        })
    }

    fn supports_alpha(&self) -> bool {
        true
    }

    fn protocol_kind(&self) -> ProtocolKind {
        ProtocolKind::Kitty
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_backend() -> KittyBackend<Vec<u8>> {
        KittyBackend::with_writer(Vec::new(), FontSize::default())
    }

    #[test]
    fn transmit_produces_escape_sequence() {
        let mut backend = test_backend();
        let pixmap = Pixmap::new(10, 10).unwrap();
        let pos = TerminalPosition::new(0, 0, 5, 5);

        let handle = backend.transmit(&pixmap, pos, 0).unwrap();
        assert_eq!(handle.protocol(), ProtocolKind::Kitty);

        // Check that output contains Kitty escape sequence markers
        let output = String::from_utf8_lossy(&backend.writer);
        assert!(output.contains("\x1b_G"));
        assert!(output.contains("\x1b\\"));
        assert!(output.contains("a=T"));
        assert!(output.contains("f=32")); // RGBA pixel format
        assert!(
            output.contains("o=z"),
            "default format should use zlib compression"
        );
        assert!(output.contains("p=1"), "should include placement ID");
        // Synchronized output wrapping
        assert!(
            output.contains("\x1b[?2026h"),
            "should begin synchronized update"
        );
        assert!(
            output.contains("\x1b[?2026l"),
            "should end synchronized update"
        );
    }

    #[test]
    fn remove_sends_delete_command() {
        let mut backend = test_backend();
        let pixmap = Pixmap::new(10, 10).unwrap();
        let pos = TerminalPosition::new(0, 0, 5, 5);

        let handle = backend.transmit(&pixmap, pos, 0).unwrap();
        let id = handle.id();

        backend.writer.clear();
        backend.remove(&handle).unwrap();

        let output = String::from_utf8_lossy(&backend.writer);
        assert!(output.contains("a=d"));
        assert!(output.contains(&format!("i={id}")));
    }

    #[test]
    fn clear_all_removes_all_images() {
        let mut backend = test_backend();
        let pixmap = Pixmap::new(10, 10).unwrap();
        let pos = TerminalPosition::new(0, 0, 5, 5);

        backend.transmit(&pixmap, pos, 0).unwrap();
        backend.transmit(&pixmap, pos, 0).unwrap();
        assert_eq!(backend.active_ids.len(), 2);

        backend.clear_all().unwrap();
        assert!(backend.active_ids.is_empty());
    }

    #[test]
    fn supports_alpha() {
        let backend = test_backend();
        assert!(backend.supports_alpha());
    }

    #[test]
    fn replace_reuses_image_id() {
        let mut backend = test_backend();
        let pixmap = Pixmap::new(10, 10).unwrap();
        let pos = TerminalPosition::new(0, 0, 5, 5);

        // Transmit initial image
        let handle = backend.transmit(&pixmap, pos, 0).unwrap();
        let original_id = handle.id();

        // Replace with new data
        backend.writer.clear();
        let new_handle = backend.replace(&handle, &pixmap, pos, 0).unwrap();

        // Must reuse the same image ID
        assert_eq!(new_handle.id(), original_id);

        // Must NOT contain a delete command
        let output = String::from_utf8_lossy(&backend.writer);
        assert!(
            !output.contains("a=d"),
            "replace should not emit a delete command"
        );
        assert!(
            output.contains("a=T"),
            "replace should emit a transmit command"
        );
        assert!(output.contains(&format!("i={original_id}")));
        assert!(
            output.contains("p=1"),
            "replace should include placement id for atomic swap"
        );
    }
}
