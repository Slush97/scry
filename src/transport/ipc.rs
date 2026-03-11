// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared-memory IPC protocol for native scry-terminal communication.
//!
//! This module defines the wire protocol used between `scry-cli` (the client)
//! and `scry-terminal` (the server) for zero-copy image overlay management.
//!
//! # Architecture
//!
//! - **Bulk data**: Raw RGBA pixels live in anonymous shared memory created
//!   via `memfd_create`. The file descriptor is passed over a Unix socket
//!   using `SCM_RIGHTS`.
//! - **Control messages**: Tiny fixed-size commands/responses/events flow
//!   over the same Unix socket as regular data.
//!
//! # Wire Format
//!
//! All messages use: `[2-byte LE length][1-byte tag][payload]`.
//! Tags 0x00–0x7F are responses, 0x80–0xBF are commands, 0xC0–0xFF are events.
//!
//! # Feature Gate
//!
//! This module requires the `native-ipc` feature.

#![allow(unsafe_code)]

use std::io::{self, Read, Write};
use std::os::unix::io::{AsRawFd, RawFd};
use std::os::unix::net::UnixStream;

// ---------------------------------------------------------------------------
// Overlay anchor
// ---------------------------------------------------------------------------

/// Where an overlay is anchored in the terminal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverlayAnchor {
    /// Anchored to an absolute grid position — scrolls with terminal content.
    /// Used by `scry chart`, `scry see`, etc.
    Canvas {
        /// Column (0-indexed).
        col: u16,
        /// Absolute row in the scrollback buffer (can be negative for history).
        row: i64,
    },
    /// Pinned to the visible viewport — stays fixed regardless of scrolling.
    /// Used for border animations, HUD elements, decorations.
    Viewport {
        /// Column relative to viewport (0-indexed).
        col: u16,
        /// Row relative to viewport (0-indexed).
        row: u16,
    },
}

// ---------------------------------------------------------------------------
// Commands (client → server)
// ---------------------------------------------------------------------------

/// Control commands sent from scry-cli to scry-terminal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IpcCommand {
    /// Transmit a new overlay image.
    /// The accompanying `memfd` is passed out-of-band via `SCM_RIGHTS`.
    Transmit {
        /// Client-assigned overlay ID.
        id: u32,
        /// Where to place the overlay.
        anchor: OverlayAnchor,
        /// Width in terminal cells.
        w_cells: u16,
        /// Height in terminal cells.
        h_cells: u16,
        /// Z-index for compositing order.
        z: i32,
        /// Pixel width of the RGBA buffer.
        px_w: u32,
        /// Pixel height of the RGBA buffer.
        px_h: u32,
        /// If true, the overlay survives client disconnection.
        /// Used for static images (charts, `scry render`) that should
        /// remain visible after the CLI process exits.
        persist: bool,
    },
    /// Re-read the existing memfd (data updated in-place). Very fast — no fd transfer.
    Refresh {
        /// Overlay ID to refresh.
        id: u32,
    },
    /// Remove an overlay.
    Remove {
        /// Overlay ID to remove.
        id: u32,
    },
    /// Remove all overlays from this client.
    ClearAll,
    /// Query terminal info (font size, grid dimensions).
    QueryInfo,
    /// Submit a serialized scene graph for server-side GPU rendering.
    ///
    /// The terminal will rasterize the scene on its GPU — no pixel data
    /// is transferred. The scene bytes are passed via `memfd` (same as
    /// `Transmit`), with `scene_len` indicating how many bytes to read.
    SubmitScene {
        /// Client-assigned overlay ID.
        id: u32,
        /// Where to place the overlay.
        anchor: OverlayAnchor,
        /// Z-index for compositing order.
        z_index: i32,
        /// Size of the serialized scene data in the accompanying memfd.
        scene_len: u32,
        /// If true, the overlay survives client disconnection.
        persist: bool,
    },
    /// Submit an animation program for terminal-autonomous rendering.
    ///
    /// The terminal deserializes the `AnimationProgram` from the memfd,
    /// then drives the animation in its own render loop. The CLI can exit
    /// immediately after submission.
    SubmitAnimation {
        /// Client-assigned overlay ID.
        id: u32,
        /// Where to place the overlay.
        anchor: OverlayAnchor,
        /// Z-index for compositing order.
        z_index: i32,
        /// Size of the serialized `AnimationProgram` data in the memfd.
        program_len: u32,
        /// Duration in seconds (0 = infinite loop).
        duration_secs: u32,
        /// Target frames per second.
        fps: u32,
        /// Pixel width for SDF rendering.
        width: u32,
        /// Pixel height for SDF rendering.
        height: u32,
        /// If true, the overlay survives client disconnection.
        persist: bool,
    },
}

// ---------------------------------------------------------------------------
// Responses (server → client, synchronous)
// ---------------------------------------------------------------------------

/// Synchronous responses from scry-terminal to scry-cli.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IpcResponse {
    /// Command succeeded.
    Ok {
        /// The overlay ID that was affected (0 for `ClearAll`).
        id: u32,
    },
    /// Terminal information.
    Info {
        /// Font cell width in pixels.
        font_w: u16,
        /// Font cell height in pixels.
        font_h: u16,
        /// Grid columns.
        cols: u16,
        /// Grid rows.
        rows: u16,
    },
    /// Command failed.
    Error {
        /// Human-readable error message.
        msg: String,
    },
}

// ---------------------------------------------------------------------------
// Events (server → client, asynchronous / pushed)
// ---------------------------------------------------------------------------

/// Asynchronous events pushed from scry-terminal to scry-cli.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IpcEvent {
    /// User clicked on the overlay (toggle pause).
    Clicked {
        /// Overlay ID that was clicked.
        id: u32,
    },
    /// Overlay scrolled in or out of the viewport.
    Visibility {
        /// Overlay ID.
        id: u32,
        /// `true` if the overlay is now visible.
        visible: bool,
    },
}

// ---------------------------------------------------------------------------
// Wire tags
// ---------------------------------------------------------------------------

// Response tags: 0x00–0x7F
const TAG_RESP_OK: u8 = 0x01;
const TAG_RESP_INFO: u8 = 0x02;
const TAG_RESP_ERROR: u8 = 0x03;

// Command tags: 0x80–0xBF
const TAG_CMD_TRANSMIT: u8 = 0x80;
const TAG_CMD_REFRESH: u8 = 0x81;
const TAG_CMD_REMOVE: u8 = 0x82;
const TAG_CMD_CLEAR_ALL: u8 = 0x83;
const TAG_CMD_QUERY_INFO: u8 = 0x84;
const TAG_CMD_SUBMIT_SCENE: u8 = 0x85;
const TAG_CMD_SUBMIT_ANIMATION: u8 = 0x86;

// Event tags: 0xC0–0xFF
const TAG_EVT_CLICKED: u8 = 0xC0;
const TAG_EVT_VISIBILITY: u8 = 0xC1;

// Anchor sub-tags
const ANCHOR_CANVAS: u8 = 0x00;
const ANCHOR_VIEWPORT: u8 = 0x01;

// ---------------------------------------------------------------------------
// Memfd — anonymous shared memory
// ---------------------------------------------------------------------------

/// Anonymous shared memory buffer backed by `memfd_create`.
///
/// The fd can be passed to another process via `SCM_RIGHTS`. Both processes
/// then `mmap` the same physical pages — true zero-copy.
pub struct Memfd {
    /// File descriptor (owned).
    fd: RawFd,
    /// Memory-mapped pointer.
    ptr: *mut u8,
    /// Mapping size in bytes.
    len: usize,
}

impl std::fmt::Debug for Memfd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Memfd")
            .field("fd", &self.fd)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

// SAFETY: Memfd owns its fd and mapping exclusively.
unsafe impl Send for Memfd {}

impl Memfd {
    /// Create a new anonymous shared memory region.
    ///
    /// Uses `memfd_create` (Linux 3.17+) for an fd-backed anonymous mapping.
    /// The name is purely for debugging (`/proc/pid/fd/` symlink).
    pub fn create(name: &str, size: usize) -> io::Result<Self> {
        let c_name = std::ffi::CString::new(name)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;

        // SAFETY: memfd_create creates an anonymous file and returns an fd.
        let fd = unsafe { libc::memfd_create(c_name.as_ptr(), libc::MFD_CLOEXEC) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }

        // Size the region.
        #[allow(clippy::cast_possible_wrap)]
        let ret = unsafe { libc::ftruncate(fd, size as libc::off_t) };
        if ret < 0 {
            let err = io::Error::last_os_error();
            unsafe { libc::close(fd) };
            return Err(err);
        }

        // Map it.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            let err = io::Error::last_os_error();
            unsafe { libc::close(fd) };
            return Err(err);
        }

        Ok(Self {
            fd,
            ptr: ptr.cast::<u8>(),
            len: size,
        })
    }

    /// Open a received fd as a read-only shared memory mapping.
    ///
    /// Used by the terminal side after receiving an fd via `SCM_RIGHTS`.
    pub fn from_fd(fd: RawFd, size: usize) -> io::Result<Self> {
        // Validate that the fd's actual size is at least `size` bytes.
        // Prevents SIGBUS from mapping beyond the file's backing storage.
        Self::validate_fd_size(fd, size)?;

        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        Ok(Self {
            fd,
            ptr: ptr.cast::<u8>(),
            len: size,
        })
    }

    /// Validate that this fd's actual backing size is at least `expected` bytes.
    ///
    /// Uses `fstat` to check — prevents mapping more memory than the fd
    /// actually backs, which would cause SIGBUS on access.
    fn validate_fd_size(fd: RawFd, expected: usize) -> io::Result<()> {
        // SAFETY: `fstat` is a well-defined POSIX call; stat is zero-initialized.
        let mut stat: libc::stat = unsafe { std::mem::zeroed() };
        let ret = unsafe { libc::fstat(fd, &mut stat) };
        if ret != 0 {
            return Err(io::Error::last_os_error());
        }
        let actual = stat.st_size as usize;
        if actual < expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("memfd size mismatch: fd has {actual} bytes, expected {expected}"),
            ));
        }
        Ok(())
    }

    /// Write pixel data into the shared memory region.
    ///
    /// # Errors
    ///
    /// Returns an error if `data.len()` exceeds the buffer capacity.
    pub fn write(&self, data: &[u8]) -> io::Result<()> {
        if data.len() > self.len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Memfd::write: data ({}) exceeds capacity ({})",
                    data.len(),
                    self.len
                ),
            ));
        }
        // SAFETY: ptr is valid and exclusively managed, data fits within bounds.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr, data.len());
        }
        Ok(())
    }

    /// Read-only view of the mapped memory.
    ///
    /// # Safety
    ///
    /// The caller must ensure no other process is concurrently writing to the
    /// same region while this slice is live.
    pub(crate) unsafe fn as_slice(&self) -> &[u8] {
        std::slice::from_raw_parts(self.ptr, self.len)
    }

    /// Copy the mapped memory into a `Vec<u8>`.
    ///
    /// Safe alternative to [`as_slice`](Self::as_slice) — produces a snapshot
    /// that is immune to concurrent writes from the client process.
    /// The slight allocation cost is negligible relative to the GPU upload that
    /// typically follows this call.
    #[must_use]
    pub fn read_bytes(&self) -> Vec<u8> {
        // SAFETY: ptr is valid for `self.len` bytes and we copy immediately.
        // Using `unsafe` here is confined to this safe wrapper so callers
        // (e.g. scry-terminal which has `deny(unsafe_code)`) can call this freely.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len).to_vec() }
    }

    /// The file descriptor (for passing via `SCM_RIGHTS`).
    pub fn as_raw_fd(&self) -> RawFd {
        self.fd
    }

    /// Size of the mapping in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the mapping is zero-sized.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for Memfd {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                libc::munmap(self.ptr.cast(), self.len);
            }
            libc::close(self.fd);
        }
    }
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

/// Write a `u16` as little-endian.
fn put_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// Write a `u32` as little-endian.
fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// Write an `i32` as little-endian.
fn put_i32(buf: &mut Vec<u8>, v: i32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// Write an `i64` as little-endian.
fn put_i64(buf: &mut Vec<u8>, v: i64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// Read a `u16` from a byte slice at the given offset.
fn get_u16(data: &[u8], off: usize) -> io::Result<u16> {
    let bytes: [u8; 2] = data
        .get(off..off + 2)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "truncated u16"))?;
    Ok(u16::from_le_bytes(bytes))
}

/// Read a `u32` from a byte slice at the given offset.
fn get_u32(data: &[u8], off: usize) -> io::Result<u32> {
    let bytes: [u8; 4] = data
        .get(off..off + 4)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "truncated u32"))?;
    Ok(u32::from_le_bytes(bytes))
}

/// Read an `i32` from a byte slice at the given offset.
fn get_i32(data: &[u8], off: usize) -> io::Result<i32> {
    let bytes: [u8; 4] = data
        .get(off..off + 4)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "truncated i32"))?;
    Ok(i32::from_le_bytes(bytes))
}

/// Read an `i64` from a byte slice at the given offset.
fn get_i64(data: &[u8], off: usize) -> io::Result<i64> {
    let bytes: [u8; 8] = data
        .get(off..off + 8)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "truncated i64"))?;
    Ok(i64::from_le_bytes(bytes))
}

// ---------------------------------------------------------------------------
// Command serialization
// ---------------------------------------------------------------------------

impl IpcCommand {
    /// Serialize a command to wire format.
    ///
    /// Returns the complete frame: `[2-byte LE length][1-byte tag][payload]`.
    #[must_use]
    pub fn serialize(&self) -> Vec<u8> {
        let mut payload = Vec::with_capacity(64);

        match self {
            Self::Transmit {
                id,
                anchor,
                w_cells,
                h_cells,
                z,
                px_w,
                px_h,
                persist,
            } => {
                payload.push(TAG_CMD_TRANSMIT);
                put_u32(&mut payload, *id);
                match anchor {
                    OverlayAnchor::Canvas { col, row } => {
                        payload.push(ANCHOR_CANVAS);
                        put_u16(&mut payload, *col);
                        put_i64(&mut payload, *row);
                    }
                    OverlayAnchor::Viewport { col, row } => {
                        payload.push(ANCHOR_VIEWPORT);
                        put_u16(&mut payload, *col);
                        put_u16(&mut payload, *row);
                    }
                }
                put_u16(&mut payload, *w_cells);
                put_u16(&mut payload, *h_cells);
                put_i32(&mut payload, *z);
                put_u32(&mut payload, *px_w);
                put_u32(&mut payload, *px_h);
                payload.push(u8::from(*persist));
            }
            Self::Refresh { id } => {
                payload.push(TAG_CMD_REFRESH);
                put_u32(&mut payload, *id);
            }
            Self::Remove { id } => {
                payload.push(TAG_CMD_REMOVE);
                put_u32(&mut payload, *id);
            }
            Self::ClearAll => {
                payload.push(TAG_CMD_CLEAR_ALL);
            }
            Self::QueryInfo => {
                payload.push(TAG_CMD_QUERY_INFO);
            }
            Self::SubmitScene {
                id,
                anchor,
                z_index,
                scene_len,
                persist,
            } => {
                payload.push(TAG_CMD_SUBMIT_SCENE);
                put_u32(&mut payload, *id);
                match anchor {
                    OverlayAnchor::Canvas { col, row } => {
                        payload.push(ANCHOR_CANVAS);
                        put_u16(&mut payload, *col);
                        put_i64(&mut payload, *row);
                    }
                    OverlayAnchor::Viewport { col, row } => {
                        payload.push(ANCHOR_VIEWPORT);
                        put_u16(&mut payload, *col);
                        put_u16(&mut payload, *row);
                    }
                }
                put_i32(&mut payload, *z_index);
                put_u32(&mut payload, *scene_len);
                payload.push(u8::from(*persist));
            }
            Self::SubmitAnimation {
                id,
                anchor,
                z_index,
                program_len,
                duration_secs,
                fps,
                width,
                height,
                persist,
            } => {
                payload.push(TAG_CMD_SUBMIT_ANIMATION);
                put_u32(&mut payload, *id);
                match anchor {
                    OverlayAnchor::Canvas { col, row } => {
                        payload.push(ANCHOR_CANVAS);
                        put_u16(&mut payload, *col);
                        put_i64(&mut payload, *row);
                    }
                    OverlayAnchor::Viewport { col, row } => {
                        payload.push(ANCHOR_VIEWPORT);
                        put_u16(&mut payload, *col);
                        put_u16(&mut payload, *row);
                    }
                }
                put_i32(&mut payload, *z_index);
                put_u32(&mut payload, *program_len);
                put_u32(&mut payload, *duration_secs);
                put_u32(&mut payload, *fps);
                put_u32(&mut payload, *width);
                put_u32(&mut payload, *height);
                payload.push(u8::from(*persist));
            }
        }

        // Prepend 2-byte LE length (of payload, not including the length field itself).
        let len = payload.len() as u16;
        let mut frame = Vec::with_capacity(2 + payload.len());
        frame.extend_from_slice(&len.to_le_bytes());
        frame.extend_from_slice(&payload);
        frame
    }

    /// Deserialize a command from the payload (after length prefix has been read).
    pub fn deserialize(payload: &[u8]) -> io::Result<Self> {
        if payload.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "empty command payload",
            ));
        }

        let tag = payload[0];
        let data = &payload[1..];

        match tag {
            TAG_CMD_TRANSMIT => {
                // id(4) + anchor_tag(1) + anchor_data(variable) + w_cells(2) + h_cells(2) + z(4) + px_w(4) + px_h(4)
                let id = get_u32(data, 0)?;
                let anchor_tag = *data.get(4).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::UnexpectedEof, "missing anchor tag")
                })?;

                let (anchor, rest_offset) = match anchor_tag {
                    ANCHOR_CANVAS => {
                        let col = get_u16(data, 5)?;
                        let row = get_i64(data, 7)?;
                        (OverlayAnchor::Canvas { col, row }, 15)
                    }
                    ANCHOR_VIEWPORT => {
                        let col = get_u16(data, 5)?;
                        let row = get_u16(data, 7)?;
                        (OverlayAnchor::Viewport { col, row }, 9)
                    }
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("unknown anchor tag: {anchor_tag:#x}"),
                        ));
                    }
                };

                let w_cells = get_u16(data, rest_offset)?;
                let h_cells = get_u16(data, rest_offset + 2)?;
                let z = get_i32(data, rest_offset + 4)?;
                let px_w = get_u32(data, rest_offset + 8)?;
                let px_h = get_u32(data, rest_offset + 12)?;
                let persist = data.get(rest_offset + 16).copied().unwrap_or(0) != 0;

                Ok(Self::Transmit {
                    id,
                    anchor,
                    w_cells,
                    h_cells,
                    z,
                    px_w,
                    px_h,
                    persist,
                })
            }
            TAG_CMD_REFRESH => {
                let id = get_u32(data, 0)?;
                Ok(Self::Refresh { id })
            }
            TAG_CMD_REMOVE => {
                let id = get_u32(data, 0)?;
                Ok(Self::Remove { id })
            }
            TAG_CMD_CLEAR_ALL => Ok(Self::ClearAll),
            TAG_CMD_QUERY_INFO => Ok(Self::QueryInfo),
            TAG_CMD_SUBMIT_SCENE => {
                let id = get_u32(data, 0)?;
                let anchor_tag = *data.get(4).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::UnexpectedEof, "missing anchor tag")
                })?;

                let (anchor, rest_offset) = match anchor_tag {
                    ANCHOR_CANVAS => {
                        let col = get_u16(data, 5)?;
                        let row = get_i64(data, 7)?;
                        (OverlayAnchor::Canvas { col, row }, 15)
                    }
                    ANCHOR_VIEWPORT => {
                        let col = get_u16(data, 5)?;
                        let row = get_u16(data, 7)?;
                        (OverlayAnchor::Viewport { col, row }, 9)
                    }
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("unknown anchor tag: {anchor_tag:#x}"),
                        ));
                    }
                };

                let z_index = get_i32(data, rest_offset)?;
                let scene_len = get_u32(data, rest_offset + 4)?;
                let persist = data.get(rest_offset + 8).copied().unwrap_or(0) != 0;

                Ok(Self::SubmitScene {
                    id,
                    anchor,
                    z_index,
                    scene_len,
                    persist,
                })
            }
            TAG_CMD_SUBMIT_ANIMATION => {
                let id = get_u32(data, 0)?;
                let anchor_tag = *data.get(4).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::UnexpectedEof, "missing anchor tag")
                })?;

                let (anchor, rest_offset) = match anchor_tag {
                    ANCHOR_CANVAS => {
                        let col = get_u16(data, 5)?;
                        let row = get_i64(data, 7)?;
                        (OverlayAnchor::Canvas { col, row }, 15)
                    }
                    ANCHOR_VIEWPORT => {
                        let col = get_u16(data, 5)?;
                        let row = get_u16(data, 7)?;
                        (OverlayAnchor::Viewport { col, row }, 9)
                    }
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("unknown anchor tag: {anchor_tag:#x}"),
                        ));
                    }
                };

                let z_index = get_i32(data, rest_offset)?;
                let program_len = get_u32(data, rest_offset + 4)?;
                let duration_secs = get_u32(data, rest_offset + 8)?;
                let fps = get_u32(data, rest_offset + 12)?;
                let width = get_u32(data, rest_offset + 16)?;
                let height = get_u32(data, rest_offset + 20)?;
                let persist = data.get(rest_offset + 24).copied().unwrap_or(0) != 0;

                Ok(Self::SubmitAnimation {
                    id,
                    anchor,
                    z_index,
                    program_len,
                    duration_secs,
                    fps,
                    width,
                    height,
                    persist,
                })
            }

            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown command tag: {tag:#x}"),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Response serialization
// ---------------------------------------------------------------------------

impl IpcResponse {
    /// Serialize a response to wire format.
    #[must_use]
    pub fn serialize(&self) -> Vec<u8> {
        let mut payload = Vec::with_capacity(32);

        match self {
            Self::Ok { id } => {
                payload.push(TAG_RESP_OK);
                put_u32(&mut payload, *id);
            }
            Self::Info {
                font_w,
                font_h,
                cols,
                rows,
            } => {
                payload.push(TAG_RESP_INFO);
                put_u16(&mut payload, *font_w);
                put_u16(&mut payload, *font_h);
                put_u16(&mut payload, *cols);
                put_u16(&mut payload, *rows);
            }
            Self::Error { msg } => {
                payload.push(TAG_RESP_ERROR);
                let bytes = msg.as_bytes();
                put_u16(&mut payload, bytes.len() as u16);
                payload.extend_from_slice(bytes);
            }
        }

        let len = payload.len() as u16;
        let mut frame = Vec::with_capacity(2 + payload.len());
        frame.extend_from_slice(&len.to_le_bytes());
        frame.extend_from_slice(&payload);
        frame
    }

    /// Deserialize a response from the payload.
    pub fn deserialize(payload: &[u8]) -> io::Result<Self> {
        if payload.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "empty response payload",
            ));
        }

        let tag = payload[0];
        let data = &payload[1..];

        match tag {
            TAG_RESP_OK => {
                let id = get_u32(data, 0)?;
                Ok(Self::Ok { id })
            }
            TAG_RESP_INFO => {
                let font_w = get_u16(data, 0)?;
                let font_h = get_u16(data, 2)?;
                let cols = get_u16(data, 4)?;
                let rows = get_u16(data, 6)?;
                Ok(Self::Info {
                    font_w,
                    font_h,
                    cols,
                    rows,
                })
            }
            TAG_RESP_ERROR => {
                let msg_len = get_u16(data, 0)? as usize;
                let bytes = data.get(2..2 + msg_len).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::UnexpectedEof, "truncated error message")
                })?;
                let msg = String::from_utf8_lossy(bytes).into_owned();
                Ok(Self::Error { msg })
            }
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown response tag: {tag:#x}"),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Event serialization
// ---------------------------------------------------------------------------

impl IpcEvent {
    /// Serialize an event to wire format.
    #[must_use]
    pub fn serialize(&self) -> Vec<u8> {
        let mut payload = Vec::with_capacity(16);

        match self {
            Self::Clicked { id } => {
                payload.push(TAG_EVT_CLICKED);
                put_u32(&mut payload, *id);
            }
            Self::Visibility { id, visible } => {
                payload.push(TAG_EVT_VISIBILITY);
                put_u32(&mut payload, *id);
                payload.push(u8::from(*visible));
            }
        }

        let len = payload.len() as u16;
        let mut frame = Vec::with_capacity(2 + payload.len());
        frame.extend_from_slice(&len.to_le_bytes());
        frame.extend_from_slice(&payload);
        frame
    }

    /// Deserialize an event from the payload.
    pub fn deserialize(payload: &[u8]) -> io::Result<Self> {
        if payload.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "empty event payload",
            ));
        }

        let tag = payload[0];
        let data = &payload[1..];

        match tag {
            TAG_EVT_CLICKED => {
                let id = get_u32(data, 0)?;
                Ok(Self::Clicked { id })
            }
            TAG_EVT_VISIBILITY => {
                let id = get_u32(data, 0)?;
                let visible = *data.get(4).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::UnexpectedEof, "missing visibility flag")
                })? != 0;
                Ok(Self::Visibility { id, visible })
            }
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown event tag: {tag:#x}"),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Socket I/O helpers
// ---------------------------------------------------------------------------

/// Read a complete framed message from a Unix socket.
///
/// Reads the 2-byte length prefix, then reads exactly that many bytes.
/// Returns the payload (tag + data).
pub fn read_frame(stream: &mut UnixStream) -> io::Result<Vec<u8>> {
    let mut len_buf = [0u8; 2];
    stream.read_exact(&mut len_buf)?;
    let len = u16::from_le_bytes(len_buf) as usize;

    if len == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "zero-length frame",
        ));
    }
    if len > 65535 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame too large: {len}"),
        ));
    }

    let mut payload = vec![0u8; len];
    stream.read_exact(&mut payload)?;
    Ok(payload)
}

/// Write a complete framed message to a Unix socket.
///
/// The frame should already include the 2-byte length prefix.
pub fn write_frame(stream: &mut UnixStream, frame: &[u8]) -> io::Result<()> {
    stream.write_all(frame)?;
    stream.flush()
}

/// Send a command with an optional file descriptor via `SCM_RIGHTS`.
///
/// The command is sent as a regular framed message. If `fd` is provided,
/// it is attached as ancillary data using `sendmsg`.
pub fn send_command_with_fd(
    stream: &mut UnixStream,
    cmd: &IpcCommand,
    fd: Option<RawFd>,
) -> io::Result<()> {
    let frame = cmd.serialize();

    if let Some(fd) = fd {
        send_with_fd(stream.as_raw_fd(), &frame, fd)
    } else {
        write_frame(stream, &frame)
    }
}

/// Receive a command with an optional file descriptor.
///
/// Returns `(command, Option<RawFd>)`. The caller owns the fd if present.
pub fn recv_command_with_fd(stream: &mut UnixStream) -> io::Result<(IpcCommand, Option<RawFd>)> {
    let (data, fd) = recv_with_fd(stream.as_raw_fd())?;

    // The data includes the 2-byte length prefix — skip it.
    if data.len() < 2 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "frame too short",
        ));
    }
    let payload = &data[2..];
    let cmd = IpcCommand::deserialize(payload)?;
    Ok((cmd, fd))
}

/// Send a response (no fd).
pub fn send_response(stream: &mut UnixStream, resp: &IpcResponse) -> io::Result<()> {
    write_frame(stream, &resp.serialize())
}

/// Receive a response (no fd).
pub fn recv_response(stream: &mut UnixStream) -> io::Result<IpcResponse> {
    let payload = read_frame(stream)?;
    IpcResponse::deserialize(&payload)
}

/// Send an event (no fd, pushed by terminal).
pub fn send_event(stream: &mut UnixStream, event: &IpcEvent) -> io::Result<()> {
    write_frame(stream, &event.serialize())
}

/// Try to receive a response or event, non-blocking.
///
/// Returns `None` if no data is available (would block).
/// Returns `Ok(Some(payload))` if a frame was read.
pub fn try_recv_frame(stream: &mut UnixStream) -> io::Result<Option<Vec<u8>>> {
    stream.set_nonblocking(true)?;
    let result = read_frame(stream);
    stream.set_nonblocking(false)?;

    match result {
        Ok(payload) => Ok(Some(payload)),
        Err(e) if e.kind() == io::ErrorKind::WouldBlock => Ok(None),
        Err(e) => Err(e),
    }
}

/// Classify a payload tag: is it a response, command, or event?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageKind {
    /// Response (tag 0x00–0x7F).
    Response,
    /// Command (tag 0x80–0xBF).
    Command,
    /// Event (tag 0xC0–0xFF).
    Event,
}

/// Classify a raw payload by its tag byte.
pub fn classify_payload(payload: &[u8]) -> Option<MessageKind> {
    let tag = *payload.first()?;
    Some(match tag {
        0x00..=0x7F => MessageKind::Response,
        0x80..=0xBF => MessageKind::Command,
        0xC0..=0xFF => MessageKind::Event,
    })
}

// ---------------------------------------------------------------------------
// Low-level fd passing (SCM_RIGHTS)
// ---------------------------------------------------------------------------

/// Send data + a file descriptor over a Unix socket using `sendmsg` + `SCM_RIGHTS`.
fn send_with_fd(sock_fd: RawFd, data: &[u8], fd: RawFd) -> io::Result<()> {
    // Build the iovec for the data.
    let iov = libc::iovec {
        iov_base: data.as_ptr() as *mut libc::c_void,
        iov_len: data.len(),
    };

    // Build the cmsg buffer for SCM_RIGHTS with one fd.
    // The buffer must be large enough for: cmsghdr + one RawFd.
    let cmsg_space = unsafe { libc::CMSG_SPACE(std::mem::size_of::<RawFd>() as u32) } as usize;
    let mut cmsg_buf = vec![0u8; cmsg_space];

    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &iov as *const libc::iovec as *mut libc::iovec;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf.as_mut_ptr().cast();
    msg.msg_controllen = cmsg_space;

    // Fill the cmsg header.
    let cmsg: *mut libc::cmsghdr = unsafe { libc::CMSG_FIRSTHDR(&msg) };
    if cmsg.is_null() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "CMSG_FIRSTHDR returned null",
        ));
    }
    unsafe {
        (*cmsg).cmsg_level = libc::SOL_SOCKET;
        (*cmsg).cmsg_type = libc::SCM_RIGHTS;
        (*cmsg).cmsg_len = libc::CMSG_LEN(std::mem::size_of::<RawFd>() as u32) as usize;

        // Copy the fd into the cmsg data area.
        let fd_ptr = libc::CMSG_DATA(cmsg).cast::<RawFd>();
        std::ptr::write_unaligned(fd_ptr, fd);
    }

    let ret = unsafe { libc::sendmsg(sock_fd, &msg, 0) };
    if ret < 0 {
        return Err(io::Error::last_os_error());
    }

    Ok(())
}

/// Receive data + optional file descriptor from a Unix socket using `recvmsg`.
fn recv_with_fd(sock_fd: RawFd) -> io::Result<(Vec<u8>, Option<RawFd>)> {
    // Max IPC frame is 2 (length prefix) + 65535 (payload) = 65537 bytes.
    // Use 66000 to detect truncation with margin.
    let mut data_buf = vec![0u8; 66_000];

    let mut iov = libc::iovec {
        iov_base: data_buf.as_mut_ptr().cast(),
        iov_len: data_buf.len(),
    };

    let cmsg_space = unsafe { libc::CMSG_SPACE(std::mem::size_of::<RawFd>() as u32) } as usize;
    let mut cmsg_buf = vec![0u8; cmsg_space];

    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf.as_mut_ptr().cast();
    msg.msg_controllen = cmsg_space;

    let n = unsafe { libc::recvmsg(sock_fd, &mut msg, 0) };
    if n < 0 {
        return Err(io::Error::last_os_error());
    }
    if n == 0 {
        return Err(io::Error::new(
            io::ErrorKind::ConnectionReset,
            "peer disconnected",
        ));
    }

    data_buf.truncate(n as usize);

    // Detect truncation — reject messages that were too large for the buffer.
    if msg.msg_flags & libc::MSG_TRUNC != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "IPC message was truncated",
        ));
    }

    // Extract fd from ancillary data.
    let mut received_fd = None;
    let cmsg = unsafe { libc::CMSG_FIRSTHDR(&msg) };
    if !cmsg.is_null() {
        unsafe {
            if (*cmsg).cmsg_level == libc::SOL_SOCKET && (*cmsg).cmsg_type == libc::SCM_RIGHTS {
                let fd_ptr = libc::CMSG_DATA(cmsg).cast::<RawFd>();
                received_fd = Some(std::ptr::read_unaligned(fd_ptr));
            }
        }
    }

    Ok((data_buf, received_fd))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_transmit_canvas_round_trip() {
        let cmd = IpcCommand::Transmit {
            id: 42,
            anchor: OverlayAnchor::Canvas { col: 10, row: -5 },
            w_cells: 80,
            h_cells: 24,
            z: 1,
            px_w: 640,
            px_h: 480,
            persist: false,
        };
        let frame = cmd.serialize();
        // Skip 2-byte length prefix.
        let payload = &frame[2..];
        let decoded = IpcCommand::deserialize(payload).unwrap();
        assert_eq!(cmd, decoded);
    }

    #[test]
    fn command_transmit_viewport_round_trip() {
        let cmd = IpcCommand::Transmit {
            id: 1,
            anchor: OverlayAnchor::Viewport { col: 0, row: 0 },
            w_cells: 120,
            h_cells: 40,
            z: -1,
            px_w: 1920,
            px_h: 1080,
            persist: false,
        };
        let frame = cmd.serialize();
        let payload = &frame[2..];
        let decoded = IpcCommand::deserialize(payload).unwrap();
        assert_eq!(cmd, decoded);
    }

    #[test]
    fn command_refresh_round_trip() {
        let cmd = IpcCommand::Refresh { id: 7 };
        let frame = cmd.serialize();
        let decoded = IpcCommand::deserialize(&frame[2..]).unwrap();
        assert_eq!(cmd, decoded);
    }

    #[test]
    fn command_remove_round_trip() {
        let cmd = IpcCommand::Remove { id: 99 };
        let frame = cmd.serialize();
        let decoded = IpcCommand::deserialize(&frame[2..]).unwrap();
        assert_eq!(cmd, decoded);
    }

    #[test]
    fn command_clear_all_round_trip() {
        let frame = IpcCommand::ClearAll.serialize();
        let decoded = IpcCommand::deserialize(&frame[2..]).unwrap();
        assert_eq!(IpcCommand::ClearAll, decoded);
    }

    #[test]
    fn command_query_info_round_trip() {
        let frame = IpcCommand::QueryInfo.serialize();
        let decoded = IpcCommand::deserialize(&frame[2..]).unwrap();
        assert_eq!(IpcCommand::QueryInfo, decoded);
    }

    #[test]
    fn ipc_submit_scene_canvas_round_trip() {
        let cmd = IpcCommand::SubmitScene {
            id: 7,
            anchor: OverlayAnchor::Canvas { col: 5, row: -3 },
            z_index: 2,
            scene_len: 42_000,
            persist: true,
        };
        let frame = cmd.serialize();
        let decoded = IpcCommand::deserialize(&frame[2..]).unwrap();
        assert_eq!(cmd, decoded);
    }

    #[test]
    fn ipc_submit_scene_viewport_round_trip() {
        let cmd = IpcCommand::SubmitScene {
            id: 99,
            anchor: OverlayAnchor::Viewport { col: 0, row: 0 },
            z_index: -1,
            scene_len: 1024,
            persist: false,
        };
        let frame = cmd.serialize();
        let decoded = IpcCommand::deserialize(&frame[2..]).unwrap();
        assert_eq!(cmd, decoded);
    }

    #[test]
    fn response_ok_round_trip() {
        let resp = IpcResponse::Ok { id: 42 };
        let frame = resp.serialize();
        let decoded = IpcResponse::deserialize(&frame[2..]).unwrap();
        assert_eq!(resp, decoded);
    }

    #[test]
    fn response_info_round_trip() {
        let resp = IpcResponse::Info {
            font_w: 8,
            font_h: 16,
            cols: 120,
            rows: 40,
        };
        let frame = resp.serialize();
        let decoded = IpcResponse::deserialize(&frame[2..]).unwrap();
        assert_eq!(resp, decoded);
    }

    #[test]
    fn response_error_round_trip() {
        let resp = IpcResponse::Error {
            msg: "something went wrong".to_string(),
        };
        let frame = resp.serialize();
        let decoded = IpcResponse::deserialize(&frame[2..]).unwrap();
        assert_eq!(resp, decoded);
    }

    #[test]
    fn event_clicked_round_trip() {
        let event = IpcEvent::Clicked { id: 5 };
        let frame = event.serialize();
        let decoded = IpcEvent::deserialize(&frame[2..]).unwrap();
        assert_eq!(event, decoded);
    }

    #[test]
    fn event_visibility_round_trip() {
        for visible in [true, false] {
            let event = IpcEvent::Visibility { id: 3, visible };
            let frame = event.serialize();
            let decoded = IpcEvent::deserialize(&frame[2..]).unwrap();
            assert_eq!(event, decoded);
        }
    }

    #[test]
    fn classify_tags() {
        assert_eq!(
            classify_payload(&[TAG_RESP_OK, 0, 0, 0, 0]),
            Some(MessageKind::Response)
        );
        assert_eq!(
            classify_payload(&[TAG_CMD_TRANSMIT]),
            Some(MessageKind::Command)
        );
        assert_eq!(
            classify_payload(&[TAG_EVT_CLICKED, 0, 0, 0, 0]),
            Some(MessageKind::Event)
        );
        assert_eq!(classify_payload(&[]), None);
    }

    #[test]
    fn memfd_create_write_read() {
        let memfd = Memfd::create("scry-test-ipc", 1024).unwrap();
        assert_eq!(memfd.len(), 1024);
        assert!(!memfd.is_empty());

        let data = vec![0xAB_u8; 512];
        memfd.write(&data).unwrap();

        let slice = unsafe { memfd.as_slice() };
        assert!(slice[..512].iter().all(|&b| b == 0xAB));
    }

    #[test]
    fn memfd_write_overflow() {
        let memfd = Memfd::create("scry-test-overflow", 64).unwrap();
        let big = vec![0u8; 128];
        assert!(memfd.write(&big).is_err());
    }

    #[test]
    fn command_deserialize_invalid_tag() {
        assert!(IpcCommand::deserialize(&[0xFF]).is_err());
    }

    #[test]
    fn response_deserialize_empty() {
        assert!(IpcResponse::deserialize(&[]).is_err());
    }

    #[test]
    fn socket_fd_passing() {
        // Create a Unix socket pair.
        let (mut tx, mut rx) = UnixStream::pair().unwrap();

        // Create a memfd with test data.
        let memfd = Memfd::create("scry-test-fdpass", 256).unwrap();
        let test_data = b"hello from shared memory!";
        memfd.write(test_data).unwrap();

        // Send a Transmit command with the memfd fd.
        let cmd = IpcCommand::Transmit {
            id: 1,
            anchor: OverlayAnchor::Canvas { col: 0, row: 0 },
            w_cells: 10,
            h_cells: 5,
            z: 0,
            px_w: 100,
            px_h: 50,
            persist: false,
        };
        send_command_with_fd(&mut tx, &cmd, Some(memfd.as_raw_fd())).unwrap();

        // Receive the command + fd on the other end.
        let (received_cmd, received_fd) = recv_command_with_fd(&mut rx).unwrap();
        assert_eq!(received_cmd, cmd);
        assert!(received_fd.is_some());

        // Map the received fd and verify the data.
        let received_memfd = Memfd::from_fd(received_fd.unwrap(), 256).unwrap();
        let slice = unsafe { received_memfd.as_slice() };
        assert_eq!(&slice[..test_data.len()], test_data);
    }
}
