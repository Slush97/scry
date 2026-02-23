// SPDX-License-Identifier: MIT OR Apache-2.0
//! Native shared-memory backend for scry-terminal.
//!
//! When `scry` (the CLI) runs inside `scry-terminal`, this backend bypasses
//! all escape-sequence encoding and uses `memfd_create` + `SCM_RIGHTS` for
//! zero-copy pixel transfer via a Unix domain socket.
//!
//! # Detection
//!
//! The [`Picker`](crate::transport::Picker) creates this backend when the
//! `SCRY_TERMINAL_SOCK` environment variable is present.
//!
//! # Feature Gate
//!
//! This module requires the `native-ipc` feature.

#![allow(unsafe_code)]

use std::collections::HashMap;
use std::io;
use std::os::unix::net::UnixStream;

use tiny_skia::Pixmap;

use crate::transport::backend::{ImageHandle, ProtocolBackend, ProtocolKind, TerminalInfoResponse, TerminalPosition};
use crate::transport::ipc::{
    classify_payload, IpcCommand, IpcEvent, IpcResponse, Memfd, MessageKind, OverlayAnchor,
};
use crate::PixelCanvasError;

// ---------------------------------------------------------------------------
// NativeBackend
// ---------------------------------------------------------------------------

/// Zero-copy shared-memory backend for `scry-terminal`.
///
/// Holds a Unix socket connection to the terminal's IPC server and a map
/// of live overlay memfds. Implements [`ProtocolBackend`] for seamless
/// integration with the existing `FrameDriver` / `Picker` infrastructure.
#[derive(Debug)]
pub struct NativeBackend {
    /// Path to the Unix socket.
    sock_path: String,
    /// Connected socket (lazy, reconnects once on error).
    stream: Option<UnixStream>,
    /// Next overlay ID to assign.
    next_id: u32,
    /// Live overlay memfds keyed by overlay ID.
    memfds: HashMap<u32, Memfd>,
    /// Pause state per overlay (toggled by `Clicked` events).
    paused: HashMap<u32, bool>,
    /// Visibility state per overlay (set by `Visibility` events).
    visible: HashMap<u32, bool>,
}

impl NativeBackend {
    /// Create a new backend that will connect to the given socket path.
    ///
    /// Connection is lazy — the socket is opened on the first `transmit()`.
    #[must_use]
    pub fn new(sock_path: &str) -> Self {
        Self {
            sock_path: sock_path.to_string(),
            stream: None,
            next_id: 1,
            memfds: HashMap::new(),
            paused: HashMap::new(),
            visible: HashMap::new(),
        }
    }

    /// Connect immediately (useful for startup validation).
    pub fn connect(sock_path: &str) -> Self {
        let mut backend = Self::new(sock_path);
        // Best-effort connect; if it fails, transmit() will retry.
        let _ = backend.ensure_connected();
        backend
    }

    /// Ensure we have a live connection, connecting or reconnecting as needed.
    fn ensure_connected(&mut self) -> Result<(), PixelCanvasError> {
        if self.stream.is_some() {
            return Ok(());
        }

        let stream = UnixStream::connect(&self.sock_path).map_err(|e| {
            PixelCanvasError::Rasterization(format!(
                "native IPC: failed to connect to scry-terminal at {}: {e}",
                self.sock_path
            ))
        })?;

        self.stream = Some(stream);
        Ok(())
    }

    /// Get the stream, returning an error if not connected.
    fn stream_mut(&mut self) -> Result<&mut UnixStream, PixelCanvasError> {
        self.stream.as_mut().ok_or_else(|| {
            PixelCanvasError::Rasterization("native IPC: not connected to scry-terminal".into())
        })
    }

    /// Allocate the next overlay ID.
    fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        id
    }

    /// Process an asynchronous event from the terminal.
    fn apply_event(&mut self, event: &IpcEvent) {
        match event {
            IpcEvent::Clicked { id } => {
                let paused = self.paused.entry(*id).or_insert(false);
                *paused = !*paused;
            }
            IpcEvent::Visibility { id, visible } => {
                self.visible.insert(*id, *visible);
            }
        }
    }

    /// Send a command (with optional fd) and receive the synchronous response.
    fn send_recv(
        &mut self,
        cmd: &IpcCommand,
        fd: Option<std::os::unix::io::RawFd>,
    ) -> Result<IpcResponse, PixelCanvasError> {
        self.ensure_connected()?;
        let stream = self.stream_mut()?;

        crate::transport::ipc::send_command_with_fd(stream, cmd, fd)
            .map_err(|e| io::Error::new(io::ErrorKind::BrokenPipe, format!("native IPC send: {e}")))?;

        // Read response, collecting any interleaved events.
        let mut pending_events = Vec::new();
        let response = loop {
            let payload = crate::transport::ipc::read_frame(stream)
                .map_err(|e| io::Error::new(io::ErrorKind::BrokenPipe, format!("native IPC recv: {e}")))?;

            match classify_payload(&payload) {
                Some(MessageKind::Response) => {
                    break IpcResponse::deserialize(&payload).map_err(|e| {
                        io::Error::new(io::ErrorKind::InvalidData, format!("native IPC: bad response: {e}"))
                    })?;
                }
                Some(MessageKind::Event) => {
                    if let Ok(event) = IpcEvent::deserialize(&payload) {
                        pending_events.push(event);
                    }
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "native IPC: unexpected message type",
                    ).into());
                }
            }
        };

        // Apply collected events now (stream borrow is released).
        for event in &pending_events {
            self.apply_event(event);
        }

        Ok(response)
    }

    // -----------------------------------------------------------------------
    // Public event API (beyond ProtocolBackend trait)
    // -----------------------------------------------------------------------

    /// Poll for asynchronous events from the terminal (non-blocking).
    ///
    /// Returns all pending events (click, visibility changes).
    /// Call this in your animation loop between frames.
    pub fn poll_events(&mut self) -> Vec<IpcEvent> {
        let mut events = Vec::new();

        let Some(stream) = self.stream.as_mut() else {
            return events;
        };

        // Collect frames without borrowing self.
        let mut raw_payloads = Vec::new();
        loop {
            match crate::transport::ipc::try_recv_frame(stream) {
                Ok(Some(payload)) => raw_payloads.push(payload),
                Ok(None) => break,
                Err(_) => break,
            }
        }

        // Parse and apply events.
        for payload in &raw_payloads {
            if classify_payload(payload) == Some(MessageKind::Event) {
                if let Ok(event) = IpcEvent::deserialize(payload) {
                    self.apply_event(&event);
                    events.push(event);
                }
            }
        }

        events
    }

    /// Whether the given overlay is currently paused (toggled by click).
    #[must_use]
    pub fn is_paused(&self, id: u32) -> bool {
        self.paused.get(&id).copied().unwrap_or(false)
    }

    /// Whether the given overlay is currently visible in the viewport.
    ///
    /// Returns `true` by default (until the terminal sends a `Visibility` event).
    #[must_use]
    pub fn is_visible(&self, id: u32) -> bool {
        self.visible.get(&id).copied().unwrap_or(true)
    }

    /// Query terminal info (font size, grid dimensions).
    pub fn query_info(&mut self) -> Result<IpcResponse, PixelCanvasError> {
        self.send_recv(&IpcCommand::QueryInfo, None)
    }

    /// Submit a scene graph for server-side GPU rendering.
    ///
    /// Instead of rasterizing locally and transferring pixels, this sends
    /// the serialized scene description to scry-terminal, which renders it
    /// on its own GPU. Falls back to [`transmit`](Self::transmit_inner) if
    /// serialization or the terminal rejects the scene.
    ///
    /// The scene is serialized with `postcard` and passed via memfd (same
    /// mechanism as pixel data, reusing `SCM_RIGHTS` fd passing).
    pub fn submit_scene(
        &mut self,
        scene: &crate::scene::PixelCanvas,
        anchor: OverlayAnchor,
        z_index: i32,
        persist: bool,
    ) -> Result<ImageHandle, PixelCanvasError> {
        let id = self.alloc_id();

        // Serialize the scene graph with postcard.
        let scene_bytes = postcard::to_allocvec(scene).map_err(|e| {
            PixelCanvasError::Rasterization(format!("native IPC: scene serialize failed: {e}"))
        })?;

        let scene_len = scene_bytes.len();

        // Write serialized bytes to a memfd.
        let memfd = Memfd::create(&format!("scry-scene-{id}"), scene_len).map_err(|e| {
            PixelCanvasError::Rasterization(format!("native IPC: memfd_create failed: {e}"))
        })?;
        memfd.write(&scene_bytes).map_err(|e| {
            PixelCanvasError::Rasterization(format!("native IPC: memfd write failed: {e}"))
        })?;

        let fd = memfd.as_raw_fd();

        let cmd = IpcCommand::SubmitScene {
            id,
            anchor,
            z_index,
            scene_len: scene_len as u32,
            persist,
        };

        let resp = self.send_recv(&cmd, Some(fd))?;

        match resp {
            IpcResponse::Ok { id: resp_id } => {
                self.memfds.insert(resp_id, memfd);
                self.visible.insert(resp_id, true);
                Ok(ImageHandle {
                    id: resp_id,
                    protocol: ProtocolKind::Native,
                })
            }
            IpcResponse::Error { msg } => Err(PixelCanvasError::Rasterization(format!(
                "native IPC: submit_scene rejected: {msg}"
            ))),
            _ => Err(PixelCanvasError::Rasterization(
                "native IPC: unexpected response to submit_scene".into(),
            )),
        }
    }

    /// Submit an animation program for terminal-autonomous rendering.
    ///
    /// The terminal will create an SDF pipeline and drive the animation
    /// in its own render loop. The CLI can exit immediately after this
    /// call succeeds.
    #[cfg(feature = "sdf")]
    pub fn submit_animation(
        &mut self,
        program: &crate::sdf::AnimationProgram,
        anchor: OverlayAnchor,
        z_index: i32,
        duration_secs: u32,
        fps: u32,
        width: u32,
        height: u32,
        persist: bool,
    ) -> Result<ImageHandle, PixelCanvasError> {
        let id = self.alloc_id();

        // Serialize the animation program with postcard.
        let program_bytes = postcard::to_allocvec(program).map_err(|e| {
            PixelCanvasError::Rasterization(format!(
                "native IPC: animation program serialize failed: {e}"
            ))
        })?;

        let program_len = program_bytes.len();

        // Write serialized bytes to a memfd.
        let memfd =
            Memfd::create(&format!("scry-anim-{id}"), program_len).map_err(|e| {
                PixelCanvasError::Rasterization(format!(
                    "native IPC: memfd_create failed: {e}"
                ))
            })?;
        memfd.write(&program_bytes).map_err(|e| {
            PixelCanvasError::Rasterization(format!(
                "native IPC: memfd write failed: {e}"
            ))
        })?;

        let fd = memfd.as_raw_fd();

        let cmd = IpcCommand::SubmitAnimation {
            id,
            anchor,
            z_index,
            program_len: program_len as u32,
            duration_secs,
            fps,
            width,
            height,
            persist,
        };

        let resp = self.send_recv(&cmd, Some(fd))?;

        match resp {
            IpcResponse::Ok { id: resp_id } => {
                self.memfds.insert(resp_id, memfd);
                self.visible.insert(resp_id, true);
                Ok(ImageHandle {
                    id: resp_id,
                    protocol: ProtocolKind::Native,
                })
            }
            IpcResponse::Error { msg } => Err(PixelCanvasError::Rasterization(
                format!("native IPC: submit_animation rejected: {msg}"),
            )),
            _ => Err(PixelCanvasError::Rasterization(
                "native IPC: unexpected response to submit_animation".into(),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// ProtocolBackend implementation
// ---------------------------------------------------------------------------

impl NativeBackend {
    /// Transmit a persistent overlay that survives client disconnection.
    ///
    /// Used for static images and charts that should remain visible after
    /// the CLI process exits. The terminal will keep the overlay until
    /// explicitly removed or the terminal is closed.
    pub fn transmit_persistent(
        &mut self,
        pixmap: &Pixmap,
        position: TerminalPosition,
        z_index: i32,
    ) -> Result<ImageHandle, PixelCanvasError> {
        self.transmit_inner(pixmap, position, z_index, true)
    }

    /// Common transmit logic shared by trait `transmit` and `transmit_persistent`.
    fn transmit_inner(
        &mut self,
        pixmap: &Pixmap,
        position: TerminalPosition,
        z_index: i32,
        persist: bool,
    ) -> Result<ImageHandle, PixelCanvasError> {
        let id = self.alloc_id();
        let rgba = pixmap.data();
        let size = rgba.len();

        // Create memfd and write RGBA data.
        let memfd = Memfd::create(&format!("scry-overlay-{id}"), size).map_err(|e| {
            PixelCanvasError::Rasterization(format!("native IPC: memfd_create failed: {e}"))
        })?;
        memfd.write(rgba).map_err(|e| {
            PixelCanvasError::Rasterization(format!("native IPC: memfd write failed: {e}"))
        })?;

        let fd = memfd.as_raw_fd();

        let cmd = IpcCommand::Transmit {
            id,
            anchor: OverlayAnchor::Viewport {
                col: position.col,
                row: position.row,
            },
            w_cells: position.width_cells,
            h_cells: position.height_cells,
            z: z_index,
            px_w: pixmap.width(),
            px_h: pixmap.height(),
            persist,
        };

        let resp = self.send_recv(&cmd, Some(fd))?;

        match resp {
            IpcResponse::Ok { id: resp_id } => {
                self.memfds.insert(resp_id, memfd);
                self.visible.insert(resp_id, true);
                Ok(ImageHandle {
                    id: resp_id,
                    protocol: ProtocolKind::Native,
                })
            }
            IpcResponse::Error { msg } => Err(PixelCanvasError::Rasterization(format!(
                "native IPC: transmit rejected: {msg}"
            ))),
            _ => Err(PixelCanvasError::Rasterization(
                "native IPC: unexpected response to transmit".into(),
            )),
        }
    }
}

impl ProtocolBackend for NativeBackend {
    fn transmit(
        &mut self,
        pixmap: &Pixmap,
        position: TerminalPosition,
        z_index: i32,
    ) -> Result<ImageHandle, PixelCanvasError> {
        self.transmit_inner(pixmap, position, z_index, false)
    }

    fn remove(&mut self, handle: &ImageHandle) -> Result<(), PixelCanvasError> {
        let resp = self.send_recv(&IpcCommand::Remove { id: handle.id }, None)?;

        self.memfds.remove(&handle.id);
        self.paused.remove(&handle.id);
        self.visible.remove(&handle.id);

        match resp {
            IpcResponse::Ok { .. } => Ok(()),
            IpcResponse::Error { msg } => Err(PixelCanvasError::Rasterization(format!(
                "native IPC: remove rejected: {msg}"
            ))),
            _ => Err(PixelCanvasError::Rasterization(
                "native IPC: unexpected response to remove".into(),
            )),
        }
    }

    fn clear_all(&mut self) -> Result<(), PixelCanvasError> {
        let resp = self.send_recv(&IpcCommand::ClearAll, None)?;

        self.memfds.clear();
        self.paused.clear();
        self.visible.clear();

        match resp {
            IpcResponse::Ok { .. } => Ok(()),
            IpcResponse::Error { msg } => Err(PixelCanvasError::Rasterization(format!(
                "native IPC: clear_all rejected: {msg}"
            ))),
            _ => Err(PixelCanvasError::Rasterization(
                "native IPC: unexpected response to clear_all".into(),
            )),
        }
    }

    fn replace(
        &mut self,
        handle: &ImageHandle,
        pixmap: &Pixmap,
        position: TerminalPosition,
        z_index: i32,
    ) -> Result<ImageHandle, PixelCanvasError> {
        let id = handle.id;
        let rgba = pixmap.data();

        // Fast path: overwrite memfd in-place if it fits.
        if let Some(memfd) = self.memfds.get(&id) {
            if memfd.len() >= rgba.len() {
                memfd.write(rgba).map_err(|e| {
                    PixelCanvasError::Rasterization(format!("native IPC: memfd overwrite failed: {e}"))
                })?;

                let resp = self.send_recv(&IpcCommand::Refresh { id }, None)?;

                return match resp {
                    IpcResponse::Ok { id: resp_id } => Ok(ImageHandle {
                        id: resp_id,
                        protocol: ProtocolKind::Native,
                    }),
                    IpcResponse::Error { msg } => Err(PixelCanvasError::Rasterization(format!(
                        "native IPC: refresh rejected: {msg}"
                    ))),
                    _ => Err(PixelCanvasError::Rasterization(
                        "native IPC: unexpected response to refresh".into(),
                    )),
                };
            }
        }

        // Slow path: pixmap size changed — remove old, create new.
        self.remove(handle)?;
        self.transmit(pixmap, position, z_index)
    }

    fn supports_alpha(&self) -> bool {
        true
    }

    fn protocol_kind(&self) -> ProtocolKind {
        ProtocolKind::Native
    }

    fn poll_events(&mut self) {
        // Delegate to the existing pub method which handles internal state updates.
        let _ = NativeBackend::poll_events(self);
    }

    fn is_overlay_paused(&self, id: u32) -> bool {
        self.is_paused(id)
    }

    fn transmit_persistent(
        &mut self,
        pixmap: &Pixmap,
        position: TerminalPosition,
        z_index: i32,
    ) -> Result<ImageHandle, PixelCanvasError> {
        self.transmit_inner(pixmap, position, z_index, true)
    }

    fn query_info(&mut self) -> Result<TerminalInfoResponse, PixelCanvasError> {
        let resp = NativeBackend::query_info(self)?;
        match resp {
            IpcResponse::Info { font_w, font_h, cols, rows } => {
                Ok(TerminalInfoResponse { font_w, font_h, cols, rows })
            }
            _ => Err(PixelCanvasError::Rasterization(
                "native IPC: unexpected response to QueryInfo".into(),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_disconnected() {
        let backend = NativeBackend::new("/tmp/nonexistent-scry.sock");
        assert!(backend.stream.is_none());
        assert_eq!(backend.protocol_kind(), ProtocolKind::Native);
        assert!(backend.supports_alpha());
    }

    #[test]
    fn connect_to_nonexistent_is_ok() {
        // connect() is best-effort — shouldn't panic.
        let backend = NativeBackend::connect("/tmp/nonexistent-scry-test.sock");
        assert!(backend.stream.is_none());
    }

    #[test]
    fn pause_visibility_defaults() {
        let backend = NativeBackend::new("/tmp/test.sock");
        assert!(!backend.is_paused(1));
        assert!(backend.is_visible(1));
    }

    #[test]
    fn handle_clicked_toggles_pause() {
        let mut backend = NativeBackend::new("/tmp/test.sock");
        assert!(!backend.is_paused(5));

        backend.apply_event(&IpcEvent::Clicked { id: 5 });
        assert!(backend.is_paused(5));

        backend.apply_event(&IpcEvent::Clicked { id: 5 });
        assert!(!backend.is_paused(5));
    }

    #[test]
    fn handle_visibility_updates_state() {
        let mut backend = NativeBackend::new("/tmp/test.sock");
        assert!(backend.is_visible(3));

        backend.apply_event(&IpcEvent::Visibility {
            id: 3,
            visible: false,
        });
        assert!(!backend.is_visible(3));

        backend.apply_event(&IpcEvent::Visibility {
            id: 3,
            visible: true,
        });
        assert!(backend.is_visible(3));
    }
}
