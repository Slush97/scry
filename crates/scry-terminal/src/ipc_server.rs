// SPDX-License-Identifier: MIT OR Apache-2.0
//! IPC server for scry-terminal — listens for overlay commands from scry-cli.
//!
//! Runs a Unix domain socket listener on a background thread. Child processes
//! (scry-cli) connect and send overlay commands (transmit, refresh, remove)
//! via the shared-memory IPC protocol defined in `scry_engine::transport::ipc`.
//!
//! Overlay operations are forwarded to the main event loop via a
//! `crossbeam_channel::Sender`, where the compositor applies them.
//!
//! Click / visibility events are dispatched to the owning client via a
//! **per-client `Sender`** stored in a shared `HashMap`. This ensures that an
//! event destined for client N is only seen by client N's handler thread,
//! fixing the MPMC fan-out bug where a shared channel would deliver each
//! event to an arbitrary handler.

use std::collections::HashMap;
use std::os::unix::io::RawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use crossbeam_channel::{Receiver, Sender};
use scry_engine::transport::ipc::{
    self, IpcCommand, IpcEvent, IpcResponse, Memfd, OverlayAnchor,
};

// ---------------------------------------------------------------------------
// Overlay operations (sent to main thread)
// ---------------------------------------------------------------------------

/// An overlay operation to be applied by the compositor on the main thread.
#[derive(Debug)]
pub enum OverlayOp {
    /// Add a new overlay from shared memory.
    Add {
        /// Overlay ID.
        id: u32,
        /// Client connection ID (for routing events back).
        client_id: u64,
        /// Where to anchor the overlay.
        anchor: OverlayAnchor,
        /// Width in terminal cells.
        w_cells: u16,
        /// Height in terminal cells.
        h_cells: u16,
        /// Z-index for compositing order.
        z: i32,
        /// Pixel width.
        px_w: u32,
        /// Pixel height.
        px_h: u32,
        /// The shared memory mapping (moved to main thread).
        memfd: Memfd,
    },
    /// Refresh an existing overlay (re-read its memfd).
    Refresh {
        /// Overlay ID.
        id: u32,
    },
    /// Remove an overlay.
    Remove {
        /// Overlay ID.
        id: u32,
    },
    /// Remove all overlays from a specific client.
    ClearAll {
        /// Client identifier (usually the thread/connection index).
        client_id: u64,
    },
}

/// Information about the terminal for `QueryInfo` responses.
#[derive(Debug, Clone, Copy)]
pub struct TerminalInfo {
    /// Font cell width in pixels.
    pub font_w: u16,
    /// Font cell height in pixels.
    pub font_h: u16,
    /// Grid columns.
    pub cols: u16,
    /// Grid rows.
    pub rows: u16,
}

impl Default for TerminalInfo {
    fn default() -> Self {
        Self {
            font_w: 8,
            font_h: 16,
            cols: 80,
            rows: 24,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-client event routing
// ---------------------------------------------------------------------------

/// Thread-safe map from `client_id` to that client's event `Sender`.
///
/// The main thread calls `send_to_client` to push events (click, visibility)
/// directly to the owning client's handler thread — no MPMC fan-out.
type ClientEventMap = Arc<Mutex<HashMap<u64, Sender<IpcEvent>>>>;

// ---------------------------------------------------------------------------
// IPC Server
// ---------------------------------------------------------------------------

/// The IPC server that listens for scry-cli connections.
pub struct IpcServer {
    /// Path to the Unix socket.
    sock_path: PathBuf,
    /// Shutdown flag.
    shutdown: Arc<AtomicBool>,
    /// Listener thread handle.
    handle: Option<thread::JoinHandle<()>>,
    /// Per-client event senders — keyed by `client_id`.
    ///
    /// Holding the mutex for a short duration to insert/remove entries is
    /// fine; it never contends with the hot path (event sends use a cloned
    /// `Sender` obtained while holding the lock briefly, not during send).
    client_events: ClientEventMap,
}

impl IpcServer {
    /// Start the IPC server.
    ///
    /// Creates a Unix socket at `$XDG_RUNTIME_DIR/scry-term-<pid>.sock`
    /// and spawns a background thread that accepts connections and forwards
    /// overlay operations to `ops_tx`.
    ///
    /// Returns the server and the socket path (to set as env var).
    pub fn start(
        ops_tx: Sender<OverlayOp>,
        info: Arc<std::sync::RwLock<TerminalInfo>>,
    ) -> Result<Self, String> {
        let sock_path = Self::socket_path();

        // Remove stale socket if it exists.
        if sock_path.exists() {
            let _ = std::fs::remove_file(&sock_path);
        }

        // Ensure parent directory exists.
        if let Some(parent) = sock_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create socket directory: {e}"))?;
        }

        let listener = UnixListener::bind(&sock_path)
            .map_err(|e| format!("failed to bind IPC socket at {}: {e}", sock_path.display()))?;

        // Set non-blocking so we can check the shutdown flag periodically.
        listener
            .set_nonblocking(true)
            .map_err(|e| format!("failed to set non-blocking: {e}"))?;

        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();
        let path_clone = sock_path.clone();

        let client_events: ClientEventMap = Arc::new(Mutex::new(HashMap::new()));
        let client_events_clone = client_events.clone();

        let handle = thread::Builder::new()
            .name("scry-ipc-server".to_string())
            .spawn(move || {
                Self::listener_loop(
                    listener,
                    ops_tx,
                    info,
                    client_events_clone,
                    shutdown_clone,
                    &path_clone,
                );
            })
            .map_err(|e| format!("failed to spawn IPC server thread: {e}"))?;

        eprintln!(
            "[scry-term] IPC server started at {}",
            sock_path.display()
        );

        Ok(Self {
            sock_path,
            shutdown,
            handle: Some(handle),
            client_events,
        })
    }

    /// The socket path for the current process.
    pub fn socket_path() -> PathBuf {
        let runtime_dir = std::env::var("XDG_RUNTIME_DIR")
            .unwrap_or_else(|_| "/tmp".to_string());
        PathBuf::from(runtime_dir).join(format!("scry-term-{}.sock", std::process::id()))
    }

    /// Get the socket path as a string (for env var).
    pub fn sock_path_str(&self) -> &str {
        self.sock_path.to_str().unwrap_or("")
    }

    /// Send an event to a specific client by `client_id`.
    ///
    /// If the client is no longer connected (its `Sender` is gone from the
    /// map), the send is silently ignored — the client has already cleaned up.
    pub fn send_to_client(&self, client_id: u64, event: IpcEvent) {
        if let Ok(map) = self.client_events.lock() {
            if let Some(tx) = map.get(&client_id) {
                let _ = tx.send(event);
            }
        }
    }

    /// Shut down the server and clean up.
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        self.cleanup_socket();
    }

    /// Remove the socket file.
    fn cleanup_socket(&self) {
        if self.sock_path.exists() {
            let _ = std::fs::remove_file(&self.sock_path);
        }
    }

    /// Main listener loop (runs on background thread).
    fn listener_loop(
        listener: UnixListener,
        ops_tx: Sender<OverlayOp>,
        info: Arc<std::sync::RwLock<TerminalInfo>>,
        client_events: ClientEventMap,
        shutdown: Arc<AtomicBool>,
        _path: &Path,
    ) {
        let mut client_counter: u64 = 0;

        while !shutdown.load(Ordering::SeqCst) {
            match listener.accept() {
                Ok((stream, _addr)) => {
                    client_counter += 1;
                    let client_id = client_counter;
                    let ops_tx = ops_tx.clone();
                    let info = info.clone();
                    let shutdown = shutdown.clone();
                    let client_events = client_events.clone();

                    // Set blocking mode for the client connection.
                    let _ = stream.set_nonblocking(false);

                    // Create a dedicated per-client (one sender, one receiver).
                    let (event_tx, event_rx) = crossbeam_channel::unbounded::<IpcEvent>();

                    // Register the sender in the shared map.
                    if let Ok(mut map) = client_events.lock() {
                        map.insert(client_id, event_tx);
                    }

                    thread::Builder::new()
                        .name(format!("scry-ipc-client-{client_id}"))
                        .spawn(move || {
                            Self::client_handler(
                                stream, client_id, ops_tx, info, event_rx, shutdown,
                                client_events,
                            );
                        })
                        .ok();
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // No pending connections — sleep briefly and retry.
                    thread::sleep(std::time::Duration::from_millis(50));
                }
                Err(e) => {
                    if !shutdown.load(Ordering::SeqCst) {
                        eprintln!("[scry-ipc] accept error: {e}");
                    }
                    break;
                }
            }
        }
    }

    /// Handle a single client connection.
    ///
    /// Uses a dedicated `event_rx` channel — only events sent via
    /// `IpcServer::send_to_client(client_id, ...)` arrive here.
    fn client_handler(
        mut stream: UnixStream,
        client_id: u64,
        ops_tx: Sender<OverlayOp>,
        info: Arc<std::sync::RwLock<TerminalInfo>>,
        event_rx: Receiver<IpcEvent>,
        shutdown: Arc<AtomicBool>,
        client_events: ClientEventMap,
    ) {
        eprintln!("[scry-ipc] client {client_id} connected");

        while !shutdown.load(Ordering::SeqCst) {
            // Drain any pending events for this client (non-blocking).
            // Since this is a per-client channel, every message here is
            // guaranteed to be for *this* client — no filtering needed.
            while let Ok(event) = event_rx.try_recv() {
                if let Err(e) = ipc::send_event(&mut stream, &event) {
                    eprintln!("[scry-ipc] client {client_id} event send error: {e}");
                    // Fall through to command loop which will detect the broken pipe.
                }
            }

            // Receive command + optional fd.
            let (cmd, fd) = match ipc::recv_command_with_fd(&mut stream) {
                Ok(result) => result,
                Err(e) => {
                    if e.kind() != std::io::ErrorKind::ConnectionReset
                        && e.kind() != std::io::ErrorKind::UnexpectedEof
                    {
                        eprintln!("[scry-ipc] client {client_id} recv error: {e}");
                    }
                    break;
                }
            };

            let response = Self::process_command(
                cmd, fd, client_id, &ops_tx, &info,
            );

            if let Err(e) = ipc::send_response(&mut stream, &response) {
                eprintln!("[scry-ipc] client {client_id} send error: {e}");
                break;
            }
        }

        // Client disconnected — clean up all its overlays and remove its sender.
        let _ = ops_tx.send(OverlayOp::ClearAll { client_id });
        if let Ok(mut map) = client_events.lock() {
            map.remove(&client_id);
        }
        eprintln!("[scry-ipc] client {client_id} disconnected");
    }

    /// Process a single command and return the response.
    fn process_command(
        cmd: IpcCommand,
        fd: Option<RawFd>,
        client_id: u64,
        ops_tx: &Sender<OverlayOp>,
        info: &Arc<std::sync::RwLock<TerminalInfo>>,
    ) -> IpcResponse {
        match cmd {
            IpcCommand::Transmit {
                id,
                anchor,
                w_cells,
                h_cells,
                z,
                px_w,
                px_h,
            } => {
                let Some(raw_fd) = fd else {
                    return IpcResponse::Error {
                        msg: "Transmit requires a memfd (no fd received)".into(),
                    };
                };

                let size = (px_w * px_h * 4) as usize; // RGBA
                match Memfd::from_fd(raw_fd, size) {
                    Ok(memfd) => {
                        let op = OverlayOp::Add {
                            id,
                            client_id,
                            anchor,
                            w_cells,
                            h_cells,
                            z,
                            px_w,
                            px_h,
                            memfd,
                        };
                        if ops_tx.send(op).is_err() {
                            return IpcResponse::Error {
                                msg: "compositor channel closed".into(),
                            };
                        }
                        IpcResponse::Ok { id }
                    }
                    Err(e) => IpcResponse::Error {
                        msg: format!("failed to mmap received fd: {e}"),
                    },
                }
            }

            IpcCommand::Refresh { id } => {
                if ops_tx.send(OverlayOp::Refresh { id }).is_err() {
                    return IpcResponse::Error {
                        msg: "compositor channel closed".into(),
                    };
                }
                IpcResponse::Ok { id }
            }

            IpcCommand::Remove { id } => {
                if ops_tx.send(OverlayOp::Remove { id }).is_err() {
                    return IpcResponse::Error {
                        msg: "compositor channel closed".into(),
                    };
                }
                IpcResponse::Ok { id }
            }

            IpcCommand::ClearAll => {
                if ops_tx
                    .send(OverlayOp::ClearAll { client_id })
                    .is_err()
                {
                    return IpcResponse::Error {
                        msg: "compositor channel closed".into(),
                    };
                }
                IpcResponse::Ok { id: 0 }
            }

            IpcCommand::QueryInfo => {
                let info = info.read().unwrap_or_else(|p| p.into_inner());
                IpcResponse::Info {
                    font_w: info.font_w,
                    font_h: info.font_h,
                    cols: info.cols,
                    rows: info.rows,
                }
            }
        }
    }
}

impl Drop for IpcServer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_client_event_routing_only_reaches_target() {
        // Simulate two clients by creating two per-client channels manually.
        let (tx1, rx1) = crossbeam_channel::unbounded::<IpcEvent>();
        let (tx2, rx2) = crossbeam_channel::unbounded::<IpcEvent>();

        let mut map: HashMap<u64, Sender<IpcEvent>> = HashMap::new();
        map.insert(1, tx1);
        map.insert(2, tx2);

        // Send an event intended for client 1 only.
        if let Some(tx) = map.get(&1) {
            tx.send(IpcEvent::Clicked { id: 42 }).unwrap();
        }

        // Client 1 should receive it.
        assert!(matches!(rx1.try_recv(), Ok(IpcEvent::Clicked { id: 42 })));
        // Client 2 should NOT receive it.
        assert!(rx2.try_recv().is_err(), "client 2 should not receive client 1's event");
    }

    #[test]
    fn client_cleanup_removes_sender() {
        let map: ClientEventMap = Arc::new(Mutex::new(HashMap::new()));
        let (tx, _rx) = crossbeam_channel::unbounded::<IpcEvent>();

        map.lock().unwrap().insert(99, tx);
        assert!(map.lock().unwrap().contains_key(&99));

        // Simulate disconnect cleanup
        map.lock().unwrap().remove(&99);
        assert!(!map.lock().unwrap().contains_key(&99));
    }
}
