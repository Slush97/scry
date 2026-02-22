// SPDX-License-Identifier: MIT OR Apache-2.0
//! PTY management — spawning, reading, writing, and resizing.
//!
//! Uses `portable-pty` for cross-platform PTY support. The PTY reader
//! runs on a dedicated thread and sends raw bytes to the main thread
//! via a bounded channel. An optional waker callback is invoked after
//! each send to wake the event loop.

use crossbeam_channel::{Receiver, Sender};
use portable_pty::{CommandBuilder, MasterPty, PtySize};
use std::io::{Read, Write};
use std::sync::Arc;

use crate::platform;

/// Message from the PTY reader thread to the main thread.
pub enum PtyEvent {
    /// Raw bytes from the PTY (child process output).
    Output(Vec<u8>),
    /// The child process has exited.
    ChildExited,
}

/// Manages the pseudo-terminal and child process.
pub struct PtyManager {
    /// Master side writer — used to send input to the child.
    writer: Box<dyn Write + Send>,
    /// Master PTY handle — used for resizing.
    master: Box<dyn MasterPty + Send>,
    /// The child process handle.
    child: Box<dyn portable_pty::Child + Send + Sync>,
    /// Receiver for PTY events (read on main thread).
    pub receiver: Receiver<PtyEvent>,
    /// Current PTY size.
    size: PtySize,
}

impl PtyManager {
    /// Spawn a new PTY with the given shell and size.
    ///
    /// Starts the child process and a background reader thread.
    pub fn spawn(
        shell: &str,
        cols: u16,
        rows: u16,
        pixel_width: u16,
        pixel_height: u16,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::spawn_internal(shell, cols, rows, pixel_width, pixel_height, None)
    }

    /// Spawn a new PTY with a waker callback.
    ///
    /// The waker is called after each successful data send to wake the
    /// event loop (e.g., via `EventLoopProxy::send_event`).
    pub fn spawn_with_waker(
        shell: &str,
        cols: u16,
        rows: u16,
        pixel_width: u16,
        pixel_height: u16,
        waker: Box<dyn Fn() + Send + Sync>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::spawn_internal(shell, cols, rows, pixel_width, pixel_height, Some(Arc::new(waker)))
    }

    /// Internal spawning logic.
    fn spawn_internal(
        shell: &str,
        cols: u16,
        rows: u16,
        pixel_width: u16,
        pixel_height: u16,
        waker: Option<Arc<Box<dyn Fn() + Send + Sync>>>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let pty_size = PtySize {
            rows,
            cols,
            pixel_width,
            pixel_height,
        };

        // Create the PTY pair
        let pty_system = portable_pty::native_pty_system();
        let pair = pty_system.openpty(pty_size)?;

        // Build the shell command
        let mut cmd = CommandBuilder::new(shell);
        platform::setup_child_env(&mut cmd);

        // Spawn the child process
        let child = pair.slave.spawn_command(cmd)?;

        // Drop the slave — we only interact with the master side
        drop(pair.slave);

        // Get the master reader (writer is obtained via take_writer)
        let reader = pair.master.try_clone_reader()?;

        // Create the channel
        let (sender, receiver) = crossbeam_channel::bounded::<PtyEvent>(256);

        // Start the reader thread
        Self::start_reader_thread(reader, sender, waker)
            .map_err(|e| -> Box<dyn std::error::Error> { Box::new(e) })?;

        Ok(Self {
            writer: pair.master.take_writer()?,
            master: pair.master,
            child,
            receiver,
            size: pty_size,
        })
    }

    /// Start the background PTY reader thread.
    fn start_reader_thread(
        mut reader: Box<dyn Read + Send>,
        sender: Sender<PtyEvent>,
        waker: Option<Arc<Box<dyn Fn() + Send + Sync>>>,
    ) -> Result<(), crate::error::TerminalError> {
        std::thread::Builder::new()
            .name("pty-reader".to_string())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut buf = vec![0u8; 8192];
                    loop {
                        match reader.read(&mut buf) {
                            Ok(0) => {
                                // EOF — child closed its end
                                let _ = sender.send(PtyEvent::ChildExited);
                                if let Some(w) = &waker {
                                    w();
                                }
                                break;
                            }
                            Ok(n) => {
                                let data = buf[..n].to_vec();
                                if sender.send(PtyEvent::Output(data)).is_err() {
                                    break; // Receiver dropped
                                }
                                // Wake the event loop so it processes the data
                                if let Some(w) = &waker {
                                    w();
                                }
                            }
                            Err(e) => {
                                // I/O error — child probably exited
                                if e.kind() != std::io::ErrorKind::Interrupted {
                                    let _ = sender.send(PtyEvent::ChildExited);
                                    if let Some(w) = &waker {
                                        w();
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }));

                if result.is_err() {
                    eprintln!("[scry-term] PTY reader thread panicked");
                    let _ = sender.send(PtyEvent::ChildExited);
                    if let Some(w) = &waker {
                        w();
                    }
                }
            })
            .map_err(crate::error::TerminalError::ThreadSpawn)?;
        Ok(())
    }

    /// Write bytes to the PTY (sends input to the child process).
    pub fn write(&mut self, data: &[u8]) -> std::io::Result<()> {
        self.writer.write_all(data)?;
        self.writer.flush()
    }

    /// Resize the PTY.
    pub fn resize(
        &mut self,
        cols: u16,
        rows: u16,
        pixel_width: u16,
        pixel_height: u16,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let new_size = PtySize {
            rows,
            cols,
            pixel_width,
            pixel_height,
        };

        if new_size.rows != self.size.rows
            || new_size.cols != self.size.cols
            || new_size.pixel_width != self.size.pixel_width
            || new_size.pixel_height != self.size.pixel_height
        {
            self.master.resize(new_size)?;
            self.size = new_size;
        }

        Ok(())
    }

    /// Check if the child process has exited (non-blocking).
    pub fn try_wait(&mut self) -> Option<portable_pty::ExitStatus> {
        self.child.try_wait().ok().flatten()
    }

    /// Current PTY size.
    pub fn size(&self) -> PtySize {
        self.size
    }
}
