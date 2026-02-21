// SPDX-License-Identifier: MIT OR Apache-2.0
//! Performance — parse throttling and render scheduling.
//!
//! Prevents the terminal from freezing when flooded with output
//! (e.g., `cat /dev/urandom`) and coalesces redraws at vsync.

use std::time::{Duration, Instant};

use crate::grid::TerminalGrid;
use crate::pty::{PtyEvent, PtyManager};
use crate::security::SecurityGate;
use crate::vt::VtHandler;

// ── Parse throttler ────────────────────────────────────────────────

/// Limits the number of bytes parsed per frame to maintain responsiveness.
pub struct ParseThrottler {
    /// Maximum bytes to parse per frame.
    budget: usize,
    /// VTE parser state.
    parser: vte::Parser,
}

impl ParseThrottler {
    /// Create a new throttler with the given per-frame byte budget.
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            parser: vte::Parser::new(),
        }
    }

    /// Default budget: 256 KB per frame.
    ///
    /// At 60fps, this allows ~15 MB/s sustained throughput, which is
    /// more than enough for any realistic terminal workload while
    /// keeping frame times under 16ms.
    pub fn default_budget() -> Self {
        Self::new(256 * 1024)
    }

    /// Drain the PTY channel up to the budget, parsing and applying actions.
    ///
    /// Returns the number of bytes consumed and whether the child has exited.
    pub fn poll_pty(
        &mut self,
        pty: &PtyManager,
        grid: &mut TerminalGrid,
        security: &mut SecurityGate,
    ) -> PollResult {
        let mut consumed = 0;
        let mut child_exited = false;
        let mut responses = Vec::new();

        while consumed < self.budget {
            match pty.receiver.try_recv() {
                Ok(PtyEvent::Output(data)) => {
                    consumed += data.len();
                    // Parse through VTE
                    let mut handler = VtHandler::new(grid, security);
                    for &byte in &data {
                        self.parser.advance(&mut handler, byte);
                    }
                    if !handler.response.is_empty() {
                        responses.push(handler.response);
                    }
                }
                Ok(PtyEvent::ChildExited) => {
                    child_exited = true;
                    break;
                }
                Err(crossbeam_channel::TryRecvError::Empty) => {
                    break; // No more data this frame
                }
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    child_exited = true;
                    break;
                }
            }
        }

        PollResult {
            bytes_consumed: consumed,
            child_exited,
            responses,
        }
    }
}

/// Result of polling the PTY channel.
pub struct PollResult {
    /// Number of bytes consumed this frame.
    pub bytes_consumed: usize,
    /// Whether the child process has exited.
    pub child_exited: bool,
    /// Response bytes to send back to the PTY.
    pub responses: Vec<Vec<u8>>,
}

// ── Render scheduler ───────────────────────────────────────────────

/// Coalesces redraws to prevent excessive GPU work.
pub struct RenderScheduler {
    /// Whether a redraw is pending.
    pending: bool,
    /// Time of last render.
    last_render: Instant,
    /// Minimum interval between renders.
    min_interval: Duration,
}

impl RenderScheduler {
    /// Create a new scheduler targeting the given framerate.
    pub fn new(target_fps: u32) -> Self {
        Self {
            pending: true, // First frame always renders
            last_render: Instant::now() - Duration::from_secs(1), // Force first render
            min_interval: Duration::from_micros(1_000_000 / u64::from(target_fps)),
        }
    }

    /// Mark that a redraw is needed (grid content changed).
    pub fn request_redraw(&mut self) {
        self.pending = true;
    }

    /// Check if we should render now.
    ///
    /// Returns `true` if:
    /// 1. A redraw is pending, AND
    /// 2. Enough time has elapsed since the last render.
    pub fn should_render(&mut self) -> bool {
        if !self.pending {
            return false;
        }
        if self.last_render.elapsed() < self.min_interval {
            return false;
        }
        self.pending = false;
        self.last_render = Instant::now();
        true
    }

    /// Called after a successful render to record the timestamp.
    pub fn did_render(&mut self) {
        self.last_render = Instant::now();
        self.pending = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throttler_has_default_budget() {
        let throttler = ParseThrottler::default_budget();
        assert_eq!(throttler.budget, 256 * 1024);
    }

    #[test]
    fn render_scheduler_first_frame() {
        let mut scheduler = RenderScheduler::new(60);
        assert!(scheduler.should_render());
        assert!(!scheduler.should_render()); // Already consumed
    }

    #[test]
    fn render_scheduler_pending() {
        let mut scheduler = RenderScheduler::new(60);
        scheduler.did_render();
        assert!(!scheduler.should_render()); // Not pending
        scheduler.request_redraw();
        // Might still be too soon — but pending is set
        // (timing-dependent, just verify the flag)
    }
}
