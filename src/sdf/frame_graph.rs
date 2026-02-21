// SPDX-License-Identifier: MIT OR Apache-2.0
//! Explicit frame state machine for SDF GPU rendering.
//!
//! [`SdfFrameGraph`] replaces the implicit double-buffer state machine
//! (7 boolean/integer fields) with a 2-variant [`FrameState`] enum.
//! It also introduces **timeout-guarded readback** — GPU poll uses a
//! deadline instead of `wgpu::Maintain::Wait`, preventing deadlocks
//! when the GPU device is lost.
//!
//! # Pipeline
//!
//! ```text
//! Frame N:    submit(scene_N) → readback(frame_N-1) → display prev
//! Frame N+1:  submit(scene_N+1) → readback(frame_N) → display prev
//! ```

use std::time::Duration;

use crate::scene::command::ImageData;
use crate::sdf::gpu_renderer::{SdfGpuContext, SdfGpuRenderer};
use crate::sdf::scene::SdfScene;

/// Default readback timeout — 2 seconds.
///
/// This is 60× longer than a typical 33ms GPU frame, so it only
/// triggers on genuine device-lost scenarios, not slow frames.
const DEFAULT_POLL_TIMEOUT: Duration = Duration::from_secs(2);

// ── Frame state ────────────────────────────────────────────────────

/// Explicit state of the frame pipeline.
#[derive(Clone, Debug)]
#[allow(unreachable_pub)]
pub enum FrameState {
    /// No GPU work in flight.
    Idle,
    /// GPU work submitted, awaiting readback.
    Submitted {
        /// Render dimensions (potentially downscaled).
        render_w: u32,
        render_h: u32,
        /// Requested output dimensions.
        output_w: u32,
        output_h: u32,
    },
}

// ── Frame graph ────────────────────────────────────────────────────

/// Explicit SDF frame state machine.
///
/// Owns the double-buffer pipeline state and readback buffer.
/// The SDF pipeline delegates GPU frame management to this struct.
#[allow(unreachable_pub)]
pub struct SdfFrameGraph {
    state: FrameState,
    /// Previous frame's GPU result (for the double-buffer pipeline).
    prev_result: Option<ImageData>,
    /// Reusable readback buffer to avoid per-frame allocation.
    readback_buf: Vec<u8>,
    /// Timeout for GPU poll — prevents deadlock on device loss.
    poll_timeout: Duration,
}

impl SdfFrameGraph {
    /// Create a new frame graph in `Idle` state.
    pub(super) fn new() -> Self {
        Self {
            state: FrameState::Idle,
            prev_result: None,
            readback_buf: Vec::new(),
            poll_timeout: DEFAULT_POLL_TIMEOUT,
        }
    }

    /// Whether there's a pending GPU submission.
    #[allow(dead_code)]
    pub(super) fn is_submitted(&self) -> bool {
        matches!(self.state, FrameState::Submitted { .. })
    }

    /// Take the previous frame result (if any).
    pub(super) fn take_prev_result(&mut self) -> Option<ImageData> {
        self.prev_result.take()
    }

    /// Readback the pending GPU frame (if any) and store in `prev_result`.
    ///
    /// This is the critical path — uses timeout-guarded polling instead
    /// of `device.poll(Maintain::Wait)` to prevent deadlocks.
    pub(super) fn readback_pending(
        &mut self,
        ctx: &mut SdfGpuContext,
        health: &crate::gpu::SharedHealthMonitor,
    ) {
        if let FrameState::Submitted {
            render_w,
            render_h,
            output_w,
            output_h,
        } = self.state
        {
            let result = if render_w == output_w && render_h == output_h {
                // No upscale needed — readback directly into buffer
                Self::readback_into_with_timeout(
                    ctx,
                    render_w,
                    render_h,
                    &mut self.readback_buf,
                    self.poll_timeout,
                )
                .map(|()| {
                    let data = std::mem::take(&mut self.readback_buf);
                    ImageData::new(output_w, output_h, data)
                })
            } else {
                // Upscale needed — readback into pixmap, then upscale
                Self::readback_with_timeout(ctx, render_w, render_h, self.poll_timeout)
                    .map(|pm| {
                        let upscaled = crate::sdf::upscale::upscale_bicubic(
                            pm.data(),
                            render_w,
                            render_h,
                            output_w,
                            output_h,
                        );
                        ImageData::new(output_w, output_h, upscaled)
                    })
            };

            match result {
                Ok(image) => {
                    self.prev_result = Some(image);
                    if let Ok(mut h) = health.lock() {
                        h.report_success();
                    }
                }
                Err(_) => {
                    // Drop this frame — don't panic, just skip
                    if let Ok(mut h) = health.lock() {
                        h.report_failure();
                    }
                }
            }
            self.state = FrameState::Idle;
        }
    }

    /// Submit new GPU work.
    ///
    /// Returns `true` if the submission succeeded and the state
    /// transitioned to `Submitted`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn submit(
        &mut self,
        ctx: &mut SdfGpuContext,
        scene: &SdfScene,
        render_w: u32,
        render_h: u32,
        output_w: u32,
        output_h: u32,
        time: f32,
        health: &crate::gpu::SharedHealthMonitor,
    ) -> bool {
        if SdfGpuRenderer::submit(ctx, scene, render_w, render_h, time).is_ok() {
            self.state = FrameState::Submitted {
                render_w,
                render_h,
                output_w,
                output_h,
            };
            if let Ok(mut h) = health.lock() {
                h.report_success();
            }
            true
        } else {
            if let Ok(mut h) = health.lock() {
                h.report_failure();
            }
            false
        }
    }

    // ── Timeout-guarded readback ───────────────────────────────────

    /// Readback into a `Pixmap` with a timeout.
    ///
    /// Uses a poll loop with `Instant::now() < deadline` instead of
    /// `device.poll(Maintain::Wait)` to prevent deadlocks on device loss.
    fn readback_with_timeout(
        ctx: &mut SdfGpuContext,
        width: u32,
        height: u32,
        timeout: Duration,
    ) -> Result<tiny_skia::Pixmap, crate::PixelCanvasError> {
        // Reuse cached pixmap if dimensions match, otherwise allocate
        let mut pixmap = match ctx.take_cached_pixmap() {
            Some(pm) if pm.width() == width && pm.height() == height => pm,
            _ => tiny_skia::Pixmap::new(width, height).ok_or_else(|| {
                crate::PixelCanvasError::PixmapCreation(format!(
                    "failed to create {width}x{height} pixmap"
                ))
            })?,
        };

        let readback_buf = ctx
            .pool()
            .get(crate::gpu::buffer_pool::BufferKey::SdfReadback)
            .unwrap();

        let readback_slice = readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        readback_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });

        // Timeout-guarded poll loop
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if ctx.device().poll(wgpu::MaintainBase::Poll).is_queue_empty() {
                break;
            }
            if std::time::Instant::now() >= deadline {
                return Err(crate::PixelCanvasError::Rasterization(
                    crate::sdf::error::SdfError::ReadbackTimeout(timeout).to_string(),
                ));
            }
            std::thread::sleep(Duration::from_millis(1));
        }

        rx.recv()
            .map_err(|e| {
                crate::PixelCanvasError::Rasterization(format!("GPU readback failed: {e}"))
            })?
            .map_err(|e| {
                crate::PixelCanvasError::Rasterization(format!("GPU buffer map failed: {e}"))
            })?;

        {
            let data = readback_slice.get_mapped_range();
            pixmap.data_mut().copy_from_slice(&data);
        }
        readback_buf.unmap();
        ctx.clear_pending_readback();

        Ok(pixmap)
    }

    /// Readback raw bytes into a buffer with a timeout.
    ///
    /// Uses the same timeout-guarded poll loop as [`readback_with_timeout`].
    fn readback_into_with_timeout(
        ctx: &mut SdfGpuContext,
        width: u32,
        height: u32,
        buf: &mut Vec<u8>,
        timeout: Duration,
    ) -> Result<(), crate::PixelCanvasError> {
        let expected = (width as usize) * (height as usize) * 4;
        buf.resize(expected, 0);

        let readback_buf = ctx
            .pool()
            .get(crate::gpu::buffer_pool::BufferKey::SdfReadback)
            .unwrap();

        let readback_slice = readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        readback_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });

        // Timeout-guarded poll loop
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if ctx.device().poll(wgpu::MaintainBase::Poll).is_queue_empty() {
                break;
            }
            if std::time::Instant::now() >= deadline {
                return Err(crate::PixelCanvasError::Rasterization(
                    crate::sdf::error::SdfError::ReadbackTimeout(timeout).to_string(),
                ));
            }
            std::thread::sleep(Duration::from_millis(1));
        }

        rx.recv()
            .map_err(|e| {
                crate::PixelCanvasError::Rasterization(format!("GPU readback failed: {e}"))
            })?
            .map_err(|e| {
                crate::PixelCanvasError::Rasterization(format!("GPU buffer map failed: {e}"))
            })?;

        {
            let data = readback_slice.get_mapped_range();
            buf.copy_from_slice(&data);
        }
        readback_buf.unmap();
        ctx.clear_pending_readback();

        Ok(())
    }

}

impl Default for SdfFrameGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SdfFrameGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SdfFrameGraph")
            .field("state", &self.state)
            .field("has_prev_result", &self.prev_result.is_some())
            .field("readback_buf_len", &self.readback_buf.len())
            .field("poll_timeout", &self.poll_timeout)
            .finish()
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_graph_starts_idle() {
        let fg = SdfFrameGraph::new();
        assert!(matches!(fg.state, FrameState::Idle));
        assert!(fg.prev_result.is_none());
    }

    #[test]
    fn frame_graph_default_timeout() {
        let fg = SdfFrameGraph::new();
        assert_eq!(fg.poll_timeout, DEFAULT_POLL_TIMEOUT);
    }
}
