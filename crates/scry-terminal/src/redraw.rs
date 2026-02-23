// SPDX-License-Identifier: MIT OR Apache-2.0
//! RedrawRequested handler, decomposed into sub-methods.

use std::time::{Duration, Instant};
use scry_engine::sdf::SdfPipeline;

#[cfg(feature = "logging")]
use tracing::{debug, info, warn, error};

#[cfg(not(feature = "logging"))]
macro_rules! debug { ($($t:tt)*) => { if false { let _ = format_args!($($t)*); } } }
#[cfg(not(feature = "logging"))]
macro_rules! info  { ($($t:tt)*) => { if false { let _ = format_args!($($t)*); } } }
#[cfg(not(feature = "logging"))]
macro_rules! warn  { ($($t:tt)*) => { if false { let _ = format_args!($($t)*); } } }
#[cfg(not(feature = "logging"))]
macro_rules! error { ($($t:tt)*) => { if false { let _ = format_args!($($t)*); } } }

use scry_terminal::config::TerminalConfig;
use scry_engine::transport::ipc::OverlayAnchor;

use super::{pty_write, ActiveOverlay, TerminalState};

impl TerminalState {
    /// Handle a RedrawRequested event. Returns `true` if the event loop should exit.
    pub(crate) fn handle_redraw(&mut self, config: &TerminalConfig) -> bool {
        // Drain mode: child has exited, flush remaining output
        if self.exit_deadline.is_some() {
            return self.handle_drain_phase();
        }

        self.poll_pty_data();

        // poll_pty_data may set exit_deadline if child exited — start
        // drain on next redraw rather than continuing normal processing.
        if self.exit_deadline.is_some() {
            return false;
        }

        self.process_ipc_overlays();
        self.process_clipboard();
        self.process_bell(config);
        self.advance_animations();

        if self.render_frame() {
            return true; // OOM
        }

        self.update_title();

        if self.check_child_exit() {
            return false; // entering drain, not exiting yet
        }

        self.schedule_animations();
        false
    }

    /// Handle drain phase (child has exited, we're flushing remaining output).
    /// Returns `true` if the event loop should exit.
    fn handle_drain_phase(&mut self) -> bool {
        let Some(deadline) = self.exit_deadline else {
            return false;
        };

        // Drain remaining PTY output during grace period
        let drain = self
            .throttler
            .poll_pty(&self.pty, &mut self.grid, &mut self.security);
        for response in &drain.responses {
            pty_write(&mut self.pty, response);
        }
        if drain.bytes_consumed > 0 {
            self.scheduler.request_redraw();
        }

        if Instant::now() >= deadline {
            debug!("drain deadline reached — exiting");
            return true;
        }

        // Still draining — render what we have and continue
        if (self.scheduler.should_render() || self.grid.has_dirty())
            && self
                .compositor
                .render_frame(&self.grid, Some(&self.selection))
                .is_ok()
        {
            self.scheduler.did_render();
            self.grid.clear_dirty();
        }
        self.window.request_redraw();
        false
    }

    fn poll_pty_data(&mut self) {
        let result = self
            .throttler
            .poll_pty(&self.pty, &mut self.grid, &mut self.security);

        // Send responses back to PTY
        for response in &result.responses {
            pty_write(&mut self.pty, response);
        }

        if result.child_exited {
            let elapsed = self.spawn_time.elapsed();
            info!(
                "child exited after {:.1}ms",
                elapsed.as_secs_f64() * 1000.0
            );
            self.child_exited = true;
            self.exit_deadline = Some(Instant::now() + Duration::from_millis(100));
            self.window.request_redraw();
        }

        if result.bytes_consumed > 0 {
            self.scheduler.request_redraw();
        }
    }

    fn process_ipc_overlays(&mut self) {
        let Some(rx) = &self.ipc_ops_rx else { return };
        while let Ok(op) = rx.try_recv() {
            match op {
                scry_terminal::ipc_server::OverlayOp::Add {
                    id,
                    memfd,
                    px_w,
                    px_h,
                    anchor,
                    w_cells,
                    h_cells,
                    persist: _,
                    ..
                } => {
                    let rgba = memfd.read_bytes();

                    // Compute pixel origin from the anchor.
                    let cw = self.compositor.cell_width();
                    let ch = self.compositor.cell_height();
                    let pad = self.compositor.padding();
                    let (origin_x, origin_y) = match &anchor {
                        OverlayAnchor::Viewport { col, row } => {
                            (pad + *col as f32 * cw, pad + *row as f32 * ch)
                        }
                        OverlayAnchor::Canvas { col, row } => {
                            (pad + *col as f32 * cw, pad + *row as f32 * ch)
                        }
                    };

                    self.compositor
                        .set_overlay_rgba(px_w, px_h, &rgba, origin_x, origin_y);

                    // Retain the overlay state so Refresh can re-read.
                    self.active_overlays.insert(id, ActiveOverlay {
                        memfd,
                        px_w,
                        px_h,
                        anchor,
                        w_cells,
                        h_cells,
                    });
                    self.scheduler.request_redraw();
                }
                scry_terminal::ipc_server::OverlayOp::Refresh { id } => {
                    // Re-read the memfd — the CLI has written
                    // new pixel data in-place for this frame.
                    if let Some(ov) = self.active_overlays.get(&id) {
                        let rgba = ov.memfd.read_bytes();
                        let cw = self.compositor.cell_width();
                        let ch = self.compositor.cell_height();
                        let pad = self.compositor.padding();
                        let (ox, oy) = match &ov.anchor {
                            OverlayAnchor::Viewport { col, row } => {
                                (pad + *col as f32 * cw, pad + *row as f32 * ch)
                            }
                            OverlayAnchor::Canvas { col, row } => {
                                (pad + *col as f32 * cw, pad + *row as f32 * ch)
                            }
                        };
                        self.compositor
                            .set_overlay_rgba(ov.px_w, ov.px_h, &rgba, ox, oy);
                    }
                    self.scheduler.request_redraw();
                }
                scry_terminal::ipc_server::OverlayOp::Remove { id } => {
                    self.active_overlays.remove(&id);
                    if self.active_overlays.is_empty() {
                        self.compositor.clear_overlay();
                    }
                    self.scheduler.request_redraw();
                }
                scry_terminal::ipc_server::OverlayOp::ClearAll { persistent_ids, .. } => {
                    if persistent_ids.is_empty() {
                        // No persistent overlays — clear everything.
                        self.active_overlays.clear();
                        self.compositor.clear_overlay();
                    } else {
                        // Keep persistent overlays, remove the rest.
                        self.active_overlays.retain(|id, _| persistent_ids.contains(id));
                        if self.active_overlays.is_empty() {
                            self.compositor.clear_overlay();
                        }
                    }
                    self.scheduler.request_redraw();
                }
                scry_terminal::ipc_server::OverlayOp::AddScene {
                    id: _,
                    anchor: _,
                    z: _,
                    scene,
                    persist: _,
                    ..
                } => {
                    // Pass the scene graph to the compositor for GPU rendering.
                    self.compositor.set_overlay_scene(Some(scene));
                    self.scheduler.request_redraw();
                }
                scry_terminal::ipc_server::OverlayOp::AddAnimation {
                    id,
                    anchor,
                    duration_secs,
                    fps,
                    width,
                    height,
                    program,
                    ..
                } => {
                    let now = Instant::now();
                    let fps_clamped = fps.max(1).min(120);
                    let anim = super::ActiveAnimation {
                        id,
                        program,
                        pipeline: SdfPipeline::new(),
                        start_time: now,
                        duration: if duration_secs > 0 {
                            Some(Duration::from_secs(u64::from(duration_secs)))
                        } else {
                            None
                        },
                        frame_interval: Duration::from_secs_f64(1.0 / f64::from(fps_clamped)),
                        last_frame: now - Duration::from_secs(1), // force immediate first frame
                        width: width.max(1),
                        height: height.max(1),
                        paused: false,
                        visible: true,
                        anchor,
                        paused_elapsed: Duration::ZERO,
                    };
                    self.active_animations.push(anim);
                    self.scheduler.request_redraw();
                }
            }
        }
    }

    fn process_clipboard(&mut self) {
        if let Some(text) = self.grid.clipboard_pending.take() {
            if let Some(clip) = &mut self.clipboard {
                let _ = clip.set_text(&text);
            }
        }
    }

    fn process_bell(&mut self, config: &TerminalConfig) {
        if self.grid.bell_pending {
            self.grid.bell_pending = false;
            if config.bell.enabled && config.bell.visual {
                self.compositor.trigger_bell();
                self.scheduler.request_redraw();
            }
        }
    }

    /// Render a frame. Returns `true` on OOM (caller should exit).
    fn render_frame(&mut self) -> bool {
        if self.scheduler.should_render() || self.grid.has_dirty() {
            match self
                .compositor
                .render_frame(&self.grid, Some(&self.selection))
            {
                Ok(()) => {
                    self.scheduler.did_render();
                    self.grid.clear_dirty();
                }
                Err(wgpu::SurfaceError::Lost) => {
                    let size = self.window.inner_size();
                    self.compositor.resize(size.width, size.height);
                }
                Err(wgpu::SurfaceError::OutOfMemory) => {
                    error!("out of GPU memory");
                    return true;
                }
                Err(e) => {
                    warn!("surface error (recovered): {e}");
                }
            }
        }
        false
    }

    fn update_title(&mut self) {
        if !self.grid.title.is_empty() {
            self.window
                .set_title(&super::sanitize_title(&self.grid.title));
        }
    }

    /// Check if child has exited. Returns `true` if entering drain mode
    /// (caller should return early to start drain on next redraw).
    fn check_child_exit(&mut self) -> bool {
        // Skip during startup grace period to avoid racing with
        // shell initialization.
        if self.spawn_time.elapsed() > Duration::from_millis(500) {
            if let Some(status) = self.pty.try_wait() {
                info!(
                    "child exit status {:?} after {:.1}ms",
                    status,
                    self.spawn_time.elapsed().as_secs_f64() * 1000.0
                );
                self.child_exited = true;
                self.exit_deadline = Some(Instant::now() + Duration::from_millis(100));
                self.window.request_redraw();
                return true;
            }
        }
        false
    }

    fn schedule_animations(&self) {
        // With ControlFlow::Wait, we don't need to continuously request
        // redraws — the PTY waker will send a user event when data arrives.
        // But if bell is active, IPC overlays are live, or animations are
        // running, keep rendering so frames are processed promptly.
        if self.compositor.is_bell_active()
            || !self.active_overlays.is_empty()
            || !self.active_animations.is_empty()
        {
            self.window.request_redraw();
        }
    }

    /// Advance active animations: build SDF scenes, render frames, manage lifecycle.
    fn advance_animations(&mut self) {
        if self.active_animations.is_empty() {
            return;
        }

        let now = Instant::now();
        let mut rendered_any = false;

        for anim in &mut self.active_animations {
            // Skip if paused or not visible.
            if anim.paused || !anim.visible {
                continue;
            }

            // Check duration expiry.
            if let Some(dur) = anim.duration {
                let elapsed = anim.paused_elapsed + (now - anim.start_time);
                if elapsed >= dur {
                    continue; // duration exceeded, will be cleaned or frozen
                }
            }

            // Frame rate limiting.
            if now.duration_since(anim.last_frame) < anim.frame_interval {
                continue;
            }

            // Compute animation time (accounting for pause time).
            let t = (now
                .duration_since(anim.start_time)
                .saturating_sub(anim.paused_elapsed))
            .as_secs_f32();

            // Build the SDF scene for this time.
            let scene = anim.program.build_scene(t);

            // Render to pixmap via the SDF pipeline.
            let result = anim.pipeline.render(&scene, anim.width, anim.height, t);

            // Convert ImageData to RGBA bytes and set as overlay.
            let rgba = result.image.data();
            let cw = self.compositor.cell_width();
            let ch = self.compositor.cell_height();
            let pad = self.compositor.padding();
            let (ox, oy) = match &anim.anchor {
                scry_engine::transport::ipc::OverlayAnchor::Viewport { col, row } => {
                    (pad + *col as f32 * cw, pad + *row as f32 * ch)
                }
                scry_engine::transport::ipc::OverlayAnchor::Canvas { col, row } => {
                    (pad + *col as f32 * cw, pad + *row as f32 * ch)
                }
            };

            self.compositor
                .set_overlay_rgba(result.width, result.height, rgba, ox, oy);

            anim.last_frame = now;
            rendered_any = true;
        }

        // Remove expired animations.
        self.active_animations.retain(|anim| {
            if let Some(dur) = anim.duration {
                let elapsed = anim.paused_elapsed
                    + Instant::now().duration_since(anim.start_time);
                elapsed < dur
            } else {
                true // infinite animations never expire
            }
        });

        if rendered_any {
            self.scheduler.request_redraw();
        }
    }
}
