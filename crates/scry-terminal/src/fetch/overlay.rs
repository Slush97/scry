// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fetch overlay controller — manages the lifecycle of the fetch splash.
//!
//! State machine: `Inactive` → `FadeIn` → `Active` → `FadeOut` → `Inactive`.
//! Handles timing, alpha computation, and scene generation.

use std::time::{Duration, Instant};

use scry_engine::scene::PixelCanvas;

use crate::config::FetchConfig;

use super::renderer::FetchRenderer;
use super::sysinfo::SysInfo;

/// Fetch overlay state.
enum FetchState {
    /// Not active — no overlay shown.
    Inactive,
    /// Fading in (0.0 → 1.0).
    FadeIn { start: Instant },
    /// Fully visible, animating.
    Active { start: Instant },
    /// Fading out (1.0 → 0.0).
    FadeOut { start: Instant },
}

/// The fetch overlay controller.
pub struct FetchOverlay {
    state: FetchState,
    renderer: FetchRenderer,
    /// Total animation start time (for continuous `t` parameter).
    anim_start: Instant,
    /// Duration to stay fully visible before auto-dismiss.
    active_duration: Duration,
    /// Duration of the fade-in.
    fade_in_duration: Duration,
    /// Duration of the fade-out.
    fade_out_duration: Duration,
    /// Cached banner height.
    banner_height: u32,
}

/// Result of a tick: the scene to render and the current alpha.
pub struct FetchFrame {
    /// The `PixelCanvas` scene for this frame.
    pub canvas: PixelCanvas,
    /// Global alpha (0.0–1.0) for compositor fade.
    pub alpha: f32,
}

impl FetchOverlay {
    /// Create a new fetch overlay from config.
    ///
    /// This collects system info (may block briefly for package count).
    /// Call `activate()` to start the display.
    pub fn new(config: &FetchConfig) -> Self {
        let info = SysInfo::collect();
        let renderer = FetchRenderer::new(config, info);

        Self {
            state: FetchState::Inactive,
            renderer,
            anim_start: Instant::now(),
            active_duration: Duration::from_secs_f32(config.duration),
            fade_in_duration: Duration::from_millis(300),
            fade_out_duration: Duration::from_secs_f32(config.fade_duration),
            banner_height: 0,
        }
    }

    /// Activate the fetch overlay (begin fade-in).
    pub fn activate(&mut self) {
        let now = Instant::now();
        self.anim_start = now;
        self.state = FetchState::FadeIn { start: now };
    }

    /// Dismiss the fetch overlay (begin fade-out). Called on keypress or timeout.
    pub fn dismiss(&mut self) {
        match self.state {
            FetchState::Active { .. } | FetchState::FadeIn { .. } => {
                self.state = FetchState::FadeOut {
                    start: Instant::now(),
                };
            }
            _ => {}
        }
    }

    /// Whether the overlay is currently showing (any state except Inactive).
    pub fn is_active(&self) -> bool {
        !matches!(self.state, FetchState::Inactive)
    }

    /// Compute the banner height for the current terminal dimensions.
    pub fn compute_banner_height(&mut self, cell_height: f32, padding: f32) -> u32 {
        self.banner_height = self.renderer.banner_height(cell_height, padding);
        self.banner_height
    }

    /// Advance the state machine and return the frame to render.
    ///
    /// Returns `None` when the overlay has fully faded out and should be removed.
    pub fn tick(&mut self, screen_width: u32) -> Option<FetchFrame> {
        let now = Instant::now();

        match self.state {
            FetchState::Inactive => return None,

            FetchState::FadeIn { start } => {
                let elapsed = now.duration_since(start);
                if elapsed >= self.fade_in_duration {
                    // Transition to Active
                    self.state = FetchState::Active { start: now };
                }
            }

            FetchState::Active { start } => {
                let elapsed = now.duration_since(start);
                if self.active_duration > Duration::ZERO && elapsed >= self.active_duration {
                    // Auto-dismiss
                    self.state = FetchState::FadeOut { start: now };
                }
            }

            FetchState::FadeOut { start } => {
                let elapsed = now.duration_since(start);
                if elapsed >= self.fade_out_duration {
                    self.state = FetchState::Inactive;
                    return None;
                }
            }
        }

        // Compute alpha
        let alpha = match &self.state {
            FetchState::Inactive => return None,
            FetchState::FadeIn { start } => {
                let p = now.duration_since(*start).as_secs_f32()
                    / self.fade_in_duration.as_secs_f32();
                ease_in_out(p.clamp(0.0, 1.0))
            }
            FetchState::Active { .. } => 1.0,
            FetchState::FadeOut { start } => {
                let p = now.duration_since(*start).as_secs_f32()
                    / self.fade_out_duration.as_secs_f32();
                1.0 - ease_in_out(p.clamp(0.0, 1.0))
            }
        };

        // Animation time (continuous from activation)
        let t = now.duration_since(self.anim_start).as_secs_f32();

        // Render the scene
        let canvas = self.renderer.render(screen_width, self.banner_height, t);

        Some(FetchFrame { canvas, alpha })
    }
}

/// Smooth ease-in-out curve (cubic).
fn ease_in_out(t: f32) -> f32 {
    if t < 0.5 {
        4.0 * t * t * t
    } else {
        1.0 - (-2.0f32).mul_add(t, 2.0).powi(3) / 2.0
    }
}
