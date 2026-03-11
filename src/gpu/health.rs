// SPDX-License-Identifier: MIT OR Apache-2.0
//! Unified GPU health monitoring and backend selection.
//!
//! [`GpuHealthMonitor`] provides a single state machine that both the SDF
//! pipeline and the 2D rasterizer consult when deciding whether to use GPU
//! or CPU rendering.  This eliminates the previous architecture where three
//! independent layers each made their own GPU/CPU decision.
//!
//! # State Machine
//!
//! ```text
//! Initializing ──► Healthy ──► Degraded ──► Dead
//!       ▲                          │           │
//!       └──────────────────────────┘           │
//!       └──────────────────────────────────────┘
//!                (recovery after cooldown)
//! ```
//!
//! # Thread Safety
//!
//! The monitor is wrapped in `Arc<Mutex<..>>` by [`GpuDevice`](super::GpuDevice)
//! and shared across all rendering contexts.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ── Configuration constants ────────────────────────────────────────

/// Number of frames to wait after GPU init before dispatching GPU work.
/// Prevents visual discontinuity at the CPU→GPU switchover.
const WARMUP_FRAMES: u32 = 3;

/// Number of consecutive GPU successes needed to recover from `Degraded`.
const RECOVERY_THRESHOLD: u32 = 3;

/// Number of consecutive GPU failures to transition from `Healthy`/`Degraded` to `Dead`.
const DEATH_THRESHOLD: u32 = 5;

/// How long to wait before attempting GPU recovery from `Dead`.
const RECOVERY_COOLDOWN: Duration = Duration::from_secs(30);

/// When the rasterizer falls back to CPU, keep it on CPU for this many
/// frames before re-checking `gpu_suitable()`.  Prevents per-frame flipping.
const RASTER_STICKY_FRAMES: u32 = 60;

// ── Types ──────────────────────────────────────────────────────────

/// GPU health state.
#[derive(Clone, Debug)]
pub enum GpuHealth {
    /// GPU device is being initialized (background thread running).
    Initializing,
    /// GPU is working normally.
    Healthy,
    /// GPU had recent failures but may recover.
    Degraded {
        /// Number of consecutive failures since entering `Degraded`.
        consecutive_failures: u32,
        /// Number of consecutive successes (resets on failure).
        consecutive_successes: u32,
    },
    /// GPU is considered offline.  CPU-only until recovery cooldown.
    Dead {
        /// When the monitor entered `Dead` state.
        since: Instant,
    },
}

/// Shared GPU health monitor.
///
/// All GPU/CPU selection decisions flow through this struct.  The SDF
/// pipeline calls [`should_use_gpu_sdf`] and the 2D rasterizer calls
/// [`should_use_gpu_raster`].
pub struct GpuHealthMonitor {
    state: GpuHealth,
    /// Warmup counter — counts down from [`WARMUP_FRAMES`] to 0 after init.
    warmup_remaining: u32,
    /// Sticky CPU counter for the 2D rasterizer — when >0, always return
    /// false from [`should_use_gpu_raster`].
    raster_cpu_sticky: u32,
}

impl GpuHealthMonitor {
    /// Create a new monitor in `Initializing` state.
    pub fn new() -> Self {
        Self {
            state: GpuHealth::Initializing,
            warmup_remaining: 0,
            raster_cpu_sticky: 0,
        }
    }

    /// Create a monitor that starts in `Dead` state (for CPU-only pipelines).
    pub fn cpu_only() -> Self {
        Self {
            state: GpuHealth::Dead {
                since: Instant::now(),
            },
            warmup_remaining: 0,
            raster_cpu_sticky: 0,
        }
    }

    /// Current health state.
    pub fn state(&self) -> &GpuHealth {
        &self.state
    }

    // ── Transitions ────────────────────────────────────────────────

    /// Mark GPU as successfully initialized.
    ///
    /// Transitions `Initializing → Healthy` with a warmup delay.
    pub fn mark_initialized(&mut self) {
        if matches!(self.state, GpuHealth::Initializing) {
            self.state = GpuHealth::Healthy;
            self.warmup_remaining = WARMUP_FRAMES;
        }
    }

    /// Mark GPU init as failed.
    ///
    /// Transitions `Initializing → Dead`.
    pub fn mark_init_failed(&mut self) {
        if matches!(self.state, GpuHealth::Initializing) {
            self.state = GpuHealth::Dead {
                since: Instant::now(),
            };
        }
    }

    /// Report a successful GPU frame.
    pub fn report_success(&mut self) {
        match &mut self.state {
            GpuHealth::Degraded {
                consecutive_successes,
                consecutive_failures,
            } => {
                *consecutive_successes += 1;
                if *consecutive_successes >= RECOVERY_THRESHOLD {
                    self.state = GpuHealth::Healthy;
                } else {
                    // Reset failure counter on success
                    *consecutive_failures = 0;
                }
            }
            GpuHealth::Healthy => {}
            // Success in other states is a no-op (shouldn't happen)
            _ => {}
        }
    }

    /// Report a failed GPU frame.
    pub fn report_failure(&mut self) {
        match &mut self.state {
            GpuHealth::Healthy => {
                self.state = GpuHealth::Degraded {
                    consecutive_failures: 1,
                    consecutive_successes: 0,
                };
            }
            GpuHealth::Degraded {
                consecutive_failures,
                consecutive_successes,
            } => {
                *consecutive_failures += 1;
                *consecutive_successes = 0;
                if *consecutive_failures >= DEATH_THRESHOLD {
                    self.state = GpuHealth::Dead {
                        since: Instant::now(),
                    };
                }
            }
            // Failure in Dead/Initializing is a no-op
            _ => {}
        }
    }

    /// Attempt recovery from `Dead` state.
    ///
    /// Returns `true` if the cooldown has elapsed and the monitor
    /// transitions to `Initializing`.
    pub fn attempt_recovery(&mut self) -> bool {
        if let GpuHealth::Dead { since } = &self.state {
            if since.elapsed() >= RECOVERY_COOLDOWN {
                self.state = GpuHealth::Initializing;
                return true;
            }
        }
        false
    }

    // ── Decision queries ───────────────────────────────────────────

    /// Should the SDF pipeline use GPU for this frame?
    ///
    /// Returns `false` during warmup, when dead, or when initializing.
    pub fn should_use_gpu_sdf(&mut self) -> bool {
        // Tick warmup
        if self.warmup_remaining > 0 {
            self.warmup_remaining -= 1;
            return false;
        }

        matches!(self.state, GpuHealth::Healthy | GpuHealth::Degraded { .. })
    }

    /// Should the 2D rasterizer use GPU for this frame?
    ///
    /// Incorporates the `gpu_suitable` flag from the canvas and applies
    /// sticky CPU hold logic to prevent per-frame flipping.
    pub fn should_use_gpu_raster(&mut self, canvas_gpu_suitable: bool) -> bool {
        // If GPU isn't healthy at all, always no
        if !matches!(self.state, GpuHealth::Healthy | GpuHealth::Degraded { .. }) {
            return false;
        }

        // Sticky CPU hold: once we fall back, stay on CPU for N frames
        if self.raster_cpu_sticky > 0 {
            self.raster_cpu_sticky -= 1;
            return false;
        }

        if !canvas_gpu_suitable {
            self.raster_cpu_sticky = RASTER_STICKY_FRAMES;
            return false;
        }

        true
    }

    /// Returns `true` if the GPU is healthy or degraded (i.e., available).
    pub fn is_gpu_available(&self) -> bool {
        matches!(self.state, GpuHealth::Healthy | GpuHealth::Degraded { .. })
    }
}

impl Default for GpuHealthMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for GpuHealthMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuHealthMonitor")
            .field("state", &self.state)
            .field("warmup_remaining", &self.warmup_remaining)
            .field("raster_cpu_sticky", &self.raster_cpu_sticky)
            .finish()
    }
}

/// Convenience type alias for shared health monitor.
pub type SharedHealthMonitor = Arc<Mutex<GpuHealthMonitor>>;

/// Create a new shared health monitor.
pub fn shared_health_monitor() -> SharedHealthMonitor {
    Arc::new(Mutex::new(GpuHealthMonitor::new()))
}

/// Create a shared health monitor in CPU-only mode.
pub fn shared_cpu_only_monitor() -> SharedHealthMonitor {
    Arc::new(Mutex::new(GpuHealthMonitor::cpu_only()))
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_starts_initializing() {
        let mut monitor = GpuHealthMonitor::new();
        assert!(matches!(monitor.state(), GpuHealth::Initializing));
        // Should NOT use GPU while initializing
        assert!(!monitor.should_use_gpu_sdf());
    }

    #[test]
    fn health_transitions_to_healthy_with_warmup() {
        let mut monitor = GpuHealthMonitor::new();
        monitor.mark_initialized();
        assert!(matches!(monitor.state(), GpuHealth::Healthy));

        // Warmup frames should return false
        for _ in 0..WARMUP_FRAMES {
            assert!(
                !monitor.should_use_gpu_sdf(),
                "should be false during warmup"
            );
        }
        // After warmup, should return true
        assert!(monitor.should_use_gpu_sdf(), "should be true after warmup");
    }

    #[test]
    fn health_degrades_on_failure() {
        let mut monitor = GpuHealthMonitor::new();
        monitor.mark_initialized();
        // Drain warmup
        for _ in 0..WARMUP_FRAMES {
            monitor.should_use_gpu_sdf();
        }

        monitor.report_failure();
        assert!(matches!(monitor.state(), GpuHealth::Degraded { .. }));
        // Still usable when degraded
        assert!(monitor.should_use_gpu_sdf());
    }

    #[test]
    fn health_dies_on_repeated_failures() {
        let mut monitor = GpuHealthMonitor::new();
        monitor.mark_initialized();
        for _ in 0..WARMUP_FRAMES {
            monitor.should_use_gpu_sdf();
        }

        for _ in 0..DEATH_THRESHOLD {
            monitor.report_failure();
        }
        assert!(matches!(monitor.state(), GpuHealth::Dead { .. }));
        assert!(!monitor.should_use_gpu_sdf());
    }

    #[test]
    fn health_recovers_after_cooldown() {
        let mut monitor = GpuHealthMonitor::new();
        // Force into Dead state with a time in the past
        monitor.state = GpuHealth::Dead {
            since: Instant::now() - RECOVERY_COOLDOWN - Duration::from_secs(1),
        };

        assert!(monitor.attempt_recovery());
        assert!(matches!(monitor.state(), GpuHealth::Initializing));
    }

    #[test]
    fn health_no_recovery_before_cooldown() {
        let mut monitor = GpuHealthMonitor::new();
        monitor.state = GpuHealth::Dead {
            since: Instant::now(),
        };

        assert!(!monitor.attempt_recovery());
        assert!(matches!(monitor.state(), GpuHealth::Dead { .. }));
    }

    #[test]
    fn raster_sticky_cpu_hold() {
        let mut monitor = GpuHealthMonitor::new();
        monitor.mark_initialized();
        for _ in 0..WARMUP_FRAMES {
            monitor.should_use_gpu_sdf();
        }

        // First call with gpu_suitable=false triggers sticky hold
        assert!(!monitor.should_use_gpu_raster(false));

        // Next RASTER_STICKY_FRAMES calls should return false even with gpu_suitable=true
        for i in 0..RASTER_STICKY_FRAMES {
            assert!(
                !monitor.should_use_gpu_raster(true),
                "frame {i} should be sticky CPU"
            );
        }

        // After sticky period, should return true again
        assert!(monitor.should_use_gpu_raster(true));
    }

    #[test]
    fn health_success_recovers_from_degraded() {
        let mut monitor = GpuHealthMonitor::new();
        monitor.mark_initialized();
        for _ in 0..WARMUP_FRAMES {
            monitor.should_use_gpu_sdf();
        }

        // Degrade
        monitor.report_failure();
        assert!(matches!(monitor.state(), GpuHealth::Degraded { .. }));

        // Recover
        for _ in 0..RECOVERY_THRESHOLD {
            monitor.report_success();
        }
        assert!(matches!(monitor.state(), GpuHealth::Healthy));
    }

    #[test]
    fn init_failed_goes_to_dead() {
        let mut monitor = GpuHealthMonitor::new();
        monitor.mark_init_failed();
        assert!(matches!(monitor.state(), GpuHealth::Dead { .. }));
        assert!(!monitor.should_use_gpu_sdf());
    }

    #[test]
    fn cpu_only_monitor_always_refuses_gpu() {
        let mut monitor = GpuHealthMonitor::cpu_only();
        assert!(!monitor.should_use_gpu_sdf());
        assert!(!monitor.should_use_gpu_raster(true));
    }

    #[test]
    fn shared_monitor_creation() {
        let shared = shared_health_monitor();
        let guard = shared.lock().unwrap();
        assert!(matches!(guard.state(), GpuHealth::Initializing));
    }
}
