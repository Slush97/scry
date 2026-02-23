// SPDX-License-Identifier: MIT OR Apache-2.0
//! Unified engine health diagnostics.
//!
//! [`EngineReport`] aggregates GPU availability, backend selection, active
//! feature flags, and scene-level warnings into a single snapshot that can
//! be printed or logged.
//!
//! # Example
//!
//! ```
//! use scry_engine::diagnostics::EngineReport;
//!
//! let report = EngineReport::snapshot();
//! println!("{report}");
//! ```

use crate::rasterize::backend::BackendKind;
use crate::scene::validate::{SceneWarning, validate_scene};

/// Unified engine health snapshot.
///
/// Collects GPU availability, backend selection, feature flags, and optional
/// scene warnings into a single report.
#[derive(Debug)]
pub struct EngineReport {
    /// Whether a GPU adapter was successfully initialized.
    pub gpu_available: bool,
    /// Human-readable GPU adapter name (e.g., "NVIDIA GeForce RTX 4090").
    pub gpu_adapter: Option<String>,
    /// Human-readable GPU health state.
    pub gpu_health: String,
    /// Which raster backend is active.
    pub active_backend: BackendKind,
    /// Compile-time features enabled in this build.
    pub features: Vec<&'static str>,
    /// Scene-level warnings (empty if no canvas was provided).
    pub scene_warnings: Vec<SceneWarning>,
}

impl EngineReport {
    /// Take a snapshot of the engine's current state (no scene analysis).
    #[must_use]
    pub fn snapshot() -> Self {
        let (gpu_available, gpu_adapter, gpu_health) = Self::probe_gpu();
        Self {
            gpu_available,
            gpu_adapter,
            gpu_health,
            active_backend: Self::detect_backend(),
            features: Self::active_features(),
            scene_warnings: Vec::new(),
        }
    }

    /// Take a snapshot **and** validate the given canvas.
    #[must_use]
    pub fn for_canvas(canvas: &crate::scene::PixelCanvas) -> Self {
        let mut report = Self::snapshot();
        report.scene_warnings = validate_scene(canvas);
        report
    }

    // ── Internal helpers ──

    fn probe_gpu() -> (bool, Option<String>, String) {
        #[cfg(feature = "gpu")]
        {
            use crate::gpu::GpuDevice;
            match GpuDevice::global() {
                Some(gpu) => {
                    let info = gpu.info();
                    let health = gpu
                        .health()
                        .lock()
                        .map(|h| format!("{:?}", h.state()))
                        .unwrap_or_else(|_| "lock_poisoned".to_string());
                    (true, Some(info.adapter_name.clone()), health)
                }
                None => (false, None, "unavailable".to_string()),
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            (false, None, "gpu feature disabled".to_string())
        }
    }

    fn detect_backend() -> BackendKind {
        #[cfg(feature = "gpu")]
        {
            use crate::gpu::GpuDevice;
            if GpuDevice::global().is_some() {
                return BackendKind::Gpu;
            }
        }
        BackendKind::Cpu
    }

    fn active_features() -> Vec<&'static str> {
        let mut feats = Vec::new();

        #[cfg(feature = "gpu")]
        feats.push("gpu");
        #[cfg(feature = "kitty")]
        feats.push("kitty");
        #[cfg(feature = "sixel")]
        feats.push("sixel");
        #[cfg(feature = "iterm2")]
        feats.push("iterm2");
        #[cfg(feature = "text")]
        feats.push("text");
        #[cfg(feature = "sdf")]
        feats.push("sdf");
        #[cfg(feature = "sdf-gpu")]
        feats.push("sdf-gpu");
        #[cfg(feature = "sdf-text")]
        feats.push("sdf-text");
        #[cfg(feature = "widget")]
        feats.push("widget");
        #[cfg(feature = "svg")]
        feats.push("svg");
        #[cfg(feature = "logging")]
        feats.push("logging");
        #[cfg(feature = "input")]
        feats.push("input");
        #[cfg(feature = "native-ipc")]
        feats.push("native-ipc");
        #[cfg(feature = "window")]
        feats.push("window");

        feats
    }
}

impl std::fmt::Display for EngineReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "─── scry-engine diagnostics ───")?;
        writeln!(f, "GPU available : {}", self.gpu_available)?;
        if let Some(name) = &self.gpu_adapter {
            writeln!(f, "GPU adapter   : {name}")?;
        }
        writeln!(f, "GPU health    : {}", self.gpu_health)?;
        writeln!(f, "Backend       : {:?}", self.active_backend)?;
        writeln!(f, "Features      : {}", self.features.join(", "))?;

        if self.scene_warnings.is_empty() {
            writeln!(f, "Scene warnings: (none)")?;
        } else {
            writeln!(f, "Scene warnings: {}", self.scene_warnings.len())?;
            for w in &self.scene_warnings {
                writeln!(f, "  {w}")?;
            }
        }

        write!(f, "───────────────────────────────")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::style::Color;
    use crate::scene::PixelCanvas;

    #[test]
    fn snapshot_succeeds() {
        let report = EngineReport::snapshot();
        // Backend should always be one of Cpu or Gpu
        assert!(
            report.active_backend == BackendKind::Cpu
                || report.active_backend == BackendKind::Gpu,
        );
        // Display should produce non-empty output
        let display = format!("{report}");
        assert!(!display.is_empty());
    }

    #[test]
    fn for_canvas_includes_warnings() {
        let canvas = PixelCanvas::new(100, 100)
            .circle(50.0, 50.0, 0.0) // zero radius
            .fill(Color::RED)
            .done();

        let report = EngineReport::for_canvas(&canvas);
        assert!(
            !report.scene_warnings.is_empty(),
            "expected warnings for zero-radius circle",
        );
    }

    #[test]
    fn display_format_is_human_readable() {
        let report = EngineReport::snapshot();
        let s = format!("{report}");
        assert!(s.contains("GPU available"));
        assert!(s.contains("Backend"));
        assert!(s.contains("Features"));
    }
}
