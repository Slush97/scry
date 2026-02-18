// SPDX-License-Identifier: MIT OR Apache-2.0
//! Per-stage profiling for the SDF ray marcher.
//!
//! Provides timing data for each rendering stage (march, shadow, normal,
//! shading, reflection, fire) and a live bar chart renderer for terminal
//! display.

use std::collections::VecDeque;
use std::time::Instant;

/// Rendering stages tracked by the SDF profiler.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SdfStage {
    /// Ray marching (sphere tracing to find surface hits).
    March,
    /// Shadow ray tests against lights.
    Shadow,
    /// Surface normal estimation.
    Normal,
    /// Phong diffuse + specular computation.
    Shading,
    /// Mirror/water reflection and refraction bounces.
    Reflection,
    /// Volumetric fire ray marching.
    Fire,
}

impl SdfStage {
    /// All stages in display order.
    pub const ALL: [Self; 6] = [
        Self::March,
        Self::Shadow,
        Self::Normal,
        Self::Shading,
        Self::Reflection,
        Self::Fire,
    ];

    /// Index into the `stage_us` arrays.
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    /// Short display name.
    pub fn label(self) -> &'static str {
        match self {
            Self::March => "march",
            Self::Shadow => "shadow",
            Self::Normal => "normal",
            Self::Shading => "shade",
            Self::Reflection => "refl",
            Self::Fire => "fire",
        }
    }

    /// ANSI color code for the bar chart.
    pub fn ansi_color(self) -> &'static str {
        match self {
            Self::March => "\x1b[34m",      // blue
            Self::Shadow => "\x1b[37m",     // white/gray
            Self::Normal => "\x1b[33m",     // orange/yellow
            Self::Shading => "\x1b[93m",    // bright yellow
            Self::Reflection => "\x1b[35m", // purple
            Self::Fire => "\x1b[31m",       // red
        }
    }
}

/// Per-row timing accumulator. Created on the stack for each scanline.
#[derive(Clone, Debug, Default)]
pub struct RowProfile {
    /// Microseconds spent in each [`SdfStage`] for this row.
    pub stage_us: [u64; 6],
}

impl RowProfile {
    /// Create a new zeroed row profile.
    #[inline]
    pub fn new() -> Self {
        Self { stage_us: [0; 6] }
    }

    /// Record elapsed microseconds for a stage.
    #[inline]
    pub fn record(&mut self, stage: SdfStage, start: Instant) {
        self.stage_us[stage.index()] += start.elapsed().as_micros() as u64;
    }

    /// Merge another row's timings into this one.
    #[inline]
    pub fn merge(&mut self, other: &Self) {
        for i in 0..6 {
            self.stage_us[i] += other.stage_us[i];
        }
    }
}

/// Frame-level profile aggregated from all rows.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SdfProfile {
    /// Total frame time in microseconds.
    pub total_us: u64,
    /// Per-stage accumulated microseconds (summed across all rows/threads).
    pub stage_us: [u64; 6],
    /// Render width.
    pub width: u32,
    /// Render height.
    pub height: u32,
}

impl SdfProfile {
    /// Create a profile with only the total frame time (no per-stage breakdown).
    ///
    /// Useful for GPU rendering where stage breakdown isn't available.
    pub fn total_only(total_us: u64, width: u32, height: u32) -> Self {
        Self {
            total_us,
            stage_us: [0; 6],
            width,
            height,
        }
    }

    /// Aggregate row profiles into a frame profile.
    pub fn from_rows(rows: &[RowProfile], total_us: u64, width: u32, height: u32) -> Self {
        let mut stage_us = [0u64; 6];
        for row in rows {
            for i in 0..6 {
                stage_us[i] += row.stage_us[i];
            }
        }
        Self {
            total_us,
            stage_us,
            width,
            height,
        }
    }

    /// Return stages sorted by descending time, filtering out zeros.
    pub fn active_stages_sorted(&self) -> Vec<(SdfStage, u64)> {
        let mut stages: Vec<(SdfStage, u64)> = SdfStage::ALL
            .iter()
            .filter(|s| self.stage_us[s.index()] > 0)
            .map(|s| (*s, self.stage_us[s.index()]))
            .collect();
        stages.sort_by(|a, b| b.1.cmp(&a.1));
        stages
    }
}

/// Smoothed profile computed from the rolling history (median per stage).
#[derive(Clone, Debug)]
pub struct SmoothedSdfProfile {
    /// Median total frame time in microseconds.
    pub total_us: u64,
    /// Median per-stage microseconds.
    pub stage_us: [u64; 6],
}

/// Rolling history of frame profiles for smoothing.
pub struct SdfProfileHistory {
    frames: VecDeque<SdfProfile>,
    capacity: usize,
}

impl SdfProfileHistory {
    /// Create a new history with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            frames: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Push a new frame profile, evicting the oldest if full.
    pub fn push(&mut self, profile: SdfProfile) {
        if self.frames.len() >= self.capacity {
            self.frames.pop_front();
        }
        self.frames.push_back(profile);
    }

    /// Compute the median of each stage across the history.
    pub fn summary(&self) -> SmoothedSdfProfile {
        if self.frames.is_empty() {
            return SmoothedSdfProfile {
                total_us: 0,
                stage_us: [0; 6],
            };
        }

        let n = self.frames.len();
        let mut stage_us = [0u64; 6];

        for i in 0..6 {
            let mut vals: Vec<u64> = self.frames.iter().map(|f| f.stage_us[i]).collect();
            vals.sort_unstable();
            stage_us[i] = vals[n / 2];
        }

        let mut totals: Vec<u64> = self.frames.iter().map(|f| f.total_us).collect();
        totals.sort_unstable();

        SmoothedSdfProfile {
            total_us: totals[n / 2],
            stage_us,
        }
    }
}

const RESET: &str = "\x1b[0m";

/// Render a colored horizontal bar chart string for the given smoothed profile.
///
/// `bar_width` is the number of block characters available for the bar.
pub fn render_profile_bar(profile: &SmoothedSdfProfile, bar_width: usize) -> String {
    let stage_total: u64 = profile.stage_us.iter().sum();
    if stage_total == 0 {
        return String::new();
    }

    let total_ms = profile.total_us as f64 / 1000.0;

    // Build bar
    let mut bar = String::with_capacity(bar_width * 4 + 128);
    bar.push('[');

    let mut chars_used = 0;
    let active: Vec<(SdfStage, u64)> = SdfStage::ALL
        .iter()
        .filter(|s| profile.stage_us[s.index()] > 0)
        .map(|s| (*s, profile.stage_us[s.index()]))
        .collect();

    for (i, (stage, us)) in active.iter().enumerate() {
        let frac = *us as f64 / stage_total as f64;
        let n = if i == active.len() - 1 {
            // Last stage gets remaining chars to avoid rounding gaps
            bar_width - chars_used
        } else {
            let n = (frac * bar_width as f64).round() as usize;
            n.min(bar_width - chars_used)
        };
        if n > 0 {
            bar.push_str(stage.ansi_color());
            for _ in 0..n {
                bar.push('\u{2588}'); // █
            }
            bar.push_str(RESET);
            chars_used += n;
        }
    }

    bar.push(']');

    // Legend
    for (stage, us) in &active {
        let ms = *us as f64 / 1000.0;
        bar.push_str(&format!(
            " {}{}{} {:.1}ms",
            stage.ansi_color(),
            stage.label(),
            RESET,
            ms
        ));
    }

    bar.push_str(&format!(" | {total_ms:.1}ms total"));

    bar
}
