//! Per-stage pipeline profiling for bottleneck identification.
//!
//! This module provides zero-overhead instrumentation that can be
//! enabled per-frame to measure exactly where time is spent in the
//! rasterization and transport pipelines.
//!
//! # Usage
//!
//! ```ignore
//! let profile = Rasterizer::rasterize_into_profiled(&canvas, &mut pixmap);
//! // profile.raster_by_type contains per-command-type timing
//! ```

use std::fmt;
use std::time::Instant;

use tiny_skia::Pixmap;

use crate::scene::command::DrawCommand;
use crate::scene::PixelCanvas;

// ---------------------------------------------------------------------------
// Command type index (matches DrawCommand discriminant order)
// ---------------------------------------------------------------------------

/// Index into the per-command-type timing array.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CommandType {
    /// `DrawCommand::Clear`
    Clear = 0,
    /// `DrawCommand::Circle`
    Circle = 1,
    /// `DrawCommand::Rectangle`
    Rectangle = 2,
    /// `DrawCommand::Ellipse`
    Ellipse = 3,
    /// `DrawCommand::Line`
    Line = 4,
    /// `DrawCommand::Path`
    Path = 5,
    /// `DrawCommand::Polyline`
    Polyline = 6,
    /// `DrawCommand::Gradient`
    Gradient = 7,
    /// `DrawCommand::Arc`
    Arc = 8,
    /// `DrawCommand::Image`
    Image = 9,
    /// `DrawCommand::Text`
    Text = 10,
    /// `DrawCommand::Group`
    Group = 11,
}

/// Total number of command types we track.
pub const NUM_COMMAND_TYPES: usize = 12;

/// All command type variants for iteration.
pub const ALL_COMMAND_TYPES: [CommandType; NUM_COMMAND_TYPES] = [
    CommandType::Clear,
    CommandType::Circle,
    CommandType::Rectangle,
    CommandType::Ellipse,
    CommandType::Line,
    CommandType::Path,
    CommandType::Polyline,
    CommandType::Gradient,
    CommandType::Arc,
    CommandType::Image,
    CommandType::Text,
    CommandType::Group,
];

impl CommandType {
    /// Classify a `DrawCommand` into its `CommandType`.
    #[must_use]
    pub fn from_command(cmd: &DrawCommand) -> Self {
        match cmd {
            DrawCommand::Clear { .. } => Self::Clear,
            DrawCommand::Circle { .. } => Self::Circle,
            DrawCommand::Rectangle { .. } => Self::Rectangle,
            DrawCommand::Ellipse { .. } => Self::Ellipse,
            DrawCommand::Line { .. } => Self::Line,
            DrawCommand::Path { .. } => Self::Path,
            DrawCommand::Polyline { .. } => Self::Polyline,
            DrawCommand::Gradient { .. } => Self::Gradient,
            DrawCommand::Arc { .. } => Self::Arc,
            DrawCommand::Image { .. } => Self::Image,
            #[cfg(feature = "text")]
            DrawCommand::Text { .. } => Self::Text,
            DrawCommand::Group { .. } => Self::Group,
        }
    }

    /// Short display name for status bar.
    #[must_use]
    pub const fn short_name(self) -> &'static str {
        match self {
            Self::Clear => "clr",
            Self::Circle => "cir",
            Self::Rectangle => "rct",
            Self::Ellipse => "ell",
            Self::Line => "lin",
            Self::Path => "pth",
            Self::Polyline => "ply",
            Self::Gradient => "grd",
            Self::Arc => "arc",
            Self::Image => "img",
            Self::Text => "txt",
            Self::Group => "grp",
        }
    }

    /// Full display name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Clear => "Clear",
            Self::Circle => "Circle",
            Self::Rectangle => "Rect",
            Self::Ellipse => "Ellipse",
            Self::Line => "Line",
            Self::Path => "Path",
            Self::Polyline => "Polyline",
            Self::Gradient => "Gradient",
            Self::Arc => "Arc",
            Self::Image => "Image",
            Self::Text => "Text",
            Self::Group => "Group",
        }
    }
}

// ---------------------------------------------------------------------------
// Per-command-type timing
// ---------------------------------------------------------------------------

/// Accumulated timing statistics for a single command type.
#[derive(Clone, Debug, Default)]
pub struct CommandTiming {
    /// Number of commands of this type rasterized.
    pub count: u32,
    /// Total time spent rasterizing this command type, in microseconds.
    pub total_us: u64,
    /// Maximum time for a single command of this type, in microseconds.
    pub max_us: u64,
}

impl CommandTiming {
    /// Record a single command's elapsed time.
    pub fn record(&mut self, elapsed_us: u64) {
        self.count += 1;
        self.total_us += elapsed_us;
        if elapsed_us > self.max_us {
            self.max_us = elapsed_us;
        }
    }

    /// Average time per command in microseconds (0 if no commands).
    #[must_use]
    pub fn avg_us(&self) -> u64 {
        if self.count == 0 { 0 } else { self.total_us / u64::from(self.count) }
    }
}

// ---------------------------------------------------------------------------
// Raster profile
// ---------------------------------------------------------------------------

/// Profiling data from a single rasterization pass.
#[derive(Clone, Debug)]
pub struct RasterProfile {
    /// Total rasterization time in microseconds (background fill + all commands).
    pub total_us: u64,
    /// Background fill time in microseconds.
    pub background_us: u64,
    /// Per-command-type timing breakdown.
    pub by_type: [CommandTiming; NUM_COMMAND_TYPES],
    /// Total number of commands processed (including recursive group children).
    pub total_commands: u32,
    /// Number of group commands that required a temporary pixmap.
    pub group_temp_count: u32,
    /// Time spent on group temp pixmap allocation/clear in microseconds.
    pub group_alloc_us: u64,
    /// Time spent on group compositing (draw_pixmap back to parent) in microseconds.
    pub group_composite_us: u64,
    /// Number of pixmaps in the pool after rasterization.
    pub pool_size: usize,
}

impl Default for RasterProfile {
    fn default() -> Self {
        Self {
            total_us: 0,
            background_us: 0,
            by_type: std::array::from_fn(|_| CommandTiming::default()),
            total_commands: 0,
            group_temp_count: 0,
            group_alloc_us: 0,
            group_composite_us: 0,
            pool_size: 0,
        }
    }
}

impl RasterProfile {
    /// Get timing for a specific command type.
    #[must_use]
    pub fn timing(&self, ct: CommandType) -> &CommandTiming {
        &self.by_type[ct as usize]
    }

    /// Iterate over command types that have non-zero count, sorted by total time (desc).
    pub fn active_types_sorted(&self) -> Vec<(CommandType, &CommandTiming)> {
        let mut active: Vec<_> = ALL_COMMAND_TYPES
            .iter()
            .copied()
            .filter_map(|ct| {
                let t = &self.by_type[ct as usize];
                if t.count > 0 { Some((ct, t)) } else { None }
            })
            .collect();
        active.sort_by(|a, b| b.1.total_us.cmp(&a.1.total_us));
        active
    }
}

/// Format microseconds compactly: `<1ms` → `800μ`, `≥1ms` → `1.2m`.
fn compact_time(us: u64) -> String {
    if us < 1000 {
        format!("{us}μ")
    } else {
        format!("{:.1}m", us as f64 / 1000.0)
    }
}

impl fmt::Display for RasterProfile {
    /// Ultra-compact one-line display for status bars.
    ///
    /// Example: `4.6m[grd6=1.8 grp17=1.0 cir39=0.5 lin23=0.3 rct62=0.2 bg=80μ]`
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[", compact_time(self.total_us))?;
        let sorted = self.active_types_sorted();
        for (i, (ct, timing)) in sorted.iter().enumerate() {
            if i > 0 { write!(f, " ")?; }
            write!(
                f,
                "{}{}={}",
                ct.short_name(),
                timing.count,
                compact_time(timing.total_us),
            )?;
        }
        if self.group_temp_count > 0 {
            write!(
                f,
                " a{}={}",
                self.group_temp_count,
                compact_time(self.group_alloc_us + self.group_composite_us),
            )?;
        }
        write!(f, "]")
    }
}

// ---------------------------------------------------------------------------
// Rolling profile history (median + P95)
// ---------------------------------------------------------------------------

use std::collections::VecDeque;

/// Default rolling window size for [`ProfileHistory`].
pub const DEFAULT_HISTORY_CAPACITY: usize = 64;

/// Per-command-type smoothed statistics.
#[derive(Clone, Debug)]
pub struct SmoothedTiming {
    /// Command type.
    pub cmd_type: CommandType,
    /// Median command count across frames.
    pub count: u32,
    /// Median total microseconds for this command type.
    pub median_us: u64,
    /// 95th-percentile total microseconds.
    pub p95_us: u64,
}

/// Smoothed profile snapshot computed from a rolling window.
#[derive(Clone, Debug)]
pub struct SmoothedProfile {
    /// Median total rasterization time in microseconds.
    pub total_median_us: u64,
    /// 95th-percentile total rasterization time.
    pub total_p95_us: u64,
    /// Per-command-type smoothed timings (only types with non-zero counts).
    pub by_type: Vec<SmoothedTiming>,
    /// Number of frames in the window used to compute these stats.
    pub frame_count: usize,
}

impl SmoothedProfile {
    /// Iterate over command types sorted by median time (descending).
    pub fn sorted_types(&self) -> Vec<&SmoothedTiming> {
        let mut sorted: Vec<&SmoothedTiming> = self.by_type.iter().collect();
        sorted.sort_by(|a, b| b.median_us.cmp(&a.median_us));
        sorted
    }
}

impl fmt::Display for SmoothedProfile {
    /// Ultra-compact one-line display for status bars.
    ///
    /// Top 3 command types show median/P95, remaining types show median only.
    ///
    /// Example: `~7.1m[grd6=3.5/3.8m grp17=1.3/1.5m cir39=540μ lin23=507μ rct62=376μ]`
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "~{}[", compact_time(self.total_median_us))?;
        let sorted = self.sorted_types();
        for (i, st) in sorted.iter().enumerate() {
            if i > 0 { write!(f, " ")?; }
            if i < 3 {
                // Top 3: show median/P95
                write!(
                    f,
                    "{}{}={}/{}",
                    st.cmd_type.short_name(),
                    st.count,
                    compact_time(st.median_us),
                    compact_time(st.p95_us),
                )?;
            } else {
                // Rest: median only
                write!(
                    f,
                    "{}{}={}",
                    st.cmd_type.short_name(),
                    st.count,
                    compact_time(st.median_us),
                )?;
            }
        }
        write!(f, "]")
    }
}

/// Rolling history of [`RasterProfile`] snapshots for computing stable
/// median and P95 statistics across frames.
///
/// This solves the problem of noisy single-frame profiler readings by
/// maintaining a sliding window and extracting robust percentiles.
///
/// # Example
///
/// ```ignore
/// let mut history = ProfileHistory::new(64);
/// // In your render loop:
/// let profile = ProfiledRasterizer::rasterize_into_profiled(&canvas, &mut pixmap);
/// history.push(profile);
/// let smoothed = history.summary();
/// println!("{smoothed}"); // ~7.1m[grd6=3.5m↑3.8 grp17=1.3m↑1.5 ...]
/// ```
#[derive(Clone, Debug)]
pub struct ProfileHistory {
    frames: VecDeque<RasterProfile>,
    capacity: usize,
}

impl Default for ProfileHistory {
    fn default() -> Self {
        Self::new(DEFAULT_HISTORY_CAPACITY)
    }
}

impl ProfileHistory {
    /// Create a new history with the given rolling window capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            frames: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Record a frame's rasterization profile.
    pub fn push(&mut self, profile: RasterProfile) {
        if self.frames.len() >= self.capacity {
            self.frames.pop_front();
        }
        self.frames.push_back(profile);
    }

    /// Number of frames currently in the history.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether the history is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Clear all stored frames.
    pub fn clear(&mut self) {
        self.frames.clear();
    }

    /// Median of `total_us` across stored frames.
    #[must_use]
    pub fn median_total_us(&self) -> u64 {
        percentile_of(&self.frames, |p| p.total_us, 50)
    }

    /// 95th percentile of `total_us` across stored frames.
    #[must_use]
    pub fn p95_total_us(&self) -> u64 {
        percentile_of(&self.frames, |p| p.total_us, 95)
    }

    /// Median `total_us` for a specific command type across stored frames.
    #[must_use]
    pub fn median_by_type(&self, ct: CommandType) -> u64 {
        let idx = ct as usize;
        percentile_of(&self.frames, |p| p.by_type[idx].total_us, 50)
    }

    /// 95th percentile `total_us` for a specific command type.
    #[must_use]
    pub fn p95_by_type(&self, ct: CommandType) -> u64 {
        let idx = ct as usize;
        percentile_of(&self.frames, |p| p.by_type[idx].total_us, 95)
    }

    /// Compute a full smoothed profile snapshot for display.
    #[must_use]
    pub fn summary(&self) -> SmoothedProfile {
        if self.frames.is_empty() {
            return SmoothedProfile {
                total_median_us: 0,
                total_p95_us: 0,
                by_type: Vec::new(),
                frame_count: 0,
            };
        }

        let mut by_type = Vec::new();
        for ct in ALL_COMMAND_TYPES {
            let idx = ct as usize;
            // Check if any frame has non-zero count for this type
            let has_data = self.frames.iter().any(|p| p.by_type[idx].count > 0);
            if has_data {
                #[allow(clippy::cast_possible_truncation)]
                let median_count = percentile_of(&self.frames, |p| u64::from(p.by_type[idx].count), 50) as u32;
                by_type.push(SmoothedTiming {
                    cmd_type: ct,
                    count: median_count,
                    median_us: self.median_by_type(ct),
                    p95_us: self.p95_by_type(ct),
                });
            }
        }

        SmoothedProfile {
            total_median_us: self.median_total_us(),
            total_p95_us: self.p95_total_us(),
            by_type,
            frame_count: self.frames.len(),
        }
    }
}

/// Extract a percentile value from a collection using a key extractor.
///
/// `pct` should be 0–100. Uses nearest-rank method.
fn percentile_of<F>(frames: &VecDeque<RasterProfile>, key: F, pct: usize) -> u64
where
    F: Fn(&RasterProfile) -> u64,
{
    if frames.is_empty() {
        return 0;
    }
    let mut values: Vec<u64> = frames.iter().map(&key).collect();
    values.sort_unstable();
    let n = values.len();
    // Ceiling-based nearest-rank: ceil(pct/100 * n) - 1
    let rank = ((pct * n + 99) / 100).saturating_sub(1).min(n - 1);
    values[rank]
}

// ---------------------------------------------------------------------------
// Transport profile
// ---------------------------------------------------------------------------

/// Profiling data from a single transport (flush) pass.
#[derive(Clone, Debug, Default)]
pub struct TransportProfile {
    /// Total transport time in microseconds.
    pub total_us: u64,
    /// Zlib compression time in microseconds (0 if not using zlib).
    pub compress_us: u64,
    /// Base64 encoding time in microseconds.
    pub encode_us: u64,
    /// Write + flush I/O time in microseconds.
    pub io_us: u64,
    /// Total wire bytes (compressed + base64-encoded payload).
    pub wire_bytes: usize,
    /// Raw pixel data size in bytes.
    pub raw_bytes: usize,
}

impl TransportProfile {
    /// Compression ratio (raw / wire). Higher = better compression.
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        if self.wire_bytes == 0 {
            0.0
        } else {
            self.raw_bytes as f64 / self.wire_bytes as f64
        }
    }
}

impl fmt::Display for TransportProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "flush {:.1}ms [zlib {:.1}ms  b64 {:.1}ms  io {:.1}ms  wire {}KB  ratio {:.1}×]",
            self.total_us as f64 / 1000.0,
            self.compress_us as f64 / 1000.0,
            self.encode_us as f64 / 1000.0,
            self.io_us as f64 / 1000.0,
            self.wire_bytes / 1024,
            self.compression_ratio(),
        )
    }
}

// ---------------------------------------------------------------------------
// Full pipeline profile
// ---------------------------------------------------------------------------

/// Complete pipeline profile combining scene build, rasterization, and transport.
#[derive(Clone, Debug, Default)]
pub struct PipelineProfile {
    /// Scene construction time in microseconds.
    pub scene_build_us: u64,
    /// Rasterization profile.
    pub raster: RasterProfile,
    /// Transport profile.
    pub transport: TransportProfile,
    /// Canvas dimensions.
    pub canvas_width: u32,
    /// Canvas dimensions.
    pub canvas_height: u32,
    /// Total pixel count.
    pub pixel_count: u64,
}

impl PipelineProfile {
    /// Total frame time in microseconds.
    #[must_use]
    pub fn total_us(&self) -> u64 {
        self.scene_build_us + self.raster.total_us + self.transport.total_us
    }

    /// Total frame time in milliseconds.
    #[must_use]
    pub fn total_ms(&self) -> f64 {
        self.total_us() as f64 / 1000.0
    }

    /// Compact one-line summary for status bar display.
    #[must_use]
    pub fn compact_summary(&self) -> String {
        format!(
            "build {:.1}ms │ {} │ {}",
            self.scene_build_us as f64 / 1000.0,
            self.raster,
            self.transport,
        )
    }
}

impl fmt::Display for PipelineProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Pipeline Profile ({:.1}ms total)", self.total_ms())?;
        writeln!(f, "  Canvas: {}×{} ({} pixels)", self.canvas_width, self.canvas_height, self.pixel_count)?;
        writeln!(f, "  Build:     {:.3}ms", self.scene_build_us as f64 / 1000.0)?;
        writeln!(f, "  Raster:    {:.3}ms ({} commands)", self.raster.total_us as f64 / 1000.0, self.raster.total_commands)?;
        for (ct, timing) in self.raster.active_types_sorted() {
            writeln!(
                f,
                "    {:<10} ×{:<3}  total {:.3}ms  avg {:.1}μs  max {:.1}μs",
                ct.name(),
                timing.count,
                timing.total_us as f64 / 1000.0,
                timing.avg_us() as f64,
                timing.max_us as f64,
            )?;
        }
        writeln!(f, "  Transport: {:.3}ms", self.transport.total_us as f64 / 1000.0)?;
        writeln!(f, "    Compress: {:.3}ms", self.transport.compress_us as f64 / 1000.0)?;
        writeln!(f, "    Encode:   {:.3}ms", self.transport.encode_us as f64 / 1000.0)?;
        writeln!(f, "    I/O:      {:.3}ms", self.transport.io_us as f64 / 1000.0)?;
        writeln!(f, "    Wire:     {} KB ({:.1}× compression)", self.transport.wire_bytes / 1024, self.transport.compression_ratio())?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Profiled rasterization
// ---------------------------------------------------------------------------

use tiny_skia::{
    FillRule, Transform as SkiaTransform,
};

use crate::scene::style::Color;

/// Profiled rasterization: wraps each `render_command` call with timing.
///
/// This is a separate implementation to ensure zero overhead in the normal
/// (non-profiled) path.
pub struct ProfiledRasterizer;

impl ProfiledRasterizer {
    /// Rasterize a canvas into an existing pixmap with per-command profiling.
    ///
    /// Returns a `RasterProfile` with timing breakdown by command type.
    ///
    /// # Panics
    ///
    /// Panics if the pixmap dimensions don't match the canvas dimensions.
    pub fn rasterize_into_profiled(
        canvas: &PixelCanvas,
        pixmap: &mut Pixmap,
    ) -> RasterProfile {
        let mut gc = crate::rasterize::skia::GradientCache::new();
        Self::rasterize_into_profiled_cached(canvas, pixmap, &mut gc)
    }

    /// Profiled rasterization with a persistent gradient cache.
    ///
    /// Same as [`rasterize_into_profiled`](Self::rasterize_into_profiled) but
    /// reuses gradient pixmaps across frames.
    #[allow(clippy::cast_precision_loss)]
    pub fn rasterize_into_profiled_cached(
        canvas: &PixelCanvas,
        pixmap: &mut Pixmap,
        grad_cache: &mut crate::rasterize::skia::GradientCache,
    ) -> RasterProfile {
        assert_eq!(
            (pixmap.width(), pixmap.height()),
            (canvas.width(), canvas.height()),
            "pixmap dimensions must match canvas dimensions"
        );

        let total_start = Instant::now();
        let mut profile = RasterProfile::default();

        // Background fill
        let bg_start = Instant::now();
        let bg = canvas.background_color();
        if bg == Color::TRANSPARENT {
            pixmap.data_mut().fill(0);
        } else if let Some(color) = bg.to_tiny_skia() {
            pixmap.fill(color);
        } else {
            pixmap.data_mut().fill(0);
        }
        profile.background_us = bg_start.elapsed().as_micros() as u64;

        // Pool of reusable pixmaps
        let mut pool: Vec<Pixmap> = Vec::new();

        // Render each command with timing
        for cmd in canvas.commands() {
            Self::render_command_profiled(
                pixmap,
                cmd,
                SkiaTransform::identity(),
                &mut pool,
                &mut profile,
                grad_cache,
            );
        }

        profile.total_us = total_start.elapsed().as_micros() as u64;
        profile.pool_size = pool.len();
        profile
    }

    /// Render a single command with profiling.
    #[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
    fn render_command_profiled(
        pixmap: &mut Pixmap,
        cmd: &DrawCommand,
        parent_transform: SkiaTransform,
        pool: &mut Vec<Pixmap>,
        profile: &mut RasterProfile,
        grad_cache: &mut crate::rasterize::skia::GradientCache,
    ) {
        let ct = CommandType::from_command(cmd);
        profile.total_commands += 1;

        let cmd_start = Instant::now();

        // Delegate to the real render logic — we re-implement dispatch here
        // to capture group-internal timing.
        match cmd {
            DrawCommand::Group {
                commands,
                transform,
                clip,
                opacity,
                blend_mode,
            } => {
                let combined = parent_transform.post_concat(transform.to_tiny_skia());
                let needs_temp = *opacity < 1.0
                    || clip.is_some()
                    || *blend_mode != crate::scene::style::BlendMode::SrcOver;

                if needs_temp {
                    profile.group_temp_count += 1;

                    // Measure allocation/clear
                    let alloc_start = Instant::now();

                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        clippy::cast_precision_loss,
                    )]
                    let (tw, th, origin_col, origin_row) = match clip {
                        Some(crate::scene::style::ClipRegion::Rect(r)) => {
                            let cw = (r.width.ceil() as u32).max(1).min(pixmap.width());
                            let ch = (r.height.ceil() as u32).max(1).min(pixmap.height());
                            (cw, ch, r.x.floor() as i32, r.y.floor() as i32)
                        }
                        _ => crate::rasterize::Rasterizer::estimate_group_bounds(
                            commands,
                            pixmap.width(),
                            pixmap.height(),
                        ),
                    };

                    let needed_area = (tw as usize) * (th as usize);
                    let mut temp = pool.pop().unwrap_or_else(|| {
                        Pixmap::new(tw, th).expect("temp pixmap for group")
                    });

                    // Right-size: discard if too small or >4× needed.
                    let pool_area = (temp.width() as usize) * (temp.height() as usize);
                    if temp.width() < tw || temp.height() < th || pool_area > needed_area * 4 {
                        temp = Pixmap::new(tw, th).expect("temp pixmap for group");
                    }

                    // Targeted clear.
                    if temp.width() == tw && temp.height() == th {
                        temp.data_mut().fill(0);
                    } else if temp.width() == tw {
                        let end = (tw as usize * th as usize * 4).min(temp.data().len());
                        temp.data_mut()[..end].fill(0);
                    } else {
                        let row_stride = temp.width() as usize * 4;
                        let row_clear = tw as usize * 4;
                        let data = temp.data_mut();
                        for row in 0..(th as usize) {
                            let start = row * row_stride;
                            let end = (start + row_clear).min(data.len());
                            if start >= data.len() { break; }
                            data[start..end].fill(0);
                        }
                    }

                    #[allow(clippy::cast_precision_loss)]
                    let child_transform = if origin_col != 0 || origin_row != 0 {
                        let offset = SkiaTransform::from_translate(
                            -(origin_col as f32),
                            -(origin_row as f32),
                        );
                        combined.post_concat(offset)
                    } else {
                        combined
                    };

                    profile.group_alloc_us +=
                        alloc_start.elapsed().as_micros() as u64;

                    // Render children (profiled recursively)
                    for child in commands {
                        Self::render_command_profiled(
                            &mut temp,
                            child,
                            child_transform,
                            pool,
                            profile,
                            grad_cache,
                        );
                    }

                    // Measure compositing
                    let composite_start = Instant::now();

                    let mask = clip.as_ref().and_then(|clip_region| match clip_region {
                        crate::scene::style::ClipRegion::Rect(_) => None,
                        crate::scene::style::ClipRegion::Path(path_data) => {
                            let mut mask =
                                tiny_skia::Mask::new(pixmap.width(), pixmap.height())?;
                            mask.fill_path(
                                path_data.path(),
                                FillRule::Winding,
                                true,
                                SkiaTransform::identity(),
                            );
                            Some(mask)
                        }
                    });

                    let paint = tiny_skia::PixmapPaint {
                        opacity: *opacity,
                        blend_mode: blend_mode.to_tiny_skia(),
                        quality: tiny_skia::FilterQuality::Nearest,
                    };
                    pixmap.draw_pixmap(
                        origin_col,
                        origin_row,
                        temp.as_ref(),
                        &paint,
                        SkiaTransform::identity(),
                        mask.as_ref(),
                    );

                    profile.group_composite_us +=
                        composite_start.elapsed().as_micros() as u64;

                    pool.push(temp);
                } else {
                    // Fast path: no compositing needed
                    for child in commands {
                        Self::render_command_profiled(
                            pixmap, child, combined, pool, profile, grad_cache,
                        );
                    }
                }
            }

            // All non-group commands: delegate to the standard rasterizer
            _ => {
                crate::rasterize::Rasterizer::render_command(
                    pixmap,
                    cmd,
                    parent_transform,
                    pool,
                    grad_cache,
                );
            }
        }

        let elapsed_us = cmd_start.elapsed().as_micros() as u64;
        profile.by_type[ct as usize].record(elapsed_us);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::style::Color;

    #[test]
    fn profile_empty_canvas() {
        let canvas = PixelCanvas::new(100, 100);
        let mut pixmap = Pixmap::new(100, 100).unwrap();
        let profile = ProfiledRasterizer::rasterize_into_profiled(&canvas, &mut pixmap);
        assert_eq!(profile.total_commands, 0);
    }

    #[test]
    fn profile_simple_scene() {
        let canvas = PixelCanvas::new(200, 200)
            .background(Color::BLACK)
            .circle(100.0, 100.0, 50.0)
            .fill(Color::RED)
            .done()
            .rect(10.0, 10.0, 80.0, 80.0)
            .fill(Color::BLUE)
            .done();

        let mut pixmap = Pixmap::new(200, 200).unwrap();
        let profile = ProfiledRasterizer::rasterize_into_profiled(&canvas, &mut pixmap);

        assert_eq!(profile.total_commands, 2);
        assert_eq!(profile.timing(CommandType::Circle).count, 1);
        assert_eq!(profile.timing(CommandType::Rectangle).count, 1);
        assert!(profile.total_us > 0);
    }

    #[test]
    fn profile_display() {
        let canvas = PixelCanvas::new(100, 100)
            .circle(50.0, 50.0, 25.0)
            .fill(Color::RED)
            .done();

        let mut pixmap = Pixmap::new(100, 100).unwrap();
        let profile = ProfiledRasterizer::rasterize_into_profiled(&canvas, &mut pixmap);

        let display = format!("{profile}");
        assert!(display.contains("["));
        assert!(display.contains("cir1="));
    }

    #[test]
    fn command_type_classification() {
        let cmd = DrawCommand::Circle {
            cx: 0.0,
            cy: 0.0,
            radius: 10.0,
            style: crate::scene::style::ShapeStyle::default(),
        };
        assert_eq!(CommandType::from_command(&cmd), CommandType::Circle);
    }

    // -- ProfileHistory tests --

    /// Build a synthetic RasterProfile with a given total_us and circle timing.
    fn synth_profile(total_us: u64, circle_us: u64) -> RasterProfile {
        let mut p = RasterProfile::default();
        p.total_us = total_us;
        p.by_type[CommandType::Circle as usize].record(circle_us);
        p
    }

    #[test]
    fn history_median_simple() {
        let mut h = ProfileHistory::new(10);
        // Push 5 values: total_us = 100, 200, 300, 400, 500
        for v in [100, 200, 300, 400, 500] {
            h.push(synth_profile(v, v / 2));
        }
        // Median of [100,200,300,400,500] at rank 50%: index 2 = 300
        assert_eq!(h.median_total_us(), 300);
        assert_eq!(h.median_by_type(CommandType::Circle), 150);
    }

    #[test]
    fn history_p95() {
        let mut h = ProfileHistory::new(64);
        // Push 17 "normal" frames at 1000us and 3 "spike" frames at 5000us
        // That's 15% spikes — P50 should be 1000, P95 should catch the 5000s.
        for _ in 0..17 {
            h.push(synth_profile(1000, 500));
        }
        for _ in 0..3 {
            h.push(synth_profile(5000, 4000));
        }
        // 20 frames total. Sorted: [1000×17, 5000×3]
        // Median: rank = ceil(50*20/100) - 1 = 10 - 1 = 9 → 1000 ✓
        assert_eq!(h.median_total_us(), 1000);
        // P95: rank = ceil(95*20/100) - 1 = 19 - 1 = 18 → 5000 ✓ (index 18 is in the spike zone: 17,18,19)
        assert_eq!(h.p95_total_us(), 5000);
    }

    #[test]
    fn history_capacity() {
        let mut h = ProfileHistory::new(3);
        h.push(synth_profile(100, 50));
        h.push(synth_profile(200, 100));
        h.push(synth_profile(300, 150));
        assert_eq!(h.len(), 3);
        // Push a 4th — oldest (100) should be evicted
        h.push(synth_profile(400, 200));
        assert_eq!(h.len(), 3);
        // Median of [200, 300, 400] at rank 1 = 300
        assert_eq!(h.median_total_us(), 300);
    }

    #[test]
    fn history_display() {
        let mut h = ProfileHistory::new(10);
        h.push(synth_profile(5000, 2000));
        h.push(synth_profile(6000, 3000));
        let smoothed = h.summary();
        let display = format!("{smoothed}");
        // Should start with ~ for smoothed indicator
        assert!(display.starts_with('~'), "got: {display}");
        // Top types show median/P95 with '/' separator
        assert!(display.contains('/'), "got: {display}");
        // Should contain circle type
        assert!(display.contains("cir"), "got: {display}");
    }
}
