// SPDX-License-Identifier: MIT OR Apache-2.0
//! Axis rendering — tick marks, gridlines, minor ticks, and labels.
//!
//! Supports all four axis positions (Top, Bottom, Left, Right), minor tick
//! subdivisions, configurable tick direction, adaptive tick density,
//! auto-skip for overlapping labels, and zero line rendering.

use std::sync::Arc;

use scry_engine::scene::PixelCanvas;
use scry_engine::style::{Color, DashPattern};

use crate::formatter::{AutoFormatter, TickFormatter};
use crate::scale::{LinearScale, Scale};

/// Axis position on the chart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum AxisSide {
    /// Bottom (X axis — default).
    Bottom,
    /// Top (secondary X axis).
    Top,
    /// Left (Y axis — default).
    Left,
    /// Right (secondary Y axis).
    Right,
}

/// Direction tick marks extend from the axis line.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum TickDirection {
    /// Ticks extend outward from the plot area.
    #[default]
    Out,
    /// Ticks extend inward into the plot area.
    In,
    /// Ticks extend both directions (centered on axis line).
    InOut,
}

/// Rotation angle for tick labels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum LabelRotation {
    /// Horizontal labels (default).
    #[default]
    Horizontal,
    /// Labels rotated 45° (diagonal).
    Diagonal,
    /// Labels rotated 90° (vertical).
    Vertical,
    /// Custom rotation angle in degrees (clamped to 0–90).
    Angle(f32),
}

impl LabelRotation {
    /// Rotation in degrees.
    #[must_use]
    pub fn degrees(self) -> f32 {
        match self {
            Self::Horizontal => 0.0,
            Self::Diagonal => 45.0,
            Self::Vertical => 90.0,
            Self::Angle(deg) => deg.clamp(0.0, 90.0),
        }
    }
}

/// Configuration for rendering an axis.
#[allow(clippy::struct_excessive_bools)]
#[non_exhaustive]
pub struct AxisConfig {
    /// Which side the axis is on.
    pub side: AxisSide,
    /// Whether the axis spine (line) is visible.
    pub visible: bool,
    /// Color of the axis line.
    pub axis_color: Color,
    /// Axis line width in pixels.
    pub axis_width: f32,

    // --- Major ticks ---
    /// Length of major tick marks in pixels.
    pub tick_length: f32,
    /// Width of major tick marks in pixels.
    pub tick_width: f32,
    /// Major tick color (defaults to axis color).
    pub tick_color: Color,
    /// Direction ticks extend from the axis line.
    pub tick_direction: TickDirection,
    /// Maximum number of major ticks (adaptive based on axis length if 0).
    pub max_ticks: usize,
    /// Fixed tick positions (overrides auto-generation).
    pub fixed_ticks: Option<Vec<f64>>,

    // --- Minor ticks ---
    /// Whether to show minor ticks between major ticks.
    pub minor_ticks: bool,
    /// Number of minor tick subdivisions between each pair of major ticks.
    pub minor_subdivisions: usize,
    /// Minor tick length in pixels.
    pub minor_tick_length: f32,
    /// Minor tick width in pixels.
    pub minor_tick_width: f32,
    /// Minor tick color.
    pub minor_tick_color: Color,
    /// Whether to show minor gridlines.
    pub minor_grid: bool,

    // --- Grid ---
    /// Whether to draw gridlines at major ticks.
    pub show_grid: bool,
    /// Color of gridlines.
    pub grid_color: Color,
    /// Grid line width in pixels.
    pub grid_width: f32,
    /// Dash pattern for grid lines (`None` = solid).
    pub grid_dash: Option<DashPattern>,

    // --- Label formatting ---
    /// Custom tick label formatter. If `None`, uses the default `AutoFormatter`.
    pub tick_formatter: Option<Arc<dyn TickFormatter>>,
    /// Pixel offset from tick to label.
    pub label_offset: f32,

    // --- Fixed step ---
    /// Fixed tick step size. When set, ticks are placed at multiples of
    /// this value within the axis domain (overrides adaptive generation,
    /// but `fixed_ticks` takes priority).
    pub tick_step: Option<f64>,

    // --- Zero line ---
    /// Color for the zero line (when domain spans zero). `None` = auto (semi-transparent axis color).
    pub zero_line_color: Option<Color>,
    /// Width of the zero line in pixels (default: 1.5).
    pub zero_line_width: f32,

    // --- Label rotation ---
    /// Rotation for tick labels (horizontal, diagonal, or vertical).
    pub tick_label_rotation: LabelRotation,
}

impl Clone for AxisConfig {
    fn clone(&self) -> Self {
        Self {
            side: self.side,
            visible: self.visible,
            axis_color: self.axis_color,
            axis_width: self.axis_width,
            tick_length: self.tick_length,
            tick_width: self.tick_width,
            tick_color: self.tick_color,
            tick_direction: self.tick_direction,
            max_ticks: self.max_ticks,
            fixed_ticks: self.fixed_ticks.clone(),
            minor_ticks: self.minor_ticks,
            minor_subdivisions: self.minor_subdivisions,
            minor_tick_length: self.minor_tick_length,
            minor_tick_width: self.minor_tick_width,
            minor_tick_color: self.minor_tick_color,
            minor_grid: self.minor_grid,
            show_grid: self.show_grid,
            grid_color: self.grid_color,
            grid_width: self.grid_width,
            grid_dash: self.grid_dash.clone(),
            tick_formatter: self.tick_formatter.clone(),
            label_offset: self.label_offset,
            tick_step: self.tick_step,
            zero_line_color: self.zero_line_color,
            zero_line_width: self.zero_line_width,
            tick_label_rotation: self.tick_label_rotation,
        }
    }
}

impl std::fmt::Debug for AxisConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AxisConfig")
            .field("side", &self.side)
            .field("visible", &self.visible)
            .field("axis_color", &self.axis_color)
            .field("show_grid", &self.show_grid)
            .field("minor_ticks", &self.minor_ticks)
            .field(
                "tick_formatter",
                &self.tick_formatter.as_ref().map(|_| ".."),
            )
            .field("tick_label_rotation", &self.tick_label_rotation)
            .finish()
    }
}

impl Default for AxisConfig {
    fn default() -> Self {
        let axis_color = Color::from_rgba8(160, 160, 180, 255);
        Self {
            side: AxisSide::Bottom,
            visible: true,
            axis_color,
            axis_width: 1.5,
            tick_length: 5.0,
            tick_width: 1.0,
            tick_color: axis_color,
            tick_direction: TickDirection::Out,
            max_ticks: 0, // 0 = adaptive
            fixed_ticks: None,
            minor_ticks: false,
            minor_subdivisions: 4,
            minor_tick_length: 3.0,
            minor_tick_color: Color::from_rgba8(120, 120, 140, 180),
            minor_tick_width: 0.7,
            minor_grid: false,
            show_grid: true,
            grid_color: Color::from_rgba8(50, 50, 65, 120),
            grid_width: 0.5,
            grid_dash: None,
            tick_formatter: None,
            label_offset: 8.0,
            tick_step: None,
            zero_line_color: None,
            zero_line_width: 1.5,
            tick_label_rotation: LabelRotation::Horizontal,
        }
    }
}

/// Compute adaptive tick count based on available axis length in pixels.
///
/// Uses the heuristic: ~1 tick per 60px for horizontal, ~1 per 50px for vertical.
#[must_use]
pub fn adaptive_tick_count(axis_length_px: f32, is_horizontal: bool) -> usize {
    adaptive_tick_count_rotated(axis_length_px, is_horizontal, LabelRotation::Horizontal)
}

/// Rotation-aware variant: rotated labels take less horizontal space,
/// so more ticks can fit.
#[must_use]
pub fn adaptive_tick_count_rotated(
    axis_length_px: f32,
    is_horizontal: bool,
    rotation: LabelRotation,
) -> usize {
    let px_per_tick = if is_horizontal {
        match rotation {
            LabelRotation::Horizontal => 60.0,
            LabelRotation::Diagonal => 45.0,
            LabelRotation::Vertical => 30.0,
            LabelRotation::Angle(deg) => {
                // Interpolate between 60 (horizontal) and 30 (vertical)
                let t = deg.clamp(0.0, 90.0) / 90.0;
                60.0 - t * 30.0
            }
        }
    } else {
        50.0
    };
    let count = (axis_length_px / px_per_tick).round() as usize;
    count.clamp(3, 12)
}

/// Generate tick values at fixed intervals within a domain.
///
/// Ticks are placed at `ceil(domain.0 / step) * step`, then incremented by `step`
/// until reaching `domain.1`. A hard cap of 200 ticks prevents infinite loops
/// from degenerate step values.
///
/// Returns a two-element fallback `[domain.0, domain.1]` if `step` is zero,
/// negative, or non-finite.
fn generate_step_ticks(domain: (f64, f64), step: f64) -> Vec<f64> {
    if step <= 0.0 || !step.is_finite() {
        return vec![domain.0, domain.1];
    }

    let start = (domain.0 / step).ceil() * step;
    let mut ticks = Vec::new();
    // Use indexed computation (`start + i * step`) instead of cumulative
    // addition to avoid accumulated floating-point drift.
    let mut i = 0usize;
    loop {
        let val = start + i as f64 * step;
        if val > domain.1 + step * 0.01 || i >= 200 {
            break;
        }
        ticks.push(val);
        i += 1;
    }
    if ticks.is_empty() {
        ticks.push(domain.0);
    }
    ticks
}
/// Draw an axis (line + ticks + gridlines + minor ticks) onto a canvas.
///
/// - `plot_area`: The pixel rectangle of the plot area `(x, y, w, h)`.
/// - `scale`: The linear scale mapping data to pixels.
/// - `config`: Axis styling.
///
/// Returns tick positions and their formatted labels for text rendering.
#[must_use]
pub fn draw_axis(
    mut canvas: PixelCanvas,
    plot_area: (f32, f32, f32, f32),
    scale: &LinearScale,
    config: &AxisConfig,
) -> (PixelCanvas, Vec<(f32, String)>) {
    let (px, py, pw, ph) = plot_area;
    let mut tick_labels = Vec::new();

    // Determine tick count
    let is_horizontal = matches!(config.side, AxisSide::Bottom | AxisSide::Top);
    let axis_length = if is_horizontal { pw } else { ph };
    let n_ticks = if config.max_ticks > 0 {
        config.max_ticks
    } else {
        adaptive_tick_count_rotated(axis_length, is_horizontal, config.tick_label_rotation)
    };

    // Get tick values: fixed_ticks > tick_step > auto
    #[allow(clippy::option_if_let_else)]
    let ticks = if let Some(ref fixed) = config.fixed_ticks {
        fixed.clone()
    } else if let Some(step) = config.tick_step {
        generate_step_ticks(scale.domain(), step)
    } else {
        scale.ticks(n_ticks)
    };

    // Batch-format all tick labels for uniform precision
    let domain = scale.domain();
    let formatter: &dyn TickFormatter = config.tick_formatter.as_deref().unwrap_or(&AutoFormatter);
    let labels = formatter.format_batch(&ticks, domain);

    // Pair ticks with labels and apply auto-skip if needed
    let tick_label_pairs: Vec<(f64, String)> =
        ticks.iter().zip(labels).map(|(&v, l)| (v, l)).collect();

    let tick_label_pairs = auto_skip_labels(
        tick_label_pairs,
        axis_length,
        is_horizontal,
        config.tick_label_rotation,
        scale,
    );

    // Draw zero line if domain spans zero
    canvas = draw_zero_line(canvas, plot_area, scale, config);

    match config.side {
        AxisSide::Bottom => {
            // Axis spine
            if config.visible {
                canvas = canvas
                    .line(px, py + ph, px + pw, py + ph)
                    .color(config.axis_color)
                    .width(config.axis_width)
                    .done();
            }

            for (t, label) in &tick_label_pairs {
                let x = scale.to_pixel(*t) as f32;
                if x < px || x > px + pw {
                    continue;
                }

                // Tick mark
                let (t_start, t_end) =
                    tick_extents(py + ph, config.tick_length, config.tick_direction, false);
                canvas = canvas
                    .line(x, t_start, x, t_end)
                    .color(config.tick_color)
                    .width(config.tick_width)
                    .done();

                // Major gridline
                if config.show_grid {
                    canvas = draw_gridline(canvas, x, py, x, py + ph, config);
                }

                tick_labels.push((x, label.clone()));
            }

            // Minor ticks (use original tick set, not skipped)
            let orig_ticks: Vec<f64> = tick_label_pairs.iter().map(|(v, _)| *v).collect();
            if config.minor_ticks {
                canvas = draw_minor_ticks_h(canvas, &orig_ticks, scale, config, plot_area);
            }
        }

        AxisSide::Top => {
            if config.visible {
                canvas = canvas
                    .line(px, py, px + pw, py)
                    .color(config.axis_color)
                    .width(config.axis_width)
                    .done();
            }

            for (t, label) in &tick_label_pairs {
                let x = scale.to_pixel(*t) as f32;
                if x < px || x > px + pw {
                    continue;
                }

                let (t_start, t_end) =
                    tick_extents(py, config.tick_length, config.tick_direction, true);
                canvas = canvas
                    .line(x, t_start, x, t_end)
                    .color(config.tick_color)
                    .width(config.tick_width)
                    .done();

                if config.show_grid {
                    canvas = draw_gridline(canvas, x, py, x, py + ph, config);
                }

                tick_labels.push((x, label.clone()));
            }

            let orig_ticks: Vec<f64> = tick_label_pairs.iter().map(|(v, _)| *v).collect();
            if config.minor_ticks {
                canvas = draw_minor_ticks_h(canvas, &orig_ticks, scale, config, plot_area);
            }
        }

        AxisSide::Left => {
            if config.visible {
                canvas = canvas
                    .line(px, py, px, py + ph)
                    .color(config.axis_color)
                    .width(config.axis_width)
                    .done();
            }

            for (t, label) in &tick_label_pairs {
                let y = scale.to_pixel(*t) as f32;
                if y < py || y > py + ph {
                    continue;
                }

                let (t_start, t_end) =
                    tick_extents(px, config.tick_length, config.tick_direction, true);
                canvas = canvas
                    .line(t_start, y, t_end, y)
                    .color(config.tick_color)
                    .width(config.tick_width)
                    .done();

                if config.show_grid {
                    canvas = draw_gridline(canvas, px, y, px + pw, y, config);
                }

                tick_labels.push((y, label.clone()));
            }

            let orig_ticks: Vec<f64> = tick_label_pairs.iter().map(|(v, _)| *v).collect();
            if config.minor_ticks {
                canvas = draw_minor_ticks_v(canvas, &orig_ticks, scale, config, plot_area);
            }
        }

        AxisSide::Right => {
            if config.visible {
                canvas = canvas
                    .line(px + pw, py, px + pw, py + ph)
                    .color(config.axis_color)
                    .width(config.axis_width)
                    .done();
            }

            for (t, label) in &tick_label_pairs {
                let y = scale.to_pixel(*t) as f32;
                if y < py || y > py + ph {
                    continue;
                }

                let (t_start, t_end) =
                    tick_extents(px + pw, config.tick_length, config.tick_direction, false);
                canvas = canvas
                    .line(t_start, y, t_end, y)
                    .color(config.tick_color)
                    .width(config.tick_width)
                    .done();

                if config.show_grid {
                    canvas = draw_gridline(canvas, px, y, px + pw, y, config);
                }

                tick_labels.push((y, label.clone()));
            }

            let orig_ticks: Vec<f64> = tick_label_pairs.iter().map(|(v, _)| *v).collect();
            if config.minor_ticks {
                canvas = draw_minor_ticks_v(canvas, &orig_ticks, scale, config, plot_area);
            }
        }
    }

    (canvas, tick_labels)
}

// ---------------------------------------------------------------------------
// Auto-skip: remove labels when they'd overlap
// ---------------------------------------------------------------------------

use crate::layout::INTER_ADVANCE_RATIO;

/// Average character width at the reference 11px font size (Inter Regular).
///
/// Derived from [`INTER_ADVANCE_RATIO`] × 11.0 so there is a single
/// source of truth for the Inter advance-width metric.
const AVG_CHAR_WIDTH: f32 = INTER_ADVANCE_RATIO * 11.0;
/// Minimum gap between adjacent labels in pixels.
const MIN_LABEL_GAP: f32 = 6.0;

/// Drop every other tick label if adjacent labels would overlap.
///
/// For horizontal axes, this compares pixel positions along X.
/// For vertical axes, it uses a fixed line height spacing.
fn auto_skip_labels(
    pairs: Vec<(f64, String)>,
    axis_length: f32,
    is_horizontal: bool,
    rotation: LabelRotation,
    scale: &LinearScale,
) -> Vec<(f64, String)> {
    if pairs.len() <= 2 {
        return pairs;
    }

    let label_span = if is_horizontal {
        // Estimate total label width for horizontal axes.
        // Rotated labels take less horizontal space.
        let max_label_chars = pairs.iter().map(|(_, l)| l.len()).max().unwrap_or(1);
        let raw_width = max_label_chars as f32 * AVG_CHAR_WIDTH;
        let effective_width = match rotation {
            LabelRotation::Horizontal => raw_width,
            LabelRotation::Diagonal => raw_width * 0.71,
            LabelRotation::Vertical => 11.0, // ~font height
            LabelRotation::Angle(deg) => {
                let rad = deg.clamp(0.0, 90.0).to_radians();
                (raw_width * rad.cos()).max(11.0)
            }
        };
        effective_width + MIN_LABEL_GAP
    } else {
        // Y-axis: spacing is vertical. Each label needs ~15px line height + gap.
        // Inter Regular at 11px has ~14.4px measured height; we add generous padding.
        15.0 + MIN_LABEL_GAP + 1.0
    };

    // Use actual pixel positions from the scale to determine minimum spacing.
    // This handles non-uniform tick spacing (e.g., endpoint insertion by nice_ticks)
    // that the old `axis_length / count` average couldn't catch.
    let pixel_positions: Vec<f32> = pairs.iter().map(|(v, _)| scale.to_pixel(*v) as f32).collect();
    let min_spacing = pixel_positions
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(f32::INFINITY, f32::min);

    // If the tightest pair of adjacent labels has enough room, keep all
    if min_spacing >= label_span {
        return pairs;
    }

    // Calculate skip factor based on the worst-case spacing
    let avg_spacing = axis_length / pairs.len().max(1) as f32;
    let skip = (label_span / avg_spacing).ceil().max(label_span / min_spacing.max(0.1)) as usize;
    let skip = skip.max(2); // must skip at least every other

    // Always preserve first and last labels so readers can see the domain extent.
    // However, only preserve the last if it won't overlap with the previous kept label.
    let total = pairs.len();
    let last_kept_before_end = if total > skip {
        // The last regularly-kept index before the final element.
        ((total - 1) / skip) * skip
    } else {
        0
    };

    // Check if the last label would overlap the previous kept label using
    // actual pixel distances.
    let keep_last = if total > 1 && last_kept_before_end < total - 1 {
        let last_px = pixel_positions[total - 1];
        let prev_px = pixel_positions[last_kept_before_end];
        (last_px - prev_px).abs() >= label_span
    } else {
        false
    };

    pairs
        .into_iter()
        .enumerate()
        .filter(|(i, _)| i % skip == 0 || *i == 0 || (*i == total - 1 && keep_last))
        .map(|(_, p)| p)
        .collect()
}

// ---------------------------------------------------------------------------
// Zero line
// ---------------------------------------------------------------------------

/// Draw a prominent line at y=0 (or x=0) when the domain spans zero.
fn draw_zero_line(
    canvas: PixelCanvas,
    plot_area: (f32, f32, f32, f32),
    scale: &LinearScale,
    config: &AxisConfig,
) -> PixelCanvas {
    let (dmin, dmax) = scale.domain();

    // Only draw if domain actually spans zero (or includes it as an endpoint)
    if dmin > 0.0 || dmax < 0.0 {
        return canvas;
    }
    // Don't draw if domain is degenerate (both endpoints effectively zero)
    if dmin.abs() < f64::EPSILON * 100.0 && dmax.abs() < f64::EPSILON * 100.0 {
        return canvas;
    }

    let (px, py, pw, ph) = plot_area;
    let zero_px = scale.to_pixel(0.0) as f32;

    let color = config
        .zero_line_color
        .unwrap_or_else(|| config.axis_color.with_alpha(0.5));
    let width = config.zero_line_width;

    match config.side {
        AxisSide::Bottom | AxisSide::Top => {
            // Vertical zero line at x=0
            if zero_px >= px && zero_px <= px + pw {
                canvas
                    .line(zero_px, py, zero_px, py + ph)
                    .color(color)
                    .width(width)
                    .done()
            } else {
                canvas
            }
        }
        AxisSide::Left | AxisSide::Right => {
            // Horizontal zero line at y=0
            if zero_px >= py && zero_px <= py + ph {
                canvas
                    .line(px, zero_px, px + pw, zero_px)
                    .color(color)
                    .width(width)
                    .done()
            } else {
                canvas
            }
        }
    }
}

/// Compute tick mark start/end positions based on direction.
///
/// `axis_pos` is the position of the axis line.
/// `inward` = true means "toward plot center" is the negative direction.
fn tick_extents(axis_pos: f32, length: f32, direction: TickDirection, inward: bool) -> (f32, f32) {
    match direction {
        TickDirection::Out => {
            if inward {
                (axis_pos - length, axis_pos)
            } else {
                (axis_pos, axis_pos + length)
            }
        }
        TickDirection::In => {
            if inward {
                (axis_pos, axis_pos + length)
            } else {
                (axis_pos - length, axis_pos)
            }
        }
        TickDirection::InOut => {
            let half = length / 2.0;
            (axis_pos - half, axis_pos + half)
        }
    }
}

/// Draw a single gridline with optional dash pattern.
fn draw_gridline(
    canvas: PixelCanvas,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    config: &AxisConfig,
) -> PixelCanvas {
    let mut line = canvas
        .line(x1, y1, x2, y2)
        .color(config.grid_color)
        .width(config.grid_width);
    if let Some(ref dash) = config.grid_dash {
        line = line.dash(dash.clone());
    }
    line.done()
}

/// Draw minor tick marks between major ticks (horizontal axis).
fn draw_minor_ticks_h(
    mut canvas: PixelCanvas,
    major_ticks: &[f64],
    scale: &LinearScale,
    config: &AxisConfig,
    plot_area: (f32, f32, f32, f32),
) -> PixelCanvas {
    let (px, py, _pw, ph) = plot_area;
    let axis_y = match config.side {
        AxisSide::Bottom => py + ph,
        AxisSide::Top => py,
        _ => return canvas,
    };
    let inward = matches!(config.side, AxisSide::Top);

    for w in major_ticks.windows(2) {
        let step = (w[1] - w[0]) / config.minor_subdivisions as f64;
        for i in 1..config.minor_subdivisions {
            let val = w[0] + step * i as f64;
            let x = scale.to_pixel(val) as f32;
            if x < px {
                continue;
            }

            let (t_start, t_end) = tick_extents(
                axis_y,
                config.minor_tick_length,
                config.tick_direction,
                inward,
            );
            canvas = canvas
                .line(x, t_start, x, t_end)
                .color(config.minor_tick_color)
                .width(config.minor_tick_width)
                .done();

            if config.minor_grid {
                let mut grid = canvas
                    .line(x, py, x, py + ph)
                    .color(config.grid_color.with_alpha(0.3))
                    .width(config.grid_width * 0.5);
                if let Some(ref dash) = config.grid_dash {
                    grid = grid.dash(dash.clone());
                }
                canvas = grid.done();
            }
        }
    }
    canvas
}

/// Draw minor tick marks between major ticks (vertical axis).
fn draw_minor_ticks_v(
    mut canvas: PixelCanvas,
    major_ticks: &[f64],
    scale: &LinearScale,
    config: &AxisConfig,
    plot_area: (f32, f32, f32, f32),
) -> PixelCanvas {
    let (px, py, pw, _ph) = plot_area;
    let axis_x = match config.side {
        AxisSide::Left => px,
        AxisSide::Right => px + pw,
        _ => return canvas,
    };
    let inward = matches!(config.side, AxisSide::Left);

    for w in major_ticks.windows(2) {
        let step = (w[1] - w[0]) / config.minor_subdivisions as f64;
        for i in 1..config.minor_subdivisions {
            let val = w[0] + step * i as f64;
            let y = scale.to_pixel(val) as f32;
            if y < py {
                continue;
            }

            let (t_start, t_end) = tick_extents(
                axis_x,
                config.minor_tick_length,
                config.tick_direction,
                inward,
            );
            canvas = canvas
                .line(t_start, y, t_end, y)
                .color(config.minor_tick_color)
                .width(config.minor_tick_width)
                .done();

            if config.minor_grid {
                let mut grid = canvas
                    .line(px, y, px + pw, y)
                    .color(config.grid_color.with_alpha(0.3))
                    .width(config.grid_width * 0.5);
                if let Some(ref dash) = config.grid_dash {
                    grid = grid.dash(dash.clone());
                }
                canvas = grid.done();
            }
        }
    }
    canvas
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_tick_count_reasonable() {
        assert!(adaptive_tick_count(300.0, true) >= 3);
        assert!(adaptive_tick_count(300.0, true) <= 12);
        assert!(adaptive_tick_count(60.0, true) >= 3);
        assert!(adaptive_tick_count(1200.0, true) <= 12);
    }

    #[test]
    fn tick_extents_out_bottom() {
        let (s, e) = tick_extents(100.0, 5.0, TickDirection::Out, false);
        assert_eq!(s, 100.0);
        assert_eq!(e, 105.0);
    }

    #[test]
    fn tick_extents_in_left() {
        let (s, e) = tick_extents(50.0, 5.0, TickDirection::In, true);
        assert_eq!(s, 50.0);
        assert_eq!(e, 55.0);
    }

    #[test]
    fn tick_extents_inout() {
        let (s, e) = tick_extents(100.0, 10.0, TickDirection::InOut, false);
        assert_eq!(s, 95.0);
        assert_eq!(e, 105.0);
    }

    #[test]
    fn auto_skip_preserves_endpoints() {
        // Create labels that would overlap: 10 labels in 200px with ~7px/char
        let pairs: Vec<(f64, String)> = (0..10).map(|i| (i as f64, format!("label_{i}"))).collect();
        // Scale maps domain 0..9 to pixel range 0..200 (evenly spaced)
        let scale = crate::scale::LinearScale::new((0.0, 9.0), (0.0, 200.0));
        let result = auto_skip_labels(pairs, 200.0, true, LabelRotation::Horizontal, &scale);
        // First label must always be preserved
        assert_eq!(result.first().unwrap().1, "label_0");
    }

    #[test]
    fn step_ticks_no_float_drift() {
        // step=0.1 over domain (0, 10): 101 ticks, last should be very close to 10.0
        let ticks = generate_step_ticks((0.0, 10.0), 0.1);
        assert!(!ticks.is_empty());
        let last = *ticks.last().unwrap();
        assert!(
            (last - 10.0).abs() < 1e-10,
            "last tick {last} drifted from 10.0"
        );
    }

    #[test]
    fn auto_skip_y_axis_uses_vertical_spacing() {
        // Y-axis with many labels in a short axis: should skip some
        let pairs: Vec<(f64, String)> = (0..20)
            .map(|i| (i as f64, format!("{}", i * 100)))
            .collect();
        // Y-axis scale: domain 0..19, range 200..0 (inverted for screen coords)
        let scale = crate::scale::LinearScale::new((0.0, 19.0), (200.0, 0.0));
        let result = auto_skip_labels(pairs, 200.0, false, LabelRotation::Horizontal, &scale);
        // With ~22px per label and 200px total, should skip some labels
        assert!(
            result.len() < 20,
            "Y-axis auto-skip should skip some labels"
        );
        assert_eq!(result.first().unwrap().1, "0");
    }

    #[test]
    fn label_rotation_angle_degrees() {
        assert_eq!(LabelRotation::Angle(30.0).degrees(), 30.0);
        assert_eq!(LabelRotation::Angle(60.0).degrees(), 60.0);
        // Clamps to 0-90 range
        assert_eq!(LabelRotation::Angle(-10.0).degrees(), 0.0);
        assert_eq!(LabelRotation::Angle(120.0).degrees(), 90.0);
    }

    #[test]
    fn auto_skip_with_angle_rotation() {
        // Custom angle should affect skip behavior via effective width
        let pairs: Vec<(f64, String)> = (0..10)
            .map(|i| (i as f64, format!("long_label_{i}")))
            .collect();
        let scale = crate::scale::LinearScale::new((0.0, 9.0), (0.0, 300.0));
        let result_horiz = auto_skip_labels(pairs.clone(), 300.0, true, LabelRotation::Horizontal, &scale);
        let result_angle = auto_skip_labels(pairs, 300.0, true, LabelRotation::Angle(60.0), &scale);
        // Angled labels take less horizontal space, so should skip fewer (or equal)
        assert!(
            result_angle.len() >= result_horiz.len(),
            "60° labels should fit as well or better than horizontal: {} vs {}",
            result_angle.len(),
            result_horiz.len()
        );
    }

    #[test]
    fn adaptive_tick_count_rotated_with_angle() {
        let count = adaptive_tick_count_rotated(300.0, true, LabelRotation::Angle(45.0));
        assert!(count >= 3);
        assert!(count <= 12);
    }
}
