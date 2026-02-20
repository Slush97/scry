// SPDX-License-Identifier: MIT OR Apache-2.0
//! Legend rendering utilities.
//!
//! Draws color swatches with labels to identify data series.
//! Supports auto-sizing, 9 placement options, and multiple swatch shapes.
//! Labels are font-metric-aware, auto-truncated, and proportionally padded.

use scry_engine::scene::PixelCanvas;
use scry_engine::style::Color;

use crate::layout::INTER_ADVANCE_RATIO;
use crate::text_utils;

// ---------------------------------------------------------------------------
// Position & Shape
// ---------------------------------------------------------------------------

/// Legend placement within (or relative to) the plot area.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum LegendPosition {
    /// Top-right corner (default).
    #[default]
    TopRight,
    /// Top-left corner.
    TopLeft,
    /// Bottom-right corner.
    BottomRight,
    /// Bottom-left corner.
    BottomLeft,
    /// Centered along the top edge.
    Top,
    /// Centered along the bottom edge.
    Bottom,
    /// Centered along the left edge.
    Left,
    /// Centered along the right edge.
    Right,
    /// Auto-detect least-overlap corner.
    Best,
    /// Outside the plot area, to the right.
    OutsideRight,
    /// Outside the plot area, below.
    OutsideBottom,
}

/// Shape of the color swatch in legend entries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum SwatchShape {
    /// Filled rounded rectangle (default — good for bar charts).
    #[default]
    Rect,
    /// Filled circle (good for scatter plots).
    Circle,
    /// Short line segment (good for line charts).
    Line,
}

/// Legend entry arrangement direction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum LegendOrientation {
    /// Entries are stacked vertically (default).
    #[default]
    Vertical,
    /// Entries are laid out horizontally in a single row.
    Horizontal,
}
// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for legend rendering.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct LegendConfig {
    /// Where to place the legend.
    pub position: LegendPosition,
    /// Optional background color behind the legend.
    pub background: Option<Color>,
    /// Optional border color around the legend.
    pub border: Option<Color>,
    /// Label text color.
    pub font_color: Color,
    /// Swatch shape for all entries.
    pub swatch_shape: SwatchShape,
    /// Color swatch size in pixels (default: 12.0).
    pub swatch_size: f32,
    /// Inner padding (px).
    pub padding: f32,
    /// Vertical spacing between entries (px).
    pub spacing: f32,
    /// Approximate character width for label measurement (px).
    pub char_width: f32,
    /// Whether to show the legend at all.
    pub visible: bool,
    /// Optional title displayed above the legend entries.
    pub title: Option<String>,
    /// Layout orientation for entries.
    pub orientation: LegendOrientation,
    /// Number of columns for vertical layout (default: 1).
    /// Only applies when `orientation` is `Vertical`.
    /// Set to 2+ for multi-column legends with many entries.
    pub columns: usize,
}

impl Default for LegendConfig {
    fn default() -> Self {
        Self {
            position: LegendPosition::TopRight,
            background: Some(Color::from_rgba8(30, 30, 40, 240)),
            border: Some(Color::from_rgba8(80, 80, 100, 200)),
            font_color: Color::from_rgba8(200, 200, 220, 255),
            swatch_shape: SwatchShape::Rect,
            swatch_size: 12.0,
            padding: 12.0,
            spacing: 8.0,
            char_width: 7.0,
            visible: true,
            title: None,
            orientation: LegendOrientation::Vertical,
            columns: 1,
        }
    }
}

impl LegendConfig {
    /// Apply theme-derived legend settings (background, border, font size)
    /// and font-metric-aware sizing from the given font_size.
    ///
    /// All dimension fields (swatch_size, padding, spacing, char_width) are
    /// computed relative to the actual rendered font size so the legend
    /// scales correctly at any canvas resolution — from 200px to 4K.
    pub fn apply_theme_and_font_size(&mut self, theme: &crate::theme::LegendTheme, font_size: f32) {
        self.background = Some(theme.background);
        self.border = theme.border;
        self.char_width = font_size * INTER_ADVANCE_RATIO;
        self.padding = (font_size * 0.6).max(4.0);
        self.swatch_size = (font_size * 0.85).max(8.0);
        self.spacing = (font_size * 0.55).max(4.0);
    }
}

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------

/// A single legend entry.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct LegendEntry {
    /// Display label.
    pub label: String,
    /// Color swatch.
    pub color: Color,
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

/// Font-aware measurement of a single legend label width.
fn measure_label_width(label: &str, char_width: f32) -> f32 {
    label.chars().count() as f32 * char_width
}

/// Measure auto-sized legend dimensions from entries.
///
/// Returns `(width, height)` in pixels. Accounts for orientation,
/// title, swatch size, and padding.
#[must_use]
pub fn measure_legend(
    entries: &[LegendEntry],
    swatch_size: f32,
    config: &LegendConfig,
) -> (f32, f32) {
    if entries.is_empty() {
        return (0.0, 0.0);
    }

    let title_height = if config.title.is_some() {
        swatch_size + config.spacing
    } else {
        0.0
    };

    match config.orientation {
        LegendOrientation::Vertical => {
            let cols = config.columns.max(1);
            let max_label_len = entries
                .iter()
                .map(|e| e.label.chars().count())
                .max()
                .unwrap_or(0);

            let label_width = max_label_len as f32 * config.char_width;
            let swatch_gap = 6.0;
            let col_width = swatch_size + swatch_gap + label_width;
            let col_gap = config.spacing * 2.0;

            // Title may be wider than entries
            let title_width = config
                .title
                .as_ref()
                .map_or(0.0, |t| measure_label_width(t, config.char_width));

            let content_width = col_width * cols as f32 + col_gap * (cols as f32 - 1.0).max(0.0);
            let width = config.padding * 2.0 + content_width.max(title_width);

            let rows = entries.len().div_ceil(cols);
            let entries_height = rows as f32 * (swatch_size + config.spacing) - config.spacing;
            let height = config.padding * 2.0 + title_height + entries_height;

            (width.max(40.0), height.max(20.0))
        }
        LegendOrientation::Horizontal => {
            let swatch_gap = 6.0;
            let entry_gap = config.spacing * 2.0; // gap between entries
            let total_width: f32 = entries
                .iter()
                .map(|e| {
                    swatch_size + swatch_gap + measure_label_width(&e.label, config.char_width)
                })
                .sum::<f32>()
                + entry_gap * (entries.len().saturating_sub(1)) as f32;

            let title_width = config
                .title
                .as_ref()
                .map_or(0.0, |t| measure_label_width(t, config.char_width));

            let width = config.padding * 2.0 + total_width.max(title_width);
            let height = config.padding * 2.0 + title_height + swatch_size;

            (width.max(40.0), height.max(20.0))
        }
    }
}

/// Estimate the right-side margin needed for an `OutsideRight` legend.
///
/// Pre-measures entry count × char_width to predict legend width before
/// `compute_plot_area` finalises the plot rectangle. Returns 0 if the
/// legend is not configured for outside-right placement or is hidden.
#[must_use]
pub fn estimate_legend_right_margin(
    config: &LegendConfig,
    entry_count: usize,
    max_label_chars: usize,
) -> f32 {
    if !config.visible || entry_count == 0 {
        return 0.0;
    }
    if config.position != LegendPosition::OutsideRight {
        return 0.0;
    }
    let swatch_gap = 6.0;
    let label_width = max_label_chars as f32 * config.char_width;
    let col_width = config.swatch_size + swatch_gap + label_width;
    // legend_width + padding on both sides + gap from plot area
    col_width + config.padding * 2.0 + config.padding
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Compute legend position within the plot area and draw it.
///
/// `data_points` is an optional flat list of `(px_x, px_y)` pixel positions
/// across all series — used by `LegendPosition::Best` to pick the corner with
/// the least overlap. Pass `None` or `&[]` if unavailable; `Best` then
/// falls back to `TopRight`.
///
/// Automatically promotes to multi-column layout when a single column
/// would exceed 40% of the plot height, keeping the legend compact.
/// If all vertical layouts overflow, switches to horizontal orientation.
/// If horizontal also overflows, truncates to top-N series with "+ N more".
///
/// Returns `(canvas, text_entries)` where text_entries is `Vec<(x, y, label)>`.
#[must_use]
pub fn draw_positioned_legend(
    mut canvas: PixelCanvas,
    entries: &[LegendEntry],
    plot: (f32, f32, f32, f32),
    config: &LegendConfig,
    _swatch_size: f32,
    _spacing: f32,
    data_points: Option<&[(f32, f32)]>,
) -> (PixelCanvas, Vec<(f32, f32, String)>) {
    if entries.is_empty() || !config.visible {
        return (canvas, Vec::new());
    }

    let (_px, _py, pw, ph) = plot;

    // Use config's swatch_size and spacing (ignore legacy parameters)
    let swatch_size = config.swatch_size;
    let spacing = config.spacing;

    // --- A2: Truncate labels that exceed 25% of plot width ---
    let max_label_w = pw * 0.25;
    let display_entries: Vec<LegendEntry> = entries
        .iter()
        .map(|e| {
            let truncated = text_utils::ellipsize(&e.label, max_label_w, config.char_width);
            LegendEntry {
                label: truncated,
                color: e.color,
            }
        })
        .collect();

    // --- A6: Auto-column promotion and overflow handling ---
    let (effective_config, effective_entries) = resolve_legend_layout(
        &display_entries,
        config,
        swatch_size,
        pw,
        ph,
    );

    let (px, py, pw, ph) = plot;
    let canvas_w = canvas.width() as f32;
    let canvas_h = canvas.height() as f32;

    // Auto-measure legend size
    let (legend_w, legend_h) = measure_legend(&effective_entries, swatch_size, &effective_config);

    // Position based on config
    let (lx, ly) = compute_position(
        effective_config.position,
        px,
        py,
        pw,
        ph,
        legend_w,
        legend_h,
        effective_config.padding,
        data_points,
    );

    // --- E2: Clamp legend to stay within canvas edges ---
    let legend_pad = effective_config.padding;
    let clamp_x_max = (canvas_w - legend_w - legend_pad).max(legend_pad);
    let clamp_y_max = (canvas_h - legend_h - legend_pad).max(legend_pad);
    let lx = lx.clamp(legend_pad, clamp_x_max);
    let ly = ly.clamp(legend_pad, clamp_y_max);

    // --- A3/A4: Draw background with proportional padding and near-opaque fill ---
    if let Some(bg) = effective_config.background {
        canvas = canvas
            .rect(lx, ly, legend_w, legend_h)
            .fill(bg)
            .corner_radius(3.0)
            .done();
    }

    // Draw border
    if let Some(border_color) = effective_config.border {
        canvas = canvas
            .rect(lx, ly, legend_w, legend_h)
            .stroke(border_color, 1.0)
            .corner_radius(3.0)
            .done();
    }

    draw_legend_entries(
        canvas,
        &effective_entries,
        lx + effective_config.padding,
        ly + effective_config.padding,
        swatch_size,
        spacing,
        effective_config.swatch_shape,
        &effective_config,
    )
}

/// Resolve legend layout: auto-promote columns, switch to horizontal,
/// or truncate entries when the legend would exceed 40% of plot height.
/// As a final escape, promotes to `OutsideRight` when all inside layouts overflow.
fn resolve_legend_layout(
    entries: &[LegendEntry],
    config: &LegendConfig,
    swatch_size: f32,
    plot_w: f32,
    plot_h: f32,
) -> (LegendConfig, Vec<LegendEntry>) {
    let max_h = plot_h * 0.4;
    let max_w = plot_w * 0.5;

    // Measure at current config
    let current_dims = measure_legend(entries, swatch_size, config);
    if current_dims.1 <= max_h && current_dims.0 <= max_w {
        return (config.clone(), entries.to_vec());
    }

    // Try vertical with increasing columns (only if currently vertical)
    if config.orientation == LegendOrientation::Vertical && entries.len() > 2 {
        // Try 2 columns
        let mut c = config.clone();
        c.columns = 2;
        let two = measure_legend(entries, swatch_size, &c);
        if two.1 <= max_h && two.0 <= max_w {
            return (c, entries.to_vec());
        }

        // Try 3 columns
        if entries.len() > 6 {
            c.columns = 3;
            let three = measure_legend(entries, swatch_size, &c);
            if three.1 <= max_h && three.0 <= max_w {
                return (c, entries.to_vec());
            }
        }
    }

    // Try horizontal layout
    let mut horiz = config.clone();
    horiz.orientation = LegendOrientation::Horizontal;
    let horiz_dims = measure_legend(entries, swatch_size, &horiz);
    if horiz_dims.0 <= max_w && horiz_dims.1 <= max_h {
        return (horiz, entries.to_vec());
    }

    // Promote to OutsideRight before truncating — preserving all entries
    // is more important than keeping the legend inside the plot area.
    let mut outside_cfg = config.clone();
    outside_cfg.position = LegendPosition::OutsideRight;
    let outside_dims = measure_legend(entries, swatch_size, &outside_cfg);
    // OutsideRight has no inside-plot height constraint, but check canvas fit
    if outside_dims.0 <= max_w {
        return (outside_cfg, entries.to_vec());
    }

    // Last resort: truncate to top-N entries with "+ N more" suffix
    for n in (3..entries.len()).rev() {
        let mut truncated: Vec<LegendEntry> = entries[..n].to_vec();
        let remaining = entries.len() - n;
        truncated.push(LegendEntry {
            label: format!("+ {remaining} more"),
            color: Color::from_rgba8(128, 128, 128, 180),
        });
        let dims = measure_legend(&truncated, swatch_size, config);
        if dims.1 <= max_h {
            return (config.clone(), truncated);
        }
    }

    // Absolute fallback: outside + truncated
    (outside_cfg, entries.to_vec())
}

/// Compute legend top-left position for a given placement.
///
/// For inside positions (TopRight, TopLeft, etc.), if `data_points` are
/// provided and the computed position would overlap data, automatically
/// falls through to [`best_corner`] to find a non-overlapping corner or
/// promote to outside placement.
#[allow(clippy::too_many_arguments)]
fn compute_position(
    position: LegendPosition,
    px: f32,
    py: f32,
    pw: f32,
    ph: f32,
    lw: f32,
    lh: f32,
    pad: f32,
    data_points: Option<&[(f32, f32)]>,
) -> (f32, f32) {
    // Outside positions are immune to data overlap — always honour them.
    match position {
        LegendPosition::OutsideRight => return (px + pw + pad, py + pad),
        LegendPosition::OutsideBottom => return (px + pad, py + ph + pad),
        _ => {}
    }

    // For Best, delegate directly.
    if matches!(position, LegendPosition::Best) {
        return best_corner(px, py, pw, ph, lw, lh, pad, data_points);
    }

    // Compute the requested inside position.
    let (lx, ly) = match position {
        LegendPosition::TopRight => (px + pw - lw - pad, py + pad),
        LegendPosition::TopLeft => (px + pad, py + pad),
        LegendPosition::BottomRight => (px + pw - lw - pad, py + ph - lh - pad),
        LegendPosition::BottomLeft => (px + pad, py + ph - lh - pad),
        LegendPosition::Top => (px + (pw - lw) / 2.0, py + pad),
        LegendPosition::Bottom => (px + (pw - lw) / 2.0, py + ph - lh - pad),
        LegendPosition::Left => (px + pad, py + (ph - lh) / 2.0),
        LegendPosition::Right => (px + pw - lw - pad, py + (ph - lh) / 2.0),
        // Already handled above.
        LegendPosition::Best | LegendPosition::OutsideRight | LegendPosition::OutsideBottom => {
            unreachable!()
        }
    };

    // If data_points are available, check whether this position overlaps.
    // If it does, let best_corner find a clear spot (or promote outside).
    if let Some(pts) = data_points {
        let overlaps = pts
            .iter()
            .any(|&(x, y)| x >= lx - 5.0 && x <= lx + lw + 5.0 && y >= ly - 5.0 && y <= ly + lh + 5.0);
        if overlaps {
            return best_corner(px, py, pw, ph, lw, lh, pad, data_points);
        }
    }

    (lx, ly)
}

/// Pick the corner with the fewest data-point overlaps.
///
/// Checks all four corners (TR, TL, BR, BL). For each, counts how many
/// data points fall inside the legend rectangle. Picks the corner with
/// the lowest count, breaking ties in the order TR → BL → TL → BR
/// (TR is the most common default; BL is the second-best for most charts).
///
/// **Academic convention (Tufte, Cleveland, matplotlib `loc='best'`):**
/// If the best inside corner still has ≥ 1 data-point collision, the legend
/// is promoted to `OutsideRight` placement so it never obscures data.
#[allow(clippy::too_many_arguments)]
fn best_corner(
    px: f32,
    py: f32,
    pw: f32,
    ph: f32,
    lw: f32,
    lh: f32,
    pad: f32,
    data_points: Option<&[(f32, f32)]>,
) -> (f32, f32) {
    let candidates = [
        (LegendPosition::TopRight, (px + pw - lw - pad, py + pad)),
        (LegendPosition::BottomLeft, (px + pad, py + ph - lh - pad)),
        (LegendPosition::TopLeft, (px + pad, py + pad)),
        (
            LegendPosition::BottomRight,
            (px + pw - lw - pad, py + ph - lh - pad),
        ),
    ];

    let points = match data_points {
        Some(pts) if !pts.is_empty() => pts,
        _ => return candidates[0].1, // no data → default to TopRight
    };

    let mut best_pos = candidates[0].1;
    let mut best_count = usize::MAX;

    for &(_label, (cx, cy)) in &candidates {
        let count = points
            .iter()
            .filter(|&&(x, y)| x >= cx - 5.0 && x <= cx + lw + 5.0 && y >= cy - 5.0 && y <= cy + lh + 5.0)
            .count();
        if count < best_count {
            best_count = count;
            best_pos = (cx, cy);
            if count == 0 {
                break; // can't do better than zero overlap
            }
        }
    }

    // Academic convention: never let the legend obscure data.
    // Promote to outside-right when even the best inside corner collides.
    if best_count > 0 {
        return compute_position(
            LegendPosition::OutsideRight,
            px, py, pw, ph, lw, lh, pad, None,
        );
    }

    best_pos
}

/// Draw legend entries with shape-aware swatches.
///
/// Renders in either vertical (stacked) or horizontal (side-by-side)
/// orientation based on `config.orientation`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn draw_legend_entries(
    mut canvas: PixelCanvas,
    entries: &[LegendEntry],
    x: f32,
    y: f32,
    swatch_size: f32,
    spacing: f32,
    shape: SwatchShape,
    config: &LegendConfig,
) -> (PixelCanvas, Vec<(f32, f32, String)>) {
    let mut text_entries = Vec::new();

    // Title rendering
    let y_offset = config.title.as_ref().map_or(0.0, |title| {
        text_entries.push((x, y, title.clone()));
        swatch_size + spacing
    });

    match config.orientation {
        LegendOrientation::Vertical => {
            let cols = config.columns.max(1);
            let max_label_len = entries
                .iter()
                .map(|e| e.label.chars().count())
                .max()
                .unwrap_or(0);
            let swatch_gap = 6.0;
            let col_gap = spacing * 2.0;
            let label_width = max_label_len as f32 * config.char_width;
            let col_width = swatch_size + swatch_gap + label_width;

            let mut current_y = y + y_offset;
            let mut col_idx = 0;
            let rows = entries.len().div_ceil(cols);
            let mut row_idx = 0;

            for entry in entries {
                let entry_x = x + (col_width + col_gap) * col_idx as f32;
                canvas = draw_swatch(canvas, entry_x, current_y, swatch_size, entry.color, shape);
                // Vertically center text with swatch: offset by half swatch height
                text_entries.push((
                    entry_x + swatch_size + swatch_gap,
                    current_y + swatch_size * 0.5 - 1.0,
                    entry.label.clone(),
                ));

                row_idx += 1;
                if row_idx >= rows {
                    row_idx = 0;
                    col_idx += 1;
                    current_y = y + y_offset;
                } else {
                    current_y += swatch_size + spacing;
                }
            }
        }
        LegendOrientation::Horizontal => {
            let mut current_x = x;
            let entry_y = y + y_offset;
            let swatch_gap = 6.0;
            let entry_gap = spacing * 2.0;
            for entry in entries {
                canvas = draw_swatch(canvas, current_x, entry_y, swatch_size, entry.color, shape);
                // Vertically center text with swatch
                text_entries.push((
                    current_x + swatch_size + swatch_gap,
                    entry_y + swatch_size * 0.5 - 1.0,
                    entry.label.clone(),
                ));
                let label_w = measure_label_width(&entry.label, config.char_width);
                current_x += swatch_size + swatch_gap + label_w + entry_gap;
            }
        }
    }

    (canvas, text_entries)
}

/// Draw a single swatch shape at (x, y).
fn draw_swatch(
    canvas: PixelCanvas,
    x: f32,
    y: f32,
    swatch_size: f32,
    color: Color,
    shape: SwatchShape,
) -> PixelCanvas {
    match shape {
        SwatchShape::Rect => canvas
            .rect(x, y, swatch_size, swatch_size)
            .fill(color)
            .corner_radius(2.0)
            .done(),
        SwatchShape::Circle => {
            let r = swatch_size / 2.0;
            canvas.circle(x + r, y + r, r).fill(color).done()
        }
        SwatchShape::Line => canvas
            .line(
                x,
                y + swatch_size / 2.0,
                x + swatch_size,
                y + swatch_size / 2.0,
            )
            .color(color)
            .width(2.5)
            .done(),
    }
}

/// Draw legend color swatches onto a canvas (backward-compat wrapper).
///
/// Returns the canvas and the entry positions for text overlay.
/// Each entry is `(x, y, label)`.
#[must_use]
pub fn draw_legend_swatches(
    canvas: PixelCanvas,
    entries: &[LegendEntry],
    x: f32,
    y: f32,
    swatch_size: f32,
    spacing: f32,
) -> (PixelCanvas, Vec<(f32, f32, String)>) {
    let config = LegendConfig::default();
    draw_legend_entries(
        canvas,
        entries,
        x,
        y,
        swatch_size,
        spacing,
        SwatchShape::Rect,
        &config,
    )
}
