// SPDX-License-Identifier: MIT OR Apache-2.0
//! Legend rendering utilities.
//!
//! Draws color swatches with labels to identify data series.
//! Supports auto-sizing, 9 placement options, and multiple swatch shapes.

use scry_engine::scene::PixelCanvas;
use scry_engine::style::Color;

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
            background: None,
            border: None,
            font_color: Color::from_rgba8(200, 200, 220, 255),
            swatch_shape: SwatchShape::Rect,
            padding: 8.0,
            spacing: 4.0,
            char_width: 7.0,
            visible: true,
            title: None,
            orientation: LegendOrientation::Vertical,
            columns: 1,
        }
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
            let swatch_gap = 4.0;
            let col_width = swatch_size + swatch_gap + label_width;
            let col_gap = config.spacing * 2.0;

            // Title may be wider than entries
            let title_width = config
                .title
                .as_ref()
                .map_or(0.0, |t| t.chars().count() as f32 * config.char_width);

            let content_width = col_width * cols as f32 + col_gap * (cols as f32 - 1.0).max(0.0);
            let width = config.padding * 2.0 + content_width.max(title_width);

            let rows = entries.len().div_ceil(cols);
            let entries_height = rows as f32 * (swatch_size + config.spacing) - config.spacing;
            let height = config.padding * 2.0 + title_height + entries_height;

            (width.max(40.0), height.max(20.0))
        }
        LegendOrientation::Horizontal => {
            let swatch_gap = 4.0;
            let entry_gap = config.spacing * 2.0; // gap between entries
            let total_width: f32 = entries
                .iter()
                .map(|e| {
                    swatch_size + swatch_gap + e.label.chars().count() as f32 * config.char_width
                })
                .sum::<f32>()
                + entry_gap * (entries.len().saturating_sub(1)) as f32;

            let title_width = config
                .title
                .as_ref()
                .map_or(0.0, |t| t.chars().count() as f32 * config.char_width);

            let width = config.padding * 2.0 + total_width.max(title_width);
            let height = config.padding * 2.0 + title_height + swatch_size;

            (width.max(40.0), height.max(20.0))
        }
    }
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
/// Returns `(canvas, text_entries)` where text_entries is `Vec<(x, y, label)>`.
#[must_use]
pub fn draw_positioned_legend(
    mut canvas: PixelCanvas,
    entries: &[LegendEntry],
    plot: (f32, f32, f32, f32),
    config: &LegendConfig,
    swatch_size: f32,
    spacing: f32,
    data_points: Option<&[(f32, f32)]>,
) -> (PixelCanvas, Vec<(f32, f32, String)>) {
    if entries.is_empty() || !config.visible {
        return (canvas, Vec::new());
    }

    let (px, py, pw, ph) = plot;

    // Auto-measure legend size
    let (legend_w, legend_h) = measure_legend(entries, swatch_size, config);

    // Position based on config
    let (lx, ly) = compute_position(
        config.position,
        px,
        py,
        pw,
        ph,
        legend_w,
        legend_h,
        config.padding,
        data_points,
    );

    // Draw background
    if let Some(bg) = config.background {
        canvas = canvas
            .rect(lx - 4.0, ly - 4.0, legend_w + 8.0, legend_h + 8.0)
            .fill(bg)
            .corner_radius(4.0)
            .done();
    }

    // Draw border
    if let Some(border_color) = config.border {
        canvas = canvas
            .rect(lx - 4.0, ly - 4.0, legend_w + 8.0, legend_h + 8.0)
            .stroke(border_color, 1.0)
            .corner_radius(4.0)
            .done();
    }

    draw_legend_entries(
        canvas,
        entries,
        lx + config.padding,
        ly + config.padding,
        swatch_size,
        spacing,
        config.swatch_shape,
        config,
    )
}

/// Compute legend top-left position for a given placement.
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
    match position {
        LegendPosition::TopRight => (px + pw - lw - pad, py + pad),
        LegendPosition::TopLeft => (px + pad, py + pad),
        LegendPosition::BottomRight => (px + pw - lw - pad, py + ph - lh - pad),
        LegendPosition::BottomLeft => (px + pad, py + ph - lh - pad),
        LegendPosition::Top => (px + (pw - lw) / 2.0, py + pad),
        LegendPosition::Bottom => (px + (pw - lw) / 2.0, py + ph - lh - pad),
        LegendPosition::Left => (px + pad, py + (ph - lh) / 2.0),
        LegendPosition::Right => (px + pw - lw - pad, py + (ph - lh) / 2.0),
        LegendPosition::OutsideRight => (px + pw + pad, py + pad),
        LegendPosition::OutsideBottom => (px + pad, py + ph + pad),
        LegendPosition::Best => best_corner(px, py, pw, ph, lw, lh, pad, data_points),
    }
}

/// Pick the corner with the fewest data-point overlaps.
///
/// Checks all four corners (TR, TL, BR, BL). For each, counts how many
/// data points fall inside the legend rectangle. Picks the corner with
/// the lowest count, breaking ties in the order TR → BL → TL → BR
/// (TR is the most common default; BL is the second-best for most charts).
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
            .filter(|&&(x, y)| x >= cx && x <= cx + lw && y >= cy && y <= cy + lh)
            .count();
        if count < best_count {
            best_count = count;
            best_pos = (cx, cy);
            if count == 0 {
                break; // can't do better than zero overlap
            }
        }
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
            let swatch_gap = 4.0;
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
                text_entries.push((
                    entry_x + swatch_size + swatch_gap,
                    current_y,
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
            let swatch_gap = 4.0;
            let entry_gap = spacing * 2.0;
            for entry in entries {
                canvas = draw_swatch(canvas, current_x, entry_y, swatch_size, entry.color, shape);
                text_entries.push((
                    current_x + swatch_size + swatch_gap,
                    entry_y,
                    entry.label.clone(),
                ));
                let label_w = entry.label.chars().count() as f32 * config.char_width;
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
