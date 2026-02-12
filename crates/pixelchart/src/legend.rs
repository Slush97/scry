//! Legend rendering utilities.
//!
//! Draws color swatches with labels to identify data series.
//! Supports configurable positioning (TopRight, TopLeft, etc.).

use ratatui_pixelcanvas::scene::PixelCanvas;
use ratatui_pixelcanvas::style::Color;

/// Legend placement within the plot area.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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
}

/// Configuration for legend rendering.
#[derive(Clone, Debug)]
pub struct LegendConfig {
    /// Where to place the legend.
    pub position: LegendPosition,
    /// Optional background color behind the legend.
    pub background: Option<Color>,
    /// Optional border color around the legend.
    pub border: Option<Color>,
}

impl Default for LegendConfig {
    fn default() -> Self {
        Self {
            position: LegendPosition::TopRight,
            background: None,
            border: None,
        }
    }
}

/// A single legend entry.
#[derive(Clone, Debug)]
pub struct LegendEntry {
    /// Display label.
    pub label: String,
    /// Color swatch.
    pub color: Color,
}

/// Compute legend position within the plot area and draw it.
///
/// Returns `(canvas, text_entries)` where text_entries is `Vec<(x, y, label)>`.
pub fn draw_positioned_legend(
    mut canvas: PixelCanvas,
    entries: &[LegendEntry],
    plot: (f32, f32, f32, f32),
    config: &LegendConfig,
    swatch_size: f32,
    spacing: f32,
) -> (PixelCanvas, Vec<(f32, f32, String)>) {
    if entries.is_empty() {
        return (canvas, Vec::new());
    }

    let (px, py, pw, ph) = plot;
    let legend_w = 80.0_f32;
    let legend_h = entries.len() as f32 * (swatch_size + spacing) - spacing;
    let padding = 8.0;

    let (lx, ly) = match config.position {
        LegendPosition::TopRight => (px + pw - legend_w - padding, py + padding),
        LegendPosition::TopLeft => (px + padding, py + padding),
        LegendPosition::BottomRight => (px + pw - legend_w - padding, py + ph - legend_h - padding * 2.0),
        LegendPosition::BottomLeft => (px + padding, py + ph - legend_h - padding * 2.0),
    };

    // Draw background if configured
    if let Some(bg) = config.background {
        canvas = canvas
            .rect(lx - 4.0, ly - 4.0, legend_w + 8.0, legend_h + 8.0)
            .fill(bg)
            .corner_radius(4.0)
            .done();
    }

    // Draw border if configured
    if let Some(border_color) = config.border {
        canvas = canvas
            .rect(lx - 4.0, ly - 4.0, legend_w + 8.0, legend_h + 8.0)
            .stroke(border_color, 1.0)
            .corner_radius(4.0)
            .done();
    }

    draw_legend_swatches(canvas, entries, lx, ly, swatch_size, spacing)
}

/// Draw legend color swatches onto a canvas.
///
/// Returns the canvas and the entry positions for text overlay.
/// Each entry is `(x, y, label)`.
pub fn draw_legend_swatches(
    mut canvas: PixelCanvas,
    entries: &[LegendEntry],
    x: f32,
    y: f32,
    swatch_size: f32,
    spacing: f32,
) -> (PixelCanvas, Vec<(f32, f32, String)>) {
    let mut text_entries = Vec::new();
    let mut current_y = y;

    for entry in entries {
        // Draw color swatch as a filled rounded rect
        canvas = canvas
            .rect(x, current_y, swatch_size, swatch_size)
            .fill(entry.color)
            .corner_radius(2.0)
            .done();

        // Record position for text label (right of swatch)
        text_entries.push((
            x + swatch_size + 4.0,
            current_y,
            entry.label.clone(),
        ));

        current_y += swatch_size + spacing;
    }

    (canvas, text_entries)
}
