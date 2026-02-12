//! Axis rendering — tick marks, gridlines, and labels.
//!
//! Draws axes on a `PixelCanvas` using the scale system.

use ratatui_pixelcanvas::scene::PixelCanvas;
use ratatui_pixelcanvas::style::Color;

use crate::scale::{LinearScale, Scale};

/// Axis orientation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AxisSide {
    /// Bottom (X axis).
    Bottom,
    /// Left (Y axis).
    Left,
}

/// Configuration for rendering an axis.
#[derive(Clone, Debug)]
pub struct AxisConfig {
    /// Which side the axis is on.
    pub side: AxisSide,
    /// Color of the axis line.
    pub axis_color: Color,
    /// Color of gridlines.
    pub grid_color: Color,
    /// Axis line width in pixels.
    pub axis_width: f32,
    /// Grid line width in pixels.
    pub grid_width: f32,
    /// Length of tick marks in pixels.
    pub tick_length: f32,
    /// Whether to draw gridlines.
    pub show_grid: bool,
}

impl Default for AxisConfig {
    fn default() -> Self {
        Self {
            side: AxisSide::Bottom,
            axis_color: Color::from_rgba8(160, 160, 180, 255),
            grid_color: Color::from_rgba8(50, 50, 65, 120),
            axis_width: 1.5,
            grid_width: 0.5,
            tick_length: 5.0,
            show_grid: true,
        }
    }
}

/// Draw an axis (line + ticks + gridlines) onto a canvas.
///
/// - `plot_area`: The pixel rectangle of the plot area `(x, y, w, h)`.
/// - `scale`: The linear scale mapping data to pixels.
/// - `config`: Axis styling.
///
/// Returns tick positions and their formatted labels for text rendering.
pub fn draw_axis(
    mut canvas: PixelCanvas,
    plot_area: (f32, f32, f32, f32),
    scale: &LinearScale,
    config: &AxisConfig,
) -> (PixelCanvas, Vec<(f32, String)>) {
    let (px, py, pw, ph) = plot_area;
    let ticks = scale.ticks(6);
    let mut tick_labels = Vec::new();

    match config.side {
        AxisSide::Bottom => {
            // Horizontal axis line along bottom of plot area
            canvas = canvas
                .line(px, py + ph, px + pw, py + ph)
                .color(config.axis_color)
                .width(config.axis_width)
                .done();

            for &t in &ticks {
                let x = scale.to_pixel(t) as f32;
                if x < px || x > px + pw {
                    continue;
                }

                // Tick mark
                canvas = canvas
                    .line(x, py + ph, x, py + ph + config.tick_length)
                    .color(config.axis_color)
                    .width(1.0)
                    .done();

                // Gridline
                if config.show_grid {
                    canvas = canvas
                        .line(x, py, x, py + ph)
                        .color(config.grid_color)
                        .width(config.grid_width)
                        .done();
                }

                tick_labels.push((x, scale.format_tick(t)));
            }
        }
        AxisSide::Left => {
            // Vertical axis line along left of plot area
            canvas = canvas
                .line(px, py, px, py + ph)
                .color(config.axis_color)
                .width(config.axis_width)
                .done();

            for &t in &ticks {
                let y = scale.to_pixel(t) as f32;
                if y < py || y > py + ph {
                    continue;
                }

                // Tick mark
                canvas = canvas
                    .line(px - config.tick_length, y, px, y)
                    .color(config.axis_color)
                    .width(1.0)
                    .done();

                // Gridline
                if config.show_grid {
                    canvas = canvas
                        .line(px, y, px + pw, y)
                        .color(config.grid_color)
                        .width(config.grid_width)
                        .done();
                }

                tick_labels.push((y, scale.format_tick(t)));
            }
        }
    }

    (canvas, tick_labels)
}
