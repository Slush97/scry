//! Cursor crosshair, nearest-point detection, and tooltip overlay.
//!
//! This module provides interactive cursor features for charts:
//! - Crosshair lines that follow the mouse position
//! - Nearest data point detection with tooltip display
//! - Data coordinate readout from pixel positions

use ratatui_pixelcanvas::scene::PixelCanvas;
use ratatui_pixelcanvas::style::Color;

use std::sync::Arc;

use crate::layout::{TextAlign, TextOverlay};
use crate::scale::{LinearScale, Scale};
use ratatui_pixelcanvas::style::DashPattern;

/// Custom tooltip formatter callback type.
///
/// Receives `(x, y, series_index)` and returns a display string.
pub type TooltipFn = Arc<dyn Fn(f64, f64, usize) -> String + Send + Sync>;

/// A data point with both data-space and pixel-space coordinates.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct DataPoint {
    /// Data X coordinate.
    pub x: f64,
    /// Data Y coordinate.
    pub y: f64,
    /// Series index this point belongs to.
    pub series_index: usize,
    /// Point index within the series.
    pub point_index: usize,
    /// Pixel X coordinate.
    pub px: f32,
    /// Pixel Y coordinate.
    pub py: f32,
}

/// Cursor state for interactive charts.
pub struct CursorState {
    /// Mouse position in pixel coordinates (within the chart canvas).
    pub pixel_pos: Option<(f32, f32)>,
    /// The data-coordinate position of the cursor.
    pub data_pos: Option<(f64, f64)>,
    /// The nearest data point to the cursor.
    pub nearest_point: Option<DataPoint>,
    /// Whether the crosshair is visible.
    pub show_crosshair: bool,
    /// Whether tooltips are enabled.
    pub show_tooltip: bool,
    /// Custom tooltip formatter. Receives `(x, y, series_index)` and returns
    /// a display string. When `None`, the default `"({x:.1}, {y:.1})"` format is used.
    pub tooltip_formatter: Option<TooltipFn>,
}

impl Clone for CursorState {
    fn clone(&self) -> Self {
        Self {
            pixel_pos: self.pixel_pos,
            data_pos: self.data_pos,
            nearest_point: self.nearest_point.clone(),
            show_crosshair: self.show_crosshair,
            show_tooltip: self.show_tooltip,
            tooltip_formatter: self.tooltip_formatter.clone(),
        }
    }
}

impl std::fmt::Debug for CursorState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CursorState")
            .field("pixel_pos", &self.pixel_pos)
            .field("data_pos", &self.data_pos)
            .field("nearest_point", &self.nearest_point)
            .field("show_crosshair", &self.show_crosshair)
            .field("show_tooltip", &self.show_tooltip)
            .field(
                "tooltip_formatter",
                &self.tooltip_formatter.as_ref().map(|_| ".."),
            )
            .finish()
    }
}

impl Default for CursorState {
    fn default() -> Self {
        Self::new()
    }
}

impl CursorState {
    /// Create a new cursor state with crosshair and tooltip enabled.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pixel_pos: None,
            data_pos: None,
            nearest_point: None,
            show_crosshair: true,
            show_tooltip: true,
            tooltip_formatter: None,
        }
    }

    /// Update cursor from a mouse position in pixel coordinates.
    ///
    /// Converts pixel position to data coordinates using the given scales,
    /// and finds the nearest data point from the provided series.
    pub fn update(
        &mut self,
        pixel_x: f32,
        pixel_y: f32,
        plot: (f32, f32, f32, f32),
        x_scale: &LinearScale,
        y_scale: &LinearScale,
        series_points: &[Vec<(f64, f64)>],
    ) {
        let (px, py, pw, ph) = plot;

        // Clamp to plot area
        if pixel_x < px || pixel_x > px + pw || pixel_y < py || pixel_y > py + ph {
            self.pixel_pos = None;
            self.data_pos = None;
            self.nearest_point = None;
            return;
        }

        self.pixel_pos = Some((pixel_x, pixel_y));

        // Convert to data coordinates
        let data_x = x_scale.to_data(pixel_x as f64);
        let data_y = y_scale.to_data(pixel_y as f64);
        self.data_pos = Some((data_x, data_y));

        // Find nearest point across all series
        let mut best: Option<(f64, DataPoint)> = None;
        for (si, series) in series_points.iter().enumerate() {
            for (pi, &(dx, dy)) in series.iter().enumerate() {
                let ppx = x_scale.to_pixel(dx) as f32;
                let ppy = y_scale.to_pixel(dy) as f32;
                let dist_sq = (ppx - pixel_x).powi(2) + (ppy - pixel_y).powi(2);

                let is_closer = match &best {
                    Some((d, _)) => (dist_sq as f64) < *d,
                    None => true,
                };

                if is_closer {
                    best = Some((
                        dist_sq as f64,
                        DataPoint {
                            x: dx,
                            y: dy,
                            series_index: si,
                            point_index: pi,
                            px: ppx,
                            py: ppy,
                        },
                    ));
                }
            }
        }

        // Only snap if within 25 pixels
        self.nearest_point = best
            .filter(|(dist_sq, _)| *dist_sq < 625.0) // 25^2
            .map(|(_, pt)| pt);
    }

    /// Clear the cursor state (e.g., when mouse leaves the chart).
    pub fn clear(&mut self) {
        self.pixel_pos = None;
        self.data_pos = None;
        self.nearest_point = None;
    }

    /// Toggle crosshair visibility.
    pub fn toggle_crosshair(&mut self) {
        self.show_crosshair = !self.show_crosshair;
    }

    /// Draw the crosshair overlay onto a canvas.
    ///
    /// Returns the canvas and any tooltip text overlays.
    #[must_use]
    pub fn draw_overlay(
        &self,
        mut canvas: PixelCanvas,
        plot: (f32, f32, f32, f32),
        crosshair_color: Color,
    ) -> (PixelCanvas, Vec<TextOverlay>) {
        let mut overlays = Vec::new();
        let (px, py, pw, ph) = plot;

        if let Some((mx, my)) = self.pixel_pos {
            // Draw crosshair
            if self.show_crosshair {
                let dash = DashPattern::new(vec![4.0, 3.0], 0.0);

                // Vertical crosshair line
                canvas = canvas
                    .line(mx, py, mx, py + ph)
                    .color(crosshair_color)
                    .width(1.0)
                    .dash(dash.clone())
                    .done();

                // Horizontal crosshair line
                canvas = canvas
                    .line(px, my, px + pw, my)
                    .color(crosshair_color)
                    .width(1.0)
                    .dash(dash)
                    .done();
            }

            // Highlight nearest point
            if let Some(ref pt) = self.nearest_point {
                // Draw highlight ring around nearest point
                canvas = canvas
                    .circle(pt.px, pt.py, 6.0)
                    .stroke(Color::from_rgba8(255, 255, 100, 200), 2.0)
                    .done();

                // Draw tooltip
                if self.show_tooltip {
                    let tooltip = self.tooltip_formatter.as_ref().map_or_else(
                        || format!("({:.1}, {:.1})", pt.x, pt.y),
                        |fmt| fmt(pt.x, pt.y, pt.series_index),
                    );
                    let text_w = tooltip.len() as f32 * 7.0 + 8.0;

                    // Flip tooltip direction when too close to edges
                    let tx = if pt.px + 12.0 + text_w > px + pw {
                        pt.px - text_w - 4.0
                    } else {
                        pt.px + 12.0
                    };
                    let ty = if pt.py - 22.0 < py {
                        pt.py + 16.0
                    } else {
                        pt.py - 12.0
                    };

                    // Tooltip background
                    canvas = canvas
                        .rect(tx - 3.0, ty - 10.0, text_w, 16.0)
                        .fill(Color::from_rgba8(30, 30, 30, 220))
                        .corner_radius(3.0)
                        .done();

                    overlays.push(TextOverlay {
                        x_px: tx,
                        y_px: ty - 4.0,
                        text: tooltip,
                        color: Color::from_rgba8(255, 255, 200, 255),
                        align: TextAlign::Left,
                        font_size: 11.0,
                        bold: false,
                        rotation_deg: 0.0,
                    });
                }
            }

            // Data coordinate readout at bottom
            if let Some((dx, dy)) = self.data_pos {
                overlays.push(TextOverlay {
                    x_px: px + pw - 2.0,
                    y_px: py + ph + 4.0,
                    text: format!("x={dx:.2} y={dy:.2}"),
                    color: crosshair_color,
                    align: TextAlign::Right,
                    font_size: 10.0,
                    bold: false,
                    rotation_deg: 0.0,
                });
            }
        }

        (canvas, overlays)
    }
}
