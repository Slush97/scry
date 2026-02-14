//! Ratatui widget integration for rendering charts.
//!
//! Provides [`ChartWidget`] and [`ChartState`], which coordinate pixel
//! rendering via `ratatui-pixelcanvas` and overlay text labels via ratatui.

use ratatui::prelude::*;
use ratatui::widgets::StatefulWidget;

use ratatui_pixelcanvas::prelude::{PixelCanvasState, PixelCanvasWidget};
use ratatui_pixelcanvas::transport::backend::FontSize;
use ratatui_pixelcanvas::transport::{self, ProtocolBackend};

use crate::chart::Chart;
use crate::layout::{self, TextAlign, TextOverlay};

// ---------------------------------------------------------------------------
// ChartWidget
// ---------------------------------------------------------------------------

/// A ratatui widget that renders a pixel-perfect chart.
///
/// # Example
///
/// ```ignore
/// use pixelchart::prelude::*;
///
/// let chart = Chart::line(&[1.0, 4.0, 2.0, 8.0, 5.0])
///     .title("My Data")
///     .build();
///
/// frame.render_stateful_widget(
///     ChartWidget::new(&chart),
///     area,
///     &mut chart_state,
/// );
/// ```
#[must_use]
pub struct ChartWidget<'a> {
    chart: &'a Chart,
    z_index: i32,
}

impl<'a> ChartWidget<'a> {
    /// Create a new chart widget referencing a chart specification.
    pub fn new(chart: &'a Chart) -> Self {
        Self { chart, z_index: -1 }
    }

    /// Set the z-index for Kitty layering.
    pub fn z_index(mut self, z: i32) -> Self {
        self.z_index = z;
        self
    }
}

// ---------------------------------------------------------------------------
// ChartState
// ---------------------------------------------------------------------------

/// Persistent state for [`ChartWidget`] across render frames.
///
/// This wraps a [`PixelCanvasState`] and manages the underlying
/// graphics protocol connection. Optionally holds cursor and zoom
/// state for interactive charts.
pub struct ChartState {
    /// Underlying pixel canvas state.
    pub(crate) pixel_state: PixelCanvasState,
    /// Cursor crosshair and nearest-point state.
    pub cursor: crate::cursor::CursorState,
    /// Zoom and pan state.
    pub zoom: Option<crate::zoom::ZoomState>,
    /// Whether interactivity is enabled.
    pub interactive: bool,
    /// Last known plot area (set during render).
    pub(crate) last_plot: Option<(f32, f32, f32, f32)>,
    /// Last known render area in terminal cell coordinates.
    pub(crate) last_area: Option<Rect>,
}

impl ChartState {
    /// Create a new chart state from a pixel canvas state.
    #[must_use]
    pub fn new(pixel_state: PixelCanvasState) -> Self {
        Self {
            pixel_state,
            cursor: crate::cursor::CursorState::new(),
            zoom: None,
            interactive: false,
            last_plot: None,
            last_area: None,
        }
    }

    /// Auto-detect the best protocol and create state with one call.
    ///
    /// This is the recommended way to create chart state:
    /// ```ignore
    /// let mut state = ChartState::auto();
    /// ```
    #[must_use]
    pub fn auto() -> Self {
        use ratatui_pixelcanvas::prelude::{Picker, ProtocolKind};

        let picker = Picker::detect();
        let backend: Box<dyn ProtocolBackend> = match picker.protocol() {
            ProtocolKind::Kitty => {
                Box::new(transport::kitty::KittyBackend::new(picker.font_size()))
            }
            _ => Box::new(transport::halfblock::HalfblockBackend::new()),
        };
        Self::new(PixelCanvasState::new(backend, picker.font_size()))
    }

    /// Enable interactivity (crosshair, tooltip, zoom/pan).
    #[must_use]
    pub fn with_interactivity(mut self) -> Self {
        self.interactive = true;
        self.cursor.show_crosshair = true;
        self.cursor.show_tooltip = true;
        self
    }

    /// Initialize zoom state from data extents.
    pub fn set_zoom_extents(&mut self, x_min: f64, x_max: f64, y_min: f64, y_max: f64) {
        self.zoom = Some(crate::zoom::ZoomState::new(x_min, x_max, y_min, y_max));
    }

    /// Get the font size for cell ↔ pixel calculations.
    #[must_use]
    pub fn font_size(&self) -> FontSize {
        self.pixel_state.font_size()
    }

    /// Handle a mouse event, updating cursor position and zoom.
    ///
    /// `col` and `row` are terminal cell coordinates from the mouse event.
    pub fn handle_mouse_move(&mut self, col: u16, row: u16) {
        if !self.interactive {
            return;
        }

        let Some(area) = self.last_area else { return };
        let Some(_plot) = self.last_plot else { return };

        // Convert terminal cell coords to pixel coords
        let font = self.font_size();
        let pixel_x = (col.saturating_sub(area.x)) as f32 * f32::from(font.width);
        let pixel_y = (row.saturating_sub(area.y)) as f32 * f32::from(font.height);

        self.cursor.pixel_pos = Some((pixel_x, pixel_y));
    }

    /// Handle a scroll event for zooming.
    pub fn handle_scroll_up(&mut self) {
        if let Some(ref mut zoom) = self.zoom {
            zoom.zoom_in();
        }
    }

    /// Handle a scroll down event for zooming out.
    pub fn handle_scroll_down(&mut self) {
        if let Some(ref mut zoom) = self.zoom {
            zoom.zoom_out();
        }
    }

    /// Get the cursor's current data-coordinate position.
    #[must_use]
    pub fn cursor_data_position(&self) -> Option<(f64, f64)> {
        self.cursor.data_pos
    }

    /// Get the nearest data point to the cursor.
    #[must_use]
    pub fn nearest_point(&self) -> Option<&crate::cursor::DataPoint> {
        self.cursor.nearest_point.as_ref()
    }

    /// Transmit any pending image to the terminal.
    ///
    /// Call this **after** `terminal.draw()` to avoid flicker.
    /// See [`PixelCanvasState::flush()`] for details.
    pub fn flush(&mut self) -> Result<(), ratatui_pixelcanvas::PixelCanvasError> {
        self.pixel_state.flush()
    }

    /// Clean up resources.
    pub fn cleanup(&mut self) {
        self.pixel_state.cleanup();
    }
}

impl std::fmt::Debug for ChartState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChartState")
            .field("pixel_state", &self.pixel_state)
            .field("interactive", &self.interactive)
            .field("cursor", &self.cursor)
            .field("zoom", &self.zoom)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// StatefulWidget implementation
// ---------------------------------------------------------------------------

impl StatefulWidget for ChartWidget<'_> {
    type State = ChartState;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::State) {
        if area.width < 4 || area.height < 4 {
            return;
        }

        let font = state.font_size();
        let pixel_w = u32::from(area.width) * u32::from(font.width);
        let pixel_h = u32::from(area.height) * u32::from(font.height);

        // Store area for event handlers
        state.last_area = Some(area);

        // Render the chart into a PixelCanvas.
        // If zoom is active, pass the viewport to avoid inlining clone logic here.
        let viewport = state.zoom.as_ref().filter(|z| z.is_zoomed()).map(|z| {
            let (x0, x1) = z.x_range();
            let (y0, y1) = z.y_range();
            (x0, x1, y0, y1)
        });
        let mut rendered =
            layout::render_chart_with_viewport(self.chart, pixel_w, pixel_h, viewport);

        // Store plot area for cursor calculations
        state.last_plot = rendered.plot_area;

        // Update cursor state with scales and series data if interactive
        if state.interactive {
            if let (Some(plot), Some(ref x_scale), Some(ref y_scale)) =
                (rendered.plot_area, &rendered.x_scale, &rendered.y_scale)
            {
                // If cursor has a pixel position, update data coordinates and nearest point
                if let Some((px, py)) = state.cursor.pixel_pos {
                    state
                        .cursor
                        .update(px, py, plot, x_scale, y_scale, &rendered.series_points);
                }

                // Draw cursor overlay (crosshair, tooltip, highlight)
                let crosshair_color =
                    ratatui_pixelcanvas::style::Color::from_rgba8(200, 200, 220, 160);
                let (canvas, cursor_overlays) =
                    state
                        .cursor
                        .draw_overlay(rendered.canvas, plot, crosshair_color);
                rendered.canvas = canvas;
                rendered.text_overlays.extend(cursor_overlays);
            }
        }

        // Render the pixel canvas via the underlying widget
        let pcw = PixelCanvasWidget::new(rendered.canvas).z_index(self.z_index);
        pcw.render(area, buf, &mut state.pixel_state);

        // Overlay text labels via ratatui
        render_text_overlays(buf, area, &rendered.text_overlays, font);
    }
}

// ---------------------------------------------------------------------------
// Text overlay rendering
// ---------------------------------------------------------------------------

/// Render text overlays on top of the chart using the ratatui buffer.
fn render_text_overlays(buf: &mut Buffer, area: Rect, overlays: &[TextOverlay], font: FontSize) {
    for overlay in overlays {
        // Convert pixel position to cell position
        let cell_x = (overlay.x_px / f32::from(font.width)) as u16;
        let cell_y = (overlay.y_px / f32::from(font.height)) as u16;

        let abs_x = area.x + cell_x;
        let abs_y = area.y + cell_y;

        // Skip if outside the buffer area
        if abs_x >= area.x + area.width || abs_y >= area.y + area.height {
            continue;
        }

        let text = &overlay.text;
        let r = (overlay.color.r * 255.0) as u8;
        let g = (overlay.color.g * 255.0) as u8;
        let b = (overlay.color.b * 255.0) as u8;
        let style = Style::default().fg(ratatui::style::Color::Rgb(r, g, b));

        if overlay.rotation_deg > 0.0 {
            // Rotated text: stack characters vertically.
            // For 90°, each character goes straight down one row.
            // For 45°, each character goes down one row AND right one column (diagonal).
            let is_diagonal = overlay.rotation_deg < 60.0;

            // For rotated labels, anchor at the tick position
            // (right-aligned overlays: start from the anchor point going down)
            for (i, ch) in text.chars().enumerate() {
                let cy = abs_y + i as u16;
                let cx = if is_diagonal { abs_x + i as u16 } else { abs_x };

                if cy >= area.y + area.height || cx >= area.x + area.width {
                    break;
                }
                if cx >= area.x && cy >= area.y {
                    if let Some(cell) = buf.cell_mut(Position::new(cx, cy)) {
                        cell.set_char(ch);
                        cell.set_style(style);
                    }
                }
            }
        } else {
            // Horizontal text (original logic)
            let text_len = text.chars().count() as u16;
            let draw_x = match overlay.align {
                TextAlign::Left => abs_x,
                TextAlign::Center => abs_x.saturating_sub(text_len / 2),
                TextAlign::Right => abs_x.saturating_sub(text_len),
            };

            for (i, ch) in text.chars().enumerate() {
                let cx = draw_x + i as u16;
                if cx >= area.x + area.width {
                    break;
                }
                if cx >= area.x && abs_y >= area.y {
                    if let Some(cell) = buf.cell_mut(Position::new(cx, abs_y)) {
                        cell.set_char(ch);
                        cell.set_style(style);
                    }
                }
            }
        }
    }
}
