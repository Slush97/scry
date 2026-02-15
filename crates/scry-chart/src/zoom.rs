//! Zoom and pan state management for interactive charts.
//!
//! Stores the current viewport as data-space x/y ranges and provides
//! methods for zoom-in, zoom-out, pan, and reset.

/// Zoom/pan state for an interactive chart.
#[derive(Clone, Debug)]
pub struct ZoomState {
    /// Original data extent (x_min, x_max, y_min, y_max).
    pub original: (f64, f64, f64, f64),
    /// Current viewport (x_min, x_max, y_min, y_max).
    pub viewport: (f64, f64, f64, f64),
    /// Zoom factor per step (e.g., 0.8 means zoom to 80% of current range).
    pub zoom_factor: f64,
    /// Pan step as fraction of current range (e.g., 0.1 = 10%).
    pub pan_fraction: f64,
}

impl ZoomState {
    /// Create a new zoom state from the original data extent.
    #[must_use]
    pub fn new(x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Self {
        Self {
            original: (x_min, x_max, y_min, y_max),
            viewport: (x_min, x_max, y_min, y_max),
            zoom_factor: 0.8,
            pan_fraction: 0.1,
        }
    }

    /// Get the current X range.
    #[must_use]
    pub fn x_range(&self) -> (f64, f64) {
        (self.viewport.0, self.viewport.1)
    }

    /// Get the current Y range.
    #[must_use]
    pub fn y_range(&self) -> (f64, f64) {
        (self.viewport.2, self.viewport.3)
    }

    /// Zoom in centered on the current viewport center.
    pub fn zoom_in(&mut self) {
        self.zoom_centered(self.zoom_factor);
    }

    /// Zoom out centered on the current viewport center.
    pub fn zoom_out(&mut self) {
        self.zoom_centered(1.0 / self.zoom_factor);
    }

    /// Zoom centered on a specific data coordinate.
    pub fn zoom_at(&mut self, center_x: f64, center_y: f64, factor: f64) {
        let (x0, x1, y0, y1) = self.viewport;
        let xw = (x1 - x0) * factor;
        let yh = (y1 - y0) * factor;

        let cx = center_x.clamp(x0, x1);
        let cy = center_y.clamp(y0, y1);

        // Fraction of viewport where center falls
        let fx = if (x1 - x0).abs() > f64::EPSILON {
            (cx - x0) / (x1 - x0)
        } else {
            0.5
        };
        let fy = if (y1 - y0).abs() > f64::EPSILON {
            (cy - y0) / (y1 - y0)
        } else {
            0.5
        };

        self.viewport = (
            cx - xw * fx,
            cx + xw * (1.0 - fx),
            cy - yh * fy,
            cy + yh * (1.0 - fy),
        );
    }

    /// Zoom centered on the viewport center.
    fn zoom_centered(&mut self, factor: f64) {
        let (x0, x1, y0, y1) = self.viewport;
        let cx = (x0 + x1) / 2.0;
        let cy = (y0 + y1) / 2.0;
        self.zoom_at(cx, cy, factor);
    }

    /// Pan the viewport in a direction.
    pub fn pan(&mut self, dx_frac: f64, dy_frac: f64) {
        let (x0, x1, y0, y1) = self.viewport;
        let xw = x1 - x0;
        let yh = y1 - y0;
        let dx = xw * dx_frac;
        let dy = yh * dy_frac;

        self.viewport = (x0 + dx, x1 + dx, y0 + dy, y1 + dy);
    }

    /// Pan left by `pan_fraction`.
    pub fn pan_left(&mut self) {
        self.pan(-self.pan_fraction, 0.0);
    }

    /// Pan right by `pan_fraction`.
    pub fn pan_right(&mut self) {
        self.pan(self.pan_fraction, 0.0);
    }

    /// Pan up by `pan_fraction`.
    pub fn pan_up(&mut self) {
        self.pan(0.0, self.pan_fraction);
    }

    /// Pan down by `pan_fraction`.
    pub fn pan_down(&mut self) {
        self.pan(0.0, -self.pan_fraction);
    }

    /// Reset to the original extent.
    pub fn reset(&mut self) {
        self.viewport = self.original;
    }

    /// Check if the viewport differs from the original.
    #[must_use]
    pub fn is_zoomed(&self) -> bool {
        let (ox0, ox1, oy0, oy1) = self.original;
        let (vx0, vx1, vy0, vy1) = self.viewport;
        (vx0 - ox0).abs() > f64::EPSILON
            || (vx1 - ox1).abs() > f64::EPSILON
            || (vy0 - oy0).abs() > f64::EPSILON
            || (vy1 - oy1).abs() > f64::EPSILON
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zoom_in_shrinks_viewport() {
        let mut z = ZoomState::new(0.0, 100.0, 0.0, 50.0);
        z.zoom_in();
        let (x0, x1) = z.x_range();
        assert!(x1 - x0 < 100.0, "x range should shrink: {} to {}", x0, x1);
        let (y0, y1) = z.y_range();
        assert!(y1 - y0 < 50.0, "y range should shrink: {} to {}", y0, y1);
    }

    #[test]
    fn zoom_out_grows_viewport() {
        let mut z = ZoomState::new(0.0, 100.0, 0.0, 50.0);
        z.zoom_in();
        z.zoom_out();
        let (x0, x1) = z.x_range();
        assert!((x1 - x0 - 100.0).abs() < 0.1, "should be ~100 after in+out");
    }

    #[test]
    fn pan_shifts_viewport() {
        let mut z = ZoomState::new(0.0, 100.0, 0.0, 50.0);
        z.pan_right();
        let (x0, x1) = z.x_range();
        assert!(x0 > 0.0, "x0 should have shifted right");
        assert!(x1 > 100.0, "x1 should have shifted right");
    }

    #[test]
    fn reset_restores_original() {
        let mut z = ZoomState::new(0.0, 100.0, 0.0, 50.0);
        z.zoom_in();
        z.pan_right();
        assert!(z.is_zoomed());
        z.reset();
        assert!(!z.is_zoomed());
    }
}
