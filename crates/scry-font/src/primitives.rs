// SPDX-License-Identifier: MIT OR Apache-2.0
//! Geometric primitives for glyph construction.
//!
//! Every glyph in Sigil Mono is composed from a small vocabulary of reusable
//! primitives — like shared libraries in a codebase. This module provides
//! those building blocks.
//!
//! All coordinates are in font units (UPM space, typically 0–1000).
//! Contours use **clockwise** winding for outer paths and **counter-clockwise**
//! for inner paths (cutouts), following TrueType conventions.

use std::f32::consts::PI;

use crate::params::FontParams;

// ── Point types ─────────────────────────────────────────────────────

/// A point on a glyph contour.
#[derive(Clone, Copy, Debug)]
pub struct Point {
    /// X coordinate in font units.
    pub x: i16,
    /// Y coordinate in font units.
    pub y: i16,
    /// Whether this is an on-curve point (true) or a quadratic Bézier
    /// control point (false).
    pub on_curve: bool,
}

impl Point {
    /// Create an on-curve point.
    pub fn on(x: i16, y: i16) -> Self {
        Self {
            x,
            y,
            on_curve: true,
        }
    }

    /// Create an off-curve (quadratic Bézier control) point.
    pub fn off(x: i16, y: i16) -> Self {
        Self {
            x,
            y,
            on_curve: false,
        }
    }

    /// Create a point from f32 coordinates, rounding to nearest integer.
    pub fn on_f(x: f32, y: f32) -> Self {
        Self::on(x.round() as i16, y.round() as i16)
    }

    /// Create an off-curve point from f32 coordinates.
    pub fn off_f(x: f32, y: f32) -> Self {
        Self::off(x.round() as i16, y.round() as i16)
    }
}

/// A closed contour — a sequence of points forming a closed path.
#[derive(Clone, Debug)]
pub struct Contour {
    /// The points of the contour in order.
    pub points: Vec<Point>,
}

impl Contour {
    /// Create a new contour from a list of points.
    pub fn new(points: Vec<Point>) -> Self {
        Self { points }
    }

    /// Reverse the winding direction of the contour.
    pub fn reverse(&mut self) {
        self.points.reverse();
    }

    /// Create a reversed copy of the contour.
    pub fn reversed(&self) -> Self {
        let mut c = self.clone();
        c.reverse();
        c
    }

    /// Translate all points by (dx, dy).
    pub fn translate(&mut self, dx: i16, dy: i16) {
        for p in &mut self.points {
            p.x += dx;
            p.y += dy;
        }
    }

    /// Create a translated copy.
    pub fn translated(&self, dx: i16, dy: i16) -> Self {
        let mut c = self.clone();
        c.translate(dx, dy);
        c
    }
}

// ── Primitive: Rectangle ────────────────────────────────────────────

/// A simple rectangle contour (clockwise winding).
///
/// Used as the basis for stems and bars.
pub fn rect(x: i16, y: i16, w: i16, h: i16) -> Contour {
    Contour::new(vec![
        Point::on(x, y),
        Point::on(x + w, y),
        Point::on(x + w, y + h),
        Point::on(x, y + h),
    ])
}

// ── Primitive: Vertical Stem ────────────────────────────────────────

/// A vertical stem — the backbone of letters like `b`, `d`, `h`, `k`, `l`.
///
/// Placed at horizontal position `x` (left edge of stroke), spanning from
/// `y_bottom` to `y_top`.
pub fn vertical_stem(x: i16, y_bottom: i16, y_top: i16, p: &FontParams) -> Contour {
    rect(x, y_bottom, p.stroke_width, y_top - y_bottom)
}

// ── Primitive: Horizontal Bar ───────────────────────────────────────

/// A horizontal bar — used for crossbars in `e`, `f`, `t`, `z`, etc.
///
/// Placed at vertical position `y` (bottom edge), spanning from
/// `x_left` to `x_right`.
pub fn horizontal_bar(x_left: i16, x_right: i16, y: i16, p: &FontParams) -> Contour {
    rect(x_left, y, x_right - x_left, p.stroke_width)
}

// ── Primitive: Arc (Quadratic Bézier) ───────────────────────────────

/// Approximate a circular arc with quadratic Bézier segments.
///
/// Returns a list of points (alternating on-curve and off-curve) that
/// traces a circular arc from `start_angle` to `end_angle` (in degrees,
/// counter-clockwise from 3-o'clock).
///
/// `cx`, `cy` = center, `radius` = radius.
/// The arc is split into 90° segments, each approximated by one quadratic
/// Bézier curve.
fn arc_points(
    cx: f32,
    cy: f32,
    radius: f32,
    start_deg: f32,
    end_deg: f32,
) -> Vec<Point> {
    let start = start_deg * PI / 180.0;
    let end = end_deg * PI / 180.0;
    let mut angle = start;
    let total = end - start;
    let dir = total.signum();
    let abs_total = total.abs();

    // Number of 90° quadratic Bézier segments needed
    let n_segments = (abs_total / (PI / 2.0)).ceil() as usize;
    if n_segments == 0 {
        return vec![Point::on_f(cx + radius * start.cos(), cy + radius * start.sin())];
    }
    let step = total / n_segments as f32;

    let mut points = Vec::with_capacity(n_segments * 2 + 1);

    // First on-curve point
    points.push(Point::on_f(
        cx + radius * angle.cos(),
        cy + radius * angle.sin(),
    ));

    for _ in 0..n_segments {
        let next_angle = angle + step;
        let mid_angle = angle + step / 2.0;

        // The off-curve control point sits on the tangent intersection
        // For quadratic Bézier approximation of a circular arc:
        // control = intersection of tangent lines at start and end of segment
        let k = (4.0 / 3.0) * ((step / 4.0).abs().tan());
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let cos_b = next_angle.cos();
        let sin_b = next_angle.sin();

        // Control point via tangent-line intersection
        let ctrl_x = cx + radius * (cos_a - k * sin_a * dir);
        let ctrl_y = cy + radius * (sin_a + k * cos_a * dir);

        // But TrueType uses quadratic Béziers, so we use the midpoint approach:
        // The off-curve point is along the bisector at radius / cos(half_step)
        let half = step.abs() / 2.0;
        let bisect_r = radius / half.cos();
        let ctrl_x = cx + bisect_r * mid_angle.cos();
        let ctrl_y = cy + bisect_r * mid_angle.sin();

        points.push(Point::off_f(ctrl_x, ctrl_y));
        points.push(Point::on_f(
            cx + radius * cos_b,
            cy + radius * sin_b,
        ));

        angle = next_angle;
    }

    points
}

/// Build a full circular arc contour (outer stroke only, open path turned into
/// a closed thick-stroked arc).
///
/// Creates a "thick arc" by tracing the outer radius forward and the inner
/// radius backward, closing the shape into a filled contour.
pub fn thick_arc(
    cx: f32,
    cy: f32,
    outer_r: f32,
    inner_r: f32,
    start_deg: f32,
    end_deg: f32,
) -> Contour {
    let outer = arc_points(cx, cy, outer_r, start_deg, end_deg);
    let inner = arc_points(cx, cy, inner_r, end_deg, start_deg); // reversed direction

    let mut points = outer;
    points.extend(inner);
    Contour::new(points)
}

/// Build a full circle contour (outer boundary, clockwise).
pub fn circle(cx: f32, cy: f32, radius: f32) -> Contour {
    let pts = arc_points(cx, cy, radius, 0.0, 360.0);
    // Remove the duplicate last point (same as first for 360°)
    let mut pts = pts;
    if pts.len() > 1 {
        pts.pop();
    }
    Contour::new(pts)
}

/// Build a filled ring (circle with hole).
///
/// Returns two contours: outer (clockwise) and inner (counter-clockwise).
pub fn ring(cx: f32, cy: f32, outer_r: f32, inner_r: f32) -> Vec<Contour> {
    let outer = circle(cx, cy, outer_r);
    let inner = circle(cx, cy, inner_r).reversed();
    vec![outer, inner]
}

// ── Primitive: Diagonal ─────────────────────────────────────────────

/// A diagonal stroke from (x0, y0) to (x1, y1) with given thickness.
///
/// Creates a parallelogram contour offset perpendicular to the stroke direction.
pub fn diagonal(x0: f32, y0: f32, x1: f32, y1: f32, thickness: f32) -> Contour {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len = dx.hypot(dy);
    if len < 0.001 {
        return Contour::new(vec![]);
    }
    // Perpendicular unit vector
    let nx = -dy / len * thickness / 2.0;
    let ny = dx / len * thickness / 2.0;

    Contour::new(vec![
        Point::on_f(x0 + nx, y0 + ny),
        Point::on_f(x1 + nx, y1 + ny),
        Point::on_f(x1 - nx, y1 - ny),
        Point::on_f(x0 - nx, y0 - ny),
    ])
}

// ── Primitive: 45° Terminal Cut ─────────────────────────────────────

/// Apply a 45° terminal cut to a rectangular contour.
///
/// `corner` selects which corner to cut:
/// - `TopRight`, `TopLeft`, `BottomRight`, `BottomLeft`
///
/// The cut removes a triangle of size `cut_size` from the specified corner,
/// creating the signature crystal-facet appearance.
#[derive(Clone, Copy, Debug)]
pub enum Corner {
    /// Top-right corner.
    TopRight,
    /// Top-left corner.
    TopLeft,
    /// Bottom-right corner.
    BottomRight,
    /// Bottom-left corner.
    BottomLeft,
}

/// Create a rectangle with a 45° chamfer (crystal-facet cut) on one corner.
///
/// This is the signature Sigil Mono detail: stroke endings are cut at 45°
/// like facets of a crystal.
pub fn chamfered_rect(x: i16, y: i16, w: i16, h: i16, corner: Corner, cut: i16) -> Contour {
    let cut = cut.min(w).min(h);
    let r = x + w;
    let t = y + h;

    match corner {
        Corner::TopRight => Contour::new(vec![
            Point::on(x, y),
            Point::on(r, y),
            Point::on(r, t - cut),
            Point::on(r - cut, t),
            Point::on(x, t),
        ]),
        Corner::TopLeft => Contour::new(vec![
            Point::on(x, y),
            Point::on(r, y),
            Point::on(r, t),
            Point::on(x + cut, t),
            Point::on(x, t - cut),
        ]),
        Corner::BottomRight => Contour::new(vec![
            Point::on(x, y),
            Point::on(r - cut, y),
            Point::on(r, y + cut),
            Point::on(r, t),
            Point::on(x, t),
        ]),
        Corner::BottomLeft => Contour::new(vec![
            Point::on(x + cut, y),
            Point::on(r, y),
            Point::on(r, t),
            Point::on(x, t),
            Point::on(x, y + cut),
        ]),
    }
}

// ── Composite helpers ───────────────────────────────────────────────

/// Merge multiple contour groups into a single flat list.
pub fn merge(groups: Vec<Vec<Contour>>) -> Vec<Contour> {
    groups.into_iter().flatten().collect()
}

/// Convenience: wrap a single contour in a Vec.
pub fn single(c: Contour) -> Vec<Contour> {
    vec![c]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_has_four_points() {
        let r = rect(0, 0, 100, 200);
        assert_eq!(r.points.len(), 4);
        assert!(r.points.iter().all(|p| p.on_curve));
    }

    #[test]
    fn arc_points_full_circle() {
        let pts = arc_points(0.0, 0.0, 100.0, 0.0, 360.0);
        // 4 quadrants × 2 points + 1 start = 9 points
        assert!(pts.len() >= 5, "full circle should have multiple points: got {}", pts.len());
        // First and last should be approximately the same
        let first = pts.first().unwrap();
        let last = pts.last().unwrap();
        assert!((first.x - last.x).abs() <= 1, "full circle should close");
        assert!((first.y - last.y).abs() <= 1, "full circle should close");
    }

    #[test]
    fn diagonal_produces_four_points() {
        let d = diagonal(0.0, 0.0, 100.0, 100.0, 20.0);
        assert_eq!(d.points.len(), 4);
    }

    #[test]
    fn chamfered_rect_has_five_points() {
        let c = chamfered_rect(0, 0, 100, 200, Corner::TopRight, 30);
        assert_eq!(c.points.len(), 5);
    }

    #[test]
    fn contour_reverse() {
        let c = rect(0, 0, 10, 10);
        let r = c.reversed();
        assert_eq!(c.points.len(), r.points.len());
        assert_eq!(c.points[0].x, r.points[3].x);
    }
}
