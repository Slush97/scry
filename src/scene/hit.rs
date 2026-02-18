// SPDX-License-Identifier: MIT OR Apache-2.0
//! Hit-testing for scene objects.
//!
//! Provides [`HitTag`], [`HitTester`], and geometry functions for determining
//! which tagged scene objects contain a given point. This enables interactive
//! terminal graphics — map a click position back to the scene object the user
//! intended to interact with.
//!
//! # Example
//!
//! ```
//! use scry_engine::scene::{PixelCanvas, Color};
//! use scry_engine::scene::hit::{HitTag, HitTester, HitTestConfig};
//! use scry_engine::scene::style::Point;
//!
//! let mut canvas = PixelCanvas::new(200, 200);
//! canvas = canvas
//!     .circle(100.0, 100.0, 50.0).fill(Color::RED).done()
//!     .with_tag(HitTag::new(1));
//!
//! let tester = HitTester::new(&canvas, HitTestConfig::default());
//! let hits = tester.test_point(Point::new(100.0, 100.0));
//! assert_eq!(hits.len(), 1);
//! assert_eq!(hits[0].tag_id, 1);
//! ```

use std::collections::HashMap;

use crate::scene::command::DrawCommand;
use crate::scene::style::{ClipRegion, Point, Rect, Transform};
use crate::scene::PixelCanvas;
use crate::transport::backend::FontSize;

// ---------------------------------------------------------------------------
// HitTag — identifier attached to a draw command
// ---------------------------------------------------------------------------

/// A tag identifying a scene object for hit-testing.
///
/// Attach tags to draw commands via [`PixelCanvas::with_tag()`] to make them
/// discoverable by [`HitTester`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct HitTag {
    /// Unique numeric identifier.
    pub id: usize,
    /// Optional human-readable label.
    pub label: Option<String>,
}

impl HitTag {
    /// Create a tag with just a numeric ID.
    #[must_use]
    pub const fn new(id: usize) -> Self {
        Self { id, label: None }
    }

    /// Create a tag with an ID and a label.
    #[must_use]
    pub fn with_label(id: usize, label: impl Into<String>) -> Self {
        Self {
            id,
            label: Some(label.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// HitResult — returned by hit-testing
// ---------------------------------------------------------------------------

/// Result of a hit test against a single tagged object.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct HitResult {
    /// The tag ID of the hit object.
    pub tag_id: usize,
    /// The label, if any, of the hit object.
    pub label: Option<String>,
    /// Index of the command in the display list.
    pub command_index: usize,
}

// ---------------------------------------------------------------------------
// HitTestConfig
// ---------------------------------------------------------------------------

/// Configuration for hit-testing behavior.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct HitTestConfig {
    /// Tolerance (in pixels) for line/stroke proximity tests.
    pub tolerance: f32,
}

impl Default for HitTestConfig {
    fn default() -> Self {
        Self { tolerance: 3.0 }
    }
}

// ---------------------------------------------------------------------------
// HitTester
// ---------------------------------------------------------------------------

/// Performs hit-testing against a tagged [`PixelCanvas`] scene.
///
/// Traverses the display list in reverse order (frontmost first) and tests
/// each tagged command's geometry against the query point.
pub struct HitTester<'a> {
    commands: &'a [DrawCommand],
    tags: &'a HashMap<usize, HitTag>,
    config: HitTestConfig,
}

impl<'a> HitTester<'a> {
    /// Create a hit tester for the given canvas.
    #[must_use]
    pub fn new(canvas: &'a PixelCanvas, config: HitTestConfig) -> Self {
        Self {
            commands: canvas.commands(),
            tags: canvas.hit_tags(),
            config,
        }
    }

    /// Test a pixel-coordinate point against all tagged objects.
    ///
    /// Returns hits in front-to-back order (last drawn = first in results).
    #[must_use]
    pub fn test_point(&self, point: Point) -> Vec<HitResult> {
        let mut results = Vec::new();
        self.test_commands(self.commands, point, Transform::IDENTITY, &mut results, 0);
        results
    }

    /// Test a terminal cell coordinate (col, row) by converting to pixel space.
    ///
    /// The center of the cell is used as the test point.
    #[must_use]
    pub fn test_cell(&self, col: u16, row: u16, font_size: FontSize) -> Vec<HitResult> {
        let px = f32::from(col) * f32::from(font_size.width) + f32::from(font_size.width) / 2.0;
        let py = f32::from(row) * f32::from(font_size.height) + f32::from(font_size.height) / 2.0;
        self.test_point(Point::new(px, py))
    }

    /// Recursively test commands, accumulating transforms for groups.
    fn test_commands(
        &self,
        commands: &[DrawCommand],
        point: Point,
        parent_transform: Transform,
        results: &mut Vec<HitResult>,
        base_index: usize,
    ) {
        // Reverse order: last drawn = frontmost
        for (i, cmd) in commands.iter().enumerate().rev() {
            let global_index = base_index + i;
            let has_tag = self.tags.contains_key(&global_index);

            if let DrawCommand::Group {
                commands: children,
                transform,
                clip,
                ..
            } = cmd
            {
                let combined = parent_transform.concat(*transform);
                // Apply inverse transform to test point
                let Some(inv) = combined.inverse() else {
                    continue;
                };
                let local_point = inv.apply_point(point);

                // Check clip region
                if let Some(clip) = clip {
                    if !point_in_clip(local_point, clip) {
                        continue;
                    }
                }

                self.test_commands(children, point, combined, results, global_index + 1);
            } else {
                if !has_tag {
                    continue;
                }

                // Apply parent inverse transform
                let test_point = if parent_transform == Transform::IDENTITY {
                    point
                } else {
                    let Some(inv) = parent_transform.inverse() else {
                        continue;
                    };
                    inv.apply_point(point)
                };

                if hit_test_command(cmd, test_point, &self.config) {
                    let tag = &self.tags[&global_index];
                    results.push(HitResult {
                        tag_id: tag.id,
                        label: tag.label.clone(),
                        command_index: global_index,
                    });
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Geometry hit-test functions
// ---------------------------------------------------------------------------

/// Test whether a point is inside a clip region.
fn point_in_clip(point: Point, clip: &ClipRegion) -> bool {
    match clip {
        ClipRegion::Rect(r) => r.contains(point),
        ClipRegion::Path(_) => {
            // Path clipping: conservative approximation (always pass)
            // Full path clipping requires flattening — acceptable trade-off.
            true
        }
    }
}

/// Test whether a point hits a specific draw command.
fn hit_test_command(cmd: &DrawCommand, point: Point, config: &HitTestConfig) -> bool {
    match cmd {
        DrawCommand::Circle {
            cx,
            cy,
            radius,
            style,
        } => {
            let dx = point.x - cx;
            let dy = point.y - cy;
            let dist_sq = dx * dx + dy * dy;

            // Test fill
            if style.fill.is_some() && dist_sq <= radius * radius {
                return true;
            }
            // Test stroke ring
            if let Some(ref stroke) = style.stroke {
                let half_w = stroke.width / 2.0 + config.tolerance;
                let outer = radius + half_w;
                let inner = (radius - half_w).max(0.0);
                dist_sq <= outer * outer && dist_sq >= inner * inner
            } else {
                false
            }
        }

        DrawCommand::Rectangle { rect, style, .. } => {
            // Test fill
            if style.fill.is_some() && rect.contains(point) {
                return true;
            }
            // Test stroke border
            if let Some(ref stroke) = style.stroke {
                let tol = stroke.width / 2.0 + config.tolerance;
                let outer = Rect::new(
                    rect.x - tol,
                    rect.y - tol,
                    rect.width + tol * 2.0,
                    rect.height + tol * 2.0,
                );
                let inner = Rect::new(
                    rect.x + tol,
                    rect.y + tol,
                    (rect.width - tol * 2.0).max(0.0),
                    (rect.height - tol * 2.0).max(0.0),
                );
                outer.contains(point) && !inner.contains(point)
            } else {
                false
            }
        }

        DrawCommand::Ellipse {
            cx,
            cy,
            rx,
            ry,
            rotation,
            style,
        } => {
            // Inverse-rotate point around center
            let (sin, cos) = (-rotation).sin_cos();
            let dx = point.x - cx;
            let dy = point.y - cy;
            let lx = cos * dx - sin * dy;
            let ly = sin * dx + cos * dy;

            let norm = if *rx > 0.0 && *ry > 0.0 {
                (lx / rx) * (lx / rx) + (ly / ry) * (ly / ry)
            } else {
                f32::INFINITY
            };

            if style.fill.is_some() && norm <= 1.0 {
                return true;
            }
            if let Some(ref stroke) = style.stroke {
                let tol = stroke.width / (2.0 * rx.min(*ry)) + config.tolerance / rx.min(*ry);
                norm <= (1.0 + tol) * (1.0 + tol) && norm >= (1.0 - tol).max(0.0).powi(2)
            } else {
                false
            }
        }

        DrawCommand::Line {
            x1,
            y1,
            x2,
            y2,
            stroke,
            ..
        } => {
            let tol = stroke.width / 2.0 + config.tolerance;
            point_to_segment_dist_sq(point, Point::new(*x1, *y1), Point::new(*x2, *y2)) <= tol * tol
        }

        DrawCommand::Polyline {
            points,
            closed,
            style,
        } => {
            // Check fill (closed polygon)
            if *closed
                && style.fill.is_some()
                && points.len() >= 3
                && winding_number_test(point, points)
            {
                return true;
            }
            // Check stroke (edge proximity)
            if let Some(ref stroke) = style.stroke {
                let tol = stroke.width / 2.0 + config.tolerance;
                let tol_sq = tol * tol;
                for pair in points.windows(2) {
                    let a = Point::new(pair[0].0, pair[0].1);
                    let b = Point::new(pair[1].0, pair[1].1);
                    if point_to_segment_dist_sq(point, a, b) <= tol_sq {
                        return true;
                    }
                }
                if *closed && points.len() >= 2 {
                    let a = Point::new(points.last().unwrap().0, points.last().unwrap().1);
                    let b = Point::new(points[0].0, points[0].1);
                    if point_to_segment_dist_sq(point, a, b) <= tol_sq {
                        return true;
                    }
                }
            }
            false
        }

        DrawCommand::Arc {
            cx,
            cy,
            radius,
            start_angle,
            sweep_angle,
            style,
        } => {
            let dx = point.x - cx;
            let dy = point.y - cy;
            let dist_sq = dx * dx + dy * dy;
            let angle = dy.atan2(dx);

            // Normalize angle to be within sweep
            let in_sweep = angle_in_sweep(angle, *start_angle, *sweep_angle);

            if style.fill.is_some() {
                // Pie-slice test: within radius AND within angle sweep
                if dist_sq <= radius * radius && in_sweep {
                    return true;
                }
            }
            if let Some(ref stroke) = style.stroke {
                let tol = stroke.width / 2.0 + config.tolerance;
                let outer = radius + tol;
                let inner = (radius - tol).max(0.0);
                dist_sq <= outer * outer && dist_sq >= inner * inner && in_sweep
            } else {
                false
            }
        }

        DrawCommand::Image { image, x, y, .. } => {
            // AABB test
            let rect = Rect::new(*x, *y, image.width() as f32, image.height() as f32);
            rect.contains(point)
        }

        DrawCommand::Gradient { rect, .. } => rect.contains(point),

        DrawCommand::Clear { .. } => {
            // Clear fills entire canvas; no meaningful hit test
            false
        }

        DrawCommand::Path { style, .. } => {
            // Path hit-testing: approximate with fill=always-hit for filled paths
            // Full winding-number test on flattened Bézier would be ideal but
            // is complex; for now, this is a best-effort stub.
            style.fill.is_some()
        }

        #[cfg(feature = "text")]
        DrawCommand::Text {
            x,
            y,
            font_size,
            text,
            ..
        } => {
            // Bounding-box approximation: width ≈ chars × font_size × 0.6
            let approx_width = text.len() as f32 * font_size * 0.6;
            let rect = Rect::new(*x, *y - font_size, approx_width, *font_size);
            rect.contains(point)
        }

        // Forward-compatible: unknown variants don't hit
        _ => false,
    }
}

/// Squared distance from a point to a line segment.
fn point_to_segment_dist_sq(p: Point, a: Point, b: Point) -> f32 {
    let ab_x = b.x - a.x;
    let ab_y = b.y - a.y;
    let ap_x = p.x - a.x;
    let ap_y = p.y - a.y;

    let ab_len_sq = ab_x * ab_x + ab_y * ab_y;
    if ab_len_sq < 1e-10 {
        return ap_x * ap_x + ap_y * ap_y;
    }

    let t = ((ap_x * ab_x + ap_y * ab_y) / ab_len_sq).clamp(0.0, 1.0);
    let proj_x = a.x + t * ab_x;
    let proj_y = a.y + t * ab_y;
    let dx = p.x - proj_x;
    let dy = p.y - proj_y;
    dx * dx + dy * dy
}

/// Winding number test for point-in-polygon.
fn winding_number_test(point: Point, vertices: &[(f32, f32)]) -> bool {
    let mut winding = 0i32;
    let n = vertices.len();
    for i in 0..n {
        let (y0, y1) = (vertices[i].1, vertices[(i + 1) % n].1);
        let (x0, x1) = (vertices[i].0, vertices[(i + 1) % n].0);
        if y0 <= point.y {
            if y1 > point.y && cross_2d(x0, y0, x1, y1, point.x, point.y) > 0.0 {
                winding += 1;
            }
        } else if y1 <= point.y && cross_2d(x0, y0, x1, y1, point.x, point.y) < 0.0 {
            winding -= 1;
        }
    }
    winding != 0
}

/// 2D cross product helper for winding number.
fn cross_2d(x0: f32, y0: f32, x1: f32, y1: f32, px: f32, py: f32) -> f32 {
    (x1 - x0) * (py - y0) - (px - x0) * (y1 - y0)
}

/// Check whether an angle falls within an arc's sweep.
fn angle_in_sweep(angle: f32, start: f32, sweep: f32) -> bool {
    let two_pi = std::f32::consts::TAU;

    // Normalize to [0, 2π)
    let normalize = |a: f32| ((a % two_pi) + two_pi) % two_pi;
    let norm_angle = normalize(angle);
    let norm_start = normalize(start);

    if sweep.abs() >= two_pi {
        return true;
    }

    if sweep >= 0.0 {
        let end = normalize(start + sweep);
        if norm_start <= end {
            norm_angle >= norm_start && norm_angle <= end
        } else {
            norm_angle >= norm_start || norm_angle <= end
        }
    } else {
        let end = normalize(start + sweep);
        if end <= norm_start {
            norm_angle <= norm_start && norm_angle >= end
        } else {
            norm_angle <= norm_start || norm_angle >= end
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::style::Color;

    #[test]
    fn hit_test_circle_center() {
        let mut canvas = PixelCanvas::new(200, 200);
        canvas = canvas
            .circle(100.0, 100.0, 50.0)
            .fill(Color::RED)
            .done()
            .with_tag(HitTag::new(1));

        let tester = HitTester::new(&canvas, HitTestConfig::default());
        let hits = tester.test_point(Point::new(100.0, 100.0));
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].tag_id, 1);
    }

    #[test]
    fn hit_test_circle_miss() {
        let mut canvas = PixelCanvas::new(200, 200);
        canvas = canvas
            .circle(100.0, 100.0, 50.0)
            .fill(Color::RED)
            .done()
            .with_tag(HitTag::new(1));

        let tester = HitTester::new(&canvas, HitTestConfig::default());
        let hits = tester.test_point(Point::new(0.0, 0.0));
        assert!(hits.is_empty());
    }

    #[test]
    fn hit_test_rectangle() {
        let mut canvas = PixelCanvas::new(200, 200);
        canvas = canvas
            .rect(10.0, 10.0, 80.0, 40.0)
            .fill(Color::BLUE)
            .done()
            .with_tag(HitTag::new(42));

        let tester = HitTester::new(&canvas, HitTestConfig::default());
        let hits = tester.test_point(Point::new(50.0, 30.0));
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].tag_id, 42);
    }

    #[test]
    fn hit_test_line_proximity() {
        let mut canvas = PixelCanvas::new(200, 200);
        canvas = canvas
            .line(0.0, 0.0, 200.0, 200.0)
            .color(Color::WHITE)
            .width(2.0)
            .done()
            .with_tag(HitTag::new(7));

        let tester = HitTester::new(&canvas, HitTestConfig::default());
        // Point near the diagonal
        let hits = tester.test_point(Point::new(100.0, 101.0));
        assert_eq!(hits.len(), 1);

        // Point far from the line
        let hits = tester.test_point(Point::new(0.0, 200.0));
        assert!(hits.is_empty());
    }

    #[test]
    fn hit_test_untagged_ignored() {
        let canvas = PixelCanvas::new(200, 200)
            .circle(100.0, 100.0, 50.0)
            .fill(Color::RED)
            .done();
        // No tag applied

        let tester = HitTester::new(&canvas, HitTestConfig::default());
        let hits = tester.test_point(Point::new(100.0, 100.0));
        assert!(hits.is_empty());
    }

    #[test]
    fn hit_test_frontmost_first() {
        let mut canvas = PixelCanvas::new(200, 200);
        canvas = canvas
            .circle(100.0, 100.0, 80.0)
            .fill(Color::RED)
            .done()
            .with_tag(HitTag::new(1));
        canvas = canvas
            .circle(100.0, 100.0, 30.0)
            .fill(Color::BLUE)
            .done()
            .with_tag(HitTag::new(2));

        let tester = HitTester::new(&canvas, HitTestConfig::default());
        let hits = tester.test_point(Point::new(100.0, 100.0));
        // Both should hit, but frontmost (tag 2) should be first
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].tag_id, 2);
        assert_eq!(hits[1].tag_id, 1);
    }

    #[test]
    fn hit_test_polygon_winding() {
        let mut canvas = PixelCanvas::new(200, 200);
        // Triangle
        canvas = canvas
            .polygon(vec![(50.0, 10.0), (90.0, 90.0), (10.0, 90.0)])
            .fill(Color::GREEN)
            .done()
            .with_tag(HitTag::new(3));

        let tester = HitTester::new(&canvas, HitTestConfig::default());
        let hits = tester.test_point(Point::new(50.0, 50.0));
        assert_eq!(hits.len(), 1);

        let hits = tester.test_point(Point::new(5.0, 5.0));
        assert!(hits.is_empty());
    }

    #[test]
    fn hit_tag_with_label() {
        let tag = HitTag::with_label(5, "my-button");
        assert_eq!(tag.id, 5);
        assert_eq!(tag.label.as_deref(), Some("my-button"));
    }

    #[test]
    fn point_to_segment_distance() {
        let a = Point::new(0.0, 0.0);
        let b = Point::new(10.0, 0.0);
        let p = Point::new(5.0, 3.0);
        let d2 = point_to_segment_dist_sq(p, a, b);
        assert!((d2 - 9.0).abs() < 0.01);
    }

    #[test]
    fn winding_number_square() {
        let square = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        assert!(winding_number_test(Point::new(5.0, 5.0), &square));
        assert!(!winding_number_test(Point::new(15.0, 5.0), &square));
    }

    #[test]
    fn test_cell_conversion() {
        let mut canvas = PixelCanvas::new(200, 200);
        canvas = canvas
            .rect(0.0, 0.0, 200.0, 200.0)
            .fill(Color::RED)
            .done()
            .with_tag(HitTag::new(1));

        let tester = HitTester::new(&canvas, HitTestConfig::default());
        let hits = tester.test_cell(1, 1, FontSize::new(8, 16));
        assert_eq!(hits.len(), 1);
    }
}
