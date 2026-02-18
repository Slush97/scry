// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shape building helpers: rounded rectangles, arcs, and bounding box estimation.

use tiny_skia::PathBuilder;

use crate::scene::command::DrawCommand;

use super::Rasterizer;

impl Rasterizer {
    /// Estimate the bounding box of a group's child commands.
    ///
    /// Returns `(width, height, origin_x, origin_y)` clamped to the parent
    /// canvas dimensions. Used to allocate a bounded temp pixmap instead of
    /// a full-canvas one for groups with blend modes or opacity.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    pub(crate) fn estimate_group_bounds(
        commands: &[DrawCommand],
        canvas_w: u32,
        canvas_h: u32,
    ) -> (u32, u32, i32, i32) {
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        for cmd in commands {
            let (x0, y0, x1, y1) = Self::estimate_command_bounds(cmd);
            if x0 < min_x {
                min_x = x0;
            }
            if y0 < min_y {
                min_y = y0;
            }
            if x1 > max_x {
                max_x = x1;
            }
            if y1 > max_y {
                max_y = y1;
            }
        }

        // If no commands or no bounds, fall back to full canvas
        if min_x >= max_x || min_y >= max_y {
            return (canvas_w, canvas_h, 0, 0);
        }

        // Add margin for stroke widths and anti-aliasing
        let margin = 8.0;
        min_x = (min_x - margin).max(0.0);
        min_y = (min_y - margin).max(0.0);
        max_x = (max_x + margin).min(canvas_w as f32);
        max_y = (max_y + margin).min(canvas_h as f32);

        let tw = ((max_x - min_x).ceil() as u32).max(1).min(canvas_w);
        let th = ((max_y - min_y).ceil() as u32).max(1).min(canvas_h);
        let origin_col = min_x.floor() as i32;
        let origin_row = min_y.floor() as i32;

        (tw, th, origin_col, origin_row)
    }

    /// Estimate the axis-aligned bounding box of a single draw command.
    /// Returns `(min_x, min_y, max_x, max_y)`.
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn estimate_command_bounds(cmd: &DrawCommand) -> (f32, f32, f32, f32) {
        match cmd {
            DrawCommand::Circle {
                cx,
                cy,
                radius,
                style,
            } => {
                let sw = style.stroke.as_ref().map_or(0.0, |s| s.width);
                let r = radius + sw;
                (cx - r, cy - r, cx + r, cy + r)
            }
            DrawCommand::Rectangle { rect, style, .. } => {
                let sw = style.stroke.as_ref().map_or(0.0, |s| s.width);
                (
                    rect.x - sw,
                    rect.y - sw,
                    rect.x + rect.width + sw,
                    rect.y + rect.height + sw,
                )
            }
            DrawCommand::Ellipse {
                cx,
                cy,
                rx,
                ry,
                style,
                ..
            } => {
                let sw = style.stroke.as_ref().map_or(0.0, |s| s.width);
                let r = rx.max(*ry) + sw;
                (cx - r, cy - r, cx + r, cy + r)
            }
            DrawCommand::Line {
                x1,
                y1,
                x2,
                y2,
                stroke,
                ..
            } => {
                let sw = stroke.width;
                (
                    x1.min(*x2) - sw,
                    y1.min(*y2) - sw,
                    x1.max(*x2) + sw,
                    y1.max(*y2) + sw,
                )
            }
            DrawCommand::Arc {
                cx,
                cy,
                radius,
                style,
                ..
            } => {
                let sw = style.stroke.as_ref().map_or(0.0, |s| s.width);
                let r = radius + sw;
                (cx - r, cy - r, cx + r, cy + r)
            }
            DrawCommand::Polyline { points, style, .. } => {
                let sw = style.stroke.as_ref().map_or(0.0, |s| s.width);
                let (mut min_x, mut min_y) = (f32::MAX, f32::MAX);
                let (mut max_x, mut max_y) = (f32::MIN, f32::MIN);
                for &(x, y) in points {
                    if x < min_x {
                        min_x = x;
                    }
                    if y < min_y {
                        min_y = y;
                    }
                    if x > max_x {
                        max_x = x;
                    }
                    if y > max_y {
                        max_y = y;
                    }
                }
                (min_x - sw, min_y - sw, max_x + sw, max_y + sw)
            }
            DrawCommand::Path { path, style } => {
                let sw = style.stroke.as_ref().map_or(0.0, |s| s.width);
                let b = path.path().bounds();
                (b.left() - sw, b.top() - sw, b.right() + sw, b.bottom() + sw)
            }
            DrawCommand::Gradient { rect, .. } => {
                (rect.x, rect.y, rect.x + rect.width, rect.y + rect.height)
            }
            DrawCommand::Image { image, x, y, .. } => {
                (*x, *y, x + image.width() as f32, y + image.height() as f32)
            }
            #[cfg(feature = "text")]
            DrawCommand::Text {
                x,
                y,
                font_size,
                text,
                ..
            } => {
                // Rough estimate: each character ~0.6 × font_size wide
                let est_w = text.len() as f32 * font_size * 0.6;
                (*x, y - font_size, x + est_w, *y + font_size * 0.3)
            }
            #[cfg(feature = "sdf")]
            DrawCommand::Sdf3D { rect, .. } => {
                (rect.x, rect.y, rect.x + rect.width, rect.y + rect.height)
            }
            DrawCommand::Clear { .. } => (f32::MAX, f32::MAX, f32::MIN, f32::MIN),
            DrawCommand::Group {
                commands: children,
                clip,
                ..
            } => {
                if let Some(crate::scene::style::ClipRegion::Rect(r)) = clip {
                    (r.x, r.y, r.x + r.width, r.y + r.height)
                } else {
                    // Recurse into child commands
                    let mut min_x = f32::MAX;
                    let mut min_y = f32::MAX;
                    let mut max_x = f32::MIN;
                    let mut max_y = f32::MIN;
                    for child in children {
                        let (x0, y0, x1, y1) = Self::estimate_command_bounds(child);
                        if x0 < min_x {
                            min_x = x0;
                        }
                        if y0 < min_y {
                            min_y = y0;
                        }
                        if x1 > max_x {
                            max_x = x1;
                        }
                        if y1 > max_y {
                            max_y = y1;
                        }
                    }
                    (min_x, min_y, max_x, max_y)
                }
            }
        }
    }

    /// Build a rounded rectangle path manually.
    #[allow(clippy::many_single_char_names)]
    pub(super) fn build_round_rect(
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        r: f32,
    ) -> Option<tiny_skia::Path> {
        // Clamp radius to half the smaller dimension
        let r = r.min(w / 2.0).min(h / 2.0);

        let mut pb = PathBuilder::new();

        // Top edge (left to right)
        pb.move_to(x + r, y);
        pb.line_to(x + w - r, y);
        // Top-right corner
        pb.quad_to(x + w, y, x + w, y + r);
        // Right edge
        pb.line_to(x + w, y + h - r);
        // Bottom-right corner
        pb.quad_to(x + w, y + h, x + w - r, y + h);
        // Bottom edge
        pb.line_to(x + r, y + h);
        // Bottom-left corner
        pb.quad_to(x, y + h, x, y + h - r);
        // Left edge
        pb.line_to(x, y + r);
        // Top-left corner
        pb.quad_to(x, y, x + r, y);
        pb.close();

        pb.finish()
    }

    /// Build an arc path using cubic Bézier approximation.
    ///
    /// Splits the arc into segments of ≤90° and uses the standard
    /// `4/3 * tan(θ/4)` control-point formula for each segment.
    #[allow(
        clippy::many_single_char_names,
        clippy::similar_names,
        clippy::suboptimal_flops,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss
    )]
    pub(super) fn build_arc_path(
        cx: f32,
        cy: f32,
        radius: f32,
        start_angle: f32,
        sweep_angle: f32,
    ) -> Option<tiny_skia::Path> {
        if sweep_angle.abs() < f32::EPSILON || radius <= 0.0 {
            return None;
        }

        let mut pb = PathBuilder::new();

        // Start point
        let sx = cx + radius * start_angle.cos();
        let sy = cy + radius * start_angle.sin();
        pb.move_to(sx, sy);

        // Split into segments of at most 90 degrees (π/2 radians)
        let max_segment = std::f32::consts::FRAC_PI_2;
        let segments = ((sweep_angle.abs() / max_segment).ceil() as usize).max(1);
        let segment_angle = sweep_angle / segments as f32;

        let mut angle = start_angle;
        for _ in 0..segments {
            let next_angle = angle + segment_angle;
            let half = segment_angle / 2.0;

            // Control point distance: 4/3 * tan(θ/4) * radius
            let alpha = (4.0 / 3.0) * (half / 2.0).tan();

            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let cos_b = next_angle.cos();
            let sin_b = next_angle.sin();

            // Control point 1 (tangent at start of segment)
            let cp1x = cx + radius * (cos_a - alpha * sin_a);
            let cp1y = cy + radius * (sin_a + alpha * cos_a);
            // Control point 2 (tangent at end of segment)
            let cp2x = cx + radius * (cos_b + alpha * sin_b);
            let cp2y = cy + radius * (sin_b - alpha * cos_b);
            // End point
            let ex = cx + radius * cos_b;
            let ey = cy + radius * sin_b;

            pb.cubic_to(cp1x, cp1y, cp2x, cp2y, ex, ey);
            angle = next_angle;
        }

        pb.finish()
    }
}
