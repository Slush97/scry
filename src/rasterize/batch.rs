// SPDX-License-Identifier: MIT OR Apache-2.0
//! Command batching for reduced `fill_path`/`stroke_path` call overhead.
//!
//! When multiple consecutive commands share the same [`ShapeStyle`] and are
//! rendered with the same transform, their geometries can be merged into a
//! single compound `tiny_skia::Path`. This reduces per-call overhead in
//! `tiny-skia` (paint setup, scanline iterator init, clip checks) by a factor
//! equal to the batch size.
//!
//! # How it works
//!
//! 1. The batcher scans the command list for runs of batchable commands
//!    (Circle, Rectangle, Arc, Ellipse, Polyline, Path) with identical styles.
//! 2. Each run is merged into a single `BatchedOp::Compound` containing a
//!    pre-built `tiny_skia::Path` with all sub-paths.
//! 3. Non-batchable commands (Gradient, Group, Line, Clear, Image, Text) pass
//!    through as `BatchedOp::Single`.
//!
//! The batcher is O(n) and allocation-free for single-command "batches".

use tiny_skia::PathBuilder;

use crate::scene::command::DrawCommand;
use crate::scene::style::ShapeStyle;

// ---------------------------------------------------------------------------
// Batched operation
// ---------------------------------------------------------------------------

/// A rendering operation that may contain multiple merged commands.
pub(crate) enum BatchedOp<'a> {
    /// A single unbatched command (pass-through).
    Single(&'a DrawCommand),
    /// Multiple same-style shape commands merged into one compound path.
    Compound {
        path: tiny_skia::Path,
        style: &'a ShapeStyle,
    },
}

// ---------------------------------------------------------------------------
// Path building helpers
// ---------------------------------------------------------------------------

/// Append a circle sub-path to the builder.
fn append_circle(pb: &mut PathBuilder, cx: f32, cy: f32, radius: f32) {
    // 4-conic approximation of a circle (same as tiny_skia::PathBuilder::from_circle)
    let kappa = 0.552_284_8; // 4/3 * (sqrt(2) - 1)
    let k = radius * kappa;

    pb.move_to(cx + radius, cy);
    pb.cubic_to(cx + radius, cy + k, cx + k, cy + radius, cx, cy + radius);
    pb.cubic_to(cx - k, cy + radius, cx - radius, cy + k, cx - radius, cy);
    pb.cubic_to(cx - radius, cy - k, cx - k, cy - radius, cx, cy - radius);
    pb.cubic_to(cx + k, cy - radius, cx + radius, cy - k, cx + radius, cy);
    pb.close();
}

/// Append a rectangle sub-path to the builder.
fn append_rect(pb: &mut PathBuilder, x: f32, y: f32, w: f32, h: f32) {
    pb.move_to(x, y);
    pb.line_to(x + w, y);
    pb.line_to(x + w, y + h);
    pb.line_to(x, y + h);
    pb.close();
}

/// Append an arc sub-path to the builder.
///
/// Approximates the arc with cubic Bézier segments (one per 90° sweep).
#[allow(clippy::similar_names)]
fn append_arc(pb: &mut PathBuilder, cx: f32, cy: f32, radius: f32, start: f32, sweep: f32) {
    let n_segs = ((sweep.abs() / std::f32::consts::FRAC_PI_2).ceil() as usize).max(1);
    let seg_sweep = sweep / n_segs as f32;

    let mut angle = start;
    let (sin0, cos0) = angle.sin_cos();
    pb.move_to(cos0.mul_add(radius, cx), sin0.mul_add(radius, cy));

    for _ in 0..n_segs {
        let end_angle = angle + seg_sweep;
        let half = seg_sweep / 2.0;
        let alpha = (4.0 * half.tan() / 3.0).abs() * sweep.signum();

        let (sin_a, cos_a) = angle.sin_cos();
        let (sin_e, cos_e) = end_angle.sin_cos();

        let cp1x = (-sin_a).mul_add(alpha * radius, cos_a.mul_add(radius, cx));
        let cp1y = cos_a.mul_add(alpha * radius, sin_a.mul_add(radius, cy));
        let cp2x = sin_e.mul_add(alpha * radius, cos_e.mul_add(radius, cx));
        let cp2y = (-cos_e).mul_add(alpha * radius, sin_e.mul_add(radius, cy));
        let ex = cos_e.mul_add(radius, cx);
        let ey = sin_e.mul_add(radius, cy);

        pb.cubic_to(cp1x, cp1y, cp2x, cp2y, ex, ey);
        angle = end_angle;
    }
}

/// Append a round-rect sub-path to the builder.
#[allow(clippy::similar_names, clippy::many_single_char_names)]
fn append_round_rect(pb: &mut PathBuilder, x: f32, y: f32, w: f32, h: f32, cr: f32) {
    let r = cr.min(w / 2.0).min(h / 2.0);
    let kappa = 0.552_284_8_f32;
    let k = r * kappa;

    pb.move_to(x + r, y);
    pb.line_to(x + w - r, y);
    pb.cubic_to(x + w - r + k, y, x + w, y + r - k, x + w, y + r);
    pb.line_to(x + w, y + h - r);
    pb.cubic_to(x + w, y + h - r + k, x + w - r + k, y + h, x + w - r, y + h);
    pb.line_to(x + r, y + h);
    pb.cubic_to(x + r - k, y + h, x, y + h - r + k, x, y + h - r);
    pb.line_to(x, y + r);
    pb.cubic_to(x, y + r - k, x + r - k, y, x + r, y);
    pb.close();
}

// ---------------------------------------------------------------------------
// Core batching logic
// ---------------------------------------------------------------------------

/// Extract the [`ShapeStyle`] from a batchable command, or `None` for
/// commands that cannot be batched (Gradient, Group, Clear, Line, Image, Text).
///
/// Lines are excluded because they use a different render path (stroke-only,
/// no fill, separate anti-alias flag).
const fn batchable_style(cmd: &DrawCommand) -> Option<&ShapeStyle> {
    match cmd {
        DrawCommand::Circle { style, .. }
        | DrawCommand::Rectangle { style, .. }
        | DrawCommand::Ellipse { style, .. }
        | DrawCommand::Arc { style, .. }
        | DrawCommand::Polyline { style, .. }
        | DrawCommand::Path { style, .. } => Some(style),
        // Not batchable: different render paths
        DrawCommand::Gradient { .. }
        | DrawCommand::Group { .. }
        | DrawCommand::Clear { .. }
        | DrawCommand::Line { .. }
        | DrawCommand::Image { .. } => None,
        #[cfg(feature = "text")]
        DrawCommand::Text { .. } => None,
    }
}

/// Append a single command's geometry to the compound path builder.
///
/// Returns `true` if the geometry was successfully appended.
fn append_command(pb: &mut PathBuilder, cmd: &DrawCommand) -> bool {
    match cmd {
        DrawCommand::Circle { cx, cy, radius, .. } => {
            append_circle(pb, *cx, *cy, *radius);
            true
        }
        DrawCommand::Rectangle {
            rect,
            corner_radius,
            ..
        } => {
            if *corner_radius > 0.0 {
                append_round_rect(pb, rect.x, rect.y, rect.width, rect.height, *corner_radius);
            } else {
                append_rect(pb, rect.x, rect.y, rect.width, rect.height);
            }
            true
        }
        DrawCommand::Ellipse {
            cx,
            cy,
            rx,
            ry,
            rotation,
            ..
        } => {
            // Only batch non-rotated ellipses; rotated ones need per-command transforms
            if rotation.abs() <= f32::EPSILON {
                // Use kappa approximation for ellipse (same approach as circle)
                let kappa = 0.552_284_8_f32;
                let kx = *rx * kappa;
                let ky = *ry * kappa;
                pb.move_to(cx + rx, *cy);
                pb.cubic_to(cx + rx, cy + ky, cx + kx, cy + ry, *cx, cy + ry);
                pb.cubic_to(cx - kx, cy + ry, cx - rx, cy + ky, cx - rx, *cy);
                pb.cubic_to(cx - rx, cy - ky, cx - kx, cy - ry, *cx, cy - ry);
                pb.cubic_to(cx + kx, cy - ry, cx + rx, cy - ky, cx + rx, *cy);
                pb.close();
                true
            } else {
                false // Rotated ellipses can't be batched
            }
        }
        DrawCommand::Arc {
            cx,
            cy,
            radius,
            start_angle,
            sweep_angle,
            ..
        } => {
            append_arc(pb, *cx, *cy, *radius, *start_angle, *sweep_angle);
            true
        }
        DrawCommand::Polyline { points, closed, .. } => {
            if points.len() >= 2 {
                pb.move_to(points[0].0, points[0].1);
                for &(x, y) in &points[1..] {
                    pb.line_to(x, y);
                }
                if *closed {
                    pb.close();
                }
                true
            } else {
                false
            }
        }
        DrawCommand::Path { path, .. } => {
            // Copy path segments into the compound builder using the
            // public PathSegment iterator (PathVerb is not re-exported).
            let src = path.path();
            for segment in src.segments() {
                match segment {
                    tiny_skia::PathSegment::MoveTo(pt) => {
                        pb.move_to(pt.x, pt.y);
                    }
                    tiny_skia::PathSegment::LineTo(pt) => {
                        pb.line_to(pt.x, pt.y);
                    }
                    tiny_skia::PathSegment::QuadTo(p1, p2) => {
                        pb.quad_to(p1.x, p1.y, p2.x, p2.y);
                    }
                    tiny_skia::PathSegment::CubicTo(p1, p2, p3) => {
                        pb.cubic_to(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
                    }
                    tiny_skia::PathSegment::Close => {
                        pb.close();
                    }
                }
            }
            true
        }
        _ => false,
    }
}

/// Batch consecutive same-style commands into compound paths.
///
/// Returns a `Vec<BatchedOp>` where runs of ≥2 consecutive batchable commands
/// with identical `ShapeStyle` are merged into `BatchedOp::Compound`.
/// Single commands and non-batchable commands pass through as `BatchedOp::Single`.
///
/// This is O(n) in the number of commands and performs no heap allocation for
/// runs of length 1.
pub(crate) fn batch_commands(commands: &[DrawCommand]) -> Vec<BatchedOp<'_>> {
    let mut result = Vec::with_capacity(commands.len());
    let mut i = 0;

    while i < commands.len() {
        let Some(style) = batchable_style(&commands[i]) else {
            result.push(BatchedOp::Single(&commands[i]));
            i += 1;
            continue;
        };

        // Scan for consecutive commands with the same style
        let run_start = i;
        i += 1;
        while i < commands.len() {
            if let Some(next_style) = batchable_style(&commands[i]) {
                if next_style == style {
                    i += 1;
                    continue;
                }
            }
            break;
        }

        let run_len = i - run_start;

        if run_len == 1 {
            // Single command — no batching overhead
            result.push(BatchedOp::Single(&commands[run_start]));
        } else {
            // Build compound path from all commands in the run
            let mut pb = PathBuilder::new();
            let mut any_appended = false;

            for cmd in &commands[run_start..i] {
                if append_command(&mut pb, cmd) {
                    any_appended = true;
                }
            }

            if any_appended {
                if let Some(path) = pb.finish() {
                    result.push(BatchedOp::Compound { path, style });
                } else {
                    // Fallback: emit individually
                    for cmd in &commands[run_start..i] {
                        result.push(BatchedOp::Single(cmd));
                    }
                }
            } else {
                for cmd in &commands[run_start..i] {
                    result.push(BatchedOp::Single(cmd));
                }
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::style::{Color, FillStyle, Rect, ShapeStyle};

    fn red_fill() -> ShapeStyle {
        ShapeStyle {
            fill: Some(FillStyle::Solid(Color::RED)),
            stroke: None,
            anti_alias: true,
        }
    }

    fn blue_fill() -> ShapeStyle {
        ShapeStyle {
            fill: Some(FillStyle::Solid(Color::BLUE)),
            stroke: None,
            anti_alias: true,
        }
    }

    #[test]
    fn single_commands_not_batched() {
        let commands = vec![DrawCommand::Circle {
            cx: 10.0,
            cy: 10.0,
            radius: 5.0,
            style: red_fill(),
        }];
        let batched = batch_commands(&commands);
        assert_eq!(batched.len(), 1);
        assert!(matches!(batched[0], BatchedOp::Single(_)));
    }

    #[test]
    fn consecutive_same_style_batched() {
        let commands = vec![
            DrawCommand::Circle {
                cx: 10.0,
                cy: 10.0,
                radius: 5.0,
                style: red_fill(),
            },
            DrawCommand::Circle {
                cx: 20.0,
                cy: 20.0,
                radius: 5.0,
                style: red_fill(),
            },
            DrawCommand::Circle {
                cx: 30.0,
                cy: 30.0,
                radius: 5.0,
                style: red_fill(),
            },
        ];
        let batched = batch_commands(&commands);
        assert_eq!(batched.len(), 1);
        assert!(matches!(batched[0], BatchedOp::Compound { .. }));
    }

    #[test]
    fn different_styles_not_batched() {
        let commands = vec![
            DrawCommand::Circle {
                cx: 10.0,
                cy: 10.0,
                radius: 5.0,
                style: red_fill(),
            },
            DrawCommand::Circle {
                cx: 20.0,
                cy: 20.0,
                radius: 5.0,
                style: blue_fill(),
            },
        ];
        let batched = batch_commands(&commands);
        assert_eq!(batched.len(), 2);
        assert!(matches!(batched[0], BatchedOp::Single(_)));
        assert!(matches!(batched[1], BatchedOp::Single(_)));
    }

    #[test]
    fn mixed_batchable_and_unbatchable() {
        let style = red_fill();
        let commands = vec![
            DrawCommand::Circle {
                cx: 10.0,
                cy: 10.0,
                radius: 5.0,
                style: style.clone(),
            },
            DrawCommand::Circle {
                cx: 20.0,
                cy: 20.0,
                radius: 5.0,
                style: style.clone(),
            },
            DrawCommand::Clear {
                color: Color::BLACK,
            },
            DrawCommand::Circle {
                cx: 30.0,
                cy: 30.0,
                radius: 5.0,
                style,
            },
        ];
        let batched = batch_commands(&commands);
        assert_eq!(batched.len(), 3); // Compound(2 circles), Single(clear), Single(1 circle)
        assert!(matches!(batched[0], BatchedOp::Compound { .. }));
        assert!(matches!(batched[1], BatchedOp::Single(_)));
        assert!(matches!(batched[2], BatchedOp::Single(_)));
    }

    #[test]
    fn mixed_shape_types_same_style_batched() {
        let style = red_fill();
        let commands = vec![
            DrawCommand::Circle {
                cx: 10.0,
                cy: 10.0,
                radius: 5.0,
                style: style.clone(),
            },
            DrawCommand::Rectangle {
                rect: Rect::new(0.0, 0.0, 20.0, 20.0),
                corner_radius: 0.0,
                style: style.clone(),
            },
            DrawCommand::Arc {
                cx: 50.0,
                cy: 50.0,
                radius: 20.0,
                start_angle: 0.0,
                sweep_angle: std::f32::consts::PI,
                style,
            },
        ];
        let batched = batch_commands(&commands);
        assert_eq!(batched.len(), 1);
        assert!(matches!(batched[0], BatchedOp::Compound { .. }));
    }

    #[test]
    fn empty_commands() {
        let batched = batch_commands(&[]);
        assert!(batched.is_empty());
    }
}
