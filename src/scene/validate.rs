// SPDX-License-Identifier: MIT OR Apache-2.0
//! Pre-rasterization scene validation.
//!
//! [`validate_scene`] inspects the [`DrawCommand`] list in a [`PixelCanvas`]
//! and reports potential issues (degenerate geometry, out-of-bounds coordinates,
//! invisible draws) **without aborting**.  This lets callers decide how to
//! handle warnings: log them, surface them in a diagnostic UI, or ignore them.
//!
//! # Example
//!
//! ```
//! use scry_engine::scene::{PixelCanvas, Color};
//! use scry_engine::scene::validate::{validate_scene, WarningSeverity};
//!
//! let canvas = PixelCanvas::new(100, 100)
//!     .circle(50.0, 50.0, 0.0) // zero radius → warning
//!         .done();
//!
//! let warnings = validate_scene(&canvas);
//! assert!(!warnings.is_empty());
//! assert!(warnings[0].severity == WarningSeverity::Warning);
//! ```

use crate::scene::command::DrawCommand;
use crate::scene::PixelCanvas;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Severity level for a scene warning.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WarningSeverity {
    /// Informational diagnostic (e.g., element partially out of bounds).
    Info,
    /// The command will produce no visible output or has suspect parameters.
    Warning,
    /// The command is structurally invalid and will be skipped by the rasterizer.
    Error,
}

/// A diagnostic warning for a single draw command.
#[derive(Clone, Debug)]
pub struct SceneWarning {
    /// Zero-based index of the command in the display list.
    pub command_index: usize,
    /// How serious this issue is.
    pub severity: WarningSeverity,
    /// Human-readable description of the problem.
    pub message: &'static str,
}

impl std::fmt::Display for SceneWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let sev = match self.severity {
            WarningSeverity::Info => "info",
            WarningSeverity::Warning => "warn",
            WarningSeverity::Error => "error",
        };
        write!(f, "[{}] cmd[{}]: {}", sev, self.command_index, self.message)
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validate every command in a canvas and return any diagnostic warnings.
///
/// This runs in O(n) over the command list with no allocations beyond the
/// returned `Vec`.  Warnings are non-fatal — a scene with warnings can
/// still be rasterized (degenerate commands will simply be skipped).
pub fn validate_scene(canvas: &PixelCanvas) -> Vec<SceneWarning> {
    let mut warnings = Vec::new();
    let cw = canvas.width() as f32;
    let ch = canvas.height() as f32;

    for (i, cmd) in canvas.commands().iter().enumerate() {
        validate_command(cmd, i, cw, ch, &mut warnings);
    }

    warnings
}

fn validate_command(
    cmd: &DrawCommand,
    index: usize,
    canvas_w: f32,
    canvas_h: f32,
    out: &mut Vec<SceneWarning>,
) {
    match cmd {
        DrawCommand::Circle {
            cx, cy, radius, style,
        } => {
            if *radius <= 0.0 {
                out.push(SceneWarning {
                    command_index: index,
                    severity: WarningSeverity::Warning,
                    message: "Circle: zero or negative radius (will be skipped)",
                });
            }
            if style.fill.is_none() && style.stroke.is_none() {
                out.push(SceneWarning {
                    command_index: index,
                    severity: WarningSeverity::Warning,
                    message: "Circle: no fill and no stroke (invisible)",
                });
            }
            if *cx + *radius < 0.0
                || *cx - *radius > canvas_w
                || *cy + *radius < 0.0
                || *cy - *radius > canvas_h
            {
                out.push(SceneWarning {
                    command_index: index,
                    severity: WarningSeverity::Info,
                    message: "Circle: entirely outside canvas bounds",
                });
            }
        }

        DrawCommand::Rectangle {
            rect, style, ..
        } => {
            if rect.width <= 0.0 || rect.height <= 0.0 {
                out.push(SceneWarning {
                    command_index: index,
                    severity: WarningSeverity::Warning,
                    message: "Rectangle: zero or negative dimensions (will be skipped)",
                });
            }
            if style.fill.is_none() && style.stroke.is_none() {
                out.push(SceneWarning {
                    command_index: index,
                    severity: WarningSeverity::Warning,
                    message: "Rectangle: no fill and no stroke (invisible)",
                });
            }
            if rect.x > canvas_w || rect.y > canvas_h
                || rect.x + rect.width < 0.0
                || rect.y + rect.height < 0.0
            {
                out.push(SceneWarning {
                    command_index: index,
                    severity: WarningSeverity::Info,
                    message: "Rectangle: entirely outside canvas bounds",
                });
            }
        }

        DrawCommand::Ellipse {
            cx: _, cy: _, rx, ry, style, ..
        } => {
            if *rx <= 0.0 || *ry <= 0.0 {
                out.push(SceneWarning {
                    command_index: index,
                    severity: WarningSeverity::Warning,
                    message: "Ellipse: zero or negative radii (will be skipped)",
                });
            }
            if style.fill.is_none() && style.stroke.is_none() {
                out.push(SceneWarning {
                    command_index: index,
                    severity: WarningSeverity::Warning,
                    message: "Ellipse: no fill and no stroke (invisible)",
                });
            }
        }

        DrawCommand::Line { stroke, .. } => {
            if stroke.width <= 0.0 {
                out.push(SceneWarning {
                    command_index: index,
                    severity: WarningSeverity::Warning,
                    message: "Line: zero or negative stroke width (invisible)",
                });
            }
        }

        DrawCommand::Polyline {
            points, style, ..
        } => {
            if points.len() < 2 {
                out.push(SceneWarning {
                    command_index: index,
                    severity: WarningSeverity::Error,
                    message: "Polyline: fewer than 2 points (will be skipped)",
                });
            }
            if style.fill.is_none() && style.stroke.is_none() {
                out.push(SceneWarning {
                    command_index: index,
                    severity: WarningSeverity::Warning,
                    message: "Polyline: no fill and no stroke (invisible)",
                });
            }
        }

        DrawCommand::Arc {
            radius, style, ..
        } => {
            if *radius <= 0.0 {
                out.push(SceneWarning {
                    command_index: index,
                    severity: WarningSeverity::Warning,
                    message: "Arc: zero or negative radius (will be skipped)",
                });
            }
            if style.fill.is_none() && style.stroke.is_none() {
                out.push(SceneWarning {
                    command_index: index,
                    severity: WarningSeverity::Warning,
                    message: "Arc: no fill and no stroke (invisible)",
                });
            }
        }

        DrawCommand::Gradient { rect, gradient, .. } => {
            if rect.width <= 0.0 || rect.height <= 0.0 {
                out.push(SceneWarning {
                    command_index: index,
                    severity: WarningSeverity::Warning,
                    message: "Gradient: zero or negative rect dimensions",
                });
            }
            // Check gradient stops are in ascending order
            let stops = &gradient.stops;
            for window in stops.windows(2) {
                if window[1].position < window[0].position {
                    out.push(SceneWarning {
                        command_index: index,
                        severity: WarningSeverity::Warning,
                        message: "Gradient: stops not in ascending offset order",
                    });
                    break;
                }
            }
        }

        DrawCommand::Group {
            commands, opacity, ..
        } => {
            if commands.is_empty() {
                out.push(SceneWarning {
                    command_index: index,
                    severity: WarningSeverity::Info,
                    message: "Group: empty (no child commands)",
                });
            }
            if *opacity <= 0.0 {
                out.push(SceneWarning {
                    command_index: index,
                    severity: WarningSeverity::Warning,
                    message: "Group: zero opacity (entirely invisible)",
                });
            }
            // Recurse into children
            for (ci, child) in commands.iter().enumerate() {
                validate_command(child, ci, canvas_w, canvas_h, out);
            }
        }

        // Clear, Image, Path, Text, Sdf3D — no degenerate-geometry checks needed
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::style::Color;

    #[test]
    fn valid_scene_produces_no_warnings() {
        let canvas = PixelCanvas::new(200, 200)
            .circle(100.0, 100.0, 50.0)
            .fill(Color::RED)
            .done();

        let warnings = validate_scene(&canvas);
        assert!(warnings.is_empty(), "expected no warnings, got: {warnings:?}");
    }

    #[test]
    fn zero_radius_circle_warns() {
        let canvas = PixelCanvas::new(100, 100)
            .circle(50.0, 50.0, 0.0)
            .fill(Color::RED)
            .done();

        let warnings = validate_scene(&canvas);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].severity, WarningSeverity::Warning);
        assert!(warnings[0].message.contains("zero or negative radius"));
    }

    #[test]
    fn negative_rect_dimensions_warns() {
        let canvas = PixelCanvas::new(100, 100)
            .rect(10.0, 10.0, -5.0, 20.0)
            .fill(Color::BLUE)
            .done();

        let warnings = validate_scene(&canvas);
        assert!(
            warnings.iter().any(|w| w.message.contains("zero or negative dimensions")),
            "expected dimension warning, got: {warnings:?}",
        );
    }

    #[test]
    fn empty_polyline_errors() {
        use crate::scene::command::DrawCommand;
        use crate::scene::style::ShapeStyle;

        let canvas = PixelCanvas::new(100, 100);
        // Manually push a polyline with 0 points (builder normally prevents this)
        let mut cmds = canvas.commands().to_vec();
        cmds.push(DrawCommand::Polyline {
            points: vec![],
            closed: false,
            style: ShapeStyle::default(),
        });
        let canvas = PixelCanvas::from_commands(100, 100, cmds, Color::TRANSPARENT);

        let warnings = validate_scene(&canvas);
        assert!(
            warnings.iter().any(|w| w.severity == WarningSeverity::Error),
            "expected error-level warning for empty polyline",
        );
    }

    #[test]
    fn circle_outside_bounds_info() {
        let canvas = PixelCanvas::new(100, 100)
            .circle(-200.0, -200.0, 10.0)
            .fill(Color::RED)
            .done();

        let warnings = validate_scene(&canvas);
        assert!(
            warnings.iter().any(|w| w.severity == WarningSeverity::Info
                && w.message.contains("outside canvas bounds")),
            "expected info about out-of-bounds circle",
        );
    }

    #[test]
    fn warning_display_format() {
        let w = SceneWarning {
            command_index: 3,
            severity: WarningSeverity::Warning,
            message: "test warning",
        };
        assert_eq!(format!("{w}"), "[warn] cmd[3]: test warning");
    }
}
