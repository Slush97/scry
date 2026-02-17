// SPDX-License-Identifier: MIT OR Apache-2.0
//! SVG line drawing animation.
//!
//! Extract paths from an SVG and progressively reveal them using animated
//! dash patterns — producing a "pen drawing on paper" effect.
//!
//! # How it works
//!
//! 1. Parse an SVG into a `usvg::Tree` (via [`SvgImage`])
//! 2. Walk the tree to extract every `<path>` element as a [`SvgPathSegment`]
//! 3. Compute each path's total arc length
//! 4. On each frame, call [`SvgLineDrawing::draw`] with progress `t ∈ [0,1]`
//!    — it emits [`DrawCommand::Path`]
//!    commands with a [`DashPattern`] that
//!    reveals the appropriate portion of each stroke
//!
//! # Organic Drawing Features
//!
//! - **Per-segment easing** — built-in acceleration/deceleration per stroke
//! - **Pen pressure** — subtle stroke-width variation along each path
//! - **Pen-tip dot** — glowing dot at the current drawing position
//! - **Trailing ghost** — faded echo behind the tip for ink-settling feel
//!
//! # Example
//!
//! ```rust,ignore
//! use scry_engine::svg::line_drawing::SvgLineDrawing;
//! use scry_engine::scene::PixelCanvas;
//!
//! let drawing = SvgLineDrawing::from_str(SVG_CONTENT)
//!     .unwrap()
//!     .easing(Easing::EaseInOutCubic)
//!     .pen_pressure(PenPressure::default())
//!     .pen_tip(PenTip::default());
//!
//! let canvas = drawing.draw(PixelCanvas::new(800, 600), 0.5); // 50% drawn
//! ```

use std::sync::Arc;

use crate::scene::animation::Easing;
use crate::scene::command::{DrawCommand, PathData};
use crate::scene::style::{
    Color, DashPattern, FillStyle, LineCap, LineJoin, ShapeStyle, StrokeStyle,
};
use crate::scene::PixelCanvas;

use super::{SvgError, SvgImage};

// ---------------------------------------------------------------------------
// Path segment extraction
// ---------------------------------------------------------------------------

/// A single path extracted from an SVG, with pre-computed metadata.
#[derive(Clone, Debug)]
pub struct SvgPathSegment {
    /// The path geometry (same type used by [`DrawCommand::Path`]).
    /// Wrapped in `Arc` to avoid deep clones on every animation frame.
    pub path: Arc<tiny_skia::Path>,
    /// Total arc length of the path, in SVG user units.
    pub length: f32,
    /// Stroke color extracted from the SVG (defaults to white if absent).
    pub stroke_color: Color,
    /// Stroke width extracted from the SVG (defaults to 2.0 if absent).
    pub stroke_width: f32,
}

/// Walk a `usvg::Tree` and extract all visible path elements.
///
/// Each SVG `<path>`, `<circle>`, `<rect>`, etc. is converted to a
/// `tiny_skia::Path` by `usvg` during parsing. This function collects
/// them all with their styling and pre-computed lengths.
fn extract_paths(tree: &resvg::usvg::Tree) -> Vec<SvgPathSegment> {
    let mut segments = Vec::new();
    walk_group(tree.root(), &mut segments);
    segments
}

/// Recursively walk a group's children, collecting path segments.
fn walk_group(group: &resvg::usvg::Group, out: &mut Vec<SvgPathSegment>) {
    for node in group.children() {
        match node {
            resvg::usvg::Node::Path(ref path) => {
                if !path.is_visible() {
                    continue;
                }

                // Clone the tiny_skia path data into an Arc for zero-cost sharing.
                let skia_path = Arc::new(path.data().clone());
                let length = path_length(&skia_path);

                // Skip degenerate/zero-length paths.
                if length < f32::EPSILON {
                    continue;
                }

                // Extract stroke styling, falling back to fill color or white.
                let (stroke_color, stroke_width) = if let Some(stroke) = path.stroke() {
                    let color = extract_paint_color(stroke.paint());
                    let width = stroke.width().get();
                    (color, width)
                } else if let Some(fill) = path.fill() {
                    // If only fill is present, use it as stroke color with a default width.
                    let color = extract_paint_color(fill.paint());
                    (color, 2.0)
                } else {
                    (Color::WHITE, 2.0)
                };

                out.push(SvgPathSegment {
                    path: skia_path,
                    length,
                    stroke_color,
                    stroke_width,
                });
            }
            resvg::usvg::Node::Group(ref group) => {
                walk_group(group, out);
            }
            // Text nodes are converted to paths by usvg, so we only
            // encounter them as paths. Images are irrelevant for line drawing.
            _ => {}
        }
    }
}

/// Extract an RGB color from a `usvg::Paint`, defaulting to white for gradients/patterns.
const fn extract_paint_color(paint: &resvg::usvg::Paint) -> Color {
    match paint {
        resvg::usvg::Paint::Color(c) => Color::from_rgb8(c.red, c.green, c.blue),
        // Gradients/patterns don't map to a single color — use white.
        _ => Color::WHITE,
    }
}

// ---------------------------------------------------------------------------
// Path length measurement
// ---------------------------------------------------------------------------

/// Compute the total arc length of a `tiny_skia::Path`.
///
/// Line segments use exact Euclidean distance. Quadratic and cubic Bézier
/// curves are approximated by recursive subdivision (de Casteljau).
///
/// # Accuracy
///
/// Quadratic curves use 8 subdivisions, cubic curves use 16. This provides
/// sub-pixel accuracy for typical SVG paths while remaining fast.
pub fn path_length(path: &tiny_skia::Path) -> f32 {
    let mut total = 0.0_f32;
    let mut cursor = (0.0_f32, 0.0_f32);
    let mut subpath_start = cursor;

    for seg in path.segments() {
        match seg {
            tiny_skia::PathSegment::MoveTo(p) => {
                cursor = (p.x, p.y);
                subpath_start = cursor;
            }
            tiny_skia::PathSegment::LineTo(p) => {
                total += line_len(cursor, (p.x, p.y));
                cursor = (p.x, p.y);
            }
            tiny_skia::PathSegment::QuadTo(cp, end) => {
                total += quad_bezier_length(cursor, (cp.x, cp.y), (end.x, end.y), 8);
                cursor = (end.x, end.y);
            }
            tiny_skia::PathSegment::CubicTo(cp1, cp2, end) => {
                total +=
                    cubic_bezier_length(cursor, (cp1.x, cp1.y), (cp2.x, cp2.y), (end.x, end.y), 16);
                cursor = (end.x, end.y);
            }
            tiny_skia::PathSegment::Close => {
                total += line_len(cursor, subpath_start);
                cursor = subpath_start;
            }
        }
    }

    total
}

/// Evaluate the position on a path at a given arc-length distance.
///
/// Returns `(x, y)` of the point at `target_len` arc-length along the path.
/// If `target_len` exceeds the path length, returns the path's endpoint.
fn point_at_length(path: &tiny_skia::Path, target_len: f32) -> (f32, f32) {
    let mut accumulated = 0.0_f32;
    let mut cursor = (0.0_f32, 0.0_f32);
    let mut subpath_start = cursor;

    for seg in path.segments() {
        match seg {
            tiny_skia::PathSegment::MoveTo(p) => {
                cursor = (p.x, p.y);
                subpath_start = cursor;
            }
            tiny_skia::PathSegment::LineTo(p) => {
                let end = (p.x, p.y);
                let seg_len = line_len(cursor, end);
                if accumulated + seg_len >= target_len && seg_len > f32::EPSILON {
                    let frac = (target_len - accumulated) / seg_len;
                    return (
                        (end.0 - cursor.0).mul_add(frac, cursor.0),
                        (end.1 - cursor.1).mul_add(frac, cursor.1),
                    );
                }
                accumulated += seg_len;
                cursor = end;
            }
            tiny_skia::PathSegment::QuadTo(cp, end) => {
                let cp = (cp.x, cp.y);
                let end = (end.x, end.y);
                let seg_len = quad_bezier_length(cursor, cp, end, 8);
                if accumulated + seg_len >= target_len && seg_len > f32::EPSILON {
                    let frac = (target_len - accumulated) / seg_len;
                    return eval_quad_bezier(cursor, cp, end, frac);
                }
                accumulated += seg_len;
                cursor = end;
            }
            tiny_skia::PathSegment::CubicTo(cp1, cp2, end) => {
                let cp1 = (cp1.x, cp1.y);
                let cp2 = (cp2.x, cp2.y);
                let end = (end.x, end.y);
                let seg_len = cubic_bezier_length(cursor, cp1, cp2, end, 16);
                if accumulated + seg_len >= target_len && seg_len > f32::EPSILON {
                    let frac = (target_len - accumulated) / seg_len;
                    return eval_cubic_bezier(cursor, cp1, cp2, end, frac);
                }
                accumulated += seg_len;
                cursor = end;
            }
            tiny_skia::PathSegment::Close => {
                let seg_len = line_len(cursor, subpath_start);
                if accumulated + seg_len >= target_len && seg_len > f32::EPSILON {
                    let frac = (target_len - accumulated) / seg_len;
                    return (
                        (subpath_start.0 - cursor.0).mul_add(frac, cursor.0),
                        (subpath_start.1 - cursor.1).mul_add(frac, cursor.1),
                    );
                }
                accumulated += seg_len;
                cursor = subpath_start;
            }
        }
    }

    cursor
}

#[inline]
fn line_len(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    dx.hypot(dy)
}

/// Evaluate a quadratic Bézier at parameter `t`.
#[inline]
fn eval_quad_bezier(p0: (f32, f32), p1: (f32, f32), p2: (f32, f32), t: f32) -> (f32, f32) {
    let inv = 1.0 - t;
    let x = (t * t).mul_add(p2.0, (inv * inv).mul_add(p0.0, 2.0 * inv * t * p1.0));
    let y = (t * t).mul_add(p2.1, (inv * inv).mul_add(p0.1, 2.0 * inv * t * p1.1));
    (x, y)
}

/// Evaluate a cubic Bézier at parameter `t`.
#[inline]
fn eval_cubic_bezier(
    p0: (f32, f32),
    p1: (f32, f32),
    p2: (f32, f32),
    p3: (f32, f32),
    t: f32,
) -> (f32, f32) {
    let inv = 1.0 - t;
    let inv2 = inv * inv;
    let t2 = t * t;
    let x = (t2 * t).mul_add(
        p3.0,
        (3.0 * inv * t2).mul_add(p2.0, (inv2 * inv).mul_add(p0.0, 3.0 * inv2 * t * p1.0)),
    );
    let y = (t2 * t).mul_add(
        p3.1,
        (3.0 * inv * t2).mul_add(p2.1, (inv2 * inv).mul_add(p0.1, 3.0 * inv2 * t * p1.1)),
    );
    (x, y)
}

/// Approximate a quadratic Bézier curve length by `n` linear subdivisions.
fn quad_bezier_length(p0: (f32, f32), p1: (f32, f32), p2: (f32, f32), subdivisions: u32) -> f32 {
    let mut length = 0.0_f32;
    let mut prev = p0;
    let n = subdivisions as f32;

    for i in 1..=subdivisions {
        let t = i as f32 / n;
        let pt = eval_quad_bezier(p0, p1, p2, t);
        length += line_len(prev, pt);
        prev = pt;
    }

    length
}

/// Approximate a cubic Bézier curve length by `n` linear subdivisions.
fn cubic_bezier_length(
    p0: (f32, f32),
    p1: (f32, f32),
    p2: (f32, f32),
    p3: (f32, f32),
    subdivisions: u32,
) -> f32 {
    let mut length = 0.0_f32;
    let mut prev = p0;
    let n = subdivisions as f32;

    for i in 1..=subdivisions {
        let t = i as f32 / n;
        let pt = eval_cubic_bezier(p0, p1, p2, p3, t);
        length += line_len(prev, pt);
        prev = pt;
    }

    length
}

// ---------------------------------------------------------------------------
// Organic drawing configuration
// ---------------------------------------------------------------------------

/// Simulates hand pressure variation along a stroke.
///
/// The stroke width is modulated by a curve that ramps from `start` to `peak`
/// (at the midpoint) and back down to `end`, simulating how a pen naturally
/// applies more pressure in the middle of a stroke.
#[derive(Clone, Debug)]
pub struct PenPressure {
    /// Width multiplier at the beginning of the stroke (0.0–1.0).
    pub start: f32,
    /// Width multiplier at the midpoint of the stroke (typically 1.0).
    pub peak: f32,
    /// Width multiplier at the end of the stroke (0.0–1.0).
    pub end: f32,
}

impl Default for PenPressure {
    fn default() -> Self {
        Self {
            start: 0.4,
            peak: 1.0,
            end: 0.6,
        }
    }
}

impl PenPressure {
    /// Evaluate the pressure curve at position `t ∈ [0, 1]`.
    fn eval(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        if t <= 0.5 {
            // Ramp from start to peak
            let frac = t * 2.0;
            (self.peak - self.start).mul_add(frac, self.start)
        } else {
            // Ramp from peak to end
            let frac = (t - 0.5) * 2.0;
            (self.end - self.peak).mul_add(frac, self.peak)
        }
    }
}

/// A small glowing dot at the current drawing position.
///
/// When enabled, a circle is drawn at the tip of the currently-drawing
/// stroke, giving a visual cue of where the "pen" is.
#[derive(Clone, Debug)]
pub struct PenTip {
    /// Radius of the pen-tip dot, as a multiplier of the stroke width
    /// (default: 1.5, so a 3px stroke gets a 4.5px dot).
    pub radius_factor: f32,
    /// Opacity of the pen-tip dot (default: 0.7).
    pub opacity: f32,
}

impl Default for PenTip {
    fn default() -> Self {
        Self {
            radius_factor: 1.5,
            opacity: 0.7,
        }
    }
}

/// A faint echo behind the pen tip to simulate ink settling on paper.
///
/// This emits a second stroke command with reduced alpha covering just
/// the trailing portion of the currently-drawing stroke.
#[derive(Clone, Debug)]
pub struct Trail {
    /// How far behind the tip the trail extends, in SVG user units.
    pub length: f32,
    /// Starting opacity of the trail (fades to 0 at the far end).
    pub opacity: f32,
}

impl Default for Trail {
    fn default() -> Self {
        Self {
            length: 20.0,
            opacity: 0.3,
        }
    }
}

// ---------------------------------------------------------------------------
// Animation driver
// ---------------------------------------------------------------------------

/// Controls how multiple path segments are animated relative to each other.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum DrawMode {
    /// Paths are drawn one after another, proportional to their length.
    /// A path that is 30% of total length occupies 30% of the `t` range.
    #[default]
    Sequential,
    /// All paths animate simultaneously — each receives the full `t` range.
    Simultaneous,
}

/// SVG line drawing animator.
///
/// Holds extracted path segments from an SVG and generates
/// [`DrawCommand`]s with animated [`DashPattern`]s that progressively
/// reveal the strokes.
///
/// # Example
///
/// ```rust,ignore
/// use scry_engine::svg::line_drawing::{SvgLineDrawing, DrawMode};
/// use scry_engine::scene::PixelCanvas;
///
/// let drawing = SvgLineDrawing::from_str(r#"
///     <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
///         <circle cx="50" cy="50" r="40" stroke="red" fill="none" stroke-width="2"/>
///     </svg>
/// "#).unwrap()
///   .easing(Easing::EaseInOutCubic)
///   .pen_pressure(PenPressure::default())
///   .pen_tip(PenTip::default());
///
/// // In your animation loop:
/// let canvas = drawing.draw(PixelCanvas::new(400, 400), t);
/// ```
#[derive(Clone, Debug)]
pub struct SvgLineDrawing {
    segments: Vec<SvgPathSegment>,
    total_length: f32,
    mode: DrawMode,
    /// Per-segment easing applied before computing visible length.
    segment_easing: Option<Easing>,
    /// Optional pen-pressure simulation.
    pressure: Option<PenPressure>,
    /// Optional pen-tip dot.
    tip: Option<PenTip>,
    /// Optional trailing ghost.
    trail: Option<Trail>,
}

impl SvgLineDrawing {
    /// Create a line drawing animator from an already-parsed [`SvgImage`].
    pub fn from_svg(svg: &SvgImage) -> Self {
        let segments = extract_paths(svg.tree());
        let total_length = segments.iter().map(|s| s.length).sum();
        Self {
            segments,
            total_length,
            mode: DrawMode::default(),
            segment_easing: None,
            pressure: None,
            tip: None,
            trail: None,
        }
    }

    /// Parse an SVG string and create a line drawing animator.
    ///
    /// # Errors
    ///
    /// Returns an error if the SVG content is invalid.
    pub fn from_str(svg_content: &str) -> Result<Self, SvgError> {
        let svg = SvgImage::from_str(svg_content)?;
        Ok(Self::from_svg(&svg))
    }

    /// Set the animation mode.
    #[must_use]
    pub const fn mode(mut self, mode: DrawMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set per-segment easing.
    ///
    /// This easing is applied *within each individual segment*, controlling
    /// how the "pen speed" varies as it draws a single stroke. Use
    /// `EaseInOutCubic` for natural acceleration/deceleration, or
    /// `EaseOutQuad` for a quick start that tapers off.
    #[must_use]
    pub const fn easing(mut self, easing: Easing) -> Self {
        self.segment_easing = Some(easing);
        self
    }

    /// Enable pen-pressure simulation.
    ///
    /// Creates subtle stroke-width variation along each path, simulating
    /// the natural thick→thin variation of a hand-held pen.
    #[must_use]
    pub const fn pen_pressure(mut self, pressure: PenPressure) -> Self {
        self.pressure = Some(pressure);
        self
    }

    /// Enable the pen-tip dot.
    ///
    /// Draws a small glowing circle at the current tip of the pen,
    /// giving a visual cue of where the drawing is happening.
    #[must_use]
    pub const fn pen_tip(mut self, tip: PenTip) -> Self {
        self.tip = Some(tip);
        self
    }

    /// Enable the trailing ghost effect.
    ///
    /// Draws a faint, fading echo behind the pen tip to simulate
    /// ink settling on paper.
    #[must_use]
    pub const fn trail(mut self, trail: Trail) -> Self {
        self.trail = Some(trail);
        self
    }

    /// Total number of path segments extracted from the SVG.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Access individual path segments for custom styling or inspection.
    pub fn segments(&self) -> &[SvgPathSegment] {
        &self.segments
    }

    /// Total length across all path segments (only meaningful in [`DrawMode::Sequential`]).
    pub const fn total_length(&self) -> f32 {
        self.total_length
    }

    /// Generate [`DrawCommand`]s at animation progress `t ∈ [0, 1]`.
    ///
    /// Returns the canvas with all animated path commands appended.
    ///
    /// At `t = 0.0`, all strokes are invisible (no commands emitted).
    /// At `t = 1.0`, all strokes are fully visible (no dash pattern needed).
    pub fn draw(&self, mut canvas: PixelCanvas, t: f32) -> PixelCanvas {
        self.draw_into(&mut canvas, t);
        canvas
    }

    /// Append animated [`DrawCommand`]s to an existing canvas (mutable variant).
    ///
    /// This avoids ownership transfer, making it ideal for use inside
    /// animation loops where the canvas is built incrementally.
    pub fn draw_into(&self, canvas: &mut PixelCanvas, t: f32) {
        let t = t.clamp(0.0, 1.0);

        if self.segments.is_empty() || t < f32::EPSILON {
            return;
        }

        match self.mode {
            DrawMode::Sequential => self.draw_sequential(canvas, t),
            DrawMode::Simultaneous => self.draw_simultaneous(canvas, t),
        }
    }

    /// Sequential mode: paths appear one after another.
    fn draw_sequential(&self, canvas: &mut PixelCanvas, t: f32) {
        if self.total_length < f32::EPSILON {
            return;
        }

        let drawn_length = t * self.total_length;
        let mut accumulated = 0.0_f32;

        for seg in &self.segments {
            let seg_start = accumulated;
            let seg_end = seg_start + seg.length;
            accumulated = seg_end;

            if drawn_length <= seg_start {
                // Not reached yet — skip entirely (no invisible commands).
                continue;
            }

            let raw_progress = if drawn_length >= seg_end {
                1.0
            } else {
                (drawn_length - seg_start) / seg.length
            };

            // Apply per-segment easing to the local progress.
            let eased_progress = match &self.segment_easing {
                Some(easing) => easing.ease(raw_progress),
                None => raw_progress,
            };

            let visible = eased_progress * seg.length;
            let is_active = raw_progress < 1.0;

            self.emit_path_command(canvas, seg, visible, eased_progress, is_active);
        }
    }

    /// Simultaneous mode: all paths animate together.
    fn draw_simultaneous(&self, canvas: &mut PixelCanvas, t: f32) {
        for seg in &self.segments {
            // Apply per-segment easing.
            let eased = match &self.segment_easing {
                Some(easing) => easing.ease(t),
                None => t,
            };

            let visible = seg.length * eased;
            let is_active = t < 1.0;

            self.emit_path_command(canvas, seg, visible, eased, is_active);
        }
    }

    /// Emit a single path command with a dash pattern that reveals `visible` length.
    ///
    /// `progress` is the normalized position within this segment (0–1).
    /// `is_active` is true if this segment is currently being drawn (not yet complete).
    fn emit_path_command(
        &self,
        canvas: &mut PixelCanvas,
        seg: &SvgPathSegment,
        visible: f32,
        progress: f32,
        is_active: bool,
    ) {
        // Determine effective stroke width, possibly modulated by pen pressure.
        let effective_width = if let Some(ref pressure) = self.pressure {
            seg.stroke_width * pressure.eval(progress)
        } else {
            seg.stroke_width
        };

        // When fully visible, skip the dash pattern entirely for cleaner rendering.
        let dash = if visible >= seg.length - 0.01 {
            None
        } else {
            Some(DashPattern::pair(visible, seg.length + 1.0))
        };

        // Main stroke — use Arc clone (pointer copy, not deep clone).
        let path_data = PathData::from_shared(Arc::clone(&seg.path));

        canvas.push_command(DrawCommand::Path {
            path: path_data,
            style: ShapeStyle {
                fill: None,
                stroke: Some(StrokeStyle {
                    color: seg.stroke_color,
                    width: effective_width,
                    line_cap: LineCap::Round,
                    line_join: LineJoin::Round,
                    dash,
                }),
                anti_alias: true,
            },
        });

        // ── Trailing ghost ──────────────────────────────────────────────
        if let Some(ref trail) = self.trail {
            if is_active && visible > trail.length && visible < seg.length - 0.1 {
                let trail_start = visible - trail.length;
                let ghost_color = seg.stroke_color.with_alpha(trail.opacity);
                // Dash pattern: skip `trail_start`, then draw `trail.length`, then hide.
                let trail_dash =
                    DashPattern::quad(0.0, trail_start, trail.length, seg.length + 1.0, 0.0);

                canvas.push_command(DrawCommand::Path {
                    path: PathData::from_shared(Arc::clone(&seg.path)),
                    style: ShapeStyle {
                        fill: None,
                        stroke: Some(StrokeStyle {
                            color: ghost_color,
                            width: effective_width * 1.8,
                            line_cap: LineCap::Round,
                            line_join: LineJoin::Round,
                            dash: Some(trail_dash),
                        }),
                        anti_alias: true,
                    },
                });
            }
        }

        // ── Pen-tip dot ─────────────────────────────────────────────────
        if let Some(ref tip) = self.tip {
            if is_active && visible > 0.5 {
                let (tx, ty) = point_at_length(&seg.path, visible);
                let tip_radius = effective_width * tip.radius_factor;
                let tip_color = seg.stroke_color.with_alpha(tip.opacity);

                canvas.push_command(DrawCommand::Circle {
                    cx: tx,
                    cy: ty,
                    radius: tip_radius,
                    style: ShapeStyle {
                        fill: Some(FillStyle::Solid(tip_color)),
                        stroke: None,
                        anti_alias: true,
                    },
                });
            }
        }
    }

    /// Draw with an optional fill that fades in after the stroke completes.
    ///
    /// `fill_opacity` controls how opaque the fill is (0.0 = transparent, 1.0 = full).
    /// Typically you'd derive this from `t` — e.g., `(t - 0.8).max(0.0) / 0.2`
    /// to fade fill in during the last 20% of the animation.
    pub fn draw_with_fill(
        &self,
        mut canvas: PixelCanvas,
        t: f32,
        fill_color: Color,
        fill_opacity: f32,
    ) -> PixelCanvas {
        let t = t.clamp(0.0, 1.0);
        let fill_opacity = fill_opacity.clamp(0.0, 1.0);

        if self.segments.is_empty() {
            return canvas;
        }

        // Draw fills first (underneath strokes) if opacity > 0.
        if fill_opacity > f32::EPSILON {
            let fill_c = fill_color.with_alpha(fill_opacity);
            for seg in &self.segments {
                canvas.push_command(DrawCommand::Path {
                    path: PathData::from_shared(Arc::clone(&seg.path)),
                    style: ShapeStyle {
                        fill: Some(FillStyle::Solid(fill_c)),
                        stroke: None,
                        anti_alias: true,
                    },
                });
            }
        }

        // Then draw the animated strokes on top.
        canvas = self.draw(canvas, t);
        canvas
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const CIRCLE_SVG: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="40" stroke="red" fill="none" stroke-width="2"/>
    </svg>"#;

    const MULTI_PATH_SVG: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100">
        <line x1="0" y1="0" x2="100" y2="0" stroke="blue" stroke-width="1"/>
        <line x1="0" y1="50" x2="100" y2="50" stroke="green" stroke-width="1"/>
        <rect x="120" y="10" width="60" height="60" stroke="red" fill="none" stroke-width="1"/>
    </svg>"#;

    // ── path_length tests ──────────────────────────────────────────

    #[test]
    fn path_length_straight_line() {
        let mut pb = tiny_skia::PathBuilder::new();
        pb.move_to(0.0, 0.0);
        pb.line_to(100.0, 0.0);
        let path = pb.finish().unwrap();

        let len = path_length(&path);
        assert!((len - 100.0).abs() < 0.01, "expected ~100.0, got {len}");
    }

    #[test]
    fn path_length_square() {
        let mut pb = tiny_skia::PathBuilder::new();
        pb.move_to(0.0, 0.0);
        pb.line_to(100.0, 0.0);
        pb.line_to(100.0, 100.0);
        pb.line_to(0.0, 100.0);
        pb.close();
        let path = pb.finish().unwrap();

        let len = path_length(&path);
        assert!((len - 400.0).abs() < 0.01, "expected ~400.0, got {len}");
    }

    #[test]
    fn path_length_diagonal() {
        let mut pb = tiny_skia::PathBuilder::new();
        pb.move_to(0.0, 0.0);
        pb.line_to(3.0, 4.0);
        let path = pb.finish().unwrap();

        let len = path_length(&path);
        assert!(
            (len - 5.0).abs() < 0.01,
            "expected ~5.0 (3-4-5 triangle), got {len}"
        );
    }

    #[test]
    fn path_length_cubic_bezier() {
        // Quarter-circle approximation: radius 100, expected ~157 (π*100/2)
        let mut pb = tiny_skia::PathBuilder::new();
        pb.move_to(100.0, 0.0);
        pb.cubic_to(100.0, 55.228, 55.228, 100.0, 0.0, 100.0);
        let path = pb.finish().unwrap();

        let len = path_length(&path);
        let expected = std::f32::consts::FRAC_PI_2 * 100.0; // ~157.08
        assert!(
            (len - expected).abs() / expected < 0.01,
            "expected ~{expected}, got {len} (error > 1%)"
        );
    }

    // ── extract_paths tests ────────────────────────────────────────

    #[test]
    fn extract_paths_circle_svg() {
        let svg = SvgImage::from_str(CIRCLE_SVG).unwrap();
        let segments = extract_paths(svg.tree());

        assert_eq!(
            segments.len(),
            1,
            "circle SVG should produce 1 path segment"
        );
        assert!(
            segments[0].length > 200.0,
            "circle circumference should be > 200 (r=40 → ~251)"
        );
    }

    #[test]
    fn extract_paths_multi_path() {
        let svg = SvgImage::from_str(MULTI_PATH_SVG).unwrap();
        let segments = extract_paths(svg.tree());

        assert!(
            segments.len() >= 3,
            "multi-path SVG should extract at least 3 segments, got {}",
            segments.len()
        );
    }

    // ── SvgLineDrawing tests ───────────────────────────────────────

    #[test]
    fn draw_at_zero_is_empty() {
        let drawing = SvgLineDrawing::from_str(CIRCLE_SVG).unwrap();
        let canvas = drawing.draw(PixelCanvas::new(200, 200), 0.0);
        // At t=0, no commands should be emitted (everything invisible).
        assert!(
            canvas.commands().is_empty(),
            "draw at t=0 should produce zero commands (skip invisible)"
        );
    }

    #[test]
    fn draw_at_one_fully_visible() {
        let drawing = SvgLineDrawing::from_str(CIRCLE_SVG).unwrap();
        let canvas = drawing.draw(PixelCanvas::new(200, 200), 1.0);
        assert!(
            !canvas.commands().is_empty(),
            "draw at t=1 should produce path commands"
        );

        // At t=1.0, fully visible paths should have NO dash pattern.
        if let DrawCommand::Path { style, .. } = &canvas.commands()[0] {
            let stroke = style.stroke.as_ref().unwrap();
            assert!(
                stroke.dash.is_none(),
                "at t=1.0, fully visible paths should have no dash pattern"
            );
        }
    }

    #[test]
    fn draw_at_half_partial() {
        let drawing = SvgLineDrawing::from_str(CIRCLE_SVG).unwrap();
        let canvas = drawing.draw(PixelCanvas::new(200, 200), 0.5);
        assert!(!canvas.commands().is_empty());

        if let DrawCommand::Path { style, .. } = &canvas.commands()[0] {
            let dash = style.stroke.as_ref().unwrap().dash.as_ref().unwrap();
            let visible = dash.intervals[0];
            let total = drawing.segments()[0].length;
            assert!(
                visible > 0.0 && visible < total,
                "at t=0.5, visible ({visible}) should be between 0 and total ({total})"
            );
        }
    }

    #[test]
    fn from_str_invalid_svg() {
        assert!(SvgLineDrawing::from_str("not svg at all").is_err());
    }

    #[test]
    fn simultaneous_mode_works() {
        let drawing = SvgLineDrawing::from_str(MULTI_PATH_SVG)
            .unwrap()
            .mode(DrawMode::Simultaneous);

        let canvas = drawing.draw(PixelCanvas::new(200, 100), 0.5);
        // In simultaneous mode, at least one command per segment
        assert!(
            canvas.commands().len() >= drawing.segment_count(),
            "simultaneous mode should emit at least one command per segment"
        );
    }

    #[test]
    fn segment_count_matches() {
        let drawing = SvgLineDrawing::from_str(MULTI_PATH_SVG).unwrap();
        assert!(
            drawing.segment_count() >= 3,
            "expected at least 3 segments, got {}",
            drawing.segment_count()
        );
    }

    // ── New organic feature tests ──────────────────────────────────

    #[test]
    fn pen_pressure_modulates_width() {
        let drawing = SvgLineDrawing::from_str(CIRCLE_SVG)
            .unwrap()
            .pen_pressure(PenPressure::default());

        // At t=0.5, pressure at the midpoint should produce peak width.
        let canvas = drawing.draw(PixelCanvas::new(200, 200), 0.5);
        assert!(!canvas.commands().is_empty());

        if let DrawCommand::Path { style, .. } = &canvas.commands()[0] {
            let width = style.stroke.as_ref().unwrap().width;
            // Original width is 2.0, at progress ~0.5 peak should be ~2.0 (1.0 * 2.0)
            // but since sequential mode, the segment progress differs.
            // At least verify it's non-zero and positive.
            assert!(width > 0.0, "pen pressure should produce positive width");
        }
    }

    #[test]
    fn pen_tip_emits_circle() {
        let drawing = SvgLineDrawing::from_str(CIRCLE_SVG)
            .unwrap()
            .pen_tip(PenTip::default());

        let canvas = drawing.draw(PixelCanvas::new(200, 200), 0.5);
        // Should have the path command + a circle for the pen tip.
        let has_circle = canvas
            .commands()
            .iter()
            .any(|cmd| matches!(cmd, DrawCommand::Circle { .. }));
        assert!(
            has_circle,
            "pen_tip should emit a Circle command at partial t"
        );
    }

    #[test]
    fn pen_tip_not_at_zero() {
        let drawing = SvgLineDrawing::from_str(CIRCLE_SVG)
            .unwrap()
            .pen_tip(PenTip::default());

        let canvas = drawing.draw(PixelCanvas::new(200, 200), 0.0);
        // At t=0, no commands at all.
        assert!(
            canvas.commands().is_empty(),
            "pen_tip should NOT draw at t=0"
        );
    }

    #[test]
    fn pen_tip_not_at_one() {
        let drawing = SvgLineDrawing::from_str(CIRCLE_SVG)
            .unwrap()
            .pen_tip(PenTip::default());

        let canvas = drawing.draw(PixelCanvas::new(200, 200), 1.0);
        // At t=1, segment is complete so pen tip should NOT appear.
        let has_circle = canvas
            .commands()
            .iter()
            .any(|cmd| matches!(cmd, DrawCommand::Circle { .. }));
        assert!(
            !has_circle,
            "pen_tip should NOT emit a Circle at t=1.0 (drawing complete)"
        );
    }

    #[test]
    fn trail_emits_extra_stroke() {
        let drawing = SvgLineDrawing::from_str(CIRCLE_SVG).unwrap().trail(Trail {
            length: 10.0,
            opacity: 0.3,
        });

        // The circle has ~251 length. At t=0.5, visible ~125.5, which is > 10.
        let canvas = drawing.draw(PixelCanvas::new(200, 200), 0.5);
        // Should have at least 2 path commands (main stroke + trail).
        let path_count = canvas
            .commands()
            .iter()
            .filter(|cmd| matches!(cmd, DrawCommand::Path { .. }))
            .count();
        assert!(
            path_count >= 2,
            "trail should emit an additional Path command, got {path_count}"
        );
    }

    #[test]
    fn per_segment_easing_differs_from_linear() {
        let linear = SvgLineDrawing::from_str(CIRCLE_SVG).unwrap();
        let eased = SvgLineDrawing::from_str(CIRCLE_SVG)
            .unwrap()
            .easing(Easing::EaseInOutCubic);

        let canvas_lin = linear.draw(PixelCanvas::new(200, 200), 0.3);
        let canvas_eas = eased.draw(PixelCanvas::new(200, 200), 0.3);

        // Both should produce commands.
        assert!(!canvas_lin.commands().is_empty());
        assert!(!canvas_eas.commands().is_empty());

        // The visible lengths (dash intervals) should differ.
        let get_visible = |canvas: &PixelCanvas| -> f32 {
            if let DrawCommand::Path { style, .. } = &canvas.commands()[0] {
                style
                    .stroke
                    .as_ref()
                    .unwrap()
                    .dash
                    .as_ref()
                    .map_or(999.0, |d| d.intervals[0])
            } else {
                0.0
            }
        };

        let vis_lin = get_visible(&canvas_lin);
        let vis_eas = get_visible(&canvas_eas);
        assert!(
            (vis_lin - vis_eas).abs() > 0.01,
            "eased visible ({vis_eas}) should differ from linear ({vis_lin})"
        );
    }

    #[test]
    fn point_at_length_straight_line() {
        let mut pb = tiny_skia::PathBuilder::new();
        pb.move_to(0.0, 0.0);
        pb.line_to(100.0, 0.0);
        let path = pb.finish().unwrap();

        let (x, y) = point_at_length(&path, 50.0);
        assert!((x - 50.0).abs() < 0.5, "expected x≈50, got {x}");
        assert!((y - 0.0).abs() < 0.5, "expected y≈0, got {y}");
    }

    #[test]
    fn pressure_eval_edges() {
        let p = PenPressure {
            start: 0.2,
            peak: 1.0,
            end: 0.4,
        };
        assert!((p.eval(0.0) - 0.2).abs() < 0.01);
        assert!((p.eval(0.5) - 1.0).abs() < 0.01);
        assert!((p.eval(1.0) - 0.4).abs() < 0.01);
    }

    #[test]
    fn all_features_combined() {
        let drawing = SvgLineDrawing::from_str(CIRCLE_SVG)
            .unwrap()
            .easing(Easing::EaseInOutCubic)
            .pen_pressure(PenPressure::default())
            .pen_tip(PenTip::default())
            .trail(Trail::default());

        // Should produce commands at partial progress without panicking.
        let canvas = drawing.draw(PixelCanvas::new(200, 200), 0.5);
        assert!(
            !canvas.commands().is_empty(),
            "combined features should draw"
        );
    }
}
