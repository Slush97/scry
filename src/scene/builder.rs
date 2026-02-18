// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fluent builder API for constructing a [`PixelCanvas`] scene.
//!
//! The builder pattern lets users construct drawing scenes without needing to
//! know `tiny-skia` internals:
//!
//! ```
//! use scry_engine::scene::PixelCanvas;
//! use scry_engine::scene::style::Color;
//!
//! let canvas = PixelCanvas::new(200, 200)
//!     .background(Color::BLACK)
//!     .circle(100.0, 100.0, 50.0)
//!         .fill(Color::BLUE)
//!         .stroke(Color::WHITE, 2.0)
//!         .done()
//!     .line(10.0, 10.0, 190.0, 190.0)
//!         .color(Color::RED)
//!         .width(3.0)
//!         .done();
//! ```

use std::hash::{Hash, Hasher};

#[cfg(feature = "input")]
use std::collections::HashMap;

#[cfg(feature = "text")]
use crate::scene::command::{FontData, TextAlign, TextStyle};
use crate::scene::command::{DrawCommand, ImageData, PathData};
use crate::scene::style::{
    BlendMode, ClipRegion, Color, DashPattern, FillRule, FillStyle, GradientDef, GradientKind,
    GradientStop, LineCap, LineJoin, Point, Rect, ShapeStyle, StrokeStyle, Transform,
};

// ---------------------------------------------------------------------------
// PixelCanvas
// ---------------------------------------------------------------------------

/// The main drawing surface. Collects [`DrawCommand`]s into a scene display list.
///
/// A `PixelCanvas` is a lightweight, allocation-only builder. No pixel work
/// happens until [`rasterize()`](crate::rasterize::Rasterizer::rasterize) is
/// called.
#[derive(Clone, Debug)]
pub struct PixelCanvas {
    commands: Vec<DrawCommand>,
    background: Color,
    width: u32,
    height: u32,
    /// Per-command hit-test tags. Keys are command indices.
    #[cfg(feature = "input")]
    hit_tags: HashMap<usize, crate::scene::hit::HitTag>,
}

impl PixelCanvas {
    /// Create a new canvas with the given pixel dimensions.
    ///
    /// The canvas starts empty with a transparent background.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            commands: Vec::new(),
            background: Color::TRANSPARENT,
            width,
            height,
            #[cfg(feature = "input")]
            hit_tags: HashMap::new(),
        }
    }

    /// Set the background color. This is drawn before all other commands.
    #[must_use]
    pub const fn background(mut self, color: Color) -> Self {
        self.background = color;
        self
    }

    /// Width of the canvas in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Height of the canvas in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// The background color.
    #[must_use]
    pub const fn background_color(&self) -> Color {
        self.background
    }

    /// The list of drawing commands in submission order.
    #[must_use]
    pub fn commands(&self) -> &[DrawCommand] {
        &self.commands
    }

    /// Returns `true` if this scene can be correctly rendered on GPU.
    ///
    /// The 2D GPU rasterizer draws all GPU primitives first, then blits CPU
    /// fallback overlays on top.  This fixed render order means **any** CPU
    /// fallback command breaks z-ordering with surrounding GPU commands.
    /// Additionally, shapes with gradient fills are silently transparent on
    /// GPU (no shader support yet).
    ///
    /// Therefore we only route to GPU when the scene is **pure GPU** — zero
    /// CPU-fallback commands.
    #[must_use]
    pub fn gpu_suitable(&self) -> bool {
        Self::all_gpu_native(&self.commands)
    }

    /// Returns `true` when every command in `cmds` can be rendered natively
    /// on the GPU without CPU fallback.
    fn all_gpu_native(cmds: &[DrawCommand]) -> bool {
        for cmd in cmds {
            match cmd {
                DrawCommand::Line { stroke, .. } => {
                    // Dashed lines, non-Butt caps, and gradient strokes fall back
                    // to CPU in the GPU rasterizer.
                    if stroke.dash.is_some()
                        || stroke.line_cap != crate::scene::style::LineCap::Butt
                        || stroke.paint.is_some()
                    {
                        return false;
                    }
                }
                DrawCommand::Gradient { .. } | DrawCommand::Clear { .. } => {}

                // Shapes are GPU-native only with solid or no fill,
                // no per-shape opacity/transform, and winding fill rule.
                DrawCommand::Circle { style, .. }
                | DrawCommand::Rectangle { style, .. }
                | DrawCommand::Ellipse { style, .. } => {
                    if style.opacity < 1.0
                        || style.transform.is_some()
                        || style.fill_rule != crate::scene::style::FillRule::Winding
                        || style.blend_mode != crate::scene::style::BlendMode::SrcOver
                    {
                        return false;
                    }
                    if matches!(
                        &style.fill,
                        Some(
                            crate::scene::style::FillStyle::LinearGradient(_)
                                | crate::scene::style::FillStyle::RadialGradient(_)
                        )
                    ) {
                        return false;
                    }
                }

                // Polylines: stroke-only or solid fill is GPU-native.
                DrawCommand::Polyline { style, .. } => {
                    if style.opacity < 1.0
                        || style.transform.is_some()
                        || style.blend_mode != crate::scene::style::BlendMode::SrcOver
                    {
                        return false;
                    }
                    // Gradient fills still require CPU
                    if matches!(
                        &style.fill,
                        Some(
                            crate::scene::style::FillStyle::LinearGradient(_)
                                | crate::scene::style::FillStyle::RadialGradient(_)
                        )
                    ) {
                        return false;
                    }
                }

                // Paths and arcs: solid fill with simple compositing is GPU-native
                // (tessellated to triangles). Gradient fills, strokes, or complex
                // compositing still fall back to CPU.
                DrawCommand::Path { style, .. } | DrawCommand::Arc { style, .. } => {
                    if style.stroke.is_some() || !solid_fill_only(style) {
                        return false;
                    }
                }

                DrawCommand::Image { .. } => return false,

                #[cfg(feature = "sdf")]
                DrawCommand::Sdf3D { .. } => return false,

                #[cfg(feature = "text")]
                DrawCommand::Text { .. } => return false,

                DrawCommand::Group {
                    commands,
                    opacity,
                    blend_mode,
                    clip,
                    transform,
                } => {
                    let needs_compositing = *opacity < 1.0
                        || clip.is_some()
                        || *blend_mode != crate::scene::style::BlendMode::SrcOver;
                    if needs_compositing || *transform != crate::scene::style::Transform::IDENTITY {
                        return false;
                    }
                    if !Self::all_gpu_native(commands) {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Consume the canvas and return the command list.
    #[must_use]
    pub fn into_commands(self) -> Vec<DrawCommand> {
        self.commands
    }

    /// Add a raw draw command. For advanced use cases where the fluent API
    /// doesn't cover your needs.
    #[must_use]
    pub fn command(mut self, cmd: DrawCommand) -> Self {
        self.commands.push(cmd);
        self
    }

    /// Add a raw draw command mutably. For conditional composition
    /// where the fluent builder pattern is inconvenient.
    ///
    /// ```
    /// use scry_engine::scene::PixelCanvas;
    /// use scry_engine::scene::command::DrawCommand;
    /// use scry_engine::scene::style::{Color, ShapeStyle, FillStyle};
    ///
    /// let mut canvas = PixelCanvas::new(200, 200);
    /// let show_border = true;
    /// if show_border {
    ///     canvas.push_command(DrawCommand::Circle {
    ///         cx: 100.0, cy: 100.0, radius: 50.0,
    ///         style: ShapeStyle { fill: Some(FillStyle::Solid(Color::RED)), stroke: None, anti_alias: true, ..ShapeStyle::default() },
    ///     });
    /// }
    /// ```
    pub fn push_command(&mut self, cmd: DrawCommand) {
        self.commands.push(cmd);
    }

    /// Remove all draw commands, keeping the canvas dimensions and background color.
    ///
    /// Useful for animation loops where you rebuild the scene each frame.
    pub fn clear(&mut self) {
        self.commands.clear();
    }

    /// The number of draw commands in this canvas.
    #[must_use]
    pub fn command_count(&self) -> usize {
        self.commands.len()
    }

    // --- Fluent shape builders ---

    /// Begin drawing a circle.
    ///
    /// Returns a [`ShapeBuilder`] for configuring fill, stroke, and anti-aliasing.
    #[must_use]
    pub const fn circle(self, cx: f32, cy: f32, radius: f32) -> ShapeBuilder {
        ShapeBuilder::new(self, ShapeKind::Circle { cx, cy, radius })
    }

    /// Begin drawing a rectangle.
    ///
    /// Returns a [`ShapeBuilder`] for configuring fill, stroke, corner radius,
    /// and anti-aliasing.
    #[must_use]
    pub const fn rect(self, x: f32, y: f32, width: f32, height: f32) -> ShapeBuilder {
        ShapeBuilder::new(
            self,
            ShapeKind::Rectangle {
                rect: Rect::new(x, y, width, height),
                corner_radius: 0.0,
            },
        )
    }

    /// Begin drawing a line.
    ///
    /// Returns a [`LineBuilder`] for configuring stroke color, width, and dashing.
    #[must_use]
    pub fn line(self, x1: f32, y1: f32, x2: f32, y2: f32) -> LineBuilder {
        LineBuilder::new(self, x1, y1, x2, y2)
    }

    /// Begin drawing an arbitrary Bézier path.
    ///
    /// The path should be constructed using `tiny_skia::PathBuilder`.
    #[must_use]
    pub const fn path(self, path: tiny_skia::Path) -> ShapeBuilder {
        ShapeBuilder::new(self, ShapeKind::Path(PathData::new(path)))
    }

    /// Begin drawing an ellipse.
    ///
    /// `rx` and `ry` are the horizontal and vertical radii. Use
    /// `rotation` (in radians) to rotate the ellipse.
    ///
    /// Returns a [`ShapeBuilder`] for configuring fill, stroke, and anti-aliasing.
    #[must_use]
    pub const fn ellipse(self, cx: f32, cy: f32, rx: f32, ry: f32) -> ShapeBuilder {
        ShapeBuilder::new(
            self,
            ShapeKind::Ellipse {
                cx,
                cy,
                rx,
                ry,
                rotation: 0.0,
            },
        )
    }

    /// Begin drawing a polyline (connected open line segments).
    ///
    /// Pass a list of `(x, y)` vertices. The path will remain open.
    /// Use `.fill()` / `.stroke()` on the returned builder.
    #[must_use]
    pub const fn polyline(self, points: Vec<(f32, f32)>) -> ShapeBuilder {
        ShapeBuilder::new(
            self,
            ShapeKind::Polyline {
                points,
                closed: false,
            },
        )
    }

    /// Begin drawing a closed polygon.
    ///
    /// Like `polyline`, but the path is closed automatically.
    #[must_use]
    pub const fn polygon(self, points: Vec<(f32, f32)>) -> ShapeBuilder {
        ShapeBuilder::new(
            self,
            ShapeKind::Polyline {
                points,
                closed: true,
            },
        )
    }

    /// Begin drawing a star (regular star polygon).
    ///
    /// Generates alternating outer/inner radius vertices at evenly spaced
    /// angles, producing a pointed star shape. Uses the existing polygon
    /// infrastructure.
    ///
    /// # Parameters
    /// - `cx`, `cy`: center coordinates
    /// - `outer_r`: radius to the star's outer points
    /// - `inner_r`: radius to the inner notches
    /// - `num_points`: number of outer points (e.g., 5 for a classic star)
    #[must_use]
    pub fn star(
        self,
        cx: f32,
        cy: f32,
        outer_r: f32,
        inner_r: f32,
        num_points: usize,
    ) -> ShapeBuilder {
        use std::f32::consts::TAU;
        let total = num_points * 2;
        let mut verts = Vec::with_capacity(total);
        for i in 0..total {
            let angle = (i as f32 / total as f32) * TAU - std::f32::consts::FRAC_PI_2;
            let r = if i % 2 == 0 { outer_r } else { inner_r };
            verts.push((cx + r * angle.cos(), cy + r * angle.sin()));
        }
        ShapeBuilder::new(
            self,
            ShapeKind::Polyline {
                points: verts,
                closed: true,
            },
        )
    }

    /// Begin drawing a circular arc (partial circle).
    ///
    /// `start_angle` and `sweep_angle` are in radians. A start angle of 0 is
    /// at the 3 o'clock position; positive sweep is counter-clockwise.
    ///
    /// Returns a [`ShapeBuilder`] for configuring fill, stroke, and anti-aliasing.
    #[must_use]
    pub const fn arc(
        self,
        cx: f32,
        cy: f32,
        radius: f32,
        start_angle: f32,
        sweep_angle: f32,
    ) -> ShapeBuilder {
        ShapeBuilder::new(
            self,
            ShapeKind::Arc {
                cx,
                cy,
                radius,
                start_angle,
                sweep_angle,
            },
        )
    }

    /// Begin drawing a gradient-filled rectangle.
    #[must_use]
    pub fn gradient(self, x: f32, y: f32, width: f32, height: f32) -> GradientBuilder {
        GradientBuilder::new(self, Rect::new(x, y, width, height))
    }

    /// Begin a transform group. All subsequent commands added via the returned
    /// [`GroupBuilder`] will have the given transform applied.
    #[must_use]
    pub fn group(self, transform: Transform) -> GroupBuilder {
        GroupBuilder::new(self, transform)
    }

    /// Begin drawing a raster image at the given position.
    ///
    /// Returns an [`ImageBuilder`] for configuring opacity.
    #[must_use]
    pub const fn image(self, image: ImageData, x: f32, y: f32) -> ImageBuilder {
        ImageBuilder::new(self, image, x, y)
    }

    /// Begin drawing text at the given baseline position.
    ///
    /// Returns a [`TextBuilder`] for configuring font, size, and color.
    /// If no font is set on the builder, the embedded default font is used.
    #[cfg(feature = "text")]
    #[must_use]
    pub fn text(self, text: &str, x: f32, y: f32) -> TextBuilder {
        TextBuilder::new(self, text.to_string(), x, y)
    }

    /// Draw text using a pre-built [`TextStyle`]. Shorthand for
    /// `canvas.text(text, x, y).style(&style).done()`.
    #[cfg(feature = "text")]
    #[must_use]
    pub fn text_styled(self, text: &str, x: f32, y: f32, style: &TextStyle) -> Self {
        self.text(text, x, y).style(style).done()
    }

    /// Add an SDF 3D scene to the canvas at the given position and size.
    ///
    /// The SDF ray marcher will render `scene` into a `(w × h)` pixel region
    /// and composite the result at `(x, y)`. The `generation` counter is used
    /// for caching — bump it whenever the scene changes.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::sync::Arc;
    /// use scry_engine::sdf::SdfScene;
    ///
    /// let scene = Arc::new(SdfScene::default());
    /// let canvas = PixelCanvas::new(800, 600)
    ///     .sdf_scene(scene, 0, 100.0, 50.0, 600.0, 400.0, 0.0);
    /// ```
    #[cfg(feature = "sdf")]
    #[must_use]
    pub fn sdf_scene(
        self,
        scene: std::sync::Arc<crate::sdf::SdfScene>,
        generation: u64,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        time: f32,
    ) -> Self {
        self.push(DrawCommand::Sdf3D {
            scene: crate::scene::command::SdfSceneRef::new(scene, generation),
            rect: Rect::new(x, y, w, h),
            time,
            render_scale: None,
        })
    }

    /// Add an SDF 3D scene with a render scale factor.
    ///
    /// Same as [`sdf_scene`](Self::sdf_scene) but renders at a reduced internal
    /// resolution and bicubic-upscales. `render_scale` is clamped to `[0.1, 1.0]`.
    ///
    /// - `1.0` — full resolution (same as `sdf_scene`)
    /// - `0.5` — half resolution (~4× faster)
    /// - `0.25` — quarter resolution (~16× faster)
    #[cfg(feature = "sdf")]
    #[must_use]
    pub fn sdf_scene_scaled(
        self,
        scene: std::sync::Arc<crate::sdf::SdfScene>,
        generation: u64,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        time: f32,
        render_scale: f32,
    ) -> Self {
        self.push(DrawCommand::Sdf3D {
            scene: crate::scene::command::SdfSceneRef::new(scene, generation),
            rect: Rect::new(x, y, w, h),
            time,
            render_scale: Some(render_scale),
        })
    }

    /// Compute a content hash of the entire scene (background + commands).
    ///
    /// This is used by the caching layer to detect when a scene has changed.
    #[must_use]
    pub fn content_hash(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.background.hash(&mut hasher);
        self.width.hash(&mut hasher);
        self.height.hash(&mut hasher);
        self.commands.hash(&mut hasher);
        hasher.finish()
    }

    /// Tag the most recently added command for hit-testing.
    ///
    /// The tag is associated with the last command in the display list.
    /// Commands without tags are ignored during hit-testing.
    ///
    /// # Panics
    ///
    /// Panics if the canvas has no commands (nothing to tag).
    #[cfg(feature = "input")]
    #[must_use]
    pub fn with_tag(mut self, tag: crate::scene::hit::HitTag) -> Self {
        assert!(
            !self.commands.is_empty(),
            "with_tag() requires at least one command"
        );
        let idx = self.commands.len() - 1;
        self.hit_tags.insert(idx, tag);
        self
    }

    /// Tag a specific command index for hit-testing.
    #[cfg(feature = "input")]
    pub fn tag_command(&mut self, index: usize, tag: crate::scene::hit::HitTag) {
        self.hit_tags.insert(index, tag);
    }

    /// Get the hit-test tag map.
    #[cfg(feature = "input")]
    #[must_use]
    pub fn hit_tags(&self) -> &HashMap<usize, crate::scene::hit::HitTag> {
        &self.hit_tags
    }

    /// Push a command internally (used by builders).
    fn push(mut self, cmd: DrawCommand) -> Self {
        self.commands.push(cmd);
        self
    }
}

/// Returns `true` if the style is compatible with GPU tessellation:
/// solid or no fill, full opacity, no transform, default blend mode, winding rule.
fn solid_fill_only(style: &ShapeStyle) -> bool {
    matches!(&style.fill, Some(FillStyle::Solid(_)) | None)
        && style.opacity >= 1.0
        && style.transform.is_none()
        && style.blend_mode == BlendMode::SrcOver
        && style.fill_rule == FillRule::Winding
}

// ---------------------------------------------------------------------------
// Internal shape kind enum (not public — just for the builder)
// ---------------------------------------------------------------------------

/// Internal enum to track what shape the builder is configuring.
enum ShapeKind {
    Circle {
        cx: f32,
        cy: f32,
        radius: f32,
    },
    Rectangle {
        rect: Rect,
        corner_radius: f32,
    },
    Ellipse {
        cx: f32,
        cy: f32,
        rx: f32,
        ry: f32,
        rotation: f32,
    },
    Path(PathData),
    Polyline {
        points: Vec<(f32, f32)>,
        closed: bool,
    },
    Arc {
        cx: f32,
        cy: f32,
        radius: f32,
        start_angle: f32,
        sweep_angle: f32,
    },
}

// ---------------------------------------------------------------------------
// ShapeBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for shapes (circle, rect, path) with fill and stroke.
///
/// Finish building by calling [`.done()`](ShapeBuilder::done) to add the
/// command to the canvas.
pub struct ShapeBuilder {
    canvas: PixelCanvas,
    kind: ShapeKind,
    fill: Option<FillStyle>,
    stroke: Option<StrokeStyle>,
    anti_alias: bool,
    corner_radius: Option<f32>,
    rotation: Option<f32>,
    fill_rule: FillRule,
    opacity: f32,
    shape_transform: Option<Transform>,
    blend_mode: BlendMode,
}

impl ShapeBuilder {
    const fn new(canvas: PixelCanvas, kind: ShapeKind) -> Self {
        Self {
            canvas,
            kind,
            fill: None,
            stroke: None,
            anti_alias: true,
            corner_radius: None,
            rotation: None,
            fill_rule: FillRule::Winding,
            opacity: 1.0,
            shape_transform: None,
            blend_mode: BlendMode::SrcOver,
        }
    }

    /// Set a solid fill color.
    #[must_use]
    pub fn fill(mut self, color: Color) -> Self {
        self.fill = Some(FillStyle::Solid(color));
        self
    }

    /// Set a linear gradient fill.
    #[must_use]
    pub fn fill_linear_gradient(mut self, gradient: GradientDef) -> Self {
        self.fill = Some(FillStyle::LinearGradient(gradient));
        self
    }

    /// Set a radial gradient fill.
    #[must_use]
    pub fn fill_radial_gradient(mut self, gradient: GradientDef) -> Self {
        self.fill = Some(FillStyle::RadialGradient(gradient));
        self
    }

    /// Set stroke color and width.
    #[must_use]
    pub fn stroke(mut self, color: Color, width: f32) -> Self {
        let s = self.stroke.get_or_insert_with(StrokeStyle::default);
        s.color = color;
        s.width = width;
        self
    }

    /// Set stroke line cap.
    #[must_use]
    pub fn line_cap(mut self, cap: LineCap) -> Self {
        self.stroke
            .get_or_insert_with(StrokeStyle::default)
            .line_cap = cap;
        self
    }

    /// Set stroke line join.
    #[must_use]
    pub fn line_join(mut self, join: LineJoin) -> Self {
        self.stroke
            .get_or_insert_with(StrokeStyle::default)
            .line_join = join;
        self
    }

    /// Set stroke dash pattern.
    #[must_use]
    pub fn dash(mut self, pattern: DashPattern) -> Self {
        self.stroke.get_or_insert_with(StrokeStyle::default).dash = Some(pattern);
        self
    }

    /// Set corner radius (only meaningful for rectangles).
    #[must_use]
    pub const fn corner_radius(mut self, radius: f32) -> Self {
        self.corner_radius = Some(radius);
        self
    }

    /// Set the rotation angle in radians (only meaningful for ellipses).
    #[must_use]
    pub const fn rotation(mut self, radians: f32) -> Self {
        self.rotation = Some(radians);
        self
    }

    /// Enable or disable anti-aliasing (default: enabled).
    #[must_use]
    pub const fn anti_alias(mut self, enabled: bool) -> Self {
        self.anti_alias = enabled;
        self
    }

    /// Set the fill rule for this shape.
    #[must_use]
    pub const fn fill_rule(mut self, rule: FillRule) -> Self {
        self.fill_rule = rule;
        self
    }

    /// Set per-shape opacity (0.0–1.0). Default: 1.0.
    #[must_use]
    pub const fn opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }

    /// Set a per-shape transform (applied before the parent/group transform).
    #[must_use]
    pub const fn transform(mut self, transform: Transform) -> Self {
        self.shape_transform = Some(transform);
        self
    }

    /// Set the miter limit for the stroke.
    #[must_use]
    pub fn miter_limit(mut self, limit: f32) -> Self {
        self.stroke
            .get_or_insert_with(StrokeStyle::default)
            .miter_limit = limit;
        self
    }

    /// Set a linear gradient paint for the stroke.
    #[must_use]
    pub fn stroke_gradient(mut self, gradient: GradientDef) -> Self {
        self.stroke.get_or_insert_with(StrokeStyle::default).paint =
            Some(FillStyle::LinearGradient(gradient));
        self
    }

    /// Set a radial gradient paint for the stroke.
    #[must_use]
    pub fn stroke_radial_gradient(mut self, gradient: GradientDef) -> Self {
        self.stroke.get_or_insert_with(StrokeStyle::default).paint =
            Some(FillStyle::RadialGradient(gradient));
        self
    }

    /// Set per-shape blend mode. Default: `SrcOver`.
    #[must_use]
    pub const fn blend_mode(mut self, mode: BlendMode) -> Self {
        self.blend_mode = mode;
        self
    }

    /// Finish building the shape and add it to the canvas.
    #[must_use]
    pub fn done(self) -> PixelCanvas {
        let style = ShapeStyle {
            fill: self.fill,
            stroke: self.stroke,
            anti_alias: self.anti_alias,
            fill_rule: self.fill_rule,
            opacity: self.opacity,
            transform: self.shape_transform,
            blend_mode: self.blend_mode,
        };

        let cmd = match self.kind {
            ShapeKind::Circle { cx, cy, radius } => DrawCommand::Circle {
                cx,
                cy,
                radius,
                style,
            },
            ShapeKind::Rectangle {
                rect,
                corner_radius,
            } => DrawCommand::Rectangle {
                rect,
                corner_radius: self.corner_radius.unwrap_or(corner_radius),
                style,
            },
            ShapeKind::Ellipse {
                cx,
                cy,
                rx,
                ry,
                rotation,
            } => DrawCommand::Ellipse {
                cx,
                cy,
                rx,
                ry,
                rotation: self.rotation.unwrap_or(rotation),
                style,
            },
            ShapeKind::Path(path) => DrawCommand::Path { path, style },
            ShapeKind::Polyline { points, closed } => DrawCommand::Polyline {
                points,
                closed,
                style,
            },
            ShapeKind::Arc {
                cx,
                cy,
                radius,
                start_angle,
                sweep_angle,
            } => DrawCommand::Arc {
                cx,
                cy,
                radius,
                start_angle,
                sweep_angle,
                style,
            },
        };

        self.canvas.push(cmd)
    }
}

// ---------------------------------------------------------------------------
// LineBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for line drawing.
///
/// Finish building by calling [`.done()`](LineBuilder::done).
pub struct LineBuilder {
    canvas: PixelCanvas,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    stroke: StrokeStyle,
    anti_alias: bool,
}

impl LineBuilder {
    fn new(canvas: PixelCanvas, x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self {
            canvas,
            x1,
            y1,
            x2,
            y2,
            stroke: StrokeStyle::default(),
            anti_alias: true,
        }
    }

    /// Set both the stroke color and width at once.
    ///
    /// This mirrors [`ShapeBuilder::stroke`] for API consistency.
    #[must_use]
    pub const fn stroke(mut self, color: Color, width: f32) -> Self {
        self.stroke.color = color;
        self.stroke.width = width;
        self
    }

    /// Set the stroke color.
    #[must_use]
    pub const fn color(mut self, color: Color) -> Self {
        self.stroke.color = color;
        self
    }

    /// Set the stroke width.
    #[must_use]
    pub const fn width(mut self, width: f32) -> Self {
        self.stroke.width = width;
        self
    }

    /// Set the line cap style.
    #[must_use]
    pub const fn line_cap(mut self, cap: LineCap) -> Self {
        self.stroke.line_cap = cap;
        self
    }

    /// Set the line join style.
    #[must_use]
    pub const fn line_join(mut self, join: LineJoin) -> Self {
        self.stroke.line_join = join;
        self
    }

    /// Set a dash pattern.
    #[must_use]
    pub fn dash(mut self, pattern: DashPattern) -> Self {
        self.stroke.dash = Some(pattern);
        self
    }

    /// Set the miter limit for the stroke.
    #[must_use]
    pub const fn miter_limit(mut self, limit: f32) -> Self {
        self.stroke.miter_limit = limit;
        self
    }

    /// Set a linear gradient paint for the stroke.
    #[must_use]
    pub fn stroke_gradient(mut self, gradient: GradientDef) -> Self {
        self.stroke.paint = Some(FillStyle::LinearGradient(gradient));
        self
    }

    /// Set a radial gradient paint for the stroke.
    #[must_use]
    pub fn stroke_radial_gradient(mut self, gradient: GradientDef) -> Self {
        self.stroke.paint = Some(FillStyle::RadialGradient(gradient));
        self
    }

    /// Enable or disable anti-aliasing (default: enabled).
    #[must_use]
    pub const fn anti_alias(mut self, enabled: bool) -> Self {
        self.anti_alias = enabled;
        self
    }

    /// Finish building the line and add it to the canvas.
    #[must_use]
    pub fn done(self) -> PixelCanvas {
        self.canvas.push(DrawCommand::Line {
            x1: self.x1,
            y1: self.y1,
            x2: self.x2,
            y2: self.y2,
            stroke: self.stroke,
            anti_alias: self.anti_alias,
        })
    }
}

// ---------------------------------------------------------------------------
// GradientBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for gradient-filled rectangles.
///
/// Finish building by calling [`.done()`](GradientBuilder::done).
pub struct GradientBuilder {
    canvas: PixelCanvas,
    rect: Rect,
    stops: Vec<GradientStop>,
    kind: GradientKind,
    anti_alias: bool,
}

impl GradientBuilder {
    fn new(canvas: PixelCanvas, rect: Rect) -> Self {
        // Default: top-to-bottom linear gradient
        Self {
            canvas,
            kind: GradientKind::Linear {
                start: Point::new(rect.x, rect.y),
                end: Point::new(rect.x, rect.y + rect.height),
            },
            rect,
            stops: Vec::new(),
            anti_alias: true,
        }
    }

    /// Add a color stop at the given position (0.0–1.0).
    #[must_use]
    pub fn stop(mut self, position: f32, color: Color) -> Self {
        self.stops.push(GradientStop { position, color });
        self
    }

    /// Set the gradient direction to linear between two points.
    #[must_use]
    pub const fn linear(mut self, start: Point, end: Point) -> Self {
        self.kind = GradientKind::Linear { start, end };
        self
    }

    /// Set the gradient to radial from a center with a given radius.
    #[must_use]
    pub const fn radial(mut self, center: Point, radius: f32) -> Self {
        self.kind = GradientKind::Radial { center, radius };
        self
    }

    /// Enable or disable anti-aliasing (default: enabled).
    #[must_use]
    pub const fn anti_alias(mut self, enabled: bool) -> Self {
        self.anti_alias = enabled;
        self
    }

    /// Finish building the gradient and add it to the canvas.
    #[must_use]
    pub fn done(self) -> PixelCanvas {
        self.canvas.push(DrawCommand::Gradient {
            rect: self.rect,
            gradient: GradientDef {
                kind: self.kind,
                stops: self.stops,
            },
            anti_alias: self.anti_alias,
        })
    }
}

// ---------------------------------------------------------------------------
// GroupBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for grouped commands with a shared transform.
///
/// Finish building by calling [`.done()`](GroupBuilder::done).
pub struct GroupBuilder {
    parent: PixelCanvas,
    child: PixelCanvas,
    transform: Transform,
    clip: Option<ClipRegion>,
    opacity: f32,
    blend_mode: BlendMode,
}

impl GroupBuilder {
    fn new(parent: PixelCanvas, transform: Transform) -> Self {
        let child = PixelCanvas::new(parent.width, parent.height);
        Self {
            parent,
            child,
            transform,
            clip: None,
            opacity: 1.0,
            blend_mode: BlendMode::SrcOver,
        }
    }

    /// Access the child canvas to add commands to the group.
    ///
    /// This consumes the inner canvas and returns a new builder with
    /// the updated canvas.
    #[must_use]
    pub fn canvas(mut self, build_fn: impl FnOnce(PixelCanvas) -> PixelCanvas) -> Self {
        self.child = build_fn(self.child);
        self
    }

    /// Clip all children to the given rectangle.
    #[must_use]
    pub fn clip_rect(mut self, rect: Rect) -> Self {
        self.clip = Some(ClipRegion::Rect(rect));
        self
    }

    /// Clip all children to the given path.
    #[must_use]
    pub fn clip_path(mut self, path: tiny_skia::Path) -> Self {
        self.clip = Some(ClipRegion::Path(PathData::new(path)));
        self
    }

    /// Set group opacity (0.0 = transparent, 1.0 = opaque). Default: 1.0.
    #[must_use]
    pub const fn opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }

    /// Set the blend mode for compositing the group. Default: `SrcOver`.
    #[must_use]
    pub const fn blend_mode(mut self, mode: BlendMode) -> Self {
        self.blend_mode = mode;
        self
    }

    /// Finish building the group and add it to the parent canvas.
    #[must_use]
    pub fn done(self) -> PixelCanvas {
        self.parent.push(DrawCommand::Group {
            commands: self.child.into_commands(),
            transform: self.transform,
            clip: self.clip,
            opacity: self.opacity,
            blend_mode: self.blend_mode,
        })
    }
}

// ---------------------------------------------------------------------------
// ImageBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for image blitting.
///
/// Finish building by calling [`.done()`](ImageBuilder::done).
pub struct ImageBuilder {
    canvas: PixelCanvas,
    image: ImageData,
    x: f32,
    y: f32,
    opacity: f32,
}

impl ImageBuilder {
    const fn new(canvas: PixelCanvas, image: ImageData, x: f32, y: f32) -> Self {
        Self {
            canvas,
            image,
            x,
            y,
            opacity: 1.0,
        }
    }

    /// Set the opacity (0.0 = transparent, 1.0 = opaque). Default: 1.0.
    #[must_use]
    pub const fn opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }

    /// Finish building the image command and add it to the canvas.
    #[must_use]
    pub fn done(self) -> PixelCanvas {
        self.canvas.push(DrawCommand::Image {
            image: self.image,
            x: self.x,
            y: self.y,
            opacity: self.opacity,
        })
    }
}

// ---------------------------------------------------------------------------
// TextBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for text rendering.
///
/// Finish building by calling [`.done()`](TextBuilder::done).
#[cfg(feature = "text")]
pub struct TextBuilder {
    canvas: PixelCanvas,
    text: String,
    x: f32,
    y: f32,
    font_size: f32,
    color: Color,
    font_data: Option<FontData>,
    align: TextAlign,
    outline_color: Option<Color>,
    outline_width: Option<f32>,
    fill_style: Option<FillStyle>,
    shadow: Option<crate::scene::command::TextShadow>,
}

#[cfg(feature = "text")]
impl TextBuilder {
    const fn new(canvas: PixelCanvas, text: String, x: f32, y: f32) -> Self {
        Self {
            canvas,
            text,
            x,
            y,
            font_size: 16.0,
            color: Color::WHITE,
            font_data: None,
            align: TextAlign::Left,
            outline_color: None,
            outline_width: None,
            fill_style: None,
            shadow: None,
        }
    }

    /// Set the font size in pixels. Default: 16.0.
    #[must_use]
    pub const fn size(mut self, size: f32) -> Self {
        self.font_size = size;
        self
    }

    /// Set the text color. Default: white.
    #[must_use]
    pub const fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Set the font data (TTF/OTF bytes).
    ///
    /// If not called, the embedded default font (Inter-Regular) is used.
    #[must_use]
    pub fn font(mut self, font_data: FontData) -> Self {
        self.font_data = Some(font_data);
        self
    }

    /// Set the horizontal text alignment. Default: `Left`.
    #[must_use]
    pub const fn align(mut self, align: TextAlign) -> Self {
        self.align = align;
        self
    }

    /// Set a text outline (rendered as multi-offset copies behind the fill).
    #[must_use]
    pub const fn outline(mut self, color: Color, width: f32) -> Self {
        self.outline_color = Some(color);
        self.outline_width = Some(width);
        self
    }

    /// Set a drop shadow.
    #[must_use]
    pub const fn shadow(mut self, color: Color, offset_x: f32, offset_y: f32) -> Self {
        self.shadow = Some(crate::scene::command::TextShadow {
            offset_x,
            offset_y,
            color,
            blur_radius: 0.0,
        });
        self
    }

    /// Set a drop shadow with blur.
    #[must_use]
    pub const fn shadow_blurred(
        mut self,
        color: Color,
        offset_x: f32,
        offset_y: f32,
        blur_radius: f32,
    ) -> Self {
        self.shadow = Some(crate::scene::command::TextShadow {
            offset_x,
            offset_y,
            color,
            blur_radius,
        });
        self
    }

    /// Set a linear gradient fill for the text (overrides `.color()`).
    #[must_use]
    pub fn fill_linear_gradient(
        mut self,
        start: Point,
        end: Point,
        stops: &[(f32, Color)],
    ) -> Self {
        self.fill_style = Some(FillStyle::LinearGradient(GradientDef {
            kind: GradientKind::Linear { start, end },
            stops: stops
                .iter()
                .map(|(pos, color)| GradientStop {
                    position: *pos,
                    color: *color,
                })
                .collect(),
        }));
        self
    }

    /// Set a radial gradient fill for the text (overrides `.color()`).
    #[must_use]
    pub fn fill_radial_gradient(
        mut self,
        center: Point,
        radius: f32,
        stops: &[(f32, Color)],
    ) -> Self {
        self.fill_style = Some(FillStyle::RadialGradient(GradientDef {
            kind: GradientKind::Radial { center, radius },
            stops: stops
                .iter()
                .map(|(pos, color)| GradientStop {
                    position: *pos,
                    color: *color,
                })
                .collect(),
        }));
        self
    }

    /// Apply a pre-built [`TextStyle`] (font, size, color, alignment).
    #[must_use]
    pub fn style(mut self, style: &TextStyle) -> Self {
        self.font_data = style.font_data.clone();
        self.font_size = style.font_size;
        self.color = style.color;
        self.align = style.align;
        self
    }

    /// Finish building the text command and add it to the canvas.
    ///
    /// Falls back to the embedded default font if no font was set via `.font()`.
    #[must_use]
    pub fn done(self) -> PixelCanvas {
        let font_data = self
            .font_data
            .unwrap_or_else(crate::rasterize::skia::text::default_font);
        self.canvas.push(DrawCommand::Text {
            text: self.text,
            x: self.x,
            y: self.y,
            font_size: self.font_size,
            color: self.color,
            font_data,
            align: self.align,
            outline_color: self.outline_color,
            outline_width: self.outline_width,
            fill_style: self.fill_style,
            shadow: self.shadow,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_canvas_has_no_commands() {
        let canvas = PixelCanvas::new(100, 100);
        assert!(canvas.commands().is_empty());
        assert_eq!(canvas.width(), 100);
        assert_eq!(canvas.height(), 100);
    }

    #[test]
    fn circle_builder_produces_command() {
        let canvas = PixelCanvas::new(200, 200)
            .circle(100.0, 100.0, 50.0)
            .fill(Color::RED)
            .done();

        assert_eq!(canvas.commands().len(), 1);
        assert!(matches!(canvas.commands()[0], DrawCommand::Circle { .. }));
    }

    #[test]
    fn line_builder_produces_command() {
        let canvas = PixelCanvas::new(200, 200)
            .line(0.0, 0.0, 100.0, 100.0)
            .color(Color::GREEN)
            .width(2.0)
            .done();

        assert_eq!(canvas.commands().len(), 1);
        assert!(matches!(canvas.commands()[0], DrawCommand::Line { .. }));
    }

    #[test]
    fn chained_commands() {
        let canvas = PixelCanvas::new(200, 200)
            .background(Color::BLACK)
            .circle(50.0, 50.0, 20.0)
            .fill(Color::RED)
            .done()
            .rect(10.0, 10.0, 80.0, 80.0)
            .fill(Color::BLUE)
            .stroke(Color::WHITE, 1.0)
            .done()
            .line(0.0, 0.0, 200.0, 200.0)
            .color(Color::GREEN)
            .done();

        assert_eq!(canvas.commands().len(), 3);
        assert_eq!(canvas.background_color(), Color::BLACK);
    }

    #[test]
    fn content_hash_changes_with_commands() {
        let a = PixelCanvas::new(100, 100)
            .circle(50.0, 50.0, 25.0)
            .fill(Color::RED)
            .done();

        let b = PixelCanvas::new(100, 100)
            .circle(50.0, 50.0, 30.0)
            .fill(Color::RED)
            .done();

        assert_ne!(a.content_hash(), b.content_hash());
    }

    #[test]
    fn content_hash_stable_for_identical_scenes() {
        let build = || {
            PixelCanvas::new(100, 100)
                .background(Color::BLACK)
                .circle(50.0, 50.0, 25.0)
                .fill(Color::RED)
                .done()
        };

        assert_eq!(build().content_hash(), build().content_hash());
    }

    #[test]
    fn gradient_builder() {
        let canvas = PixelCanvas::new(200, 200)
            .gradient(0.0, 0.0, 200.0, 200.0)
            .stop(0.0, Color::RED)
            .stop(1.0, Color::BLUE)
            .done();

        assert_eq!(canvas.commands().len(), 1);
        assert!(matches!(canvas.commands()[0], DrawCommand::Gradient { .. }));
    }

    #[test]
    fn rect_with_corner_radius() {
        let canvas = PixelCanvas::new(200, 200)
            .rect(10.0, 10.0, 100.0, 50.0)
            .fill(Color::WHITE)
            .corner_radius(8.0)
            .done();

        assert_eq!(canvas.commands().len(), 1);
        if let DrawCommand::Rectangle { corner_radius, .. } = &canvas.commands()[0] {
            assert!((*corner_radius - 8.0).abs() < f32::EPSILON);
        } else {
            panic!("Expected Rectangle command");
        }
    }

    #[test]
    fn gpu_suitable_pure_shapes() {
        let canvas = PixelCanvas::new(100, 100)
            .circle(50.0, 50.0, 20.0)
            .fill(Color::RED)
            .done()
            .rect(10.0, 10.0, 30.0, 30.0)
            .fill(Color::BLUE)
            .done()
            .line(0.0, 0.0, 100.0, 100.0)
            .color(Color::WHITE)
            .done();
        assert!(canvas.gpu_suitable());
    }

    #[test]
    fn gpu_suitable_true_with_solid_fill_path() {
        let mut canvas = PixelCanvas::new(100, 100);
        let mut pb = tiny_skia::PathBuilder::new();
        pb.move_to(10.0, 10.0);
        pb.line_to(90.0, 10.0);
        pb.line_to(90.0, 90.0);
        pb.close();
        if let Some(path) = pb.finish() {
            canvas.push_command(crate::scene::command::DrawCommand::Path {
                path: crate::scene::command::PathData::new(path),
                style: crate::scene::style::ShapeStyle {
                    fill: Some(crate::scene::style::FillStyle::Solid(Color::RED)),
                    ..crate::scene::style::ShapeStyle::default()
                },
            });
        }
        assert!(canvas.gpu_suitable());
    }

    #[test]
    fn gpu_suitable_false_with_stroked_path() {
        let mut canvas = PixelCanvas::new(100, 100);
        let mut pb = tiny_skia::PathBuilder::new();
        pb.move_to(0.0, 0.0);
        pb.line_to(100.0, 100.0);
        if let Some(path) = pb.finish() {
            canvas.push_command(crate::scene::command::DrawCommand::Path {
                path: crate::scene::command::PathData::new(path),
                style: crate::scene::style::ShapeStyle {
                    stroke: Some(crate::scene::style::StrokeStyle::default()),
                    ..crate::scene::style::ShapeStyle::default()
                },
            });
        }
        assert!(!canvas.gpu_suitable());
    }

    #[test]
    fn gpu_suitable_false_with_arc() {
        let canvas = PixelCanvas::new(100, 100)
            .arc(50.0, 50.0, 30.0, 0.0, 1.5)
            .stroke(Color::RED, 2.0)
            .done();
        assert!(!canvas.gpu_suitable());
    }

    #[test]
    fn gpu_suitable_false_with_transform_group() {
        let canvas = PixelCanvas::new(100, 100)
            .group(crate::scene::style::Transform::rotate_at(0.5, 50.0, 50.0))
            .canvas(|inner| inner.rect(20.0, 20.0, 60.0, 60.0).fill(Color::GREEN).done())
            .done();
        assert!(!canvas.gpu_suitable());
    }
}
