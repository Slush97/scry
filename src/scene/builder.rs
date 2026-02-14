//! Fluent builder API for constructing a [`PixelCanvas`] scene.
//!
//! The builder pattern lets users construct drawing scenes without needing to
//! know `tiny-skia` internals:
//!
//! ```
//! use ratatui_pixelcanvas::scene::PixelCanvas;
//! use ratatui_pixelcanvas::scene::style::Color;
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

#[cfg(feature = "text")]
use crate::scene::command::FontData;
use crate::scene::command::{DrawCommand, ImageData, PathData};
use crate::scene::style::{
    BlendMode, ClipRegion, Color, DashPattern, FillStyle, GradientDef, GradientKind, GradientStop,
    LineCap, LineJoin, Point, Rect, ShapeStyle, StrokeStyle, Transform,
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
}

impl PixelCanvas {
    /// Create a new canvas with the given pixel dimensions.
    ///
    /// The canvas starts empty with a transparent background.
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            commands: Vec::new(),
            background: Color::TRANSPARENT,
            width,
            height,
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
    /// use ratatui_pixelcanvas::scene::PixelCanvas;
    /// use ratatui_pixelcanvas::scene::command::DrawCommand;
    /// use ratatui_pixelcanvas::scene::style::{Color, ShapeStyle, FillStyle};
    ///
    /// let mut canvas = PixelCanvas::new(200, 200);
    /// let show_border = true;
    /// if show_border {
    ///     canvas.push_command(DrawCommand::Circle {
    ///         cx: 100.0, cy: 100.0, radius: 50.0,
    ///         style: ShapeStyle { fill: Some(FillStyle::Solid(Color::RED)), stroke: None, anti_alias: true },
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
    pub const fn group(self, transform: Transform) -> GroupBuilder {
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
    #[cfg(feature = "text")]
    #[must_use]
    pub fn text(self, text: &str, x: f32, y: f32) -> TextBuilder {
        TextBuilder::new(self, text.to_string(), x, y)
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

    /// Push a command internally (used by builders).
    fn push(mut self, cmd: DrawCommand) -> Self {
        self.commands.push(cmd);
        self
    }
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

    /// Finish building the shape and add it to the canvas.
    #[must_use]
    pub fn done(self) -> PixelCanvas {
        let style = ShapeStyle {
            fill: self.fill,
            stroke: self.stroke,
            anti_alias: self.anti_alias,
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
    const fn new(parent: PixelCanvas, transform: Transform) -> Self {
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
    #[must_use]
    pub fn font(mut self, font_data: FontData) -> Self {
        self.font_data = Some(font_data);
        self
    }

    /// Finish building the text command and add it to the canvas.
    ///
    /// # Panics
    ///
    /// Panics if no font data was set via `.font()`.
    #[must_use]
    pub fn done(self) -> PixelCanvas {
        let font_data = self
            .font_data
            .expect("TextBuilder requires .font() to be called before .done()");
        self.canvas.push(DrawCommand::Text {
            text: self.text,
            x: self.x,
            y: self.y,
            font_size: self.font_size,
            color: self.color,
            font_data,
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
}
