// SPDX-License-Identifier: MIT OR Apache-2.0
//! Drawing commands that form the scene display list.
//!
//! Each variant of [`DrawCommand`] represents a single drawing instruction.
//! Commands are collected into a list by [`PixelCanvas`](crate::scene::PixelCanvas)
//! and later rasterized into a pixel buffer.

use std::cell::OnceCell;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::scene::style::{
    BlendMode, ClipRegion, Color, GradientDef, Rect, ShapeStyle, StrokeStyle, Transform,
};

// ---------------------------------------------------------------------------
// Serialized path (for hashing)
// ---------------------------------------------------------------------------

/// A wrapper around `tiny_skia::Path` that supports `Hash` and `Eq`.
///
/// Since `tiny_skia::Path` does not implement `Hash`, we lazily compute a
/// hash from the full path geometry (verbs + control points) on first access.
/// This avoids the overhead of serializing the path at construction time —
/// if the scene is never hashed (e.g., caching is disabled), no work is done.
#[derive(Clone, Debug)]
pub struct PathData {
    /// The underlying tiny-skia path.
    path: tiny_skia::Path,
    /// Lazily computed hash of the path geometry.
    cached_hash: OnceCell<u64>,
}

impl PathData {
    /// Create a `PathData` from a `tiny_skia::Path`.
    ///
    /// No hashing work is done at construction time — the geometry hash is
    /// computed lazily on first call to `Hash::hash()`.
    #[must_use]
    pub const fn new(path: tiny_skia::Path) -> Self {
        Self {
            path,
            cached_hash: OnceCell::new(),
        }
    }

    /// Create a `PathData` from a shared `Arc<tiny_skia::Path>`.
    ///
    /// If the `Arc` has a single strong reference, the path is unwrapped
    /// in-place (zero-cost move). Otherwise, it falls back to a deep clone.
    /// This is the preferred constructor in animation hot loops where paths
    /// are stored as `Arc` for reuse across frames.
    #[must_use]
    pub fn from_shared(arc: Arc<tiny_skia::Path>) -> Self {
        let path = Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone());
        Self {
            path,
            cached_hash: OnceCell::new(),
        }
    }

    /// Get a reference to the underlying `tiny_skia::Path`.
    #[must_use]
    pub const fn path(&self) -> &tiny_skia::Path {
        &self.path
    }

    /// Compute and cache the geometry hash.
    fn compute_hash(&self) -> u64 {
        *self.cached_hash.get_or_init(|| {
            use std::hash::DefaultHasher;
            let mut hasher = DefaultHasher::new();
            for verb in self.path.verbs() {
                (*verb as u8).hash(&mut hasher);
            }
            for point in self.path.points() {
                point.x.to_bits().hash(&mut hasher);
                point.y.to_bits().hash(&mut hasher);
            }
            hasher.finish()
        })
    }
}

/// Note: equality is based on geometry hash plus verb/point counts.
/// False positives are theoretically possible but astronomically unlikely.
impl PartialEq for PathData {
    fn eq(&self, other: &Self) -> bool {
        self.path.verbs().len() == other.path.verbs().len()
            && self.path.points().len() == other.path.points().len()
            && self.compute_hash() == other.compute_hash()
    }
}

impl Eq for PathData {}

impl Hash for PathData {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.compute_hash().hash(state);
    }
}

// ---------------------------------------------------------------------------
// Image data (for hashing)
// ---------------------------------------------------------------------------

/// Raw RGBA image data for blitting onto the canvas.
///
/// Stores pixel data as a flat RGBA byte buffer alongside dimensions.
/// Supports `Hash` and `Eq` by hashing the raw bytes.
#[derive(Clone, Debug)]
pub struct ImageData {
    /// Width in pixels.
    width: u32,
    /// Height in pixels.
    height: u32,
    /// Raw RGBA pixel data (4 bytes per pixel, row-major).
    data: Vec<u8>,
}

impl ImageData {
    /// Create a new `ImageData` from raw RGBA bytes.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != width * height * 4`.
    #[must_use]
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        assert_eq!(
            data.len(),
            (width as usize) * (height as usize) * 4,
            "RGBA data length must equal width * height * 4"
        );
        Self {
            width,
            height,
            data,
        }
    }

    /// Width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// The raw RGBA pixel data.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }
}

impl PartialEq for ImageData {
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width && self.height == other.height && self.data == other.data
    }
}

impl Eq for ImageData {}

impl Hash for ImageData {
    /// Sampled hash: dimensions + first/middle/last 64 bytes.
    ///
    /// This is O(1) instead of O(pixels), which matters for large images
    /// (e.g., a 1920×1080 image has 8MB of RGBA data). The sampling provides
    /// strong-enough identity for caching purposes — collisions require two
    /// different images with identical dimensions AND identical bytes at all
    /// three sample positions.
    fn hash<H: Hasher>(&self, state: &mut H) {
        const SAMPLE: usize = 64;

        self.width.hash(state);
        self.height.hash(state);
        self.data.len().hash(state);

        let len = self.data.len();

        // First SAMPLE bytes
        let head = len.min(SAMPLE);
        self.data[..head].hash(state);

        // Middle SAMPLE bytes
        if len > SAMPLE * 2 {
            let mid_start = len / 2 - SAMPLE / 2;
            self.data[mid_start..mid_start + SAMPLE].hash(state);
        }

        // Last SAMPLE bytes
        if len > SAMPLE {
            self.data[len - SAMPLE..].hash(state);
        }
    }
}

// ---------------------------------------------------------------------------
// Text shadow
// ---------------------------------------------------------------------------

/// Shadow effect for text rendering.
#[cfg(feature = "text")]
#[derive(Clone, Debug, PartialEq)]
pub struct TextShadow {
    /// Horizontal offset in pixels (positive = right).
    pub offset_x: f32,
    /// Vertical offset in pixels (positive = down).
    pub offset_y: f32,
    /// Shadow color (typically semi-transparent black).
    pub color: crate::scene::style::Color,
    /// Blur radius (0.0 = sharp shadow). Approximated via box-downsample.
    pub blur_radius: f32,
}

#[cfg(feature = "text")]
impl Eq for TextShadow {}

#[cfg(feature = "text")]
impl Hash for TextShadow {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.offset_x.to_bits().hash(state);
        self.offset_y.to_bits().hash(state);
        self.color.hash(state);
        self.blur_radius.to_bits().hash(state);
    }
}

// ---------------------------------------------------------------------------
// Text alignment
// ---------------------------------------------------------------------------

/// Horizontal alignment for text rendering.
#[cfg(feature = "text")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TextAlign {
    /// Left-aligned (default). The x coordinate is the left edge.
    #[default]
    Left,
    /// Center-aligned. The x coordinate is the center of the text.
    Center,
    /// Right-aligned. The x coordinate is the right edge.
    Right,
}

// ---------------------------------------------------------------------------
// Font data (for text rendering)
// ---------------------------------------------------------------------------

/// Shared font data for text rendering.
///
/// Wraps font file bytes in an `Arc` for cheap cloning. Equality and hashing
/// are based on pointer identity — two `FontData` values are equal only if
/// they point to the same allocation.
#[cfg(feature = "text")]
#[derive(Clone, Debug)]
pub struct FontData(Arc<Vec<u8>>);

#[cfg(feature = "text")]
impl FontData {
    /// Create a new `FontData` from TTF or OTF font file bytes.
    #[must_use]
    pub fn new(bytes: Vec<u8>) -> Self {
        Self(Arc::new(bytes))
    }

    /// Get a reference to the raw font bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.0
    }

    /// Get the pointer identity of the underlying `Arc` allocation.
    /// Used as a cache key for the thread-local font cache.
    #[must_use]
    pub fn arc_ptr(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }
}

#[cfg(feature = "text")]
impl PartialEq for FontData {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

#[cfg(feature = "text")]
impl Eq for FontData {}

#[cfg(feature = "text")]
impl Hash for FontData {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash by pointer identity for performance
        Arc::as_ptr(&self.0).hash(state);
    }
}

// ---------------------------------------------------------------------------
// Text metrics
// ---------------------------------------------------------------------------

/// Measurement results from [`measure_text`](crate::rasterize::skia::text::measure_text).
#[cfg(feature = "text")]
#[derive(Clone, Copy, Debug)]
pub struct TextMetrics {
    /// Total advance width of the text string in pixels.
    pub width: f32,
    /// Font height (ascent + descent) in pixels.
    pub height: f32,
    /// Distance from the baseline to the top of the tallest glyph.
    pub ascent: f32,
    /// Distance from the baseline to the bottom of the lowest glyph (positive downward).
    pub descent: f32,
}

// ---------------------------------------------------------------------------
// Text style (convenience bundle)
// ---------------------------------------------------------------------------

/// A reusable bundle of text rendering parameters.
///
/// Use with [`PixelCanvas::text_styled`] to avoid repeating font/size/color
/// on every text call.
#[cfg(feature = "text")]
#[derive(Clone, Debug)]
pub struct TextStyle {
    /// Font data (TTF/OTF bytes). Uses the embedded default if `None`.
    pub font_data: Option<FontData>,
    /// Font size in pixels.
    pub font_size: f32,
    /// Text color.
    pub color: Color,
    /// Horizontal alignment.
    pub align: TextAlign,
}

#[cfg(feature = "text")]
impl Default for TextStyle {
    fn default() -> Self {
        Self {
            font_data: None,
            font_size: 16.0,
            color: Color::WHITE,
            align: TextAlign::Left,
        }
    }
}

// ---------------------------------------------------------------------------
// SDF scene reference (for hashing)
// ---------------------------------------------------------------------------

/// A reference to an [`SdfScene`](crate::sdf::SdfScene) with a generation counter.
///
/// Because `SdfScene` doesn't implement `Hash` or `Eq`, we use a globally unique
/// generation counter for identity. Each `SdfSceneRef` created via [`new()`](Self::new)
/// receives a unique generation, so two distinct scenes will never collide.
#[cfg(feature = "sdf")]
#[derive(Clone, Debug)]
pub struct SdfSceneRef {
    /// The shared scene.
    scene: Arc<crate::sdf::SdfScene>,
    /// Globally unique generation counter.
    generation: u64,
}

#[cfg(feature = "sdf")]
impl SdfSceneRef {
    /// Global monotonic counter for unique generation IDs.
    fn next_generation() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT_GEN: AtomicU64 = AtomicU64::new(1);
        NEXT_GEN.fetch_add(1, Ordering::Relaxed)
    }

    /// Create a new scene reference with an auto-assigned unique generation.
    #[must_use]
    pub fn new(scene: Arc<crate::sdf::SdfScene>) -> Self {
        Self {
            scene,
            generation: Self::next_generation(),
        }
    }

    /// Create a new scene reference with an explicit generation counter.
    ///
    /// Use this when you need deterministic generation values (e.g., for tests
    /// or when synchronizing with external state).
    #[must_use]
    pub fn with_generation(scene: Arc<crate::sdf::SdfScene>, generation: u64) -> Self {
        Self { scene, generation }
    }

    /// Get a reference to the underlying scene.
    #[must_use]
    pub fn scene(&self) -> &crate::sdf::SdfScene {
        &self.scene
    }

    /// Get the generation counter.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

#[cfg(feature = "sdf")]
impl PartialEq for SdfSceneRef {
    fn eq(&self, other: &Self) -> bool {
        self.generation == other.generation
    }
}

#[cfg(feature = "sdf")]
impl Eq for SdfSceneRef {}

#[cfg(feature = "sdf")]
impl Hash for SdfSceneRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.generation.hash(state);
    }
}

// ---------------------------------------------------------------------------
// Draw commands
// ---------------------------------------------------------------------------

/// A single drawing instruction in the scene display list.
///
/// Commands are created by the fluent API on [`PixelCanvas`](crate::scene::PixelCanvas)
/// and consumed by the rasterizer.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum DrawCommand {
    /// Fill the entire canvas with a solid color.
    Clear {
        /// The color to fill with.
        color: Color,
    },

    /// Draw a circle.
    Circle {
        /// Center X coordinate.
        cx: f32,
        /// Center Y coordinate.
        cy: f32,
        /// Radius.
        radius: f32,
        /// Fill and stroke style.
        style: ShapeStyle,
    },

    /// Draw an axis-aligned rectangle, optionally with rounded corners.
    Rectangle {
        /// Bounding rectangle.
        rect: Rect,
        /// Corner radius for rounded rectangles. 0.0 = sharp corners.
        corner_radius: f32,
        /// Fill and stroke style.
        style: ShapeStyle,
    },

    /// Draw an ellipse (or rotated ellipse).
    Ellipse {
        /// Center X coordinate.
        cx: f32,
        /// Center Y coordinate.
        cy: f32,
        /// Horizontal radius.
        rx: f32,
        /// Vertical radius.
        ry: f32,
        /// Rotation angle in radians.
        rotation: f32,
        /// Fill and stroke style.
        style: ShapeStyle,
    },

    /// Draw a straight line between two points.
    Line {
        /// Start X.
        x1: f32,
        /// Start Y.
        y1: f32,
        /// End X.
        x2: f32,
        /// End Y.
        y2: f32,
        /// Stroke style.
        stroke: StrokeStyle,
        /// Whether anti-aliasing is enabled.
        anti_alias: bool,
    },

    /// Draw an arbitrary Bézier path.
    Path {
        /// The path data.
        path: PathData,
        /// Fill and stroke style.
        style: ShapeStyle,
    },

    /// Draw a polyline (connected line segments) or polygon.
    Polyline {
        /// Ordered list of (x, y) vertices.
        points: Vec<(f32, f32)>,
        /// Whether to close the path (polygon) or leave it open (polyline).
        closed: bool,
        /// Fill and stroke style.
        style: ShapeStyle,
    },

    /// Fill a rectangle with a gradient.
    Gradient {
        /// Bounding rectangle.
        rect: Rect,
        /// Gradient definition.
        gradient: GradientDef,
        /// Whether anti-aliasing is enabled.
        anti_alias: bool,
    },

    /// Draw a circular arc (partial circle).
    Arc {
        /// Center X coordinate.
        cx: f32,
        /// Center Y coordinate.
        cy: f32,
        /// Radius.
        radius: f32,
        /// Start angle in radians (0 = 3 o'clock, counter-clockwise).
        start_angle: f32,
        /// Sweep angle in radians (positive = counter-clockwise).
        sweep_angle: f32,
        /// Fill and stroke style.
        style: ShapeStyle,
    },

    /// Draw a raster image at the given position.
    Image {
        /// The image pixel data.
        image: ImageData,
        /// Destination X coordinate.
        x: f32,
        /// Destination Y coordinate.
        y: f32,
        /// Opacity (0.0 = fully transparent, 1.0 = fully opaque).
        opacity: f32,
    },

    /// Draw text at the given position.
    #[cfg(feature = "text")]
    Text {
        /// The text string to render.
        text: String,
        /// Baseline X coordinate.
        x: f32,
        /// Baseline Y coordinate.
        y: f32,
        /// Font size in pixels.
        font_size: f32,
        /// Text color (used when `fill_style` is `None`).
        color: Color,
        /// Font data (TTF/OTF bytes).
        font_data: FontData,
        /// Horizontal text alignment. Default: `Left`.
        align: TextAlign,
        /// Rotation in degrees (0 = horizontal, positive = counter-clockwise).
        /// The text is rotated around its anchor point (alignment-dependent).
        rotation: f32,
        /// Optional outline color.
        outline_color: Option<Color>,
        /// Outline width in pixels.
        outline_width: Option<f32>,
        /// Optional gradient or solid fill style (overrides `color` when set).
        fill_style: Option<crate::scene::style::FillStyle>,
        /// Optional drop shadow.
        shadow: Option<TextShadow>,
    },

    /// Render an SDF 3D scene into a rectangular region.
    ///
    /// The SDF ray marcher renders the scene into a pixmap at the given
    /// dimensions and composites it at `(rect.x, rect.y)`.
    #[cfg(feature = "sdf")]
    Sdf3D {
        /// The shared SDF scene with generation counter.
        scene: SdfSceneRef,
        /// Destination rectangle (position + render dimensions).
        rect: crate::scene::style::Rect,
        /// Animation time parameter passed to the renderer.
        time: f32,
        /// Render scale factor (0.1–1.0). `None` = full resolution.
        /// When set, renders at reduced resolution and bicubic-upscales.
        render_scale: Option<f32>,
    },

    /// A group of commands with a shared transform.
    Group {
        /// Child commands.
        commands: Vec<Self>,
        /// Transform applied to all children.
        transform: Transform,
        /// Optional clipping region.
        clip: Option<ClipRegion>,
        /// Group opacity (0.0–1.0). Default: 1.0.
        opacity: f32,
        /// Blend mode for compositing the group. Default: `SrcOver`.
        blend_mode: BlendMode,
    },
}

impl Eq for DrawCommand {}

impl Hash for DrawCommand {
    #[allow(clippy::too_many_lines)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Self::Clear { color } => color.hash(state),
            Self::Circle {
                cx,
                cy,
                radius,
                style,
            } => {
                cx.to_bits().hash(state);
                cy.to_bits().hash(state);
                radius.to_bits().hash(state);
                style.hash(state);
            }
            Self::Rectangle {
                rect,
                corner_radius,
                style,
            } => {
                rect.hash(state);
                corner_radius.to_bits().hash(state);
                style.hash(state);
            }
            Self::Ellipse {
                cx,
                cy,
                rx,
                ry,
                rotation,
                style,
            } => {
                cx.to_bits().hash(state);
                cy.to_bits().hash(state);
                rx.to_bits().hash(state);
                ry.to_bits().hash(state);
                rotation.to_bits().hash(state);
                style.hash(state);
            }
            Self::Line {
                x1,
                y1,
                x2,
                y2,
                stroke,
                anti_alias,
            } => {
                x1.to_bits().hash(state);
                y1.to_bits().hash(state);
                x2.to_bits().hash(state);
                y2.to_bits().hash(state);
                stroke.hash(state);
                anti_alias.hash(state);
            }
            Self::Path { path, style } => {
                path.hash(state);
                style.hash(state);
            }
            Self::Polyline {
                points,
                closed,
                style,
            } => {
                for (x, y) in points {
                    x.to_bits().hash(state);
                    y.to_bits().hash(state);
                }
                closed.hash(state);
                style.hash(state);
            }
            Self::Gradient {
                rect,
                gradient,
                anti_alias,
            } => {
                rect.hash(state);
                gradient.hash(state);
                anti_alias.hash(state);
            }
            Self::Group {
                commands,
                transform,
                clip,
                opacity,
                blend_mode,
            } => {
                commands.hash(state);
                transform.hash(state);
                clip.hash(state);
                opacity.to_bits().hash(state);
                blend_mode.hash(state);
            }
            Self::Arc {
                cx,
                cy,
                radius,
                start_angle,
                sweep_angle,
                style,
            } => {
                cx.to_bits().hash(state);
                cy.to_bits().hash(state);
                radius.to_bits().hash(state);
                start_angle.to_bits().hash(state);
                sweep_angle.to_bits().hash(state);
                style.hash(state);
            }
            Self::Image {
                image,
                x,
                y,
                opacity,
            } => {
                image.hash(state);
                x.to_bits().hash(state);
                y.to_bits().hash(state);
                opacity.to_bits().hash(state);
            }
            #[cfg(feature = "text")]
            Self::Text {
                text,
                x,
                y,
                font_size,
                color,
                font_data,
                align,
                rotation,
                outline_color,
                outline_width,
                fill_style,
                shadow,
            } => {
                text.hash(state);
                x.to_bits().hash(state);
                y.to_bits().hash(state);
                font_size.to_bits().hash(state);
                color.hash(state);
                font_data.hash(state);
                align.hash(state);
                rotation.to_bits().hash(state);
                outline_color.hash(state);
                outline_width.map(f32::to_bits).hash(state);
                fill_style.hash(state);
                shadow.hash(state);
            }
            #[cfg(feature = "sdf")]
            Self::Sdf3D {
                scene,
                rect,
                time,
                render_scale,
            } => {
                scene.hash(state);
                rect.hash(state);
                time.to_bits().hash(state);
                render_scale.map(f32::to_bits).hash(state);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;

    use super::*;

    fn hash_of(cmd: &DrawCommand) -> u64 {
        let mut hasher = DefaultHasher::new();
        cmd.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn same_commands_produce_same_hash() {
        let a = DrawCommand::Circle {
            cx: 50.0,
            cy: 50.0,
            radius: 25.0,
            style: ShapeStyle::default(),
        };
        let b = a.clone();
        assert_eq!(hash_of(&a), hash_of(&b));
    }

    #[test]
    fn different_commands_produce_different_hash() {
        let a = DrawCommand::Circle {
            cx: 50.0,
            cy: 50.0,
            radius: 25.0,
            style: ShapeStyle::default(),
        };
        let b = DrawCommand::Circle {
            cx: 51.0,
            cy: 50.0,
            radius: 25.0,
            style: ShapeStyle::default(),
        };
        assert_ne!(hash_of(&a), hash_of(&b));
    }

    #[test]
    fn clear_and_circle_are_different() {
        let a = DrawCommand::Clear {
            color: Color::BLACK,
        };
        let b = DrawCommand::Circle {
            cx: 0.0,
            cy: 0.0,
            radius: 0.0,
            style: ShapeStyle::default(),
        };
        assert_ne!(hash_of(&a), hash_of(&b));
    }
}
