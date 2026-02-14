//! Style primitives for drawing commands.
//!
//! This module defines the color, fill, stroke, and gradient types used to
//! describe how shapes are rendered on a [`PixelCanvas`](crate::scene::PixelCanvas).

use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// Color
// ---------------------------------------------------------------------------

/// A 32-bit RGBA color stored as straight (non-premultiplied) floating-point
/// components.
///
/// Premultiplication is handled automatically when converting to `tiny-skia`
/// types via [`to_tiny_skia()`](Color::to_tiny_skia).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    /// Red channel (0.0–1.0).
    pub r: f32,
    /// Green channel (0.0–1.0).
    pub g: f32,
    /// Blue channel (0.0–1.0).
    pub b: f32,
    /// Alpha channel (0.0–1.0).
    pub a: f32,
}

impl Eq for Color {}

impl Hash for Color {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.r.to_bits().hash(state);
        self.g.to_bits().hash(state);
        self.b.to_bits().hash(state);
        self.a.to_bits().hash(state);
    }
}

impl Color {
    /// Transparent black.
    pub const TRANSPARENT: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };

    /// Fully opaque black.
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };

    /// Fully opaque white.
    pub const WHITE: Self = Self {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };

    /// Fully opaque red.
    pub const RED: Self = Self {
        r: 1.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };

    /// Fully opaque green.
    pub const GREEN: Self = Self {
        r: 0.0,
        g: 1.0,
        b: 0.0,
        a: 1.0,
    };

    /// Fully opaque blue.
    pub const BLUE: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    };

    /// Create a color from 8-bit RGBA components.
    ///
    /// # Examples
    ///
    /// ```
    /// use ratatui_pixelcanvas::style::Color;
    ///
    /// let steel_blue = Color::from_rgba8(70, 130, 180, 255);
    /// ```
    #[must_use]
    pub const fn from_rgba8(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self {
            r: r as f32 / 255.0,
            g: g as f32 / 255.0,
            b: b as f32 / 255.0,
            a: a as f32 / 255.0,
        }
    }

    /// Create a fully opaque color from 8-bit RGB components.
    #[must_use]
    pub const fn from_rgb8(r: u8, g: u8, b: u8) -> Self {
        Self::from_rgba8(r, g, b, 255)
    }

    /// Create a color from floating-point RGBA components (each 0.0–1.0).
    #[must_use]
    pub const fn from_rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Return this color with a different alpha value.
    #[must_use]
    pub const fn with_alpha(mut self, a: f32) -> Self {
        self.a = a;
        self
    }

    /// Create a color from HSLA components.
    ///
    /// - `h`: Hue in degrees (0–360)
    /// - `s`: Saturation (0.0–1.0)
    /// - `l`: Lightness (0.0–1.0)
    /// - `a`: Alpha (0.0–1.0)
    #[must_use]
    #[allow(clippy::many_single_char_names)]
    pub fn from_hsla(hue: f32, sat: f32, light: f32, alpha: f32) -> Self {
        let hue = ((hue % 360.0) + 360.0) % 360.0; // Normalize to 0–360
        let sat = sat.clamp(0.0, 1.0);
        let light = light.clamp(0.0, 1.0);
        let alpha = alpha.clamp(0.0, 1.0);

        let chroma = (1.0 - (2.0f32).mul_add(light, -1.0).abs()) * sat;
        let x_val = chroma * (1.0 - ((hue / 60.0) % 2.0 - 1.0).abs());
        let match_val = light - chroma * 0.5;

        let (r1, g1, b1) = if hue < 60.0 {
            (chroma, x_val, 0.0)
        } else if hue < 120.0 {
            (x_val, chroma, 0.0)
        } else if hue < 180.0 {
            (0.0, chroma, x_val)
        } else if hue < 240.0 {
            (0.0, x_val, chroma)
        } else if hue < 300.0 {
            (x_val, 0.0, chroma)
        } else {
            (chroma, 0.0, x_val)
        };

        Self {
            r: r1 + match_val,
            g: g1 + match_val,
            b: b1 + match_val,
            a: alpha,
        }
    }

    /// Create a fully opaque color from HSL components.
    #[must_use]
    pub fn from_hsl(h: f32, s: f32, l: f32) -> Self {
        Self::from_hsla(h, s, l, 1.0)
    }

    /// Linearly interpolate between two colors in Oklab perceptual space.
    ///
    /// `t = 0.0` returns `self`, `t = 1.0` returns `other`.
    /// This is the same as the [`Lerp`](crate::scene::animation::Lerp) trait
    /// but available directly on `Color` without importing the trait.
    ///
    /// Oklab interpolation produces perceptually uniform gradients — unlike
    /// linear RGB, it avoids muddy midpoints (e.g., red→green won't go
    /// through brown).
    #[must_use]
    pub fn mix(self, other: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let (l1, a1, b1) = self.to_oklab();
        let (l2, a2, b2) = other.to_oklab();
        Self::from_oklab(
            (l2 - l1).mul_add(t, l1),
            (a2 - a1).mul_add(t, a1),
            (b2 - b1).mul_add(t, b1),
            (other.a - self.a).mul_add(t, self.a),
        )
    }

    /// Linearly interpolate in raw RGB space (no perceptual correction).
    ///
    /// Use this when you need exact RGB blending or when Oklab overhead
    /// is not desired.
    #[must_use]
    pub fn mix_rgb(self, other: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            r: (other.r - self.r).mul_add(t, self.r),
            g: (other.g - self.g).mul_add(t, self.g),
            b: (other.b - self.b).mul_add(t, self.b),
            a: (other.a - self.a).mul_add(t, self.a),
        }
    }

    // --- Oklab color space conversion ---

    /// Convert sRGB to Oklab perceptual color space.
    ///
    /// Returns `(L, a, b)` where L is lightness `[0,1]`, a is green-red, b is blue-yellow.
    /// Reference: Björn Ottosson, "A perceptual color space for image processing" (2020).
    #[must_use]
    pub fn to_oklab(self) -> (f32, f32, f32) {
        // sRGB → linear
        let rl = srgb_to_linear(self.r);
        let gl = srgb_to_linear(self.g);
        let bl = srgb_to_linear(self.b);

        // Linear RGB → LMS (using Oklab matrix)
        let l = 0.412_165_6_f32.mul_add(rl, 0.536_275_2_f32.mul_add(gl, 0.051_457_57 * bl));
        let m = 0.211_859_1_f32.mul_add(rl, 0.680_719_9_f32.mul_add(gl, 0.107_406_58 * bl));
        let s = 0.088_309_78_f32.mul_add(rl, 0.281_847_42_f32.mul_add(gl, 0.629_927_6 * bl));

        // Cube root
        let l_ = l.cbrt();
        let m_ = m.cbrt();
        let s_ = s.cbrt();

        // LMS → Lab
        let ok_l = 0.210_454_26_f32.mul_add(l_, 0.793_617_8_f32.mul_add(m_, -0.004_072_047 * s_));
        let ok_a = 1.977_998_5_f32.mul_add(l_, (-2.428_592_2_f32).mul_add(m_, 0.450_593_7 * s_));
        let ok_b = 0.025_904_037_f32.mul_add(l_, 0.782_771_8_f32.mul_add(m_, -0.808_675_77 * s_));

        (ok_l, ok_a, ok_b)
    }

    /// Create a color from Oklab components plus alpha.
    #[must_use]
    pub fn from_oklab(l: f32, a_ok: f32, b_ok: f32, alpha: f32) -> Self {
        // Lab → LMS
        let l_ = 0.396_337_78_f32.mul_add(a_ok, 0.215_803_76_f32.mul_add(b_ok, l));
        let m_ = (-0.105_561_346_f32).mul_add(a_ok, (-0.063_854_17_f32).mul_add(b_ok, l));
        let s_ = (-0.089_484_18_f32).mul_add(a_ok, (-1.291_485_5_f32).mul_add(b_ok, l));

        // Cube
        let l_lin = l_ * l_ * l_;
        let m_lin = m_ * m_ * m_;
        let s_lin = s_ * s_ * s_;

        // LMS → linear RGB
        let rl = 4.076_741_7_f32.mul_add(
            l_lin,
            (-3.307_711_6_f32).mul_add(m_lin, 0.230_969_94 * s_lin),
        );
        let gl =
            (-1.268_438_f32).mul_add(l_lin, 2.609_757_4_f32.mul_add(m_lin, -0.341_319_38 * s_lin));
        let bl = (-0.004_196_086_3_f32).mul_add(
            l_lin,
            (-0.703_418_6_f32).mul_add(m_lin, 1.707_614_7 * s_lin),
        );

        Self {
            r: linear_to_srgb(rl),
            g: linear_to_srgb(gl),
            b: linear_to_srgb(bl),
            a: alpha,
        }
    }

    /// Adjust lightness by a factor.
    ///
    /// Values > 1.0 lighten, values < 1.0 darken. Clamped to [0.0, 1.0] range.
    #[must_use]
    pub fn with_lightness(self, factor: f32) -> Self {
        Self {
            r: (self.r * factor).clamp(0.0, 1.0),
            g: (self.g * factor).clamp(0.0, 1.0),
            b: (self.b * factor).clamp(0.0, 1.0),
            a: self.a,
        }
    }

    /// Convert to a `tiny_skia::Color`.
    #[must_use]
    pub fn to_tiny_skia(self) -> Option<tiny_skia::Color> {
        tiny_skia::Color::from_rgba(self.r, self.g, self.b, self.a)
    }
}

// --- sRGB transfer functions ---

/// sRGB gamma → linear (inverse OETF).
#[inline]
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear → sRGB gamma (OETF).
#[inline]
fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055_f32.mul_add(c.powf(1.0 / 2.4), -0.055)
    }
}

impl Default for Color {
    fn default() -> Self {
        Self::BLACK
    }
}

#[cfg(feature = "widget")]
impl From<ratatui::style::Color> for Color {
    /// Convert from a `ratatui::style::Color`.
    ///
    /// - `Rgb(r, g, b)` maps directly.
    /// - Named colors (Red, Blue, etc.) use standard ANSI values.
    /// - `Indexed` attempts the standard 256-color palette for the first
    ///   16 entries; others default to white.
    fn from(c: ratatui::style::Color) -> Self {
        match c {
            ratatui::style::Color::Rgb(r, g, b) => Self::from_rgb8(r, g, b),
            ratatui::style::Color::Black => Self::BLACK,
            ratatui::style::Color::Red => Self::from_rgb8(205, 0, 0),
            ratatui::style::Color::Green => Self::from_rgb8(0, 205, 0),
            ratatui::style::Color::Blue => Self::from_rgb8(0, 0, 238),
            ratatui::style::Color::Yellow => Self::from_rgb8(205, 205, 0),
            ratatui::style::Color::Magenta => Self::from_rgb8(205, 0, 205),
            ratatui::style::Color::Cyan => Self::from_rgb8(0, 205, 205),
            ratatui::style::Color::Gray => Self::from_rgb8(128, 128, 128),
            ratatui::style::Color::DarkGray => Self::from_rgb8(85, 85, 85),
            ratatui::style::Color::LightRed => Self::from_rgb8(255, 0, 0),
            ratatui::style::Color::LightGreen => Self::from_rgb8(0, 255, 0),
            ratatui::style::Color::LightBlue => Self::from_rgb8(92, 92, 255),
            ratatui::style::Color::LightYellow => Self::from_rgb8(255, 255, 0),
            ratatui::style::Color::LightMagenta => Self::from_rgb8(255, 0, 255),
            ratatui::style::Color::LightCyan => Self::from_rgb8(0, 255, 255),
            _ => Self::WHITE, // White, Reset, Indexed fallback
        }
    }
}

#[cfg(feature = "widget")]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
impl From<Color> for ratatui::style::Color {
    /// Convert to a `ratatui::style::Color::Rgb`.
    fn from(c: Color) -> Self {
        Self::Rgb(
            (c.r * 255.0) as u8,
            (c.g * 255.0) as u8,
            (c.b * 255.0) as u8,
        )
    }
}

// ---------------------------------------------------------------------------
// Line cap & join
// ---------------------------------------------------------------------------

/// How the ends of a stroked line are drawn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum LineCap {
    /// Flat edge at the endpoint.
    #[default]
    Butt,
    /// Rounded edge extending past the endpoint.
    Round,
    /// Square edge extending past the endpoint.
    Square,
}

/// How corners in a stroked path are drawn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum LineJoin {
    /// Sharp corner.
    #[default]
    Miter,
    /// Rounded corner.
    Round,
    /// Beveled (flat) corner.
    Bevel,
}

// ---------------------------------------------------------------------------
// Dash pattern
// ---------------------------------------------------------------------------

/// A repeating dash pattern for stroked lines.
#[derive(Clone, Debug, PartialEq)]
pub struct DashPattern {
    /// Alternating lengths of dash and gap segments.
    pub intervals: Vec<f32>,
    /// Offset into the pattern to start drawing from.
    pub offset: f32,
}

impl Eq for DashPattern {}

impl Hash for DashPattern {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for &v in &self.intervals {
            v.to_bits().hash(state);
        }
        self.offset.to_bits().hash(state);
    }
}

impl DashPattern {
    /// Create a simple dash pattern (e.g., `[5.0, 3.0]` = 5px dash, 3px gap).
    #[must_use]
    pub const fn new(intervals: Vec<f32>, offset: f32) -> Self {
        Self { intervals, offset }
    }

    /// Create a 2-element dash pattern (dash, gap) — the most common case
    /// for line-drawing animation.
    #[must_use]
    pub fn pair(dash: f32, gap: f32) -> Self {
        Self {
            intervals: vec![dash, gap],
            offset: 0.0,
        }
    }

    /// Create a 4-element dash pattern with a custom offset.
    ///
    /// Used by trailing ghost effects: `[skip, trail_start, trail_len, hide]`.
    #[must_use]
    pub fn quad(a: f32, b: f32, c: f32, d: f32, offset: f32) -> Self {
        Self {
            intervals: vec![a, b, c, d],
            offset,
        }
    }
}

// ---------------------------------------------------------------------------
// Stroke style
// ---------------------------------------------------------------------------

/// Describes how the outline of a shape is rendered.
#[derive(Clone, Debug, PartialEq)]
pub struct StrokeStyle {
    /// Stroke color.
    pub color: Color,
    /// Stroke width in pixels.
    pub width: f32,
    /// How line endpoints are drawn.
    pub line_cap: LineCap,
    /// How line corners are drawn.
    pub line_join: LineJoin,
    /// Optional dash pattern.
    pub dash: Option<DashPattern>,
}

impl Eq for StrokeStyle {}

impl Hash for StrokeStyle {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.color.hash(state);
        self.width.to_bits().hash(state);
        self.line_cap.hash(state);
        self.line_join.hash(state);
        self.dash.hash(state);
    }
}

impl Default for StrokeStyle {
    fn default() -> Self {
        Self {
            color: Color::WHITE,
            width: 1.0,
            line_cap: LineCap::default(),
            line_join: LineJoin::default(),
            dash: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Gradient
// ---------------------------------------------------------------------------

/// A color stop in a gradient.
#[derive(Clone, Debug, PartialEq)]
pub struct GradientStop {
    /// Position along the gradient (0.0–1.0).
    pub position: f32,
    /// Color at this stop.
    pub color: Color,
}

impl Eq for GradientStop {}

impl Hash for GradientStop {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.position.to_bits().hash(state);
        self.color.hash(state);
    }
}

/// Definition of a gradient (linear or radial).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GradientDef {
    /// The type of gradient.
    pub kind: GradientKind,
    /// Color stops, sorted by position.
    pub stops: Vec<GradientStop>,
}

/// Whether a gradient is linear or radial.
#[derive(Clone, Debug, PartialEq)]
pub enum GradientKind {
    /// A gradient interpolated along a line between two points.
    Linear {
        /// Start point.
        start: Point,
        /// End point.
        end: Point,
    },
    /// A gradient radiating from a center point.
    Radial {
        /// Center of the gradient.
        center: Point,
        /// Radius of the gradient.
        radius: f32,
    },
}

impl Eq for GradientKind {}

impl Hash for GradientKind {
    fn hash<H: Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Self::Linear { start, end } => {
                start.hash(state);
                end.hash(state);
            }
            Self::Radial { center, radius } => {
                center.hash(state);
                radius.to_bits().hash(state);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fill style
// ---------------------------------------------------------------------------

/// Describes how the interior of a shape is filled.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum FillStyle {
    /// A single solid color.
    Solid(Color),
    /// A linear gradient fill.
    LinearGradient(GradientDef),
    /// A radial gradient fill.
    RadialGradient(GradientDef),
}

impl Default for FillStyle {
    fn default() -> Self {
        Self::Solid(Color::WHITE)
    }
}

// ---------------------------------------------------------------------------
// Shape style (fill + stroke combined)
// ---------------------------------------------------------------------------

/// Combined fill and stroke configuration for a shape.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ShapeStyle {
    /// How the interior is filled. `None` means no fill.
    pub fill: Option<FillStyle>,
    /// How the outline is stroked. `None` means no stroke.
    pub stroke: Option<StrokeStyle>,
    /// Whether anti-aliasing is enabled.
    pub anti_alias: bool,
}

// ---------------------------------------------------------------------------
// Geometry primitives
// ---------------------------------------------------------------------------

/// A 2D point.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Point {
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
}

impl Eq for Point {}

impl Hash for Point {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.to_bits().hash(state);
        self.y.to_bits().hash(state);
    }
}

impl Point {
    /// Create a new point.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// A 2D axis-aligned rectangle.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Rect {
    /// X coordinate of the top-left corner.
    pub x: f32,
    /// Y coordinate of the top-left corner.
    pub y: f32,
    /// Width.
    pub width: f32,
    /// Height.
    pub height: f32,
}

impl Eq for Rect {}

impl Hash for Rect {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.to_bits().hash(state);
        self.y.to_bits().hash(state);
        self.width.to_bits().hash(state);
        self.height.to_bits().hash(state);
    }
}

impl Rect {
    /// Create a new rectangle.
    #[must_use]
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Create a rectangle from the origin with the given dimensions.
    #[must_use]
    pub const fn from_size(width: f32, height: f32) -> Self {
        Self::new(0.0, 0.0, width, height)
    }

    /// X coordinate of the right edge.
    #[must_use]
    pub fn right(&self) -> f32 {
        self.x + self.width
    }

    /// Y coordinate of the bottom edge.
    #[must_use]
    pub fn bottom(&self) -> f32 {
        self.y + self.height
    }

    /// Center point of the rectangle.
    #[must_use]
    pub fn center(&self) -> Point {
        Point::new(
            self.width.mul_add(0.5, self.x),
            self.height.mul_add(0.5, self.y),
        )
    }

    /// Whether this rectangle contains the given point.
    #[must_use]
    pub fn contains(&self, point: Point) -> bool {
        point.x >= self.x
            && point.x <= self.right()
            && point.y >= self.y
            && point.y <= self.bottom()
    }

    /// Whether this rectangle overlaps with another.
    #[must_use]
    pub fn intersects(&self, other: &Self) -> bool {
        self.x < other.right()
            && self.right() > other.x
            && self.y < other.bottom()
            && self.bottom() > other.y
    }

    /// Compute the bounding box that contains both rectangles.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        let right = self.right().max(other.right());
        let bottom = self.bottom().max(other.bottom());
        Self::new(x, y, right - x, bottom - y)
    }

    /// Area of the rectangle.
    #[must_use]
    pub fn area(&self) -> f32 {
        self.width * self.height
    }
}

// ---------------------------------------------------------------------------
// Clipping
// ---------------------------------------------------------------------------

/// A region that restricts drawing to its interior.
///
/// Used on [`Group`](crate::scene::command::DrawCommand::Group) to clip
/// all child drawing commands.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ClipRegion {
    /// Clip to a rectangle.
    Rect(Rect),
    /// Clip to an arbitrary path.
    Path(crate::scene::command::PathData),
}

// ---------------------------------------------------------------------------
// Blend modes
// ---------------------------------------------------------------------------

/// Compositing blend mode for shapes and groups.
///
/// Determines how source pixels are combined with destination pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BlendMode {
    /// Normal alpha compositing (default).
    #[default]
    SrcOver,
    /// Multiplies source and destination colors.
    Multiply,
    /// Inverse multiply — lightens.
    Screen,
    /// Combines multiply and screen based on destination.
    Overlay,
    /// Keeps the darker of source and destination.
    Darken,
    /// Keeps the lighter of source and destination.
    Lighten,
}

impl BlendMode {
    /// Convert to the corresponding `tiny_skia::BlendMode`.
    #[must_use]
    pub const fn to_tiny_skia(self) -> tiny_skia::BlendMode {
        match self {
            Self::SrcOver => tiny_skia::BlendMode::SourceOver,
            Self::Multiply => tiny_skia::BlendMode::Multiply,
            Self::Screen => tiny_skia::BlendMode::Screen,
            Self::Overlay => tiny_skia::BlendMode::Overlay,
            Self::Darken => tiny_skia::BlendMode::Darken,
            Self::Lighten => tiny_skia::BlendMode::Lighten,
        }
    }
}

/// A 2D affine transform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    /// Scale X.
    pub sx: f32,
    /// Skew Y.
    pub kx: f32,
    /// Skew X.
    pub ky: f32,
    /// Scale Y.
    pub sy: f32,
    /// Translate X.
    pub tx: f32,
    /// Translate Y.
    pub ty: f32,
}

impl Eq for Transform {}

impl Hash for Transform {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.sx.to_bits().hash(state);
        self.kx.to_bits().hash(state);
        self.ky.to_bits().hash(state);
        self.sy.to_bits().hash(state);
        self.tx.to_bits().hash(state);
        self.ty.to_bits().hash(state);
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    /// The identity transform (no transformation).
    pub const IDENTITY: Self = Self {
        sx: 1.0,
        kx: 0.0,
        ky: 0.0,
        sy: 1.0,
        tx: 0.0,
        ty: 0.0,
    };

    /// Create an identity transform (no transformation).
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            sx: 1.0,
            kx: 0.0,
            ky: 0.0,
            sy: 1.0,
            tx: 0.0,
            ty: 0.0,
        }
    }

    /// Create a translation transform.
    #[must_use]
    pub const fn translate(tx: f32, ty: f32) -> Self {
        Self {
            sx: 1.0,
            kx: 0.0,
            ky: 0.0,
            sy: 1.0,
            tx,
            ty,
        }
    }

    /// Create a uniform scale transform.
    #[must_use]
    pub const fn scale(s: f32) -> Self {
        Self {
            sx: s,
            kx: 0.0,
            ky: 0.0,
            sy: s,
            tx: 0.0,
            ty: 0.0,
        }
    }

    /// Create a non-uniform scale transform.
    #[must_use]
    pub const fn scale_xy(sx: f32, sy: f32) -> Self {
        Self {
            sx,
            kx: 0.0,
            ky: 0.0,
            sy,
            tx: 0.0,
            ty: 0.0,
        }
    }

    /// Create a rotation transform (angle in radians, around the origin).
    #[must_use]
    pub fn rotate(angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self {
            sx: cos,
            kx: sin,
            ky: -sin,
            sy: cos,
            tx: 0.0,
            ty: 0.0,
        }
    }

    /// Create a rotation transform around a given center point.
    #[must_use]
    pub fn rotate_at(angle: f32, cx: f32, cy: f32) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self {
            sx: cos,
            kx: sin,
            ky: -sin,
            sy: cos,
            tx: cos.mul_add(-cx, sin.mul_add(cy, cx)),
            ty: sin.mul_add(-cx, cos.mul_add(-cy, cy)),
        }
    }

    /// Create a skew (shear) transform.
    ///
    /// `kx` skews along the X axis, `ky` along the Y axis (both in radians).
    #[must_use]
    pub fn skew(kx: f32, ky: f32) -> Self {
        Self {
            sx: 1.0,
            kx: kx.tan(),
            ky: ky.tan(),
            sy: 1.0,
            tx: 0.0,
            ty: 0.0,
        }
    }

    /// Construct a transform from all six matrix components directly.
    #[must_use]
    pub const fn from_matrix(sx: f32, kx: f32, ky: f32, sy: f32, tx: f32, ty: f32) -> Self {
        Self {
            sx,
            kx,
            ky,
            sy,
            tx,
            ty,
        }
    }

    /// Concatenate (multiply) this transform with another.
    ///
    /// The resulting transform applies `self` first, then `other`.
    #[must_use]
    pub fn concat(self, other: Self) -> Self {
        Self {
            sx: self.sx.mul_add(other.sx, self.kx * other.ky),
            kx: self.sx.mul_add(other.kx, self.kx * other.sy),
            ky: self.ky.mul_add(other.sx, self.sy * other.ky),
            sy: self.ky.mul_add(other.kx, self.sy * other.sy),
            tx: self
                .tx
                .mul_add(other.sx, self.ty.mul_add(other.ky, other.tx)),
            ty: self
                .tx
                .mul_add(other.kx, self.ty.mul_add(other.sy, other.ty)),
        }
    }

    /// Compute the determinant of the transformation matrix.
    ///
    /// A zero determinant means the transform is degenerate (non-invertible).
    #[must_use]
    pub fn determinant(self) -> f32 {
        self.sx.mul_add(self.sy, -(self.kx * self.ky))
    }

    /// Compute the inverse of this transform.
    ///
    /// Returns `None` if the transform is degenerate (determinant ≈ 0).
    #[must_use]
    pub fn inverse(self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < 1e-10 {
            return None;
        }
        let inv_det = 1.0 / det;
        Some(Self {
            sx: self.sy * inv_det,
            kx: -self.kx * inv_det,
            ky: -self.ky * inv_det,
            sy: self.sx * inv_det,
            tx: self.kx.mul_add(self.ty, -(self.sy * self.tx)) * inv_det,
            ty: self.ky.mul_add(self.tx, -(self.sx * self.ty)) * inv_det,
        })
    }

    /// Apply this transform to a point.
    #[must_use]
    pub fn apply_point(self, p: Point) -> Point {
        Point {
            x: self.sx.mul_add(p.x, self.ky.mul_add(p.y, self.tx)),
            y: self.kx.mul_add(p.x, self.sy.mul_add(p.y, self.ty)),
        }
    }

    /// Convert to a `tiny_skia::Transform`.
    #[must_use]
    pub const fn to_tiny_skia(self) -> tiny_skia::Transform {
        tiny_skia::Transform {
            sx: self.sx,
            kx: self.kx,
            ky: self.ky,
            sy: self.sy,
            tx: self.tx,
            ty: self.ty,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn transform_identity() {
        let t = Transform::IDENTITY;
        assert_eq!(t.sx, 1.0);
        assert_eq!(t.sy, 1.0);
        assert_eq!(t.kx, 0.0);
        assert_eq!(t.ky, 0.0);
        assert_eq!(t.tx, 0.0);
        assert_eq!(t.ty, 0.0);
    }

    #[test]
    fn transform_rotate_90_degrees() {
        let t = Transform::rotate(FRAC_PI_2);
        assert!(approx_eq(t.sx, 0.0));
        assert!(approx_eq(t.kx, 1.0));
        assert!(approx_eq(t.ky, -1.0));
        assert!(approx_eq(t.sy, 0.0));
    }

    #[test]
    fn transform_scale_xy() {
        let t = Transform::scale_xy(2.0, 3.0);
        assert_eq!(t.sx, 2.0);
        assert_eq!(t.sy, 3.0);
        assert_eq!(t.kx, 0.0);
        assert_eq!(t.ky, 0.0);
    }

    #[test]
    fn transform_concat_identity() {
        let t = Transform::translate(10.0, 20.0);
        let result = t.concat(Transform::IDENTITY);
        assert!(approx_eq(result.tx, 10.0));
        assert!(approx_eq(result.ty, 20.0));
        assert!(approx_eq(result.sx, 1.0));
    }

    #[test]
    fn transform_concat_translate_then_scale() {
        let t = Transform::translate(10.0, 20.0).concat(Transform::scale(2.0));
        assert!(approx_eq(t.sx, 2.0));
        assert!(approx_eq(t.tx, 20.0)); // 10 * 2
        assert!(approx_eq(t.ty, 40.0)); // 20 * 2
    }

    #[test]
    fn transform_skew() {
        let t = Transform::skew(FRAC_PI_4, 0.0);
        assert!(approx_eq(t.kx, 1.0)); // tan(45°) = 1
        assert!(approx_eq(t.ky, 0.0));
        assert!(approx_eq(t.sx, 1.0));
        assert!(approx_eq(t.sy, 1.0));
    }

    #[test]
    fn transform_rotate_at_center() {
        let t = Transform::rotate_at(FRAC_PI_2, 50.0, 50.0);
        // Rotating (50, 0) around (50, 50) by 90° should give (100, 50)
        // Apply: x' = sx*x + ky*y + tx, y' = kx*x + sy*y + ty
        let x = t.sx.mul_add(50.0, t.ky * 0.0) + t.tx;
        let y = t.kx.mul_add(50.0, t.sy * 0.0) + t.ty;
        assert!(approx_eq(x, 100.0));
        assert!(approx_eq(y, 50.0));
    }

    #[test]
    fn transform_from_matrix_roundtrip() {
        let t = Transform::from_matrix(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        assert_eq!(t.sx, 1.0);
        assert_eq!(t.kx, 2.0);
        assert_eq!(t.ky, 3.0);
        assert_eq!(t.sy, 4.0);
        assert_eq!(t.tx, 5.0);
        assert_eq!(t.ty, 6.0);
    }
}
