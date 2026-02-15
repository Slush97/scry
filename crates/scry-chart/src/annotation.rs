//! Annotation system for placing labels at data coordinates.
//!
//! Annotations are text labels placed at specific data-space coordinates.
//! They can optionally have arrows, backgrounds, and borders.

use scry_engine::style::Color;

/// An annotation placed at specific data coordinates on a chart.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[must_use]
#[non_exhaustive]
pub struct Annotation {
    /// X data coordinate.
    pub x: f64,
    /// Y data coordinate.
    pub y: f64,
    /// The annotation text.
    pub text: String,
    /// Whether to draw an arrow from text to the data point.
    pub arrow: bool,
    /// Visual style for the annotation.
    pub style: AnnotationStyle,
}

/// Visual style for an annotation.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct AnnotationStyle {
    /// Text color.
    pub text_color: Color,
    /// Optional background fill behind the text.
    pub background: Option<Color>,
    /// Optional border color.
    pub border: Option<Color>,
    /// Pixel offset from the data point (dx, dy).
    pub offset: (f32, f32),
}

impl Default for AnnotationStyle {
    fn default() -> Self {
        Self {
            text_color: Color::from_rgba8(220, 220, 220, 255),
            background: None,
            border: None,
            offset: (10.0, -15.0),
        }
    }
}

impl Annotation {
    /// Create a simple annotation at the given data point.
    pub fn new(x: f64, y: f64, text: impl Into<String>) -> Self {
        Self {
            x,
            y,
            text: text.into(),
            arrow: false,
            style: AnnotationStyle::default(),
        }
    }

    /// Enable an arrow from the text to the data point.
    pub fn with_arrow(mut self) -> Self {
        self.arrow = true;
        self
    }

    /// Set a background color behind the annotation text.
    pub fn with_background(mut self, color: Color) -> Self {
        self.style.background = Some(color);
        self
    }

    /// Set the pixel offset from the data point.
    pub fn with_offset(mut self, dx: f32, dy: f32) -> Self {
        self.style.offset = (dx, dy);
        self
    }

    /// Set the text color.
    pub fn with_color(mut self, color: Color) -> Self {
        self.style.text_color = color;
        self
    }
}
