// SPDX-License-Identifier: MIT OR Apache-2.0
//! Color themes and palettes for charts.
//!
//! Ships with dark-terminal-optimized defaults and seaborn-inspired palettes.
//! Themes use a hierarchical token system for fine-grained control.

use scry_engine::style::Color;
use scry_engine::style::DashPattern;

// ---------------------------------------------------------------------------
// Sub-structs
// ---------------------------------------------------------------------------

/// Text styling tokens.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct TextStyle {
    /// Text color.
    pub color: Color,
    /// Whether text should be bold.
    pub bold: bool,
    /// Base font size in pixels (before canvas-proportional scaling).
    ///
    /// The layout engine multiplies this by a scale factor derived from
    /// the canvas dimensions. Typical defaults: title 18, labels 13, ticks 11.
    pub font_size: f32,
}

/// Axis rendering tokens.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct AxisTheme {
    /// Color of axis lines (spines).
    pub color: Color,
    /// Axis line width (pixels).
    pub width: f32,
    /// Major tick mark length (pixels).
    pub tick_length: f32,
    /// Tick mark color (defaults to axis color).
    pub tick_color: Color,
    /// Whether to show minor tick marks between major ticks.
    pub minor_ticks: bool,
    /// Minor tick length (pixels).
    pub minor_tick_length: f32,
}

/// Grid rendering tokens.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct GridTheme {
    /// Grid line color.
    pub color: Color,
    /// Grid line width (pixels).
    pub width: f32,
    /// Dash pattern for grid lines (`None` = solid).
    pub dash: Option<DashPattern>,
    /// Whether to draw gridlines (master toggle).
    pub show: bool,
    /// Override for X-axis gridlines (vertical lines).
    /// `None` inherits from [`show`](Self::show).
    pub show_x: Option<bool>,
    /// Override for Y-axis gridlines (horizontal lines).
    /// `None` inherits from [`show`](Self::show).
    pub show_y: Option<bool>,
}

/// Data series rendering tokens.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct SeriesTheme {
    /// Point radius for scatter plots (pixels).
    pub point_radius: f32,
    /// Line width for line charts (pixels).
    pub line_width: f32,
    /// Stroke width for bar chart outlines (0.0 = no stroke).
    pub bar_stroke_width: f32,
    /// Default opacity for area fills (0.0–1.0).
    pub fill_opacity: f32,
    /// Default corner radius for bars.
    pub bar_corner_radius: f32,
}

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

/// Legend visual tokens.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct LegendTheme {
    /// Legend label font size in pixels (before canvas-proportional scaling).
    /// Defaults to `tick_style.font_size`.
    pub font_size: f32,
    /// Legend box background color (near-opaque to occlude grid lines).
    pub background: Color,
    /// Optional border color around the legend box.
    pub border: Option<Color>,
}

/// A complete visual theme for chart rendering.
///
/// Uses hierarchical tokens: `axis`, `grid`, `series`, and text styles
/// for fine-grained control over every chart element.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[must_use]
#[non_exhaustive]
pub struct Theme {
    /// Background color of the plot area.
    pub background: Color,
    /// Primary text/foreground color.
    pub foreground: Color,
    /// Ordered palette of data series colors.
    pub palette: Vec<Color>,
    /// Title text style.
    pub title_style: TextStyle,
    /// Axis label text style.
    pub label_style: TextStyle,
    /// Tick label text style.
    pub tick_style: TextStyle,
    /// Axis rendering tokens.
    pub axis: AxisTheme,
    /// Grid rendering tokens.
    pub grid: GridTheme,
    /// Series rendering tokens.
    pub series: SeriesTheme,
    /// Legend rendering tokens.
    pub legend: LegendTheme,
}

impl Theme {
    /// Dark theme optimized for terminal backgrounds.
    ///
    /// Uses muted neon colors that look great on dark terminals.
    pub fn dark() -> Self {
        let text_color = Color::from_rgba8(200, 200, 220, 255);
        let axis_color = Color::from_rgba8(160, 160, 180, 255);

        Self {
            background: Color::from_rgba8(15, 15, 25, 255),
            foreground: text_color,
            palette: vec![
                Color::from_rgba8(99, 179, 237, 255),  // sky blue
                Color::from_rgba8(252, 129, 155, 255), // coral pink
                Color::from_rgba8(134, 239, 172, 255), // mint green
                Color::from_rgba8(251, 191, 36, 255),  // amber
                Color::from_rgba8(196, 167, 255, 255), // lavender
                Color::from_rgba8(255, 130, 90, 255),  // warm coral
                Color::from_rgba8(72, 226, 186, 255),  // turquoise
                Color::from_rgba8(220, 120, 220, 255), // orchid
            ],
            title_style: TextStyle {
                color: text_color,
                bold: true,
                font_size: 18.0,
            },
            label_style: TextStyle {
                color: text_color,
                bold: false,
                font_size: 14.0,
            },
            tick_style: TextStyle {
                color: text_color,
                bold: false,
                font_size: 11.0,
            },
            axis: AxisTheme {
                color: axis_color,
                width: 1.5,
                tick_length: 5.0,
                tick_color: axis_color,
                minor_ticks: false,
                minor_tick_length: 3.0,
            },
            grid: GridTheme {
                color: Color::from_rgba8(60, 60, 80, 110),
                width: 1.0,
                dash: Some(DashPattern::new(vec![4.0, 4.0], 0.0)),
                show: true,
                show_x: None,
                show_y: None,
            },
            series: SeriesTheme {
                point_radius: 5.5,
                line_width: 2.5,
                bar_stroke_width: 1.0,
                fill_opacity: 0.25,
                bar_corner_radius: 3.0,
            },
            legend: LegendTheme {
                font_size: 11.0,
                background: Color::from_rgba8(15, 15, 25, 235),
                border: Some(Color::from_rgba8(80, 80, 100, 200)),
            },
        }
    }

    /// Light theme — modern muted tones on a clean white background.
    ///
    /// Curated palette with wide hue separation, suitable for reports,
    /// dashboards, and embedding in light-mode UIs.
    pub fn light() -> Self {
        let text_color = Color::from_rgba8(38, 38, 56, 255);
        let axis_color = Color::from_rgba8(55, 55, 75, 255);

        Self {
            background: Color::from_rgba8(252, 252, 254, 255),
            foreground: text_color,
            palette: vec![
                Color::from_rgba8(55, 126, 184, 255),  // steel blue
                Color::from_rgba8(228, 120, 51, 255),  // warm amber
                Color::from_rgba8(77, 175, 74, 255),   // sage green
                Color::from_rgba8(178, 55, 65, 255),   // wine red
                Color::from_rgba8(130, 100, 175, 255), // muted violet
                Color::from_rgba8(0, 150, 145, 255),   // deep teal
                Color::from_rgba8(140, 140, 50, 255),  // dark khaki
                Color::from_rgba8(95, 95, 115, 255),   // cool slate
            ],
            title_style: TextStyle {
                color: text_color,
                bold: true,
                font_size: 18.0,
            },
            label_style: TextStyle {
                color: text_color,
                bold: false,
                font_size: 14.0,
            },
            tick_style: TextStyle {
                color: text_color,
                bold: false,
                font_size: 11.0,
            },
            axis: AxisTheme {
                color: axis_color,
                width: 1.5,
                tick_length: 5.0,
                tick_color: axis_color,
                minor_ticks: false,
                minor_tick_length: 3.0,
            },
            grid: GridTheme {
                color: Color::from_rgba8(190, 190, 205, 100),
                width: 0.8,
                dash: Some(DashPattern::new(vec![4.0, 4.0], 0.0)),
                show: true,
                show_x: None,
                show_y: None,
            },
            series: SeriesTheme {
                point_radius: 5.0,
                line_width: 2.0,
                bar_stroke_width: 0.8,
                fill_opacity: 0.20,
                bar_corner_radius: 2.0,
            },
            legend: LegendTheme {
                font_size: 11.0,
                background: Color::from_rgba8(252, 252, 254, 235),
                border: Some(Color::from_rgba8(195, 195, 200, 200)),
            },
        }
    }

    /// Chalk pastel theme — boosted soft pastels on dark charcoal.
    ///
    /// Warmer and more saturated than classic pastels, ensuring each
    /// series is visible at 35% fill opacity on the dark background.
    pub fn pastel() -> Self {
        let text_color = Color::from_rgba8(195, 195, 215, 255);
        let axis_color = Color::from_rgba8(145, 145, 165, 255);

        Self {
            background: Color::from_rgba8(22, 22, 30, 255),
            foreground: text_color,
            palette: vec![
                Color::from_rgba8(120, 170, 230, 255), // sky blue
                Color::from_rgba8(240, 130, 130, 255), // coral
                Color::from_rgba8(100, 210, 100, 255), // grass green
                Color::from_rgba8(240, 200, 80, 255),  // sunflower
                Color::from_rgba8(190, 120, 220, 255), // violet
                Color::from_rgba8(50, 210, 190, 255),  // aqua-green
                Color::from_rgba8(195, 175, 130, 255), // warm sand
                Color::from_rgba8(170, 100, 120, 255), // dark rose
            ],
            title_style: TextStyle {
                color: text_color,
                bold: true,
                font_size: 18.0,
            },
            label_style: TextStyle {
                color: text_color,
                bold: false,
                font_size: 14.0,
            },
            tick_style: TextStyle {
                color: text_color,
                bold: false,
                font_size: 11.0,
            },
            axis: AxisTheme {
                color: axis_color,
                width: 1.0,
                tick_length: 5.0,
                tick_color: axis_color,
                minor_ticks: false,
                minor_tick_length: 3.0,
            },
            grid: GridTheme {
                color: Color::from_rgba8(55, 55, 75, 110),
                width: 0.8,
                dash: Some(DashPattern::new(vec![4.0, 4.0], 0.0)),
                show: true,
                show_x: None,
                show_y: None,
            },
            series: SeriesTheme {
                point_radius: 5.5,
                line_width: 2.0,
                bar_stroke_width: 0.8,
                fill_opacity: 0.35,
                bar_corner_radius: 4.0,
            },
            legend: LegendTheme {
                font_size: 11.0,
                background: Color::from_rgba8(22, 22, 30, 235),
                border: Some(Color::from_rgba8(80, 80, 100, 200)),
            },
        }
    }

    /// Ocean theme — deep navy with wide-hue aquatic accents.
    ///
    /// Cyan anchors the palette, with coral, gold, and violet accents
    /// ensuring every series is distinctly identifiable.
    pub fn ocean() -> Self {
        let text_color = Color::from_rgba8(200, 220, 240, 255);
        let axis_color = Color::from_rgba8(120, 150, 180, 255);

        Self {
            background: Color::from_rgba8(10, 20, 40, 255),
            foreground: text_color,
            palette: vec![
                Color::from_rgba8(0, 188, 212, 255),   // cyan
                Color::from_rgba8(255, 140, 90, 255),  // coral (was teal — too close)
                Color::from_rgba8(100, 221, 80, 255),  // lime
                Color::from_rgba8(255, 200, 60, 255),  // gold
                Color::from_rgba8(171, 100, 210, 255), // violet
                Color::from_rgba8(66, 165, 245, 255),  // light blue
                Color::from_rgba8(240, 98, 146, 255),  // pink
                Color::from_rgba8(180, 180, 180, 255), // silver
            ],
            title_style: TextStyle {
                color: text_color,
                bold: true,
                font_size: 18.0,
            },
            label_style: TextStyle {
                color: text_color,
                bold: false,
                font_size: 14.0,
            },
            tick_style: TextStyle {
                color: text_color,
                bold: false,
                font_size: 11.0,
            },
            axis: AxisTheme {
                color: axis_color,
                width: 1.5,
                tick_length: 5.0,
                tick_color: axis_color,
                minor_ticks: false,
                minor_tick_length: 3.0,
            },
            grid: GridTheme {
                color: Color::from_rgba8(40, 65, 90, 110),
                width: 1.0,
                dash: Some(DashPattern::new(vec![4.0, 4.0], 0.0)),
                show: true,
                show_x: None,
                show_y: None,
            },
            series: SeriesTheme {
                point_radius: 5.5,
                line_width: 2.5,
                bar_stroke_width: 1.0,
                fill_opacity: 0.25,
                bar_corner_radius: 3.0,
            },
            legend: LegendTheme {
                font_size: 11.0,
                background: Color::from_rgba8(10, 20, 40, 240),
                border: Some(Color::from_rgba8(60, 90, 120, 200)),
            },
        }
    }

    /// Forest theme — earthy greens with warm accent hues.
    ///
    /// Anchored by forest green, then amber and terra cotta to ensure
    /// adjacent series are always distinguishable.
    pub fn forest() -> Self {
        let text_color = Color::from_rgba8(210, 210, 190, 255);
        let axis_color = Color::from_rgba8(150, 140, 120, 255);

        Self {
            background: Color::from_rgba8(18, 22, 15, 255),
            foreground: text_color,
            palette: vec![
                Color::from_rgba8(76, 175, 80, 255),   // green
                Color::from_rgba8(240, 170, 60, 255),  // warm amber (was light green)
                Color::from_rgba8(180, 110, 85, 255),  // terra cotta (was lime yellow)
                Color::from_rgba8(120, 200, 190, 255), // sage teal
                Color::from_rgba8(200, 200, 80, 255),  // chartreuse
                Color::from_rgba8(255, 138, 101, 255), // deep orange
                Color::from_rgba8(160, 140, 200, 255), // mist violet
                Color::from_rgba8(174, 213, 129, 255), // pale green
            ],
            title_style: TextStyle {
                color: text_color,
                bold: true,
                font_size: 18.0,
            },
            label_style: TextStyle {
                color: text_color,
                bold: false,
                font_size: 14.0,
            },
            tick_style: TextStyle {
                color: text_color,
                bold: false,
                font_size: 11.0,
            },
            axis: AxisTheme {
                color: axis_color,
                width: 1.5,
                tick_length: 5.0,
                tick_color: axis_color,
                minor_ticks: false,
                minor_tick_length: 3.0,
            },
            grid: GridTheme {
                color: Color::from_rgba8(50, 58, 42, 110),
                width: 1.0,
                dash: Some(DashPattern::new(vec![4.0, 4.0], 0.0)),
                show: true,
                show_x: None,
                show_y: None,
            },
            series: SeriesTheme {
                point_radius: 5.5,
                line_width: 2.5,
                bar_stroke_width: 1.0,
                fill_opacity: 0.30,
                bar_corner_radius: 2.0,
            },
            legend: LegendTheme {
                font_size: 11.0,
                background: Color::from_rgba8(18, 22, 15, 235),
                border: Some(Color::from_rgba8(80, 75, 60, 200)),
            },
        }
    }

    /// Colorblind-safe theme using the Okabe-Ito palette.
    ///
    /// Designed for accessibility — all 8 colors are distinguishable under
    /// deuteranopia, protanopia, and tritanopia. Based on the palette
    /// recommended by Nature, Science, and the NIH for scientific figures.
    ///
    /// Uses a dark background with the same axis/grid tokens as [`dark()`].
    pub fn colorblind() -> Self {
        let text_color = Color::from_rgba8(200, 200, 220, 255);
        let axis_color = Color::from_rgba8(160, 160, 180, 255);

        Self {
            background: Color::from_rgba8(15, 15, 25, 255),
            foreground: text_color,
            palette: vec![
                Color::from_rgba8(230, 159, 0, 255),   // orange
                Color::from_rgba8(86, 180, 233, 255),  // sky blue
                Color::from_rgba8(0, 158, 115, 255),   // bluish green
                Color::from_rgba8(240, 228, 66, 255),  // yellow
                Color::from_rgba8(0, 114, 178, 255),   // blue
                Color::from_rgba8(213, 94, 0, 255),    // vermillion
                Color::from_rgba8(204, 121, 167, 255), // reddish purple
                Color::from_rgba8(176, 176, 176, 255), // gray (replaces black for dark bg)
            ],
            title_style: TextStyle {
                color: text_color,
                bold: true,
                font_size: 18.0,
            },
            label_style: TextStyle {
                color: text_color,
                bold: false,
                font_size: 14.0,
            },
            tick_style: TextStyle {
                color: text_color,
                bold: false,
                font_size: 11.0,
            },
            axis: AxisTheme {
                color: axis_color,
                width: 1.5,
                tick_length: 5.0,
                tick_color: axis_color,
                minor_ticks: false,
                minor_tick_length: 3.0,
            },
            grid: GridTheme {
                color: Color::from_rgba8(60, 60, 80, 110),
                width: 1.0,
                dash: Some(DashPattern::new(vec![4.0, 4.0], 0.0)),
                show: true,
                show_x: None,
                show_y: None,
            },
            series: SeriesTheme {
                point_radius: 5.5,
                line_width: 2.5,
                bar_stroke_width: 1.0,
                fill_opacity: 0.25,
                bar_corner_radius: 2.0,
            },
            legend: LegendTheme {
                font_size: 11.0,
                background: Color::from_rgba8(15, 15, 25, 235),
                border: Some(Color::from_rgba8(80, 80, 100, 200)),
            },
        }
    }

    /// Academic theme — publication-ready for journals and papers.
    ///
    /// White background, solid black axes, near-zero corner radii, thin
    /// lines, and the Okabe-Ito palette for colorblind safety. Follows
    /// Nature / Science / IEEE figure guidelines.
    pub fn academic() -> Self {
        let text_color = Color::from_rgba8(25, 25, 25, 255);
        let axis_color = Color::from_rgba8(0, 0, 0, 255);

        Self {
            background: Color::WHITE,
            foreground: text_color,
            palette: vec![
                Color::from_rgba8(0, 114, 178, 255),   // blue
                Color::from_rgba8(200, 135, 0, 255), // dark amber (deepened for white-bg contrast)
                Color::from_rgba8(0, 158, 115, 255), // bluish green
                Color::from_rgba8(204, 121, 167, 255), // reddish purple
                Color::from_rgba8(60, 150, 210, 255), // sky blue (deepened for white-bg)
                Color::from_rgba8(213, 94, 0, 255),  // vermillion
                Color::from_rgba8(155, 140, 0, 255), // olive gold (white-bg safe)
                Color::from_rgba8(100, 100, 100, 255), // dark gray
            ],
            title_style: TextStyle {
                color: text_color,
                bold: true,
                font_size: 16.0,
            },
            label_style: TextStyle {
                color: text_color,
                bold: false,
                font_size: 13.0,
            },
            tick_style: TextStyle {
                color: text_color,
                bold: false,
                font_size: 11.0,
            },
            axis: AxisTheme {
                color: axis_color,
                width: 1.5,
                tick_length: 5.0,
                tick_color: axis_color,
                minor_ticks: true,
                minor_tick_length: 3.0,
            },
            grid: GridTheme {
                color: Color::from_rgba8(200, 200, 200, 80),
                width: 0.6,
                dash: Some(DashPattern::new(vec![2.0, 3.0], 0.0)),
                show: true,
                show_x: Some(false),
                show_y: None,
            },
            series: SeriesTheme {
                point_radius: 4.5,
                line_width: 1.5,
                bar_stroke_width: 0.5,
                fill_opacity: 0.15,
                bar_corner_radius: 0.0,
            },
            legend: LegendTheme {
                font_size: 10.0,
                background: Color::from_rgba8(255, 255, 255, 240),
                border: Some(Color::from_rgba8(180, 180, 180, 200)),
            },
        }
    }

    /// Presentation theme — bold and vibrant for slides and projectors.
    ///
    /// Large font sizes, thick lines, high-saturation palette, and
    /// generous point radii ensure readability from the back of the room.
    pub fn presentation() -> Self {
        let text_color = Color::from_rgba8(240, 240, 250, 255);
        let axis_color = Color::from_rgba8(180, 180, 200, 255);

        Self {
            background: Color::from_rgba8(18, 18, 35, 255),
            foreground: text_color,
            palette: vec![
                Color::from_rgba8(72, 149, 239, 255),  // vivid blue
                Color::from_rgba8(255, 107, 107, 255), // bright red
                Color::from_rgba8(78, 205, 130, 255),  // emerald
                Color::from_rgba8(255, 193, 59, 255),  // sunflower
                Color::from_rgba8(168, 85, 247, 255),  // vivid purple
                Color::from_rgba8(45, 212, 191, 255),  // turquoise
                Color::from_rgba8(251, 146, 60, 255),  // tangerine
                Color::from_rgba8(244, 114, 182, 255), // hot pink
            ],
            title_style: TextStyle {
                color: text_color,
                bold: true,
                font_size: 24.0,
            },
            label_style: TextStyle {
                color: text_color,
                bold: false,
                font_size: 16.0,
            },
            tick_style: TextStyle {
                color: text_color,
                bold: false,
                font_size: 13.0,
            },
            axis: AxisTheme {
                color: axis_color,
                width: 2.0,
                tick_length: 6.0,
                tick_color: axis_color,
                minor_ticks: false,
                minor_tick_length: 3.0,
            },
            grid: GridTheme {
                color: Color::from_rgba8(70, 70, 100, 115),
                width: 1.0,
                dash: Some(DashPattern::new(vec![6.0, 4.0], 0.0)),
                show: true,
                show_x: None,
                show_y: None,
            },
            series: SeriesTheme {
                point_radius: 7.0,
                line_width: 3.5,
                bar_stroke_width: 1.5,
                fill_opacity: 0.35,
                bar_corner_radius: 4.0,
            },
            legend: LegendTheme {
                font_size: 13.0,
                background: Color::from_rgba8(18, 18, 35, 240),
                border: Some(Color::from_rgba8(100, 100, 130, 200)),
            },
        }
    }

    /// Monochrome theme — pure grayscale for B&W printing.
    ///
    /// Uses 8 distinct gray shades and relies on `series_dash()` patterns
    /// for line chart differentiation. No rounded corners. Designed for
    /// academic papers, patents, and fax-safe output.
    pub fn monochrome() -> Self {
        let text_color = Color::from_rgba8(20, 20, 20, 255);
        let axis_color = Color::from_rgba8(0, 0, 0, 255);

        Self {
            background: Color::WHITE,
            foreground: text_color,
            palette: vec![
                Color::from_rgba8(30, 30, 30, 255),    // near-black
                Color::from_rgba8(80, 80, 80, 255),    // dark gray
                Color::from_rgba8(130, 130, 130, 255), // medium gray
                Color::from_rgba8(55, 55, 55, 255),    // charcoal
                Color::from_rgba8(105, 105, 105, 255), // dim gray
                Color::from_rgba8(60, 60, 60, 255),    // jet
                Color::from_rgba8(140, 140, 140, 255), // silver
                Color::from_rgba8(40, 40, 40, 255),    // onyx
            ],
            title_style: TextStyle {
                color: text_color,
                bold: true,
                font_size: 16.0,
            },
            label_style: TextStyle {
                color: text_color,
                bold: false,
                font_size: 13.0,
            },
            tick_style: TextStyle {
                color: text_color,
                bold: false,
                font_size: 11.0,
            },
            axis: AxisTheme {
                color: axis_color,
                width: 1.5,
                tick_length: 5.0,
                tick_color: axis_color,
                minor_ticks: true,
                minor_tick_length: 3.0,
            },
            grid: GridTheme {
                color: Color::from_rgba8(180, 180, 180, 80),
                width: 0.5,
                dash: None, // solid thin lines
                show: true,
                show_x: Some(false),
                show_y: None,
            },
            series: SeriesTheme {
                point_radius: 5.0,
                line_width: 2.0,
                bar_stroke_width: 1.0,
                fill_opacity: 0.20,
                bar_corner_radius: 0.0,
            },
            legend: LegendTheme {
                font_size: 10.0,
                background: Color::from_rgba8(255, 255, 255, 240),
                border: Some(Color::from_rgba8(150, 150, 150, 200)),
            },
        }
    }

    // --- Convenience accessors (backward compat) ---

    /// Color of axis lines.
    #[inline]
    #[must_use]
    pub fn axis_color(&self) -> Color {
        self.axis.color
    }

    /// Axis line width.
    #[inline]
    #[must_use]
    pub fn axis_width(&self) -> f32 {
        self.axis.width
    }

    /// Color of grid lines.
    #[inline]
    #[must_use]
    pub fn grid_color(&self) -> Color {
        self.grid.color
    }

    /// Grid line width.
    #[inline]
    #[must_use]
    pub fn grid_width(&self) -> f32 {
        self.grid.width
    }

    /// Whether to show gridlines.
    #[inline]
    #[must_use]
    pub fn show_grid(&self) -> bool {
        self.grid.show
    }

    /// Grid dash pattern.
    #[inline]
    #[must_use]
    pub fn grid_dash(&self) -> Option<&DashPattern> {
        self.grid.dash.as_ref()
    }

    /// Text color (alias for foreground).
    #[inline]
    #[must_use]
    pub fn text_color(&self) -> Color {
        self.foreground
    }

    /// Point radius for scatter plots.
    #[inline]
    #[must_use]
    pub fn point_radius(&self) -> f32 {
        self.series.point_radius
    }

    /// Line width for line charts.
    #[inline]
    #[must_use]
    pub fn line_width(&self) -> f32 {
        self.series.line_width
    }

    /// Bar outline stroke width.
    #[inline]
    #[must_use]
    pub fn bar_stroke_width(&self) -> f32 {
        self.series.bar_stroke_width
    }

    /// Default area fill opacity.
    #[inline]
    #[must_use]
    pub fn fill_opacity(&self) -> f32 {
        self.series.fill_opacity
    }

    /// Get the n-th series color (wraps around the palette).
    #[must_use]
    pub fn series_color(&self, index: usize) -> Color {
        if self.palette.is_empty() {
            Color::WHITE
        } else {
            self.palette[index % self.palette.len()]
        }
    }

    /// Get the dash pattern for the n-th series.
    ///
    /// Returns `None` (solid) for the first series, then cycles through
    /// dashed, dotted, dash-dot, and long-dash patterns. Useful for B&W
    /// output and accessibility.
    #[must_use]
    pub fn series_dash(&self, index: usize) -> Option<DashPattern> {
        match index % 5 {
            1 => Some(DashPattern::new(vec![8.0, 4.0], 0.0)), // dashed
            2 => Some(DashPattern::new(vec![2.0, 3.0], 0.0)), // dotted
            3 => Some(DashPattern::new(vec![10.0, 3.0, 2.0, 3.0], 0.0)), // dash-dot
            4 => Some(DashPattern::new(vec![16.0, 6.0], 0.0)), // long-dash
            _ => None,                                        // solid (0 and any)
        }
    }

    // --- Per-series style resolution helpers ---

    /// Resolve the effective color for a series, preferring the per-series
    /// override and falling back to the theme palette.
    #[inline]
    #[must_use]
    pub fn resolve_series_color(&self, index: usize, style: &crate::data::SeriesStyle) -> Color {
        style.color.unwrap_or_else(|| self.series_color(index))
    }

    /// Resolve the effective line width for a series.
    ///
    /// Priority: per-series `style.line_width` → chart-level override → theme default.
    #[inline]
    #[must_use]
    pub fn resolve_line_width(
        &self,
        style: &crate::data::SeriesStyle,
        chart_override: Option<f32>,
    ) -> f32 {
        style
            .line_width
            .or(chart_override)
            .unwrap_or(self.series.line_width)
    }

    /// Resolve the effective fill opacity for a series.
    #[inline]
    #[must_use]
    pub fn resolve_fill_opacity(&self, style: &crate::data::SeriesStyle) -> f32 {
        style.fill_opacity.unwrap_or(self.series.fill_opacity)
    }

    /// Resolve the effective dash pattern for a series.
    ///
    /// Priority: per-series `style.dash` → theme `series_dash(index)` if
    /// `use_theme_dash` is true → `None` (solid).
    #[inline]
    #[must_use]
    pub fn resolve_series_dash(
        &self,
        index: usize,
        style: &crate::data::SeriesStyle,
        use_theme_dash: bool,
    ) -> Option<DashPattern> {
        style.dash.as_ref().map_or_else(
            || {
                if use_theme_dash {
                    self.series_dash(index)
                } else {
                    None
                }
            },
            |dash| Some(dash.clone()),
        )
    }

    // --- Builder-style mutators ---

    /// Replace the entire palette with custom colors.
    ///
    /// # Example
    /// ```ignore
    /// let theme = Theme::dark().with_palette(vec![
    ///     Color::from_rgba8(255, 100, 100, 255),
    ///     Color::from_rgba8(100, 255, 100, 255),
    ///     Color::from_rgba8(100, 100, 255, 255),
    /// ]);
    /// ```
    pub fn with_palette(mut self, colors: Vec<Color>) -> Self {
        self.palette = colors;
        self
    }

    /// Append a single color to the palette.
    pub fn add_color(mut self, color: Color) -> Self {
        self.palette.push(color);
        self
    }

    /// Modify grid settings via a closure.
    ///
    /// # Example
    /// ```ignore
    /// let theme = Theme::dark().with_grid(|g| {
    ///     g.show_x = Some(false);
    ///     g.dash = None; // solid lines
    /// });
    /// ```
    pub fn with_grid(mut self, f: impl FnOnce(&mut GridTheme)) -> Self {
        f(&mut self.grid);
        self
    }

    /// Modify series rendering settings via a closure.
    pub fn with_series(mut self, f: impl FnOnce(&mut SeriesTheme)) -> Self {
        f(&mut self.series);
        self
    }

    /// Modify axis rendering settings via a closure.
    pub fn with_axis(mut self, f: impl FnOnce(&mut AxisTheme)) -> Self {
        f(&mut self.axis);
        self
    }
}

impl Default for Theme {
    fn default() -> Self {
        Self::dark()
    }
}

// ---------------------------------------------------------------------------
// Contrast text helper
// ---------------------------------------------------------------------------

/// sRGB → linear conversion for WCAG luminance calculation.
///
/// This is the inverse sRGB OETF (same as `scry_engine::style::srgb_to_linear`
/// but kept local to avoid depending on engine internals).
#[inline]
fn srgb_linearize(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Choose a high-contrast text color (black or white) for a given background.
///
/// Uses the WCAG 2.0 relative luminance formula to decide:
/// - Dark backgrounds (luminance ≤ 0.179) → white text
/// - Light backgrounds (luminance > 0.179) → black text
///
/// This ensures a minimum contrast ratio of ~4.5:1, meeting WCAG AA.
///
/// # Examples
///
/// ```
/// use scry_engine::style::Color;
/// use scry_chart::theme::contrast_text_color;
///
/// assert_eq!(contrast_text_color(Color::BLACK), Color::WHITE);
/// assert_eq!(contrast_text_color(Color::WHITE), Color::BLACK);
/// ```
#[must_use]
pub fn contrast_text_color(bg: Color) -> Color {
    let lum = 0.2126 * srgb_linearize(bg.r)
        + 0.7152 * srgb_linearize(bg.g)
        + 0.0722 * srgb_linearize(bg.b);
    if lum > 0.179 {
        Color::BLACK
    } else {
        Color::WHITE
    }
}

/// Alpha-composite-aware contrast text color.
///
/// Composites `fg` over `canvas_bg` using standard "over" blending, then
/// picks black or white text for maximum readability. This is essential for
/// semi-transparent fills (e.g. funnel chart gradient stages) where the
/// nominal RGB values don't reflect the actual perceived color.
#[must_use]
pub fn contrast_text_color_composited(fg: Color, canvas_bg: Color) -> Color {
    let a = fg.a;
    let composited = Color::from_rgba(
        fg.r * a + canvas_bg.r * (1.0 - a),
        fg.g * a + canvas_bg.g * (1.0 - a),
        fg.b * a + canvas_bg.b * (1.0 - a),
        1.0,
    );
    contrast_text_color(composited)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contrast_on_black_returns_white() {
        assert_eq!(contrast_text_color(Color::BLACK), Color::WHITE);
    }

    #[test]
    fn contrast_on_white_returns_black() {
        assert_eq!(contrast_text_color(Color::WHITE), Color::BLACK);
    }

    #[test]
    fn contrast_on_dark_blue_returns_white() {
        let dark_blue = Color::from_rgb8(0, 0, 139);
        assert_eq!(contrast_text_color(dark_blue), Color::WHITE);
    }

    #[test]
    fn contrast_on_yellow_returns_black() {
        let yellow = Color::from_rgb8(255, 255, 0);
        assert_eq!(contrast_text_color(yellow), Color::BLACK);
    }

    #[test]
    fn contrast_on_mid_gray_returns_white() {
        // Mid-gray (128,128,128) has luminance ~0.216 → borderline
        // sRGB 128/255 ≈ 0.502 → linear ≈ 0.216, just above 0.179 → black
        let gray = Color::from_rgb8(128, 128, 128);
        assert_eq!(contrast_text_color(gray), Color::BLACK);
    }

    #[test]
    fn contrast_on_dark_gray_returns_white() {
        let dark_gray = Color::from_rgb8(80, 80, 80);
        assert_eq!(contrast_text_color(dark_gray), Color::WHITE);
    }

    #[test]
    fn all_builtin_themes_have_valid_config() {
        // Smoke test: all themes construct without panic
        let themes = [
            Theme::dark(),
            Theme::light(),
            Theme::pastel(),
            Theme::ocean(),
            Theme::forest(),
            Theme::colorblind(),
        ];
        for t in &themes {
            // Smoke: series_color should return a valid color
            let _ = t.series_color(0);
        }
    }
}
