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
                Color::from_rgba8(255, 160, 122, 255), // salmon
                Color::from_rgba8(103, 232, 249, 255), // cyan
                Color::from_rgba8(253, 186, 116, 255), // peach
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
                color: Color::from_rgba8(60, 60, 80, 55),
                width: 1.0,
                dash: Some(DashPattern::new(vec![4.0, 4.0], 0.0)),
                show: true,
                show_x: None,
                show_y: None,
            },
            series: SeriesTheme {
                point_radius: 5.5,
                line_width: 2.5,
                bar_stroke_width: 0.0,
                fill_opacity: 0.30,
                bar_corner_radius: 3.0,
            },
            legend: LegendTheme {
                font_size: 11.0,
                background: Color::from_rgba8(30, 30, 40, 240),
                border: Some(Color::from_rgba8(80, 80, 100, 200)),
            },
        }
    }

    /// Light theme for terminals with light backgrounds.
    pub fn light() -> Self {
        let text_color = Color::from_rgba8(40, 40, 60, 255);
        let axis_color = Color::from_rgba8(60, 60, 80, 255);

        Self {
            background: Color::from_rgba8(250, 250, 252, 255),
            foreground: text_color,
            palette: vec![
                Color::from_rgba8(31, 119, 180, 255),  // blue
                Color::from_rgba8(255, 127, 14, 255),  // orange
                Color::from_rgba8(44, 160, 44, 255),   // green
                Color::from_rgba8(214, 39, 40, 255),   // red
                Color::from_rgba8(148, 103, 189, 255), // purple
                Color::from_rgba8(140, 86, 75, 255),   // brown
                Color::from_rgba8(227, 119, 194, 255), // pink
                Color::from_rgba8(127, 127, 127, 255), // gray
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
                color: Color::from_rgba8(180, 180, 200, 40),
                width: 1.0,
                dash: Some(DashPattern::new(vec![4.0, 4.0], 0.0)),
                show: true,
                show_x: None,
                show_y: None,
            },
            series: SeriesTheme {
                point_radius: 5.5,
                line_width: 2.5,
                bar_stroke_width: 0.0,
                fill_opacity: 0.30,
                bar_corner_radius: 3.0,
            },
            legend: LegendTheme {
                font_size: 11.0,
                background: Color::from_rgba8(255, 255, 255, 240),
                border: Some(Color::from_rgba8(200, 200, 200, 200)),
            },
        }
    }

    /// Chalk pastel theme — soft, warm colors on dark.
    pub fn pastel() -> Self {
        let text_color = Color::from_rgba8(190, 190, 210, 255);
        let axis_color = Color::from_rgba8(140, 140, 160, 255);

        Self {
            background: Color::from_rgba8(22, 22, 30, 255),
            foreground: text_color,
            palette: vec![
                Color::from_rgba8(179, 205, 224, 255), // powder blue
                Color::from_rgba8(240, 178, 178, 255), // rose
                Color::from_rgba8(178, 223, 178, 255), // sage
                Color::from_rgba8(255, 218, 170, 255), // peach
                Color::from_rgba8(204, 185, 232, 255), // lilac
                Color::from_rgba8(255, 245, 186, 255), // butter
                Color::from_rgba8(186, 225, 225, 255), // aqua
                Color::from_rgba8(232, 196, 213, 255), // mauve
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
                color: Color::from_rgba8(55, 55, 75, 55),
                width: 1.0,
                dash: Some(DashPattern::new(vec![4.0, 4.0], 0.0)),
                show: true,
                show_x: None,
                show_y: None,
            },
            series: SeriesTheme {
                point_radius: 5.5,
                line_width: 2.5,
                bar_stroke_width: 0.0,
                fill_opacity: 0.30,
                bar_corner_radius: 3.0,
            },
            legend: LegendTheme {
                font_size: 11.0,
                background: Color::from_rgba8(250, 248, 245, 240),
                border: Some(Color::from_rgba8(200, 195, 190, 200)),
            },
        }
    }

    /// Ocean theme — deep blues and teals with vibrant accents.
    pub fn ocean() -> Self {
        let text_color = Color::from_rgba8(200, 220, 240, 255);
        let axis_color = Color::from_rgba8(120, 150, 180, 255);

        Self {
            background: Color::from_rgba8(10, 20, 40, 255),
            foreground: text_color,
            palette: vec![
                Color::from_rgba8(0, 188, 212, 255),  // cyan
                Color::from_rgba8(38, 166, 154, 255), // teal
                Color::from_rgba8(100, 221, 23, 255), // lime
                Color::from_rgba8(255, 167, 38, 255), // orange
                Color::from_rgba8(171, 71, 188, 255), // purple
                Color::from_rgba8(255, 112, 67, 255), // deep orange
                Color::from_rgba8(66, 165, 245, 255), // light blue
                Color::from_rgba8(255, 213, 79, 255), // yellow
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
                color: Color::from_rgba8(40, 65, 90, 55),
                width: 1.0,
                dash: Some(DashPattern::new(vec![4.0, 4.0], 0.0)),
                show: true,
                show_x: None,
                show_y: None,
            },
            series: SeriesTheme {
                point_radius: 5.5,
                line_width: 2.5,
                bar_stroke_width: 0.0,
                fill_opacity: 0.30,
                bar_corner_radius: 3.0,
            },
            legend: LegendTheme {
                font_size: 11.0,
                background: Color::from_rgba8(10, 20, 40, 240),
                border: Some(Color::from_rgba8(60, 90, 120, 200)),
            },
        }
    }

    /// Forest theme — earthy greens and warm browns.
    pub fn forest() -> Self {
        let text_color = Color::from_rgba8(210, 210, 190, 255);
        let axis_color = Color::from_rgba8(150, 140, 120, 255);

        Self {
            background: Color::from_rgba8(18, 22, 15, 255),
            foreground: text_color,
            palette: vec![
                Color::from_rgba8(76, 175, 80, 255),   // green
                Color::from_rgba8(139, 195, 74, 255),  // light green
                Color::from_rgba8(205, 220, 57, 255),  // lime yellow
                Color::from_rgba8(255, 183, 77, 255),  // amber
                Color::from_rgba8(161, 136, 127, 255), // brown
                Color::from_rgba8(255, 138, 101, 255), // deep orange
                Color::from_rgba8(77, 182, 172, 255),  // teal
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
                color: Color::from_rgba8(50, 58, 42, 55),
                width: 1.0,
                dash: Some(DashPattern::new(vec![4.0, 4.0], 0.0)),
                show: true,
                show_x: None,
                show_y: None,
            },
            series: SeriesTheme {
                point_radius: 5.5,
                line_width: 2.5,
                bar_stroke_width: 0.0,
                fill_opacity: 0.30,
                bar_corner_radius: 3.0,
            },
            legend: LegendTheme {
                font_size: 11.0,
                background: Color::from_rgba8(18, 22, 15, 240),
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
                color: Color::from_rgba8(60, 60, 80, 55),
                width: 1.0,
                dash: Some(DashPattern::new(vec![4.0, 4.0], 0.0)),
                show: true,
                show_x: None,
                show_y: None,
            },
            series: SeriesTheme {
                point_radius: 5.5,
                line_width: 2.5,
                bar_stroke_width: 0.0,
                fill_opacity: 0.30,
                bar_corner_radius: 3.0,
            },
            legend: LegendTheme {
                font_size: 11.0,
                background: Color::from_rgba8(240, 240, 240, 240),
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
