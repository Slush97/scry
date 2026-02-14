//! Color themes and palettes for charts.
//!
//! Ships with dark-terminal-optimized defaults and seaborn-inspired palettes.
//! Themes use a hierarchical token system for fine-grained control.

use ratatui_pixelcanvas::style::Color;
use ratatui_pixelcanvas::style::DashPattern;

// ---------------------------------------------------------------------------
// Sub-structs
// ---------------------------------------------------------------------------

/// Text styling tokens.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TextStyle {
    /// Text color.
    pub color: Color,
    /// Whether text should be bold.
    pub bold: bool,
}

/// Axis rendering tokens.
#[derive(Clone, Debug)]
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

/// A complete visual theme for chart rendering.
///
/// Uses hierarchical tokens: `axis`, `grid`, `series`, and text styles
/// for fine-grained control over every chart element.
#[derive(Clone, Debug)]
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
            },
            label_style: TextStyle {
                color: text_color,
                bold: false,
            },
            tick_style: TextStyle {
                color: text_color,
                bold: false,
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
                color: Color::from_rgba8(60, 60, 80, 180),
                width: 1.0,
                dash: Some(DashPattern::new(vec![4.0, 4.0], 0.0)),
                show: true,
                show_x: None,
                show_y: None,
            },
            series: SeriesTheme {
                point_radius: 5.5,
                line_width: 2.5,
                bar_stroke_width: 1.5,
                fill_opacity: 0.30,
                bar_corner_radius: 3.0,
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
            },
            label_style: TextStyle {
                color: text_color,
                bold: false,
            },
            tick_style: TextStyle {
                color: text_color,
                bold: false,
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
                color: Color::from_rgba8(180, 180, 200, 160),
                width: 1.0,
                dash: Some(DashPattern::new(vec![4.0, 4.0], 0.0)),
                show: true,
                show_x: None,
                show_y: None,
            },
            series: SeriesTheme {
                point_radius: 5.5,
                line_width: 2.5,
                bar_stroke_width: 1.5,
                fill_opacity: 0.30,
                bar_corner_radius: 3.0,
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
            },
            label_style: TextStyle {
                color: text_color,
                bold: false,
            },
            tick_style: TextStyle {
                color: text_color,
                bold: false,
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
                color: Color::from_rgba8(55, 55, 75, 180),
                width: 1.0,
                dash: Some(DashPattern::new(vec![4.0, 4.0], 0.0)),
                show: true,
                show_x: None,
                show_y: None,
            },
            series: SeriesTheme {
                point_radius: 5.5,
                line_width: 2.5,
                bar_stroke_width: 1.5,
                fill_opacity: 0.30,
                bar_corner_radius: 3.0,
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
            },
            label_style: TextStyle {
                color: text_color,
                bold: false,
            },
            tick_style: TextStyle {
                color: text_color,
                bold: false,
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
                color: Color::from_rgba8(40, 65, 90, 180),
                width: 1.0,
                dash: Some(DashPattern::new(vec![4.0, 4.0], 0.0)),
                show: true,
                show_x: None,
                show_y: None,
            },
            series: SeriesTheme {
                point_radius: 5.5,
                line_width: 2.5,
                bar_stroke_width: 1.5,
                fill_opacity: 0.30,
                bar_corner_radius: 3.0,
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
            },
            label_style: TextStyle {
                color: text_color,
                bold: false,
            },
            tick_style: TextStyle {
                color: text_color,
                bold: false,
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
                color: Color::from_rgba8(50, 58, 42, 180),
                width: 1.0,
                dash: Some(DashPattern::new(vec![4.0, 4.0], 0.0)),
                show: true,
                show_x: None,
                show_y: None,
            },
            series: SeriesTheme {
                point_radius: 5.5,
                line_width: 2.5,
                bar_stroke_width: 1.5,
                fill_opacity: 0.30,
                bar_corner_radius: 3.0,
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
            0 => None,                                                   // solid
            1 => Some(DashPattern::new(vec![8.0, 4.0], 0.0)),            // dashed
            2 => Some(DashPattern::new(vec![2.0, 3.0], 0.0)),            // dotted
            3 => Some(DashPattern::new(vec![10.0, 3.0, 2.0, 3.0], 0.0)), // dash-dot
            4 => Some(DashPattern::new(vec![16.0, 6.0], 0.0)),           // long-dash
            _ => None,
        }
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
