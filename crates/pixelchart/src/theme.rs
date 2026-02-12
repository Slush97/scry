//! Color themes and palettes for charts.
//!
//! Ships with dark-terminal-optimized defaults and seaborn-inspired palettes.

use ratatui_pixelcanvas::style::Color;

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

/// A complete visual theme for chart rendering.
#[derive(Clone, Debug)]
pub struct Theme {
    /// Background color of the plot area.
    pub background: Color,
    /// Color of axis lines.
    pub axis_color: Color,
    /// Color of grid lines.
    pub grid_color: Color,
    /// Color of axis label text.
    pub text_color: Color,
    /// Ordered palette of data series colors.
    pub palette: Vec<Color>,
    /// Point radius for scatter plots (pixels).
    pub point_radius: f32,
    /// Line width for line charts (pixels).
    pub line_width: f32,
    /// Axis line width (pixels).
    pub axis_width: f32,
    /// Grid line width (pixels).
    pub grid_width: f32,
    /// Whether to draw gridlines.
    pub show_grid: bool,
}

impl Theme {
    /// Dark theme optimized for terminal backgrounds.
    ///
    /// Uses muted neon colors that look great on dark terminals.
    pub fn dark() -> Self {
        Self {
            background: Color::from_rgba8(15, 15, 25, 255),
            axis_color: Color::from_rgba8(160, 160, 180, 255),
            grid_color: Color::from_rgba8(50, 50, 65, 120),
            text_color: Color::from_rgba8(200, 200, 220, 255),
            palette: vec![
                Color::from_rgba8(99, 179, 237, 255),  // sky blue
                Color::from_rgba8(252, 129, 155, 255),  // coral pink
                Color::from_rgba8(134, 239, 172, 255),  // mint green
                Color::from_rgba8(251, 191, 36, 255),   // amber
                Color::from_rgba8(196, 167, 255, 255),  // lavender
                Color::from_rgba8(255, 160, 122, 255),  // salmon
                Color::from_rgba8(103, 232, 249, 255),  // cyan
                Color::from_rgba8(253, 186, 116, 255),  // peach
            ],
            point_radius: 4.0,
            line_width: 2.5,
            axis_width: 1.5,
            grid_width: 0.5,
            show_grid: true,
        }
    }

    /// Light theme for terminals with light backgrounds.
    pub fn light() -> Self {
        Self {
            background: Color::from_rgba8(250, 250, 252, 255),
            axis_color: Color::from_rgba8(60, 60, 80, 255),
            grid_color: Color::from_rgba8(200, 200, 210, 100),
            text_color: Color::from_rgba8(40, 40, 60, 255),
            palette: vec![
                Color::from_rgba8(31, 119, 180, 255),   // blue
                Color::from_rgba8(255, 127, 14, 255),   // orange
                Color::from_rgba8(44, 160, 44, 255),    // green
                Color::from_rgba8(214, 39, 40, 255),    // red
                Color::from_rgba8(148, 103, 189, 255),  // purple
                Color::from_rgba8(140, 86, 75, 255),    // brown
                Color::from_rgba8(227, 119, 194, 255),  // pink
                Color::from_rgba8(127, 127, 127, 255),  // gray
            ],
            point_radius: 4.0,
            line_width: 2.5,
            axis_width: 1.5,
            grid_width: 0.5,
            show_grid: true,
        }
    }

    /// Chalk pastel theme — soft, warm colors on dark.
    pub fn pastel() -> Self {
        Self {
            background: Color::from_rgba8(22, 22, 30, 255),
            axis_color: Color::from_rgba8(140, 140, 160, 255),
            grid_color: Color::from_rgba8(45, 45, 60, 100),
            text_color: Color::from_rgba8(190, 190, 210, 255),
            palette: vec![
                Color::from_rgba8(179, 205, 224, 255),  // powder blue
                Color::from_rgba8(240, 178, 178, 255),  // rose
                Color::from_rgba8(178, 223, 178, 255),  // sage
                Color::from_rgba8(255, 218, 170, 255),  // peach
                Color::from_rgba8(204, 185, 232, 255),  // lilac
                Color::from_rgba8(255, 245, 186, 255),  // butter
                Color::from_rgba8(186, 225, 225, 255),  // aqua
                Color::from_rgba8(232, 196, 213, 255),  // mauve
            ],
            point_radius: 5.0,
            line_width: 2.0,
            axis_width: 1.0,
            grid_width: 0.5,
            show_grid: true,
        }
    }

    /// Get the n-th series color (wraps around the palette).
    pub fn series_color(&self, index: usize) -> Color {
        if self.palette.is_empty() {
            Color::WHITE
        } else {
            self.palette[index % self.palette.len()]
        }
    }
}

impl Default for Theme {
    fn default() -> Self {
        Self::dark()
    }
}
