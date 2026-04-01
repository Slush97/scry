// SPDX-License-Identifier: MIT OR Apache-2.0
//! Theme and styling for Mermaid diagrams.

use scry_engine::scene::style::Color;

/// Visual theme for diagram rendering.
#[derive(Clone, Debug)]
pub struct MermaidTheme {
    /// Canvas background.
    pub background: Color,
    /// Default node fill.
    pub node_fill: Color,
    /// Node border color.
    pub node_stroke: Color,
    /// Node border width.
    pub node_stroke_width: f32,
    /// Node corner radius (for rounded rects).
    pub node_corner_radius: f32,
    /// Text color inside nodes.
    pub node_text_color: Color,
    /// Edge (arrow) color.
    pub edge_color: Color,
    /// Edge line width.
    pub edge_width: f32,
    /// Edge label text color.
    pub edge_label_color: Color,
    /// Font size for node labels.
    pub node_font_size: f32,
    /// Font size for edge labels.
    pub edge_font_size: f32,
    /// Decision node (diamond) fill.
    pub decision_fill: Color,
    /// Stadium/pill node fill.
    pub stadium_fill: Color,
}

impl MermaidTheme {
    /// Dark theme optimized for terminal backgrounds.
    #[must_use]
    pub fn dark() -> Self {
        Self {
            background: Color::from_rgba8(22, 22, 30, 255),
            node_fill: Color::from_rgba8(40, 44, 68, 255),
            node_stroke: Color::from_rgba8(100, 120, 200, 255),
            node_stroke_width: 2.0,
            node_corner_radius: 8.0,
            node_text_color: Color::from_rgba8(220, 225, 240, 255),
            edge_color: Color::from_rgba8(140, 150, 180, 255),
            edge_width: 2.0,
            edge_label_color: Color::from_rgba8(180, 185, 200, 255),
            node_font_size: 14.0,
            edge_font_size: 11.0,
            decision_fill: Color::from_rgba8(55, 45, 65, 255),
            stadium_fill: Color::from_rgba8(35, 55, 55, 255),
        }
    }

    /// Light theme for bright backgrounds.
    #[must_use]
    pub fn light() -> Self {
        Self {
            background: Color::from_rgba8(250, 250, 252, 255),
            node_fill: Color::from_rgba8(225, 230, 245, 255),
            node_stroke: Color::from_rgba8(80, 100, 180, 255),
            node_stroke_width: 2.0,
            node_corner_radius: 8.0,
            node_text_color: Color::from_rgba8(30, 30, 50, 255),
            edge_color: Color::from_rgba8(100, 110, 140, 255),
            edge_width: 2.0,
            edge_label_color: Color::from_rgba8(60, 65, 80, 255),
            node_font_size: 14.0,
            edge_font_size: 11.0,
            decision_fill: Color::from_rgba8(240, 230, 245, 255),
            stadium_fill: Color::from_rgba8(220, 240, 240, 255),
        }
    }
}

impl Default for MermaidTheme {
    fn default() -> Self {
        Self::dark()
    }
}

/// Layout configuration for diagrams.
#[derive(Clone, Debug)]
pub struct LayoutConfig {
    /// Horizontal spacing between nodes.
    pub node_spacing_x: f32,
    /// Vertical spacing between layers.
    pub node_spacing_y: f32,
    /// Minimum node width.
    pub min_node_width: f32,
    /// Minimum node height.
    pub min_node_height: f32,
    /// Horizontal padding inside nodes.
    pub node_padding_x: f32,
    /// Vertical padding inside nodes.
    pub node_padding_y: f32,
    /// Margin around the entire diagram.
    pub margin: f32,
    /// Arrowhead size.
    pub arrow_size: f32,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            node_spacing_x: 60.0,
            node_spacing_y: 80.0,
            min_node_width: 100.0,
            min_node_height: 40.0,
            node_padding_x: 20.0,
            node_padding_y: 12.0,
            margin: 30.0,
            arrow_size: 10.0,
        }
    }
}
