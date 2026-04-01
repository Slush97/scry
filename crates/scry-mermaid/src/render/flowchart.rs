// SPDX-License-Identifier: MIT OR Apache-2.0
//! Flowchart renderer — converts a parsed + laid-out flowchart into a `PixelCanvas`.

use scry_engine::scene::style::{Color, DashPattern};
use scry_engine::scene::PixelCanvas;

use crate::layout::flowchart::{FlowchartLayout, NodeLayout};
use crate::parser::flowchart::{Direction, EdgeStyle, FlowchartAst, NodeShape, Subgraph};
use crate::theme::{LayoutConfig, MermaidTheme};

use super::RenderedDiagram;

/// Render a flowchart AST into a `RenderedDiagram`.
///
/// If the natural layout exceeds `max_width` or `max_height`, all coordinates
/// and sizes are uniformly scaled down to fit within the bounds.
pub fn render(
    ast: &FlowchartAst,
    theme: &MermaidTheme,
    config: &LayoutConfig,
    max_width: u32,
    max_height: u32,
) -> RenderedDiagram {
    let layout = crate::layout::flowchart::layout(ast, config);

    // Count back-edges to determine routing margin.
    let num_back_edges = count_back_edges(ast, &layout);
    let back_edge_margin = if num_back_edges > 0 {
        20.0 + num_back_edges as f32 * 15.0
    } else {
        0.0
    };
    let natural_w = layout.width + back_edge_margin;
    let natural_h = layout.height + back_edge_margin;

    // Compute uniform scale factor to fit within bounds.
    let scale = if max_width > 0 && max_height > 0 {
        let sx = max_width as f32 / natural_w;
        let sy = max_height as f32 / natural_h;
        sx.min(sy).min(1.0) // never upscale
    } else {
        1.0
    };

    let w = (natural_w * scale).ceil() as u32;
    let h = (natural_h * scale).ceil() as u32;

    // Build a scaled theme so fonts, strokes, and arrows shrink proportionally.
    let scaled_theme = ScaledTheme::new(theme, config, scale);

    let mut canvas = PixelCanvas::new(w, h).background(theme.background);

    // Draw edges first (below nodes). Track back-edge index for staggered routing.
    let mut back_edge_idx: usize = 0;
    for edge in &ast.edges {
        let from_layout = match layout.nodes.get(&edge.from) {
            Some(l) => l,
            None => continue,
        };
        let to_layout = match layout.nodes.get(&edge.to) {
            Some(l) => l,
            None => continue,
        };
        let kind = classify_edge(from_layout, to_layout);
        let current_back_idx = if kind == EdgeKind::Back {
            let idx = back_edge_idx;
            back_edge_idx += 1;
            idx
        } else {
            0
        };
        canvas = draw_edge(
            canvas,
            from_layout,
            to_layout,
            &edge.style,
            edge.label.as_deref(),
            ast.direction,
            theme,
            config,
            &layout,
            current_back_idx,
            &scaled_theme,
        );
    }

    // Draw subgraph bounding boxes (behind nodes, above edges).
    for sg in &ast.subgraphs {
        canvas = draw_subgraph(canvas, sg, &layout, theme, &scaled_theme);
    }

    // Draw nodes on top.
    for node in &ast.nodes {
        if let Some(nl) = layout.nodes.get(&node.id) {
            canvas = draw_node(canvas, nl, &node.label, node.shape, theme, &scaled_theme);
        }
    }

    RenderedDiagram {
        canvas,
        width: w,
        height: h,
    }
}

/// Pre-computed scaled values for rendering at a given scale factor.
struct ScaledTheme {
    scale: f32,
    node_stroke_width: f32,
    node_corner_radius: f32,
    node_font_size: f32,
    edge_width: f32,
    edge_font_size: f32,
    arrow_size: f32,
}

impl ScaledTheme {
    fn new(theme: &MermaidTheme, config: &LayoutConfig, scale: f32) -> Self {
        Self {
            scale,
            node_stroke_width: theme.node_stroke_width * scale,
            node_corner_radius: theme.node_corner_radius * scale,
            node_font_size: theme.node_font_size * scale,
            edge_width: theme.edge_width * scale,
            edge_font_size: theme.edge_font_size * scale,
            arrow_size: config.arrow_size * scale,
        }
    }

    /// Scale a layout coordinate.
    fn s(&self, v: f32) -> f32 {
        v * self.scale
    }
}

/// Count how many back-edges exist (for margin calculation).
fn count_back_edges(ast: &FlowchartAst, layout: &FlowchartLayout) -> usize {
    ast.edges
        .iter()
        .filter(|e| {
            let fl = layout.nodes.get(&e.from).map(|n| n.layer);
            let tl = layout.nodes.get(&e.to).map(|n| n.layer);
            matches!((fl, tl), (Some(f), Some(t)) if f >= t)
        })
        .count()
}

fn draw_node(
    canvas: PixelCanvas,
    nl: &NodeLayout,
    label: &str,
    shape: NodeShape,
    theme: &MermaidTheme,
    st: &ScaledTheme,
) -> PixelCanvas {
    let r = &nl.rect;
    let x = st.s(r.left());
    let y = st.s(r.top());
    let w = st.s(r.w);
    let h = st.s(r.h);
    let cx = st.s(r.cx);
    let cy = st.s(r.cy);

    let canvas = match shape {
        NodeShape::Rectangle | NodeShape::Subroutine => canvas
            .rect(x, y, w, h)
            .fill(theme.node_fill)
            .stroke(theme.node_stroke, st.node_stroke_width)
            .corner_radius(2.0 * st.scale)
            .done(),
        NodeShape::Rounded => canvas
            .rect(x, y, w, h)
            .fill(theme.node_fill)
            .stroke(theme.node_stroke, st.node_stroke_width)
            .corner_radius(st.node_corner_radius)
            .done(),
        NodeShape::Stadium => canvas
            .rect(x, y, w, h)
            .fill(theme.stadium_fill)
            .stroke(theme.node_stroke, st.node_stroke_width)
            .corner_radius(h / 2.0)
            .done(),
        NodeShape::Diamond => {
            let points = diamond_points(cx, cy, w / 2.0, h / 2.0);
            canvas
                .polygon(points)
                .fill(theme.decision_fill)
                .stroke(theme.node_stroke, st.node_stroke_width)
                .done()
        }
        NodeShape::Circle => {
            let radius = w.min(h) / 2.0;
            canvas
                .circle(cx, cy, radius)
                .fill(theme.node_fill)
                .stroke(theme.node_stroke, st.node_stroke_width)
                .done()
        }
        NodeShape::Cylinder => {
            draw_cylinder(canvas, cx, cy, w, h, theme, st)
        }
    };

    // Draw subroutine double borders.
    let canvas = if shape == NodeShape::Subroutine {
        let inset = 4.0 * st.scale;
        canvas
            .rect(x + inset, y + inset, w - inset * 2.0, h - inset * 2.0)
            .stroke(theme.node_stroke, 1.0 * st.scale)
            .corner_radius(2.0 * st.scale)
            .done()
    } else {
        canvas
    };

    // Draw label text centered in the node.
    canvas
        .text(label, cx, cy + st.node_font_size * 0.35)
        .size(st.node_font_size)
        .color(theme.node_text_color)
        .align(scry_engine::scene::TextAlign::Center)
        .done()
}

/// Draw a cylinder (database shape): rect body + elliptical top and bottom caps.
fn draw_cylinder(
    canvas: PixelCanvas,
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    theme: &MermaidTheme,
    st: &ScaledTheme,
) -> PixelCanvas {
    let cap_ry = 8.0 * st.scale; // vertical radius of the elliptical caps
    let rx = w / 2.0;
    let body_top = cy - h / 2.0 + cap_ry;
    let body_bottom = cy + h / 2.0 - cap_ry;
    let body_h = body_bottom - body_top;

    // Body rectangle (no stroke — sides drawn separately so caps overlap cleanly).
    let canvas = canvas
        .rect(cx - rx, body_top, w, body_h)
        .fill(theme.node_fill)
        .done();

    // Side lines.
    let canvas = canvas
        .line(cx - rx, body_top, cx - rx, body_bottom)
        .color(theme.node_stroke)
        .width(st.node_stroke_width)
        .done();
    let canvas = canvas
        .line(cx + rx, body_top, cx + rx, body_bottom)
        .color(theme.node_stroke)
        .width(st.node_stroke_width)
        .done();

    // Bottom cap (filled, behind body).
    let canvas = canvas
        .ellipse(cx, body_bottom, rx, cap_ry)
        .fill(theme.node_fill)
        .stroke(theme.node_stroke, st.node_stroke_width)
        .done();

    // Top cap (filled, on top).
    canvas
        .ellipse(cx, body_top, rx, cap_ry)
        .fill(theme.node_fill)
        .stroke(theme.node_stroke, st.node_stroke_width)
        .done()
}

/// Draw a subgraph bounding box with a label.
fn draw_subgraph(
    canvas: PixelCanvas,
    sg: &Subgraph,
    layout: &FlowchartLayout,
    theme: &MermaidTheme,
    st: &ScaledTheme,
) -> PixelCanvas {
    // Compute bounding box of all member nodes.
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    let mut found = false;
    for id in &sg.node_ids {
        if let Some(nl) = layout.nodes.get(id) {
            found = true;
            min_x = min_x.min(nl.rect.left());
            min_y = min_y.min(nl.rect.top());
            max_x = max_x.max(nl.rect.right());
            max_y = max_y.max(nl.rect.bottom());
        }
    }

    if !found {
        return canvas;
    }

    // Add padding around the group.
    let pad = 20.0;
    let label_h = 20.0; // space for the label at the top
    min_x -= pad;
    min_y -= pad + label_h;
    max_x += pad;
    max_y += pad;

    // Scale to pixel space.
    let x = st.s(min_x);
    let y = st.s(min_y);
    let w = st.s(max_x - min_x);
    let h = st.s(max_y - min_y);

    // Semi-transparent fill for the group background.
    let bg = Color {
        r: theme.node_stroke.r,
        g: theme.node_stroke.g,
        b: theme.node_stroke.b,
        a: 0.1,
    };

    let canvas = canvas
        .rect(x, y, w, h)
        .fill(bg)
        .stroke(theme.node_stroke, st.node_stroke_width * 0.7)
        .corner_radius(st.node_corner_radius)
        .done();

    // Label at top-left of the box.
    let label_x = x + 10.0 * st.scale;
    let label_y = y + st.node_font_size * 0.9;
    canvas
        .text(&sg.label, label_x, label_y)
        .size(st.node_font_size * 0.85)
        .color(theme.edge_label_color)
        .done()
}

fn diamond_points(cx: f32, cy: f32, rx: f32, ry: f32) -> Vec<(f32, f32)> {
    vec![
        (cx, cy - ry), // top
        (cx + rx, cy), // right
        (cx, cy + ry), // bottom
        (cx - rx, cy), // left
    ]
}

/// Classify an edge based on the layer relationship between source and target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EdgeKind {
    /// Source layer < target layer (normal forward edge).
    Forward,
    /// Source layer >= target layer (back-edge / cycle).
    Back,
    /// Source and target are the same node.
    SelfLoop,
}

fn classify_edge(from: &NodeLayout, to: &NodeLayout) -> EdgeKind {
    if from.rect.cx == to.rect.cx && from.rect.cy == to.rect.cy {
        EdgeKind::SelfLoop
    } else if from.layer >= to.layer {
        EdgeKind::Back
    } else {
        EdgeKind::Forward
    }
}

/// Draw an edge (line + arrowhead + optional label).
#[allow(clippy::too_many_arguments)]
fn draw_edge(
    canvas: PixelCanvas,
    from: &NodeLayout,
    to: &NodeLayout,
    style: &EdgeStyle,
    label: Option<&str>,
    direction: Direction,
    theme: &MermaidTheme,
    config: &LayoutConfig,
    layout: &FlowchartLayout,
    back_edge_idx: usize,
    st: &ScaledTheme,
) -> PixelCanvas {
    let kind = classify_edge(from, to);
    let edge_color = theme.edge_color;
    let width = match style {
        EdgeStyle::ThickArrow => st.edge_width * 1.8,
        _ => st.edge_width,
    };

    match kind {
        EdgeKind::SelfLoop => draw_self_loop(canvas, from, direction, style, edge_color, width, st),
        EdgeKind::Back => draw_back_edge(
            canvas, from, to, style, label, direction, theme, config, layout, edge_color, width,
            back_edge_idx, st,
        ),
        EdgeKind::Forward => draw_forward_edge(
            canvas, from, to, style, label, direction, theme, edge_color, width, st,
        ),
    }
}

/// Draw a normal forward edge (source layer < target layer).
///
/// Uses orthogonal routing: for non-aligned nodes, draws 3 segments
/// (primary → lateral → primary) instead of diagonals that cross through nodes.
#[allow(clippy::too_many_arguments)]
fn draw_forward_edge(
    canvas: PixelCanvas,
    from: &NodeLayout,
    to: &NodeLayout,
    style: &EdgeStyle,
    label: Option<&str>,
    direction: Direction,
    theme: &MermaidTheme,
    edge_color: Color,
    width: f32,
    st: &ScaledTheme,
) -> PixelCanvas {
    let (x1, y1) = exit_point_forward(&from.rect, direction);
    let (x2, y2) = entry_point_forward(&to.rect, direction);

    let is_aligned = match direction {
        Direction::TB | Direction::BT => (x1 - x2).abs() < 5.0,
        Direction::LR | Direction::RL => (y1 - y2).abs() < 5.0,
    };

    // Build the list of waypoints (in layout space, scaled later).
    let pts: Vec<(f32, f32)> = if is_aligned {
        vec![(x1, y1), (x2, y2)]
    } else {
        match direction {
            Direction::TB | Direction::BT => {
                let mid_y = (y1 + y2) / 2.0;
                vec![(x1, y1), (x1, mid_y), (x2, mid_y), (x2, y2)]
            }
            Direction::LR | Direction::RL => {
                let mid_x = (x1 + x2) / 2.0;
                vec![(x1, y1), (mid_x, y1), (mid_x, y2), (x2, y2)]
            }
        }
    };

    // Scale all points.
    let points: Vec<(f32, f32)> = pts.iter().map(|&(px, py)| (st.s(px), st.s(py))).collect();

    // Draw all segments.
    let mut canvas = canvas;
    for i in 0..points.len() - 1 {
        canvas = draw_line_segment(
            canvas,
            points[i].0, points[i].1,
            points[i + 1].0, points[i + 1].1,
            style, edge_color, width, st,
        );
    }

    // Arrowhead on the final segment.
    let n = points.len();
    let canvas = match style {
        EdgeStyle::SolidLine => canvas,
        _ => draw_arrowhead(
            canvas,
            points[n - 2].0, points[n - 2].1,
            points[n - 1].0, points[n - 1].1,
            edge_color, st.arrow_size,
        ),
    };

    // Label at midpoint of the full edge span.
    draw_edge_label(canvas, st.s(x1), st.s(y1), st.s(x2), st.s(y2), label, theme, st)
}

/// Draw a back-edge (cycle edge going backward in layers).
/// Routes around the outside of the diagram to avoid crossing through nodes.
#[allow(clippy::too_many_arguments)]
fn draw_back_edge(
    canvas: PixelCanvas,
    from: &NodeLayout,
    to: &NodeLayout,
    style: &EdgeStyle,
    label: Option<&str>,
    direction: Direction,
    theme: &MermaidTheme,
    config: &LayoutConfig,
    layout: &FlowchartLayout,
    edge_color: Color,
    width: f32,
    back_edge_idx: usize,
    st: &ScaledTheme,
) -> PixelCanvas {
    let base_offset = 15.0;
    let stagger = back_edge_idx as f32 * 15.0;

    // For back-edges, exit from the side and route around (layout space).
    let (raw_points, raw_lx, raw_ly) = match direction {
        Direction::TB | Direction::BT => {
            let exit_x = from.rect.right();
            let exit_y = from.rect.cy;
            let enter_x = to.rect.right();
            let enter_y = to.rect.cy;
            let route_x = layout.width - config.margin / 2.0 + base_offset + stagger;

            let pts = vec![
                (exit_x, exit_y),
                (route_x, exit_y),
                (route_x, enter_y),
                (enter_x, enter_y),
            ];
            (pts, route_x, (exit_y + enter_y) / 2.0)
        }
        Direction::LR | Direction::RL => {
            let exit_x = from.rect.cx;
            let exit_y = from.rect.bottom();
            let enter_x = to.rect.cx;
            let enter_y = to.rect.bottom();
            let route_y = layout.height - config.margin / 2.0 + base_offset + stagger;

            let pts = vec![
                (exit_x, exit_y),
                (exit_x, route_y),
                (enter_x, route_y),
                (enter_x, enter_y),
            ];
            (pts, (exit_x + enter_x) / 2.0, route_y)
        }
    };

    // Scale to pixel space.
    let points: Vec<(f32, f32)> = raw_points.iter().map(|&(px, py)| (st.s(px), st.s(py))).collect();
    let label_x = st.s(raw_lx);
    let label_y = st.s(raw_ly);

    // Draw the polyline segments.
    let mut canvas = canvas;
    for i in 0..points.len() - 1 {
        let (ax, ay) = points[i];
        let (bx, by) = points[i + 1];
        canvas = draw_line_segment(canvas, ax, ay, bx, by, style, edge_color, width, st);
    }

    // Arrowhead on the last segment.
    let canvas = match style {
        EdgeStyle::SolidLine => canvas,
        _ => {
            let n = points.len();
            let (px, py) = points[n - 2];
            let (qx, qy) = points[n - 1];
            draw_arrowhead(canvas, px, py, qx, qy, edge_color, st.arrow_size)
        }
    };

    // Label near the routing path.
    if let Some(text) = label {
        let label_w = text.len() as f32 * st.edge_font_size * 0.55 + 10.0 * st.scale;
        let label_h = st.edge_font_size + 6.0 * st.scale;
        let canvas = canvas
            .rect(
                label_x - label_w / 2.0,
                label_y - label_h / 2.0,
                label_w,
                label_h,
            )
            .fill(theme.background)
            .corner_radius(3.0 * st.scale)
            .done();
        canvas
            .text(text, label_x, label_y + st.edge_font_size * 0.35)
            .size(st.edge_font_size)
            .color(theme.edge_label_color)
            .align(scry_engine::scene::TextAlign::Center)
            .done()
    } else {
        canvas
    }
}

/// Draw a self-loop (edge from a node to itself).
fn draw_self_loop(
    canvas: PixelCanvas,
    node: &NodeLayout,
    direction: Direction,
    style: &EdgeStyle,
    edge_color: Color,
    width: f32,
    st: &ScaledTheme,
) -> PixelCanvas {
    let r = &node.rect;
    let loop_size = 20.0;

    // Build waypoints in layout space.
    let (raw_points, raw_arrow_from, raw_arrow_to) = match direction {
        Direction::TB | Direction::BT => {
            let x1 = r.right();
            let y1 = r.cy - 8.0;
            let x2 = r.right();
            let y2 = r.cy + 8.0;
            let px = x1 + loop_size;
            (
                vec![(x1, y1), (px, y1), (px, y2), (x2, y2)],
                (px, y2),
                (x2, y2),
            )
        }
        Direction::LR | Direction::RL => {
            let x1 = r.cx - 8.0;
            let y1 = r.bottom();
            let x2 = r.cx + 8.0;
            let y2 = r.bottom();
            let py = y1 + loop_size;
            (
                vec![(x1, y1), (x1, py), (x2, py), (x2, y2)],
                (x2, py),
                (x2, y2),
            )
        }
    };

    // Scale to pixel space.
    let points: Vec<(f32, f32)> = raw_points.iter().map(|&(px, py)| (st.s(px), st.s(py))).collect();
    let arrow_from = (st.s(raw_arrow_from.0), st.s(raw_arrow_from.1));
    let arrow_to = (st.s(raw_arrow_to.0), st.s(raw_arrow_to.1));

    let mut canvas = canvas;
    for i in 0..points.len() - 1 {
        let (ax, ay) = points[i];
        let (bx, by) = points[i + 1];
        canvas = draw_line_segment(canvas, ax, ay, bx, by, style, edge_color, width, st);
    }

    match style {
        EdgeStyle::SolidLine => canvas,
        _ => draw_arrowhead(
            canvas,
            arrow_from.0,
            arrow_from.1,
            arrow_to.0,
            arrow_to.1,
            edge_color,
            st.arrow_size,
        ),
    }
}

/// Draw a single line segment with the appropriate style (solid, dotted, thick).
///
/// Coordinates are expected to be already in scaled pixel space.
fn draw_line_segment(
    canvas: PixelCanvas,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    style: &EdgeStyle,
    color: Color,
    width: f32,
    st: &ScaledTheme,
) -> PixelCanvas {
    match style {
        EdgeStyle::DottedArrow => canvas
            .line(x1, y1, x2, y2)
            .color(color)
            .width(width)
            .dash(DashPattern {
                intervals: vec![6.0 * st.scale, 4.0 * st.scale],
                offset: 0.0,
            })
            .done(),
        _ => canvas.line(x1, y1, x2, y2).color(color).width(width).done(),
    }
}

/// Draw an optional edge label at the midpoint between two points.
///
/// Coordinates are expected to be already in scaled pixel space.
fn draw_edge_label(
    canvas: PixelCanvas,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    label: Option<&str>,
    theme: &MermaidTheme,
    st: &ScaledTheme,
) -> PixelCanvas {
    if let Some(text) = label {
        let mx = (x1 + x2) / 2.0;
        let my = (y1 + y2) / 2.0;

        let label_w = text.len() as f32 * st.edge_font_size * 0.55 + 10.0 * st.scale;
        let label_h = st.edge_font_size + 6.0 * st.scale;
        let canvas = canvas
            .rect(
                mx - label_w / 2.0,
                my - label_h / 2.0,
                label_w,
                label_h,
            )
            .fill(theme.background)
            .corner_radius(3.0 * st.scale)
            .done();

        canvas
            .text(text, mx, my + st.edge_font_size * 0.35)
            .size(st.edge_font_size)
            .color(theme.edge_label_color)
            .align(scry_engine::scene::TextAlign::Center)
            .done()
    } else {
        canvas
    }
}

/// Compute where a forward edge exits a node.
fn exit_point_forward(
    rect: &crate::layout::PositionedRect,
    dir: Direction,
) -> (f32, f32) {
    match dir {
        Direction::TB => (rect.cx, rect.bottom()),
        Direction::BT => (rect.cx, rect.top()),
        Direction::LR => (rect.right(), rect.cy),
        Direction::RL => (rect.left(), rect.cy),
    }
}

/// Compute where a forward edge enters a node.
fn entry_point_forward(
    rect: &crate::layout::PositionedRect,
    dir: Direction,
) -> (f32, f32) {
    match dir {
        Direction::TB => (rect.cx, rect.top()),
        Direction::BT => (rect.cx, rect.bottom()),
        Direction::LR => (rect.left(), rect.cy),
        Direction::RL => (rect.right(), rect.cy),
    }
}

/// Draw an arrowhead triangle at the end of a line.
fn draw_arrowhead(
    canvas: PixelCanvas,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    color: Color,
    size: f32,
) -> PixelCanvas {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 0.01 {
        return canvas;
    }

    let ux = dx / len;
    let uy = dy / len;

    // Perpendicular.
    let px = -uy;
    let py = ux;

    let tip_x = x2;
    let tip_y = y2;
    let base_x = x2 - ux * size;
    let base_y = y2 - uy * size;

    let half = size * 0.4;
    let left_x = base_x + px * half;
    let left_y = base_y + py * half;
    let right_x = base_x - px * half;
    let right_y = base_y - py * half;

    canvas
        .polygon(vec![(tip_x, tip_y), (left_x, left_y), (right_x, right_y)])
        .fill(color)
        .done()
}
