// SPDX-License-Identifier: MIT OR Apache-2.0
//! Layered graph layout for flowcharts (simplified Sugiyama).
//!
//! Steps:
//! 1. Assign each node to a layer (longest-path from sources).
//! 2. Order nodes within layers to minimize edge crossings (barycenter heuristic).
//! 3. Assign concrete (x, y) coordinates based on layer/position.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::parser::flowchart::{Direction, FlowchartAst, NodeShape};
use crate::theme::LayoutConfig;

use super::PositionedRect;

/// Result of laying out a flowchart.
#[derive(Clone, Debug)]
pub struct FlowchartLayout {
    /// Node positions keyed by node id.
    pub nodes: HashMap<String, NodeLayout>,
    /// Total canvas width required.
    pub width: f32,
    /// Total canvas height required.
    pub height: f32,
}

/// Layout data for a single node.
#[derive(Clone, Debug)]
pub struct NodeLayout {
    /// Pixel-space bounding box.
    pub rect: PositionedRect,
    /// Which layer (rank) this node is in.
    pub layer: usize,
    /// Position within the layer.
    pub order: usize,
}

/// Compute layout for a flowchart AST.
pub fn layout(ast: &FlowchartAst, config: &LayoutConfig) -> FlowchartLayout {
    let node_ids: Vec<&str> = ast.nodes.iter().map(|n| n.id.as_str()).collect();
    let id_to_idx: HashMap<&str, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();

    // Build adjacency list.
    let n = node_ids.len();
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];

    for edge in &ast.edges {
        if let (Some(&from), Some(&to)) = (
            id_to_idx.get(edge.from.as_str()),
            id_to_idx.get(edge.to.as_str()),
        ) {
            adj[from].push(to);
        }
    }

    // Detect and remove back-edges (cycles) before layer assignment.
    let back_edges = detect_back_edges(n, &adj);
    let dag_adj = remove_edges(n, &adj, &back_edges);

    let mut dag_in_deg = vec![0usize; n];
    for neighbors in &dag_adj {
        for &v in neighbors {
            dag_in_deg[v] += 1;
        }
    }

    // Step 1: Assign layers via longest-path from sources (topological order on the DAG).
    let layers = assign_layers(n, &dag_adj, &dag_in_deg);

    // Build layer → [node indices] mapping.
    let max_layer = layers.iter().copied().max().unwrap_or(0);
    let mut layer_members: Vec<Vec<usize>> = vec![vec![]; max_layer + 1];
    for (i, &layer) in layers.iter().enumerate() {
        layer_members[layer].push(i);
    }

    // Step 2: Order nodes within layers (barycenter heuristic).
    order_layers(&mut layer_members, &adj, n);

    // Compute node sizes based on labels and shapes.
    let node_sizes: Vec<(f32, f32)> = ast
        .nodes
        .iter()
        .map(|node| compute_node_size(node.label.as_str(), node.shape, config))
        .collect();

    // Step 3: Assign coordinates.
    let is_horizontal = matches!(ast.direction, Direction::LR | Direction::RL);
    let (node_layouts, width, height) = assign_coordinates(
        &layer_members,
        &node_sizes,
        &node_ids,
        config,
        is_horizontal,
    );

    // If RL or BT, mirror the coordinates.
    let mut node_map: HashMap<String, NodeLayout> = node_layouts;
    match ast.direction {
        Direction::RL => {
            for nl in node_map.values_mut() {
                nl.rect.cx = width - nl.rect.cx;
            }
        }
        Direction::BT => {
            for nl in node_map.values_mut() {
                nl.rect.cy = height - nl.rect.cy;
            }
        }
        _ => {}
    }

    FlowchartLayout {
        nodes: node_map,
        width,
        height,
    }
}

/// Detect back-edges via DFS. Returns set of (from, to) edges that form cycles.
fn detect_back_edges(n: usize, adj: &[Vec<usize>]) -> HashSet<(usize, usize)> {
    let mut back_edges = HashSet::new();
    // 0 = unvisited, 1 = in current DFS path (gray), 2 = finished (black)
    let mut state = vec![0u8; n];

    for start in 0..n {
        if state[start] == 0 {
            dfs_back_edges(start, adj, &mut state, &mut back_edges);
        }
    }
    back_edges
}

fn dfs_back_edges(
    u: usize,
    adj: &[Vec<usize>],
    state: &mut [u8],
    back_edges: &mut HashSet<(usize, usize)>,
) {
    state[u] = 1; // gray — in current path
    for &v in &adj[u] {
        match state[v] {
            0 => dfs_back_edges(v, adj, state, back_edges),
            1 => {
                // v is an ancestor in the current DFS path → back-edge
                back_edges.insert((u, v));
            }
            _ => {} // already finished, cross/forward edge — fine
        }
    }
    state[u] = 2; // black — finished
}

/// Build a new adjacency list with specified edges removed.
fn remove_edges(n: usize, adj: &[Vec<usize>], remove: &HashSet<(usize, usize)>) -> Vec<Vec<usize>> {
    let mut result = vec![vec![]; n];
    for (u, neighbors) in adj.iter().enumerate() {
        for &v in neighbors {
            if !remove.contains(&(u, v)) {
                result[u].push(v);
            }
        }
    }
    result
}

/// Longest-path layer assignment using topological sort on a DAG.
///
/// Back-edges must be removed before calling this function.
fn assign_layers(n: usize, adj: &[Vec<usize>], in_deg: &[usize]) -> Vec<usize> {
    let mut layers = vec![0usize; n];
    let mut deg = in_deg.to_vec();
    let mut queue: VecDeque<usize> = VecDeque::new();

    // Seed with all source nodes (in_deg == 0).
    for i in 0..n {
        if deg[i] == 0 {
            queue.push_back(i);
        }
    }

    let mut visited = vec![false; n];
    while let Some(u) = queue.pop_front() {
        if visited[u] {
            continue;
        }
        visited[u] = true;
        for &v in &adj[u] {
            layers[v] = layers[v].max(layers[u] + 1);
            deg[v] = deg[v].saturating_sub(1);
            if deg[v] == 0 {
                queue.push_back(v);
            }
        }
    }

    // Any unvisited nodes (isolated or in pure cycles that survived edge removal)
    // get assigned based on their connection to visited nodes.
    for i in 0..n {
        if !visited[i] {
            // Place after any visited predecessor, or layer 0.
            let max_pred_layer = adj
                .iter()
                .enumerate()
                .filter(|(_, neighbors)| neighbors.contains(&i))
                .filter(|(u, _)| visited[*u])
                .map(|(u, _)| layers[u])
                .max();
            layers[i] = max_pred_layer.map_or(0, |l| l + 1);
        }
    }

    layers
}

/// Barycenter heuristic for ordering nodes within layers.
fn order_layers(layer_members: &mut [Vec<usize>], adj: &[Vec<usize>], n: usize) {
    // Build reverse adjacency for looking up predecessors.
    let mut rev_adj: Vec<Vec<usize>> = vec![vec![]; n];
    for (u, neighbors) in adj.iter().enumerate() {
        for &v in neighbors {
            rev_adj[v].push(u);
        }
    }

    // Multiple passes for refinement.
    for _ in 0..4 {
        // Forward pass: order each layer based on predecessor positions.
        for l in 1..layer_members.len() {
            let prev_order: HashMap<usize, usize> = layer_members[l - 1]
                .iter()
                .enumerate()
                .map(|(pos, &node)| (node, pos))
                .collect();

            let mut scored: Vec<(f64, usize)> = layer_members[l]
                .iter()
                .map(|&node| {
                    let preds: Vec<f64> = rev_adj[node]
                        .iter()
                        .filter_map(|&p| prev_order.get(&p).map(|&pos| pos as f64))
                        .collect();
                    let bary = if preds.is_empty() {
                        node as f64
                    } else {
                        preds.iter().sum::<f64>() / preds.len() as f64
                    };
                    (bary, node)
                })
                .collect();

            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            layer_members[l] = scored.into_iter().map(|(_, node)| node).collect();
        }

        // Backward pass.
        for l in (0..layer_members.len().saturating_sub(1)).rev() {
            let next_order: HashMap<usize, usize> = layer_members[l + 1]
                .iter()
                .enumerate()
                .map(|(pos, &node)| (node, pos))
                .collect();

            let mut scored: Vec<(f64, usize)> = layer_members[l]
                .iter()
                .map(|&node| {
                    let succs: Vec<f64> = adj[node]
                        .iter()
                        .filter_map(|&s| next_order.get(&s).map(|&pos| pos as f64))
                        .collect();
                    let bary = if succs.is_empty() {
                        node as f64
                    } else {
                        succs.iter().sum::<f64>() / succs.len() as f64
                    };
                    (bary, node)
                })
                .collect();

            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            layer_members[l] = scored.into_iter().map(|(_, node)| node).collect();
        }
    }
}

/// Compute the pixel size of a node based on its label text and shape.
fn compute_node_size(label: &str, shape: NodeShape, config: &LayoutConfig) -> (f32, f32) {
    // Inter font advance ratio is ~0.59 (calibrated in scry-chart).
    const INTER_ADVANCE_RATIO: f32 = 0.59;
    let font_size = 14.0; // matches theme.node_font_size default
    let char_width = font_size * INTER_ADVANCE_RATIO;
    let text_w = label.len() as f32 * char_width;
    let text_h = font_size;

    let w = (text_w + config.node_padding_x * 2.0).max(config.min_node_width);
    let h = (text_h + config.node_padding_y * 2.0).max(config.min_node_height);

    match shape {
        NodeShape::Diamond => {
            // Diamonds need more space due to rotation.
            (w * 1.5, h * 1.5)
        }
        NodeShape::Circle => {
            let diameter = w.max(h);
            (diameter, diameter)
        }
        NodeShape::Cylinder => {
            // Cylinders need extra vertical space for the elliptical caps.
            (w, h + 16.0)
        }
        _ => (w, h),
    }
}

/// Assign concrete pixel coordinates to each node.
fn assign_coordinates(
    layer_members: &[Vec<usize>],
    node_sizes: &[(f32, f32)],
    node_ids: &[&str],
    config: &LayoutConfig,
    horizontal: bool,
) -> (HashMap<String, NodeLayout>, f32, f32) {
    let mut result = HashMap::new();

    // Compute per-layer dimensions.
    let mut layer_widths: Vec<f32> = Vec::new();
    let mut layer_heights: Vec<f32> = Vec::new();

    for members in layer_members {
        if horizontal {
            // In horizontal mode, "width" of a layer is the max node width,
            // and "height" is the sum of node heights + spacing.
            let max_w = members
                .iter()
                .map(|&i| node_sizes[i].0)
                .fold(0.0f32, f32::max);
            let total_h: f32 = members.iter().map(|&i| node_sizes[i].1).sum::<f32>()
                + (members.len().saturating_sub(1)) as f32 * config.node_spacing_y;
            layer_widths.push(max_w);
            layer_heights.push(total_h);
        } else {
            let total_w: f32 = members.iter().map(|&i| node_sizes[i].0).sum::<f32>()
                + (members.len().saturating_sub(1)) as f32 * config.node_spacing_x;
            let max_h = members
                .iter()
                .map(|&i| node_sizes[i].1)
                .fold(0.0f32, f32::max);
            layer_widths.push(total_w);
            layer_heights.push(max_h);
        }
    }

    if horizontal {
        // LR layout: layers go left-to-right, nodes stacked vertically within each layer.
        let total_width = layer_widths.iter().sum::<f32>()
            + (layer_widths.len().saturating_sub(1)) as f32 * config.node_spacing_x
            + config.margin * 2.0;
        let max_layer_height = layer_heights.iter().cloned().fold(0.0f32, f32::max);
        let total_height = max_layer_height + config.margin * 2.0;

        let mut x_offset = config.margin;
        for (l, members) in layer_members.iter().enumerate() {
            let layer_w = layer_widths[l];
            let layer_cx = x_offset + layer_w / 2.0;

            // Center nodes vertically within the canvas.
            let layer_h = layer_heights[l];
            let y_start = config.margin + (max_layer_height - layer_h) / 2.0;
            let mut y_offset = y_start;

            for (order, &node_idx) in members.iter().enumerate() {
                let (nw, nh) = node_sizes[node_idx];
                let cy = y_offset + nh / 2.0;
                result.insert(
                    node_ids[node_idx].to_string(),
                    NodeLayout {
                        rect: PositionedRect {
                            cx: layer_cx,
                            cy,
                            w: nw,
                            h: nh,
                        },
                        layer: l,
                        order,
                    },
                );
                y_offset += nh + config.node_spacing_y;
            }

            x_offset += layer_w + config.node_spacing_x;
        }

        (result, total_width, total_height)
    } else {
        // TB layout: layers go top-to-bottom, nodes spread horizontally.
        let max_layer_width = layer_widths.iter().cloned().fold(0.0f32, f32::max);
        let total_width = max_layer_width + config.margin * 2.0;
        let total_height = layer_heights.iter().sum::<f32>()
            + (layer_heights.len().saturating_sub(1)) as f32 * config.node_spacing_y
            + config.margin * 2.0;

        let mut y_offset = config.margin;
        for (l, members) in layer_members.iter().enumerate() {
            let layer_h = layer_heights[l];
            let layer_cy = y_offset + layer_h / 2.0;

            // Center nodes horizontally.
            let layer_w = layer_widths[l];
            let x_start = config.margin + (max_layer_width - layer_w) / 2.0;
            let mut x_off = x_start;

            for (order, &node_idx) in members.iter().enumerate() {
                let (nw, nh) = node_sizes[node_idx];
                let cx = x_off + nw / 2.0;
                result.insert(
                    node_ids[node_idx].to_string(),
                    NodeLayout {
                        rect: PositionedRect {
                            cx,
                            cy: layer_cy,
                            w: nw,
                            h: nh,
                        },
                        layer: l,
                        order,
                    },
                );
                x_off += nw + config.node_spacing_x;
            }

            y_offset += layer_h + config.node_spacing_y;
        }

        (result, total_width, total_height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::flowchart::parse_flowchart;

    #[test]
    fn layout_simple_chain() {
        let src = "graph TD\n    A --> B --> C";
        let ast = parse_flowchart(src).unwrap();
        let config = LayoutConfig::default();
        let result = layout(&ast, &config);

        // Three nodes in three layers.
        assert_eq!(result.nodes.len(), 3);
        let a = &result.nodes["A"];
        let b = &result.nodes["B"];
        let c = &result.nodes["C"];

        // A should be above B, B above C.
        assert!(a.rect.cy < b.rect.cy);
        assert!(b.rect.cy < c.rect.cy);

        // All in the same column (centered).
        assert!((a.rect.cx - b.rect.cx).abs() < 1.0);
        assert!((b.rect.cx - c.rect.cx).abs() < 1.0);
    }

    #[test]
    fn layout_lr_direction() {
        let src = "flowchart LR\n    A --> B --> C";
        let ast = parse_flowchart(src).unwrap();
        let config = LayoutConfig::default();
        let result = layout(&ast, &config);

        let a = &result.nodes["A"];
        let b = &result.nodes["B"];
        let c = &result.nodes["C"];

        // Horizontal: A to the left of B, B to the left of C.
        assert!(a.rect.cx < b.rect.cx);
        assert!(b.rect.cx < c.rect.cx);

        // All at the same vertical position.
        assert!((a.rect.cy - b.rect.cy).abs() < 1.0);
    }

    #[test]
    fn layout_branching() {
        let src = "graph TD\n    A --> B\n    A --> C";
        let ast = parse_flowchart(src).unwrap();
        let config = LayoutConfig::default();
        let result = layout(&ast, &config);

        let a = &result.nodes["A"];
        let b = &result.nodes["B"];
        let c = &result.nodes["C"];

        // A in layer 0, B and C in layer 1.
        assert_eq!(a.layer, 0);
        assert_eq!(b.layer, 1);
        assert_eq!(c.layer, 1);

        // B and C should be side by side.
        assert!((b.rect.cy - c.rect.cy).abs() < 1.0);
        assert!((b.rect.cx - c.rect.cx).abs() > 10.0);
    }

    #[test]
    fn layout_with_back_edge() {
        // D → B is a back-edge. Should not break layering.
        let src = "graph TD\n    A --> B\n    B --> C\n    C --> D\n    D --> B";
        let ast = parse_flowchart(src).unwrap();
        let config = LayoutConfig::default();
        let result = layout(&ast, &config);

        let a = &result.nodes["A"];
        let b = &result.nodes["B"];
        let c = &result.nodes["C"];
        let d = &result.nodes["D"];

        // A should be topmost, then B, C, D in order.
        assert!(a.rect.cy < b.rect.cy, "A should be above B");
        assert!(b.rect.cy < c.rect.cy, "B should be above C");
        assert!(c.rect.cy < d.rect.cy, "C should be above D");
    }

    #[test]
    fn layout_with_cycle_to_root() {
        // E → A is a back-edge to the root.
        let src = "graph TD\n    A --> B\n    B --> C\n    C --> D\n    D --> E\n    E --> A";
        let ast = parse_flowchart(src).unwrap();
        let config = LayoutConfig::default();
        let result = layout(&ast, &config);

        let a = &result.nodes["A"];
        let e = &result.nodes["E"];

        // A should still be at the top (layer 0).
        assert_eq!(a.layer, 0);
        // E should be below A.
        assert!(a.rect.cy < e.rect.cy, "A should be above E");
    }

    #[test]
    fn layout_self_loop() {
        let src = "graph TD\n    A --> B\n    B --> B\n    B --> C";
        let ast = parse_flowchart(src).unwrap();
        let config = LayoutConfig::default();
        let result = layout(&ast, &config);

        // Self-loop should not affect layering.
        let a = &result.nodes["A"];
        let b = &result.nodes["B"];
        let c = &result.nodes["C"];

        assert!(a.rect.cy < b.rect.cy, "A above B");
        assert!(b.rect.cy < c.rect.cy, "B above C");
    }

    #[test]
    fn layout_complex_ci_pipeline() {
        let src = r#"graph TD
    A[Push] --> B{Tests?}
    B -->|Yes| C[Build]
    B -->|No| D[Notify]
    D --> E[Fix]
    E --> A
    C --> F{Deploy?}
    F -->|Yes| G[Staging]
    G --> H{Smoke?}
    H -->|Yes| I[Prod]
    H -->|No| D
    F -->|No| J[End]"#;
        let ast = parse_flowchart(src).unwrap();
        let config = LayoutConfig::default();
        let result = layout(&ast, &config);

        // Height should be substantially more than width for a TD layout.
        assert!(
            result.height > result.width * 0.5,
            "TD layout should be taller than wide: {}x{}",
            result.width,
            result.height
        );

        // A should be at the top.
        let a = &result.nodes["A"];
        let j = &result.nodes["J"];
        assert!(a.rect.cy < j.rect.cy, "A should be above J");
    }

    #[test]
    fn layout_disconnected_nodes() {
        let src = "graph TD\n    A --> B\n    C --> D";
        let ast = parse_flowchart(src).unwrap();
        let config = LayoutConfig::default();
        let result = layout(&ast, &config);

        // Both chains should have proper layering.
        assert_eq!(result.nodes["A"].layer, 0);
        assert_eq!(result.nodes["B"].layer, 1);
        assert_eq!(result.nodes["C"].layer, 0);
        assert_eq!(result.nodes["D"].layer, 1);
    }

    #[test]
    fn layout_single_node() {
        let src = "graph TD\n    A[Alone]";
        let ast = parse_flowchart(src).unwrap();
        let config = LayoutConfig::default();
        let result = layout(&ast, &config);

        assert_eq!(result.nodes.len(), 1);
        assert!(result.width > 0.0);
        assert!(result.height > 0.0);
    }
}
