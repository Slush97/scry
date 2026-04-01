// SPDX-License-Identifier: MIT OR Apache-2.0
//! Flowchart parser.
//!
//! Parses `graph TD` / `flowchart LR` style diagrams into a structured AST.
//!
//! Supported syntax:
//! - Directions: `TD`, `TB`, `LR`, `RL`, `BT`
//! - Node shapes: `A[text]` (rect), `A(text)` (rounded), `A{text}` (diamond),
//!   `A([text])` (stadium), `A[[text]]` (subroutine), `A((text))` (circle),
//!   `A[(text)]` (cylinder/database)
//! - Edge types: `-->`, `---`, `-.->`, `==>`, `--text-->`, `-->|text|`

use crate::error::MermaidError;

/// Parsed flowchart AST.
#[derive(Clone, Debug)]
pub struct FlowchartAst {
    /// Graph direction.
    pub direction: Direction,
    /// All nodes (definition order).
    pub nodes: Vec<Node>,
    /// All edges.
    pub edges: Vec<Edge>,
    /// Subgraph groupings.
    pub subgraphs: Vec<Subgraph>,
}

/// A named group of nodes rendered with a bounding box.
#[derive(Clone, Debug)]
pub struct Subgraph {
    /// Display label.
    pub label: String,
    /// Node IDs that belong to this subgraph.
    pub node_ids: Vec<String>,
}

/// Graph direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Top to bottom.
    TB,
    /// Left to right.
    LR,
    /// Right to left.
    RL,
    /// Bottom to top.
    BT,
}

/// A node in the flowchart.
#[derive(Clone, Debug)]
pub struct Node {
    /// Unique identifier (e.g., `A`, `start`).
    pub id: String,
    /// Display text (falls back to `id` if not set).
    pub label: String,
    /// Visual shape.
    pub shape: NodeShape,
}

/// Node shape variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeShape {
    /// `[text]` — standard rectangle.
    Rectangle,
    /// `(text)` — rounded rectangle.
    Rounded,
    /// `{text}` — diamond / rhombus.
    Diamond,
    /// `([text])` — stadium / pill.
    Stadium,
    /// `[[text]]` — subroutine (double-border).
    Subroutine,
    /// `((text))` — circle.
    Circle,
    /// `[(text)]` — cylinder (database).
    Cylinder,
}

/// An edge between two nodes.
#[derive(Clone, Debug)]
pub struct Edge {
    /// Source node id.
    pub from: String,
    /// Target node id.
    pub to: String,
    /// Optional label on the edge.
    pub label: Option<String>,
    /// Edge style.
    pub style: EdgeStyle,
}

/// Visual style of an edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeStyle {
    /// `-->` solid line with arrow.
    SolidArrow,
    /// `---` solid line, no arrow.
    SolidLine,
    /// `-.->` dotted line with arrow.
    DottedArrow,
    /// `==>` thick line with arrow.
    ThickArrow,
}

/// Parse a flowchart source string.
pub fn parse_flowchart(source: &str) -> Result<FlowchartAst, MermaidError> {
    let mut lines = source.lines().enumerate().peekable();
    let mut nodes: Vec<Node> = Vec::new();
    let mut edges: Vec<Edge> = Vec::new();
    let mut node_ids = std::collections::HashSet::new();
    let mut subgraphs: Vec<Subgraph> = Vec::new();

    // Stack of open subgraphs (label, collected node IDs).
    let mut subgraph_stack: Vec<(String, Vec<String>)> = Vec::new();

    // Parse direction from first line.
    let (first_lineno, first_line) = lines
        .find(|(_, l)| !l.trim().is_empty())
        .ok_or_else(|| MermaidError::Parse {
            line: 1,
            message: "empty input".into(),
        })?;

    let direction = parse_direction(first_line.trim()).ok_or_else(|| MermaidError::Parse {
        line: first_lineno + 1,
        message: format!("expected 'graph/flowchart TD|TB|LR|RL|BT', got: {first_line}"),
    })?;

    // Parse remaining lines for nodes and edges.
    for (lineno, raw_line) in lines {
        let line = strip_comment(raw_line.trim());
        if line.is_empty() {
            continue;
        }

        // Handle subgraph open/close.
        if line.starts_with("subgraph ") {
            let label = line["subgraph ".len()..].trim().to_string();
            subgraph_stack.push((label, Vec::new()));
            continue;
        }
        if line == "end" {
            if let Some((label, member_ids)) = subgraph_stack.pop() {
                subgraphs.push(Subgraph {
                    label,
                    node_ids: member_ids,
                });
            }
            continue;
        }

        // Skip style/class/click directives.
        if line.starts_with("style ")
            || line.starts_with("class ")
            || line.starts_with("classDef ")
            || line.starts_with("click ")
        {
            continue;
        }

        // Record node count before parsing this statement.
        let node_count_before = nodes.len();

        // Try to parse as edge line (may contain inline node defs).
        parse_statement(line, lineno + 1, &mut nodes, &mut edges, &mut node_ids)?;

        // Any new nodes belong to the innermost open subgraph.
        if let Some((_label, ref mut member_ids)) = subgraph_stack.last_mut() {
            for node in &nodes[node_count_before..] {
                if !member_ids.contains(&node.id) {
                    member_ids.push(node.id.clone());
                }
            }
        }
    }

    // Close any unclosed subgraphs gracefully.
    while let Some((label, member_ids)) = subgraph_stack.pop() {
        subgraphs.push(Subgraph {
            label,
            node_ids: member_ids,
        });
    }

    Ok(FlowchartAst {
        direction,
        nodes,
        edges,
        subgraphs,
    })
}

fn parse_direction(line: &str) -> Option<Direction> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 2 {
        return None;
    }
    match parts[0] {
        "graph" | "flowchart" => {}
        _ => return None,
    }
    match parts[1] {
        "TD" | "TB" => Some(Direction::TB),
        "LR" => Some(Direction::LR),
        "RL" => Some(Direction::RL),
        "BT" => Some(Direction::BT),
        _ => None,
    }
}

fn strip_comment(line: &str) -> &str {
    // Mermaid uses %% for comments.
    if let Some(pos) = line.find("%%") {
        line[..pos].trim_end()
    } else {
        line
    }
}

/// Parse a single statement line, which may define nodes and/or edges.
///
/// Examples:
///   `A[Start]`              — node definition only
///   `A --> B`               — edge (nodes auto-created if unseen)
///   `A[Start] --> B[End]`   — inline node defs + edge
///   `A --> B --> C`          — chained edges
fn parse_statement(
    line: &str,
    lineno: usize,
    nodes: &mut Vec<Node>,
    edges: &mut Vec<Edge>,
    node_ids: &mut std::collections::HashSet<String>,
) -> Result<(), MermaidError> {
    // Tokenize: split into node-refs and edge-arrows, preserving order.
    let tokens = tokenize(line, lineno)?;

    if tokens.is_empty() {
        return Ok(());
    }

    // Walk tokens: NodeRef, Edge, NodeRef, Edge, NodeRef ...
    let mut i = 0;
    while i < tokens.len() {
        match &tokens[i] {
            Token::NodeRef(nr) => {
                ensure_node(nr, nodes, node_ids);
                // Check if next token is an edge.
                if i + 2 < tokens.len() {
                    if let Token::Edge(es, label) = &tokens[i + 1] {
                        if let Token::NodeRef(nr2) = &tokens[i + 2] {
                            ensure_node(nr2, nodes, node_ids);
                            edges.push(Edge {
                                from: nr.id.clone(),
                                to: nr2.id.clone(),
                                label: label.clone(),
                                style: *es,
                            });
                            i += 1; // advance past edge; loop +1 lands on target for chaining
                        } else {
                            return Err(MermaidError::Parse {
                                line: lineno,
                                message: "expected node after edge arrow".into(),
                            });
                        }
                    } else {
                        // Standalone node def, fine.
                    }
                }
            }
            Token::Edge(_, _) => {
                return Err(MermaidError::Parse {
                    line: lineno,
                    message: "unexpected edge arrow without preceding node".into(),
                });
            }
        }
        i += 1;
    }

    Ok(())
}

fn ensure_node(
    nr: &NodeRefData,
    nodes: &mut Vec<Node>,
    node_ids: &mut std::collections::HashSet<String>,
) {
    if node_ids.contains(&nr.id) {
        // Update label/shape if this definition provides them.
        if nr.has_shape {
            if let Some(n) = nodes.iter_mut().find(|n| n.id == nr.id) {
                n.label = nr.label.clone();
                n.shape = nr.shape;
            }
        }
        return;
    }
    node_ids.insert(nr.id.clone());
    nodes.push(Node {
        id: nr.id.clone(),
        label: nr.label.clone(),
        shape: nr.shape,
    });
}

// ---- Tokenizer ----

#[derive(Debug)]
enum Token {
    NodeRef(NodeRefData),
    Edge(EdgeStyle, Option<String>),
}

#[derive(Debug)]
struct NodeRefData {
    id: String,
    label: String,
    shape: NodeShape,
    has_shape: bool,
}

fn tokenize(line: &str, lineno: usize) -> Result<Vec<Token>, MermaidError> {
    let mut tokens = Vec::new();
    let bytes = line.as_bytes();
    let mut pos = 0;

    while pos < bytes.len() {
        skip_spaces(bytes, &mut pos);
        if pos >= bytes.len() {
            break;
        }

        // Try edge arrow.
        if let Some((style, label, end)) = try_edge(line, pos) {
            tokens.push(Token::Edge(style, label));
            pos = end;
            continue;
        }

        // Try node ref.
        if let Some((nr, end)) = try_node_ref(line, pos, lineno)? {
            tokens.push(Token::NodeRef(nr));
            pos = end;
            continue;
        }

        // If we're stuck, skip a character to avoid infinite loop.
        // This handles semicolons and other separators.
        if bytes[pos] == b';' {
            pos += 1;
            continue;
        }

        return Err(MermaidError::Parse {
            line: lineno,
            message: format!("unexpected character '{}' at position {pos}", bytes[pos] as char),
        });
    }

    Ok(tokens)
}

fn skip_spaces(bytes: &[u8], pos: &mut usize) {
    while *pos < bytes.len() && bytes[*pos] == b' ' {
        *pos += 1;
    }
}

/// Try to match an edge arrow starting at `pos`.
/// Returns `(style, optional_label, end_position)`.
fn try_edge(line: &str, pos: usize) -> Option<(EdgeStyle, Option<String>, usize)> {
    let rest = &line[pos..];

    // Thick arrow with pipe label: ==>|text|
    if rest.starts_with("==>|") {
        if let Some(end) = rest[4..].find('|') {
            let label = rest[4..4 + end].to_string();
            let label = if label.is_empty() { None } else { Some(label) };
            return Some((EdgeStyle::ThickArrow, label, pos + 5 + end));
        }
    }

    // Solid arrow with pipe label: -->|text|
    if rest.starts_with("-->|") {
        if let Some(end) = rest[4..].find('|') {
            let label = rest[4..4 + end].to_string();
            let label = if label.is_empty() { None } else { Some(label) };
            return Some((EdgeStyle::SolidArrow, label, pos + 5 + end));
        }
    }

    // Solid arrow with inline label: --text-->
    if rest.starts_with("--") && !rest.starts_with("-->") && !rest.starts_with("---") {
        if let Some(arrow_pos) = rest[2..].find("-->") {
            let label = rest[2..2 + arrow_pos].trim().to_string();
            let label = if label.is_empty() { None } else { Some(label) };
            return Some((EdgeStyle::SolidArrow, label, pos + 2 + arrow_pos + 3));
        }
    }

    // Dotted with pipe label: -.->|text| (must check before plain -.->)
    if rest.starts_with("-.->|") {
        if let Some(end) = rest[5..].find('|') {
            let label = rest[5..5 + end].to_string();
            let label = if label.is_empty() { None } else { Some(label) };
            return Some((EdgeStyle::DottedArrow, label, pos + 6 + end));
        }
    }
    // Dotted arrow: -.->
    if rest.starts_with("-.->") {
        return Some((EdgeStyle::DottedArrow, None, pos + 4));
    }

    // Simple arrows (order matters — longest match first).
    if rest.starts_with("==>") {
        return Some((EdgeStyle::ThickArrow, None, pos + 3));
    }
    if rest.starts_with("-->") {
        return Some((EdgeStyle::SolidArrow, None, pos + 3));
    }
    if rest.starts_with("---") {
        return Some((EdgeStyle::SolidLine, None, pos + 3));
    }

    None
}

/// Try to match a node reference (with optional shape) starting at `pos`.
fn try_node_ref(
    line: &str,
    pos: usize,
    lineno: usize,
) -> Result<Option<(NodeRefData, usize)>, MermaidError> {
    let bytes = line.as_bytes();

    // Node ID: alphanumeric + underscore.
    let id_start = pos;
    let mut p = pos;
    while p < bytes.len() && (bytes[p].is_ascii_alphanumeric() || bytes[p] == b'_') {
        p += 1;
    }
    if p == id_start {
        return Ok(None);
    }
    let id = line[id_start..p].to_string();

    // Check for shape delimiter.
    if p >= bytes.len() {
        return Ok(Some((
            NodeRefData {
                label: id.clone(),
                id,
                shape: NodeShape::Rectangle,
                has_shape: false,
            },
            p,
        )));
    }

    let (shape, open, close) = match bytes[p] {
        b'[' => {
            if p + 1 < bytes.len() && bytes[p + 1] == b'(' {
                // [(text)] — cylinder / database.
                (NodeShape::Cylinder, "[(", ")]")
            } else if p + 1 < bytes.len() && bytes[p + 1] == b'[' {
                // [[text]] — subroutine.
                (NodeShape::Subroutine, "[[", "]]")
            } else {
                (NodeShape::Rectangle, "[", "]")
            }
        }
        b'(' => {
            if p + 1 < bytes.len() && bytes[p + 1] == b'[' {
                (NodeShape::Stadium, "([", "])")
            } else if p + 1 < bytes.len() && bytes[p + 1] == b'(' {
                (NodeShape::Circle, "((", "))")
            } else {
                (NodeShape::Rounded, "(", ")")
            }
        }
        b'{' => (NodeShape::Diamond, "{", "}"),
        _ => {
            return Ok(Some((
                NodeRefData {
                    label: id.clone(),
                    id,
                    shape: NodeShape::Rectangle,
                    has_shape: false,
                },
                p,
            )));
        }
    };

    let content_start = p + open.len();
    let remaining = &line[content_start..];
    let close_pos = remaining.find(close).ok_or_else(|| MermaidError::Parse {
        line: lineno,
        message: format!("unclosed '{open}' for node '{id}'"),
    })?;
    let label = remaining[..close_pos].trim().to_string();
    let end = content_start + close_pos + close.len();

    Ok(Some((
        NodeRefData {
            label: if label.is_empty() { id.clone() } else { label },
            id,
            shape,
            has_shape: true,
        },
        end,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_flowchart() {
        let src = r"graph TD
            A[Start] --> B[Process]
            B --> C{Decision}
            C -->|Yes| D[End]
            C -->|No| B";
        let ast = parse_flowchart(src).unwrap();
        assert_eq!(ast.direction, Direction::TB);
        assert_eq!(ast.nodes.len(), 4);
        assert_eq!(ast.edges.len(), 4);
        assert_eq!(ast.nodes[0].label, "Start");
        assert_eq!(ast.nodes[2].shape, NodeShape::Diamond);
        assert_eq!(ast.edges[2].label.as_deref(), Some("Yes"));
    }

    #[test]
    fn parse_lr_direction() {
        let src = "flowchart LR\n    A --> B";
        let ast = parse_flowchart(src).unwrap();
        assert_eq!(ast.direction, Direction::LR);
    }

    #[test]
    fn parse_shapes() {
        let src = r"graph TD
            A[rect]
            B(rounded)
            C{diamond}
            D([stadium])
            E[[subroutine]]
            F((circle))
            G[(cylinder)]";
        let ast = parse_flowchart(src).unwrap();
        assert_eq!(ast.nodes[0].shape, NodeShape::Rectangle);
        assert_eq!(ast.nodes[1].shape, NodeShape::Rounded);
        assert_eq!(ast.nodes[2].shape, NodeShape::Diamond);
        assert_eq!(ast.nodes[3].shape, NodeShape::Stadium);
        assert_eq!(ast.nodes[4].shape, NodeShape::Subroutine);
        assert_eq!(ast.nodes[5].shape, NodeShape::Circle);
        assert_eq!(ast.nodes[6].shape, NodeShape::Cylinder);
        assert_eq!(ast.nodes[6].label, "cylinder");
    }

    #[test]
    fn parse_edge_styles() {
        let src = r"graph TD
            A --> B
            B --- C
            C -.-> D
            D ==> E";
        let ast = parse_flowchart(src).unwrap();
        assert_eq!(ast.edges[0].style, EdgeStyle::SolidArrow);
        assert_eq!(ast.edges[1].style, EdgeStyle::SolidLine);
        assert_eq!(ast.edges[2].style, EdgeStyle::DottedArrow);
        assert_eq!(ast.edges[3].style, EdgeStyle::ThickArrow);
    }

    #[test]
    fn parse_chained_edges() {
        let src = "graph TD\n    A --> B --> C";
        let ast = parse_flowchart(src).unwrap();
        assert_eq!(ast.edges.len(), 2);
        assert_eq!(ast.edges[0].from, "A");
        assert_eq!(ast.edges[0].to, "B");
        assert_eq!(ast.edges[1].from, "B");
        assert_eq!(ast.edges[1].to, "C");
    }

    #[test]
    fn parse_inline_label() {
        let src = "graph TD\n    A --yes--> B";
        let ast = parse_flowchart(src).unwrap();
        assert_eq!(ast.edges[0].label.as_deref(), Some("yes"));
    }

    #[test]
    fn comments_ignored() {
        let src = "graph TD\n    A --> B %% this is a comment";
        let ast = parse_flowchart(src).unwrap();
        assert_eq!(ast.edges.len(), 1);
    }

    #[test]
    fn dotted_arrow_with_label() {
        let src = "graph TD\n    A -.->|maybe| B";
        let ast = parse_flowchart(src).unwrap();
        assert_eq!(ast.edges[0].style, EdgeStyle::DottedArrow);
        assert_eq!(ast.edges[0].label.as_deref(), Some("maybe"));
    }

    #[test]
    fn thick_arrow_with_label() {
        let src = "graph TD\n    A ==>|fast| B";
        let ast = parse_flowchart(src).unwrap();
        assert_eq!(ast.edges[0].style, EdgeStyle::ThickArrow);
        assert_eq!(ast.edges[0].label.as_deref(), Some("fast"));
    }

    #[test]
    fn node_id_with_underscores() {
        let src = "graph TD\n    my_node[Hello] --> other_node[World]";
        let ast = parse_flowchart(src).unwrap();
        assert_eq!(ast.nodes[0].id, "my_node");
        assert_eq!(ast.nodes[1].id, "other_node");
    }

    #[test]
    fn semicolon_separated() {
        let src = "graph TD\n    A --> B; B --> C";
        let ast = parse_flowchart(src).unwrap();
        assert_eq!(ast.edges.len(), 2);
    }

    #[test]
    fn standalone_node_def() {
        let src = "graph TD\n    A[Start]\n    B[End]\n    A --> B";
        let ast = parse_flowchart(src).unwrap();
        assert_eq!(ast.nodes.len(), 2);
        assert_eq!(ast.nodes[0].label, "Start");
        assert_eq!(ast.edges.len(), 1);
    }

    #[test]
    fn node_redefinition_updates_label() {
        let src = "graph TD\n    A --> B\n    A[Renamed]";
        let ast = parse_flowchart(src).unwrap();
        assert_eq!(ast.nodes.iter().find(|n| n.id == "A").unwrap().label, "Renamed");
    }

    #[test]
    fn empty_lines_and_whitespace() {
        let src = "graph TD\n\n    A --> B\n\n    B --> C\n\n";
        let ast = parse_flowchart(src).unwrap();
        assert_eq!(ast.edges.len(), 2);
    }

    #[test]
    fn back_edge_doesnt_crash() {
        // Cycle: A → B → C → A
        let src = "graph TD\n    A --> B\n    B --> C\n    C --> A";
        let ast = parse_flowchart(src).unwrap();
        assert_eq!(ast.edges.len(), 3);
    }

    #[test]
    fn self_loop_doesnt_crash() {
        let src = "graph TD\n    A --> A";
        let ast = parse_flowchart(src).unwrap();
        assert_eq!(ast.edges.len(), 1);
        assert_eq!(ast.edges[0].from, "A");
        assert_eq!(ast.edges[0].to, "A");
    }

    #[test]
    fn parse_subgraphs() {
        let src = r#"graph TD
    subgraph Backend
        A[API] --> B[(DB)]
    end
    subgraph Frontend
        C[React] --> D[Redux]
    end
    B --> C"#;
        let ast = parse_flowchart(src).unwrap();
        assert_eq!(ast.subgraphs.len(), 2);
        assert_eq!(ast.subgraphs[0].label, "Backend");
        assert_eq!(ast.subgraphs[0].node_ids, vec!["A", "B"]);
        assert_eq!(ast.subgraphs[1].label, "Frontend");
        assert_eq!(ast.subgraphs[1].node_ids, vec!["C", "D"]);
    }

    #[test]
    fn unclosed_subgraph() {
        let src = "graph TD\n    subgraph Oops\n        A --> B";
        let ast = parse_flowchart(src).unwrap();
        assert_eq!(ast.subgraphs.len(), 1);
        assert_eq!(ast.subgraphs[0].label, "Oops");
    }

    #[test]
    fn skip_directives() {
        let src = "graph TD\n    classDef red fill:#f00\n    class A red\n    style A fill:#0f0\n    click A href\n    A --> B";
        let ast = parse_flowchart(src).unwrap();
        assert_eq!(ast.edges.len(), 1);
    }

    #[test]
    fn bt_direction() {
        let src = "graph BT\n    A --> B";
        let ast = parse_flowchart(src).unwrap();
        assert_eq!(ast.direction, Direction::BT);
    }

    #[test]
    fn rl_direction() {
        let src = "flowchart RL\n    A --> B";
        let ast = parse_flowchart(src).unwrap();
        assert_eq!(ast.direction, Direction::RL);
    }
}
