// SPDX-License-Identifier: MIT OR Apache-2.0
//! Mermaid diagram parser.
//!
//! Parses a subset of the Mermaid language into an AST. Currently supports
//! flowcharts (`graph` / `flowchart`).

pub mod flowchart;

use crate::error::MermaidError;

/// Top-level parsed diagram.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Diagram {
    /// A flowchart / graph diagram.
    Flowchart(flowchart::FlowchartAst),
}

/// Parse a Mermaid source string into a [`Diagram`].
pub fn parse(source: &str) -> Result<Diagram, MermaidError> {
    let trimmed = source.trim();

    // Detect diagram type from the first non-empty line.
    let first_line = trimmed
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim();

    if first_line.starts_with("graph ") || first_line.starts_with("flowchart ") {
        let ast = flowchart::parse_flowchart(trimmed)?;
        Ok(Diagram::Flowchart(ast))
    } else {
        let kind = first_line.split_whitespace().next().unwrap_or("(empty)");
        Err(MermaidError::UnsupportedDiagram(kind.to_string()))
    }
}
