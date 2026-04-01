// SPDX-License-Identifier: MIT OR Apache-2.0
//! # scry-mermaid
//!
//! Mermaid diagram rendering for terminals, built on
//! [`scry-engine`](https://docs.rs/scry-engine).
//!
//! Parses Mermaid syntax, computes graph layout, and renders pixel-perfect
//! diagrams via scry-engine's `PixelCanvas`. Output can be displayed in any
//! Kitty/Sixel/iTerm2 terminal, exported as PNG, or embedded in a Ratatui TUI.
//!
//! ## Quick start
//!
//! ```
//! use scry_mermaid::Mermaid;
//!
//! let diagram = Mermaid::parse("graph TD\n    A[Start] --> B[End]").unwrap();
//! let rendered = diagram.render(800, 600);
//! // rendered.canvas is a scry_engine::PixelCanvas ready for rasterization
//! ```
//!
//! ## Supported diagram types
//!
//! - **Flowchart** (`graph` / `flowchart`) — nodes, edges, shapes, labels
//!
//! More diagram types (sequence, class, state) planned.

#![warn(missing_docs)]
#![deny(unsafe_code)]

pub mod error;
pub mod layout;
pub mod parser;
pub mod render;
pub mod theme;
#[cfg(feature = "widget")]
pub mod widget;

use error::MermaidError;
use parser::Diagram;
use render::RenderedDiagram;
use theme::{LayoutConfig, MermaidTheme};

/// Main entry point for parsing and rendering Mermaid diagrams.
#[derive(Debug)]
pub struct Mermaid {
    diagram: Diagram,
    theme: MermaidTheme,
    config: LayoutConfig,
}

impl Mermaid {
    /// Parse a Mermaid source string.
    pub fn parse(source: &str) -> Result<Self, MermaidError> {
        let diagram = parser::parse(source)?;
        Ok(Self {
            diagram,
            theme: MermaidTheme::default(),
            config: LayoutConfig::default(),
        })
    }

    /// Set the visual theme.
    #[must_use]
    pub fn theme(mut self, theme: MermaidTheme) -> Self {
        self.theme = theme;
        self
    }

    /// Set the layout configuration.
    #[must_use]
    pub fn layout_config(mut self, config: LayoutConfig) -> Self {
        self.config = config;
        self
    }

    /// Render the diagram at the given pixel dimensions.
    ///
    /// The diagram is laid out to fit naturally, then uniformly scaled down
    /// if it exceeds `max_width` or `max_height`. The actual rendered size
    /// may be smaller than the bounds.
    #[must_use]
    pub fn render(&self, max_width: u32, max_height: u32) -> RenderedDiagram {
        match &self.diagram {
            Diagram::Flowchart(ast) => {
                render::flowchart::render(ast, &self.theme, &self.config, max_width, max_height)
            }
        }
    }

    /// Render and rasterize to a PNG byte buffer.
    pub fn render_to_png(&self, max_width: u32, max_height: u32) -> Result<Vec<u8>, MermaidError> {
        let rendered = self.render(max_width, max_height);
        let pixmap = scry_engine::rasterize::Rasterizer::rasterize(&rendered.canvas)
            .map_err(|e| MermaidError::Render(e.to_string()))?;
        pixmap
            .encode_png()
            .map_err(|e| MermaidError::Render(e.to_string()))
    }

    /// Access the parsed diagram AST.
    pub fn diagram(&self) -> &Diagram {
        &self.diagram
    }

    /// Create a clone of this diagram with a different theme.
    #[must_use]
    pub fn clone_with_theme(&self, theme: MermaidTheme) -> Self {
        Self {
            diagram: self.diagram.clone(),
            theme,
            config: self.config.clone(),
        }
    }
}

/// Convenience re-exports.
pub mod prelude {
    pub use crate::error::MermaidError;
    pub use crate::parser::flowchart::{Direction, EdgeStyle, NodeShape};
    pub use crate::render::RenderedDiagram;
    pub use crate::theme::{LayoutConfig, MermaidTheme};
    pub use crate::Mermaid;
    #[cfg(feature = "widget")]
    pub use crate::widget::{MermaidState, MermaidWidget};
}
