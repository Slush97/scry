// SPDX-License-Identifier: MIT OR Apache-2.0
//! Diagram rendering to scry-engine `PixelCanvas`.

pub mod flowchart;

use scry_engine::scene::PixelCanvas;

/// Result of rendering a diagram.
#[derive(Clone, Debug)]
pub struct RenderedDiagram {
    /// The rendered scene.
    pub canvas: PixelCanvas,
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
}
