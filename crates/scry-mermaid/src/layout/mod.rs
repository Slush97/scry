// SPDX-License-Identifier: MIT OR Apache-2.0
//! Graph layout algorithms for Mermaid diagrams.

pub mod flowchart;

/// A positioned rectangle in pixel space.
#[derive(Clone, Debug)]
pub struct PositionedRect {
    /// Center X coordinate.
    pub cx: f32,
    /// Center Y coordinate.
    pub cy: f32,
    /// Width.
    pub w: f32,
    /// Height.
    pub h: f32,
}

impl PositionedRect {
    /// Left edge.
    pub fn left(&self) -> f32 {
        self.cx - self.w / 2.0
    }
    /// Right edge.
    pub fn right(&self) -> f32 {
        self.cx + self.w / 2.0
    }
    /// Top edge.
    pub fn top(&self) -> f32 {
        self.cy - self.h / 2.0
    }
    /// Bottom edge.
    pub fn bottom(&self) -> f32 {
        self.cy + self.h / 2.0
    }
}
