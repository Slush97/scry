// SPDX-License-Identifier: MIT OR Apache-2.0
//! Reference lines, annotations, and trend-line configuration.

use crate::annotation::Annotation;
use crate::chart::ReferenceLine;

/// Overlay elements drawn on top of the plot area.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct OverlayConfig {
    /// Horizontal reference lines.
    pub h_lines: Vec<ReferenceLine>,
    /// Vertical reference lines.
    pub v_lines: Vec<ReferenceLine>,
    /// Data-coordinate annotations.
    pub annotations: Vec<Annotation>,
    /// Whether to show a trend line (linear regression).
    pub show_trend: bool,
}
