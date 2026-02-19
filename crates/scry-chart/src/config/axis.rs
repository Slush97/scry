// SPDX-License-Identifier: MIT OR Apache-2.0
//! Axis range and inversion configuration.

/// Manual axis domain overrides and axis direction.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct AxisRangeConfig {
    /// Manual x-axis domain override (min, max).
    pub x_range: Option<(f64, f64)>,
    /// Manual y-axis domain override (min, max).
    pub y_range: Option<(f64, f64)>,
    /// Whether to invert (reverse) the X axis.
    pub x_inverted: bool,
    /// Whether to invert (reverse) the Y axis.
    pub y_inverted: bool,
}
