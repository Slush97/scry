// SPDX-License-Identifier: MIT OR Apache-2.0
//! Axis range, inversion, and aspect ratio configuration.

/// Aspect ratio constraint for the plot area.
///
/// Controls whether the X and Y axes use the same data-to-pixel ratio,
/// following Cleveland (1985) and Wilkinson (2005) recommendations
/// for equal-unit axes.
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AspectRatio {
    /// Fill available space (default). Axes scale independently.
    #[default]
    Auto,
    /// 1:1 data-unit to pixel ratio. Both axes use the same scale factor.
    Equal,
    /// Custom ratio: `x_units_per_pixel / y_units_per_pixel`.
    Fixed(f64),
}

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
    /// Aspect ratio constraint for the plot area.
    pub aspect_ratio: AspectRatio,
}
