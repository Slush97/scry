//! Configurable chart margins / padding.
//!
//! Use [`Margin`] to control the extra whitespace around the plot area,
//! beyond the default proportional margins computed by the layout engine.

/// Extra whitespace (in pixels) added around the plot area.
///
/// These values are additive — they are applied *on top of* the automatic
/// proportional margins. Set all to `0.0` for no extra space.
///
/// # Example
///
/// ```
/// use scry_chart::margin::Margin;
///
/// // 20px on all sides
/// let m = Margin::uniform(20.0);
///
/// // Custom per-side
/// let m = Margin::new(10.0, 20.0, 10.0, 30.0);
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct Margin {
    /// Extra space above the chart (pixels).
    pub top: f32,
    /// Extra space to the right (pixels).
    pub right: f32,
    /// Extra space below the chart (pixels).
    pub bottom: f32,
    /// Extra space to the left (pixels).
    pub left: f32,
}

impl Margin {
    /// Create a margin with specific values per side.
    #[must_use]
    pub fn new(top: f32, right: f32, bottom: f32, left: f32) -> Self {
        Self {
            top: top.max(0.0),
            right: right.max(0.0),
            bottom: bottom.max(0.0),
            left: left.max(0.0),
        }
    }

    /// Create a margin with the same value on all sides.
    #[must_use]
    pub fn uniform(px: f32) -> Self {
        Self::new(px, px, px, px)
    }
}

impl Default for Margin {
    fn default() -> Self {
        Self {
            top: 0.0,
            right: 0.0,
            bottom: 0.0,
            left: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn margin_uniform() {
        let m = Margin::uniform(10.0);
        assert_eq!(m.top, 10.0);
        assert_eq!(m.right, 10.0);
        assert_eq!(m.bottom, 10.0);
        assert_eq!(m.left, 10.0);
    }

    #[test]
    fn margin_default_is_zero() {
        let m = Margin::default();
        assert_eq!(m.top, 0.0);
        assert_eq!(m.right, 0.0);
        assert_eq!(m.bottom, 0.0);
        assert_eq!(m.left, 0.0);
    }

    #[test]
    fn margin_clamps_negative() {
        let m = Margin::new(-5.0, -10.0, -1.0, -0.1);
        assert_eq!(m.top, 0.0);
        assert_eq!(m.right, 0.0);
        assert_eq!(m.bottom, 0.0);
        assert_eq!(m.left, 0.0);
    }

    #[test]
    fn margin_per_side() {
        let m = Margin::new(5.0, 10.0, 15.0, 20.0);
        assert_eq!(m.top, 5.0);
        assert_eq!(m.right, 10.0);
        assert_eq!(m.bottom, 15.0);
        assert_eq!(m.left, 20.0);
    }
}
