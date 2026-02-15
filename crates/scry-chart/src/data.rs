//! Data types for chart input.
//!
//! Provides [`Series`] — a named sequence of values that can be fed into charts,
//! and [`SeriesStyle`] — optional per-series visual overrides.

use scry_engine::style::{Color, DashPattern};

// ---------------------------------------------------------------------------
// Gradient & pattern types
// ---------------------------------------------------------------------------

/// Gradient direction for area fills under line charts.
///
/// When set on a [`SeriesStyle`], overrides the default top→bottom gradient
/// that is applied when `LineChart::filled()` is enabled.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum GradientFill {
    /// Gradient from top of the curve down to the baseline (default behavior).
    TopToBottom,
    /// Gradient from the baseline up to the curve.
    BottomToTop,
    /// Custom gradient stops: `(position, color)` where position is 0.0–1.0.
    ///
    /// Position 0.0 = top of curve, 1.0 = baseline.
    Custom(Vec<(f32, Color)>),
}

/// Fill pattern for bar chart accessibility.
///
/// Overlays geometric hatch marks on top of bar fills, making series
/// distinguishable without relying on color alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FillPattern {
    /// No pattern overlay (solid fill, default).
    Solid,
    /// Horizontal lines at regular spacing.
    Hatched,
    /// Horizontal + vertical crosshatch lines.
    CrossHatched,
    /// Grid of small dots.
    Dotted,
    /// 45° diagonal lines.
    Diagonal,
}

// ---------------------------------------------------------------------------
// GapPolicy — how line charts handle NaN/missing data
// ---------------------------------------------------------------------------

/// Policy for handling NaN (missing) values in line chart data series.
///
/// When a [`Series`] contains `NaN` values, `GapPolicy` controls whether the
/// line renderer breaks the line, interpolates across the gap, or substitutes
/// zero.
///
/// # Examples
///
/// ```
/// use scry_chart::data::GapPolicy;
///
/// let policy = GapPolicy::default();
/// assert_eq!(policy, GapPolicy::Skip);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GapPolicy {
    /// Break the line at NaN values, leaving a visual gap (default).
    #[default]
    Skip,
    /// Linearly interpolate across NaN gaps using the nearest finite neighbors.
    Interpolate,
    /// Treat NaN values as 0.0.
    Zero,
}

// ---------------------------------------------------------------------------
// SeriesStyle — per-series visual overrides
// ---------------------------------------------------------------------------

/// Per-series visual overrides.
///
/// Every field is optional — `None` means "inherit from the theme".
/// This allows surgical customization of individual series without
/// replacing the entire theme palette.
///
/// # Examples
///
/// ```
/// use scry_chart::data::{Series, SeriesStyle};
/// use scry_engine::style::Color;
///
/// let s = Series::new("Revenue", vec![10.0, 20.0, 30.0])
///     .style(SeriesStyle::new().color(Color::from_rgba8(255, 0, 0, 255)));
/// assert!(s.series_style().color.is_some());
/// ```
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct SeriesStyle {
    /// Override the series color (replaces the theme palette entry).
    pub color: Option<Color>,
    /// Override the line width for this series (line/radar charts).
    pub line_width: Option<f32>,
    /// Override the dash pattern for this series.
    pub dash: Option<DashPattern>,
    /// Override the area fill opacity for this series (0.0–1.0).
    pub fill_opacity: Option<f32>,
    /// Override the gradient direction for filled line/area charts.
    pub fill_gradient: Option<GradientFill>,
    /// Override the bar fill pattern (accessibility overlay).
    pub fill_pattern: Option<FillPattern>,
}

impl SeriesStyle {
    /// Create an empty style (all fields inherit from theme).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the series color.
    #[must_use]
    pub fn color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }

    /// Set the line width for this series.
    #[must_use]
    pub fn line_width(mut self, width: f32) -> Self {
        self.line_width = Some(width);
        self
    }

    /// Set the dash pattern for this series.
    #[must_use]
    pub fn dash(mut self, pattern: DashPattern) -> Self {
        self.dash = Some(pattern);
        self
    }

    /// Set the area fill opacity (0.0–1.0).
    #[must_use]
    pub fn fill_opacity(mut self, opacity: f32) -> Self {
        self.fill_opacity = Some(opacity.clamp(0.0, 1.0));
        self
    }

    /// Set the gradient direction for filled line/area charts.
    #[must_use]
    pub fn fill_gradient(mut self, gradient: GradientFill) -> Self {
        self.fill_gradient = Some(gradient);
        self
    }

    /// Set a fill pattern overlay for bar charts (accessibility).
    #[must_use]
    pub fn fill_pattern(mut self, pattern: FillPattern) -> Self {
        self.fill_pattern = Some(pattern);
        self
    }

    /// Whether this style has any overrides at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.color.is_none()
            && self.line_width.is_none()
            && self.dash.is_none()
            && self.fill_opacity.is_none()
            && self.fill_gradient.is_none()
            && self.fill_pattern.is_none()
    }
}

// ---------------------------------------------------------------------------
// Series
// ---------------------------------------------------------------------------

/// A named sequence of floating-point data values.
///
/// # Examples
///
/// ```
/// use scry_chart::data::Series;
///
/// let s = Series::new("Temperature", vec![20.0, 22.5, 19.0, 25.0]);
/// assert_eq!(s.label(), "Temperature");
/// assert_eq!(s.len(), 4);
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[must_use]
pub struct Series {
    label: String,
    values: Vec<f64>,
    /// Symmetric error values (±error) for each data point.
    error_y: Option<Vec<f64>>,
    /// Per-series visual overrides.
    #[cfg_attr(feature = "serde", serde(skip))]
    style: SeriesStyle,
}

impl Series {
    /// Create a new series with a label and values.
    ///
    /// Non-finite values (NaN, Infinity) are preserved in storage but
    /// excluded from statistical calculations (min, max, mean, extent).
    pub fn new(label: impl Into<String>, values: Vec<f64>) -> Self {
        Self {
            label: label.into(),
            values,
            error_y: None,
            style: SeriesStyle::default(),
        }
    }

    /// Create an unlabeled series.
    pub fn from_values(values: Vec<f64>) -> Self {
        Self {
            label: String::new(),
            values,
            error_y: None,
            style: SeriesStyle::default(),
        }
    }

    /// Attach symmetric error values (±error) to this series.
    ///
    /// The error vector must have the same length as the data values.
    /// Each error value represents the magnitude of the error bar.
    pub fn with_error(mut self, errors: Vec<f64>) -> Self {
        self.error_y = Some(errors);
        self
    }

    /// Get the error values, if set.
    #[must_use]
    pub fn error_values(&self) -> Option<&[f64]> {
        self.error_y.as_deref()
    }

    /// Return a new Series with all non-finite values (NaN, Infinity) removed.
    /// Set per-series visual overrides.
    ///
    /// Any field left as `None` inherits from the chart theme.
    pub fn style(mut self, style: SeriesStyle) -> Self {
        self.style = style;
        self
    }

    /// Get the per-series style overrides.
    #[must_use]
    pub fn series_style(&self) -> &SeriesStyle {
        &self.style
    }

    /// Return a new Series with all non-finite values (NaN, Infinity) removed.
    pub fn sanitized(&self) -> Self {
        Self {
            label: self.label.clone(),
            values: self
                .values
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .collect(),
            error_y: None, // loses error association on sanitize
            style: self.style.clone(),
        }
    }

    /// Returns only the finite values (filtering NaN and Infinity).
    pub fn finite_values(&self) -> impl Iterator<Item = f64> + '_ {
        self.values.iter().copied().filter(|v| v.is_finite())
    }

    /// Count of non-finite values in the series.
    #[must_use]
    pub fn non_finite_count(&self) -> usize {
        self.values.iter().filter(|v| !v.is_finite()).count()
    }

    /// The series label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// The data values.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Number of data points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Count of finite (non-NaN, non-Infinity) values.
    #[must_use]
    pub fn finite_count(&self) -> usize {
        self.values.iter().filter(|v| v.is_finite()).count()
    }

    /// Whether the series is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The minimum finite value, or `None` if no finite values exist.
    #[must_use]
    pub fn min(&self) -> Option<f64> {
        self.finite_values().reduce(f64::min)
    }

    /// The maximum finite value, or `None` if no finite values exist.
    #[must_use]
    pub fn max(&self) -> Option<f64> {
        self.finite_values().reduce(f64::max)
    }

    /// (min, max) extent of the data.
    #[must_use]
    pub fn extent(&self) -> Option<(f64, f64)> {
        match (self.min(), self.max()) {
            (Some(lo), Some(hi)) => Some((lo, hi)),
            _ => None,
        }
    }

    /// Mean of the finite data values.
    #[must_use]
    pub fn mean(&self) -> Option<f64> {
        let finite: Vec<f64> = self.finite_values().collect();
        if finite.is_empty() {
            None
        } else {
            Some(finite.iter().sum::<f64>() / finite.len() as f64)
        }
    }
}

impl From<Vec<f64>> for Series {
    fn from(values: Vec<f64>) -> Self {
        Self::from_values(values)
    }
}

impl From<&[f64]> for Series {
    fn from(values: &[f64]) -> Self {
        Self::from_values(values.to_vec())
    }
}

/// Convert a slice of f32 values into a Series (widening to f64).
impl From<&[f32]> for Series {
    fn from(values: &[f32]) -> Self {
        Self::from_values(values.iter().map(|&v| f64::from(v)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn series_basic() {
        let s = Series::new("test", vec![1.0, 5.0, 3.0]);
        assert_eq!(s.label(), "test");
        assert_eq!(s.len(), 3);
        assert!(!s.is_empty());
    }

    #[test]
    fn series_extent() {
        let s = Series::new("", vec![3.0, 1.0, 5.0, 2.0]);
        assert_eq!(s.min(), Some(1.0));
        assert_eq!(s.max(), Some(5.0));
        assert_eq!(s.extent(), Some((1.0, 5.0)));
    }

    #[test]
    fn series_mean() {
        let s = Series::new("", vec![2.0, 4.0, 6.0]);
        assert_eq!(s.mean(), Some(4.0));
    }

    #[test]
    fn empty_series() {
        let s = Series::from_values(vec![]);
        assert!(s.is_empty());
        assert_eq!(s.min(), None);
        assert_eq!(s.extent(), None);
        assert_eq!(s.mean(), None);
    }

    #[test]
    fn nan_infinity_filtered() {
        let s = Series::new(
            "dirty",
            vec![1.0, f64::NAN, 3.0, f64::INFINITY, 2.0, f64::NEG_INFINITY],
        );
        assert_eq!(s.len(), 6); // raw length includes non-finite
        assert_eq!(s.non_finite_count(), 3);
        assert_eq!(s.min(), Some(1.0));
        assert_eq!(s.max(), Some(3.0));
        assert_eq!(s.extent(), Some((1.0, 3.0)));
        assert_eq!(s.mean(), Some(2.0));

        let clean = s.sanitized();
        assert_eq!(clean.len(), 3);
        assert_eq!(clean.values(), &[1.0, 3.0, 2.0]);
    }

    #[test]
    fn all_nan_series() {
        let s = Series::new("all_nan", vec![f64::NAN, f64::NAN]);
        assert_eq!(s.min(), None);
        assert_eq!(s.max(), None);
        assert_eq!(s.extent(), None);
        assert_eq!(s.mean(), None);
    }

    #[test]
    fn series_style_defaults() {
        let style = SeriesStyle::default();
        assert!(style.color.is_none());
        assert!(style.line_width.is_none());
        assert!(style.dash.is_none());
        assert!(style.fill_opacity.is_none());
        assert!(style.is_empty());
    }

    #[test]
    fn series_style_builder() {
        let red = Color::from_rgba8(255, 0, 0, 255);
        let style = SeriesStyle::new()
            .color(red)
            .line_width(3.0)
            .fill_opacity(0.5);
        assert_eq!(style.color, Some(red));
        assert_eq!(style.line_width, Some(3.0));
        assert_eq!(style.fill_opacity, Some(0.5));
        assert!(!style.is_empty());
    }

    #[test]
    fn series_style_opacity_clamped() {
        let style = SeriesStyle::new().fill_opacity(2.0);
        assert_eq!(style.fill_opacity, Some(1.0));
        let style = SeriesStyle::new().fill_opacity(-0.5);
        assert_eq!(style.fill_opacity, Some(0.0));
    }

    #[test]
    fn series_with_style() {
        let red = Color::from_rgba8(255, 0, 0, 255);
        let s = Series::new("styled", vec![1.0, 2.0])
            .style(SeriesStyle::new().color(red));
        assert_eq!(s.series_style().color, Some(red));
    }

    #[test]
    fn series_style_preserved_on_sanitize() {
        let red = Color::from_rgba8(255, 0, 0, 255);
        let s = Series::new("dirty", vec![1.0, f64::NAN, 3.0])
            .style(SeriesStyle::new().color(red));
        let clean = s.sanitized();
        assert_eq!(clean.series_style().color, Some(red));
        assert_eq!(clean.len(), 2);
    }

    #[test]
    fn series_style_gradient_fill() {
        let style = SeriesStyle::new().fill_gradient(GradientFill::TopToBottom);
        assert!(style.fill_gradient.is_some());
        assert!(!style.is_empty());
    }

    #[test]
    fn series_style_gradient_bottom_to_top() {
        let style = SeriesStyle::new().fill_gradient(GradientFill::BottomToTop);
        assert!(matches!(
            style.fill_gradient,
            Some(GradientFill::BottomToTop)
        ));
    }

    #[test]
    fn series_style_gradient_custom_stops() {
        let red = Color::from_rgba8(255, 0, 0, 255);
        let blue = Color::from_rgba8(0, 0, 255, 255);
        let style = SeriesStyle::new()
            .fill_gradient(GradientFill::Custom(vec![(0.0, red), (1.0, blue)]));
        assert!(!style.is_empty());
        if let Some(GradientFill::Custom(stops)) = &style.fill_gradient {
            assert_eq!(stops.len(), 2);
        } else {
            panic!("expected Custom gradient");
        }
    }

    #[test]
    fn series_style_fill_pattern() {
        let style = SeriesStyle::new().fill_pattern(FillPattern::Diagonal);
        assert_eq!(style.fill_pattern, Some(FillPattern::Diagonal));
        assert!(!style.is_empty());
    }

    #[test]
    fn series_style_all_patterns() {
        for pattern in [
            FillPattern::Solid,
            FillPattern::Hatched,
            FillPattern::CrossHatched,
            FillPattern::Dotted,
            FillPattern::Diagonal,
        ] {
            let style = SeriesStyle::new().fill_pattern(pattern);
            assert_eq!(style.fill_pattern, Some(pattern));
        }
    }

    #[test]
    fn series_style_combined_overrides() {
        let red = Color::from_rgba8(255, 0, 0, 255);
        let style = SeriesStyle::new()
            .color(red)
            .line_width(2.0)
            .fill_opacity(0.7)
            .fill_gradient(GradientFill::TopToBottom)
            .fill_pattern(FillPattern::Hatched);
        assert!(!style.is_empty());
        assert!(style.color.is_some());
        assert!(style.line_width.is_some());
        assert!(style.fill_opacity.is_some());
        assert!(style.fill_gradient.is_some());
        assert!(style.fill_pattern.is_some());
    }

    #[test]
    fn gap_policy_default_is_skip() {
        assert_eq!(GapPolicy::default(), GapPolicy::Skip);
    }

    #[test]
    fn gap_policy_equality_and_copy() {
        let a = GapPolicy::Interpolate;
        let b = a; // Copy
        assert_eq!(a, b);
        assert_ne!(GapPolicy::Skip, GapPolicy::Zero);
        assert_ne!(GapPolicy::Zero, GapPolicy::Interpolate);
    }
}

