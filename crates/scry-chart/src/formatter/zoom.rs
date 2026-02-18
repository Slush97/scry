// SPDX-License-Identifier: MIT OR Apache-2.0
//! Semantic zoom formatting — adapts tick label formatting to zoom level.

use crate::formatter::{
    AutoFormatter, FixedDecimalFormatter, ScientificFormatter, SiFormatter, TickFormatter,
};

// ---------------------------------------------------------------------------
// ZoomLevel
// ---------------------------------------------------------------------------

/// Classification of the current zoom level based on the ratio of the
/// visible domain span to the full data span.
///
/// Used by [`SemanticZoomFormatter`] to select an appropriate child
/// formatter for the current zoom level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ZoomLevel {
    /// Very wide view (ratio ≥ 0.75) — compact summary labels.
    Overview,
    /// Normal view (0.25 ≤ ratio < 0.75) — standard adaptive formatting.
    Standard,
    /// Zoomed in (0.05 ≤ ratio < 0.25) — detailed fixed-decimal labels.
    Detailed,
    /// Extreme zoom (ratio < 0.05) — scientific notation.
    Microscope,
}

impl ZoomLevel {
    /// Classify the current zoom level from the visible domain span and the
    /// full (original) data span.
    ///
    /// Returns [`Overview`](Self::Overview) for the widest view,
    /// [`Microscope`](Self::Microscope) for the most zoomed-in view.
    ///
    /// # Example
    ///
    /// ```
    /// use scry_chart::formatter::ZoomLevel;
    ///
    /// // Viewing the full dataset
    /// assert_eq!(ZoomLevel::from_domain_span(100.0, 100.0), ZoomLevel::Overview);
    ///
    /// // Zoomed to 10% of the data
    /// assert_eq!(ZoomLevel::from_domain_span(10.0, 100.0), ZoomLevel::Detailed);
    /// ```
    #[must_use]
    pub fn from_domain_span(span: f64, full_span: f64) -> Self {
        if full_span <= 0.0 || !full_span.is_finite() || !span.is_finite() {
            return Self::Standard;
        }
        let ratio = (span / full_span).clamp(0.0, 1.0);
        if ratio >= 0.75 {
            Self::Overview
        } else if ratio >= 0.25 {
            Self::Standard
        } else if ratio >= 0.05 {
            Self::Detailed
        } else {
            Self::Microscope
        }
    }
}

// ---------------------------------------------------------------------------
// SemanticZoomFormatter
// ---------------------------------------------------------------------------

/// A formatter that selects among child formatters based on the current
/// zoom level.
///
/// At overview zoom (wide out), uses compact SI labels. As the user
/// zooms in, progressively switches to standard, fixed-decimal, and
/// finally scientific notation for maximum precision.
///
/// # Example
///
/// ```
/// use scry_chart::formatter::{SemanticZoomFormatter, TickFormatter};
///
/// let fmt = SemanticZoomFormatter::default();
/// // At wide domain (overview), delegates to SiFormatter
/// let labels = fmt.format_batch(&[1_000_000.0], (0.0, 10_000_000.0));
/// assert!(labels[0].contains('M') || labels[0].contains('K') || labels[0] == "1000000");
/// ```
#[derive(Clone)]
pub struct SemanticZoomFormatter {
    /// Formatter for [`ZoomLevel::Overview`].
    pub overview: std::sync::Arc<dyn TickFormatter>,
    /// Formatter for [`ZoomLevel::Standard`].
    pub standard: std::sync::Arc<dyn TickFormatter>,
    /// Formatter for [`ZoomLevel::Detailed`].
    pub detailed: std::sync::Arc<dyn TickFormatter>,
    /// Formatter for [`ZoomLevel::Microscope`].
    pub microscope: std::sync::Arc<dyn TickFormatter>,
    /// Full data span (used to classify zoom level). Set by the builder.
    pub full_span: f64,
}

impl Default for SemanticZoomFormatter {
    fn default() -> Self {
        Self {
            overview: std::sync::Arc::new(SiFormatter::default()),
            standard: std::sync::Arc::new(AutoFormatter),
            detailed: std::sync::Arc::new(FixedDecimalFormatter(2)),
            microscope: std::sync::Arc::new(ScientificFormatter::default()),
            full_span: 0.0, // 0 means "auto-detect from first batch"
        }
    }
}

impl std::fmt::Debug for SemanticZoomFormatter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SemanticZoomFormatter")
            .field("full_span", &self.full_span)
            .finish()
    }
}

impl TickFormatter for SemanticZoomFormatter {
    fn format_batch(&self, values: &[f64], domain: (f64, f64)) -> Vec<String> {
        let span = (domain.1 - domain.0).abs();
        let full = if self.full_span > 0.0 {
            self.full_span
        } else {
            // When full_span is unknown, assume current domain IS the full span
            // (i.e., Overview level).
            span
        };
        let level = ZoomLevel::from_domain_span(span, full);
        match level {
            ZoomLevel::Overview => self.overview.format_batch(values, domain),
            ZoomLevel::Standard => self.standard.format_batch(values, domain),
            ZoomLevel::Detailed => self.detailed.format_batch(values, domain),
            ZoomLevel::Microscope => self.microscope.format_batch(values, domain),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zoom_level_from_span() {
        // Full view → Overview
        assert_eq!(
            ZoomLevel::from_domain_span(100.0, 100.0),
            ZoomLevel::Overview
        );
        // 80% → still Overview
        assert_eq!(
            ZoomLevel::from_domain_span(80.0, 100.0),
            ZoomLevel::Overview
        );
        // 50% → Standard
        assert_eq!(
            ZoomLevel::from_domain_span(50.0, 100.0),
            ZoomLevel::Standard
        );
        // 10% → Detailed
        assert_eq!(
            ZoomLevel::from_domain_span(10.0, 100.0),
            ZoomLevel::Detailed
        );
        // 2% → Microscope
        assert_eq!(
            ZoomLevel::from_domain_span(2.0, 100.0),
            ZoomLevel::Microscope
        );
        // Edge: zero full_span → Standard (safe fallback)
        assert_eq!(ZoomLevel::from_domain_span(10.0, 0.0), ZoomLevel::Standard);
        // Edge: NaN → Standard
        assert_eq!(
            ZoomLevel::from_domain_span(f64::NAN, 100.0),
            ZoomLevel::Standard
        );
    }

    #[test]
    fn semantic_zoom_overview() {
        let mut fmt = SemanticZoomFormatter::default();
        fmt.full_span = 1_000_000.0;
        // Full domain → Overview → SiFormatter
        let labels = fmt.format_batch(&[1_000_000.0], (0.0, 1_000_000.0));
        assert_eq!(labels[0], "1M");
    }

    #[test]
    fn semantic_zoom_detailed() {
        let mut fmt = SemanticZoomFormatter::default();
        fmt.full_span = 1000.0;
        // 15% of span → Detailed → FixedDecimalFormatter(2)
        let labels = fmt.format_batch(&[3.14159], (0.0, 150.0));
        assert_eq!(labels[0], "3.14");
    }

    #[test]
    fn semantic_zoom_microscope() {
        let mut fmt = SemanticZoomFormatter::default();
        fmt.full_span = 1000.0;
        // 1% of span → Microscope → ScientificFormatter
        let labels = fmt.format_batch(&[0.001234], (0.0, 10.0));
        assert!(
            labels[0].contains('e'),
            "expected scientific notation, got: {}",
            labels[0]
        );
    }
}
