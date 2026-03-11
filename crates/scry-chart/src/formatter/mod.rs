// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tick formatting system for chart axes.
//!
//! Provides a [`TickFormatter`] trait and built-in implementations for
//! common formatting needs. The default [`AutoFormatter`] produces
//! uniform-precision labels with SI suffixes for large values.

mod date;
pub(crate) mod locale;
pub mod numeric;
mod semantic;
/// Semantic zoom formatting — adapts tick labels to the current zoom level.
pub mod zoom;

pub use date::*;
use locale::apply_locale_batch;
pub use locale::LocaleConfig;
pub use numeric::*;
pub use semantic::*;
pub use zoom::{SemanticZoomFormatter, ZoomLevel};

use std::sync::Arc;

use crate::scale::format_tick_adaptive;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// A formatter that converts tick values to display strings.
///
/// The key method is `format_batch` which formats an entire axis's ticks
/// at once, enabling uniform precision across all labels on an axis.
pub trait TickFormatter: Send + Sync + std::fmt::Debug {
    /// Format a single tick value (convenience method).
    ///
    /// Default implementation delegates to `format_batch` with a
    /// single-element slice and domain `(value, value)`.
    fn format(&self, value: f64) -> String {
        self.format_batch(&[value], (value, value))
            .into_iter()
            .next()
            .unwrap_or_default()
    }

    /// Format a batch of tick values with uniform precision.
    ///
    /// All returned strings should use the same number of decimal places
    /// so labels on the same axis look consistent. The domain `(min, max)`
    /// is the full axis range, not just the tick range.
    fn format_batch(&self, values: &[f64], domain: (f64, f64)) -> Vec<String>;
}

/// Erase a `TickFormatter` into a shared trait object.
pub fn boxed_formatter(f: impl TickFormatter + 'static) -> Arc<dyn TickFormatter> {
    Arc::new(f)
}

// ---------------------------------------------------------------------------
// AutoFormatter — default, replaces format_tick_adaptive
// ---------------------------------------------------------------------------

/// Adaptive formatter that selects SI suffixes, integer, or decimal
/// formatting based on domain span. Guarantees uniform precision across
/// all ticks on an axis via `format_batch`.
///
/// Auto-detects percentage data: when all values are in `[0, 1]` or `[0, 100]`
/// and the axis label hints at percentages, values are formatted with `%`.
#[derive(Clone, Debug, Default)]
pub struct AutoFormatter;

impl TickFormatter for AutoFormatter {
    fn format_batch(&self, values: &[f64], domain: (f64, f64)) -> Vec<String> {
        if values.is_empty() {
            return Vec::new();
        }

        // Format each tick using the adaptive algorithm
        let labels: Vec<String> = values
            .iter()
            .map(|&v| format_tick_adaptive(v, domain.0, domain.1))
            .collect();

        // Ensure uniform precision: find the max decimal places used
        // and re-format any shorter labels to match
        uniform_precision(labels)
    }
}

/// Detect whether tick values represent percentages (all in [0,1] or [0,100])
/// and format them as `XX.X%`.
///
/// Returns `None` if the values don't look like percentages.
#[must_use]
pub fn try_format_as_percent(values: &[f64], domain: (f64, f64)) -> Option<Vec<String>> {
    if values.is_empty() {
        return None;
    }

    let (lo, hi) = domain;
    // Check [0, 1] range (fraction mode)
    if lo >= -0.001 && hi <= 1.001 {
        let labels: Vec<String> = values
            .iter()
            .map(|&v| {
                let pct = v * 100.0;
                if (pct - pct.round()).abs() < 0.05 {
                    format!("{}%", pct.round() as i64)
                } else {
                    format!("{pct:.1}%")
                }
            })
            .collect();
        return Some(labels);
    }

    // Check [0, 100] range
    if lo >= -0.1 && hi <= 100.1 {
        let labels: Vec<String> = values
            .iter()
            .map(|&v| {
                if (v - v.round()).abs() < 0.05 {
                    format!("{}%", v.round() as i64)
                } else {
                    format!("{v:.1}%")
                }
            })
            .collect();
        return Some(labels);
    }

    None
}

// ---------------------------------------------------------------------------
// NullFormatter — hides labels but keeps tick marks
// ---------------------------------------------------------------------------

/// Formats all tick labels as empty strings.
///
/// Useful for secondary axes where you want tick marks but no labels,
/// or for heatmaps where categorical labels are used instead.
#[derive(Clone, Debug, Default)]
pub struct NullFormatter;

impl TickFormatter for NullFormatter {
    fn format_batch(&self, values: &[f64], _domain: (f64, f64)) -> Vec<String> {
        vec![String::new(); values.len()]
    }
}

// ---------------------------------------------------------------------------
// FnFormatter — wraps a closure
// ---------------------------------------------------------------------------

/// Wraps a user-provided closure as a `TickFormatter`.
///
/// This is the simplest way to create a custom formatter without
/// implementing the trait manually.
///
/// # Example
///
/// ```
/// use scry_chart::formatter::FnFormatter;
///
/// let fmt = FnFormatter::new(|v| format!("{:.1}°C", v));
/// ```
pub struct FnFormatter {
    f: Box<dyn Fn(f64) -> String + Send + Sync>,
}

impl FnFormatter {
    /// Create a formatter from a closure.
    pub fn new(f: impl Fn(f64) -> String + Send + Sync + 'static) -> Self {
        Self { f: Box::new(f) }
    }
}

impl std::fmt::Debug for FnFormatter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FnFormatter").finish()
    }
}

impl TickFormatter for FnFormatter {
    fn format_batch(&self, values: &[f64], _domain: (f64, f64)) -> Vec<String> {
        values.iter().map(|&v| (self.f)(v)).collect()
    }
}

// ---------------------------------------------------------------------------
// LocaleFormatter — wraps any formatter with locale-aware post-processing
// ---------------------------------------------------------------------------

/// Wraps any [`TickFormatter`] with locale-aware number formatting.
///
/// Applies thousands grouping and decimal separator substitution
/// as a post-processing step after the inner formatter runs.
///
/// # Example
///
/// ```
/// use scry_chart::formatter::{LocaleFormatter, LocaleConfig, FixedDecimalFormatter, TickFormatter};
///
/// let fmt = LocaleFormatter::new(FixedDecimalFormatter(2), LocaleConfig::european());
/// let labels = fmt.format_batch(&[1234.56], (0.0, 2000.0));
/// assert_eq!(labels[0], "1.234,56");
/// ```
#[derive(Clone, Debug)]
pub struct LocaleFormatter<F: TickFormatter> {
    inner: F,
    locale: LocaleConfig,
}

impl<F: TickFormatter> LocaleFormatter<F> {
    /// Wrap a formatter with locale post-processing.
    #[must_use]
    pub fn new(inner: F, locale: LocaleConfig) -> Self {
        Self { inner, locale }
    }
}

impl<F: TickFormatter + 'static> TickFormatter for LocaleFormatter<F> {
    fn format_batch(&self, values: &[f64], domain: (f64, f64)) -> Vec<String> {
        let labels = self.inner.format_batch(values, domain);
        apply_locale_batch(labels, &self.locale)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Ensure all labels in a batch have consistent decimal precision.
///
/// Finds the maximum number of decimal digits in any label, then
/// pads shorter labels to match. Only operates on labels that look
/// like plain numbers (no SI suffix, no special formatting).
pub(crate) fn uniform_precision(labels: Vec<String>) -> Vec<String> {
    // --- Phase 1: Harmonize mixed SI/plain labels ---
    // If some labels use SI suffixes (K, M, G) and others are plain numbers,
    // convert the plain numbers to the same SI format for visual consistency.
    let labels = harmonize_si_labels(labels);

    // --- Phase 2: Uniform decimal precision for plain numbers ---
    // Find max decimal places among "plain number" labels
    let max_decimals = labels
        .iter()
        .filter_map(|l| {
            // Skip labels with suffixes (K, M, G, %, $, e, etc.)
            if l.bytes()
                .any(|b| (b.is_ascii_alphabetic() && b != b'e' && b != b'E') || b == b'%')
            {
                return None;
            }
            l.find('.').map(|dot| l.len() - dot - 1)
        })
        .max()
        .unwrap_or(0);

    if max_decimals == 0 {
        return labels;
    }

    labels
        .into_iter()
        .map(|l| {
            // Only pad plain-number labels
            if l.bytes()
                .any(|b| (b.is_ascii_alphabetic() && b != b'e' && b != b'E') || b == b'%')
            {
                return l;
            }

            if let Some(dot) = l.find('.') {
                let current_decimals = l.len() - dot - 1;
                if current_decimals < max_decimals {
                    format!("{l}{}", "0".repeat(max_decimals - current_decimals))
                } else {
                    l
                }
            } else {
                // Integer — add decimal point and zeros
                format!("{l}.{}", "0".repeat(max_decimals))
            }
        })
        .collect()
}

/// Ensure all plain-number labels use the same number of significant digits.
///
/// Useful when tick values span different magnitudes (e.g., 0.1, 0.2, 0.30000000000000004).
/// Formats each value to `sig_digits` significant figures, then pads to uniform
/// decimal precision.
#[must_use]
pub fn uniform_significant_digits(values: &[f64], sig_digits: usize) -> Vec<String> {
    if values.is_empty() {
        return Vec::new();
    }
    let sig = sig_digits.max(1).min(6);
    let labels: Vec<String> = values
        .iter()
        .map(|&v| {
            if v == 0.0 || !v.is_finite() {
                return "0".to_string();
            }
            let magnitude = v.abs().log10().floor() as i32;
            let decimals = (sig as i32 - 1 - magnitude).max(0) as usize;
            format!("{v:.prec$}", prec = decimals)
        })
        .collect();
    uniform_precision(labels)
}

/// Detect and harmonize mixed SI/plain number labels.
///
/// When an axis has both SI-suffixed labels (e.g., "10K") and plain numbers
/// (e.g., "5000"), converts the plain numbers to match the SI format so all
/// labels on the axis use the same style.
fn harmonize_si_labels(labels: Vec<String>) -> Vec<String> {
    if labels.len() < 2 {
        return labels;
    }

    // Count SI vs plain labels
    let mut si_count = 0usize;
    let mut plain_count = 0usize;
    let mut dominant_suffix = None;

    for l in &labels {
        let trimmed = l.trim_start_matches('-');
        if trimmed.ends_with('K') || trimmed.ends_with('M') || trimmed.ends_with('G') {
            si_count += 1;
            let suffix = trimmed.chars().last().unwrap();
            dominant_suffix = Some(suffix);
        } else if !l.is_empty()
            && !l
                .bytes()
                .any(|b| (b.is_ascii_alphabetic() && b != b'e' && b != b'E') || b == b'%')
        {
            plain_count += 1;
        }
    }

    // Only harmonize if we have a genuine mix (both SI and plain present)
    if si_count == 0 || plain_count == 0 {
        return labels;
    }

    let Some(suffix) = dominant_suffix else {
        return labels;
    };
    let divisor = match suffix {
        'K' => 1e3,
        'M' => 1e6,
        'G' => 1e9,
        _ => return labels,
    };

    labels
        .into_iter()
        .map(|l| {
            // Already has an SI suffix — keep as-is
            let trimmed = l.trim_start_matches('-');
            if trimmed.ends_with('K') || trimmed.ends_with('M') || trimmed.ends_with('G') {
                return l;
            }
            // Not a plain number — keep as-is
            if l.bytes()
                .any(|b| (b.is_ascii_alphabetic() && b != b'e' && b != b'E') || b == b'%')
            {
                return l;
            }
            // Convert plain number to SI
            l.parse::<f64>().map_or(l, |v| {
                if !v.is_finite() {
                    return format!("{v}");
                }
                let scaled = v / divisor;
                if (scaled - scaled.round()).abs() < 0.05 && scaled.round().abs() <= i64::MAX as f64
                {
                    format!("{}{suffix}", scaled.round() as i64)
                } else {
                    format!("{scaled:.1}{suffix}")
                }
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_formatter_uniform_precision() {
        let fmt = AutoFormatter;
        let labels = fmt.format_batch(&[0.0, 0.5, 1.0, 1.5, 2.0], (0.0, 2.0));
        // All should have the same decimal count
        let decimals: Vec<_> = labels
            .iter()
            .filter_map(|l| l.find('.').map(|d| l.len() - d - 1))
            .collect();
        assert!(
            decimals.windows(2).all(|w| w[0] == w[1]),
            "Inconsistent precision: {labels:?} → decimals: {decimals:?}"
        );
    }

    #[test]
    fn auto_formatter_negative_zero() {
        let fmt = AutoFormatter;
        let labels = fmt.format_batch(&[-0.0, 0.5, 1.0], (0.0, 1.0));
        assert!(
            !labels[0].starts_with('-'),
            "Negative zero displayed: {:?}",
            labels[0]
        );
    }

    #[test]
    fn auto_formatter_zero_domain() {
        let fmt = AutoFormatter;
        let labels = fmt.format_batch(&[0.0], (0.0, 0.0));
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0], "0");
    }

    #[test]
    fn fixed_decimal_formatter() {
        let fmt = FixedDecimalFormatter(2);
        let labels = fmt.format_batch(&[0.0, 1.5, 10.0], (0.0, 10.0));
        assert_eq!(labels, vec!["0.00", "1.50", "10.00"]);
    }

    #[test]
    fn scientific_formatter() {
        let fmt = ScientificFormatter::default();
        let labels = fmt.format_batch(&[1e6, 2.5e6], (1e6, 3e6));
        assert!(
            labels[0].contains('e'),
            "Expected scientific notation: {}",
            labels[0]
        );
    }

    #[test]
    fn null_formatter() {
        let fmt = NullFormatter;
        let labels = fmt.format_batch(&[1.0, 2.0, 3.0], (1.0, 3.0));
        assert!(labels.iter().all(|l| l.is_empty()));
    }

    #[test]
    fn fn_formatter() {
        let fmt = FnFormatter::new(|v| format!("{v:.0}°"));
        let labels = fmt.format_batch(&[20.0, 25.0, 30.0], (20.0, 30.0));
        assert_eq!(labels, vec!["20°", "25°", "30°"]);
    }

    #[test]
    fn si_formatter_full_range() {
        let fmt = SiFormatter::default();
        let labels = fmt.format_batch(&[0.0, 1_000.0, 1_000_000.0, 1e9, 1e12, 1e15], (0.0, 1e15));
        assert_eq!(labels[0], "0");
        assert_eq!(labels[1], "1K");
        assert_eq!(labels[2], "1M");
        assert_eq!(labels[3], "1G");
        assert_eq!(labels[4], "1T");
        assert_eq!(labels[5], "1P");
    }

    #[test]
    fn uniform_precision_pads_integers() {
        let result = uniform_precision(vec!["0".to_string(), "0.5".to_string(), "1".to_string()]);
        assert_eq!(result, vec!["0.0", "0.5", "1.0"]);
    }

    #[test]
    fn uniform_precision_skips_si_labels() {
        let result = uniform_precision(vec!["1K".to_string(), "2K".to_string()]);
        // Should NOT pad SI-suffixed labels
        assert_eq!(result, vec!["1K", "2K"]);
    }

    #[test]
    fn nice_step_zero_guard() {
        // This previously crashed with log10(0) = -inf
        let step = crate::scale::nice_step(0.0);
        assert!(
            step.is_finite(),
            "nice_step(0.0) returned non-finite: {step}"
        );
        assert!(step > 0.0, "nice_step(0.0) should be positive: {step}");
    }

    #[test]
    fn nice_ticks_zero_domain() {
        let ticks = crate::scale::nice_ticks(0.0, 0.0, 5);
        assert!(
            !ticks.is_empty(),
            "zero domain should produce at least one tick"
        );
        assert!(ticks[0].is_finite());
    }

    #[test]
    fn format_tick_negative_zero() {
        let label = crate::scale::format_tick_adaptive(-0.0, -1.0, 1.0);
        assert!(
            !label.starts_with('-'),
            "Negative zero should not display: got {label:?}"
        );
    }

    #[test]
    fn harmonize_si_mixed_labels() {
        // Mix of plain and SI labels should all become SI
        let result = harmonize_si_labels(vec![
            "5000".to_string(),
            "7500".to_string(),
            "10K".to_string(),
            "12.5K".to_string(),
        ]);
        assert_eq!(result[0], "5K");
        assert_eq!(result[1], "7.5K");
        assert_eq!(result[2], "10K");
        assert_eq!(result[3], "12.5K");
    }

    #[test]
    fn harmonize_si_all_plain_noop() {
        // All plain — no harmonization needed
        let result = harmonize_si_labels(vec![
            "100".to_string(),
            "200".to_string(),
            "300".to_string(),
        ]);
        assert_eq!(result, vec!["100", "200", "300"]);
    }

    #[test]
    fn thousands_formatter_basic() {
        let _labels =
            ThousandsFormatter.format_batch(&[0.0, 1000.0, 1234567.0], (0.0, 2_000_000.0));
        // Verify comma insertion via format_batch on small values
        let labels = ThousandsFormatter.format_batch(&[0.0, 1000.0, 5000.0], (0.0, 5000.0));
        assert_eq!(labels[1], "1,000");
        assert_eq!(labels[2], "5,000");
    }

    // --- BinarySiFormatter tests ---

    #[test]
    fn binary_si_formatter_powers() {
        let fmt = BinarySiFormatter::default();
        let labels = fmt.format_batch(&[0.0, 1024.0, 1048576.0, 1073741824.0], (0.0, 1073741824.0));
        assert_eq!(labels[0], "0");
        assert_eq!(labels[1], "1 KiB");
        assert_eq!(labels[2], "1 MiB");
        assert_eq!(labels[3], "1 GiB");
    }

    #[test]
    fn binary_si_formatter_fractional() {
        let fmt = BinarySiFormatter::default();
        let labels = fmt.format_batch(&[1.5 * 1048576.0], (0.0, 2.0 * 1048576.0));
        assert_eq!(labels[0], "1.5 MiB");
    }

    #[test]
    fn binary_si_formatter_small_values() {
        let fmt = BinarySiFormatter::default();
        let labels = fmt.format_batch(&[512.0, 100.0], (0.0, 1024.0));
        assert_eq!(labels[0], "512");
        assert_eq!(labels[1], "100");
    }

    // --- EngineeringFormatter tests ---

    #[test]
    fn engineering_formatter_basic() {
        let fmt = EngineeringFormatter::default();
        assert_eq!(fmt.format(47000.0), "47.00e3");
        assert_eq!(fmt.format(1500.0), "1.50e3");
    }

    #[test]
    fn engineering_formatter_small() {
        let fmt = EngineeringFormatter::default();
        assert_eq!(fmt.format(0.0025), "2.50e-3");
        assert_eq!(fmt.format(0.00047), "470.00e-6");
    }

    #[test]
    fn engineering_formatter_zero() {
        let fmt = EngineeringFormatter::default();
        assert_eq!(fmt.format(0.0), "0");
        assert_eq!(fmt.format(-0.0), "0");
    }

    #[test]
    fn engineering_formatter_unity_range() {
        let fmt = EngineeringFormatter::default();
        // Values 1-999 should have no exponent
        assert_eq!(fmt.format(1.0), "1.00");
        assert_eq!(fmt.format(42.5), "42.50");
        assert_eq!(fmt.format(999.0), "999.00");
    }

    // --- Phase 3: Formatter edge case hardening ---

    #[test]
    fn auto_formatter_all_identical_values() {
        let fmt = AutoFormatter;
        let labels = fmt.format_batch(&[5.0, 5.0, 5.0], (5.0, 5.0));
        // All labels should be identical and readable
        assert!(
            labels.iter().all(|l| l == &labels[0]),
            "All-identical values should produce identical labels: {labels:?}"
        );
        assert!(!labels[0].is_empty(), "Labels should not be empty");
    }

    #[test]
    fn si_formatter_boundary_999_5() {
        let fmt = SiFormatter { decimals: 0 };
        // 999.5 is right at the K boundary — should round to 1K or stay as 1000
        let label = fmt.format(999.5);
        // Either "1K" or "1000" are acceptable — just shouldn't panic or produce garbage
        assert!(
            label == "1K" || label == "1000" || label == "999",
            "SI boundary 999.5 produced unexpected: {label}"
        );
    }

    #[test]
    fn si_formatter_exact_boundaries() {
        let fmt = SiFormatter { decimals: 1 };
        assert_eq!(fmt.format(1000.0), "1K");
        assert_eq!(fmt.format(1_000_000.0), "1M");
        assert_eq!(fmt.format(1_000_000_000.0), "1G");
    }

    #[test]
    fn auto_formatter_large_span_consistency() {
        let fmt = AutoFormatter;
        let labels = fmt.format_batch(&[0.0, 100000.0, 200000.0, 300000.0], (0.0, 300000.0));
        // All labels should use the same format style (all SI or all plain)
        let si_count = labels
            .iter()
            .filter(|l| l.contains('K') || l.contains('M'))
            .count();
        let non_zero_count = labels.iter().filter(|l| *l != "0" && *l != "0K").count();
        // Either all SI or all plain — no mixing (except zero)
        assert!(
            si_count == 0 || si_count >= non_zero_count,
            "Mixed SI/plain labels: {labels:?}"
        );
    }
}
