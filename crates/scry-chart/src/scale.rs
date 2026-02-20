// SPDX-License-Identifier: MIT OR Apache-2.0
//! Scale system for mapping data domains to pixel ranges.
//!
//! Scales are the core arithmetic of chart rendering — they translate from
//! data coordinates (e.g., 0.0–100.0) to pixel coordinates (e.g., 40–760px).

use std::fmt;

// ---------------------------------------------------------------------------
// Scale trait
// ---------------------------------------------------------------------------

/// A mapping from data domain → pixel range.
pub trait Scale: fmt::Debug {
    /// Map a data value to a pixel coordinate.
    fn to_pixel(&self, value: f64) -> f64;

    /// Map a pixel coordinate back to a data value.
    fn to_data(&self, pixel: f64) -> f64;

    /// Generate nicely-spaced tick values within the domain.
    fn ticks(&self, target_count: usize) -> Vec<f64>;

    /// Format a tick value for display.
    fn format_tick(&self, value: f64) -> String;
}

// ---------------------------------------------------------------------------
// LinearScale
// ---------------------------------------------------------------------------

/// A linear mapping from `[domain_min, domain_max]` → `[range_min, range_max]`.
#[derive(Clone, Debug)]
pub struct LinearScale {
    pub(crate) domain_min: f64,
    pub(crate) domain_max: f64,
    range_min: f64,
    range_max: f64,
}

impl LinearScale {
    /// Create a new linear scale.
    #[must_use]
    pub fn new(domain: (f64, f64), range: (f64, f64)) -> Self {
        Self {
            domain_min: domain.0,
            domain_max: domain.1,
            range_min: range.0,
            range_max: range.1,
        }
    }

    /// Create a scale with nice domain bounds that include the data extent
    /// with some padding.
    ///
    /// Handles edge cases:
    /// - **Degenerate (single point):** pads by ±10% of |value| or ±5 if near zero.
    /// - **Micro-ranges (span < 1.0):** uses span-relative padding so the domain
    ///   stays tight around the data instead of rounding to distant integers.
    /// - **Normal ranges:** 5% padding + nice rounding.
    #[must_use]
    pub fn nice(extent: (f64, f64), range: (f64, f64)) -> Self {
        let (lo, hi) = extent;
        let span = hi - lo;

        if span.abs() < f64::EPSILON {
            // Degenerate: single value or all-equal data.
            // Pad by ±10% of |value|, minimum ±5.0 so the axis isn't trivial.
            let pad = (lo.abs() * 0.1).max(5.0);
            let nice_lo = nice_floor(lo - pad);
            let nice_hi = nice_ceil(hi + pad);
            return Self::new((nice_lo, nice_hi), range);
        }

        let padding = span * 0.05;
        let padded_lo = lo - padding;
        let padded_hi = hi + padding;

        // For micro-ranges (span < 1.0), use span-relative nice rounding
        // instead of absolute magnitude rounding which would clobber the range.
        let (nice_lo, nice_hi) = if span < 1.0 {
            let step = nice_step(span / 5.0);
            let nlo = (padded_lo / step).floor() * step;
            let nhi = (padded_hi / step).ceil() * step;
            (nlo, nhi)
        } else {
            (nice_floor(padded_lo), nice_ceil(padded_hi))
        };

        // Zero-snap heuristic: if data is all-positive (or all-negative) and
        // the nice bound is close to 0 (within 15% of the data span), snap to
        // 0 so the axis origin is clean. This matches D3/matplotlib behavior.
        // Skip for micro-ranges where snapping to 0 would clobber the data.
        //
        // Threshold is 15% (not 25%) to avoid snapping too aggressively,
        // which wastes plot area when data doesn't start near zero.
        let nice_lo = if span >= 1.0 && lo >= 0.0 && nice_lo.abs() < span * 0.15 {
            0.0
        } else {
            nice_lo
        };
        let nice_hi = if span >= 1.0 && hi <= 0.0 && nice_hi.abs() < span * 0.15 {
            0.0
        } else {
            nice_hi
        };

        Self::new((nice_lo, nice_hi), range)
    }

    /// Like `nice()` but ensures `0.0` is always within the domain and
    /// anchors the domain at zero — padding is only applied to the
    /// non-zero end so bars sit flush on the axis.
    #[must_use]
    pub fn nice_zero(extent: (f64, f64), range: (f64, f64)) -> Self {
        let (lo, hi) = extent;
        // Force domain to include zero
        let (adj_lo, adj_hi) = (lo.min(0.0), hi.max(0.0));

        // Nice-round, but anchor the zero side exactly at 0 to avoid
        // the axis gap caused by padding pushing the domain away from zero.
        let scale = Self::nice((adj_lo, adj_hi), range);
        let (d_lo, d_hi) = scale.domain();

        let final_lo = if adj_lo >= 0.0 { 0.0 } else { d_lo.min(0.0) };
        let final_hi = if adj_hi <= 0.0 { 0.0 } else { d_hi.max(0.0) };

        Self::new((final_lo, final_hi), range)
    }

    /// The domain bounds.
    #[must_use]
    pub fn domain(&self) -> (f64, f64) {
        (self.domain_min, self.domain_max)
    }

    /// The pixel range.
    #[must_use]
    pub fn range(&self) -> (f64, f64) {
        (self.range_min, self.range_max)
    }

    /// Return a new scale with the range endpoints swapped, effectively
    /// reversing the axis direction.
    ///
    /// For Y axes this makes high values appear at the bottom; for X axes
    /// it makes high values appear on the left.
    #[must_use]
    pub fn inverted(&self) -> Self {
        Self {
            domain_min: self.domain_min,
            domain_max: self.domain_max,
            range_min: self.range_max,
            range_max: self.range_min,
        }
    }
}

impl Scale for LinearScale {
    fn to_pixel(&self, value: f64) -> f64 {
        let domain_span = self.domain_max - self.domain_min;
        if domain_span.abs() < f64::EPSILON {
            return (self.range_min + self.range_max) / 2.0;
        }
        let t = (value - self.domain_min) / domain_span;
        self.range_min + t * (self.range_max - self.range_min)
    }

    fn to_data(&self, pixel: f64) -> f64 {
        let range_span = self.range_max - self.range_min;
        if range_span.abs() < f64::EPSILON {
            return (self.domain_min + self.domain_max) / 2.0;
        }
        let t = (pixel - self.range_min) / range_span;
        self.domain_min + t * (self.domain_max - self.domain_min)
    }

    fn ticks(&self, target_count: usize) -> Vec<f64> {
        nice_ticks(self.domain_min, self.domain_max, target_count)
    }

    fn format_tick(&self, value: f64) -> String {
        format_tick_adaptive(value, self.domain_min, self.domain_max)
    }
}

// ---------------------------------------------------------------------------
// CategoricalScale
// ---------------------------------------------------------------------------

/// Maps categorical labels to evenly-spaced pixel positions.
#[derive(Clone, Debug)]
pub struct CategoricalScale {
    labels: Vec<String>,
    range_min: f64,
    range_max: f64,
}

impl CategoricalScale {
    /// Create a categorical scale from labels.
    #[must_use]
    pub fn new(labels: Vec<String>, range: (f64, f64)) -> Self {
        Self {
            labels,
            range_min: range.0,
            range_max: range.1,
        }
    }

    /// The category labels.
    #[must_use]
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Get the center pixel position for a given category index.
    #[must_use]
    pub fn center(&self, index: usize) -> f64 {
        let n = self.labels.len();
        if n == 0 {
            return (self.range_min + self.range_max) / 2.0;
        }
        let band_width = (self.range_max - self.range_min) / n as f64;
        self.range_min + band_width * (index as f64 + 0.5)
    }

    /// Width of each category band in pixels.
    #[must_use]
    pub fn band_width(&self) -> f64 {
        let n = self.labels.len();
        if n == 0 {
            return 0.0;
        }
        (self.range_max - self.range_min) / n as f64
    }
}

// ---------------------------------------------------------------------------
// LogScale
// ---------------------------------------------------------------------------

/// A logarithmic (base-10) mapping from `[domain_min, domain_max]` → `[range_min, range_max]`.
///
/// Values ≤ 0 are clamped to a small positive epsilon for safety.
#[derive(Clone, Debug)]
pub struct LogScale {
    domain_min: f64,
    domain_max: f64,
    range_min: f64,
    range_max: f64,
    log_lo: f64,
    log_hi: f64,
}

impl LogScale {
    /// Create a new log scale. Domain values must be positive.
    #[must_use]
    pub fn new(domain: (f64, f64), range: (f64, f64)) -> Self {
        let lo = domain.0.max(f64::EPSILON);
        let hi = domain.1.max(lo * 10.0);
        Self {
            domain_min: lo,
            domain_max: hi,
            range_min: range.0,
            range_max: range.1,
            log_lo: lo.log10(),
            log_hi: hi.log10(),
        }
    }

    /// Create a log scale with nice domain bounds.
    #[must_use]
    pub fn nice(extent: (f64, f64), range: (f64, f64)) -> Self {
        let lo = extent.0.max(f64::EPSILON);
        let hi = extent.1.max(lo * 10.0);

        // Round to nearest power of 10
        let nice_lo = 10.0_f64.powf(lo.log10().floor());
        let nice_hi = 10.0_f64.powf(hi.log10().ceil());

        Self::new((nice_lo, nice_hi), range)
    }

    /// The domain bounds.
    #[must_use]
    pub fn domain(&self) -> (f64, f64) {
        (self.domain_min, self.domain_max)
    }
}

impl Scale for LogScale {
    fn to_pixel(&self, value: f64) -> f64 {
        let v = value.max(f64::EPSILON);
        let log_span = self.log_hi - self.log_lo;
        if log_span.abs() < f64::EPSILON {
            return (self.range_min + self.range_max) / 2.0;
        }
        let t = (v.log10() - self.log_lo) / log_span;
        self.range_min + t * (self.range_max - self.range_min)
    }

    fn to_data(&self, pixel: f64) -> f64 {
        let range_span = self.range_max - self.range_min;
        if range_span.abs() < f64::EPSILON {
            return (self.domain_min + self.domain_max) / 2.0;
        }
        let t = (pixel - self.range_min) / range_span;
        let log_val = self.log_lo + t * (self.log_hi - self.log_lo);
        10.0_f64.powf(log_val)
    }

    fn ticks(&self, target_count: usize) -> Vec<f64> {
        log_ticks(self.domain_min, self.domain_max, target_count)
    }

    fn format_tick(&self, value: f64) -> String {
        if value >= 1_000_000.0 {
            format!("{:.0e}", value)
        } else if value >= 1.0 && (value - value.round()).abs() < value * 0.001 {
            format!("{}", value.round() as i64)
        } else if value >= 0.01 {
            format!("{value:.2}")
        } else {
            format!("{value:.1e}")
        }
    }
}

/// Generate tick values for a log-scale axis.
///
/// Adapts sub-decade multipliers based on the number of decades spanned:
/// - ≥4 decades: decade boundaries only `{1}`
/// - 2–3 decades: `{1, 2, 5}` (classic)
/// - <2 decades: finer `{1, 2, 3, 5, 7}` for better resolution
fn log_ticks(lo: f64, hi: f64, _target_count: usize) -> Vec<f64> {
    let lo = lo.max(f64::EPSILON);
    let hi = hi.max(lo);

    let log_lo_f = lo.log10().floor();
    let log_hi_f = hi.log10().ceil();

    // Guard: non-finite logs or extreme range → just return endpoints
    if !log_lo_f.is_finite() || !log_hi_f.is_finite() {
        return vec![lo, hi];
    }

    let log_lo = (log_lo_f as i32).max(-20);
    let log_hi = (log_hi_f as i32).min(20);
    let decades = (log_hi - log_lo).unsigned_abs() as usize;

    // Adaptive multipliers based on range span
    let multipliers: &[f64] = if decades >= 4 {
        &[1.0] // decade boundaries only — avoid crowding
    } else if decades >= 2 {
        &[1.0, 2.0, 5.0] // classic — good for 2-3 decades
    } else {
        &[1.0, 2.0, 3.0, 5.0, 7.0] // finer — sub-decade resolution
    };

    let mut ticks = Vec::new();
    for exp in log_lo..=log_hi {
        let base = 10.0_f64.powi(exp);
        for mult in multipliers {
            let v = base * mult;
            if v >= lo * 0.99 && v <= hi * 1.01 {
                ticks.push(v);
            }
        }
    }

    if ticks.is_empty() {
        ticks.push(lo);
    }

    ticks
}

// ---------------------------------------------------------------------------
// Nice tick generation
// ---------------------------------------------------------------------------

/// Generate "nice" tick values for a given range and target count.
pub(crate) fn nice_ticks(lo: f64, hi: f64, target_count: usize) -> Vec<f64> {
    let target = target_count.max(2);
    let span = hi - lo;
    if span.abs() < f64::EPSILON || !span.is_finite() {
        return if lo.is_finite() { vec![lo] } else { vec![0.0] };
    }

    let rough_step = span / (target - 1) as f64;
    let step = nice_step(rough_step);

    // Guard: if step is not finite or is zero, bail out
    if !step.is_finite() || step <= 0.0 {
        return vec![lo, hi];
    }

    let start = (lo / step).ceil() * step;
    let mut ticks = Vec::new();
    let mut v = start;
    // Cap iterations to prevent infinite loops from floating-point precision
    // loss where `v + step == v` for very large values.
    let max_ticks = (target * 3).max(20);
    while v <= hi + step * 0.001 && ticks.len() < max_ticks {
        ticks.push(v);
        let next = v + step;
        if next <= v {
            break; // precision exhausted
        }
        v = next;
    }

    // M1: Guarantee domain endpoints are included so readers always
    // see the data range, even if nice rounding missed them.
    // Use 0.3×step threshold to avoid near-overlapping labels when
    // endpoints are close to nice ticks (e.g., 0.95 next to 1.0).
    if let Some(&first) = ticks.first() {
        if (first - lo).abs() > step * 0.3 && lo < first {
            ticks.insert(0, lo);
        }
    }
    if let Some(&last) = ticks.last() {
        if (last - hi).abs() > step * 0.3 && hi > last {
            ticks.push(hi);
        }
    }

    ticks
}

/// Find a "nice" step size close to the given rough step.
pub(crate) fn nice_step(rough: f64) -> f64 {
    // Guard against zero or near-zero step (would produce log10(0) = -inf)
    if rough.abs() < f64::EPSILON * 100.0 {
        return 1.0;
    }
    let magnitude = 10.0_f64.powf(rough.abs().log10().floor());
    let fraction = rough / magnitude;

    let nice_fraction = if fraction <= 1.0 {
        1.0
    } else if fraction <= 2.0 {
        2.0
    } else if fraction <= 3.0 {
        2.5
    } else if fraction <= 5.0 {
        5.0
    } else {
        10.0
    };

    nice_fraction * magnitude
}

/// Round down to a "nice" number.
fn nice_floor(v: f64) -> f64 {
    if v == 0.0 {
        return 0.0;
    }
    let abs = v.abs();
    if abs < f64::EPSILON {
        return 0.0;
    }
    let magnitude = 10.0_f64.powf(abs.log10().floor());
    if v > 0.0 {
        (v / magnitude).floor() * magnitude
    } else {
        // For negative values, floor goes more negative
        -((-v / magnitude).ceil() * magnitude)
    }
}

/// Round up to a "nice" number.
fn nice_ceil(v: f64) -> f64 {
    if v == 0.0 {
        return 0.0;
    }
    let abs = v.abs();
    if abs < f64::EPSILON {
        return 0.0;
    }
    let magnitude = 10.0_f64.powf(abs.log10().floor());
    if v > 0.0 {
        (v / magnitude).ceil() * magnitude
    } else {
        // For negative values, ceil goes toward zero
        -((-v / magnitude).floor() * magnitude)
    }
}

/// Format a tick value adaptively based on the domain span.
///
/// Uses SI suffixes (K, M, G) for large values and span-relative
/// decimal precision for small ranges.
pub(crate) fn format_tick_adaptive(value: f64, domain_min: f64, domain_max: f64) -> String {
    // Canonicalize negative zero so no tick ever displays as "-0".
    // Use abs() < epsilon to also catch -0.0 from float arithmetic.
    let value = if value == 0.0 || value.abs() < f64::EPSILON * 100.0 {
        0.0
    } else {
        value
    };
    let span = (domain_max - domain_min).abs();
    let abs = value.abs();

    // SI suffix formatting for large numbers
    if span >= 1_000.0 {
        if abs >= 1e9 {
            let v = value / 1e9;
            return if (v - v.round()).abs() < 0.05 && v.round().abs() <= i64::MAX as f64 {
                format!("{}G", v.round() as i64)
            } else {
                format!("{v:.1}G")
            };
        }
        if abs >= 1e6 {
            let v = value / 1e6;
            return if (v - v.round()).abs() < 0.05 && v.round().abs() <= i64::MAX as f64 {
                format!("{}M", v.round() as i64)
            } else {
                format!("{v:.1}M")
            };
        }
        if abs >= 1e4 {
            let v = value / 1e3;
            return if (v - v.round()).abs() < 0.05 && v.round().abs() <= i64::MAX as f64 {
                format!("{}K", v.round() as i64)
            } else {
                format!("{v:.1}K")
            };
        }
    }

    // Scientific notation for very small non-zero values
    if abs > 0.0 && abs < 0.01 && span < 1.0 {
        return format!("{value:.2e}");
    }

    // Integer formatting when value is whole and span is reasonable
    if (value - value.round()).abs() < f64::EPSILON * 100.0 && span >= 1.0 {
        // Guard against i64 overflow for huge values
        let rounded = value.round();
        if rounded >= i64::MIN as f64 && rounded <= i64::MAX as f64 {
            return format!("{}", rounded as i64);
        }
        return format!("{value:.0}");
    }

    // Span-adaptive decimal precision
    if span < f64::EPSILON {
        return format!("{value}");
    }

    // Compute the tick step to decide decimals needed
    let step = nice_step(span / 5.0);
    let decimals = if step >= 1.0 {
        usize::from((value - value.round()).abs() >= f64::EPSILON * 100.0)
    } else {
        // Count decimals needed to represent the step
        -step.log10().floor() as usize
    };

    // If we need more than 6 decimal places, switch to scientific notation.
    // This handles sub-micro values like 1e-7, 2.5e-8, etc.
    if decimals > 6 {
        // Use engineering-style: show significant digits relative to step
        let sig_digits = if step < 1e-15 {
            6
        } else {
            let step_digits = (-step.log10().floor() as usize)
                .saturating_sub(-abs.max(1e-300).log10().floor() as usize);
            step_digits.max(1).min(4)
        };
        return match sig_digits {
            1 => format!("{value:.1e}"),
            3 => format!("{value:.3e}"),
            _ => format!("{value:.2e}"),
        };
    }

    match decimals {
        0 => {
            let rounded = value.round();
            if rounded >= i64::MIN as f64 && rounded <= i64::MAX as f64 {
                format!("{}", rounded as i64)
            } else {
                format!("{value:.0}")
            }
        }
        1 => format!("{value:.1}"),
        2 => format!("{value:.2}"),
        3 => format!("{value:.3}"),
        4 => format!("{value:.4}"),
        5 => format!("{value:.5}"),
        _ => format!("{value:.6}"),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_scale_identity() {
        let s = LinearScale::new((0.0, 1.0), (0.0, 1.0));
        assert!((s.to_pixel(0.5) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn linear_scale_maps() {
        let s = LinearScale::new((0.0, 100.0), (0.0, 500.0));
        assert!((s.to_pixel(50.0) - 250.0).abs() < f64::EPSILON);
        assert!((s.to_data(250.0) - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn linear_scale_inverted_range() {
        // Y axis: domain 0→100, range 500→0 (screen Y is inverted)
        let s = LinearScale::new((0.0, 100.0), (500.0, 0.0));
        assert!((s.to_pixel(0.0) - 500.0).abs() < f64::EPSILON);
        assert!((s.to_pixel(100.0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn inverted_scale_swaps_range() {
        let s = LinearScale::new((0.0, 100.0), (0.0, 500.0));
        let inv = s.inverted();
        assert_eq!(inv.domain(), (0.0, 100.0));
        assert_eq!(inv.range(), (500.0, 0.0));
    }

    #[test]
    fn inverted_scale_preserves_domain() {
        let s = LinearScale::nice((10.0, 90.0), (40.0, 760.0));
        let inv = s.inverted();
        assert_eq!(s.domain(), inv.domain());
        // Range is swapped
        assert_eq!(inv.range(), (760.0, 40.0));
    }

    #[test]
    fn inverted_scale_reverses_mapping() {
        let s = LinearScale::new((0.0, 100.0), (0.0, 500.0));
        let inv = s.inverted();
        // 0 maps to 500 (right→left) and 100 maps to 0
        assert!((inv.to_pixel(0.0) - 500.0).abs() < f64::EPSILON);
        assert!((inv.to_pixel(100.0) - 0.0).abs() < f64::EPSILON);
        assert!((inv.to_pixel(50.0) - 250.0).abs() < f64::EPSILON);
    }

    #[test]
    fn nice_ticks_output() {
        let ticks = nice_ticks(0.0, 100.0, 5);
        assert!(!ticks.is_empty());
        assert!(ticks[0] >= 0.0);
        assert!(*ticks.last().unwrap() <= 110.0);
    }

    #[test]
    fn categorical_scale_centers() {
        let s = CategoricalScale::new(vec!["A".into(), "B".into(), "C".into()], (0.0, 300.0));
        assert!((s.center(0) - 50.0).abs() < f64::EPSILON);
        assert!((s.center(1) - 150.0).abs() < f64::EPSILON);
        assert!((s.center(2) - 250.0).abs() < f64::EPSILON);
        assert!((s.band_width() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn log_scale_maps() {
        let s = LogScale::new((1.0, 1000.0), (0.0, 300.0));
        // log10(1) = 0, log10(1000) = 3, so midpoint log10(~31.6) ≈ 1.5 → 150px
        let mid = s.to_pixel(10.0_f64.powf(1.5)); // √1000 ≈ 31.6
        assert!((mid - 150.0).abs() < 1.0, "mid was {mid}");
    }

    #[test]
    fn log_scale_round_trip() {
        let s = LogScale::new((1.0, 10000.0), (100.0, 500.0));
        let original = 42.0;
        let pixel = s.to_pixel(original);
        let back = s.to_data(pixel);
        assert!(
            (back - original).abs() < 0.01,
            "round trip: {back} != {original}"
        );
    }

    #[test]
    fn log_scale_nice_bounds() {
        let s = LogScale::nice((3.5, 850.0), (0.0, 400.0));
        assert!(
            (s.domain_min - 1.0).abs() < 1e-9,
            "domain_min: {}",
            s.domain_min
        );
        assert!(
            (s.domain_max - 1000.0).abs() < 1e-9,
            "domain_max: {}",
            s.domain_max
        );
    }

    #[test]
    fn log_ticks_decades() {
        let ticks = log_ticks(1.0, 1000.0, 5);
        // Use tolerance checks since Miri's soft-float can produce
        // values like 9.999999... instead of exactly 10.0.
        let has_near = |target: f64| ticks.iter().any(|t| (t - target).abs() < target * 1e-9);
        assert!(has_near(1.0), "missing ~1.0 in {ticks:?}");
        assert!(has_near(10.0), "missing ~10.0 in {ticks:?}");
        assert!(has_near(100.0), "missing ~100.0 in {ticks:?}");
        assert!(has_near(1000.0), "missing ~1000.0 in {ticks:?}");
    }

    #[test]
    fn nice_step_2_5_bucket() {
        // Rough step ~2.5 → should pick 2.5, not 5.0
        assert!((nice_step(2.5) - 2.5).abs() < f64::EPSILON);
        assert!((nice_step(3.0) - 2.5).abs() < f64::EPSILON);
        // 3.1 should fall to next bucket (5.0)
        assert!((nice_step(3.1) - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn nice_ticks_include_endpoints() {
        // Domain 3..97: nice ticks should include values near both endpoints
        let ticks = nice_ticks(3.0, 97.0, 5);
        assert!(!ticks.is_empty());
        // lo should be first tick (or inserted)
        assert!(
            *ticks.first().unwrap() <= 3.0 + 1.0,
            "first tick {} > 4.0",
            ticks.first().unwrap()
        );
        // hi should be last tick (or inserted)
        assert!(
            *ticks.last().unwrap() >= 96.0,
            "last tick {} < 96.0",
            ticks.last().unwrap()
        );
    }

    #[test]
    fn log_ticks_wide_range_decade_only() {
        // 5+ decades: should only have decade boundaries
        let ticks = log_ticks(1.0, 1e6, 5);
        // All ticks should be powers of 10
        for t in &ticks {
            let log = t.log10();
            assert!(
                (log - log.round()).abs() < 0.01,
                "tick {t} is not a power of 10 (log10={log})"
            );
        }
    }

    #[test]
    fn log_ticks_narrow_range_has_fine_multipliers() {
        // <2 decades: should include fine multipliers like 3 and 7
        let ticks = log_ticks(10.0, 100.0, 5);
        let has_30_ish = ticks.iter().any(|t| (t - 30.0).abs() < 1.0);
        let has_70_ish = ticks.iter().any(|t| (t - 70.0).abs() < 1.0);
        assert!(has_30_ish, "missing ~30 in narrow log ticks: {ticks:?}");
        assert!(has_70_ish, "missing ~70 in narrow log ticks: {ticks:?}");
    }
}
