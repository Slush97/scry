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
pub(crate) fn log_ticks(lo: f64, hi: f64, _target_count: usize) -> Vec<f64> {
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

    // Safety: filter any non-finite values that slipped through
    // floating-point arithmetic edge cases.
    ticks.retain(|v| v.is_finite());

    ticks
}

/// Find a "nice" step size close to the given rough step.
pub(crate) fn nice_step(rough: f64) -> f64 {
    // Guard against zero, near-zero, NaN, or Infinity
    if rough.abs() < f64::EPSILON * 100.0 || !rough.is_finite() {
        return 1.0;
    }
    let magnitude = 10.0_f64.powf(rough.abs().log10().floor());
    // Always use absolute value for bucketing so negative inputs
    // (e.g. inverted axes) produce correct step sizes.
    let fraction = rough.abs() / magnitude;

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
    // Pass through non-finite values (NaN, ±Infinity) rather than
    // producing garbage from log10 arithmetic.
    if !v.is_finite() {
        return v;
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
    // Pass through non-finite values (NaN, ±Infinity).
    if !v.is_finite() {
        return v;
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
    // Guard: non-finite values get a human-readable representation
    // rather than propagating garbage through downstream arithmetic.
    if !value.is_finite() {
        return if value.is_nan() {
            "NaN".to_string()
        } else if value.is_sign_positive() {
            "∞".to_string()
        } else {
            "-∞".to_string()
        };
    }
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
        if abs >= 1e15 {
            let v = value / 1e15;
            return if (v - v.round()).abs() < 0.05 && v.round().abs() <= i64::MAX as f64 {
                format!("{}P", v.round() as i64)
            } else {
                format!("{v:.1}P")
            };
        }
        if abs >= 1e12 {
            let v = value / 1e12;
            return if (v - v.round()).abs() < 0.05 && v.round().abs() <= i64::MAX as f64 {
                format!("{}T", v.round() as i64)
            } else {
                format!("{v:.1}T")
            };
        }
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
        // Count decimals needed to represent the step.
        // Clamp to 0 before casting to avoid wrapping on negative log values.
        (-step.log10().floor()).max(0.0) as usize
    };

    // If we need more than 6 decimal places, switch to scientific notation.
    // This handles sub-micro values like 1e-7, 2.5e-8, etc.
    if decimals > 6 {
        // Use engineering-style: show significant digits relative to step
        let sig_digits = if step < 1e-15 {
            6
        } else {
            let raw_step_d = (-step.log10().floor()).max(0.0) as usize;
            let raw_abs_d = (-abs.max(1e-300).log10().floor()).max(0.0) as usize;
            let step_digits = raw_step_d.saturating_sub(raw_abs_d);
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
// SymlogScale — symmetric logarithmic
// ---------------------------------------------------------------------------

/// A symmetric logarithmic mapping that handles zero-crossing data.
///
/// Uses the transform `symlog(x) = sign(x) × log₁₀(1 + |x| / threshold)`,
/// which transitions smoothly from linear near zero to logarithmic at large
/// magnitudes. This makes it ideal for data with both positive and negative
/// values spanning many orders of magnitude (e.g., financial PnL, temperature
/// anomalies, seismic data).
///
/// Equivalent to matplotlib's `SymLogNorm` / D3's `scaleSymlog`.
///
/// # Parameters
///
/// - `linear_threshold` (default `1.0`): Controls where the linear-to-log
///   transition occurs. Values within `[-threshold, threshold]` appear
///   roughly linear; values beyond are compressed logarithmically.
#[derive(Clone, Debug)]
pub struct SymlogScale {
    domain_min: f64,
    domain_max: f64,
    range_min: f64,
    range_max: f64,
    /// The linear-to-logarithmic transition threshold.
    linear_threshold: f64,
}

impl SymlogScale {
    /// Create a new symlog scale with the given domain, range, and linear
    /// threshold.
    ///
    /// `threshold` must be positive and finite; invalid values default to 1.0.
    #[must_use]
    pub fn new(domain: (f64, f64), range: (f64, f64), threshold: f64) -> Self {
        Self {
            domain_min: domain.0,
            domain_max: domain.1,
            range_min: range.0,
            range_max: range.1,
            linear_threshold: if threshold > 0.0 && threshold.is_finite() {
                threshold
            } else {
                1.0
            },
        }
    }

    /// Create a symlog scale with the default threshold of 1.0.
    #[must_use]
    pub fn with_default_threshold(domain: (f64, f64), range: (f64, f64)) -> Self {
        Self::new(domain, range, 1.0)
    }

    /// Create a symlog scale with nice domain bounds.
    #[must_use]
    pub fn nice(extent: (f64, f64), range: (f64, f64), threshold: f64) -> Self {
        let thresh = if threshold > 0.0 && threshold.is_finite() {
            threshold
        } else {
            1.0
        };
        let (lo, hi) = extent;

        // For degenerate domains, use linear nice rounding
        if !lo.is_finite() || !hi.is_finite() || (hi - lo).abs() < f64::EPSILON {
            let lin = LinearScale::nice(extent, range);
            return Self::new(lin.domain(), range, thresh);
        }

        // Nice-round each side independently in symlog space
        let nice_lo = symlog_nice_bound(lo, thresh, false);
        let nice_hi = symlog_nice_bound(hi, thresh, true);

        Self::new((nice_lo, nice_hi), range, thresh)
    }

    /// The domain bounds.
    #[must_use]
    pub fn domain(&self) -> (f64, f64) {
        (self.domain_min, self.domain_max)
    }

    /// The linear threshold.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.linear_threshold
    }
}

/// The symlog transform: `sign(x) × log₁₀(1 + |x| / threshold)`.
fn symlog_transform(x: f64, threshold: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    x.signum() * (1.0 + x.abs() / threshold).log10()
}

/// The inverse symlog transform.
fn symlog_inverse(y: f64, threshold: f64) -> f64 {
    if !y.is_finite() {
        return y;
    }
    y.signum() * threshold * (10.0_f64.powf(y.abs()) - 1.0)
}

/// Round a value to a "nice" boundary in symlog space.
fn symlog_nice_bound(val: f64, threshold: f64, is_upper: bool) -> f64 {
    let abs = val.abs();

    // Within the linear region — snap to 0 or ±threshold
    if abs <= threshold {
        if is_upper {
            return if val >= 0.0 { threshold } else { 0.0 };
        }
        return if val <= 0.0 { -threshold } else { 0.0 };
    }

    // In the log region — round to a nice power of 10
    let magnitude = 10.0_f64.powf(abs.log10().floor());
    let nice = if is_upper {
        if val > 0.0 {
            (val / magnitude).ceil() * magnitude
        } else {
            -(((-val) / magnitude).floor() * magnitude)
        }
    } else if val < 0.0 {
        -(((-val) / magnitude).ceil() * magnitude)
    } else {
        (val / magnitude).floor() * magnitude
    };
    nice
}

impl Scale for SymlogScale {
    fn to_pixel(&self, value: f64) -> f64 {
        let s_lo = symlog_transform(self.domain_min, self.linear_threshold);
        let s_hi = symlog_transform(self.domain_max, self.linear_threshold);
        let s_span = s_hi - s_lo;
        if s_span.abs() < f64::EPSILON {
            return (self.range_min + self.range_max) / 2.0;
        }
        let s_val = symlog_transform(value, self.linear_threshold);
        let t = (s_val - s_lo) / s_span;
        self.range_min + t * (self.range_max - self.range_min)
    }

    fn to_data(&self, pixel: f64) -> f64 {
        let range_span = self.range_max - self.range_min;
        if range_span.abs() < f64::EPSILON {
            return (self.domain_min + self.domain_max) / 2.0;
        }
        let t = (pixel - self.range_min) / range_span;
        let s_lo = symlog_transform(self.domain_min, self.linear_threshold);
        let s_hi = symlog_transform(self.domain_max, self.linear_threshold);
        let s_val = s_lo + t * (s_hi - s_lo);
        symlog_inverse(s_val, self.linear_threshold)
    }

    fn ticks(&self, target_count: usize) -> Vec<f64> {
        symlog_ticks(
            self.domain_min,
            self.domain_max,
            self.linear_threshold,
            target_count,
        )
    }

    fn format_tick(&self, value: f64) -> String {
        format_tick_adaptive(value, self.domain_min, self.domain_max)
    }
}

/// Generate tick values for a symlog-scale axis.
///
/// Places ticks symmetrically around zero at:
/// `0, ±threshold, ±2t, ±5t, ±10t, ±20t, ±50t, ±100t, ...`
pub(crate) fn symlog_ticks(lo: f64, hi: f64, threshold: f64, target_count: usize) -> Vec<f64> {
    let target = target_count.max(3);

    if !lo.is_finite() || !hi.is_finite() {
        return vec![0.0];
    }

    let mut ticks: Vec<f64> = Vec::new();

    // Always include 0 if domain crosses zero
    let crosses_zero = lo <= 0.0 && hi >= 0.0;
    if crosses_zero {
        ticks.push(0.0);
    }

    // Generate positive ticks
    let nice_mults: &[f64] = &[1.0, 2.0, 5.0];
    if hi > 0.0 {
        let start = if lo > 0.0 { lo } else { threshold };
        let mag_start = if start <= threshold {
            threshold.log10().floor() as i32
        } else {
            start.log10().floor() as i32
        };
        let mag_end = if hi > threshold {
            hi.log10().ceil() as i32
        } else {
            mag_start + 1
        };

        for exp in mag_start..=mag_end {
            for &m in nice_mults {
                let v = m * 10.0_f64.powi(exp);
                if v >= lo * 0.99 && v <= hi * 1.01 && v > 0.0 {
                    ticks.push(v);
                }
            }
        }
    }

    // Generate negative ticks (mirror of positive)
    if lo < 0.0 {
        let neg_hi = hi.min(0.0).abs().max(threshold);
        let neg_lo = lo.abs();
        let mag_start = if neg_hi <= threshold {
            threshold.log10().floor() as i32
        } else {
            neg_hi.log10().floor() as i32
        };
        let mag_end = neg_lo.log10().ceil() as i32;

        for exp in mag_start..=mag_end {
            for &m in nice_mults {
                let v = -(m * 10.0_f64.powi(exp));
                if v >= lo * 1.01 && v <= hi.max(0.0) * 1.01 && v < 0.0 {
                    ticks.push(v);
                }
            }
        }
    }

    // Sort and deduplicate
    ticks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ticks.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON * 100.0);

    // Trim to target count if way too many
    while ticks.len() > target * 2 && ticks.len() > 3 {
        // Remove every other interior tick
        let mut i = 1;
        while i < ticks.len() - 1 && ticks.len() > target * 2 {
            ticks.remove(i);
            i += 1; // skip one, remove next
        }
    }

    if ticks.is_empty() {
        ticks.push(lo);
        ticks.push(hi);
    }

    ticks
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

    // -----------------------------------------------------------------------
    // Edge case tests (Phase 1 hardening)
    // -----------------------------------------------------------------------

    #[test]
    fn nice_step_negative_input() {
        // Negative rough step (e.g., inverted axis) should produce a positive step
        let step = nice_step(-5.0);
        assert!(step > 0.0, "nice_step(-5.0) = {step}");
        assert!(step.is_finite());
        assert!((step - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn nice_step_nan() {
        let step = nice_step(f64::NAN);
        assert!(step.is_finite(), "nice_step(NaN) = {step}");
        assert!(step > 0.0);
    }

    #[test]
    fn nice_step_infinity() {
        let step = nice_step(f64::INFINITY);
        assert!(step.is_finite(), "nice_step(Inf) = {step}");
        assert!(step > 0.0);
    }

    #[test]
    fn nice_floor_nan() {
        let v = nice_floor(f64::NAN);
        assert!(v.is_nan(), "nice_floor(NaN) should be NaN, got {v}");
    }

    #[test]
    fn nice_floor_infinity() {
        assert_eq!(nice_floor(f64::INFINITY), f64::INFINITY);
        assert_eq!(nice_floor(f64::NEG_INFINITY), f64::NEG_INFINITY);
    }

    #[test]
    fn nice_ceil_nan() {
        let v = nice_ceil(f64::NAN);
        assert!(v.is_nan(), "nice_ceil(NaN) should be NaN, got {v}");
    }

    #[test]
    fn nice_ceil_infinity() {
        assert_eq!(nice_ceil(f64::INFINITY), f64::INFINITY);
        assert_eq!(nice_ceil(f64::NEG_INFINITY), f64::NEG_INFINITY);
    }

    #[test]
    fn nice_ticks_nan_domain() {
        let ticks = nice_ticks(f64::NAN, f64::NAN, 5);
        assert!(!ticks.is_empty());
        // All output ticks must be finite
        for t in &ticks {
            assert!(t.is_finite(), "non-finite tick in NaN domain: {t}");
        }
    }

    #[test]
    fn nice_ticks_inf_domain() {
        let ticks = nice_ticks(f64::NEG_INFINITY, f64::INFINITY, 5);
        assert!(!ticks.is_empty());
        for t in &ticks {
            assert!(t.is_finite(), "non-finite tick in Inf domain: {t}");
        }
    }

    #[test]
    fn nice_ticks_huge_values() {
        // Near i64::MAX — should not panic from overflow
        let ticks = nice_ticks(1e18, 1e18 + 100.0, 5);
        assert!(!ticks.is_empty());
        for t in &ticks {
            assert!(t.is_finite(), "non-finite tick in huge domain: {t}");
        }
    }

    #[test]
    fn nice_ticks_micro_range() {
        let ticks = nice_ticks(0.001, 0.002, 5);
        assert!(!ticks.is_empty());
        assert!(*ticks.first().unwrap() >= 0.0009);
        assert!(*ticks.last().unwrap() <= 0.0025);
    }

    #[test]
    fn format_tick_nan() {
        let s = format_tick_adaptive(f64::NAN, 0.0, 100.0);
        assert_eq!(s, "NaN");
    }

    #[test]
    fn format_tick_infinity() {
        let s = format_tick_adaptive(f64::INFINITY, 0.0, 100.0);
        assert_eq!(s, "∞");
        let s = format_tick_adaptive(f64::NEG_INFINITY, 0.0, 100.0);
        assert_eq!(s, "-∞");
    }

    #[test]
    fn format_tick_huge_negative() {
        // Should not panic from i64 overflow
        let s = format_tick_adaptive(-1e19, -2e19, 0.0);
        assert!(!s.is_empty());
        assert!(!s.contains("NaN"));
    }

    #[test]
    fn format_tick_trillion() {
        let s = format_tick_adaptive(1e12, 0.0, 2e12);
        assert_eq!(s, "1T");
        let s = format_tick_adaptive(2.5e12, 0.0, 5e12);
        assert_eq!(s, "2.5T");
    }

    #[test]
    fn format_tick_peta() {
        let s = format_tick_adaptive(1e15, 0.0, 2e15);
        assert_eq!(s, "1P");
        let s = format_tick_adaptive(2.5e15, 0.0, 5e15);
        assert_eq!(s, "2.5P");
    }

    #[test]
    fn format_tick_100_trillion() {
        // 1e14 = 100T (below P threshold)
        let s = format_tick_adaptive(1e14, 0.0, 2e14);
        assert_eq!(s, "100T");
    }

    #[test]
    fn format_tick_subnormal() {
        // Smallest positive subnormal float — should not panic
        let s = format_tick_adaptive(5e-324, 0.0, 1e-320);
        assert!(!s.is_empty());
    }

    // -----------------------------------------------------------------------
    // SymlogScale tests
    // -----------------------------------------------------------------------

    #[test]
    fn symlog_transform_zero() {
        assert!((symlog_transform(0.0, 1.0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn symlog_transform_positive() {
        // symlog(10, 1) = sign(10) * log10(1 + 10/1) = log10(11) ≈ 1.041
        let v = symlog_transform(10.0, 1.0);
        assert!((v - 11.0_f64.log10()).abs() < 1e-10);
    }

    #[test]
    fn symlog_transform_negative() {
        // symlog(-10, 1) = -log10(11) ≈ -1.041
        let v = symlog_transform(-10.0, 1.0);
        assert!((v + 11.0_f64.log10()).abs() < 1e-10);
    }

    #[test]
    fn symlog_round_trip() {
        let threshold = 1.0;
        for &val in &[-1000.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 1000.0] {
            let transformed = symlog_transform(val, threshold);
            let back = symlog_inverse(transformed, threshold);
            assert!(
                (back - val).abs() < 1e-8,
                "round trip failed for {val}: got {back}"
            );
        }
    }

    #[test]
    fn symlog_scale_maps_zero() {
        // Domain [-100, 100], zero should map to the center of pixel range
        let s = SymlogScale::with_default_threshold((-100.0, 100.0), (0.0, 400.0));
        let px = s.to_pixel(0.0);
        assert!(
            (px - 200.0).abs() < 1.0,
            "zero should map near center, got {px}"
        );
    }

    #[test]
    fn symlog_scale_round_trip() {
        let s = SymlogScale::new((-1000.0, 1000.0), (0.0, 800.0), 1.0);
        for &val in &[-500.0, -1.0, 0.0, 1.0, 500.0] {
            let px = s.to_pixel(val);
            let back = s.to_data(px);
            assert!(
                (back - val).abs() < 0.5,
                "round trip: {val} → px={px} → {back}"
            );
        }
    }

    #[test]
    fn symlog_ticks_cross_zero() {
        let ticks = symlog_ticks(-1000.0, 1000.0, 1.0, 7);
        assert!(!ticks.is_empty());
        // Must include 0
        assert!(
            ticks.iter().any(|t| t.abs() < f64::EPSILON),
            "symlog ticks should include 0: {ticks:?}"
        );
        // Should have both positive and negative ticks
        assert!(ticks.iter().any(|t| *t > 0.0), "should have positive ticks");
        assert!(ticks.iter().any(|t| *t < 0.0), "should have negative ticks");
    }

    #[test]
    fn symlog_ticks_positive_only() {
        let ticks = symlog_ticks(1.0, 10000.0, 1.0, 5);
        assert!(!ticks.is_empty());
        for t in &ticks {
            assert!(*t > 0.0, "positive-only domain should have no negative ticks");
        }
    }

    #[test]
    fn symlog_nice_bounds() {
        let s = SymlogScale::nice((-73.0, 850.0), (0.0, 400.0), 1.0);
        let (lo, hi) = s.domain();
        assert!(lo <= -73.0, "nice lo {lo} should be <= -73");
        assert!(hi >= 850.0, "nice hi {hi} should be >= 850");
    }

    #[test]
    fn symlog_scale_degenerate_domain() {
        // Same value for lo and hi
        let s = SymlogScale::nice((5.0, 5.0), (0.0, 400.0), 1.0);
        let px = s.to_pixel(5.0);
        assert!(px.is_finite(), "degenerate domain should produce finite pixel");
    }

    #[test]
    fn symlog_ticks_nan_domain() {
        let ticks = symlog_ticks(f64::NAN, f64::NAN, 1.0, 5);
        assert!(!ticks.is_empty());
        for t in &ticks {
            assert!(t.is_finite(), "symlog ticks should be finite even for NaN domain");
        }
    }

    #[test]
    fn symlog_scale_invalid_threshold() {
        // Negative threshold should default to 1.0
        let s = SymlogScale::new((-100.0, 100.0), (0.0, 400.0), -5.0);
        assert!((s.threshold() - 1.0).abs() < f64::EPSILON);
    }
}
