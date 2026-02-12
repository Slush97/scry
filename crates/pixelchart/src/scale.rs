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
    domain_min: f64,
    domain_max: f64,
    range_min: f64,
    range_max: f64,
}

impl LinearScale {
    /// Create a new linear scale.
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
    pub fn nice(extent: (f64, f64), range: (f64, f64)) -> Self {
        let (lo, hi) = extent;
        let span = hi - lo;
        let padding = if span.abs() < f64::EPSILON {
            1.0
        } else {
            span * 0.05
        };

        let nice_lo = nice_floor(lo - padding);
        let nice_hi = nice_ceil(hi + padding);

        Self::new((nice_lo, nice_hi), range)
    }

    /// The domain bounds.
    pub fn domain(&self) -> (f64, f64) {
        (self.domain_min, self.domain_max)
    }

    /// The pixel range.
    pub fn range(&self) -> (f64, f64) {
        (self.range_min, self.range_max)
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
        // Adaptive formatting: integers if whole, otherwise reasonable precision
        if (value - value.round()).abs() < f64::EPSILON * 100.0 {
            format!("{}", value as i64)
        } else if value.abs() >= 100.0 {
            format!("{value:.1}")
        } else if value.abs() >= 1.0 {
            format!("{value:.2}")
        } else {
            format!("{value:.3}")
        }
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
    pub fn new(labels: Vec<String>, range: (f64, f64)) -> Self {
        Self {
            labels,
            range_min: range.0,
            range_max: range.1,
        }
    }

    /// The category labels.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Get the center pixel position for a given category index.
    pub fn center(&self, index: usize) -> f64 {
        let n = self.labels.len();
        if n == 0 {
            return (self.range_min + self.range_max) / 2.0;
        }
        let band_width = (self.range_max - self.range_min) / n as f64;
        self.range_min + band_width * (index as f64 + 0.5)
    }

    /// Width of each category band in pixels.
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
    pub fn nice(extent: (f64, f64), range: (f64, f64)) -> Self {
        let lo = extent.0.max(f64::EPSILON);
        let hi = extent.1.max(lo * 10.0);

        // Round to nearest power of 10
        let nice_lo = 10.0_f64.powf(lo.log10().floor());
        let nice_hi = 10.0_f64.powf(hi.log10().ceil());

        Self::new((nice_lo, nice_hi), range)
    }

    /// The domain bounds.
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
fn log_ticks(lo: f64, hi: f64, _target_count: usize) -> Vec<f64> {
    let lo = lo.max(f64::EPSILON);
    let hi = hi.max(lo);

    let log_lo = lo.log10().floor() as i32;
    let log_hi = hi.log10().ceil() as i32;

    let mut ticks = Vec::new();
    for exp in log_lo..=log_hi {
        let base = 10.0_f64.powi(exp);
        // For small ranges, add sub-decade ticks
        for mult in &[1.0, 2.0, 5.0] {
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
// Adaptive tick count
// ---------------------------------------------------------------------------

/// Compute a reasonable number of ticks based on axis pixel length.
///
/// Uses ~60px spacing for X axes and ~40px spacing for Y axes.
pub fn adaptive_tick_count(axis_length_px: f32, is_horizontal: bool) -> usize {
    let spacing = if is_horizontal { 60.0 } else { 40.0 };
    let count = (axis_length_px / spacing).floor() as usize;
    count.clamp(2, 15)
}

// ---------------------------------------------------------------------------
// Nice tick generation
// ---------------------------------------------------------------------------

/// Generate "nice" tick values for a given range and target count.
fn nice_ticks(lo: f64, hi: f64, target_count: usize) -> Vec<f64> {
    let target = target_count.max(2);
    let span = hi - lo;
    if span.abs() < f64::EPSILON {
        return vec![lo];
    }

    let rough_step = span / (target - 1) as f64;
    let step = nice_step(rough_step);

    let start = (lo / step).ceil() * step;
    let mut ticks = Vec::new();
    let mut v = start;
    while v <= hi + step * 0.001 {
        ticks.push(v);
        v += step;
    }

    ticks
}

/// Find a "nice" step size close to the given rough step.
fn nice_step(rough: f64) -> f64 {
    let magnitude = 10.0_f64.powf(rough.abs().log10().floor());
    let fraction = rough / magnitude;

    let nice_fraction = if fraction <= 1.0 {
        1.0
    } else if fraction <= 2.0 {
        2.0
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
    let magnitude = 10.0_f64.powf(v.abs().log10().floor());
    let normalized = v / magnitude;
    (normalized.floor()) * magnitude
}

/// Round up to a "nice" number.
fn nice_ceil(v: f64) -> f64 {
    if v == 0.0 {
        return 0.0;
    }
    let magnitude = 10.0_f64.powf(v.abs().log10().floor());
    let normalized = v / magnitude;
    (normalized.ceil()) * magnitude
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
    fn nice_ticks_output() {
        let ticks = nice_ticks(0.0, 100.0, 5);
        assert!(!ticks.is_empty());
        assert!(ticks[0] >= 0.0);
        assert!(*ticks.last().unwrap() <= 110.0);
    }

    #[test]
    fn categorical_scale_centers() {
        let s = CategoricalScale::new(
            vec!["A".into(), "B".into(), "C".into()],
            (0.0, 300.0),
        );
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
        assert!((back - original).abs() < 0.01, "round trip: {back} != {original}");
    }

    #[test]
    fn log_scale_nice_bounds() {
        let s = LogScale::nice((3.5, 850.0), (0.0, 400.0));
        assert_eq!(s.domain_min, 1.0);
        assert_eq!(s.domain_max, 1000.0);
    }

    #[test]
    fn log_ticks_decades() {
        let ticks = log_ticks(1.0, 1000.0, 5);
        assert!(ticks.contains(&1.0));
        assert!(ticks.contains(&10.0));
        assert!(ticks.contains(&100.0));
        assert!(ticks.contains(&1000.0));
    }

    #[test]
    fn adaptive_tick_count_small() {
        assert_eq!(adaptive_tick_count(100.0, true), 2); // 100/60 = 1.6 → clamped to 2
    }

    #[test]
    fn adaptive_tick_count_large() {
        assert_eq!(adaptive_tick_count(800.0, true), 13); // 800/60 = 13
    }
}
