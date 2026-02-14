//! Time-aware scale for mapping Unix timestamps to pixel coordinates.
//!
//! Provides [`TimeScale`] — wraps a [`LinearScale`] with time-aware tick
//! generation that automatically selects granularity (second → year) and
//! formats labels accordingly.
//!
//! # Example
//!
//! ```ignore
//! use pixelchart::time_scale::TimeScale;
//!
//! // 24 hours of data
//! let now = 1_700_000_000.0;
//! let scale = TimeScale::nice((now, now + 86400.0), (0.0, 800.0));
//! // Tick labels like "00:00", "06:00", "12:00", "18:00"
//! ```

use crate::scale::{LinearScale, Scale};

// ---------------------------------------------------------------------------
// Time granularity
// ---------------------------------------------------------------------------

/// The granularity of time ticks — auto-selected based on the span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeGranularity {
    /// 1-second ticks (span < 2 minutes)
    Seconds,
    /// 1-minute ticks (span < 2 hours)
    Minutes,
    /// 1-hour ticks (span < 3 days)
    Hours,
    /// 1-day ticks (span < 90 days)
    Days,
    /// 1-month ticks (span < 3 years)
    Months,
    /// 1-year ticks (span ≥ 3 years)
    Years,
}

impl TimeGranularity {
    /// Select the best granularity for a given time span in seconds.
    #[must_use]
    pub fn from_span(span_secs: f64) -> Self {
        let span = span_secs.abs();
        if span < 120.0 {
            Self::Seconds
        } else if span < 7_200.0 {
            Self::Minutes
        } else if span < 259_200.0 {
            Self::Hours
        } else if span < 7_776_000.0 {
            Self::Days
        } else if span < 94_608_000.0 {
            Self::Months
        } else {
            Self::Years
        }
    }

    /// The "natural" step size in seconds for this granularity.
    fn base_step(self) -> f64 {
        match self {
            Self::Seconds => 1.0,
            Self::Minutes => 60.0,
            Self::Hours => 3_600.0,
            Self::Days => 86_400.0,
            Self::Months => 2_592_000.0, // ~30 days
            Self::Years => 31_536_000.0, // 365 days
        }
    }

    /// Select a nice step size for this granularity targeting ~`target_ticks`.
    fn nice_step(self, span_secs: f64, target_ticks: usize) -> f64 {
        let base = self.base_step();
        let target = target_ticks.max(2) as f64;
        let rough_mult = span_secs / (base * target);

        // Pick from human-friendly multipliers
        let multipliers: &[f64] = match self {
            Self::Seconds | Self::Minutes => &[1.0, 2.0, 5.0, 10.0, 15.0, 30.0],
            Self::Hours => &[1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0],
            Self::Days => &[1.0, 2.0, 7.0, 14.0, 28.0],
            Self::Months => &[1.0, 2.0, 3.0, 6.0, 12.0],
            Self::Years => &[1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
        };

        let mut best = multipliers[0];
        for &m in multipliers {
            if m >= rough_mult {
                best = m;
                break;
            }
            best = m;
        }

        base * best
    }
}

// ---------------------------------------------------------------------------
// TimeScale
// ---------------------------------------------------------------------------

/// A time-aware axis scale that maps Unix epoch seconds to pixel coordinates.
///
/// Automatically selects tick granularity and formatting based on the span.
#[derive(Clone, Debug)]
pub struct TimeScale {
    inner: LinearScale,
    granularity: TimeGranularity,
}

impl TimeScale {
    /// Create a new time scale.
    ///
    /// `domain` is `(epoch_start, epoch_end)` in Unix seconds.
    /// `range` is `(pixel_start, pixel_end)`.
    #[must_use]
    pub fn new(domain: (f64, f64), range: (f64, f64)) -> Self {
        let span = (domain.1 - domain.0).abs();
        let granularity = TimeGranularity::from_span(span);
        Self {
            inner: LinearScale::new(domain, range),
            granularity,
        }
    }

    /// Create a time scale with nice domain bounds.
    #[must_use]
    pub fn nice(extent: (f64, f64), range: (f64, f64)) -> Self {
        let span = (extent.1 - extent.0).abs();
        let granularity = TimeGranularity::from_span(span);
        // Round domain to nice time boundaries
        let step = granularity.base_step();
        let nice_lo = (extent.0 / step).floor() * step;
        let nice_hi = (extent.1 / step).ceil() * step;
        Self {
            inner: LinearScale::new((nice_lo, nice_hi), range),
            granularity,
        }
    }

    /// The auto-detected granularity.
    #[must_use]
    pub const fn granularity(&self) -> TimeGranularity {
        self.granularity
    }

    /// Generate nice tick positions for this time scale.
    #[must_use]
    pub fn time_ticks(&self, target_count: usize) -> Vec<f64> {
        let (lo, hi) = self.inner.domain();
        let span = hi - lo;
        let step = self.granularity.nice_step(span, target_count);

        if step <= 0.0 || !step.is_finite() {
            return vec![lo, hi];
        }

        let start = (lo / step).ceil() * step;
        let mut ticks = Vec::new();
        let mut v = start;
        let max_ticks = (target_count * 3).max(20);

        while v <= hi + step * 0.001 && ticks.len() < max_ticks {
            ticks.push(v);
            let next = v + step;
            if next <= v {
                break;
            }
            v = next;
        }

        ticks
    }

    /// Create a [`DateTimeFormatter`](crate::formatter::DateTimeFormatter) matching this scale.
    #[must_use]
    pub fn formatter(&self) -> crate::formatter::DateTimeFormatter {
        crate::formatter::DateTimeFormatter
    }

    /// Convert to the inner linear scale for axis drawing.
    #[must_use]
    pub const fn as_linear(&self) -> &LinearScale {
        &self.inner
    }
}

impl Scale for TimeScale {
    fn to_pixel(&self, value: f64) -> f64 {
        self.inner.to_pixel(value)
    }

    fn to_data(&self, pixel: f64) -> f64 {
        self.inner.to_data(pixel)
    }

    fn ticks(&self, target_count: usize) -> Vec<f64> {
        self.time_ticks(target_count)
    }

    fn format_tick(&self, value: f64) -> String {
        use crate::formatter::TickFormatter;
        let domain = self.inner.domain();
        let formatter = crate::formatter::DateTimeFormatter;
        formatter
            .format_batch(&[value], domain)
            .into_iter()
            .next()
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Calendar helpers (used by tests)
// ---------------------------------------------------------------------------

/// Convert Unix epoch seconds to calendar parts (year, month, day, hour, min, sec).
///
/// Simplified civil calendar computation (Gregorian, UTC).
#[allow(dead_code)]
fn epoch_to_parts(epoch: i64) -> (i32, u32, u32, u32, u32, u32) {
    let secs_per_day: i64 = 86_400;
    let total_days = epoch.div_euclid(secs_per_day);
    let day_seconds = epoch.rem_euclid(secs_per_day);

    let hour = (day_seconds / 3600) as u32;
    let minute = ((day_seconds % 3600) / 60) as u32;
    let second = (day_seconds % 60) as u32;

    let z = total_days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    (y as i32, m, d, hour, minute, second)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formatter::TickFormatter;

    #[test]
    fn granularity_from_span() {
        assert_eq!(TimeGranularity::from_span(30.0), TimeGranularity::Seconds);
        assert_eq!(TimeGranularity::from_span(300.0), TimeGranularity::Minutes);
        assert_eq!(TimeGranularity::from_span(7200.0), TimeGranularity::Hours);
        assert_eq!(TimeGranularity::from_span(86400.0), TimeGranularity::Hours);
        assert_eq!(TimeGranularity::from_span(500_000.0), TimeGranularity::Days);
        assert_eq!(
            TimeGranularity::from_span(8_000_000.0),
            TimeGranularity::Months
        );
        assert_eq!(
            TimeGranularity::from_span(200_000_000.0),
            TimeGranularity::Years
        );
    }

    #[test]
    fn time_scale_ticks_hourly() {
        let now = 1_700_000_000.0;
        let scale = TimeScale::nice((now, now + 86400.0), (0.0, 800.0));
        assert_eq!(scale.granularity(), TimeGranularity::Hours);

        let ticks = scale.time_ticks(8);
        assert!(!ticks.is_empty());
        for t in &ticks {
            assert_eq!(*t as i64 % 3600, 0, "tick {} is not on an hour boundary", t);
        }
    }

    #[test]
    fn time_scale_ticks_daily() {
        let start = 1_700_000_000.0;
        let scale = TimeScale::nice((start, start + 30.0 * 86400.0), (0.0, 800.0));
        assert_eq!(scale.granularity(), TimeGranularity::Days);
        let ticks = scale.time_ticks(8);
        assert!(!ticks.is_empty());
    }

    #[test]
    fn time_scale_ticks_yearly() {
        let start = 1_500_000_000.0;
        let end = start + 5.0 * 365.25 * 86400.0;
        let scale = TimeScale::nice((start, end), (0.0, 800.0));
        assert_eq!(scale.granularity(), TimeGranularity::Years);
        let ticks = scale.time_ticks(6);
        assert!(!ticks.is_empty());
    }

    #[test]
    fn time_format_via_formatter() {
        let formatter = crate::formatter::DateTimeFormatter;
        let labels = formatter.format_batch(&[1_699_920_045.0], (1_699_920_000.0, 1_699_923_600.0));
        assert_eq!(labels.len(), 1);
        assert!(
            labels[0].contains(':'),
            "should contain colon: {}",
            labels[0]
        );
    }

    #[test]
    fn time_format_hours_via_formatter() {
        let formatter = crate::formatter::DateTimeFormatter;
        let t0 = 1_699_920_000.0;
        let labels = formatter.format_batch(&[t0], (t0, t0 + 43200.0));
        assert_eq!(labels.len(), 1);
        assert!(
            labels[0].contains(':'),
            "hourly format should contain colon, got {}",
            labels[0]
        );
    }

    #[test]
    fn time_format_days_via_formatter() {
        let formatter = crate::formatter::DateTimeFormatter;
        let t0 = 1_699_920_000.0;
        let labels = formatter.format_batch(&[t0], (t0, t0 + 30.0 * 86400.0));
        assert_eq!(labels.len(), 1);
        assert!(
            labels[0].starts_with("Nov") || labels[0].starts_with("Dec"),
            "daily format should start with month abbrev, got {}",
            labels[0]
        );
    }

    #[test]
    fn time_format_years_via_formatter() {
        let formatter = crate::formatter::DateTimeFormatter;
        let t0 = 1_609_459_200.0;
        let labels = formatter.format_batch(&[t0], (t0, t0 + 5.0 * 365.25 * 86400.0));
        assert_eq!(labels.len(), 1);
        assert!(
            labels[0].contains("2021"),
            "should contain year, got {}",
            labels[0]
        );
    }

    #[test]
    fn time_scale_nice_rounding() {
        let start = 1_700_000_123.0;
        let end = start + 3600.0;
        let scale = TimeScale::nice((start, end), (0.0, 400.0));
        let (lo, _hi) = scale.inner.domain();
        assert!((lo - lo.floor()).abs() < 0.001);
    }

    #[test]
    fn time_scale_implements_scale_trait() {
        let scale = TimeScale::new((0.0, 100.0), (0.0, 200.0));
        let px = scale.to_pixel(50.0);
        assert!((px - 100.0).abs() < 0.001);
        let data = scale.to_data(100.0);
        assert!((data - 50.0).abs() < 0.001);
    }

    #[test]
    fn epoch_to_parts_known_date() {
        let (y, m, d, h, min, s) = epoch_to_parts(1_609_459_200);
        assert_eq!((y, m, d, h, min, s), (2021, 1, 1, 0, 0, 0));
    }

    #[test]
    fn epoch_to_parts_epoch_zero() {
        let (y, m, d, h, min, s) = epoch_to_parts(0);
        assert_eq!((y, m, d, h, min, s), (1970, 1, 1, 0, 0, 0));
    }
}
