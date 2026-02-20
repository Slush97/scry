// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tick locator system — determines *where* ticks are placed on an axis.
//!
//! This separates tick *location* from tick *formatting* (handled by
//! [`TickFormatter`](crate::formatter::TickFormatter)), following the
//! architecture used by matplotlib (`Locator` vs `Formatter`) and D3
//! (`scale.ticks()` vs `scale.tickFormat()`).
//!
//! # Built-in Locators
//!
//! | Locator | Description |
//! |---------|-------------|
//! | [`AutoLocator`] | Nice-number algorithm (default) |
//! | [`MaxNLocator`] | Strict maximum N ticks with nice numbers |
//! | [`MultipleLocator`] | Ticks at multiples of a base value |
//! | [`FixedLocator`] | User-specified tick positions |
//! | [`LogLocator`] | Logarithmic (base-10) tick placement |

use std::sync::Arc;

use crate::scale::{log_ticks, nice_ticks, symlog_ticks};

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Determines tick positions for a given data domain.
///
/// Implementations receive the axis domain `(min, max)` and a target tick
/// count (a *suggestion*, not a hard limit). They return a sorted `Vec<f64>`
/// of data-space positions where ticks should be placed.
pub trait TickLocator: Send + Sync + std::fmt::Debug {
    /// Compute tick positions for the given domain.
    ///
    /// `target_count` is a suggestion — implementations may return fewer
    /// or more ticks depending on the algorithm.
    fn tick_values(&self, domain: (f64, f64), target_count: usize) -> Vec<f64>;
}

/// Erase a `TickLocator` into a shared trait object.
pub fn boxed_locator(l: impl TickLocator + 'static) -> Arc<dyn TickLocator> {
    Arc::new(l)
}

// ---------------------------------------------------------------------------
// AutoLocator — default, wraps nice_ticks()
// ---------------------------------------------------------------------------

/// Automatic tick placement using the "nice numbers" algorithm.
///
/// This is the default locator, equivalent to matplotlib's `AutoLocator`
/// or D3's `scale.ticks()`. It selects tick positions at round multiples
/// of 1, 2, 2.5, 5, or 10 × 10^n.
#[derive(Clone, Debug, Default)]
pub struct AutoLocator;

impl TickLocator for AutoLocator {
    fn tick_values(&self, domain: (f64, f64), target_count: usize) -> Vec<f64> {
        nice_ticks(domain.0, domain.1, target_count)
    }
}

// ---------------------------------------------------------------------------
// MaxNLocator — strict max with nice numbers
// ---------------------------------------------------------------------------

/// Places at most `max_n` ticks using nice numbers.
///
/// Unlike [`AutoLocator`] which may exceed the target count when inserting
/// domain endpoints, `MaxNLocator` strictly enforces a maximum tick count
/// by pruning excess ticks from the interior.
///
/// # Example
///
/// ```
/// use scry_chart::locator::{MaxNLocator, TickLocator};
///
/// let loc = MaxNLocator::new(5);
/// let ticks = loc.tick_values((0.0, 100.0), 10);
/// assert!(ticks.len() <= 5);
/// ```
#[derive(Clone, Debug)]
pub struct MaxNLocator {
    max_n: usize,
}

impl MaxNLocator {
    /// Create a locator with a strict maximum tick count.
    #[must_use]
    pub fn new(max_n: usize) -> Self {
        Self { max_n: max_n.max(2) }
    }
}

impl TickLocator for MaxNLocator {
    fn tick_values(&self, domain: (f64, f64), _target_count: usize) -> Vec<f64> {
        let mut ticks = nice_ticks(domain.0, domain.1, self.max_n);
        // Prune from the interior if we have too many, keeping first and last
        while ticks.len() > self.max_n && ticks.len() > 2 {
            // Remove the tick closest to the middle of the array
            let mid = ticks.len() / 2;
            ticks.remove(mid);
        }
        ticks
    }
}

// ---------------------------------------------------------------------------
// MultipleLocator — fixed interval
// ---------------------------------------------------------------------------

/// Places ticks at multiples of a base value.
///
/// Equivalent to matplotlib's `MultipleLocator(base)`. Ticks are placed at
/// `ceil(domain_min / base) * base`, incremented by `base`.
///
/// # Example
///
/// ```
/// use scry_chart::locator::{MultipleLocator, TickLocator};
///
/// let loc = MultipleLocator::new(25.0);
/// let ticks = loc.tick_values((0.0, 100.0), 5);
/// assert_eq!(ticks, vec![0.0, 25.0, 50.0, 75.0, 100.0]);
/// ```
#[derive(Clone, Debug)]
pub struct MultipleLocator {
    base: f64,
}

impl MultipleLocator {
    /// Create a locator with ticks at multiples of `base`.
    ///
    /// `base` must be positive and finite; invalid values default to 1.0.
    #[must_use]
    pub fn new(base: f64) -> Self {
        Self {
            base: if base > 0.0 && base.is_finite() {
                base
            } else {
                1.0
            },
        }
    }
}

impl TickLocator for MultipleLocator {
    fn tick_values(&self, domain: (f64, f64), _target_count: usize) -> Vec<f64> {
        let (lo, hi) = domain;
        if !lo.is_finite() || !hi.is_finite() {
            return vec![0.0];
        }

        let start = (lo / self.base).ceil() * self.base;
        let mut ticks = Vec::new();
        let mut i = 0usize;
        loop {
            let val = start + i as f64 * self.base;
            if val > hi + self.base * 0.01 || i >= 200 {
                break;
            }
            ticks.push(val);
            i += 1;
        }
        if ticks.is_empty() {
            ticks.push(lo);
        }
        ticks
    }
}

// ---------------------------------------------------------------------------
// FixedLocator — user-specified positions
// ---------------------------------------------------------------------------

/// Places ticks at user-specified positions.
///
/// Equivalent to matplotlib's `FixedLocator`. Ignores the target count
/// and domain entirely — ticks appear exactly where specified.
///
/// # Example
///
/// ```
/// use scry_chart::locator::{FixedLocator, TickLocator};
///
/// let loc = FixedLocator::new(vec![0.0, 33.3, 66.6, 100.0]);
/// let ticks = loc.tick_values((0.0, 100.0), 5);
/// assert_eq!(ticks, vec![0.0, 33.3, 66.6, 100.0]);
/// ```
#[derive(Clone, Debug)]
pub struct FixedLocator {
    positions: Vec<f64>,
}

impl FixedLocator {
    /// Create a locator with fixed tick positions.
    #[must_use]
    pub fn new(positions: Vec<f64>) -> Self {
        Self { positions }
    }
}

impl TickLocator for FixedLocator {
    fn tick_values(&self, _domain: (f64, f64), _target_count: usize) -> Vec<f64> {
        self.positions.clone()
    }
}

// ---------------------------------------------------------------------------
// LogLocator — logarithmic tick placement
// ---------------------------------------------------------------------------

/// Logarithmic (base-10) tick placement.
///
/// Equivalent to matplotlib's `LogLocator`. Adapts sub-decade multipliers
/// based on the number of decades spanned.
#[derive(Clone, Debug, Default)]
pub struct LogLocator;

impl TickLocator for LogLocator {
    fn tick_values(&self, domain: (f64, f64), target_count: usize) -> Vec<f64> {
        log_ticks(domain.0, domain.1, target_count)
    }
}

// ---------------------------------------------------------------------------
// SymlogLocator — symmetric logarithmic tick placement
// ---------------------------------------------------------------------------

/// Symmetric logarithmic tick placement for zero-crossing data.
///
/// Equivalent to matplotlib's `SymmetricalLogLocator`. Places ticks at
/// nice values in symlog space: `0, ±1, ±2, ±5, ±10, ±20, ±50, ...`
///
/// # Example
///
/// ```
/// use scry_chart::locator::{SymlogLocator, TickLocator};
///
/// let loc = SymlogLocator::new(1.0);
/// let ticks = loc.tick_values((-1000.0, 1000.0), 7);
/// assert!(ticks.iter().any(|t| t.abs() < f64::EPSILON)); // includes 0
/// ```
#[derive(Clone, Debug)]
pub struct SymlogLocator {
    threshold: f64,
}

impl SymlogLocator {
    /// Create a symlog locator with the given linear threshold.
    ///
    /// `threshold` must be positive and finite; invalid values default to 1.0.
    #[must_use]
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold: if threshold > 0.0 && threshold.is_finite() {
                threshold
            } else {
                1.0
            },
        }
    }
}

impl Default for SymlogLocator {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl TickLocator for SymlogLocator {
    fn tick_values(&self, domain: (f64, f64), target_count: usize) -> Vec<f64> {
        symlog_ticks(domain.0, domain.1, self.threshold, target_count)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_locator_produces_nice_ticks() {
        let loc = AutoLocator;
        let ticks = loc.tick_values((0.0, 100.0), 5);
        assert!(!ticks.is_empty());
        assert!(*ticks.first().unwrap() >= -1.0);
        assert!(*ticks.last().unwrap() <= 110.0);
    }

    #[test]
    fn max_n_locator_enforces_limit() {
        let loc = MaxNLocator::new(4);
        let ticks = loc.tick_values((0.0, 1000.0), 10);
        assert!(
            ticks.len() <= 4,
            "MaxNLocator(4) produced {} ticks: {:?}",
            ticks.len(),
            ticks
        );
    }

    #[test]
    fn multiple_locator_basic() {
        let loc = MultipleLocator::new(25.0);
        let ticks = loc.tick_values((0.0, 100.0), 5);
        assert_eq!(ticks, vec![0.0, 25.0, 50.0, 75.0, 100.0]);
    }

    #[test]
    fn multiple_locator_invalid_base() {
        let loc = MultipleLocator::new(-5.0);
        // Invalid base defaults to 1.0
        let ticks = loc.tick_values((0.0, 3.0), 5);
        assert!(ticks.contains(&0.0));
        assert!(ticks.contains(&1.0));
        assert!(ticks.contains(&2.0));
        assert!(ticks.contains(&3.0));
    }

    #[test]
    fn multiple_locator_nan_domain() {
        let loc = MultipleLocator::new(10.0);
        let ticks = loc.tick_values((f64::NAN, f64::NAN), 5);
        assert!(!ticks.is_empty());
    }

    #[test]
    fn fixed_locator_returns_positions() {
        let loc = FixedLocator::new(vec![1.0, 3.14, 42.0]);
        let ticks = loc.tick_values((0.0, 100.0), 5);
        assert_eq!(ticks, vec![1.0, 3.14, 42.0]);
    }

    #[test]
    fn log_locator_basic() {
        let loc = LogLocator;
        let ticks = loc.tick_values((1.0, 1000.0), 5);
        assert!(ticks.iter().any(|t| (*t - 10.0).abs() < 0.1));
        assert!(ticks.iter().any(|t| (*t - 100.0).abs() < 1.0));
    }

    // --- SymlogLocator tests ---

    #[test]
    fn symlog_locator_includes_zero() {
        let loc = SymlogLocator::new(1.0);
        let ticks = loc.tick_values((-1000.0, 1000.0), 7);
        assert!(
            ticks.iter().any(|t| t.abs() < f64::EPSILON),
            "symlog locator should include zero: {ticks:?}"
        );
    }

    #[test]
    fn symlog_locator_symmetric_ticks() {
        let loc = SymlogLocator::new(1.0);
        let ticks = loc.tick_values((-100.0, 100.0), 7);
        // Should have roughly symmetric ticks
        let pos_count = ticks.iter().filter(|t| **t > 0.0).count();
        let neg_count = ticks.iter().filter(|t| **t < 0.0).count();
        assert!(
            pos_count.abs_diff(neg_count) <= 2,
            "symlog ticks should be roughly symmetric: pos={pos_count}, neg={neg_count}, ticks={ticks:?}"
        );
    }

    #[test]
    fn symlog_locator_positive_only() {
        let loc = SymlogLocator::new(1.0);
        let ticks = loc.tick_values((1.0, 10000.0), 5);
        for t in &ticks {
            assert!(*t > 0.0, "positive domain should have no negative ticks: {ticks:?}");
        }
    }

    #[test]
    fn symlog_locator_invalid_threshold() {
        // Invalid threshold defaults to 1.0
        let loc = SymlogLocator::new(-5.0);
        let ticks = loc.tick_values((-100.0, 100.0), 5);
        assert!(!ticks.is_empty());
    }

    #[test]
    fn symlog_locator_default() {
        let loc = SymlogLocator::default();
        let ticks = loc.tick_values((-10.0, 10.0), 5);
        assert!(!ticks.is_empty());
    }
}
