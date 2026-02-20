// SPDX-License-Identifier: MIT OR Apache-2.0
//! Semantic formatters: percent and currency.

use super::numeric::format_si;
use super::TickFormatter;

// ---------------------------------------------------------------------------
// PercentFormatter
// ---------------------------------------------------------------------------

/// Formats values as percentages.
///
/// If `is_fraction` is true (default), the value is multiplied by 100
/// (i.e., 0.5 → "50%"). If false, the value is used as-is (50.0 → "50%").
#[derive(Clone, Debug)]
pub struct PercentFormatter {
    /// Number of decimal places (default: 0).
    pub decimals: usize,
    /// Whether input values are fractions (0.0–1.0) that need ×100.
    pub is_fraction: bool,
}

impl Default for PercentFormatter {
    fn default() -> Self {
        Self {
            decimals: 0,
            is_fraction: true,
        }
    }
}

impl TickFormatter for PercentFormatter {
    fn format_batch(&self, values: &[f64], _domain: (f64, f64)) -> Vec<String> {
        values
            .iter()
            .map(|&v| {
                let v = if v.abs() < f64::EPSILON * 100.0 {
                    0.0
                } else {
                    v
                };
                let pct = if self.is_fraction { v * 100.0 } else { v };
                format!("{pct:.prec$}%", prec = self.decimals)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// CurrencyFormatter
// ---------------------------------------------------------------------------

/// Formats values with a currency prefix.
///
/// Supports optional SI suffix for large values (e.g., "$1.2M").
#[derive(Clone, Debug)]
pub struct CurrencyFormatter {
    /// Currency symbol (default: "$").
    pub symbol: String,
    /// Number of decimal places (default: 0 for large values, 2 otherwise).
    pub decimals: usize,
    /// Whether to use SI suffixes for large values (default: true).
    pub si_suffixes: bool,
    /// Use accounting-style negatives: `($100)` instead of `-$100`.
    pub accounting: bool,
}

impl Default for CurrencyFormatter {
    fn default() -> Self {
        Self {
            symbol: "$".to_string(),
            decimals: 0,
            si_suffixes: true,
            accounting: false,
        }
    }
}

impl TickFormatter for CurrencyFormatter {
    fn format_batch(&self, values: &[f64], _domain: (f64, f64)) -> Vec<String> {
        values
            .iter()
            .map(|&v| {
                let v = if v.abs() < f64::EPSILON * 100.0 {
                    0.0
                } else {
                    v
                };
                let is_neg = v < 0.0;
                let abs_v = v.abs();
                let raw = if self.si_suffixes && abs_v >= 1e3 {
                    format!("{}{}", self.symbol, format_si(abs_v, self.decimals))
                } else {
                    format!("{}{abs_v:.prec$}", self.symbol, prec = self.decimals)
                };
                if is_neg {
                    if self.accounting {
                        format!("({raw})")
                    } else {
                        format!("-{raw}")
                    }
                } else {
                    raw
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percent_formatter_fractions() {
        let fmt = PercentFormatter::default();
        let labels = fmt.format_batch(&[0.0, 0.25, 0.5, 0.75, 1.0], (0.0, 1.0));
        assert_eq!(labels, vec!["0%", "25%", "50%", "75%", "100%"]);
    }

    #[test]
    fn percent_formatter_raw() {
        let fmt = PercentFormatter {
            decimals: 1,
            is_fraction: false,
        };
        let labels = fmt.format_batch(&[0.0, 25.0, 50.0], (0.0, 100.0));
        assert_eq!(labels, vec!["0.0%", "25.0%", "50.0%"]);
    }

    #[test]
    fn currency_formatter_si() {
        let fmt = CurrencyFormatter::default();
        let labels = fmt.format_batch(&[0.0, 500_000.0, 1_000_000.0], (0.0, 1_000_000.0));
        assert_eq!(labels[0], "$0");
        assert_eq!(labels[1], "$500K");
        assert_eq!(labels[2], "$1M");
    }

    #[test]
    fn currency_formatter_accounting_negatives() {
        let fmt = CurrencyFormatter {
            accounting: true,
            ..CurrencyFormatter::default()
        };
        let labels = fmt.format_batch(
            &[-500.0, 0.0, 1000.0, -2_000_000.0],
            (-2_000_000.0, 1_000_000.0),
        );
        assert_eq!(labels[0], "($500)");
        assert_eq!(labels[1], "$0");
        assert_eq!(labels[2], "$1K");
        assert_eq!(labels[3], "($2M)");
    }

    #[test]
    fn currency_formatter_negative_zero() {
        let fmt = CurrencyFormatter::default();
        let labels = fmt.format_batch(&[-0.0], (-100.0, 100.0));
        // -0.0 should produce "$0", not "-$0" or "(-$0)"
        assert_eq!(labels[0], "$0", "Negative zero should display as $0");
    }

    #[test]
    fn percent_formatter_nan_inf() {
        let fmt = PercentFormatter::default();
        // Should not panic on NaN/Inf inputs
        let labels = fmt.format_batch(
            &[f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.5],
            (0.0, 1.0),
        );
        assert_eq!(labels.len(), 4);
        // NaN and Inf produce *some* output, not crash
        assert!(!labels[0].is_empty(), "NaN should produce a label");
        assert!(!labels[1].is_empty(), "Inf should produce a label");
        assert!(!labels[2].is_empty(), "-Inf should produce a label");
        assert_eq!(labels[3], "50%");
    }
}
