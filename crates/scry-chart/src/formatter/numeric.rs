//! Numeric formatting: SI prefixes, binary SI, fixed decimal, scientific,
//! engineering notation, and thousands separators.

use crate::scale::format_tick_adaptive;

use super::{uniform_precision, TickFormatter};

// ---------------------------------------------------------------------------
// SiFormatter
// ---------------------------------------------------------------------------

/// Always uses SI suffixes (K, M, G, T, P) regardless of domain span.
///
/// Useful when you know the data is in large units (e.g., bytes, population).
#[derive(Clone, Debug)]
pub struct SiFormatter {
    /// Number of decimal places after the SI suffix (default: 1).
    pub decimals: usize,
}

impl Default for SiFormatter {
    fn default() -> Self {
        Self { decimals: 1 }
    }
}

impl TickFormatter for SiFormatter {
    fn format_batch(&self, values: &[f64], _domain: (f64, f64)) -> Vec<String> {
        values
            .iter()
            .map(|&v| format_si(v, self.decimals))
            .collect()
    }
}

/// Format a value with SI suffix.
pub(crate) fn format_si(value: f64, decimals: usize) -> String {
    let value = if value.abs() < f64::EPSILON * 100.0 {
        0.0
    } else {
        value
    };
    let abs = value.abs();

    let (scaled, suffix) = if abs >= 1e15 {
        (value / 1e15, "P")
    } else if abs >= 1e12 {
        (value / 1e12, "T")
    } else if abs >= 1e9 {
        (value / 1e9, "G")
    } else if abs >= 1e6 {
        (value / 1e6, "M")
    } else if abs >= 1e3 {
        (value / 1e3, "K")
    } else {
        (value, "")
    };

    if suffix.is_empty() {
        if (scaled - scaled.round()).abs() < f64::EPSILON * 100.0 {
            format!("{}", scaled as i64)
        } else {
            format!("{scaled:.prec$}", prec = decimals)
        }
    } else if decimals == 0 || (scaled - scaled.round()).abs() < 0.05 {
        format!("{}{suffix}", scaled.round() as i64)
    } else {
        format!("{scaled:.prec$}{suffix}", prec = decimals)
    }
}

// ---------------------------------------------------------------------------
// BinarySiFormatter
// ---------------------------------------------------------------------------

/// Formats values using binary SI (IEC) prefixes: KiB, MiB, GiB, TiB, PiB.
///
/// Uses powers of 1024 instead of 1000, matching standard byte-count
/// conventions for memory, file sizes, and network throughput.
///
/// # Example
///
/// ```
/// use scry_chart::formatter::{BinarySiFormatter, TickFormatter};
///
/// let fmt = BinarySiFormatter::default();
/// let labels = fmt.format_batch(&[0.0, 1024.0, 1048576.0], (0.0, 1048576.0));
/// assert_eq!(labels, vec!["0", "1 KiB", "1 MiB"]);
/// ```
#[derive(Clone, Debug)]
pub struct BinarySiFormatter {
    /// Number of decimal places for fractional values (default: 1).
    pub decimals: usize,
}

impl Default for BinarySiFormatter {
    fn default() -> Self {
        Self { decimals: 1 }
    }
}
const KIB: f64 = 1024.0;
const MIB: f64 = 1024.0 * 1024.0;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const TIB: f64 = 1024.0 * 1024.0 * 1024.0 * 1024.0;
const PIB: f64 = 1024.0 * 1024.0 * 1024.0 * 1024.0 * 1024.0;

impl BinarySiFormatter {
    /// Format a single value with binary SI prefix.
    fn format_binary_si(value: f64, decimals: usize) -> String {
        let value = if value.abs() < f64::EPSILON * 100.0 {
            0.0
        } else {
            value
        };
        let abs = value.abs();

        let (scaled, suffix) = if abs >= PIB {
            (value / PIB, " PiB")
        } else if abs >= TIB {
            (value / TIB, " TiB")
        } else if abs >= GIB {
            (value / GIB, " GiB")
        } else if abs >= MIB {
            (value / MIB, " MiB")
        } else if abs >= KIB {
            (value / KIB, " KiB")
        } else {
            return if (value - value.round()).abs() < 0.05 {
                format!("{}", value.round() as i64)
            } else {
                format!("{value:.prec$}", prec = decimals)
            };
        };

        if decimals == 0 || (scaled - scaled.round()).abs() < 0.05 {
            format!("{}{suffix}", scaled.round() as i64)
        } else {
            format!("{scaled:.prec$}{suffix}", prec = decimals)
        }
    }
}

impl TickFormatter for BinarySiFormatter {
    fn format_batch(&self, values: &[f64], _domain: (f64, f64)) -> Vec<String> {
        values
            .iter()
            .map(|&v| Self::format_binary_si(v, self.decimals))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// FixedDecimalFormatter
// ---------------------------------------------------------------------------

/// Always uses a fixed number of decimal places.
///
/// `FixedDecimalFormatter(2)` formats 0.0 as "0.00", 1.5 as "1.50".
#[derive(Clone, Debug)]
pub struct FixedDecimalFormatter(pub usize);

impl TickFormatter for FixedDecimalFormatter {
    fn format_batch(&self, values: &[f64], _domain: (f64, f64)) -> Vec<String> {
        values
            .iter()
            .map(|&v| {
                let v = if v.abs() < f64::EPSILON * 100.0 {
                    0.0
                } else {
                    v
                };
                format!("{v:.prec$}", prec = self.0)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ScientificFormatter
// ---------------------------------------------------------------------------

/// Always uses scientific notation (e.g., "1.23e4").
#[derive(Clone, Debug)]
pub struct ScientificFormatter {
    /// Number of significant decimal digits (default: 2).
    pub precision: usize,
}

impl Default for ScientificFormatter {
    fn default() -> Self {
        Self { precision: 2 }
    }
}

impl TickFormatter for ScientificFormatter {
    fn format_batch(&self, values: &[f64], _domain: (f64, f64)) -> Vec<String> {
        values
            .iter()
            .map(|&v| {
                let v = if v.abs() < f64::EPSILON * 100.0 {
                    0.0
                } else {
                    v
                };
                format!("{v:.prec$e}", prec = self.precision)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// EngineeringFormatter
// ---------------------------------------------------------------------------

/// Formats values in engineering notation where exponents are always multiples of 3.
///
/// This aligns exponents with metric prefixes (kilo, mega, giga, etc.),
/// making values easier to read in scientific and industrial contexts.
///
/// # Example
///
/// ```
/// use scry_chart::formatter::{EngineeringFormatter, TickFormatter};
///
/// let fmt = EngineeringFormatter::default();
/// assert_eq!(fmt.format(47000.0), "47.00e3");
/// assert_eq!(fmt.format(0.0025), "2.50e-3");
/// ```
#[derive(Clone, Debug)]
pub struct EngineeringFormatter {
    /// Number of decimal digits in the mantissa (default: 2).
    pub precision: usize,
}

impl Default for EngineeringFormatter {
    fn default() -> Self {
        Self { precision: 2 }
    }
}

impl TickFormatter for EngineeringFormatter {
    fn format_batch(&self, values: &[f64], _domain: (f64, f64)) -> Vec<String> {
        values
            .iter()
            .map(|&v| {
                let v = if v.abs() < f64::EPSILON * 100.0 {
                    0.0
                } else {
                    v
                };
                if v == 0.0 {
                    return "0".to_string();
                }

                let exp = v.abs().log10().floor() as i32;
                // Snap to nearest multiple of 3 (towards zero)
                let eng_exp = exp - exp.rem_euclid(3);
                let mantissa = v / 10_f64.powi(eng_exp);

                if eng_exp == 0 {
                    format!("{mantissa:.prec$}", prec = self.precision)
                } else {
                    format!("{mantissa:.prec$}e{eng_exp}", prec = self.precision)
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ThousandsFormatter — comma-separated thousands
// ---------------------------------------------------------------------------

/// Formats numbers with comma-separated thousands (e.g., `1,234,567`).
///
/// # Example
///
/// ```
/// use scry_chart::formatter::{ThousandsFormatter, TickFormatter};
///
/// let fmt = ThousandsFormatter;
/// // Use a domain where values stay below the SI-suffix threshold (< 10K)
/// let labels = fmt.format_batch(&[0.0, 1000.0, 5000.0], (0.0, 5000.0));
/// assert_eq!(labels[0], "0");
/// assert_eq!(labels[1], "1,000");
/// assert_eq!(labels[2], "5,000");
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ThousandsFormatter;

impl ThousandsFormatter {
    /// Insert comma separators into the integer part of a number string.
    fn add_thousands_separators(s: &str) -> String {
        let (sign, rest) = s
            .strip_prefix('-')
            .map_or(("", s), |stripped| ("-", stripped));

        let (int_part, frac_part) = rest
            .find('.')
            .map_or((rest, ""), |pos| (&rest[..pos], &rest[pos..]));

        let mut result = String::with_capacity(s.len() + int_part.len() / 3);
        result.push_str(sign);

        for (i, ch) in int_part.chars().enumerate() {
            if i > 0 && (int_part.len() - i) % 3 == 0 {
                result.push(',');
            }
            result.push(ch);
        }

        result.push_str(frac_part);
        result
    }
}

impl TickFormatter for ThousandsFormatter {
    fn format_batch(&self, values: &[f64], domain: (f64, f64)) -> Vec<String> {
        let raw: Vec<String> = values
            .iter()
            .map(|&v| format_tick_adaptive(v, domain.0, domain.1))
            .collect();
        let unified = uniform_precision(raw);
        unified
            .into_iter()
            .map(|l| {
                // Only add separators to plain numbers (not SI-suffixed)
                if l.bytes().any(|b| b.is_ascii_alphabetic() || b == b'%') {
                    l
                } else {
                    Self::add_thousands_separators(&l)
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_thousands_separators() {
        let sep = ThousandsFormatter::add_thousands_separators("1234567");
        assert_eq!(sep, "1,234,567");
        let sep2 = ThousandsFormatter::add_thousands_separators("-42000.50");
        assert_eq!(sep2, "-42,000.50");
        let sep3 = ThousandsFormatter::add_thousands_separators("999");
        assert_eq!(sep3, "999");
    }
}
