//! Tick formatting system for chart axes.
//!
//! Provides a [`TickFormatter`] trait and built-in implementations for
//! common formatting needs. The default [`AutoFormatter`] produces
//! uniform-precision labels with SI suffixes for large values.

use std::sync::Arc;

use crate::scale::format_tick_adaptive;

// ---------------------------------------------------------------------------
// Locale configuration
// ---------------------------------------------------------------------------

/// Locale-aware number formatting configuration.
///
/// Controls decimal separator and digit grouping characters. Applied as a
/// post-processing step after the core formatting logic, so all formatters
/// can benefit without duplicating locale logic.
///
/// # Example
///
/// ```
/// use scry_chart::formatter::LocaleConfig;
///
/// let locale = LocaleConfig::european();
/// assert_eq!(locale.decimal_separator, ',');
/// assert_eq!(locale.thousands_separator, '.');
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LocaleConfig {
    /// Character between integer and fractional parts (default: `.`).
    pub decimal_separator: char,
    /// Character between digit groups (default: `,`).
    pub thousands_separator: char,
    /// Number of digits per group (default: 3).
    pub group_size: usize,
}

impl Default for LocaleConfig {
    fn default() -> Self {
        Self::us()
    }
}

impl LocaleConfig {
    /// US/English convention: period decimal, comma grouping.
    #[must_use]
    pub fn us() -> Self {
        Self {
            decimal_separator: '.',
            thousands_separator: ',',
            group_size: 3,
        }
    }

    /// European convention: comma decimal, period grouping.
    #[must_use]
    pub fn european() -> Self {
        Self {
            decimal_separator: ',',
            thousands_separator: '.',
            group_size: 3,
        }
    }

    /// Swiss convention: period decimal, apostrophe grouping.
    #[must_use]
    pub fn swiss() -> Self {
        Self {
            decimal_separator: '.',
            thousands_separator: '\'',
            group_size: 3,
        }
    }

    /// Indian convention: period decimal, comma grouping, mixed group sizes
    /// (first group of 3, then groups of 2).
    #[must_use]
    pub fn indian() -> Self {
        Self {
            decimal_separator: '.',
            thousands_separator: ',',
            group_size: 2, // after initial 3-digit group
        }
    }
}

/// Apply locale formatting to a numeric string.
///
/// Swaps the decimal separator and inserts thousands grouping characters
/// into the integer part. Skips strings that contain alphabetic characters
/// or `%` (SI-suffixed, percent, or currency labels are handled by their
/// formatters and should not be double-processed).
fn apply_locale(s: &str, locale: &LocaleConfig) -> String {
    // Skip non-numeric labels (SI suffixes, %, currency symbols, etc.)
    if s.bytes().any(|b| b.is_ascii_alphabetic() || b == b'%') {
        // Still swap decimal separator for labels like "1.5K"
        if locale.decimal_separator != '.' {
            return s.replace('.', &locale.decimal_separator.to_string());
        }
        return s.to_string();
    }

    let (sign, rest) = s
        .strip_prefix('-')
        .map_or(("", s), |stripped| ("-", stripped));
    let (int_part, frac_with_dot) = rest
        .find('.')
        .map_or((rest, ""), |pos| (&rest[..pos], &rest[pos + 1..]));

    // Insert thousands grouping into integer part
    let mut grouped = String::with_capacity(int_part.len() + int_part.len() / 3 + 1);
    grouped.push_str(sign);
    let digits: Vec<char> = int_part.chars().collect();
    let n = digits.len();
    for (i, &ch) in digits.iter().enumerate() {
        if i > 0 && (n - i) % locale.group_size == 0 {
            grouped.push(locale.thousands_separator);
        }
        grouped.push(ch);
    }

    // Append fractional part with locale decimal separator
    if !frac_with_dot.is_empty() {
        grouped.push(locale.decimal_separator);
        grouped.push_str(frac_with_dot);
    }

    grouped
}

/// Apply locale formatting to a batch of labels.
fn apply_locale_batch(labels: Vec<String>, locale: &LocaleConfig) -> Vec<String> {
    labels
        .into_iter()
        .map(|l| apply_locale(&l, locale))
        .collect()
}

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
fn format_si(value: f64, decimals: usize) -> String {
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

// ---------------------------------------------------------------------------
// DateTimeFormatter — Unix timestamps → date/time strings
// ---------------------------------------------------------------------------

/// Formats Unix timestamps (seconds since 1970-01-01) as human-readable
/// date/time strings.
///
/// Uses a simple built-in formatter that does not require external date
/// libraries. The format adapts based on the time span:
/// - Span ≤ 1 hour → `HH:MM:SS`
/// - Span ≤ 1 day  → `HH:MM`
/// - Span ≤ 90 days → `Mon DD`
/// - Otherwise     → `YYYY-MM-DD`
///
/// # Example
///
/// ```
/// use scry_chart::formatter::DateTimeFormatter;
///
/// let fmt = DateTimeFormatter;
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DateTimeFormatter;

impl DateTimeFormatter {
    /// Format a Unix timestamp to a date/time string based on span.
    fn format_timestamp(ts: f64, span_secs: f64) -> String {
        let ts = ts as i64;
        let secs_per_day: i64 = 86400;
        let secs_per_hour: i64 = 3600;

        if span_secs <= secs_per_hour as f64 {
            // HH:MM:SS
            let h = (ts % secs_per_day) / secs_per_hour;
            let m = (ts % secs_per_hour) / 60;
            let s = ts % 60;
            format!("{h:02}:{m:02}:{s:02}")
        } else if span_secs <= secs_per_day as f64 {
            // HH:MM
            let h = ((ts % secs_per_day) + secs_per_day) % secs_per_day / secs_per_hour;
            let m = (ts % secs_per_hour + secs_per_hour) % secs_per_hour / 60;
            format!("{h:02}:{m:02}")
        } else if span_secs <= 90.0 * secs_per_day as f64 {
            // Mon DD (approximate)
            let days = ts / secs_per_day;
            let (_, month, day) = Self::days_to_ymd(days);
            let month_name = Self::month_abbr(month);
            format!("{month_name} {day}")
        } else {
            // YYYY-MM-DD
            let days = ts / secs_per_day;
            let (year, month, day) = Self::days_to_ymd(days);
            format!("{year}-{month:02}-{day:02}")
        }
    }

    /// Convert days since epoch to (year, month, day).
    /// Simple civil calendar conversion (no leap second handling).
    fn days_to_ymd(days: i64) -> (i64, u32, u32) {
        // Algorithm from Howard Hinnant's chrono-compatible date library
        let z = days + 719_468;
        let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
        let doe = (z - era * 146_097) as u32;
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
        let y = yoe as i64 + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        let y = if m <= 2 { y + 1 } else { y };
        (y, m, d)
    }

    fn month_abbr(month: u32) -> &'static str {
        match month {
            1 => "Jan",
            2 => "Feb",
            3 => "Mar",
            4 => "Apr",
            5 => "May",
            6 => "Jun",
            7 => "Jul",
            8 => "Aug",
            9 => "Sep",
            10 => "Oct",
            11 => "Nov",
            12 => "Dec",
            _ => "???",
        }
    }
}

impl TickFormatter for DateTimeFormatter {
    fn format_batch(&self, values: &[f64], domain: (f64, f64)) -> Vec<String> {
        let span = (domain.1 - domain.0).abs();
        values
            .iter()
            .map(|&v| Self::format_timestamp(v, span))
            .collect()
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
// SemanticZoomFormatter — adapts formatting to zoom level
// ---------------------------------------------------------------------------

/// Classification of the current zoom level based on the ratio of the
/// visible domain span to the full data span.
///
/// Used by [`SemanticZoomFormatter`] to select an appropriate child
/// formatter for the current zoom level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Ensure all labels in a batch have consistent decimal precision.
///
/// Finds the maximum number of decimal digits in any label, then
/// pads shorter labels to match. Only operates on labels that look
/// like plain numbers (no SI suffix, no special formatting).
fn uniform_precision(labels: Vec<String>) -> Vec<String> {
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
            if l.bytes().any(|b| b.is_ascii_alphabetic() || b == b'%') {
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
            if l.bytes().any(|b| b.is_ascii_alphabetic() || b == b'%') {
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
        } else if !l.is_empty() && !l.bytes().any(|b| b.is_ascii_alphabetic() || b == b'%') {
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
            if l.bytes().any(|b| b.is_ascii_alphabetic() || b == b'%') {
                return l;
            }
            // Convert plain number to SI
            l.parse::<f64>().map_or(l, |v| {
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
        // All are >= 10K → SI suffixed, no comma changes needed
        // For smaller values, verify comma insertion
        let sep = ThousandsFormatter::add_thousands_separators("1234567");
        assert_eq!(sep, "1,234,567");
        let sep2 = ThousandsFormatter::add_thousands_separators("-42000.50");
        assert_eq!(sep2, "-42,000.50");
        let sep3 = ThousandsFormatter::add_thousands_separators("999");
        assert_eq!(sep3, "999");
    }

    #[test]
    fn datetime_formatter_adapts_to_span() {
        use crate::formatter::TickFormatter;
        let fmt = DateTimeFormatter;
        // 2 hour span → HH:MM format
        let ts = 1700000000.0; // some timestamp
        let labels = fmt.format_batch(&[ts, ts + 3600.0], (ts, ts + 7200.0));
        assert!(labels[0].contains(':'), "expected HH:MM, got {}", labels[0]);
        // Multi-year span → YYYY-MM-DD format
        let labels = fmt.format_batch(&[0.0, 365.0 * 86400.0 * 5.0], (0.0, 365.0 * 86400.0 * 5.0));
        assert!(
            labels[0].contains('-'),
            "expected YYYY-MM-DD, got {}",
            labels[0]
        );
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

    // --- LocaleConfig / LocaleFormatter tests ---

    #[test]
    fn locale_european_decimal() {
        let locale = LocaleConfig::european();
        let fmt = LocaleFormatter::new(FixedDecimalFormatter(2), locale);
        let labels = fmt.format_batch(&[1234.56, -42000.5, 0.0], (0.0, 50000.0));
        assert_eq!(labels[0], "1.234,56");
        assert_eq!(labels[1], "-42.000,50");
        assert_eq!(labels[2], "0,00");
    }

    #[test]
    fn locale_swiss_grouping() {
        let locale = LocaleConfig::swiss();
        let fmt = LocaleFormatter::new(FixedDecimalFormatter(0), locale);
        let labels = fmt.format_batch(&[1234567.0], (0.0, 2_000_000.0));
        assert_eq!(labels[0], "1'234'567");
    }

    #[test]
    fn locale_none_is_default() {
        // Without locale, AutoFormatter output is unchanged
        let fmt = AutoFormatter;
        let labels = fmt.format_batch(&[1000.0, 2000.0, 3000.0], (0.0, 3000.0));
        // Should produce SI-suffixed or plain numbers — no grouping chars
        for label in &labels {
            assert!(
                !label.contains(','),
                "default should not add commas: {label}"
            );
            assert!(
                !label.contains('\''),
                "default should not add apostrophes: {label}"
            );
        }
    }

    #[test]
    fn locale_with_si_labels() {
        // European locale should swap decimal in SI labels like "1.5K" → "1,5K"
        let locale = LocaleConfig::european();
        let result = apply_locale("1.5K", &locale);
        assert_eq!(result, "1,5K");
    }

    #[test]
    fn locale_auto_formatter_composition() {
        // LocaleFormatter wrapping FixedDecimalFormatter for a deterministic check
        let fmt = LocaleFormatter::new(FixedDecimalFormatter(1), LocaleConfig::european());
        let labels = fmt.format_batch(&[500.0, 1000.0, 1500.0], (0.0, 1500.0));
        // European locale: "500,0", "1.000,0", "1.500,0"
        assert!(
            labels[0].contains(','),
            "european locale should use comma decimal: {}",
            labels[0]
        );
        assert!(
            !labels[0].contains('.'),
            "500 should not have period: {}",
            labels[0]
        );
    }

    // --- SemanticZoomFormatter tests ---

    #[test]
    fn zoom_level_from_span() {
        // Full view → Overview
        assert_eq!(ZoomLevel::from_domain_span(100.0, 100.0), ZoomLevel::Overview);
        // 80% → still Overview
        assert_eq!(ZoomLevel::from_domain_span(80.0, 100.0), ZoomLevel::Overview);
        // 50% → Standard
        assert_eq!(ZoomLevel::from_domain_span(50.0, 100.0), ZoomLevel::Standard);
        // 10% → Detailed
        assert_eq!(ZoomLevel::from_domain_span(10.0, 100.0), ZoomLevel::Detailed);
        // 2% → Microscope
        assert_eq!(ZoomLevel::from_domain_span(2.0, 100.0), ZoomLevel::Microscope);
        // Edge: zero full_span → Standard (safe fallback)
        assert_eq!(ZoomLevel::from_domain_span(10.0, 0.0), ZoomLevel::Standard);
        // Edge: NaN → Standard
        assert_eq!(ZoomLevel::from_domain_span(f64::NAN, 100.0), ZoomLevel::Standard);
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
