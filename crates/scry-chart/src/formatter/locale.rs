// SPDX-License-Identifier: MIT OR Apache-2.0
//! Locale-aware number formatting configuration.

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
pub(crate) fn apply_locale(s: &str, locale: &LocaleConfig) -> String {
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
pub(crate) fn apply_locale_batch(labels: Vec<String>, locale: &LocaleConfig) -> Vec<String> {
    labels
        .into_iter()
        .map(|l| apply_locale(&l, locale))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formatter::{FixedDecimalFormatter, LocaleFormatter, TickFormatter};

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
        use crate::formatter::AutoFormatter;
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
        use crate::formatter::FixedDecimalFormatter;
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
}
