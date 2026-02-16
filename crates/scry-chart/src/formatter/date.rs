//! Date/time formatting for Unix timestamps.

use super::TickFormatter;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn datetime_formatter_adapts_to_span() {
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
}
