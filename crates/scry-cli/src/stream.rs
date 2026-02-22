// SPDX-License-Identifier: MIT OR Apache-2.0
//! `scry stream` — live streaming chart from stdin.
//!
//! Reads numeric data line-by-line from stdin and renders a live-updating
//! chart inline in the terminal using the auto-detected graphics protocol.
//!
//! ```bash
//! # Single series
//! seq 1 100 | scry stream --title "Counting"
//!
//! # Multi-column (space-delimited)
//! vmstat 1 | scry stream --columns 0,1,2
//!
//! # Custom delimiter and window
//! cat sensor.csv | scry stream --delimiter "," --window 200
//! ```

use std::io::{self, BufRead, Write};
use std::time::{Duration, Instant};

use scry_chart::streaming::StreamingChart;

use crate::display;

// ---------------------------------------------------------------------------
// CLI arguments
// ---------------------------------------------------------------------------

/// Live streaming chart from data sources.
#[derive(Debug, clap::Args)]
pub struct StreamArgs {
    /// Window size (number of points to display)
    #[arg(long, default_value = "100")]
    window: usize,

    /// Chart width in pixels
    #[arg(short = 'W', long, default_value = "800")]
    width: u32,

    /// Chart height in pixels
    #[arg(short = 'H', long, default_value = "400")]
    height: u32,

    /// Update interval in milliseconds
    #[arg(long, default_value = "200")]
    interval: u64,

    /// Chart title
    #[arg(long)]
    title: Option<String>,

    /// Y-axis range (min:max)
    #[arg(long)]
    y_range: Option<String>,

    /// Column delimiter for multi-column input
    #[arg(long, default_value = " ")]
    delimiter: String,

    /// Which columns to plot (0-indexed, comma-separated). Default: all numeric.
    #[arg(long)]
    columns: Option<String>,

    /// Series labels (comma-separated, for legend)
    #[arg(long)]
    labels: Option<String>,
}

// ---------------------------------------------------------------------------
// Line parsing
// ---------------------------------------------------------------------------

/// Parse a line of text into numeric values using the given delimiter.
///
/// Non-numeric tokens are silently skipped.
fn parse_line(line: &str, delimiter: &str) -> Vec<f64> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    // Split by delimiter (special-case: whitespace collapses runs)
    let tokens: Vec<&str> = if delimiter == " " || delimiter == "\t" {
        trimmed.split_whitespace().collect()
    } else {
        trimmed.split(delimiter).collect()
    };

    tokens
        .iter()
        .filter_map(|t| t.trim().parse::<f64>().ok())
        .collect()
}

/// Parse selected columns from a set of values.
fn select_columns(values: &[f64], columns: &Option<Vec<usize>>) -> Vec<f64> {
    match columns {
        Some(cols) => cols
            .iter()
            .filter_map(|&i| values.get(i).copied())
            .collect(),
        None => values.to_vec(),
    }
}

/// Parse a "min:max" range string into (f64, f64).
fn parse_y_range(s: &str) -> Result<(f64, f64), String> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 2 {
        return Err(format!("invalid y-range '{s}': expected 'min:max'"));
    }
    let min: f64 = parts[0]
        .parse()
        .map_err(|_| format!("invalid y-range min: '{}'", parts[0]))?;
    let max: f64 = parts[1]
        .parse()
        .map_err(|_| format!("invalid y-range max: '{}'", parts[1]))?;
    Ok((min, max))
}

/// Parse column indices from a comma-separated string like "0,2,5".
fn parse_columns(s: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(|c| {
            c.trim()
                .parse::<usize>()
                .map_err(|_| format!("invalid column index: '{c}'"))
        })
        .collect()
}

/// Check if a value looks like a Unix timestamp (seconds since epoch).
/// Heuristic: value > 1e9 (Sep 2001) and < 1e11 (year 5138).
fn looks_like_timestamp(v: f64) -> bool {
    v > 1e9 && v < 1e11
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Estimate the number of terminal rows an image of the given pixel height
/// occupies, assuming ~16px per terminal row (reasonable default).
fn estimate_terminal_rows(pixel_height: u32) -> u16 {
    // Try to get actual cell height, fall back to 16px
    let cell_height = 16u32;
    pixel_height.div_ceil(cell_height) as u16
}

/// Render one frame: move cursor up to overwrite previous frame, then display.
fn render_frame(
    driver: &mut display::FrameDriver,
    chart: &StreamingChart,
    width: u32,
    height: u32,
    frame_number: u64,
) -> Result<(), String> {
    let png_data = chart
        .render(width, height)
        .map_err(|e| format!("render failed: {e}"))?;

    let mut stdout = io::stdout().lock();

    // If not the first frame, move cursor up to overwrite previous image
    if frame_number > 0 {
        let rows = estimate_terminal_rows(height);
        // Move cursor up and clear each line
        write!(stdout, "\x1b[{}A", rows + 1).map_err(|e| e.to_string())?;
        for _ in 0..=rows {
            write!(stdout, "\x1b[2K\x1b[1B").map_err(|e| e.to_string())?;
        }
        write!(stdout, "\x1b[{}A", rows + 1).map_err(|e| e.to_string())?;
        stdout.flush().map_err(|e| e.to_string())?;
    }

    driver.display_png(&png_data)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

pub fn run(args: &StreamArgs) -> Result<(), String> {
    // Parse optional y-range
    let y_range = args
        .y_range
        .as_ref()
        .map(|s| parse_y_range(s))
        .transpose()?;

    // Parse optional column selection
    let columns = args
        .columns
        .as_ref()
        .map(|s| parse_columns(s))
        .transpose()?;

    // Parse optional labels
    let labels: Option<Vec<String>> = args
        .labels
        .as_ref()
        .map(|s| s.split(',').map(|l| l.trim().to_string()).collect());

    // We'll determine n_series from the first data line
    let mut chart: Option<StreamingChart> = None;
    let mut point_counter: u64 = 0;
    let mut frame_number: u64 = 0;

    let interval = Duration::from_millis(args.interval);
    let mut last_render = Instant::now();

    let mut driver = display::FrameDriver::detect();

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = line.map_err(|e| format!("stdin read error: {e}"))?;
        let all_values = parse_line(&line, &args.delimiter);

        if all_values.is_empty() {
            continue;
        }

        let values = select_columns(&all_values, &columns);
        if values.is_empty() {
            continue;
        }

        // Initialize chart on first data line
        let c = chart.get_or_insert_with(|| {
            // Detect timestamp mode: if first value of multi-column data looks like a timestamp
            let n_series = if values.len() > 1 && looks_like_timestamp(values[0]) {
                values.len() - 1 // first col is X
            } else {
                values.len()
            };

            let mut sc = StreamingChart::new()
                .window_size(args.window)
                .n_series(n_series);

            if let Some(ref title) = args.title {
                sc = sc.title(title.as_str());
            }
            if let Some((min, max)) = y_range {
                sc = sc.y_range(min, max);
            }
            if let Some(ref l) = labels {
                sc = sc.labels(l.clone());
            }

            sc
        });

        // Push data
        if values.len() == 1 {
            c.push_now(values[0]);
        } else if values.len() > 1 && looks_like_timestamp(values[0]) {
            // First column is timestamp, rest are series values
            let x = values[0];
            for (i, &v) in values[1..].iter().enumerate() {
                c.push_series(i, x, v);
            }
        } else {
            // All columns are series values, auto-increment X
            for (i, &v) in values.iter().enumerate() {
                c.push_series(i, point_counter as f64, v);
            }
        }

        point_counter += 1;

        // Rate-limited rendering
        if last_render.elapsed() >= interval {
            render_frame(&mut driver, c, args.width, args.height, frame_number)?;
            frame_number += 1;
            last_render = Instant::now();
        }
    }

    // Final render after stdin closes (show last state)
    if let Some(ref c) = chart {
        if c.total_points() > 0 {
            render_frame(&mut driver, c, args.width, args.height, frame_number)?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_line_single_value() {
        assert_eq!(parse_line("42.5", " "), vec![42.5]);
    }

    #[test]
    fn parse_line_multi_value() {
        assert_eq!(parse_line("42.5 18.3 95.0", " "), vec![42.5, 18.3, 95.0]);
    }

    #[test]
    fn parse_line_comma_delim() {
        assert_eq!(parse_line("42.5,18.3", ","), vec![42.5, 18.3]);
    }

    #[test]
    fn parse_line_with_junk() {
        let result = parse_line("not a number", " ");
        assert!(result.is_empty());
    }

    #[test]
    fn parse_line_mixed() {
        assert_eq!(parse_line("42.5 foo 18.3", " "), vec![42.5, 18.3]);
    }

    #[test]
    fn parse_line_empty() {
        assert!(parse_line("", " ").is_empty());
        assert!(parse_line("   ", " ").is_empty());
    }

    #[test]
    fn parse_line_tab_delim() {
        assert_eq!(parse_line("1.0\t2.0\t3.0", "\t"), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn parse_line_whitespace_collapse() {
        assert_eq!(
            parse_line("  42.5   18.3   95.0  ", " "),
            vec![42.5, 18.3, 95.0]
        );
    }

    #[test]
    fn y_range_parsing() {
        assert_eq!(parse_y_range("0:100"), Ok((0.0, 100.0)));
        assert_eq!(parse_y_range("-50:50.5"), Ok((-50.0, 50.5)));
        assert!(parse_y_range("abc:100").is_err());
        assert!(parse_y_range("100").is_err());
    }

    #[test]
    fn column_parsing() {
        assert_eq!(parse_columns("0,2,5"), Ok(vec![0, 2, 5]));
        assert_eq!(parse_columns("1"), Ok(vec![1]));
        assert!(parse_columns("a,b").is_err());
    }

    #[test]
    fn select_columns_some() {
        let values = vec![10.0, 20.0, 30.0, 40.0];
        let cols = Some(vec![0, 2]);
        assert_eq!(select_columns(&values, &cols), vec![10.0, 30.0]);
    }

    #[test]
    fn select_columns_none() {
        let values = vec![10.0, 20.0, 30.0];
        assert_eq!(select_columns(&values, &None), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn select_columns_out_of_bounds() {
        let values = vec![10.0, 20.0];
        let cols = Some(vec![0, 5]); // index 5 doesn't exist
        assert_eq!(select_columns(&values, &cols), vec![10.0]);
    }

    #[test]
    fn timestamp_detection() {
        assert!(looks_like_timestamp(1_708_099_200.0));
        assert!(!looks_like_timestamp(42.5));
        assert!(!looks_like_timestamp(0.0));
        assert!(!looks_like_timestamp(-1.0));
    }

    #[test]
    fn integration_streaming_chart_push() {
        let mut chart = StreamingChart::new().window_size(50).title("Test Stream");

        for i in 0..10 {
            chart.push(i as f64, (i * i) as f64);
        }

        let snap = chart.snapshot();
        assert!(snap.is_some());
    }
}
