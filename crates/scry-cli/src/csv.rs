// SPDX-License-Identifier: MIT OR Apache-2.0
//! CSV data ingestion for the `scry chart plot` command.
//!
//! Parses CSV/TSV from stdin or files, extracts columns by header name,
//! supports multi-series, and auto-detects chart type based on data shape.

use crate::spec::{ChartData, ChartSpec, ChartType, GroupSpec, SeriesSpec};
use std::io::Read;

// ---------------------------------------------------------------------------
// Parse options
// ---------------------------------------------------------------------------

/// Configuration for CSV parsing behaviour.
#[derive(Debug, Clone)]
pub struct CsvParseOptions {
    /// Field delimiter byte (default: b',').
    pub delimiter: u8,
    /// Whether the first row is a header.  When `false`, synthetic headers
    /// `col0`, `col1`, … are generated.
    pub has_header: bool,
    /// Skip this many data rows after the header (useful for comment lines).
    pub skip_rows: usize,
    /// Maximum number of data rows to read (safety cap).
    pub max_rows: usize,
}

impl Default for CsvParseOptions {
    fn default() -> Self {
        Self {
            delimiter: b',',
            has_header: true,
            skip_rows: 0,
            max_rows: 100_000,
        }
    }
}

// ---------------------------------------------------------------------------
// CsvData — parsed CSV contents
// ---------------------------------------------------------------------------

/// Parsed CSV data with headers and typed column access.
pub struct CsvData {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl CsvData {
    /// Parse CSV from any reader with default options (comma-separated, has header).
    #[allow(dead_code)]
    pub fn from_reader(rdr: impl Read) -> Result<Self, String> {
        Self::from_reader_with_opts(rdr, &CsvParseOptions::default())
    }

    /// Parse CSV from any reader with the given options.
    pub fn from_reader_with_opts(rdr: impl Read, opts: &CsvParseOptions) -> Result<Self, String> {
        let mut csv_rdr = csv::ReaderBuilder::new()
            .delimiter(opts.delimiter)
            .has_headers(opts.has_header)
            .flexible(true)
            .trim(csv::Trim::All)
            .from_reader(rdr);

        let headers: Vec<String> = if opts.has_header {
            let h = csv_rdr
                .headers()
                .map_err(|e| format!("failed to read CSV headers: {e}"))?;
            h.iter().map(|s| s.to_string()).collect()
        } else {
            // Peek at the first record to determine column count, then generate
            // synthetic headers.  We'll store the first record below.
            Vec::new() // filled after first record peek
        };

        let mut rows: Vec<Vec<String>> = Vec::new();
        let mut skipped = 0usize;

        for result in csv_rdr.records() {
            let record =
                result.map_err(|e| format!("CSV parse error at row {}: {e}", rows.len() + 1))?;
            if skipped < opts.skip_rows {
                skipped += 1;
                continue;
            }
            if rows.len() >= opts.max_rows {
                eprintln!(
                    "warning: capped at {} rows (use --max-rows to increase)",
                    opts.max_rows
                );
                break;
            }
            rows.push(record.iter().map(|s| s.to_string()).collect());
        }

        // Handle no-header mode: generate synthetic names from the first row width
        let headers = if headers.is_empty() {
            if let Some(first) = rows.first() {
                (0..first.len()).map(|i| format!("col{i}")).collect()
            } else {
                return Err("CSV is empty (no data)".into());
            }
        } else {
            headers
        };

        if headers.is_empty() {
            return Err("CSV has no columns".into());
        }
        if rows.is_empty() {
            return Err("CSV has headers but no data rows".into());
        }

        Ok(Self { headers, rows })
    }

    // -- Column access -------------------------------------------------------

    /// Get the index of a column by header name (case-insensitive).
    fn column_index(&self, name: &str) -> Option<usize> {
        self.headers
            .iter()
            .position(|h| h.eq_ignore_ascii_case(name))
    }

    /// Extract a column as strings.
    pub fn string_column(&self, name: &str) -> Result<Vec<String>, String> {
        let idx = self.column_index(name).ok_or_else(|| {
            format!(
                "column '{}' not found. Available: {}",
                name,
                self.headers.join(", ")
            )
        })?;
        let expected = self.headers.len();
        let mut ragged_warned = false;
        Ok(self
            .rows
            .iter()
            .enumerate()
            .map(|(row_num, row)| {
                if !ragged_warned && row.len() != expected {
                    eprintln!(
                        "warning: row {} has {} fields but header has {} — data may be misaligned",
                        row_num + 1,
                        row.len(),
                        expected
                    );
                    ragged_warned = true;
                }
                row.get(idx).cloned().unwrap_or_default()
            })
            .collect())
    }

    /// Extract a column as f64 values, skipping unparseable entries.
    /// Prints a warning to stderr when values are skipped.
    pub fn numeric_column(&self, name: &str) -> Result<Vec<f64>, String> {
        let strings = self.string_column(name)?;
        let total = strings.len();
        let values: Vec<f64> = strings
            .iter()
            .filter_map(|s| s.trim().parse::<f64>().ok())
            .collect();

        if values.is_empty() {
            return Err(format!("column '{}' contains no numeric values", name));
        }

        let skipped = total - values.len();
        if skipped > 0 {
            eprintln!("warning: skipped {skipped} non-numeric value(s) in column '{name}'");
        }

        Ok(values)
    }

    /// Check if a column is predominantly numeric (>50% of values parse as f64).
    fn is_numeric_column(&self, idx: usize) -> bool {
        let total = self.rows.len();
        if total == 0 {
            return false;
        }
        let numeric_count = self
            .rows
            .iter()
            .filter(|row| {
                row.get(idx)
                    .map(|s| s.trim().parse::<f64>().is_ok())
                    .unwrap_or(false)
            })
            .count();
        numeric_count * 2 > total
    }

    /// Return headers of columns that are predominantly numeric.
    pub fn numeric_column_names(&self) -> Vec<String> {
        self.headers
            .iter()
            .enumerate()
            .filter(|(i, _)| self.is_numeric_column(*i))
            .map(|(_, h)| h.clone())
            .collect()
    }

    /// Available column headers.
    pub fn headers(&self) -> &[String] {
        &self.headers
    }

    /// Number of data rows.
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Sort rows by a column's numeric value (ascending).
    /// Non-numeric values in the sort column sort to the end.
    pub fn sort_by_column(&mut self, name: &str) -> Result<(), String> {
        let idx = self.column_index(name).ok_or_else(|| {
            format!(
                "sort column '{}' not found. Available: {}",
                name,
                self.headers.join(", ")
            )
        })?;
        self.rows.sort_by(|a, b| {
            let va = a
                .get(idx)
                .and_then(|s| s.trim().parse::<f64>().ok())
                .unwrap_or(f64::MAX);
            let vb = b
                .get(idx)
                .and_then(|s| s.trim().parse::<f64>().ok())
                .unwrap_or(f64::MAX);
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Conversion to ChartSpec
// ---------------------------------------------------------------------------

/// Options for building a ChartSpec from CSV data.
pub struct PlotOptions<'a> {
    pub chart_type: Option<ChartType>,
    pub x_col: Option<&'a str>,
    pub y_cols: &'a [String],
    pub title: Option<String>,
    pub x_label: Option<String>,
    pub y_label: Option<String>,
    pub theme: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub x_range: Option<(f64, f64)>,
    pub y_range: Option<(f64, f64)>,
    pub bins: Option<usize>,
    pub transparent_bg: bool,
}

/// Convert CSV data + user options into a ChartSpec for rendering.
pub fn csv_to_chart_spec(data: &CsvData, opts: PlotOptions<'_>) -> Result<ChartSpec, String> {
    let y_col_single = if opts.y_cols.len() == 1 {
        Some(opts.y_cols[0].as_str())
    } else {
        None
    };

    // Determine the chart type and build chart data
    let (resolved_type, chart_data) = if opts.y_cols.len() >= 2 {
        // Multi-series mode
        let ct = opts.chart_type.unwrap_or(ChartType::Line);
        let cd = build_multi_series(data, ct, opts.x_col, opts.y_cols)?;
        (ct, cd)
    } else if let Some(ct) = opts.chart_type {
        let cd = build_chart_data_for_type(data, ct, opts.x_col, y_col_single)?;
        (ct, cd)
    } else {
        auto_detect(data, opts.x_col, y_col_single)?
    };

    let effective_x_label = opts.x_label.or_else(|| opts.x_col.map(|s| s.to_string()));
    let effective_y_label = opts.y_label.or_else(|| {
        if opts.y_cols.len() == 1 {
            Some(opts.y_cols[0].clone())
        } else {
            None
        }
    });

    Ok(ChartSpec {
        chart_type: resolved_type,
        data: chart_data,
        title: opts.title,
        x_label: effective_x_label,
        y_label: effective_y_label,
        theme: opts.theme,
        width: opts.width,
        height: opts.height,
        x_range: opts.x_range,
        y_range: opts.y_range,
        bins: opts.bins,
        transparent_bg: opts.transparent_bg,
    })
}

// ---------------------------------------------------------------------------
// Multi-series construction
// ---------------------------------------------------------------------------

/// Build ChartData with multiple named series.
fn build_multi_series(
    data: &CsvData,
    ct: ChartType,
    x_col: Option<&str>,
    y_cols: &[String],
) -> Result<ChartData, String> {
    match ct {
        ChartType::Line | ChartType::Scatter => {
            let series: Vec<SeriesSpec> = y_cols
                .iter()
                .map(|name| {
                    let values = data.numeric_column(name)?;
                    Ok(SeriesSpec {
                        label: name.clone(),
                        values,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;

            let x = match x_col {
                Some(xn) => Some(data.numeric_column(xn)?),
                None => None,
            };

            Ok(ChartData {
                y: None,
                x,
                series: Some(series),
                labels: None,
                values: None,
                grid: None,
                groups: None,
                filled: None,
                points: None,
                smooth: None,
                step: None,
                ..Default::default()
            })
        }
        ChartType::Bar => {
            // Multi-series bar: each y column is a series, x is labels
            let x_name = x_col.ok_or("multi-series bar chart requires --x <label-column>")?;
            let labels = data.string_column(x_name)?;
            let series: Vec<SeriesSpec> = y_cols
                .iter()
                .map(|name| {
                    let values = data.numeric_column(name)?;
                    Ok(SeriesSpec {
                        label: name.clone(),
                        values,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;

            Ok(ChartData {
                y: None,
                x: None,
                series: Some(series),
                labels: Some(labels),
                values: None,
                grid: None,
                groups: None,
                filled: None,
                points: None,
                smooth: None,
                step: None,
                ..Default::default()
            })
        }
        ChartType::Boxplot | ChartType::Violin => {
            // Each y column becomes a group
            let groups: Vec<GroupSpec> = y_cols
                .iter()
                .map(|name| {
                    let values = data.numeric_column(name)?;
                    Ok(GroupSpec {
                        label: name.clone(),
                        values,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;

            Ok(ChartData {
                y: None,
                x: None,
                series: None,
                labels: None,
                values: None,
                grid: None,
                groups: Some(groups),
                filled: None,
                points: None,
                smooth: None,
                step: None,
                ..Default::default()
            })
        }
        _ => Err(format!(
            "multi-series is not supported for {ct} charts. Use a single --y column."
        )),
    }
}

// ---------------------------------------------------------------------------
// Single-series construction
// ---------------------------------------------------------------------------

/// Build ChartData for a specific chart type given the CSV and column selections.
fn build_chart_data_for_type(
    data: &CsvData,
    ct: ChartType,
    x_col: Option<&str>,
    y_col: Option<&str>,
) -> Result<ChartData, String> {
    match ct {
        ChartType::Line => {
            let y_name = y_col.ok_or(
                "line chart requires --y <column>. Available: ".to_string()
                    + &data.numeric_column_names().join(", "),
            )?;
            let y = data.numeric_column(y_name)?;
            let x = match x_col {
                Some(xn) => Some(data.numeric_column(xn)?),
                None => None,
            };
            Ok(ChartData {
                y: Some(y),
                x,
                series: None,
                labels: None,
                values: None,
                grid: None,
                groups: None,
                filled: None,
                points: None,
                smooth: None,
                step: None,
                ..Default::default()
            })
        }
        ChartType::Scatter => {
            let x_name = x_col.ok_or("scatter chart requires --x <column>")?;
            let y_name = y_col.ok_or("scatter chart requires --y <column>")?;
            let x = data.numeric_column(x_name)?;
            let y = data.numeric_column(y_name)?;
            Ok(ChartData {
                y: Some(y),
                x: Some(x),
                series: None,
                labels: None,
                values: None,
                grid: None,
                groups: None,
                filled: None,
                points: None,
                smooth: None,
                step: None,
                ..Default::default()
            })
        }
        ChartType::Bar => {
            let x_name = x_col.ok_or("bar chart requires --x <label-column>")?;
            let y_name = y_col.ok_or("bar chart requires --y <value-column>")?;
            let labels = data.string_column(x_name)?;
            let values = data.numeric_column(y_name)?;
            Ok(ChartData {
                y: None,
                x: None,
                series: None,
                labels: Some(labels),
                values: Some(values),
                grid: None,
                groups: None,
                filled: None,
                points: None,
                smooth: None,
                step: None,
                ..Default::default()
            })
        }
        ChartType::Histogram => {
            let col_name = y_col
                .or(x_col)
                .ok_or("histogram requires --y <column> (or --x)")?;
            let values = data.numeric_column(col_name)?;
            Ok(ChartData {
                y: None,
                x: None,
                series: None,
                labels: None,
                values: Some(values),
                grid: None,
                groups: None,
                filled: None,
                points: None,
                smooth: None,
                step: None,
                ..Default::default()
            })
        }
        ChartType::Boxplot => {
            let numeric_cols = data.numeric_column_names();
            if numeric_cols.is_empty() {
                return Err("boxplot requires at least one numeric column".into());
            }
            let groups: Vec<GroupSpec> = numeric_cols
                .iter()
                .map(|name| {
                    let values = data.numeric_column(name).unwrap_or_default();
                    GroupSpec {
                        label: name.clone(),
                        values,
                    }
                })
                .collect();
            Ok(ChartData {
                y: None,
                x: None,
                series: None,
                labels: None,
                values: None,
                grid: None,
                groups: Some(groups),
                filled: None,
                points: None,
                smooth: None,
                step: None,
                ..Default::default()
            })
        }
        ChartType::Pie => {
            let x_name = x_col.ok_or("pie chart requires --x <label-column>")?;
            let y_name = y_col.ok_or("pie chart requires --y <value-column>")?;
            let labels = data.string_column(x_name)?;
            let values = data.numeric_column(y_name)?;
            Ok(ChartData {
                y: None,
                x: None,
                series: None,
                labels: Some(labels),
                values: Some(values),
                grid: None,
                groups: None,
                filled: None,
                points: None,
                smooth: None,
                step: None,
                ..Default::default()
            })
        }
        ChartType::Heatmap => {
            Err("heatmap is not supported for CSV input (requires 2D grid data)".into())
        }
        ChartType::Waterfall | ChartType::Funnel | ChartType::Lollipop => {
            // Same shape as bar: labels + values
            let x_name = x_col.ok_or(format!("{ct} chart requires --x <label-column>"))?;
            let y_name = y_col.ok_or(format!("{ct} chart requires --y <value-column>"))?;
            let labels = data.string_column(x_name)?;
            let values = data.numeric_column(y_name)?;
            Ok(ChartData {
                labels: Some(labels),
                values: Some(values),
                ..Default::default()
            })
        }
        ChartType::Violin => {
            // Same shape as boxplot: groups of raw values
            let numeric_cols = data.numeric_column_names();
            if numeric_cols.is_empty() {
                return Err("violin plot requires at least one numeric column".into());
            }
            let groups: Vec<GroupSpec> = numeric_cols
                .iter()
                .map(|name| {
                    let values = data.numeric_column(name).unwrap_or_default();
                    GroupSpec {
                        label: name.clone(),
                        values,
                    }
                })
                .collect();
            Ok(ChartData {
                groups: Some(groups),
                ..Default::default()
            })
        }
        ChartType::Sparkline => {
            // Same as histogram: a single column of values
            let col_name = y_col
                .or(x_col)
                .ok_or("sparkline requires --y <column> (or --x)")?;
            let values = data.numeric_column(col_name)?;
            Ok(ChartData {
                values: Some(values),
                ..Default::default()
            })
        }
        ChartType::Radar
        | ChartType::Candlestick
        | ChartType::Bubble
        | ChartType::Gauge
        | ChartType::Gantt => Err(format!(
            "{ct} chart is not supported for CSV input (requires specialized data shape). \
                 Use JSON input with `scry chart render` instead."
        )),
    }
}

// ---------------------------------------------------------------------------
// Auto-detection
// ---------------------------------------------------------------------------

/// Auto-detect chart type from data shape and column selections.
fn auto_detect(
    data: &CsvData,
    x_col: Option<&str>,
    y_col: Option<&str>,
) -> Result<(ChartType, ChartData), String> {
    let numeric_cols = data.numeric_column_names();

    match (x_col, y_col) {
        (Some(x_name), Some(_)) => {
            let x_is_numeric = data.numeric_column(x_name).is_ok();
            if x_is_numeric {
                let cd = build_chart_data_for_type(data, ChartType::Scatter, x_col, y_col)?;
                Ok((ChartType::Scatter, cd))
            } else {
                let cd = build_chart_data_for_type(data, ChartType::Bar, x_col, y_col)?;
                Ok((ChartType::Bar, cd))
            }
        }
        (None, Some(_)) => {
            let cd = build_chart_data_for_type(data, ChartType::Line, None, y_col)?;
            Ok((ChartType::Line, cd))
        }
        (Some(x_name), None) => {
            let cd = build_chart_data_for_type(data, ChartType::Line, None, Some(x_name))?;
            Ok((ChartType::Line, cd))
        }
        (None, None) => {
            if numeric_cols.len() == 1 {
                let name = &numeric_cols[0];
                let cd = build_chart_data_for_type(data, ChartType::Histogram, None, Some(name))?;
                Ok((ChartType::Histogram, cd))
            } else if numeric_cols.len() >= 2 {
                let cd = build_chart_data_for_type(
                    data,
                    ChartType::Scatter,
                    Some(&numeric_cols[0]),
                    Some(&numeric_cols[1]),
                )?;
                eprintln!(
                    "auto-detected: scatter plot of '{}' vs '{}'",
                    numeric_cols[0], numeric_cols[1]
                );
                Ok((ChartType::Scatter, cd))
            } else {
                Err(
                    "no numeric columns found in CSV. Use --x and --y to specify columns.\nAvailable columns: "
                        .to_string()
                        + &data.headers().join(", "),
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sample_csv() -> &'static str {
        "month,revenue,expenses\nJan,12.4,8.2\nFeb,15.8,9.1\nMar,14.2,8.8\n"
    }

    // -- Basic parsing -------------------------------------------------------

    #[test]
    fn parse_csv_basic() {
        let data = CsvData::from_reader(Cursor::new(sample_csv())).unwrap();
        assert_eq!(data.headers(), &["month", "revenue", "expenses"]);
        assert_eq!(data.row_count(), 3);
    }

    #[test]
    fn numeric_column_extraction() {
        let data = CsvData::from_reader(Cursor::new(sample_csv())).unwrap();
        let rev = data.numeric_column("revenue").unwrap();
        assert_eq!(rev, vec![12.4, 15.8, 14.2]);
    }

    #[test]
    fn string_column_extraction() {
        let data = CsvData::from_reader(Cursor::new(sample_csv())).unwrap();
        let months = data.string_column("month").unwrap();
        assert_eq!(months, vec!["Jan", "Feb", "Mar"]);
    }

    #[test]
    fn numeric_column_names_detected() {
        let data = CsvData::from_reader(Cursor::new(sample_csv())).unwrap();
        let nums = data.numeric_column_names();
        assert_eq!(nums, vec!["revenue", "expenses"]);
    }

    #[test]
    fn case_insensitive_lookup() {
        let data = CsvData::from_reader(Cursor::new(sample_csv())).unwrap();
        assert!(data.numeric_column("Revenue").is_ok());
        assert!(data.numeric_column("REVENUE").is_ok());
    }

    #[test]
    fn missing_column_error() {
        let data = CsvData::from_reader(Cursor::new(sample_csv())).unwrap();
        let err = data.numeric_column("nonexistent").unwrap_err();
        assert!(err.contains("not found"));
        assert!(err.contains("revenue"));
    }

    // -- Auto-detection ------------------------------------------------------

    #[test]
    fn auto_detect_single_numeric() {
        let csv = "label,value\nA,1.0\nB,2.0\nC,3.0\n";
        let data = CsvData::from_reader(Cursor::new(csv)).unwrap();
        let (ct, _) = auto_detect(&data, None, None).unwrap();
        assert_eq!(ct, ChartType::Histogram);
    }

    #[test]
    fn auto_detect_two_numeric() {
        let data = CsvData::from_reader(Cursor::new(sample_csv())).unwrap();
        let (ct, _) = auto_detect(&data, None, None).unwrap();
        assert_eq!(ct, ChartType::Scatter);
    }

    #[test]
    fn auto_detect_y_only_is_line() {
        let data = CsvData::from_reader(Cursor::new(sample_csv())).unwrap();
        let (ct, _) = auto_detect(&data, None, Some("revenue")).unwrap();
        assert_eq!(ct, ChartType::Line);
    }

    #[test]
    fn auto_detect_string_x_is_bar() {
        let data = CsvData::from_reader(Cursor::new(sample_csv())).unwrap();
        let (ct, _) = auto_detect(&data, Some("month"), Some("revenue")).unwrap();
        assert_eq!(ct, ChartType::Bar);
    }

    #[test]
    fn csv_to_chart_spec_builds_correctly() {
        let data = CsvData::from_reader(Cursor::new(sample_csv())).unwrap();
        let spec = csv_to_chart_spec(
            &data,
            PlotOptions {
                chart_type: Some(ChartType::Line),
                x_col: None,
                y_cols: &["revenue".to_string()],
                title: Some("Test".into()),
                x_label: None,
                y_label: None,
                theme: None,
                width: None,
                height: None,
                x_range: None,
                y_range: None,
                bins: None,
                transparent_bg: false,
            },
        )
        .unwrap();
        assert_eq!(spec.chart_type, ChartType::Line);
        assert_eq!(spec.title.as_deref(), Some("Test"));
        assert!(spec.data.y.is_some());
    }

    // -- Phase 1: Delimiter & no-header --------------------------------------

    #[test]
    fn tsv_delimiter() {
        let tsv = "x\ty\n1\t10\n2\t20\n3\t30\n";
        let opts = CsvParseOptions {
            delimiter: b'\t',
            ..Default::default()
        };
        let data = CsvData::from_reader_with_opts(Cursor::new(tsv), &opts).unwrap();
        assert_eq!(data.headers(), &["x", "y"]);
        assert_eq!(data.numeric_column("y").unwrap(), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn pipe_delimiter() {
        let psv = "a|b|c\n1|2|3\n4|5|6\n";
        let opts = CsvParseOptions {
            delimiter: b'|',
            ..Default::default()
        };
        let data = CsvData::from_reader_with_opts(Cursor::new(psv), &opts).unwrap();
        assert_eq!(data.headers(), &["a", "b", "c"]);
        assert_eq!(data.numeric_column("b").unwrap(), vec![2.0, 5.0]);
    }

    #[test]
    fn no_header_mode() {
        let csv = "1,10\n2,20\n3,30\n";
        let opts = CsvParseOptions {
            has_header: false,
            ..Default::default()
        };
        let data = CsvData::from_reader_with_opts(Cursor::new(csv), &opts).unwrap();
        assert_eq!(data.headers(), &["col0", "col1"]);
        assert_eq!(data.row_count(), 3);
        assert_eq!(data.numeric_column("col1").unwrap(), vec![10.0, 20.0, 30.0]);
    }

    // -- Phase 1: Multi-series -----------------------------------------------

    #[test]
    fn multi_series_line() {
        let data = CsvData::from_reader(Cursor::new(sample_csv())).unwrap();
        let y_cols = vec!["revenue".to_string(), "expenses".to_string()];
        let (ct, cd) = {
            let ct = ChartType::Line;
            let cd = build_multi_series(&data, ct, None, &y_cols).unwrap();
            (ct, cd)
        };
        assert_eq!(ct, ChartType::Line);
        let series = cd.series.unwrap();
        assert_eq!(series.len(), 2);
        assert_eq!(series[0].label, "revenue");
        assert_eq!(series[1].label, "expenses");
    }

    #[test]
    fn multi_series_auto_labels() {
        let data = CsvData::from_reader(Cursor::new(sample_csv())).unwrap();
        let y_cols = vec!["revenue".to_string(), "expenses".to_string()];
        let cd = build_multi_series(&data, ChartType::Line, None, &y_cols).unwrap();
        let series = cd.series.unwrap();
        assert_eq!(series[0].label, "revenue");
        assert_eq!(series[1].label, "expenses");
        assert_eq!(series[0].values, vec![12.4, 15.8, 14.2]);
    }

    // -- Phase 2: Axis ranges / bins -----------------------------------------

    #[test]
    fn axis_range_passthrough() {
        let data = CsvData::from_reader(Cursor::new(sample_csv())).unwrap();
        let spec = csv_to_chart_spec(
            &data,
            PlotOptions {
                chart_type: Some(ChartType::Line),
                x_col: None,
                y_cols: &["revenue".to_string()],
                title: None,
                x_label: None,
                y_label: None,
                theme: None,
                width: None,
                height: None,
                x_range: Some((0.0, 100.0)),
                y_range: Some((0.0, 50.0)),
                bins: None,
                transparent_bg: false,
            },
        )
        .unwrap();
        assert_eq!(spec.x_range, Some((0.0, 100.0)));
        assert_eq!(spec.y_range, Some((0.0, 50.0)));
    }

    #[test]
    fn bins_passthrough() {
        let data = CsvData::from_reader(Cursor::new(sample_csv())).unwrap();
        let spec = csv_to_chart_spec(
            &data,
            PlotOptions {
                chart_type: Some(ChartType::Histogram),
                x_col: None,
                y_cols: &["revenue".to_string()],
                title: None,
                x_label: None,
                y_label: None,
                theme: None,
                width: None,
                height: None,
                x_range: None,
                y_range: None,
                bins: Some(5),
                transparent_bg: false,
            },
        )
        .unwrap();
        assert_eq!(spec.bins, Some(5));
    }

    // -- Phase 3: Skip rows / max rows / sort --------------------------------

    #[test]
    fn skip_rows() {
        let csv = "x,y\n1,10\n2,20\n3,30\n4,40\n";
        let opts = CsvParseOptions {
            skip_rows: 2,
            ..Default::default()
        };
        let data = CsvData::from_reader_with_opts(Cursor::new(csv), &opts).unwrap();
        assert_eq!(data.row_count(), 2);
        assert_eq!(data.numeric_column("x").unwrap(), vec![3.0, 4.0]);
    }

    #[test]
    fn max_rows_cap() {
        let csv = "x,y\n1,10\n2,20\n3,30\n4,40\n5,50\n";
        let opts = CsvParseOptions {
            max_rows: 3,
            ..Default::default()
        };
        let data = CsvData::from_reader_with_opts(Cursor::new(csv), &opts).unwrap();
        assert_eq!(data.row_count(), 3);
    }

    #[test]
    fn sort_by_column_works() {
        let csv = "name,value\nC,30\nA,10\nB,20\n";
        let mut data = CsvData::from_reader(Cursor::new(csv)).unwrap();
        data.sort_by_column("value").unwrap();
        assert_eq!(
            data.numeric_column("value").unwrap(),
            vec![10.0, 20.0, 30.0]
        );
        assert_eq!(data.string_column("name").unwrap(), vec!["A", "B", "C"]);
    }

    // -- Edge cases ----------------------------------------------------------

    #[test]
    fn numeric_column_with_nulls() {
        let csv = "x,y\n1,10\n2,N/A\n3,30\n4,\n5,null\n";
        let data = CsvData::from_reader(Cursor::new(csv)).unwrap();
        let vals = data.numeric_column("y").unwrap();
        assert_eq!(vals, vec![10.0, 30.0]);
    }

    #[test]
    fn empty_csv_error() {
        let csv = "";
        let result = CsvData::from_reader(Cursor::new(csv));
        assert!(result.is_err());
    }

    #[test]
    fn single_row_csv() {
        let csv = "x,y\n42,99\n";
        let data = CsvData::from_reader(Cursor::new(csv)).unwrap();
        assert_eq!(data.row_count(), 1);
        assert_eq!(data.numeric_column("x").unwrap(), vec![42.0]);
    }

    #[test]
    fn unicode_headers() {
        let csv = "名前,値\nA,1\nB,2\n";
        let data = CsvData::from_reader(Cursor::new(csv)).unwrap();
        assert_eq!(data.headers(), &["名前", "値"]);
        assert_eq!(data.numeric_column("値").unwrap(), vec![1.0, 2.0]);
    }
}
