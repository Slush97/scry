// SPDX-License-Identifier: MIT OR Apache-2.0
//! Chart subcommand group — `scry chart render|plot|example|show`.
//!
//! This module wraps all chart-related CLI functionality that was previously
//! the entire CLI.  Each subcommand is a handler function that reads input,
//! builds a `scry_chart::Chart`, renders it to PNG, and either saves to a
//! file or displays it inline in the terminal.

use clap::Subcommand;
use std::io::Read;

use crate::csv;
use crate::examples;
use crate::inline;
use crate::spec::{ChartSpec, ChartType};

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

#[derive(Subcommand, Debug)]
pub enum ChartCommands {
    /// Render a chart from JSON (stdin or --data)
    Render {
        /// Chart type (overrides "type" field in JSON)
        #[arg(short = 't', long)]
        r#type: Option<ChartType>,

        /// JSON data string (if omitted, reads from stdin)
        #[arg(short, long)]
        data: Option<String>,

        /// Chart title (overrides "title" field in JSON)
        #[arg(long)]
        title: Option<String>,

        /// X-axis label
        #[arg(long)]
        x_label: Option<String>,

        /// Y-axis label
        #[arg(long)]
        y_label: Option<String>,

        /// Image width in pixels
        #[arg(short = 'W', long, default_value = "800")]
        width: u32,

        /// Image height in pixels
        #[arg(short = 'H', long, default_value = "500")]
        height: u32,

        /// Theme: "dark" or "light"
        #[arg(long, default_value = "dark")]
        theme: String,

        /// Output file path (if omitted, displays inline in terminal)
        #[arg(short, long)]
        output: Option<String>,

        /// Export DPI (default: 144, use 288 for Retina)
        #[arg(long, default_value = "144")]
        dpi: u32,
    },

    /// Display an existing PNG image inline in the terminal
    Show {
        /// Path to PNG file
        path: String,
    },

    /// Render a built-in example chart (no JSON needed)
    Example {
        /// Chart type to demo (omit for gallery of all types)
        chart_type: Option<ChartType>,

        /// Image width in pixels
        #[arg(short = 'W', long, default_value = "800")]
        width: u32,

        /// Image height in pixels
        #[arg(short = 'H', long, default_value = "500")]
        height: u32,

        /// Theme: "dark" or "light"
        #[arg(long, default_value = "dark")]
        theme: String,

        /// Output file path (if omitted, displays inline in terminal)
        #[arg(short, long)]
        output: Option<String>,

        /// Export DPI (default: 144, use 288 for Retina)
        #[arg(long, default_value = "144")]
        dpi: u32,
    },

    /// Plot data from CSV (stdin or --csv file)
    Plot {
        /// Chart type (auto-detected if omitted)
        #[arg(short = 't', long)]
        r#type: Option<ChartType>,

        /// CSV file path (reads stdin if omitted)
        #[arg(long)]
        csv: Option<String>,

        /// Column name for X axis
        #[arg(short = 'x', long)]
        x: Option<String>,

        /// Column name(s) for Y axis (comma-separated for multi-series)
        #[arg(short = 'y', long, value_delimiter = ',')]
        y: Vec<String>,

        /// Chart title
        #[arg(long)]
        title: Option<String>,

        /// X-axis label (defaults to column name)
        #[arg(long)]
        x_label: Option<String>,

        /// Y-axis label (defaults to column name)
        #[arg(long)]
        y_label: Option<String>,

        /// Image width in pixels
        #[arg(short = 'W', long, default_value = "800")]
        width: u32,

        /// Image height in pixels
        #[arg(short = 'H', long, default_value = "500")]
        height: u32,

        /// Theme: "dark", "light", "pastel", "ocean", "forest"
        #[arg(long, default_value = "dark")]
        theme: String,

        /// Output file path (displays inline if omitted)
        #[arg(short, long)]
        output: Option<String>,

        /// Field delimiter character (default: comma). Use 'tab' for TSV.
        #[arg(long, default_value = ",")]
        delimiter: String,

        /// CSV has no header row (columns will be named col0, col1, ...)
        #[arg(long)]
        no_header: bool,

        /// Override Y-axis minimum value
        #[arg(long)]
        y_min: Option<f64>,

        /// Override Y-axis maximum value
        #[arg(long)]
        y_max: Option<f64>,

        /// Override X-axis minimum value
        #[arg(long)]
        x_min: Option<f64>,

        /// Override X-axis maximum value
        #[arg(long)]
        x_max: Option<f64>,

        /// Number of histogram bins (default: auto)
        #[arg(long)]
        bins: Option<usize>,

        /// Skip this many data rows after the header
        #[arg(long, default_value = "0")]
        skip_rows: usize,

        /// Maximum number of data rows to read
        #[arg(long, default_value = "100000")]
        max_rows: usize,

        /// Sort data by X column before plotting
        #[arg(long)]
        sort: bool,

        /// Transparent background (no fill behind chart)
        #[arg(long)]
        no_bg: bool,

        /// Export DPI (default: 144, use 288 for Retina)
        #[arg(long, default_value = "144")]
        dpi: u32,
    },

    /// Print terminal capabilities and chart type info
    Info,
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

pub fn run(cmd: ChartCommands) -> Result<(), String> {
    match cmd {
        ChartCommands::Render {
            r#type,
            data,
            title,
            x_label,
            y_label,
            width,
            height,
            theme,
            output,
            dpi,
        } => cmd_render(
            r#type, data, title, x_label, y_label, width, height, theme, output, dpi,
        ),
        ChartCommands::Show { path } => cmd_show(&path),
        ChartCommands::Example {
            chart_type,
            width,
            height,
            theme,
            output,
            dpi,
        } => cmd_example(chart_type, width, height, theme, output, dpi),
        ChartCommands::Plot {
            r#type,
            csv,
            x,
            y,
            title,
            x_label,
            y_label,
            width,
            height,
            theme,
            output,
            delimiter,
            no_header,
            y_min,
            y_max,
            x_min,
            x_max,
            bins,
            skip_rows,
            max_rows,
            sort,
            no_bg,
            dpi,
        } => cmd_plot(
            r#type, csv, x, y, title, x_label, y_label, width, height, theme, output, delimiter,
            no_header, y_min, y_max, x_min, x_max, bins, skip_rows, max_rows, sort, no_bg, dpi,
        ),
        ChartCommands::Info => cmd_info(),
    }
}

// ---------------------------------------------------------------------------
// Command handlers (moved from main.rs)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn cmd_render(
    chart_type: Option<ChartType>,
    data: Option<String>,
    title: Option<String>,
    x_label: Option<String>,
    y_label: Option<String>,
    width: u32,
    height: u32,
    theme: String,
    output: Option<String>,
    dpi: u32,
) -> Result<(), String> {
    // Read JSON from --data flag or stdin
    let json_str = if let Some(d) = data {
        d
    } else {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format!("failed to read stdin: {e}"))?;
        buf
    };

    // Parse as ChartSpec
    let mut spec: ChartSpec =
        serde_json::from_str(&json_str).map_err(|e| format!("invalid JSON: {e}"))?;

    // CLI flags override JSON fields
    if let Some(t) = chart_type {
        spec.chart_type = t;
    }
    if title.is_some() {
        spec.title = title;
    }
    if x_label.is_some() {
        spec.x_label = x_label;
    }
    if y_label.is_some() {
        spec.y_label = y_label;
    }
    if theme != "dark" {
        spec.theme = Some(theme);
    }
    spec.width = Some(spec.width.unwrap_or(width));
    spec.height = Some(spec.height.unwrap_or(height));

    let w = spec.width.unwrap_or(800);
    let h = spec.height.unwrap_or(500);

    // Build chart
    let mut chart = spec.into_chart()?;

    // Inject DPI into chart config
    if dpi != 144 {
        if let Some(cfg) = chart.config_mut() { cfg.export.dpi = dpi; }
    }

    // Render to PNG
    let png_data = scry_chart::export::render_to_png(&chart, w, h)?;

    if let Some(path) = output {
        // Save to file
        std::fs::write(&path, &png_data).map_err(|e| format!("failed to write {path}: {e}"))?;
        eprintln!("✓ Saved {path} ({w}×{h}, {} bytes)", png_data.len());
    } else {
        // Display inline in terminal
        inline::display_inline_auto(&png_data)
            .map_err(|e| format!("inline display failed: {e}"))?;
    }

    Ok(())
}

fn cmd_show(path: &str) -> Result<(), String> {
    let png_data = std::fs::read(path).map_err(|e| format!("failed to read {path}: {e}"))?;

    inline::display_inline_auto(&png_data).map_err(|e| format!("inline display failed: {e}"))?;

    Ok(())
}

fn cmd_example(
    chart_type: Option<ChartType>,
    width: u32,
    height: u32,
    theme_name: String,
    output: Option<String>,
    dpi: u32,
) -> Result<(), String> {
    let theme = crate::spec::resolve_theme(Some(theme_name.as_str()));

    let types: Vec<ChartType> = if let Some(t) = chart_type {
        vec![t]
    } else {
        examples::all_types().to_vec()
    };

    for (i, ct) in types.iter().enumerate() {
        let mut chart = examples::build_example(*ct, theme.clone());
        if dpi != 144 {
            if let Some(cfg) = chart.config_mut() { cfg.export.dpi = dpi; }
        }
        let png_data = scry_chart::export::render_to_png(&chart, width, height)?;

        if let Some(ref path) = output {
            // For gallery mode, suffix the filename with the chart type
            let actual_path = if types.len() > 1 {
                let stem = path.trim_end_matches(".png");
                format!("{stem}_{ct}.png")
            } else {
                path.clone()
            };
            std::fs::write(&actual_path, &png_data)
                .map_err(|e| format!("failed to write {actual_path}: {e}"))?;
            eprintln!("✓ Saved {actual_path} ({width}×{height})");
        } else {
            if types.len() > 1 {
                eprintln!("── {ct} ──");
            }
            inline::display_inline_auto(&png_data)
                .map_err(|e| format!("inline display failed: {e}"))?;
        }

        // Small separator between gallery items (except last)
        if types.len() > 1 && i < types.len() - 1 && output.is_none() {
            eprintln!();
        }
    }

    Ok(())
}

fn cmd_info() -> Result<(), String> {
    println!("scry chart — pixel-perfect terminal charts");
    println!();

    // Terminal info
    let term = std::env::var("TERM").unwrap_or_else(|_| "unknown".into());
    let term_program = std::env::var("TERM_PROGRAM").unwrap_or_else(|_| "unknown".into());
    let kitty_pid = std::env::var("KITTY_PID").ok();

    println!("Terminal:");
    println!("  TERM={term}");
    println!("  TERM_PROGRAM={term_program}");
    if let Some(pid) = &kitty_pid {
        println!("  KITTY_PID={pid}");
    }
    println!(
        "  Inline images: {}",
        if inline::terminal_supports_inline() {
            "✓ supported"
        } else {
            "✗ not detected (will attempt Kitty protocol anyway)"
        }
    );
    println!();

    // Chart types
    println!("Supported chart types (17):");
    println!();
    println!("  line         Line chart from y values");
    println!("               {{\"type\":\"line\",\"data\":{{\"y\":[1,4,2,8,5]}}}}");
    println!();
    println!("  scatter      Scatter plot from x,y pairs");
    println!("               {{\"type\":\"scatter\",\"data\":{{\"x\":[1,2,3],\"y\":[4,5,6]}}}}");
    println!();
    println!("  bar          Bar chart with labels");
    println!("               {{\"type\":\"bar\",\"data\":{{\"labels\":[\"A\",\"B\"],\"values\":[10,20]}}}}");
    println!();
    println!("  histogram    Histogram from raw values");
    println!("               {{\"type\":\"histogram\",\"data\":{{\"values\":[1.2,3.4,2.1,5.6]}}}}");
    println!();
    println!("  boxplot      Box plot from grouped data");
    println!("               {{\"type\":\"boxplot\",\"data\":{{\"groups\":[{{\"label\":\"A\",\"values\":[1,2,3]}}]}}}}");
    println!();
    println!("  heatmap      Heatmap from 2D grid");
    println!("               {{\"type\":\"heatmap\",\"data\":{{\"grid\":[[1,2],[3,4]]}}}}");
    println!();
    println!("  pie          Pie chart with labels");
    println!("               {{\"type\":\"pie\",\"data\":{{\"labels\":[\"A\",\"B\"],\"values\":[60,40]}}}}");
    println!();
    println!("  radar        Radar/spider chart");
    println!("               {{\"type\":\"radar\",\"data\":{{\"axes\":[\"A\",\"B\",\"C\"],\"radar_series\":[{{\"label\":\"S1\",\"values\":[1,2,3]}}]}}}}");
    println!();
    println!("  candlestick  OHLC candlestick chart");
    println!("               {{\"type\":\"candlestick\",\"data\":{{\"ohlc\":[{{\"open\":10,\"high\":15,\"low\":8,\"close\":12}}]}}}}");
    println!();
    println!("  bubble       Bubble chart (x,y,size)");
    println!("               {{\"type\":\"bubble\",\"data\":{{\"x\":[1,2],\"y\":[3,4],\"sizes\":[10,20]}}}}");
    println!();
    println!("  violin       Violin plot from grouped data");
    println!("               {{\"type\":\"violin\",\"data\":{{\"groups\":[{{\"label\":\"A\",\"values\":[1,2,3]}}]}}}}");
    println!();
    println!("  sparkline    Minimal inline sparkline");
    println!("               {{\"type\":\"sparkline\",\"data\":{{\"values\":[3,7,4,8,2,9]}}}}");
    println!();
    println!("  waterfall    Waterfall (P&L) chart");
    println!("               {{\"type\":\"waterfall\",\"data\":{{\"labels\":[\"Rev\",\"Cost\"],\"values\":[500,-200]}}}}");
    println!();
    println!("  funnel       Funnel / conversion pipeline");
    println!("               {{\"type\":\"funnel\",\"data\":{{\"labels\":[\"Visit\",\"Signup\"],\"values\":[1000,500]}}}}");
    println!();
    println!("  gauge        KPI gauge / speedometer");
    println!("               {{\"type\":\"gauge\",\"data\":{{\"value\":73}}}}");
    println!();
    println!("  lollipop     Lollipop / dot plot");
    println!("               {{\"type\":\"lollipop\",\"data\":{{\"labels\":[\"A\",\"B\"],\"values\":[10,20]}}}}");
    println!();

    // Options
    println!("Common options:");
    println!("  --title TEXT           Chart title");
    println!("  --x-label TEXT         X-axis label");
    println!("  --y-label TEXT         Y-axis label");
    println!("  --theme NAME           Color theme (default: dark)");
    println!("  --width PIXELS         Image width (default: 800)");
    println!("  --height PIXELS        Image height (default: 500)");
    println!("  --dpi VALUE            Export DPI (default: 144, 288 for Retina)");
    println!("  --output FILE          Save to PNG instead of inline display");
    println!();
    println!("Themes: dark (default), light, pastel, ocean, forest, colorblind");

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_plot(
    chart_type: Option<ChartType>,
    csv_path: Option<String>,
    x_col: Option<String>,
    y_cols: Vec<String>,
    title: Option<String>,
    x_label: Option<String>,
    y_label: Option<String>,
    width: u32,
    height: u32,
    theme: String,
    output: Option<String>,
    delimiter: String,
    no_header: bool,
    y_min: Option<f64>,
    y_max: Option<f64>,
    x_min: Option<f64>,
    x_max: Option<f64>,
    bins: Option<usize>,
    skip_rows: usize,
    max_rows: usize,
    sort: bool,
    no_bg: bool,
    dpi: u32,
) -> Result<(), String> {
    // Parse delimiter
    let delim_byte = match delimiter.as_str() {
        "tab" | "\t" => b'\t',
        s if s.len() == 1 => s.as_bytes()[0],
        _ => {
            return Err(format!(
                "invalid delimiter: '{}' (use a single character or 'tab')",
                delimiter
            ))
        }
    };

    let parse_opts = csv::CsvParseOptions {
        delimiter: delim_byte,
        has_header: !no_header,
        skip_rows,
        max_rows,
    };

    // Read CSV from file or stdin
    let mut csv_data = if let Some(ref path) = csv_path {
        let file = std::fs::File::open(path).map_err(|e| format!("failed to open {path}: {e}"))?;
        csv::CsvData::from_reader_with_opts(file, &parse_opts)?
    } else {
        let stdin = std::io::stdin();
        csv::CsvData::from_reader_with_opts(stdin.lock(), &parse_opts)?
    };

    eprintln!(
        "read {} rows × {} columns ({})",
        csv_data.row_count(),
        csv_data.headers().len(),
        csv_data.headers().join(", "),
    );

    // Sort by x column if requested
    if sort {
        if let Some(ref x) = x_col {
            csv_data.sort_by_column(x)?;
        } else if !y_cols.is_empty() {
            // Sort by first y column if no x specified
            csv_data.sort_by_column(&y_cols[0])?;
        }
    }

    // Build axis range overrides
    let x_range = match (x_min, x_max) {
        (Some(min), Some(max)) => Some((min, max)),
        (Some(min), None) => Some((min, f64::INFINITY)),
        (None, Some(max)) => Some((f64::NEG_INFINITY, max)),
        (None, None) => None,
    };
    let y_range = match (y_min, y_max) {
        (Some(min), Some(max)) => Some((min, max)),
        (Some(min), None) => Some((min, f64::INFINITY)),
        (None, Some(max)) => Some((f64::NEG_INFINITY, max)),
        (None, None) => None,
    };

    // Convert CSV to chart spec
    let theme_str = if theme == "dark" { None } else { Some(theme) };
    let spec = csv::csv_to_chart_spec(
        &csv_data,
        csv::PlotOptions {
            chart_type,
            x_col: x_col.as_deref(),
            y_cols: &y_cols,
            title,
            x_label,
            y_label,
            theme: theme_str,
            width: Some(width),
            height: Some(height),
            x_range,
            y_range,
            bins,
            transparent_bg: no_bg,
        },
    )?;

    // Build chart
    let mut chart = spec.into_chart()?;

    // Inject DPI
    if dpi != 144 {
        if let Some(cfg) = chart.config_mut() { cfg.export.dpi = dpi; }
    }

    // Render to PNG
    let png_data = scry_chart::export::render_to_png(&chart, width, height)?;

    if let Some(path) = output {
        std::fs::write(&path, &png_data).map_err(|e| format!("failed to write {path}: {e}"))?;
        eprintln!(
            "✓ Saved {path} ({width}×{height}, {} bytes)",
            png_data.len()
        );
    } else {
        inline::display_inline_auto(&png_data)
            .map_err(|e| format!("inline display failed: {e}"))?;
    }

    Ok(())
}
