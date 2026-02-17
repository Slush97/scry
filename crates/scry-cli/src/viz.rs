// SPDX-License-Identifier: MIT OR Apache-2.0
//! Visualization subcommand group — `scry viz 3d-scatter`.
//!
//! 3D visualization commands that use the `Chart3D` pipeline directly,
//! bypassing the 2D `ChartSpec → Chart` path.

use clap::Subcommand;

use crate::csv::{CsvData, CsvParseOptions};

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

/// Visualization commands for 3D charts and ML data exploration.
#[derive(Subcommand, Debug)]
pub enum VizCommands {
    /// Render an interactive 3D scatter plot from CSV data
    ///
    /// Points are plotted using three numeric columns for X, Y, and Z axes.
    /// Optionally color points by a categorical column.
    ///
    /// Examples:
    ///   scry viz 3d-scatter data.csv --x sepal_length --y sepal_width --z petal_length
    ///   scry viz 3d-scatter iris.csv --x f1 --y f2 --z f3 --color-by species
    #[command(name = "3d-scatter")]
    Scatter3D {
        /// CSV file path (reads stdin if omitted)
        csv: Option<String>,

        /// Column name for X axis
        #[arg(short = 'x', long)]
        x: String,

        /// Column name for Y axis
        #[arg(short = 'y', long)]
        y: String,

        /// Column name for Z axis
        #[arg(short = 'z', long)]
        z: String,

        /// Column name to color points by (categorical or numeric class)
        #[arg(long)]
        color_by: Option<String>,

        /// Chart title
        #[arg(long)]
        title: Option<String>,

        /// Image width in pixels
        #[arg(short = 'W', long, default_value = "800")]
        width: u32,

        /// Image height in pixels
        #[arg(short = 'H', long, default_value = "600")]
        height: u32,

        /// Output file path (interactive display if omitted)
        #[arg(short, long)]
        output: Option<String>,

        /// Point size in pixels
        #[arg(long, default_value = "6.0")]
        point_size: f32,

        /// Disable the XZ grid plane
        #[arg(long)]
        no_grid: bool,

        /// CSV delimiter character (default: comma). Use 'tab' for TSV.
        #[arg(long, default_value = ",")]
        delimiter: String,

        /// CSV has no header row (columns named col0, col1, ...)
        #[arg(long)]
        no_header: bool,
    },
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Run a viz subcommand.
pub fn run(cmd: VizCommands) -> Result<(), String> {
    match cmd {
        VizCommands::Scatter3D {
            csv,
            x,
            y,
            z,
            color_by,
            title,
            width,
            height,
            output,
            point_size,
            no_grid,
            delimiter,
            no_header,
        } => cmd_scatter_3d(
            csv, &x, &y, &z, color_by.as_deref(), title, width, height,
            output, point_size, no_grid, &delimiter, no_header,
        ),
    }
}

// ---------------------------------------------------------------------------
// 3D scatter handler
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn cmd_scatter_3d(
    csv_path: Option<String>,
    x_col: &str,
    y_col: &str,
    z_col: &str,
    color_by: Option<&str>,
    title: Option<String>,
    width: u32,
    height: u32,
    output: Option<String>,
    point_size: f32,
    no_grid: bool,
    delimiter: &str,
    no_header: bool,
) -> Result<(), String> {
    // Parse delimiter
    let delim_byte = match delimiter {
        "tab" | "\t" => b'\t',
        s if s.len() == 1 => s.as_bytes()[0],
        _ => {
            return Err(format!(
                "invalid delimiter: '{}' (use a single character or 'tab')",
                delimiter
            ))
        }
    };

    let parse_opts = CsvParseOptions {
        delimiter: delim_byte,
        has_header: !no_header,
        skip_rows: 0,
        max_rows: 1_000_000,
    };

    // Read CSV from file or stdin
    let csv_data = if let Some(ref path) = csv_path {
        let file =
            std::fs::File::open(path).map_err(|e| format!("failed to open {path}: {e}"))?;
        CsvData::from_reader_with_opts(file, &parse_opts)?
    } else {
        let stdin = std::io::stdin();
        CsvData::from_reader_with_opts(stdin.lock(), &parse_opts)?
    };



    // Extract numeric columns for X, Y, Z
    let x_data = csv_data.numeric_column(x_col)?;
    let y_data = csv_data.numeric_column(y_col)?;
    let z_data = csv_data.numeric_column(z_col)?;

    if x_data.is_empty() || y_data.is_empty() || z_data.is_empty() {
        return Err("one or more columns produced no numeric data".into());
    }

    // Build the Chart3D
    let mut chart = scry_chart::chart3d::Chart3D::scatter(&x_data, &y_data, &z_data)
        .x_label(x_col)
        .y_label(y_col)
        .z_label(z_col)
        .point_size(point_size)
        .grid(!no_grid);

    // Apply title
    if let Some(ref t) = title {
        chart = chart.title(t.as_str());
    } else if let Some(ref path) = csv_path {
        // Default title from filename
        let stem = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("3D Scatter");
        chart = chart.title(format!("3D Scatter — {stem}"));
    } else {
        chart = chart.title("3D Scatter");
    }

    // Color by categorical column
    if let Some(col) = color_by {
        let labels = csv_data.string_column(col)?;
        chart = chart.color_by_labels(&labels);
    }

    // Render or display
    if let Some(ref path) = output {
        eprintln!(
            "read {} rows × {} columns ({})",
            csv_data.row_count(),
            csv_data.headers().len(),
            csv_data.headers().join(", "),
        );
        if color_by.is_some() {
            eprintln!("coloring by '{}' ({} unique values)",
                color_by.unwrap_or("?"),
                {
                    let labels_col = csv_data.string_column(color_by.unwrap_or(""));
                    labels_col.map_or(0, |l| {
                        let mut uniq: Vec<&str> = l.iter().map(|s| s.as_str()).collect();
                        uniq.sort_unstable();
                        uniq.dedup();
                        uniq.len()
                    })
                });
        }
        chart
            .save_png(width, height, path)
            .map_err(|e| format!("failed to save PNG: {e}"))?;
        eprintln!("✓ Saved {path} ({width}×{height})");
    } else {
        chart.show().map_err(|e| format!("display failed: {e}"))?;
    }

    Ok(())
}
