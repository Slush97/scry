// SPDX-License-Identifier: MIT OR Apache-2.0
//! # scry — pixel-perfect terminal graphics
//!
//! Unified CLI for charts, splash animations, image rendering, and animation presets.
//!
//! ```bash
//! # Chart from JSON
//! echo '{"type":"line","data":{"y":[1,4,2,8,5]}}' | scry chart render
//!
//! # Chart from CSV
//! cat data.csv | scry chart plot -y revenue,expenses
//!
//! # Display an image inline
//! scry render image.png
//!
//! # Terminal info
//! scry info
//! ```

mod chart;
mod csv;
mod display;
mod examples;

mod play;
mod render_image;
mod see;
mod spec;
mod stream;

mod viz;

use clap::{Parser, Subcommand};

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

/// scry — pixel-perfect terminal graphics, charts, and animations
#[derive(Parser, Debug)]
#[command(name = "scry", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Charting commands (render, plot, example, show, info)
    Chart {
        #[command(subcommand)]
        cmd: Box<chart::ChartCommands>,
    },

    /// Display an image file inline in the terminal
    Render(render_image::RenderArgs),

    /// Play interactive fullscreen animations
    Play(play::PlayArgs),

    /// View SDF shapes inline (`scry see cube`, `scry see torus`, …)
    See(see::SeeArgs),

    /// Live streaming chart from stdin
    Stream(stream::StreamArgs),

    /// 3D visualization commands (scatter, etc.)
    Viz {
        #[command(subcommand)]
        cmd: Box<viz::VizCommands>,
    },


    /// Print terminal capabilities and supported features
    Info,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Chart { cmd } => chart::run(*cmd),

        Commands::Render(args) => render_image::run(&args),
        Commands::Play(args)   => play::run(&args),
        Commands::See(args)    => see::run(&args),
        Commands::Stream(args) => stream::run(&args),
        Commands::Viz { cmd }  => viz::run(*cmd),
        Commands::Info         => cmd_info(),
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Info command (top-level)
// ---------------------------------------------------------------------------

fn cmd_info() -> Result<(), String> {
    println!("scry — pixel-perfect terminal graphics");
    println!("  Engine: scry-engine v{}", env!("CARGO_PKG_VERSION"));
    println!();

    // Terminal info
    let term = std::env::var("TERM").unwrap_or_else(|_| "unknown".into());
    let term_program = std::env::var("TERM_PROGRAM").unwrap_or_else(|_| "unknown".into());

    println!("Terminal:");
    println!("  TERM={term}");
    println!("  TERM_PROGRAM={term_program}");
    if std::env::var("SCRY_TERMINAL_SOCK").is_ok() {
        println!("  scry-terminal: ✓ (native shared-memory IPC)");
    }
    if let Some(pid) = std::env::var("KITTY_PID").ok() {
        println!("  KITTY_PID={pid}");
    }
    let driver = display::FrameDriver::detect();
    println!("  Protocol: {}", driver.protocol());
    println!(
        "  Inline images: {}",
        if driver.supports_inline() {
            "✓ supported"
        } else {
            "✗ not detected (halfblock fallback)"
        }
    );
    println!();

    // Available commands
    println!("Available commands:");
    println!("  scry chart render    Render chart from JSON");
    println!("  scry chart plot      Plot data from CSV");
    println!("  scry chart example   Built-in demo charts");
    println!("  scry chart show      Display PNG inline");
    println!("  scry render          Display image inline");
    println!("  scry stream          Live streaming chart from stdin");
    println!("  scry play            Interactive TUI animations & illusions");
    println!("  scry play -p illusion  Render optical illusions inline");
    println!("  scry see <shape>     View SDF shapes (cube, torus, gyroid, …)");
    println!("  scry see cube --low-res  Low-res mode (200×200)");
    println!("  scry info            Terminal capabilities");
    println!();

    // Chart types
    println!("Supported chart types (17):");
    println!("  line, scatter, bar, histogram, boxplot, heatmap, pie,");
    println!("  radar, candlestick, bubble, violin, sparkline,");
    println!("  waterfall, funnel, gauge, lollipop");
    println!();

    // Themes
    println!("Themes: dark (default), light, pastel, ocean, forest, colorblind");

    Ok(())
}
