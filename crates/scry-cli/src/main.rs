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
//! # Splash animation
//! scry splash --preset geometry
//!
//! # Terminal info
//! scry info
//! ```

mod chart;
mod csv;
mod examples;
mod inline;
mod play;
mod render_image;
mod spec;
mod splash;

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

    /// Display a startup splash animation in the terminal
    Splash(splash::SplashArgs),

    /// Display an image file inline in the terminal
    Render(render_image::RenderArgs),

    /// Play interactive fullscreen animations
    Play(play::PlayArgs),

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
        Commands::Splash(args) => splash::run(&args),
        Commands::Render(args) => render_image::run(&args),
        Commands::Play(args) => play::run(&args),
        Commands::Info => cmd_info(),
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

    // Available commands
    println!("Available commands:");
    println!("  scry chart render    Render chart from JSON");
    println!("  scry chart plot      Plot data from CSV");
    println!("  scry chart example   Built-in demo charts");
    println!("  scry chart show      Display PNG inline");
    println!("  scry splash          Startup splash animation");
    println!("  scry render          Display image inline");
    println!("  scry play            Interactive TUI animations");
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
