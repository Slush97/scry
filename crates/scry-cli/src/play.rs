// SPDX-License-Identifier: MIT OR Apache-2.0
//! Play subcommand — `scry play`.
//!
//! Interactive TUI animation viewer.  Runs fullscreen animations using
//! the pixel canvas engine with ratatui and crossterm.

use clap::ValueEnum;

// ---------------------------------------------------------------------------
// CLI types
// ---------------------------------------------------------------------------

/// Available animation presets.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum PlayPreset {
    /// Sacred geometry animation
    Geometry,
    /// Wave interference pattern
    Wave,
    /// Fractal zoomer
    Fractal,
    /// Aurora borealis
    Aurora,
}

impl std::fmt::Display for PlayPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Geometry => write!(f, "geometry"),
            Self::Wave => write!(f, "wave"),
            Self::Fractal => write!(f, "fractal"),
            Self::Aurora => write!(f, "aurora"),
        }
    }
}

/// CLI arguments for the play subcommand.
#[derive(Debug, clap::Args)]
pub struct PlayArgs {
    /// Animation preset to play
    #[arg(short, long, default_value = "geometry")]
    pub preset: PlayPreset,

    /// Auto-exit after this many seconds (0 = run until 'q')
    #[arg(short, long, default_value = "0")]
    pub duration: u64,

    /// Target frames per second
    #[arg(long, default_value = "30")]
    pub fps: u32,
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

pub fn run(args: &PlayArgs) -> Result<(), String> {
    eprintln!("scry play: interactive TUI animations");
    eprintln!();
    eprintln!("  Preset: {}", args.preset);
    eprintln!("  FPS:    {}", args.fps);
    if args.duration > 0 {
        eprintln!("  Duration: {}s", args.duration);
    } else {
        eprintln!("  Duration: until 'q' is pressed");
    }
    eprintln!();
    eprintln!("  This feature is coming soon. For now, try the examples:");
    eprintln!("    cargo run --example sacred_geometry");
    eprintln!("    cargo run --example wave_interference");
    eprintln!("    cargo run --example aurora_borealis");
    eprintln!();
    eprintln!("Available presets:");
    eprintln!("  geometry   Sacred geometry (Flower of Life, Metatron's Cube)");
    eprintln!("  wave       Wave interference pattern");
    eprintln!("  fractal    Fractal zoomer");
    eprintln!("  aurora     Aurora borealis");
    Ok(())
}
