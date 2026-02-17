// SPDX-License-Identifier: MIT OR Apache-2.0
//! Splash subcommand — `scry splash`.
//!
//! Productized version of the `startup_anim` example.  Renders a sacred
//! geometry animation in the top rows of the terminal and forks to the
//! background so the shell prompt appears immediately.

use clap::ValueEnum;

// ---------------------------------------------------------------------------
// CLI types
// ---------------------------------------------------------------------------

/// Available splash animation presets.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum SplashPreset {
    /// Sacred geometry animation (Flower of Life, Metatron's Cube)
    Geometry,
    /// Sine-wave interference pattern
    Wave,
    /// Floating neon particles
    Particles,
    /// Minimal loading dots
    Minimal,
}

impl std::fmt::Display for SplashPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Geometry => write!(f, "geometry"),
            Self::Wave => write!(f, "wave"),
            Self::Particles => write!(f, "particles"),
            Self::Minimal => write!(f, "minimal"),
        }
    }
}

/// CLI arguments for the splash subcommand.
#[derive(Debug, clap::Args)]
pub struct SplashArgs {
    /// Animation preset to play
    #[arg(short, long, default_value = "geometry")]
    pub preset: SplashPreset,

    /// Number of terminal rows to use for the animation
    #[arg(short, long, default_value = "12")]
    pub rows: u16,

    /// Duration in seconds (0 = loop forever until stdout pipe closes)
    #[arg(short, long, default_value = "0")]
    pub duration: u64,

    /// Color palette
    #[arg(long, default_value = "default")]
    pub palette: String,

    /// Skip fastfetch integration (just show the animation)
    #[arg(long)]
    pub no_fastfetch: bool,

    /// Run in foreground instead of forking to background
    #[arg(long)]
    pub foreground: bool,
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

pub fn run(args: &SplashArgs) -> Result<(), String> {
    match args.preset {
        SplashPreset::Geometry => {
            eprintln!("scry splash: geometry preset");
            eprintln!();
            eprintln!("  This preset requires a Kitty-compatible terminal and renders a");
            eprintln!("  sacred geometry animation in the terminal's top rows.");
            eprintln!();
            eprintln!("  For the full experience, run the standalone example:");
            eprintln!("    cargo run --example fastfetch_anim");
            eprintln!();
            eprintln!("  Standalone CLI integration coming soon.");
            Ok(())
        }
        preset => {
            eprintln!("scry splash: preset '{}' coming soon.", preset);
            eprintln!();
            eprintln!("Available presets:");
            eprintln!("  geometry   Sacred geometry (Flower of Life, Metatron's Cube)");
            eprintln!("  wave       Sine-wave interference pattern");
            eprintln!("  particles  Floating neon particles");
            eprintln!("  minimal    Minimal loading dots");
            Ok(())
        }
    }
}
