// SPDX-License-Identifier: MIT OR Apache-2.0
//! See subcommand — `scry see <shape>`.
//!
//! Quick SDF shape viewer. Renders a single SDF object inline in the
//! terminal at native resolution (auto-detected, 200×200 with `--low-res`).

use crate::play::{self, sdf_default_res, SdfRunParams, SDF_LOW_RES};
use clap::ValueEnum;

// ---------------------------------------------------------------------------
// CLI types
// ---------------------------------------------------------------------------

/// SDF shapes available via `scry see <shape>`.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum SeeShape {
    /// Spinning 3D cube with rainbow gradient
    Cube,
    /// Hypnotic glass torus with fire core
    Vortex,
    /// Breathing organic orb (smooth-blended spheres)
    Pulse,
    /// Cosmic orbital system (mirror sphere + orbiting bodies)
    Orbit,
    /// Rainbow chrome torus sliced to reveal trippy swirl
    Torus,
    /// Psychedelic mirror with swirling 3D text
    Mirror,
    /// Mandelbulb fractal in rainbow chrome
    Mandelbulb,
    /// Glass Menger sponge with fire inside
    Menger,
    /// Gyroid minimal surface in rainbow
    Gyroid,
    /// Gradient descent on a 3D loss landscape
    GradientDescent,
    /// Neural network layers pulsing with signal propagation
    NeuralNet,
    /// K-Means clustering with converging centroids
    KMeans,
    /// GPU SDF 3D extruded text (use --message to set text)
    Text,
}

/// Material preset for the text shape.
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum TextMaterial {
    /// Animated rainbow spectrum
    #[default]
    Rainbow,
    /// Mirror chrome finish
    Chrome,
    /// Transparent glass with refraction
    Glass,
    /// Volumetric fire
    Fire,
    /// Clean white matte
    Matte,
}

/// CLI arguments for `scry see`.
#[derive(Debug, clap::Args)]
pub struct SeeArgs {
    /// Shape to render
    pub shape: SeeShape,

    /// Use low-resolution rendering (200×200)
    #[arg(long)]
    pub low_res: bool,

    /// Output width in pixels (overrides default)
    #[arg(short = 'W', long)]
    pub width: Option<u32>,

    /// Output height in pixels (overrides default)
    #[arg(short = 'H', long)]
    pub height: Option<u32>,

    /// Target frames per second
    #[arg(long, default_value = "30")]
    pub fps: u32,

    /// Auto-exit after this many seconds (0 = run until any key)
    #[arg(short, long, default_value = "0")]
    pub duration: u64,

    /// Text to render (for the `text` shape)
    #[arg(trailing_var_arg = true, default_value = "SCRY")]
    pub message: Vec<String>,

    /// Material preset for the `text` shape
    #[arg(long, value_enum, default_value = "rainbow")]
    pub material: TextMaterial,

    /// Enable live animation loop (arrow keys to rotate)
    #[arg(long)]
    pub animate: bool,

    /// Add wiggle effect to the text (gentle oscillation)
    #[arg(long)]
    pub wiggle: bool,

    /// Add warp effect to the text (psychedelic distortion)
    #[arg(long)]
    pub warp: bool,
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

pub fn run(args: &SeeArgs) -> Result<(), String> {
    let default = if args.low_res {
        SDF_LOW_RES
    } else {
        sdf_default_res()
    };
    let params = SdfRunParams {
        width: args.width.unwrap_or(default),
        height: args.height.unwrap_or(default),
        fps: args.fps,
        duration: args.duration,
    };

    match args.shape {
        SeeShape::Cube => play::run_cube(&params),
        SeeShape::Vortex => play::run_vortex(&params),
        SeeShape::Pulse => play::run_pulse(&params),
        SeeShape::Orbit => play::run_orbit(&params),
        SeeShape::Torus => play::run_torus(&params),
        SeeShape::Mirror => play::run_mirror(&params),
        SeeShape::Mandelbulb => play::run_mandelbulb(&params),
        SeeShape::Menger => play::run_menger(&params),
        SeeShape::Gyroid => play::run_gyroid(&params),
        SeeShape::GradientDescent => play::run_gradient_descent(&params),
        SeeShape::NeuralNet => play::run_neural_net(&params),
        SeeShape::KMeans => play::run_kmeans(&params),
        SeeShape::Text => play::run_text(
            &params,
            &play::TextOptions {
                message: args.message.join(" "),
                material: args.material,
                animate: args.animate,
                wiggle: args.wiggle,
                warp: args.warp,
            },
        ),
    }
}
