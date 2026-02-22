// SPDX-License-Identifier: MIT OR Apache-2.0
//! Render subcommand — `scry render FILE`.
//!
//! Display images (PNG, JPEG) inline in the terminal using the
//! auto-detected graphics protocol.

use crate::display;

/// CLI arguments for the render subcommand.
#[derive(Debug, clap::Args)]
pub struct RenderArgs {
    /// Path to the image file (PNG, JPEG)
    pub path: String,

    /// Limit display width (terminal columns)
    #[arg(short = 'W', long)]
    pub width: Option<u32>,

    /// Limit display height (terminal rows)
    #[arg(short = 'H', long)]
    pub height: Option<u32>,
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

pub fn run(args: &RenderArgs) -> Result<(), String> {
    let path = &args.path;

    // Read the file
    let data = std::fs::read(path).map_err(|e| format!("failed to read {path}: {e}"))?;

    // Detect format by extension or magic bytes
    let lower = path.to_lowercase();
    let png_data = if lower.ends_with(".png") {
        data
    } else if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        // For JPEG: decode → re-encode as PNG for inline display
        reencode_as_png(&data)?
    } else if data.starts_with(b"\x89PNG") {
        data
    } else if data.starts_with(b"\xFF\xD8\xFF") {
        reencode_as_png(&data)?
    } else {
        // Try displaying as-is — some terminals handle raw image data
        data
    };

    let mut driver = display::FrameDriver::detect();
    driver.display_png(&png_data)?;

    Ok(())
}

/// Re-encode JPEG image data as PNG using tiny-skia for pixel decoding.
fn reencode_as_png(jpeg_data: &[u8]) -> Result<Vec<u8>, String> {
    // For now, we only support PNG natively.
    // JPEG re-encoding requires an image decoder (not in our deps yet).
    // Fall through to try displaying the raw bytes.
    eprintln!("warning: JPEG re-encoding not yet supported; attempting raw display");
    Ok(jpeg_data.to_vec())
}
