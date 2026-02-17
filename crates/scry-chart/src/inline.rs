// SPDX-License-Identifier: MIT OR Apache-2.0
//! Inline terminal image display for charts.
//!
//! Displays PNG images directly in the terminal using Kitty, iTerm2,
//! or Sixel graphics protocols. Includes frame-overwrite support for
//! smooth live-updating charts.
//!
//! Gated on the `inline` feature.

use std::io::{self, Write};

const CHUNK_SIZE: usize = 65_536;

/// Display a PNG image inline using the Kitty graphics protocol.
pub fn display_kitty_inline(png_data: &[u8]) -> io::Result<()> {
    use base64::Engine;

    let encoded = base64::engine::general_purpose::STANDARD.encode(png_data);
    let mut stdout = io::stdout().lock();

    let total = encoded.len();
    let n_chunks = total.div_ceil(CHUNK_SIZE).max(1);

    for i in 0..n_chunks {
        let start = i * CHUNK_SIZE;
        let end = (start + CHUNK_SIZE).min(total);
        let chunk = &encoded[start..end];
        let more = u8::from(i < n_chunks - 1);

        if i == 0 {
            write!(stdout, "\x1b_Ga=T,q=2,f=100,m={more};{chunk}\x1b\\")?;
        } else {
            write!(stdout, "\x1b_Gm={more};{chunk}\x1b\\")?;
        }
    }

    writeln!(stdout)?;
    stdout.flush()?;
    Ok(())
}

/// Display a PNG image inline using the iTerm2 inline image protocol.
pub fn display_iterm2_inline(png_data: &[u8]) -> io::Result<()> {
    use base64::Engine;

    let encoded = base64::engine::general_purpose::STANDARD.encode(png_data);
    let size = png_data.len();
    let mut stdout = io::stdout().lock();

    write!(
        stdout,
        "\x1b]1337;File=inline=1;size={size};preserveAspectRatio=1:{encoded}\x07"
    )?;

    writeln!(stdout)?;
    stdout.flush()?;
    Ok(())
}

/// Auto-detect the best inline display method and show the image.
pub fn display_inline_auto(png_data: &[u8]) -> io::Result<()> {
    let term_program = std::env::var("TERM_PROGRAM").unwrap_or_default();
    let term = std::env::var("TERM").unwrap_or_default();
    let kitty_pid = std::env::var("KITTY_PID").ok();

    if term_program == "iTerm.app" || term_program == "WezTerm" {
        display_iterm2_inline(png_data)
    } else if term == "xterm-kitty" || kitty_pid.is_some() {
        display_kitty_inline(png_data)
    } else {
        // Default to Kitty — supported by Kitty, Ghostty, WezTerm, Konsole
        display_kitty_inline(png_data)
    }
}

/// Probe whether the current terminal likely supports inline images.
#[must_use]
pub fn terminal_supports_inline() -> bool {
    let term = std::env::var("TERM").unwrap_or_default();
    let term_program = std::env::var("TERM_PROGRAM").unwrap_or_default();
    let kitty_pid = std::env::var("KITTY_PID").ok();

    term == "xterm-kitty"
        || kitty_pid.is_some()
        || term_program == "iTerm.app"
        || term_program == "WezTerm"
        || term_program == "ghostty"
}

/// Estimate the number of terminal rows an image of the given pixel height
/// occupies, assuming ~16px per terminal row.
fn estimate_terminal_rows(pixel_height: u32) -> u16 {
    let cell_height = 16u32;
    pixel_height.div_ceil(cell_height) as u16
}

/// Overwrite a previous frame by moving the cursor up, then display a new frame.
///
/// On `frame_number == 0` no cursor movement happens (first frame).
/// For subsequent frames, the cursor is moved up to overwrite the previous image.
pub fn display_frame(png_data: &[u8], pixel_height: u32, frame_number: u64) -> io::Result<()> {
    let mut stdout = io::stdout().lock();

    if frame_number > 0 {
        let rows = estimate_terminal_rows(pixel_height);
        write!(stdout, "\x1b[{}A", rows + 1)?;
        for _ in 0..=rows {
            write!(stdout, "\x1b[2K\x1b[1B")?;
        }
        write!(stdout, "\x1b[{}A", rows + 1)?;
        stdout.flush()?;
    }

    drop(stdout); // release lock before display_inline_auto takes it
    display_inline_auto(png_data)
}
