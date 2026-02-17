// SPDX-License-Identifier: MIT OR Apache-2.0
//! Inline terminal image display.
//!
//! Displays PNG images directly in the terminal using Kitty, iTerm2,
//! or Sixel graphics protocols. This is a simplified, standalone
//! implementation that doesn't need ratatui — just raw escape sequences
//! to stdout.

use std::io::{self, Write};

/// Display a PNG image inline in the terminal using the Kitty graphics protocol.
///
/// The image is displayed at the current cursor position. The terminal will
/// automatically determine the appropriate display size based on the image
/// dimensions and cell size.
pub fn display_kitty_inline(png_data: &[u8]) -> io::Result<()> {
    use base64::Engine;

    let encoded = base64::engine::general_purpose::STANDARD.encode(png_data);
    let mut stdout = io::stdout().lock();

    // Chunk size for Kitty protocol (64KB of base64 text)
    const CHUNK_SIZE: usize = 65_536;
    let total = encoded.len();
    let n_chunks = total.div_ceil(CHUNK_SIZE).max(1);

    for i in 0..n_chunks {
        let start = i * CHUNK_SIZE;
        let end = (start + CHUNK_SIZE).min(total);
        let chunk = &encoded[start..end];
        let more = if i != n_chunks - 1 { 1 } else { 0 };

        if i == 0 {
            // First chunk: include format and action headers
            // a=T: transmit and display
            // f=100: PNG format
            // q=2: suppress terminal responses
            write!(stdout, "\x1b_Ga=T,q=2,f=100,m={more};{chunk}\x1b\\")?;
        } else {
            // Continuation chunks
            write!(stdout, "\x1b_Gm={more};{chunk}\x1b\\")?;
        }
    }

    // Newline after the image so the shell prompt appears below
    writeln!(stdout)?;
    stdout.flush()?;

    Ok(())
}

/// Display a PNG image inline using the iTerm2 inline image protocol.
///
/// Works in iTerm2, WezTerm, and other compatible terminals.
pub fn display_iterm2_inline(png_data: &[u8]) -> io::Result<()> {
    use base64::Engine;

    let encoded = base64::engine::general_purpose::STANDARD.encode(png_data);
    let size = png_data.len();
    let mut stdout = io::stdout().lock();

    // iTerm2 inline image protocol
    write!(
        stdout,
        "\x1b]1337;File=inline=1;size={size};preserveAspectRatio=1:{encoded}\x07"
    )?;

    writeln!(stdout)?;
    stdout.flush()?;

    Ok(())
}

/// Auto-detect the best inline display method and show the image.
///
/// Detection order:
/// 1. `TERM_PROGRAM=iTerm.app` → iTerm2 protocol
/// 2. `TERM=xterm-kitty` or `KITTY_PID` set → Kitty protocol
/// 3. `TERM_PROGRAM=WezTerm` → iTerm2 protocol (WezTerm supports it)
/// 4. Fallback → Kitty protocol (widely supported: Kitty, Ghostty, WezTerm)
pub fn display_inline_auto(png_data: &[u8]) -> io::Result<()> {
    let term_program = std::env::var("TERM_PROGRAM").unwrap_or_default();
    let term = std::env::var("TERM").unwrap_or_default();
    let kitty_pid = std::env::var("KITTY_PID").ok();

    if term_program == "iTerm.app" {
        display_iterm2_inline(png_data)
    } else if term == "xterm-kitty" || kitty_pid.is_some() {
        display_kitty_inline(png_data)
    } else if term_program == "WezTerm" {
        display_iterm2_inline(png_data)
    } else {
        // Default to Kitty — it's supported by Kitty, Ghostty, WezTerm, Konsole
        display_kitty_inline(png_data)
    }
}

/// Probe whether the current terminal likely supports inline images.
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
