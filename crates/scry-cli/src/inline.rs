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

/// Display a PNG image inline using the Kitty graphics protocol with explicit
/// display size in terminal columns (`c=`) and rows (`r=`).
pub fn display_kitty_inline_sized(
    png_data: &[u8],
    width: Option<u32>,
    height: Option<u32>,
) -> io::Result<()> {
    use base64::Engine;

    let encoded = base64::engine::general_purpose::STANDARD.encode(png_data);
    let mut stdout = io::stdout().lock();

    // Build the size parameters string
    let mut size_params = String::new();
    if let Some(w) = width {
        size_params.push_str(&format!(",c={w}"));
    }
    if let Some(h) = height {
        size_params.push_str(&format!(",r={h}"));
    }

    const CHUNK_SIZE: usize = 65_536;
    let total = encoded.len();
    let n_chunks = total.div_ceil(CHUNK_SIZE).max(1);

    for i in 0..n_chunks {
        let start = i * CHUNK_SIZE;
        let end = (start + CHUNK_SIZE).min(total);
        let chunk = &encoded[start..end];
        let more = if i != n_chunks - 1 { 1 } else { 0 };

        if i == 0 {
            write!(
                stdout,
                "\x1b_Ga=T,q=2,f=100,m={more}{size_params};{chunk}\x1b\\"
            )?;
        } else {
            write!(stdout, "\x1b_Gm={more};{chunk}\x1b\\")?;
        }
    }

    writeln!(stdout)?;
    stdout.flush()?;
    Ok(())
}

/// Auto-detect the best inline display method and show the image with size hints.
pub fn display_inline_auto_sized(
    png_data: &[u8],
    width: Option<u32>,
    height: Option<u32>,
) -> io::Result<()> {
    let term_program = std::env::var("TERM_PROGRAM").unwrap_or_default();
    let term = std::env::var("TERM").unwrap_or_default();
    let kitty_pid = std::env::var("KITTY_PID").ok();

    if term_program == "iTerm.app" || term_program == "WezTerm" {
        // iTerm2 protocol supports width= and height= params but we
        // fall back to the unsized variant for simplicity.
        display_iterm2_inline(png_data)
    } else if term == "xterm-kitty" || kitty_pid.is_some() {
        display_kitty_inline_sized(png_data, width, height)
    } else {
        display_kitty_inline_sized(png_data, width, height)
    }
}

/// Display one animation frame using the Kitty graphics protocol.
///
/// On the first frame (`frame == 0`) the terminal is scrolled to reserve
/// enough rows for the image, then the cursor is moved back up to the
/// top of that reserved area.  All frames use explicit cursor positioning
/// (`\x1b[row;colH`) so we never rely on fragile cursor save/restore.
///
/// A fixed image ID (`i=1, p=1`) is used so the terminal replaces the
/// previous placement.  Cell-size hints (`c=`, `r=`) tell the terminal
/// exactly how large the image should be.
pub fn display_kitty_animation_frame(png_data: &[u8], frame: u64) -> io::Result<()> {
    use base64::Engine;
    use std::sync::atomic::{AtomicU16, Ordering};

    /// Row where the image starts (1-indexed, set on frame 0).
    static ANCHOR_ROW: AtomicU16 = AtomicU16::new(0);

    let encoded = base64::engine::general_purpose::STANDARD.encode(png_data);
    let mut stdout = io::stdout().lock();

    // Figure out how many terminal rows the image will occupy.
    let (term_cols, _term_rows) = crossterm::terminal::size().unwrap_or((120, 40));
    let cell_w: u16 = 8; // reasonable default cell width in px
    let cell_h: u16 = 16; // reasonable default cell height in px
    // Read image dimensions from the PNG header (width/height at bytes 16..24).
    let (img_w, img_h) = png_dimensions(png_data).unwrap_or((400, 400));
    // The terminal will scale the image to fit `c` columns and `r` rows.
    // We want the image to fit within the terminal width.
    let img_cols = (img_w / u32::from(cell_w)).max(1) as u16;
    let cols_to_use = img_cols.min(term_cols);
    let scale = f64::from(cols_to_use) / f64::from(img_cols);
    let img_rows = ((f64::from(img_h) / f64::from(cell_h)) * scale).ceil() as u16;
    let img_rows = img_rows.max(1);

    if frame == 0 {
        // Print enough newlines to force the terminal to scroll and
        // allocate vertical space for the image.
        for _ in 0..img_rows {
            writeln!(stdout)?;
        }
        // Move cursor back up to the top of the reserved area.
        write!(stdout, "\x1b[{}A", img_rows)?;
        // Query where we are now (use crossterm if available, else estimate).
        // We'll use a simple heuristic: read cursor position via crossterm.
        let anchor = crossterm::cursor::position()
            .map(|(_, row)| row + 1) // convert 0-indexed to 1-indexed
            .unwrap_or(1);
        ANCHOR_ROW.store(anchor, Ordering::Relaxed);
    } else {
        // Delete the old placement before drawing the new one.
        write!(stdout, "\x1b_Ga=d,d=i,i=1,q=2;\x1b\\")?;
    }

    let anchor = ANCHOR_ROW.load(Ordering::Relaxed).max(1);

    // Position cursor at the anchor row, column 1.
    write!(stdout, "\x1b[{anchor};1H")?;

    const CHUNK_SIZE: usize = 65_536;
    let total = encoded.len();
    let n_chunks = total.div_ceil(CHUNK_SIZE).max(1);

    // Begin synchronized update (atomic draw — no flicker).
    write!(stdout, "\x1b[?2026h")?;

    for i in 0..n_chunks {
        let start = i * CHUNK_SIZE;
        let end = (start + CHUNK_SIZE).min(total);
        let chunk = &encoded[start..end];
        let more = if i != n_chunks - 1 { 1 } else { 0 };

        if i == 0 {
            // a=T: transmit+display, f=100: PNG, i=1,p=1: stable IDs,
            // q=2: suppress responses, C=1: don't move cursor,
            // c/r: cell dimensions for proper sizing.
            write!(
                stdout,
                "\x1b_Ga=T,q=2,f=100,i=1,p=1,C=1,c={cols_to_use},r={img_rows},m={more};{chunk}\x1b\\"
            )?;
        } else {
            write!(stdout, "\x1b_Gm={more};{chunk}\x1b\\")?;
        }
    }

    // End synchronized update.
    write!(stdout, "\x1b[?2026l")?;

    // Move cursor below the image so the prompt appears there after exit.
    write!(stdout, "\x1b[{};1H", anchor as u32 + img_rows as u32)?;

    stdout.flush()?;

    Ok(())
}

/// Extract width and height from a PNG file's IHDR chunk.
///
/// The PNG spec places width (4 bytes BE) at offset 16 and height at offset 20.
fn png_dimensions(data: &[u8]) -> Option<(u32, u32)> {
    if data.len() < 24 {
        return None;
    }
    let w = u32::from_be_bytes([data[16], data[17], data[18], data[19]]);
    let h = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);
    Some((w, h))
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
