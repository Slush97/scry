// SPDX-License-Identifier: MIT OR Apache-2.0
//! Inline terminal image display.
//!
//! Displays PNG images directly in the terminal using Kitty, iTerm2,
//! or Sixel graphics protocols. This is a simplified, standalone
//! implementation that doesn't need ratatui — just raw escape sequences
//! to stdout.

use std::io::{self, Write};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Terminal cell-size detection
// ---------------------------------------------------------------------------

/// Cached cell size so we only call the ioctl once.
static CELL_SIZE: OnceLock<(u16, u16)> = OnceLock::new();

/// Detect the actual terminal cell size in pixels (width, height).
///
/// Uses `Picker::detect_font_size()` which queries `TIOCGWINSZ` on Unix.
/// Falls back to (8, 16) if detection fails.
pub fn detect_cell_size() -> (u16, u16) {
    *CELL_SIZE.get_or_init(|| {
        let font = scry_engine::transport::Picker::detect().font_size();
        // Font detection returns (0,0) or default (8,16) if the ioctl fails.
        // Validate and use real values when available.
        if font.width > 0 && font.height > 0 {
            (font.width, font.height)
        } else {
            (8, 16)
        }
    })
}

/// Compute the terminal's visible pixel dimensions.
///
/// Returns `(pixel_width, pixel_height)` based on the number of columns/rows
/// multiplied by the real cell pixel size.  This tells you how large an image
/// the terminal can display at 1:1 pixel mapping.
pub fn terminal_pixel_size() -> (u32, u32) {
    let (cols, rows) = crossterm::terminal::size().unwrap_or((120, 40));
    let (cw, ch) = detect_cell_size();
    (u32::from(cols) * u32::from(cw), u32::from(rows) * u32::from(ch))
}

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

    // Figure out how many terminal cells the image should occupy.
    let (term_cols, term_rows) = crossterm::terminal::size().unwrap_or((120, 40));
    // Read image dimensions from the PNG header (width/height at bytes 16..24).
    let (img_w, img_h) = png_dimensions(png_data).unwrap_or((400, 400));
    let aspect = img_h as f64 / img_w as f64;
    let (cw, ch) = detect_cell_size();

    // Compute how many columns the image would occupy at 1:1 pixel
    // mapping, then cap at 90% of terminal width so there's margin.
    // This prevents Kitty from rescaling the image (which can cause
    // overflow / clipping when the render is larger than the display).
    let max_rows = term_rows.saturating_sub(2).max(1);
    let native_cols = (img_w as f64 / cw as f64).ceil() as u16;
    let max_cols = ((term_cols as f64) * 0.9).round() as u16;
    let mut cols_to_use = native_cols.min(max_cols).max(20).min(term_cols);

    // Compute how many rows this would require at the given aspect ratio.
    let display_px_w = cols_to_use as f64 * cw as f64;
    let display_px_h = display_px_w * aspect;
    let mut img_rows = (display_px_h / ch as f64).ceil() as u16;

    // If it overflows vertically, shrink columns proportionally to fit.
    if img_rows > max_rows {
        img_rows = max_rows;
        // Back-compute columns from the row budget.
        let fitted_px_h = img_rows as f64 * ch as f64;
        let fitted_px_w = fitted_px_h / aspect;
        cols_to_use = (fitted_px_w / cw as f64).floor() as u16;
        cols_to_use = cols_to_use.max(20).min(term_cols);
    }
    let img_rows = img_rows.max(1);

    if frame == 0 {
        // Query current cursor row (works reliably in raw mode).
        // Falls back to bottom-of-terminal if query fails (piped output).
        let cursor_row = crossterm::cursor::position()
            .map(|(_, row)| row + 1) // 0-indexed → 1-indexed
            .unwrap_or(term_rows.saturating_sub(img_rows).max(1));

        // Print enough newlines to force the terminal to scroll and
        // allocate vertical space for the image.
        for _ in 0..img_rows {
            writeln!(stdout)?;
        }
        // Move cursor back up to the top of the reserved area.
        write!(stdout, "\x1b[{}A", img_rows)?;

        // If near the bottom, the terminal scrolled — anchor shifts up.
        let space_below = term_rows.saturating_sub(cursor_row);
        let anchor = if space_below >= img_rows {
            cursor_row // enough room, place right at cursor
        } else {
            // terminal scrolled by (img_rows - space_below) lines
            cursor_row.saturating_sub(img_rows.saturating_sub(space_below)).max(1)
        };
        ANCHOR_ROW.store(anchor, Ordering::Relaxed);
    } else {
        // Delete the old placement before drawing the new one.
        write!(stdout, "\x1b_Ga=d,d=i,i=1,q=2;\x1b\\")?;
    }

    let anchor = ANCHOR_ROW.load(Ordering::Relaxed).max(1);

    // Position the image at 25% of the horizontal margin.
    let col = (term_cols / 4).max(1); // 1-indexed column

    // Position cursor at the anchor row, centered column.
    write!(stdout, "\x1b[{anchor};{col}H")?;

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
