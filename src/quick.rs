// SPDX-License-Identifier: MIT OR Apache-2.0
//! One-shot convenience renderer for quick prototyping.
//!
//! Displays a [`PixelCanvas`] in the terminal with minimal boilerplate — no
//! Ratatui `Terminal`, no manual backend setup. Useful for examples, debugging,
//! and "hello world" programs.
//!
//! # Example
//!
//! ```no_run
//! use scry_engine::scene::{PixelCanvas, Color};
//!
//! scry_engine::quick::show(|w, h| {
//!     PixelCanvas::new(w, h)
//!         .background(Color::BLACK)
//!         .circle(w as f32 / 2.0, h as f32 / 2.0, 80.0)
//!             .fill(Color::from_rgba8(70, 130, 180, 255))
//!             .done()
//! }).unwrap();
//! ```

use crate::render::IncrementalRenderer;
use crate::scene::PixelCanvas;
use crate::transport::Picker;
use crate::PixelCanvasError;

/// Render a canvas to the terminal, wait for a keypress, then clean up.
///
/// The closure receives `(pixel_width, pixel_height)` computed from the
/// terminal size and font metrics, so you can build a canvas that fills
/// the screen without measuring anything yourself.
///
/// After rendering, the function enters raw mode and waits for any key
/// event before restoring the terminal and returning.
///
/// # Errors
///
/// Returns [`PixelCanvasError`] if protocol detection, rasterization,
/// or transmission fails. Also wraps I/O errors from terminal control.
pub fn show<F>(build: F) -> Result<(), PixelCanvasError>
where
    F: FnOnce(u32, u32) -> PixelCanvas,
{
    use crossterm::event::{self, Event};
    use crossterm::terminal;

    let picker = Picker::detect();
    let font_size = picker.font_size();

    // Get terminal size in cells, then convert to pixels.
    let (cols, rows) = terminal::size().map_err(PixelCanvasError::Transmission)?;
    let px_w = u32::from(cols) * u32::from(font_size.width);
    // Reserve 1 row for the status line.
    let px_h = u32::from(rows.saturating_sub(1)) * u32::from(font_size.height);

    if px_w == 0 || px_h == 0 {
        return Err(PixelCanvasError::FontSizeUnknown);
    }

    let canvas = build(px_w, px_h);

    let mut renderer = IncrementalRenderer::from_picker(&picker);
    renderer.render_canvas(&canvas)?;

    // Print hint on the last row and wait for input.
    {
        use std::io::Write;
        let mut stdout = std::io::stdout().lock();
        write!(stdout, "\x1b[{rows};1H\x1b[2mpress any key\x1b[0m")?;
        stdout.flush()?;
    }

    terminal::enable_raw_mode().map_err(PixelCanvasError::Transmission)?;
    loop {
        if let Ok(ev) = event::read() {
            if matches!(ev, Event::Key(_)) {
                break;
            }
        }
    }
    terminal::disable_raw_mode().map_err(PixelCanvasError::Transmission)?;

    // Clean up protocol images and restore cursor.
    drop(renderer);
    {
        use std::io::Write;
        let mut stdout = std::io::stdout().lock();
        write!(stdout, "\x1b[{rows};1H\x1b[2K")?;
        stdout.flush()?;
    }

    Ok(())
}
