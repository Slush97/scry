//! Simple shapes example.
//!
//! Renders various shapes to demonstrate the `PixelCanvas` drawing API
//! and displays them using the Kitty graphics protocol.
//!
//! Run with: `cargo run --example simple_shapes`

use std::io::stdout;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

// Use our crate's Color type (not ratatui's)
use ratatui_pixelcanvas::prelude::{
    PixelCanvasState, PixelCanvasWidget, Picker, ProtocolKind,
};
use ratatui_pixelcanvas::scene::style::Point;
use ratatui_pixelcanvas::scene::PixelCanvas;
use ratatui_pixelcanvas::style::Color as PxColor;
use ratatui_pixelcanvas::transport;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup terminal
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    // Detect protocol & font size
    let picker = Picker::detect();
    eprintln!(
        "Detected protocol: {:?}, font size: {:?}",
        picker.protocol(),
        picker.font_size()
    );

    // Create backend based on detection
    let backend: Box<dyn transport::ProtocolBackend> = match picker.protocol() {
        ProtocolKind::Kitty => Box::new(transport::kitty::KittyBackend::new(picker.font_size())),
        _ => Box::new(transport::halfblock::HalfblockBackend::new()),
    };

    let mut state = PixelCanvasState::new(backend, picker.font_size());

    // Main loop
    loop {
        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(3)])
                .split(frame.area());

            let drawing_area = chunks[0];

            // Build the scene
            let canvas = build_demo_scene(drawing_area, &state);

            // Render the pixel canvas widget
            frame.render_stateful_widget(
                PixelCanvasWidget::new(canvas).z_index(-1),
                drawing_area,
                &mut state,
            );

            // Status bar
            let status = Paragraph::new(" Press 'q' to quit")
                .block(Block::default().borders(Borders::TOP));
            frame.render_widget(status, chunks[1]);
        })?;
        state.flush()?;

        // Handle input
        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press && key.code == KeyCode::Char('q') {
                    break;
                }
            }
        }
    }

    // Cleanup
    state.cleanup();
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn build_demo_scene(area: Rect, state: &PixelCanvasState) -> PixelCanvas {
    let font = state.font_size();
    let w = u32::from(area.width) * u32::from(font.width);
    let h = u32::from(area.height) * u32::from(font.height);

    let w_f = w as f32;
    let h_f = h as f32;

    PixelCanvas::new(w, h)
        // Dark background
        .background(PxColor::from_rgba8(15, 15, 25, 255))
        // Large circle in the center
        .circle(w_f / 2.0, h_f / 2.0, h_f * 0.3)
        .fill(PxColor::from_rgba8(70, 130, 180, 200))
        .stroke(PxColor::from_rgba8(200, 220, 255, 255), 3.0)
        .done()
        // Smaller circle top-left
        .circle(w_f * 0.2, h_f * 0.3, h_f * 0.12)
        .fill(PxColor::from_rgba8(255, 100, 100, 180))
        .done()
        // Rectangle bottom-right
        .rect(w_f * 0.6, h_f * 0.6, w_f * 0.25, h_f * 0.25)
        .fill(PxColor::from_rgba8(100, 255, 100, 180))
        .corner_radius(12.0)
        .stroke(PxColor::from_rgba8(255, 255, 255, 100), 2.0)
        .done()
        // Diagonal line
        .line(w_f * 0.1, h_f * 0.9, w_f * 0.9, h_f * 0.1)
        .color(PxColor::from_rgba8(255, 200, 50, 200))
        .width(2.0)
        .done()
        // Gradient rectangle at top
        .gradient(w_f * 0.1, h_f * 0.05, w_f * 0.8, h_f * 0.08)
        .linear(
            Point::new(w_f * 0.1, h_f * 0.05),
            Point::new(w_f * 0.9, h_f * 0.05),
        )
        .stop(0.0, PxColor::from_rgba8(255, 50, 100, 255))
        .stop(0.5, PxColor::from_rgba8(150, 50, 255, 255))
        .stop(1.0, PxColor::from_rgba8(50, 150, 255, 255))
        .done()
}
