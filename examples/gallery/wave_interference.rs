//! Wave Interference — interactive physics visualization.
//!
//! Two or more point sources emit circular waves that create
//! interference patterns with constructive and destructive zones.
//! Click to place new sources. Color-mapped with smooth gradients.
//!
//! Controls:
//! - Left click — place a new wave source
//! - `c` — clear all sources (reset to two)
//! - `q` — quit
//!
//! Run with: `cargo run --example wave_interference --release`

use std::io::stdout;
use std::time::Instant;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, MouseButton, MouseEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use scry_engine::prelude::{Picker, PixelCanvasState, PixelCanvasWidget, ProtocolKind};
use scry_engine::scene::PixelCanvas;
use scry_engine::style::Color as PxColor;
use scry_engine::transport;

/// A wave source with position and phase offset.
struct WaveSource {
    x: f32,
    y: f32,
    phase: f32,
}

fn default_sources() -> Vec<WaveSource> {
    vec![
        WaveSource {
            x: 0.35,
            y: 0.5,
            phase: 0.0,
        },
        WaveSource {
            x: 0.65,
            y: 0.5,
            phase: 0.0,
        },
    ]
}

// ═══════════════════════════════════════════════════════════════════
// Window mode
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "window")]
fn run_window() -> Result<(), Box<dyn std::error::Error>> {
    use scry_engine::rasterize::Rasterizer;
    use scry_engine::transport::window::{run_loop_continuous, LoopAction};
    use winit::keyboard::KeyCode as WKey;

    let mut sources = default_sources();
    // Add a few extra sources in interesting positions for the window demo
    sources.push(WaveSource {
        x: 0.5,
        y: 0.25,
        phase: 0.5,
    });
    sources.push(WaveSource {
        x: 0.5,
        y: 0.75,
        phase: 1.0,
    });
    let start = Instant::now();

    run_loop_continuous(
        960,
        640,
        "Wave Interference",
        true,
        move |backend, keys, (w, h)| {
            for key in keys {
                if !key.pressed {
                    continue;
                }
                match key.code {
                    WKey::Escape | WKey::KeyQ => return LoopAction::Exit,
                    WKey::KeyC => {
                        sources = default_sources();
                        sources.push(WaveSource {
                            x: 0.5,
                            y: 0.25,
                            phase: 0.5,
                        });
                        sources.push(WaveSource {
                            x: 0.5,
                            y: 0.75,
                            phase: 1.0,
                        });
                    }
                    _ => {}
                }
            }

            let t = start.elapsed().as_secs_f32();
            let canvas = build_wave_pattern(w, h, &sources, t);
            if let Ok(pixmap) = Rasterizer::rasterize(&canvas) {
                let _ = backend.blit(&pixmap);
            }
            LoopAction::Continue
        },
    )?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let use_window = std::env::args().any(|a| a == "--window");
    if use_window {
        #[cfg(feature = "window")]
        {
            return run_window();
        }
        #[cfg(not(feature = "window"))]
        {
            eprintln!("error: --window requires the `window` feature");
            std::process::exit(1);
        }
    }

    enable_raw_mode()?;
    crossterm::execute!(
        stdout(),
        EnterAlternateScreen,
        crossterm::event::EnableMouseCapture
    )?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let picker = Picker::detect();
    let backend: Box<dyn transport::ProtocolBackend> = match picker.protocol() {
        ProtocolKind::Kitty => Box::new(transport::kitty::KittyBackend::new(picker.font_size())),
        _ => Box::new(transport::halfblock::HalfblockBackend::new()),
    };

    let mut state = PixelCanvasState::new(backend, picker.font_size());
    let start = Instant::now();

    // Start with two sources
    let mut sources = default_sources();

    // Track the main render area for mouse coordinate mapping
    let mut render_area = Rect::default();

    loop {
        let t = start.elapsed().as_secs_f32();

        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(3)])
                .split(frame.area());

            render_area = chunks[0];
            let area = chunks[0];
            let font = state.font_size();
            let w = u32::from(area.width) * u32::from(font.width);
            let h = u32::from(area.height) * u32::from(font.height);

            let canvas = build_wave_pattern(w, h, &sources, t);

            frame.render_stateful_widget(
                PixelCanvasWidget::new(canvas).skip_cache(),
                area,
                &mut state,
            );

            let status = Paragraph::new(format!(
                " ▸ wave_interference | {} sources | click to add, 'c' = clear, 'q' = quit",
                sources.len()
            ))
            .style(Style::default().fg(Color::DarkGray))
            .block(
                Block::default()
                    .borders(Borders::TOP)
                    .border_style(Style::default().fg(Color::DarkGray)),
            );
            frame.render_widget(status, chunks[1]);
        })?;
        state.flush()?;

        if event::poll(std::time::Duration::from_millis(16))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Char('c') => {
                        sources = default_sources();
                    }
                    _ => {}
                },
                Event::Mouse(mouse) => {
                    if mouse.kind == MouseEventKind::Down(MouseButton::Left) {
                        // Convert terminal coordinates to normalized coordinates
                        if render_area.width > 0 && render_area.height > 0 {
                            let nx = f32::from(mouse.column.saturating_sub(render_area.x))
                                / f32::from(render_area.width);
                            let ny = f32::from(mouse.row.saturating_sub(render_area.y))
                                / f32::from(render_area.height);
                            sources.push(WaveSource {
                                x: nx.clamp(0.0, 1.0),
                                y: ny.clamp(0.0, 1.0),
                                phase: t,
                            });
                        }
                    }
                }
                _ => {}
            }
        }
    }

    state.cleanup();
    crossterm::execute!(
        stdout(),
        crossterm::event::DisableMouseCapture,
        LeaveAlternateScreen
    )?;
    disable_raw_mode()?;
    Ok(())
}

/// Build the interference pattern as a pixel-by-pixel rendering.
///
/// Uses a sampling grid (1 pixel = 4x4 block) for performance, then fills
/// rectangles with the computed color.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn build_wave_pattern(w: u32, h: u32, sources: &[WaveSource], t: f32) -> PixelCanvas {
    let wf = w as f32;
    let hf = h as f32;

    // Sample at lower resolution for performance
    let step = 4u32;
    let cols = w / step;
    let rows = h / step;

    let mut canvas = PixelCanvas::new(w, h).background(PxColor::from_rgba8(5, 5, 15, 255));

    let wavelength = 30.0; // pixels
    let speed = 4.0; // wave speed
    let k = 2.0 * std::f32::consts::PI / wavelength;
    let omega = speed * k;

    for gy in 0..rows {
        for gx in 0..cols {
            let px = (gx * step) as f32 + step as f32 / 2.0;
            let py = (gy * step) as f32 + step as f32 / 2.0;

            // Sum wave amplitudes from all sources
            let mut amplitude = 0.0_f32;
            for src in sources {
                let sx = src.x * wf;
                let sy = src.y * hf;
                let dist = (px - sx).hypot(py - sy);
                // Decaying wave: sin(k*r - ω*t + phase) / sqrt(r)
                let decay = 1.0 / (1.0 + dist * 0.02);
                amplitude += (k * dist - omega * (t - src.phase)).sin() * decay;
            }

            // Normalize amplitude to color
            let normalized = (amplitude / sources.len().max(1) as f32).clamp(-1.0, 1.0);

            // Map to a cool color palette:
            // negative = deep blue/purple, zero = dark, positive = cyan/white
            let (r, g, b) = if normalized >= 0.0 {
                let v = normalized;
                (
                    (v * v * 100.0) as u8,
                    (v * 200.0 + v * v * 55.0) as u8,
                    (v * 180.0 + v * v * 75.0) as u8,
                )
            } else {
                let v = -normalized;
                (
                    (v * 80.0 + v * v * 100.0) as u8,
                    (v * 20.0) as u8,
                    (v * 120.0 + v * v * 100.0) as u8,
                )
            };

            canvas = canvas
                .rect(
                    (gx * step) as f32,
                    (gy * step) as f32,
                    step as f32,
                    step as f32,
                )
                .fill(PxColor::from_rgba8(r, g, b, 255))
                .done();
        }
    }

    // Draw source markers
    for src in sources {
        let sx = src.x * wf;
        let sy = src.y * hf;
        canvas = canvas
            .circle(sx, sy, 6.0)
            .fill(PxColor::from_rgba8(255, 255, 255, 200))
            .done()
            .circle(sx, sy, 3.0)
            .fill(PxColor::from_rgba8(255, 200, 50, 255))
            .done();
    }

    canvas
}
