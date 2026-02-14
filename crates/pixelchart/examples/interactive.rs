//! Interactive chart demo — crosshair, tooltip, zoom/pan.
//!
//! Run with: `cargo run -p pixelchart --example interactive`
//!
//! Controls:
//!   Mouse move  — crosshair follows cursor
//!   Scroll      — zoom in/out
//!   h/j/k/l     — pan left/down/up/right
//!   r           — reset zoom
//!   c           — toggle crosshair
//!   q / Esc     — quit

use std::io::stdout;

use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, MouseEventKind,
    },
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use pixelchart::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    stdout().execute(EnableMouseCapture)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let mut state = ChartState::auto().with_interactivity();

    // Generate sample data: damped sine wave
    let n = 60;
    let x_data: Vec<f64> = (0..n).map(|i| i as f64 * 0.2).collect();
    let y_data: Vec<f64> = x_data.iter().map(|&x| x.sin() * (-x * 0.1).exp()).collect();

    let y2_data: Vec<f64> = x_data
        .iter()
        .map(|&x| x.cos() * (-x * 0.15).exp() * 0.8)
        .collect();

    // Initialize zoom from data extents
    let x_min = 0.0;
    let x_max = (n as f64 - 1.0) * 0.2;
    let y_min = -1.1;
    let y_max = 1.1;
    state.set_zoom_extents(x_min, x_max, y_min, y_max);

    loop {
        // Build chart with current zoom range
        let mut chart_builder = Chart::scatter(&x_data, &y_data)
            .title("Interactive: Damped Oscillation")
            .x_label("time (s)")
            .y_label("amplitude")
            .connected()
            .h_line(0.0)
            .add_series(
                Series::new("cos-decay", x_data.clone()),
                Series::new("", y2_data.clone()),
            )
            .annotate(0.0, 0.0, "origin")
            .trend_line()
            .theme(Theme::dark());

        // Apply zoom range if zoomed
        if let Some(ref zoom) = state.zoom {
            let (zx0, zx1) = zoom.x_range();
            let (zy0, zy1) = zoom.y_range();
            chart_builder = chart_builder.x_range(zx0, zx1).y_range(zy0, zy1);
        }

        let chart = chart_builder.build();

        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(3)])
                .split(frame.area());

            frame.render_stateful_widget(ChartWidget::new(&chart), chunks[0], &mut state);

            // Status bar
            let zoom_info = if let Some(ref zoom) = state.zoom {
                if zoom.is_zoomed() {
                    format!(
                        " [ZOOMED] x:{:.1}..{:.1} y:{:.1}..{:.1}",
                        zoom.x_range().0,
                        zoom.x_range().1,
                        zoom.y_range().0,
                        zoom.y_range().1
                    )
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            let cursor_info = if let Some((x, y)) = state.cursor_data_position() {
                format!("  cursor=({x:.2}, {y:.2})")
            } else {
                String::new()
            };

            let status = Paragraph::new(format!(
                " h/j/k/l=pan  scroll=zoom  r=reset  c=crosshair  q=quit{zoom_info}{cursor_info}"
            ))
            .block(Block::default().borders(Borders::TOP));
            frame.render_widget(status, chunks[1]);
        })?;
        state.flush()?;

        if event::poll(std::time::Duration::from_millis(50))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Char('h') => {
                        if let Some(ref mut zoom) = state.zoom {
                            zoom.pan_left();
                        }
                    }
                    KeyCode::Char('l') => {
                        if let Some(ref mut zoom) = state.zoom {
                            zoom.pan_right();
                        }
                    }
                    KeyCode::Char('k') => {
                        if let Some(ref mut zoom) = state.zoom {
                            zoom.pan_up();
                        }
                    }
                    KeyCode::Char('j') => {
                        if let Some(ref mut zoom) = state.zoom {
                            zoom.pan_down();
                        }
                    }
                    KeyCode::Char('r') => {
                        if let Some(ref mut zoom) = state.zoom {
                            zoom.reset();
                        }
                    }
                    KeyCode::Char('c') => {
                        state.cursor.toggle_crosshair();
                    }
                    _ => {}
                },
                Event::Mouse(mouse) => match mouse.kind {
                    MouseEventKind::Moved => {
                        state.handle_mouse_move(mouse.column, mouse.row);
                    }
                    MouseEventKind::ScrollUp => {
                        state.handle_scroll_up();
                    }
                    MouseEventKind::ScrollDown => {
                        state.handle_scroll_down();
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }

    state.cleanup();
    disable_raw_mode()?;
    stdout().execute(DisableMouseCapture)?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}
