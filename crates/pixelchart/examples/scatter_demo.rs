//! Scatter plot demo — interactive pixel-perfect scatter chart.
//!
//! Run with: `cargo run -p pixelchart --example scatter_demo`

use std::io::stdout;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use pixelchart::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let mut state = ChartState::auto();

    // Sample data: y = sin(x) + noise
    let n = 40;
    let x_data: Vec<f64> = (0..n).map(|i| i as f64 * 0.3).collect();
    let y_data: Vec<f64> = x_data
        .iter()
        .enumerate()
        .map(|(i, &x)| x.sin() + (i as f64 * 0.7).cos() * 0.3)
        .collect();

    let y2_data: Vec<f64> = x_data.iter().map(|&x| x.cos() * 0.8).collect();

    let chart = Chart::scatter(&x_data, &y_data)
        .title("sin(x) vs cos(x)")
        .x_label("x")
        .y_label("amplitude")
        .connected()
        .h_line(0.0) // zero line
        .add_series(Series::new("cos", x_data.clone()), Series::new("", y2_data))
        .theme(Theme::dark())
        .build();

    loop {
        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(3)])
                .split(frame.area());

            frame.render_stateful_widget(ChartWidget::new(&chart), chunks[0], &mut state);

            let status =
                Paragraph::new(" Press 'q' to quit").block(Block::default().borders(Borders::TOP));
            frame.render_widget(status, chunks[1]);
        })?;
        state.flush()?;

        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press && key.code == KeyCode::Char('q') {
                    break;
                }
            }
        }
    }

    state.cleanup();
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}
