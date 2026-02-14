//! Dashboard demo — multiple charts in a split layout.
//!
//! Run with: `cargo run -p pixelchart --example dashboard`

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

    // ChartState::auto() — one liner per chart
    let mut line_state = ChartState::auto();
    let mut bar_state = ChartState::auto();
    let mut hist_state = ChartState::auto();
    let mut scatter_state = ChartState::auto();

    // --- Build charts ---

    // Line chart with fill + reference line
    let line_chart = Chart::line(&[2.0, 5.0, 3.0, 8.0, 4.0, 7.0, 6.0, 9.0])
        .title("Revenue (M$)")
        .x_label("Quarter")
        .filled()
        .with_points()
        .h_line(5.0) // target line
        .add_series(Series::new(
            "Costs",
            vec![1.0, 3.0, 2.5, 5.0, 3.5, 4.0, 5.0, 6.0],
        ))
        .theme(Theme::dark())
        .build();

    // Bar chart with pastel theme
    let bar_chart = Chart::bar(
        vec![
            "Rust".into(),
            "Python".into(),
            "Go".into(),
            "C++".into(),
            "JS".into(),
        ],
        &[85.0, 78.0, 72.0, 68.0, 65.0],
    )
    .title("Developer Satisfaction")
    .y_label("Score")
    .y_range(0.0, 100.0) // explicit axis range
    .theme(Theme::pastel())
    .build();

    // Histogram
    let hist_data: Vec<f64> = (0..200)
        .map(|i| {
            let x = i as f64 * 0.05;
            (x * 2.7).sin() * 3.0 + 5.0 + (x * 0.3).cos()
        })
        .collect();
    let hist_chart = Chart::histogram(&hist_data)
        .title("Distribution")
        .x_label("Value")
        .y_label("Count")
        .bins(20)
        .theme(Theme::dark())
        .build();

    // Scatter chart with markers
    let n = 30;
    let sx: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
    let sy: Vec<f64> = sx
        .iter()
        .map(|&x| x.sqrt() * 2.0 + (x * 0.8).sin())
        .collect();
    let scatter_chart = Chart::scatter(&sx, &sy)
        .title("Growth Curve")
        .x_label("Time")
        .y_label("Value")
        .connected()
        .marker(Marker::Diamond)
        .theme(Theme::dark())
        .build();

    loop {
        terminal.draw(|frame| {
            let main_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Percentage(50),
                    Constraint::Percentage(50),
                    Constraint::Length(3),
                ])
                .split(frame.area());

            let top = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(main_chunks[0]);

            let bottom = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(main_chunks[1]);

            frame.render_stateful_widget(ChartWidget::new(&line_chart), top[0], &mut line_state);
            frame.render_stateful_widget(ChartWidget::new(&bar_chart), top[1], &mut bar_state);
            frame.render_stateful_widget(ChartWidget::new(&hist_chart), bottom[0], &mut hist_state);
            frame.render_stateful_widget(
                ChartWidget::new(&scatter_chart),
                bottom[1],
                &mut scatter_state,
            );

            let status = Paragraph::new(" pixelchart dashboard — press 'q' to quit")
                .block(Block::default().borders(Borders::TOP));
            frame.render_widget(status, main_chunks[2]);
        })?;
        line_state.flush()?;
        bar_state.flush()?;
        hist_state.flush()?;
        scatter_state.flush()?;

        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press && key.code == KeyCode::Char('q') {
                    break;
                }
            }
        }
    }

    line_state.cleanup();
    bar_state.cleanup();
    hist_state.cleanup();
    scatter_state.cleanup();
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}
