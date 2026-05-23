mod app;
mod data;
mod input;
mod panel;
mod ring;
mod theme;

use std::io::stdout;
use std::time::Duration;

use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Terminal setup
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let rx = data::spawn_poller();
    let mut app = app::App::new();

    loop {
        // Receive any pending snapshots (non-blocking)
        while let Ok(snap) = rx.try_recv() {
            app.update(&snap);
        }

        // Render
        terminal.draw(|frame| {
            app.render(frame);
        })?;

        // Flush chart pixel data to terminal
        app.flush_all()?;

        // Handle input with ~30fps frame budget
        match input::poll(Duration::from_millis(33))? {
            input::Action::Quit => break,
            input::Action::TogglePause => app.paused = !app.paused,
            input::Action::CycleSort => app.cycle_sort(),
            input::Action::None => {}
        }
    }

    // Cleanup
    app.cleanup();
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}
