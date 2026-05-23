use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use std::time::Duration;

pub enum Action {
    Quit,
    TogglePause,
    CycleSort,
    None,
}

pub fn poll(timeout: Duration) -> std::io::Result<Action> {
    if event::poll(timeout)? {
        if let Event::Key(key) = event::read()? {
            if key.kind != KeyEventKind::Press {
                return Ok(Action::None);
            }
            return Ok(match key.code {
                KeyCode::Char('q') | KeyCode::Esc => Action::Quit,
                KeyCode::Char('p') => Action::TogglePause,
                KeyCode::Char('s') => Action::CycleSort,
                _ => Action::None,
            });
        }
    }
    Ok(Action::None)
}
