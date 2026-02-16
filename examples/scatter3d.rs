//! Interactive 3D scatter plot in the terminal.
//!
//! Demonstrates both rendering modes:
//! - **Inline** (default): `cargo run --example scatter3d`
//! - **TUI**: `cargo run --example scatter3d -- --tui`
//!
//! Controls:
//! - Arrow keys / WASD — rotate
//! - +/- — zoom in/out
//! - Q / Esc — quit

use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let tui_mode = args.iter().any(|a| a == "--tui");

    // Sample 3D data: two clusters
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();

    // Cluster 1: centered at (0.2, 0.3, 0.2)
    let mut rng = fastrand::Rng::with_seed(42);
    for _ in 0..40 {
        xs.push(0.2 + rng.f64() * 0.3);
        ys.push(0.3 + rng.f64() * 0.3);
        zs.push(0.2 + rng.f64() * 0.3);
    }

    // Cluster 2: centered at (0.7, 0.7, 0.7)
    for _ in 0..40 {
        xs.push(0.7 + rng.f64() * 0.2);
        ys.push(0.7 + rng.f64() * 0.2);
        zs.push(0.7 + rng.f64() * 0.2);
    }

    let chart = scry_chart::chart3d::Chart3D::scatter(&xs, &ys, &zs)
        .title("3D Scatter — Two Clusters");

    if tui_mode {
        run_tui(chart);
    } else {
        // Inline mode — direct to terminal, no ratatui
        if let Err(e) = chart.show() {
            eprintln!("Error: {e}");
        }
    }
}

fn run_tui(chart: scry_chart::chart3d::Chart3D) {
    use std::io::stdout;

    use crossterm::{
        event::{self, Event, KeyCode, KeyEventKind},
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
        ExecutableCommand,
    };

    use ratatui::widgets::{Block, Borders};

    use scry_chart::prelude::{Chart3DState, Chart3DWidget};
    use scry_chart::chart3d::camera::{Camera3D, Vec3};

    enable_raw_mode().expect("enable raw mode");
    stdout()
        .execute(EnterAlternateScreen)
        .expect("enter alt screen");

    let backend = ratatui::backend::CrosstermBackend::new(stdout());
    let mut terminal = ratatui::Terminal::new(backend).expect("terminal");

    let mut state = Chart3DState::auto();
    let mut angle_y: f64 = 0.45;
    let mut angle_x: f64 = 0.35;
    let mut distance: f64 = 2.5;

    loop {
        let cam = Camera3D::orbiting(
            Vec3::new(0.5, 0.5, 0.5),
            distance as f32,
            angle_y as f32,
            angle_x as f32,
        );
        let current_chart = chart.clone().camera(cam);

        terminal
            .draw(|frame| {
                let area = frame.area();
                let block = Block::default()
                    .borders(Borders::ALL)
                    .title(" 3D Scatter — ←→↑↓ rotate | +/- zoom | Q quit ");
                let inner = block.inner(area);
                frame.render_widget(block, area);
                frame.render_stateful_widget(
                    Chart3DWidget::new(&current_chart),
                    inner,
                    &mut state,
                );
            })
            .expect("draw");

        let _ = state.flush();

        if event::poll(std::time::Duration::from_millis(50)).unwrap_or(false) {
            if let Ok(Event::Key(key)) = event::read() {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Left | KeyCode::Char('a') => angle_y -= 0.12,
                    KeyCode::Right | KeyCode::Char('d') => angle_y += 0.12,
                    KeyCode::Up | KeyCode::Char('w') => angle_x -= 0.12,
                    KeyCode::Down | KeyCode::Char('s') => angle_x += 0.12,
                    KeyCode::Char('+') | KeyCode::Char('=') => {
                        distance = (distance - 0.2).max(0.5);
                    }
                    KeyCode::Char('-') => {
                        distance = (distance + 0.2).min(10.0);
                    }
                    _ => {}
                }
            }
        }
    }

    state.cleanup();
    disable_raw_mode().expect("disable raw mode");
    stdout()
        .execute(LeaveAlternateScreen)
        .expect("leave alt screen");
}
