//! scry-viz — pixel-perfect terminal music visualizer.
//!
//! Captures the default sink monitor (whatever is playing) via
//! PulseAudio/PipeWire and renders reactive vector graphics through scry.

mod analysis;
mod audio;
mod dsp;
mod theme;
mod viz;

use std::io::stdout;
use std::time::{Duration, Instant};

use clap::Parser;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::*;
use ratatui::widgets::Paragraph;

use scry_engine::prelude::{Picker, PixelCanvasState, PixelCanvasWidget, ProtocolKind};
use scry_engine::transport;

#[derive(Parser)]
#[command(name = "scry-viz", about = "Pixel-perfect terminal music visualizer")]
struct Args {
    /// Visual mode
    #[arg(short, long, value_enum, default_value = "silk")]
    mode: viz::Mode,

    /// Color theme (neon, aurora, sunset, matrix, ice, ember)
    #[arg(short, long, default_value = "neon")]
    theme: String,

    /// PulseAudio/PipeWire source name (default: monitor of the default sink)
    #[arg(short, long)]
    device: Option<String>,

    /// Number of spectrum bands
    #[arg(short, long, default_value_t = 64)]
    bands: usize,

    /// Frame rate cap
    #[arg(long, default_value_t = 60)]
    fps: u32,

    /// Start with the status HUD hidden.
    #[arg(long)]
    no_hud: bool,

    /// Alias for --no-hud for clean visual-only launchers.
    #[arg(long)]
    visual_only: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let mut theme_idx = theme::THEMES
        .iter()
        .position(|t| t.name == args.theme)
        .ok_or_else(|| {
            let names: Vec<_> = theme::THEMES.iter().map(|t| t.name).collect();
            format!(
                "unknown theme '{}' (themes: {})",
                args.theme,
                names.join(", ")
            )
        })?;
    let bands = args.bands.clamp(8, 256);

    let audio = audio::spawn_capture(args.device)?;
    let mut analyzer = dsp::Analyzer::new(bands);
    let mut vstate = viz::VizState::new();
    let mut mode = args.mode;
    let mut show_status = !(args.no_hud || args.visual_only);

    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let picker = Picker::detect();
    let backend: Box<dyn transport::ProtocolBackend> = match picker.protocol() {
        ProtocolKind::Kitty => Box::new(transport::kitty::KittyBackend::new(picker.font_size())),
        _ => Box::new(transport::halfblock::HalfblockBackend::new()),
    };
    let mut state = PixelCanvasState::new(backend, picker.font_size());

    let frame_budget = Duration::from_secs(1) / args.fps.clamp(10, 240);
    let start = Instant::now();
    let mut last_frame = Instant::now();
    let mut paused = false;
    let mut fps = 0.0f64;

    loop {
        let frame_start = Instant::now();
        let dt = (frame_start - last_frame).as_secs_f32().min(0.1);
        last_frame = frame_start;
        fps = fps * 0.9 + 0.1 / f64::from(dt.max(1e-6));

        if !paused {
            analyzer.update(&audio, dt);
        }

        let theme = &theme::THEMES[theme_idx];
        terminal.draw(|frame| {
            let status_rows = u16::from(show_status);
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(status_rows)])
                .split(frame.area());

            let area = chunks[0];
            let font = state.font_size();
            let w = u32::from(area.width) * u32::from(font.width);
            let h = u32::from(area.height) * u32::from(font.height);

            let canvas = mode.build(
                &mut vstate,
                w,
                h,
                &analyzer.frame,
                theme,
                start.elapsed().as_secs_f32(),
                dt,
            );
            frame.render_stateful_widget(
                PixelCanvasWidget::new(canvas).skip_cache(),
                area,
                &mut state,
            );

            if show_status {
                let beat = if analyzer.frame.beat.envelope > 0.5 {
                    "●"
                } else {
                    "○"
                };
                let status = Paragraph::new(format!(
                    " {beat} scry-viz | {} | {} | {fps:.0} fps | tab: mode  1-9/0/-: direct  t: theme  s: hud  space: pause  q: quit",
                    mode.name(),
                    theme.name,
                ))
                .style(Style::default().fg(Color::DarkGray));
                frame.render_widget(status, chunks[1]);
            }
        })?;
        state.flush()?;

        let poll_for = frame_budget.saturating_sub(frame_start.elapsed());
        if event::poll(poll_for.max(Duration::from_millis(1)))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Tab => mode = mode.next(),
                        KeyCode::Char('1') => mode = viz::Mode::Silk,
                        KeyCode::Char('2') => mode = viz::Mode::Ridge,
                        KeyCode::Char('3') => mode = viz::Mode::Mandala,
                        KeyCode::Char('4') => mode = viz::Mode::Nova,
                        KeyCode::Char('5') => mode = viz::Mode::Bars,
                        KeyCode::Char('6') => mode = viz::Mode::Radial,
                        KeyCode::Char('7') => mode = viz::Mode::Wave,
                        KeyCode::Char('8') => mode = viz::Mode::Spectrogram,
                        KeyCode::Char('9') => mode = viz::Mode::Vortex,
                        KeyCode::Char('0') => mode = viz::Mode::Constellation,
                        KeyCode::Char('-') => mode = viz::Mode::Prism,
                        KeyCode::Char('t') => theme_idx = (theme_idx + 1) % theme::THEMES.len(),
                        KeyCode::Char('s') => show_status = !show_status,
                        KeyCode::Char(' ') => paused = !paused,
                        _ => {}
                    }
                }
            }
        }
    }

    state.cleanup();
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}
