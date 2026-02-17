//! **Spring + Sequence + Preset Animation Showcase**
//!
//! Demonstrates scry's novel animation system — the first coroutine-style
//! animation sequencing in any Rust graphics library.
//!
//! **Pages:**
//! 1. **Physics Springs** — 5 named spring configs racing to a target, showing
//!    overshoot/damping differences. Press `r` to restart the race.
//! 2. **Coroutine Sequence** — A multi-step screenplay: fade-in → wait →
//!    spring pop → parallel slide, all driven by `SequencePlayer`.
//! 3. **Preset Gallery** — Side-by-side preset animations (fade, pop, pulse,
//!    shake, slide) firing in sequence with stagger.
//!
//! Run with: `cargo run --example spring_sequence_demo`
//! Window:   `cargo run --example spring_sequence_demo --features window -- --window`

use std::io::stdout;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use scry_engine::prelude::*;
use scry_engine::scene::PixelCanvas;
use scry_engine::style::Color as C;
use scry_engine::transport;

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

// ═══════════════════════════════════════════════════════════════════
// Pages
// ═══════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, PartialEq, Eq)]
enum Page {
    SpringRace,
    SequenceScreenplay,
    PresetGallery,
}

impl Page {
    const ALL: [Self; 3] = [
        Self::SpringRace,
        Self::SequenceScreenplay,
        Self::PresetGallery,
    ];
    const fn label(self) -> &'static str {
        match self {
            Self::SpringRace => "Physics Springs Race",
            Self::SequenceScreenplay => "Coroutine Sequence",
            Self::PresetGallery => "Preset Gallery",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Window mode
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "window")]
fn run_window() -> Result<(), Box<dyn std::error::Error>> {
    use scry_engine::rasterize::Rasterizer;
    use scry_engine::transport::window::{run_loop_continuous, LoopAction};
    use winit::keyboard::KeyCode as WKey;

    let mut page_idx: usize = 0;
    let mut last_frame = Instant::now();

    let mut springs = create_spring_race();
    let mut seq_player = create_screenplay();
    let mut preset_players = create_preset_gallery();
    let mut preset_start = Instant::now();

    run_loop_continuous(
        960,
        640,
        "Spring + Sequence Demo",
        true,
        move |backend, keys, (w, h)| {
            let now = Instant::now();
            let dt = now - last_frame;
            last_frame = now;
            let page = Page::ALL[page_idx];

            for key in keys {
                if !key.pressed {
                    continue;
                }
                match key.code {
                    WKey::Escape | WKey::KeyQ => return LoopAction::Exit,
                    WKey::ArrowRight | WKey::KeyL => {
                        page_idx = (page_idx + 1) % Page::ALL.len();
                    }
                    WKey::ArrowLeft | WKey::KeyH => {
                        page_idx = page_idx.checked_sub(1).unwrap_or(Page::ALL.len() - 1);
                    }
                    WKey::KeyR => {
                        springs = create_spring_race();
                        seq_player = create_screenplay();
                        preset_players = create_preset_gallery();
                        preset_start = Instant::now();
                    }
                    _ => {}
                }
            }

            // Tick active page
            match page {
                Page::SpringRace => {
                    for (spring, _, _) in &mut springs {
                        spring.advance(dt);
                    }
                }
                Page::SequenceScreenplay => {
                    seq_player.advance(dt);
                    if seq_player.is_complete() {
                        seq_player = create_screenplay();
                    }
                }
                Page::PresetGallery => {
                    for player in &mut preset_players {
                        player.advance(dt);
                    }
                    if now.duration_since(preset_start) > Duration::from_secs(4) {
                        preset_players = create_preset_gallery();
                        preset_start = now;
                    }
                }
            }

            let canvas = match page {
                Page::SpringRace => build_spring_race_page(w, h, &springs),
                Page::SequenceScreenplay => build_screenplay_page(w, h, &seq_player),
                Page::PresetGallery => build_preset_page(w, h, &preset_players),
            };

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
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let picker = Picker::detect();
    let backend: Box<dyn transport::ProtocolBackend> = match picker.protocol() {
        ProtocolKind::Kitty => Box::new(transport::kitty::KittyBackend::new(picker.font_size())),
        _ => Box::new(transport::halfblock::HalfblockBackend::new()),
    };
    let mut state = PixelCanvasState::new(backend, picker.font_size());

    let mut page_idx: usize = 0;
    let start = Instant::now();
    let mut last_frame = start;

    // FPS tracking
    let mut frame_times: Vec<Duration> = Vec::with_capacity(60);
    let mut fps: f64 = 0.0;

    // Page 1: Spring race state
    let mut springs = create_spring_race();

    // Page 2: Sequence player
    let mut seq_player = create_screenplay();

    // Page 3: Preset players
    let mut preset_players = create_preset_gallery();
    let mut preset_start = Instant::now();

    loop {
        let now = Instant::now();
        let dt = now - last_frame;
        last_frame = now;
        let page = Page::ALL[page_idx];

        // FPS
        frame_times.push(dt);
        if frame_times.len() > 60 {
            frame_times.remove(0);
        }
        if !frame_times.is_empty() {
            let total: Duration = frame_times.iter().sum();
            fps = frame_times.len() as f64 / total.as_secs_f64();
        }

        // Tick active page
        match page {
            Page::SpringRace => {
                for (spring, _, _) in &mut springs {
                    spring.advance(dt);
                }
            }
            Page::SequenceScreenplay => {
                seq_player.advance(dt);
                if seq_player.is_complete() {
                    seq_player = create_screenplay();
                }
            }
            Page::PresetGallery => {
                for player in &mut preset_players {
                    player.advance(dt);
                }
                // Auto-restart presets every 4 seconds
                if now.duration_since(preset_start) > Duration::from_secs(4) {
                    preset_players = create_preset_gallery();
                    preset_start = now;
                }
            }
        }

        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(3)])
                .split(frame.area());

            let area = chunks[0];
            let font = state.font_size();
            let w = u32::from(area.width) * u32::from(font.width);
            let h = u32::from(area.height) * u32::from(font.height);

            let canvas = match page {
                Page::SpringRace => build_spring_race_page(w, h, &springs),
                Page::SequenceScreenplay => build_screenplay_page(w, h, &seq_player),
                Page::PresetGallery => build_preset_page(w, h, &preset_players),
            };

            frame.render_stateful_widget(
                PixelCanvasWidget::new(canvas).z_index(-1),
                area,
                &mut state,
            );

            let status = Paragraph::new(format!(
                " ← → page ({}/{})  |  {}  |  {fps:.0} fps  |  r restart  |  q quit",
                page_idx + 1,
                Page::ALL.len(),
                page.label(),
            ))
            .block(Block::default().borders(Borders::TOP));
            frame.render_widget(status, chunks[1]);
        })?;

        state.flush()?;

        if event::poll(Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Right | KeyCode::Char('l') => {
                            page_idx = (page_idx + 1) % Page::ALL.len();
                        }
                        KeyCode::Left | KeyCode::Char('h') => {
                            page_idx = page_idx.checked_sub(1).unwrap_or(Page::ALL.len() - 1);
                        }
                        KeyCode::Char('r') => {
                            springs = create_spring_race();
                            seq_player = create_screenplay();
                            preset_players = create_preset_gallery();
                            preset_start = Instant::now();
                        }
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

// ═══════════════════════════════════════════════════════════════════
// Page 1: Spring Race
// ═══════════════════════════════════════════════════════════════════

type SpringEntry = (Spring<f32>, &'static str, C);

fn create_spring_race() -> Vec<SpringEntry> {
    vec![
        (
            Spring::new(0.0, 1.0, SpringConfig::GENTLE),
            "GENTLE",
            C::from_hsl(200.0, 0.8, 0.6),
        ),
        (
            Spring::new(0.0, 1.0, SpringConfig::BOUNCY),
            "BOUNCY",
            C::from_hsl(340.0, 0.9, 0.6),
        ),
        (
            Spring::new(0.0, 1.0, SpringConfig::STIFF),
            "STIFF",
            C::from_hsl(160.0, 0.8, 0.5),
        ),
        (
            Spring::new(0.0, 1.0, SpringConfig::SLOW),
            "SLOW",
            C::from_hsl(40.0, 0.9, 0.6),
        ),
        (
            Spring::new(0.0, 1.0, SpringConfig::SNAPPY),
            "SNAPPY",
            C::from_hsl(270.0, 0.8, 0.6),
        ),
    ]
}

#[allow(clippy::cast_precision_loss)]
fn build_spring_race_page(w: u32, h: u32, springs: &[SpringEntry]) -> PixelCanvas {
    let w_f = w as f32;
    let h_f = h as f32;

    let mut canvas = PixelCanvas::new(w, h).background(C::from_rgba8(10, 10, 22, 255));

    let margin_left = w_f * 0.12;
    let margin_right = w_f * 0.05;
    let track_width = w_f - margin_left - margin_right;
    let row_height = h_f / (springs.len() as f32 + 1.5);
    let radius = (row_height * 0.3).min(14.0);

    // Target line (vertical dashed)
    let target_x = margin_left + track_width;
    canvas = canvas
        .line(target_x, 10.0, target_x, h_f - 10.0)
        .color(C::from_rgba8(60, 60, 90, 150))
        .width(2.0)
        .done();

    for (i, (spring, label, color)) in springs.iter().enumerate() {
        let y = row_height * (i as f32 + 1.0);
        let val = spring.value();
        let x = margin_left + track_width * val;

        // Track line
        canvas = canvas
            .line(margin_left, y, margin_left + track_width, y)
            .color(C::from_rgba8(30, 30, 50, 255))
            .width(1.0)
            .done();

        // Trail behind ball
        let trail_w = (x - margin_left).max(0.0);
        if trail_w > 1.0 {
            canvas = canvas
                .rect(margin_left, y - radius * 0.25, trail_w, radius * 0.5)
                .fill(color.with_alpha(0.12))
                .done();
        }

        // Overshoot zone (if past target)
        if val > 1.0 {
            let overshoot_start = margin_left + track_width;
            let overshoot_w = (x - overshoot_start).max(0.0);
            canvas = canvas
                .rect(
                    overshoot_start,
                    y - radius * 0.35,
                    overshoot_w,
                    radius * 0.7,
                )
                .fill(C::from_rgba8(255, 60, 60, 40))
                .done();
        }

        // Ball
        let glow_alpha = if spring.is_settled() { 0.0 } else { 0.15 };
        canvas = canvas
            .circle(x, y, radius * 1.5)
            .fill(color.with_alpha(glow_alpha))
            .done()
            .circle(x, y, radius)
            .fill(*color)
            .done();

        // Settled indicator
        if spring.is_settled() {
            canvas = canvas
                .circle(margin_left - 12.0, y, 4.0)
                .fill(C::from_hsl(120.0, 0.8, 0.5))
                .done();
        }

        // Label (small rect as a visual tag)
        let _label_w = label.len() as f32 * 6.0;
        canvas = canvas
            .rect(8.0, y - 3.0, margin_left - 20.0, 6.0)
            .fill(color.with_alpha(0.3))
            .corner_radius(3.0)
            .done();
    }

    canvas
}

// ═══════════════════════════════════════════════════════════════════
// Page 2: Coroutine Sequence Screenplay
// ═══════════════════════════════════════════════════════════════════

fn create_screenplay() -> SequencePlayer {
    let seq = AnimationSequence::new()
        // Act 1: Fade in
        .tween("opacity", 0.0, 1.0, ms(400), Easing::EaseOutCubic)
        .tween("radius", 5.0, 25.0, ms(400), Easing::EaseOutCubic)
        // Beat: pause
        .wait(ms(200))
        // Act 2: Spring pop to full size
        .spring_to("radius", 25.0, 50.0, SpringConfig::BOUNCY)
        // Beat: pause
        .wait(ms(150))
        // Act 3: Parallel — slide right + rotate hue
        .parallel(|p| {
            p.branch(|b| b.tween("x_norm", 0.2, 0.8, ms(800), Easing::EaseInOutCubic))
                .branch(|b| b.tween("hue", 200.0, 360.0, ms(800), Easing::Linear))
        })
        // Beat
        .wait(ms(200))
        // Act 4: Shrink + fade out simultaneously
        .parallel(|p| {
            p.branch(|b| b.tween("opacity", 1.0, 0.0, ms(500), Easing::EaseInCubic))
                .branch(|b| b.tween("radius", 50.0, 5.0, ms(500), Easing::EaseInQuad))
                .branch(|b| b.tween("x_norm", 0.8, 0.5, ms(500), Easing::EaseInCubic))
        });

    SequencePlayer::new(seq)
}

#[allow(clippy::cast_precision_loss)]
fn build_screenplay_page(w: u32, h: u32, player: &SequencePlayer) -> PixelCanvas {
    let w_f = w as f32;
    let h_f = h as f32;

    let mut canvas = PixelCanvas::new(w, h).background(C::from_rgba8(8, 8, 18, 255));

    let opacity = player.get("opacity").unwrap_or(0.0);
    let radius = player.get("radius").unwrap_or(5.0);
    let x_norm = player.get("x_norm").unwrap_or(0.2);
    let hue = player.get("hue").unwrap_or(200.0);

    let cx = w_f * x_norm;
    let cy = h_f * 0.45;
    let color = C::from_hsla(hue, 0.85, 0.55, opacity);

    // Stage reference — subtle center line
    canvas = canvas
        .line(w_f * 0.1, cy, w_f * 0.9, cy)
        .color(C::from_rgba8(30, 30, 50, 80))
        .width(1.0)
        .done();

    // Stage markers at key positions
    for &mark_x in &[0.2_f32, 0.5, 0.8] {
        canvas = canvas
            .circle(w_f * mark_x, cy, 3.0)
            .fill(C::from_rgba8(50, 50, 80, 120))
            .done();
    }

    // Glow
    if radius > 10.0 {
        canvas = canvas
            .circle(cx, cy, radius * 2.0)
            .fill(color.with_alpha(opacity * 0.06))
            .done()
            .circle(cx, cy, radius * 1.4)
            .fill(color.with_alpha(opacity * 0.12))
            .done();
    }

    // Main circle
    canvas = canvas
        .circle(cx, cy, radius)
        .fill(color)
        .stroke(C::WHITE.with_alpha(opacity * 0.4), 2.0)
        .done();

    // Step indicator at bottom — show which values are active
    let indicators = [
        ("opacity", opacity),
        ("radius", radius / 50.0),
        ("x_pos", x_norm),
        ("hue", hue / 360.0),
    ];

    let bar_w = w_f * 0.15;
    let bar_h = 8.0;
    let bar_y = h_f * 0.85;
    let total_w = bar_w * indicators.len() as f32 + 10.0 * (indicators.len() as f32 - 1.0);
    let start_x = (w_f - total_w) / 2.0;

    for (i, (_name, val)) in indicators.iter().enumerate() {
        let bx = start_x + (bar_w + 10.0) * i as f32;
        let indicator_hue = 200.0 + 40.0 * i as f32;

        // Background
        canvas = canvas
            .rect(bx, bar_y, bar_w, bar_h)
            .fill(C::from_rgba8(25, 25, 45, 255))
            .corner_radius(4.0)
            .done();

        // Fill
        let fill_w = (bar_w * val.clamp(0.0, 1.0)).max(0.0);
        if fill_w > 0.5 {
            canvas = canvas
                .rect(bx, bar_y, fill_w, bar_h)
                .fill(C::from_hsl(indicator_hue, 0.7, 0.5))
                .corner_radius(4.0)
                .done();
        }
    }

    canvas
}

// ═══════════════════════════════════════════════════════════════════
// Page 3: Preset Gallery
// ═══════════════════════════════════════════════════════════════════

fn create_preset_gallery() -> Vec<SequencePlayer> {
    vec![
        SequencePlayer::new(preset::fade_in("v", ms(600))),
        SequencePlayer::new(preset::pop_in("v")),
        SequencePlayer::new(preset::bounce_in("v")),
        SequencePlayer::new(preset::pulse("v", ms(800))),
        SequencePlayer::new(preset::shake("v", 1.0, ms(600))),
    ]
}

#[allow(clippy::cast_precision_loss)]
fn build_preset_page(w: u32, h: u32, players: &[SequencePlayer]) -> PixelCanvas {
    let w_f = w as f32;
    let h_f = h as f32;

    let mut canvas = PixelCanvas::new(w, h).background(C::from_rgba8(10, 10, 22, 255));

    let names = ["fade_in", "pop_in", "bounce_in", "pulse", "shake"];
    let colors = [
        C::from_hsl(200.0, 0.8, 0.6),
        C::from_hsl(340.0, 0.9, 0.6),
        C::from_hsl(160.0, 0.8, 0.5),
        C::from_hsl(50.0, 0.9, 0.6),
        C::from_hsl(270.0, 0.8, 0.6),
    ];

    let n = players.len() as f32;
    let col_w = w_f / n;
    let base_radius = (col_w * 0.25).min(30.0);

    for (i, (player, _name)) in players.iter().zip(names.iter()).enumerate() {
        let cx = col_w * (i as f32 + 0.5);
        let cy = h_f * 0.45;
        let val = player.get("v").unwrap_or(0.0);
        let color = colors[i];

        // Each preset drives the value differently:
        match i {
            0 => {
                // fade_in: val = opacity (0→1)
                let opacity = val.clamp(0.0, 1.0);
                canvas = canvas
                    .circle(cx, cy, base_radius * 1.4)
                    .fill(color.with_alpha(opacity * 0.1))
                    .done()
                    .circle(cx, cy, base_radius)
                    .fill(color.with_alpha(opacity))
                    .done();
            }
            1 | 2 => {
                // pop_in / bounce_in: val = scale (0→1 with overshoot)
                let scale = val.max(0.0);
                let r = base_radius * scale;
                if r > 0.5 {
                    canvas = canvas
                        .circle(cx, cy, r * 1.5)
                        .fill(color.with_alpha(0.08))
                        .done()
                        .circle(cx, cy, r)
                        .fill(color)
                        .done();
                }
            }
            3 => {
                // pulse: val = scale (1.0 → 1.2 → 1.0)
                let scale = val.max(0.0);
                let r = base_radius * scale;
                canvas = canvas
                    .circle(cx, cy, r * 1.3)
                    .fill(color.with_alpha(0.1))
                    .done()
                    .circle(cx, cy, r)
                    .fill(color)
                    .done();
            }
            4 => {
                // shake: val = x-offset (-1..1 normalized)
                let offset = val * base_radius * 2.0;
                canvas = canvas
                    .circle(cx + offset, cy, base_radius)
                    .fill(color)
                    .done();
            }
            _ => {}
        }

        // Label tag (colored bar below each)
        let tag_w = col_w * 0.6;
        let tag_x = cx - tag_w / 2.0;
        canvas = canvas
            .rect(tag_x, h_f * 0.72, tag_w, 4.0)
            .fill(color.with_alpha(0.6))
            .corner_radius(2.0)
            .done();

        // Completion indicator
        if player.is_complete() {
            canvas = canvas
                .circle(cx, h_f * 0.80, 4.0)
                .fill(C::from_hsl(120.0, 0.8, 0.5))
                .done();
        }
    }

    canvas
}
