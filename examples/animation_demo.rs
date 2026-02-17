//! Animation system demo — showcases every animation primitive.
//!
//! Demonstrates `Transition`, `Keyframes`, `AnimationState`, and all
//! major easing curves by animating shapes, colors, and transforms
//! in real time.
//!
//! Run with: `cargo run --example animation_demo`
//! Window:   `cargo run --example animation_demo --features window -- --window`

use std::io::stdout;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use scry_engine::prelude::{
    AnimationState, Easing, Keyframe, Keyframes, Lerp, Picker, PixelCanvasState, PixelCanvasWidget,
    ProtocolKind,
};
use scry_engine::scene::style::Point;
use scry_engine::scene::PixelCanvas;
use scry_engine::style::Color as PxColor;
use scry_engine::transport;

// ───────────────────────────────────────────────────────────────────
// Demo pages
// ───────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Page {
    EasingShowcase,
    KeyframeTimeline,
    AnimationOrchestrator,
    ColorTransitions,
}

impl Page {
    const ALL: [Self; 4] = [
        Self::EasingShowcase,
        Self::KeyframeTimeline,
        Self::AnimationOrchestrator,
        Self::ColorTransitions,
    ];

    const fn label(self) -> &'static str {
        match self {
            Self::EasingShowcase => "Easing Showcase",
            Self::KeyframeTimeline => "Keyframe Timeline",
            Self::AnimationOrchestrator => "Animation Orchestrator",
            Self::ColorTransitions => "Color Transitions",
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// Window mode
// ───────────────────────────────────────────────────────────────────

#[cfg(feature = "window")]
fn run_window() -> Result<(), Box<dyn std::error::Error>> {
    use scry_engine::rasterize::Rasterizer;
    use scry_engine::transport::window::{run_loop_continuous, LoopAction};
    use winit::keyboard::KeyCode as WKey;

    let mut page_idx: usize = 0;
    let start = Instant::now();
    let mut last_frame = start;

    let mut anim_state = AnimationState::new();
    setup_orchestrator_anims(&mut anim_state);

    run_loop_continuous(
        960,
        640,
        "Animation Demo",
        true,
        move |backend, keys, (w, h)| {
            let now = Instant::now();
            let dt = now - last_frame;
            last_frame = now;
            let elapsed = now.duration_since(start);
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
                        setup_orchestrator_anims(&mut anim_state);
                    }
                    _ => {}
                }
            }

            if page == Page::AnimationOrchestrator {
                anim_state.tick(dt);
                if anim_state.is_idle() {
                    setup_orchestrator_anims(&mut anim_state);
                }
            }

            let canvas = match page {
                Page::EasingShowcase => build_easing_page(w, h, elapsed),
                Page::KeyframeTimeline => build_keyframe_page(w, h, elapsed),
                Page::AnimationOrchestrator => build_orchestrator_page(w, h, &anim_state),
                Page::ColorTransitions => build_color_page(w, h, elapsed),
            };

            if let Ok(pixmap) = Rasterizer::rasterize(&canvas) {
                let _ = backend.blit(&pixmap);
            }
            LoopAction::Continue
        },
    )?;

    Ok(())
}

// ───────────────────────────────────────────────────────────────────
// Main
// ───────────────────────────────────────────────────────────────────

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

    // FPS tracking — rolling window of last 60 frame times
    let mut frame_times: Vec<Duration> = Vec::with_capacity(60);
    let mut fps: f64 = 0.0;
    let mut draw_ms: f64 = 0.0;
    let mut flush_ms: f64 = 0.0;

    // AnimationState for the orchestrator page
    let mut anim_state = AnimationState::new();
    setup_orchestrator_anims(&mut anim_state);

    loop {
        let now = Instant::now();
        let dt = now - last_frame;
        last_frame = now;
        let elapsed = now.duration_since(start);
        let page = Page::ALL[page_idx];

        // Update FPS counter
        frame_times.push(dt);
        if frame_times.len() > 60 {
            frame_times.remove(0);
        }
        if !frame_times.is_empty() {
            let total: Duration = frame_times.iter().sum();
            fps = frame_times.len() as f64 / total.as_secs_f64();
        }

        // Tick the orchestrator
        if page == Page::AnimationOrchestrator {
            anim_state.tick(dt);
            if anim_state.is_idle() {
                setup_orchestrator_anims(&mut anim_state);
            }
        }

        // --- Timed: terminal.draw (scene build + rasterize) ---
        let draw_start = Instant::now();
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
                Page::EasingShowcase => build_easing_page(w, h, elapsed),
                Page::KeyframeTimeline => build_keyframe_page(w, h, elapsed),
                Page::AnimationOrchestrator => {
                    build_orchestrator_page(w, h, &anim_state)
                }
                Page::ColorTransitions => build_color_page(w, h, elapsed),
            };

            frame.render_stateful_widget(
                PixelCanvasWidget::new(canvas).z_index(-1),
                area,
                &mut state,
            );

            let status = Paragraph::new(format!(
                " ← → page ({}/{})  |  {}  |  {fps:.1} fps  |  draw {draw_ms:.0}ms flush {flush_ms:.0}ms  |  r restart  |  q quit",
                page_idx + 1,
                Page::ALL.len(),
                page.label(),
                draw_ms = draw_ms,
                flush_ms = flush_ms,
            ))
            .block(Block::default().borders(Borders::TOP));
            frame.render_widget(status, chunks[1]);
        })?;
        let draw_elapsed = draw_start.elapsed();

        // --- Timed: flush (protocol transmission) ---
        let flush_start = Instant::now();
        state.flush()?;
        let flush_elapsed = flush_start.elapsed();

        // Update timing for next frame's display
        draw_ms = draw_elapsed.as_secs_f64() * 1000.0;
        flush_ms = flush_elapsed.as_secs_f64() * 1000.0;

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
                            setup_orchestrator_anims(&mut anim_state);
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

// ───────────────────────────────────────────────────────────────────
// Page 1: Easing Showcase
// ───────────────────────────────────────────────────────────────────

/// Animates circles across the screen, each using a different easing curve.
/// The animation loops every 3 seconds.
#[allow(clippy::cast_precision_loss)]
fn build_easing_page(w: u32, h: u32, elapsed: Duration) -> PixelCanvas {
    let w_f = w as f32;
    let h_f = h as f32;

    let easings: Vec<(&str, Easing)> = vec![
        ("Linear", Easing::Linear),
        ("EaseInQuad", Easing::EaseInQuad),
        ("EaseOutCubic", Easing::EaseOutCubic),
        ("EaseInOutQuart", Easing::EaseInOutQuart),
        ("EaseInSine", Easing::EaseInSine),
        ("EaseOutExpo", Easing::EaseOutExpo),
        ("EaseInOutCirc", Easing::EaseInOutCirc),
        ("Spring", Easing::BACK),
        ("Bounce", Easing::Bounce),
        ("Elastic", Easing::Elastic),
        ("CSS ease", Easing::CSS_EASE),
    ];

    // Loop period = 3s (ping-pong)
    let period = 3.0_f32;
    let raw_t = (elapsed.as_secs_f32() % (period * 2.0)) / period;
    let linear_t = if raw_t > 1.0 { 2.0 - raw_t } else { raw_t };

    let margin_left = w_f * 0.15;
    let margin_right = w_f * 0.05;
    let track_width = w_f - margin_left - margin_right;
    let row_height = h_f / (easings.len() as f32 + 1.0);
    let radius = (row_height * 0.3).min(12.0);

    let mut canvas = PixelCanvas::new(w, h).background(PxColor::from_rgba8(12, 12, 24, 255));

    // Track lines and animated circles
    for (i, (_, easing)) in easings.iter().enumerate() {
        let y = row_height * (i as f32 + 1.0);
        let eased_t = easing.ease(linear_t);
        let x = margin_left + track_width * eased_t;

        // Track line (dim)
        canvas = canvas
            .line(margin_left, y, margin_left + track_width, y)
            .color(PxColor::from_rgba8(40, 40, 60, 255))
            .width(1.0)
            .done();

        // Hue cycles through the rainbow per row
        let hue = 360.0 * (i as f32 / easings.len() as f32);
        let color = PxColor::from_hsl(hue, 0.8, 0.6);

        // Trail (fading rect behind the ball)
        let trail_w = (x - margin_left).max(0.0);
        if trail_w > 1.0 {
            canvas = canvas
                .rect(margin_left, y - radius * 0.3, trail_w, radius * 0.6)
                .fill(color.with_alpha(0.15))
                .done();
        }

        // Animated circle
        canvas = canvas.circle(x, y, radius).fill(color).done();
    }

    canvas
}

// ───────────────────────────────────────────────────────────────────
// Page 2: Keyframe Timeline
// ───────────────────────────────────────────────────────────────────

/// Demonstrates `Keyframes<T>` with a multi-stop position + color timeline.
#[allow(clippy::cast_precision_loss)]
fn build_keyframe_page(w: u32, h: u32, elapsed: Duration) -> PixelCanvas {
    let w_f = w as f32;
    let h_f = h as f32;

    // Keyframe timeline: position traces a diamond path
    let pos_kf = Keyframes::new(vec![
        Keyframe {
            position: 0.0,
            value: Point::new(w_f * 0.2, h_f * 0.5),
            easing: Easing::EaseInOutCubic,
        },
        Keyframe {
            position: 0.25,
            value: Point::new(w_f * 0.5, h_f * 0.15),
            easing: Easing::EaseInOutCubic,
        },
        Keyframe {
            position: 0.5,
            value: Point::new(w_f * 0.8, h_f * 0.5),
            easing: Easing::EaseInOutCubic,
        },
        Keyframe {
            position: 0.75,
            value: Point::new(w_f * 0.5, h_f * 0.85),
            easing: Easing::EaseInOutCubic,
        },
        Keyframe {
            position: 1.0,
            value: Point::new(w_f * 0.2, h_f * 0.5),
            easing: Easing::Linear,
        },
    ]);

    // Color keyframes: cycle through rainbow
    let color_kf = Keyframes::new(vec![
        Keyframe {
            position: 0.0,
            value: PxColor::from_hsl(0.0, 0.9, 0.6),
            easing: Easing::Linear,
        },
        Keyframe {
            position: 0.33,
            value: PxColor::from_hsl(120.0, 0.9, 0.6),
            easing: Easing::Linear,
        },
        Keyframe {
            position: 0.66,
            value: PxColor::from_hsl(240.0, 0.9, 0.6),
            easing: Easing::Linear,
        },
        Keyframe {
            position: 1.0,
            value: PxColor::from_hsl(360.0, 0.9, 0.6),
            easing: Easing::Linear,
        },
    ]);

    // Size keyframes: breathe effect
    let size_kf = Keyframes::new(vec![
        Keyframe {
            position: 0.0,
            value: 15.0_f32,
            easing: Easing::EaseInOutSine,
        },
        Keyframe {
            position: 0.25,
            value: 30.0,
            easing: Easing::EaseInOutSine,
        },
        Keyframe {
            position: 0.5,
            value: 15.0,
            easing: Easing::EaseInOutSine,
        },
        Keyframe {
            position: 0.75,
            value: 40.0,
            easing: Easing::EaseInOutSine,
        },
        Keyframe {
            position: 1.0,
            value: 15.0,
            easing: Easing::Linear,
        },
    ]);

    let period = 4.0_f32;
    let t = (elapsed.as_secs_f32() % period) / period;

    let pos = pos_kf.value_at(t);
    let color = color_kf.value_at(t);
    let size = size_kf.value_at(t);

    let mut canvas = PixelCanvas::new(w, h).background(PxColor::from_rgba8(8, 8, 18, 255));

    // Draw path preview (diamond outline)
    let diamond = [
        (w_f * 0.2, h_f * 0.5),
        (w_f * 0.5, h_f * 0.15),
        (w_f * 0.8, h_f * 0.5),
        (w_f * 0.5, h_f * 0.85),
    ];
    canvas = canvas
        .polygon(diamond.to_vec())
        .stroke(PxColor::from_rgba8(50, 50, 80, 120), 1.5)
        .done();

    // Draw keyframe markers
    for &(x, y) in &diamond {
        canvas = canvas
            .circle(x, y, 4.0)
            .fill(PxColor::from_rgba8(100, 100, 140, 180))
            .done();
    }

    // Trail: draw 20 ghost positions behind
    let trail_count = 20;
    for i in 0..trail_count {
        let trail_t = (i as f32).mul_add(-0.01, t).rem_euclid(1.0);
        let trail_pos = pos_kf.value_at(trail_t);
        let trail_color = color_kf.value_at(trail_t);
        let alpha = 0.3 * (1.0 - i as f32 / trail_count as f32);
        canvas = canvas
            .circle(trail_pos.x, trail_pos.y, size * 0.4)
            .fill(trail_color.with_alpha(alpha))
            .done();
    }

    // Main animated circle
    canvas = canvas
        .circle(pos.x, pos.y, size)
        .fill(color.with_alpha(0.8))
        .stroke(PxColor::WHITE.with_alpha(0.6), 2.0)
        .done();

    // Progress bar at bottom
    let bar_y = h_f * 0.95;
    let bar_w = w_f * 0.8;
    let bar_x = w_f * 0.1;
    canvas = canvas
        .rect(bar_x, bar_y, bar_w, 4.0)
        .fill(PxColor::from_rgba8(30, 30, 50, 255))
        .done()
        .rect(bar_x, bar_y, bar_w * t, 4.0)
        .fill(color)
        .done();

    canvas
}

// ───────────────────────────────────────────────────────────────────
// Page 3: AnimationState Orchestrator
// ───────────────────────────────────────────────────────────────────

fn setup_orchestrator_anims(anim: &mut AnimationState) {
    anim.cancel_all();
    anim.start(
        "x",
        0.0_f32,
        1.0_f32,
        Duration::from_secs(2),
        Easing::EaseInOutCubic,
    );
    anim.start(
        "y",
        0.0_f32,
        1.0_f32,
        Duration::from_millis(3000),
        Easing::Bounce,
    );
    anim.start(
        "scale",
        0.3_f32,
        1.5_f32,
        Duration::from_millis(2500),
        Easing::EaseInOutQuart,
    );
    anim.start(
        "hue",
        0.0_f32,
        360.0_f32,
        Duration::from_secs(4),
        Easing::Linear,
    );
    anim.start(
        "opacity",
        0.2_f32,
        1.0_f32,
        Duration::from_millis(1500),
        Easing::EaseOutExpo,
    );
}

/// Shows `AnimationState` managing 5 independent named transitions.
#[allow(clippy::cast_precision_loss)]
fn build_orchestrator_page(w: u32, h: u32, anim: &AnimationState) -> PixelCanvas {
    let w_f = w as f32;
    let h_f = h as f32;

    let x_norm = anim.get::<f32>("x").unwrap_or(0.0);
    let y_norm = anim.get::<f32>("y").unwrap_or(0.0);
    let scale = anim.get::<f32>("scale").unwrap_or(1.0);
    let hue = anim.get::<f32>("hue").unwrap_or(0.0);
    let opacity = anim.get::<f32>("opacity").unwrap_or(1.0);

    let margin = 60.0;
    let cx = 2.0f32.mul_add(-margin, w_f).mul_add(x_norm, margin);
    let cy = 2.0f32.mul_add(-margin, h_f).mul_add(y_norm, margin);
    let radius = 25.0 * scale;

    let color = PxColor::from_hsla(hue, 0.85, 0.55, opacity);

    let mut canvas = PixelCanvas::new(w, h).background(PxColor::from_rgba8(10, 10, 22, 255));

    // Grid lines for reference
    for i in 0..=10 {
        let frac = i as f32 / 10.0;
        let gx = 2.0f32.mul_add(-margin, w_f).mul_add(frac, margin);
        let gy = 2.0f32.mul_add(-margin, h_f).mul_add(frac, margin);
        canvas = canvas
            .line(gx, margin, gx, h_f - margin)
            .color(PxColor::from_rgba8(25, 25, 40, 255))
            .width(0.5)
            .done()
            .line(margin, gy, w_f - margin, gy)
            .color(PxColor::from_rgba8(25, 25, 40, 255))
            .width(0.5)
            .done();
    }

    // Glow ring behind main circle
    canvas = canvas
        .circle(cx, cy, radius * 1.8)
        .fill(color.with_alpha(0.08))
        .done()
        .circle(cx, cy, radius * 1.3)
        .fill(color.with_alpha(0.15))
        .done();

    // Main animated circle
    canvas = canvas
        .circle(cx, cy, radius)
        .fill(color)
        .stroke(PxColor::WHITE.with_alpha(0.4), 2.0)
        .done();

    // Status indicators for each animation (bottom of screen)
    let names = ["x", "y", "scale", "hue", "opacity"];
    let bar_h = 6.0;
    let bar_spacing = w_f * 0.8 / names.len() as f32;
    let bar_start = w_f * 0.1;

    for (i, &name) in names.iter().enumerate() {
        let bx = bar_start + bar_spacing * i as f32;
        let by = h_f - 30.0;
        let bar_w = bar_spacing * 0.8;

        // Background bar
        canvas = canvas
            .rect(bx, by, bar_w, bar_h)
            .fill(PxColor::from_rgba8(30, 30, 50, 255))
            .corner_radius(2.0)
            .done();

        // Fill based on current value
        let fill_frac = match name {
            "x" => x_norm,
            "y" => y_norm,
            "scale" => (scale - 0.3) / 1.2,
            "hue" => hue / 360.0,
            "opacity" => opacity,
            _ => 0.0,
        };

        let bar_color = PxColor::from_hsl(360.0 * i as f32 / names.len() as f32, 0.7, 0.5);
        canvas = canvas
            .rect(bx, by, bar_w * fill_frac.clamp(0.0, 1.0), bar_h)
            .fill(bar_color)
            .corner_radius(2.0)
            .done();
    }

    canvas
}

// ───────────────────────────────────────────────────────────────────
// Page 4: Color Transitions
// ───────────────────────────────────────────────────────────────────

/// Demonstrates `Color::mix`, `Color::from_hsla`, and `Transition<Color>`.
#[allow(clippy::cast_precision_loss)]
fn build_color_page(w: u32, h: u32, elapsed: Duration) -> PixelCanvas {
    let w_f = w as f32;
    let h_f = h as f32;

    let mut canvas = PixelCanvas::new(w, h).background(PxColor::from_rgba8(8, 8, 16, 255));

    // Section 1: HSL color wheel (top half)
    let center_x = w_f * 0.3;
    let center_y = h_f * 0.35;
    let ring_radius = (h_f * 0.25).min(w_f * 0.2);
    let dot_radius = (ring_radius * 0.08).max(4.0);
    let n_dots = 36;

    let time_offset = elapsed.as_secs_f32() * 30.0; // slowly rotate

    for i in 0..n_dots {
        let hue = (360.0 * i as f32 / n_dots as f32 + time_offset) % 360.0;
        let angle = std::f32::consts::TAU * i as f32 / n_dots as f32;
        let x = center_x + ring_radius * angle.cos();
        let y = center_y + ring_radius * angle.sin();
        let color = PxColor::from_hsl(hue, 0.9, 0.6);

        canvas = canvas.circle(x, y, dot_radius).fill(color).done();
    }

    // Inner breathing circle using HSL
    let breath_t = (elapsed.as_secs_f32() * 0.5).sin().mul_add(0.5, 0.5); // 0..1
    let inner_hue = elapsed.as_secs_f32() * 60.0 % 360.0;
    let inner_color = PxColor::from_hsla(inner_hue, 0.8, 0.4f32.mul_add(breath_t, 0.3), 0.7);
    canvas = canvas
        .circle(
            center_x,
            center_y,
            ring_radius * 0.5 * 0.4f32.mul_add(breath_t, 0.6),
        )
        .fill(inner_color)
        .done();

    // Section 2: Color.mix gradient strips (right side)
    let gradient_x = w_f * 0.55;
    let gradient_w = w_f * 0.4;
    let strip_h = h_f * 0.06;
    let strip_gap = h_f * 0.02;

    let color_pairs: Vec<(PxColor, PxColor, &str)> = vec![
        (
            PxColor::from_hsl(0.0, 0.9, 0.5),
            PxColor::from_hsl(240.0, 0.9, 0.5),
            "Red→Blue",
        ),
        (
            PxColor::from_hsl(120.0, 0.9, 0.5),
            PxColor::from_hsl(60.0, 0.9, 0.9),
            "Green→Yellow",
        ),
        (PxColor::BLACK, PxColor::WHITE, "Black→White"),
        (
            PxColor::from_hsl(280.0, 0.9, 0.4),
            PxColor::from_hsl(30.0, 1.0, 0.6),
            "Purple→Orange",
        ),
    ];

    let n_segments = 32;
    let seg_width = gradient_w / n_segments as f32;

    for (row, (from, to, _label)) in color_pairs.iter().enumerate() {
        let y = h_f.mul_add(0.1, (strip_h + strip_gap) * row as f32);

        for seg in 0..n_segments {
            let t = seg as f32 / (n_segments - 1) as f32;
            let mixed = from.mix(*to, t);
            let x = gradient_x + seg_width * seg as f32;

            canvas = canvas
                .rect(x, y, seg_width + 0.5, strip_h)
                .fill(mixed)
                .done();
        }
    }

    // Section 3: Animated color transition (bottom)
    let transition_y = h_f * 0.7;
    let period = 5.0;
    let t = (elapsed.as_secs_f32() % period) / period;
    let eased_t = Easing::EaseInOutCubic.ease(t);

    // Lerp between two colors and show a large animated swatch
    let from = PxColor::from_hsl(200.0, 0.9, 0.4);
    let to = PxColor::from_hsl(350.0, 0.9, 0.6);
    let current = from.lerp(&to, eased_t);

    let swatch_w = w_f * 0.35;
    let swatch_h = h_f * 0.18;

    // "From" swatch
    canvas = canvas
        .rect(w_f * 0.1, transition_y, swatch_w * 0.3, swatch_h)
        .fill(from)
        .corner_radius(6.0)
        .done();

    // Animated swatch (center, large)
    canvas = canvas
        .rect(w_f * 0.3, transition_y - 10.0, swatch_w, swatch_h + 20.0)
        .fill(current)
        .corner_radius(10.0)
        .stroke(PxColor::WHITE.with_alpha(0.3), 2.0)
        .done();

    // "To" swatch
    canvas = canvas
        .rect(w_f * 0.7, transition_y, swatch_w * 0.3, swatch_h)
        .fill(to)
        .corner_radius(6.0)
        .done();

    // Arrow indicators
    let arrow_y = transition_y + swatch_h * 0.5;
    canvas = canvas
        .line(w_f * 0.22, arrow_y, w_f * 0.28, arrow_y)
        .color(PxColor::from_rgba8(180, 180, 200, 200))
        .width(2.0)
        .done()
        .line(w_f * 0.67, arrow_y, w_f * 0.73, arrow_y)
        .color(PxColor::from_rgba8(180, 180, 200, 200))
        .width(2.0)
        .done();

    canvas
}
