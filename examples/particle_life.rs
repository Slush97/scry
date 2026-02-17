//! Particle Life — emergent behavior from simple rules.
//!
//! Hundreds of colored particles interact through attraction/repulsion
//! rules, creating mesmerizing organic-looking patterns. Each particle
//! type has a different interaction with each other type.
//!
//! Demonstrates real-time rendering capability and the visual impact
//! of anti-aliased circles at small scales.
//!
//! Controls:
//! - `r` — randomize interaction rules
//! - `q` — quit
//!
//! Run with: `cargo run --example particle_life --release`
//! Window:   `cargo run --example particle_life --release --features window -- --window`

use std::io::stdout;
use std::time::Instant;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use scry_engine::prelude::{Picker, PixelCanvasState, PixelCanvasWidget, ProtocolKind};
use scry_engine::scene::PixelCanvas;
use scry_engine::style::Color as PxColor;
use scry_engine::transport;

const NUM_TYPES: usize = 5;
const PARTICLES_PER_TYPE: usize = 60;
const FORCE_RANGE: f32 = 80.0;
const FRICTION: f32 = 0.92;

/// Particle with position, velocity, and type index.
struct Particle {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    kind: usize,
}

/// Simulation state.
struct Simulation {
    particles: Vec<Particle>,
    /// Interaction matrix: rules[a][b] = force of type `b` on type `a`.
    /// Positive = attraction, negative = repulsion.
    rules: [[f32; NUM_TYPES]; NUM_TYPES],
    width: f32,
    height: f32,
}

impl Simulation {
    fn new(width: f32, height: f32) -> Self {
        let mut particles = Vec::with_capacity(NUM_TYPES * PARTICLES_PER_TYPE);
        for kind in 0..NUM_TYPES {
            for _ in 0..PARTICLES_PER_TYPE {
                particles.push(Particle {
                    x: fastrand::f32() * width,
                    y: fastrand::f32() * height,
                    vx: 0.0,
                    vy: 0.0,
                    kind,
                });
            }
        }

        let mut sim = Self {
            particles,
            rules: [[0.0; NUM_TYPES]; NUM_TYPES],
            width,
            height,
        };
        sim.randomize_rules();
        sim
    }

    fn randomize_rules(&mut self) {
        for a in 0..NUM_TYPES {
            for b in 0..NUM_TYPES {
                self.rules[a][b] = fastrand::f32().mul_add(2.0, -1.0); // -1 to +1
            }
        }
    }

    const fn resize(&mut self, w: f32, h: f32) {
        self.width = w;
        self.height = h;
    }

    fn step(&mut self, dt: f32) {
        let n = self.particles.len();

        // Collect forces (avoid borrow conflict)
        let mut forces: Vec<(f32, f32)> = vec![(0.0, 0.0); n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dx = self.particles[j].x - self.particles[i].x;
                let dy = self.particles[j].y - self.particles[i].y;
                let dist = dx.hypot(dy);

                if dist > 0.5 && dist < FORCE_RANGE {
                    let rule = self.rules[self.particles[i].kind][self.particles[j].kind];
                    // Force function: repel at very close range, then apply rule
                    let force = if dist < FORCE_RANGE * 0.3 {
                        // Strong repulsion at close range
                        -1.0 / dist.max(1.0) * 5.0
                    } else {
                        rule / dist * 3.0
                    };
                    let fx = dx / dist * force;
                    let fy = dy / dist * force;
                    forces[i].0 += fx;
                    forces[i].1 += fy;
                }
            }
        }

        // Apply forces and friction
        for (i, p) in self.particles.iter_mut().enumerate() {
            p.vx = forces[i].0.mul_add(dt, p.vx) * FRICTION;
            p.vy = forces[i].1.mul_add(dt, p.vy) * FRICTION;
            p.x += p.vx * dt;
            p.y += p.vy * dt;

            // Wrap around edges
            if p.x < 0.0 {
                p.x += self.width;
            }
            if p.x > self.width {
                p.x -= self.width;
            }
            if p.y < 0.0 {
                p.y += self.height;
            }
            if p.y > self.height {
                p.y -= self.height;
            }
        }
    }
}

const fn type_color(kind: usize) -> PxColor {
    match kind {
        0 => PxColor::from_rgba8(255, 80, 100, 255), // Coral red
        1 => PxColor::from_rgba8(80, 200, 255, 255), // Sky blue
        2 => PxColor::from_rgba8(120, 255, 120, 255), // Lime green
        3 => PxColor::from_rgba8(255, 200, 60, 255), // Gold
        4 => PxColor::from_rgba8(200, 120, 255, 255), // Purple
        _ => PxColor::from_rgba8(255, 255, 255, 255),
    }
}

// ═══════════════════════════════════════════════════════════════════
// Scene builder
// ═══════════════════════════════════════════════════════════════════

fn build_particle_scene(w: u32, h: u32, sim: &Simulation) -> PixelCanvas {
    let mut canvas = PixelCanvas::new(w, h).background(PxColor::from_rgba8(8, 8, 14, 255));

    // Draw particles with glow effect
    for p in &sim.particles {
        let color = type_color(p.kind);
        // Outer glow
        canvas = canvas
            .circle(p.x, p.y, 5.0)
            .fill(color.with_alpha(0.15))
            .done();
        // Inner body
        canvas = canvas.circle(p.x, p.y, 2.5).fill(color).done();
    }

    canvas
}

// ═══════════════════════════════════════════════════════════════════
// Window mode
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "window")]
fn run_window() -> Result<(), Box<dyn std::error::Error>> {
    use scry_engine::rasterize::Rasterizer;
    use scry_engine::transport::window::{run_loop_continuous, LoopAction};
    use winit::keyboard::KeyCode as WKey;

    let mut sim = Simulation::new(960.0, 640.0);
    let mut last_frame = Instant::now();

    run_loop_continuous(
        960,
        640,
        "Particle Life",
        true,
        move |backend, keys, (w, h)| {
            let now = Instant::now();
            let dt = (now - last_frame).as_secs_f32().min(0.05);
            last_frame = now;

            for key in keys {
                if !key.pressed {
                    continue;
                }
                match key.code {
                    WKey::Escape | WKey::KeyQ => return LoopAction::Exit,
                    WKey::KeyR => sim.randomize_rules(),
                    _ => {}
                }
            }

            sim.resize(w as f32, h as f32);
            sim.step(dt * 20.0);

            let canvas = build_particle_scene(w, h, &sim);
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
    let mut sim = Simulation::new(400.0, 300.0);
    let start = Instant::now();
    let mut last_frame = Instant::now();

    loop {
        let now = Instant::now();
        let dt = (now - last_frame).as_secs_f32().min(0.05);
        last_frame = now;

        // Run physics
        sim.step(dt * 20.0);

        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(3)])
                .split(frame.area());

            let area = chunks[0];
            let font = state.font_size();
            let w = u32::from(area.width) * u32::from(font.width);
            let h = u32::from(area.height) * u32::from(font.height);

            sim.resize(w as f32, h as f32);

            let canvas = build_particle_scene(w, h, &sim);

            frame.render_stateful_widget(
                PixelCanvasWidget::new(canvas).skip_cache(),
                area,
                &mut state,
            );

            let elapsed = start.elapsed().as_secs_f64();
            let fps = 1.0 / f64::from(dt);
            let status = Paragraph::new(format!(
                " ▸ particle_life | {} particles | {fps:.0} FPS | {elapsed:.0}s | 'r' = randomize, 'q' = quit",
                sim.particles.len()
            ))
            .style(Style::default().fg(Color::DarkGray))
            .block(Block::default().borders(Borders::TOP).border_style(Style::default().fg(Color::DarkGray)));
            frame.render_widget(status, chunks[1]);
        })?;
        state.flush()?;

        if event::poll(std::time::Duration::from_millis(1))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Char('r') => sim.randomize_rules(),
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
