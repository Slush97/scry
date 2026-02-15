//! **Fluid Symphony** — curl-noise vector field with flowing particle trails.
//!
//! Thousands of particles flow through an animated 2D noise field, leaving
//! fading trails. The velocity maps to color (HSL hue), creating organic,
//! flowing patterns reminiscent of Van Gogh's Starry Night.
//!
//! Controls:
//!   `1`–`3`  — preset: Starry Night / Ink in Water / Solar Wind
//!   `r`      — reset particles
//!   `+`/`-`  — particle count
//!   `Space`  — pause/resume
//!   `q`      — quit
//!
//! Run with: `cargo run --example fluid_symphony --release`

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::similar_names,
    clippy::needless_range_loop
)]

use std::io::stdout;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use scry_engine::prelude::{Picker, PixelCanvasState, PixelCanvasWidget, ProtocolKind};
use scry_engine::scene::PixelCanvas;
use scry_engine::style::Color as C;
use scry_engine::transport;

// ═══════════════════════════════════════════════════════════════════
// Simplex-style noise (2D, self-contained — no external deps)
// ═══════════════════════════════════════════════════════════════════

/// Permutation table (doubled for wrapping)
const PERM: [u8; 512] = {
    let base: [u8; 256] = [
        151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30,
        69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94,
        252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171,
        168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60,
        211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1,
        216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86,
        164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118,
        126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170,
        213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39,
        253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34,
        242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49,
        192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254,
        138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
    ];
    let mut out = [0u8; 512];
    let mut i = 0;
    while i < 512 {
        out[i] = base[i & 255];
        i += 1;
    }
    out
};

/// Fast gradient noise (value noise with smooth interpolation).
fn noise2d(x: f32, y: f32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let xf = x - x.floor();
    let yf = y - y.floor();

    // Smooth interpolation
    let u = xf * xf * 2.0f32.mul_add(-xf, 3.0);
    let v = yf * yf * 2.0f32.mul_add(-yf, 3.0);

    let aa = PERM[(i32::from(PERM[(xi & 255) as usize]) + (yi & 255)) as usize & 511];
    let ab = PERM[(i32::from(PERM[(xi & 255) as usize]) + ((yi + 1) & 255)) as usize & 511];
    let ba = PERM[(i32::from(PERM[((xi + 1) & 255) as usize]) + (yi & 255)) as usize & 511];
    let bb = PERM[(i32::from(PERM[((xi + 1) & 255) as usize]) + ((yi + 1) & 255)) as usize & 511];

    let a = f32::from(aa) / 255.0;
    let b = f32::from(ba) / 255.0;
    let c = f32::from(ab) / 255.0;
    let d = f32::from(bb) / 255.0;

    let x1 = a + u * (b - a);
    let x2 = c + u * (d - c);
    x1 + v * (x2 - x1)
}

/// Multi-octave fractal noise (fBm).
fn fbm(x: f32, y: f32, octaves: u32) -> f32 {
    let mut value = 0.0;
    let mut amp = 0.5;
    let mut freq = 1.0;
    for _ in 0..octaves {
        value += amp * noise2d(x * freq, y * freq);
        freq *= 2.0;
        amp *= 0.5;
    }
    value
}

/// Curl of the noise field (divergence-free velocity field).
fn curl_noise(x: f32, y: f32, time: f32) -> (f32, f32) {
    let eps = 0.01;
    let n = fbm(x, time.mul_add(0.2, y), 4);
    let dx = (fbm(x + eps, time.mul_add(0.2, y), 4) - n) / eps;
    let dy = (fbm(x, time.mul_add(0.2, y + eps), 4) - n) / eps;
    // Curl: rotate gradient 90° for divergence-free flow
    (-dy, dx)
}

// ═══════════════════════════════════════════════════════════════════
// Particle system
// ═══════════════════════════════════════════════════════════════════

const TRAIL_LEN: usize = 20;

struct Particle {
    trail: [(f32, f32); TRAIL_LEN],
    trail_head: usize,
    trail_count: usize,
    speed: f32,
    life: f32,     // 0..1, fades out
    max_life: f32, // Total lifetime
}

impl Particle {
    const fn new(x: f32, y: f32, max_life: f32) -> Self {
        let mut trail = [(0.0_f32, 0.0_f32); TRAIL_LEN];
        trail[0] = (x, y);
        Self {
            trail,
            trail_head: 0,
            trail_count: 1,
            speed: 0.0,
            life: 0.0,
            max_life,
        }
    }

    const fn x(&self) -> f32 {
        self.trail[self.trail_head].0
    }

    const fn y(&self) -> f32 {
        self.trail[self.trail_head].1
    }

    fn update(&mut self, dt: f32, time: f32, noise_scale: f32, flow_strength: f32) {
        self.life += dt;
        if self.life > self.max_life {
            return;
        }

        let nx = self.x() * noise_scale;
        let ny = self.y() * noise_scale;
        let (vx, vy) = curl_noise(nx, ny, time);

        self.speed = vx.hypot(vy);

        let new_x = (vx * flow_strength).mul_add(dt, self.x());
        let new_y = (vy * flow_strength).mul_add(dt, self.y());

        // Advance trail ring buffer
        self.trail_head = (self.trail_head + 1) % TRAIL_LEN;
        self.trail[self.trail_head] = (new_x, new_y);
        self.trail_count = (self.trail_count + 1).min(TRAIL_LEN);
    }

    fn is_dead(&self) -> bool {
        self.life > self.max_life
    }

    fn is_oob(&self, w: f32, h: f32) -> bool {
        let margin = 20.0;
        self.x() < -margin || self.x() > w + margin || self.y() < -margin || self.y() > h + margin
    }

    /// Get trail points from oldest to newest.
    fn trail_points(&self) -> Vec<(f32, f32)> {
        let mut pts = Vec::with_capacity(self.trail_count);
        for i in 0..self.trail_count {
            let idx = (self.trail_head + TRAIL_LEN - self.trail_count + 1 + i) % TRAIL_LEN;
            pts.push(self.trail[idx]);
        }
        pts
    }
}

// ═══════════════════════════════════════════════════════════════════
// Presets
// ═══════════════════════════════════════════════════════════════════

#[derive(Clone, Copy)]
struct FlowPreset {
    name: &'static str,
    noise_scale: f32,
    flow_strength: f32,
    base_hue: f32,
    hue_range: f32,
    bg: (u8, u8, u8),
    trail_width: f32,
}

const PRESETS: [FlowPreset; 3] = [
    // Starry Night — deep indigos, warm yellows, electric teals
    FlowPreset {
        name: "Starry Night",
        noise_scale: 0.004,
        flow_strength: 120.0,
        base_hue: 220.0,
        hue_range: 160.0,
        bg: (8, 5, 20),
        trail_width: 1.8,
    },
    // Ink in Water — black ink on white with color bleeding
    FlowPreset {
        name: "Ink in Water",
        noise_scale: 0.006,
        flow_strength: 80.0,
        base_hue: 200.0,
        hue_range: 80.0,
        bg: (12, 12, 18),
        trail_width: 2.2,
    },
    // Solar Wind — reds, oranges, white-hot streamers
    FlowPreset {
        name: "Solar Wind",
        noise_scale: 0.003,
        flow_strength: 160.0,
        base_hue: 15.0,
        hue_range: 50.0,
        bg: (5, 2, 8),
        trail_width: 1.5,
    },
];

// ═══════════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════════

struct FluidState {
    particles: Vec<Particle>,
    preset_idx: usize,
    target_count: usize,
    paused: bool,
    canvas_w: f32,
    canvas_h: f32,
    /// Ever-incrementing counter for unique spawn seeding.
    spawn_counter: u32,
}

impl FluidState {
    const fn new() -> Self {
        Self {
            particles: Vec::new(),
            preset_idx: 0,
            target_count: 800,
            paused: false,
            canvas_w: 800.0,
            canvas_h: 600.0,
            spawn_counter: 0,
        }
    }

    const fn preset(&self) -> &FlowPreset {
        &PRESETS[self.preset_idx]
    }

    fn spawn_particle(&mut self) {
        // Use ever-incrementing counter for unique spatial scatter.
        // Using particles.len() causes clumping when the count stabilizes
        // because the same seed produces the same position repeatedly.
        let seed = self.spawn_counter as f32;
        self.spawn_counter = self.spawn_counter.wrapping_add(1);

        let x = (seed.mul_add(0.618_034, 0.3) % 1.0) * self.canvas_w; // Golden ratio scatter
        let y = (seed.mul_add(0.381_966_02, 0.7) % 1.0) * self.canvas_h; // 1 - phi
        let life = (seed * 0.171573 % 1.0).mul_add(5.0, 3.0);
        self.particles.push(Particle::new(x, y, life));
    }

    fn reset(&mut self) {
        self.particles.clear();
    }

    fn update(&mut self, dt: f32, time: f32) {
        if self.paused {
            return;
        }

        let p = self.preset();
        let ns = p.noise_scale;
        let fs = p.flow_strength;

        // Update existing particles
        for particle in &mut self.particles {
            particle.update(dt, time, ns, fs);
        }

        // Remove dead / oob particles
        let w = self.canvas_w;
        let h = self.canvas_h;
        self.particles.retain(|p| !p.is_dead() && !p.is_oob(w, h));

        // Spawn new particles to reach target count
        let spawn_rate = 20.min(self.target_count.saturating_sub(self.particles.len()));
        for _ in 0..spawn_rate {
            self.spawn_particle();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

fn main() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let picker = Picker::detect();
    let backend: Box<dyn transport::ProtocolBackend> = match picker.protocol() {
        ProtocolKind::Kitty => Box::new(transport::kitty::KittyBackend::new(picker.font_size())),
        _ => Box::new(transport::halfblock::HalfblockBackend::new()),
    };
    let mut px_state = PixelCanvasState::new(backend, picker.font_size());

    let mut fluid = FluidState::new();
    let start = Instant::now();
    let mut last_frame = Instant::now();
    let mut frozen_time = 0.0_f32;

    loop {
        let now = Instant::now();
        let dt = now.duration_since(last_frame).as_secs_f32().min(0.05); // Cap dt
        last_frame = now;

        let elapsed = if fluid.paused {
            frozen_time
        } else {
            let e = now.duration_since(start).as_secs_f32();
            frozen_time = e;
            e
        };

        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(3)])
                .split(frame.area());

            let area = chunks[0];

            // Update canvas dimensions
            let font = px_state.font_size();
            fluid.canvas_w = (u32::from(area.width) * u32::from(font.width)) as f32;
            fluid.canvas_h = (u32::from(area.height) * u32::from(font.height)) as f32;

            fluid.update(dt, elapsed);

            let canvas = build_fluid_scene(area, &px_state, &fluid, elapsed);

            frame.render_stateful_widget(
                PixelCanvasWidget::new(canvas).skip_cache().z_index(-1),
                area,
                &mut px_state,
            );

            let status_text = format!(
                " {} │ {} particles │ [1-3] preset [r] reset [+/-] count [space] pause [q] quit",
                fluid.preset().name,
                fluid.particles.len(),
            );
            let status = Paragraph::new(status_text).block(Block::default().borders(Borders::TOP));
            frame.render_widget(status, chunks[1]);
        })?;
        px_state.flush()?;

        if event::poll(Duration::from_millis(1))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Char('1') => {
                            fluid.preset_idx = 0;
                            fluid.reset();
                        }
                        KeyCode::Char('2') => {
                            fluid.preset_idx = 1;
                            fluid.reset();
                        }
                        KeyCode::Char('3') => {
                            fluid.preset_idx = 2;
                            fluid.reset();
                        }
                        KeyCode::Char('r') => fluid.reset(),
                        KeyCode::Char('+' | '=') => {
                            fluid.target_count = (fluid.target_count + 200).min(3000);
                        }
                        KeyCode::Char('-') => {
                            fluid.target_count = fluid.target_count.saturating_sub(200).max(100);
                        }
                        KeyCode::Char(' ') => fluid.paused = !fluid.paused,
                        _ => {}
                    }
                }
            }
        }
    }

    px_state.cleanup();
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════
// Scene builder
// ═══════════════════════════════════════════════════════════════════

fn build_fluid_scene(
    area: Rect,
    px_state: &PixelCanvasState,
    fluid: &FluidState,
    _time: f32,
) -> PixelCanvas {
    let font = px_state.font_size();
    let w = u32::from(area.width) * u32::from(font.width);
    let h = u32::from(area.height) * u32::from(font.height);
    if w == 0 || h == 0 {
        return PixelCanvas::new(1, 1);
    }

    let preset = fluid.preset();
    let bg = C::from_rgba8(preset.bg.0, preset.bg.1, preset.bg.2, 255);

    let mut canvas = PixelCanvas::new(w, h).background(bg);

    // Draw each particle's trail as a polyline with decreasing alpha
    for particle in &fluid.particles {
        if particle.trail_count < 2 {
            continue;
        }

        let trail = particle.trail_points();
        let life_frac = 1.0 - (particle.life / particle.max_life);
        let base_alpha = life_frac.clamp(0.0, 1.0);

        // Color based on speed
        let speed_norm = (particle.speed / 3.0).clamp(0.0, 1.0);
        let hue = (preset.base_hue + speed_norm * preset.hue_range) % 360.0;
        let sat = 0.3f32.mul_add(speed_norm, 0.7);
        let light = 0.3f32.mul_add(speed_norm, 0.35);

        let color = C::from_hsla(hue, sat, light, base_alpha * 0.7);

        // Draw the trail as a polyline
        canvas = canvas
            .polyline(trail.clone())
            .stroke(color, preset.trail_width * 0.5f32.mul_add(life_frac, 0.5))
            .done();

        // Bright head dot
        if let Some(&(hx, hy)) = trail.last() {
            let head_color = C::from_hsla(hue, sat, light + 0.15, base_alpha * 0.9);
            canvas = canvas
                .circle(hx, hy, preset.trail_width * 0.8)
                .fill(head_color)
                .done();
        }
    }

    canvas
}
