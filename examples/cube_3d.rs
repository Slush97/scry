//! 3D wireframe cube — rendered entirely from scratch.
//!
//! No math libraries, no GPU, no 3D engine. Just sin, cos, and lines.
//!
//! Run with: `cargo run --example cube_3d`

use std::io::stdout;

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

// ───────────────────────────────────────────────────────────────────
// 3D math from scratch — that's it, this is the entire "engine"
// ───────────────────────────────────────────────────────────────────

/// A point in 3D space.
#[derive(Clone, Copy)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

/// Rotate a point around the Y axis.
fn rotate_y(p: Vec3, angle: f32) -> Vec3 {
    let (s, c) = angle.sin_cos();
    Vec3 {
        x: p.x.mul_add(c, p.z * s),
        y: p.y,
        z: (-p.x).mul_add(s, p.z * c),
    }
}

/// Rotate a point around the X axis.
fn rotate_x(p: Vec3, angle: f32) -> Vec3 {
    let (s, c) = angle.sin_cos();
    Vec3 {
        x: p.x,
        y: p.y.mul_add(c, -(p.z * s)),
        z: p.y.mul_add(s, p.z * c),
    }
}

/// Perspective projection: 3D → 2D screen coordinates.
fn project(p: Vec3, cx: f32, cy: f32, fov: f32) -> (f32, f32) {
    let z = p.z + 4.0; // push the cube away from the camera
    let scale = fov / z;
    (p.x.mul_add(scale, cx), p.y.mul_add(scale, cy))
}

// ───────────────────────────────────────────────────────────────────
// Cube definition — 8 vertices, 12 edges
// ───────────────────────────────────────────────────────────────────

/// Unit cube vertices centered at origin.
const VERTICES: [Vec3; 8] = [
    Vec3 {
        x: -1.0,
        y: -1.0,
        z: -1.0,
    },
    Vec3 {
        x: 1.0,
        y: -1.0,
        z: -1.0,
    },
    Vec3 {
        x: 1.0,
        y: 1.0,
        z: -1.0,
    },
    Vec3 {
        x: -1.0,
        y: 1.0,
        z: -1.0,
    },
    Vec3 {
        x: -1.0,
        y: -1.0,
        z: 1.0,
    },
    Vec3 {
        x: 1.0,
        y: -1.0,
        z: 1.0,
    },
    Vec3 {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    },
    Vec3 {
        x: -1.0,
        y: 1.0,
        z: 1.0,
    },
];

/// The 12 edges of a cube, as pairs of vertex indices.
const EDGES: [(usize, usize); 12] = [
    // Front face
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    // Back face
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    // Connecting edges
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

// ───────────────────────────────────────────────────────────────────
// Main loop
// ───────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let picker = Picker::detect();
    let backend: Box<dyn transport::ProtocolBackend> = match picker.protocol() {
        ProtocolKind::Kitty => Box::new(transport::kitty::KittyBackend::new(picker.font_size())),
        _ => Box::new(transport::halfblock::HalfblockBackend::new()),
    };
    let mut state = PixelCanvasState::new(backend, picker.font_size());

    let mut angle_y: f32 = 0.0;
    let mut angle_x: f32 = 0.3; // slight tilt so we see the top

    loop {
        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(3)])
                .split(frame.area());

            let area = chunks[0];
            let canvas = build_cube_scene(area, &state, angle_y, angle_x);

            frame.render_stateful_widget(
                PixelCanvasWidget::new(canvas).z_index(-1),
                area,
                &mut state,
            );

            let status = Paragraph::new(" ← → rotate Y  |  ↑ ↓ rotate X  |  q quit")
                .block(Block::default().borders(Borders::TOP));
            frame.render_widget(status, chunks[1]);
        })?;
        state.flush()?;

        // Animate: auto-rotate slowly
        angle_y += 0.02;

        if event::poll(std::time::Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Left => angle_y -= 0.15,
                        KeyCode::Right => angle_y += 0.15,
                        KeyCode::Up => angle_x -= 0.15,
                        KeyCode::Down => angle_x += 0.15,
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
// Scene builder
// ───────────────────────────────────────────────────────────────────

#[allow(clippy::cast_precision_loss)]
fn build_cube_scene(
    area: Rect,
    state: &PixelCanvasState,
    angle_y: f32,
    angle_x: f32,
) -> PixelCanvas {
    let font = state.font_size();
    let w = u32::from(area.width) * u32::from(font.width);
    let h = u32::from(area.height) * u32::from(font.height);

    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let fov = h as f32 * 0.8; // field of view scales with window

    // 1. Transform all vertices
    let projected: Vec<(f32, f32)> = VERTICES
        .iter()
        .map(|&v| {
            let rotated = rotate_x(rotate_y(v, angle_y), angle_x);
            project(rotated, cx, cy, fov)
        })
        .collect();

    // 2. Build the scene
    let mut canvas = PixelCanvas::new(w, h).background(PxColor::from_rgba8(10, 10, 20, 255));

    // Edge colors: front=cyan, back=dim blue, connecting=purple
    let front_color = PxColor::from_rgba8(0, 220, 240, 255);
    let back_color = PxColor::from_rgba8(60, 80, 160, 200);
    let connect_color = PxColor::from_rgba8(180, 100, 255, 220);

    for (i, &(a, b)) in EDGES.iter().enumerate() {
        let (x1, y1) = projected[a];
        let (x2, y2) = projected[b];

        let color = if i < 4 {
            front_color
        } else if i < 8 {
            back_color
        } else {
            connect_color
        };

        canvas = canvas.line(x1, y1, x2, y2).color(color).width(2.5).done();
    }

    // Draw vertices as small circles
    for &(x, y) in &projected {
        canvas = canvas.circle(x, y, 4.0).fill(PxColor::WHITE).done();
    }

    canvas
}
