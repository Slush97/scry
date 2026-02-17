//! Visual showcase — renders **every** drawing primitive and style option.
//!
//! This serves as both a visual regression test and a feature catalogue.
//! The screen is divided into a 4-column × 4-row grid. Each cell demonstrates
//! one feature with a label.
//!
//! **Primitives covered:** Clear, Circle, Rectangle, Ellipse, Rotated Ellipse,
//! Line, Path (Bézier), Polyline, Polygon, Gradient (linear + radial), Group.
//!
//! **Styles covered:** fill, stroke, corner radius, rotation, `LineCap::Round`,
//! `LineJoin::Bevel`, `DashPattern`, `fill_radial_gradient`,
//! `anti_alias(false)`, and `Transform` (translate + scale).
//!
//! Run with: `cargo run --example showcase --features widget`

// Example code — suppress pedantic clippy lints that don't matter here
#![allow(
    clippy::suboptimal_flops,
    clippy::items_after_statements,
    clippy::cast_precision_loss,
    clippy::unreadable_literal,
    clippy::similar_names
)]

use std::io::stdout;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use scry_engine::prelude::{Picker, PixelCanvasState, PixelCanvasWidget, ProtocolKind};
use scry_engine::scene::style::{
    Color as C, DashPattern, GradientDef, GradientKind, GradientStop, LineCap, LineJoin, Point,
    Transform,
};
use scry_engine::scene::PixelCanvas;
use scry_engine::transport;

#[cfg(feature = "window")]
fn run_window() -> Result<(), Box<dyn std::error::Error>> {
    use scry_engine::rasterize::Rasterizer;
    use scry_engine::transport::window::{run_loop_continuous, LoopAction};
    use winit::keyboard::KeyCode as WKey;

    run_loop_continuous(
        960,
        640,
        "Visual Showcase",
        true,
        move |backend, keys, (w, h)| {
            for key in keys {
                if !key.pressed {
                    continue;
                }
                match key.code {
                    WKey::Escape | WKey::KeyQ => return LoopAction::Exit,
                    _ => {}
                }
            }

            let canvas = build_showcase(w, h);
            if let Ok(pixmap) = Rasterizer::rasterize(&canvas) {
                let _ = backend.blit(&pixmap);
            }
            LoopAction::Continue
        },
    )?;
    Ok(())
}

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

    loop {
        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(3)])
                .split(frame.area());

            let font = state.font_size();
            let w = u32::from(chunks[0].width) * u32::from(font.width);
            let h = u32::from(chunks[0].height) * u32::from(font.height);
            let canvas = build_showcase(w, h);
            frame.render_stateful_widget(
                PixelCanvasWidget::new(canvas).z_index(-1),
                chunks[0],
                &mut state,
            );

            let status = Paragraph::new(
                " Full showcase: all 9 DrawCommand variants + style options  |  'q' quit",
            )
            .block(Block::default().borders(Borders::TOP));
            frame.render_widget(status, chunks[1]);
        })?;
        state.flush()?;

        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press && key.code == KeyCode::Char('q') {
                    break;
                }
            }
        }
    }

    state.cleanup();
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: cell bounds in a 4×4 grid
// ─────────────────────────────────────────────────────────────────────────────
struct Cell {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    cx: f32,
    cy: f32,
}

fn cell(col: usize, row: usize, total_w: f32, total_h: f32) -> Cell {
    let w = total_w / 4.0;
    let h = total_h / 4.0;
    let x = col as f32 * w;
    let y = row as f32 * h;
    Cell {
        x,
        y,
        w,
        h,
        cx: x + w / 2.0,
        cy: y + h / 2.0,
    }
}

#[allow(clippy::too_many_lines)]
fn build_showcase(w: u32, h: u32) -> PixelCanvas {
    let wf = w as f32;
    let hf = h as f32;
    let pad = 8.0;

    let mut canvas = PixelCanvas::new(w, h).background(C::from_rgba8(18, 18, 28, 255));

    // ═══════════════════════════════════════════════════════════════
    // Row 0 — Basic shapes
    // ═══════════════════════════════════════════════════════════════

    // (0,0) Circle — solid fill + stroke
    {
        let c = cell(0, 0, wf, hf);
        let r = (c.w.min(c.h) * 0.35) - pad;
        canvas = canvas
            .circle(c.cx, c.cy, r)
            .fill(C::from_rgba8(100, 149, 237, 220))
            .stroke(C::from_rgba8(200, 220, 255, 255), 2.5)
            .done();
    }

    // (1,0) Rectangle — rounded corners
    {
        let c = cell(1, 0, wf, hf);
        canvas = canvas
            .rect(c.x + pad, c.y + pad, c.w - pad * 2.0, c.h - pad * 2.0)
            .fill(C::from_rgba8(60, 179, 113, 200))
            .corner_radius(12.0)
            .stroke(C::WHITE, 1.5)
            .done();
    }

    // (2,0) Ellipse — axis-aligned
    {
        let c = cell(2, 0, wf, hf);
        canvas = canvas
            .ellipse(c.cx, c.cy, c.w * 0.4 - pad, c.h * 0.25 - pad)
            .fill(C::from_rgba8(255, 165, 0, 200))
            .stroke(C::from_rgba8(255, 220, 150, 255), 2.0)
            .done();
    }

    // (3,0) Rotated ellipse
    {
        let c = cell(3, 0, wf, hf);
        canvas = canvas
            .ellipse(c.cx, c.cy, c.w * 0.38 - pad, c.h * 0.15 - pad)
            .rotation(std::f32::consts::FRAC_PI_4)
            .fill(C::from_rgba8(186, 85, 211, 200))
            .stroke(C::from_rgba8(230, 180, 255, 255), 2.0)
            .done();
    }

    // ═══════════════════════════════════════════════════════════════
    // Row 1 — Lines, Polyline, Polygon
    // ═══════════════════════════════════════════════════════════════

    // (0,1) Line — basic X-cross
    {
        let c = cell(0, 1, wf, hf);
        canvas = canvas
            .line(c.x + pad, c.y + pad, c.x + c.w - pad, c.y + c.h - pad)
            .stroke(C::from_rgba8(255, 215, 0, 255), 3.0)
            .done()
            .line(c.x + c.w - pad, c.y + pad, c.x + pad, c.y + c.h - pad)
            .stroke(C::from_rgba8(255, 100, 100, 255), 3.0)
            .done();
    }

    // (1,1) Dashed line with round cap
    {
        let c = cell(1, 1, wf, hf);
        canvas = canvas
            .line(c.x + pad, c.cy, c.x + c.w - pad, c.cy)
            .stroke(C::from_rgba8(0, 255, 200, 255), 4.0)
            .line_cap(LineCap::Round)
            .dash(DashPattern::new(vec![12.0, 6.0, 4.0, 6.0], 0.0))
            .done();
        // Second dashed line with square cap below
        canvas = canvas
            .line(
                c.x + pad,
                c.cy + pad * 2.0,
                c.x + c.w - pad,
                c.cy + pad * 2.0,
            )
            .stroke(C::from_rgba8(255, 150, 50, 255), 3.0)
            .line_cap(LineCap::Square)
            .dash(DashPattern::new(vec![8.0, 4.0], 0.0))
            .done();
    }

    // (2,1) Polyline — zigzag (open)
    {
        let c = cell(2, 1, wf, hf);
        let pts = vec![
            (c.x + pad, c.y + c.h - pad),
            (c.x + c.w * 0.25, c.y + pad),
            (c.x + c.w * 0.5, c.y + c.h - pad * 3.0),
            (c.x + c.w * 0.75, c.y + pad),
            (c.x + c.w - pad, c.y + c.h - pad),
        ];
        canvas = canvas
            .polyline(pts)
            .stroke(C::from_rgba8(0, 255, 255, 255), 2.5)
            .done();
    }

    // (3,1) Polygon — filled triangle with bevel join
    {
        let c = cell(3, 1, wf, hf);
        let pts = vec![
            (c.cx, c.y + pad),
            (c.x + pad, c.y + c.h - pad),
            (c.x + c.w - pad, c.y + c.h - pad),
        ];
        canvas = canvas
            .polygon(pts)
            .fill(C::from_rgba8(255, 99, 71, 200))
            .stroke(C::WHITE, 2.0)
            .line_join(LineJoin::Bevel)
            .done();
    }

    // ═══════════════════════════════════════════════════════════════
    // Row 2 — Path, Gradients, Gradient fill on shapes
    // ═══════════════════════════════════════════════════════════════

    // (0,2) Path — Bézier curve (heart shape)
    {
        let c = cell(0, 2, wf, hf);
        let s = (c.w.min(c.h) * 0.35).min(60.0);
        let cx = c.cx;
        let top = c.cy - s * 0.3;
        let bot = c.cy + s * 0.7;

        let mut pb = tiny_skia::PathBuilder::new();
        pb.move_to(cx, bot);
        pb.cubic_to(cx - s, top - s * 0.5, cx - s * 0.3, top - s, cx, top);
        pb.cubic_to(cx + s * 0.3, top - s, cx + s, top - s * 0.5, cx, bot);
        pb.close();

        if let Some(path) = pb.finish() {
            canvas = canvas
                .path(path)
                .fill(C::from_rgba8(220, 50, 80, 220))
                .stroke(C::from_rgba8(255, 180, 190, 255), 1.5)
                .done();
        }
    }

    // (1,2) Linear gradient rect
    {
        let c = cell(1, 2, wf, hf);
        canvas = canvas
            .gradient(c.x + pad, c.y + pad, c.w - pad * 2.0, c.h - pad * 2.0)
            .linear(
                Point::new(c.x + pad, c.y + pad),
                Point::new(c.x + c.w - pad, c.y + c.h - pad),
            )
            .stop(0.0, C::from_rgba8(255, 0, 128, 255))
            .stop(0.5, C::from_rgba8(128, 0, 255, 255))
            .stop(1.0, C::from_rgba8(0, 128, 255, 255))
            .done();
    }

    // (2,2) Radial gradient rect
    {
        let c = cell(2, 2, wf, hf);
        let r = (c.w.min(c.h) * 0.5) - pad;
        canvas = canvas
            .gradient(c.x + pad, c.y + pad, c.w - pad * 2.0, c.h - pad * 2.0)
            .radial(Point::new(c.cx, c.cy), r)
            .stop(0.0, C::from_rgba8(255, 255, 200, 255))
            .stop(0.5, C::from_rgba8(255, 140, 0, 255))
            .stop(1.0, C::from_rgba8(40, 0, 60, 255))
            .done();
    }

    // (3,2) Circle with radial gradient fill
    {
        let c = cell(3, 2, wf, hf);
        let r = (c.w.min(c.h) * 0.35) - pad;
        let grad = GradientDef {
            kind: GradientKind::Radial {
                center: Point::new(c.cx, c.cy),
                radius: r,
            },
            stops: vec![
                GradientStop {
                    position: 0.0,
                    color: C::from_rgba8(255, 255, 255, 255),
                },
                GradientStop {
                    position: 0.5,
                    color: C::from_rgba8(0, 180, 255, 255),
                },
                GradientStop {
                    position: 1.0,
                    color: C::from_rgba8(0, 30, 80, 255),
                },
            ],
        };
        canvas = canvas
            .circle(c.cx, c.cy, r)
            .fill_radial_gradient(grad)
            .stroke(C::from_rgba8(100, 200, 255, 200), 2.0)
            .done();
    }

    // ═══════════════════════════════════════════════════════════════
    // Row 3 — Group transform, anti_alias off, linear gradient fill,
    //         push_command mutable API
    // ═══════════════════════════════════════════════════════════════

    // (0,3) Group — translate + scale
    {
        let c = cell(0, 3, wf, hf);
        let transform = Transform {
            sx: 0.6,
            kx: 0.0,
            ky: 0.0,
            sy: 0.6,
            tx: c.x + c.w * 0.2,
            ty: c.y + c.h * 0.2,
        };
        canvas = canvas
            .group(transform)
            .canvas(|inner| {
                inner
                    .circle(c.w * 0.5, c.h * 0.5, c.w.min(c.h) * 0.3)
                    .fill(C::from_rgba8(255, 200, 0, 200))
                    .stroke(C::WHITE, 3.0)
                    .done()
                    .rect(c.w * 0.2, c.h * 0.2, c.w * 0.3, c.h * 0.3)
                    .fill(C::from_rgba8(100, 100, 255, 150))
                    .done()
            })
            .done();
    }

    // (1,3) anti_alias(false) — pixel art style
    {
        let c = cell(1, 3, wf, hf);
        let r = (c.w.min(c.h) * 0.3) - pad;
        canvas = canvas
            .circle(c.cx, c.cy, r)
            .fill(C::from_rgba8(255, 50, 150, 255))
            .stroke(C::WHITE, 2.0)
            .anti_alias(false)
            .done();
        // A small aliased rect
        canvas = canvas
            .rect(c.cx - r * 0.4, c.cy - r * 0.4, r * 0.8, r * 0.8)
            .fill(C::from_rgba8(50, 255, 150, 255))
            .anti_alias(false)
            .done();
    }

    // (2,3) Rectangle with linear gradient fill
    {
        let c = cell(2, 3, wf, hf);
        let inner_w = c.w - pad * 2.0;
        let inner_h = c.h - pad * 2.0;
        let grad = GradientDef {
            kind: GradientKind::Linear {
                start: Point::new(c.x + pad, c.y + pad),
                end: Point::new(c.x + pad + inner_w, c.y + pad + inner_h),
            },
            stops: vec![
                GradientStop {
                    position: 0.0,
                    color: C::from_rgba8(0, 200, 100, 255),
                },
                GradientStop {
                    position: 1.0,
                    color: C::from_rgba8(200, 0, 200, 255),
                },
            ],
        };
        canvas = canvas
            .rect(c.x + pad, c.y + pad, inner_w, inner_h)
            .fill_linear_gradient(grad)
            .corner_radius(8.0)
            .stroke(C::from_rgba8(220, 220, 220, 150), 1.5)
            .done();
    }

    // (3,3) push_command() mutable API — star polygon
    {
        let c = cell(3, 3, wf, hf);
        let r_outer = (c.w.min(c.h) * 0.38) - pad;
        let r_inner = r_outer * 0.4;
        let spikes = 5;
        let mut star_pts = Vec::with_capacity(spikes * 2);
        for i in 0..(spikes * 2) {
            let angle =
                -std::f32::consts::FRAC_PI_2 + std::f32::consts::PI * i as f32 / spikes as f32;
            let r = if i % 2 == 0 { r_outer } else { r_inner };
            star_pts.push((c.cx + r * angle.cos(), c.cy + r * angle.sin()));
        }
        // Use mutable push_command instead of fluent builder
        use scry_engine::scene::command::DrawCommand;
        use scry_engine::scene::style::{FillStyle, ShapeStyle};
        canvas.push_command(DrawCommand::Polyline {
            points: star_pts,
            closed: true,
            style: ShapeStyle {
                fill: Some(FillStyle::Solid(C::from_rgba8(255, 215, 0, 220))),
                stroke: Some(scry_engine::scene::style::StrokeStyle {
                    color: C::WHITE,
                    width: 2.0,
                    line_cap: LineCap::Butt,
                    line_join: LineJoin::Miter,
                    dash: None,
                }),
                anti_alias: true,
            },
        });
    }

    canvas
}
