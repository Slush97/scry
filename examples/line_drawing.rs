//! SVG line drawing animation with performance diagnostics.
//!
//! Demonstrates the `SvgLineDrawing` API with organic pen effects and
//! real-time frame-timing breakdown.
//!
//! Controls:
//!   r  — restart animation
//!   m  — toggle sequential / simultaneous mode
//!   p  — toggle pen pressure
//!   d  — toggle pen-tip dot
//!   t  — toggle trailing ghost
//!   e  — cycle easing (none → `EaseInOutCubic` → `EaseOutQuart`)
//!   q  — quit
//!
//! Status bar shows: draw (command build µs), rast (rasterize+transmit µs),
//! total (frame µs), and FPS.

#![allow(clippy::cast_precision_loss)]

use std::collections::VecDeque;
use std::io::stdout;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color as TuiColor, Style};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Terminal;

use ratatui_pixelcanvas::prelude::*;
use ratatui_pixelcanvas::transport;

// ───────────────────────────────────────────────────────────────────
// Showcase SVG
//
// This SVG exercises every path primitive that the line-drawing
// engine handles: <rect> with rx/ry (→ cubic arcs), <circle> &
// <ellipse> (→ 4 cubic arcs each), <polygon>/<polyline> (→ lineTo),
// <line> (→ single lineTo), <path> with M/C/Q/L/A commands, and
// nested transforms.  The varied stroke widths and colors ensure
// pen-pressure and trail effects are clearly visible.
// ───────────────────────────────────────────────────────────────────

const SVG_CONTENT: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500">
  <!-- 1. Rounded frame (rect with rx/ry → cubic arcs at corners) -->
  <rect x="10" y="10" width="480" height="480" rx="40" ry="40"
        fill="none" stroke="#4ECDC4" stroke-width="3"/>

  <!-- 2. Five-pointed star (polygon → lineTo × 10 + close) -->
  <polygon points="250,45 280,170 410,170 305,240 335,365 250,290 165,365 195,240 90,170 220,170"
           fill="none" stroke="#FF6B6B" stroke-width="3.5"/>

  <!-- 3. Concentric circles (each → 4 cubic arcs) -->
  <circle cx="250" cy="250" r="185" fill="none" stroke="#45B7D1" stroke-width="2"/>
  <circle cx="250" cy="250" r="140" fill="none" stroke="#96CEB4" stroke-width="2"/>
  <circle cx="250" cy="250" r="95"  fill="none" stroke="#9B59B6" stroke-width="2"/>

  <!-- 4. Spiral (cubic Béziers → curved segments) -->
  <path d="M 250,250
           C 250,190 320,190 320,250
           C 320,330 170,330 170,250
           C 170,150 350,150 350,250
           C 350,360 130,360 130,250
           C 130,120 380,120 380,250"
        fill="none" stroke="#E056A0" stroke-width="3"/>

  <!-- 5. Quadratic Bézier wave (Q commands) -->
  <path d="M 40,400 Q 120,340 200,400 Q 280,460 360,400 Q 440,340 460,400"
        fill="none" stroke="#F7DC6F" stroke-width="2.5"/>

  <!-- 6. Diagonal cross (line → single lineTo each) -->
  <line x1="60"  y1="60"  x2="440" y2="440" stroke="#E74C3C" stroke-width="1.5"/>
  <line x1="440" y1="60"  x2="60"  y2="440" stroke="#E74C3C" stroke-width="1.5"/>

  <!-- 7. Triangle (polygon → 3 lineTo + close) -->
  <polygon points="250,70 420,410 80,410"
           fill="none" stroke="#3498DB" stroke-width="2"/>

  <!-- 8. Small decorative circles (tests many short paths) -->
  <circle cx="100" cy="100" r="15" fill="none" stroke="#FF6B6B" stroke-width="2"/>
  <circle cx="400" cy="100" r="15" fill="none" stroke="#FF6B6B" stroke-width="2"/>
  <circle cx="100" cy="400" r="15" fill="none" stroke="#45B7D1" stroke-width="2"/>
  <circle cx="400" cy="400" r="15" fill="none" stroke="#45B7D1" stroke-width="2"/>
  <circle cx="250" cy="250" r="10" fill="none" stroke="#E056A0" stroke-width="3"/>

  <!-- 9. Sinusoidal arc path (A commands → arc-to) -->
  <path d="M 50,250 A 100,50 0 0 1 150,250 A 100,50 0 0 0 250,250
           A 100,50 0 0 1 350,250 A 100,50 0 0 0 450,250"
        fill="none" stroke="#2ECC71" stroke-width="2"/>

  <!-- 10. Ellipse (different rx/ry → tests elliptical arc decomposition) -->
  <ellipse cx="250" cy="250" rx="220" ry="100"
           fill="none" stroke="#F39C12" stroke-width="1.5" stroke-dasharray="none"/>
</svg>"##;

// ───────────────────────────────────────────────────────────────────
// Frame-timing ring buffer (60 frames ≈ 1s of history)
// ───────────────────────────────────────────────────────────────────

struct FrameTimings {
    draw_us: VecDeque<u64>,
    render_us: VecDeque<u64>,
    total_us: VecDeque<u64>,
    cap: usize,
}

impl FrameTimings {
    fn new(cap: usize) -> Self {
        Self {
            draw_us: VecDeque::with_capacity(cap),
            render_us: VecDeque::with_capacity(cap),
            total_us: VecDeque::with_capacity(cap),
            cap,
        }
    }

    fn push(&mut self, draw: u64, render: u64, total: u64) {
        if self.draw_us.len() >= self.cap {
            self.draw_us.pop_front();
            self.render_us.pop_front();
            self.total_us.pop_front();
        }
        self.draw_us.push_back(draw);
        self.render_us.push_back(render);
        self.total_us.push_back(total);
    }

    fn avg(ring: &VecDeque<u64>) -> f64 {
        if ring.is_empty() {
            return 0.0;
        }
        ring.iter().sum::<u64>() as f64 / ring.len() as f64
    }

    fn summary(&self) -> String {
        let draw = Self::avg(&self.draw_us);
        let rast = Self::avg(&self.render_us);
        let total = Self::avg(&self.total_us);
        let fps = if total > 0.0 {
            1_000_000.0 / total
        } else {
            0.0
        };
        format!("draw:{draw:.0}µs rast:{rast:.0}µs total:{total:.0}µs fps:{fps:.0}")
    }
}

// ───────────────────────────────────────────────────────────────────
// Main
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

    // ── Parse SVG ONCE — the path geometry is immutable across frames ──
    let base_drawing = SvgLineDrawing::from_str(SVG_CONTENT)?;

    // Feature toggles (organic features ON by default)
    let mut mode = DrawMode::Sequential;
    let mut use_pressure = true;
    let mut use_tip = true;
    let mut use_trail = true;
    let mut easing_idx: u8 = 1; // 0=none, 1=EaseInOutCubic, 2=EaseOutQuart

    let anim_duration = Duration::from_millis(6000);
    let mut anim_start = Instant::now();
    let mut timings = FrameTimings::new(60);

    loop {
        let frame_start = Instant::now();

        let elapsed = anim_start.elapsed();
        let raw_t = (elapsed.as_secs_f32() / anim_duration.as_secs_f32()).min(1.0);
        let t = Easing::EaseOutQuad.ease(raw_t);

        let term_size = terminal.size()?;
        let term_rect = Rect::new(0, 0, term_size.width, term_size.height);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(2)])
            .split(term_rect);
        let canvas_area = chunks[0];
        let status_area = chunks[1];

        let font = state.font_size();
        let w = u32::from(canvas_area.width) * u32::from(font.width);
        let h = u32::from(canvas_area.height) * u32::from(font.height);

        if w == 0 || h == 0 {
            std::thread::sleep(Duration::from_millis(16));
            continue;
        }

        // ── BUILD DRAW COMMANDS ──────────────────────────────────────
        // Time complexity:  O(S) where S = segment count
        //   - clone: S × Arc::clone (atomic inc, no geometry copy)
        //   - easing eval: S × 1 Easing::ease call (O(1) math)
        //   - DashPattern: S × pair()/quad() (2-4 element Vec, tiny alloc)
        //   - point_at_length: 1 call for the active segment (O(V) walk)
        //   - pen_tip circle: 1 circle command
        let draw_start = Instant::now();

        // Clone base_drawing: copies Arc pointers (not path geometry)
        // then configure features per this frame's toggle state.
        let mut drawing = base_drawing.clone().mode(mode);

        match easing_idx {
            1 => drawing = drawing.easing(Easing::EaseInOutCubic),
            2 => drawing = drawing.easing(Easing::EaseOutQuart),
            _ => {}
        }
        if use_pressure {
            drawing = drawing.pen_pressure(PenPressure::default());
        }
        if use_tip {
            drawing = drawing.pen_tip(PenTip::default());
        }
        if use_trail {
            drawing = drawing.trail(Trail::default());
        }

        // Fit the SVG viewBox into the canvas with 10% padding
        let svg_size = 500.0_f32;
        let scale = (w as f32 / svg_size).min(h as f32 / svg_size) * 0.90;
        let offset_x = svg_size.mul_add(-scale, w as f32) / 2.0;
        let offset_y = svg_size.mul_add(-scale, h as f32) / 2.0;

        let transform = ratatui_pixelcanvas::scene::style::Transform {
            sx: scale,
            kx: 0.0,
            ky: 0.0,
            sy: scale,
            tx: offset_x,
            ty: offset_y,
        };

        let mut canvas = PixelCanvas::new(w, h).background(Color::from_rgba8(10, 10, 18, 255));

        let drawing_ref = &drawing;
        canvas = canvas
            .group(transform)
            .canvas(move |mut inner| {
                drawing_ref.draw_into(&mut inner, t);
                inner
            })
            .done();

        // Slim progress bar
        let bar_y = h as f32 - 4.0;
        let bar_w = w as f32 * 0.5;
        let bar_x = (w as f32 - bar_w) / 2.0;
        if raw_t > 0.001 {
            canvas = canvas
                .rect(bar_x, bar_y, bar_w * raw_t, 2.0)
                .fill(Color::from_rgba8(78, 205, 196, 180))
                .done();
        }

        let draw_elapsed = draw_start.elapsed();

        // ── RASTERIZE + TRANSMIT ─────────────────────────────────────
        // Uses `skip_cache()` to avoid the O(S×V) content_hash walk.
        // The rasterizer reuses the same pixmap (no per-frame allocation).
        let render_start = Instant::now();

        let mode_str = match mode {
            DrawMode::Sequential => "seq",
            DrawMode::Simultaneous => "sim",
        };
        let easing_str = match easing_idx {
            0 => "none",
            1 => "cubic",
            2 => "quart",
            _ => "?",
        };
        let segs = drawing.segment_count();
        let features = format!(
            "{}{}{}",
            if use_pressure { "P" } else { "-" },
            if use_tip { "T" } else { "-" },
            if use_trail { "G" } else { "-" },
        );
        let timing_str = timings.summary();

        terminal.draw(|frame| {
            frame.render_stateful_widget(
                // skip_cache: skip O(S×V) content_hash — always re-rasterize.
                PixelCanvasWidget::new(canvas).z_index(-1).skip_cache(),
                canvas_area,
                &mut state,
            );

            let status = format!(
                " ✏ {segs}p t={raw_t:.2} {mode_str} ease:{easing_str} [{features}] │ {timing_str} │ r m p d t e q"
            );
            let status_widget = Paragraph::new(status)
                .style(Style::default().fg(TuiColor::DarkGray))
                .block(Block::default().borders(Borders::TOP));
            frame.render_widget(status_widget, status_area);
        })?;
        state.flush()?;

        let render_elapsed = render_start.elapsed();

        #[allow(clippy::cast_possible_truncation)]
        let draw_us = draw_elapsed.as_micros() as u64;
        #[allow(clippy::cast_possible_truncation)]
        let render_us = render_elapsed.as_micros() as u64;
        #[allow(clippy::cast_possible_truncation)]
        let total_us = frame_start.elapsed().as_micros() as u64;

        timings.push(draw_us, render_us, total_us);

        // Auto-restart after completion + pause
        if elapsed > anim_duration + Duration::from_millis(2000) {
            anim_start = Instant::now();
        }

        if event::poll(Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Char('r') => anim_start = Instant::now(),
                        KeyCode::Char('m') => {
                            mode = match mode {
                                DrawMode::Sequential => DrawMode::Simultaneous,
                                DrawMode::Simultaneous => DrawMode::Sequential,
                            };
                            anim_start = Instant::now();
                        }
                        KeyCode::Char('p') => use_pressure = !use_pressure,
                        KeyCode::Char('d') => use_tip = !use_tip,
                        KeyCode::Char('t') => use_trail = !use_trail,
                        KeyCode::Char('e') => {
                            easing_idx = (easing_idx + 1) % 3;
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
