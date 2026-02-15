//! Comprehensive feature showcase — demonstrates **every** library feature.
//!
//! This example covers features not shown in other demos:
//!
//! | Page | Features |
//! |------|----------|
//! | 1 — Advanced Shapes | Ellipse rotation, Polyline (star/spiral), custom Bézier paths, anti-alias toggle |
//! | 2 — Color & Style | `Color.with_lightness()`, `mix_rgb()`, gradient fills on shapes, `Transform.skew()` |
//! | 3 — Clip Paths | Path-based clipping (not just rect), nested clips |
//! | 4 — Image Blitting | Procedural `ImageData` generation + opacity layering |
//! | 5 — Transition Lifecycle | `reverse()`, `reset()`, `remaining()`, progress accessors, `is_active()/cancel_all()` |
//!
//! Navigate: ←→ or h/l to switch pages, q to quit.
//!
//! ```bash
//! cargo run --example feature_showcase
//! ```

use std::io::stdout;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use scry_engine::prelude::{
    AnimationState, Easing, Keyframe, Keyframes, Picker, PixelCanvasState, PixelCanvasWidget,
    ProfileHistory, ProfiledRasterizer, ProtocolKind,
};
use scry_engine::scene::animation::Transition;
use scry_engine::scene::command::ImageData;
use scry_engine::scene::style::{
    Color as C, GradientDef, GradientKind, GradientStop, Point, Rect as PxRect, Transform,
};
use scry_engine::scene::PixelCanvas;
use scry_engine::transport;

// ─────────────────────────────────────────────────────────────────────────────
// Pages
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Page {
    AdvancedShapes,
    ColorAndStyle,
    ClipPaths,
    ImageBlitting,
    TransitionLifecycle,
}

impl Page {
    const ALL: [Self; 5] = [
        Self::AdvancedShapes,
        Self::ColorAndStyle,
        Self::ClipPaths,
        Self::ImageBlitting,
        Self::TransitionLifecycle,
    ];

    const fn label(self) -> &'static str {
        match self {
            Self::AdvancedShapes => "1: Advanced Shapes",
            Self::ColorAndStyle => "2: Color & Style",
            Self::ClipPaths => "3: Clip Paths",
            Self::ImageBlitting => "4: Image Blitting",
            Self::TransitionLifecycle => "5: Transition Lifecycle",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn pixel_size(area: Rect, state: &PixelCanvasState) -> (u32, u32, f32, f32) {
    let font = state.font_size();
    let w = u32::from(area.width) * u32::from(font.width);
    let h = u32::from(area.height) * u32::from(font.height);
    (w, h, w as f32, h as f32)
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

#[allow(clippy::cast_precision_loss)]
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

    let mut page_idx: usize = 0;
    let start = Instant::now();
    let mut last_frame = start;

    // FPS & timing
    let mut frame_times: Vec<Duration> = Vec::with_capacity(60);
    let mut fps: f64 = 0.0;
    let mut draw_ms: f64 = 0.0;
    let mut flush_ms: f64 = 0.0;
    #[allow(unused_assignments)]
    let mut cmd_count: usize = 0;

    // Profiling (toggle with 'p')
    let mut profiling = false;
    let mut profile_history = ProfileHistory::default();
    let mut last_profile_str = String::new();

    // Animation state for Page 5 (Transition Lifecycle)
    let mut anim = AnimationState::new();
    let mut bounce_transition =
        Transition::new(0.0_f32, 1.0_f32, Duration::from_secs(2)).easing(Easing::EaseInOutCubic);
    let mut reverse_count: u32 = 0;

    loop {
        let now = Instant::now();
        let dt = now - last_frame;
        last_frame = now;
        let elapsed = now.duration_since(start);
        let page = Page::ALL[page_idx];

        // FPS tracking (rolling window of 60 frames)
        frame_times.push(dt);
        if frame_times.len() > 60 {
            frame_times.remove(0);
        }
        if !frame_times.is_empty() {
            let total: Duration = frame_times.iter().sum();
            fps = frame_times.len() as f64 / total.as_secs_f64();
        }

        // Advance the bounce transition
        if bounce_transition.advance(dt) {
            bounce_transition.reverse();
            reverse_count += 1;
        }

        // Tick orchestrator
        anim.tick(dt);

        // Start animations on page 5 if idle
        if page == Page::TransitionLifecycle && anim.is_idle() {
            anim.start(
                "pulse",
                0.3_f32,
                1.0_f32,
                Duration::from_secs(2),
                Easing::EaseInOutSine,
            );
            anim.start(
                "slide",
                0.0_f32,
                300.0_f32,
                Duration::from_secs(3),
                Easing::EaseOutExpo,
            );
            anim.start(
                "spin",
                0.0_f32,
                std::f32::consts::TAU,
                Duration::from_secs(4),
                Easing::Linear,
            );
        }

        let draw_start = Instant::now();

        // Build scene outside terminal.draw() so we can optionally profile it
        let term_size = terminal.size()?;
        let term_rect = Rect::new(0, 0, term_size.width, term_size.height);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(2)])
            .split(term_rect);
        let canvas_area = chunks[0];
        let status_area = chunks[1];

        let scene_start = Instant::now();
        let canvas = match page {
            Page::AdvancedShapes => build_advanced_shapes(canvas_area, &state, elapsed),
            Page::ColorAndStyle => build_color_style(canvas_area, &state, elapsed),
            Page::ClipPaths => build_clip_paths(canvas_area, &state, elapsed),
            Page::ImageBlitting => build_image_blitting(canvas_area, &state),
            Page::TransitionLifecycle => build_transition_lifecycle(
                canvas_area,
                &state,
                &bounce_transition,
                reverse_count,
                &anim,
            ),
        };
        let scene_build_us = scene_start.elapsed().as_micros() as u64;
        cmd_count = canvas.command_count();

        // When profiling, rasterize manually with per-command timing
        if profiling {
            let (pixmap_entry, gc) = state
                .cache_mut()
                .get_or_insert_with_grad_cache(canvas.width(), canvas.height());
            if let Some(pixmap) = pixmap_entry {
                let rp = ProfiledRasterizer::rasterize_into_profiled_cached(&canvas, pixmap, gc);
                profile_history.push(rp);
                let smoothed = profile_history.summary();
                last_profile_str = format!(
                    " \u{1F50D} {fps:.0}fps \u{2502} bld={:.1}m rast={smoothed} flsh={flush_ms:.1}m \u{2502} {cmd_count}cmd f{} \u{2502} p:off q:quit",
                    scene_build_us as f64 / 1000.0,
                    smoothed.frame_count,
                );
            }
            use std::sync::atomic::{AtomicU64, Ordering};
            static FRAME_SEQ: AtomicU64 = AtomicU64::new(1);
            state
                .cache_mut()
                .mark_valid(FRAME_SEQ.fetch_add(1, Ordering::Relaxed));
        }

        let profile_line = last_profile_str.clone();
        let is_profiling = profiling;

        terminal.draw(|frame| {
            frame.render_stateful_widget(
                PixelCanvasWidget::new(canvas).z_index(-1),
                canvas_area,
                &mut state,
            );

            let status_text = if is_profiling {
                profile_line.clone()
            } else {
                let tab_str: String = Page::ALL
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        if i == page_idx {
                            format!("[{}]", p.label())
                        } else {
                            p.label().to_string()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                format!(
                    " {tab_str} \u{2502} {fps:.0}fps \u{2502} draw {draw_ms:.1}ms \u{2502} flush {flush_ms:.1}ms \u{2502} {cmd_count}cmd \u{2502} ←→:page p:profile q:quit",
                )
            };
            frame.render_widget(
                Paragraph::new(status_text)
                    .block(Block::default().borders(Borders::TOP)),
                status_area,
            );
        })?;

        let draw_elapsed = draw_start.elapsed();

        let flush_start = Instant::now();
        state.flush()?;
        let flush_elapsed = flush_start.elapsed();

        draw_ms = draw_elapsed.as_secs_f64() * 1000.0;
        flush_ms = flush_elapsed.as_secs_f64() * 1000.0;

        // Input
        if event::poll(Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Char('p') => profiling = !profiling,
                        KeyCode::Right | KeyCode::Char('l') => {
                            page_idx = (page_idx + 1) % Page::ALL.len();
                        }
                        KeyCode::Left | KeyCode::Char('h') => {
                            page_idx = page_idx.checked_sub(1).unwrap_or(Page::ALL.len() - 1);
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

// ─────────────────────────────────────────────────────────────────────────────
// Page 1: Advanced Shapes
// ─────────────────────────────────────────────────────────────────────────────

/// Demonstrates: Ellipse with rotation, Polyline (star + spiral),
/// custom Bézier Path, `anti_alias(false)` comparison.
#[allow(clippy::cast_precision_loss)]
fn build_advanced_shapes(area: Rect, pxstate: &PixelCanvasState, elapsed: Duration) -> PixelCanvas {
    let (w, h, wf, hf) = pixel_size(area, pxstate);
    let t = elapsed.as_secs_f32();

    let mut canvas = PixelCanvas::new(w, h).background(C::from_rgba8(15, 15, 25, 255));

    // --- Section 1: Rotating ellipses (top-left quadrant) ---
    let cx = wf * 0.2;
    let cy = hf * 0.3;
    for i in 0..6 {
        let angle = t.mul_add(0.5, (i as f32) * std::f32::consts::FRAC_PI_6);
        let hue = (i as f32) / 6.0 * 360.0;
        let color = C::from_hsla(hue, 0.8, 0.6, 0.7);
        canvas = canvas
            .ellipse(cx, cy, 60.0, 25.0)
            .rotation(angle)
            .fill(color)
            .stroke(C::WHITE.with_alpha(0.3), 1.0)
            .done();
    }

    // --- Section 2: Polyline star (top-right quadrant) ---
    let star_cx = wf * 0.7;
    let star_cy = hf * 0.25;
    let star_r_outer = 70.0;
    let star_r_inner = 30.0;
    let pi_5 = std::f32::consts::PI / 5.0;
    let star_points: Vec<(f32, f32)> = (0..10)
        .map(|i| {
            let angle = t.mul_add(0.3, (i as f32).mul_add(pi_5, -std::f32::consts::FRAC_PI_2));
            let r = if i % 2 == 0 {
                star_r_outer
            } else {
                star_r_inner
            };
            (
                angle.cos().mul_add(r, star_cx),
                angle.sin().mul_add(r, star_cy),
            )
        })
        .collect();
    canvas = canvas
        .polygon(star_points)
        .fill(C::from_hsla(306.0, 0.9, 0.5, 0.9))
        .stroke(C::from_rgba8(255, 200, 100, 255), 2.0)
        .done();

    // --- Section 3: Polyline spiral (bottom-left) ---
    let spiral_cx = wf * 0.25;
    let spiral_cy = hf * 0.75;
    let spiral_points: Vec<(f32, f32)> = (0..120)
        .map(|i| {
            let angle = (i as f32).mul_add(0.15, t);
            let r = (i as f32).mul_add(0.7, 3.0);
            (
                angle.cos().mul_add(r, spiral_cx),
                angle.sin().mul_add(r, spiral_cy),
            )
        })
        .collect();
    canvas = canvas
        .polyline(spiral_points)
        .stroke(C::from_hsla(180.0, 0.8, 0.7, 1.0), 2.0)
        .done();

    // --- Section 4: Custom Bézier path — heart shape (bottom-center) ---
    let path_x = wf * 0.5;
    let path_y = hf * 0.65;
    let mut pb = tiny_skia::PathBuilder::new();
    pb.move_to(path_x, path_y + 20.0);
    pb.cubic_to(
        path_x - 50.0,
        path_y - 30.0,
        path_x - 90.0,
        path_y + 30.0,
        path_x,
        path_y + 80.0,
    );
    pb.move_to(path_x, path_y + 20.0);
    pb.cubic_to(
        path_x + 50.0,
        path_y - 30.0,
        path_x + 90.0,
        path_y + 30.0,
        path_x,
        path_y + 80.0,
    );
    if let Some(path) = pb.finish() {
        canvas = canvas
            .path(path)
            .fill(C::from_rgba8(220, 40, 60, 255))
            .stroke(C::from_rgba8(255, 100, 120, 255), 2.0)
            .done();
    }

    // --- Section 5: Anti-alias comparison (bottom-right) ---
    let aa_x = wf * 0.75;
    let aa_y = hf * 0.72;
    // With anti-aliasing
    canvas = canvas
        .circle(aa_x - 35.0, aa_y, 25.0)
        .fill(C::from_rgba8(100, 200, 255, 255))
        .anti_alias(true)
        .done();
    // Without anti-aliasing — visible jagged edges
    canvas = canvas
        .circle(aa_x + 35.0, aa_y, 25.0)
        .fill(C::from_rgba8(100, 200, 255, 255))
        .anti_alias(false)
        .done();

    canvas
}

// ─────────────────────────────────────────────────────────────────────────────
// Page 2: Color & Style
// ─────────────────────────────────────────────────────────────────────────────

/// Demonstrates: `Color.with_lightness()`, `Color.mix_rgb()` vs `mix()`,
/// gradient fills on shapes (`fill_linear_gradient`, `fill_radial_gradient`),
/// `Transform.skew()`, `Rect.intersects()`, `Rect.union()`.
#[allow(clippy::cast_precision_loss)]
fn build_color_style(area: Rect, pxstate: &PixelCanvasState, elapsed: Duration) -> PixelCanvas {
    let (w, h, wf, hf) = pixel_size(area, pxstate);
    let t = elapsed.as_secs_f32();

    let mut canvas = PixelCanvas::new(w, h).background(C::from_rgba8(10, 10, 20, 255));

    // --- Row 1: Color.with_lightness() gradient ---
    let base = C::from_rgba8(50, 120, 220, 255);
    for i in 0..12 {
        let factor = (i as f32).mul_add(0.15, 0.2);
        let color = base.with_lightness(factor);
        let x = wf.mul_add(0.05, (i as f32) * (wf * 0.075));
        canvas = canvas
            .rect(x, hf * 0.05, wf * 0.065, hf * 0.1)
            .fill(color)
            .corner_radius(4.0)
            .done();
    }

    // --- Row 2: mix_rgb() vs mix() (Oklab) comparison ---
    let c1 = C::from_rgba8(255, 0, 0, 255);
    let c2 = C::from_rgba8(0, 255, 0, 255);
    for i in 0..16 {
        let t_mix = (i as f32) / 15.0;
        let x = wf.mul_add(0.05, (i as f32) * (wf * 0.055));

        // Top row: mix_rgb (linear) — goes through muddy brown
        let rgb_blend = c1.mix_rgb(c2, t_mix);
        canvas = canvas
            .rect(x, hf * 0.22, wf * 0.045, hf * 0.08)
            .fill(rgb_blend)
            .done();

        // Bottom row: mix (Oklab) — perceptually smooth
        let oklab_blend = c1.mix(c2, t_mix);
        canvas = canvas
            .rect(x, hf * 0.32, wf * 0.045, hf * 0.08)
            .fill(oklab_blend)
            .done();
    }

    // --- Row 3: Gradient fills on shapes ---
    // Circle with linear gradient fill
    let grad_lin = GradientDef {
        kind: GradientKind::Linear {
            start: Point::new(0.0, 0.0),
            end: Point::new(100.0, 100.0),
        },
        stops: vec![
            GradientStop {
                position: 0.0,
                color: C::from_rgba8(255, 100, 200, 255),
            },
            GradientStop {
                position: 1.0,
                color: C::from_rgba8(100, 200, 255, 255),
            },
        ],
    };
    canvas = canvas
        .circle(wf * 0.2, hf * 0.58, 40.0)
        .fill_linear_gradient(grad_lin)
        .stroke(C::WHITE.with_alpha(0.5), 2.0)
        .done();

    // Rect with radial gradient fill
    let grad_rad = GradientDef {
        kind: GradientKind::Radial {
            center: Point::new(50.0, 40.0),
            radius: 60.0,
        },
        stops: vec![
            GradientStop {
                position: 0.0,
                color: C::from_rgba8(255, 255, 100, 255),
            },
            GradientStop {
                position: 0.5,
                color: C::from_rgba8(255, 100, 50, 255),
            },
            GradientStop {
                position: 1.0,
                color: C::from_rgba8(50, 0, 100, 255),
            },
        ],
    };
    canvas = canvas
        .rect(wf * 0.4, hf * 0.48, 100.0, 80.0)
        .fill_radial_gradient(grad_rad)
        .corner_radius(12.0)
        .done();

    // --- Row 3 right: Transform.skew() ---
    let skew_x = (t * 0.5).sin() * 0.3;
    let skew_transform = Transform::skew(skew_x, 0.0);
    canvas = canvas
        .group(Transform::translate(wf * 0.72, hf * 0.55).concat(skew_transform))
        .canvas(|c| {
            c.rect(-40.0, -30.0, 80.0, 60.0)
                .fill(C::from_hsla(54.0, 0.9, 0.5, 0.9))
                .stroke(C::WHITE, 2.0)
                .corner_radius(6.0)
                .done()
                .rect(-30.0, -20.0, 60.0, 40.0)
                .fill(C::from_hsla(216.0, 0.9, 0.5, 0.7))
                .corner_radius(4.0)
                .done()
        })
        .done();

    // --- Row 4: Rect.intersects() and Rect.union() visualization ---
    let r1 = PxRect::new(wf * 0.15, hf * 0.78, 120.0, 60.0);
    let r2_x = (t * 0.7).sin().mul_add(40.0, wf.mul_add(0.15, 60.0));
    let r2 = PxRect::new(r2_x, hf.mul_add(0.78, 20.0), 100.0, 50.0);
    let intersects = r1.intersects(&r2);
    let union_rect = r1.union(&r2);

    // Union bounding box (dim outline)
    canvas = canvas
        .rect(
            union_rect.x,
            union_rect.y,
            union_rect.width,
            union_rect.height,
        )
        .stroke(C::from_rgba8(80, 80, 80, 255), 1.0)
        .done();
    // Rect 1
    canvas = canvas
        .rect(r1.x, r1.y, r1.width, r1.height)
        .fill(C::from_rgba8(100, 150, 255, 128))
        .stroke(C::from_rgba8(100, 150, 255, 255), 2.0)
        .done();
    // Rect 2 — changes color when intersecting
    let r2_color = if intersects {
        C::from_rgba8(255, 100, 100, 128)
    } else {
        C::from_rgba8(100, 255, 100, 128)
    };
    let r2_stroke = if intersects {
        C::from_rgba8(255, 100, 100, 255)
    } else {
        C::from_rgba8(100, 255, 100, 255)
    };
    canvas = canvas
        .rect(r2.x, r2.y, r2.width, r2.height)
        .fill(r2_color)
        .stroke(r2_stroke, 2.0)
        .done();

    canvas
}

// ─────────────────────────────────────────────────────────────────────────────
// Page 3: Clip Paths
// ─────────────────────────────────────────────────────────────────────────────

/// Demonstrates: `ClipRegion::Path` (path-based clipping), nested groups with
/// different clip regions.
#[allow(clippy::cast_precision_loss)]
fn build_clip_paths(area: Rect, pxstate: &PixelCanvasState, elapsed: Duration) -> PixelCanvas {
    let (w, h, wf, hf) = pixel_size(area, pxstate);
    let t = elapsed.as_secs_f32();

    let mut canvas = PixelCanvas::new(w, h).background(C::from_rgba8(20, 15, 30, 255));

    // --- Left: Clip to circle path ---
    let clip_cx = wf * 0.25;
    let clip_cy = hf * 0.4;
    let clip_r = 80.0;
    if let Some(clip_path) = tiny_skia::PathBuilder::from_circle(clip_cx, clip_cy, clip_r) {
        canvas = canvas
            .group(Transform::identity())
            .clip_path(clip_path)
            .canvas(|c| {
                // Draw a grid of rainbow bars that gets clipped to circle
                let mut inner = c;
                let step = 15.0;
                let mut y = clip_cy - clip_r;
                while y < clip_cy + clip_r {
                    let hue = ((y - clip_cy + clip_r) / (2.0 * clip_r)) * 120.0;
                    inner = inner
                        .rect(clip_cx - clip_r, y, clip_r * 2.0, step * 0.8)
                        .fill(C::from_hsla(t.mul_add(30.0, hue), 0.8, 0.5, 1.0))
                        .done();
                    y += step;
                }
                // Animated circle bouncing inside the clip
                let ball_x = (t * 2.0).sin().mul_add(50.0, clip_cx);
                let ball_y = (t * 1.5).cos().mul_add(50.0, clip_cy);
                inner.circle(ball_x, ball_y, 20.0).fill(C::WHITE).done()
            })
            .done();

        // Clip border for visibility
        canvas = canvas
            .circle(clip_cx, clip_cy, clip_r)
            .stroke(C::from_rgba8(200, 200, 200, 255), 2.0)
            .done();
    }

    // --- Right: Clip to star path ---
    let star_cx = wf * 0.65;
    let star_cy = hf * 0.4;
    let pi_5 = std::f32::consts::PI / 5.0;
    let mut pb = tiny_skia::PathBuilder::new();
    for i in 0..10 {
        let angle = (i as f32).mul_add(pi_5, -std::f32::consts::FRAC_PI_2);
        let r = if i % 2 == 0 { 90.0 } else { 40.0 };
        let px = angle.cos().mul_add(r, star_cx);
        let py = angle.sin().mul_add(r, star_cy);
        if i == 0 {
            pb.move_to(px, py);
        } else {
            pb.line_to(px, py);
        }
    }
    pb.close();
    if let Some(star_path) = pb.finish() {
        canvas = canvas
            .group(Transform::identity())
            .clip_path(star_path)
            .canvas(|c| {
                // Radial gradient background clipped to star
                c.gradient(star_cx - 90.0, star_cy - 90.0, 180.0, 180.0)
                    .stop(0.0, C::from_rgba8(255, 200, 50, 255))
                    .stop(0.5, C::from_rgba8(255, 50, 100, 255))
                    .stop(1.0, C::from_rgba8(50, 0, 150, 255))
                    .radial(Point::new(90.0, 90.0), 120.0)
                    .done()
                    // Animated lines
                    .line(
                        star_cx - 80.0,
                        (t * 3.0).sin().mul_add(60.0, star_cy),
                        star_cx + 80.0,
                        (t * 2.5).cos().mul_add(60.0, star_cy),
                    )
                    .stroke(C::WHITE, 3.0)
                    .done()
            })
            .done();
    }

    // --- Bottom: Nested clip (circle with rotating content) ---
    let nest_x = wf * 0.5;
    let nest_y = hf * 0.8;
    if let Some(outer_clip) = tiny_skia::PathBuilder::from_circle(nest_x, nest_y, 50.0) {
        canvas = canvas
            .group(Transform::identity())
            .clip_path(outer_clip)
            .opacity(0.9)
            .canvas(|c| {
                let mut inner = c
                    .rect(nest_x - 60.0, nest_y - 60.0, 120.0, 120.0)
                    .fill(C::from_rgba8(60, 30, 80, 255))
                    .done();
                // Rotating rectangles inside
                for i in 0..6 {
                    let angle = (i as f32).mul_add(std::f32::consts::FRAC_PI_3, t);
                    let dx = angle.cos() * 30.0;
                    let dy = angle.sin() * 30.0;
                    inner = inner
                        .rect(nest_x + dx - 10.0, nest_y + dy - 10.0, 20.0, 20.0)
                        .fill(C::from_hsla((i as f32) / 6.0 * 360.0, 0.9, 0.6, 0.8))
                        .corner_radius(3.0)
                        .done();
                }
                inner
            })
            .done();
    }

    canvas
}

// ─────────────────────────────────────────────────────────────────────────────
// Page 4: Image Blitting
// ─────────────────────────────────────────────────────────────────────────────

/// Demonstrates: `ImageData` procedural generation, .`image()` builder,
/// .`opacity()` on images, multiple image compositing.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn build_image_blitting(area: Rect, pxstate: &PixelCanvasState) -> PixelCanvas {
    let (w, h, wf, hf) = pixel_size(area, pxstate);

    let mut canvas = PixelCanvas::new(w, h).background(C::from_rgba8(15, 20, 30, 255));

    // --- Procedural checkerboard image ---
    let img_w: u32 = 64;
    let img_h: u32 = 64;
    let mut checker_data = vec![0u8; (img_w * img_h * 4) as usize];
    for y in 0..img_h {
        for x in 0..img_w {
            let idx = ((y * img_w + x) * 4) as usize;
            let is_dark = ((x / 8) + (y / 8)) % 2 == 0;
            let (r, g, b) = if is_dark {
                (40, 40, 60)
            } else {
                (200, 180, 220)
            };
            checker_data[idx] = r;
            checker_data[idx + 1] = g;
            checker_data[idx + 2] = b;
            checker_data[idx + 3] = 255;
        }
    }
    let checker = ImageData::new(img_w, img_h, checker_data);

    // Draw checkerboard at 3 positions with different opacities
    canvas = canvas
        .image(checker.clone(), wf * 0.1, hf * 0.1)
        .done()
        .image(checker.clone(), wf * 0.3, hf * 0.1)
        .opacity(0.7)
        .done()
        .image(checker.clone(), wf * 0.5, hf * 0.1)
        .opacity(0.4)
        .done();

    // --- Procedural gradient image ---
    let grad_w: u32 = 128;
    let grad_h: u32 = 64;
    let mut grad_data = vec![0u8; (grad_w * grad_h * 4) as usize];
    for y in 0..grad_h {
        for x in 0..grad_w {
            let idx = ((y * grad_w + x) * 4) as usize;
            let tx = x as f32 / grad_w as f32;
            let ty = y as f32 / grad_h as f32;
            grad_data[idx] = (tx * 255.0) as u8;
            grad_data[idx + 1] = ((1.0 - ty) * 200.0) as u8;
            grad_data[idx + 2] = (ty * 255.0) as u8;
            grad_data[idx + 3] = 255;
        }
    }
    let grad_img = ImageData::new(grad_w, grad_h, grad_data);
    canvas = canvas.image(grad_img, wf * 0.1, hf * 0.45).done();

    // --- Procedural plasma pattern ---
    let plasma_w: u32 = 96;
    let plasma_h: u32 = 96;
    let mut plasma_data = vec![0u8; (plasma_w * plasma_h * 4) as usize];
    for y in 0..plasma_h {
        for x in 0..plasma_w {
            let idx = ((y * plasma_w + x) * 4) as usize;
            let fx = x as f32 / plasma_w as f32;
            let fy = y as f32 / plasma_h as f32;
            let v1 = (fx * 10.0).sin();
            let v2 = fy.mul_add(8.0, fx * 6.0).sin();
            let v3 = (fx * 5.0).hypot(fy * 5.0).sin();
            let v = ((v1 + v2 + v3) / 3.0).mul_add(0.5, 0.5);
            plasma_data[idx] = (v * 200.0 + 55.0) as u8;
            plasma_data[idx + 1] = (1.0 - v).mul_add(150.0, 50.0) as u8;
            plasma_data[idx + 2] = (v * 2.0).sin().abs().mul_add(200.0, 55.0) as u8;
            plasma_data[idx + 3] = 255;
        }
    }
    let plasma_img = ImageData::new(plasma_w, plasma_h, plasma_data);
    canvas = canvas.image(plasma_img, wf * 0.55, hf * 0.35).done();

    // Overlay: tiled checkerboard at 30% opacity on top of plasma
    canvas = canvas
        .image(checker.clone(), wf * 0.55, hf * 0.35)
        .opacity(0.3)
        .done();

    // --- Ring of images with decreasing opacity ---
    for i in 0..8 {
        let angle = (i as f32) * std::f32::consts::FRAC_PI_4;
        let ix = wf.mul_add(0.5, angle.cos() * 100.0);
        let iy = hf.mul_add(0.75, angle.sin() * 60.0);
        let alpha = (i as f32).mul_add(-0.1, 1.0);
        canvas = canvas
            .image(checker.clone(), ix - 16.0, iy - 16.0)
            .opacity(alpha)
            .done();
    }

    canvas
}

// ─────────────────────────────────────────────────────────────────────────────
// Page 5: Transition Lifecycle
// ─────────────────────────────────────────────────────────────────────────────

/// Demonstrates: `Transition.reverse()`, .`remaining()`,
/// .`linear_progress()`, .`eased_progress()`, `AnimationState.is_active()`,
/// .`active_count()`.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn build_transition_lifecycle(
    area: Rect,
    pxstate: &PixelCanvasState,
    bounce: &Transition<f32>,
    _reverse_count: u32,
    anim: &AnimationState,
) -> PixelCanvas {
    let (w, h, wf, hf) = pixel_size(area, pxstate);

    let mut canvas = PixelCanvas::new(w, h).background(C::from_rgba8(10, 15, 25, 255));

    // --- Section 1: Bounce transition with reverse() ---
    let bounce_val = bounce.value();
    let bounce_x = wf.mul_add(0.1, bounce_val * (wf * 0.35));
    canvas = canvas
        .circle(bounce_x, hf * 0.12, 20.0)
        .fill(C::from_hsla(bounce_val * 120.0, 0.9, 0.6, 1.0))
        .done();

    // Progress bars: linear vs eased vs remaining
    let lin_prog = bounce.linear_progress();
    let eas_prog = bounce.eased_progress();
    let bar_w = wf * 0.35;
    let bar_x = wf * 0.55;

    // Linear progress bar
    canvas = canvas
        .rect(bar_x, hf * 0.06, bar_w, 10.0)
        .fill(C::from_rgba8(40, 40, 60, 255))
        .done()
        .rect(bar_x, hf * 0.06, bar_w * lin_prog, 10.0)
        .fill(C::from_rgba8(100, 150, 255, 255))
        .done();
    // Eased progress bar
    canvas = canvas
        .rect(bar_x, hf * 0.11, bar_w, 10.0)
        .fill(C::from_rgba8(40, 40, 60, 255))
        .done()
        .rect(bar_x, hf * 0.11, bar_w * eas_prog.clamp(0.0, 1.0), 10.0)
        .fill(C::from_rgba8(255, 150, 100, 255))
        .done();

    // Remaining time visualization
    let remaining_ms = bounce.remaining().as_millis() as f32;
    let total_ms = bounce.duration().as_millis() as f32;
    let remaining_frac = if total_ms > 0.0 {
        remaining_ms / total_ms
    } else {
        0.0
    };
    canvas = canvas
        .rect(bar_x, hf * 0.16, bar_w, 10.0)
        .fill(C::from_rgba8(40, 40, 60, 255))
        .done()
        .rect(bar_x, hf * 0.16, bar_w * remaining_frac, 10.0)
        .fill(C::from_rgba8(100, 255, 150, 255))
        .done();

    // --- Section 2: AnimationState orchestrator visualization ---
    // Pulse circle
    let pulse_val: f32 = anim.get("pulse").unwrap_or(0.3);
    canvas = canvas
        .circle(wf * 0.2, hf * 0.42, 30.0 * pulse_val)
        .fill(C::from_hsla(198.0, 0.8, 0.6, pulse_val))
        .stroke(C::WHITE.with_alpha(0.5), 1.5)
        .done();

    // Slide rect
    let slide_val: f32 = anim.get("slide").unwrap_or(0.0);
    canvas = canvas
        .rect(wf.mul_add(0.1, slide_val * 0.5), hf * 0.52, 40.0, 25.0)
        .fill(C::from_rgba8(255, 100, 200, 255))
        .corner_radius(6.0)
        .done();

    // Spin indicator
    let spin_val: f32 = anim.get("spin").unwrap_or(0.0);
    let spin_cx = wf * 0.7;
    let spin_cy = hf * 0.42;
    canvas = canvas
        .group(Transform::rotate_at(spin_val, spin_cx, spin_cy))
        .canvas(|c| {
            c.rect(spin_cx - 25.0, spin_cy - 25.0, 50.0, 50.0)
                .fill(C::from_hsla(108.0, 0.8, 0.5, 0.8))
                .stroke(C::WHITE, 2.0)
                .corner_radius(8.0)
                .done()
        })
        .done();

    // is_active() indicator dots
    let names = ["pulse", "slide", "spin"];
    for (i, name) in names.iter().enumerate() {
        let is_running = anim.is_active(name);
        let dot_color = if is_running {
            C::from_rgba8(50, 255, 100, 255)
        } else {
            C::from_rgba8(100, 50, 50, 255)
        };
        canvas = canvas
            .circle(wf.mul_add(0.35, (i as f32) * 30.0), hf * 0.62, 8.0)
            .fill(dot_color)
            .done();
    }

    // --- Section 3: Keyframes with value_at() curve visualization ---
    let kf = Keyframes::new(vec![
        Keyframe {
            position: 0.0,
            value: 0.0_f32,
            easing: Easing::Linear,
        },
        Keyframe {
            position: 0.3,
            value: hf * 0.15,
            easing: Easing::Bounce,
        },
        Keyframe {
            position: 0.6,
            value: hf * 0.05,
            easing: Easing::EaseInOutCubic,
        },
        Keyframe {
            position: 1.0,
            value: hf * 0.12,
            easing: Easing::EaseOutExpo,
        },
    ]);

    // Visualize the full keyframe curve as a dot trail
    let curve_x_start = wf * 0.1;
    let curve_x_end = wf * 0.9;
    let curve_y = hf * 0.76;
    let n_samples = 80;
    for i in 0..n_samples {
        let t_sample = i as f32 / n_samples as f32;
        let val = kf.value_at(t_sample);
        let x = curve_x_start + t_sample * (curve_x_end - curve_x_start);
        canvas = canvas
            .circle(x, curve_y + val * 0.8, 2.0)
            .fill(C::from_hsla(t_sample * 180.0, 0.9, 0.7, 1.0))
            .done();
    }

    canvas
}
