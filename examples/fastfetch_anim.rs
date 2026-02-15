//! **Fastfetch Animation** — replace the ASCII logo with a live pixel animation.
//!
//! Runs `fastfetch --logo none` to capture system info text, then renders
//! a sacred-geometry animation on the left alongside your system info
//! using ratatui's inline viewport (output stays visible after exit).
//!
//! Run with: `cargo run --example fastfetch_anim`

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::unreadable_literal
)]

use std::f32::consts::{FRAC_PI_3, TAU};
use std::io::stdout;
use std::process::Command;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode},
    ExecutableCommand,
};
use ratatui::prelude::*;
use ratatui::widgets::Paragraph;

use scry_engine::prelude::{Picker, PixelCanvasState, PixelCanvasWidget, ProtocolKind};
use scry_engine::scene::PixelCanvas;
use scry_engine::style::Color as C;
use scry_engine::transport;

// ═══════════════════════════════════════════════════════════════════
// Theme colors — matching the user's fastfetch pastel palette
// ═══════════════════════════════════════════════════════════════════

const SOFT_BLUE: C = C::from_rgb8(158, 193, 255); // #9EC1FF
const SOFT_PINK: C = C::from_rgb8(242, 181, 212); // #F2B5D4
const SOFT_PURPLE: C = C::from_rgb8(203, 182, 255); // #CBB6FF
const SOFT_GREEN: C = C::from_rgb8(168, 213, 186); // #A8D5BA
const SOFT_PEACH: C = C::from_rgb8(244, 192, 149); // #F4C095
const SOFT_CREAM: C = C::from_rgb8(243, 231, 179); // #F3E7B3

const PALETTE: [C; 6] = [
    SOFT_BLUE, SOFT_PINK, SOFT_PURPLE, SOFT_GREEN, SOFT_PEACH, SOFT_CREAM,
];

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Capture fastfetch text output
    let ff_text = capture_fastfetch();
    let ff_lines: Vec<&str> = ff_text.lines().collect();
    let info_height = ff_lines.len().max(8) as u16 + 2; // +2 for padding

    // 2. Set up ratatui with INLINE viewport — output stays after exit
    enable_raw_mode()?;
    // Move cursor down to create space, then back up
    stdout().execute(crossterm::cursor::Hide)?;

    let options = ratatui::TerminalOptions {
        viewport: ratatui::Viewport::Inline(info_height),
    };
    let mut terminal = Terminal::with_options(CrosstermBackend::new(stdout()), options)?;

    // 3. Detect protocol & set up scry-engine
    let picker = Picker::detect();
    let backend: Box<dyn transport::ProtocolBackend> = match picker.protocol() {
        ProtocolKind::Kitty => Box::new(transport::kitty::KittyBackend::new(picker.font_size())),
        _ => Box::new(transport::halfblock::HalfblockBackend::new()),
    };
    let mut px_state = PixelCanvasState::new(backend, picker.font_size());

    // 4. Animation loop — runs until q is pressed
    let start = Instant::now();

    loop {
        let now = Instant::now();
        let elapsed = now.duration_since(start);
        let t = elapsed.as_secs_f32();

        terminal.draw(|frame| {
            let area = frame.area();

            // Layout: [animation | spacing | text]
            let logo_cols = (info_height as u16).saturating_mul(2).min(area.width / 3).max(14);
            let chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Length(logo_cols),
                    Constraint::Length(1), // spacer
                    Constraint::Min(1),
                ])
                .split(area);

            let anim_area = chunks[0];
            let text_area = chunks[2];

            // ─── Left: Animated sacred geometry orb ───
            let canvas = build_anim_scene(anim_area, &px_state, t);
            frame.render_stateful_widget(
                PixelCanvasWidget::new(canvas).z_index(-1),
                anim_area,
                &mut px_state,
            );

            // ─── Right: Fastfetch text with styled colors ───
            let styled_text = build_styled_text(&ff_lines, t);
            let para = Paragraph::new(styled_text);
            frame.render_widget(para, text_area);
        })?;
        px_state.flush()?;

        // Handle input — q to quit early
        if event::poll(Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press && key.code == KeyCode::Char('q') {
                    break;
                }
            }
        }
    }

    // 5. Clean exit — inline viewport keeps content visible
    px_state.cleanup();
    disable_raw_mode()?;
    stdout().execute(crossterm::cursor::Show)?;

    // Print a newline so the prompt appears below the output
    println!();

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════
// Fastfetch capture
// ═══════════════════════════════════════════════════════════════════

fn capture_fastfetch() -> String {
    Command::new("fastfetch")
        .arg("--logo")
        .arg("none")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_else(|_| {
            // Fallback if fastfetch isn't installed
            "user@host\nOS Unknown\nKernel ???\nUptime ???\nShell ???\nTerminal ???\nPackages ???\nMemory ???\n".to_string()
        })
}

// ═══════════════════════════════════════════════════════════════════
// Styled text builder — pastel colors with fade-in per line
// ═══════════════════════════════════════════════════════════════════

fn build_styled_text<'a>(lines: &[&str], t: f32) -> Vec<Line<'a>> {
    let colors = [
        ratatui::style::Color::Rgb(168, 213, 186), // green  — title
        ratatui::style::Color::Rgb(168, 213, 186), // green  — os
        ratatui::style::Color::Rgb(158, 193, 255), // blue   — kernel
        ratatui::style::Color::Rgb(244, 192, 149), // peach  — uptime
        ratatui::style::Color::Rgb(243, 231, 179), // cream  — shell
        ratatui::style::Color::Rgb(203, 182, 255), // purple — terminal
        ratatui::style::Color::Rgb(168, 213, 186), // green  — packages
        ratatui::style::Color::Rgb(242, 181, 212), // pink   — memory
    ];

    let mut result = Vec::new();

    // Top padding
    result.push(Line::default());

    for (i, line) in lines.iter().enumerate() {
        // Staggered fade-in: each line appears 0.15s after the previous
        let line_start = i as f32 * 0.15;
        let alpha = ((t - line_start) / 0.3).clamp(0.0, 1.0);

        if alpha <= 0.0 {
            result.push(Line::default());
            continue;
        }

        let color = colors.get(i).copied().unwrap_or(ratatui::style::Color::Rgb(232, 226, 215));

        // The separator from fastfetch config
        let sep_color = ratatui::style::Color::Rgb(203, 182, 255);

        // First line is the title (user@host)
        if i == 0 {
            let parts: Vec<&str> = line.splitn(2, '@').collect();
            if parts.len() == 2 {
                let user_span = Span::styled(
                    parts[0].to_string(),
                    Style::default().fg(ratatui::style::Color::Rgb(168, 213, 186)),
                );
                let at_span = Span::styled(
                    "@".to_string(),
                    Style::default().fg(ratatui::style::Color::Rgb(154, 146, 135)),
                );
                let host_span = Span::styled(
                    parts[1].to_string(),
                    Style::default().fg(ratatui::style::Color::Rgb(242, 181, 212)),
                );
                result.push(Line::from(vec![
                    Span::raw("  "),
                    user_span,
                    at_span,
                    host_span,
                ]));
            } else {
                result.push(Line::from(vec![
                    Span::raw("  "),
                    Span::styled(line.to_string(), Style::default().fg(color)),
                ]));
            }
        } else {
            // Info lines: use separator style
            result.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(" ▎", Style::default().fg(sep_color)),
                Span::styled(
                    format!(" {line}"),
                    Style::default().fg(color),
                ),
            ]));
        }
    }

    result
}

// ═══════════════════════════════════════════════════════════════════
// Animation scene — pulsing sacred geometry orb
// ═══════════════════════════════════════════════════════════════════

fn build_anim_scene(area: Rect, px_state: &PixelCanvasState, t: f32) -> PixelCanvas {
    let font = px_state.font_size();
    let w = u32::from(area.width) * u32::from(font.width);
    let h = u32::from(area.height) * u32::from(font.height);

    if w == 0 || h == 0 {
        return PixelCanvas::new(1, 1);
    }

    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let radius = (w.min(h) as f32) * 0.42;

    // Transparent background — blends with terminal
    let mut canvas = PixelCanvas::new(w, h);

    // ─── Phase timing — clean 6s loop ───
    // Phase 0.0–1.0: Build up (rings + flower + geometry fade in)
    // Phase 1.0–5.0: Steady state (everything visible, rotating)
    // Phase 5.0–6.0: Fade out, then restart
    let cycle = 6.0_f32;
    let phase = t % cycle;

    // Fade in over first second, sustain, fade out in last second
    let envelope = if phase < 1.0 {
        phase // 0→1 fade in
    } else if phase < 5.0 {
        1.0 // sustain
    } else {
        cycle - phase // 1→0 fade out
    };

    let intro = (phase / 0.6).min(1.0) * envelope;
    let flower = ((phase - 0.2) / 0.5).clamp(0.0, 1.0) * envelope;
    let geometry = ((phase - 0.4) / 0.5).clamp(0.0, 1.0) * envelope;
    let radiance = ((phase - 0.7) / 0.3).clamp(0.0, 1.0) * envelope;

    // Slow rotation
    let rot = t * 0.4;

    // Breathing effect
    let breath = 0.03f32.mul_add((t * 1.8).sin(), 1.0);

    // ─── Background glow ───
    if intro > 0.0 {
        let glow_alpha = intro * 0.08;
        canvas = canvas
            .circle(cx, cy, radius * 1.3 * breath)
            .fill(SOFT_PURPLE.with_alpha(glow_alpha))
            .done();
        canvas = canvas
            .circle(cx, cy, radius * 1.1 * breath)
            .fill(SOFT_BLUE.with_alpha(glow_alpha * 1.5))
            .done();
    }

    // ─── Outer concentric rings ───
    if intro > 0.0 {
        for i in 0..3 {
            let ring_r = radius * (1.0 - i as f32 * 0.08) * breath;
            let alpha = intro * (0.4 - i as f32 * 0.1);
            let color = palette_color(i, t).with_alpha(alpha);
            canvas = canvas
                .circle(cx, cy, ring_r)
                .stroke(color, 1.2)
                .done();
        }
    }

    // ─── Flower of Life circles ───
    if flower > 0.0 {
        let r = radius / 4.0;

        // Ring 0: center
        let rings: Vec<(f32, f32, usize)> = {
            let mut centers = Vec::new();
            centers.push((cx, cy, 0));

            // Ring 1: 6 circles
            for i in 0..6 {
                let angle = i as f32 * FRAC_PI_3 + rot;
                let x = (r * breath).mul_add(angle.cos(), cx);
                let y = (r * breath).mul_add(angle.sin(), cy);
                centers.push((x, y, 1));
            }

            // Ring 2: 6 circles at 2r
            for i in 0..6 {
                let angle = i as f32 * FRAC_PI_3 + rot;
                let x = (2.0 * r * breath).mul_add(angle.cos(), cx);
                let y = (2.0 * r * breath).mul_add(angle.sin(), cy);
                centers.push((x, y, 2));
            }

            // Ring 2 intermediates
            for i in 0..6 {
                let angle = (i as f32).mul_add(FRAC_PI_3, FRAC_PI_3 / 2.0) + rot;
                let sqrt3 = 3.0_f32.sqrt();
                let x = (sqrt3 * r * breath).mul_add(angle.cos(), cx);
                let y = (sqrt3 * r * breath).mul_add(angle.sin(), cy);
                centers.push((x, y, 2));
            }

            centers
        };

        let max_rings = 2;
        let rings_revealed = flower * (max_rings as f32 + 1.0);

        for &(x, y, ring) in &rings {
            let ring_progress = (rings_revealed - ring as f32).clamp(0.0, 1.0);
            if ring_progress <= 0.0 {
                continue;
            }

            let stroke_color = palette_color(ring, t).with_alpha(ring_progress * 0.7);
            let fill_color = palette_color(ring + 2, t).with_alpha(ring_progress * 0.05);

            let scale = ring_progress;
            let current_r = r * scale;

            // Glow halo
            if ring_progress > 0.5 {
                let halo_alpha = (ring_progress - 0.5) * 0.08;
                canvas = canvas
                    .circle(x, y, current_r * 1.4)
                    .fill(palette_color(ring + 1, t).with_alpha(halo_alpha))
                    .done();
            }

            canvas = canvas
                .circle(x, y, current_r)
                .fill(fill_color)
                .stroke(stroke_color, 1.2)
                .done();
        }
    }

    // ─── Inner hexagonal geometry ───
    if geometry > 0.0 {
        // Hexagonal frame
        let hex_r = radius * 0.55 * breath;
        let hex_points: Vec<(f32, f32)> = (0..6)
            .map(|i| {
                let angle = (i as f32).mul_add(FRAC_PI_3, rot);
                (
                    hex_r.mul_add(angle.cos(), cx),
                    hex_r.mul_add(angle.sin(), cy),
                )
            })
            .collect();

        let hex_color = SOFT_BLUE.with_alpha(geometry * 0.5);
        canvas = canvas
            .polygon(hex_points.clone())
            .stroke(hex_color, 1.0)
            .done();

        // Star of David — two overlapping triangles
        let star_r = radius * 0.4 * breath;
        let tri_alpha = geometry * 0.6;

        // Upward triangle
        let up_tri: Vec<(f32, f32)> = (0..3)
            .map(|i| {
                let angle = (i as f32).mul_add(TAU / 3.0, rot - std::f32::consts::FRAC_PI_2);
                (
                    star_r.mul_add(angle.cos(), cx),
                    star_r.mul_add(angle.sin(), cy),
                )
            })
            .collect();
        canvas = canvas
            .polygon(up_tri)
            .stroke(SOFT_PINK.with_alpha(tri_alpha), 1.2)
            .fill(SOFT_PINK.with_alpha(tri_alpha * 0.04))
            .done();

        // Downward triangle
        let down_tri: Vec<(f32, f32)> = (0..3)
            .map(|i| {
                let angle = (i as f32).mul_add(TAU / 3.0, rot + std::f32::consts::FRAC_PI_2);
                (
                    star_r.mul_add(angle.cos(), cx),
                    star_r.mul_add(angle.sin(), cy),
                )
            })
            .collect();
        canvas = canvas
            .polygon(down_tri)
            .stroke(SOFT_PURPLE.with_alpha(tri_alpha), 1.2)
            .fill(SOFT_PURPLE.with_alpha(tri_alpha * 0.04))
            .done();

        // Inner connecting lines (hex vertices to center)
        for &(hx, hy) in &hex_points {
            let line_alpha = geometry * 0.2;
            canvas = canvas
                .line(cx, cy, hx, hy)
                .color(SOFT_CREAM.with_alpha(line_alpha))
                .width(0.6)
                .done();
        }
    }

    // ─── Central bindu (sacred point) ───
    if geometry > 0.3 {
        let bindu_reveal = ((geometry - 0.3) / 0.7).clamp(0.0, 1.0);
        let bindu_r = radius * 0.04 * breath;

        // Glow rings
        for i in 0..3 {
            let gr = bindu_r * (i as f32).mul_add(2.5, 3.0);
            let ga = bindu_reveal * 0.06 / (i as f32).mul_add(0.5, 1.0);
            canvas = canvas
                .circle(cx, cy, gr)
                .fill(SOFT_PINK.with_alpha(ga))
                .done();
        }

        canvas = canvas
            .circle(cx, cy, bindu_r)
            .fill(C::WHITE.with_alpha(bindu_reveal * 0.9))
            .done();
    }

    // ─── Radiance rays ───
    if radiance > 0.0 {
        let pulse = (t * 3.0).sin().mul_add(0.5, 0.5);
        let n_rays = 12;
        for i in 0..n_rays {
            let angle = t.mul_add(0.3, i as f32 * TAU / n_rays as f32);
            let len = radius * 0.7 * 0.25f32.mul_add(t.mul_add(1.5, i as f32).sin(), 0.75);
            let x2 = cx + len * angle.cos();
            let y2 = cy + len * angle.sin();

            let ray_color = palette_color(i % PALETTE.len(), t)
                .with_alpha(radiance * 0.1 * 0.4f32.mul_add(pulse, 0.6));

            canvas = canvas
                .line(cx, cy, x2, y2)
                .color(ray_color)
                .width(1.0)
                .done();
        }
    }

    canvas
}

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

fn palette_color(idx: usize, time: f32) -> C {
    // Slowly shift through palette with time
    let shifted = idx + (time * 0.5) as usize;
    PALETTE[shifted % PALETTE.len()]
}
