//! **Scry Fetch** — custom system info display with live pixel animation.
//!
//! A fully native replacement for `fastfetch --logo none`.  System info is
//! gathered from `/proc`, `/etc/os-release`, and environment variables — no
//! external binary required.  A sacred-geometry animation renders on the left
//! alongside your system info using ratatui's inline viewport (output stays
//! visible after exit).
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

#[path = "sysinfo_fetch.rs"]
mod sysinfo_fetch;

use std::f32::consts::{FRAC_PI_3, TAU};
use std::io::stdout;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode},
    ExecutableCommand,
};
use ratatui::prelude::*;
use ratatui::widgets::{Cell, Paragraph, Row, Table};

use scry_engine::prelude::{Picker, PixelCanvasState, PixelCanvasWidget, ProtocolKind};
use scry_engine::scene::PixelCanvas;
use scry_engine::style::Color as C;
use scry_engine::transport;

// ═══════════════════════════════════════════════════════════════════
// Theme colors — pastel palette
// ═══════════════════════════════════════════════════════════════════

const SOFT_BLUE: C = C::from_rgb8(158, 193, 255); // #9EC1FF
const SOFT_PINK: C = C::from_rgb8(242, 181, 212); // #F2B5D4
const SOFT_PURPLE: C = C::from_rgb8(203, 182, 255); // #CBB6FF
const SOFT_GREEN: C = C::from_rgb8(168, 213, 186); // #A8D5BA
const SOFT_PEACH: C = C::from_rgb8(244, 192, 149); // #F4C095
const SOFT_CREAM: C = C::from_rgb8(243, 231, 179); // #F3E7B3

const PALETTE: [C; 6] = [
    SOFT_BLUE,
    SOFT_PINK,
    SOFT_PURPLE,
    SOFT_GREEN,
    SOFT_PEACH,
    SOFT_CREAM,
];

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Collect system info — pure Rust, no fastfetch binary needed
    let info = sysinfo_fetch::SysInfo::collect();
    let title = info.user_at_host.clone();
    let rows  = info.rows(); // Vec<(icon, label, value)>

    // Total display height = title + separator + rows + padding
    let display_height = (rows.len() as u16 + 4).max(8);

    // 2. Set up ratatui with INLINE viewport — output stays after exit
    enable_raw_mode()?;
    stdout().execute(crossterm::cursor::Hide)?;

    let options = ratatui::TerminalOptions {
        viewport: ratatui::Viewport::Inline(display_height),
    };
    let mut terminal = Terminal::with_options(CrosstermBackend::new(stdout()), options)?;

    // 3. Detect protocol & set up scry-engine pixel canvas
    let picker = Picker::detect();
    let backend: Box<dyn transport::ProtocolBackend> = match picker.protocol() {
        ProtocolKind::Kitty => Box::new(transport::kitty::KittyBackend::new(picker.font_size())),
        _ => Box::new(transport::halfblock::HalfblockBackend::new()),
    };
    let mut px_state = PixelCanvasState::new(backend, picker.font_size());

    // 4. Animation loop — runs until q or Esc is pressed
    let start = Instant::now();

    loop {
        let t = start.elapsed().as_secs_f32();

        terminal.draw(|frame| {
            let area = frame.area();

            // Layout: [animation | spacer | sysinfo]
            let logo_cols = display_height
                .saturating_mul(2)
                .min(area.width / 3)
                .max(14);
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

            // ─── Right: Native sysinfo with styled colors ───
            // Split vertically: header (title + sep + blank) | table rows
            let v = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3), // blank + title + separator
                    Constraint::Min(1),
                ])
                .split(text_area);
            frame.render_widget(Paragraph::new(build_header(&title, t)), v[0]);
            frame.render_widget(build_info_table(&rows, t), v[1]);
        })?;
        px_state.flush()?;

        // Handle input — q / Esc to quit
        if event::poll(Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press
                    && matches!(key.code, KeyCode::Char('q') | KeyCode::Esc)
                {
                    break;
                }
            }
        }
    }

    // 5. Clean exit — inline viewport keeps content visible
    px_state.cleanup();
    disable_raw_mode()?;
    stdout().execute(crossterm::cursor::Show)?;
    println!();

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════
// Header: blank line + user@host + separator rule
// ═══════════════════════════════════════════════════════════════════

fn build_header(title: &str, t: f32) -> Vec<Line<'static>> {
    let green  = ratatui::style::Color::Rgb(168, 213, 186);
    let pink   = ratatui::style::Color::Rgb(242, 181, 212);
    let muted  = ratatui::style::Color::Rgb(120, 115, 110);
    let purple = ratatui::style::Color::Rgb(203, 182, 255);

    let mut out = vec![Line::default()];

    let fade = (t / 0.3).clamp(0.0, 1.0);
    let title_line = if fade > 0.0 {
        let parts: Vec<&str> = title.splitn(2, '@').collect();
        if parts.len() == 2 {
            Line::from(vec![
                Span::raw("  "),
                Span::styled(parts[0].to_string(), Style::default().fg(green).bold()),
                Span::styled("@", Style::default().fg(muted)),
                Span::styled(parts[1].to_string(), Style::default().fg(pink).bold()),
            ])
        } else {
            Line::from(vec![
                Span::raw("  "),
                Span::styled(title.to_string(), Style::default().fg(green).bold()),
            ])
        }
    } else {
        Line::default()
    };
    out.push(title_line);

    // Separator rule — same visual width as title
    let sep = "─".repeat(title.chars().count() + 2);
    out.push(Line::from(vec![
        Span::raw("  "),
        Span::styled(sep, Style::default().fg(purple)),
    ]));

    out
}

// ═══════════════════════════════════════════════════════════════════
// Info table — fixed columns guarantee horizontal alignment
//
//  col 0  │ col 1    │ col 2      │ col 3
//  " ▎ "  │ icon + " "│ label      │ value
//  (4)     │ (3)       │ (10)       │ (Min)
// ═══════════════════════════════════════════════════════════════════

fn build_info_table(rows: &[(&str, &str, &str)], t: f32) -> Table<'static> {
    let label_colors = [
        ratatui::style::Color::Rgb(168, 213, 186), // green
        ratatui::style::Color::Rgb(158, 193, 255), // blue
        ratatui::style::Color::Rgb(244, 192, 149), // peach
        ratatui::style::Color::Rgb(243, 231, 179), // cream
        ratatui::style::Color::Rgb(203, 182, 255), // purple
        ratatui::style::Color::Rgb(242, 181, 212), // pink
    ];
    let sep_color  = ratatui::style::Color::Rgb(203, 182, 255);
    let icon_color = ratatui::style::Color::Rgb(203, 182, 255);
    let val_color  = ratatui::style::Color::Rgb(232, 226, 215);

    let table_rows: Vec<Row<'static>> = rows
        .iter()
        .enumerate()
        .map(|(i, (icon, label, value))| {
            let row_start = i as f32 * 0.12;
            let alpha = ((t - row_start) / 0.25).clamp(0.0, 1.0);

            if alpha <= 0.0 {
                // Invisible placeholder row — keeps spacing
                return Row::new(vec![
                    Cell::from(""),
                    Cell::from(""),
                    Cell::from(""),
                    Cell::from(""),
                ]);
            }

            let label_col = label_colors[i % label_colors.len()];

            Row::new(vec![
                // col 0: separator bar
                Cell::from(Span::styled(
                    " ▎ ",
                    Style::default().fg(sep_color),
                )),
                // col 1: icon (always 1 glyph; Table clips to the column width)
                Cell::from(Span::styled(
                    icon.to_string(),
                    Style::default().fg(icon_color),
                )),
                // col 2: label — Table pads/clips to Constraint::Length(10)
                Cell::from(Span::styled(
                    label.to_string(),
                    Style::default().fg(label_col).bold(),
                )),
                // col 3: value
                Cell::from(Span::styled(
                    value.to_string(),
                    Style::default().fg(val_color),
                )),
            ])
        })
        .collect();

    // Column widths are enforced by the Table widget — alignment is exact.
    Table::new(
        table_rows,
        [
            Constraint::Length(4),  // " ▎ "
            Constraint::Length(3),  // icon (2-wide glyph + 1 space padding)
            Constraint::Length(10), // label (longest is "Packages" = 8 + pad)
            Constraint::Min(1),     // value — takes remaining width
        ],
    )
    .column_spacing(1)
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

    let mut canvas = PixelCanvas::new(w, h);

    // Phase timing — clean 6s loop
    let cycle = 6.0_f32;
    let phase = t % cycle;

    let envelope = if phase < 1.0 {
        phase
    } else if phase < 5.0 {
        1.0
    } else {
        cycle - phase
    };

    let intro    = (phase / 0.6).min(1.0) * envelope;
    let flower   = ((phase - 0.2) / 0.5).clamp(0.0, 1.0) * envelope;
    let geometry = ((phase - 0.4) / 0.5).clamp(0.0, 1.0) * envelope;
    let radiance = ((phase - 0.7) / 0.3).clamp(0.0, 1.0) * envelope;

    let rot    = t * 0.4;
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
            let alpha  = intro * (0.4 - i as f32 * 0.1);
            let color  = palette_color(i, t).with_alpha(alpha);
            canvas = canvas.circle(cx, cy, ring_r).stroke(color, 1.2).done();
        }
    }

    // ─── Flower of Life circles ───
    if flower > 0.0 {
        let r = radius / 4.0;

        let mut centers: Vec<(f32, f32, usize)> = vec![(cx, cy, 0)];

        for i in 0..6 {
            let angle = i as f32 * FRAC_PI_3 + rot;
            centers.push((
                (r * breath).mul_add(angle.cos(), cx),
                (r * breath).mul_add(angle.sin(), cy),
                1,
            ));
        }
        for i in 0..6 {
            let angle = i as f32 * FRAC_PI_3 + rot;
            centers.push((
                (2.0 * r * breath).mul_add(angle.cos(), cx),
                (2.0 * r * breath).mul_add(angle.sin(), cy),
                2,
            ));
        }
        for i in 0..6 {
            let angle = (i as f32).mul_add(FRAC_PI_3, FRAC_PI_3 / 2.0) + rot;
            let sqrt3 = 3.0_f32.sqrt();
            centers.push((
                (sqrt3 * r * breath).mul_add(angle.cos(), cx),
                (sqrt3 * r * breath).mul_add(angle.sin(), cy),
                2,
            ));
        }

        let rings_revealed = flower * 3.0;

        for &(x, y, ring) in &centers {
            let ring_progress = (rings_revealed - ring as f32).clamp(0.0, 1.0);
            if ring_progress <= 0.0 {
                continue;
            }
            let stroke_color = palette_color(ring, t).with_alpha(ring_progress * 0.7);
            let fill_color   = palette_color(ring + 2, t).with_alpha(ring_progress * 0.05);
            let current_r    = r * ring_progress;

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
        let hex_r = radius * 0.55 * breath;
        let hex_points: Vec<(f32, f32)> = (0..6)
            .map(|i| {
                let angle = (i as f32).mul_add(FRAC_PI_3, rot);
                (hex_r.mul_add(angle.cos(), cx), hex_r.mul_add(angle.sin(), cy))
            })
            .collect();

        canvas = canvas
            .polygon(hex_points.clone())
            .stroke(SOFT_BLUE.with_alpha(geometry * 0.5), 1.0)
            .done();

        let star_r    = radius * 0.4 * breath;
        let tri_alpha = geometry * 0.6;

        let up_tri: Vec<(f32, f32)> = (0..3)
            .map(|i| {
                let angle = (i as f32).mul_add(TAU / 3.0, rot - std::f32::consts::FRAC_PI_2);
                (star_r.mul_add(angle.cos(), cx), star_r.mul_add(angle.sin(), cy))
            })
            .collect();
        canvas = canvas
            .polygon(up_tri)
            .stroke(SOFT_PINK.with_alpha(tri_alpha), 1.2)
            .fill(SOFT_PINK.with_alpha(tri_alpha * 0.04))
            .done();

        let down_tri: Vec<(f32, f32)> = (0..3)
            .map(|i| {
                let angle = (i as f32).mul_add(TAU / 3.0, rot + std::f32::consts::FRAC_PI_2);
                (star_r.mul_add(angle.cos(), cx), star_r.mul_add(angle.sin(), cy))
            })
            .collect();
        canvas = canvas
            .polygon(down_tri)
            .stroke(SOFT_PURPLE.with_alpha(tri_alpha), 1.2)
            .fill(SOFT_PURPLE.with_alpha(tri_alpha * 0.04))
            .done();

        for &(hx, hy) in &hex_points {
            canvas = canvas
                .line(cx, cy, hx, hy)
                .color(SOFT_CREAM.with_alpha(geometry * 0.2))
                .width(0.6)
                .done();
        }
    }

    // ─── Central bindu ───
    if geometry > 0.3 {
        let bindu_reveal = ((geometry - 0.3) / 0.7).clamp(0.0, 1.0);
        let bindu_r = radius * 0.04 * breath;

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
        let pulse  = (t * 3.0).sin().mul_add(0.5, 0.5);
        let n_rays = 12;
        for i in 0..n_rays {
            let angle = t.mul_add(0.3, i as f32 * TAU / n_rays as f32);
            let len   = radius * 0.7 * 0.25f32.mul_add(t.mul_add(1.5, i as f32).sin(), 0.75);
            let x2    = cx + len * angle.cos();
            let y2    = cy + len * angle.sin();
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
    let shifted = idx + (time * 0.5) as usize;
    PALETTE[shifted % PALETTE.len()]
}
