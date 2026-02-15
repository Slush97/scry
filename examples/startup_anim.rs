//! **Startup Animation** — looping Kitty terminal splash screen.
//!
//! Renders a sacred geometry flower of life animation directly via the Kitty
//! graphics protocol alongside fastfetch system info. The animation forks to
//! the background so your shell prompt appears immediately — you can type
//! right away while the animation loops in the top rows.
//!
//! The animation automatically stops when it detects that the shell cursor
//! has moved past the animation region (i.e. you've typed enough commands to
//! push it up). The last frame is left in place as a static image — Kitty
//! keeps it visible on the scrolled-off cells with zero CPU cost.
//!
//! ## Usage
//!
//! ```bash
//! # Build the release binary
//! cargo build --release --example startup_anim
//!
//! # Run it (animation forks to background, prompt appears immediately)
//! ./target/release/examples/startup_anim
//!
//! # Install to ~/.local/bin for shell RC integration
//! cp target/release/examples/startup_anim ~/.local/bin/
//! ```
//!
//! ## Shell Integration
//!
//! Add to `~/.bashrc` or `~/.zshrc`:
//! ```bash
//! if [ -n "$KITTY_PID" ] && command -v startup_anim &>/dev/null; then
//!     startup_anim
//! fi
//! ```
//!
//! Or use a Kitty session file (`~/.config/kitty/startup.session`):
//! ```text
//! launch --type=background startup_anim
//! ```

#![allow(
    unsafe_code,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::unreadable_literal
)]

use std::f32::consts::{FRAC_PI_3, TAU};
use std::io::{self, Write};
use std::process::Command;
use std::time::{Duration, Instant};

use scry_engine::rasterize::Rasterizer;
use scry_engine::scene::PixelCanvas;
use scry_engine::style::Color as C;
use scry_engine::transport::kitty::KittyBackend;
use scry_engine::transport::{Picker, ProtocolBackend, TerminalPosition};

// ═══════════════════════════════════════════════════════════════════
// Theme — matched to fastfetch config.jsonc + kitty.conf
// ═══════════════════════════════════════════════════════════════════

// Fastfetch palette (from config.jsonc)
const SOFT_BLUE: C = C::from_rgb8(158, 193, 255);   // #9EC1FF — keys
const SOFT_PINK: C = C::from_rgb8(242, 181, 212);   // #F2B5D4 — title
const SOFT_PURPLE: C = C::from_rgb8(203, 182, 255); // #CBB6FF — separator
const SOFT_GREEN: C = C::from_rgb8(168, 213, 186);  // #A8D5BA — user/os
const SOFT_PEACH: C = C::from_rgb8(244, 192, 149);  // #F4C095 — uptime
const SOFT_CREAM: C = C::from_rgb8(243, 231, 179);  // #F3E7B3 — shell
const WARM_WHITE: C = C::from_rgb8(232, 226, 215);  // #E8E2D7 — output

// Kitty terminal accent colors (from kitty.conf)
const TERM_BLUE: C = C::from_rgb8(74, 118, 230);    // #4A76E6 — bright blue
const TERM_TEAL: C = C::from_rgb8(74, 163, 163);    // #4AA3A3 — bright cyan
const TERM_PURPLE: C = C::from_rgb8(176, 74, 162);   // #B04AA2 — bright magenta

const PALETTE: [C; 9] = [
    SOFT_BLUE, SOFT_PINK, SOFT_PURPLE, SOFT_GREEN, SOFT_PEACH, SOFT_CREAM,
    TERM_BLUE, TERM_TEAL, TERM_PURPLE,
];

/// How many terminal rows tall the animation region is.
const ANIM_ROWS: u16 = 8;

/// Animation cycle duration (seconds) — loops seamlessly.
const CYCLE: f32 = 6.0;

/// Target FPS.
const FPS: u64 = 30;

/// Safety-net lifetime for the background process (seconds).
/// The primary exit signal is a broken stdout pipe (terminal closed or
/// process detached). This timeout is a fallback in case stdout never errors.
const ANIM_LIFETIME: f32 = 120.0;

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

fn main() {
    // 1. Detect protocol + font size
    let picker = Picker::detect();
    let font = picker.font_size();

    // 2. Get terminal size
    let (term_cols, _term_rows) = crossterm::terminal::size().unwrap_or((80, 24));

    // 3. Decide layout: [animation | gap | fastfetch text]
    let anim_cols = (ANIM_ROWS * 2).min(term_cols / 3).max(12);
    let text_col = anim_cols + 2; // 2-column gap

    // 4. Capture fastfetch text
    let ff_text = capture_fastfetch();
    let ff_lines: Vec<&str> = ff_text.lines().collect();

    // 5. Print blank lines to reserve space for the animation region
    for _ in 0..ANIM_ROWS {
        println!();
    }

    // Move cursor up to the start of our reserved region
    print!("\x1b[{ANIM_ROWS}A");
    io::stdout().flush().unwrap();

    // 7. Print fastfetch text on the right side (static — printed once)
    print_fastfetch_text(&ff_lines, text_col);

    // Move cursor below the animation region so the shell prompt starts there
    print!("\x1b[{};1H", 1u32 + u32::from(ANIM_ROWS));
    io::stdout().flush().unwrap();

    // 8. Fork to background — parent returns to shell immediately
    #[cfg(unix)]
    unsafe {
        // Install SIGHUP handler before forking so child cleans up on terminal close
        libc::signal(libc::SIGHUP, libc::SIG_DFL);

        let pid = libc::fork();
        match pid.cmp(&0) {
            std::cmp::Ordering::Less => {
                // Fork failed — just run in foreground as fallback
                eprintln!("startup_anim: fork failed, running in foreground");
            }
            std::cmp::Ordering::Greater => {
                // Parent — exit immediately, shell prompt appears
                return;
            }
            std::cmp::Ordering::Equal => {
                // Child — detach from terminal session
                libc::setsid();
            }
        }
    }

    // 9. Set up the Kitty backend (child process only in fork case)
    let mut backend = KittyBackend::new(font);
    let position = TerminalPosition::new(0, 0, anim_cols, ANIM_ROWS);

    // 10. Animation loop — runs until the region scrolls off-screen,
    //     the lifetime expires, or stdout breaks (terminal closed).
    //     When the user scrolls past the animation region (by typing
    //     commands), the loop exits and the last frame persists as a
    //     static Kitty image with zero CPU cost.
    let start = Instant::now();
    let frame_dur = Duration::from_millis(1000 / FPS);
    let mut handle = None;
    let mut frame_count: u64 = 0;
    let mut scrolled_away = false;

    /// How often (in frames) to poll the cursor position for scroll detection.
    /// At 30 FPS this is roughly every 500 ms — frequent enough to stop
    /// quickly, infrequent enough to avoid excessive CSI 6n round-trips.
    const SCROLL_POLL_INTERVAL: u64 = 15;

    loop {
        let t = start.elapsed().as_secs_f32();

        // Exit after lifetime
        if t > ANIM_LIFETIME {
            break;
        }

        // ── Scroll detection ──
        // Every SCROLL_POLL_INTERVAL frames, query the shell cursor row.
        // If it has moved past the animation region the user has typed
        // enough to push the animation off-screen — stop rendering.
        if frame_count % SCROLL_POLL_INTERVAL == 0 && frame_count > 0 {
            if let Ok((_, row)) = crossterm::cursor::position() {
                if row >= ANIM_ROWS {
                    scrolled_away = true;
                    break;
                }
            }
        }
        frame_count += 1;

        // Build the scene
        let canvas = build_scene(
            u32::from(anim_cols) * u32::from(font.width),
            u32::from(ANIM_ROWS) * u32::from(font.height),
            t,
        );

        // Rasterize to pixmap
        let Ok(pixmap) = Rasterizer::rasterize(&canvas) else {
            break;
        };

        // Transmit or replace via Kitty protocol
        // Save/restore cursor so we don't interfere with the shell
        print!("\x1b[s"); // save cursor
        print!("\x1b[1;1H"); // move to animation origin (row 1)
        if let Some(ref h) = handle {
            match backend.replace(h, &pixmap, position, -1) {
                Ok(new_h) => handle = Some(new_h),
                Err(_) => break,
            }
        } else {
            match backend.transmit(&pixmap, position, -1) {
                Ok(h) => handle = Some(h),
                Err(_) => break,
            }
        }
        print!("\x1b[u"); // restore cursor

        // Flush stdout — if this fails (broken pipe, terminal closed),
        // exit cleanly. This is the primary exit signal.
        if io::stdout().flush().is_err() {
            break;
        }

        // Sleep for remainder of frame
        let elapsed = start.elapsed().as_secs_f32() - t;
        let target = frame_dur.as_secs_f32();
        if elapsed < target {
            std::thread::sleep(Duration::from_secs_f32(target - elapsed));
        }
    }

    // 11. Clean exit.
    //     • If the animation scrolled away, leave the last frame in place —
    //       Kitty keeps it as a static image on the scrolled-off cells.
    //     • Otherwise (timeout / pipe break), actively remove the image.
    if !scrolled_away {
        if let Some(ref h) = handle {
            let _ = backend.remove(h);
        }
    }
    let _ = io::stdout().flush();
}

// ═══════════════════════════════════════════════════════════════════
// Fastfetch capture
// ═══════════════════════════════════════════════════════════════════

fn capture_fastfetch() -> String {
    Command::new("fastfetch")
        .arg("--logo")
        .arg("none")
        .output()
        .map_or_else(
            |_| "user@host\nOS Unknown\nKernel ???\nUptime ???\nShell ???\nTerminal ???\n".to_string(),
            |o| String::from_utf8_lossy(&o.stdout).to_string(),
        )
}

// ═══════════════════════════════════════════════════════════════════
// Print fastfetch text at a column offset — static, printed once
// ═══════════════════════════════════════════════════════════════════

fn print_fastfetch_text(lines: &[&str], start_col: u16) {
    // Exact colors from fastfetch config.jsonc outputColor per module
    let colors = [
        "\x1b[38;2;168;213;186m", // #A8D5BA — title (user)
        "\x1b[38;2;168;213;186m", // #A8D5BA — os
        "\x1b[38;2;158;193;255m", // #9EC1FF — kernel
        "\x1b[38;2;244;192;149m", // #F4C095 — uptime
        "\x1b[38;2;243;231;179m", // #F3E7B3 — shell
        "\x1b[38;2;203;182;255m", // #CBB6FF — terminal
        "\x1b[38;2;168;213;186m", // #A8D5BA — packages
        "\x1b[38;2;242;181;212m", // #F2B5D4 — memory
    ];
    let reset = "\x1b[0m";
    let sep_color = "\x1b[38;2;203;182;255m"; // #CBB6FF

    for (i, line) in lines.iter().enumerate().take(ANIM_ROWS as usize) {
        let color = colors.get(i).unwrap_or(&"\x1b[38;2;232;226;215m"); // #E8E2D7 fallback

        // Position cursor: row i+1 (1-indexed), column start_col+1
        print!("\x1b[{};{}H", i + 1, start_col + 1);

        if i == 0 {
            // Title line: user@host with special coloring
            let parts: Vec<&str> = line.splitn(2, '@').collect();
            if parts.len() == 2 {
                print!(
                    "  \x1b[38;2;168;213;186m{}\x1b[38;2;154;146;135m@\x1b[38;2;242;181;212m{}{}",
                    parts[0], parts[1], reset
                );
            } else {
                print!("  {color}{line}{reset}");
            }
        } else {
            print!(
                "  {sep_color}▎{reset} {color}{line}{reset}"
            );
        }
    }

    // Flush once
    io::stdout().flush().unwrap();
}

// ═══════════════════════════════════════════════════════════════════
// Animation scene — sacred geometry flower of life
// ═══════════════════════════════════════════════════════════════════

fn build_scene(w: u32, h: u32, t: f32) -> PixelCanvas {
    if w == 0 || h == 0 {
        return PixelCanvas::new(1, 1);
    }

    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let radius = (w.min(h) as f32) * 0.42;

    let mut canvas = PixelCanvas::new(w, h);

    // Phase timing — clean cycle with seamless looping
    let phase = t % CYCLE;

    // Envelope: fade in → sustain → fade out → seamless restart
    let envelope = if phase < 0.8 {
        phase / 0.8
    } else if phase < (CYCLE - 0.8) {
        1.0
    } else {
        (CYCLE - phase) / 0.8
    };

    let intro = (phase / 0.6).min(1.0) * envelope;
    let flower = ((phase - 0.2) / 0.5).clamp(0.0, 1.0) * envelope;
    let geometry = ((phase - 0.4) / 0.5).clamp(0.0, 1.0) * envelope;
    let radiance = ((phase - 0.7) / 0.3).clamp(0.0, 1.0) * envelope;

    let rot = t * 0.3;
    let breath = 0.02f32.mul_add((t * 1.5).sin(), 1.0);

    // ─── Background glow (barely visible — no dark outlines) ───
    if intro > 0.0 {
        let glow_alpha = intro * 0.015;
        canvas = canvas
            .circle(cx, cy, radius * 0.95 * breath)
            .fill(SOFT_PURPLE.with_alpha(glow_alpha))
            .done();
        canvas = canvas
            .circle(cx, cy, radius * 0.7 * breath)
            .fill(SOFT_BLUE.with_alpha(glow_alpha))
            .done();
    }

    // ─── Outer concentric rings (thin, no dark outline) ───
    if intro > 0.0 {
        for i in 0..3 {
            let ring_r = radius * (1.0 - i as f32 * 0.08) * breath;
            let alpha = intro * (0.18 - i as f32 * 0.05);
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

        let rings: Vec<(f32, f32, usize)> = {
            let mut centers = Vec::new();
            centers.push((cx, cy, 0));

            for i in 0..6 {
                let angle = i as f32 * FRAC_PI_3 + rot;
                let x = (r * breath).mul_add(angle.cos(), cx);
                let y = (r * breath).mul_add(angle.sin(), cy);
                centers.push((x, y, 1));
            }

            for i in 0..6 {
                let angle = i as f32 * FRAC_PI_3 + rot;
                let x = (2.0 * r * breath).mul_add(angle.cos(), cx);
                let y = (2.0 * r * breath).mul_add(angle.sin(), cy);
                centers.push((x, y, 2));
            }

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

            let stroke_color = palette_color(ring, t).with_alpha(ring_progress * 0.5);
            let fill_color = palette_color(ring + 2, t).with_alpha(ring_progress * 0.03);
            let scale = ring_progress;
            let current_r = r * scale;

            canvas = canvas
                .circle(x, y, current_r)
                .fill(fill_color)
                .stroke(stroke_color, 1.5)
                .done();
        }
    }

    // ─── Inner hexagonal geometry ───
    if geometry > 0.0 {
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

        let hex_color = SOFT_BLUE.with_alpha(geometry * 0.35);
        canvas = canvas
            .polygon(hex_points.clone())
            .stroke(hex_color, 1.0)
            .done();

        // Star of David — two overlapping triangles
        let star_r = radius * 0.4 * breath;
        let tri_alpha = geometry * 0.4;

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
            .fill(SOFT_PINK.with_alpha(tri_alpha * 0.03))
            .done();

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
            .stroke(TERM_PURPLE.with_alpha(tri_alpha), 1.2)
            .fill(TERM_PURPLE.with_alpha(tri_alpha * 0.03))
            .done();

        for &(hx, hy) in &hex_points {
            let line_alpha = geometry * 0.18;
            canvas = canvas
                .line(cx, cy, hx, hy)
                .color(WARM_WHITE.with_alpha(line_alpha))
                .width(0.8)
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

    // ─── Radiance rays (subtle) ───
    if radiance > 0.0 {
        let pulse = (t * 2.0).sin().mul_add(0.5, 0.5);
        let n_rays = 12;
        for i in 0..n_rays {
            let angle = t.mul_add(0.2, i as f32 * TAU / n_rays as f32);
            let len = radius * 0.5 * 0.2f32.mul_add(t.mul_add(1.2, i as f32).sin(), 0.8);
            let x2 = cx + len * angle.cos();
            let y2 = cy + len * angle.sin();

            let ray_color = palette_color(i % PALETTE.len(), t)
                .with_alpha(radiance * 0.06 * 0.3f32.mul_add(pulse, 0.7));

            canvas = canvas
                .line(cx, cy, x2, y2)
                .color(ray_color)
                .width(0.8)
                .done();
        }
    }

    // ─── Floating sparkle particles ───
    if radiance > 0.0 {
        let n_sparkles = 8;
        for i in 0..n_sparkles {
            // Each sparkle has its own orbit and phase
            let orbit_r = radius * (0.3 + 0.4 * ((i as f32 * 1.7 + 0.5).sin() * 0.5 + 0.5));
            let speed = 0.15 + (i as f32 * 0.37).sin().abs() * 0.15;
            let sparkle_angle = t * speed + i as f32 * TAU / n_sparkles as f32;
            let sx = cx + orbit_r * sparkle_angle.cos() * breath;
            let sy = cy + orbit_r * sparkle_angle.sin() * breath;

            // Twinkle: each sparkle pulses in and out
            let twinkle = ((t * 2.5 + i as f32 * 1.3).sin() * 0.5 + 0.5).powi(2);
            let sparkle_alpha = radiance * twinkle * 0.35;
            let sparkle_r = radius * 0.012;

            let sparkle_color = palette_color(i % PALETTE.len(), t).with_alpha(sparkle_alpha);
            canvas = canvas
                .circle(sx, sy, sparkle_r)
                .fill(sparkle_color)
                .done();

            // Tiny glow around each sparkle
            if twinkle > 0.5 {
                canvas = canvas
                    .circle(sx, sy, sparkle_r * 3.0)
                    .fill(sparkle_color.with_alpha(sparkle_alpha * 0.15))
                    .done();
            }
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
