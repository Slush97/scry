// SPDX-License-Identifier: MIT OR Apache-2.0
//! Logo selection and rendering for the fetch splash.
//!
//! Supports three modes:
//! - **Auto**: detect the distro from `/etc/os-release` and render its vector logo.
//! - **Geometry**: animated sacred geometry (flower of life, hexagons, star of David).
//! - **None**: no logo.

use std::f32::consts::{FRAC_PI_3, TAU};

use scry_engine::scene::PixelCanvas;
use scry_engine::style::Color as C;

use super::sysinfo::SysInfo;

// ── Palette ─────────────────────────────────────────────────────

const SOFT_BLUE: C = C::from_rgb8(158, 193, 255);
const SOFT_PINK: C = C::from_rgb8(242, 181, 212);
const SOFT_PURPLE: C = C::from_rgb8(203, 182, 255);
const SOFT_GREEN: C = C::from_rgb8(168, 213, 186);
const SOFT_PEACH: C = C::from_rgb8(244, 192, 149);
const SOFT_CREAM: C = C::from_rgb8(243, 231, 179);

const ANIM_PALETTE: [C; 6] = [
    SOFT_BLUE, SOFT_PINK, SOFT_PURPLE, SOFT_GREEN, SOFT_PEACH, SOFT_CREAM,
];

fn anim_palette(idx: usize, time: f32) -> C {
    ANIM_PALETTE[(idx + (time * 0.5) as usize) % ANIM_PALETTE.len()]
}

// ── Logo source ─────────────────────────────────────────────────

/// Which logo to render.
#[derive(Debug, Clone)]
pub(crate) enum LogoSource {
    /// Auto-detected distro logo.
    Distro(String),
    /// Sacred geometry animation.
    Geometry,
    /// No logo.
    None,
}

impl LogoSource {
    /// Resolve the logo source from the config string.
    pub(super) fn resolve(config_value: &str) -> Self {
        match config_value {
            "none" => Self::None,
            "geometry" => Self::Geometry,
            "auto" => Self::Distro(SysInfo::distro_id()),
            other => {
                // Treat as distro name override
                Self::Distro(other.to_lowercase())
            }
        }
    }
}

// ── Logo rendering ──────────────────────────────────────────────

/// Render the logo onto a `PixelCanvas`.
///
/// `canvas_w` and `canvas_h` are the full canvas pixel dimensions.
/// The logo renders centered in the left `canvas_h`-wide region.
/// `t` is the animation time in seconds.
pub(crate) fn render_logo(source: &LogoSource, canvas_w: u32, canvas_h: u32, t: f32) -> PixelCanvas {
    if canvas_w == 0 || canvas_h == 0 {
        return PixelCanvas::new(1, 1);
    }

    // Logo occupies a square region on the left (width = height, capped at 1/3 of canvas)
    let logo_w = canvas_h.min(canvas_w / 3);

    match source {
        LogoSource::None => {
            let _ = logo_w;
            PixelCanvas::new(canvas_w, canvas_h)
        }
        LogoSource::Geometry => render_sacred_geometry(logo_w, canvas_h, t, canvas_w),
        LogoSource::Distro(id) => render_distro_logo(id, logo_w, canvas_h, t, canvas_w),
    }
}

// ── Distro logos ────────────────────────────────────────────────

fn render_distro_logo(distro_id: &str, w: u32, h: u32, t: f32, canvas_w: u32) -> PixelCanvas {
    match distro_id {
        "arch" | "archlinux" | "artix" | "endeavouros" => render_arch_logo(w, h, t, canvas_w),
        "ubuntu" => render_ubuntu_logo(w, h, t, canvas_w),
        "fedora" => render_fedora_logo(w, h, t, canvas_w),
        "debian" => render_debian_logo(w, h, t, canvas_w),
        "nixos" | "nix" => render_nix_logo(w, h, t, canvas_w),
        "void" | "voidlinux" => render_void_logo(w, h, t, canvas_w),
        "gentoo" => render_gentoo_logo(w, h, t, canvas_w),
        _ => render_sacred_geometry(w, h, t, canvas_w),
    }
}

/// Arch Linux — stylized triangle "A" shape.
fn render_arch_logo(w: u32, h: u32, t: f32, canvas_w: u32) -> PixelCanvas {
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let r = (w.min(h) as f32) * 0.4;
    let breath = 0.02f32.mul_add((t * 1.5).sin(), 1.0);

    let mut canvas = PixelCanvas::new(canvas_w, h);

    // Background glow
    canvas = canvas
        .circle(cx, cy, r * 1.3 * breath)
        .fill(C::from_rgb8(23, 147, 209).with_alpha(0.06))
        .done();

    // Outer triangle (Arch shape)
    let top = (cx, cy - r * breath);
    let bl = (cx - r * 0.8 * breath, cy + r * 0.7 * breath);
    let br = (cx + r * 0.8 * breath, cy + r * 0.7 * breath);
    canvas = canvas
        .polygon(vec![top, bl, br])
        .stroke(C::from_rgb8(23, 147, 209).with_alpha(0.8), 2.5)
        .done();

    // Inner notch (the "gap" in the A)
    let notch_y = cy + r * 0.15 * breath;
    let nl = (cx - r * 0.3 * breath, notch_y);
    let nr = (cx + r * 0.3 * breath, notch_y);
    let nb = (cx, cy + r * 0.45 * breath);
    canvas = canvas
        .polygon(vec![nl, nb, nr])
        .stroke(C::from_rgb8(23, 147, 209).with_alpha(0.6), 1.8)
        .done();

    // Subtle center dot
    canvas = canvas
        .circle(cx, cy - r * 0.2 * breath, r * 0.04)
        .fill(C::from_rgb8(23, 147, 209).with_alpha(0.5))
        .done();

    canvas
}

/// Ubuntu — circle with three "friends" dots.
fn render_ubuntu_logo(w: u32, h: u32, t: f32, canvas_w: u32) -> PixelCanvas {
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let r = (w.min(h) as f32) * 0.35;
    let breath = 0.02f32.mul_add((t * 1.5).sin(), 1.0);

    let mut canvas = PixelCanvas::new(canvas_w, h);

    // Main circle
    canvas = canvas
        .circle(cx, cy, r * breath)
        .stroke(C::from_rgb8(233, 84, 32).with_alpha(0.7), 2.0)
        .done();

    // Inner circle
    canvas = canvas
        .circle(cx, cy, r * 0.5 * breath)
        .stroke(C::from_rgb8(233, 84, 32).with_alpha(0.4), 1.5)
        .done();

    // Three "friends" dots around the circle
    for i in 0..3 {
        let angle = i as f32 * TAU / 3.0 - std::f32::consts::FRAC_PI_2;
        let dx = (r * 0.75 * breath) * angle.cos();
        let dy = (r * 0.75 * breath) * angle.sin();
        canvas = canvas
            .circle(cx + dx, cy + dy, r * 0.12)
            .fill(C::from_rgb8(233, 84, 32).with_alpha(0.8))
            .done();
    }

    canvas
}

/// Fedora — stylized "f" in a circle.
fn render_fedora_logo(w: u32, h: u32, t: f32, canvas_w: u32) -> PixelCanvas {
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let r = (w.min(h) as f32) * 0.35;
    let breath = 0.02f32.mul_add((t * 1.5).sin(), 1.0);

    let mut canvas = PixelCanvas::new(canvas_w, h);

    // Outer circle
    canvas = canvas
        .circle(cx, cy, r * breath)
        .stroke(C::from_rgb8(60, 110, 180).with_alpha(0.7), 2.0)
        .done();

    // Infinity-esque inner shape (simplified)
    canvas = canvas
        .circle(cx, cy, r * 0.55 * breath)
        .stroke(C::from_rgb8(60, 110, 180).with_alpha(0.4), 1.5)
        .done();

    // Vertical bar of the "f"
    let bar_x = cx - r * 0.1;
    canvas = canvas
        .line(bar_x, cy - r * 0.4 * breath, bar_x, cy + r * 0.4 * breath)
        .color(C::from_rgb8(60, 110, 180).with_alpha(0.8))
        .width(2.5)
        .done();

    // Horizontal bar of the "f"
    canvas = canvas
        .line(cx - r * 0.3, cy - r * 0.1, cx + r * 0.2, cy - r * 0.1)
        .color(C::from_rgb8(60, 110, 180).with_alpha(0.7))
        .width(2.0)
        .done();

    canvas
}

/// Debian — stylized swirl.
fn render_debian_logo(w: u32, h: u32, t: f32, canvas_w: u32) -> PixelCanvas {
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let r = (w.min(h) as f32) * 0.35;
    let breath = 0.02f32.mul_add((t * 1.5).sin(), 1.0);
    let rot = t * 0.2;

    let mut canvas = PixelCanvas::new(canvas_w, h);

    // Spiral approximation using arcs
    let color = C::from_rgb8(215, 7, 81);
    for i in 0..24 {
        let a1 = i as f32 * TAU / 24.0 + rot;
        let a2 = (i + 1) as f32 * TAU / 24.0 + rot;
        let sr = r * (0.3 + 0.7 * (i as f32 / 24.0)) * breath;
        let x1 = sr.mul_add(a1.cos(), cx);
        let y1 = sr.mul_add(a1.sin(), cy);
        let x2 = (r * (0.3 + 0.7 * ((i + 1) as f32 / 24.0)) * breath).mul_add(a2.cos(), cx);
        let y2 = (r * (0.3 + 0.7 * ((i + 1) as f32 / 24.0)) * breath).mul_add(a2.sin(), cy);
        canvas = canvas
            .line(x1, y1, x2, y2)
            .color(color.with_alpha(0.5 + 0.3 * (i as f32 / 24.0)))
            .width(2.0)
            .done();
    }

    canvas
}

/// NixOS — stylized snowflake / lambda.
fn render_nix_logo(w: u32, h: u32, t: f32, canvas_w: u32) -> PixelCanvas {
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let r = (w.min(h) as f32) * 0.35;
    let breath = 0.02f32.mul_add((t * 1.5).sin(), 1.0);
    let rot = t * 0.15;

    let mut canvas = PixelCanvas::new(canvas_w, h);

    let color1 = C::from_rgb8(126, 186, 228);
    let color2 = C::from_rgb8(80, 145, 207);

    // Six-pointed snowflake arms
    for i in 0..6 {
        let angle = i as f32 * FRAC_PI_3 + rot;
        let x2 = (r * 0.85 * breath).mul_add(angle.cos(), cx);
        let y2 = (r * 0.85 * breath).mul_add(angle.sin(), cy);
        canvas = canvas
            .line(cx, cy, x2, y2)
            .color(if i % 2 == 0 { color1 } else { color2 }.with_alpha(0.7))
            .width(2.5)
            .done();

        // Small cross-bars at the tips
        let perp = angle + std::f32::consts::FRAC_PI_2;
        let tip_x = (r * 0.7 * breath).mul_add(angle.cos(), cx);
        let tip_y = (r * 0.7 * breath).mul_add(angle.sin(), cy);
        let bar = r * 0.15 * breath;
        canvas = canvas
            .line(
                tip_x - bar * perp.cos(), tip_y - bar * perp.sin(),
                tip_x + bar * perp.cos(), tip_y + bar * perp.sin(),
            )
            .color(color1.with_alpha(0.5))
            .width(1.5)
            .done();
    }

    canvas
}

/// Void Linux — stylized "V" void.
fn render_void_logo(w: u32, h: u32, t: f32, canvas_w: u32) -> PixelCanvas {
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let r = (w.min(h) as f32) * 0.35;
    let breath = 0.02f32.mul_add((t * 1.5).sin(), 1.0);

    let mut canvas = PixelCanvas::new(canvas_w, h);

    let color = C::from_rgb8(72, 130, 65);

    // Outer circle
    canvas = canvas
        .circle(cx, cy, r * breath)
        .stroke(color.with_alpha(0.6), 2.0)
        .done();

    // "V" shape inside
    let v_top_l = (cx - r * 0.35 * breath, cy - r * 0.3 * breath);
    let v_top_r = (cx + r * 0.35 * breath, cy - r * 0.3 * breath);
    let v_bot = (cx, cy + r * 0.35 * breath);
    canvas = canvas
        .polygon(vec![v_top_l, v_bot, v_top_r])
        .stroke(color.with_alpha(0.8), 2.5)
        .done();

    canvas
}

/// Gentoo — stylized "g" / atom.
fn render_gentoo_logo(w: u32, h: u32, t: f32, canvas_w: u32) -> PixelCanvas {
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let r = (w.min(h) as f32) * 0.35;
    let breath = 0.02f32.mul_add((t * 1.5).sin(), 1.0);
    let rot = t * 0.3;

    let mut canvas = PixelCanvas::new(canvas_w, h);

    let color = C::from_rgb8(180, 160, 220);

    // Three orbital ellipses (approximated as rotated circles)
    for i in 0..3 {
        let angle = i as f32 * FRAC_PI_3 + rot;
        // Draw an arc of points to simulate tilted ellipse
        let points: Vec<(f32, f32)> = (0..32)
            .map(|j| {
                let a = j as f32 * TAU / 32.0;
                let ex = r * 0.8 * breath * a.cos();
                let ey = r * 0.3 * breath * a.sin();
                // Rotate by orbit angle
                let rx = ex * angle.cos() - ey * angle.sin();
                let ry = ex * angle.sin() + ey * angle.cos();
                (cx + rx, cy + ry)
            })
            .collect();
        canvas = canvas
            .polygon(points)
            .stroke(color.with_alpha(0.5), 1.5)
            .done();
    }

    // Center nucleus
    canvas = canvas
        .circle(cx, cy, r * 0.08)
        .fill(color.with_alpha(0.7))
        .done();

    canvas
}

// ── Sacred geometry ─────────────────────────────────────────────

/// Animated sacred geometry (flower of life, hexagons, star of David).
/// Ported from the CLI fetch implementation.
#[allow(clippy::similar_names, clippy::too_many_lines)]
fn render_sacred_geometry(w: u32, h: u32, t: f32, canvas_w: u32) -> PixelCanvas {
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let radius = (w.min(h) as f32) * 0.42;
    let mut canvas = PixelCanvas::new(canvas_w, h);

    let cycle = 6.0_f32;
    let phase = t % cycle;
    let envelope = if phase < 1.0 {
        phase
    } else if phase < 5.0 {
        1.0
    } else {
        cycle - phase
    };

    let intro = (phase / 0.6).min(1.0) * envelope;
    let flower = ((phase - 0.2) / 0.5).clamp(0.0, 1.0) * envelope;
    let geometry = ((phase - 0.4) / 0.5).clamp(0.0, 1.0) * envelope;
    let radiance = ((phase - 0.7) / 0.3).clamp(0.0, 1.0) * envelope;
    let rot = t * 0.4;
    let breath = 0.03f32.mul_add((t * 1.8).sin(), 1.0);

    // Background glow
    if intro > 0.0 {
        let ga = intro * 0.08;
        canvas = canvas
            .circle(cx, cy, radius * 1.3 * breath)
            .fill(SOFT_PURPLE.with_alpha(ga))
            .done();
        canvas = canvas
            .circle(cx, cy, radius * 1.1 * breath)
            .fill(SOFT_BLUE.with_alpha(ga * 1.5))
            .done();
    }

    // Concentric rings
    if intro > 0.0 {
        for i in 0..3 {
            let rr = radius * (1.0 - i as f32 * 0.08) * breath;
            let alpha = intro * (0.4 - i as f32 * 0.1);
            canvas = canvas
                .circle(cx, cy, rr)
                .stroke(anim_palette(i, t).with_alpha(alpha), 1.2)
                .done();
        }
    }

    // Flower of Life
    if flower > 0.0 {
        let r = radius / 4.0;
        let mut centers: Vec<(f32, f32, usize)> = vec![(cx, cy, 0)];
        for i in 0..6 {
            let a = i as f32 * FRAC_PI_3 + rot;
            centers.push((
                (r * breath).mul_add(a.cos(), cx),
                (r * breath).mul_add(a.sin(), cy),
                1,
            ));
        }
        for i in 0..6 {
            let a = i as f32 * FRAC_PI_3 + rot;
            centers.push((
                (2.0 * r * breath).mul_add(a.cos(), cx),
                (2.0 * r * breath).mul_add(a.sin(), cy),
                2,
            ));
        }
        for i in 0..6 {
            let a = (i as f32).mul_add(FRAC_PI_3, FRAC_PI_3 / 2.0) + rot;
            let s = 3.0_f32.sqrt();
            centers.push((
                (s * r * breath).mul_add(a.cos(), cx),
                (s * r * breath).mul_add(a.sin(), cy),
                2,
            ));
        }
        let rev = flower * 3.0;
        for &(x, y, ring) in &centers {
            let rp = (rev - ring as f32).clamp(0.0, 1.0);
            if rp <= 0.0 {
                continue;
            }
            let cur_r = r * rp;
            if rp > 0.5 {
                canvas = canvas
                    .circle(x, y, cur_r * 1.4)
                    .fill(anim_palette(ring + 1, t).with_alpha((rp - 0.5) * 0.08))
                    .done();
            }
            canvas = canvas
                .circle(x, y, cur_r)
                .fill(anim_palette(ring + 2, t).with_alpha(rp * 0.05))
                .stroke(anim_palette(ring, t).with_alpha(rp * 0.7), 1.2)
                .done();
        }
    }

    // Hexagon + Star of David
    if geometry > 0.0 {
        let hex_r = radius * 0.55 * breath;
        let hex: Vec<(f32, f32)> = (0..6)
            .map(|i| {
                let a = (i as f32).mul_add(FRAC_PI_3, rot);
                (hex_r.mul_add(a.cos(), cx), hex_r.mul_add(a.sin(), cy))
            })
            .collect();
        canvas = canvas
            .polygon(hex.clone())
            .stroke(SOFT_BLUE.with_alpha(geometry * 0.5), 1.0)
            .done();

        let star_r = radius * 0.4 * breath;
        let ta = geometry * 0.6;
        let up: Vec<(f32, f32)> = (0..3)
            .map(|i| {
                let a =
                    (i as f32).mul_add(TAU / 3.0, rot - std::f32::consts::FRAC_PI_2);
                (star_r.mul_add(a.cos(), cx), star_r.mul_add(a.sin(), cy))
            })
            .collect();
        canvas = canvas
            .polygon(up)
            .stroke(SOFT_PINK.with_alpha(ta), 1.2)
            .fill(SOFT_PINK.with_alpha(ta * 0.04))
            .done();
        let dn: Vec<(f32, f32)> = (0..3)
            .map(|i| {
                let a =
                    (i as f32).mul_add(TAU / 3.0, rot + std::f32::consts::FRAC_PI_2);
                (star_r.mul_add(a.cos(), cx), star_r.mul_add(a.sin(), cy))
            })
            .collect();
        canvas = canvas
            .polygon(dn)
            .stroke(SOFT_PURPLE.with_alpha(ta), 1.2)
            .fill(SOFT_PURPLE.with_alpha(ta * 0.04))
            .done();

        for &(hx, hy) in &hex {
            canvas = canvas
                .line(cx, cy, hx, hy)
                .color(SOFT_CREAM.with_alpha(geometry * 0.2))
                .width(0.6)
                .done();
        }

        // Bindu
        if geometry > 0.3 {
            let br = ((geometry - 0.3) / 0.7).clamp(0.0, 1.0);
            let bd = radius * 0.04 * breath;
            for i in 0..3 {
                canvas = canvas
                    .circle(cx, cy, bd * (i as f32).mul_add(2.5, 3.0))
                    .fill(SOFT_PINK.with_alpha(br * 0.06 / (i as f32).mul_add(0.5, 1.0)))
                    .done();
            }
            canvas = canvas
                .circle(cx, cy, bd)
                .fill(C::WHITE.with_alpha(br * 0.9))
                .done();
        }
    }

    // Radiance rays
    if radiance > 0.0 {
        let pulse = (t * 3.0).sin().mul_add(0.5, 0.5);
        for i in 0..12_usize {
            let a = t.mul_add(0.3, i as f32 * TAU / 12.0);
            let len =
                radius * 0.7 * 0.25f32.mul_add(t.mul_add(1.5, i as f32).sin(), 0.75);
            canvas = canvas
                .line(cx, cy, cx + len * a.cos(), cy + len * a.sin())
                .color(
                    anim_palette(i % ANIM_PALETTE.len(), t)
                        .with_alpha(radiance * 0.1 * 0.4f32.mul_add(pulse, 0.6)),
                )
                .width(1.0)
                .done();
        }
    }

    canvas
}
