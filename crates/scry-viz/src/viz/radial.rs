//! Radial spectrum: mirrored bands around a slowly rotating ring that
//! pulses with the bass.

use std::f32::consts::{FRAC_PI_2, TAU};

use scry_engine::scene::PixelCanvas;
use scry_engine::style::LineCap;

use crate::analysis::AnalysisFrame;
use crate::theme::Theme;

pub(super) fn build(
    mut canvas: PixelCanvas,
    w: u32,
    h: u32,
    s: &AnalysisFrame,
    theme: &Theme,
    t: f32,
) -> PixelCanvas {
    let (w, h) = (w as f32, h as f32);
    let (cx, cy) = (w * 0.5, h * 0.5);
    let min_dim = w.min(h);
    let base_r = min_dim * 0.18 * (1.0 + 0.10 * s.beat.envelope + 0.06 * s.bass);
    let max_len = min_dim * 0.30;
    let rotation = t * 0.15 - FRAC_PI_2;

    let n = s.bands.len();
    let spoke_w = (base_r * TAU / (2 * n) as f32 * 0.55).clamp(1.0, 6.0);

    // Mirror the spectrum across the vertical axis for a symmetric mandala.
    for (i, &v) in s.bands.iter().enumerate() {
        let frac = i as f32 / n as f32;
        let color = theme.sample(frac);
        let len = base_r * 0.08 + v * max_len;
        for angle in [rotation + frac * (TAU / 2.0), rotation - frac * (TAU / 2.0)] {
            let (sin, cos) = angle.sin_cos();
            canvas = canvas
                .line(
                    cx + cos * base_r,
                    cy + sin * base_r,
                    cx + cos * (base_r + len),
                    cy + sin * (base_r + len),
                )
                .stroke(color, spoke_w)
                .line_cap(LineCap::Round)
                .done();
        }
    }

    // Core: glow that breathes with the beat, thin ring at the spoke base.
    canvas = canvas
        .circle(cx, cy, base_r * 0.82)
        .fill(theme.accent.with_alpha(0.06 + 0.30 * s.beat.envelope))
        .done()
        .circle(cx, cy, base_r * 0.92)
        .stroke(theme.accent.with_alpha(0.5 + 0.5 * s.beat.envelope), 1.5)
        .done();

    canvas
}
