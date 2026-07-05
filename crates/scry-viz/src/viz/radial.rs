//! Radial spectrum: mirrored bands around a slowly rotating ring that
//! pulses with the bass, each spoke wrapped in a soft glow.

use std::f32::consts::{FRAC_PI_2, TAU};

use scry_engine::scene::style::{BlendMode, GradientDef, GradientKind, GradientStop, Point};
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
    let beat = s.beat.envelope;
    let base_r = min_dim * 0.18 * (1.0 + 0.10 * beat + 0.06 * s.bass);
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
            let (x0, y0) = (cx + cos * base_r, cy + sin * base_r);
            let (x1, y1) = (cx + cos * (base_r + len), cy + sin * (base_r + len));

            canvas = canvas
                .polyline(vec![(x0, y0), (x1, y1)])
                .stroke(
                    color.with_alpha(0.10 + 0.22 * v),
                    spoke_w * (2.4 + 0.8 * beat),
                )
                .line_cap(LineCap::Round)
                .blend_mode(BlendMode::Screen)
                .done()
                .line(x0, y0, x1, y1)
                .stroke(color, spoke_w)
                .line_cap(LineCap::Round)
                .done()
                .circle(x1, y1, spoke_w * (0.5 + 0.6 * v))
                .fill(color.with_lightness(1.3).with_alpha(0.4 + 0.4 * v))
                .blend_mode(BlendMode::Screen)
                .done();
        }
    }

    // Core: radial glow that breathes with the beat, a thin ring at the
    // spoke base, and a bright center dot.
    let glow_r = base_r * (1.05 + 0.12 * beat);
    canvas
        .circle(cx, cy, glow_r)
        .fill_radial_gradient(GradientDef {
            kind: GradientKind::Radial {
                center: Point { x: cx, y: cy },
                radius: glow_r,
            },
            stops: vec![
                GradientStop {
                    position: 0.0,
                    color: theme
                        .accent
                        .with_lightness(1.3)
                        .with_alpha(0.45 + 0.25 * beat),
                },
                GradientStop {
                    position: 0.55,
                    color: theme.accent.with_alpha(0.14),
                },
                GradientStop {
                    position: 1.0,
                    color: theme.accent.with_alpha(0.0),
                },
            ],
        })
        .blend_mode(BlendMode::Screen)
        .done()
        .circle(cx, cy, base_r * 0.9)
        .stroke(theme.accent.with_alpha(0.4 + 0.5 * beat), 1.5)
        .done()
        .circle(cx, cy, base_r * 0.14)
        .fill(theme.accent.with_lightness(1.6).with_alpha(0.9))
        .done()
}
