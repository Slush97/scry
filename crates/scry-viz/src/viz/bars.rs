//! Spectrum bars with soft glow, bright tips, and glowing peak caps.

use scry_engine::scene::style::{BlendMode, GradientDef, GradientKind, GradientStop, Point};
use scry_engine::scene::PixelCanvas;

use crate::analysis::AnalysisFrame;
use crate::theme::Theme;

pub(super) fn build(
    mut canvas: PixelCanvas,
    w: u32,
    h: u32,
    s: &AnalysisFrame,
    theme: &Theme,
) -> PixelCanvas {
    let (w, h) = (w as f32, h as f32);
    let n = s.bands.len();
    let gap = (w / n as f32 * 0.22).clamp(1.0, 6.0);
    let bw = (w - gap * (n + 1) as f32) / n as f32;
    let max_h = h * 0.94;
    let beat = s.beat.envelope;

    // Atmospheric floor that lifts with the bass. Screen-blended and
    // transparent at the top so the terminal still shows through.
    canvas = canvas
        .rect(0.0, h * 0.72, w, h * 0.28)
        .fill_linear_gradient(GradientDef {
            kind: GradientKind::Linear {
                start: Point {
                    x: 0.0,
                    y: h * 0.72,
                },
                end: Point { x: 0.0, y: h },
            },
            stops: vec![
                GradientStop {
                    position: 0.0,
                    color: theme.bg.with_alpha(0.0),
                },
                GradientStop {
                    position: 1.0,
                    color: theme.accent.with_alpha(0.05 + 0.10 * s.bass),
                },
            ],
        })
        .blend_mode(BlendMode::Screen)
        .done();

    for (i, &v) in s.bands.iter().enumerate() {
        let t = i as f32 / (n - 1) as f32;
        let color = theme.sample(t);
        let x = gap + i as f32 * (bw + gap);
        let bh = (v * max_h).max(1.0);
        let y = h - bh;

        // Soft glow halo behind active bars.
        if v > 0.02 {
            canvas = canvas
                .rect(x - bw * 0.4, y - bw * 0.7, bw * 1.8, bh + bw * 0.7)
                .corner_radius((bw * 0.6).min(7.0))
                .fill(color.with_alpha(0.05 + 0.13 * v * (1.0 + 0.4 * beat)))
                .blend_mode(BlendMode::Screen)
                .done();
        }

        // Bar body.
        canvas = canvas
            .rect(x, y, bw, bh)
            .corner_radius((bw * 0.3).min(4.0))
            .fill_linear_gradient(GradientDef {
                kind: GradientKind::Linear {
                    start: Point { x, y },
                    end: Point { x, y: h },
                },
                stops: vec![
                    GradientStop {
                        position: 0.0,
                        color,
                    },
                    GradientStop {
                        position: 1.0,
                        color: color.with_lightness(0.45).with_alpha(0.8),
                    },
                ],
            })
            .done();

        // Bright tip highlight at the crest.
        let tip_h = (bh * 0.14).clamp(1.0, 6.0);
        canvas = canvas
            .rect(x, y, bw, tip_h)
            .corner_radius((bw * 0.3).min(4.0))
            .fill(
                color
                    .with_lightness(1.35 + 0.3 * beat)
                    .with_alpha(0.6 + 0.4 * v),
            )
            .blend_mode(BlendMode::Screen)
            .done();

        // Falling peak cap with a soft glow.
        let py = h - s.peaks[i] * max_h - 3.0;
        if py < y - 2.0 {
            canvas = canvas
                .rect(x - bw * 0.18, py - 1.0, bw * 1.36, 4.0)
                .fill(color.with_lightness(1.4).with_alpha(0.18))
                .blend_mode(BlendMode::Screen)
                .done()
                .rect(x, py, bw, 2.0)
                .fill(color.with_lightness(1.4))
                .done();
        }
    }

    canvas
}
