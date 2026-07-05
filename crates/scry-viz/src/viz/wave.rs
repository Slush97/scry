//! Oscilloscope: layered glow strokes over the raw waveform.

use scry_engine::scene::style::{GradientDef, GradientKind, GradientStop, Point};
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
    let mid = h * 0.5;
    let amp = h * 0.40;
    // Normalize to the window peak so quiet tracks still fill the scope
    // without clipping, capped so silence doesn't amplify noise.
    let peak = s.waveform.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let gain = (0.9 / peak.max(0.02)).min(20.0);

    let n = s.waveform.len();
    let step = (n / (w as usize).max(1)).max(1);
    let points: Vec<(f32, f32)> = s
        .waveform
        .iter()
        .step_by(step)
        .enumerate()
        .map(|(i, &v)| {
            let x = i as f32 * step as f32 / (n - 1) as f32 * w;
            (x, mid - (v * gain).clamp(-1.0, 1.0) * amp)
        })
        .collect();

    let gradient = |alpha: f32| GradientDef {
        kind: GradientKind::Linear {
            start: Point { x: 0.0, y: mid },
            end: Point { x: w, y: mid },
        },
        stops: [0.0, 0.5, 1.0]
            .iter()
            .map(|&p| GradientStop {
                position: p,
                color: theme.sample(p).with_alpha(alpha),
            })
            .collect(),
    };

    canvas = canvas
        .line(0.0, mid, w, mid)
        .stroke(theme.accent.with_alpha(0.12), 1.0)
        .done();

    // Glow: same polyline at decreasing width and increasing opacity.
    for (width, alpha) in [(7.0, 0.10), (3.5, 0.30), (1.6, 1.0)] {
        canvas = canvas
            .polyline(points.clone())
            .stroke(theme.accent, width * (1.0 + 0.6 * s.beat.envelope))
            .stroke_gradient(gradient(alpha))
            .done();
    }

    canvas
}
