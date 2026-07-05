//! Silk: layered smoothed spectrum ribbons with glow and a reflection.

use scry_engine::scene::style::{BlendMode, GradientDef, GradientKind, GradientStop, Point};
use scry_engine::scene::PixelCanvas;

use crate::analysis::AnalysisFrame;
use crate::theme::Theme;

fn x_gradient(w: f32, theme: &Theme, alpha: f32) -> GradientDef {
    GradientDef {
        kind: GradientKind::Linear {
            start: Point { x: 0.0, y: 0.0 },
            end: Point { x: w, y: 0.0 },
        },
        stops: [0.0f32, 0.35, 0.65, 1.0]
            .iter()
            .map(|&p| GradientStop {
                position: p,
                color: theme.sample(p).with_alpha(alpha),
            })
            .collect(),
    }
}

pub(super) fn build(
    mut canvas: PixelCanvas,
    w: u32,
    h: u32,
    s: &AnalysisFrame,
    theme: &Theme,
) -> PixelCanvas {
    let (w, h) = (w as f32, h as f32);
    let baseline = h * 0.62;
    let amp = h * 0.52 * (1.0 + 0.04 * s.beat.envelope);
    let bands = super::smooth3(&s.bands);
    let n = bands.len();

    let curve = |squash: f32| -> Vec<(f32, f32)> {
        bands
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let x = i as f32 / (n - 1) as f32 * w;
                (x, baseline - v * amp * squash)
            })
            .collect()
    };
    let main = curve(1.0);

    // Reflection: mirrored, squashed, dim.
    let mirror: Vec<(f32, f32)> = main
        .iter()
        .map(|&(x, y)| (x, baseline + (baseline - y) * 0.40))
        .collect();
    if let Some(path) = super::spline(&mirror, Some(baseline)) {
        canvas = canvas
            .path(path.clone())
            .fill_linear_gradient(x_gradient(w, theme, 0.05))
            .done()
            .path(path)
            .stroke(theme.accent, 1.2)
            .stroke_gradient(x_gradient(w, theme, 0.14))
            .done();
    }

    // Depth ribbons behind the main curve: same spectrum squashed toward
    // the baseline, screened so crossings brighten.
    for (squash, alpha) in [(0.35, 0.20), (0.62, 0.30)] {
        if let Some(path) = super::spline(&curve(squash), None) {
            canvas = canvas
                .path(path)
                .stroke(theme.accent, 1.6)
                .stroke_gradient(x_gradient(w, theme, alpha))
                .blend_mode(BlendMode::Screen)
                .done();
        }
    }

    // Area fill under the main curve, fading toward the baseline.
    if let Some(path) = super::spline(&main, Some(baseline)) {
        canvas = canvas
            .path(path)
            .fill_linear_gradient(x_gradient(w, theme, 0.13))
            .done();
    }

    // Main curve: two glow passes then the bright core line.
    let glow_boost = 1.0 + 0.45 * s.beat.envelope;
    for (width, alpha, blend) in [
        (9.0, 0.08, BlendMode::Screen),
        (4.0, 0.20, BlendMode::Screen),
        (2.0, 1.0, BlendMode::SrcOver),
    ] {
        if let Some(path) = super::spline(&main, None) {
            canvas = canvas
                .path(path)
                .stroke(theme.accent, width * glow_boost)
                .stroke_gradient(x_gradient(w, theme, alpha))
                .blend_mode(blend)
                .done();
        }
    }

    canvas
        .line(0.0, baseline, w, baseline)
        .stroke(theme.accent.with_alpha(0.18), 1.0)
        .done()
}
