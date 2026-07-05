//! Prism: a perspective spectrum mesh with crystalline waveform edges.

use scry_engine::scene::style::{BlendMode, GradientDef, GradientKind, GradientStop, Point};
use scry_engine::scene::PixelCanvas;

use crate::analysis::AnalysisFrame;
use crate::theme::Theme;

const COLS: usize = 32;
const DEPTH: usize = 16;

fn sample_band(bands: &[f32], t: f32) -> f32 {
    if bands.is_empty() {
        return 0.0;
    }

    let x = t.clamp(0.0, 1.0) * (bands.len() - 1) as f32;
    let i = x.floor() as usize;
    let frac = x - i as f32;
    bands[i] + (bands[(i + 1).min(bands.len() - 1)] - bands[i]) * frac
}

#[allow(clippy::many_single_char_names)]
pub(super) fn build(
    mut canvas: PixelCanvas,
    w: u32,
    h: u32,
    s: &AnalysisFrame,
    theme: &Theme,
    t: f32,
) -> PixelCanvas {
    let (w, h) = (w.max(1) as f32, h.max(1) as f32);
    let cx = w * 0.5;
    let horizon = h * 0.20;
    let beat = 1.0 + 0.20 * s.beat.envelope;

    canvas = canvas
        .rect(0.0, horizon, w, h - horizon)
        .fill_linear_gradient(GradientDef {
            kind: GradientKind::Linear {
                start: Point { x: 0.0, y: horizon },
                end: Point { x: 0.0, y: h },
            },
            stops: vec![
                GradientStop {
                    position: 0.0,
                    color: theme.bg.with_alpha(0.0),
                },
                GradientStop {
                    position: 0.72,
                    color: theme.accent.with_alpha(0.035 + 0.055 * s.rms),
                },
                GradientStop {
                    position: 1.0,
                    color: theme.bg.with_alpha(0.0),
                },
            ],
        })
        .blend_mode(BlendMode::Screen)
        .done();

    let mut grid = vec![vec![(0.0f32, 0.0f32, 0.0f32); COLS]; DEPTH];
    for (d, row) in grid.iter_mut().enumerate() {
        let z = d as f32 / (DEPTH - 1) as f32;
        let near = z.powf(1.55);
        let span = w * (0.20 + 0.78 * near);
        let base_y = horizon + h * 0.70 * near;
        let skew = (z - 0.5) * w * 0.12 * (t * 0.22).sin();
        let amp = h * (0.035 + 0.19 * near) * beat;

        for (c, point) in row.iter_mut().enumerate() {
            let f = c as f32 / (COLS - 1) as f32;
            let shifted = (f + 0.035 * d as f32 + 0.018 * t).fract();
            let band = sample_band(&s.bands, shifted).powf(0.76);
            let ripple = (f * std::f32::consts::TAU * 3.0 + t * 1.4 + d as f32 * 0.65).sin()
                * h
                * 0.012
                * s.high_mid;
            let x = cx + (f - 0.5) * span + skew;
            let y = base_y - band * amp + ripple;
            *point = (x, y, band);
        }
    }

    for d in 0..DEPTH - 1 {
        let z = d as f32 / (DEPTH - 1) as f32;
        for c in 0..COLS - 1 {
            let p00 = grid[d][c];
            let p10 = grid[d][c + 1];
            let p11 = grid[d + 1][c + 1];
            let p01 = grid[d + 1][c];
            let level = ((p00.2 + p10.2 + p11.2 + p01.2) * 0.25).clamp(0.0, 1.0);
            if level < 0.018 && d % 3 != 0 {
                continue;
            }

            let freq = c as f32 / (COLS - 1) as f32;
            let color = theme
                .sample((0.78 * freq + 0.20 * z + 0.05 * s.treble).fract())
                .with_lightness(0.62 + 1.10 * level)
                .with_alpha(0.025 + 0.24 * level + 0.08 * z);
            canvas = canvas
                .polygon(vec![
                    (p00.0, p00.1),
                    (p10.0, p10.1),
                    (p11.0, p11.1),
                    (p01.0, p01.1),
                ])
                .fill(color)
                .blend_mode(BlendMode::Screen)
                .done();
        }
    }

    for d in (0..DEPTH).step_by(3) {
        let z = d as f32 / (DEPTH - 1) as f32;
        let points: Vec<(f32, f32)> = grid[d].iter().map(|&(x, y, _)| (x, y)).collect();
        if let Some(path) = super::spline(&points, None) {
            canvas = canvas
                .path(path)
                .stroke(
                    theme
                        .sample(z)
                        .with_alpha(0.12 + 0.38 * z + 0.16 * s.transient),
                    0.7 + 1.8 * z,
                )
                .blend_mode(BlendMode::Screen)
                .done();
        }
    }

    for c in (0..COLS).step_by(4) {
        for d in 0..DEPTH - 1 {
            let z = d as f32 / (DEPTH - 1) as f32;
            let p0 = grid[d][c];
            let p1 = grid[d + 1][c];
            canvas = canvas
                .line(p0.0, p0.1, p1.0, p1.1)
                .stroke(
                    theme
                        .sample(c as f32 / COLS as f32)
                        .with_alpha(0.05 + 0.20 * z),
                    0.7,
                )
                .done();
        }
    }

    let front: Vec<(f32, f32)> = grid[DEPTH - 1].iter().map(|&(x, y, _)| (x, y)).collect();
    if let Some(path) = super::spline(&front, None) {
        canvas = canvas
            .path(path)
            .stroke(
                theme
                    .accent
                    .with_lightness(1.45)
                    .with_alpha(0.62 + 0.26 * s.beat.envelope),
                2.0,
            )
            .blend_mode(BlendMode::Screen)
            .done();
    }

    canvas
}
