//! Vortex: spectrum-history rings twisted into a perspective tunnel.

use std::f32::consts::TAU;

use scry_engine::scene::style::{BlendMode, GradientDef, GradientKind, GradientStop, Point};
use scry_engine::scene::PixelCanvas;
use scry_engine::style::LineCap;

use super::VizState;
use crate::analysis::AnalysisFrame;
use crate::theme::Theme;

const SLICES: usize = 72;
const RINGS: usize = 24;
const PUSH_INTERVAL: f32 = 0.045;

fn push_ring(st: &mut VizState, s: &AnalysisFrame, dt: f32) {
    st.vortex_acc += dt;
    if st.vortex_acc < PUSH_INTERVAL && !st.vortex_rows.is_empty() {
        return;
    }

    st.vortex_acc = 0.0;
    let grouped = super::group_bands(&s.bands, SLICES);
    st.vortex_rows.push_back(super::smooth3(&grouped));
    while st.vortex_rows.len() > RINGS {
        st.vortex_rows.pop_front();
    }
}

pub(super) fn build(
    mut canvas: PixelCanvas,
    st: &mut VizState,
    w: u32,
    h: u32,
    s: &AnalysisFrame,
    theme: &Theme,
    t: f32,
    dt: f32,
) -> PixelCanvas {
    push_ring(st, s, dt);

    let (w, h) = (w.max(1) as f32, h.max(1) as f32);
    let (cx, cy) = (w * 0.5, h * 0.5);
    let scale = w.min(h);

    let core = scale * (0.12 + 0.08 * s.bass + 0.05 * s.beat.envelope);
    canvas = canvas
        .circle(cx, cy, core)
        .fill_radial_gradient(GradientDef {
            kind: GradientKind::Radial {
                center: Point { x: cx, y: cy },
                radius: core,
            },
            stops: vec![
                GradientStop {
                    position: 0.0,
                    color: theme.accent.with_lightness(1.4).with_alpha(0.36),
                },
                GradientStop {
                    position: 0.62,
                    color: theme.accent.with_alpha(0.12),
                },
                GradientStop {
                    position: 1.0,
                    color: theme.accent.with_alpha(0.0),
                },
            ],
        })
        .blend_mode(BlendMode::Screen)
        .done();

    let mut rings: Vec<Vec<(f32, f32)>> = Vec::with_capacity(st.vortex_rows.len());
    for (idx, row) in st.vortex_rows.iter().enumerate() {
        let depth = if RINGS > 1 {
            (idx + RINGS - st.vortex_rows.len()) as f32 / (RINGS - 1) as f32
        } else {
            1.0
        };
        let near = depth.powf(1.45);
        let radius = scale * (0.06 + 0.46 * near);
        let amp = scale * (0.015 + 0.16 * near) * (1.0 + 0.25 * s.beat.envelope);
        let twist = t * (0.38 + 0.28 * s.high_mid) + depth * TAU * (0.30 + 0.22 * s.treble);
        let vertical = (depth - 0.50) * h * 0.11;

        let pts: Vec<(f32, f32)> = row
            .iter()
            .enumerate()
            .map(|(slice, &value)| {
                let a = twist + slice as f32 / SLICES as f32 * TAU;
                let r = radius + value.powf(0.74) * amp;
                (
                    cx + a.cos() * r * (1.0 + 0.08 * near),
                    cy + a.sin() * r * (0.54 + 0.10 * near) + vertical,
                )
            })
            .collect();

        let color = theme.sample(0.12 + 0.78 * depth);
        canvas = canvas
            .polygon(pts.clone())
            .stroke(
                color.with_alpha(0.08 + 0.50 * near),
                0.5 + 2.2 * near + 0.8 * s.beat.envelope,
            )
            .blend_mode(BlendMode::Screen)
            .done();
        rings.push(pts);
    }

    for slice in (0..SLICES).step_by(6) {
        for pair in rings.windows(2) {
            let p0 = pair[0][slice];
            let p1 = pair[1][slice];
            let y_mid = (p0.1 + p1.1) * 0.5 / h;
            canvas = canvas
                .line(p0.0, p0.1, p1.0, p1.1)
                .stroke(
                    theme.sample(y_mid).with_alpha(0.12 + 0.24 * s.high_mid),
                    0.7 + 0.8 * s.transient,
                )
                .line_cap(LineCap::Round)
                .done();
        }
    }

    let beat_r = scale * (0.18 + 0.42 * s.beat.envelope);
    canvas
        .circle(cx, cy, beat_r)
        .stroke(
            theme
                .accent
                .with_lightness(1.3)
                .with_alpha(0.10 + 0.30 * s.beat.envelope),
            1.0 + 2.5 * s.beat.envelope,
        )
        .blend_mode(BlendMode::Screen)
        .done()
}
