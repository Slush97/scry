//! Mandala: kaleidoscopic petals replicated with rotation transforms.
//! Outer petals track bass/mids, a counter-rotating inner layer tracks
//! highs, dashed rings pulse with overall energy.

use std::f32::consts::TAU;

use scry_engine::scene::style::{
    BlendMode, DashPattern, GradientDef, GradientKind, GradientStop, Point, Transform,
};
use scry_engine::scene::PixelCanvas;

use crate::analysis::AnalysisFrame;
use crate::theme::Theme;

const WEDGES: usize = 12;

/// Teardrop petal pointing up from `(cx, cy)`, tip at `len` above.
fn petal(cx: f32, cy: f32, len: f32) -> Option<tiny_skia::Path> {
    let side = len * 0.16;
    let inner = len * 0.10;
    let mut pb = tiny_skia::PathBuilder::new();
    pb.move_to(cx, cy - inner);
    pb.quad_to(cx - side, cy - len * 0.55, cx, cy - len);
    pb.quad_to(cx + side, cy - len * 0.55, cx, cy - inner);
    pb.close();
    pb.finish()
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
    let (w, h) = (w as f32, h as f32);
    let (cx, cy) = (w * 0.5, h * 0.5);
    let r_max = w.min(h) * 0.46 * (1.0 + 0.05 * s.beat.envelope);

    // Mirror-symmetric values so the pattern reads as a kaleidoscope:
    // wedge j and wedge WEDGES-1-j share a band group.
    let groups = super::group_bands(&s.bands, WEDGES / 2 + 1);
    let value =
        |j: usize| groups[(WEDGES / 2) - (j as i32 - WEDGES as i32 / 2).unsigned_abs() as usize];

    let rot = t * 0.10;
    for j in 0..WEDGES {
        let v = value(j);
        let len = r_max * (0.30 + 0.70 * v);
        let color = theme.sample(j as f32 / (WEDGES - 1) as f32);
        let angle = rot + j as f32 * TAU / WEDGES as f32;

        canvas = canvas
            .group(Transform::rotate_at(angle, cx, cy))
            .canvas(|c| {
                let mut c = c;
                if let Some(p) = petal(cx, cy, len) {
                    c = c
                        .path(p.clone())
                        .fill(color.with_alpha(0.30 + 0.35 * v))
                        .blend_mode(BlendMode::Screen)
                        .done()
                        .path(p)
                        .stroke(color.with_alpha(0.85), 1.2)
                        .done();
                }
                c.circle(cx, cy - len, 1.5 + 2.5 * v)
                    .fill(color.with_lightness(1.3).with_alpha(0.9))
                    .done()
            })
            .done();
    }

    // Inner counter-rotating layer driven by the top of the spectrum.
    let hi = super::group_bands(&s.bands[s.bands.len() / 2..], WEDGES / 2 + 1);
    let hi_value =
        |j: usize| hi[(WEDGES / 2) - (j as i32 - WEDGES as i32 / 2).unsigned_abs() as usize];
    for j in 0..WEDGES {
        let v = hi_value(j);
        let len = r_max * (0.12 + 0.30 * v);
        let color = theme.sample(1.0 - j as f32 / (WEDGES - 1) as f32);
        let angle = -rot * 1.6 + (j as f32 + 0.5) * TAU / WEDGES as f32;

        canvas = canvas
            .group(Transform::rotate_at(angle, cx, cy))
            .canvas(|c| {
                let mut c = c;
                if let Some(p) = petal(cx, cy, len) {
                    c = c
                        .path(p)
                        .fill(color.with_alpha(0.30 + 0.40 * v))
                        .blend_mode(BlendMode::Screen)
                        .done();
                }
                c
            })
            .done();
    }

    // Dashed rings pulsing with the mid bands, counter-rotating.
    let mids = super::group_bands(&s.bands, 3);
    for (i, (frac, dir)) in [(0.52f32, 1.0f32), (0.72, -1.0), (0.92, 1.0)]
        .iter()
        .enumerate()
    {
        let r = r_max * frac * (1.0 + 0.10 * mids[i]);
        let dash = r * TAU / (WEDGES * 2) as f32;
        canvas = canvas
            .group(Transform::rotate_at(rot * 2.0 * dir, cx, cy))
            .canvas(|c| {
                c.circle(cx, cy, r)
                    .stroke(
                        theme.sample(*frac).with_alpha(0.20 + 0.55 * mids[i]),
                        1.0 + 1.5 * mids[i],
                    )
                    .dash(DashPattern::pair(dash, dash * 0.6))
                    .done()
            })
            .done();
    }

    // Core glow.
    let core = r_max * 0.16 * (1.0 + 0.35 * s.beat.envelope + 0.20 * s.bass);
    canvas
        .circle(cx, cy, core)
        .fill_radial_gradient(GradientDef {
            kind: GradientKind::Radial {
                center: Point { x: cx, y: cy },
                radius: core,
            },
            stops: vec![
                GradientStop {
                    position: 0.0,
                    color: theme.accent.with_alpha(0.55),
                },
                GradientStop {
                    position: 0.6,
                    color: theme.accent.with_alpha(0.15),
                },
                GradientStop {
                    position: 1.0,
                    color: theme.accent.with_alpha(0.0),
                },
            ],
        })
        .blend_mode(BlendMode::Screen)
        .done()
        .circle(cx, cy, core * 0.18)
        .fill(theme.accent.with_lightness(1.5).with_alpha(0.9))
        .done()
}
