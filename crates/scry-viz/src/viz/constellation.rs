//! Constellation: a reactive node field that forms and breaks musical chords.

use scry_engine::scene::style::BlendMode;
use scry_engine::scene::PixelCanvas;
use scry_engine::style::LineCap;

use super::VizState;
use crate::analysis::AnalysisFrame;
use crate::theme::Theme;

const NODE_COUNT: usize = 58;

pub(super) struct Node {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    seed: f32,
    pulse: f32,
}

fn init(st: &mut VizState) {
    if !st.constellation_nodes.is_empty() {
        return;
    }

    st.constellation_nodes.reserve(NODE_COUNT);
    for _ in 0..NODE_COUNT {
        let a = st.rng.f32() * std::f32::consts::TAU;
        let r = st.rng.f32().sqrt() * 0.42;
        st.constellation_nodes.push(Node {
            x: 0.5 + a.cos() * r,
            y: 0.5 + a.sin() * r,
            vx: (st.rng.f32() - 0.5) * 0.05,
            vy: (st.rng.f32() - 0.5) * 0.05,
            seed: st.rng.f32(),
            pulse: st.rng.f32() * 0.3,
        });
    }
}

fn update(st: &mut VizState, s: &AnalysisFrame, t: f32, dt: f32) {
    init(st);

    let beat_edge = s.beat.onset > 0.5 && st.constellation_prev_beat <= 0.5;
    st.constellation_prev_beat = s.beat.onset;
    if beat_edge {
        for node in &mut st.constellation_nodes {
            if st.rng.f32() < 0.42 + 0.28 * s.beat.confidence {
                node.pulse = 1.0;
                let dx = node.x - 0.5;
                let dy = node.y - 0.5;
                let d = dx.hypot(dy).max(0.01);
                node.vx += dx / d * (0.22 + 0.20 * s.bass);
                node.vy += dy / d * (0.22 + 0.20 * s.bass);
            }
        }
    }

    let swirl = 0.10 + 0.50 * s.high_mid + 0.20 * s.treble;
    let breathe = 0.05 + 0.20 * s.bass + 0.16 * s.beat.envelope;
    let drag = (-1.75 * dt).exp();

    for node in &mut st.constellation_nodes {
        let dx = node.x - 0.5;
        let dy = node.y - 0.5;
        let dist = dx.hypot(dy).max(0.015);
        let drift = (t * (0.7 + node.seed) + node.seed * 23.0).sin() * 0.018;

        node.vx += (-dy * swirl + dx / dist * breathe + drift) * dt;
        node.vy += (dx * swirl + dy / dist * breathe - drift * 0.7) * dt;
        node.vx *= drag;
        node.vy *= drag;
        node.x += node.vx * dt;
        node.y += node.vy * dt;
        node.pulse *= (-3.6 * dt).exp();

        let dx = node.x - 0.5;
        let dy = node.y - 0.5;
        let dist = dx.hypot(dy);
        if dist > 0.49 {
            let pull = (dist - 0.49) / dist;
            node.x -= dx * pull;
            node.y -= dy * pull;
            node.vx *= -0.25;
            node.vy *= -0.25;
        }
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
    update(st, s, t, dt);

    let (w, h) = (w.max(1) as f32, h.max(1) as f32);
    let scale = w.min(h);
    let threshold = scale * (0.145 + 0.075 * s.rms + 0.050 * s.beat.envelope);
    let energy = (s.low_mid * 0.45 + s.high_mid * 0.35 + s.treble * 0.20).clamp(0.0, 1.0);

    for i in 0..st.constellation_nodes.len() {
        let a = &st.constellation_nodes[i];
        let ax = a.x * w;
        let ay = a.y * h;
        for b in &st.constellation_nodes[i + 1..] {
            let bx = b.x * w;
            let by = b.y * h;
            let dist = (bx - ax).hypot(by - ay);
            if dist > threshold {
                continue;
            }

            let closeness = 1.0 - dist / threshold;
            let pulse = a.pulse.max(b.pulse);
            let color = theme
                .sample(((a.seed + b.seed) * 0.5 + 0.15 * s.treble).fract())
                .with_alpha((0.05 + 0.38 * closeness * energy + 0.24 * pulse).min(0.70));
            canvas = canvas
                .line(ax, ay, bx, by)
                .stroke(color, 0.45 + 1.25 * closeness + 0.85 * pulse)
                .line_cap(LineCap::Round)
                .done();
        }
    }

    for node in &st.constellation_nodes {
        let x = node.x * w;
        let y = node.y * h;
        let radius = scale * (0.0038 + 0.0060 * energy) + node.pulse * scale * 0.012;
        let color = theme.sample((node.seed + s.treble * 0.25).fract());
        canvas = canvas
            .circle(x, y, radius * 3.6)
            .fill(color.with_alpha(0.05 + 0.16 * node.pulse))
            .blend_mode(BlendMode::Screen)
            .done()
            .circle(x, y, radius)
            .fill(
                color
                    .with_lightness(1.25 + 0.55 * node.pulse)
                    .with_alpha(0.68 + 0.25 * energy),
            )
            .done();
    }

    canvas
}
