//! Nova: a bass-driven core with orbiting spark particles and beat
//! shockwaves.

use std::f32::consts::TAU;

use scry_engine::scene::style::{BlendMode, GradientDef, GradientKind, GradientStop, Point};
use scry_engine::scene::PixelCanvas;

use super::VizState;
use crate::analysis::AnalysisFrame;
use crate::theme::Theme;

const MAX_PARTICLES: usize = 450;
const WAVE_LIFE: f32 = 0.9;

pub(super) struct Particle {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    life: f32,
    max_life: f32,
    ci: f32,
}

fn spawn(st: &mut VizState, cx: f32, cy: f32, radius: f32, speed: f32) {
    let a = st.rng.f32() * TAU;
    let sp = speed * (0.6 + 0.8 * st.rng.f32());
    let life = 1.0 + 1.4 * st.rng.f32();
    st.particles.push(Particle {
        x: cx + a.cos() * radius,
        y: cy + a.sin() * radius,
        vx: a.cos() * sp - a.sin() * sp * 0.6,
        vy: a.sin() * sp + a.cos() * sp * 0.6,
        life,
        max_life: life,
        ci: st.rng.f32(),
    });
}

fn update(st: &mut VizState, s: &AnalysisFrame, dt: f32, cx: f32, cy: f32, scale: f32) {
    let treble = s.treble;
    let core_r = scale * 0.10;

    st.spawn_acc += (12.0 + 130.0 * treble) * dt;
    while st.spawn_acc >= 1.0 && st.particles.len() < MAX_PARTICLES {
        st.spawn_acc -= 1.0;
        spawn(st, cx, cy, core_r, scale * (0.10 + 0.35 * s.rms));
    }

    // Beat onset: shockwave plus a burst of fast sparks.
    if s.beat.onset > 0.5 && st.prev_beat <= 0.5 {
        st.wave_ages.push(0.0);
        for _ in 0..24 {
            if st.particles.len() < MAX_PARTICLES {
                spawn(st, cx, cy, core_r, scale * 0.65);
            }
        }
    }
    st.prev_beat = s.beat.onset;

    let swirl = 1.4 + 4.0 * s.bass;
    let drag = (-0.9 * dt).exp();
    st.particles.retain_mut(|p| {
        let dx = p.x - cx;
        let dy = p.y - cy;
        let dist = dx.hypot(dy).max(1.0);
        // Tangential swirl plus gentle outward push from the bass.
        p.vx += (-dy / dist * swirl * 60.0 + dx / dist * s.bass * 50.0) * dt;
        p.vy += (dx / dist * swirl * 60.0 + dy / dist * s.bass * 50.0) * dt;
        p.vx *= drag;
        p.vy *= drag;
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.life -= dt;
        p.life > 0.0
    });

    for age in &mut st.wave_ages {
        *age += dt;
    }
    st.wave_ages.retain(|&a| a < WAVE_LIFE);
}

pub(super) fn build(
    mut canvas: PixelCanvas,
    st: &mut VizState,
    w: u32,
    h: u32,
    s: &AnalysisFrame,
    theme: &Theme,
    dt: f32,
) -> PixelCanvas {
    let (w, h) = (w as f32, h as f32);
    let (cx, cy) = (w * 0.5, h * 0.5);
    let scale = w.min(h);
    update(st, s, dt, cx, cy, scale);

    // Shockwaves: eased expanding rings.
    for &age in &st.wave_ages {
        let f = age / WAVE_LIFE;
        let ease = 1.0 - (1.0 - f) * (1.0 - f);
        let r = scale * (0.08 + 0.45 * ease);
        let fade = 1.0 - f;
        canvas = canvas
            .circle(cx, cy, r)
            .stroke(theme.accent.with_alpha(0.55 * fade), 0.5 + 2.5 * fade)
            .blend_mode(BlendMode::Screen)
            .done()
            .circle(cx, cy, r * 0.93)
            .stroke(theme.sample(0.5).with_alpha(0.25 * fade), 1.0)
            .blend_mode(BlendMode::Screen)
            .done();
    }

    // Particles: motion trail, soft glow, bright core.
    for p in &st.particles {
        let fade = p.life / p.max_life;
        let color = theme.sample(p.ci);
        canvas = canvas
            .line(p.x - p.vx * 0.11, p.y - p.vy * 0.11, p.x, p.y)
            .stroke(color.with_alpha(0.45 * fade), 1.2)
            .done()
            .circle(p.x, p.y, 4.2)
            .fill(color.with_alpha(0.30 * fade))
            .blend_mode(BlendMode::Screen)
            .done()
            .circle(p.x, p.y, 1.3)
            .fill(color.with_lightness(1.3).with_alpha(0.95 * fade))
            .done();
    }

    // Core: radial glow breathing with the bass.
    let core = scale * 0.13 * (1.0 + 0.55 * s.bass + 0.25 * s.beat.envelope);
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
                    color: theme.accent.with_lightness(1.3).with_alpha(0.60),
                },
                GradientStop {
                    position: 0.45,
                    color: theme.accent.with_alpha(0.22),
                },
                GradientStop {
                    position: 1.0,
                    color: theme.accent.with_alpha(0.0),
                },
            ],
        })
        .blend_mode(BlendMode::Screen)
        .done()
        .circle(cx, cy, core * 0.16)
        .fill(theme.accent.with_lightness(1.6).with_alpha(0.9))
        .done()
}
