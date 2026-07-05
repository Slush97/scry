//! Visual modes: each builds a `PixelCanvas` from one analysis frame.
//!
//! All modes render on a transparent background so the terminal shows
//! through (kitty transmits the alpha channel).

mod bars;
mod constellation;
mod mandala;
mod nova;
mod prism;
mod radial;
mod ridge;
mod silk;
mod spectrogram;
mod vortex;
mod wave;

use std::collections::VecDeque;

use clap::ValueEnum;
use scry_engine::scene::PixelCanvas;

use crate::analysis::AnalysisFrame;
use crate::theme::Theme;

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Mode {
    Silk,
    Ridge,
    Mandala,
    Nova,
    Bars,
    Radial,
    Wave,
    Spectrogram,
    Vortex,
    Constellation,
    Prism,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Silk => "silk",
            Self::Ridge => "ridge",
            Self::Mandala => "mandala",
            Self::Nova => "nova",
            Self::Bars => "bars",
            Self::Radial => "radial",
            Self::Wave => "wave",
            Self::Spectrogram => "spectrogram",
            Self::Vortex => "vortex",
            Self::Constellation => "constellation",
            Self::Prism => "prism",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Silk => Self::Ridge,
            Self::Ridge => Self::Mandala,
            Self::Mandala => Self::Nova,
            Self::Nova => Self::Bars,
            Self::Bars => Self::Radial,
            Self::Radial => Self::Wave,
            Self::Wave => Self::Spectrogram,
            Self::Spectrogram => Self::Vortex,
            Self::Vortex => Self::Constellation,
            Self::Constellation => Self::Prism,
            Self::Prism => Self::Silk,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build(
        self,
        st: &mut VizState,
        w: u32,
        h: u32,
        s: &AnalysisFrame,
        theme: &Theme,
        t: f32,
        dt: f32,
    ) -> PixelCanvas {
        let canvas = PixelCanvas::new(w, h);
        match self {
            Self::Silk => silk::build(canvas, w, h, s, theme),
            Self::Ridge => ridge::build(canvas, st, w, h, s, theme, dt),
            Self::Mandala => mandala::build(canvas, w, h, s, theme, t),
            Self::Nova => nova::build(canvas, st, w, h, s, theme, dt),
            Self::Bars => bars::build(canvas, w, h, s, theme),
            Self::Radial => radial::build(canvas, w, h, s, theme, t),
            Self::Wave => wave::build(canvas, w, h, s, theme),
            Self::Spectrogram => spectrogram::build(canvas, st, w, h, s, theme, dt),
            Self::Vortex => vortex::build(canvas, st, w, h, s, theme, t, dt),
            Self::Constellation => constellation::build(canvas, st, w, h, s, theme, t, dt),
            Self::Prism => prism::build(canvas, w, h, s, theme, t),
        }
    }
}

/// Per-mode mutable state (band history, particle systems).
pub struct VizState {
    ridge_rows: VecDeque<Vec<f32>>,
    ridge_acc: f32,
    particles: Vec<nova::Particle>,
    wave_ages: Vec<f32>,
    spectro_columns: VecDeque<Vec<f32>>,
    spectro_acc: f32,
    vortex_rows: VecDeque<Vec<f32>>,
    vortex_acc: f32,
    constellation_nodes: Vec<constellation::Node>,
    constellation_prev_beat: f32,
    spawn_acc: f32,
    prev_beat: f32,
    rng: fastrand::Rng,
}

impl VizState {
    pub fn new() -> Self {
        Self {
            ridge_rows: VecDeque::new(),
            ridge_acc: 0.0,
            particles: Vec::new(),
            wave_ages: Vec::new(),
            spectro_columns: VecDeque::new(),
            spectro_acc: 0.0,
            vortex_rows: VecDeque::new(),
            vortex_acc: 0.0,
            constellation_nodes: Vec::new(),
            constellation_prev_beat: 0.0,
            spawn_acc: 0.0,
            prev_beat: 0.0,
            rng: fastrand::Rng::with_seed(42),
        }
    }
}

/// Catmull-Rom spline through `pts`, optionally closed down to a baseline
/// (for area fills). Returns `None` for degenerate input.
fn spline(pts: &[(f32, f32)], close_to_y: Option<f32>) -> Option<tiny_skia::Path> {
    if pts.len() < 2 {
        return None;
    }
    let mut pb = tiny_skia::PathBuilder::new();
    pb.move_to(pts[0].0, pts[0].1);
    for i in 0..pts.len() - 1 {
        let p0 = pts[i.saturating_sub(1)];
        let p1 = pts[i];
        let p2 = pts[i + 1];
        let p3 = pts[(i + 2).min(pts.len() - 1)];
        pb.cubic_to(
            p1.0 + (p2.0 - p0.0) / 6.0,
            p1.1 + (p2.1 - p0.1) / 6.0,
            p2.0 - (p3.0 - p1.0) / 6.0,
            p2.1 - (p3.1 - p1.1) / 6.0,
            p2.0,
            p2.1,
        );
    }
    if let Some(y) = close_to_y {
        pb.line_to(pts[pts.len() - 1].0, y);
        pb.line_to(pts[0].0, y);
        pb.close();
    }
    pb.finish()
}

/// Average `bands` into `k` equal groups.
fn group_bands(bands: &[f32], k: usize) -> Vec<f32> {
    (0..k)
        .map(|i| {
            let a = i * bands.len() / k;
            let b = (((i + 1) * bands.len() / k).max(a + 1)).min(bands.len());
            bands[a..b].iter().sum::<f32>() / (b - a) as f32
        })
        .collect()
}

/// 3-tap spatial smoothing to soften band-to-band jaggies.
fn smooth3(values: &[f32]) -> Vec<f32> {
    let n = values.len();
    (0..n)
        .map(|i| {
            let l = values[i.saturating_sub(1)];
            let r = values[(i + 1).min(n - 1)];
            0.25 * l + 0.5 * values[i] + 0.25 * r
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::SAMPLE_RATE;
    use crate::dsp::WAVEFORM_LEN;

    #[test]
    fn every_mode_builds_from_fixed_analysis_frame() {
        let frame = AnalysisFrame::fixture(64, WAVEFORM_LEN, SAMPLE_RATE);

        for mode in [
            Mode::Silk,
            Mode::Ridge,
            Mode::Mandala,
            Mode::Nova,
            Mode::Bars,
            Mode::Radial,
            Mode::Wave,
            Mode::Spectrogram,
            Mode::Vortex,
            Mode::Constellation,
            Mode::Prism,
        ] {
            let mut state = VizState::new();
            let canvas = mode.build(
                &mut state,
                640,
                360,
                &frame,
                &crate::theme::THEMES[0],
                1.25,
                1.0 / 60.0,
            );

            assert_eq!(canvas.width(), 640, "mode {}", mode.name());
            assert_eq!(canvas.height(), 360, "mode {}", mode.name());
            assert!(canvas.command_count() > 0, "mode {}", mode.name());
        }
    }
}
