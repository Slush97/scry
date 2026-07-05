//! Ridge: spectrum-history waterfall in the Unknown Pleasures style.
//! Rows scroll toward the viewer; fills occlude the rows behind them.

use scry_engine::scene::PixelCanvas;

use super::VizState;
use crate::analysis::AnalysisFrame;
use crate::theme::Theme;

const ROWS: usize = 26;
const POINTS: usize = 56;
const PUSH_INTERVAL: f32 = 0.05;

pub(super) fn build(
    mut canvas: PixelCanvas,
    st: &mut VizState,
    w: u32,
    h: u32,
    s: &AnalysisFrame,
    theme: &Theme,
    dt: f32,
) -> PixelCanvas {
    st.ridge_acc += dt;
    if st.ridge_acc >= PUSH_INTERVAL || st.ridge_rows.is_empty() {
        st.ridge_acc = 0.0;
        // Resample bands to a fixed width with center emphasis so rows
        // peak in the middle like the original pulsar plot.
        let bands = super::smooth3(&s.bands);
        let row: Vec<f32> = (0..POINTS)
            .map(|i| {
                let f = i as f32 / (POINTS - 1) as f32;
                let idx = f * (bands.len() - 1) as f32;
                let a = idx as usize;
                let v = bands[a] + (bands[(a + 1).min(bands.len() - 1)] - bands[a]) * idx.fract();
                let center = (std::f32::consts::PI * f).sin();
                v * (0.25 + 0.75 * center * center)
            })
            .collect();
        st.ridge_rows.push_back(row);
        if st.ridge_rows.len() > ROWS {
            st.ridge_rows.pop_front();
        }
    }

    let (w, h) = (w as f32, h as f32);
    let margin = w * 0.10;
    let span = w - 2.0 * margin;

    let count = st.ridge_rows.len();
    for (idx, row) in st.ridge_rows.iter().enumerate() {
        // Oldest row at the top/back, newest at the bottom/front.
        let d = (idx + ROWS - count) as f32 / (ROWS - 1) as f32;
        let y_base = h * (0.14 + 0.76 * d);
        let amp = h * 0.30 * (0.25 + 0.75 * d);

        let pts: Vec<(f32, f32)> = row
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let x = margin + i as f32 / (POINTS - 1) as f32 * span;
                (x, y_base - v * amp)
            })
            .collect();

        let Some(fill) = super::spline(&pts, Some(y_base + 1.0)) else {
            continue;
        };
        let Some(stroke) = super::spline(&pts, None) else {
            continue;
        };
        let brightness = 0.30 + 0.70 * d * d;
        canvas = canvas
            .path(fill)
            .fill(theme.bg.with_alpha(0.93))
            .done()
            .path(stroke)
            .stroke(
                theme.sample(0.15 + 0.70 * d).with_alpha(brightness),
                1.0 + 0.9 * d,
            )
            .done();
    }

    canvas
}
