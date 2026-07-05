//! Spectrogram: scrolling frequency heat with a centroid trace and beat scan.

use scry_engine::scene::style::{BlendMode, GradientDef, GradientKind, GradientStop, Point};
use scry_engine::scene::PixelCanvas;

use super::VizState;
use crate::analysis::AnalysisFrame;
use crate::theme::Theme;

const ROWS: usize = 38;
const MAX_COLUMNS: usize = 76;
const PUSH_INTERVAL: f32 = 0.035;

fn push_column(st: &mut VizState, s: &AnalysisFrame, dt: f32) {
    st.spectro_acc += dt;
    if st.spectro_acc < PUSH_INTERVAL && !st.spectro_columns.is_empty() {
        return;
    }

    st.spectro_acc = 0.0;
    let mut column = super::group_bands(&s.bands, ROWS);
    for value in &mut column {
        *value = value.powf(0.68).clamp(0.0, 1.0);
    }

    st.spectro_columns.push_back(column);
    while st.spectro_columns.len() > MAX_COLUMNS {
        st.spectro_columns.pop_front();
    }
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
    push_column(st, s, dt);

    let (w, h) = (w.max(1) as f32, h.max(1) as f32);
    let cell_w = w / MAX_COLUMNS as f32;
    let row_h = h / ROWS as f32;
    let x0 = w - st.spectro_columns.len() as f32 * cell_w;

    canvas = canvas
        .rect(0.0, h * 0.68, w, h * 0.32)
        .fill_linear_gradient(GradientDef {
            kind: GradientKind::Linear {
                start: Point {
                    x: 0.0,
                    y: h * 0.68,
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
                    color: theme.accent.with_alpha(0.08 + 0.10 * s.bass),
                },
            ],
        })
        .blend_mode(BlendMode::Screen)
        .done();

    for (col_idx, column) in st.spectro_columns.iter().enumerate() {
        let age = col_idx as f32 / (MAX_COLUMNS - 1) as f32;
        let x = x0 + col_idx as f32 * cell_w;
        for (row_idx, &value) in column.iter().enumerate() {
            if value < 0.025 {
                continue;
            }

            let freq = row_idx as f32 / (ROWS - 1) as f32;
            let y = h - (row_idx + 1) as f32 * row_h;
            let alpha = (0.04 + 0.72 * value) * (0.22 + 0.78 * age);
            let color = theme
                .sample(0.08 + 0.86 * freq)
                .with_lightness(0.55 + 1.45 * value)
                .with_alpha(alpha);

            canvas = canvas
                .rect(
                    x + cell_w * 0.08,
                    y + row_h * 0.08,
                    (cell_w * 0.86).max(0.6),
                    (row_h * 0.84).max(0.6),
                )
                .fill(color)
                .blend_mode(BlendMode::Screen)
                .done();
        }
    }

    if let Some(centroid) = s.tonal.spectral_centroid {
        let y = h - centroid.clamp(0.0, 1.0) * h;
        canvas = canvas
            .line(0.0, y, w, y)
            .stroke(
                theme.accent.with_alpha(0.18 + 0.50 * s.treble),
                1.0 + 1.2 * s.treble,
            )
            .done();
    }

    let scan_x = (w - cell_w * 1.5).max(0.0);
    canvas
        .line(scan_x, 0.0, scan_x, h)
        .stroke(
            theme
                .accent
                .with_lightness(1.5)
                .with_alpha(0.18 + 0.55 * s.beat.envelope),
            1.0 + 2.0 * s.beat.envelope,
        )
        .done()
}
