//! Classic spectrum bars with falling peak caps.

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
    let n = s.bands.len();
    let gap = (w / n as f32 * 0.22).clamp(1.0, 6.0);
    let bw = (w - gap * (n + 1) as f32) / n as f32;
    let max_h = h * 0.94;

    for (i, &v) in s.bands.iter().enumerate() {
        let t = i as f32 / (n - 1) as f32;
        let color = theme.sample(t);
        let x = gap + i as f32 * (bw + gap);
        let bh = (v * max_h).max(1.0);
        let y = h - bh;

        canvas = canvas
            .rect(x, y, bw, bh)
            .corner_radius((bw * 0.3).min(4.0))
            .fill_linear_gradient(GradientDef {
                kind: GradientKind::Linear {
                    start: Point { x, y },
                    end: Point { x, y: h },
                },
                stops: vec![
                    GradientStop {
                        position: 0.0,
                        color,
                    },
                    GradientStop {
                        position: 1.0,
                        color: color.with_lightness(0.45).with_alpha(0.8),
                    },
                ],
            })
            .done();

        let py = h - s.peaks[i] * max_h - 3.0;
        if py < y - 2.0 {
            canvas = canvas
                .rect(x, py, bw, 2.0)
                .fill(color.with_lightness(1.4))
                .done();
        }
    }

    canvas
}
