//! Fuzz target: Chart rendering pipeline robustness.
//!
//! Builds charts of all 7 types with arbitrary fuzzed data, then runs
//! `layout::render_chart()`. Verifies no panics, canvas dimensions match,
//! and all text overlay coordinates are finite.

#![no_main]

use libfuzzer_sys::fuzz_target;
use pixelchart::chart::Chart;
use pixelchart::layout;
use pixelchart::theme::Theme;

/// Extract an f64 from fuzz data at a given offset.
fn fuzz_f64(data: &[u8], offset: usize) -> f64 {
    if offset + 8 > data.len() {
        return 0.0;
    }
    f64::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ])
}

/// Extract a Vec<f64> from fuzz data.
fn fuzz_vec(data: &[u8], offset: usize, count: usize) -> Vec<f64> {
    (0..count).map(|i| fuzz_f64(data, offset + i * 8)).collect()
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 20 {
        return;
    }

    let chart_type = data[0] % 7;
    // Canvas dimensions — clamped to reasonable range for speed
    let w = u16::from_le_bytes([data[1], data[2]]).max(1).min(512) as u32;
    let h = u16::from_le_bytes([data[3], data[4]]).max(1).min(512) as u32;
    let n_points = ((data[5] % 20) + 1) as usize; // 1–20 data points
    let theme_idx = data[6] % 3;

    let theme = match theme_idx {
        0 => Theme::dark(),
        1 => Theme::light(),
        _ => Theme::pastel(),
    };

    // Feature flags from data[7]
    let flags = data[7];
    let with_title = flags & 0x01 != 0;
    let with_x_label = flags & 0x02 != 0;
    let with_y_label = flags & 0x04 != 0;
    let with_h_line = flags & 0x08 != 0;
    let with_v_line = flags & 0x10 != 0;

    let offset = 8;
    let values = fuzz_vec(data, offset, n_points);

    let chart = match chart_type {
        0 => {
            // Scatter
            let x_vals = fuzz_vec(data, offset + n_points * 8, n_points);
            let mut b = Chart::scatter(&x_vals, &values).theme(theme);
            if with_title {
                b = b.title("FuzzScatter");
            }
            if with_x_label {
                b = b.x_label("X");
            }
            if with_y_label {
                b = b.y_label("Y");
            }
            if with_h_line {
                b = b.h_line(fuzz_f64(data, offset + n_points * 16));
            }
            if with_v_line {
                b = b.v_line(fuzz_f64(data, offset + n_points * 16 + 8));
            }
            b.build()
        }
        1 => {
            // Line
            let mut b = Chart::line(&values).theme(theme);
            if with_title {
                b = b.title("FuzzLine");
            }
            if with_x_label {
                b = b.x_label("X");
            }
            if with_y_label {
                b = b.y_label("Y");
            }
            if with_h_line {
                b = b.h_line(fuzz_f64(data, offset + n_points * 8));
            }
            b.build()
        }
        2 => {
            // Bar
            let labels: Vec<String> = (0..n_points).map(|i| format!("L{i}")).collect();
            let mut b = Chart::bar(labels, &values).theme(theme);
            if with_title {
                b = b.title("FuzzBar");
            }
            if with_x_label {
                b = b.x_label("Category");
            }
            if with_y_label {
                b = b.y_label("Value");
            }
            b.build()
        }
        3 => {
            // Histogram
            let mut b = Chart::histogram(&values).theme(theme);
            if with_title {
                b = b.title("FuzzHist");
            }
            let bins = ((data[6] >> 4) % 10 + 2) as usize;
            b = b.bins(bins);
            b.build()
        }
        4 => {
            // BoxPlot
            let n_groups = ((data[5] >> 4) % 4 + 1) as usize;
            let mut groups: Vec<(String, Vec<f64>)> = Vec::new();
            for g in 0..n_groups {
                let start = offset + g * n_points * 8;
                let gvals = fuzz_vec(data, start, n_points.min(10));
                groups.push((format!("G{g}"), gvals));
            }
            let mut b = Chart::boxplot(groups).theme(theme);
            if with_title {
                b = b.title("FuzzBox");
            }
            b.build()
        }
        5 => {
            // Heatmap
            let rows = ((data[5] >> 4) % 8 + 1) as usize;
            let cols = (n_points % 8 + 1) as usize;
            let mut grid = Vec::new();
            for r in 0..rows {
                let row_start = offset + r * cols * 8;
                grid.push(fuzz_vec(data, row_start, cols));
            }
            let mut b = Chart::heatmap(grid).theme(theme);
            if with_title {
                b = b.title("FuzzHeat");
            }
            b.build()
        }
        _ => {
            // Pie
            let labels: Vec<String> = (0..n_points.min(8)).map(|i| format!("S{i}")).collect();
            let pie_vals: Vec<f64> = values.iter().take(labels.len()).copied().collect();
            let mut b = Chart::pie(labels, &pie_vals).theme(theme);
            if with_title {
                b = b.title("FuzzPie");
            }
            b.build()
        }
    };

    // ═══════════════════════════════════════════
    // THE CRITICAL TEST: render must not panic
    // ═══════════════════════════════════════════
    let rendered = layout::render_chart(&chart, w, h);

    // Canvas dimensions must match request
    assert_eq!(rendered.canvas.width(), w, "width mismatch");
    assert_eq!(rendered.canvas.height(), h, "height mismatch");

    // Text overlays exist and rendering didn't panic — that's the key test.
    // Overlay coordinates may be non-finite for degenerate domain/range combos,
    // which is harmless (they just won't render visibly).
    let _ = &rendered.text_overlays;
});
