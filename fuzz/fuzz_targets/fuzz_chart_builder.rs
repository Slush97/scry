//! Fuzz target: Chart builder API robustness.
//!
//! Exercises the full builder chain for each chart type with arbitrary
//! combinations of builder methods. Catches panics in builder validation
//! or downstream rendering.

#![no_main]

use libfuzzer_sys::fuzz_target;
use pixelchart::chart::{Chart, LineChart};
use pixelchart::data::Series;
use pixelchart::layout;
use pixelchart::prelude::Marker;
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

fuzz_target!(|data: &[u8]| {
    if data.len() < 32 {
        return;
    }

    let builder_type = data[0] % 5;
    let flags = data[1];
    let n_series = ((data[2] % 4) + 1) as usize;
    let n_points = ((data[3] % 15) + 2) as usize;

    // Feature flags
    let add_title = flags & 0x01 != 0;
    let add_x_label = flags & 0x02 != 0;
    let add_y_label = flags & 0x04 != 0;
    let add_x_range = flags & 0x08 != 0;
    let add_y_range = flags & 0x10 != 0;
    let add_h_line = flags & 0x20 != 0;
    let add_v_line = flags & 0x40 != 0;
    let add_annotation = flags & 0x80 != 0;

    let theme_idx = data[4] % 3;
    let theme = match theme_idx {
        0 => Theme::dark(),
        1 => Theme::light(),
        _ => Theme::pastel(),
    };

    let offset = 5;

    let chart = match builder_type {
        0 => {
            // Scatter builder with every option
            let x: Vec<f64> = (0..n_points)
                .map(|i| fuzz_f64(data, offset + i * 8))
                .collect();
            let y: Vec<f64> = (0..n_points)
                .map(|i| fuzz_f64(data, offset + n_points * 8 + i * 8))
                .collect();

            let mut b = Chart::scatter(&x, &y).theme(theme);

            if add_title {
                b = b.title("Fuzz Scatter");
            }
            if add_x_label {
                b = b.x_label("X Axis");
            }
            if add_y_label {
                b = b.y_label("Y Axis");
            }
            if add_x_range {
                let r0 = fuzz_f64(data, offset + n_points * 16);
                let r1 = fuzz_f64(data, offset + n_points * 16 + 8);
                b = b.x_range(r0, r1);
            }
            if add_y_range {
                let r0 = fuzz_f64(data, offset + n_points * 16 + 16);
                let r1 = fuzz_f64(data, offset + n_points * 16 + 24);
                b = b.y_range(r0, r1);
            }
            if add_h_line {
                b = b.h_line(fuzz_f64(data, 20));
            }
            if add_v_line {
                b = b.v_line(fuzz_f64(data, 28));
            }
            if add_annotation {
                b = b.annotate(
                    fuzz_f64(data, 20),
                    fuzz_f64(data, 28),
                    "FuzzAnnotation",
                );
            }

            // Marker variants
            let marker = match data[4] >> 4 {
                0 => Marker::Circle,
                1 => Marker::Square,
                2 => Marker::Diamond,
                3 => Marker::Cross,
                _ => Marker::Triangle,
            };
            b = b.marker(marker);

            // Connected + trend
            if data[3] & 0x80 != 0 {
                b = b.connected();
            }
            if data[3] & 0x40 != 0 {
                b = b.trend_line();
            }

            // Extra series
            for s in 1..n_series {
                let sx: Vec<f64> = (0..n_points)
                    .map(|i| fuzz_f64(data, offset + (s * 2) * n_points * 8 + i * 8))
                    .collect();
                let sy: Vec<f64> = (0..n_points)
                    .map(|i| fuzz_f64(data, offset + (s * 2 + 1) * n_points * 8 + i * 8))
                    .collect();
                b = b.add_series(
                    Series::new(format!("S{s}"), sx),
                    Series::new(format!("S{s}Y"), sy),
                );
            }

            b.build()
        }
        1 => {
            // Line builder — multi-series with all options
            let series: Vec<Series> = (0..n_series)
                .map(|s| {
                    let vals: Vec<f64> = (0..n_points)
                        .map(|i| fuzz_f64(data, offset + (s * n_points + i) * 8))
                        .collect();
                    Series::new(format!("Line{s}"), vals)
                })
                .collect();

            let mut b = LineChart::new(series).theme(theme);

            if add_title {
                b = b.title("Fuzz Line");
            }
            if add_x_label {
                b = b.x_label("Time");
            }
            if add_y_label {
                b = b.y_label("Value");
            }
            if data[2] & 0x80 != 0 {
                b = b.filled();
            }
            if data[2] & 0x40 != 0 {
                b = b.with_points();
            }
            if add_h_line {
                b = b.h_line(fuzz_f64(data, 20));
            }
            if add_v_line {
                b = b.v_line(fuzz_f64(data, 28));
            }
            if add_annotation {
                b = b.annotate(
                    fuzz_f64(data, 20),
                    fuzz_f64(data, 28),
                    "FuzzNote",
                );
            }
            if data[2] & 0x20 != 0 {
                b = b.trend_line();
            }

            b.build()
        }
        2 => {
            // Bar builder — stacked, horizontal, grouped
            let labels: Vec<String> = (0..n_points.min(10))
                .map(|i| format!("Cat{i}"))
                .collect();
            let vals: Vec<f64> = (0..labels.len())
                .map(|i| fuzz_f64(data, offset + i * 8))
                .collect();

            let mut b = Chart::bar(labels, &vals).theme(theme);

            if add_title {
                b = b.title("Fuzz Bar");
            }
            if add_x_label {
                b = b.x_label("Category");
            }
            if add_y_label {
                b = b.y_label("Amount");
            }
            if data[2] & 0x80 != 0 {
                b = b.stacked();
            }
            if data[2] & 0x40 != 0 {
                b = b.horizontal();
            }
            if add_h_line {
                b = b.h_line(fuzz_f64(data, 20));
            }

            // Extra series
            for s in 1..n_series {
                let sv: Vec<f64> = (0..n_points.min(10))
                    .map(|i| fuzz_f64(data, offset + (s * 10 + i) * 8))
                    .collect();
                b = b.add_series(Series::new(format!("Extra{s}"), sv));
            }

            b.build()
        }
        3 => {
            // Histogram builder
            let vals: Vec<f64> = (0..n_points * 5)
                .map(|i| fuzz_f64(data, offset + (i % n_points) * 8))
                .collect();

            let mut b = Chart::histogram(&vals).theme(theme);

            if add_title {
                b = b.title("Fuzz Hist");
            }
            let bins = ((data[3] >> 4) % 15 + 2) as usize;
            b = b.bins(bins);
            if data[2] & 0x80 != 0 {
                b = b.density();
            }
            if add_h_line {
                b = b.h_line(fuzz_f64(data, 20));
            }

            b.build()
        }
        _ => {
            // Heatmap builder
            let rows = (n_points % 10 + 1).min(8);
            let cols = ((data[4] >> 2) % 8 + 1) as usize;
            let grid: Vec<Vec<f64>> = (0..rows)
                .map(|r| {
                    (0..cols)
                        .map(|c| fuzz_f64(data, offset + (r * cols + c) * 8))
                        .collect()
                })
                .collect();

            let mut b = Chart::heatmap(grid).theme(theme);
            if add_title {
                b = b.title("Fuzz Heat");
            }
            if data[2] & 0x80 != 0 {
                b = b.values(false);
            }

            b.build()
        }
    };

    // Render at multiple sizes to stress layout
    let sizes = [(200, 150), (50, 30), (800, 600)];
    for (w, h) in sizes {
        let rendered = layout::render_chart(&chart, w, h);

        assert_eq!(rendered.canvas.width(), w);
        assert_eq!(rendered.canvas.height(), h);

        // Overlay coordinates may be non-finite for degenerate data, which is harmless.
        let _ = &rendered.text_overlays;
    }
});
