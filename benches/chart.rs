//! Investor-grade chart benchmarks for `scry-chart`.
//!
//! Exercises **every dimension** of the charting library:
//!
//! | Group | Dimension |
//! |-------|-----------|
//! | 1 | All 9 chart types — build + PNG render |
//! | 2 | Multi-series scaling (1 → 20 series) |
//! | 3 | Data-volume scaling (100 → 100 000 pts) |
//! | 4 | All 5 themes on the same chart |
//! | 5 | Triple export path (PNG, SVG, RGBA) |
//! | 6 | Resolution scaling (400×300 → 3840×2160) |
//! | 7 | Kitchen-sink builder stress test |
//! | 8 | Financial dashboard workload |
//! | 9 | Correlation heatmap (science/ML workload) |
//! | 10 | Isolated layout engine |
//!
//! Run with: `cargo bench --bench chart`

#![allow(missing_docs)]


use std::time::Duration as StdDuration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use scry_chart::chart::{
    BarChart, BoxPlot, CandlestickChart, Chart, Heatmap, Histogram, LineChart, OhlcEntry,
    PieChart, RadarChart, ScatterChart,
};
use scry_chart::data::Series;
use scry_chart::export::{render_to_png, render_to_rgba};
use scry_chart::formatter::{CurrencyFormatter, SiFormatter};
use scry_chart::svg_export::render_to_svg;
use scry_chart::theme::Theme;

// ─────────────────────────────────────────────────────────────────
// Deterministic data generators
// ─────────────────────────────────────────────────────────────────

fn gen_f64(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = i as f64 * 0.01;
            (x * 3.7).sin() * 50.0 + x.cos() * 30.0 + (x * 0.1).powi(2)
        })
        .collect()
}

fn gen_xy(n: usize) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y = gen_f64(n);
    (x, y)
}

fn gen_categories(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("Cat-{}", i + 1)).collect()
}

fn gen_ohlc(n: usize) -> Vec<OhlcEntry> {
    let mut price = 100.0;
    (0..n)
        .map(|i| {
            let open = price;
            let delta = (i as f64 * 0.37).sin() * 5.0;
            let close = open + delta;
            let high = open.max(close) + (i as f64 * 0.53).sin().abs() * 3.0;
            let low = open.min(close) - (i as f64 * 0.71).cos().abs() * 3.0;
            price = close;
            OhlcEntry::new(i as f64, open, high, low, close)
        })
        .collect()
}

fn gen_heatmap(rows: usize, cols: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| {
                    let x = r as f64 / rows as f64;
                    let y = c as f64 / cols as f64;
                    (x * 3.0).sin() * (y * 2.0).cos() + (x * y * 5.0).sin() * 0.5
                })
                .collect()
        })
        .collect()
}

fn gen_correlation(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|r| {
            (0..n)
                .map(|c| {
                    if r == c {
                        1.0
                    } else {
                        let v = ((r * 7 + c * 13) as f64 * 0.1).sin();
                        (v * 100.0).round() / 100.0
                    }
                })
                .collect()
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────

const W: u32 = 800;
const H: u32 = 500;

// ─────────────────────────────────────────────────────────────────
// 1. All Nine Chart Types — build + PNG
// ─────────────────────────────────────────────────────────────────

fn bench_all_chart_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("all_chart_types");

    let (x, y) = gen_xy(500);
    let labels = gen_categories(10);
    let vals: Vec<f64> = (0..10).map(|i| (i as f64 + 1.0) * 12.5).collect();
    let raw = gen_f64(2000);
    let ohlc = gen_ohlc(60);
    let hmap = gen_heatmap(8, 10);
    let radar_axes: Vec<String> = vec![
        "Speed", "Power", "Range", "Accuracy", "Efficiency", "Stealth",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    // --- Scatter ---
    let scatter = Chart::scatter(&x[..500], &y[..500])
        .title("Scatter")
        .theme(Theme::dark())
        .build();
    group.bench_function("scatter_500pts", |b| {
        b.iter(|| render_to_png(black_box(&scatter), W, H).unwrap());
    });

    // --- Line ---
    let line = Chart::line(&y[..500]).title("Line").theme(Theme::dark()).build();
    group.bench_function("line_500pts", |b| {
        b.iter(|| render_to_png(black_box(&line), W, H).unwrap());
    });

    // --- Bar ---
    let bar = Chart::bar(labels.clone(), &vals)
        .title("Bar")
        .theme(Theme::dark())
        .build();
    group.bench_function("bar_10cat", |b| {
        b.iter(|| render_to_png(black_box(&bar), W, H).unwrap());
    });

    // --- Histogram ---
    let hist = Chart::histogram(&raw)
        .title("Histogram")
        .theme(Theme::dark())
        .build();
    group.bench_function("histogram_2k", |b| {
        b.iter(|| render_to_png(black_box(&hist), W, H).unwrap());
    });

    // --- BoxPlot ---
    let boxplot_data: Vec<(String, Vec<f64>)> = (0..5)
        .map(|i| {
            let vals: Vec<f64> = (0..100)
                .map(|j| (j as f64 * 0.1 + i as f64).sin() * 20.0 + i as f64 * 10.0)
                .collect();
            (format!("Group {}", i + 1), vals)
        })
        .collect();
    let bp = Chart::boxplot(boxplot_data)
        .title("Box Plot")
        .theme(Theme::dark())
        .build();
    group.bench_function("boxplot_5grp", |b| {
        b.iter(|| render_to_png(black_box(&bp), W, H).unwrap());
    });

    // --- Heatmap ---
    let hm = Chart::heatmap(hmap.clone())
        .title("Heatmap")
        .theme(Theme::dark())
        .build();
    group.bench_function("heatmap_8x10", |b| {
        b.iter(|| render_to_png(black_box(&hm), W, H).unwrap());
    });

    // --- Pie ---
    let pie = Chart::pie(labels.clone(), &vals)
        .title("Pie")
        .theme(Theme::dark())
        .build();
    group.bench_function("pie_10slc", |b| {
        b.iter(|| render_to_png(black_box(&pie), W, H).unwrap());
    });

    // --- Candlestick ---
    let candle = Chart::candlestick(ohlc.clone())
        .title("Candlestick")
        .theme(Theme::dark())
        .build();
    group.bench_function("candlestick_60", |b| {
        b.iter(|| render_to_png(black_box(&candle), W, H).unwrap());
    });

    // --- Radar ---
    let radar = Chart::radar(radar_axes.clone())
        .add_series("Alpha", &[0.8, 0.6, 0.9, 0.4, 0.7, 0.5])
        .add_series("Beta", &[0.5, 0.9, 0.3, 0.8, 0.6, 0.7])
        .title("Radar")
        .theme(Theme::dark())
        .build();
    group.bench_function("radar_2ser", |b| {
        b.iter(|| render_to_png(black_box(&radar), W, H).unwrap());
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// 2. Multi-Series Scaling (1 → 20 series)
// ─────────────────────────────────────────────────────────────────

fn bench_multi_series(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_series_scaling");

    for &n_series in &[1usize, 5, 10, 20] {
        let mut builder = Chart::line(&gen_f64(1000))
            .title("Multi-Series")
            .theme(Theme::dark());

        for i in 1..n_series {
            let data: Vec<f64> = gen_f64(1000)
                .iter()
                .map(|v| v + (i as f64) * 20.0)
                .collect();
            builder = builder.add_named_series(format!("S{}", i + 1), &data);
        }
        let chart = builder.build();

        group.bench_with_input(
            BenchmarkId::new("line_1k_pts", format!("{n_series}_series")),
            &chart,
            |b, chart| {
                b.iter(|| render_to_png(black_box(chart), W, H).unwrap());
            },
        );
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// 3. Data-Volume Scaling (100 → 100k data points)
// ─────────────────────────────────────────────────────────────────

fn bench_data_volume(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_volume_scaling");

    for &n in &[100usize, 1_000, 10_000, 100_000] {
        let (x, y) = gen_xy(n);
        let chart = Chart::scatter(&x, &y)
            .title("Volume Stress")
            .theme(Theme::dark())
            .build();

        group.bench_with_input(
            BenchmarkId::new("scatter", format!("{n}_pts")),
            &chart,
            |b, chart| {
                b.iter(|| render_to_png(black_box(chart), W, H).unwrap());
            },
        );
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// 4. All Five Themes
// ─────────────────────────────────────────────────────────────────

fn bench_all_themes(c: &mut Criterion) {
    let mut group = c.benchmark_group("theme_comparison");
    let y = gen_f64(500);

    let themes: Vec<(&str, Theme)> = vec![
        ("dark", Theme::dark()),
        ("light", Theme::light()),
        ("pastel", Theme::pastel()),
        ("ocean", Theme::ocean()),
        ("forest", Theme::forest()),
    ];

    for (name, theme) in &themes {
        let chart = Chart::line(&y)
            .title("Theme Bench")
            .theme(theme.clone())
            .build();

        group.bench_with_input(BenchmarkId::new("line_500", *name), &chart, |b, chart| {
            b.iter(|| render_to_png(black_box(chart), W, H).unwrap());
        });
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// 5. Triple Export Path (PNG, SVG, RGBA)
// ─────────────────────────────────────────────────────────────────

fn bench_export_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("export_paths");

    let (x, y) = gen_xy(500);
    let chart = Chart::scatter(&x, &y)
        .title("Export Bench")
        .theme(Theme::dark())
        .build();

    group.bench_function("png_800x500", |b| {
        b.iter(|| render_to_png(black_box(&chart), W, H).unwrap());
    });

    group.bench_function("svg_800x500", |b| {
        b.iter(|| {
            let svg = render_to_svg(black_box(&chart), W, H);
            black_box(svg);
        });
    });

    group.bench_function("rgba_800x500", |b| {
        b.iter(|| render_to_rgba(black_box(&chart), W, H).unwrap());
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// 6. Resolution Scaling (400×300 → 3840×2160)
// ─────────────────────────────────────────────────────────────────

fn bench_resolution_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("resolution_scaling");

    let y = gen_f64(1000);
    let chart = Chart::line(&y)
        .title("Res Scaling")
        .theme(Theme::dark())
        .with_points()
        .build();

    for &(w, h) in &[(400u32, 300u32), (800, 500), (1280, 720), (1920, 1080), (3840, 2160)] {
        group.bench_with_input(
            BenchmarkId::new("line_1k", format!("{w}x{h}")),
            &(w, h),
            |b, &(w, h)| {
                b.iter(|| render_to_png(black_box(&chart), w, h).unwrap());
            },
        );
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// 7. Kitchen-Sink Builder Stress Test
// ─────────────────────────────────────────────────────────────────

fn bench_kitchen_sink(c: &mut Criterion) {
    let mut group = c.benchmark_group("kitchen_sink");
    let y = gen_f64(500);

    let chart = Chart::line(&y)
        .title("Kitchen Sink")
        .x_label("Time (s)")
        .y_label("Value ($)")
        .theme(Theme::dark())
        .smooth()
        .filled()
        .with_points()
        .stacked()
        .dash_lines()
        .show_values()
        .x_ticks_diagonal()
        .y_grid_only()
        .legend_title("Series")
        .legend_outside_right()
        .h_line(25.0)
        .h_line_styled(50.0, scry_engine::style::Color::from_rgba8(255, 0, 0, 128))
        .annotate(250.0, 40.0, "Peak")
        .trend_line()
        .x_formatter(SiFormatter::default())
        .y_formatter(CurrencyFormatter::default())
        .add_named_series("Revenue", &gen_f64(500))
        .add_named_series("Cost", &gen_f64(500))
        .build();

    group.bench_function("line_all_features", |b| {
        b.iter(|| render_to_png(black_box(&chart), W, H).unwrap());
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// 8. Financial Dashboard Workload
// ─────────────────────────────────────────────────────────────────

fn bench_financial_dashboard(c: &mut Criterion) {
    let mut group = c.benchmark_group("financial_dashboard");

    let ohlc = gen_ohlc(200);
    let chart = Chart::candlestick(ohlc)
        .title("AAPL — 200 Day")
        .x_label("Day")
        .y_label("Price ($)")
        .theme(Theme::dark())
        .h_line(100.0)
        .h_line_styled(120.0, scry_engine::style::Color::from_rgba8(255, 165, 0, 180))
        .annotate(50.0, 115.0, "Resistance")
        .annotate(150.0, 85.0, "Support")
        .y_formatter(CurrencyFormatter::default())
        .build();

    group.bench_function("candlestick_200_full", |b| {
        b.iter(|| render_to_png(black_box(&chart), 1280, 720).unwrap());
    });

    // Same chart at 4K for high-res investor slide
    group.bench_function("candlestick_200_4k", |b| {
        b.iter(|| render_to_png(black_box(&chart), 3840, 2160).unwrap());
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// 9. Correlation Heatmap (Science / ML Workload)
// ─────────────────────────────────────────────────────────────────

fn bench_correlation_heatmap(c: &mut Criterion) {
    let mut group = c.benchmark_group("correlation_heatmap");

    // 10×10 correlation matrix
    let labels_10: Vec<String> = (0..10).map(|i| format!("F{}", i + 1)).collect();
    let corr_10 = Heatmap::correlation(gen_correlation(10), labels_10)
        .title("10×10 Correlation")
        .theme(Theme::dark())
        .build();
    group.bench_function("corr_10x10", |b| {
        b.iter(|| render_to_png(black_box(&corr_10), W, H).unwrap());
    });

    // 20×20 correlation matrix — stress test
    let labels_20: Vec<String> = (0..20).map(|i| format!("Feat-{}", i + 1)).collect();
    let corr_20 = Heatmap::correlation(gen_correlation(20), labels_20)
        .title("20×20 Correlation")
        .theme(Theme::ocean())
        .cell_gap(1.0)
        .cell_radius(1.0)
        .build();
    group.bench_function("corr_20x20", |b| {
        b.iter(|| render_to_png(black_box(&corr_20), 1000, 1000).unwrap());
    });

    // 50×50 dense heatmap — extreme
    let hmap_50 = Chart::heatmap(gen_heatmap(50, 50))
        .title("50×50 Dense")
        .theme(Theme::dark())
        .build();
    group.bench_function("heatmap_50x50", |b| {
        b.iter(|| render_to_png(black_box(&hmap_50), 1200, 1200).unwrap());
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// 10. Isolated Layout Engine
// ─────────────────────────────────────────────────────────────────

fn bench_layout_engine(c: &mut Criterion) {
    let mut group = c.benchmark_group("layout_engine");

    // Build-only: time the chart construction (no rendering)
    group.bench_function("build_line_1k", |b| {
        let y = gen_f64(1000);
        b.iter(|| {
            let chart = Chart::line(black_box(&y))
                .title("Layout Only")
                .theme(Theme::dark())
                .smooth()
                .filled()
                .with_points()
                .build();
            let _ = black_box(chart);
        });
    });

    group.bench_function("build_scatter_10k", |b| {
        let (x, y) = gen_xy(10_000);
        b.iter(|| {
            let chart = Chart::scatter(black_box(&x), black_box(&y))
                .title("Layout Only")
                .theme(Theme::dark())
                .build();
            let _ = black_box(chart);
        });
    });

    group.bench_function("build_bar_50cat", |b| {
        let labels = gen_categories(50);
        let vals: Vec<f64> = (0..50).map(|i| (i as f64 + 1.0) * 3.3).collect();
        b.iter(|| {
            let chart = Chart::bar(black_box(labels.clone()), black_box(&vals))
                .title("Layout Only")
                .theme(Theme::dark())
                .build();
            let _ = black_box(chart);
        });
    });

    group.bench_function("build_candlestick_500", |b| {
        let ohlc = gen_ohlc(500);
        b.iter(|| {
            let chart = Chart::candlestick(black_box(ohlc.clone()))
                .title("Layout Only")
                .theme(Theme::dark())
                .build();
            let _ = black_box(chart);
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────

fn investor_config() -> Criterion {
    Criterion::default()
        .warm_up_time(StdDuration::from_secs(2))
        .measurement_time(StdDuration::from_secs(5))
        .sample_size(30)
}

criterion_group! {
    name = benches;
    config = investor_config();
    targets =
        bench_all_chart_types,
        bench_multi_series,
        bench_data_volume,
        bench_all_themes,
        bench_export_paths,
        bench_resolution_scaling,
        bench_kitchen_sink,
        bench_financial_dashboard,
        bench_correlation_heatmap,
        bench_layout_engine,
}
criterion_main!(benches);
