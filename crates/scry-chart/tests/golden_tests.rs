//! Golden image regression tests for scry-chart.
//!
//! Renders every chart type to PNG at a fixed size, then compares the output
//! against committed golden reference images in `tests/golden/`.
//!
//! # Generating golden images
//!
//! Run with `GOLDEN_UPDATE=1` to write/overwrite the reference images:
//!
//! ```sh
//! GOLDEN_UPDATE=1 cargo test -p scry-chart --test golden_tests
//! ```
//!
//! # Pixel-diff tolerance
//!
//! Anti-aliasing can vary across platforms and tiny-skia versions, so we
//! allow a small per-pixel tolerance (configurable via `MAX_DIFF_PERCENT`).

use scry_chart::chart::{Chart, Charts, LineChart, OhlcEntry};
use scry_chart::data::Series;
use scry_chart::export::render_to_png;
use scry_chart::theme::Theme;
use std::path::{Path, PathBuf};

/// Maximum percentage of pixels that may differ before a test fails.
const MAX_DIFF_PERCENT: f64 = 0.5;

/// Per-channel tolerance for individual pixels (0–255 scale).
const CHANNEL_TOLERANCE: u8 = 8;

/// Standard render size for golden tests.
const WIDTH: u32 = 800;
const HEIGHT: u32 = 500;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn should_update() -> bool {
    std::env::var("GOLDEN_UPDATE").is_ok()
}

/// Decode a PNG file into raw RGBA pixels + dimensions.
fn decode_png(data: &[u8]) -> (Vec<u8>, u32, u32) {
    let decoder = png::Decoder::new(std::io::Cursor::new(data));
    let mut reader = decoder.read_info().expect("failed to read PNG header");
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("failed to decode PNG");
    buf.truncate(info.buffer_size());
    (buf, info.width, info.height)
}

/// Compare two RGBA buffers. Returns the percentage of pixels that differ
/// beyond `CHANNEL_TOLERANCE` on any channel.
fn pixel_diff_percent(a: &[u8], b: &[u8], _width: u32, _height: u32) -> f64 {
    assert_eq!(a.len(), b.len(), "image buffers must be the same size");
    let total_pixels = a.len() / 4;
    if total_pixels == 0 {
        return 0.0;
    }

    let diff_pixels = a
        .chunks_exact(4)
        .zip(b.chunks_exact(4))
        .filter(|(pa, pb)| {
            pa.iter()
                .zip(pb.iter())
                .any(|(&ca, &cb)| ca.abs_diff(cb) > CHANNEL_TOLERANCE)
        })
        .count();

    (diff_pixels as f64 / total_pixels as f64) * 100.0
}

/// Core test runner: render a chart, compare against golden, or update golden.
fn assert_golden(name: &str, chart: Chart) {
    let png_bytes = render_to_png(&chart, WIDTH, HEIGHT).expect("render_to_png failed");

    let golden_path = golden_dir().join(format!("{name}.png"));

    if should_update() {
        std::fs::create_dir_all(golden_dir()).ok();
        std::fs::write(&golden_path, &png_bytes)
            .unwrap_or_else(|e| panic!("failed to write golden {}: {e}", golden_path.display()));
        eprintln!("  ✓ updated golden: {}", golden_path.display());
        return;
    }

    if !golden_path.exists() {
        panic!(
            "Golden image not found: {}\n\
             Run with GOLDEN_UPDATE=1 to generate it:\n\
             GOLDEN_UPDATE=1 cargo test -p scry-chart --test golden_tests",
            golden_path.display()
        );
    }

    let golden_data = std::fs::read(&golden_path)
        .unwrap_or_else(|e| panic!("failed to read golden {}: {e}", golden_path.display()));

    let (actual_rgba, aw, ah) = decode_png(&png_bytes);
    let (golden_rgba, gw, gh) = decode_png(&golden_data);

    assert_eq!(
        (aw, ah),
        (gw, gh),
        "golden image dimensions mismatch for {name}: actual={aw}x{ah}, golden={gw}x{gh}"
    );

    let diff = pixel_diff_percent(&actual_rgba, &golden_rgba, aw, ah);
    assert!(
        diff <= MAX_DIFF_PERCENT,
        "pixel diff {diff:.2}% exceeds threshold {MAX_DIFF_PERCENT}% for golden '{name}'\n\
         Run with GOLDEN_UPDATE=1 to update the golden image if the change is intentional."
    );
}

// ===========================================================================
// Golden tests — one per chart type
// ===========================================================================

#[test]
fn golden_line_basic() {
    let chart = Charts::line(&[1.0, 4.0, 2.0, 8.0, 5.0, 3.0, 7.0, 6.0])
        .title("Line Chart")
        .x_label("Time")
        .y_label("Value")
        .theme(Theme::dark())
        .build();
    assert_golden("line_basic", chart);
}

#[test]
fn golden_line_smooth_filled() {
    let chart = Charts::line(&[1.0, 4.0, 2.0, 8.0, 5.0, 3.0, 7.0])
        .smooth()
        .filled()
        .with_points()
        .title("Smooth Filled Line")
        .theme(Theme::dark())
        .build();
    assert_golden("line_smooth_filled", chart);
}

#[test]
fn golden_line_multi_series() {
    let chart = LineChart::new(vec![
        Series::new("Revenue", vec![10.0, 30.0, 20.0, 50.0, 40.0]),
        Series::new("Expenses", vec![15.0, 25.0, 35.0, 30.0, 45.0]),
        Series::new("Profit", vec![-5.0, 5.0, -15.0, 20.0, -5.0]),
    ])
    .title("Financial Overview")
    .theme(Theme::dark())
    .build();
    assert_golden("line_multi_series", chart);
}

#[test]
fn golden_scatter() {
    let chart = Charts::scatter(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2.0, 4.0, 1.0, 8.0, 5.0, 9.0, 3.0, 7.0],
    )
    .title("Scatter Plot")
    .x_label("X")
    .y_label("Y")
    .theme(Theme::dark())
    .build();
    assert_golden("scatter", chart);
}

#[test]
fn golden_bar() {
    let chart = Charts::bar(
        vec![
            "Mon".into(),
            "Tue".into(),
            "Wed".into(),
            "Thu".into(),
            "Fri".into(),
        ],
        &[12.0, 19.0, 8.0, 15.0, 22.0],
    )
    .title("Weekly Sales")
    .y_label("Units")
    .theme(Theme::dark())
    .build();
    assert_golden("bar", chart);
}

#[test]
fn golden_bar_grouped() {
    let chart = Charts::bar(
        vec![
            "Q1".into(),
            "Q2".into(),
            "Q3".into(),
            "Q4".into(),
        ],
        &[10.0, 15.0, 12.0, 18.0],
    )
    .add_series(Series::new("Product B", vec![8.0, 12.0, 14.0, 16.0]))
    .title("Grouped Bars")
    .theme(Theme::dark())
    .build();
    assert_golden("bar_grouped", chart);
}

#[test]
fn golden_histogram() {
    let data: Vec<f64> = (0..200)
        .map(|i| (i as f64 * 0.05).sin() * 50.0 + 50.0)
        .collect();
    let chart = Charts::histogram(&data)
        .bins(20)
        .title("Distribution")
        .x_label("Value")
        .y_label("Frequency")
        .theme(Theme::dark())
        .build();
    assert_golden("histogram", chart);
}

#[test]
fn golden_boxplot() {
    let chart = Charts::boxplot(vec![
        ("Group A", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        ("Group B", vec![3.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 12.0, 15.0]),
        ("Group C", vec![0.5, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 20.0]),
    ])
    .title("Box Plot")
    .y_label("Score")
    .theme(Theme::dark())
    .build();
    assert_golden("boxplot", chart);
}

#[test]
fn golden_heatmap() {
    let chart = Charts::heatmap(vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0],
        vec![9.0, 10.0, 11.0, 12.0],
    ])
    .title("Heatmap")
    .row_labels(vec!["R1".into(), "R2".into(), "R3".into()])
    .col_labels(vec!["C1".into(), "C2".into(), "C3".into(), "C4".into()])
    .theme(Theme::dark())
    .build();
    assert_golden("heatmap", chart);
}

#[test]
fn golden_pie() {
    let chart = Charts::pie(
        vec![
            "Rent".into(),
            "Food".into(),
            "Transport".into(),
            "Entertainment".into(),
            "Savings".into(),
        ],
        &[35.0, 25.0, 15.0, 10.0, 15.0],
    )
    .title("Monthly Budget")
    .theme(Theme::dark())
    .build();
    assert_golden("pie", chart);
}

#[test]
fn golden_radar() {
    let chart = Charts::radar(vec!["Speed", "Power", "Range", "Defense", "Magic"])
        .add_series("Hero", &[0.8, 0.6, 0.9, 0.4, 0.7])
        .add_series("Villain", &[0.5, 0.9, 0.3, 0.8, 0.6])
        .title("Character Stats")
        .theme(Theme::dark())
        .build();
    assert_golden("radar", chart);
}

#[test]
fn golden_candlestick() {
    let data = vec![
        OhlcEntry { x: 1.0, open: 100.0, high: 110.0, low: 95.0, close: 105.0 },
        OhlcEntry { x: 2.0, open: 105.0, high: 115.0, low: 100.0, close: 98.0 },
        OhlcEntry { x: 3.0, open: 98.0, high: 108.0, low: 92.0, close: 106.0 },
        OhlcEntry { x: 4.0, open: 106.0, high: 120.0, low: 104.0, close: 118.0 },
        OhlcEntry { x: 5.0, open: 118.0, high: 125.0, low: 112.0, close: 110.0 },
    ];
    let chart = Charts::candlestick(data)
        .title("OHLC Chart")
        .theme(Theme::dark())
        .build();
    assert_golden("candlestick", chart);
}

#[test]
fn golden_violin() {
    let chart = Charts::violin(vec![
        ("A", vec![1.0, 2.0, 2.5, 3.0, 3.0, 3.5, 4.0, 5.0, 6.0]),
        ("B", vec![2.0, 3.0, 3.5, 4.0, 4.0, 4.5, 5.0, 5.5, 8.0]),
    ])
    .title("Violin Plot")
    .theme(Theme::dark())
    .build();
    assert_golden("violin", chart);
}

#[test]
fn golden_waterfall() {
    let chart = Charts::waterfall(
        vec![
            "Revenue".into(),
            "COGS".into(),
            "OpEx".into(),
            "Tax".into(),
            "Net".into(),
        ],
        &[100.0, -40.0, -25.0, -10.0, 25.0],
    )
    .title("Waterfall Chart")
    .theme(Theme::dark())
    .build();
    assert_golden("waterfall", chart);
}

#[test]
fn golden_bubble() {
    let chart = Charts::bubble(
        &[1.0, 3.0, 5.0, 7.0, 9.0],
        &[2.0, 8.0, 4.0, 6.0, 3.0],
        &[10.0, 30.0, 20.0, 40.0, 15.0],
    )
    .title("Bubble Chart")
    .theme(Theme::dark())
    .build();
    assert_golden("bubble", chart);
}

#[test]
fn golden_lollipop() {
    let chart = Charts::lollipop(
        vec!["A".into(), "B".into(), "C".into(), "D".into()],
        &[15.0, 30.0, 22.0, 40.0],
    )
    .title("Lollipop Chart")
    .theme(Theme::dark())
    .build();
    assert_golden("lollipop", chart);
}

#[test]
fn golden_funnel() {
    let chart = Charts::funnel(
        vec![
            "Visitors".into(),
            "Leads".into(),
            "Qualified".into(),
            "Deals".into(),
        ],
        &[1000.0, 600.0, 300.0, 100.0],
    )
    .title("Sales Funnel")
    .theme(Theme::dark())
    .build();
    assert_golden("funnel", chart);
}

#[test]
fn golden_gauge() {
    let chart = Charts::gauge(73.0)
        .title("CPU Usage")
        .theme(Theme::dark())
        .build();
    assert_golden("gauge", chart);
}

// ===========================================================================
// Theme variants
// ===========================================================================

#[test]
fn golden_light_theme() {
    let chart = Charts::line(&[1.0, 4.0, 2.0, 8.0, 5.0, 3.0])
        .title("Light Theme")
        .theme(Theme::light())
        .build();
    assert_golden("theme_light", chart);
}

#[test]
fn golden_colorblind_theme() {
    let chart = LineChart::new(vec![
        Series::new("A", vec![1.0, 3.0, 2.0, 5.0]),
        Series::new("B", vec![2.0, 1.0, 4.0, 3.0]),
        Series::new("C", vec![3.0, 5.0, 1.0, 4.0]),
    ])
    .title("Colorblind Theme")
    .theme(Theme::colorblind())
    .build();
    assert_golden("theme_colorblind", chart);
}
