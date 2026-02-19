//! Visual benchmark suite — renders every chart type under stress conditions
//! and produces a self-contained HTML report with embedded SVGs.
//!
//! Run: cargo run -p scry-chart --example visual_benchmark
//!
//! Output: /tmp/scry_visual_benchmark.html  (open in any browser)

use scry_chart::chart::OhlcEntry;
use scry_chart::data::Series;
use scry_chart::prelude::*;
use scry_chart::svg_export::render_to_svg;

use std::io::Write;
use std::time::Instant;

// ---------------------------------------------------------------------------
// HTML template
// ---------------------------------------------------------------------------

const CSS: &str = r#"
:root {
    --bg: #111118;
    --card: #1a1a24;
    --border: #2a2a3a;
    --fg: #d0d0e0;
    --accent: #63b3ed;
    --dim: #888;
    --pass: #68d391;
    --warn: #fbbf24;
    --fail: #fc8181;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg);
    color: var(--fg);
    padding: 2rem;
    max-width: 1600px;
    margin: 0 auto;
}
h1 { font-size: 2rem; margin-bottom: 0.25rem; color: var(--accent); }
h2 {
    font-size: 1.25rem;
    margin: 2.5rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
    color: var(--accent);
}
h3 {
    font-size: 0.9rem;
    margin: 1.5rem 0 0.75rem;
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.meta { color: var(--dim); font-size: 0.85rem; margin-bottom: 2rem; }
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
    gap: 1.25rem;
}
.grid-sm {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1rem;
}
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    overflow: hidden;
}
.card svg { width: 100%; height: auto; display: block; border-radius: 4px; }
.card .label {
    font-size: 0.78rem;
    color: var(--dim);
    margin-top: 0.5rem;
    display: flex;
    justify-content: space-between;
}
.rule-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    font-size: 0.85rem;
    line-height: 1.6;
}
.rule-box b { color: var(--accent); }
.rule-box .src { color: var(--dim); font-style: italic; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td {
    text-align: left; padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
}
th { color: var(--accent); font-weight: 600; }
"#;

// ---------------------------------------------------------------------------
// Chart factories
// ---------------------------------------------------------------------------

fn type_gallery() -> Vec<(&'static str, Chart)> {
    vec![
        ("Line", Charts::line(&[1.0, 4.0, 2.0, 8.0, 5.0, 3.0, 7.0, 6.0])
            .title("Line Chart").x_label("Time").y_label("Value")
            .theme(Theme::dark()).build()),
        ("Area (filled)", Charts::area(&[3.0, 7.0, 4.0, 9.0, 6.0, 8.0, 5.0])
            .title("Area Chart").x_label("Time").y_label("Revenue")
            .theme(Theme::dark()).build()),
        ("Scatter", Charts::scatter(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2.0, 4.0, 1.5, 8.0, 5.0, 7.5, 3.0, 9.0],
        ).title("Scatter Plot").x_label("X").y_label("Y")
            .theme(Theme::dark()).build()),
        ("Bar", Charts::bar(
            vec!["Mon".into(), "Tue".into(), "Wed".into(), "Thu".into(), "Fri".into()],
            &[12.0, 19.0, 8.0, 15.0, 22.0],
        ).title("Bar Chart").y_label("Units")
            .theme(Theme::dark()).build()),
        ("Histogram", {
            let data: Vec<f64> = (0..200).map(|i| {
                let x = i as f64 * 0.05;
                x.sin() * 50.0 + 50.0
            }).collect();
            Charts::histogram(&data).bins(20).title("Histogram")
                .x_label("Value").y_label("Frequency")
                .theme(Theme::dark()).build()
        }),
        ("Box Plot", Charts::boxplot(vec![
            ("Group A", (0..40).map(|i| 10.0 + (i as f64 * 0.5).sin() * 3.0).collect()),
            ("Group B", (0..40).map(|i| 15.0 + (i as f64 * 0.3).cos() * 4.0).collect()),
            ("Group C", (0..40).map(|i| 8.0 + (i as f64 * 0.7).sin() * 2.0 + i as f64 * 0.15).collect()),
        ]).title("Box Plot").y_label("Score")
            .theme(Theme::dark()).build()),
        ("Heatmap", Charts::heatmap(vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![5.0, 4.0, 3.0, 2.0, 1.0],
            vec![2.0, 4.0, 6.0, 4.0, 2.0],
            vec![3.0, 1.0, 5.0, 7.0, 3.0],
        ]).title("Heatmap")
            .row_labels(vec!["R1".into(), "R2".into(), "R3".into(), "R4".into()])
            .col_labels(vec!["C1".into(), "C2".into(), "C3".into(), "C4".into(), "C5".into()])
            .theme(Theme::dark()).build()),
        ("Pie", Charts::pie(
            vec!["Rust".into(), "Python".into(), "Go".into(), "TypeScript".into(), "Other".into()],
            &[35.0, 25.0, 15.0, 15.0, 10.0],
        ).title("Pie Chart").theme(Theme::dark()).build()),
        ("Radar", Charts::radar(vec!["Speed", "Power", "Range", "Defense", "Magic"])
            .add_series("Hero", &[0.8, 0.6, 0.9, 0.4, 0.7])
            .add_series("Villain", &[0.5, 0.9, 0.3, 0.8, 0.6])
            .title("Radar Chart").theme(Theme::dark()).build()),
        ("Candlestick", Charts::candlestick(vec![
            OhlcEntry { x: 1.0, open: 100.0, high: 110.0, low: 95.0, close: 105.0 },
            OhlcEntry { x: 2.0, open: 105.0, high: 115.0, low: 100.0, close: 98.0 },
            OhlcEntry { x: 3.0, open: 98.0, high: 108.0, low: 92.0, close: 106.0 },
            OhlcEntry { x: 4.0, open: 106.0, high: 120.0, low: 104.0, close: 118.0 },
            OhlcEntry { x: 5.0, open: 118.0, high: 125.0, low: 112.0, close: 110.0 },
        ]).title("Candlestick Chart").theme(Theme::dark()).build()),
        ("Violin", Charts::violin(vec![
            ("A", (0..80).map(|i| 10.0 + (i as f64 * 0.2).sin() * 5.0).collect()),
            ("B", (0..80).map(|i| 15.0 + (i as f64 * 0.15).cos() * 6.0).collect()),
        ]).title("Violin Plot").theme(Theme::dark()).build()),
        ("Waterfall", Charts::waterfall(
            vec!["Revenue".into(), "COGS".into(), "OpEx".into(), "Tax".into(), "Net".into()],
            &[100.0, -40.0, -25.0, -10.0, 25.0],
        ).title("Waterfall Chart").theme(Theme::dark()).build()),
        ("Bubble", Charts::bubble(
            &[1.0, 3.0, 5.0, 7.0, 9.0],
            &[2.0, 8.0, 4.0, 6.0, 3.0],
            &[10.0, 30.0, 20.0, 40.0, 15.0],
        ).title("Bubble Chart").theme(Theme::dark()).build()),
        ("Lollipop", Charts::lollipop(
            vec!["A".into(), "B".into(), "C".into(), "D".into(), "E".into()],
            &[15.0, 30.0, 22.0, 40.0, 18.0],
        ).title("Lollipop Chart").theme(Theme::dark()).build()),
        ("Funnel", Charts::funnel(
            vec!["Visitors".into(), "Leads".into(), "Qualified".into(), "Deals".into()],
            &[1000.0, 600.0, 300.0, 100.0],
        ).title("Sales Funnel").theme(Theme::dark()).build()),
        ("Gauge", Charts::gauge(73.0)
            .range(0.0, 100.0).title("CPU Usage")
            .theme(Theme::dark()).build()),
        ("Contour", Charts::contour({
            let n = 20;
            (0..n).map(|r| {
                let y = r as f64 / n as f64 * 4.0 - 2.0;
                (0..n).map(|c| {
                    let x = c as f64 / n as f64 * 4.0 - 2.0;
                    (-(x * x + y * y)).exp()
                }).collect()
            }).collect()
        }).levels(8).filled().title("Contour Plot")
            .theme(Theme::dark()).build()),
        ("Sparkline", Charts::sparkline(&[3.0, 7.0, 4.0, 8.0, 2.0, 9.0, 5.0, 6.0, 3.0])
            .build()),
    ]
}

fn theme_sweep() -> Vec<(&'static str, Chart)> {
    let data = &[1.0, 4.0, 2.0, 8.0, 5.0, 3.0, 7.0];
    let themes: Vec<(&str, Theme)> = vec![
        ("Dark", Theme::dark()),
        ("Light", Theme::light()),
        ("Pastel", Theme::pastel()),
        ("Ocean", Theme::ocean()),
        ("Forest", Theme::forest()),
        ("Colorblind", Theme::colorblind()),
    ];
    themes.into_iter().map(|(name, theme)| {
        let chart = Charts::line(data)
            .title(&format!("{name} Theme"))
            .x_label("Time").y_label("Value")
            .theme(theme).build();
        (name, chart)
    }).collect()
}

fn text_stress() -> Vec<(&'static str, Chart)> {
    vec![
        ("Long title", Charts::line(&[1.0, 4.0, 2.0, 8.0, 5.0])
            .title("A Very Long Chart Title That Should Probably Wrap or Ellipsize on Small Canvases")
            .subtitle("And This Is an Equally Long Subtitle Testing Hierarchical Font Sizing")
            .x_label("X-Axis Label With Extra Detail (thousands of units)")
            .y_label("Y-Axis Label With Units (μg/mL)")
            .theme(Theme::dark()).build()),
        ("20 categories", Charts::bar(
            (0..20).map(|i| format!("Category {}", i + 1)).collect(),
            &(0..20).map(|i| (i as f64 + 1.0) * 5.0).collect::<Vec<_>>(),
        ).title("Many Categories — Collision Test")
            .y_label("Value")
            .theme(Theme::dark()).build()),
        ("Long cat labels", Charts::bar(
            vec![
                "United States of America".into(),
                "United Kingdom".into(),
                "Republic of Korea".into(),
                "People's Republic of China".into(),
                "Russian Federation".into(),
            ],
            &[320.0, 67.0, 51.0, 1400.0, 144.0],
        ).title("Long Category Labels")
            .y_label("Population (M)")
            .theme(Theme::dark()).build()),
        ("Unicode labels", Charts::bar(
            vec!["日本語".into(), "中文".into(), "한국어".into(), "العربية".into(), "Ελληνικά".into()],
            &[125.0, 1400.0, 51.0, 420.0, 10.0],
        ).title("Unicode Category Labels")
            .y_label("Millions")
            .theme(Theme::dark()).build()),
    ]
}

fn data_density() -> Vec<(&'static str, Chart)> {
    vec![
        ("8 series", {
            let series: Vec<Series> = (0..8).map(|s| {
                let values: Vec<f64> = (0..20).map(|i| {
                    (i as f64 * 0.3 + s as f64 * 0.5).sin() * 10.0 + s as f64 * 5.0
                }).collect();
                Series::new(format!("Series {}", (b'A' + s) as char), values)
            }).collect();
            LineChart::new(series)
                .title("8 Overlapping Series — Legend Stress")
                .x_label("Time").y_label("Value")
                .theme(Theme::dark()).build()
        }),
        ("10K scatter", {
            let n = 10_000;
            let x: Vec<f64> = (0..n).map(|i| {
                let t = i as f64 / n as f64 * 20.0;
                t + (t * 3.0).sin() * 0.5
            }).collect();
            let y: Vec<f64> = (0..n).map(|i| {
                let t = i as f64 / n as f64 * 20.0;
                t.sin() * 10.0 + ((i as u64 * 2654435761) % 1000) as f64 * 0.003
            }).collect();
            Charts::scatter(&x, &y)
                .title("10K Points — Density Stress")
                .x_label("X").y_label("Y")
                .theme(Theme::dark()).build()
        }),
        ("Dense line (5K)", {
            let data: Vec<f64> = (0..5000).map(|i| {
                let t = i as f64 * 0.01;
                t.sin() + (t * 7.0).sin() * 0.3
            }).collect();
            Charts::line(&data)
                .title("5K Points — Line Density")
                .x_label("Sample").y_label("Amplitude")
                .theme(Theme::dark()).build()
        }),
    ]
}

fn extreme_values() -> Vec<(&'static str, Chart)> {
    vec![
        ("Huge (1e15)", Charts::line(&[1e15, 2e15, 1.5e15, 3e15, 2.5e15])
            .title("Huge Values (×10¹⁵)").x_label("Step").y_label("Count")
            .theme(Theme::dark()).build()),
        ("Tiny (1e-12)", Charts::line(&[1e-12, 2e-12, 1.5e-12, 3e-12, 2.5e-12])
            .title("Tiny Values (×10⁻¹²)").x_label("Step").y_label("ppm²")
            .theme(Theme::dark()).build()),
        ("All negative", Charts::bar(
            vec!["Q1".into(), "Q2".into(), "Q3".into(), "Q4".into()],
            &[-50.0, -30.0, -80.0, -20.0],
        ).title("All Negative — Bar Baseline").y_label("Loss ($K)")
            .theme(Theme::dark()).build()),
        ("Mixed ±", Charts::bar(
            vec!["Jan".into(), "Feb".into(), "Mar".into(), "Apr".into(), "May".into(), "Jun".into()],
            &[10.0, -5.0, 15.0, -8.0, 20.0, -12.0],
        ).title("Mixed Positive/Negative").y_label("P&L ($K)")
            .theme(Theme::dark()).build()),
        ("Constant", Charts::line(&[42.0, 42.0, 42.0, 42.0, 42.0])
            .title("Constant Data (all 42)").x_label("Step").y_label("Value")
            .theme(Theme::dark()).build()),
        ("Near-zero range", Charts::line(&[100.001, 100.002, 100.0015, 100.003, 100.0025])
            .title("Near-Zero Range (100.001–100.003)").x_label("Step").y_label("Measurement")
            .theme(Theme::dark()).build()),
    ]
}

fn size_stress() -> Vec<(&'static str, u32, u32, Chart)> {
    let make = |_w, _h| {
        Charts::line(&[1.0, 4.0, 2.0, 8.0, 5.0, 3.0, 7.0])
            .title("Size Stress Test")
            .x_label("Time").y_label("Value")
            .theme(Theme::dark()).build()
    };
    vec![
        ("200 × 150 (tiny)", 200, 150, make(200, 150)),
        ("400 × 300 (small)", 400, 300, make(400, 300)),
        ("800 × 500 (standard)", 800, 500, make(800, 500)),
        ("1200 × 750 (large)", 1200, 750, make(1200, 750)),
        ("1600 × 1000 (HD)", 1600, 1000, make(1600, 1000)),
    ]
}

// ---------------------------------------------------------------------------
// HTML writer
// ---------------------------------------------------------------------------

struct HtmlReport {
    buf: Vec<u8>,
    chart_count: usize,
}

impl HtmlReport {
    fn new() -> Self {
        Self { buf: Vec::with_capacity(1 << 20), chart_count: 0 }
    }

    fn header(&mut self, total_charts: usize) {
        write!(self.buf, r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>scry-chart Visual Benchmark</title>
<style>{CSS}</style>
</head>
<body>
<h1>scry-chart Visual Benchmark</h1>
<p class="meta">{total_charts} charts rendered · formatting audit report</p>
"#).unwrap();
    }

    fn section(&mut self, title: &str) {
        write!(self.buf, "<h2>{title}</h2>\n").unwrap();
    }

    fn rule_box(&mut self, rule: &str, source: &str, desc: &str) {
        write!(self.buf, r#"<div class="rule-box"><b>{rule}</b> — {desc} <span class="src">({source})</span></div>"#).unwrap();
        self.buf.push(b'\n');
    }

    fn subsection(&mut self, title: &str) {
        write!(self.buf, "<h3>{title}</h3>\n").unwrap();
    }

    fn open_grid(&mut self) {
        write!(self.buf, r#"<div class="grid">"#).unwrap();
        self.buf.push(b'\n');
    }

    fn open_grid_sm(&mut self) {
        write!(self.buf, r#"<div class="grid-sm">"#).unwrap();
        self.buf.push(b'\n');
    }

    fn close_grid(&mut self) {
        write!(self.buf, "</div>\n").unwrap();
    }

    fn card(&mut self, label: &str, svg: &str) {
        self.chart_count += 1;
        write!(self.buf, r#"<div class="card">{svg}<div class="label"><span>{label}</span><span>#{}</span></div></div>"#, self.chart_count).unwrap();
        self.buf.push(b'\n');
    }

    fn table(&mut self, headers: &[&str], rows: &[Vec<String>]) {
        write!(self.buf, "<table><thead><tr>").unwrap();
        for h in headers { write!(self.buf, "<th>{h}</th>").unwrap(); }
        write!(self.buf, "</tr></thead><tbody>").unwrap();
        for row in rows {
            write!(self.buf, "<tr>").unwrap();
            for cell in row { write!(self.buf, "<td>{cell}</td>").unwrap(); }
            write!(self.buf, "</tr>").unwrap();
        }
        write!(self.buf, "</tbody></table>\n").unwrap();
    }

    fn footer(&mut self) {
        write!(self.buf, r#"
<p class="meta" style="margin-top: 3rem; text-align: center;">
Generated by scry-chart visual benchmark · {} charts
</p>
</body></html>"#, self.chart_count).unwrap();
    }

    fn finish(self) -> Vec<u8> {
        self.buf
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let t0 = Instant::now();

    // Pre-generate all charts.
    let gallery = type_gallery();
    let themes = theme_sweep();
    let text = text_stress();
    let density = data_density();
    let extremes = extreme_values();
    let sizes = size_stress();

    let total = gallery.len() + themes.len() + text.len() + density.len()
        + extremes.len() + sizes.len();

    let mut report = HtmlReport::new();
    report.header(total);

    // --- Formatting rules reference table ---
    report.section("Formatting Audit Checklist");
    report.table(
        &["#", "Rule", "Source", "What to Check"],
        &[
            vec!["T1".into(), "Data-Ink Ratio".into(), "Tufte".into(),
                 "Maximize data ink, minimize non-data ink".into()],
            vec!["T2".into(), "No Chartjunk".into(), "Tufte".into(),
                 "No decorative elements that don't convey data".into()],
            vec!["T3".into(), "Proportional".into(), "Tufte".into(),
                 "Visual magnitude matches numeric magnitude".into()],
            vec!["C1".into(), "Legend Placement".into(), "Cleveland".into(),
                 "Legend outside data region, not overlapping data".into()],
            vec!["C2".into(), "Minimal Ticks".into(), "Cleveland".into(),
                 "Only necessary tick marks, no visual noise".into()],
            vec!["C3".into(), "Banking to 45°".into(), "Cleveland".into(),
                 "Line chart aspect ratios favor ~45° slopes".into()],
            vec!["M1".into(), "Font Hierarchy".into(), "Material".into(),
                 "Title (18) → Label (13) → Tick (11), ≥2pt steps".into()],
            vec!["M2".into(), "Sentence Case".into(), "Material".into(),
                 "UI text uses sentence case, not ALL CAPS".into()],
            vec!["M3".into(), "WCAG AA Contrast".into(), "Material".into(),
                 "Text:background ≥ 4.5:1 contrast ratio".into()],
            vec!["M4".into(), "Bar Zero Baseline".into(), "Material".into(),
                 "Bar charts start at zero (avoid truncation bias)".into()],
            vec!["G1".into(), "Label Collision".into(), "Best Practice".into(),
                 "Long labels wrap, rotate, or ellipsize — never overlap".into()],
            vec!["G2".into(), "Axis Units".into(), "Best Practice".into(),
                 "Axis labels include units of measurement".into()],
            vec!["G3".into(), "Consistent Margins".into(), "Best Practice".into(),
                 "Equal whitespace, no cramped edges".into()],
            vec!["G4".into(), "Grayscale Safe".into(), "Best Practice".into(),
                 "Color palettes remain distinguishable without color".into()],
        ],
    );

    // --- Section 1: Type Gallery ---
    report.section("§1 · Type Gallery (all chart types, dark theme)");
    report.rule_box("T1+T2", "Tufte",
        "Check each chart type for unnecessary chrome. Data should be the star.");
    report.open_grid();
    for (name, chart) in &gallery {
        let svg = render_to_svg(chart, 800, 500);
        report.card(&format!("{name}"), &svg);
    }
    report.close_grid();

    // --- Section 2: Theme Sweep ---
    report.section("§2 · Theme Sweep (same data, 6 themes)");
    report.rule_box("M3", "Material",
        "Verify text is readable against each background. Check WCAG AA.");
    report.open_grid();
    for (name, chart) in &themes {
        let svg = render_to_svg(chart, 800, 500);
        report.card(&format!("{name} theme"), &svg);
    }
    report.close_grid();

    // --- Section 3: Text Stress ---
    report.section("§3 · Text Stress (long labels, many categories)");
    report.rule_box("G1+M1", "Cleveland + Material",
        "Labels must not overlap. Font hierarchy must remain clear even with long text.");
    report.open_grid();
    for (name, chart) in &text {
        let svg = render_to_svg(chart, 800, 500);
        report.card(name, &svg);
    }
    report.close_grid();

    // --- Section 4: Data Density ---
    report.section("§4 · Data Density (many series, large N)");
    report.rule_box("C1", "Cleveland",
        "Legend should not overlap the data. 8+ series should remain distinguishable.");
    report.open_grid();
    for (name, chart) in &density {
        let svg = render_to_svg(chart, 800, 500);
        report.card(name, &svg);
    }
    report.close_grid();

    // --- Section 5: Extreme Values ---
    report.section("§5 · Extreme Values (huge, tiny, negative, constant)");
    report.rule_box("M4+T3", "Material + Tufte",
        "Bar charts must have zero baseline. Tick labels must format extreme numbers cleanly.");
    report.open_grid();
    for (name, chart) in &extremes {
        let svg = render_to_svg(chart, 800, 500);
        report.card(name, &svg);
    }
    report.close_grid();

    // --- Section 6: Size Stress ---
    report.section("§6 · Canvas Size Stress (same chart, 5 resolutions)");
    report.rule_box("M1+G3", "Material",
        "Font sizes must scale proportionally. Margins should not collapse or become huge.");
    report.open_grid_sm();
    for (name, w, h, chart) in &sizes {
        let svg = render_to_svg(chart, *w, *h);
        report.card(&format!("{name} — {}×{}", w, h), &svg);
    }
    report.close_grid();

    report.footer();

    // Write output.
    let html = report.finish();
    let out_path = "/tmp/scry_visual_benchmark.html";
    std::fs::write(out_path, &html).expect("failed to write HTML");

    let elapsed = t0.elapsed();
    eprintln!("✓ Generated {total} charts in {:.2}s", elapsed.as_secs_f64());
    eprintln!("  Output: {out_path}");
    eprintln!("  Size: {:.1} KB", html.len() as f64 / 1024.0);
    eprintln!("  Open in browser: xdg-open {out_path}");
}
