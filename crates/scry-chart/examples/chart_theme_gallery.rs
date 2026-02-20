//! Render every chart type × every theme to PNG + HTML gallery with feedback.
//!
//! Generates a complete visual audit of chart formatting across all
//! themes. Outputs to `/tmp/scry_gallery/` with an `index.html` for
//! easy browsing. Each chart card includes a feedback textarea.
//! Feedback can be exported as `feedback.json` for automated review.
//!
//! Run: cargo run -p scry-chart --example chart_theme_gallery --release

use scry_chart::chart::OhlcEntry;
use scry_chart::data::Series;
use scry_chart::export::save_png;
use scry_chart::prelude::*;

struct ChartEntry {
    name: String,
    label: String,
    width: u32,
    height: u32,
}

fn main() -> Result<(), String> {
    let dir = "/tmp/scry_gallery";
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;

    let themes: Vec<(&str, Theme)> = vec![
        ("dark", Theme::dark()),
        ("light", Theme::light()),
        ("pastel", Theme::pastel()),
        ("ocean", Theme::ocean()),
        ("forest", Theme::forest()),
        ("colorblind", Theme::colorblind()),
    ];

    let mut html_entries: Vec<ChartEntry> = Vec::new();
    let mut count = 0u32;

    for (theme_name, theme) in &themes {
        let mut idx = 0u32;
        let mut emit = |label: &str, chart: Chart, w: u32, h: u32| -> Result<(), String> {
            idx += 1;
            let name = format!("{theme_name}_{idx:02}_{}", label.replace(' ', "_").replace('—', "-").replace('±', "pm").to_lowercase());
            save_png(&chart, w, h, format!("{dir}/{name}.png"))?;
            html_entries.push(ChartEntry {
                name: name.clone(),
                label: label.to_string(),
                width: w,
                height: h,
            });
            count += 1;
            Ok(())
        };

        // =============================================================
        // CORE CHART TYPES (1–23)
        // =============================================================

        // 1. Line — basic
        emit("Line — Basic",
            Charts::line(&[1.0, 4.0, 2.0, 8.0, 5.0, 3.0, 7.0, 6.0, 9.0, 4.5])
                .title("Line Chart — Basic")
                .x_label("Time").y_label("Value")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 2. Line — filled + points
        emit("Line — Filled + Points",
            Charts::line(&[10.0, 25.0, 18.0, 35.0, 28.0, 42.0, 30.0])
                .title("Line Chart — Filled + Points")
                .x_label("Month").y_label("Revenue ($K)")
                .filled().with_points()
                .theme(theme.clone()).build(),
            800, 500)?;

        // 3. Multi-series line
        emit("Multi-Series Line",
            LineChart::new(vec![
                Series::new("Series A", vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 5.5]),
                Series::new("Series B", vec![2.0, 1.5, 4.0, 3.0, 5.0, 4.5, 7.0]),
                Series::new("Series C", vec![3.0, 2.0, 1.0, 4.0, 3.5, 5.0, 6.0]),
            ])
                .title("Multi-Series Line")
                .x_label("Time").y_label("Value")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 4. Area chart
        emit("Area Chart",
            Charts::area(&[3.0, 7.0, 4.0, 9.0, 6.0, 8.0, 5.0, 10.0, 7.0])
                .title("Area Chart")
                .x_label("Time").y_label("Revenue")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 5. Scatter plot
        emit("Scatter Plot",
            Charts::scatter(
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                &[2.0, 4.0, 1.5, 8.0, 5.0, 7.5, 3.0, 9.0],
            )
                .title("Scatter Plot")
                .x_label("X Axis").y_label("Y Axis")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 6. Bar — vertical
        emit("Bar — Vertical",
            Charts::bar(
                vec!["Mon".into(), "Tue".into(), "Wed".into(), "Thu".into(), "Fri".into()],
                &[12.0, 19.0, 8.0, 15.0, 22.0],
            )
                .title("Bar Chart — Vertical")
                .y_label("Units Sold")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 7. Bar — horizontal
        emit("Bar — Horizontal",
            Charts::bar(
                vec!["Alpha".into(), "Beta".into(), "Gamma".into(), "Delta".into()],
                &[30.0, 50.0, 20.0, 45.0],
            )
                .title("Bar Chart — Horizontal")
                .x_label("Score").horizontal()
                .theme(theme.clone()).build(),
            800, 500)?;

        // 8. Bar — negative values
        emit("Bar — Negative ±",
            Charts::bar(
                vec!["Q1".into(), "Q2".into(), "Q3".into(), "Q4".into(), "Q5".into(), "Q6".into()],
                &[10.0, -5.0, 15.0, -8.0, 20.0, -12.0],
            )
                .title("Bar — Mixed Positive/Negative")
                .y_label("Profit/Loss ($K)")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 9. Histogram
        let hist_data: Vec<f64> = (0..200)
            .map(|i| {
                let mut sum = 0.0;
                for k in 0..6u64 {
                    sum += ((i as u64 * 2654435761 + k * 7919) % 10000) as f64 / 10000.0;
                }
                (sum - 3.0) * 2.0 + 10.0
            })
            .collect();
        emit("Histogram",
            Charts::histogram(&hist_data)
                .title("Histogram")
                .x_label("Value").y_label("Count")
                .bins(20)
                .theme(theme.clone()).build(),
            800, 500)?;

        // 10. Box plot
        emit("Box Plot",
            Charts::boxplot(vec![
                ("Group A", (0..40).map(|i| 10.0 + (i as f64 * 0.5).sin() * 3.0 + i as f64 * 0.1).collect()),
                ("Group B", (0..40).map(|i| 15.0 + (i as f64 * 0.3).cos() * 4.0).collect()),
                ("Group C", (0..40).map(|i| 8.0 + (i as f64 * 0.7).sin() * 2.0 + i as f64 * 0.15).collect()),
            ])
                .title("Box Plot")
                .y_label("Score")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 11. Heatmap
        emit("Heatmap",
            Charts::heatmap(vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0],
                vec![5.0, 4.0, 3.0, 2.0, 1.0],
                vec![2.0, 4.0, 6.0, 4.0, 2.0],
                vec![3.0, 1.0, 5.0, 7.0, 3.0],
            ])
                .title("Heatmap")
                .row_labels(vec!["R1".into(), "R2".into(), "R3".into(), "R4".into()])
                .col_labels(vec!["C1".into(), "C2".into(), "C3".into(), "C4".into(), "C5".into()])
                .theme(theme.clone()).build(),
            800, 500)?;

        // 12. Pie chart
        emit("Pie Chart",
            Charts::pie(
                vec!["Rust".into(), "Python".into(), "Go".into(), "TypeScript".into(), "Other".into()],
                &[35.0, 25.0, 15.0, 15.0, 10.0],
            )
                .title("Pie Chart")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 13. Candlestick
        emit("Candlestick",
            Charts::candlestick(vec![
                OhlcEntry { x: 1.0, open: 100.0, high: 110.0, low: 95.0, close: 105.0 },
                OhlcEntry { x: 2.0, open: 105.0, high: 115.0, low: 100.0, close: 98.0 },
                OhlcEntry { x: 3.0, open: 98.0, high: 108.0, low: 92.0, close: 106.0 },
                OhlcEntry { x: 4.0, open: 106.0, high: 120.0, low: 104.0, close: 118.0 },
                OhlcEntry { x: 5.0, open: 118.0, high: 125.0, low: 112.0, close: 110.0 },
                OhlcEntry { x: 6.0, open: 110.0, high: 116.0, low: 105.0, close: 114.0 },
                OhlcEntry { x: 7.0, open: 114.0, high: 122.0, low: 108.0, close: 109.0 },
            ])
                .title("Candlestick Chart")
                .x_label("Day").y_label("Price ($)")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 14. Radar chart
        emit("Radar Chart",
            Charts::radar(vec!["Speed", "Power", "Range", "Defense", "Magic"])
                .add_series("Hero", &[0.8, 0.6, 0.9, 0.4, 0.7])
                .add_series("Villain", &[0.5, 0.9, 0.3, 0.8, 0.6])
                .title("Radar Chart")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 15. Bubble chart
        emit("Bubble Chart",
            Charts::bubble(
                &[1.0, 3.0, 5.0, 7.0, 9.0, 2.0, 6.0],
                &[2.0, 8.0, 4.0, 6.0, 3.0, 5.0, 7.0],
                &[10.0, 30.0, 20.0, 40.0, 15.0, 25.0, 35.0],
            )
                .title("Bubble Chart")
                .x_label("X").y_label("Y")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 16. Violin plot
        emit("Violin Plot",
            Charts::violin(vec![
                ("Group A", (0..80).map(|i| 10.0 + (i as f64 * 0.2).sin() * 5.0).collect()),
                ("Group B", (0..80).map(|i| 15.0 + (i as f64 * 0.15).cos() * 6.0).collect()),
                ("Group C", (0..80).map(|i| 12.0 + (i as f64 * 0.3).sin() * 3.0 + i as f64 * 0.05).collect()),
            ])
                .title("Violin Plot")
                .y_label("Value")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 17. Waterfall chart
        emit("Waterfall Chart",
            Charts::waterfall(
                vec!["Revenue".into(), "COGS".into(), "OpEx".into(), "Tax".into(), "Net".into()],
                &[100.0, -40.0, -25.0, -10.0, 25.0],
            )
                .title("Waterfall Chart")
                .y_label("Amount ($K)")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 18. Funnel chart
        emit("Funnel Chart",
            Charts::funnel(
                vec!["Visitors".into(), "Leads".into(), "Qualified".into(), "Proposals".into(), "Deals".into()],
                &[1000.0, 600.0, 350.0, 150.0, 50.0],
            )
                .title("Sales Funnel")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 19. Gauge chart
        emit("Gauge Chart",
            Charts::gauge(73.0)
                .range(0.0, 100.0)
                .title("CPU Usage")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 20. Lollipop chart
        emit("Lollipop Chart",
            Charts::lollipop(
                vec!["A".into(), "B".into(), "C".into(), "D".into(), "E".into(), "F".into()],
                &[15.0, 30.0, 22.0, 40.0, 18.0, 35.0],
            )
                .title("Lollipop Chart")
                .y_label("Score")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 21. Contour plot
        emit("Contour Plot", {
            let n = 20;
            Charts::contour(
                (0..n).map(|r| {
                    let y = r as f64 / n as f64 * 4.0 - 2.0;
                    (0..n).map(|c| {
                        let x = c as f64 / n as f64 * 4.0 - 2.0;
                        (-(x * x + y * y)).exp()
                    }).collect()
                }).collect()
            )
                .levels(8).filled()
                .title("Contour Plot")
                .theme(theme.clone()).build()
        }, 800, 500)?;

        // 22. Gantt chart
        emit("Gantt Chart",
            Charts::gantt(vec![
                GanttTask::new("Research", 0.0, 3.0).group("Phase 1").progress(1.0),
                GanttTask::new("Design", 2.0, 6.0).group("Phase 1").progress(0.8),
                GanttTask::new("Implement", 5.0, 12.0).group("Phase 2").progress(0.4),
                GanttTask::new("Test", 10.0, 15.0).group("Phase 2").progress(0.1),
                GanttTask::new("Deploy", 14.0, 16.0).group("Phase 3"),
            ])
                .title("Project Timeline")
                .x_label("Week")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 23. Sparkline
        emit("Sparkline",
            Charts::sparkline(&[3.0, 7.0, 4.0, 8.0, 2.0, 9.0, 5.0, 6.0, 3.0, 7.0, 4.0, 8.0])
                .build(),
            400, 100)?;

        // =============================================================
        // EDGE CASE STRESS TESTS (24–40)
        // =============================================================

        // 24. Large values (SI suffix test)
        emit("Large Values (SI)",
            Charts::line(&[150000.0, 250000.0, 180000.0, 350000.0, 280000.0, 400000.0])
                .title("Line — Large Values (SI Suffix)")
                .x_label("Quarter").y_label("Revenue")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 25. Micro range
        emit("Micro Range",
            Charts::line(&[0.001, 0.0015, 0.0012, 0.0018, 0.0014, 0.0016])
                .title("Line — Micro Range")
                .x_label("Sample").y_label("PPM")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 26. All-negative bar
        emit("All-Negative Bar",
            Charts::bar(
                vec!["Q1".into(), "Q2".into(), "Q3".into(), "Q4".into()],
                &[-50.0, -30.0, -80.0, -20.0],
            )
                .title("All Negative — Zero Baseline Test")
                .y_label("Loss ($K)")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 27. Constant data
        emit("Constant Data",
            Charts::line(&[42.0, 42.0, 42.0, 42.0, 42.0])
                .title("Constant Data (all 42)")
                .x_label("Step").y_label("Value")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 28. Near-zero range
        emit("Near-Zero Range",
            Charts::line(&[100.001, 100.002, 100.0015, 100.003, 100.0025])
                .title("Near-Zero Range (100.001–100.003)")
                .x_label("Step").y_label("Measurement")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 29. Small canvas
        emit("Small Canvas",
            Charts::line(&[1.0, 4.0, 2.0, 8.0, 5.0])
                .title("Small Canvas (300×200)")
                .x_label("X").y_label("Y")
                .theme(theme.clone()).build(),
            300, 200)?;

        // 30. Large canvas
        emit("Large Canvas",
            Charts::scatter(
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                &[2.0, 4.0, 1.5, 8.0, 5.0, 7.5, 3.0, 9.0, 6.0, 4.0],
            )
                .title("Large Canvas (1600×1000)")
                .x_label("X").y_label("Y")
                .theme(theme.clone()).build(),
            1600, 1000)?;

        // 31. Many labels (collision test)
        let many_labels: Vec<String> = (0..20).map(|i| format!("Cat_{i}")).collect();
        let many_values: Vec<f64> = (0..20).map(|i| 10.0 + (i as f64 * 0.7).sin() * 5.0).collect();
        emit("Many Labels (20)",
            Charts::bar(many_labels, &many_values)
                .title("Many Labels — Auto-Rotation Test")
                .y_label("Value")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 32. Long category labels
        emit("Long Labels",
            Charts::bar(
                vec![
                    "United States of America".into(),
                    "United Kingdom".into(),
                    "Republic of Korea".into(),
                    "People's Republic of China".into(),
                    "Russian Federation".into(),
                ],
                &[320.0, 67.0, 51.0, 1400.0, 144.0],
            )
                .title("Long Category Labels")
                .y_label("Population (M)")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 33. Unicode labels
        emit("Unicode Labels",
            Charts::bar(
                vec!["日本語".into(), "中文".into(), "한국어".into(), "العربية".into(), "Ελληνικά".into()],
                &[125.0, 1400.0, 51.0, 420.0, 10.0],
            )
                .title("Unicode Category Labels")
                .y_label("Millions")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 34. 8 overlapping series (legend stress)
        emit("8 Series Legend", {
            let series: Vec<Series> = (0..8).map(|s| {
                let vals: Vec<f64> = (0..20).map(|i| {
                    (i as f64 * 0.3 + s as f64 * 0.5).sin() * 10.0 + s as f64 * 5.0
                }).collect();
                Series::new(format!("Series {}", (b'A' + s) as char), vals)
            }).collect();
            LineChart::new(series)
                .title("8 Overlapping Series — Legend Stress")
                .x_label("Time").y_label("Value")
                .theme(theme.clone()).build()
        }, 800, 500)?;

        // 35. Dense line (1K points)
        emit("Dense Line (1K)", {
            let data: Vec<f64> = (0..1000).map(|i| {
                let t = i as f64 * 0.01;
                t.sin() + (t * 7.0).sin() * 0.3
            }).collect();
            Charts::line(&data)
                .title("1K Points — Line Density")
                .x_label("Sample").y_label("Amplitude")
                .theme(theme.clone()).build()
        }, 800, 500)?;

        // 36. Huge values (1e15)
        emit("Huge Values (1e15)",
            Charts::line(&[1e15, 2e15, 1.5e15, 3e15, 2.5e15])
                .title("Huge Values (×10¹⁵)")
                .x_label("Step").y_label("Count")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 37. Tiny values (1e-12)
        emit("Tiny Values (1e-12)",
            Charts::line(&[1e-12, 2e-12, 1.5e-12, 3e-12, 2.5e-12])
                .title("Tiny Values (×10⁻¹²)")
                .x_label("Step").y_label("ppm²")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 38. Many pie slices
        emit("Many Pie Slices (10)",
            Charts::pie(
                vec!["A".into(), "B".into(), "C".into(), "D".into(), "E".into(),
                     "F".into(), "G".into(), "H".into(), "I".into(), "J".into()],
                &[15.0, 12.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0],
            )
                .title("Crowded Pie (10 Slices)")
                .theme(theme.clone()).build(),
            800, 500)?;

        // 39. Large heatmap (10×10)
        emit("Large Heatmap (10x10)", {
            let grid: Vec<Vec<f64>> = (0..10).map(|r| {
                (0..10).map(|c| {
                    let x = c as f64 / 9.0 * 4.0 - 2.0;
                    let y = r as f64 / 9.0 * 4.0 - 2.0;
                    (-(x * x + y * y) / 2.0).exp() * 10.0
                }).collect()
            }).collect();
            let rl: Vec<String> = (0..10).map(|i| format!("R{i}")).collect();
            let cl: Vec<String> = (0..10).map(|i| format!("C{i}")).collect();
            Charts::heatmap(grid)
                .title("Large Heatmap (10×10)")
                .row_labels(rl).col_labels(cl)
                .theme(theme.clone()).build()
        }, 800, 500)?;

        // 40. Single data point (degenerate case)
        emit("Single Point",
            Charts::line(&[42.0])
                .title("Single Data Point")
                .x_label("X").y_label("Y")
                .theme(theme.clone()).build(),
            800, 500)?;

        eprintln!("  ✓ Theme: {theme_name} — {idx} charts");
    }

    // ---------------------------------------------------------------
    // Generate HTML gallery with feedback
    // ---------------------------------------------------------------
    let charts_per_theme = html_entries.len() / themes.len();
    let theme_names = ["dark", "light", "pastel", "ocean", "forest", "colorblind"];

    let mut html = String::with_capacity(256 * 1024);
    html.push_str(&format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scry Chart Gallery — All Themes</title>
<style>
  :root {{
    --bg: #0d1117;
    --card: #161b22;
    --border: #30363d;
    --fg: #c9d1d9;
    --dim: #8b949e;
    --accent: #58a6ff;
    --accent2: #7ee787;
    --input-bg: #0d1117;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: var(--bg); color: var(--fg);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    padding: 20px 24px;
    max-width: 1800px;
    margin: 0 auto;
  }}
  h1 {{
    text-align: center; margin: 24px 0 8px;
    font-size: 2.2em; font-weight: 700;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }}
  .meta {{
    text-align: center; color: var(--dim); font-size: 0.9em; margin-bottom: 16px;
  }}

  /* Theme filter nav */
  .filters {{
    display: flex; gap: 8px; justify-content: center;
    flex-wrap: wrap; margin: 16px 0 24px;
  }}
  .filters button {{
    background: var(--card); color: var(--fg); border: 1px solid var(--border);
    padding: 6px 16px; border-radius: 20px; cursor: pointer;
    font-size: 0.85em; transition: all 0.2s;
  }}
  .filters button:hover, .filters button.active {{
    background: var(--accent); color: #000; border-color: var(--accent);
  }}

  /* Section headers */
  h2 {{
    margin: 36px 0 16px; padding: 10px 0;
    border-bottom: 1px solid var(--border);
    color: var(--accent); font-size: 1.3em; font-weight: 600;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
    gap: 16px;
  }}

  /* Chart cards */
  .card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; overflow: hidden;
    transition: border-color 0.2s, box-shadow 0.2s;
  }}
  .card:hover {{
    border-color: var(--accent);
    box-shadow: 0 0 16px rgba(88,166,255,0.15);
  }}
  .card img {{ width: 100%; display: block; }}
  .card-info {{
    padding: 10px 14px;
    display: flex; justify-content: space-between; align-items: center;
  }}
  .card-info .label {{ font-size: 0.85em; color: var(--dim); }}
  .card-info .size {{ font-size: 0.75em; color: var(--dim); opacity: 0.7; }}

  /* Feedback textarea */
  .feedback {{
    padding: 0 14px 14px;
  }}
  .feedback textarea {{
    width: 100%; min-height: 50px; max-height: 200px;
    background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 8px 10px; font-size: 0.82em;
    font-family: inherit; resize: vertical;
    transition: border-color 0.2s;
  }}
  .feedback textarea:focus {{
    outline: none; border-color: var(--accent);
  }}
  .feedback textarea::placeholder {{ color: var(--dim); opacity: 0.6; }}
  .feedback textarea.has-content {{
    border-color: var(--accent2);
  }}

  /* Floating action bar */
  .fab {{
    position: fixed; bottom: 24px; right: 24px;
    display: flex; gap: 10px; align-items: center; z-index: 100;
  }}
  .fab button {{
    background: var(--accent); color: #000;
    border: none; padding: 12px 20px; border-radius: 28px;
    font-size: 0.9em; font-weight: 600; cursor: pointer;
    box-shadow: 0 4px 16px rgba(88,166,255,0.35);
    transition: all 0.2s;
  }}
  .fab button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(88,166,255,0.5);
  }}
  .fb-badge {{
    background: var(--accent2); color: #000;
    padding: 4px 10px; border-radius: 14px;
    font-size: 0.8em; font-weight: 600;
  }}

  /* Toast */
  .toast {{
    position: fixed; top: 24px; right: 24px;
    background: var(--accent2); color: #000;
    padding: 12px 20px; border-radius: 8px;
    font-weight: 600; font-size: 0.9em;
    opacity: 0; transform: translateY(-10px);
    transition: all 0.3s; pointer-events: none;
    z-index: 200;
  }}
  .toast.show {{ opacity: 1; transform: translateY(0); }}

  /* Theme sections */
  .theme-section {{ display: block; }}
  .theme-section.hidden {{ display: none; }}
</style>
</head>
<body>
<h1>Scry Chart Gallery</h1>
<p class="meta">{count} charts — {themes} themes × {per} chart types + edge cases</p>
"#,
    themes = theme_names.len(),
    per = charts_per_theme,
    count = count,
    ));

    // Filter buttons
    html.push_str("<div class=\"filters\">\n");
    html.push_str("<button class=\"active\" onclick=\"filterTheme('all')\">All Themes</button>\n");
    for tn in &theme_names {
        html.push_str(&format!(
            "<button onclick=\"filterTheme('{tn}')\">{}</button>\n",
            capitalize(tn)
        ));
    }
    html.push_str("</div>\n");

    // Group entries by theme
    for (ti, tn) in theme_names.iter().enumerate() {
        let start = ti * charts_per_theme;
        let end = start + charts_per_theme;
        let entries = &html_entries[start..end];

        html.push_str(&format!("<div class=\"theme-section\" data-theme=\"{tn}\">\n"));
        html.push_str(&format!("<h2>Theme: {}</h2>\n<div class=\"grid\">\n", capitalize(tn)));

        for entry in entries {
            let chart_id = &entry.name;
            html.push_str(&format!(
                r#"<div class="card">
<img src="{chart_id}.png" loading="lazy" alt="{label}">
<div class="card-info">
  <span class="label">{label}</span>
  <span class="size">{w}×{h}</span>
</div>
<div class="feedback">
  <textarea id="fb_{chart_id}" placeholder="Feedback for this chart…" oninput="onFeedback(this)"></textarea>
</div>
</div>
"#,
                label = entry.label,
                w = entry.width,
                h = entry.height,
            ));
        }
        html.push_str("</div>\n</div>\n");
    }

    // Floating action bar + toast
    html.push_str(r#"
<div class="fab">
  <span class="fb-badge" id="fbCount">0 feedback</span>
  <button onclick="exportFeedback()">⬇ Export Feedback</button>
</div>
<div class="toast" id="toast">Feedback exported!</div>

<script>
// Theme filter
function filterTheme(theme) {
  document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('.theme-section').forEach(s => {
    s.classList.toggle('hidden', theme !== 'all' && s.dataset.theme !== theme);
  });
}

// Track feedback
function onFeedback(el) {
  el.classList.toggle('has-content', el.value.trim().length > 0);
  updateCount();
}
function updateCount() {
  const n = document.querySelectorAll('textarea.has-content').length;
  document.getElementById('fbCount').textContent = n + ' feedback';
}

// Export all feedback as JSON download
function exportFeedback() {
  const feedback = {};
  document.querySelectorAll('textarea[id^="fb_"]').forEach(ta => {
    if (ta.value.trim()) {
      const id = ta.id.replace('fb_', '');
      const parts = id.split('_');
      const theme = parts[0];
      const chartType = parts.slice(2).join('_');
      feedback[id] = {
        theme: theme,
        chart_type: chartType,
        feedback: ta.value.trim()
      };
    }
  });
  const json = JSON.stringify(feedback, null, 2);
  const blob = new Blob([json], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'feedback.json';
  a.click();
  URL.revokeObjectURL(url);
  showToast();
}
function showToast() {
  const t = document.getElementById('toast');
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2500);
}
</script>
</body>
</html>
"#);

    std::fs::write(format!("{dir}/index.html"), &html).map_err(|e| e.to_string())?;

    eprintln!("\n✓ Gallery complete: {count} charts rendered to {dir}/");
    eprintln!("  Open: file://{dir}/index.html");
    eprintln!("  Each chart has a feedback textarea — click 'Export Feedback' to save as JSON");

    Ok(())
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().to_string() + c.as_str(),
    }
}
