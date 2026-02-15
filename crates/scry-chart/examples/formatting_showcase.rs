//! Formatting Showcase — exercises every Tier 2 formatting feature.
//!
//! Navigate: ← → or h/l     Quit: q
//!
//! ```bash
//! cargo run -p scry-chart --example formatting_showcase
//! ```

#![allow(
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::cast_possible_truncation
)]

use std::io::stdout;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use scry_chart::formatter::{
    CurrencyFormatter, FixedDecimalFormatter, FnFormatter, PercentFormatter, ScientificFormatter,
    SiFormatter,
};
use scry_chart::prelude::*;
use scry_engine::style::Color;

// ─────────────────────────────────────────────────────────────────────────────
// Page catalogue
// ─────────────────────────────────────────────────────────────────────────────

const NUM_PAGES: usize = 18;

struct PageInfo {
    section: &'static str,
    title: &'static str,
    features: &'static str,
}

const PAGES: [PageInfo; NUM_PAGES] = [
    // Section 1: Tick Formatters (0-5)
    PageInfo {
        section: "Fmt",
        title: "Currency Y-Axis ($K, $M)",
        features: "y_formatter(CurrencyFormatter)",
    },
    PageInfo {
        section: "Fmt",
        title: "Percent Y-Axis",
        features: "y_formatter(PercentFormatter)",
    },
    PageInfo {
        section: "Fmt",
        title: "SI Suffix Y-Axis (K, M, G)",
        features: "y_formatter(SiFormatter)",
    },
    PageInfo {
        section: "Fmt",
        title: "Fixed Decimal (2dp)",
        features: "y_formatter(FixedDecimalFormatter(2))",
    },
    PageInfo {
        section: "Fmt",
        title: "Scientific Notation",
        features: "y_formatter(ScientificFormatter)",
    },
    PageInfo {
        section: "Fmt",
        title: "Custom °C Formatter",
        features: "y_formatter(FnFormatter)",
    },
    // Section 2: Tick Steps (6-7)
    PageInfo {
        section: "Step",
        title: "Y-Axis Step = 25",
        features: "y_tick_step(25.0)",
    },
    PageInfo {
        section: "Step",
        title: "X Step = π, Y Step = 0.5",
        features: "x_tick_step(π) · y_tick_step(0.5)",
    },
    // Section 3: Grid Toggles (8-10)
    PageInfo {
        section: "Grid",
        title: "Y-Grid Only (Horizontal Lines)",
        features: "y_grid_only()",
    },
    PageInfo {
        section: "Grid",
        title: "X-Grid Only (Vertical Lines)",
        features: "x_grid_only()",
    },
    PageInfo {
        section: "Grid",
        title: "No Grid",
        features: "no_grid()",
    },
    // Section 4: Tick Rotation (11-12)
    PageInfo {
        section: "Rot",
        title: "Diagonal (45°) Tick Labels",
        features: "x_ticks_diagonal()",
    },
    PageInfo {
        section: "Rot",
        title: "Vertical (90°) Tick Labels",
        features: "x_ticks_vertical()",
    },
    // Section 5: Legend Features (13-17)
    PageInfo {
        section: "Leg",
        title: "Legend Title + Horizontal",
        features: "legend_title · legend_horizontal",
    },
    PageInfo {
        section: "Leg",
        title: "Multi-Column Legend (2 cols)",
        features: "legend_columns(2)",
    },
    PageInfo {
        section: "Leg",
        title: "Multi-Column Legend (3 cols)",
        features: "legend_columns(3)",
    },
    PageInfo {
        section: "Leg",
        title: "Custom Palette",
        features: "Theme::dark().with_palette()",
    },
    PageInfo {
        section: "Leg",
        title: "Combined: Fmt + Step + Grid + Rotation",
        features: "all Tier 2 features",
    },
];

// ─────────────────────────────────────────────────────────────────────────────
// Chart builders
// ─────────────────────────────────────────────────────────────────────────────

fn build_chart(idx: usize) -> Chart {
    match idx {
        // ── Currency formatter ───────────────────────────────────────
        0 => Chart::bar(
            vec![
                "2021".into(),
                "2022".into(),
                "2023".into(),
                "2024".into(),
                "2025".into(),
            ],
            &[250_000.0, 480_000.0, 720_000.0, 1_100_000.0, 1_850_000.0],
        )
        .title("Annual Revenue")
        .y_label("Revenue")
        .y_formatter(CurrencyFormatter::default())
        .show_values()
        .theme(Theme::dark())
        .build(),

        // ── Percent formatter ────────────────────────────────────────
        1 => {
            let x: Vec<f64> = (0..12).map(|i| i as f64).collect();
            let conv: Vec<f64> = x
                .iter()
                .map(|&v| 0.02 + v * 0.008 + (v * 0.5).sin() * 0.01)
                .collect();
            let bounce: Vec<f64> = x
                .iter()
                .map(|&v| 0.65 - v * 0.02 + (v * 0.3).cos() * 0.05)
                .collect();
            Chart::line(&conv)
                .title("Conversion & Bounce Rates")
                .x_label("Month")
                .y_label("Rate")
                .x_values(x)
                .add_named_series("Bounce Rate", &bounce)
                .y_formatter(PercentFormatter::default())
                .with_points()
                .theme(Theme::ocean())
                .build()
        }

        // ── SI suffix formatter ──────────────────────────────────────
        2 => Chart::line(&[
            100.0,
            1_500.0,
            12_000.0,
            85_000.0,
            320_000.0,
            1_200_000.0,
            5_800_000.0,
            22_000_000.0,
        ])
        .title("Exponential User Growth")
        .x_label("Quarter")
        .y_label("Users")
        .y_formatter(SiFormatter::default())
        .filled()
        .theme(Theme::forest())
        .build(),

        // ── Fixed decimal formatter ──────────────────────────────────
        3 => {
            let voltages: Vec<f64> = (0..20)
                .map(|i| 3.30 + (i as f64 * 0.1).sin() * 0.05)
                .collect();
            Chart::line(&voltages)
                .title("Voltage Monitor — Precision Readout")
                .x_label("Sample")
                .y_label("V")
                .y_formatter(FixedDecimalFormatter(2))
                .with_points()
                .h_line_styled(3.30, Color::from_rgba8(100, 255, 100, 150))
                .theme(Theme::dark())
                .build()
        }

        // ── Scientific notation ──────────────────────────────────────
        4 => {
            let data: Vec<f64> = (0..15).map(|i| (i as f64 * 0.5).exp() * 1e-6).collect();
            Chart::line(&data)
                .title("Quantum Decoherence Rate")
                .x_label("Time (ns)")
                .y_label("Rate (s⁻¹)")
                .y_formatter(ScientificFormatter { precision: 2 })
                .smooth()
                .theme(Theme::dark())
                .build()
        }

        // ── Custom FnFormatter ───────────────────────────────────────
        5 => {
            let temps: Vec<f64> = (0..24)
                .map(|h| {
                    15.0 + 12.0
                        * ((h as f64 - 14.0) * std::f64::consts::PI / 12.0)
                            .cos()
                            .abs()
                        - 5.0 * (h as f64 / 24.0)
                })
                .collect();
            Chart::line(&temps)
                .title("24-Hour Temperature Profile")
                .x_label("Hour")
                .y_label("Temperature")
                .y_formatter(FnFormatter::new(|v| format!("{v:.1}°C")))
                .x_formatter(FnFormatter::new(|v| format!("{:.0}h", v)))
                .filled()
                .theme(Theme::pastel())
                .build()
        }

        // ── Y tick step = 25 ─────────────────────────────────────────
        6 => Chart::bar(
            vec![
                "Mon".into(),
                "Tue".into(),
                "Wed".into(),
                "Thu".into(),
                "Fri".into(),
            ],
            &[42.0, 78.0, 55.0, 93.0, 67.0],
        )
        .title("Daily Tasks Completed")
        .y_label("Tasks")
        .y_tick_step(25.0)
        .y_range(0.0, 100.0)
        .show_values()
        .theme(Theme::ocean())
        .build(),

        // ── X step = π, Y step = 0.5 ────────────────────────────────
        7 => {
            let x: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
            let y: Vec<f64> = x.iter().map(|&v| v.sin()).collect();
            Chart::line_xy(&x, &y)
                .title("sin(x) with π-Spaced X Ticks")
                .x_label("x (radians)")
                .y_label("sin(x)")
                .x_tick_step(std::f64::consts::PI)
                .y_tick_step(0.5)
                .x_formatter(FnFormatter::new(|v| {
                    let pi = std::f64::consts::PI;
                    let n = (v / pi).round();
                    if n.abs() < 0.01 {
                        "0".to_string()
                    } else if (n - 1.0).abs() < 0.01 {
                        "π".to_string()
                    } else if (n + 1.0).abs() < 0.01 {
                        "-π".to_string()
                    } else {
                        format!("{n:.0}π")
                    }
                }))
                .smooth()
                .theme(Theme::dark())
                .build()
        }

        // ── Y-grid only ──────────────────────────────────────────────
        8 => {
            let y: Vec<f64> = (0..30)
                .map(|i| (i as f64 * 0.3).sin() * 5.0 + 10.0)
                .collect();
            Chart::line(&y)
                .title("Y-Grid Only — Clean Horizontal Lines")
                .x_label("Sample")
                .y_label("Value")
                .y_grid_only()
                .filled()
                .theme(Theme::dark())
                .build()
        }

        // ── X-grid only ──────────────────────────────────────────────
        9 => {
            let y: Vec<f64> = (0..30)
                .map(|i| (i as f64 * 0.2).cos() * 8.0 + 12.0)
                .collect();
            Chart::line(&y)
                .title("X-Grid Only — Vertical Reference Lines")
                .x_label("Time")
                .y_label("Amplitude")
                .x_grid_only()
                .with_points()
                .theme(Theme::ocean())
                .build()
        }

        // ── No grid ─────────────────────────────────────────────────
        10 => {
            let (sx, sy) = scatter_data();
            Chart::scatter(&sx, &sy)
                .title("Scatter — No Grid (Minimal)")
                .x_label("X")
                .y_label("Y")
                .no_grid()
                .marker(Marker::Circle)
                .connected()
                .theme(Theme::pastel())
                .build()
        }

        // ── Diagonal tick rotation ───────────────────────────────────
        11 => Chart::bar(
            vec![
                "January".into(),
                "February".into(),
                "March".into(),
                "April".into(),
                "May".into(),
                "June".into(),
                "July".into(),
                "August".into(),
            ],
            &[120.0, 200.0, 150.0, 280.0, 190.0, 310.0, 250.0, 340.0],
        )
        .title("Monthly Revenue — Diagonal Labels")
        .y_label("Revenue ($K)")
        .x_ticks_diagonal()
        .y_formatter(CurrencyFormatter {
            symbol: "$".into(),
            decimals: 0,
            si_suffixes: false,
            accounting: false,
        })
        .theme(Theme::dark())
        .build(),

        // ── Vertical tick rotation ───────────────────────────────────
        12 => {
            let labels: Vec<String> = (1..=12)
                .map(|m| {
                    [
                        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                        "Nov", "Dec",
                    ][m - 1]
                        .into()
                })
                .collect();
            let vals: Vec<f64> = (1..=12)
                .map(|m| 50.0 + (m as f64 * 0.5).sin() * 30.0)
                .collect();
            Chart::bar(labels, &vals)
                .title("12 Months — Vertical Labels")
                .y_label("Units")
                .x_ticks_vertical()
                .theme(Theme::forest())
                .build()
        }

        // ── Legend title + horizontal ────────────────────────────────
        13 => {
            let x = linspace(0.0, 10.0, 40);
            let y1: Vec<f64> = x.iter().map(|&v| v.sin() * 3.0 + 5.0).collect();
            let y2: Vec<f64> = x.iter().map(|&v| v.cos() * 3.0 + 5.0).collect();
            let y3: Vec<f64> = x.iter().map(|&v| (v * 1.5).sin() * 2.0 + 5.0).collect();
            Chart::line(&y1)
                .title("Waveforms — Horizontal Legend with Title")
                .x_label("Time (s)")
                .y_label("Amplitude")
                .x_values(x)
                .add_named_series("cos(t)", &y2)
                .add_named_series("sin(1.5t)", &y3)
                .legend_title("Signals")
                .legend_horizontal()
                .legend_position(LegendPosition::Bottom)
                .with_points()
                .theme(Theme::dark())
                .build()
        }

        // ── Multi-column legend (2 cols) ─────────────────────────────
        14 => {
            let base: Vec<f64> = (0..20).map(|i| i as f64).collect();
            let mut builder =
                Chart::line(&base.iter().map(|&v| v * 1.0 + 10.0).collect::<Vec<_>>())
                    .title("6-Series Chart — 2-Column Legend")
                    .x_label("X")
                    .y_label("Y");

            for i in 1..6 {
                let series: Vec<f64> = base
                    .iter()
                    .map(|&v| v * (1.0 + i as f64 * 0.3) + 10.0 + i as f64 * 5.0)
                    .collect();
                builder = builder
                    .add_named_series(&format!("Series {}", (b'A' + i as u8) as char), &series);
            }

            builder
                .legend_columns(2)
                .legend_title("Data Sets")
                .y_grid_only()
                .theme(Theme::dark())
                .build()
        }

        // ── Multi-column legend (3 cols) ─────────────────────────────
        15 => {
            let labels: Vec<String> = (0..9)
                .map(|i| format!("Cat {}", (b'A' + i as u8) as char))
                .collect();
            let vals: Vec<f64> = vec![85.0, 92.0, 78.0, 95.0, 88.0, 72.0, 91.0, 83.0, 76.0];
            let mut builder =
                Chart::bar(labels, &vals).title("Performance Scores — 3-Column Legend");

            let extra: Vec<f64> = vec![70.0, 85.0, 90.0, 65.0, 78.0, 88.0, 82.0, 75.0, 94.0];
            builder = builder.add_named_series("Q2", &extra);
            let extra2: Vec<f64> = vec![88.0, 76.0, 82.0, 90.0, 72.0, 95.0, 68.0, 92.0, 80.0];
            builder = builder.add_named_series("Q3", &extra2);

            builder
                .legend_columns(3)
                .legend_title("Quarters")
                .legend_position(LegendPosition::BottomLeft)
                .y_formatter(FnFormatter::new(|v| format!("{v:.0}%")))
                .y_range(0.0, 100.0)
                .theme(Theme::pastel())
                .build()
        }

        // ── Custom palette ───────────────────────────────────────────
        16 => {
            let x = linspace(0.0, 8.0, 30);
            let y1: Vec<f64> = x.iter().map(|&v| v.sin() * 4.0 + 6.0).collect();
            let y2: Vec<f64> = x.iter().map(|&v| (v + 1.0).cos() * 3.0 + 8.0).collect();
            let y3: Vec<f64> = x.iter().map(|&v| (v * 0.7).sin() * 2.5 + 5.0).collect();

            let custom_theme = Theme::dark().with_palette(vec![
                Color::from_rgba8(255, 107, 107, 255), // coral red
                Color::from_rgba8(78, 205, 196, 255),  // teal
                Color::from_rgba8(255, 230, 109, 255), // golden yellow
            ]);

            Chart::line(&y1)
                .title("Custom Palette — Coral / Teal / Gold")
                .x_label("X")
                .y_label("Y")
                .x_values(x)
                .add_named_series("Teal", &y2)
                .add_named_series("Gold", &y3)
                .smooth()
                .filled()
                .with_points()
                .legend_title("Custom Colors")
                .theme(custom_theme)
                .build()
        }

        // ── Combined: all features ───────────────────────────────────
        17 => {
            let x = linspace(0.0, 100.0, 50);
            let revenue: Vec<f64> = x
                .iter()
                .map(|&v| 500_000.0 + v * 15_000.0 + (v * 0.1).sin() * 80_000.0)
                .collect();
            let costs: Vec<f64> = x
                .iter()
                .map(|&v| 300_000.0 + v * 8_000.0 + (v * 0.15).cos() * 40_000.0)
                .collect();
            let margin: Vec<f64> = revenue
                .iter()
                .zip(costs.iter())
                .map(|(r, c)| (r - c) / r)
                .collect();

            // Can't mix different y-axis formatters, so show revenue+costs
            Chart::line(&revenue)
                .title("Combined Showcase — All Tier 2 Features")
                .x_label("Day")
                .y_label("Amount")
                .x_values(x)
                .add_named_series("Costs", &costs)
                .add_named_series(
                    "Margin (scaled)",
                    &margin.iter().map(|v| v * 2_000_000.0).collect::<Vec<_>>(),
                )
                .y_formatter(CurrencyFormatter::default())
                .x_tick_step(20.0)
                .y_grid_only()
                .x_ticks_diagonal()
                .legend_title("Financials")
                .legend_horizontal()
                .legend_position(LegendPosition::Bottom)
                .filled()
                .with_points()
                .h_line_styled(1_000_000.0, Color::from_rgba8(255, 100, 100, 180))
                .theme(Theme::dark())
                .build()
        }

        _ => unreachable!(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| lo + (hi - lo) * i as f64 / (n - 1).max(1) as f64)
        .collect()
}

fn scatter_data() -> (Vec<f64>, Vec<f64>) {
    let sx: Vec<f64> = (0..40).map(|i| i as f64 * 0.5).collect();
    let sy: Vec<f64> = sx
        .iter()
        .map(|&v| v.sqrt() * 3.0 + (v * 0.7).sin() * 2.0)
        .collect();
    (sx, sy)
}

// ─────────────────────────────────────────────────────────────────────────────
// Main TUI loop
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let mut state = ChartState::auto();
    let mut page: usize = 0;
    let mut prev_page: usize = usize::MAX;

    loop {
        terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(2)])
                .split(frame.area());

            let chart = build_chart(page);
            frame.render_stateful_widget(ChartWidget::new(&chart), chunks[0], &mut state);

            let info = &PAGES[page];
            let status = Paragraph::new(format!(
                " {:>2}/{NUM_PAGES} │ [{}] {} │ {} │ ← → navigate · q quit",
                page + 1,
                info.section,
                info.title,
                info.features,
            ))
            .block(Block::default().borders(Borders::TOP))
            .style(Style::default().fg(ratatui::style::Color::DarkGray));
            frame.render_widget(status, chunks[1]);
        })?;

        if page != prev_page {
            state.cleanup();
            prev_page = page;
        }
        state.flush()?;

        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Right | KeyCode::Char('l') => {
                        page = (page + 1) % NUM_PAGES;
                    }
                    KeyCode::Left | KeyCode::Char('h') => {
                        page = (page + NUM_PAGES - 1) % NUM_PAGES;
                    }
                    KeyCode::Home => page = 0,
                    KeyCode::End => page = NUM_PAGES - 1,
                    KeyCode::Char('1') => page = 0,  // Formatters
                    KeyCode::Char('2') => page = 6,  // Tick Steps
                    KeyCode::Char('3') => page = 8,  // Grid
                    KeyCode::Char('4') => page = 11, // Rotation
                    KeyCode::Char('5') => page = 13, // Legend
                    KeyCode::Char('6') => page = 16, // Palette
                    KeyCode::Char('7') => page = 17, // Combined
                    _ => {}
                }
            }
        }
    }

    state.cleanup();
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}
