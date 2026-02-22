// SPDX-License-Identifier: MIT OR Apache-2.0
//! `scry fetch` — animated system-info display.
//!
//! A fully native, highly configurable replacement for fastfetch.
//! Config lives at `~/.config/scry/fetch.toml` (auto-created on first run).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::unreadable_literal
)]

use std::f32::consts::{FRAC_PI_3, TAU};
use std::io::{stdout, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode},
    ExecutableCommand,
};
use ratatui::prelude::*;
use ratatui::widgets::{Cell, Paragraph, Row, StatefulWidget, Table};
use serde::Deserialize;

use scry_engine::prelude::{Picker, PixelCanvasState, PixelCanvasWidget};
use scry_engine::scene::PixelCanvas;
use scry_engine::style::Color as C;
use scry_engine::transport::ProtocolKind;

use super::sysinfo_fetch::SysInfo;

// ═══════════════════════════════════════════════════════════════════
//  Built-in palette
// ═══════════════════════════════════════════════════════════════════

const SOFT_BLUE: C   = C::from_rgb8(158, 193, 255);
const SOFT_PINK: C   = C::from_rgb8(242, 181, 212);
const SOFT_PURPLE: C = C::from_rgb8(203, 182, 255);
const SOFT_GREEN: C  = C::from_rgb8(168, 213, 186);
const SOFT_PEACH: C  = C::from_rgb8(244, 192, 149);
const SOFT_CREAM: C  = C::from_rgb8(243, 231, 179);

const ANIM_PALETTE: [C; 6] = [
    SOFT_BLUE, SOFT_PINK, SOFT_PURPLE, SOFT_GREEN, SOFT_PEACH, SOFT_CREAM,
];

// ═══════════════════════════════════════════════════════════════════
//  Config types
// ═══════════════════════════════════════════════════════════════════

/// The default `fetch.toml` written on first run (embedded at compile time).
const DEFAULT_CONFIG_TOML: &str = include_str!("fetch_default.toml");

// ── Deserialisable structs ───────────────────────────────────────────

#[derive(Debug, Deserialize, Clone, Default)]
#[serde(default)]
pub struct FetchConfig {
    pub display:    DisplayConfig,
    pub theme:      ThemeConfig,
    pub fields:     FieldsConfig,
    /// Per-field label/icon/color, keyed by `id`.
    #[serde(rename = "field")]
    pub field_defs: Vec<FieldDef>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct DisplayConfig {
    /// Animation preset: `"geometry"` | `"none"`.
    pub animation: String,
    /// Auto-exit after this many seconds (0 = press key only).
    pub duration:  u64,
    /// Ratatui inline-viewport height (0 = auto).
    pub rows:      u16,
    /// Separator character under `user@host`.
    pub separator: String,
}

impl Default for DisplayConfig {
    fn default() -> Self {
        Self {
            animation: "geometry".into(),
            duration:  3,
            rows:      0,
            separator: "─".into(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct ThemeConfig {
    pub title_user: String,
    pub title_host: String,
    pub title_at:   String,
    pub separator:  String,
    pub icon:       String,
    pub value:      String,
}

impl Default for ThemeConfig {
    fn default() -> Self {
        Self {
            title_user: "#A8D5BA".into(),
            title_host: "#F2B5D4".into(),
            title_at:   "#787268".into(),
            separator:  "#CBB6FF".into(),
            icon:       "#CBB6FF".into(),
            value:      "#E8E2D7".into(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct FieldsConfig {
    pub order: Vec<String>,
}

impl Default for FieldsConfig {
    fn default() -> Self {
        Self {
            order: vec![
                "os".into(), "kernel".into(), "uptime".into(), "shell".into(),
                "terminal".into(), "de_wm".into(), "packages".into(),
                "memory".into(), "cpu".into(),
            ],
        }
    }
}

#[derive(Debug, Deserialize, Clone, Default)]
#[serde(default)]
pub struct FieldDef {
    pub id:    String,
    pub label: String,
    pub icon:  String,
    pub color: String,
}

// ── Built-in field defaults (populated when no config exists) ────────

fn builtin_field(id: &str) -> FieldDef {
    let (label, icon, color) = match id {
        "os"       => ("OS",       "󰀧",  "#A8D5BA"),
        "kernel"   => ("Kernel",   "",  "#9EC1FF"),
        "uptime"   => ("Uptime",   "󰔟",  "#F4C095"),
        "shell"    => ("Shell",    "",  "#F3E7B3"),
        "terminal" => ("Terminal", "",  "#CBB6FF"),
        "de_wm"    => ("DE / WM",  "󰖲",  "#F2B5D4"),
        "packages" => ("Packages", "󰏗",  "#A8D5BA"),
        "memory"   => ("Memory",   "",  "#9EC1FF"),
        "cpu"      => ("CPU",      "󰘚",  "#CBB6FF"),
        other      => (other, "●", "#E8E2D7"),
    };
    FieldDef { id: id.into(), label: label.into(), icon: icon.into(), color: color.into() }
}

// ── Colour helpers ───────────────────────────────────────────────────

/// Parse a `#RRGGBB` hex string into a ratatui colour.
fn parse_hex(s: &str) -> ratatui::style::Color {
    let s = s.trim_start_matches('#');
    if s.len() == 6 {
        if let (Ok(r), Ok(g), Ok(b)) = (
            u8::from_str_radix(&s[0..2], 16),
            u8::from_str_radix(&s[2..4], 16),
            u8::from_str_radix(&s[4..6], 16),
        ) {
            return ratatui::style::Color::Rgb(r, g, b);
        }
    }
    ratatui::style::Color::Reset
}

// ═══════════════════════════════════════════════════════════════════
//  Config loading
// ═══════════════════════════════════════════════════════════════════

fn default_config_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".config/scry/fetch.toml")
}

fn load_config(custom: Option<&Path>) -> FetchConfig {
    let path = custom
        .map(PathBuf::from)
        .unwrap_or_else(default_config_path);

    if !path.exists() {
        // Auto-create on first run
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&path, DEFAULT_CONFIG_TOML);
        eprintln!("scry fetch: created config at {}", path.display());
    }

    let content = std::fs::read_to_string(&path).unwrap_or_default();
    toml::from_str(&content).unwrap_or_else(|e| {
        eprintln!("scry fetch: config parse error ({e}), using defaults");
        default_config()
    })
}

fn default_config() -> FetchConfig {
    toml::from_str(DEFAULT_CONFIG_TOML).expect("built-in config is valid TOML")
}

// ═══════════════════════════════════════════════════════════════════
//  CLI
// ═══════════════════════════════════════════════════════════════════

/// Show animated system information (add `scry fetch` to .bashrc / .zshrc).
#[derive(Debug, clap::Args)]
pub struct FetchArgs {
    /// Path to fetch.toml config (default: ~/.config/scry/fetch.toml)
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Skip animation, just print system info as plain text
    #[arg(long)]
    pub no_anim: bool,

    /// Animation preset: geometry | none
    #[arg(long)]
    pub preset: Option<String>,

    /// Auto-exit after N seconds (0 = press key only)
    #[arg(long)]
    pub duration: Option<u64>,

    /// Inline viewport height in terminal rows (0 = auto)
    #[arg(long)]
    pub rows: Option<u16>,
}

// ═══════════════════════════════════════════════════════════════════
//  Entry point
// ═══════════════════════════════════════════════════════════════════

pub fn run(args: &FetchArgs) -> Result<(), String> {
    let mut cfg = load_config(args.config.as_deref());

    // CLI flags override config
    if args.no_anim {
        cfg.display.animation = "none".into();
    }
    if let Some(preset) = &args.preset {
        cfg.display.animation = preset.clone();
    }
    if let Some(dur) = args.duration {
        cfg.display.duration = dur;
    }
    if let Some(rows) = args.rows {
        cfg.display.rows = rows;
    }

    // Collect system info
    let info = SysInfo::collect();

    // Build ordered rows from config
    let rows: Vec<(String, String, String)> = build_rows(&cfg, &info);

    let picker = Picker::detect();
    let protocol = picker.protocol();

    // Three-way dispatch:
    //  • "none" or Halfblock → instant plain print, no raw mode (fastfetch behaviour)
    //  • Native              → print text + IPC overlay animation, no raw mode
    //  • Kitty / Sixel       → ratatui inline viewport animated path
    if cfg.display.animation == "none" || protocol == ProtocolKind::Halfblock {
        run_plain(&cfg, &info, &rows);
        return Ok(());
    }

    if protocol == ProtocolKind::Native {
        return run_native_animated(&cfg, &info, &rows, picker)
            .map_err(|e| e.to_string());
    }

    run_animated(&cfg, &info, &rows).map_err(|e| e.to_string())
}

// ═══════════════════════════════════════════════════════════════════
//  Row builder  (icon, label, value)
// ═══════════════════════════════════════════════════════════════════

fn build_rows(cfg: &FetchConfig, info: &SysInfo) -> Vec<(String, String, String)> {
    // Build id → FieldDef lookup from config
    let mut lookup: std::collections::HashMap<String, FieldDef> = cfg
        .field_defs
        .iter()
        .map(|f| (f.id.clone(), f.clone()))
        .collect();

    // Build id → value lookup from SysInfo
    let values: std::collections::HashMap<&str, &str> = [
        ("os",       info.os.as_str()),
        ("kernel",   info.kernel.as_str()),
        ("uptime",   info.uptime.as_str()),
        ("shell",    info.shell.as_str()),
        ("terminal", info.terminal.as_str()),
        ("de_wm",    info.de_wm.as_str()),
        ("packages", info.packages.as_str()),
        ("memory",   info.memory.as_str()),
        ("cpu",      info.cpu.as_str()),
    ]
    .into_iter()
    .collect();

    cfg.fields
        .order
        .iter()
        .filter_map(|id| {
            let def = lookup.remove(id.as_str()).unwrap_or_else(|| builtin_field(id));
            let value = values.get(id.as_str()).copied().unwrap_or("?");
            Some((def.icon, def.label, value.to_string()))
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════
//  Plain-text mode  (--no-anim)
// ═══════════════════════════════════════════════════════════════════

fn run_plain(cfg: &FetchConfig, info: &SysInfo, rows: &[(String, String, String)]) {
    let t = &cfg.theme;
    println!();
    println!("  {}", info.user_at_host);
    let sep = cfg.display.separator.repeat(info.user_at_host.chars().count() + 2);
    println!("  {sep}");
    for (icon, label, value) in rows {
        println!("  ▎ {icon} {label:<10}  {value}");
    }
    println!();
    // Suppress unused warning on theme in plain mode
    let _ = t;
}

// ═══════════════════════════════════════════════════════════════════
//  Native animated mode (scry-terminal: IPC overlay, NO raw mode)
// ═══════════════════════════════════════════════════════════════════

/// Runs inside `scry-terminal`.  Prints sysinfo text to stdout immediately
/// (stays in scrollback), then drives an IPC overlay animation for the
/// configured duration — zero raw mode, zero keyboard interception.
fn run_native_animated(
    cfg: &FetchConfig,
    info: &SysInfo,
    rows: &[(String, String, String)],
    picker: Picker,
) -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Print sysinfo to stdout immediately ──────────────────────
    //    Record the row BEFORE printing so the overlay lines up.
    let (_, start_row) = crossterm::cursor::position().unwrap_or((0, 0));

    {
        let stdout = std::io::stdout();
        let mut w = BufWriter::new(stdout.lock());
        let t = &cfg.theme;
        writeln!(w)?;
        writeln!(w, "  {}", info.user_at_host)?;
        let sep = cfg.display.separator.repeat(info.user_at_host.chars().count() + 2);
        writeln!(w, "  {sep}")?;
        for (icon, label, value) in rows {
            writeln!(w, "  \u{258e} {icon} {label:<10}  {value}")?;
        }
        writeln!(w)?;
        let _ = t; // theme used by animated path only
    }

    let (_, end_row) = crossterm::cursor::position().unwrap_or((0, 0));
    let display_height = end_row.saturating_sub(start_row).max(4);

    // ── 2. Set up animation overlay ─────────────────────────────────
    let font = picker.font_size();
    let backend = picker.create_backend();
    let mut px_state = PixelCanvasState::new(backend, font);

    // Animation column width: same formula as ratatui path
    let (term_cols, _) = crossterm::terminal::size().unwrap_or((120, 40));
    let anim_cols = display_height
        .saturating_mul(2)
        .min(term_cols / 3)
        .max(14);

    // Overlay positioned at the text we just printed
    let area = ratatui::layout::Rect {
        x: 0,
        y: start_row,
        width: anim_cols,
        height: display_height,
    };

    let duration = if cfg.display.duration > 0 {
        Duration::from_secs(cfg.display.duration)
    } else {
        Duration::from_secs(3)
    };

    // ── 3. Animation loop — no raw mode, no event polling ───────────
    let start = Instant::now();
    while start.elapsed() < duration {
        let t = start.elapsed().as_secs_f32();
        let canvas = build_anim_scene(area, &px_state, t);

        // Render into a scratch buffer (only needed for position metadata;
        // NativeBackend ignores buffer cells and uses the IPC position from `area`).
        let mut buf = ratatui::buffer::Buffer::empty(area);
        PixelCanvasWidget::new(canvas)
            .z_index(-1)
            .skip_cache()
            .render(area, &mut buf, &mut px_state);
        px_state.flush()?;

        std::thread::sleep(Duration::from_millis(33)); // ~30 fps
    }

    // ── 4. Clean up overlay ─────────────────────────────────────────
    px_state.cleanup();
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════
//  Animated mode — Kitty / Sixel (ratatui inline viewport)
// ═══════════════════════════════════════════════════════════════════

fn run_animated(
    cfg: &FetchConfig,
    info: &SysInfo,
    rows: &[(String, String, String)],
) -> Result<(), Box<dyn std::error::Error>> {
    // Viewport height: config rows, else auto from content.
    // +6 headroom: 3 header rows + rows + 2 padding + 1 safety margin.
    let display_height = if cfg.display.rows > 0 {
        cfg.display.rows
    } else {
        (rows.len() as u16 + 6).max(10)
    };

    enable_raw_mode()?;
    stdout().execute(crossterm::cursor::Hide)?;

    let options = ratatui::TerminalOptions {
        viewport: ratatui::Viewport::Inline(display_height),
    };
    let mut terminal = Terminal::with_options(CrosstermBackend::new(stdout()), options)?;

    // Set up scry-engine pixel canvas.
    // Picker::detect() auto-selects the best available protocol:
    //   - Native (zero-copy IPC) inside scry-terminal
    //   - Kitty, Sixel, iTerm2, or Halfblock depending on the host terminal.
    let picker = Picker::detect();
    let backend = picker.create_backend();
    let mut px_state = PixelCanvasState::new(backend, picker.font_size());

    let start = Instant::now();
    let duration_limit = if cfg.display.duration > 0 {
        Some(Duration::from_secs(cfg.display.duration))
    } else {
        None
    };

    let title  = info.user_at_host.clone();
    let theme  = cfg.theme.clone();
    let sep_ch = cfg.display.separator.clone();

    loop {
        let elapsed = start.elapsed();
        let t = elapsed.as_secs_f32();

        terminal.draw(|frame: &mut ratatui::Frame| {
            let area = frame.area();

            // Layout: [animation | spacer | sysinfo]
            let logo_cols = display_height
                .saturating_mul(2)
                .min(area.width / 3)
                .max(14);
            let h_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Length(logo_cols),
                    Constraint::Length(1),
                    Constraint::Min(1),
                ])
                .split(area);

            let anim_area = h_chunks[0];
            let text_area = h_chunks[2];

            // ─── Left: animation ───
            let canvas = build_anim_scene(anim_area, &px_state, t);
            frame.render_stateful_widget(
                PixelCanvasWidget::new(canvas).z_index(-1).skip_cache(),
                anim_area,
                &mut px_state,
            );

            // ─── Right: sysinfo ───
            let v = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(3), Constraint::Min(1)])
                .split(text_area);

            frame.render_widget(
                Paragraph::new(build_header(&title, &theme, &sep_ch, t)),
                v[0],
            );
            frame.render_widget(build_info_table(rows, &theme, t), v[1]);
        })?;
        px_state.flush()?;

        // Auto-exit when duration elapses
        if let Some(limit) = duration_limit {
            if elapsed >= limit {
                break;
            }
        }

        // Key handler — q / Esc always exit
        if event::poll(Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press
                    && matches!(key.code, KeyCode::Char('q') | KeyCode::Esc)
                {
                    break;
                }
            }
        }
    }

    px_state.cleanup();
    disable_raw_mode()?;
    stdout().execute(crossterm::cursor::Show)?;
    println!();

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════
//  UI builders
// ═══════════════════════════════════════════════════════════════════

fn build_header(
    title: &str,
    theme: &ThemeConfig,
    sep_ch: &str,
    t: f32,
) -> Vec<Line<'static>> {
    let user_col = parse_hex(&theme.title_user);
    let host_col = parse_hex(&theme.title_host);
    let at_col   = parse_hex(&theme.title_at);
    let sep_col  = parse_hex(&theme.separator);

    let mut out = vec![Line::default()];

    let fade = (t / 0.3).clamp(0.0, 1.0);
    let title_line = if fade > 0.0 {
        let parts: Vec<&str> = title.splitn(2, '@').collect();
        if parts.len() == 2 {
            Line::from(vec![
                Span::raw("  "),
                Span::styled(parts[0].to_string(), Style::default().fg(user_col).bold()),
                Span::styled("@",                  Style::default().fg(at_col)),
                Span::styled(parts[1].to_string(), Style::default().fg(host_col).bold()),
            ])
        } else {
            Line::from(vec![
                Span::raw("  "),
                Span::styled(title.to_string(), Style::default().fg(user_col).bold()),
            ])
        }
    } else {
        Line::default()
    };
    out.push(title_line);

    // Separator rule
    let rule = sep_ch.repeat(title.chars().count() + 2);
    out.push(Line::from(vec![
        Span::raw("  "),
        Span::styled(rule, Style::default().fg(sep_col)),
    ]));

    out
}

/// Build the aligned info table using ratatui's Table widget.
///
/// Column layout (guaranteed pixel-exact alignment):
/// ```
///  ┌─ col 0 ─┬─ col 1 ─┬─ col 2 ─────┬─ col 3 ─────────────────┐
///  │ " ▎ " 4 │ icon  3 │ label    10 │ value              Min     │
///  └─────────┴─────────┴─────────────┴──────────────────────────-┘
/// ```
fn build_info_table(
    rows: &[(String, String, String)],
    theme: &ThemeConfig,
    t: f32,
) -> Table<'static> {
    let sep_col  = parse_hex(&theme.separator);
    let icon_col = parse_hex(&theme.icon);
    let val_col  = parse_hex(&theme.value);

    // Default label color cycle (overridden per-field via config color)
    let default_label_colors = [
        ratatui::style::Color::Rgb(168, 213, 186),
        ratatui::style::Color::Rgb(158, 193, 255),
        ratatui::style::Color::Rgb(244, 192, 149),
        ratatui::style::Color::Rgb(243, 231, 179),
        ratatui::style::Color::Rgb(203, 182, 255),
        ratatui::style::Color::Rgb(242, 181, 212),
    ];

    let table_rows: Vec<Row<'static>> = rows
        .iter()
        .enumerate()
        .map(|(i, (icon, label, value))| {
            let alpha = ((t - i as f32 * 0.12) / 0.25).clamp(0.0, 1.0);
            if alpha <= 0.0 {
                return Row::new(vec![
                    Cell::from(""),
                    Cell::from(""),
                    Cell::from(""),
                    Cell::from(""),
                ]);
            }
            let label_col = default_label_colors[i % default_label_colors.len()];
            Row::new(vec![
                Cell::from(Span::styled(" ▎ ", Style::default().fg(sep_col))),
                Cell::from(Span::styled(icon.clone(),  Style::default().fg(icon_col))),
                Cell::from(Span::styled(label.clone(), Style::default().fg(label_col).bold())),
                Cell::from(Span::styled(value.clone(), Style::default().fg(val_col))),
            ])
        })
        .collect();

    Table::new(
        table_rows,
        [
            Constraint::Length(4),  // " ▎ "
            Constraint::Length(3),  // icon
            Constraint::Length(10), // label
            Constraint::Min(1),     // value
        ],
    )
    .column_spacing(1)
}

// ═══════════════════════════════════════════════════════════════════
//  Animation — sacred geometry
// ═══════════════════════════════════════════════════════════════════

fn build_anim_scene(area: Rect, px_state: &PixelCanvasState, t: f32) -> PixelCanvas {
    let font = px_state.font_size();
    let w = u32::from(area.width) * u32::from(font.width);
    let h = u32::from(area.height) * u32::from(font.height);
    if w == 0 || h == 0 {
        return PixelCanvas::new(1, 1);
    }

    let cx     = w as f32 / 2.0;
    let cy     = h as f32 / 2.0;
    let radius = (w.min(h) as f32) * 0.42;
    let mut canvas = PixelCanvas::new(w, h);

    let cycle = 6.0_f32;
    let phase = t % cycle;
    let envelope = if phase < 1.0 { phase } else if phase < 5.0 { 1.0 } else { cycle - phase };

    let intro    = (phase / 0.6).min(1.0) * envelope;
    let flower   = ((phase - 0.2) / 0.5).clamp(0.0, 1.0) * envelope;
    let geometry = ((phase - 0.4) / 0.5).clamp(0.0, 1.0) * envelope;
    let radiance = ((phase - 0.7) / 0.3).clamp(0.0, 1.0) * envelope;
    let rot    = t * 0.4;
    let breath = 0.03f32.mul_add((t * 1.8).sin(), 1.0);

    // Background glow
    if intro > 0.0 {
        let ga = intro * 0.08;
        canvas = canvas.circle(cx, cy, radius * 1.3 * breath).fill(SOFT_PURPLE.with_alpha(ga)).done();
        canvas = canvas.circle(cx, cy, radius * 1.1 * breath).fill(SOFT_BLUE.with_alpha(ga * 1.5)).done();
    }

    // Concentric rings
    if intro > 0.0 {
        for i in 0..3 {
            let rr    = radius * (1.0 - i as f32 * 0.08) * breath;
            let alpha = intro * (0.4 - i as f32 * 0.1);
            canvas = canvas.circle(cx, cy, rr).stroke(anim_palette(i, t).with_alpha(alpha), 1.2).done();
        }
    }

    // Flower of Life
    if flower > 0.0 {
        let r = radius / 4.0;
        let mut centers: Vec<(f32, f32, usize)> = vec![(cx, cy, 0)];
        for i in 0..6 {
            let a = i as f32 * FRAC_PI_3 + rot;
            centers.push(((r * breath).mul_add(a.cos(), cx), (r * breath).mul_add(a.sin(), cy), 1));
        }
        for i in 0..6 {
            let a = i as f32 * FRAC_PI_3 + rot;
            centers.push(((2.0 * r * breath).mul_add(a.cos(), cx), (2.0 * r * breath).mul_add(a.sin(), cy), 2));
        }
        for i in 0..6 {
            let a = (i as f32).mul_add(FRAC_PI_3, FRAC_PI_3 / 2.0) + rot;
            let s = 3.0_f32.sqrt();
            centers.push(((s * r * breath).mul_add(a.cos(), cx), (s * r * breath).mul_add(a.sin(), cy), 2));
        }
        let rev = flower * 3.0;
        for &(x, y, ring) in &centers {
            let rp = (rev - ring as f32).clamp(0.0, 1.0);
            if rp <= 0.0 { continue; }
            let cur_r = r * rp;
            if rp > 0.5 {
                canvas = canvas.circle(x, y, cur_r * 1.4)
                    .fill(anim_palette(ring + 1, t).with_alpha((rp - 0.5) * 0.08)).done();
            }
            canvas = canvas.circle(x, y, cur_r)
                .fill(anim_palette(ring + 2, t).with_alpha(rp * 0.05))
                .stroke(anim_palette(ring, t).with_alpha(rp * 0.7), 1.2).done();
        }
    }

    // Hexagon + Star of David
    if geometry > 0.0 {
        let hex_r = radius * 0.55 * breath;
        let hex: Vec<(f32, f32)> = (0..6)
            .map(|i| { let a = (i as f32).mul_add(FRAC_PI_3, rot); (hex_r.mul_add(a.cos(), cx), hex_r.mul_add(a.sin(), cy)) })
            .collect();
        canvas = canvas.polygon(hex.clone()).stroke(SOFT_BLUE.with_alpha(geometry * 0.5), 1.0).done();

        let star_r = radius * 0.4 * breath;
        let ta = geometry * 0.6;
        let up: Vec<(f32, f32)> = (0..3).map(|i| {
            let a = (i as f32).mul_add(TAU / 3.0, rot - std::f32::consts::FRAC_PI_2);
            (star_r.mul_add(a.cos(), cx), star_r.mul_add(a.sin(), cy))
        }).collect();
        canvas = canvas.polygon(up).stroke(SOFT_PINK.with_alpha(ta), 1.2).fill(SOFT_PINK.with_alpha(ta * 0.04)).done();
        let dn: Vec<(f32, f32)> = (0..3).map(|i| {
            let a = (i as f32).mul_add(TAU / 3.0, rot + std::f32::consts::FRAC_PI_2);
            (star_r.mul_add(a.cos(), cx), star_r.mul_add(a.sin(), cy))
        }).collect();
        canvas = canvas.polygon(dn).stroke(SOFT_PURPLE.with_alpha(ta), 1.2).fill(SOFT_PURPLE.with_alpha(ta * 0.04)).done();

        for &(hx, hy) in &hex {
            canvas = canvas.line(cx, cy, hx, hy).color(SOFT_CREAM.with_alpha(geometry * 0.2)).width(0.6).done();
        }

        // Bindu
        if geometry > 0.3 {
            let br = ((geometry - 0.3) / 0.7).clamp(0.0, 1.0);
            let bd = radius * 0.04 * breath;
            for i in 0..3 {
                canvas = canvas.circle(cx, cy, bd * (i as f32).mul_add(2.5, 3.0))
                    .fill(SOFT_PINK.with_alpha(br * 0.06 / (i as f32).mul_add(0.5, 1.0))).done();
            }
            canvas = canvas.circle(cx, cy, bd).fill(C::WHITE.with_alpha(br * 0.9)).done();
        }
    }

    // Radiance rays
    if radiance > 0.0 {
        let pulse = (t * 3.0).sin().mul_add(0.5, 0.5);
        for i in 0..12_usize {
            let a   = t.mul_add(0.3, i as f32 * TAU / 12.0);
            let len = radius * 0.7 * 0.25f32.mul_add(t.mul_add(1.5, i as f32).sin(), 0.75);
            canvas = canvas.line(cx, cy, cx + len * a.cos(), cy + len * a.sin())
                .color(anim_palette(i % ANIM_PALETTE.len(), t).with_alpha(radiance * 0.1 * 0.4f32.mul_add(pulse, 0.6)))
                .width(1.0).done();
        }
    }

    canvas
}

fn anim_palette(idx: usize, time: f32) -> C {
    ANIM_PALETTE[(idx + (time * 0.5) as usize) % ANIM_PALETTE.len()]
}
