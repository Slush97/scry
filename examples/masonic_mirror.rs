//! **Masonic Mirror** — SDF 3D mirror room with checkerboard floor, pillars, and sphere.
//!
//! Controls:
//!   `Space` — pause/resume
//!   `g`     — toggle GPU/CPU SDF rendering (requires `sdf-gpu` feature)
//!   `s`     — cycle render scale: 100% → 75% → 50% → 25% → 100%
//!   `p`     — toggle profiling (saves benchmark on stop)
//!   `q`     — quit
//!
//! Run with:
//!   `cargo run --example masonic_mirror --features "sdf-gpu,widget" --release`
//!   CPU-only: `cargo run --example masonic_mirror --features "sdf,widget" --release`
//!   SHM:  `cargo run --example masonic_mirror --features "sdf-gpu,widget,shm" --release -- --shm`

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::items_after_statements
)]

use std::io::stdout;
use std::io::Write as IoWrite;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use scry_engine::prelude::{
    Picker, PixelCanvasState, PixelCanvasWidget,
    ProfileHistory, ProfiledRasterizer,
};
use scry_engine::scene::style::Color;
use scry_engine::scene::PixelCanvas;
use scry_engine::scene::command::ImageData;
use scry_engine::sdf::{
    Material, SdfCamera, SdfLight, SdfObject, SdfScene, SdfShape, Vec3,
};
use scry_engine::sdf::pipeline::SdfPipeline;

// ═══════════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════════

struct MasonicState {
    paused: bool,
    profiling: bool,
    profile_history: ProfileHistory,
    frame_times: Vec<f32>,
    profile_start: Option<Instant>,
    last_profile_str: String,
    canvas_w: u32,
    canvas_h: u32,
    /// Unified SDF rendering pipeline (auto-detects GPU, handles double-
    /// buffered pipelining and CPU fallback internally).
    sdf_pipeline: SdfPipeline,
}

impl MasonicState {
    fn new() -> Self {
        Self {
            paused: false,
            profiling: false,
            profile_history: ProfileHistory::default(),
            frame_times: Vec::new(),
            profile_start: None,
            last_profile_str: String::new(),
            canvas_w: 0,
            canvas_h: 0,
            sdf_pipeline: SdfPipeline::new(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// SDF Scene
// ═══════════════════════════════════════════════════════════════════

fn build_sdf_scene(time: f32) -> SdfScene {
    let angle = time * 0.15;
    let cam_radius = 6.0;
    let cam_height = 3.0;

    // Rainbow disc rotation — steady geometric spin
    let disc_angle = time * 1.2;
    // Hue offset for the rainbow so colors rotate with time
    let hue_spin = time * 0.8;

    SdfScene::new()
        // Checkerboard floor — tighter scale for moiré through the glass sphere
        .object(SdfObject::new(
            SdfShape::Plane,
            Material::Checkerboard {
                color_a: Color::from_rgba8(20, 20, 25, 255),
                color_b: Color::from_rgba8(180, 180, 190, 255),
                scale: 0.8,
                reflectivity: 0.3,
                specular: 32.0,
            },
        ))
        // Central glass sphere — semi-transparent with chromatic dispersion
        .object(
            SdfObject::new(
                SdfShape::Sphere { radius: 1.2 },
                Material::glass_dispersive(
                    Color::from_rgba8(230, 230, 255, 255), // slight cool tint
                    1.45,  // glass-like IOR
                    0.04,  // chromatic dispersion for prismatic edges
                ),
            )
            .at(Vec3::new(0.0, 1.2, 0.0)),
        )
        // Rainbow disc rotating inside the sphere — thin cylinder as disc
        .object(
            SdfObject::new(
                SdfShape::Cylinder {
                    radius: 0.95,       // fits inside the 1.2-radius sphere
                    half_height: 0.015,  // razor-thin disc
                },
                Material::rainbow_animated(hue_spin),
            )
            .at(Vec3::new(0.0, 1.2, 0.0))
            .rotate_y(disc_angle),
        )
        // Second disc — perpendicular, creates cross-pattern illusion
        .object(
            SdfObject::new(
                SdfShape::Cylinder {
                    radius: 0.85,
                    half_height: 0.012,
                },
                Material::Rainbow {
                    saturation: 0.85,
                    lightness: 0.45,
                    hue_offset: hue_spin + std::f32::consts::FRAC_PI_2,
                    specular: 48.0,
                },
            )
            .at(Vec3::new(0.0, 1.2, 0.0))
            .rotate_y(disc_angle + std::f32::consts::FRAC_PI_2),
        )
        // Left pillar (Jachin)
        .object(
            SdfObject::new(
                SdfShape::Cylinder {
                    radius: 0.25,
                    half_height: 2.0,
                },
                Material::Solid {
                    color: Color::from_rgba8(180, 160, 120, 255),
                    reflectivity: 0.15,
                    specular: 16.0,
                },
            )
            .at(Vec3::new(-2.5, 2.0, 0.0)),
        )
        // Right pillar (Boaz)
        .object(
            SdfObject::new(
                SdfShape::Cylinder {
                    radius: 0.25,
                    half_height: 2.0,
                },
                Material::Solid {
                    color: Color::from_rgba8(60, 60, 80, 255),
                    reflectivity: 0.15,
                    specular: 16.0,
                },
            )
            .at(Vec3::new(2.5, 2.0, 0.0)),
        )
        // Dramatic lighting from above
        .light(SdfLight::new(
            Vec3::new(0.0, 10.0, 3.0),
            Color::from_rgba8(255, 240, 200, 255),
            0.9,
        ))
        .light(SdfLight::new(
            Vec3::new(-4.0, 5.0, -2.0),
            Color::from_rgba8(100, 120, 200, 255),
            0.3,
        ))
        .camera(SdfCamera::new(
            Vec3::new(
                angle.cos() * cam_radius,
                cam_height,
                angle.sin() * cam_radius,
            ),
            Vec3::new(0.0, 1.0, 0.0),
            50.0,
        ))
        .max_bounces(3)  // extra bounce for glass refraction chains
        .sky_color(Color::from_rgba8(5, 5, 15, 255))
}

// ═══════════════════════════════════════════════════════════════════
// Scene assembly
// ═══════════════════════════════════════════════════════════════════

fn build_scene(
    w: u32,
    h: u32,
    sdf_image: ImageData,
) -> PixelCanvas {
    if w == 0 || h == 0 {
        return PixelCanvas::new(1, 1);
    }

    let mut canvas = PixelCanvas::new(w, h).background(Color::from_rgba8(5, 5, 15, 255));

    // SDF 3D layer — pre-rendered by SdfPipeline, blit as image
    canvas = canvas.image(sdf_image, 0.0, 0.0).done();

    canvas
}

// ═══════════════════════════════════════════════════════════════════
// Benchmark report
// ═══════════════════════════════════════════════════════════════════

fn percentile(values: &mut [f32], pct: usize) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    let rank = (pct * n).div_ceil(100).saturating_sub(1).min(n - 1);
    values[rank]
}

fn save_benchmark_report(state: &MasonicState, protocol: &str) -> std::io::Result<String> {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let filename = format!("masonic_mirror_bench_{secs}.txt");

    let duration = state
        .profile_start
        .map_or(0.0, |s| s.elapsed().as_secs_f64());
    let frame_count = state.frame_times.len();

    let mut f = std::fs::File::create(&filename)?;

    // Header
    writeln!(f, "Masonic Mirror — Benchmark Report")?;
    writeln!(f, "=================================")?;
    writeln!(f)?;

    // Environment
    writeln!(f, "Environment")?;
    writeln!(f, "-----------")?;
    writeln!(f, "  Resolution:  {}x{}", state.canvas_w, state.canvas_h)?;
    writeln!(f, "  Pixels:      {}", u64::from(state.canvas_w) * u64::from(state.canvas_h))?;
    writeln!(f, "  Protocol:    {protocol}")?;
    writeln!(f, "  SDF render:  {} (scale {}%)",
        state.sdf_pipeline.backend_name(),
        (state.sdf_pipeline.get_render_scale() * 100.0) as u32)?;
    writeln!(f, "  Duration:    {duration:.1}s")?;
    writeln!(f, "  Frames:      {frame_count}")?;
    writeln!(f)?;

    // FPS Statistics
    writeln!(f, "FPS Statistics")?;
    writeln!(f, "--------------")?;
    if frame_count > 1 {
        let dts = state.frame_times.clone();
        let mean_dt: f32 = dts.iter().sum::<f32>() / dts.len() as f32;
        let median_dt = percentile(&mut dts.clone(), 50);
        let p5_dt = percentile(&mut dts.clone(), 5);
        let p95_dt = percentile(&mut dts.clone(), 95);
        let min_dt = dts.iter().copied().fold(f32::INFINITY, f32::min);
        let max_dt = dts.iter().copied().fold(0.0_f32, f32::max);

        let safe_fps = |dt: f32| if dt > 0.0 { 1.0 / dt } else { 0.0 };

        writeln!(f, "  Mean:    {:.1} fps  ({:.2}ms/frame)", safe_fps(mean_dt), mean_dt * 1000.0)?;
        writeln!(f, "  Median:  {:.1} fps  ({:.2}ms/frame)", safe_fps(median_dt), median_dt * 1000.0)?;
        writeln!(f, "  P5:      {:.1} fps  (fast frames)", safe_fps(p5_dt))?;
        writeln!(f, "  P95:     {:.1} fps  (slow frames)", safe_fps(p95_dt))?;
        writeln!(f, "  Min:     {:.1} fps", safe_fps(max_dt))?;
        writeln!(f, "  Max:     {:.1} fps", safe_fps(min_dt))?;
    } else {
        writeln!(f, "  (insufficient frames)")?;
    }
    writeln!(f)?;

    // Rasterization Profile
    let summary = state.profile_history.summary();
    writeln!(f, "Rasterization Profile (median over {} frames)", summary.frame_count)?;
    writeln!(f, "-------------------------------------------")?;
    writeln!(f, "  Total raster:  {:.3}ms (median)  {:.3}ms (P95)",
        summary.total_median_us as f64 / 1000.0,
        summary.total_p95_us as f64 / 1000.0,
    )?;
    writeln!(f)?;

    let sorted = summary.sorted_types();
    if !sorted.is_empty() {
        writeln!(f, "  {:<12} {:>5}  {:>10}  {:>10}  {:>6}", "Type", "Count", "Median", "P95", "% Tot")?;
        writeln!(f, "  {}", "-".repeat(50))?;
        for st in &sorted {
            let pct = if summary.total_median_us > 0 {
                (st.median_us as f64 / summary.total_median_us as f64) * 100.0
            } else {
                0.0
            };
            writeln!(f, "  {:<12} {:>5}  {:>8.3}ms  {:>8.3}ms  {:>5.1}%",
                st.cmd_type.name(),
                st.count,
                st.median_us as f64 / 1000.0,
                st.p95_us as f64 / 1000.0,
                pct,
            )?;
        }
    }
    writeln!(f)?;

    // Bottleneck Analysis
    writeln!(f, "Bottleneck Analysis")?;
    writeln!(f, "-------------------")?;
    if sorted.is_empty() {
        writeln!(f, "  (no data)")?;
    } else {
        let mut found_bottleneck = false;
        for st in &sorted {
            if summary.total_median_us == 0 {
                break;
            }
            let pct = (st.median_us as f64 / summary.total_median_us as f64) * 100.0;
            if pct > 30.0 {
                writeln!(f, "  DOMINATES: {} ({:.1}% of raster time, {}x count)",
                    st.cmd_type.name(), pct, st.count)?;
                found_bottleneck = true;
            } else if pct > 15.0 {
                writeln!(f, "  SIGNIFICANT: {} ({:.1}% of raster time, {}x count)",
                    st.cmd_type.name(), pct, st.count)?;
                found_bottleneck = true;
            }
        }
        if !found_bottleneck {
            writeln!(f, "  No single command type dominates (>30%) or is significant (>15%).")?;
            writeln!(f, "  Raster load is well-distributed.")?;
        }
    }
    writeln!(f)?;

    Ok(filename)
}

// ═══════════════════════════════════════════════════════════════════
// Window mode
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "window")]
fn run_window() -> Result<(), Box<dyn std::error::Error>> {
    use scry_engine::transport::window::{run_loop_continuous, LoopAction};
    use winit::keyboard::KeyCode as WKey;

    let mut state = MasonicState::new();
    let start = Instant::now();
    let mut frozen_time = 0.0_f32;
    let mut last_time = 0.0_f32;

    run_loop_continuous(960, 640, "Masonic Mirror", true, move |backend, keys, (w, h)| {
        for key in keys {
            if !key.pressed {
                continue;
            }
            match key.code {
                WKey::Escape | WKey::KeyQ => return LoopAction::Exit,
                WKey::Space => state.paused = !state.paused,
                _ => {}
            }
        }

        let elapsed = if state.paused {
            frozen_time
        } else {
            let e = start.elapsed().as_secs_f32();
            frozen_time = e;
            e
        };

        let _dt = elapsed - last_time;
        last_time = elapsed;

        let sdf_scene = build_sdf_scene(elapsed);
        let sdf_result = state.sdf_pipeline.render_sync(&sdf_scene, w, h, elapsed);
        let canvas = build_scene(w, h, sdf_result.image);
        if let Ok(pixmap) = scry_engine::rasterize::RasterPipeline::new().rasterize(&canvas) {
            let _ = backend.blit(&pixmap);
        }
        LoopAction::Continue
    })?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let use_window = std::env::args().any(|a| a == "--window");
    if use_window {
        #[cfg(feature = "window")]
        {
            return run_window();
        }
        #[cfg(not(feature = "window"))]
        {
            eprintln!("error: --window requires the `window` feature");
            std::process::exit(1);
        }
    }

    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let picker = Picker::detect();
    let protocol_name = format!("{:?}", picker.protocol());
    let backend = picker.create_backend();
    let mut px_state = PixelCanvasState::new(backend, picker.font_size());

    let mut state = MasonicState::new();
    let start = Instant::now();
    let mut last_frame = Instant::now();
    let mut frozen_time = 0.0_f32;
    let mut frame_num = 0u64;

    eprintln!("[masonic_mirror] SDF backend: {}", state.sdf_pipeline.backend_name());

    loop {
        frame_num += 1;
        let now = Instant::now();
        let dt = now.duration_since(last_frame);
        last_frame = now;

        let elapsed = if state.paused {
            frozen_time
        } else {
            let e = now.duration_since(start).as_secs_f32();
            frozen_time = e;
            e
        };

        let fps = if dt.as_secs_f32() > 0.0 { 1.0 / dt.as_secs_f32() } else { 0.0 };

        // Build scene outside terminal.draw() so we can optionally profile it
        let term_size = terminal.size()?;
        let term_rect = ratatui::layout::Rect::new(0, 0, term_size.width, term_size.height);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(3)])
            .split(term_rect);

        let area = chunks[0];
        let font = px_state.font_size();
        let w = u32::from(area.width) * u32::from(font.width);
        let h = u32::from(area.height) * u32::from(font.height);

        // ── Render SDF via pipeline (handles GPU pipelining internally) ──
        let sdf_scene = build_sdf_scene(elapsed);
        let sdf_result = state.sdf_pipeline.render(&sdf_scene, w, h, elapsed);
        let canvas = build_scene(w, h, sdf_result.image);
        let cmd_count = canvas.command_count();

        // Diagnostic: log first frame info
        if frame_num == 1 {
            eprintln!("[diag] frame=1 w={w} h={h} cmds={cmd_count} backend={} scale={:.0}%",
                state.sdf_pipeline.backend_name(),
                state.sdf_pipeline.get_render_scale() * 100.0);
        }

        // Track resolution for report
        state.canvas_w = w;
        state.canvas_h = h;

        // When profiling, rasterize manually with per-command timing
        if state.profiling {
            state.frame_times.push(dt.as_secs_f32());

            let (pixmap_entry, gc) = px_state
                .cache_mut()
                .get_or_insert_with_grad_cache(canvas.width(), canvas.height());
            if let Some(pixmap) = pixmap_entry {
                let rp = ProfiledRasterizer::rasterize_into_profiled_cached(&canvas, pixmap, gc);
                state.profile_history.push(rp);
                let smoothed = state.profile_history.summary();
                state.last_profile_str = format!(
                    " \u{1F50D} {fps:.0}fps {w}x{h} | rast={smoothed} | {cmd_count}cmd f{} | [p] stop & save  [q] quit",
                    smoothed.frame_count,
                );
            }
            static FRAME_SEQ: AtomicU64 = AtomicU64::new(1);
            px_state
                .cache_mut()
                .mark_valid(FRAME_SEQ.fetch_add(1, Ordering::Relaxed));
        }

        let profile_line = state.last_profile_str.clone();
        let is_profiling = state.profiling;
        let render_mode = state.sdf_pipeline.backend_name();
        let scale_pct = (state.sdf_pipeline.get_render_scale() * 100.0) as u32;

        terminal.draw(|frame| {
            frame.render_stateful_widget(
                PixelCanvasWidget::new(canvas).skip_cache().z_index(-1),
                area,
                &mut px_state,
            );

            let status_text = if is_profiling {
                profile_line.clone()
            } else {
                format!(
                    " Masonic Mirror | {render_mode} {scale_pct}% | {fps:.0}fps | {elapsed:.1}s | [g] gpu [s] scale [p] profile [q] quit",
                )
            };
            let status = Paragraph::new(status_text).block(Block::default().borders(Borders::TOP));
            frame.render_widget(status, chunks[1]);
        })?;
        px_state.flush()?;

        // Flush pipeline — readback GPU result to overlap with terminal I/O
        state.sdf_pipeline.flush();

        if event::poll(Duration::ZERO)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => {
                            if state.profiling {
                                let _ = save_benchmark_report(&state, &protocol_name);
                            }
                            break;
                        }
                        KeyCode::Char(' ') => state.paused = !state.paused,
                        KeyCode::Char('g') => {
                            let currently_gpu = state.sdf_pipeline.is_gpu_active();
                            state.sdf_pipeline.set_gpu_active(!currently_gpu);
                        }
                        KeyCode::Char('s') => {
                            // Cycle: 1.0 → 0.75 → 0.5 → 0.25 → 1.0
                            let new_scale = match (state.sdf_pipeline.get_render_scale() * 100.0) as u32 {
                                100 => 0.75,
                                75 => 0.5,
                                50 => 0.25,
                                _ => 1.0,
                            };
                            state.sdf_pipeline.set_render_scale(new_scale);
                        }
                        KeyCode::Char('p') => {
                            if state.profiling {
                                // Stop profiling — save report
                                state.profiling = false;
                                match save_benchmark_report(&state, &protocol_name) {
                                    Ok(filename) => {
                                        state.last_profile_str = format!(" Saved: {filename}");
                                    }
                                    Err(e) => {
                                        state.last_profile_str = format!(" Save error: {e}");
                                    }
                                }
                            } else {
                                // Start profiling — reset accumulators
                                state.profiling = true;
                                state.profile_history.clear();
                                state.frame_times.clear();
                                state.profile_start = Some(Instant::now());
                                state.last_profile_str = " Profiling started...".to_string();
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    px_state.cleanup();
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}
