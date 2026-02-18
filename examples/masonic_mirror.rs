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
//!   CPU:  `cargo run --example masonic_mirror --features "sdf,text,widget" --release`
//!   GPU:  `cargo run --example masonic_mirror --features "sdf-gpu,text,widget" --release`

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
use std::sync::Arc;
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
    ProfileHistory, ProfiledRasterizer, ProtocolKind,
};
use scry_engine::scene::style::Color;
use scry_engine::scene::PixelCanvas;
use scry_engine::scene::command::ImageData;
use scry_engine::sdf::{
    Material, SdfCamera, SdfLight, SdfObject, SdfScene, SdfShape, Vec3,
};
use scry_engine::transport;

#[cfg(feature = "sdf-gpu")]
use scry_engine::sdf::gpu_renderer::{SdfGpuContext, SdfGpuRenderer};

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
    #[cfg(feature = "sdf-gpu")]
    gpu_ctx: Option<SdfGpuContext>,
    gpu_active: bool,
    render_scale: f32,
}

impl MasonicState {
    fn new() -> Self {
        #[cfg(feature = "sdf-gpu")]
        let gpu_ctx = match SdfGpuContext::new() {
            Ok(ctx) => {
                eprintln!("[masonic_mirror] GPU SDF renderer initialized");
                Some(ctx)
            }
            Err(e) => {
                eprintln!("[masonic_mirror] GPU not available ({e}), using CPU");
                None
            }
        };
        #[cfg(feature = "sdf-gpu")]
        let gpu_active = gpu_ctx.is_some();
        #[cfg(not(feature = "sdf-gpu"))]
        let gpu_active = false;

        Self {
            paused: false,
            profiling: false,
            profile_history: ProfileHistory::default(),
            frame_times: Vec::new(),
            profile_start: None,
            last_profile_str: String::new(),
            canvas_w: 0,
            canvas_h: 0,
            #[cfg(feature = "sdf-gpu")]
            gpu_ctx,
            gpu_active,
            render_scale: 1.0,
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

    SdfScene::new()
        // Checkerboard floor
        .object(SdfObject::new(
            SdfShape::Plane,
            Material::Checkerboard {
                color_a: Color::from_rgba8(20, 20, 25, 255),
                color_b: Color::from_rgba8(180, 180, 190, 255),
                scale: 1.0,
                reflectivity: 0.3,
                specular: 32.0,
            },
        ))
        // Central mirror sphere
        .object(
            SdfObject::new(
                SdfShape::Sphere { radius: 1.2 },
                Material::mirror(Color::from_rgba8(200, 200, 220, 255), 0.9),
            )
            .at(Vec3::new(0.0, 1.2, 0.0)),
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
        .sky_color(Color::from_rgba8(5, 5, 15, 255))
}

// ═══════════════════════════════════════════════════════════════════
// Scene assembly
// ═══════════════════════════════════════════════════════════════════

fn build_scene(
    w: u32,
    h: u32,
    state: &MasonicState,
    time: f32,
    sdf_image: Option<ImageData>,
) -> PixelCanvas {
    if w == 0 || h == 0 {
        return PixelCanvas::new(1, 1);
    }

    let wf = w as f32;
    let hf = h as f32;

    let mut canvas = PixelCanvas::new(w, h).background(Color::from_rgba8(5, 5, 15, 255));

    // SDF 3D layer — full canvas
    if let Some(img) = sdf_image {
        // GPU pre-rendered: blit as image
        canvas = canvas.image(img, 0.0, 0.0).done();
    } else {
        // CPU fallback: inline SDF render
        let sdf_scene = build_sdf_scene(time);
        if state.render_scale < 1.0 {
            canvas = canvas.sdf_scene_scaled(
                Arc::new(sdf_scene),
                (time * 60.0) as u64,
                0.0,
                0.0,
                wf,
                hf,
                time,
                state.render_scale,
            );
        } else {
            canvas = canvas.sdf_scene(
                Arc::new(sdf_scene),
                (time * 60.0) as u64,
                0.0,
                0.0,
                wf,
                hf,
                time,
            );
        }
    }

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
        if state.gpu_active { "GPU" } else { "CPU" },
        (state.render_scale * 100.0) as u32)?;
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
    use scry_engine::rasterize::Rasterizer;
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

        let canvas = build_scene(w, h, &state, elapsed, None);
        if let Ok(pixmap) = Rasterizer::rasterize(&canvas) {
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
    let backend: Box<dyn transport::ProtocolBackend> = match picker.protocol() {
        ProtocolKind::Kitty => {
            let kb = transport::kitty::KittyBackend::new(picker.font_size());
            #[cfg(feature = "shm")]
            let kb = if std::env::args().any(|a| a == "--shm") {
                eprintln!("[masonic_mirror] Using shared memory transport");
                kb.format(transport::kitty::TransmitFormat::SharedMemory)
            } else {
                kb
            };
            Box::new(kb)
        }
        _ => Box::new(transport::halfblock::HalfblockBackend::new()),
    };
    let mut px_state = PixelCanvasState::new(backend, picker.font_size());

    let mut state = MasonicState::new();
    let start = Instant::now();
    let mut last_frame = Instant::now();
    let mut frozen_time = 0.0_f32;
    let mut frame_num = 0u64;
    // Pipelined rendering: display frame N-1's SDF image while GPU computes frame N.
    let mut prev_sdf_image: Option<ImageData> = None;
    // Track render dimensions for readback.
    let mut pending_render_w = 0u32;
    let mut pending_render_h = 0u32;
    let mut pending_full_w = 0u32;
    let mut pending_full_h = 0u32;
    let mut gpu_submitted = false;
    // Reusable buffer for GPU readback (avoids per-frame allocation).
    let mut readback_buf: Vec<u8> = Vec::new();

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

        // ── Pipeline step 1: submit GPU work for THIS frame ──
        {
            #[cfg(feature = "sdf-gpu")]
            if state.gpu_active && h > 0 && w > 0 {
                let sdf_scene = build_sdf_scene(elapsed);
                let render_w = if state.render_scale < 1.0 {
                    ((w as f32 * state.render_scale) as u32).max(1)
                } else {
                    w
                };
                let render_h = if state.render_scale < 1.0 {
                    ((h as f32 * state.render_scale) as u32).max(1)
                } else {
                    h
                };
                if let Some(ctx) = state.gpu_ctx.as_mut() {
                    if SdfGpuRenderer::submit(ctx, &sdf_scene, render_w, render_h, elapsed).is_ok()
                    {
                        pending_render_w = render_w;
                        pending_render_h = render_h;
                        pending_full_w = w;
                        pending_full_h = h;
                        gpu_submitted = true;
                    }
                }
            }

            #[cfg(not(feature = "sdf-gpu"))]
            { let _ = &state.gpu_active; }
        }

        // ── Pipeline step 2: build scene with PREVIOUS frame's SDF image ──
        // On the very first frame prev_sdf_image is None, so CPU SDF fallback runs.
        // For frame 2+ we display the previous GPU result (1 frame latency).
        let sdf_image: Option<ImageData> = {
            #[cfg(feature = "sdf-gpu")]
            { if state.gpu_active { prev_sdf_image.take() } else { None } }

            #[cfg(not(feature = "sdf-gpu"))]
            { None }
        };

        let canvas = build_scene(w, h, &state, elapsed, sdf_image);
        let cmd_count = canvas.command_count();

        // Diagnostic: log first frame info
        if frame_num == 1 {
            eprintln!("[diag] frame=1 w={w} h={h} cmds={cmd_count} gpu={} scale={:.0}%",
                state.gpu_active, state.render_scale * 100.0);
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
        let render_mode = if state.gpu_active { "GPU" } else { "CPU" };
        let scale_pct = (state.render_scale * 100.0) as u32;

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

        // ── Pipeline step 3: readback GPU result (now that draw+flush overlapped) ──
        {
            #[cfg(feature = "sdf-gpu")]
            if gpu_submitted {
                if let Some(ctx) = state.gpu_ctx.as_mut() {
                    if pending_render_w == pending_full_w
                        && pending_render_h == pending_full_h
                    {
                        // No upscale needed — read directly into reusable buffer
                        if SdfGpuRenderer::readback_into(
                            ctx,
                            pending_render_w,
                            pending_render_h,
                            &mut readback_buf,
                        )
                        .is_ok()
                        {
                            // Move buffer into ImageData, replace with empty vec
                            // (will be re-allocated on next readback_into)
                            let data = std::mem::take(&mut readback_buf);
                            prev_sdf_image = Some(ImageData::new(
                                pending_full_w,
                                pending_full_h,
                                data,
                            ));
                        }
                    } else if let Ok(pm) = SdfGpuRenderer::readback(
                        ctx,
                        pending_render_w,
                        pending_render_h,
                    ) {
                        let upscaled = scry_engine::sdf::upscale::upscale_bicubic(
                            pm.data(),
                            pending_render_w,
                            pending_render_h,
                            pending_full_w,
                            pending_full_h,
                        );
                        prev_sdf_image =
                            Some(ImageData::new(pending_full_w, pending_full_h, upscaled));
                    }
                    gpu_submitted = false;
                }
            }

            #[cfg(not(feature = "sdf-gpu"))]
            { let _ = gpu_submitted; }
        }

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
                            #[cfg(feature = "sdf-gpu")]
                            {
                                if state.gpu_ctx.is_some() {
                                    state.gpu_active = !state.gpu_active;
                                }
                            }
                        }
                        KeyCode::Char('s') => {
                            // Cycle: 1.0 → 0.75 → 0.5 → 0.25 → 1.0
                            state.render_scale = match (state.render_scale * 100.0) as u32 {
                                100 => 0.75,
                                75 => 0.5,
                                50 => 0.25,
                                _ => 1.0,
                            };
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
