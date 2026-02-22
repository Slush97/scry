// SPDX-License-Identifier: MIT OR Apache-2.0
//! Scry Terminal — entry point.
//!
//! A GPU-accelerated terminal emulator powered by scry-engine's rendering
//! infrastructure. Runs a shell in a pseudo-terminal and renders output
//! in a native window using wgpu.

use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::Key;
use winit::window::{Window, WindowAttributes, WindowId};

use scry_terminal::compositor::Compositor;
use scry_terminal::config::TerminalConfig;
use scry_terminal::grid::TerminalGrid;
use scry_terminal::input;
use scry_terminal::performance::{ParseThrottler, RenderScheduler};
use scry_terminal::platform::{self, TerminalSize};
use scry_terminal::pty::PtyManager;
use scry_terminal::security::{ResponsePolicy, SecurityGate};
use scry_terminal::selection::Selection;

/// Custom event sent from the PTY reader thread to wake the event loop.
#[derive(Debug, Clone)]
enum TerminalEvent {
    /// New PTY data is available for reading.
    PtyDataReady,
}

/// The terminal application.
struct TerminalApp {
    /// Terminal configuration.
    config: TerminalConfig,
    /// Window + GPU state (initialized on resume).
    state: Option<TerminalState>,
    /// Proxy for waking the event loop from the PTY reader thread.
    proxy: EventLoopProxy<TerminalEvent>,
}

/// Active terminal state (created after window is available).
struct TerminalState {
    /// The window.
    window: Arc<Window>,
    /// GPU compositor.
    compositor: Compositor,
    /// Terminal grid.
    grid: TerminalGrid,
    /// Security gate.
    security: SecurityGate,
    /// PTY manager.
    pty: PtyManager,
    /// Parse throttler.
    throttler: ParseThrottler,
    /// Render scheduler.
    scheduler: RenderScheduler,
    /// Current modifier state.
    modifiers: winit::keyboard::ModifiersState,
    /// Currently held mouse button.
    mouse_button: Option<MouseButton>,
    /// Current mouse cell position.
    mouse_cell: (u16, u16),
    /// Text selection state.
    selection: Selection,
    /// System clipboard handle.
    clipboard: Option<arboard::Clipboard>,
    /// Whether the child has exited.
    child_exited: bool,
    /// When the PTY was spawned (for startup grace period).
    spawn_time: Instant,
    /// Deadline to exit after child exits (drain period).
    exit_deadline: Option<Instant>,
    /// Original font size for Ctrl+0 reset.
    original_font_size: f32,
    /// Content padding in pixels.
    padding: f32,
}

impl TerminalApp {
    const fn new(config: TerminalConfig, proxy: EventLoopProxy<TerminalEvent>) -> Self {
        Self {
            config,
            state: None,
            proxy,
        }
    }
}

impl ApplicationHandler<TerminalEvent> for TerminalApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return; // Already initialized
        }

        eprintln!("[scry-term] resumed: creating window...");

        // Create window
        let pad = f64::from(self.config.window.padding);
        let attrs = WindowAttributes::default()
            .with_title("Scry Terminal")
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.config.window.columns as f64 * 8.0 + 2.0 * pad,
                self.config.window.rows as f64 * 16.0 + 2.0 * pad,
            ));

        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                eprintln!("[scry-term] failed to create window: {e}");
                event_loop.exit();
                return;
            }
        };

        eprintln!("[scry-term] window created, initializing GPU...");

        // Create compositor (initializes wgpu)
        let compositor = match Compositor::new(window.clone(), &self.config) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[scry-term] GPU initialization failed: {e}");
                event_loop.exit();
                return;
            }
        };

        eprintln!(
            "[scry-term] GPU initialized. cell={}x{}",
            compositor.cell_width(),
            compositor.cell_height()
        );

        // Compute initial terminal size
        let inner = window.inner_size();
        let padding = self.config.window.padding;
        let term_size = TerminalSize::from_window(
            inner.width,
            inner.height,
            compositor.cell_width(),
            compositor.cell_height(),
            padding,
        );

        eprintln!(
            "[scry-term] terminal size: {}x{} ({}x{} px)",
            term_size.cols, term_size.rows, inner.width, inner.height
        );

        // Create grid
        let grid = TerminalGrid::new(
            term_size.cols,
            term_size.rows,
            self.config.scrollback.lines,
        );

        // Create security gate
        let security = SecurityGate::new(ResponsePolicy::default());

        // Determine shell
        let shell = self
            .config
            .shell
            .clone()
            .unwrap_or_else(platform::default_shell);

        eprintln!("[scry-term] spawning shell: {shell}");

        // Create a waker that sends a PtyDataReady event to the event loop
        let proxy = self.proxy.clone();
        let waker = Box::new(move || {
            let _ = proxy.send_event(TerminalEvent::PtyDataReady);
        });

        // Spawn PTY with waker
        let pty = match PtyManager::spawn_with_waker(
            &shell,
            term_size.cols,
            term_size.rows,
            term_size.pixel_width,
            term_size.pixel_height,
            waker,
        ) {
            Ok(pty) => pty,
            Err(e) => {
                eprintln!("[scry-term] failed to spawn PTY: {e}");
                event_loop.exit();
                return;
            }
        };

        eprintln!("[scry-term] PTY spawned successfully, installing signals...");

        // Install signal handlers
        platform::install_signal_handlers();

        let throttler = ParseThrottler::default_budget();
        let scheduler = RenderScheduler::new(60);

        // Initialize clipboard (may fail on headless systems)
        let clipboard = arboard::Clipboard::new().ok();

        self.state = Some(TerminalState {
            window: window.clone(),
            compositor,
            grid,
            security,
            pty,
            throttler,
            scheduler,
            modifiers: winit::keyboard::ModifiersState::empty(),
            mouse_button: None,
            mouse_cell: (0, 0),
            selection: Selection::default(),
            clipboard,
            child_exited: false,
            spawn_time: Instant::now(),
            exit_deadline: None,
            original_font_size: self.config.font.size,
            padding,
        });

        eprintln!("[scry-term] initialization complete, requesting first redraw...");

        // Kick-start the render loop
        window.request_redraw();
    }

    /// Handle custom user events (PTY data ready).
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: TerminalEvent) {
        match event {
            TerminalEvent::PtyDataReady => {
                // PTY has data — request a redraw to process it
                if let Some(state) = &self.state {
                    state.window.request_redraw();
                }
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    state.compositor.resize(new_size.width, new_size.height);

                    let term_size = TerminalSize::from_window(
                        new_size.width,
                        new_size.height,
                        state.compositor.cell_width(),
                        state.compositor.cell_height(),
                        state.padding,
                    );

                    state.grid.resize(term_size.cols, term_size.rows);

                    let _ = state.pty.resize(
                        term_size.cols,
                        term_size.rows,
                        term_size.pixel_width,
                        term_size.pixel_height,
                    );

                    state.scheduler.request_redraw();
                }
            }

            WindowEvent::ModifiersChanged(new_modifiers) => {
                state.modifiers = new_modifiers.state();
            }

            WindowEvent::KeyboardInput { event, .. } => {
                // Intercept Shift+PageUp/Down for viewport scrolling
                if event.state == ElementState::Pressed && state.modifiers.shift_key() {
                    match &event.logical_key {
                        Key::Named(winit::keyboard::NamedKey::PageUp) => {
                            let half = (state.grid.rows() / 2).max(1) as usize;
                            state.grid.scroll_viewport_up(half);
                            state.scheduler.request_redraw();
                            return;
                        }
                        Key::Named(winit::keyboard::NamedKey::PageDown) => {
                            let half = (state.grid.rows() / 2).max(1) as usize;
                            state.grid.scroll_viewport_down(half);
                            state.scheduler.request_redraw();
                            return;
                        }
                        _ => {}
                    }
                }

                // Intercept Ctrl+Shift+C (copy) and Ctrl+Shift+V (paste)
                if event.state == ElementState::Pressed
                    && state.modifiers.control_key()
                    && state.modifiers.shift_key()
                {
                    match &event.logical_key {
                        Key::Character(ch) if ch.eq_ignore_ascii_case("c") => {
                            // Copy selection to clipboard
                            if !state.selection.is_empty() {
                                let text = state.selection.selected_text(&state.grid);
                                if let Some(clip) = &mut state.clipboard {
                                    let _ = clip.set_text(&text);
                                }
                            }
                            // Consume the event — don't pass to PTY
                            return;
                        }
                        Key::Character(ch) if ch.eq_ignore_ascii_case("v") => {
                            // Paste from clipboard
                            if let Some(clip) = &mut state.clipboard {
                                if let Ok(text) = clip.get_text() {
                                    let bytes = input::encode_paste(
                                        &text,
                                        state.grid.bracketed_paste,
                                    );
                                    pty_write(&mut state.pty,&bytes);
                                }
                            }
                            return;
                        }
                        _ => {}
                    }
                }

                // Zoom: Ctrl+= / Ctrl+- / Ctrl+0
                if event.state == ElementState::Pressed && state.modifiers.control_key() {
                    match &event.logical_key {
                        Key::Character(ch) if ch.as_str() == "=" || ch.as_str() == "+" => {
                            let new_size = state.compositor.font_size() + 1.0;
                            zoom_to(state, new_size);
                            return;
                        }
                        Key::Character(ch) if ch.as_str() == "-" => {
                            let new_size = (state.compositor.font_size() - 1.0).max(8.0);
                            zoom_to(state, new_size);
                            return;
                        }
                        Key::Character(ch) if ch.as_str() == "0" => {
                            zoom_to(state, state.original_font_size);
                            return;
                        }
                        _ => {}
                    }
                }

                if let Some(bytes) = input::encode_key(
                    &event.logical_key,
                    event.physical_key,
                    event.state,
                    state.modifiers,
                    state.grid.app_cursor_keys,
                    state.grid.app_keypad,
                ) {
                    // User is typing — snap to bottom and reset cursor blink
                    state.grid.snap_to_bottom();
                    state.compositor.reset_blink();
                    pty_write(&mut state.pty, &bytes);
                }
            }

            WindowEvent::MouseInput {
                state: button_state,
                button,
                ..
            } => {
                // Selection tracking: left button starts/ends selection
                // when mouse reporting is NOT active
                if button == MouseButton::Left
                    && state.grid.mouse_mode == scry_terminal::grid::MouseMode::None
                {
                    if button_state == ElementState::Pressed {
                        let anchor = scry_terminal::selection::SelectionAnchor::new(
                            state.mouse_cell.0,
                            state.mouse_cell.1 as i64,
                        );
                        state.selection.begin(anchor);
                        state.scheduler.request_redraw();
                    } else {
                        state.selection.finalize();
                    }
                }

                if button_state == ElementState::Pressed {
                    state.mouse_button = Some(button);
                } else {
                    state.mouse_button = None;
                }

                if let Some(bytes) = input::encode_mouse_button(
                    button,
                    button_state,
                    state.mouse_cell.0,
                    state.mouse_cell.1,
                    state.grid.mouse_mode,
                    state.grid.mouse_encoding,
                ) {
                    pty_write(&mut state.pty,&bytes);
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let col =
                    ((position.x as f32 - state.padding) / state.compositor.cell_width()).max(0.0)
                        as u16;
                let row =
                    ((position.y as f32 - state.padding) / state.compositor.cell_height()).max(0.0)
                        as u16;
                let col = col.min(state.grid.cols().saturating_sub(1));
                let row = row.min(state.grid.rows().saturating_sub(1));

                if (col, row) != state.mouse_cell {
                    state.mouse_cell = (col, row);

                    // Update selection if dragging
                    if state.selection.active && state.mouse_button == Some(MouseButton::Left) {
                        let anchor = scry_terminal::selection::SelectionAnchor::new(
                            col,
                            row as i64,
                        );
                        state.selection.update(anchor);
                        state.scheduler.request_redraw();
                    }

                    if let Some(bytes) = input::encode_mouse_motion(
                        col,
                        row,
                        state.mouse_button,
                        state.grid.mouse_mode,
                        state.grid.mouse_encoding,
                    ) {
                        pty_write(&mut state.pty,&bytes);
                    }
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let up = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y > 0.0,
                    MouseScrollDelta::PixelDelta(pos) => pos.y > 0.0,
                };

                // When mouse reporting is off, scroll the viewport
                if state.grid.mouse_mode == scry_terminal::grid::MouseMode::None {
                    let scroll_lines = 3;
                    if up {
                        state.grid.scroll_viewport_up(scroll_lines);
                    } else {
                        state.grid.scroll_viewport_down(scroll_lines);
                    }
                    state.scheduler.request_redraw();
                } else if let Some(bytes) = input::encode_mouse_scroll(
                    up,
                    state.mouse_cell.0,
                    state.mouse_cell.1,
                    state.grid.mouse_mode,
                    state.grid.mouse_encoding,
                ) {
                    pty_write(&mut state.pty,&bytes);
                }
            }

            WindowEvent::RedrawRequested => {
                // Check if drain period has expired → exit
                if let Some(deadline) = state.exit_deadline {
                    // Drain remaining PTY output during grace period
                    let drain = state
                        .throttler
                        .poll_pty(&state.pty, &mut state.grid, &mut state.security);
                    for response in &drain.responses {
                        pty_write(&mut state.pty,response);
                    }
                    if drain.bytes_consumed > 0 {
                        state.scheduler.request_redraw();
                    }

                    if Instant::now() >= deadline {
                        event_loop.exit();
                        return;
                    }
                    // Still draining — render what we have and continue
                    if (state.scheduler.should_render() || state.grid.has_dirty())
                        && state.compositor.render_frame(&state.grid, Some(&state.selection)).is_ok()
                    {
                        state.scheduler.did_render();
                        state.grid.clear_dirty();
                    }
                    state.window.request_redraw();
                    return;
                }

                // Poll PTY for new data
                let result = state
                    .throttler
                    .poll_pty(&state.pty, &mut state.grid, &mut state.security);

                // Send responses back to PTY
                for response in &result.responses {
                    pty_write(&mut state.pty,response);
                }

                if result.child_exited {
                    state.child_exited = true;
                    state.exit_deadline =
                        Some(Instant::now() + Duration::from_millis(100));
                    state.window.request_redraw();
                    return;
                }

                if result.bytes_consumed > 0 {
                    state.scheduler.request_redraw();
                }

                // Check clipboard paste (OSC 52)
                if let Some(text) = state.grid.clipboard_pending.take() {
                    if let Some(clip) = &mut state.clipboard {
                        let _ = clip.set_text(&text);
                    }
                }

                // Check visual bell
                if state.grid.bell_pending {
                    state.grid.bell_pending = false;
                    state.compositor.trigger_bell();
                    state.scheduler.request_redraw();
                }

                // Render
                if state.scheduler.should_render() || state.grid.has_dirty() {
                    match state.compositor.render_frame(&state.grid, Some(&state.selection)) {
                        Ok(()) => {
                            state.scheduler.did_render();
                            state.grid.clear_dirty();
                        }
                        Err(wgpu::SurfaceError::Lost) => {
                            let size = state.window.inner_size();
                            state.compositor.resize(size.width, size.height);
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            eprintln!("scry-terminal: out of GPU memory");
                            event_loop.exit();
                            return;
                        }
                        Err(e) => {
                            eprintln!("[scry-term] surface error (recovered): {e}");
                        }
                    }
                }

                // Update window title
                if !state.grid.title.is_empty() {
                    state.window.set_title(&state.grid.title);
                }

                // Check if child has exited (non-blocking)
                // Skip during startup grace period to avoid racing with
                // shell initialization.
                if state.spawn_time.elapsed() > Duration::from_millis(500) {
                    if let Some(_status) = state.pty.try_wait() {
                        state.child_exited = true;
                        state.exit_deadline =
                            Some(Instant::now() + Duration::from_millis(100));
                        state.window.request_redraw();
                        return;
                    }
                }

                // With ControlFlow::Wait, we don't need to continuously request
                // redraws — the PTY waker will send a user event when data arrives.
                // But if cursor blink or bell is active, keep rendering.
                if state.compositor.next_blink_deadline().is_some()
                    && state.grid.cursor.blink
                    && state.grid.cursor.visible
                {
                    state.window.request_redraw();
                }
            }

            WindowEvent::ScaleFactorChanged {
                scale_factor,
                inner_size_writer: _,
            } => {
                // HiDPI: recalculate on scale factor change
                let size = state.window.inner_size();
                if size.width > 0 && size.height > 0 {
                    state.compositor.resize(size.width, size.height);

                    let term_size = TerminalSize::from_window(
                        size.width,
                        size.height,
                        state.compositor.cell_width(),
                        state.compositor.cell_height(),
                        state.padding,
                    );

                    state.grid.resize(term_size.cols, term_size.rows);

                    let _ = state.pty.resize(
                        term_size.cols,
                        term_size.rows,
                        term_size.pixel_width,
                        term_size.pixel_height,
                    );

                    state.scheduler.request_redraw();
                }
                let _ = scale_factor; // silence unused warning
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Use WaitUntil for cursor blink timing
        if let Some(state) = &self.state {
            if state.grid.cursor.blink && state.grid.cursor.visible {
                if let Some(deadline) = state.compositor.next_blink_deadline() {
                    event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
                    return;
                }
            }
        }
        event_loop.set_control_flow(ControlFlow::Wait);
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        eprintln!("[scry-term] SUSPENDED event received");
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        eprintln!("[scry-term] EXITING event received");
    }
}

/// Write to PTY with error logging instead of silently ignoring failures.
fn pty_write(pty: &mut PtyManager, data: &[u8]) {
    if let Err(e) = pty.write(data) {
        eprintln!("[scry-term] PTY write failed: {e}");
    }
}

/// Apply a zoom (font size change), recompute grid, and save to config.
fn zoom_to(state: &mut TerminalState, new_size: f32) {
    let (_cw, _ch) = state.compositor.set_font_size(new_size);

    let inner = state.window.inner_size();
    let term_size = TerminalSize::from_window(
        inner.width,
        inner.height,
        state.compositor.cell_width(),
        state.compositor.cell_height(),
        state.padding,
    );

    state.grid.resize(term_size.cols, term_size.rows);
    let _ = state.pty.resize(
        term_size.cols,
        term_size.rows,
        term_size.pixel_width,
        term_size.pixel_height,
    );

    state.scheduler.request_redraw();
    state.window.request_redraw();

    eprintln!("[scry-term] font size: {new_size}px (saved)");
    TerminalConfig::save_font_size(new_size);
}

fn main() {
    // Load config
    let config = TerminalConfig::load();

    // Create event loop with user events (for PTY wakeup)
    let event_loop = match EventLoop::<TerminalEvent>::with_user_event().build() {
        Ok(el) => el,
        Err(e) => {
            eprintln!("scry-terminal: failed to create event loop: {e}");
            std::process::exit(1);
        }
    };
    event_loop.set_control_flow(ControlFlow::Wait);

    let proxy = event_loop.create_proxy();

    // Run
    let mut app = TerminalApp::new(config, proxy);
    match event_loop.run_app(&mut app) {
        Ok(()) => {}
        Err(e) => {
            // ExitFailure is normal when event_loop.exit() is called
            eprintln!("scry-terminal: event loop exited: {e}");
        }
    }
}
