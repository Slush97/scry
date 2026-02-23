// SPDX-License-Identifier: MIT OR Apache-2.0
//! Scry Terminal — entry point.
//!
//! A GPU-accelerated terminal emulator powered by scry-engine's rendering
//! infrastructure. Runs a shell in a pseudo-terminal and renders output
//! in a native window using wgpu.

#[cfg(feature = "logging")]
use tracing::{debug, info, trace, warn, error};

#[cfg(not(feature = "logging"))]
macro_rules! trace { ($($t:tt)*) => { if false { let _ = format_args!($($t)*); } } }
#[cfg(not(feature = "logging"))]
macro_rules! debug { ($($t:tt)*) => { if false { let _ = format_args!($($t)*); } } }
#[cfg(not(feature = "logging"))]
macro_rules! info  { ($($t:tt)*) => { if false { let _ = format_args!($($t)*); } } }
#[cfg(not(feature = "logging"))]
macro_rules! warn  { ($($t:tt)*) => { if false { let _ = format_args!($($t)*); } } }
#[cfg(not(feature = "logging"))]
macro_rules! error { ($($t:tt)*) => { if false { let _ = format_args!($($t)*); } } }

use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel::Receiver;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowAttributes, WindowId};

use scry_terminal::compositor::Compositor;
use scry_terminal::config::TerminalConfig;
use scry_terminal::grid::TerminalGrid;
use scry_terminal::ipc_server::{IpcServer, OverlayOp, TerminalInfo};
use scry_engine::transport::ipc::{Memfd, OverlayAnchor};
use scry_terminal::performance::{ParseThrottler, RenderScheduler};
use scry_terminal::platform::{self, TerminalSize};
use scry_terminal::pty::PtyManager;
use scry_terminal::security::{ResponsePolicy, SecurityGate};
use scry_terminal::selection::Selection;

mod handlers;
mod redraw;

/// Custom event sent from the PTY reader thread to wake the event loop.
#[derive(Debug, Clone)]
enum TerminalEvent {
    /// New PTY data is available for reading.
    PtyDataReady,
    /// New IPC overlay data is available.
    OverlayReady,
}

/// State for the currently active IPC overlay.
///
/// Retains the memfd so `Refresh` commands can re-read updated pixel data
/// without needing a new file descriptor transfer.
#[allow(dead_code)]
struct ActiveOverlay {
    /// The shared memory mapping (kept alive for in-place updates).
    memfd: Memfd,
    /// Pixel width of the overlay.
    px_w: u32,
    /// Pixel height of the overlay.
    px_h: u32,
    /// Anchor position.
    anchor: OverlayAnchor,
    /// Width in terminal cells.
    w_cells: u16,
    /// Height in terminal cells.
    h_cells: u16,
}

/// An active terminal-autonomous animation.
///
/// Created when the IPC server receives a `SubmitAnimation` command.
/// The terminal drives this animation in its render loop — the CLI
/// can exit immediately after submission.
#[allow(dead_code)]
struct ActiveAnimation {
    /// Overlay ID.
    id: u32,
    /// The animation program to evaluate each frame.
    program: scry_engine::sdf::AnimationProgram,
    /// SDF rendering pipeline (owns GPU/CPU context).
    pipeline: scry_engine::sdf::SdfPipeline,
    /// When the animation started.
    start_time: Instant,
    /// Optional duration limit (None = infinite).
    duration: Option<Duration>,
    /// Target frame interval.
    frame_interval: Duration,
    /// When the last frame was rendered.
    last_frame: Instant,
    /// Pixel width for SDF rendering.
    width: u32,
    /// Pixel height for SDF rendering.
    height: u32,
    /// Whether the animation is paused (toggled by click).
    paused: bool,
    /// Whether the animation is visible in the viewport.
    visible: bool,
    /// Anchor position.
    anchor: OverlayAnchor,
    /// Elapsed time when paused (to resume from correct point).
    paused_elapsed: Duration,
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
    mouse_button: Option<winit::event::MouseButton>,
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
    /// IPC server for scry-cli overlay communication.
    _ipc_server: Option<IpcServer>,
    /// Receiver for overlay operations from the IPC server.
    ipc_ops_rx: Option<Receiver<OverlayOp>>,
    /// Terminal info shared with the IPC server.
    ipc_info: Option<Arc<std::sync::RwLock<TerminalInfo>>>,
    /// Currently active IPC overlays (retains memfds for Refresh re-reads).
    ///
    /// Keyed by overlay ID — supports multiple concurrent overlays from
    /// different CLI commands or tabs.
    active_overlays: std::collections::HashMap<u32, ActiveOverlay>,
    /// Active terminal-autonomous animations.
    active_animations: Vec<ActiveAnimation>,
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
        let resumed_start = Instant::now();
        if self.state.is_some() {
            return; // Already initialized
        }

        debug!("creating window");

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
                error!("failed to create window: {e}");
                event_loop.exit();
                return;
            }
        };

        debug!("window created, initializing GPU");

        // Create compositor (initializes wgpu)
        let compositor = match Compositor::new(window.clone(), &self.config) {
            Ok(c) => c,
            Err(e) => {
                error!("GPU initialization failed: {e}");
                event_loop.exit();
                return;
            }
        };

        debug!(
            "GPU initialized. cell={}x{}",
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

        debug!(
            "terminal size: {}x{} ({}x{} px)",
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

        // ── IPC server ─────────────────────────────────────────────
        // Start the IPC server so that scry-cli (and other tools) can
        // send overlay operations via shared memory.
        let (ipc_ops_tx, ipc_ops_rx) = crossbeam_channel::unbounded::<OverlayOp>();
        let ipc_info = Arc::new(std::sync::RwLock::new(TerminalInfo {
            font_w: compositor.cell_width() as u16,
            font_h: compositor.cell_height() as u16,
            cols: term_size.cols,
            rows: term_size.rows,
        }));

        // Create a waker that pokes the winit event loop whenever an IPC
        // overlay operation arrives.  Without this, overlay data sits in
        // the channel until the next keyboard / PTY event wakes winit.
        let ipc_proxy = self.proxy.clone();
        let ipc_waker: Box<dyn Fn() + Send + Sync> = Box::new(move || {
            let _ = ipc_proxy.send_event(TerminalEvent::OverlayReady);
        });

        let ipc_server: Option<IpcServer> = match IpcServer::start(
            ipc_ops_tx, ipc_info.clone(), Some(ipc_waker),
        ) {
            Ok(server) => {
                info!("IPC server started at {}", server.sock_path_str());
                Some(server)
            }
            Err(e) => {
                warn!("IPC server failed to start: {e} (overlays disabled)");
                None
            }
        };

        // Get the socket path for the child environment
        let sock_path: Option<String> = ipc_server
            .as_ref()
            .map(|s: &IpcServer| s.sock_path_str().to_string());

        // Determine shell
        let shell = self
            .config
            .shell
            .clone()
            .unwrap_or_else(platform::default_shell);

        info!("spawning shell: {shell}");

        // Create a waker that sends a PtyDataReady event to the event loop
        let proxy = self.proxy.clone();
        let waker = Box::new(move || {
            let _ = proxy.send_event(TerminalEvent::PtyDataReady);
        });

        // Spawn PTY with waker and IPC socket path
        let pty = match PtyManager::spawn_with_waker(
            &shell,
            term_size.cols,
            term_size.rows,
            term_size.pixel_width,
            term_size.pixel_height,
            waker,
            sock_path.as_deref(),
        ) {
            Ok(pty) => pty,
            Err(e) => {
                error!("failed to spawn PTY: {e}");
                event_loop.exit();
                return;
            }
        };

        debug!("PTY spawned, installing signals");

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
            _ipc_server: ipc_server,
            ipc_ops_rx: Some(ipc_ops_rx),
            ipc_info: Some(ipc_info),
            active_overlays: std::collections::HashMap::new(),
            active_animations: Vec::new(),
        });

        info!(
            "initialization complete in {:.1}ms",
            resumed_start.elapsed().as_secs_f64() * 1000.0
        );

        // Kick-start the render loop
        window.request_redraw();
    }

    /// Handle custom user events (PTY data ready, IPC overlay ready).
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: TerminalEvent) {
        trace!("user_event: {event:?}");
        match event {
            TerminalEvent::PtyDataReady | TerminalEvent::OverlayReady => {
                // PTY or IPC overlay data is available — request
                // a redraw to process it.
                if let Some(state) = &self.state {
                    state.window.request_redraw();
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else {
            return;
        };

        trace!("window_event: {event:?}");

        match event {
            WindowEvent::CloseRequested => {
                debug!("CloseRequested — exiting");
                event_loop.exit();
            }
            WindowEvent::Resized(s) => state.handle_resize(s),
            WindowEvent::ModifiersChanged(m) => state.modifiers = m.state(),
            WindowEvent::KeyboardInput { event, .. } => {
                state.handle_keyboard(&event);
            }
            WindowEvent::MouseInput {
                state: button_state,
                button,
                ..
            } => state.handle_mouse_input(button_state, button),
            WindowEvent::CursorMoved { position, .. } => state.handle_cursor_moved(position),
            WindowEvent::MouseWheel { delta, .. } => state.handle_mouse_wheel(delta),
            WindowEvent::RedrawRequested => {
                if state.handle_redraw(&self.config) {
                    event_loop.exit();
                }
            }
            WindowEvent::ScaleFactorChanged { .. } => state.handle_scale_factor_changed(),
            WindowEvent::Focused(focused) => state.handle_focused(focused),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        trace!("about_to_wait");
        event_loop.set_control_flow(ControlFlow::Wait);
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        debug!("SUSPENDED, state={}", if self.state.is_some() { "Some" } else { "None" });
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        debug!("EXITING event received");
    }
}

/// Sanitize window title: strip control characters, truncate to 256 chars.
///
/// Prevents malicious programs from injecting control sequences or
/// extremely long strings into the window titlebar.
fn sanitize_title(title: &str) -> String {
    title
        .chars()
        .filter(|c| !c.is_control())
        .take(256)
        .collect()
}

/// Write to PTY with error logging instead of silently ignoring failures.
fn pty_write(pty: &mut PtyManager, data: &[u8]) {
    if let Err(e) = pty.write(data) {
        warn!("PTY write failed: {e}");
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

    debug!("font size: {new_size}px (saved)");
    TerminalConfig::save_font_size(new_size);
}

fn main() {
    // Initialize tracing subscriber when the `logging` feature is enabled.
    #[cfg(feature = "logging")]
    {
        use tracing_subscriber::EnvFilter;
        tracing_subscriber::fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .with_target(false)
            .init();
    }

    debug!("main() entered, pid={}", std::process::id());

    // Load config
    let config = TerminalConfig::load();
    debug!("config loaded");

    // Create event loop with user events (for PTY wakeup)
    let event_loop = match EventLoop::<TerminalEvent>::with_user_event().build() {
        Ok(el) => el,
        Err(e) => {
            eprintln!("scry-terminal: failed to create event loop: {e}"); // Fatal — must print even without logging
            std::process::exit(1);
        }
    };
    debug!("event loop created");
    event_loop.set_control_flow(ControlFlow::Wait);

    let proxy = event_loop.create_proxy();

    // Run
    let mut app = TerminalApp::new(config, proxy);
    debug!("calling run_app");
    let result = event_loop.run_app(&mut app);
    debug!("run_app returned: {result:?}");
    if let Err(e) = result {
        // winit returns ExitFailure(1) as normal behavior when
        // event_loop.exit() is called — this is not an actual error.
        let msg = format!("{e}");
        if !msg.contains("Exit Failure") {
            eprintln!("scry-terminal: event loop error: {e}"); // Fatal — must print even without logging
        }
    }
    debug!("main() exiting");
}
