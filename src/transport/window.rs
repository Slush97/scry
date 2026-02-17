// SPDX-License-Identifier: MIT OR Apache-2.0
//! Native OS window backend via `winit` + `softbuffer`.
//!
//! Bypasses all terminal protocol overhead by blitting tiny-skia pixels
//! directly into a CPU framebuffer. No GPU required.
//!
//! # Usage
//!
//! Because `winit` requires the event loop to own the main thread, this
//! backend provides [`run_loop`] for static scenes and [`run_loop_continuous`]
//! for animations with keyboard input.
//!
//! ```no_run
//! use scry_engine::transport::window::{run_loop_continuous, LoopAction};
//! use scry_engine::scene::{PixelCanvas, Color};
//! use scry_engine::rasterize::Rasterizer;
//!
//! run_loop_continuous(400, 300, "My App", true, |backend, keys, (w, h)| {
//!     let canvas = PixelCanvas::new(w, h)
//!         .background(Color::from_rgba8(30, 30, 40, 255))
//!         .circle(200.0, 150.0, 80.0)
//!             .fill(Color::from_rgba8(70, 130, 180, 255))
//!             .done();
//!     let pixmap = Rasterizer::rasterize(&canvas).unwrap();
//!     backend.blit(&pixmap).unwrap();
//!     LoopAction::Continue
//! }).unwrap();
//! ```

use std::num::NonZeroU32;
use std::sync::Arc;

use tiny_skia::Pixmap;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::transport::backend::{ImageHandle, ProtocolBackend, ProtocolKind, TerminalPosition};
use crate::PixelCanvasError;

// ---------------------------------------------------------------------------
// WindowBackend
// ---------------------------------------------------------------------------

/// Backend that blits pixels into a native OS window via `softbuffer`.
///
/// Unlike terminal backends, this does not serialize pixels through escape
/// sequences — it copies RGBA data directly into a CPU framebuffer.
#[derive(Debug)]
pub struct WindowBackend {
    surface: softbuffer::Surface<Arc<Window>, Arc<Window>>,
    window: Arc<Window>,
    width: u32,
    height: u32,
    next_id: u32,
}

impl WindowBackend {
    /// Create a new window backend from an existing winit window.
    ///
    /// # Errors
    ///
    /// Returns an error if the softbuffer context or surface cannot be created.
    pub fn new(window: Arc<Window>, width: u32, height: u32) -> Result<Self, PixelCanvasError> {
        let context = softbuffer::Context::new(window.clone()).map_err(|e| {
            PixelCanvasError::Rasterization(format!("softbuffer context failed: {e}"))
        })?;
        let mut surface = softbuffer::Surface::new(&context, window.clone()).map_err(|e| {
            PixelCanvasError::Rasterization(format!("softbuffer surface failed: {e}"))
        })?;

        if let (Some(w), Some(h)) = (NonZeroU32::new(width), NonZeroU32::new(height)) {
            surface.resize(w, h).map_err(|e| {
                PixelCanvasError::Rasterization(format!("surface resize failed: {e}"))
            })?;
        }

        Ok(Self {
            surface,
            window,
            width,
            height,
            next_id: 1,
        })
    }

    /// Blit a tiny-skia pixmap directly into the window framebuffer.
    ///
    /// Converts RGBA → 0RGB packed u32 and presents immediately.
    ///
    /// # Errors
    ///
    /// Returns an error if the surface buffer cannot be acquired or presented.
    pub fn blit(&mut self, pixmap: &Pixmap) -> Result<(), PixelCanvasError> {
        let pw = pixmap.width();
        let ph = pixmap.height();

        // Resize surface if pixmap dimensions changed
        if pw != self.width || ph != self.height {
            if let (Some(w), Some(h)) = (NonZeroU32::new(pw), NonZeroU32::new(ph)) {
                self.surface.resize(w, h).map_err(|e| {
                    PixelCanvasError::Rasterization(format!("surface resize failed: {e}"))
                })?;
                self.width = pw;
                self.height = ph;
            }
        }

        let mut buf = self.surface.buffer_mut().map_err(|e| {
            PixelCanvasError::Rasterization(format!("surface buffer_mut failed: {e}"))
        })?;

        let data = pixmap.data();
        let pixel_count = (pw * ph) as usize;

        for i in 0..pixel_count {
            let offset = i * 4;
            let r = data[offset] as u32;
            let g = data[offset + 1] as u32;
            let b = data[offset + 2] as u32;
            // softbuffer uses 0RGB: bits [23:16]=R, [15:8]=G, [7:0]=B
            buf[i] = (r << 16) | (g << 8) | b;
        }

        buf.present()
            .map_err(|e| PixelCanvasError::Rasterization(format!("surface present failed: {e}")))?;

        Ok(())
    }

    /// Get a reference to the underlying winit window.
    pub fn window(&self) -> &Window {
        &self.window
    }

    /// Current surface width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Current surface height.
    pub fn height(&self) -> u32 {
        self.height
    }
}

impl ProtocolBackend for WindowBackend {
    fn transmit(
        &mut self,
        pixmap: &Pixmap,
        _position: TerminalPosition,
        _z_index: i32,
    ) -> Result<ImageHandle, PixelCanvasError> {
        self.blit(pixmap)?;
        let id = self.next_id;
        self.next_id += 1;
        Ok(ImageHandle {
            id,
            protocol: ProtocolKind::Window,
        })
    }

    fn remove(&mut self, _handle: &ImageHandle) -> Result<(), PixelCanvasError> {
        // No-op: the window just shows the latest frame.
        Ok(())
    }

    fn clear_all(&mut self) -> Result<(), PixelCanvasError> {
        // No-op: the window just shows the latest frame.
        Ok(())
    }

    fn supports_alpha(&self) -> bool {
        true
    }

    fn protocol_kind(&self) -> ProtocolKind {
        ProtocolKind::Window
    }
}

// ---------------------------------------------------------------------------
// LoopAction + WindowKeyEvent
// ---------------------------------------------------------------------------

/// Action returned by the render callback to control the loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LoopAction {
    /// Keep running.
    Continue,
    /// Exit the loop.
    Exit,
}

/// A keyboard event forwarded from winit to the render callback.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct WindowKeyEvent {
    /// The physical key code.
    pub code: KeyCode,
    /// Whether this is a press (true) or release (false).
    pub pressed: bool,
}

// ---------------------------------------------------------------------------
// run_loop — static scenes (Wait control flow)
// ---------------------------------------------------------------------------

/// Open a native window and run a rendering loop.
///
/// The `render` callback is called on each frame (driven by `RedrawRequested`).
/// It receives a `&mut WindowBackend` that can blit pixmaps into the window.
/// The loop exits when the user closes the window.
///
/// # Errors
///
/// Returns an error if the event loop or window cannot be created.
///
/// # Example
///
/// ```no_run
/// use scry_engine::transport::window::run_loop;
/// use scry_engine::scene::{PixelCanvas, Color};
/// use scry_engine::rasterize::Rasterizer;
///
/// run_loop(400, 300, "scry", |backend| {
///     let canvas = PixelCanvas::new(400, 300)
///         .background(Color::BLACK);
///     let pixmap = Rasterizer::rasterize(&canvas).unwrap();
///     backend.blit(&pixmap);
/// }).unwrap();
/// ```
pub fn run_loop<F>(width: u32, height: u32, title: &str, render: F) -> Result<(), PixelCanvasError>
where
    F: FnMut(&mut WindowBackend) + 'static,
{
    let event_loop = EventLoop::new()
        .map_err(|e| PixelCanvasError::Rasterization(format!("event loop creation failed: {e}")))?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);

    let mut app = App {
        width,
        height,
        title: title.to_string(),
        backend: None,
        render: Box::new(render),
        resizable: false,
    };

    event_loop
        .run_app(&mut app)
        .map_err(|e| PixelCanvasError::Rasterization(format!("event loop failed: {e}")))?;

    Ok(())
}

struct App {
    width: u32,
    height: u32,
    title: String,
    backend: Option<WindowBackend>,
    render: Box<dyn FnMut(&mut WindowBackend)>,
    resizable: bool,
}

impl std::fmt::Debug for App {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("App")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("title", &self.title)
            .finish_non_exhaustive()
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.backend.is_some() {
            return;
        }

        let attrs = Window::default_attributes()
            .with_title(&self.title)
            .with_inner_size(winit::dpi::PhysicalSize::new(self.width, self.height))
            .with_resizable(self.resizable);

        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                eprintln!("scry: failed to create window: {e}");
                event_loop.exit();
                return;
            }
        };

        match WindowBackend::new(window.clone(), self.width, self.height) {
            Ok(backend) => {
                self.backend = Some(backend);
                window.request_redraw();
            }
            Err(e) => {
                eprintln!("scry: failed to create window backend: {e}");
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let Some(ref mut backend) = self.backend {
                    (self.render)(backend);
                }
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// run_loop_continuous — animated scenes (Poll control flow + keyboard)
// ---------------------------------------------------------------------------

/// Open a native window with continuous rendering and keyboard input.
///
/// Unlike [`run_loop`], this uses `ControlFlow::Poll` for continuous frame
/// delivery and forwards keyboard events to the render callback.
///
/// The `render` callback receives:
/// - `&mut WindowBackend` for blitting
/// - `&[WindowKeyEvent]` — keyboard events since last frame
/// - `(u32, u32)` — current window size in pixels (updates on resize)
///
/// Return [`LoopAction::Exit`] to close the window.
///
/// # Errors
///
/// Returns an error if the event loop or window cannot be created.
pub fn run_loop_continuous<F>(
    width: u32,
    height: u32,
    title: &str,
    resizable: bool,
    render: F,
) -> Result<(), PixelCanvasError>
where
    F: FnMut(&mut WindowBackend, &[WindowKeyEvent], (u32, u32)) -> LoopAction + 'static,
{
    let event_loop = EventLoop::new()
        .map_err(|e| PixelCanvasError::Rasterization(format!("event loop creation failed: {e}")))?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = ContinuousApp {
        width,
        height,
        title: title.to_string(),
        backend: None,
        render: Box::new(render),
        pending_keys: Vec::new(),
        resizable,
    };

    event_loop
        .run_app(&mut app)
        .map_err(|e| PixelCanvasError::Rasterization(format!("event loop failed: {e}")))?;

    Ok(())
}

struct ContinuousApp {
    width: u32,
    height: u32,
    title: String,
    backend: Option<WindowBackend>,
    render: Box<dyn FnMut(&mut WindowBackend, &[WindowKeyEvent], (u32, u32)) -> LoopAction>,
    pending_keys: Vec<WindowKeyEvent>,
    resizable: bool,
}

impl std::fmt::Debug for ContinuousApp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContinuousApp")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("title", &self.title)
            .finish_non_exhaustive()
    }
}

impl ApplicationHandler for ContinuousApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.backend.is_some() {
            return;
        }

        let attrs = Window::default_attributes()
            .with_title(&self.title)
            .with_inner_size(winit::dpi::PhysicalSize::new(self.width, self.height))
            .with_resizable(self.resizable);

        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                eprintln!("scry: failed to create window: {e}");
                event_loop.exit();
                return;
            }
        };

        match WindowBackend::new(window.clone(), self.width, self.height) {
            Ok(backend) => {
                self.backend = Some(backend);
                window.request_redraw();
            }
            Err(e) => {
                eprintln!("scry: failed to create window backend: {e}");
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                self.width = size.width.max(1);
                self.height = size.height.max(1);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.pending_keys.push(WindowKeyEvent {
                        code,
                        pressed: event.state.is_pressed(),
                    });
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(ref mut backend) = self.backend {
                    let keys: Vec<WindowKeyEvent> = self.pending_keys.drain(..).collect();
                    let action = (self.render)(backend, &keys, (self.width, self.height));
                    if action == LoopAction::Exit {
                        event_loop.exit();
                        return;
                    }
                    backend.window().request_redraw();
                }
            }
            _ => {}
        }
    }
}
