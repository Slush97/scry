// SPDX-License-Identifier: MIT OR Apache-2.0
//! Protocol backend trait for terminal graphics transmission.
//!
//! This module defines the [`ProtocolBackend`] trait that all graphics protocol
//! implementations must satisfy, along with shared types for image handles and
//! terminal positioning.

use tiny_skia::Pixmap;

use crate::PixelCanvasError;

// ---------------------------------------------------------------------------
// Terminal position
// ---------------------------------------------------------------------------

/// Where to place an image in terminal cell coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct TerminalPosition {
    /// Column (0-indexed).
    pub col: u16,
    /// Row (0-indexed).
    pub row: u16,
    /// Width in character cells.
    pub width_cells: u16,
    /// Height in character cells.
    pub height_cells: u16,
}

impl TerminalPosition {
    /// Create a new terminal position.
    #[must_use]
    pub const fn new(col: u16, row: u16, width_cells: u16, height_cells: u16) -> Self {
        Self {
            col,
            row,
            width_cells,
            height_cells,
        }
    }
}

// ---------------------------------------------------------------------------
// Image handle
// ---------------------------------------------------------------------------

/// Identifies a transmitted image for lifecycle management.
///
/// Different backends may use different internal identification schemes,
/// but all return an `ImageHandle` that can be used to remove or replace
/// the image later.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImageHandle {
    /// Internal image ID (protocol-specific).
    pub(crate) id: u32,
    /// Which protocol created this handle.
    pub(crate) protocol: ProtocolKind,
}

impl ImageHandle {
    /// The internal image ID.
    #[must_use]
    pub const fn id(&self) -> u32 {
        self.id
    }

    /// Which protocol this handle belongs to.
    #[must_use]
    pub const fn protocol(&self) -> ProtocolKind {
        self.protocol
    }
}

// ---------------------------------------------------------------------------
// Protocol kind
// ---------------------------------------------------------------------------

/// Enumeration of supported graphics protocols.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ProtocolKind {
    /// Kitty graphics protocol.
    Kitty,
    /// Sixel graphics protocol.
    Sixel,
    /// iTerm2 inline image protocol.
    Iterm2,
    /// Unicode halfblock fallback (no protocol needed).
    Halfblock,
    /// Native OS window via softbuffer (CPU framebuffer).
    Window,
    /// Native shared-memory IPC with scry-terminal.
    Native,
}

impl std::fmt::Display for ProtocolKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Kitty => write!(f, "Kitty"),
            Self::Sixel => write!(f, "Sixel"),
            Self::Iterm2 => write!(f, "iTerm2"),
            Self::Halfblock => write!(f, "Halfblock"),
            Self::Window => write!(f, "Window"),
            Self::Native => write!(f, "Native"),
        }
    }
}

// ---------------------------------------------------------------------------
// Font size
// ---------------------------------------------------------------------------

/// Terminal font size in pixels per character cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct FontSize {
    /// Width of a single character cell in pixels.
    pub width: u16,
    /// Height of a single character cell in pixels.
    pub height: u16,
}

impl FontSize {
    /// Create a new font size.
    #[must_use]
    pub const fn new(width: u16, height: u16) -> Self {
        Self { width, height }
    }
}

impl Default for FontSize {
    /// Reasonable default: 8×16 pixels per cell.
    fn default() -> Self {
        Self {
            width: 8,
            height: 16,
        }
    }
}

// ---------------------------------------------------------------------------
// Protocol backend trait
// ---------------------------------------------------------------------------

/// Abstraction over terminal graphics protocols.
///
/// Each implementation (Kitty, Sixel, Halfblock) knows how to transmit
/// pixel data to the terminal, manage image lifecycles, and clean up
/// when images are no longer needed.
///
/// # Implementors
///
/// - [`KittyBackend`](crate::transport::kitty::KittyBackend) — Kitty graphics protocol
/// - [`HalfblockBackend`](crate::transport::halfblock::HalfblockBackend) — Unicode fallback
pub trait ProtocolBackend: std::fmt::Debug + Send {
    /// Transmit a rendered pixmap to the terminal.
    ///
    /// Returns an opaque handle for later removal or replacement.
    fn transmit(
        &mut self,
        pixmap: &Pixmap,
        position: TerminalPosition,
        z_index: i32,
    ) -> Result<ImageHandle, PixelCanvasError>;

    /// Remove a previously transmitted image from the terminal.
    fn remove(&mut self, handle: &ImageHandle) -> Result<(), PixelCanvasError>;

    /// Remove all images managed by this backend.
    fn clear_all(&mut self) -> Result<(), PixelCanvasError>;

    /// Replace an existing image with new pixel data.
    ///
    /// Backends that support atomic replacement (e.g. Kitty) should override
    /// this to reuse the same image ID, avoiding any visual gap between
    /// the old and new image.
    ///
    /// The default implementation falls back to `remove` + `transmit`.
    fn replace(
        &mut self,
        handle: &ImageHandle,
        pixmap: &Pixmap,
        position: TerminalPosition,
        z_index: i32,
    ) -> Result<ImageHandle, PixelCanvasError> {
        self.remove(handle)?;
        self.transmit(pixmap, position, z_index)
    }

    /// Transmit only the changed tiles of a pixmap.
    ///
    /// Each [`DirtyTile`](crate::rasterize::DirtyTile) specifies a sub-region of the pixmap that has
    /// changed since the last frame. The backend transmits only those
    /// regions, dramatically reducing bandwidth for partially-animated
    /// scenes (e.g. dashboards where one chart updates at a time).
    ///
    /// The default implementation ignores tiles and falls back to a full
    /// [`replace()`](Self::replace).
    fn transmit_tiles(
        &mut self,
        handle: &ImageHandle,
        pixmap: &Pixmap,
        position: TerminalPosition,
        z_index: i32,
        _dirty_tiles: &[crate::rasterize::DirtyTile],
    ) -> Result<ImageHandle, PixelCanvasError> {
        self.replace(handle, pixmap, position, z_index)
    }

    /// Whether this backend supports alpha transparency.
    fn supports_alpha(&self) -> bool;

    /// The protocol kind this backend implements.
    fn protocol_kind(&self) -> ProtocolKind;

    /// Poll for asynchronous events from the terminal (non-blocking).
    ///
    /// Only meaningful for `NativeBackend` — other backends are no-ops.
    /// Call this in animation loops between frames to process click and
    /// visibility events. Updates internal pause/visibility state.
    fn poll_events(&mut self) {}

    /// Whether the given overlay is currently paused (click-to-toggle).
    ///
    /// Only meaningful for `NativeBackend` — other backends always return false.
    fn is_overlay_paused(&self, _id: u32) -> bool {
        false
    }
}
