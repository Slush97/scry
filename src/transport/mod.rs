//! Terminal graphics transport layer.
//!
//! This module provides the [`ProtocolBackend`] trait and its implementations
//! for transmitting pixel data to the terminal. It also provides the [`Picker`]
//! for auto-detecting the best available protocol.
//!
//! # Backends
//!
//! | Backend | Feature | Quality | Transparency |
//! |---------|---------|---------|-------------|
//! | [`kitty::KittyBackend`] | `kitty` | Pixel-perfect | ✅ |
//! | `sixel::SixelBackend` | `sixel` | 256-color quantized | ❌ |
//! | `iterm2::Iterm2Backend` | `iterm2` | PNG inline | ✅ |
//! | [`halfblock::HalfblockBackend`] | always | 1×2 per cell | ❌ |

pub mod backend;
#[cfg(feature = "kitty")]
pub mod kitty;

pub mod halfblock;
pub mod picker;

#[cfg(feature = "sixel")]
pub mod sixel;

#[cfg(feature = "iterm2")]
pub mod iterm2;

#[cfg(feature = "shm")]
pub(crate) mod shm;

pub use backend::{FontSize, ImageHandle, ProtocolBackend, ProtocolKind, TerminalPosition};
pub use picker::Picker;
