// SPDX-License-Identifier: MIT OR Apache-2.0
//! Terminal graphics transport layer.
//!
//! This module provides the [`ProtocolBackend`] trait and its implementations
//! for transmitting pixel data to the terminal. It also provides the [`Picker`]
//! for auto-detecting the best available protocol.
//!
//! # Protocol Selection
//!
//! Use [`Picker::detect()`] to auto-detect the best protocol at runtime.
//! The detection order is: **Kitty → iTerm2 → Sixel → Halfblock**.
//!
//! For manual control, construct backends directly:
//!
//! ```no_run
//! use scry_engine::transport::{Picker, FontSize};
//!
//! let picker = Picker::detect();
//! println!("Detected: {:?} at {:?}", picker.protocol(), picker.font_size());
//! ```
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
pub mod capabilities;
#[cfg(feature = "kitty")]
pub mod kitty;

pub mod halfblock;
pub mod picker;
pub mod probe;

#[cfg(feature = "sixel")]
pub mod sixel;

#[cfg(feature = "iterm2")]
pub mod iterm2;

#[cfg(feature = "window")]
pub mod window;

#[cfg(feature = "shm")]
pub(crate) mod shm;

pub use backend::{FontSize, ImageHandle, ProtocolBackend, ProtocolKind, TerminalPosition};
pub use capabilities::{
    DetectionMethod, KittyFeatures, Multiplexer, ProbeConfig, SixelFeatures, TerminalCapabilities,
};
pub use picker::Picker;
