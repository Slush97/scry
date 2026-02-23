// SPDX-License-Identifier: MIT OR Apache-2.0
//! Scry Terminal — a GPU-accelerated terminal emulator.
//!
//! This is the library root that exposes all modules for the terminal.

pub mod compositor;
pub mod config;
pub mod error;
pub mod grid;
pub mod input;
#[cfg(unix)]
pub mod ipc_server;
pub mod performance;
pub mod platform;
pub mod pty;
pub mod security;
pub mod selection;
pub mod text;
pub mod vt;
