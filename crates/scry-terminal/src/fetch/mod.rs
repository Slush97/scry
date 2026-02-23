// SPDX-License-Identifier: MIT OR Apache-2.0
//! Native fetch splash — GPU-composited system info display.
//!
//! Shows a banner with a logo (distro auto-detect, sacred geometry, or custom)
//! on the left and system information on the right. Renders as a compositor
//! overlay that fades in on startup and fades out after a configurable duration
//! or on any keypress.

mod logo;
mod overlay;
mod renderer;
mod sysinfo;

pub use overlay::FetchOverlay;
