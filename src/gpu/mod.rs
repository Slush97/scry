// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared GPU device and pipeline registry.
//!
//! This module provides [`GpuDevice`] — the single mandatory GPU entry
//! point — and [`PipelineRegistry`] which lazily compiles GPU pipelines
//! on first access.

pub mod error;
mod device;
pub mod buffer_pool;
pub mod health;
pub mod pipeline_registry;
pub mod pipelines_3d;

pub use buffer_pool::BufferPool;
pub use device::{GpuDevice, GpuInfo};
pub use error::GpuError;
pub use health::{GpuHealth, GpuHealthMonitor, SharedHealthMonitor};
pub use pipeline_registry::{PipelineRegistry, Pipelines2D, PipelinesSdf};
pub use pipelines_3d::Pipelines3D;
