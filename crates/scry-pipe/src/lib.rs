// SPDX-License-Identifier: MIT OR Apache-2.0
//! scry-pipe: cross-language feature engineering compiler.
//!
//! Define ML feature pipelines once. Execute at Rust speed.
//! Compile to standalone zero-dependency Rust or WASM binaries.
//!
//! # Overview
//!
//! scry-pipe bridges the gap between Python-based ML training and
//! production deployment. Data scientists define feature engineering
//! pipelines (scaling, encoding, imputation, clipping, etc.) in Python;
//! scry-pipe compiles them into standalone executables with all fitted
//! parameters baked in, guaranteeing exact numerical parity between
//! training and serving.
//!
//! # Modules
//!
//! - [`ir`] — Intermediate representation: [`TransformOp`], [`PipelineDef`],
//!   and supporting types.
//! - [`engine`] — Runtime transform engine: [`PipelineEngine`].
//! - [`error`] — Crate error type: [`PipeError`].

pub mod engine;
pub mod error;
pub mod ir;

#[cfg(feature = "codegen")]
pub mod codegen;

pub use engine::PipelineEngine;
pub use error::PipeError;
pub use ir::{DType, FeatureSpec, ImputeStrategy, PipelineDef, PipelineStep, TransformOp};

#[cfg(feature = "codegen")]
pub use codegen::RustCodegen;

