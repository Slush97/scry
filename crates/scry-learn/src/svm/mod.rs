// SPDX-License-Identifier: MIT OR Apache-2.0
//! Support Vector Machine classifiers and regressors.
//!
//! Provides [`LinearSVC`] and [`LinearSVR`] using Pegasos SGD, plus
//! [`KernelSVC`] with SMO for non-linear classification and [`KernelSVR`]
//! for non-linear regression.

pub mod linear;
pub mod kernel;
pub mod kernel_svr;

pub use linear::{LinearSVC, LinearSVR};
pub use kernel::{KernelSVC, Kernel, Gamma};
pub use kernel_svr::KernelSVR;
