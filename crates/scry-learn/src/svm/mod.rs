// SPDX-License-Identifier: MIT OR Apache-2.0
//! Support Vector Machine classifiers and regressors.
//!
//! Provides [`LinearSVC`] and [`LinearSVR`] using Pegasos SGD, plus
//! [`KernelSVC`] with SMO for non-linear classification and [`KernelSVR`]
//! for non-linear regression.

pub mod kernel;
pub mod kernel_svr;
pub mod linear;

pub use kernel::{Gamma, Kernel, KernelSVC};
pub use kernel_svr::KernelSVR;
pub use linear::{LinearSVC, LinearSVR};
