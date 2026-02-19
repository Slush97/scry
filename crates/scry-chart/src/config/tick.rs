// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tick label formatting and stepping configuration.

use std::sync::Arc;

use crate::axis::LabelRotation;
use crate::formatter::{LocaleConfig, TickFormatter};

/// Tick label formatting, rotation, and step configuration.
#[derive(Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct TickConfig {
    /// Rotation for X-axis tick labels.
    pub x_tick_rotation: LabelRotation,
    /// Custom tick formatter for the X axis.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub x_tick_formatter: Option<Arc<dyn TickFormatter>>,
    /// Custom tick formatter for the Y axis.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub y_tick_formatter: Option<Arc<dyn TickFormatter>>,
    /// Fixed tick step for the X axis (overrides adaptive generation).
    pub x_tick_step: Option<f64>,
    /// Fixed tick step for the Y axis (overrides adaptive generation).
    pub y_tick_step: Option<f64>,
    /// Locale configuration for number formatting.
    pub locale: Option<LocaleConfig>,
}

impl std::fmt::Debug for TickConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TickConfig")
            .field("x_tick_rotation", &self.x_tick_rotation)
            .field(
                "x_tick_formatter",
                &self.x_tick_formatter.as_ref().map(|_| ".."),
            )
            .field(
                "y_tick_formatter",
                &self.y_tick_formatter.as_ref().map(|_| ".."),
            )
            .field("x_tick_step", &self.x_tick_step)
            .field("y_tick_step", &self.y_tick_step)
            .field("locale", &self.locale)
            .finish()
    }
}
