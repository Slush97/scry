// SPDX-License-Identifier: MIT OR Apache-2.0
//! Secondary (dual) Y-axis configuration.

use std::sync::Arc;

use crate::formatter::TickFormatter;

/// Configuration for the secondary (right) Y axis.
#[derive(Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct SecondaryAxisConfig {
    /// Secondary Y-axis label (rendered on the right side).
    pub label: Option<String>,
    /// Manual secondary Y-axis domain override.
    pub range: Option<(f64, f64)>,
    /// Custom tick formatter for the secondary Y axis.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub formatter: Option<Arc<dyn TickFormatter>>,
    /// Series indices to plot against the secondary Y axis.
    pub series_indices: Vec<usize>,
}

impl std::fmt::Debug for SecondaryAxisConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecondaryAxisConfig")
            .field("label", &self.label)
            .field("range", &self.range)
            .field("formatter", &self.formatter.as_ref().map(|_| ".."))
            .field("series_indices", &self.series_indices)
            .finish()
    }
}
