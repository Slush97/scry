// SPDX-License-Identifier: MIT OR Apache-2.0
//! Data value label configuration.

use std::sync::Arc;

use crate::formatter::TickFormatter;

/// Configuration for data value labels on bars, points, etc.
#[derive(Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct DataLabelConfig {
    /// Whether to show data value labels.
    pub show: bool,
    /// Custom formatter for data value labels.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub formatter: Option<Arc<dyn TickFormatter>>,
}

impl std::fmt::Debug for DataLabelConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DataLabelConfig")
            .field("show", &self.show)
            .field("formatter", &self.formatter.as_ref().map(|_| ".."))
            .finish()
    }
}
