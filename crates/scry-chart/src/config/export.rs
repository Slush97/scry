// SPDX-License-Identifier: MIT OR Apache-2.0
//! Export DPI configuration.

/// Export-related settings.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct ExportConfig {
    /// Export DPI (dots per inch). Default: 144.
    ///
    /// The export functions scale the output pixel dimensions by `dpi / 144`.
    /// Set to 288 for 2x (Retina) resolution, 72 for lower-res, etc.
    pub dpi: u32,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self { dpi: 144 }
    }
}
