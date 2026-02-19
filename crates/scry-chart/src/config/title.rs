// SPDX-License-Identifier: MIT OR Apache-2.0
//! Title, subtitle, footer, and axis label configuration.

/// Text labels for the chart chrome (title, subtitle, footer, axis labels).
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct TitleConfig {
    /// Chart title.
    pub title: Option<String>,
    /// Chart subtitle (rendered below the title in smaller text).
    pub subtitle: Option<String>,
    /// Chart footer (rendered at the bottom edge).
    pub footer: Option<String>,
    /// X-axis label.
    pub x_label: Option<String>,
    /// Y-axis label.
    pub y_label: Option<String>,
}
