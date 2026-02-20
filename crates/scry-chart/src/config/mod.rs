// SPDX-License-Identifier: MIT OR Apache-2.0
//! Decomposed chart configuration sub-structs.
//!
//! [`ChartConfig`](crate::chart::ChartConfig) delegates to these focused sub-structs,
//! each owning a single concern (titles, axis ranges, tick formatting, etc.).

mod axis;
mod data_labels;
mod export;
mod overlay;
mod secondary;
mod tick;
mod title;

pub use axis::{AspectRatio, AxisRangeConfig};
pub use data_labels::DataLabelConfig;
pub use export::ExportConfig;
pub use overlay::OverlayConfig;
pub use secondary::SecondaryAxisConfig;
pub use tick::TickConfig;
pub use title::TitleConfig;
