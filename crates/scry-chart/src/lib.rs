// SPDX-License-Identifier: MIT OR Apache-2.0
//! # scry-chart
//!
//! Pixel-perfect TUI charts built on [`scry-engine`](https://docs.rs/scry-engine).
//!
//! Draw anti-aliased scatter plots, line charts, bar charts, and histograms
//! directly in the terminal — rendered as actual pixels via Kitty/Sixel,
//! with graceful halfblock fallback.
//!
//! ## Quick Start
//!
//! ```ignore
//! use scry_chart::prelude::*;
//!
//! let chart = Chart::line(&[1.0, 4.0, 2.0, 8.0, 5.0])
//!     .title("My Data")
//!     .theme(Theme::dark())
//!     .build();
//!
//! frame.render_stateful_widget(chart.widget(), area, &mut chart_state);
//! ```

#![warn(missing_docs)]
#![deny(unsafe_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_fields_in_debug)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::redundant_pub_crate)]
#![allow(clippy::use_self)]
#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(clippy::similar_names)]
#![allow(clippy::suspicious_operation_groupings)]

pub mod annotation;
pub mod axis;
pub mod chart;
pub mod chart3d;
pub mod colormap;
pub mod cursor;
pub mod data;
pub mod error;
pub mod export;
pub mod formatter;
#[cfg(feature = "inline")]
pub mod inline;
pub mod layout;
pub mod legend;
pub mod margin;
pub mod scale;
pub mod streaming;
pub mod subplot;
pub mod svg_export;
pub mod theme;
pub mod time_scale;
#[cfg(feature = "widget")]
pub mod widget;
pub mod zoom;

/// Convenience re-exports for common usage.
pub mod prelude {
    pub use crate::annotation::Annotation;
    pub use crate::axis::LabelRotation;
    pub use crate::chart::scatter::Marker;
    pub use crate::chart::{
        BarChart, BoxPlot, BubbleChart, CandlestickChart, Chart, FunnelChart, GaugeChart, Heatmap,
        Histogram, LineChart, LollipopChart, OhlcEntry, PieChart, RadarChart, ReferenceLine,
        ScatterChart, Sparkline, SparklineKind, ViolinPlot, WaterfallChart,
    };
    pub use crate::chart3d::camera::Camera3D;
    #[cfg(feature = "gpu")]
    pub use crate::chart3d::wgpu_backend::{WgpuContext, WgpuRasterizer3D};
    pub use crate::chart3d::{Chart3D, Rasterizer3D};
    pub use crate::colormap::{colormap_from_name, Colormap};
    pub use crate::cursor::{CursorState, DataPoint};
    pub use crate::data::{FillPattern, GapPolicy, GradientFill, Series, SeriesStyle};
    pub use crate::error::ChartError;
    pub use crate::export::{render_to_png, save_png};
    pub use crate::legend::{LegendConfig, LegendOrientation, LegendPosition};
    pub use crate::margin::Margin;
    pub use crate::streaming::StreamingChart;
    pub use crate::subplot::{SharedAxisMode, SubplotGrid};
    pub use crate::svg_export::{render_to_svg, save_svg};
    pub use crate::theme::Theme;
    #[cfg(feature = "widget")]
    pub use crate::widget::{Chart3DState, Chart3DWidget, ChartState, ChartWidget};
    pub use crate::zoom::ZoomState;
}
