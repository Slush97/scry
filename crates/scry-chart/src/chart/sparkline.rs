// SPDX-License-Identifier: MIT OR Apache-2.0
//! Sparkline chart type — minimal inline data visualizations.
//!
//! Sparklines are chrome-free: no axes, no title, no margins. Pure data ink.
//! Three kinds are supported: Line (polyline), Bar (thin bars), and WinLoss
//! (bars above/below center).

use crate::chart::{Chart, ChartConfig};
use crate::spec::ChartSpec;
use scry_engine::style::Color;

/// The visual kind of sparkline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SparklineKind {
    /// Polyline connecting data points (default).
    Line,
    /// Thin vertical bars.
    Bar,
    /// Win/loss: bars above center for positive, below for negative.
    WinLoss,
}

/// A sparkline — a minimal inline chart with no chrome.
///
/// # Examples
///
/// ```
/// use scry_chart::chart::Charts;
///
/// let chart = Charts::sparkline(&[3.0, 7.0, 4.0, 8.0, 2.0, 9.0, 5.0])
///     .build();
/// ```
#[derive(Clone, Debug)]
#[must_use]
pub struct Sparkline {
    /// Data values.
    pub(crate) values: Vec<f64>,
    /// Visual kind.
    pub(crate) kind: SparklineKind,
    /// Line/bar color override (uses theme series color if None).
    pub(crate) color: Option<Color>,
    /// Whether to fill the area under the line.
    pub(crate) fill: bool,
    /// Line width (default: 1.5).
    pub(crate) line_width: f32,
    /// Shared config (mostly unused, but needed for theme).
    pub(crate) config: ChartConfig,
}

impl Sparkline {
    /// Create a new line sparkline from data values.
    pub fn new(values: Vec<f64>) -> Self {
        Self {
            values,
            kind: SparklineKind::Line,
            color: None,
            fill: false,
            line_width: 1.5,
            config: ChartConfig::default(),
        }
    }

    /// Override the sparkline color.
    pub fn color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }

    /// Fill the area under the line.
    pub fn filled(mut self) -> Self {
        self.fill = true;
        self
    }

    /// Set the line width.
    pub fn line_width(mut self, width: f32) -> Self {
        self.line_width = width;
        self
    }

    /// Render as bar sparkline.
    pub fn bar(mut self) -> Self {
        self.kind = SparklineKind::Bar;
        self
    }

    /// Render as win/loss sparkline.
    pub fn win_loss(mut self) -> Self {
        self.kind = SparklineKind::WinLoss;
        self
    }

    /// Build into a Chart enum variant.
    #[must_use]
    pub fn build(self) -> Chart {
        Box::new(self) as Chart
    }

    /// Build with validation.
    pub fn try_build(self) -> Result<Chart, crate::error::ChartError> {
        if self.values.is_empty() {
            return Err(crate::error::ChartError::EmptyData);
        }
        Ok(Box::new(self) as Chart)
    }
}

impl ChartSpec for Sparkline {
    fn render(&self, w: u32, h: u32) -> crate::layout::RenderedChart {
        crate::layout::sparkline::render_sparkline(self, w, h)
    }
    fn config(&self) -> Option<&ChartConfig> {
        Some(&self.config)
    }
    fn config_mut(&mut self) -> Option<&mut ChartConfig> {
        Some(&mut self.config)
    }
    fn clone_boxed(&self) -> Box<dyn ChartSpec> {
        Box::new(self.clone())
    }
}
