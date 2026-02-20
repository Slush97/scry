// SPDX-License-Identifier: MIT OR Apache-2.0
//! Gantt chart type — project schedule / timeline visualization.
//!
//! Horizontal bars span `[start, end]` on the X axis, positioned on a
//! categorical Y axis (task name). Tasks can carry an optional `group`
//! label for color-coding and an optional `progress` fraction (0.0–1.0)
//! for partial-completion shading.

use crate::chart::config_builder::{
    chart_config_axis_labels, chart_config_core, chart_config_formatters, chart_config_grid,
    chart_config_legend, chart_config_locale, chart_config_margin, chart_config_subtitle_footer,
    chart_config_tick_rotation, chart_config_tick_steps,
};
use crate::chart::{Chart, ChartConfig};
use crate::spec::ChartSpec;

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

/// A single task bar in a Gantt chart.
///
/// # Examples
///
/// ```
/// use scry_chart::chart::GanttTask;
///
/// let task = GanttTask::new("Design", 0.0, 5.0)
///     .group("Engineering")
///     .progress(0.8);
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GanttTask {
    /// Task name (displayed on the Y axis).
    pub label: String,
    /// Start position on the X axis (numeric — day index, epoch, etc.).
    pub start: f64,
    /// End position on the X axis.
    pub end: f64,
    /// Optional grouping key for color-coding.
    pub group: Option<String>,
    /// Optional completion fraction (0.0–1.0) rendered as a darker overlay.
    pub progress: Option<f32>,
}

impl GanttTask {
    /// Create a new Gantt task.
    pub fn new(label: impl Into<String>, start: f64, end: f64) -> Self {
        Self {
            label: label.into(),
            start,
            end,
            group: None,
            progress: None,
        }
    }

    /// Set the group for color-coding.
    pub fn group(mut self, group: impl Into<String>) -> Self {
        self.group = Some(group.into());
        self
    }

    /// Set the progress fraction (0.0–1.0).
    pub fn progress(mut self, pct: f32) -> Self {
        self.progress = Some(pct.clamp(0.0, 1.0));
        self
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// A Gantt chart — horizontal bars on a categorical Y axis for scheduling.
///
/// # Examples
///
/// ```
/// use scry_chart::chart::{GanttChart, GanttTask};
///
/// let chart = GanttChart::new(vec![
///     GanttTask::new("Research", 0.0, 3.0).group("Phase 1"),
///     GanttTask::new("Design", 2.0, 6.0).group("Phase 1"),
///     GanttTask::new("Implement", 5.0, 12.0).group("Phase 2"),
///     GanttTask::new("Test", 10.0, 14.0).group("Phase 2"),
///     GanttTask::new("Deploy", 14.0, 15.0).group("Phase 3"),
/// ])
/// .title("Project Timeline")
/// .build();
/// ```
#[derive(Clone, Debug)]
#[must_use]
pub struct GanttChart {
    /// Task list.
    pub(crate) tasks: Vec<GanttTask>,
    /// Shared config.
    pub(crate) config: ChartConfig,
    /// Bar height as a fraction of row height (0.0–1.0, default 0.6).
    pub(crate) bar_height: f32,
    /// Whether to render task labels on bars (default: true).
    pub(crate) show_labels: bool,
    /// Whether to show progress overlay bars (default: true).
    pub(crate) show_progress: bool,
    /// Whether to show numeric date labels on bar edges (default: false).
    pub(crate) show_dates: bool,
    /// Whether to use time-based formatting on the X axis (default: false).
    /// When true, X values are interpreted as Unix epoch seconds and the
    /// axis labels show human-readable dates/times.
    pub(crate) use_time_axis: bool,
}

impl GanttChart {
    /// Create a new Gantt chart from a list of tasks.
    pub fn new(tasks: Vec<GanttTask>) -> Self {
        Self {
            tasks,
            config: ChartConfig::default(),
            bar_height: 0.6,
            show_labels: true,
            show_progress: true,
            show_dates: false,
            use_time_axis: false,
        }
    }

    // --- Generated common methods ---
    chart_config_core!();
    chart_config_axis_labels!();
    chart_config_legend!();
    chart_config_grid!();
    chart_config_tick_rotation!();
    chart_config_formatters!();
    chart_config_locale!();
    chart_config_tick_steps!();
    chart_config_subtitle_footer!();
    chart_config_margin!();

    /// Set bar height as fraction of row height (clamped 0.2–0.95).
    pub fn bar_height(mut self, frac: f32) -> Self {
        self.bar_height = frac.clamp(0.2, 0.95);
        self
    }

    /// Hide task labels on bars.
    pub fn no_labels(mut self) -> Self {
        self.show_labels = false;
        self
    }

    /// Disable progress overlay rendering.
    pub fn no_progress(mut self) -> Self {
        self.show_progress = false;
        self
    }

    /// Show numeric start/end date labels on bar edges.
    pub fn show_dates(mut self) -> Self {
        self.show_dates = true;
        self
    }

    /// Enable time-based X-axis formatting.
    ///
    /// When enabled, X values are treated as Unix epoch seconds and the
    /// axis labels display human-readable dates/times. The granularity
    /// (seconds, minutes, hours, days, months, years) is auto-selected
    /// based on the time span.
    pub fn time_axis(mut self) -> Self {
        self.use_time_axis = true;
        self
    }

    /// Add a task.
    pub fn add_task(mut self, task: GanttTask) -> Self {
        self.tasks.push(task);
        self
    }

    /// Build into a Chart.
    pub fn build(self) -> Chart {
        Box::new(self) as Chart
    }

    /// Build with validation.
    pub fn try_build(self) -> Result<Chart, crate::error::ChartError> {
        if self.tasks.is_empty() {
            return Err(crate::error::ChartError::EmptyData);
        }
        Ok(Box::new(self) as Chart)
    }
}

impl ChartSpec for GanttChart {
    fn render(&self, w: u32, h: u32) -> crate::layout::RenderedChart {
        crate::layout::gantt::render_gantt(self, w, h)
    }

    fn render_with_viewport(
        &self,
        w: u32,
        h: u32,
        vp: Option<(f64, f64, f64, f64)>,
    ) -> crate::layout::RenderedChart {
        if let Some((x0, x1, _y0, _y1)) = vp {
            let mut c = self.clone();
            c.config.axes.x_range = Some((x0, x1));
            c.render(w, h)
        } else {
            self.render(w, h)
        }
    }

    fn config(&self) -> Option<&ChartConfig> {
        Some(&self.config)
    }

    fn config_mut(&mut self) -> Option<&mut ChartConfig> {
        Some(&mut self.config)
    }

    fn data_extent(&self) -> Option<(f64, f64, f64, f64)> {
        if self.tasks.is_empty() {
            return None;
        }
        let x_min = self
            .tasks
            .iter()
            .map(|t| t.start)
            .fold(f64::INFINITY, f64::min);
        let x_max = self
            .tasks
            .iter()
            .map(|t| t.end)
            .fold(f64::NEG_INFINITY, f64::max);
        let n = self.tasks.len();
        Some((x_min, x_max, 0.0, (n.saturating_sub(1)) as f64))
    }

    fn clone_boxed(&self) -> Box<dyn ChartSpec> {
        Box::new(self.clone())
    }
}
