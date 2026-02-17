// SPDX-License-Identifier: MIT OR Apache-2.0
//! Live training visualization callback.
//!
//! Renders a live-updating loss curve in the terminal during `fit()`,
//! so you can watch training converge in real time.
//!
//! Gated on the `live-plot` feature.
//!
//! # Example
//!
//! ```ignore
//! use scry_learn::prelude::*;
//! use scry_learn::neural::LivePlotCallback;
//!
//! let mut clf = MLPClassifier::new()
//!     .hidden_layers(&[64, 32])
//!     .max_iter(100)
//!     .callback(Box::new(LivePlotCallback::new()));
//! clf.fit(&train)?;
//! ```

use scry_chart::streaming::StreamingChart;
use scry_chart::theme::Theme;

use crate::neural::callback::{CallbackAction, EpochMetrics, TrainingCallback};

/// Configuration for live training plot display.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct LivePlotConfig {
    /// Chart width in pixels.
    pub width: u32,
    /// Chart height in pixels.
    pub height: u32,
    /// Visual theme.
    pub theme: Theme,
    /// Whether to show validation loss (if available).
    pub show_val_loss: bool,
    /// Minimum milliseconds between renders (rate limiting).
    pub render_interval_ms: u64,
    /// Window size (max epochs to display).
    pub window_size: usize,
}

impl Default for LivePlotConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 300,
            theme: Theme::default(),
            show_val_loss: true,
            render_interval_ms: 100,
            window_size: 500,
        }
    }
}

impl LivePlotConfig {
    /// Set chart width in pixels. Default: 800.
    #[must_use]
    pub fn width(mut self, w: u32) -> Self {
        self.width = w;
        self
    }

    /// Set chart height in pixels. Default: 300.
    #[must_use]
    pub fn height(mut self, h: u32) -> Self {
        self.height = h;
        self
    }

    /// Set visual theme.
    #[must_use]
    pub fn theme(mut self, t: Theme) -> Self {
        self.theme = t;
        self
    }

    /// Set minimum render interval in milliseconds. Default: 100.
    #[must_use]
    pub fn render_interval_ms(mut self, ms: u64) -> Self {
        self.render_interval_ms = ms;
        self
    }

    /// Set the epoch window size. Default: 500.
    #[must_use]
    pub fn window_size(mut self, n: usize) -> Self {
        self.window_size = n;
        self
    }
}

/// A training callback that renders a live loss curve in the terminal.
///
/// Pushes `train_loss` (and optionally `val_loss`) to a [`StreamingChart`]
/// each epoch and renders it inline using the terminal graphics protocol.
///
/// Rendering failures are silently ignored, making this CI-safe.
pub struct LivePlotCallback {
    chart: StreamingChart,
    config: LivePlotConfig,
    frame_number: u64,
    last_render: std::time::Instant,
}

impl LivePlotCallback {
    /// Create a new live plot callback with default settings.
    pub fn new() -> Self {
        Self::with_config(LivePlotConfig::default())
    }

    /// Create a new live plot callback with custom configuration.
    pub fn with_config(config: LivePlotConfig) -> Self {
        let chart = StreamingChart::new()
            .window_size(config.window_size)
            .n_series(2)
            .labels(vec!["train_loss", "val_loss"])
            .title("Training Loss")
            .theme(config.theme.clone());

        Self {
            chart,
            config,
            frame_number: 0,
            last_render: std::time::Instant::now(),
        }
    }
}

impl Default for LivePlotCallback {
    fn default() -> Self {
        Self::new()
    }
}

impl LivePlotCallback {
    /// Render a final frame. Called automatically when training ends.
    fn finish(&mut self) {
        if self.frame_number > 0 {
            let _ =
                self.chart
                    .render_frame(self.config.width, self.config.height, self.frame_number);
            self.frame_number += 1;
        }
    }
}

impl TrainingCallback for LivePlotCallback {
    fn on_epoch_end(&mut self, metrics: &EpochMetrics) -> CallbackAction {
        let epoch = metrics.epoch as f64;

        // Always push data.
        self.chart.push_series(0, epoch, metrics.train_loss);

        if self.config.show_val_loss {
            if let Some(val_loss) = metrics.val_loss {
                self.chart.push_series(1, epoch, val_loss);
            }
        }

        // Rate-limited rendering (always render first frame).
        let elapsed = self.last_render.elapsed();
        if self.frame_number == 0 || elapsed.as_millis() as u64 >= self.config.render_interval_ms {
            // Silently ignore render failures (non-terminal, CI, etc.).
            let _ =
                self.chart
                    .render_frame(self.config.width, self.config.height, self.frame_number);
            self.frame_number += 1;
            self.last_render = std::time::Instant::now();
        }

        CallbackAction::Continue
    }

    fn on_training_end(&mut self) {
        self.finish();
    }
}
