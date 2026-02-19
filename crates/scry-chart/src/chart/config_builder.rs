// SPDX-License-Identifier: MIT OR Apache-2.0
//! Macro for generating common chart builder methods.
//!
//! Eliminates boilerplate across chart builders by generating the
//! shared methods that delegate to `self.config: ChartConfig`.

/// Generate the core builder methods shared by **every** chart type.
///
/// Requires `self.config: ChartConfig` on the struct.
/// Generates: `title`, `theme`.
macro_rules! chart_config_core {
    () => {
        /// Set the chart title.
        pub fn title(mut self, title: impl Into<String>) -> Self {
            self.config.titles.title = Some(title.into());
            self
        }

        /// Set the visual theme.
        pub fn theme(mut self, theme: $crate::theme::Theme) -> Self {
            self.config.theme = theme;
            self
        }

        /// Set the export DPI (dots per inch).
        ///
        /// Default is 144. Higher values produce larger images:
        /// 288 DPI = 2× resolution (Retina), 72 DPI = 0.5× resolution.
        pub fn dpi(mut self, dpi: u32) -> Self {
            self.config.export.dpi = dpi.max(36); // floor at 36 to avoid degenerate sizes
            self
        }
    };
}

/// Generate axis label methods (`x_label`, `y_label`).
macro_rules! chart_config_axis_labels {
    () => {
        /// Set the x-axis label.
        pub fn x_label(mut self, label: impl Into<String>) -> Self {
            self.config.titles.x_label = Some(label.into());
            self
        }

        /// Set the y-axis label.
        pub fn y_label(mut self, label: impl Into<String>) -> Self {
            self.config.titles.y_label = Some(label.into());
            self
        }
    };
}

/// Generate axis range override methods (`x_range`, `y_range`).
macro_rules! chart_config_ranges {
    (x) => {
        /// Override the x-axis range.
        pub fn x_range(mut self, min: f64, max: f64) -> Self {
            self.config.axes.x_range = Some((min, max));
            self
        }
    };
    (y) => {
        /// Override the y-axis range.
        pub fn y_range(mut self, min: f64, max: f64) -> Self {
            self.config.axes.y_range = Some((min, max));
            self
        }
    };
    (xy) => {
        chart_config_ranges!(x);
        chart_config_ranges!(y);
    };
}

/// Generate horizontal reference line methods.
macro_rules! chart_config_h_lines {
    () => {
        /// Add a horizontal reference line.
        pub fn h_line(mut self, value: f64) -> Self {
            self.config
                .overlays
                .h_lines
                .push($crate::chart::ReferenceLine::new(value));
            self
        }

        /// Add a horizontal reference line with color.
        pub fn h_line_styled(mut self, value: f64, color: scry_engine::style::Color) -> Self {
            self.config
                .overlays
                .h_lines
                .push($crate::chart::ReferenceLine::new(value).color(color));
            self
        }
    };
}

/// Generate vertical reference line methods.
macro_rules! chart_config_v_lines {
    () => {
        /// Add a vertical reference line.
        pub fn v_line(mut self, value: f64) -> Self {
            self.config
                .overlays
                .v_lines
                .push($crate::chart::ReferenceLine::new(value));
            self
        }

        /// Add a vertical reference line with color.
        pub fn v_line_styled(mut self, value: f64, color: scry_engine::style::Color) -> Self {
            self.config
                .overlays
                .v_lines
                .push($crate::chart::ReferenceLine::new(value).color(color));
            self
        }
    };
}

/// Generate grid control methods.
macro_rules! chart_config_grid {
    () => {
        /// Show only Y-axis gridlines (horizontal lines); hide X-axis grid.
        pub fn y_grid_only(mut self) -> Self {
            self.config.theme.grid.show_x = Some(false);
            self.config.theme.grid.show_y = Some(true);
            self
        }

        /// Show only X-axis gridlines (vertical lines); hide Y-axis grid.
        pub fn x_grid_only(mut self) -> Self {
            self.config.theme.grid.show_x = Some(true);
            self.config.theme.grid.show_y = Some(false);
            self
        }

        /// Hide all gridlines.
        pub fn no_grid(mut self) -> Self {
            self.config.theme.grid.show = false;
            self
        }

        /// Explicitly control X-axis gridlines (vertical lines).
        pub fn show_x_grid(mut self, show: bool) -> Self {
            self.config.theme.grid.show_x = Some(show);
            self
        }

        /// Explicitly control Y-axis gridlines (horizontal lines).
        pub fn show_y_grid(mut self, show: bool) -> Self {
            self.config.theme.grid.show_y = Some(show);
            self
        }
    };
}

/// Generate legend toggle method.
macro_rules! chart_config_legend {
    () => {
        /// Hide the legend.
        pub fn no_legend(mut self) -> Self {
            self.config.show_legend = false;
            self
        }

        /// Set the legend title.
        pub fn legend_title(mut self, title: impl Into<String>) -> Self {
            self.config.legend.title = Some(title.into());
            self
        }

        /// Use a horizontal legend layout.
        pub fn legend_horizontal(mut self) -> Self {
            self.config.legend.orientation = $crate::legend::LegendOrientation::Horizontal;
            self
        }

        /// Set the legend position.
        pub fn legend_position(mut self, pos: $crate::legend::LegendPosition) -> Self {
            self.config.legend.position = pos;
            self
        }

        /// Modify legend config via a closure.
        pub fn legend(mut self, f: impl FnOnce(&mut $crate::legend::LegendConfig)) -> Self {
            f(&mut self.config.legend);
            self
        }

        /// Set the number of columns for vertical legend layout.
        ///
        /// Only affects vertical orientation. Use with many entries to
        /// keep the legend compact (e.g., `legend_columns(2)` for side-by-side).
        pub fn legend_columns(mut self, columns: usize) -> Self {
            self.config.legend.columns = columns.max(1);
            self
        }

        /// Place the legend outside the plot area, to the right.
        pub fn legend_outside_right(mut self) -> Self {
            self.config.legend.position = $crate::legend::LegendPosition::OutsideRight;
            self
        }

        /// Place the legend outside the plot area, below.
        pub fn legend_outside_bottom(mut self) -> Self {
            self.config.legend.position = $crate::legend::LegendPosition::OutsideBottom;
            self
        }
    };
}

/// Generate annotation and trend line methods.
macro_rules! chart_config_annotations {
    () => {
        /// Add an annotation at the given data coordinates.
        pub fn annotate(mut self, x: f64, y: f64, text: impl Into<String>) -> Self {
            self.config
                .overlays
                .annotations
                .push($crate::annotation::Annotation::new(x, y, text));
            self
        }

        /// Show a linear regression trend line.
        pub fn trend_line(mut self) -> Self {
            self.config.overlays.show_trend = true;
            self
        }
    };
}

/// Generate X-axis tick label rotation methods.
macro_rules! chart_config_tick_rotation {
    () => {
        /// Rotate X-axis tick labels to 45° (diagonal) to reduce overlap.
        pub fn x_ticks_diagonal(mut self) -> Self {
            self.config.ticks.x_tick_rotation = $crate::axis::LabelRotation::Diagonal;
            self
        }

        /// Rotate X-axis tick labels to 90° (vertical) for maximum density.
        pub fn x_ticks_vertical(mut self) -> Self {
            self.config.ticks.x_tick_rotation = $crate::axis::LabelRotation::Vertical;
            self
        }

        /// Set X-axis tick label rotation explicitly.
        pub fn x_tick_rotation(mut self, rotation: $crate::axis::LabelRotation) -> Self {
            self.config.ticks.x_tick_rotation = rotation;
            self
        }

        /// Set X-axis tick label rotation to a custom angle in degrees (0–90).
        pub fn x_tick_angle(mut self, degrees: f32) -> Self {
            self.config.ticks.x_tick_rotation = $crate::axis::LabelRotation::Angle(degrees);
            self
        }
    };
}

/// Generate tick formatter methods (`x_formatter`, `y_formatter`).
macro_rules! chart_config_formatters {
    () => {
        /// Set a custom tick formatter for the X axis.
        pub fn x_formatter(mut self, fmt: impl $crate::formatter::TickFormatter + 'static) -> Self {
            self.config.ticks.x_tick_formatter = Some(std::sync::Arc::new(fmt));
            self
        }

        /// Set a custom tick formatter for the Y axis.
        pub fn y_formatter(mut self, fmt: impl $crate::formatter::TickFormatter + 'static) -> Self {
            self.config.ticks.y_tick_formatter = Some(std::sync::Arc::new(fmt));
            self
        }
    };
}

/// Generate tick step methods (`x_tick_step`, `y_tick_step`).
macro_rules! chart_config_tick_steps {
    () => {
        /// Set a fixed tick step for the X axis.
        pub fn x_tick_step(mut self, step: f64) -> Self {
            self.config.ticks.x_tick_step = Some(step);
            self
        }

        /// Set a fixed tick step for the Y axis.
        pub fn y_tick_step(mut self, step: f64) -> Self {
            self.config.ticks.y_tick_step = Some(step);
            self
        }
    };
}

/// Generate locale configuration methods.
macro_rules! chart_config_locale {
    () => {
        /// Set locale for number formatting (decimal/thousands separators).
        pub fn locale(mut self, locale: $crate::formatter::LocaleConfig) -> Self {
            self.config.ticks.locale = Some(locale);
            self
        }

        /// Set European locale (comma decimal, period grouping).
        pub fn european_locale(mut self) -> Self {
            self.config.ticks.locale = Some($crate::formatter::LocaleConfig::european());
            self
        }

        /// Set Swiss locale (period decimal, apostrophe grouping).
        pub fn swiss_locale(mut self) -> Self {
            self.config.ticks.locale = Some($crate::formatter::LocaleConfig::swiss());
            self
        }
    };
}

/// Generate subtitle and footer builder methods.
macro_rules! chart_config_subtitle_footer {
    () => {
        /// Set a subtitle, rendered below the title in smaller text.
        pub fn subtitle(mut self, subtitle: impl Into<String>) -> Self {
            self.config.titles.subtitle = Some(subtitle.into());
            self
        }

        /// Set a footer, rendered at the bottom edge of the chart.
        pub fn footer(mut self, footer: impl Into<String>) -> Self {
            self.config.titles.footer = Some(footer.into());
            self
        }
    };
}

/// Generate margin / padding builder methods.
macro_rules! chart_config_margin {
    () => {
        /// Set custom margins (top, right, bottom, left) in pixels.
        pub fn margin(mut self, top: f32, right: f32, bottom: f32, left: f32) -> Self {
            self.config.margin = Some($crate::margin::Margin::new(top, right, bottom, left));
            self
        }

        /// Set uniform margins on all sides (pixels).
        pub fn margin_all(mut self, px: f32) -> Self {
            self.config.margin = Some($crate::margin::Margin::uniform(px));
            self
        }
    };
}

/// Generate axis inversion builder methods.
macro_rules! chart_config_invert {
    () => {
        /// Reverse the X axis (high values on the left).
        pub fn x_inverted(mut self) -> Self {
            self.config.axes.x_inverted = true;
            self
        }

        /// Reverse the Y axis (high values at the bottom).
        pub fn y_inverted(mut self) -> Self {
            self.config.axes.y_inverted = true;
            self
        }
    };
}

/// Generate secondary (dual) Y-axis builder methods.
macro_rules! chart_config_secondary_y {
    () => {
        /// Set the label for the secondary (right) Y-axis.
        pub fn secondary_y_label(mut self, label: impl Into<String>) -> Self {
            self.config.secondary.label = Some(label.into());
            self
        }

        /// Override the secondary Y-axis range.
        pub fn secondary_y_range(mut self, min: f64, max: f64) -> Self {
            self.config.secondary.range = Some((min, max));
            self
        }

        /// Set a custom tick formatter for the secondary Y-axis.
        pub fn secondary_y_formatter(
            mut self,
            f: impl $crate::formatter::TickFormatter + 'static,
        ) -> Self {
            self.config.secondary.formatter = Some(
                std::sync::Arc::new(f) as std::sync::Arc<dyn $crate::formatter::TickFormatter>
            );
            self
        }

        /// Assign one or more series indices to the secondary (right) Y-axis.
        pub fn secondary_axis(mut self, indices: &[usize]) -> Self {
            self.config.secondary.series_indices = indices.to_vec();
            self
        }
    };
}

/// Generate semantic zoom formatter methods for X and Y axes.
macro_rules! chart_config_semantic_zoom {
    () => {
        /// Enable semantic zoom formatting for the X axis.
        pub fn semantic_zoom_x(mut self) -> Self {
            self.config.ticks.x_tick_formatter = Some(std::sync::Arc::new(
                $crate::formatter::SemanticZoomFormatter::default(),
            ));
            self
        }

        /// Enable semantic zoom formatting for the Y axis.
        pub fn semantic_zoom_y(mut self) -> Self {
            self.config.ticks.y_tick_formatter = Some(std::sync::Arc::new(
                $crate::formatter::SemanticZoomFormatter::default(),
            ));
            self
        }
    };
}

// Re-export macros for use in sibling modules.
pub(crate) use chart_config_annotations;
pub(crate) use chart_config_axis_labels;
pub(crate) use chart_config_core;
pub(crate) use chart_config_formatters;
pub(crate) use chart_config_grid;
pub(crate) use chart_config_h_lines;
pub(crate) use chart_config_invert;
pub(crate) use chart_config_legend;
pub(crate) use chart_config_locale;
pub(crate) use chart_config_margin;
pub(crate) use chart_config_ranges;
pub(crate) use chart_config_secondary_y;
pub(crate) use chart_config_semantic_zoom;
pub(crate) use chart_config_subtitle_footer;
pub(crate) use chart_config_tick_rotation;
pub(crate) use chart_config_tick_steps;
pub(crate) use chart_config_v_lines;
