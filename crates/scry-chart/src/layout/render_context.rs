// SPDX-License-Identifier: MIT OR Apache-2.0
//! `RenderContext` — shared rendering infrastructure for chart layout.

use scry_engine::scene::PixelCanvas;
use scry_engine::style::Color;

use crate::axis::{self, AxisSide, LabelRotation};
use crate::chart::ChartConfig;
use crate::scale::{LinearScale, Scale};

use super::{
    axis_config_from_theme, compute_plot_area, scaled_font_size, x_tick_label_offset,
    y_tick_label_offset, RenderedChart, TextAlign, TextOverlay,
};

/// Shared state for chart rendering, eliminating boilerplate across chart types.
pub(crate) struct RenderContext {
    /// The pixel canvas being drawn to (taken temporarily during builder calls).
    pub canvas: Option<PixelCanvas>,
    /// Accumulated text overlays.
    pub overlays: Vec<TextOverlay>,
    /// Plot area rectangle (x, y, w, h).
    pub plot: (f32, f32, f32, f32),
    /// Scales for cursor interaction (populated by chart renderers).
    pub x_scale: Option<LinearScale>,
    /// Y scale for cursor interaction.
    pub y_scale: Option<LinearScale>,
    /// Series data points for cursor nearest-point detection.
    pub series_points: Vec<Vec<(f64, f64)>>,
}

impl RenderContext {
    /// Create a new render context with background and computed plot area.
    ///
    /// `y_extent` is the (min, max) of the Y data domain. When provided,
    /// the layout pre-generates tick labels and measures the widest one
    /// to reserve exactly the right amount of horizontal space.
    pub fn new(config: &ChartConfig, w: u32, h: u32, y_extent: Option<(f64, f64)>) -> Self {
        let plot = compute_plot_area(w, h, config, y_extent);
        let canvas = PixelCanvas::new(w, h).background(config.theme.background);
        Self {
            canvas: Some(canvas),
            overlays: Vec::new(),
            plot,
            x_scale: None,
            y_scale: None,
            series_points: Vec::new(),
        }
    }

    /// Temporarily take the canvas, apply a transform, and put it back.
    ///
    /// This makes the take-and-return atomic, eliminating the panic risk
    /// of the old `take_canvas()` + manual `ctx.canvas = Some(...)` pattern.
    pub(crate) fn draw(&mut self, f: impl FnOnce(PixelCanvas) -> PixelCanvas) {
        // SAFETY: `canvas` is always `Some` between public API calls — the
        // take-and-return pattern in draw()/draw_with() is atomic w.r.t. the
        // caller, so no external code can observe the `None` state.
        let canvas = self.canvas.take().expect("canvas was already taken");
        self.canvas = Some(f(canvas));
    }

    /// Like [`draw`](Self::draw), but the closure also returns extra data.
    ///
    /// Useful for operations like `axis::draw_axis` that return both the
    /// canvas and tick label positions.
    pub(crate) fn draw_with<T>(&mut self, f: impl FnOnce(PixelCanvas) -> (PixelCanvas, T)) -> T {
        // SAFETY: same invariant as `draw()` — canvas is always Some here.
        let canvas = self.canvas.take().expect("canvas was already taken");
        let (canvas, result) = f(canvas);
        self.canvas = Some(canvas);
        result
    }

    /// Canvas width.
    pub(super) fn width(&self) -> u32 {
        self.canvas.as_ref().unwrap().width()
    }

    /// Canvas height.
    pub(super) fn height(&self) -> u32 {
        self.canvas.as_ref().unwrap().height()
    }

    /// Draw X and Y axes, collecting tick labels as text overlays.
    pub fn draw_axes(
        &mut self,
        config: &ChartConfig,
        x_scale: &LinearScale,
        y_scale: &LinearScale,
    ) {
        let plot = self.plot;
        let x_cfg = axis_config_from_theme(config, AxisSide::Bottom);
        let y_cfg = axis_config_from_theme(config, AxisSide::Left);

        let x_ticks = self.draw_with(|c| axis::draw_axis(c, plot, x_scale, &x_cfg));
        let y_ticks = self.draw_with(|c| axis::draw_axis(c, plot, y_scale, &y_cfg));

        let w = self.width();
        let h = self.height();
        let tick_fs = scaled_font_size(config.theme.tick_style.font_size, w, h);

        self.add_tick_overlays(
            &x_ticks,
            &y_ticks,
            config.theme.foreground,
            config.x_tick_rotation,
            tick_fs,
        );
    }

    /// Draw only the Y axis (for categorical X charts like bar/boxplot).
    pub fn draw_y_axis(
        &mut self,
        config: &ChartConfig,
        y_scale: &LinearScale,
    ) -> Vec<(f32, String)> {
        let plot = self.plot;
        let cfg = axis_config_from_theme(config, AxisSide::Left);
        self.draw_with(|c| axis::draw_axis(c, plot, y_scale, &cfg))
    }

    /// Draw the X axis line (for categorical charts).
    pub fn draw_x_axis_line(&mut self, config: &ChartConfig) {
        let (px, py, pw, ph) = self.plot;
        let color = config.theme.axis.color;
        let width = config.theme.axis.width;
        self.draw(|c| {
            c.line(px, py + ph, px + pw, py + ph)
                .color(color)
                .width(width)
                .done()
        });
    }

    /// Draw only the X value-axis on the bottom (for horizontal bar charts).
    ///
    /// Uses the shared `draw_axis` / `axis_config_from_theme` infrastructure
    /// so tick marks, gridlines, and label offsets are consistent with other charts.
    pub fn draw_x_value_axis(&mut self, config: &ChartConfig, x_scale: &LinearScale) {
        let plot = self.plot;
        let cfg = axis_config_from_theme(config, AxisSide::Bottom);
        let x_ticks = self.draw_with(|c| axis::draw_axis(c, plot, x_scale, &cfg));

        let (_px, py, _pw, ph) = self.plot;
        let rot_deg = config.x_tick_rotation.degrees();
        let align = if rot_deg > 0.0 {
            TextAlign::Right
        } else {
            TextAlign::Center
        };
        let w = self.width();
        let h = self.height();
        let tick_fs = scaled_font_size(config.theme.tick_style.font_size, w, h);
        for (x, label) in &x_ticks {
            self.overlays.push(TextOverlay {
                x_px: *x,
                y_px: py + ph + x_tick_label_offset(h),
                text: label.clone(),
                color: config.theme.foreground,
                align,
                font_size: tick_fs,
                bold: false,
                rotation_deg: rot_deg,
            });
        }
    }

    /// Draw reference lines on the canvas.
    pub fn draw_reference_lines(
        &mut self,
        config: &ChartConfig,
        x_scale: &LinearScale,
        y_scale: &LinearScale,
    ) {
        let (px, py, pw, ph) = self.plot;
        let dash = scry_engine::style::DashPattern::new(vec![8.0, 5.0], 0.0);

        // Horizontal reference lines
        for rl in &config.h_lines {
            let y = y_scale.to_pixel(rl.value) as f32;
            if y >= py && y <= py + ph {
                let color = rl.color;
                let width = rl.width;
                let dashed = rl.dashed;
                let d = dash.clone();
                self.draw(|c| {
                    let mut b = c.line(px, y, px + pw, y).color(color).width(width);
                    if dashed {
                        b = b.dash(d);
                    }
                    b.done()
                });
            }
        }

        // Vertical reference lines
        for rl in &config.v_lines {
            let x = x_scale.to_pixel(rl.value) as f32;
            if x >= px && x <= px + pw {
                let color = rl.color;
                let width = rl.width;
                let dashed = rl.dashed;
                let d = dash.clone();
                self.draw(|c| {
                    let mut b = c.line(x, py, x, py + ph).color(color).width(width);
                    if dashed {
                        b = b.dash(d);
                    }
                    b.done()
                });
            }
        }
    }

    /// Add tick label overlays from axis rendering.
    pub(super) fn add_tick_overlays(
        &mut self,
        x_ticks: &[(f32, String)],
        y_ticks: &[(f32, String)],
        color: Color,
        x_rotation: LabelRotation,
        tick_font_size: f32,
    ) {
        let (px, py, _pw, ph) = self.plot;

        let w = self.width();
        let h = self.height();
        let x_off = x_tick_label_offset(h);
        let y_off = y_tick_label_offset(w);

        let rot_deg = x_rotation.degrees();
        // For rotated labels, right-align at the tick position so the
        // label "hangs" from the tick mark rather than centering.
        let x_align = if rot_deg > 0.0 {
            TextAlign::Right
        } else {
            TextAlign::Center
        };

        for (x, label) in x_ticks {
            self.overlays.push(TextOverlay {
                x_px: *x,
                y_px: py + ph + x_off,
                text: label.clone(),
                color,
                align: x_align,
                font_size: tick_font_size,
                bold: false,
                rotation_deg: rot_deg,
            });
        }

        for (y, label) in y_ticks {
            self.overlays.push(TextOverlay {
                x_px: px - y_off,
                y_px: *y,
                text: label.clone(),
                color,
                align: TextAlign::Right,
                font_size: tick_font_size,
                bold: false,
                rotation_deg: 0.0,
            });
        }
    }

    /// Finalize into a `RenderedChart`.
    pub fn finish(self) -> RenderedChart {
        RenderedChart {
            canvas: self.canvas.unwrap(),
            text_overlays: self.overlays,
            plot_area: Some(self.plot),
            x_scale: self.x_scale,
            y_scale: self.y_scale,
            series_points: self.series_points,
        }
    }
}
