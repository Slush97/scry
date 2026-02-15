//! Layout engine — translates chart specs into `PixelCanvas` scenes.
//!
//! This is the core rendering layer. It takes a fully-configured [`Chart`],
//! measures the components (margins, axes, legend), and emits
//! `PixelCanvas` drawing commands.

mod bar;
mod boxplot;
mod bubble;
mod candlestick;
mod funnel;
mod gauge;
mod heatmap;
mod histogram;
mod line;
mod lollipop;
mod pie;
mod radar;
mod scatter;
mod sparkline;
mod violin;
mod waterfall;

use scry_engine::scene::PixelCanvas;
use std::sync::Arc;

use crate::scale::LinearScale;
use scry_engine::style::{Color, DashPattern};

use crate::axis::{self, AxisConfig, AxisSide, LabelRotation};
use crate::chart::{Chart, ChartConfig};
use crate::formatter::{AutoFormatter, LocaleConfig, LocaleFormatter, TickFormatter};
use crate::scale::{self, Scale};

// ---------------------------------------------------------------------------
// Measurement-based layout
// ---------------------------------------------------------------------------

/// Minimum margin (pixels). Scales proportionally with canvas size.
const MIN_MARGIN: f32 = 4.0;
/// Margin as a fraction of the shorter canvas dimension.
const MARGIN_FRAC: f32 = 0.04;

/// Proportional pixel offset from X-axis spine to X tick labels (downward).
fn x_tick_label_offset(h: u32) -> f32 {
    let h = h as f32;
    (h * 0.015).max(4.0).min(14.0)
}

/// Proportional pixel offset from Y-axis spine to Y tick labels (leftward).
pub(crate) fn y_tick_label_offset(w: u32) -> f32 {
    let w = w as f32;
    (w * 0.012).max(4.0).min(14.0)
}

/// Compute proportional margin based on canvas size.
pub(crate) fn proportional_margin(w: u32, h: u32) -> f32 {
    let short = (w.min(h)) as f32;
    (short * MARGIN_FRAC).max(MIN_MARGIN).min(24.0)
}

/// Average character width at 11px font size (Inter Regular)
/// for estimating tick label pixel widths without loading fontdue.
///
/// NOTE: Prefer `char_width_for_size()` for dynamic calculations.
/// Kept for backward compatibility with `proportional_x_tick_height_with_chars`.
#[allow(dead_code)]
const AVG_CHAR_WIDTH_11PX: f32 = 6.5;

/// Average advance-width ratio for the Inter typeface.
///
/// `char_width = font_size × INTER_ADVANCE_RATIO`.
/// Calibrated from Inter Regular at multiple sizes (0.59 ≈ 6.49 / 11).
pub(crate) const INTER_ADVANCE_RATIO: f32 = 0.59;

/// Reference canvas area for scale factor = 1.0.
///
/// 400 × 300 = 120 000 — the default test canvas size.
const REFERENCE_AREA: f32 = 120_000.0;

/// Compute a canvas-proportional font scale factor.
///
/// Returns a multiplier that grows with the square root of the canvas area
/// relative to [`REFERENCE_AREA`]. The square root provides perceptually
/// linear scaling — doubling the canvas area increases font size by √2 ≈ 1.41×.
///
/// Clamped to `[0.6, 2.0]` so text never becomes unreadable on tiny canvases
/// or comically large on 4K exports.
#[must_use]
pub fn font_scale_factor(w: u32, h: u32) -> f32 {
    let area = w as f32 * h as f32;
    (area / REFERENCE_AREA).sqrt().clamp(0.6, 2.0)
}

/// Compute an effective font size scaled by canvas dimensions.
///
/// Applies [`font_scale_factor`] to a base size and clamps the result
/// to `[7.0, 48.0]` for readability.
#[must_use]
pub fn scaled_font_size(base: f32, w: u32, h: u32) -> f32 {
    (base * font_scale_factor(w, h)).clamp(7.0, 48.0)
}

/// Average character width for a given font size (Inter Regular).
///
/// Replaces the fixed [`AVG_CHAR_WIDTH_11PX`] constant with a parameterized
/// version. Returns `font_size × INTER_ADVANCE_RATIO`.
#[inline]
#[must_use]
pub fn char_width_for_size(font_size: f32) -> f32 {
    font_size * INTER_ADVANCE_RATIO
}

/// Estimate the pixel width of the widest Y-axis tick label by
/// pre-generating tick values and formatting them.
///
/// This is the key to the two-pass layout: we compute what the tick labels
/// WILL be before deciding how much space to reserve for them.
#[allow(dead_code)]
fn estimate_y_tick_width(
    y_extent: Option<(f64, f64)>,
    h: u32,
    formatter: Option<&dyn TickFormatter>,
    locale: Option<&LocaleConfig>,
) -> f32 {
    let fallback = 30.0_f32; // safe default when no extent is known

    let Some((y_lo, y_hi)) = y_extent else {
        return fallback;
    };

    // Same tick generation logic as axis::draw_axis uses
    let plot_h = h as f64 * 0.7; // rough estimate of plot pixel height
    let target_ticks = (plot_h / 40.0).clamp(3.0, 12.0) as usize;
    let ticks = scale::nice_ticks(y_lo, y_hi, target_ticks);

    // Use the provided formatter (or AutoFormatter) to measure labels,
    // matching what draw_axis will actually produce.
    let auto_fmt = AutoFormatter;
    let locale_auto_fmt;
    let fmt: &dyn TickFormatter = match (formatter, locale) {
        (Some(f), _) => f,
        (None, Some(loc)) => {
            locale_auto_fmt = LocaleFormatter::new(AutoFormatter, loc.clone());
            &locale_auto_fmt
        }
        _ => &auto_fmt,
    };
    let labels = fmt.format_batch(&ticks, (y_lo, y_hi));
    let max_chars = labels.iter().map(|l| l.len()).max().unwrap_or(3);

    // Convert character count to pixel width + padding for the tick-to-label gap
    let text_px = max_chars as f32 * AVG_CHAR_WIDTH_11PX;
    // Add the y_tick_label_offset (gap between axis spine and label right edge)
    (text_px + 8.0).max(24.0).min(100.0)
}

/// Estimate the Y-axis tick width using a specific effective font size.
///
/// Like [`estimate_y_tick_width`] but uses `char_width_for_size(font_size)`
/// instead of the fixed 11px constant.
fn estimate_y_tick_width_scaled(
    y_extent: Option<(f64, f64)>,
    h: u32,
    formatter: Option<&dyn TickFormatter>,
    locale: Option<&LocaleConfig>,
    font_size: f32,
) -> f32 {
    let fallback = 30.0_f32;

    let Some((y_lo, y_hi)) = y_extent else {
        return fallback;
    };

    let plot_h = h as f64 * 0.7;
    let target_ticks = (plot_h / 40.0).clamp(3.0, 12.0) as usize;
    let ticks = scale::nice_ticks(y_lo, y_hi, target_ticks);

    let auto_fmt = AutoFormatter;
    let locale_auto_fmt;
    let fmt: &dyn TickFormatter = match (formatter, locale) {
        (Some(f), _) => f,
        (None, Some(loc)) => {
            locale_auto_fmt = LocaleFormatter::new(AutoFormatter, loc.clone());
            &locale_auto_fmt
        }
        _ => &auto_fmt,
    };
    let labels = fmt.format_batch(&ticks, (y_lo, y_hi));
    let max_chars = labels.iter().map(|l| l.len()).max().unwrap_or(3);

    let text_px = max_chars as f32 * char_width_for_size(font_size);
    (text_px + 8.0).max(24.0).min(120.0)
}

/// Estimate the pixel width needed for the Y-axis label text.
///
/// The Y-axis label is rendered rotated 90°, so the horizontal space
/// consumed is roughly the font ascent plus a gap, NOT the text pixel
/// width (which becomes the vertical extent).
///
/// Uses [`scaled_font_size`] so the reserved width scales with canvas
/// dimensions instead of being a fixed 19 px.
fn estimate_y_label_width(label: Option<&str>, w: u32, h: u32, label_size: f32) -> f32 {
    match label {
        Some(text) if !text.is_empty() => {
            // Rotated 90°: width ≈ font ascent + breathing-room gap
            let _ = text; // label length affects vertical extent, not width
            let fs = scaled_font_size(label_size, w, h);
            // ascent ≈ fs (at Inter the ascent/em ratio ≈ 0.93, round up)
            // gap ≈ 0.7 × fs for tick-label clearance
            fs + fs * 0.7
        }
        _ => 0.0,
    }
}

/// Fallback proportional Y-axis width when no y_extent is available.
fn proportional_y_axis_width(w: u32) -> f32 {
    let w = w as f32;
    (w * 0.08).max(24.0).min(60.0)
}

/// Compute proportional title height based on canvas height.
fn proportional_title_height(h: u32) -> f32 {
    let h = h as f32;
    (h * 0.06).max(14.0).min(30.0)
}

/// Compute X-axis tick label height based on canvas height and label rotation.
///
/// For rotated labels, the vertical space needed depends on both the angle
/// and the estimated label length. Uses `max_label_chars` (default 6) for
/// a reasonable estimate when the actual tick labels aren't known yet.
fn proportional_x_tick_height(h: u32, rotation: LabelRotation) -> f32 {
    proportional_x_tick_height_with_chars(h, rotation, 6)
}

/// Inner implementation that accepts an estimated max label character count.
fn proportional_x_tick_height_with_chars(
    h: u32,
    rotation: LabelRotation,
    max_label_chars: usize,
) -> f32 {
    let base = h as f32;
    // Estimated pixel width of the longest label
    let label_px = max_label_chars as f32 * char_width_for_size(11.0);
    match rotation {
        LabelRotation::Horizontal => (base * 0.05).max(12.0).min(22.0),
        LabelRotation::Diagonal => {
            // 45° rotation: height ≈ label_width * sin(45°) ≈ 0.71 * label_width.
            let needed = label_px * 0.71 + 4.0; // +4 for tick gap
            needed.max(22.0).min(60.0)
        }
        LabelRotation::Vertical => {
            // 90° rotation: height = full label width in pixels.
            let needed = label_px + 4.0;
            needed.max(30.0).min(80.0)
        }
        LabelRotation::Angle(deg) => {
            // Height = label_width * sin(angle) + font_height * cos(angle)
            let rad = deg.clamp(0.0, 90.0).to_radians();
            let needed = label_px * rad.sin() + 14.0 * rad.cos() + 4.0;
            needed.max(12.0).min(80.0)
        }
    }
}

/// Compute X-axis label height.
fn proportional_x_label_height(h: u32) -> f32 {
    let h = h as f32;
    (h * 0.05).max(12.0).min(22.0)
}

// ---------------------------------------------------------------------------
// Rendered chart output
// ---------------------------------------------------------------------------

/// The result of laying out a chart — a `PixelCanvas` plus text overlays.
#[derive(Debug)]
pub struct RenderedChart {
    /// The pixel canvas with all vector graphics.
    pub canvas: PixelCanvas,
    /// Text labels to render via ratatui.
    pub text_overlays: Vec<TextOverlay>,
    /// Plot area rectangle for cursor calculations `(x, y, w, h)` in pixels.
    pub plot_area: Option<(f32, f32, f32, f32)>,
    /// X scale for cursor coordinate conversion.
    pub x_scale: Option<LinearScale>,
    /// Y scale for cursor coordinate conversion.
    pub y_scale: Option<LinearScale>,
    /// Series data points for nearest-point detection (one Vec per series).
    pub series_points: Vec<Vec<(f64, f64)>>,
}

/// A text label to be rendered via ratatui on top of the chart.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TextOverlay {
    /// X position in pixels (from left of canvas).
    pub x_px: f32,
    /// Y position in pixels (from top of canvas).
    pub y_px: f32,
    /// The text to display.
    pub text: String,
    /// Text color.
    pub color: Color,
    /// Horizontal alignment.
    pub align: TextAlign,
    /// Font size in pixels. Defaults to 12.0.
    ///
    /// **Limitation:** This field is only meaningful for SVG and PNG export
    /// paths. In the terminal widget path ([`crate::widget::render_text_overlays`]),
    /// all text is rendered at the terminal's fixed character-cell size —
    /// character-cell terminals cannot vary glyph size, so this value is
    /// ignored.
    pub font_size: f32,
    /// Whether the text is bold.
    pub bold: bool,
    /// Rotation in degrees (0 = horizontal, positive = counter-clockwise).
    /// Only supported in the PNG export path; the widget path approximates
    /// rotation by stacking characters vertically.
    pub rotation_deg: f32,
}

/// Text alignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TextAlign {
    /// Left-aligned (default).
    Left,
    /// Center-aligned.
    Center,
    /// Right-aligned.
    Right,
}

// ---------------------------------------------------------------------------
// RenderContext — shared rendering infrastructure
// ---------------------------------------------------------------------------

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
    fn width(&self) -> u32 {
        self.canvas.as_ref().unwrap().width()
    }

    /// Canvas height.
    fn height(&self) -> u32 {
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
        let dash = DashPattern::new(vec![8.0, 5.0], 0.0);

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

    /// Add title, subtitle, footer, x-label, y-label text overlays.
    pub fn add_common_overlays(&mut self, config: &ChartConfig) {
        let (px, py, pw, ph) = self.plot;
        let w = self.canvas.as_ref().unwrap().width();
        let h = self.canvas.as_ref().unwrap().height();
        let margin = proportional_margin(w, h);

        // Scaled font sizes from theme
        let title_fs = scaled_font_size(config.theme.title_style.font_size, w, h);
        let label_fs = scaled_font_size(config.theme.label_style.font_size, w, h);
        // Subtitle: ~67% of title size; footer: ~91% of tick size
        let subtitle_fs = scaled_font_size(config.theme.title_style.font_size * 0.67, w, h);
        let footer_fs = scaled_font_size(config.theme.tick_style.font_size * 0.91, w, h);

        // Extra user margins for positioning
        let extra_top = config.margin.as_ref().map_or(0.0, |m| m.top);

        let title_h = if config.title.is_some() {
            proportional_title_height(h)
        } else {
            0.0
        };

        if let Some(ref title) = config.title {
            self.overlays.push(TextOverlay {
                x_px: px + pw / 2.0,
                y_px: margin + extra_top + 4.0,
                text: title.clone(),
                color: config.theme.title_style.color,
                align: TextAlign::Center,
                font_size: title_fs,
                bold: true,
                rotation_deg: 0.0,
            });
        }

        // Subtitle: positioned below the title, smaller and not bold.
        if let Some(ref subtitle) = config.subtitle {
            let sub_y = margin + extra_top + title_h + 2.0;
            self.overlays.push(TextOverlay {
                x_px: px + pw / 2.0,
                y_px: sub_y,
                text: subtitle.clone(),
                color: config.theme.label_style.color,
                align: TextAlign::Center,
                font_size: subtitle_fs,
                bold: false,
                rotation_deg: 0.0,
            });
        }

        if let Some(ref label) = config.x_label {
            let x_tick_h = proportional_x_tick_height(h, config.x_tick_rotation);
            let x_label_h = proportional_x_label_height(h);
            self.overlays.push(TextOverlay {
                x_px: px + pw / 2.0,
                y_px: py + ph + x_tick_h + x_label_h * 0.7,
                text: label.clone(),
                color: config.theme.label_style.color,
                align: TextAlign::Center,
                font_size: label_fs,
                bold: false,
                rotation_deg: 0.0,
            });
        }

        if let Some(ref label) = config.y_label {
            // Y-axis label is rotated 90° so it reads vertically.
            let y_label_w = estimate_y_label_width(Some(label), w, h, config.theme.label_style.font_size);
            self.overlays.push(TextOverlay {
                x_px: margin + y_label_w / 2.0,
                y_px: py + ph / 2.0,
                text: label.clone(),
                color: config.theme.label_style.color,
                align: TextAlign::Center,
                font_size: label_fs,
                bold: false,
                rotation_deg: 90.0,
            });
        }

        // Secondary Y-axis label (right side, rotated -90°).
        if let Some(ref label) = config.secondary_y_label {
            let sec_label_w = estimate_y_label_width(Some(label), w, h, config.theme.label_style.font_size);
            self.overlays.push(TextOverlay {
                x_px: (w as f32) - margin - sec_label_w / 2.0,
                y_px: py + ph / 2.0,
                text: label.clone(),
                color: config.theme.label_style.color,
                align: TextAlign::Center,
                font_size: label_fs,
                bold: false,
                rotation_deg: -90.0,
            });
        }

        // Footer: small text at bottom center.
        if let Some(ref footer) = config.footer {
            let extra_bottom = config.margin.as_ref().map_or(0.0, |m| m.bottom);
            self.overlays.push(TextOverlay {
                x_px: px + pw / 2.0,
                y_px: (h as f32) - margin - extra_bottom + 2.0,
                text: footer.clone(),
                color: config.theme.label_style.color,
                align: TextAlign::Center,
                font_size: footer_fs,
                bold: false,
                rotation_deg: 0.0,
            });
        }
    }

    /// Add tick label overlays from axis rendering.
    fn add_tick_overlays(
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

    /// Add y-axis tick overlays (for categorical charts that do axes manually).
    pub fn add_y_tick_overlays(&mut self, y_ticks: &[(f32, String)], color: Color) {
        let (px, _py, _pw, _ph) = self.plot;
        let w = self.width();
        let h = self.height();
        let y_off = y_tick_label_offset(w);
        let tick_fs = scaled_font_size(11.0, w, h);
        for (pos, label) in y_ticks {
            self.overlays.push(TextOverlay {
                x_px: px - y_off,
                y_px: *pos,
                text: label.clone(),
                color,
                align: TextAlign::Right,
                font_size: tick_fs,
                bold: false,
                rotation_deg: 0.0,
            });
        }
    }

    /// Draw category labels along the X axis for bar/boxplot charts.
    ///
    /// Draws centered labels at each category position below the plot area,
    /// using the same offset as `add_tick_overlays` for consistency.
    pub fn draw_categorical_x_labels(
        &mut self,
        config: &ChartConfig,
        cat_scale: &crate::scale::CategoricalScale,
        labels: &[String],
    ) {
        let (_px, py, _pw, ph) = self.plot;
        let theme = &config.theme;
        let w = self.width();
        let h = self.height();
        let x_off = x_tick_label_offset(h);
        let rot_deg = config.x_tick_rotation.degrees();
        let align = if rot_deg > 0.0 {
            TextAlign::Right
        } else {
            TextAlign::Center
        };
        let tick_fs = scaled_font_size(theme.tick_style.font_size, w, h);

        for (ci, label) in labels.iter().enumerate() {
            self.overlays.push(TextOverlay {
                x_px: cat_scale.center(ci) as f32,
                y_px: py + ph + x_off,
                text: label.clone(),
                color: theme.text_color(),
                align,
                font_size: tick_fs,
                bold: false,
                rotation_deg: rot_deg,
            });
        }
    }

    /// Draw annotations at data coordinates.
    pub fn draw_annotations(
        &mut self,
        config: &ChartConfig,
        x_scale: &LinearScale,
        y_scale: &LinearScale,
    ) {
        let w = self.width();
        let h = self.height();
        let ann_fs = scaled_font_size(config.theme.tick_style.font_size, w, h);
        for ann in &config.annotations {
            let px = x_scale.to_pixel(ann.x) as f32;
            let py = y_scale.to_pixel(ann.y) as f32;
            let (dx, dy) = ann.style.offset;
            let text_x = px + dx;
            let text_y = py + dy;

            // Draw arrow from text to data point
            if ann.arrow {
                let arrow_color = ann.style.text_color;
                self.draw(|c| {
                    c.line(text_x, text_y + 6.0, px, py)
                        .color(arrow_color)
                        .width(1.0)
                        .done()
                });
            }

            // Draw background rect if configured
            if let Some(bg) = ann.style.background {
                let text_w = ann.text.len() as f32 * char_width_for_size(ann_fs) + 8.0;
                self.draw(|c| {
                    c.rect(text_x - 2.0, text_y - 2.0, text_w, 16.0)
                        .fill(bg)
                        .corner_radius(3.0)
                        .done()
                });
            }

            self.overlays.push(TextOverlay {
                x_px: text_x,
                y_px: text_y,
                text: ann.text.clone(),
                color: ann.style.text_color,
                align: TextAlign::Left,
                font_size: ann_fs,
                bold: false,
                rotation_deg: 0.0,
            });
        }
    }

    /// Draw a linear regression trend line.
    pub fn draw_trend_line(
        &mut self,
        x_vals: &[f64],
        y_vals: &[f64],
        x_scale: &LinearScale,
        y_scale: &LinearScale,
        color: Color,
    ) {
        let n = x_vals.len().min(y_vals.len());
        if n < 2 {
            return;
        }

        // Least squares linear regression
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        for i in 0..n {
            let x = x_vals[i];
            let y = y_vals[i];
            if !x.is_finite() || !y.is_finite() {
                continue;
            }
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }

        let nf = n as f64;
        let denom = nf * sum_x2 - sum_x * sum_x;
        if denom.abs() < f64::EPSILON {
            return;
        }

        let slope = (nf * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / nf;

        // Draw line from x_min to x_max
        let (x_lo, x_hi) = x_scale.domain();
        let y_lo = slope * x_lo + intercept;
        let y_hi = slope * x_hi + intercept;

        let px1 = x_scale.to_pixel(x_lo) as f32;
        let py1 = y_scale.to_pixel(y_lo) as f32;
        let px2 = x_scale.to_pixel(x_hi) as f32;
        let py2 = y_scale.to_pixel(y_hi) as f32;

        let trend_color = color.with_alpha(0.6);
        self.draw(|c| {
            c.line(px1, py1, px2, py2)
                .color(trend_color)
                .width(2.0)
                .dash(DashPattern::new(vec![12.0, 6.0], 0.0))
                .done()
        });
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

// ---------------------------------------------------------------------------
// Main layout function
// ---------------------------------------------------------------------------

/// Lay out and render a chart into a `PixelCanvas` of the given pixel dimensions.
#[must_use]
pub fn render_chart(chart: &Chart, width: u32, height: u32) -> RenderedChart {
    render_chart_with_viewport(chart, width, height, None)
}

/// Render a chart with an optional viewport override (for zoom/pan).
///
/// When `viewport` is `Some((x_min, x_max, y_min, y_max))`, those ranges
/// are used instead of the chart's own `x_range`/`y_range` config — this
/// avoids the need to clone the entire `Chart` every frame just to inject
/// zoom coordinates.
#[must_use]
pub fn render_chart_with_viewport(
    chart: &Chart,
    width: u32,
    height: u32,
    viewport: Option<(f64, f64, f64, f64)>,
) -> RenderedChart {
    // If a viewport is provided, we need to temporarily inject the ranges
    // into the chart's config. Since we only have a &Chart, we clone the
    // config once (cheap — no data) rather than the entire Chart (expensive).
    let chart_ref;
    let mut owned_chart;
    let effective_chart: &Chart = if let Some((x0, x1, y0, y1)) = viewport {
        owned_chart = chart.clone();
        let cfg = owned_chart.config_mut();
        cfg.x_range = Some((x0, x1));
        cfg.y_range = Some((y0, y1));
        chart_ref = &owned_chart;
        chart_ref
    } else {
        chart
    };

    match effective_chart {
        Chart::Scatter(sc) => scatter::render_scatter(sc, width, height),
        Chart::Line(lc) => line::render_line(lc, width, height),
        Chart::Bar(bc) => bar::render_bar(bc, width, height),
        Chart::Histogram(hc) => histogram::render_histogram(hc, width, height),
        Chart::BoxPlot(bp) => boxplot::render_boxplot(bp, width, height),
        Chart::Heatmap(hm) => heatmap::render_heatmap(hm, width, height),
        Chart::Pie(pc) => pie::render_pie(pc, width, height),
        Chart::Candlestick(cc) => candlestick::render_candlestick(cc, width, height),
        Chart::Radar(rc) => radar::render_radar(rc, width, height),
        Chart::Bubble(bc) => bubble::render_bubble(bc, width, height),
        Chart::Violin(vp) => violin::render_violin(vp, width, height),
        Chart::Sparkline(sp) => sparkline::render_sparkline(sp, width, height),
        Chart::Waterfall(wc) => waterfall::render_waterfall(wc, width, height),
        Chart::Funnel(fc) => funnel::render_funnel(fc, width, height),
        Chart::Gauge(gc) => gauge::render_gauge(gc, width, height),
        Chart::Lollipop(lc) => lollipop::render_lollipop(lc, width, height),
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Calculate the plot area rectangle given canvas dimensions.
///
/// When `y_extent` is provided, the Y-axis width is measured from
/// actual tick labels instead of guessed from a fixed percentage.
///
/// Handles Session 1 features:
/// - Custom margins (additive on top of auto margins)
/// - Subtitle height (below title)
/// - Footer height (at bottom)
/// - Secondary Y-axis gutter (right side, for dual-axis charts)
pub(crate) fn compute_plot_area(
    w: u32,
    h: u32,
    config: &ChartConfig,
    y_extent: Option<(f64, f64)>,
) -> (f32, f32, f32, f32) {
    let margin = proportional_margin(w, h);

    // Apply user-specified extra margins
    let user_margin = config.margin.as_ref();
    let extra_top = user_margin.map_or(0.0, |m| m.top);
    let extra_right = user_margin.map_or(0.0, |m| m.right);
    let extra_bottom = user_margin.map_or(0.0, |m| m.bottom);
    let extra_left = user_margin.map_or(0.0, |m| m.left);

    // Two-pass Y-axis width: measure tick labels when possible,
    // fall back to proportional guess otherwise.
    let tick_fs = scaled_font_size(config.theme.tick_style.font_size, w, h);
    let y_axis_w = if y_extent.is_some() {
        estimate_y_tick_width_scaled(
            y_extent,
            h,
            config.y_tick_formatter.as_deref(),
            config.locale.as_ref(),
            tick_fs,
        )
    } else {
        proportional_y_axis_width(w)
    };

    let title_h = if config.title.is_some() {
        proportional_title_height(h)
    } else {
        0.0
    };

    // Subtitle: smaller text below title
    let subtitle_h = if config.subtitle.is_some() {
        (proportional_title_height(h) * 0.65).max(10.0).min(20.0)
    } else {
        0.0
    };

    // Footer: small text at the very bottom
    let footer_h = if config.footer.is_some() {
        (proportional_title_height(h) * 0.55).max(10.0).min(18.0)
    } else {
        0.0
    };

    let y_label_w = estimate_y_label_width(config.y_label.as_deref(), w, h, config.theme.label_style.font_size);
    let x_tick_h = proportional_x_tick_height(h, config.x_tick_rotation);
    let x_label_h = if config.x_label.is_some() {
        proportional_x_label_height(h)
    } else {
        0.0
    };

    // Secondary Y-axis gutter (right side) — allocated when any secondary
    // Y-axis config is present.
    let has_secondary = config.secondary_y_label.is_some()
        || config.secondary_y_range.is_some()
        || config.secondary_y_formatter.is_some();
    let secondary_y_w = if has_secondary {
        let sec_label_w = estimate_y_label_width(config.secondary_y_label.as_deref(), w, h, config.theme.label_style.font_size);
        let sec_ticks_w = proportional_y_axis_width(w);
        sec_label_w + sec_ticks_w
    } else {
        0.0
    };

    let x = margin + extra_left + y_label_w + y_axis_w;
    let y = margin + extra_top + title_h + subtitle_h;
    let pw = w as f32 - x - margin - extra_right - secondary_y_w;
    let ph = h as f32 - y - x_tick_h - x_label_h - margin - extra_bottom - footer_h;
    (x, y, pw.max(1.0), ph.max(1.0))
}

/// Build an axis config from a theme.
///
/// Resolves per-axis grid visibility: X-axis (Bottom/Top) checks
/// `grid.show_x` before falling back to `grid.show`, and Y-axis
/// (Left/Right) checks `grid.show_y`.
pub(crate) fn axis_config_from_theme(config: &ChartConfig, side: AxisSide) -> AxisConfig {
    let theme = &config.theme;
    let is_x_axis = matches!(side, AxisSide::Bottom | AxisSide::Top);
    let show_grid = if is_x_axis {
        theme.grid.show_x.unwrap_or(theme.grid.show)
    } else {
        theme.grid.show_y.unwrap_or(theme.grid.show)
    };
    // Only apply label rotation to X axes (Bottom/Top)
    let rotation = if is_x_axis {
        config.x_tick_rotation
    } else {
        LabelRotation::Horizontal
    };
    // Resolve formatter and tick_step from ChartConfig based on axis orientation.
    // If a locale is set and no custom formatter is provided, wrap the default
    // AutoFormatter with locale-aware post-processing.
    let tick_formatter = if is_x_axis {
        match (&config.x_tick_formatter, &config.locale) {
            (Some(f), _) => Some(f.clone()),
            (None, Some(loc)) => Some(Arc::new(LocaleFormatter::new(AutoFormatter, loc.clone()))
                as Arc<dyn TickFormatter>),
            _ => None,
        }
    } else {
        match (&config.y_tick_formatter, &config.locale) {
            (Some(f), _) => Some(f.clone()),
            (None, Some(loc)) => Some(Arc::new(LocaleFormatter::new(AutoFormatter, loc.clone()))
                as Arc<dyn TickFormatter>),
            _ => None,
        }
    };
    let tick_step = if is_x_axis {
        config.x_tick_step
    } else {
        config.y_tick_step
    };
    AxisConfig {
        side,
        axis_color: theme.axis.color,
        grid_color: theme.grid.color,
        axis_width: theme.axis.width,
        grid_width: theme.grid.width,
        show_grid,
        grid_dash: theme.grid.dash.clone(),
        tick_label_rotation: rotation,
        tick_formatter,
        tick_step,
        tick_length: theme.axis.tick_length,
        tick_color: theme.axis.tick_color,
        minor_ticks: theme.axis.minor_ticks,
        minor_subdivisions: if theme.axis.minor_ticks { 4 } else { 0 },
        ..Default::default()
    }
}

/// Build an axis config for the secondary (right) Y axis.
///
/// Uses `AxisSide::Right`, disables grid (the primary Y already draws it),
/// and picks up `secondary_y_formatter` if set.
pub(crate) fn axis_config_from_theme_secondary(config: &ChartConfig) -> AxisConfig {
    let theme = &config.theme;
    let tick_formatter = match (&config.secondary_y_formatter, &config.locale) {
        (Some(f), _) => Some(f.clone()),
        (None, Some(loc)) => Some(Arc::new(LocaleFormatter::new(AutoFormatter, loc.clone()))
            as Arc<dyn TickFormatter>),
        _ => None,
    };
    AxisConfig {
        side: AxisSide::Right,
        axis_color: theme.axis.color,
        grid_color: theme.grid.color,
        axis_width: theme.axis.width,
        grid_width: 0.0, // No grid — primary Y axis already draws it
        show_grid: false,
        grid_dash: theme.grid.dash.clone(),
        tick_label_rotation: LabelRotation::Horizontal,
        tick_formatter,
        tick_step: None,
        tick_length: theme.axis.tick_length,
        tick_color: theme.axis.tick_color,
        minor_ticks: theme.axis.minor_ticks,
        minor_subdivisions: if theme.axis.minor_ticks { 4 } else { 0 },
        ..Default::default()
    }
}

/// Resolve the effective x extent: merge config override with data extent.
///
/// Handles partial overrides: when only one bound is set (the other is
/// ±INFINITY), the unset bound falls back to the data extent.
pub(crate) fn resolve_x_extent(config: &ChartConfig, data_extent: (f64, f64)) -> (f64, f64) {
    match config.x_range {
        Some((lo, hi)) => {
            let eff_lo = if lo.is_finite() { lo } else { data_extent.0 };
            let eff_hi = if hi.is_finite() { hi } else { data_extent.1 };
            (eff_lo, eff_hi)
        }
        None => data_extent,
    }
}

/// Resolve the effective y extent: merge config override with data extent.
///
/// Handles partial overrides: when only one bound is set (the other is
/// ±INFINITY), the unset bound falls back to the data extent.
pub(crate) fn resolve_y_extent(config: &ChartConfig, data_extent: (f64, f64)) -> (f64, f64) {
    match config.y_range {
        Some((lo, hi)) => {
            let eff_lo = if lo.is_finite() { lo } else { data_extent.0 };
            let eff_hi = if hi.is_finite() { hi } else { data_extent.1 };
            (eff_lo, eff_hi)
        }
        None => data_extent,
    }
}
