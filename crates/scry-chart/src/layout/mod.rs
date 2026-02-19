// SPDX-License-Identifier: MIT OR Apache-2.0
//! Layout engine — translates chart specs into `PixelCanvas` scenes.
//!
//! This is the core rendering layer. It takes a fully-configured [`Chart`],
//! measures the components (margins, axes, legend), and emits
//! `PixelCanvas` drawing commands.

pub(crate) mod bar;
pub(crate) mod boxplot;
pub(crate) mod bubble;
pub(crate) mod candlestick;
mod common_overlays;
pub(crate) mod contour;
pub(crate) mod funnel;
pub(crate) mod gauge;
pub(crate) mod heatmap;
pub(crate) mod histogram;
pub(crate) mod line;
pub(crate) mod lollipop;
pub(crate) mod pie;
pub(crate) mod radar;
pub(crate) mod render_context;
pub(crate) mod scatter;
pub(crate) mod sparkline;
pub(crate) mod violin;
pub(crate) mod waterfall;

pub(crate) use render_context::RenderContext;

use scry_engine::scene::PixelCanvas;
use std::sync::Arc;

use crate::scale::LinearScale;

use crate::axis::{AxisConfig, AxisSide, LabelRotation};
use crate::chart::{Chart, ChartConfig};
use crate::formatter::{AutoFormatter, LocaleConfig, LocaleFormatter, TickFormatter};
use crate::scale;

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
    (h * 0.018).max(7.0).min(16.0)
}

/// Proportional pixel offset from Y-axis spine to Y tick labels (leftward).
pub(crate) fn y_tick_label_offset(w: u32) -> f32 {
    let w = w as f32;
    (w * 0.015).max(7.0).min(16.0)
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
    pub color: scry_engine::style::Color,
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
    chart.render_with_viewport(width, height, viewport)
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
            config.ticks.y_tick_formatter.as_deref(),
            config.ticks.locale.as_ref(),
            tick_fs,
        )
    } else {
        proportional_y_axis_width(w)
    };

    let title_h = if config.titles.title.is_some() {
        proportional_title_height(h)
    } else {
        0.0
    };

    // Subtitle: smaller text below title
    let subtitle_h = if config.titles.subtitle.is_some() {
        (proportional_title_height(h) * 0.65).max(10.0).min(20.0)
    } else {
        0.0
    };

    // Footer: small text at the very bottom
    let footer_h = if config.titles.footer.is_some() {
        (proportional_title_height(h) * 0.55).max(10.0).min(18.0)
    } else {
        0.0
    };

    let y_label_w = estimate_y_label_width(
        config.titles.y_label.as_deref(),
        w,
        h,
        config.theme.label_style.font_size,
    );
    let x_tick_h = proportional_x_tick_height(h, config.ticks.x_tick_rotation);
    let x_label_h = if config.titles.x_label.is_some() {
        proportional_x_label_height(h)
    } else {
        0.0
    };

    // Secondary Y-axis gutter (right side) — allocated when any secondary
    // Y-axis config is present.
    let has_secondary = config.secondary.label.is_some()
        || config.secondary.range.is_some()
        || config.secondary.formatter.is_some();
    let secondary_y_w = if has_secondary {
        let sec_label_w = estimate_y_label_width(
            config.secondary.label.as_deref(),
            w,
            h,
            config.theme.label_style.font_size,
        );
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
        config.ticks.x_tick_rotation
    } else {
        LabelRotation::Horizontal
    };
    // Resolve formatter and tick_step from ChartConfig based on axis orientation.
    // If a locale is set and no custom formatter is provided, wrap the default
    // AutoFormatter with locale-aware post-processing.
    let tick_formatter = if is_x_axis {
        match (&config.ticks.x_tick_formatter, &config.ticks.locale) {
            (Some(f), _) => Some(f.clone()),
            (None, Some(loc)) => Some(Arc::new(LocaleFormatter::new(AutoFormatter, loc.clone()))
                as Arc<dyn TickFormatter>),
            _ => None,
        }
    } else {
        match (&config.ticks.y_tick_formatter, &config.ticks.locale) {
            (Some(f), _) => Some(f.clone()),
            (None, Some(loc)) => Some(Arc::new(LocaleFormatter::new(AutoFormatter, loc.clone()))
                as Arc<dyn TickFormatter>),
            _ => None,
        }
    };
    let tick_step = if is_x_axis {
        config.ticks.x_tick_step
    } else {
        config.ticks.y_tick_step
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
    let tick_formatter =
        match (&config.secondary.formatter, &config.ticks.locale) {
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
    match config.axes.x_range {
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
    match config.axes.y_range {
        Some((lo, hi)) => {
            let eff_lo = if lo.is_finite() { lo } else { data_extent.0 };
            let eff_hi = if hi.is_finite() { hi } else { data_extent.1 };
            (eff_lo, eff_hi)
        }
        None => data_extent,
    }
}
