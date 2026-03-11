// SPDX-License-Identifier: MIT OR Apache-2.0
//! Common overlay rendering: title, subtitle, footer, axis labels,
//! categorical X labels, annotations, and trend lines.

use scry_engine::style::{Color, DashPattern};

use crate::chart::ChartConfig;
use crate::scale::{LinearScale, Scale};

use super::render_context::RenderContext;
use super::{
    char_width_for_size, estimate_y_label_width, proportional_margin, proportional_title_height,
    proportional_x_label_height, proportional_x_tick_height, scaled_font_size, x_tick_label_offset,
    y_tick_label_offset, TextAlign,
};

impl RenderContext {
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

        let _title_h = if config.titles.title.is_some() {
            proportional_title_height(h)
        } else {
            0.0
        };

        // Font-relative gaps between title elements
        let title_gap = title_fs * 0.3;
        let subtitle_gap = subtitle_fs * 0.2;

        if let Some(ref title) = config.titles.title {
            self.add_text(
                px + pw / 2.0,
                margin + extra_top + title_gap,
                title,
                config.theme.title_style.color,
                TextAlign::Center,
                title_fs,
                true,
                0.0,
            );
        }

        // Subtitle: positioned below the title, smaller and not bold.
        if let Some(ref subtitle) = config.titles.subtitle {
            let sub_y = margin + extra_top + title_gap + title_fs + subtitle_gap;
            self.add_text(
                px + pw / 2.0,
                sub_y,
                subtitle,
                config.theme.label_style.color,
                TextAlign::Center,
                subtitle_fs,
                false,
                0.0,
            );
        }

        if let Some(ref label) = config.titles.x_label {
            let x_tick_h = proportional_x_tick_height(w, h, config.ticks.x_tick_rotation);
            let x_label_h = proportional_x_label_height(h);
            // Center the label within the reserved x-label strip.
            // 0.45× positions the baseline within bounds, leaving room.
            self.add_text(
                px + pw / 2.0,
                py + ph + x_tick_h + x_label_h * 0.45,
                label,
                config.theme.label_style.color,
                TextAlign::Center,
                label_fs,
                false,
                0.0,
            );
        }

        if let Some(ref label) = config.titles.y_label {
            // Y-axis label is rotated 90° so it reads vertically.
            let y_label_w =
                estimate_y_label_width(Some(label), w, h, config.theme.label_style.font_size);
            self.add_text(
                margin + y_label_w / 2.0,
                py + ph / 2.0,
                label,
                config.theme.label_style.color,
                TextAlign::Center,
                label_fs,
                false,
                90.0,
            );
        }

        // Secondary Y-axis label (right side, rotated -90°).
        if let Some(ref label) = config.secondary.label {
            let sec_label_w =
                estimate_y_label_width(Some(label), w, h, config.theme.label_style.font_size);
            self.add_text(
                (w as f32) - margin - sec_label_w / 2.0,
                py + ph / 2.0,
                label,
                config.theme.label_style.color,
                TextAlign::Center,
                label_fs,
                false,
                -90.0,
            );
        }

        // Footer: small text at bottom center.
        if let Some(ref footer) = config.titles.footer {
            let extra_bottom = config.margin.as_ref().map_or(0.0, |m| m.bottom);
            let footer_gap = footer_fs * 0.25;
            self.add_text(
                px + pw / 2.0,
                (h as f32) - margin - extra_bottom + footer_gap,
                footer,
                config.theme.label_style.color,
                TextAlign::Center,
                footer_fs,
                false,
                0.0,
            );
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
            self.add_text(
                px - y_off,
                *pos,
                label,
                color,
                TextAlign::Right,
                tick_fs,
                false,
                0.0,
            );
        }
    }

    /// Draw category labels along the X axis for bar/boxplot charts.
    ///
    /// Implements an automatic collision-avoidance cascade:
    /// 1. Try horizontal labels
    /// 2. Stagger — alternate labels drop to a second row
    /// 3. Rotate 45°
    /// 4. Rotate 90°
    /// 5. Skip every Nth (keeping first + last)
    /// 6. Truncate individual labels that exceed band width
    ///
    /// If the user has explicitly set `x_tick_rotation`, that rotation is
    /// used instead of steps 1–4, but skipping and truncation still apply.
    pub fn draw_categorical_x_labels(
        &mut self,
        config: &ChartConfig,
        cat_scale: &crate::scale::CategoricalScale,
        labels: &[String],
    ) {
        if labels.is_empty() {
            return;
        }

        let (_px, py, _pw, ph) = self.plot;
        let theme = &config.theme;
        let w = self.width();
        let h = self.height();
        let x_off = x_tick_label_offset(h);
        let tick_fs = scaled_font_size(theme.tick_style.font_size, w, h);
        let char_w = char_width_for_size(tick_fs);

        let band = cat_scale.band_width() as f32;
        let max_chars = labels.iter().map(|l| l.chars().count()).max().unwrap_or(1);
        let label_w = max_chars as f32 * char_w;

        // Minimum gap between adjacent labels (px).
        let gap = 4.0_f32;

        // --- Determine rotation and stagger strategy ---
        let user_set_rotation =
            config.ticks.x_tick_rotation != crate::axis::LabelRotation::Horizontal;

        let (rot_deg, stagger) = if user_set_rotation {
            // User explicitly chose a rotation — respect it, no stagger.
            (config.ticks.x_tick_rotation.degrees(), false)
        } else if label_w + gap <= band {
            // Labels fit horizontally — no changes needed.
            (0.0, false)
        } else if label_w + gap <= band * 2.0 && labels.len() > 1 {
            // Labels would overlap horizontally but fit if staggered
            // (alternate labels on a second row, each gets ~2× band of room).
            (0.0, true)
        } else if label_w * 0.71 + gap <= band {
            // 45° rotation makes them fit.
            (45.0, false)
        } else {
            // 90° rotation (maximum density).
            (90.0, false)
        };

        let align = if rot_deg > 0.0 {
            TextAlign::Right
        } else {
            TextAlign::Center
        };

        // Effective horizontal space per label (staggered labels span 2 bands).
        let effective_band = if stagger { band * 2.0 } else { band };

        // --- Skip every Nth if still too dense (after rotation) ---
        let effective_label_w = if rot_deg >= 89.0 {
            tick_fs + gap // vertical: width ≈ font height
        } else if rot_deg > 0.0 {
            let rad = rot_deg.to_radians();
            label_w * rad.cos() + gap
        } else {
            label_w + gap
        };

        let skip = if effective_label_w > effective_band && labels.len() > 1 {
            ((effective_label_w / effective_band).ceil() as usize).max(2)
        } else {
            1
        };

        // --- Truncate long individual labels ---
        let max_visible_chars = if rot_deg > 0.0 {
            // Rotated labels have diagonal space — be generous.
            usize::MAX
        } else {
            let available = effective_band / char_w;
            (available.floor() as usize).max(4)
        };

        let total = labels.len();
        for (ci, label) in labels.iter().enumerate() {
            // Skip logic: keep first, last, and every Nth.
            if skip > 1 && ci % skip != 0 && ci != 0 && ci != total - 1 {
                continue;
            }

            // Truncate if needed using text_utils::ellipsize.
            let max_label_px = max_visible_chars as f32 * char_w;
            let display_label = crate::text_utils::ellipsize(label, max_label_px, char_w);

            // Stagger: odd-indexed labels get an extra vertical offset.
            let stagger_offset = if stagger && ci % 2 == 1 {
                tick_fs + 2.0
            } else {
                0.0
            };

            self.add_text(
                cat_scale.center(ci) as f32,
                py + ph + x_off + stagger_offset,
                &display_label,
                theme.text_color(),
                align,
                tick_fs,
                false,
                rot_deg,
            );
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
        for ann in &config.overlays.annotations {
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

            self.add_text(
                text_x,
                text_y,
                &ann.text,
                ann.style.text_color,
                TextAlign::Left,
                ann_fs,
                false,
                0.0,
            );
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

        // Clamp to plot area so the trend line never bleeds outside axes.
        let (plot_x, plot_y, plot_w, plot_h) = self.plot;
        let Some((px1, py1, px2, py2)) = clip_line_to_rect(
            px1,
            py1,
            px2,
            py2,
            plot_x,
            plot_y,
            plot_x + plot_w,
            plot_y + plot_h,
        ) else {
            return; // entirely outside
        };

        let trend_color = color.with_alpha(0.6);
        self.draw(|c| {
            c.line(px1, py1, px2, py2)
                .color(trend_color)
                .width(2.0)
                .dash(DashPattern::new(vec![12.0, 6.0], 0.0))
                .done()
        });
    }
}

// ---------------------------------------------------------------------------
// Cohen-Sutherland line clipping
// ---------------------------------------------------------------------------

const INSIDE: u8 = 0;
const LEFT: u8 = 1;
const RIGHT: u8 = 2;
const BOTTOM: u8 = 4;
const TOP: u8 = 8;

fn outcode(x: f32, y: f32, xmin: f32, ymin: f32, xmax: f32, ymax: f32) -> u8 {
    let mut code = INSIDE;
    if x < xmin {
        code |= LEFT;
    } else if x > xmax {
        code |= RIGHT;
    }
    if y < ymin {
        code |= TOP;
    } else if y > ymax {
        code |= BOTTOM;
    }
    code
}

/// Clip a line segment `(x0,y0)→(x1,y1)` to the rectangle
/// `[xmin,xmax] × [ymin,ymax]`.  Returns `None` if the entire segment
/// is outside.
fn clip_line_to_rect(
    mut x0: f32,
    mut y0: f32,
    mut x1: f32,
    mut y1: f32,
    xmin: f32,
    ymin: f32,
    xmax: f32,
    ymax: f32,
) -> Option<(f32, f32, f32, f32)> {
    let mut code0 = outcode(x0, y0, xmin, ymin, xmax, ymax);
    let mut code1 = outcode(x1, y1, xmin, ymin, xmax, ymax);

    for _ in 0..20 {
        if (code0 | code1) == 0 {
            return Some((x0, y0, x1, y1)); // both inside
        }
        if (code0 & code1) != 0 {
            return None; // both outside same side
        }
        let out = if code0 != 0 { code0 } else { code1 };
        let (x, y);
        if out & TOP != 0 {
            x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0);
            y = ymin;
        } else if out & BOTTOM != 0 {
            x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0);
            y = ymax;
        } else if out & RIGHT != 0 {
            y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0);
            x = xmax;
        } else {
            y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0);
            x = xmin;
        }
        if out == code0 {
            x0 = x;
            y0 = y;
            code0 = outcode(x0, y0, xmin, ymin, xmax, ymax);
        } else {
            x1 = x;
            y1 = y;
            code1 = outcode(x1, y1, xmin, ymin, xmax, ymax);
        }
    }
    Some((x0, y0, x1, y1))
}
