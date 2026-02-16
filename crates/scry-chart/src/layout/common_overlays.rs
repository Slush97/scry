//! Common overlay rendering: title, subtitle, footer, axis labels,
//! categorical X labels, annotations, and trend lines.

use scry_engine::style::{Color, DashPattern};

use crate::chart::ChartConfig;
use crate::scale::{LinearScale, Scale};

use super::render_context::RenderContext;
use super::{
    char_width_for_size, estimate_y_label_width, proportional_margin,
    proportional_title_height, proportional_x_label_height, proportional_x_tick_height,
    scaled_font_size, x_tick_label_offset, y_tick_label_offset, TextAlign, TextOverlay,
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
                y_px: py + ph + x_tick_h + x_label_h,
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
}
