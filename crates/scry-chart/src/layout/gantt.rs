// SPDX-License-Identifier: MIT OR Apache-2.0
//! Gantt chart rendering — horizontal task bars on a categorical Y axis.

use crate::chart::gantt::GanttChart;
use crate::data::SeriesStyle;
use crate::scale::{CategoricalScale, LinearScale, Scale};
use crate::theme::contrast_text_color;
use crate::time_scale::TimeScale;
use scry_engine::style::Color;

use super::{resolve_x_extent, RenderContext, RenderedChart, TextAlign};

/// Darken a color by subtracting `amt` from each RGB channel (u8 space).
fn darken(c: Color, amt: u8) -> Color {
    let r = (c.r * 255.0).round() as u8;
    let g = (c.g * 255.0).round() as u8;
    let b = (c.b * 255.0).round() as u8;
    Color::from_rgba8(
        r.saturating_sub(amt),
        g.saturating_sub(amt),
        b.saturating_sub(amt),
        255,
    )
}

/// Render a Gantt chart.
pub(crate) fn render_gantt(gc: &GanttChart, w: u32, h: u32) -> RenderedChart {
    let config = &gc.config;
    let theme = &config.theme;

    let n = gc.tasks.len();
    if n == 0 {
        let ctx = RenderContext::new(config, w, h, None);
        return ctx.finish();
    }

    // ── Collect task labels (Y axis, bottom → top) ──
    let labels: Vec<String> = gc.tasks.iter().rev().map(|t| t.label.clone()).collect();

    // ── X extent from task start/end values ──
    let x_lo = gc
        .tasks
        .iter()
        .map(|t| t.start.min(t.end))
        .fold(f64::INFINITY, f64::min);
    let x_hi = gc
        .tasks
        .iter()
        .map(|t| t.start.max(t.end))
        .fold(f64::NEG_INFINITY, f64::max);

    let x_extent = resolve_x_extent(config, (x_lo, x_hi));

    // ── Resolve group → color mapping (needed early for legend sizing) ──
    let groups: Vec<String> = {
        let mut seen = Vec::new();
        for t in &gc.tasks {
            let g = t.group.clone().unwrap_or_default();
            if !seen.contains(&g) {
                seen.push(g);
            }
        }
        seen
    };

    let group_color = |g: &str| -> Color {
        let idx = groups.iter().position(|x| x == g).unwrap_or(0);
        theme.resolve_series_color(idx, &SeriesStyle::default())
    };

    let visible_groups: Vec<&String> = groups.iter().filter(|g| !g.is_empty()).collect();
    let has_legend = config.show_legend && visible_groups.len() > 1;

    // ── Build render context ──
    let mut ctx = RenderContext::new(config, w, h, None);
    let (px, py, pw, ph) = ctx.plot;

    // ── Measure longest Y-axis label and widen left margin ──
    let label_fs = super::scaled_font_size(10.0, w, h);
    let char_w = super::char_width_for_size(label_fs);
    let max_label_w = labels
        .iter()
        .map(|l| l.len() as f32 * char_w)
        .fold(0.0_f32, f32::max);
    let label_padding = 10.0; // gap between label text and plot edge
    let needed_left = max_label_w + label_padding;
    let left_shift = (needed_left - px).max(0.0);
    let px = px + left_shift;
    let pw = (pw - left_shift).max(10.0);

    // ── Reserve right-side space for legend (outside plot area) ──
    let legend_fs = super::scaled_font_size(9.0, w, h);
    let swatch_size = (legend_fs * 0.9).max(6.0);
    let legend_pad = 6.0_f32;
    let line_h = swatch_size + 4.0;

    let legend_w = if has_legend {
        visible_groups
            .iter()
            .map(|g| g.len() as f32 * super::char_width_for_size(legend_fs) + swatch_size + 12.0)
            .fold(0.0_f32, f32::max)
            + legend_pad * 2.0
    } else {
        0.0
    };
    let legend_gap = if has_legend { 12.0 } else { 0.0 };
    let pw = (pw - legend_w - legend_gap).max(10.0);

    // Update the context's plot area so overlays align
    ctx.plot = (px, py, pw, ph);

    // ── Scales ──
    let time_scale: Option<TimeScale>;
    let x_scale: LinearScale;
    if gc.use_time_axis {
        let ts = TimeScale::nice(x_extent, (px as f64, (px + pw) as f64));
        x_scale = ts.as_linear().clone();
        time_scale = Some(ts);
    } else {
        x_scale = LinearScale::nice(x_extent, (px as f64, (px + pw) as f64));
        time_scale = None;
    }

    // Categorical Y: map labels to rows (bottom → top)
    let cat_scale = CategoricalScale::new(labels.clone(), (py as f64, (py + ph) as f64));

    // ── Draw X axis ──
    if let Some(ref ts) = time_scale {
        draw_time_x_axis(&mut ctx, config, ts);
    } else {
        ctx.draw_x_value_axis(config, &x_scale);
    }

    // ── Draw Y categorical labels (task names on the left) ──
    {
        let label_x = px - label_padding;
        for (i, lbl) in labels.iter().enumerate() {
            let y = cat_scale.center(i) as f32 + label_fs * 0.35;
            ctx.add_text(
                label_x,
                y,
                lbl,
                theme.text_color(),
                TextAlign::Right,
                label_fs,
                false,
                0.0,
            );
        }
    }

    // ── Y-axis spine (left edge) ──
    ctx.draw(|c| {
        c.line(px, py, px, py + ph)
            .color(theme.axis.color)
            .width(theme.axis.width)
            .done()
    });

    // ── Horizontal row separators (subtle grid) ──
    {
        let grid_show = theme.grid.show_y.unwrap_or(theme.grid.show);
        if grid_show {
            let row_h = cat_scale.band_width() as f32;
            for i in 1..n {
                let sep_y = py + (i as f32) * row_h;
                ctx.draw(|c| {
                    c.line(px, sep_y, px + pw, sep_y)
                        .color(theme.grid.color)
                        .width(theme.grid.width.min(0.5))
                        .done()
                });
            }
        }
    }

    // ── Draw bars ──
    let row_h = cat_scale.band_width() as f32;
    let bar_h = row_h * gc.bar_height;
    let bar_fs = super::scaled_font_size(9.0, w, h);
    let corner_r = (bar_h * 0.15).min(4.0);

    for (data_i, task) in gc.tasks.iter().enumerate() {
        let cat_i = n - 1 - data_i;
        let row_center = cat_scale.center(cat_i) as f32;

        let x0 = x_scale.to_pixel(task.start.min(task.end)) as f32;
        let x1 = x_scale.to_pixel(task.start.max(task.end)) as f32;
        let bar_w = (x1 - x0).max(1.0);
        let bar_y = row_center - bar_h / 2.0;

        let color = group_color(task.group.as_deref().unwrap_or(""));

        // Main bar
        let fill = Color {
            r: color.r,
            g: color.g,
            b: color.b,
            a: 0.86,
        };
        ctx.draw(|c| {
            c.rect(x0, bar_y, bar_w, bar_h)
                .fill(fill)
                .corner_radius(corner_r)
                .done()
        });

        // Bar outline
        let stroke = darken(color, 30);
        ctx.draw(|c| {
            c.rect(x0, bar_y, bar_w, bar_h)
                .stroke(stroke, 1.0)
                .corner_radius(corner_r)
                .done()
        });

        // ── Progress overlay ──
        if gc.show_progress {
            if let Some(pct) = task.progress {
                let prog_w = bar_w * pct;
                if prog_w > 0.5 {
                    let prog_fill = darken(color, 40);
                    ctx.draw(|c| {
                        c.rect(x0, bar_y, prog_w, bar_h)
                            .fill(prog_fill)
                            .corner_radius(corner_r)
                            .done()
                    });
                }
            }
        }

        // ── On-bar label ──
        if gc.show_labels {
            let label_w = task.label.len() as f32 * super::char_width_for_size(bar_fs);
            // Vertical center of the bar (baseline-adjusted)
            let bar_mid_y = bar_y + bar_h / 2.0 + bar_fs * 0.35;

            if label_w < bar_w - 8.0 {
                // Label fits inside the bar — draw centered
                let lx = x0 + bar_w / 2.0;
                let txt_color = contrast_text_color(color);
                ctx.add_text(
                    lx,
                    bar_mid_y,
                    &task.label,
                    txt_color,
                    TextAlign::Center,
                    bar_fs,
                    true,
                    0.0,
                );
            } else {
                // Label doesn't fit inside — try placing to the right.
                // Only draw if there's enough room before the legend/edge.
                let right_edge = px + pw; // plot area right boundary
                let space_right = right_edge - x1 - 5.0;
                if label_w < space_right {
                    let lx = x1 + 5.0;
                    ctx.add_text(
                        lx,
                        bar_mid_y,
                        &task.label,
                        theme.text_color(),
                        TextAlign::Left,
                        bar_fs,
                        false,
                        0.0,
                    );
                }
                // If it doesn't fit to the right either, the Y-axis label
                // on the left already identifies the task, so we skip.
            }
        }

        // ── Date labels on bar edges ──
        if gc.show_dates {
            let date_fs = super::scaled_font_size(8.0, w, h);
            let date_y = bar_y - 3.0;
            let start_label = if let Some(ref ts) = time_scale {
                ts.format_tick(task.start)
            } else {
                format_compact(task.start)
            };
            let end_label = if let Some(ref ts) = time_scale {
                ts.format_tick(task.end)
            } else {
                format_compact(task.end)
            };
            ctx.add_text(
                x0,
                date_y,
                &start_label,
                theme.text_color(),
                TextAlign::Left,
                date_fs,
                false,
                0.0,
            );
            ctx.add_text(
                x1,
                date_y,
                &end_label,
                theme.text_color(),
                TextAlign::Right,
                date_fs,
                false,
                0.0,
            );
        }
    }

    // ── Legend (outside plot area, to the right) ──
    if has_legend {
        let legend_h = visible_groups.len() as f32 * line_h + legend_pad * 2.0;
        let lx = px + pw + legend_gap;
        let ly = py + 4.0; // align near top of plot

        // Legend background
        let bg = Color {
            r: theme.background.r,
            g: theme.background.g,
            b: theme.background.b,
            a: 0.85,
        };
        ctx.draw(|c| {
            c.rect(lx, ly, legend_w, legend_h)
                .fill(bg)
                .corner_radius(3.0)
                .done()
        });
        // Legend border
        ctx.draw(|c| {
            c.rect(lx, ly, legend_w, legend_h)
                .stroke(theme.axis.color, 0.5)
                .corner_radius(3.0)
                .done()
        });

        for (i, g) in visible_groups.iter().enumerate() {
            let row_y = ly + legend_pad + i as f32 * line_h;
            let color = group_color(g);

            // Swatch
            ctx.draw(|c| {
                c.rect(lx + legend_pad, row_y, swatch_size, swatch_size)
                    .fill(color)
                    .corner_radius(2.0)
                    .done()
            });

            // Label
            ctx.add_text(
                lx + legend_pad + swatch_size + 5.0,
                row_y + swatch_size * 0.85,
                g,
                theme.text_color(),
                TextAlign::Left,
                legend_fs,
                false,
                0.0,
            );
        }
    }

    ctx.add_common_overlays(config);
    ctx.finish()
}

// ---------------------------------------------------------------------------
// Time-based X axis
// ---------------------------------------------------------------------------

/// Draw a time-formatted X axis using the TimeScale.
fn draw_time_x_axis(
    ctx: &mut RenderContext,
    config: &crate::chart::ChartConfig,
    ts: &TimeScale,
) {
    let theme = &config.theme;
    let (px, py, pw, ph) = ctx.plot;
    let w = ctx.width();
    let h = ctx.height();

    // Bottom axis spine
    ctx.draw(|c| {
        c.line(px, py + ph, px + pw, py + ph)
            .color(theme.axis.color)
            .width(theme.axis.width)
            .done()
    });

    // Generate time ticks
    let tick_count = crate::axis::adaptive_tick_count(pw, true);
    let ticks = ts.time_ticks(tick_count);
    let tick_fs = super::scaled_font_size(theme.tick_style.font_size, w, h);
    let tick_y = py + ph + super::x_tick_label_offset(h);

    let grid_show = theme.grid.show_x.unwrap_or(theme.grid.show);

    for tick_val in &ticks {
        let x = ts.to_pixel(*tick_val) as f32;
        if x < px || x > px + pw {
            continue;
        }

        // Tick mark
        let tick_len = 4.0_f32;
        ctx.draw(|c| {
            c.line(x, py + ph, x, py + ph + tick_len)
                .color(theme.axis.color)
                .width(0.8)
                .done()
        });

        // Grid line
        if grid_show {
            ctx.draw(|c| {
                c.line(x, py, x, py + ph)
                    .color(theme.grid.color)
                    .width(theme.grid.width)
                    .done()
            });
        }

        // Tick label
        let label = ts.format_tick(*tick_val);
        ctx.add_text(
            x,
            tick_y,
            &label,
            theme.foreground,
            TextAlign::Center,
            tick_fs,
            false,
            0.0,
        );
    }
}

/// Compact number formatter for non-time date labels.
fn format_compact(v: f64) -> String {
    if v.fract().abs() < 1e-9 {
        format!("{}", v as i64)
    } else {
        format!("{v:.1}")
    }
}
