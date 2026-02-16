//! SVG export for charts.
//!
//! Generates standalone SVG documents from any [`Chart`] by walking the
//! `PixelCanvas` scene graph commands and emitting corresponding SVG elements.
//! No external dependencies are required — SVG is generated as plain strings.
//!
//! Text overlays (titles, axis labels, tick labels) are emitted as `<text>`
//! elements with font-family matching the Inter font used in PNG export.
//!
//! # Example
//!
//! ```ignore
//! let chart = Chart::line(&[1.0, 4.0, 2.0, 8.0]).title("Demo").build();
//! let svg = scry_chart::svg_export::render_to_svg(&chart, 800, 500);
//! std::fs::write("chart.svg", svg).unwrap();
//! ```

use std::fmt::Write;

use scry_engine::scene::command::DrawCommand;
use scry_engine::scene::style::{
    Color, DashPattern, FillStyle, GradientDef, GradientKind, LineCap, LineJoin, ShapeStyle,
    StrokeStyle,
};

use crate::chart::Chart;
use crate::layout::{self, TextAlign, TextOverlay};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Render a chart to an SVG string.
///
/// Returns a complete, standalone SVG document suitable for embedding in
/// HTML, saving to a file, or opening in a browser.
#[must_use]
pub fn render_to_svg(chart: &Chart, width: u32, height: u32) -> String {
    let rendered = layout::render_chart(chart, width, height);
    let mut svg = String::with_capacity(8192);
    let mut defs = String::new();
    let mut grad_id = 0_u32;

    // SVG header
    let _ = write!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">"#,
        width, height, width, height
    );
    svg.push('\n');

    // Render all drawing commands
    let mut body = String::with_capacity(4096);
    for cmd in rendered.canvas.commands() {
        emit_command(&mut body, &mut defs, &mut grad_id, cmd);
    }

    // Render text overlays
    for overlay in &rendered.text_overlays {
        emit_text_overlay(&mut body, overlay);
    }

    // Insert defs block if we have any gradient definitions
    if !defs.is_empty() {
        svg.push_str("<defs>\n");
        svg.push_str(&defs);
        svg.push_str("</defs>\n");
    }

    svg.push_str(&body);
    svg.push_str("</svg>\n");
    svg
}

/// Render a chart and save directly to an SVG file.
///
/// # Errors
///
/// Returns an error if file I/O fails.
pub fn save_svg(
    chart: &Chart,
    width: u32,
    height: u32,
    path: impl AsRef<std::path::Path>,
) -> Result<(), String> {
    let data = render_to_svg(chart, width, height);
    std::fs::write(path.as_ref(), data)
        .map_err(|e| format!("failed to write {}: {e}", path.as_ref().display()))
}

// ---------------------------------------------------------------------------
// Command → SVG element translation
// ---------------------------------------------------------------------------

#[allow(clippy::match_wildcard_for_single_variants)]
fn emit_command(body: &mut String, defs: &mut String, grad_id: &mut u32, cmd: &DrawCommand) {
    match cmd {
        DrawCommand::Clear { color } => {
            let _ = write!(
                body,
                r#"<rect width="100%" height="100%" fill="{}"/>"#,
                svg_color(*color)
            );
            body.push('\n');
        }

        DrawCommand::Circle {
            cx,
            cy,
            radius,
            style,
        } => {
            let fill_attr = svg_fill_attr(style, defs, grad_id);
            let stroke_attr = svg_stroke_attr(style);
            let _ = write!(
                body,
                r#"<circle cx="{:.1}" cy="{:.1}" r="{:.1}"{}{}/>"#,
                cx, cy, radius, fill_attr, stroke_attr
            );
            body.push('\n');
        }

        DrawCommand::Rectangle {
            rect,
            corner_radius,
            style,
        } => {
            let fill_attr = svg_fill_attr(style, defs, grad_id);
            let stroke_attr = svg_stroke_attr(style);
            let rx = if *corner_radius > 0.0 {
                format!(r#" rx="{:.1}" ry="{:.1}""#, corner_radius, corner_radius)
            } else {
                String::new()
            };
            let _ = write!(
                body,
                r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}"{}{}{}/>"#,
                rect.x, rect.y, rect.width, rect.height, rx, fill_attr, stroke_attr
            );
            body.push('\n');
        }

        DrawCommand::Ellipse {
            cx,
            cy,
            rx,
            ry,
            rotation,
            style,
        } => {
            let fill_attr = svg_fill_attr(style, defs, grad_id);
            let stroke_attr = svg_stroke_attr(style);
            let transform = if rotation.abs() > 0.001 {
                format!(
                    r#" transform="rotate({:.1} {:.1} {:.1})""#,
                    rotation.to_degrees(),
                    cx,
                    cy
                )
            } else {
                String::new()
            };
            let _ = write!(
                body,
                r#"<ellipse cx="{:.1}" cy="{:.1}" rx="{:.1}" ry="{:.1}"{}{}{}/>"#,
                cx, cy, rx, ry, transform, fill_attr, stroke_attr
            );
            body.push('\n');
        }

        DrawCommand::Line {
            x1,
            y1,
            x2,
            y2,
            stroke,
            ..
        } => {
            let attrs = svg_stroke_style_attr(stroke);
            let _ = write!(
                body,
                r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}"{}/>"#,
                x1, y1, x2, y2, attrs
            );
            body.push('\n');
        }

        DrawCommand::Polyline {
            points,
            closed,
            style,
        } => {
            if points.is_empty() {
                return;
            }
            let pts: String = points
                .iter()
                .map(|(x, y)| format!("{:.1},{:.1}", x, y))
                .collect::<Vec<_>>()
                .join(" ");

            let fill_attr = svg_fill_attr(style, defs, grad_id);
            let stroke_attr = svg_stroke_attr(style);

            if *closed {
                let _ = write!(
                    body,
                    r#"<polygon points="{}"{}{}/>"#,
                    pts, fill_attr, stroke_attr
                );
            } else {
                // Open polylines typically have no fill
                let fill = if style.fill.is_some() {
                    fill_attr
                } else {
                    r#" fill="none""#.to_string()
                };
                let _ = write!(
                    body,
                    r#"<polyline points="{}"{}{}/>"#,
                    pts, fill, stroke_attr
                );
            }
            body.push('\n');
        }

        DrawCommand::Gradient { rect, gradient, .. } => {
            let id = alloc_gradient(defs, grad_id, gradient);
            let _ = write!(
                body,
                r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="url(#{})"/>"#,
                rect.x, rect.y, rect.width, rect.height, id
            );
            body.push('\n');
        }

        DrawCommand::Arc {
            cx,
            cy,
            radius,
            start_angle,
            sweep_angle,
            style,
        } => {
            emit_arc(
                body,
                defs,
                grad_id,
                *cx,
                *cy,
                *radius,
                *start_angle,
                *sweep_angle,
                style,
            );
        }

        DrawCommand::Group {
            commands,
            transform,
            opacity,
            ..
        } => {
            let mut group_attrs = String::new();

            // Transform
            let ts = transform;
            if (ts.sx - 1.0).abs() > 0.001
                || ts.kx.abs() > 0.001
                || ts.ky.abs() > 0.001
                || (ts.sy - 1.0).abs() > 0.001
                || ts.tx.abs() > 0.001
                || ts.ty.abs() > 0.001
            {
                let _ = write!(
                    group_attrs,
                    r#" transform="matrix({:.4} {:.4} {:.4} {:.4} {:.1} {:.1})""#,
                    ts.sx, ts.kx, ts.ky, ts.sy, ts.tx, ts.ty
                );
            }

            if (*opacity - 1.0).abs() > 0.001 {
                let _ = write!(group_attrs, r#" opacity="{:.2}""#, opacity);
            }

            let _ = write!(body, "<g{}>", group_attrs);
            body.push('\n');
            for child in commands {
                emit_command(body, defs, grad_id, child);
            }
            body.push_str("</g>\n");
        }

        // Path commands from arbitrary Bézier paths — emit as SVG <path>
        DrawCommand::Path { path, style } => {
            let fill_attr = svg_fill_attr(style, defs, grad_id);
            let stroke_attr = svg_stroke_attr(style);
            let d = path_data_to_svg_d(path);
            let _ = write!(body, r#"<path d="{}"{}{}/>"#, d, fill_attr, stroke_attr);
            body.push('\n');
        }

        // Image is not handled in SVG output.
        DrawCommand::Image { .. } => {}

        // Feature-gated variants (e.g. Text behind "text" feature).
        #[allow(unreachable_patterns, clippy::match_same_arms, clippy::match_wildcard_for_single_variants)]
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Arc rendering
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn emit_arc(
    body: &mut String,
    defs: &mut String,
    grad_id: &mut u32,
    cx: f32,
    cy: f32,
    radius: f32,
    start_angle: f32,
    sweep_angle: f32,
    style: &ShapeStyle,
) {
    let fill_attr = svg_fill_attr(style, defs, grad_id);
    let stroke_attr = svg_stroke_attr(style);

    // SVG arc path: M start A rx ry rotation large-arc sweep end
    let x1 = cx + radius * start_angle.cos();
    let y1 = cy + radius * start_angle.sin();
    let end_angle = start_angle + sweep_angle;
    let x2 = cx + radius * end_angle.cos();
    let y2 = cy + radius * end_angle.sin();

    let large_arc = i32::from(sweep_angle.abs() > std::f32::consts::PI);
    let sweep_flag = i32::from(sweep_angle > 0.0);

    // If there's a fill, draw a pie slice (move to center, line to start, arc, close)
    if style.fill.is_some() {
        let _ = write!(
            body,
            r#"<path d="M {:.1} {:.1} L {:.1} {:.1} A {:.1} {:.1} 0 {} {} {:.1} {:.1} Z"{}{}/>"#,
            cx, cy, x1, y1, radius, radius, large_arc, sweep_flag, x2, y2, fill_attr, stroke_attr
        );
    } else {
        let _ = write!(
            body,
            r#"<path d="M {:.1} {:.1} A {:.1} {:.1} 0 {} {} {:.1} {:.1}"{}{}/>"#,
            x1, y1, radius, radius, large_arc, sweep_flag, x2, y2, fill_attr, stroke_attr
        );
    }
    body.push('\n');
}

// ---------------------------------------------------------------------------
// Path data conversion
// ---------------------------------------------------------------------------

fn path_data_to_svg_d(path_data: &scry_engine::scene::command::PathData) -> String {
    let path = path_data.path();
    let mut d = String::with_capacity(256);

    // Use the public PathSegment iterator (PathVerb is not re-exported).
    for seg in path.segments() {
        match seg {
            tiny_skia::PathSegment::MoveTo(pt) => {
                let _ = write!(d, "M {:.1} {:.1} ", pt.x, pt.y);
            }
            tiny_skia::PathSegment::LineTo(pt) => {
                let _ = write!(d, "L {:.1} {:.1} ", pt.x, pt.y);
            }
            tiny_skia::PathSegment::QuadTo(p1, p2) => {
                let _ = write!(d, "Q {:.1} {:.1} {:.1} {:.1} ", p1.x, p1.y, p2.x, p2.y);
            }
            tiny_skia::PathSegment::CubicTo(p1, p2, p3) => {
                let _ = write!(
                    d,
                    "C {:.1} {:.1} {:.1} {:.1} {:.1} {:.1} ",
                    p1.x, p1.y, p2.x, p2.y, p3.x, p3.y
                );
            }
            tiny_skia::PathSegment::Close => {
                d.push_str("Z ");
            }
        }
    }
    d.trim_end().to_string()
}

// ---------------------------------------------------------------------------
// Text overlay → <text> element
//
// Baseline approach: all `<text>` elements use `dominant-baseline="central"`
// uniformly (titles, labels, ticks, data labels). The layout engine in
// `layout/mod.rs` positions each `TextOverlay` so that `y_px` is the
// *vertical center* of the text line, not the alphabetic baseline. Using
// `"central"` aligns the SVG rendering with the PNG export and terminal
// widget paths, which both assume center-anchored placement.
// ---------------------------------------------------------------------------

fn emit_text_overlay(body: &mut String, overlay: &TextOverlay) {
    let anchor = match overlay.align {
        TextAlign::Left => "start",
        TextAlign::Center => "middle",
        TextAlign::Right => "end",
    };

    let weight = if overlay.bold { "bold" } else { "normal" };

    let transform = if overlay.rotation_deg.abs() > 0.01 {
        format!(
            r#" transform="rotate({:.1} {:.1} {:.1})""#,
            -overlay.rotation_deg, overlay.x_px, overlay.y_px
        )
    } else {
        String::new()
    };

    let color = svg_color(overlay.color);

    // Escape XML special characters in text
    let text = xml_escape(&overlay.text);

    // NOTE: font-size uses explicit `px` units for SVG standards compliance.
    write!(
        body,
        r#"<text x="{:.1}" y="{:.1}" font-family="Inter, system-ui, sans-serif" font-size="{:.0}px" font-weight="{}" fill="{}" text-anchor="{}" dominant-baseline="central"{}>{}</text>"#,
        overlay.x_px, overlay.y_px, overlay.font_size, weight, color, anchor, transform, text
    )
    .unwrap();
    body.push('\n');
}

// ---------------------------------------------------------------------------
// Style helpers
// ---------------------------------------------------------------------------

fn svg_color(color: Color) -> String {
    let r = (color.r * 255.0) as u8;
    let g = (color.g * 255.0) as u8;
    let b = (color.b * 255.0) as u8;

    if (color.a - 1.0).abs() < 0.001 {
        format!("rgb({},{},{})", r, g, b)
    } else if color.a < 0.001 {
        "none".to_string()
    } else {
        format!("rgba({},{},{},{:.2})", r, g, b, color.a)
    }
}

#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
fn svg_fill_attr(style: &ShapeStyle, defs: &mut String, grad_id: &mut u32) -> String {
    match &style.fill {
        None => r#" fill="none""#.to_string(),
        Some(FillStyle::Solid(color)) => {
            format!(r#" fill="{}""#, svg_color(*color))
        }
        Some(FillStyle::LinearGradient(grad) | FillStyle::RadialGradient(grad)) => {
            let id = alloc_gradient(defs, grad_id, grad);
            format!(r#" fill="url(#{})""#, id)
        }
    }
}

fn svg_stroke_attr(style: &ShapeStyle) -> String {
    style.stroke.as_ref().map_or_else(String::new, svg_stroke_style_attr)
}

fn svg_stroke_style_attr(stroke: &StrokeStyle) -> String {
    let mut attrs = format!(
        r#" stroke="{}" stroke-width="{:.1}""#,
        svg_color(stroke.color),
        stroke.width
    );

    match stroke.line_cap {
        LineCap::Butt => {} // SVG default
        LineCap::Round => attrs.push_str(r#" stroke-linecap="round""#),
        LineCap::Square => attrs.push_str(r#" stroke-linecap="square""#),
    }

    match stroke.line_join {
        LineJoin::Miter => {} // SVG default
        LineJoin::Round => attrs.push_str(r#" stroke-linejoin="round""#),
        LineJoin::Bevel => attrs.push_str(r#" stroke-linejoin="bevel""#),
    }

    if let Some(ref dash) = stroke.dash {
        let _ = write!(attrs, r#" stroke-dasharray="{}""#, svg_dash_pattern(dash));
        if dash.offset.abs() > 0.001 {
            let _ = write!(attrs, r#" stroke-dashoffset="{:.1}""#, dash.offset);
        }
    }

    attrs
}

fn svg_dash_pattern(dash: &DashPattern) -> String {
    dash.intervals
        .iter()
        .map(|v| format!("{:.1}", v))
        .collect::<Vec<_>>()
        .join(",")
}

// ---------------------------------------------------------------------------
// Gradient allocation into <defs>
// ---------------------------------------------------------------------------

fn alloc_gradient(defs: &mut String, grad_id: &mut u32, gradient: &GradientDef) -> String {
    let id = format!("grad{}", grad_id);
    *grad_id += 1;

    match &gradient.kind {
        GradientKind::Linear { start, end } => {
            let _ = write!(
                defs,
                r#"<linearGradient id="{}" x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" gradientUnits="userSpaceOnUse">"#,
                id, start.x, start.y, end.x, end.y
            );
            defs.push('\n');
            for stop in &gradient.stops {
                let _ = write!(
                    defs,
                    r#"<stop offset="{:.2}" stop-color="{}" stop-opacity="{:.2}"/>"#,
                    stop.position,
                    svg_color(Color::from_rgba(
                        stop.color.r,
                        stop.color.g,
                        stop.color.b,
                        1.0
                    )),
                    stop.color.a
                );
                defs.push('\n');
            }
            defs.push_str("</linearGradient>\n");
        }
        GradientKind::Radial { center, radius } => {
            let _ = write!(
                defs,
                r#"<radialGradient id="{}" cx="{:.1}" cy="{:.1}" r="{:.1}" gradientUnits="userSpaceOnUse">"#,
                id, center.x, center.y, radius
            );
            defs.push('\n');
            for stop in &gradient.stops {
                let _ = write!(
                    defs,
                    r#"<stop offset="{:.2}" stop-color="{}" stop-opacity="{:.2}"/>"#,
                    stop.position,
                    svg_color(Color::from_rgba(
                        stop.color.r,
                        stop.color.g,
                        stop.color.b,
                        1.0
                    )),
                    stop.color.a
                );
                defs.push('\n');
            }
            defs.push_str("</radialGradient>\n");
        }
    }

    id
}

// ---------------------------------------------------------------------------
// XML utilities
// ---------------------------------------------------------------------------

fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chart::Chart;
    use crate::theme::Theme;

    #[test]
    fn svg_contains_svg_root() {
        let chart = Chart::line(&[1.0, 4.0, 2.0, 8.0]).title("SVG Test").build();
        let svg = render_to_svg(&chart, 400, 300);
        assert!(svg.starts_with(r#"<svg xmlns="http://www.w3.org/2000/svg""#));
        assert!(svg.ends_with("</svg>\n"));
    }

    #[test]
    fn svg_has_text_overlays() {
        let chart = Chart::line(&[1.0, 4.0, 2.0, 8.0])
            .title("Test Title")
            .x_label("X Axis")
            .y_label("Y Axis")
            .build();
        let svg = render_to_svg(&chart, 400, 300);
        assert!(svg.contains("Test Title"));
        assert!(svg.contains("X Axis"));
        assert!(svg.contains("Y Axis"));
    }

    #[test]
    fn svg_contains_line_elements() {
        let chart = Chart::line(&[1.0, 4.0, 2.0, 8.0])
            .theme(Theme::dark())
            .build();
        let svg = render_to_svg(&chart, 400, 300);
        // Should have polyline for the data series
        assert!(svg.contains("<polyline") || svg.contains("<line"));
    }

    #[test]
    fn svg_contains_rect_for_bar_chart() {
        let chart = Chart::bar(
            vec!["A".into(), "B".into(), "C".into()],
            &[10.0, 20.0, 30.0],
        )
        .build();
        let svg = render_to_svg(&chart, 400, 300);
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn svg_contains_circle_for_scatter() {
        let chart = Chart::scatter(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).build();
        let svg = render_to_svg(&chart, 400, 300);
        assert!(svg.contains("<circle"));
    }

    #[test]
    fn svg_gradient_defs() {
        let chart = Chart::line(&[1.0, 4.0, 2.0, 8.0])
            .filled()
            .theme(Theme::dark())
            .build();
        let svg = render_to_svg(&chart, 400, 300);
        // Filled line chart should have gradient defs
        assert!(svg.contains("<defs>"));
        assert!(svg.contains("<linearGradient"));
    }

    #[test]
    fn svg_is_valid_structure() {
        let chart = Chart::scatter(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0])
            .title("Scatter")
            .build();
        let svg = render_to_svg(&chart, 400, 300);

        // Basic well-formedness checks
        assert!(svg.contains("viewBox="));
        assert!(svg.contains("width="));
        assert!(svg.contains("height="));

        // Count open/close tags are balanced (rough check)
        let opens = svg.matches("<svg").count();
        let closes = svg.matches("</svg>").count();
        assert_eq!(opens, closes);
    }

    #[test]
    fn svg_dash_pattern_renders() {
        let chart = Chart::line(&[1.0, 4.0, 2.0, 8.0])
            .h_line(3.0)
            .theme(Theme::dark())
            .build();
        let svg = render_to_svg(&chart, 400, 300);
        // Reference lines are dashed — check for stroke-dasharray
        assert!(
            svg.contains("stroke-dasharray"),
            "dashed reference lines should produce stroke-dasharray"
        );
    }

    #[test]
    fn svg_xml_escapes_special_chars() {
        let escaped = xml_escape("x < y && a > b \"quoted\" & 'apos'");
        assert_eq!(
            escaped,
            "x &lt; y &amp;&amp; a &gt; b &quot;quoted&quot; &amp; &apos;apos&apos;"
        );
    }

    #[test]
    fn save_svg_creates_file() {
        let chart = Chart::line(&[1.0, 2.0, 3.0]).build();
        let path = std::env::temp_dir().join("scry_chart_test_save.svg");
        save_svg(&chart, 200, 150, &path).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("<svg"));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn svg_color_formats() {
        assert_eq!(svg_color(Color::RED), "rgb(255,0,0)");
        assert_eq!(svg_color(Color::TRANSPARENT), "none");
        let semi = Color::from_rgba(1.0, 0.0, 0.0, 0.5);
        assert_eq!(svg_color(semi), "rgba(255,0,0,0.50)");
    }

    #[test]
    fn svg_pie_chart_renders() {
        let chart = Chart::pie(
            vec!["A".into(), "B".into(), "C".into()],
            &[30.0, 50.0, 20.0],
        )
        .build();
        let svg = render_to_svg(&chart, 400, 400);
        // Pie chart renders as filled polygons (Polyline closed=true)
        assert!(svg.contains("<polygon") || svg.contains("<path") || svg.contains("<circle"));
    }

    #[test]
    fn svg_heatmap_has_rects() {
        let chart = Chart::heatmap(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).build();
        let svg = render_to_svg(&chart, 300, 250);
        // Heatmap should have rectangles for cells
        assert!(svg.contains("<rect"));
    }
}
