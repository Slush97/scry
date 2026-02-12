//! Rasterization of scenes into pixel buffers via `tiny-skia`.
//!
//! The rasterizer walks the [`DrawCommand`] list
//! produced by a [`PixelCanvas`] and translates each
//! command into the corresponding `tiny-skia` drawing calls.

use tiny_skia::{
    FillRule, LineCap as SkiaLineCap, LineJoin as SkiaLineJoin, Paint, PathBuilder, Pixmap,
    Stroke as SkiaStroke, Transform as SkiaTransform,
};

use crate::scene::command::DrawCommand;
#[cfg(feature = "text")]
use crate::scene::command::FontData;
use crate::scene::style::{
    Color, FillStyle, GradientKind, LineCap, LineJoin, ShapeStyle, StrokeStyle,
};
use crate::scene::PixelCanvas;
use crate::PixelCanvasError;

/// Rasterizes a [`PixelCanvas`] scene into a `tiny_skia::Pixmap`.
pub struct Rasterizer;

impl Rasterizer {
    /// Rasterize a canvas scene into a pixel buffer.
    ///
    /// # Errors
    ///
    /// Returns [`PixelCanvasError::PixmapCreation`] if the pixmap dimensions
    /// are invalid (zero or too large).
    pub fn rasterize(canvas: &PixelCanvas) -> Result<Pixmap, PixelCanvasError> {
        let mut pixmap = Pixmap::new(canvas.width(), canvas.height()).ok_or_else(|| {
            PixelCanvasError::PixmapCreation(format!(
                "failed to create {}x{} pixmap",
                canvas.width(),
                canvas.height()
            ))
        })?;

        Self::rasterize_into_pixmap(canvas, &mut pixmap);
        Ok(pixmap)
    }

    /// Rasterize into an existing pixmap, avoiding allocation.
    ///
    /// The pixmap is cleared and fully redrawn. This is **significantly faster**
    /// for animation loops where the pixmap dimensions don't change between
    /// frames — it avoids the ~640 KB allocation that [`rasterize()`](Self::rasterize)
    /// creates for a 400×400 canvas.
    ///
    /// # Panics
    ///
    /// Panics if the pixmap dimensions don't match the canvas dimensions.
    /// If the canvas may change size between frames, check dimensions first
    /// or use [`rasterize()`](Self::rasterize) which always allocates.
    pub fn rasterize_into(canvas: &PixelCanvas, pixmap: &mut Pixmap) {
        assert_eq!(
            (pixmap.width(), pixmap.height()),
            (canvas.width(), canvas.height()),
            "pixmap dimensions {}×{} must match canvas dimensions {}×{}",
            pixmap.width(),
            pixmap.height(),
            canvas.width(),
            canvas.height()
        );
        Self::rasterize_into_pixmap(canvas, pixmap);
    }

    /// Internal: clear and render into a pixmap (shared by both public methods).
    fn rasterize_into_pixmap(canvas: &PixelCanvas, pixmap: &mut Pixmap) {
        // Clear to transparent
        for byte in pixmap.data_mut() {
            *byte = 0;
        }

        // Fill background
        let bg = canvas.background_color();
        if bg != Color::TRANSPARENT {
            if let Some(color) = bg.to_tiny_skia() {
                pixmap.fill(color);
            }
        }

        // Render each command
        for cmd in canvas.commands() {
            Self::render_command(pixmap, cmd, SkiaTransform::identity());
        }
    }

    #[allow(clippy::too_many_lines)]
    fn render_command(pixmap: &mut Pixmap, cmd: &DrawCommand, parent_transform: SkiaTransform) {
        match cmd {
            DrawCommand::Clear { color } => {
                if let Some(c) = color.to_tiny_skia() {
                    pixmap.fill(c);
                }
            }

            DrawCommand::Circle {
                cx,
                cy,
                radius,
                style,
            } => {
                if let Some(path) = PathBuilder::from_circle(*cx, *cy, *radius) {
                    Self::render_shape(pixmap, &path, style, parent_transform);
                }
            }

            DrawCommand::Rectangle {
                rect,
                corner_radius,
                style,
            } => {
                let skia_rect =
                    tiny_skia::Rect::from_xywh(rect.x, rect.y, rect.width, rect.height);
                if let Some(r) = skia_rect {
                    if *corner_radius > 0.0 {
                        if let Some(path) = Self::build_round_rect(
                            rect.x, rect.y, rect.width, rect.height, *corner_radius,
                        ) {
                            Self::render_shape(pixmap, &path, style, parent_transform);
                        }
                    } else {
                        let path = PathBuilder::from_rect(r);
                        Self::render_shape(pixmap, &path, style, parent_transform);
                    }
                }
            }

            DrawCommand::Line {
                x1,
                y1,
                x2,
                y2,
                stroke,
                anti_alias,
            } => {
                let mut pb = PathBuilder::new();
                pb.move_to(*x1, *y1);
                pb.line_to(*x2, *y2);
                if let Some(path) = pb.finish() {
                    let mut paint = Paint::default();
                    if let Some(c) = stroke.color.to_tiny_skia() {
                        paint.set_color(c);
                    }
                    paint.anti_alias = *anti_alias;

                    let skia_stroke = Self::to_skia_stroke(stroke);
                    pixmap.stroke_path(&path, &paint, &skia_stroke, parent_transform, None);
                }
            }

            DrawCommand::Ellipse {
                cx,
                cy,
                rx,
                ry,
                rotation,
                style,
            } => {
                // Build ellipse as a transformed circle path.
                // tiny-skia has PathBuilder::from_circle, so we scale from
                // a unit circle approach using oval rect.
                let oval = tiny_skia::Rect::from_xywh(
                    cx - rx,
                    cy - ry,
                    rx * 2.0,
                    ry * 2.0,
                );
                if let Some(r) = oval {
                    if let Some(path) = PathBuilder::from_oval(r) {
                        if rotation.abs() > f32::EPSILON {
                            // Apply rotation transform around center
                            let rot = SkiaTransform::from_rotate_at(rotation.to_degrees(), *cx, *cy);
                            let combined = parent_transform.post_concat(rot);
                            Self::render_shape(pixmap, &path, style, combined);
                        } else {
                            Self::render_shape(pixmap, &path, style, parent_transform);
                        }
                    }
                }
            }

            DrawCommand::Path { path, style } => {
                Self::render_shape(pixmap, path.path(), style, parent_transform);
            }

            DrawCommand::Polyline {
                points,
                closed,
                style,
            } => {
                if points.len() >= 2 {
                    let mut pb = PathBuilder::new();
                    pb.move_to(points[0].0, points[0].1);
                    for &(x, y) in &points[1..] {
                        pb.line_to(x, y);
                    }
                    if *closed {
                        pb.close();
                    }
                    if let Some(path) = pb.finish() {
                        Self::render_shape(pixmap, &path, style, parent_transform);
                    }
                }
            }

            DrawCommand::Gradient {
                rect,
                gradient,
                anti_alias,
            } => {
                let skia_rect =
                    tiny_skia::Rect::from_xywh(rect.x, rect.y, rect.width, rect.height);
                if let Some(r) = skia_rect {
                    let path = PathBuilder::from_rect(r);
                    let mut paint = Self::gradient_to_paint(gradient, rect);
                    paint.anti_alias = *anti_alias;
                    pixmap.fill_path(
                        &path,
                        &paint,
                        FillRule::Winding,
                        parent_transform,
                        None,
                    );
                }
            }

            DrawCommand::Arc {
                cx,
                cy,
                radius,
                start_angle,
                sweep_angle,
                style,
            } => {
                if let Some(path) = Self::build_arc_path(*cx, *cy, *radius, *start_angle, *sweep_angle) {
                    Self::render_shape(pixmap, &path, style, parent_transform);
                }
            }

            DrawCommand::Image {
                image,
                x,
                y,
                opacity,
            } => {
                if let Some(src) = tiny_skia::PixmapRef::from_bytes(
                    image.data(),
                    image.width(),
                    image.height(),
                ) {
                    let paint = tiny_skia::PixmapPaint {
                        opacity: *opacity,
                        blend_mode: tiny_skia::BlendMode::SourceOver,
                        quality: tiny_skia::FilterQuality::Bilinear,
                    };
                    let translate = SkiaTransform::from_translate(*x, *y);
                    let combined = parent_transform.post_concat(translate);
                    pixmap.draw_pixmap(0, 0, src, &paint, combined, None);
                }
            }

            #[cfg(feature = "text")]
            DrawCommand::Text {
                text,
                x,
                y,
                font_size,
                color,
                font_data,
            } => {
                Self::render_text(pixmap, text, *x, *y, *font_size, color, font_data, parent_transform);
            }

            DrawCommand::Group {
                commands,
                transform,
                clip,
                opacity,
                blend_mode,
            } => {
                let combined = parent_transform.post_concat(transform.to_tiny_skia());
                let needs_temp = *opacity < 1.0 || clip.is_some() || *blend_mode != crate::scene::style::BlendMode::SrcOver;

                if needs_temp {
                    // Render children to a temporary pixmap
                    let mut temp = Pixmap::new(pixmap.width(), pixmap.height())
                        .expect("temp pixmap same size as canvas");
                    for child in commands {
                        Self::render_command(&mut temp, child, combined);
                    }

                    // Build clip mask if needed
                    let mask = clip.as_ref().and_then(|clip_region| {
                        let mut mask = tiny_skia::Mask::new(pixmap.width(), pixmap.height())?;
                        match clip_region {
                            crate::scene::style::ClipRegion::Rect(rect) => {
                                let clip_rect = tiny_skia::Rect::from_xywh(
                                    rect.x, rect.y, rect.width, rect.height,
                                )?;
                                let clip_path = PathBuilder::from_rect(clip_rect);
                                mask.fill_path(
                                    &clip_path,
                                    FillRule::Winding,
                                    true,
                                    SkiaTransform::identity(),
                                );
                            }
                            crate::scene::style::ClipRegion::Path(path_data) => {
                                mask.fill_path(
                                    path_data.path(),
                                    FillRule::Winding,
                                    true,
                                    SkiaTransform::identity(),
                                );
                            }
                        }
                        Some(mask)
                    });

                    let paint = tiny_skia::PixmapPaint {
                        opacity: *opacity,
                        blend_mode: blend_mode.to_tiny_skia(),
                        quality: tiny_skia::FilterQuality::Nearest,
                    };
                    pixmap.draw_pixmap(
                        0,
                        0,
                        temp.as_ref(),
                        &paint,
                        SkiaTransform::identity(),
                        mask.as_ref(),
                    );
                } else {
                    // Fast path: no compositing needed
                    for child in commands {
                        Self::render_command(pixmap, child, combined);
                    }
                }
            }
        }
    }

    fn render_shape(
        pixmap: &mut Pixmap,
        path: &tiny_skia::Path,
        style: &ShapeStyle,
        transform: SkiaTransform,
    ) {
        // Fill first (if specified)
        if let Some(fill) = &style.fill {
            let mut paint = match fill {
                FillStyle::Solid(color) => {
                    let mut p = Paint::default();
                    if let Some(c) = color.to_tiny_skia() {
                        p.set_color(c);
                    }
                    p
                }
                FillStyle::LinearGradient(grad) | FillStyle::RadialGradient(grad) => {
                    let bounds = path.bounds();
                    let r = crate::scene::style::Rect::new(
                        bounds.left(),
                        bounds.top(),
                        bounds.width(),
                        bounds.height(),
                    );
                    Self::gradient_to_paint(grad, &r)
                }
            };
            paint.anti_alias = style.anti_alias;
            pixmap.fill_path(path, &paint, FillRule::Winding, transform, None);
        }

        // Then stroke (if specified)
        if let Some(stroke_style) = &style.stroke {
            let mut paint = Paint::default();
            if let Some(c) = stroke_style.color.to_tiny_skia() {
                paint.set_color(c);
            }
            paint.anti_alias = style.anti_alias;

            let skia_stroke = Self::to_skia_stroke(stroke_style);
            pixmap.stroke_path(path, &paint, &skia_stroke, transform, None);
        }
    }

    fn to_skia_stroke(style: &StrokeStyle) -> SkiaStroke {
        let line_cap = match style.line_cap {
            LineCap::Butt => SkiaLineCap::Butt,
            LineCap::Round => SkiaLineCap::Round,
            LineCap::Square => SkiaLineCap::Square,
        };
        let line_join = match style.line_join {
            LineJoin::Miter => SkiaLineJoin::Miter,
            LineJoin::Round => SkiaLineJoin::Round,
            LineJoin::Bevel => SkiaLineJoin::Bevel,
        };
        SkiaStroke {
            width: style.width,
            line_cap,
            line_join,
            ..SkiaStroke::default()
        }
    }

    fn gradient_to_paint(
        gradient: &crate::scene::style::GradientDef,
        _bounds: &crate::scene::style::Rect,
    ) -> Paint<'static> {
        let stops: Vec<tiny_skia::GradientStop> = gradient
            .stops
            .iter()
            .filter_map(|s| {
                s.color
                    .to_tiny_skia()
                    .map(|c| tiny_skia::GradientStop::new(s.position, c))
            })
            .collect();

        let mut paint = Paint::default();

        if stops.len() < 2 {
            // Fallback: use first stop color as solid, or transparent
            if let Some(first) = gradient.stops.first() {
                if let Some(c) = first.color.to_tiny_skia() {
                    paint.set_color(c);
                }
            }
            return paint;
        }

        match &gradient.kind {
            GradientKind::Linear { start, end } => {
                if let Some(shader) = tiny_skia::LinearGradient::new(
                    tiny_skia::Point::from_xy(start.x, start.y),
                    tiny_skia::Point::from_xy(end.x, end.y),
                    stops,
                    tiny_skia::SpreadMode::Pad,
                    SkiaTransform::identity(),
                ) {
                    paint.shader = shader;
                }
            }
            GradientKind::Radial { center, radius } => {
                if let Some(shader) = tiny_skia::RadialGradient::new(
                    tiny_skia::Point::from_xy(center.x, center.y),
                    tiny_skia::Point::from_xy(center.x, center.y),
                    *radius,
                    stops,
                    tiny_skia::SpreadMode::Pad,
                    SkiaTransform::identity(),
                ) {
                    paint.shader = shader;
                }
            }
        }

        paint
    }

    /// Build a rounded rectangle path manually.
    #[allow(clippy::many_single_char_names)]
    fn build_round_rect(x: f32, y: f32, w: f32, h: f32, r: f32) -> Option<tiny_skia::Path> {
        // Clamp radius to half the smaller dimension
        let r = r.min(w / 2.0).min(h / 2.0);

        let mut pb = PathBuilder::new();

        // Top edge (left to right)
        pb.move_to(x + r, y);
        pb.line_to(x + w - r, y);
        // Top-right corner
        pb.quad_to(x + w, y, x + w, y + r);
        // Right edge
        pb.line_to(x + w, y + h - r);
        // Bottom-right corner
        pb.quad_to(x + w, y + h, x + w - r, y + h);
        // Bottom edge
        pb.line_to(x + r, y + h);
        // Bottom-left corner
        pb.quad_to(x, y + h, x, y + h - r);
        // Left edge
        pb.line_to(x, y + r);
        // Top-left corner
        pb.quad_to(x, y, x + r, y);
        pb.close();

        pb.finish()
    }

    /// Build an arc path using cubic Bézier approximation.
    ///
    /// Splits the arc into segments of ≤90° and uses the standard
    /// `4/3 * tan(θ/4)` control-point formula for each segment.
    #[allow(clippy::many_single_char_names, clippy::similar_names, clippy::suboptimal_flops, clippy::cast_sign_loss, clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn build_arc_path(
        cx: f32,
        cy: f32,
        radius: f32,
        start_angle: f32,
        sweep_angle: f32,
    ) -> Option<tiny_skia::Path> {
        if sweep_angle.abs() < f32::EPSILON || radius <= 0.0 {
            return None;
        }

        let mut pb = PathBuilder::new();

        // Start point
        let sx = cx + radius * start_angle.cos();
        let sy = cy + radius * start_angle.sin();
        pb.move_to(sx, sy);

        // Split into segments of at most 90 degrees (π/2 radians)
        let max_segment = std::f32::consts::FRAC_PI_2;
        let segments = ((sweep_angle.abs() / max_segment).ceil() as usize).max(1);
        let segment_angle = sweep_angle / segments as f32;

        let mut angle = start_angle;
        for _ in 0..segments {
            let next_angle = angle + segment_angle;
            let half = segment_angle / 2.0;

            // Control point distance: 4/3 * tan(θ/4) * radius
            let alpha = (4.0 / 3.0) * (half / 2.0).tan();

            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let cos_b = next_angle.cos();
            let sin_b = next_angle.sin();

            // Control point 1 (tangent at start of segment)
            let cp1x = cx + radius * (cos_a - alpha * sin_a);
            let cp1y = cy + radius * (sin_a + alpha * cos_a);
            // Control point 2 (tangent at end of segment)
            let cp2x = cx + radius * (cos_b + alpha * sin_b);
            let cp2y = cy + radius * (sin_b - alpha * cos_b);
            // End point
            let ex = cx + radius * cos_b;
            let ey = cy + radius * sin_b;

            pb.cubic_to(cp1x, cp1y, cp2x, cp2y, ex, ey);
            angle = next_angle;
        }

        pb.finish()
    }

    /// Render text by rasterizing each glyph with `fontdue` and painting
    /// the coverage bitmap onto the canvas.
    #[cfg(feature = "text")]
    #[allow(clippy::too_many_arguments, clippy::many_single_char_names, clippy::cast_sign_loss, clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_lossless)]
    fn render_text(
        pixmap: &mut Pixmap,
        text: &str,
        x: f32,
        y: f32,
        font_size: f32,
        color: &crate::scene::style::Color,
        font_data: &FontData,
        _parent_transform: SkiaTransform,
    ) {
        let Ok(font) = fontdue::Font::from_bytes(
            font_data.bytes(),
            fontdue::FontSettings::default(),
        ) else {
            return;
        };

        let r = (color.r * 255.0) as u8;
        let g = (color.g * 255.0) as u8;
        let b = (color.b * 255.0) as u8;
        let a = (color.a * 255.0) as u8;

        let mut cursor_x = x;
        let canvas_w = pixmap.width() as i32;
        let canvas_h = pixmap.height() as i32;

        for ch in text.chars() {
            let (metrics, bitmap) = font.rasterize(ch, font_size);

            if metrics.width > 0 && metrics.height > 0 {
                let gx = cursor_x as i32 + metrics.xmin;
                let gy = y as i32 - metrics.height as i32 - metrics.ymin;

                // Paint each pixel of the glyph coverage bitmap
                let data = pixmap.data_mut();
                for row in 0..metrics.height {
                    for col in 0..metrics.width {
                        let px = gx + col as i32;
                        let py = gy + row as i32;
                        if px < 0 || py < 0 || px >= canvas_w || py >= canvas_h {
                            continue;
                        }
                        let coverage = bitmap[row * metrics.width + col];
                        if coverage == 0 {
                            continue;
                        }

                        let idx = ((py as usize) * (canvas_w as usize) + (px as usize)) * 4;
                        let ca = ((a as u16) * (coverage as u16) / 255) as u8;

                        // Alpha-blend (source-over)
                        let inv_a = 255u16 - ca as u16;
                        data[idx] =
                            ((r as u16 * ca as u16 + data[idx] as u16 * inv_a) / 255) as u8;
                        data[idx + 1] =
                            ((g as u16 * ca as u16 + data[idx + 1] as u16 * inv_a) / 255) as u8;
                        data[idx + 2] =
                            ((b as u16 * ca as u16 + data[idx + 2] as u16 * inv_a) / 255) as u8;
                        data[idx + 3] =
                            ((ca as u16 + data[idx + 3] as u16 * inv_a / 255) as u8).max(data[idx + 3]);
                    }
                }
            }

            cursor_x += metrics.advance_width;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::style::Color;

    #[test]
    fn rasterize_empty_canvas() {
        let canvas = PixelCanvas::new(100, 100);
        let pixmap = Rasterizer::rasterize(&canvas).unwrap();
        assert_eq!(pixmap.width(), 100);
        assert_eq!(pixmap.height(), 100);
    }

    #[test]
    fn rasterize_with_background() {
        let canvas = PixelCanvas::new(10, 10).background(Color::RED);
        let pixmap = Rasterizer::rasterize(&canvas).unwrap();

        // Check that all pixels are red-ish
        let data = pixmap.data();
        // First pixel: R channel should be 255
        assert_eq!(data[0], 255); // R
        assert_eq!(data[1], 0); // G
        assert_eq!(data[2], 0); // B
        assert_eq!(data[3], 255); // A
    }

    #[test]
    fn rasterize_circle() {
        let canvas = PixelCanvas::new(100, 100)
            .circle(50.0, 50.0, 30.0)
            .fill(Color::BLUE)
            .done();

        let pixmap = Rasterizer::rasterize(&canvas).unwrap();

        // Center pixel should be blue
        let idx = (50 * 100 + 50) * 4;
        let data = pixmap.data();
        assert_eq!(data[idx], 0); // R
        assert_eq!(data[idx + 1], 0); // G
        assert_eq!(data[idx + 2], 255); // B
        assert_eq!(data[idx + 3], 255); // A
    }

    #[test]
    fn rasterize_line() {
        let canvas = PixelCanvas::new(100, 100)
            .line(0.0, 50.0, 100.0, 50.0)
            .color(Color::WHITE)
            .width(1.0)
            .anti_alias(false)
            .done();

        let pixmap = Rasterizer::rasterize(&canvas).unwrap();

        // A pixel on the horizontal line should be white
        let idx = (50 * 100 + 50) * 4;
        let data = pixmap.data();
        assert_eq!(data[idx], 255); // R
        assert_eq!(data[idx + 1], 255); // G
        assert_eq!(data[idx + 2], 255); // B
    }

    #[test]
    fn rasterize_zero_size_fails() {
        let canvas = PixelCanvas::new(0, 100);
        let result = Rasterizer::rasterize(&canvas);
        assert!(result.is_err());
    }

    #[test]
    fn rasterize_round_rect() {
        let canvas = PixelCanvas::new(100, 100)
            .rect(10.0, 10.0, 80.0, 60.0)
            .fill(Color::GREEN)
            .corner_radius(10.0)
            .done();

        let pixmap = Rasterizer::rasterize(&canvas).unwrap();

        // Center of the rect should be green
        let idx = (40 * 100 + 50) * 4;
        let data = pixmap.data();
        assert_eq!(data[idx], 0); // R
        assert!(data[idx + 1] > 200); // G (may not be exactly 255 due to anti-aliasing)
        assert_eq!(data[idx + 2], 0); // B
    }
}
