// SPDX-License-Identifier: MIT OR Apache-2.0
//! Rasterization of scenes into pixel buffers via `tiny-skia`.
//!
//! The rasterizer walks the [`DrawCommand`] list
//! produced by a [`PixelCanvas`] and translates each
//! command into the corresponding `tiny-skia` drawing calls.

mod gradients;
mod shapes;
/// Text rendering via fontdue.
pub mod text;

use std::collections::HashMap;

use tiny_skia::{
    FillRule, LineCap as SkiaLineCap, LineJoin as SkiaLineJoin, Paint, PathBuilder, Pixmap,
    Stroke as SkiaStroke, Transform as SkiaTransform,
};

use crate::scene::command::DrawCommand;
use crate::scene::style::{Color, FillStyle, LineCap, LineJoin, ShapeStyle, StrokeStyle};
use crate::scene::PixelCanvas;
use crate::PixelCanvasError;

/// Cache of pre-rendered gradient pixmaps, keyed by a u64 hash of
/// `(GradientDef, Rect-dimensions)`. Blitting a cached pixmap is ~10×
/// faster than re-evaluating the gradient shader per pixel.
pub(crate) type GradientCache = HashMap<u64, Pixmap>;

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

        let mut gc = GradientCache::new();
        Self::rasterize_into_pixmap(canvas, &mut pixmap, &mut gc);
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
        let mut gc = GradientCache::new();
        Self::rasterize_into_pixmap(canvas, pixmap, &mut gc);
    }

    /// Rasterize into an existing pixmap, using a persistent gradient cache.
    ///
    /// Same as [`rasterize_into`](Self::rasterize_into) but reuses gradient
    /// pixmaps across frames — identical gradients are rendered once and
    /// blitted thereafter (~10× faster for repeated gradients).
    ///
    /// # Panics
    ///
    /// Panics if the pixmap dimensions don't match the canvas dimensions.
    pub fn rasterize_into_cached(
        canvas: &PixelCanvas,
        pixmap: &mut Pixmap,
        grad_cache: &mut GradientCache,
    ) {
        assert_eq!(
            (pixmap.width(), pixmap.height()),
            (canvas.width(), canvas.height()),
            "pixmap dimensions {}×{} must match canvas dimensions {}×{}",
            pixmap.width(),
            pixmap.height(),
            canvas.width(),
            canvas.height()
        );
        Self::rasterize_into_pixmap(canvas, pixmap, grad_cache);
    }

    /// Internal: clear and render into a pixmap (shared by both public methods).
    fn rasterize_into_pixmap(
        canvas: &PixelCanvas,
        pixmap: &mut Pixmap,
        grad_cache: &mut GradientCache,
    ) {
        // Fill background in a single pass (avoids double-write).
        let bg = canvas.background_color();
        if bg == Color::TRANSPARENT {
            pixmap.data_mut().fill(0);
        } else if let Some(color) = bg.to_tiny_skia() {
            pixmap.fill(color);
        } else {
            pixmap.data_mut().fill(0);
        }

        // Pool of reusable pixmaps for Group temp buffers.
        // Created once per rasterize call, shared across all recursive render_command calls.
        let mut pool: Vec<Pixmap> = Vec::new();

        // Batch consecutive same-style commands into compound paths for fewer
        // fill_path/stroke_path calls. O(n) preprocessing, big win for scenes
        // with many same-style primitives (e.g., 39 circles → 1 compound path).
        let batched = crate::rasterize::batch::batch_commands(canvas.commands());

        for op in &batched {
            match op {
                crate::rasterize::batch::BatchedOp::Single(cmd) => {
                    Self::render_command(
                        pixmap,
                        cmd,
                        SkiaTransform::identity(),
                        &mut pool,
                        grad_cache,
                    );
                }
                crate::rasterize::batch::BatchedOp::Compound { path, style } => {
                    Self::render_shape(pixmap, path, style, SkiaTransform::identity());
                }
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    pub(crate) fn render_command(
        pixmap: &mut Pixmap,
        cmd: &DrawCommand,
        parent_transform: SkiaTransform,
        pool: &mut Vec<Pixmap>,
        grad_cache: &mut GradientCache,
    ) {
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
                let skia_rect = tiny_skia::Rect::from_xywh(rect.x, rect.y, rect.width, rect.height);
                if let Some(r) = skia_rect {
                    if *corner_radius > 0.0 {
                        if let Some(path) = Self::build_round_rect(
                            rect.x,
                            rect.y,
                            rect.width,
                            rect.height,
                            *corner_radius,
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
                    let mut paint = if let Some(ref fill_paint) = stroke.paint {
                        match fill_paint {
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
                        }
                    } else {
                        let mut p = Paint::default();
                        if let Some(c) = stroke.color.to_tiny_skia() {
                            p.set_color(c);
                        }
                        p
                    };
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
                let oval = tiny_skia::Rect::from_xywh(cx - rx, cy - ry, rx * 2.0, ry * 2.0);
                if let Some(r) = oval {
                    if let Some(path) = PathBuilder::from_oval(r) {
                        if rotation.abs() > f32::EPSILON {
                            // Apply rotation transform around center
                            let rot =
                                SkiaTransform::from_rotate_at(rotation.to_degrees(), *cx, *cy);
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
                // Cache key: hash of (gradient definition + rect dimensions).
                // Position (rect.x, rect.y) is NOT part of the key because
                // we cache the rendered gradient at (0,0) and blit it.
                let cache_key = Self::gradient_cache_key(gradient, rect);

                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let tw = rect.width.ceil() as u32;
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let th = rect.height.ceil() as u32;

                if tw == 0 || th == 0 {
                    return;
                }

                if let Some(cached) = grad_cache.get(&cache_key) {
                    // Cache hit: blit pre-rendered gradient (~10× faster).
                    let paint = tiny_skia::PixmapPaint {
                        opacity: 1.0,
                        blend_mode: tiny_skia::BlendMode::SourceOver,
                        quality: tiny_skia::FilterQuality::Nearest,
                    };
                    #[allow(clippy::cast_possible_truncation)]
                    pixmap.draw_pixmap(
                        rect.x.floor() as i32,
                        rect.y.floor() as i32,
                        cached.as_ref(),
                        &paint,
                        parent_transform,
                        None,
                    );
                } else {
                    // Cache miss: render gradient into a temp pixmap, cache it,
                    // then blit to the destination.
                    if let Some(mut tile) = Pixmap::new(tw, th) {
                        let origin_rect =
                            crate::scene::style::Rect::new(0.0, 0.0, rect.width, rect.height);
                        let tile_skia_rect =
                            tiny_skia::Rect::from_xywh(0.0, 0.0, rect.width, rect.height);
                        if let Some(tr) = tile_skia_rect {
                            let mut paint = Self::gradient_to_paint(gradient, &origin_rect);
                            paint.anti_alias = *anti_alias;
                            tile.fill_rect(tr, &paint, SkiaTransform::identity(), None);
                        }

                        // Blit to destination
                        let blit_paint = tiny_skia::PixmapPaint {
                            opacity: 1.0,
                            blend_mode: tiny_skia::BlendMode::SourceOver,
                            quality: tiny_skia::FilterQuality::Nearest,
                        };
                        #[allow(clippy::cast_possible_truncation)]
                        pixmap.draw_pixmap(
                            rect.x.floor() as i32,
                            rect.y.floor() as i32,
                            tile.as_ref(),
                            &blit_paint,
                            parent_transform,
                            None,
                        );

                        grad_cache.insert(cache_key, tile);
                    }
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
                if let Some(path) =
                    Self::build_arc_path(*cx, *cy, *radius, *start_angle, *sweep_angle)
                {
                    Self::render_shape(pixmap, &path, style, parent_transform);
                }
            }

            #[cfg(feature = "sdf")]
            DrawCommand::Sdf3D {
                scene,
                rect,
                time,
                render_scale,
            } => {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let w = rect.width.ceil() as u32;
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let h = rect.height.ceil() as u32;
                if w > 0 && h > 0 {
                    let sdf_result = match render_scale {
                        Some(scale) => crate::sdf::SdfRenderer::render_to_pixmap_upscaled(
                            scene.scene(),
                            w,
                            h,
                            *scale,
                            *time,
                        ),
                        None => {
                            crate::sdf::SdfRenderer::render_to_pixmap(scene.scene(), w, h, *time)
                        }
                    };
                    if let Ok(sdf_pixmap) = sdf_result {
                        let paint = tiny_skia::PixmapPaint {
                            opacity: 1.0,
                            blend_mode: tiny_skia::BlendMode::SourceOver,
                            quality: tiny_skia::FilterQuality::Bilinear,
                        };
                        #[allow(clippy::cast_possible_truncation)]
                        pixmap.draw_pixmap(
                            rect.x.floor() as i32,
                            rect.y.floor() as i32,
                            sdf_pixmap.as_ref(),
                            &paint,
                            parent_transform,
                            None,
                        );
                    }
                }
            }

            DrawCommand::Image {
                image,
                x,
                y,
                opacity,
            } => {
                if let Some(src) =
                    tiny_skia::PixmapRef::from_bytes(image.data(), image.width(), image.height())
                {
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
                align,
                outline_color,
                outline_width,
                fill_style,
                shadow,
            } => {
                Self::render_rich_text(
                    pixmap,
                    text,
                    *x,
                    *y,
                    *font_size,
                    color,
                    font_data,
                    parent_transform,
                    *align,
                    outline_color.as_ref(),
                    *outline_width,
                    fill_style.as_ref(),
                    shadow.as_ref(),
                    grad_cache,
                );
            }

            DrawCommand::Group {
                commands,
                transform,
                clip,
                opacity,
                blend_mode,
            } => {
                let combined = parent_transform.post_concat(transform.to_tiny_skia());
                let needs_temp = *opacity < 1.0
                    || clip.is_some()
                    || *blend_mode != crate::scene::style::BlendMode::SrcOver;

                if needs_temp {
                    // Determine temp pixmap size: use clip rect bounds when
                    // available, otherwise estimate from child command bounds
                    // to avoid a full-canvas allocation.
                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        clippy::cast_precision_loss
                    )]
                    let (tw, th, origin_col, origin_row) = match clip {
                        Some(crate::scene::style::ClipRegion::Rect(r)) => {
                            let cw = (r.width.ceil() as u32).max(1).min(pixmap.width());
                            let ch = (r.height.ceil() as u32).max(1).min(pixmap.height());
                            (cw, ch, r.x.floor() as i32, r.y.floor() as i32)
                        }
                        _ => {
                            // Estimate bounding box from child commands to avoid
                            // allocating a full-canvas temp pixmap.
                            Self::estimate_group_bounds(commands, pixmap.width(), pixmap.height())
                        }
                    };

                    // Reuse a temp pixmap from the pool if it's appropriately sized.
                    let needed_area = (tw as usize) * (th as usize);
                    let mut temp = pool
                        .pop()
                        // SAFETY: tw and th are clamped to parent pixmap dims
                        // (which were already validated), so Pixmap::new succeeds.
                        .unwrap_or_else(|| Pixmap::new(tw, th).expect("temp pixmap for group"));

                    // Right-size: if pooled pixmap is too small OR >4× the
                    // needed area, recreate. Prevents oversized buffers from
                    // lingering and inflating clear costs.
                    let pool_area = (temp.width() as usize) * (temp.height() as usize);
                    if temp.width() < tw || temp.height() < th || pool_area > needed_area * 4 {
                        // SAFETY: same bound reasoning as above.
                        temp = Pixmap::new(tw, th).expect("temp pixmap for group");
                    }

                    // Clear only the region we'll actually render into.
                    if temp.width() == tw && temp.height() == th {
                        // Perfect fit: contiguous memset (fastest path).
                        temp.data_mut().fill(0);
                    } else if temp.width() == tw {
                        // Width matches: clear first th rows contiguously.
                        let end = (tw as usize * th as usize * 4).min(temp.data().len());
                        temp.data_mut()[..end].fill(0);
                    } else {
                        // Width mismatch: clear per-row with stride.
                        let row_stride = temp.width() as usize * 4;
                        let row_clear = tw as usize * 4;
                        let data = temp.data_mut();
                        for row in 0..(th as usize) {
                            let start = row * row_stride;
                            let end = (start + row_clear).min(data.len());
                            if start >= data.len() {
                                break;
                            }
                            data[start..end].fill(0);
                        }
                    }

                    // When rendering into a bounded temp, offset child
                    // coordinates so (origin_col, origin_row) maps to (0, 0).
                    #[allow(clippy::cast_precision_loss)]
                    let child_transform = if origin_col != 0 || origin_row != 0 {
                        let offset = SkiaTransform::from_translate(
                            -(origin_col as f32),
                            -(origin_row as f32),
                        );
                        combined.post_concat(offset)
                    } else {
                        combined
                    };

                    for child in commands {
                        Self::render_command(&mut temp, child, child_transform, pool, grad_cache);
                    }

                    // Build clip mask if needed (only for non-rect clips or
                    // when using a full-canvas temp — rect clips are inherently
                    // handled by the bounded temp size).
                    let mask = clip.as_ref().and_then(|clip_region| {
                        match clip_region {
                            crate::scene::style::ClipRegion::Rect(_) => {
                                // Clip is handled by the bounded temp pixmap
                                // dimensions — no mask needed.
                                None
                            }
                            crate::scene::style::ClipRegion::Path(path_data) => {
                                let mut mask =
                                    tiny_skia::Mask::new(pixmap.width(), pixmap.height())?;
                                mask.fill_path(
                                    path_data.path(),
                                    FillRule::Winding,
                                    true,
                                    SkiaTransform::identity(),
                                );
                                Some(mask)
                            }
                        }
                    });

                    let paint = tiny_skia::PixmapPaint {
                        opacity: *opacity,
                        blend_mode: blend_mode.to_tiny_skia(),
                        quality: tiny_skia::FilterQuality::Nearest,
                    };
                    pixmap.draw_pixmap(
                        origin_col,
                        origin_row,
                        temp.as_ref(),
                        &paint,
                        SkiaTransform::identity(),
                        mask.as_ref(),
                    );

                    // Return to pool for reuse
                    pool.push(temp);
                } else {
                    // Fast path: no compositing needed
                    for child in commands {
                        Self::render_command(pixmap, child, combined, pool, grad_cache);
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
        // Compose per-shape transform with parent transform.
        let transform = match &style.transform {
            Some(t) => transform.post_concat(t.to_tiny_skia()),
            None => transform,
        };

        let fill_rule = style.fill_rule.to_tiny_skia();

        // If per-shape opacity < 1.0 or non-default blend mode, render to a
        // temp pixmap and composite.
        let needs_compositing =
            style.opacity < 1.0 || style.blend_mode != crate::scene::style::BlendMode::SrcOver;
        if needs_compositing {
            let bounds = path.bounds();
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let tw = (bounds.width().ceil() as u32 + 2).min(pixmap.width());
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let th = (bounds.height().ceil() as u32 + 2).min(pixmap.height());
            if tw == 0 || th == 0 {
                return;
            }
            if let Some(mut temp) = Pixmap::new(tw, th) {
                #[allow(clippy::cast_precision_loss)]
                let offset_x = bounds.left().floor();
                #[allow(clippy::cast_precision_loss)]
                let offset_y = bounds.top().floor();
                let child_transform =
                    transform.post_concat(SkiaTransform::from_translate(-offset_x, -offset_y));

                Self::render_shape_inner(&mut temp, path, style, child_transform, fill_rule);

                let paint = tiny_skia::PixmapPaint {
                    opacity: style.opacity,
                    blend_mode: style.blend_mode.to_tiny_skia(),
                    quality: tiny_skia::FilterQuality::Nearest,
                };
                #[allow(clippy::cast_possible_truncation)]
                pixmap.draw_pixmap(
                    offset_x as i32,
                    offset_y as i32,
                    temp.as_ref(),
                    &paint,
                    SkiaTransform::identity(),
                    None,
                );
            }
            return;
        }

        Self::render_shape_inner(pixmap, path, style, transform, fill_rule);
    }

    /// Inner render without opacity compositing.
    fn render_shape_inner(
        pixmap: &mut Pixmap,
        path: &tiny_skia::Path,
        style: &ShapeStyle,
        transform: SkiaTransform,
        fill_rule: FillRule,
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
            pixmap.fill_path(path, &paint, fill_rule, transform, None);
        }

        // Then stroke (if specified)
        if let Some(stroke_style) = &style.stroke {
            let mut paint = if let Some(ref fill_paint) = stroke_style.paint {
                match fill_paint {
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
                }
            } else {
                let mut p = Paint::default();
                if let Some(c) = stroke_style.color.to_tiny_skia() {
                    p.set_color(c);
                }
                p
            };
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

        // Convert our DashPattern to tiny_skia's StrokeDash.
        let dash = style
            .dash
            .as_ref()
            .and_then(|dp| tiny_skia::StrokeDash::new(dp.intervals.clone(), dp.offset));

        SkiaStroke {
            width: style.width,
            miter_limit: style.miter_limit,
            line_cap,
            line_join,
            dash,
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
