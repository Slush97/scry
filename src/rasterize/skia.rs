//! Rasterization of scenes into pixel buffers via `tiny-skia`.
//!
//! The rasterizer walks the [`DrawCommand`] list
//! produced by a [`PixelCanvas`] and translates each
//! command into the corresponding `tiny-skia` drawing calls.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use tiny_skia::{
    FillRule, LineCap as SkiaLineCap, LineJoin as SkiaLineJoin, Paint, PathBuilder, Pixmap,
    Stroke as SkiaStroke, Transform as SkiaTransform,
};

use crate::scene::command::DrawCommand;
#[cfg(feature = "text")]
use crate::scene::command::FontData;
use crate::scene::style::{
    Color, FillStyle, GradientDef, GradientKind, LineCap, LineJoin, ShapeStyle, StrokeStyle,
};
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
    fn rasterize_into_pixmap(canvas: &PixelCanvas, pixmap: &mut Pixmap, grad_cache: &mut GradientCache) {
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
                    Self::render_command(pixmap, cmd, SkiaTransform::identity(), &mut pool, grad_cache);
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
                        let origin_rect = crate::scene::style::Rect::new(
                            0.0, 0.0, rect.width, rect.height,
                        );
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
                    // Determine temp pixmap size: use clip rect bounds when
                    // available, otherwise estimate from child command bounds
                    // to avoid a full-canvas allocation.
                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        clippy::cast_precision_loss,
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
                            Self::estimate_group_bounds(
                                commands,
                                pixmap.width(),
                                pixmap.height(),
                            )
                        }
                    };

                    // Reuse a temp pixmap from the pool if it's appropriately sized.
                    let needed_area = (tw as usize) * (th as usize);
                    let mut temp = pool.pop().unwrap_or_else(|| {
                        Pixmap::new(tw, th).expect("temp pixmap for group")
                    });

                    // Right-size: if pooled pixmap is too small OR >4× the
                    // needed area, recreate. Prevents oversized buffers from
                    // lingering and inflating clear costs.
                    let pool_area = (temp.width() as usize) * (temp.height() as usize);
                    if temp.width() < tw || temp.height() < th || pool_area > needed_area * 4 {
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
                            if start >= data.len() { break; }
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
                                let mut mask = tiny_skia::Mask::new(
                                    pixmap.width(),
                                    pixmap.height(),
                                )?;
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

    /// Estimate the bounding box of a group's child commands.
    ///
    /// Returns `(width, height, origin_x, origin_y)` clamped to the parent
    /// canvas dimensions. Used to allocate a bounded temp pixmap instead of
    /// a full-canvas one for groups with blend modes or opacity.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
    )]
    pub(crate) fn estimate_group_bounds(
        commands: &[DrawCommand],
        canvas_w: u32,
        canvas_h: u32,
    ) -> (u32, u32, i32, i32) {
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        for cmd in commands {
            let (x0, y0, x1, y1) = Self::estimate_command_bounds(cmd);
            if x0 < min_x { min_x = x0; }
            if y0 < min_y { min_y = y0; }
            if x1 > max_x { max_x = x1; }
            if y1 > max_y { max_y = y1; }
        }

        // If no commands or no bounds, fall back to full canvas
        if min_x >= max_x || min_y >= max_y {
            return (canvas_w, canvas_h, 0, 0);
        }

        // Add margin for stroke widths and anti-aliasing
        let margin = 8.0;
        min_x = (min_x - margin).max(0.0);
        min_y = (min_y - margin).max(0.0);
        max_x = (max_x + margin).min(canvas_w as f32);
        max_y = (max_y + margin).min(canvas_h as f32);

        let tw = ((max_x - min_x).ceil() as u32).max(1).min(canvas_w);
        let th = ((max_y - min_y).ceil() as u32).max(1).min(canvas_h);
        let origin_col = min_x.floor() as i32;
        let origin_row = min_y.floor() as i32;

        (tw, th, origin_col, origin_row)
    }

    /// Estimate the axis-aligned bounding box of a single draw command.
    /// Returns `(min_x, min_y, max_x, max_y)`.
    #[allow(clippy::cast_precision_loss)]
    fn estimate_command_bounds(cmd: &DrawCommand) -> (f32, f32, f32, f32) {
        match cmd {
            DrawCommand::Circle { cx, cy, radius, style } => {
                let sw = style.stroke.as_ref().map_or(0.0, |s| s.width);
                let r = radius + sw;
                (cx - r, cy - r, cx + r, cy + r)
            }
            DrawCommand::Rectangle { rect, style, .. } => {
                let sw = style.stroke.as_ref().map_or(0.0, |s| s.width);
                (
                    rect.x - sw,
                    rect.y - sw,
                    rect.x + rect.width + sw,
                    rect.y + rect.height + sw,
                )
            }
            DrawCommand::Ellipse { cx, cy, rx, ry, style, .. } => {
                let sw = style.stroke.as_ref().map_or(0.0, |s| s.width);
                let r = rx.max(*ry) + sw;
                (cx - r, cy - r, cx + r, cy + r)
            }
            DrawCommand::Line { x1, y1, x2, y2, stroke, .. } => {
                let sw = stroke.width;
                (
                    x1.min(*x2) - sw,
                    y1.min(*y2) - sw,
                    x1.max(*x2) + sw,
                    y1.max(*y2) + sw,
                )
            }
            DrawCommand::Arc { cx, cy, radius, style, .. } => {
                let sw = style.stroke.as_ref().map_or(0.0, |s| s.width);
                let r = radius + sw;
                (cx - r, cy - r, cx + r, cy + r)
            }
            DrawCommand::Polyline { points, style, .. } => {
                let sw = style.stroke.as_ref().map_or(0.0, |s| s.width);
                let (mut min_x, mut min_y) = (f32::MAX, f32::MAX);
                let (mut max_x, mut max_y) = (f32::MIN, f32::MIN);
                for &(x, y) in points {
                    if x < min_x { min_x = x; }
                    if y < min_y { min_y = y; }
                    if x > max_x { max_x = x; }
                    if y > max_y { max_y = y; }
                }
                (min_x - sw, min_y - sw, max_x + sw, max_y + sw)
            }
            DrawCommand::Path { path, style } => {
                let sw = style.stroke.as_ref().map_or(0.0, |s| s.width);
                let b = path.path().bounds();
                (b.left() - sw, b.top() - sw, b.right() + sw, b.bottom() + sw)
            }
            DrawCommand::Gradient { rect, .. } => {
                (rect.x, rect.y, rect.x + rect.width, rect.y + rect.height)
            }
            DrawCommand::Image { image, x, y, .. } => {
                (*x, *y, x + image.width() as f32, y + image.height() as f32)
            }
            #[cfg(feature = "text")]
            DrawCommand::Text { x, y, font_size, text, .. } => {
                // Rough estimate: each character ~0.6 × font_size wide
                let est_w = text.len() as f32 * font_size * 0.6;
                (*x, y - font_size, x + est_w, *y + font_size * 0.3)
            }
            DrawCommand::Clear { .. } => (f32::MAX, f32::MAX, f32::MIN, f32::MIN),
            DrawCommand::Group { commands: children, clip, .. } => {
                if let Some(crate::scene::style::ClipRegion::Rect(r)) = clip {
                    (r.x, r.y, r.x + r.width, r.y + r.height)
                } else {
                    // Recurse into child commands
                    let mut min_x = f32::MAX;
                    let mut min_y = f32::MAX;
                    let mut max_x = f32::MIN;
                    let mut max_y = f32::MIN;
                    for child in children {
                        let (x0, y0, x1, y1) = Self::estimate_command_bounds(child);
                        if x0 < min_x { min_x = x0; }
                        if y0 < min_y { min_y = y0; }
                        if x1 > max_x { max_x = x1; }
                        if y1 > max_y { max_y = y1; }
                    }
                    (min_x, min_y, max_x, max_y)
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

    /// Compute a cache key for a gradient: hash of (gradient def + rect
    /// width/height). Position is excluded because we render at origin
    /// and blit to the target position.
    fn gradient_cache_key(gradient: &GradientDef, rect: &crate::scene::style::Rect) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        gradient.hash(&mut hasher);
        rect.width.to_bits().hash(&mut hasher);
        rect.height.to_bits().hash(&mut hasher);
        hasher.finish()
    }

    /// Build a gradient `Paint`, using stack-allocated stops to avoid
    /// heap allocation in the hot rasterization loop.
    fn gradient_to_paint(
        gradient: &crate::scene::style::GradientDef,
        _bounds: &crate::scene::style::Rect,
    ) -> Paint<'static> {
        // Stack-allocated array: 8 stops covers virtually all real-world
        // gradients without a heap allocation.
        let mut stops = arrayvec::ArrayVec::<tiny_skia::GradientStop, 8>::new();
        for s in &gradient.stops {
            if let Some(c) = s.color.to_tiny_skia() {
                if stops.is_full() { break; }
                stops.push(tiny_skia::GradientStop::new(s.position, c));
            }
        }

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

        // tiny-skia gradient constructors take Vec<GradientStop>;
        // we convert from our stack array only here.
        let stops_vec: Vec<_> = stops.into_iter().collect();

        match &gradient.kind {
            GradientKind::Linear { start, end } => {
                if let Some(shader) = tiny_skia::LinearGradient::new(
                    tiny_skia::Point::from_xy(start.x, start.y),
                    tiny_skia::Point::from_xy(end.x, end.y),
                    stops_vec,
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
                    stops_vec,
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
    /// via tiny-skia's SIMD-optimized `draw_pixmap`.
    ///
    /// Font objects are cached per-thread by `FontData` pointer identity
    /// to avoid re-parsing TTF/OTF files on every text command.
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
        parent_transform: SkiaTransform,
    ) {
        use std::cell::RefCell;
        use std::collections::HashMap;

        // Thread-local font cache keyed by Arc pointer identity.
        // Avoids re-parsing the entire TTF/OTF file (~1-5 ms) on
        // every DrawCommand::Text using the same font.
        thread_local! {
            static FONT_CACHE: RefCell<HashMap<usize, fontdue::Font>> =
                RefCell::new(HashMap::new());
        }

        let font_key = font_data.arc_ptr();

        let font_ok = FONT_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if !cache.contains_key(&font_key) {
                match fontdue::Font::from_bytes(
                    font_data.bytes(),
                    fontdue::FontSettings::default(),
                ) {
                    Ok(font) => { cache.insert(font_key, font); }
                    Err(_) => return false,
                }
            }
            true
        });
        if !font_ok {
            return;
        }

        let r = (color.r * 255.0) as u8;
        let g = (color.g * 255.0) as u8;
        let b = (color.b * 255.0) as u8;
        let a = (color.a * 255.0) as u8;

        let mut cursor_x = x;

        FONT_CACHE.with(|cache| {
            let cache = cache.borrow();
            let font = cache.get(&font_key).expect("font was just inserted");

            for ch in text.chars() {
                let (metrics, bitmap) = font.rasterize(ch, font_size);

                if metrics.width > 0 && metrics.height > 0 {
                    let gw = metrics.width as u32;
                    let gh = metrics.height as u32;

                    // Build glyph pixmap with the text color × coverage alpha
                    if let Some(mut glyph_pm) = Pixmap::new(gw, gh) {
                        let glyph_data = glyph_pm.data_mut();
                        for (i, &coverage) in bitmap.iter().enumerate() {
                            if coverage == 0 {
                                continue;
                            }
                            let ca = ((a as u16) * (coverage as u16) / 255) as u8;
                            let idx = i * 4;
                            // Premultiplied RGBA (tiny-skia expects this)
                            glyph_data[idx]     = ((r as u16 * ca as u16) / 255) as u8;
                            glyph_data[idx + 1] = ((g as u16 * ca as u16) / 255) as u8;
                            glyph_data[idx + 2] = ((b as u16 * ca as u16) / 255) as u8;
                            glyph_data[idx + 3] = ca;
                        }

                        // Position: baseline-relative
                        let gx = cursor_x as i32 + metrics.xmin;
                        let gy = y as i32 - metrics.height as i32 - metrics.ymin;

                        let paint = tiny_skia::PixmapPaint {
                            opacity: 1.0,
                            blend_mode: tiny_skia::BlendMode::SourceOver,
                            quality: tiny_skia::FilterQuality::Nearest,
                        };

                        // Use draw_pixmap for SIMD-optimized alpha blending.
                        // Also respects parent_transform (rotation, scale, etc.).
                        pixmap.draw_pixmap(
                            gx,
                            gy,
                            glyph_pm.as_ref(),
                            &paint,
                            parent_transform,
                            None,
                        );
                    }
                }

                cursor_x += metrics.advance_width;
            }
        });
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
