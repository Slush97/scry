// SPDX-License-Identifier: MIT OR Apache-2.0
//! Gradient rendering helpers.

use std::hash::{Hash, Hasher};

use tiny_skia::{Paint, Transform as SkiaTransform};

use crate::scene::style::{GradientDef, GradientKind};

use super::Rasterizer;

impl Rasterizer {
    /// Compute a cache key for a gradient: hash of (gradient def + rect
    /// width/height). Position is excluded because we render at origin
    /// and blit to the target position.
    pub(super) fn gradient_cache_key(
        gradient: &GradientDef,
        rect: &crate::scene::style::Rect,
    ) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        gradient.hash(&mut hasher);
        rect.width.to_bits().hash(&mut hasher);
        rect.height.to_bits().hash(&mut hasher);
        hasher.finish()
    }

    /// Build a gradient `Paint` from a gradient definition.
    pub(super) fn gradient_to_paint(
        gradient: &GradientDef,
        _bounds: &crate::scene::style::Rect,
    ) -> Paint<'static> {
        let mut stops = Vec::with_capacity(gradient.stops.len());
        for s in &gradient.stops {
            if let Some(c) = s.color.to_tiny_skia() {
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

        let stops_vec = stops;

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
}
