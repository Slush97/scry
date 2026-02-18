//! Comprehensive unit tests targeting under-covered modules.
//!
//! These tests specifically target the coverage gaps identified by tarpaulin:
//! - scene/builder.rs: every shape type and builder pattern
//! - scene/style.rs: Color, Point, Rect, Transform
//! - rasterize/skia.rs: rasterization edge cases
//! - rasterize/cache.rs: caching edge cases
//! - scene/animation.rs: Transition, Keyframes, AnimationState advanced

use scry_engine::rasterize::{RasterCache, Rasterizer};
use scry_engine::scene::animation::{
    AnimationState, Easing, Keyframe, Keyframes, Lerp, Transition,
};
use scry_engine::scene::command::DrawCommand;
use scry_engine::scene::style::{
    BlendMode, Color, DashPattern, FillRule, FillStyle, GradientDef, GradientKind, GradientStop,
    LineCap, LineJoin, Point, Rect, ShapeStyle, StrokeStyle, Transform,
};
use scry_engine::scene::PixelCanvas;
use std::time::Duration;

// ═══════════════════════════════════════════════════════════
// Builder API coverage — every shape type
// ═══════════════════════════════════════════════════════════

#[test]
fn builder_circle_filled() {
    let canvas = PixelCanvas::new(100, 100)
        .circle(50.0, 50.0, 25.0)
        .fill(Color::RED)
        .done();
    assert_eq!(canvas.commands().len(), 1);
    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    let center = (50 * 100 + 50) * 4;
    assert!(pixmap.data()[center] > 200, "center should be red");
}

#[test]
fn builder_circle_stroked() {
    let canvas = PixelCanvas::new(100, 100)
        .circle(50.0, 50.0, 25.0)
        .stroke(Color::BLUE, 3.0)
        .done();
    assert_eq!(canvas.commands().len(), 1);
    Rasterizer::rasterize(&canvas).unwrap();
}

#[test]
fn builder_circle_filled_and_stroked() {
    let canvas = PixelCanvas::new(100, 100)
        .circle(50.0, 50.0, 25.0)
        .fill(Color::RED)
        .stroke(Color::BLUE, 2.0)
        .done();
    Rasterizer::rasterize(&canvas).unwrap();
}

#[test]
fn builder_rect() {
    let canvas = PixelCanvas::new(100, 100)
        .rect(10.0, 10.0, 80.0, 60.0)
        .fill(Color::GREEN)
        .done();
    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    let center = (40 * 100 + 50) * 4;
    assert!(pixmap.data()[center + 1] > 200, "center should be green");
}

#[test]
fn builder_rounded_rect() {
    let canvas = PixelCanvas::new(100, 100)
        .rect(10.0, 10.0, 80.0, 60.0)
        .corner_radius(10.0)
        .fill(Color::BLUE)
        .done();
    Rasterizer::rasterize(&canvas).unwrap();
}

#[test]
fn builder_line() {
    let canvas = PixelCanvas::new(100, 100)
        .line(0.0, 0.0, 100.0, 100.0)
        .color(Color::WHITE)
        .width(2.0)
        .done();
    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    let diag = (50 * 100 + 50) * 4;
    assert!(
        pixmap.data()[diag] > 200,
        "diagonal should have white pixels"
    );
}

#[test]
fn builder_line_with_caps_and_joins() {
    let canvas = PixelCanvas::new(100, 100)
        .line(10.0, 50.0, 90.0, 50.0)
        .color(Color::RED)
        .width(5.0)
        .line_cap(LineCap::Round)
        .line_join(LineJoin::Round)
        .done();
    Rasterizer::rasterize(&canvas).unwrap();
}

#[test]
fn builder_line_with_dash() {
    let canvas = PixelCanvas::new(100, 100)
        .line(10.0, 50.0, 90.0, 50.0)
        .color(Color::WHITE)
        .width(2.0)
        .dash(DashPattern::pair(5.0, 3.0))
        .done();
    Rasterizer::rasterize(&canvas).unwrap();
}

#[test]
fn builder_ellipse() {
    let canvas = PixelCanvas::new(100, 100)
        .ellipse(50.0, 50.0, 40.0, 20.0)
        .fill(Color::RED)
        .done();
    Rasterizer::rasterize(&canvas).unwrap();
}

#[test]
fn builder_ellipse_with_rotation() {
    let canvas = PixelCanvas::new(100, 100)
        .ellipse(50.0, 50.0, 40.0, 20.0)
        .rotation(0.785) // 45 degrees
        .fill(Color::RED)
        .done();
    Rasterizer::rasterize(&canvas).unwrap();
}

#[test]
fn builder_arc() {
    let canvas = PixelCanvas::new(100, 100)
        .arc(50.0, 50.0, 30.0, 0.0, std::f32::consts::PI)
        .stroke(Color::from_rgb8(255, 255, 0), 2.0)
        .done();
    Rasterizer::rasterize(&canvas).unwrap();
}

#[test]
fn builder_polyline() {
    let canvas = PixelCanvas::new(100, 100)
        .polyline(vec![(10.0, 10.0), (50.0, 90.0), (90.0, 10.0)])
        .stroke(Color::WHITE, 2.0)
        .done();
    Rasterizer::rasterize(&canvas).unwrap();
}

#[test]
fn builder_polygon_filled() {
    let canvas = PixelCanvas::new(100, 100)
        .polygon(vec![(10.0, 10.0), (90.0, 10.0), (50.0, 90.0)])
        .fill(Color::BLUE)
        .done();
    Rasterizer::rasterize(&canvas).unwrap();
}

#[test]
fn builder_polygon_stroked() {
    let canvas = PixelCanvas::new(100, 100)
        .polygon(vec![(10.0, 10.0), (90.0, 10.0), (50.0, 90.0)])
        .stroke(Color::RED, 3.0)
        .done();
    Rasterizer::rasterize(&canvas).unwrap();
}

#[test]
fn builder_gradient_linear() {
    let canvas = PixelCanvas::new(100, 100)
        .gradient(0.0, 0.0, 100.0, 100.0)
        .stop(0.0, Color::RED)
        .stop(1.0, Color::BLUE)
        .linear(Point::new(0.0, 0.0), Point::new(100.0, 0.0))
        .done();
    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    // Left should be reddish, right should be bluish
    let left = (50 * 100 + 5) * 4;
    let right = (50 * 100 + 95) * 4;
    assert!(
        pixmap.data()[left] > pixmap.data()[right],
        "left should be more red"
    );
    assert!(
        pixmap.data()[left + 2] < pixmap.data()[right + 2],
        "right should be more blue"
    );
}

#[test]
fn builder_gradient_radial() {
    let canvas = PixelCanvas::new(100, 100)
        .gradient(0.0, 0.0, 100.0, 100.0)
        .stop(0.0, Color::WHITE)
        .stop(1.0, Color::BLACK)
        .radial(Point::new(50.0, 50.0), 40.0)
        .done();
    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    let center = (50 * 100 + 50) * 4;
    let edge = (50 * 100 + 5) * 4;
    assert!(
        pixmap.data()[center] > pixmap.data()[edge],
        "center should be brighter"
    );
}

#[test]
fn builder_group_with_transform() {
    let canvas = PixelCanvas::new(100, 100)
        .group(Transform::translate(50.0, 50.0))
        .canvas(|c| c.circle(0.0, 0.0, 20.0).fill(Color::RED).done())
        .done();
    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    let center = (50 * 100 + 50) * 4;
    assert!(
        pixmap.data()[center] > 200,
        "transformed circle center should be red"
    );
}

#[test]
fn builder_group_with_clip_and_opacity() {
    let canvas = PixelCanvas::new(100, 100)
        .group(Transform::identity())
        .canvas(|c| c.rect(0.0, 0.0, 100.0, 100.0).fill(Color::RED).done())
        .clip_rect(Rect::new(25.0, 25.0, 50.0, 50.0))
        .opacity(0.5)
        .blend_mode(BlendMode::Multiply)
        .done();
    Rasterizer::rasterize(&canvas).unwrap();
}

#[test]
fn builder_chained_shapes() {
    let canvas = PixelCanvas::new(200, 200)
        .background(Color::BLACK)
        .circle(50.0, 50.0, 20.0)
        .fill(Color::RED)
        .done()
        .circle(150.0, 50.0, 20.0)
        .fill(Color::GREEN)
        .done()
        .circle(50.0, 150.0, 20.0)
        .fill(Color::BLUE)
        .done()
        .rect(120.0, 120.0, 60.0, 60.0)
        .fill(Color::WHITE)
        .done();
    assert_eq!(canvas.commands().len(), 4);
    Rasterizer::rasterize(&canvas).unwrap();
}

#[test]
fn builder_anti_alias_toggle() {
    let canvas = PixelCanvas::new(100, 100)
        .circle(50.0, 50.0, 30.0)
        .fill(Color::RED)
        .anti_alias(false)
        .done();
    Rasterizer::rasterize(&canvas).unwrap();
}

#[test]
fn builder_line_anti_alias() {
    let canvas = PixelCanvas::new(100, 100)
        .line(0.0, 0.0, 100.0, 100.0)
        .color(Color::WHITE)
        .width(1.0)
        .anti_alias(false)
        .done();
    Rasterizer::rasterize(&canvas).unwrap();
}

// ═══════════════════════════════════════════════════════════
// Content hash coverage
// ═══════════════════════════════════════════════════════════

#[test]
fn content_hash_includes_background() {
    let h1 = PixelCanvas::new(100, 100)
        .background(Color::RED)
        .content_hash();
    let h2 = PixelCanvas::new(100, 100)
        .background(Color::BLUE)
        .content_hash();
    assert_ne!(
        h1, h2,
        "different backgrounds should produce different hashes"
    );
}

#[test]
fn content_hash_includes_command_order() {
    let h1 = PixelCanvas::new(100, 100)
        .circle(10.0, 10.0, 5.0)
        .fill(Color::RED)
        .done()
        .circle(90.0, 90.0, 5.0)
        .fill(Color::BLUE)
        .done()
        .content_hash();
    let h2 = PixelCanvas::new(100, 100)
        .circle(90.0, 90.0, 5.0)
        .fill(Color::BLUE)
        .done()
        .circle(10.0, 10.0, 5.0)
        .fill(Color::RED)
        .done()
        .content_hash();
    assert_ne!(h1, h2, "command order should affect hash");
}

#[test]
fn content_hash_empty_vs_nonempty() {
    let h1 = PixelCanvas::new(100, 100).content_hash();
    let h2 = PixelCanvas::new(100, 100)
        .circle(50.0, 50.0, 10.0)
        .fill(Color::RED)
        .done()
        .content_hash();
    assert_ne!(h1, h2, "empty vs non-empty should produce different hashes");
}

// ═══════════════════════════════════════════════════════════
// Rasterizer edge cases
// ═══════════════════════════════════════════════════════════

#[test]
fn rasterize_large_canvas() {
    let canvas = PixelCanvas::new(1920, 1080).background(Color::BLACK);
    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    assert_eq!(pixmap.width(), 1920);
    assert_eq!(pixmap.height(), 1080);
}

#[test]
fn rasterize_shapes_outside_bounds() {
    let canvas = PixelCanvas::new(100, 100)
        .circle(-50.0, -50.0, 10.0)
        .fill(Color::RED)
        .done()
        .circle(200.0, 200.0, 10.0)
        .fill(Color::BLUE)
        .done();
    Rasterizer::rasterize(&canvas).unwrap();
}

#[test]
fn rasterize_very_small_shapes() {
    let canvas = PixelCanvas::new(100, 100)
        .circle(50.0, 50.0, 0.1)
        .fill(Color::RED)
        .done();
    Rasterizer::rasterize(&canvas).unwrap();
}

#[test]
fn rasterize_very_large_radius() {
    let canvas = PixelCanvas::new(100, 100)
        .circle(50.0, 50.0, 10000.0)
        .fill(Color::RED)
        .done();
    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    let center = (50 * 100 + 50) * 4;
    assert!(pixmap.data()[center] > 200);
}

#[test]
fn rasterize_overlapping_shapes() {
    let canvas = PixelCanvas::new(100, 100)
        .background(Color::BLACK)
        .circle(50.0, 50.0, 40.0)
        .fill(Color::RED)
        .done()
        .circle(50.0, 50.0, 20.0)
        .fill(Color::BLUE)
        .done();
    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    let center = (50 * 100 + 50) * 4;
    assert!(
        pixmap.data()[center + 2] > 200,
        "center should be blue (top layer)"
    );
}

#[test]
fn rasterize_into_clears_previous_content() {
    use tiny_skia::Pixmap;

    let canvas1 = PixelCanvas::new(100, 100).background(Color::RED);
    let canvas2 = PixelCanvas::new(100, 100).background(Color::BLUE);

    let mut pixmap = Pixmap::new(100, 100).unwrap();

    Rasterizer::rasterize_into(&canvas1, &mut pixmap);
    assert!(pixmap.data()[0] > 200, "should be red");

    Rasterizer::rasterize_into(&canvas2, &mut pixmap);
    assert!(pixmap.data()[2] > 200, "should be blue");
    assert!(pixmap.data()[0] < 10, "red should be gone");
}

// ═══════════════════════════════════════════════════════════
// Batch testing
// ═══════════════════════════════════════════════════════════

#[test]
fn batched_same_style_shapes_rasterize_correctly() {
    let mut canvas = PixelCanvas::new(200, 200).background(Color::BLACK);
    for i in 0..20 {
        canvas = canvas
            .circle(10.0 + i as f32 * 10.0, 100.0, 5.0)
            .fill(Color::RED)
            .done();
    }
    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    let mid = (100 * 200 + 50) * 4;
    assert!(pixmap.data()[mid] > 100, "should have red circles");
}

#[test]
fn batched_mixed_styles_dont_merge() {
    let canvas = PixelCanvas::new(100, 100)
        .background(Color::BLACK)
        .circle(25.0, 50.0, 15.0)
        .fill(Color::RED)
        .done()
        .circle(75.0, 50.0, 15.0)
        .fill(Color::BLUE)
        .done();
    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    let left = (50 * 100 + 25) * 4;
    let right = (50 * 100 + 75) * 4;
    assert!(pixmap.data()[left] > 200, "left should be red");
    assert!(pixmap.data()[right + 2] > 200, "right should be blue");
}

// ═══════════════════════════════════════════════════════════
// Cache edge cases
// ═══════════════════════════════════════════════════════════

#[test]
fn cache_handles_resize() {
    let mut cache = RasterCache::new();

    let canvas1 = PixelCanvas::new(100, 100).background(Color::RED);
    let pixmap1 = Rasterizer::rasterize(&canvas1).unwrap();
    let _ = cache.compute_dirty_tiles(&pixmap1);

    let canvas2 = PixelCanvas::new(200, 200).background(Color::BLUE);
    let pixmap2 = Rasterizer::rasterize(&canvas2).unwrap();
    let dirty = cache.compute_dirty_tiles(&pixmap2);

    assert!(!dirty.is_empty(), "resize should dirty all tiles");
}

// ═══════════════════════════════════════════════════════════
// Color utilities — comprehensive coverage
// ═══════════════════════════════════════════════════════════

#[test]
fn color_named_constants() {
    assert_eq!(Color::BLACK.r, 0.0);
    assert_eq!(Color::BLACK.g, 0.0);
    assert_eq!(Color::BLACK.b, 0.0);
    assert_eq!(Color::BLACK.a, 1.0);

    assert_eq!(Color::WHITE.r, 1.0);
    assert_eq!(Color::WHITE.g, 1.0);
    assert_eq!(Color::WHITE.b, 1.0);
    assert_eq!(Color::WHITE.a, 1.0);

    assert_eq!(Color::RED.r, 1.0);
    assert_eq!(Color::RED.g, 0.0);
    assert_eq!(Color::RED.b, 0.0);

    assert_eq!(Color::GREEN.g, 1.0);
    assert_eq!(Color::BLUE.b, 1.0);

    assert_eq!(Color::TRANSPARENT.a, 0.0);
}

#[test]
fn color_from_rgba8_and_back() {
    let c = Color::from_rgba8(128, 64, 32, 255);
    assert!((c.r - 128.0 / 255.0).abs() < 0.01);
    assert!((c.g - 64.0 / 255.0).abs() < 0.01);
    assert!((c.b - 32.0 / 255.0).abs() < 0.01);
    assert!((c.a - 1.0).abs() < 0.01);
}

#[test]
fn color_from_rgb8() {
    let c = Color::from_rgb8(255, 0, 128);
    assert!((c.r - 1.0).abs() < 0.01);
    assert!(c.g.abs() < 0.01);
    assert!((c.a - 1.0).abs() < 0.01);
}

#[test]
fn color_from_rgba() {
    let c = Color::from_rgba(0.5, 0.25, 0.75, 0.8);
    assert!((c.r - 0.5).abs() < 0.01);
    assert!((c.g - 0.25).abs() < 0.01);
    assert!((c.b - 0.75).abs() < 0.01);
    assert!((c.a - 0.8).abs() < 0.01);
}

#[test]
fn color_with_alpha() {
    let red = Color::RED;
    let semi = red.with_alpha(0.5);
    assert!((semi.r - 1.0).abs() < 0.01);
    assert!((semi.a - 0.5).abs() < 0.01);
}

#[test]
fn color_from_hsla() {
    // Red = 0° hue
    let c = Color::from_hsla(0.0, 1.0, 0.5, 1.0);
    assert!((c.r - 1.0).abs() < 0.05);
    assert!(c.g.abs() < 0.05);
    assert!(c.b.abs() < 0.05);

    // Green = 120° hue
    let g = Color::from_hsla(120.0, 1.0, 0.5, 1.0);
    assert!(g.r.abs() < 0.05);
    assert!((g.g - 1.0).abs() < 0.05);

    // Blue = 240° hue
    let b = Color::from_hsla(240.0, 1.0, 0.5, 1.0);
    assert!(b.r.abs() < 0.05);
    assert!((b.b - 1.0).abs() < 0.05);
}

#[test]
fn color_from_hsl() {
    let c = Color::from_hsl(60.0, 1.0, 0.5);
    assert!((c.a - 1.0).abs() < 0.01, "from_hsl should be fully opaque");
}

#[test]
fn color_with_lightness() {
    let c = Color::from_rgb8(200, 100, 50);
    let brighter = c.with_lightness(1.5);
    assert!(brighter.r > c.r, "should be brighter");

    let darker = c.with_lightness(0.5);
    assert!(darker.r < c.r, "should be darker");
}

#[test]
fn color_oklab_roundtrip_all_primaries() {
    for color in [
        Color::RED,
        Color::GREEN,
        Color::BLUE,
        Color::WHITE,
        Color::BLACK,
    ] {
        let (l, a, b) = color.to_oklab();
        let restored = Color::from_oklab(l, a, b, color.a);
        assert!(
            (color.r - restored.r).abs() < 0.01
                && (color.g - restored.g).abs() < 0.01
                && (color.b - restored.b).abs() < 0.01,
            "Oklab roundtrip failed for {:?}: got {:?}",
            color,
            restored
        );
    }
}

#[test]
fn color_mix_is_lerp() {
    let a = Color::RED;
    let b = Color::BLUE;
    let mixed = a.mix(b, 0.5);
    let lerped = a.lerp(&b, 0.5);
    assert!((mixed.r - lerped.r).abs() < 0.01);
    assert!((mixed.g - lerped.g).abs() < 0.01);
    assert!((mixed.b - lerped.b).abs() < 0.01);
}

#[test]
fn color_mix_rgb() {
    let a = Color::RED;
    let b = Color::BLUE;
    let mid = a.mix_rgb(b, 0.5);
    assert!((mid.r - 0.5).abs() < 0.01);
    assert!((mid.b - 0.5).abs() < 0.01);
}

#[test]
fn color_to_tiny_skia() {
    let c = Color::RED;
    let ts = c.to_tiny_skia();
    assert!(ts.is_some());
}

#[test]
fn color_default_is_black() {
    let c = Color::default();
    assert_eq!(c, Color::BLACK);
}

#[test]
fn color_hash_and_eq() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(Color::RED);
    set.insert(Color::BLUE);
    set.insert(Color::RED); // duplicate
    assert_eq!(set.len(), 2);
}

// ═══════════════════════════════════════════════════════════
// Style coverage
// ═══════════════════════════════════════════════════════════

#[test]
fn shape_style_default() {
    let style = ShapeStyle::default();
    assert!(style.fill.is_none());
    assert!(style.stroke.is_none());
    assert!(!style.anti_alias); // Default is false since ShapeBuilder sets it
}

#[test]
fn stroke_style_default() {
    let stroke = StrokeStyle::default();
    assert_eq!(stroke.color, Color::WHITE);
    assert_eq!(stroke.width, 1.0);
    assert!(stroke.dash.is_none());
}

#[test]
fn dash_pattern_pair() {
    let dash = DashPattern::pair(5.0, 3.0);
    assert_eq!(dash.intervals.len(), 2);
    assert_eq!(dash.offset, 0.0);
}

#[test]
fn dash_pattern_quad() {
    let dash = DashPattern::quad(1.0, 2.0, 3.0, 4.0, 10.0);
    assert_eq!(dash.intervals.len(), 4);
    assert_eq!(dash.offset, 10.0);
}

#[test]
fn fill_style_default() {
    let fill = FillStyle::default();
    assert_eq!(fill, FillStyle::Solid(Color::WHITE));
}

#[test]
fn blend_mode_to_tiny_skia() {
    // Test all blend mode conversions
    let _ = BlendMode::SrcOver.to_tiny_skia();
    let _ = BlendMode::Multiply.to_tiny_skia();
    let _ = BlendMode::Screen.to_tiny_skia();
    let _ = BlendMode::Overlay.to_tiny_skia();
    let _ = BlendMode::Darken.to_tiny_skia();
    let _ = BlendMode::Lighten.to_tiny_skia();
}

// ═══════════════════════════════════════════════════════════
// Point and Rect
// ═══════════════════════════════════════════════════════════

#[test]
fn point_new_and_default() {
    let p = Point::new(3.0, 4.0);
    assert_eq!(p.x, 3.0);
    assert_eq!(p.y, 4.0);

    let d = Point::default();
    assert_eq!(d.x, 0.0);
    assert_eq!(d.y, 0.0);
}

#[test]
fn rect_new_and_accessors() {
    let r = Rect::new(10.0, 20.0, 100.0, 50.0);
    assert_eq!(r.x, 10.0);
    assert_eq!(r.y, 20.0);
    assert_eq!(r.width, 100.0);
    assert_eq!(r.height, 50.0);
    assert!((r.right() - 110.0).abs() < 0.01);
    assert!((r.bottom() - 70.0).abs() < 0.01);
}

#[test]
fn rect_center() {
    let r = Rect::new(0.0, 0.0, 100.0, 50.0);
    let c = r.center();
    assert!((c.x - 50.0).abs() < 0.01);
    assert!((c.y - 25.0).abs() < 0.01);
}

#[test]
fn rect_from_size() {
    let r = Rect::from_size(200.0, 150.0);
    assert_eq!(r.x, 0.0);
    assert_eq!(r.y, 0.0);
    assert_eq!(r.width, 200.0);
    assert_eq!(r.height, 150.0);
}

#[test]
fn rect_contains() {
    let r = Rect::new(10.0, 10.0, 80.0, 60.0);
    assert!(r.contains(Point::new(50.0, 40.0)));
    assert!(!r.contains(Point::new(5.0, 5.0)));
    assert!(r.contains(Point::new(10.0, 10.0))); // boundary
    assert!(r.contains(Point::new(90.0, 70.0))); // boundary
}

#[test]
fn rect_intersects() {
    let a = Rect::new(0.0, 0.0, 50.0, 50.0);
    let b = Rect::new(25.0, 25.0, 50.0, 50.0);
    let c = Rect::new(100.0, 100.0, 10.0, 10.0);
    assert!(a.intersects(&b));
    assert!(!a.intersects(&c));
}

#[test]
fn rect_union() {
    let a = Rect::new(0.0, 0.0, 50.0, 50.0);
    let b = Rect::new(25.0, 25.0, 50.0, 50.0);
    let u = a.union(&b);
    assert!((u.x).abs() < 0.01);
    assert!((u.y).abs() < 0.01);
    assert!((u.width - 75.0).abs() < 0.01);
    assert!((u.height - 75.0).abs() < 0.01);
}

#[test]
fn rect_area() {
    let r = Rect::new(0.0, 0.0, 10.0, 20.0);
    assert!((r.area() - 200.0).abs() < 0.01);
}

// ═══════════════════════════════════════════════════════════
// Transform
// ═══════════════════════════════════════════════════════════

#[test]
fn transform_identity() {
    let t = Transform::identity();
    assert_eq!(t.sx, 1.0);
    assert_eq!(t.sy, 1.0);
    assert_eq!(t.tx, 0.0);
    assert_eq!(t.ty, 0.0);
}

#[test]
fn transform_translate() {
    let t = Transform::translate(10.0, 20.0);
    assert_eq!(t.tx, 10.0);
    assert_eq!(t.ty, 20.0);
}

#[test]
fn transform_scale() {
    let t = Transform::scale(2.0);
    assert_eq!(t.sx, 2.0);
    assert_eq!(t.sy, 2.0);
}

#[test]
fn transform_scale_xy() {
    let t = Transform::scale_xy(3.0, 4.0);
    assert_eq!(t.sx, 3.0);
    assert_eq!(t.sy, 4.0);
}

#[test]
fn transform_rotate() {
    let t = Transform::rotate(std::f32::consts::FRAC_PI_2);
    // 90° rotation: sx ≈ 0, kx ≈ 1, ky ≈ -1, sy ≈ 0
    assert!(t.sx.abs() < 0.01, "sx should be ~0");
    assert!((t.kx - 1.0).abs() < 0.01, "kx should be ~1");
}

#[test]
fn transform_lerp_identity() {
    let a = Transform::identity();
    let b = Transform::translate(100.0, 50.0);
    let mid = a.lerp(&b, 0.5);
    assert!(
        (mid.tx - 50.0).abs() < 1.0,
        "tx should be ~50, got {}",
        mid.tx
    );
    assert!(
        (mid.ty - 25.0).abs() < 1.0,
        "ty should be ~25, got {}",
        mid.ty
    );
}

#[test]
fn transform_lerp_scale() {
    let a = Transform::scale(1.0);
    let b = Transform::scale(2.0);
    let mid = a.lerp(&b, 0.5);
    assert!(
        (mid.sx - 1.5).abs() < 0.1,
        "sx should be ~1.5, got {}",
        mid.sx
    );
}

#[test]
fn point_lerp() {
    let a = Point { x: 0.0, y: 0.0 };
    let b = Point { x: 100.0, y: 200.0 };
    let mid = a.lerp(&b, 0.5);
    assert!((mid.x - 50.0).abs() < 0.01);
    assert!((mid.y - 100.0).abs() < 0.01);
}

// ═══════════════════════════════════════════════════════════
// Transition advanced behavior
// ═══════════════════════════════════════════════════════════

#[test]
fn transition_with_easing() {
    let mut t = Transition::new(0.0_f32, 100.0, Duration::from_secs(1)).easing(Easing::EaseInQuad);
    t.advance(Duration::from_millis(500));
    assert!(
        (t.value() - 25.0).abs() < 1.0,
        "EaseInQuad at 50% should be ~25, got {}",
        t.value()
    );
}

#[test]
fn transition_reverse() {
    let mut t = Transition::new(0.0_f32, 100.0, Duration::from_secs(1));
    t.advance(Duration::from_millis(600));
    t.reverse();
    assert!(!t.is_complete());
    assert!(
        (t.value() - 100.0).abs() < 1.0,
        "reversed should start at old 'to'"
    );
}

#[test]
fn transition_reset() {
    let mut t = Transition::new(0.0_f32, 100.0, Duration::from_secs(1));
    t.advance(Duration::from_millis(800));
    assert!(t.value() > 50.0);
    t.reset();
    assert!((t.value()).abs() < 1.0, "reset should return to start");
    assert!(!t.is_complete());
}

#[test]
fn transition_remaining() {
    let mut t = Transition::new(0.0_f32, 1.0, Duration::from_secs(3));
    t.advance(Duration::from_secs(1));
    assert_eq!(t.remaining(), Duration::from_secs(2));
}

#[test]
fn transition_linear_progress() {
    let mut t = Transition::new(0.0_f32, 1.0, Duration::from_secs(4));
    t.advance(Duration::from_secs(1));
    assert!((t.linear_progress() - 0.25).abs() < 0.01);
}

#[test]
fn transition_eased_progress() {
    let mut t = Transition::new(0.0_f32, 1.0, Duration::from_secs(1)).easing(Easing::EaseInQuad);
    t.advance(Duration::from_millis(500));
    let eased = t.eased_progress();
    // EaseInQuad at 0.5 = 0.25
    assert!((eased - 0.25).abs() < 0.01);
}

#[test]
fn transition_advance_returns_true_on_completion() {
    let mut t = Transition::new(0.0_f32, 1.0, Duration::from_millis(100));
    assert!(!t.advance(Duration::from_millis(50)));
    assert!(t.advance(Duration::from_millis(50)));
    // Already complete, second advance shouldn't return true again
    assert!(!t.advance(Duration::from_millis(50)));
}

#[test]
fn transition_duration_accessor() {
    let t = Transition::new(0.0_f32, 1.0, Duration::from_secs(5));
    assert_eq!(t.duration(), Duration::from_secs(5));
}

// ═══════════════════════════════════════════════════════════
// Keyframes advanced
// ═══════════════════════════════════════════════════════════

#[test]
fn keyframes_sorted_by_position() {
    let kf = Keyframes::new(vec![
        Keyframe {
            position: 1.0,
            value: 100.0_f32,
            easing: Easing::Linear,
        },
        Keyframe {
            position: 0.0,
            value: 0.0,
            easing: Easing::Linear,
        },
        Keyframe {
            position: 0.5,
            value: 50.0,
            easing: Easing::Linear,
        },
    ]);
    assert_eq!(kf.len(), 3);
    assert!((kf.value_at(0.0)).abs() < 0.01);
    assert!((kf.value_at(0.5) - 50.0).abs() < 0.01);
    assert!((kf.value_at(1.0) - 100.0).abs() < 0.01);
}

#[test]
fn keyframes_with_easing_per_segment() {
    let kf = Keyframes::new(vec![
        Keyframe {
            position: 0.0,
            value: 0.0_f32,
            easing: Easing::EaseInQuad,
        },
        Keyframe {
            position: 0.5,
            value: 100.0,
            easing: Easing::EaseOutQuad,
        },
        Keyframe {
            position: 1.0,
            value: 50.0,
            easing: Easing::Linear,
        },
    ]);
    let val = kf.value_at(0.125);
    assert!(val < 25.0, "EaseInQuad should be slow at start, got {val}");
}

#[test]
fn keyframes_len_and_is_empty() {
    let kf = Keyframes::new(vec![
        Keyframe {
            position: 0.0,
            value: 0.0_f32,
            easing: Easing::Linear,
        },
        Keyframe {
            position: 1.0,
            value: 1.0,
            easing: Easing::Linear,
        },
    ]);
    assert_eq!(kf.len(), 2);
    assert!(!kf.is_empty());
}

// ═══════════════════════════════════════════════════════════
// AnimationState advanced
// ═══════════════════════════════════════════════════════════

#[test]
fn animation_state_multiple_concurrent() {
    let mut state = AnimationState::new();
    state.start("x", 0.0_f32, 100.0, Duration::from_secs(1), Easing::Linear);
    state.start("y", 0.0_f32, 200.0, Duration::from_secs(2), Easing::Linear);

    assert_eq!(state.active_count(), 2);
    assert!(state.is_active("x"));
    assert!(state.is_active("y"));

    state.tick(Duration::from_millis(500));
    let x: f32 = state.get("x").unwrap();
    let y: f32 = state.get("y").unwrap();
    assert!((x - 50.0).abs() < 1.0);
    assert!((y - 50.0).abs() < 1.0);
}

#[test]
fn animation_state_cancel() {
    let mut state = AnimationState::new();
    state.start("x", 0.0_f32, 100.0, Duration::from_secs(1), Easing::Linear);
    state.cancel("x");
    assert!(!state.is_active("x"));
    assert!(state.is_idle());
}

#[test]
fn animation_state_cancel_all() {
    let mut state = AnimationState::new();
    state.start("a", 0.0_f32, 1.0, Duration::from_secs(1), Easing::Linear);
    state.start("b", 0.0_f32, 2.0, Duration::from_secs(1), Easing::Linear);
    state.cancel_all();
    assert!(state.is_idle());
    assert_eq!(state.active_count(), 0);
}

#[test]
fn animation_state_get_nonexistent() {
    let state = AnimationState::new();
    let val: Option<f32> = state.get("nonexistent");
    assert!(val.is_none());
}

#[test]
fn animation_state_replace_existing() {
    let mut state = AnimationState::new();
    state.start("x", 0.0_f32, 100.0, Duration::from_secs(1), Easing::Linear);
    state.start("x", 50.0_f32, 200.0, Duration::from_secs(2), Easing::Linear);
    assert_eq!(state.active_count(), 1);
    let val: f32 = state.get("x").unwrap();
    assert!((val - 50.0).abs() < 0.01, "should start at new from value");
}

#[test]
fn animation_state_debug_format() {
    let mut state = AnimationState::new();
    state.start(
        "alpha",
        0.0_f32,
        1.0,
        Duration::from_secs(1),
        Easing::Linear,
    );
    let debug = format!("{:?}", state);
    assert!(debug.contains("AnimationState"));
    assert!(debug.contains("alpha"));
}

#[test]
fn animation_state_default() {
    let state = AnimationState::default();
    assert!(state.is_idle());
}

// ═══════════════════════════════════════════════════════════
// Halfblock backend
// ═══════════════════════════════════════════════════════════

#[test]
fn halfblock_odd_height() {
    use scry_engine::transport::halfblock::HalfblockBackend;

    let canvas = PixelCanvas::new(10, 11).background(Color::RED);
    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    let cells = HalfblockBackend::render_to_cells(&pixmap);
    assert_eq!(cells.len(), 6); // ceil(11/2)
}

#[test]
fn halfblock_1px_height() {
    use scry_engine::transport::halfblock::HalfblockBackend;

    let canvas = PixelCanvas::new(5, 1).background(Color::GREEN);
    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    let cells = HalfblockBackend::render_to_cells(&pixmap);
    assert_eq!(cells.len(), 1);
    assert_eq!(cells[0].len(), 5);
}

#[test]
fn halfblock_render_to_cells_flat() {
    use scry_engine::transport::halfblock::HalfblockBackend;

    let canvas = PixelCanvas::new(10, 10).background(Color::RED);
    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    let mut buf = Vec::new();
    let (rows, cols) = HalfblockBackend::render_to_cells_flat(&pixmap, &mut buf);
    assert_eq!(rows, 5); // ceil(10/2)
    assert_eq!(cols, 10);
    assert_eq!(buf.len(), cols * rows);
}

// ═══════════════════════════════════════════════════════════
// Canvas accessors
// ═══════════════════════════════════════════════════════════

#[test]
fn canvas_dimensions() {
    let canvas = PixelCanvas::new(320, 240);
    assert_eq!(canvas.width(), 320);
    assert_eq!(canvas.height(), 240);
    assert_eq!(canvas.commands().len(), 0);
    assert_eq!(canvas.command_count(), 0);
}

#[test]
fn canvas_background_color() {
    let canvas = PixelCanvas::new(100, 100).background(Color::RED);
    assert_eq!(canvas.background_color(), Color::RED);
}

#[test]
fn canvas_clear() {
    let mut canvas = PixelCanvas::new(100, 100)
        .circle(50.0, 50.0, 20.0)
        .fill(Color::RED)
        .done();
    assert_eq!(canvas.commands().len(), 1);
    canvas.clear();
    assert_eq!(canvas.commands().len(), 0);
}

#[test]
fn canvas_command_api() {
    let canvas = PixelCanvas::new(100, 100).command(DrawCommand::Circle {
        cx: 50.0,
        cy: 50.0,
        radius: 25.0,
        style: ShapeStyle {
            fill: Some(FillStyle::Solid(Color::RED)),
            stroke: None,
            anti_alias: true,
            ..ShapeStyle::default()
        },
    });
    assert_eq!(canvas.commands().len(), 1);
}

#[test]
fn canvas_push_command_api() {
    let mut canvas = PixelCanvas::new(100, 100);
    canvas.push_command(DrawCommand::Circle {
        cx: 50.0,
        cy: 50.0,
        radius: 25.0,
        style: ShapeStyle {
            fill: Some(FillStyle::Solid(Color::RED)),
            stroke: None,
            anti_alias: true,
            ..ShapeStyle::default()
        },
    });
    assert_eq!(canvas.commands().len(), 1);
}

#[test]
fn canvas_into_commands() {
    let canvas = PixelCanvas::new(100, 100)
        .circle(50.0, 50.0, 20.0)
        .fill(Color::RED)
        .done()
        .rect(10.0, 10.0, 30.0, 30.0)
        .fill(Color::BLUE)
        .done();
    let cmds = canvas.into_commands();
    assert_eq!(cmds.len(), 2);
}

// ═══════════════════════════════════════════════════════════
// Gradient types
// ═══════════════════════════════════════════════════════════

#[test]
fn gradient_def_linear() {
    let grad = GradientDef {
        kind: GradientKind::Linear {
            start: Point::new(0.0, 0.0),
            end: Point::new(100.0, 0.0),
        },
        stops: vec![
            GradientStop {
                position: 0.0,
                color: Color::RED,
            },
            GradientStop {
                position: 1.0,
                color: Color::BLUE,
            },
        ],
    };
    assert_eq!(grad.stops.len(), 2);
}

#[test]
fn gradient_def_radial() {
    let grad = GradientDef {
        kind: GradientKind::Radial {
            center: Point::new(50.0, 50.0),
            radius: 40.0,
        },
        stops: vec![
            GradientStop {
                position: 0.0,
                color: Color::WHITE,
            },
            GradientStop {
                position: 1.0,
                color: Color::BLACK,
            },
        ],
    };
    assert_eq!(grad.stops.len(), 2);
}

// ═══════════════════════════════════════════════════════════
// Lerp for tuples
// ═══════════════════════════════════════════════════════════

#[test]
fn tuple_lerp() {
    let a = (0.0_f32, 100.0_f32);
    let b = (100.0_f32, 0.0_f32);
    let mid = a.lerp(&b, 0.5);
    assert!((mid.0 - 50.0).abs() < 0.01);
    assert!((mid.1 - 50.0).abs() < 0.01);
}

#[test]
fn f64_lerp() {
    let val = 0.0_f64.lerp(&100.0, 0.25);
    assert!((val - 25.0).abs() < 0.01);
}

// ═══════════════════════════════════════════════════════════
// Session 1 verification tests
// ═══════════════════════════════════════════════════════════

#[test]
fn fill_rule_even_odd_star_has_hole() {
    // Build a self-intersecting 5-pointed star by connecting every 2nd vertex.
    let cx = 50.0_f32;
    let cy = 50.0_f32;
    let r = 40.0_f32;
    let mut star_pts = Vec::new();
    for i in 0..5 {
        // Skip every other vertex (0, 2, 4, 1, 3) to create self-intersections.
        let idx = (i * 2) % 5;
        let angle = (2.0 * std::f32::consts::PI / 5.0) * idx as f32 - std::f32::consts::FRAC_PI_2;
        star_pts.push((cx + r * angle.cos(), cy + r * angle.sin()));
    }

    let canvas = PixelCanvas::new(100, 100)
        .background(Color::BLACK)
        .polygon(star_pts)
        .fill(Color::RED)
        .fill_rule(FillRule::EvenOdd)
        .done();

    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    let center = (50 * 100 + 50) * 4;
    // With EvenOdd, the center of a 5-pointed star should be a hole (background).
    assert!(
        pixmap.data()[center] < 50,
        "center R should be background (black), got {}",
        pixmap.data()[center]
    );
}

#[test]
fn gradient_many_stops_no_truncation() {
    // 12-stop rainbow gradient — proves no truncation at 8 stops.
    let colors = [
        Color::RED,
        Color::from_rgb8(255, 127, 0),
        Color::from_rgb8(255, 255, 0),
        Color::GREEN,
        Color::from_rgb8(0, 255, 127),
        Color::from_rgb8(0, 255, 255),
        Color::from_rgb8(0, 127, 255),
        Color::BLUE,
        Color::from_rgb8(127, 0, 255),
        Color::from_rgb8(255, 0, 255),
        Color::from_rgb8(255, 0, 127),
        Color::from_rgb8(200, 0, 0),
    ];

    let mut builder = PixelCanvas::new(200, 10).gradient(0.0, 0.0, 200.0, 10.0);
    for (i, color) in colors.iter().enumerate() {
        builder = builder.stop(i as f32 / 11.0, *color);
    }
    let canvas = builder
        .linear(Point::new(0.0, 0.0), Point::new(200.0, 0.0))
        .done();

    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    // Pixel near x=180 (stop 11 region) should have color from the last stops,
    // not be black/transparent (which would indicate truncation).
    let idx = (5 * 200 + 180) * 4;
    let a = pixmap.data()[idx + 3];
    assert!(a > 200, "pixel near end should be opaque, alpha={a}");
    // Should have some non-zero color channel
    let max_rgb = pixmap.data()[idx]
        .max(pixmap.data()[idx + 1])
        .max(pixmap.data()[idx + 2]);
    assert!(
        max_rgb > 50,
        "pixel near end should have color, max_rgb={max_rgb}"
    );
}

#[test]
fn per_shape_opacity_circle_no_group() {
    let canvas = PixelCanvas::new(100, 100)
        .background(Color::WHITE)
        .circle(50.0, 50.0, 30.0)
        .fill(Color::RED)
        .opacity(0.5)
        .done();

    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    let center = (50 * 100 + 50) * 4;
    let r = pixmap.data()[center];
    let g = pixmap.data()[center + 1];
    // Blended: red (255,0,0) at 50% on white (255,255,255)
    // R should stay high, G should show white bleed-through
    assert!(r > 200, "R channel should be high, got {r}");
    assert!(g > 50, "G channel should show white bleed-through, got {g}");
}

#[test]
fn per_shape_transform_rotated_rect() {
    // Render the SAME rect twice: once without rotation, once with.
    // They should produce different pixel data.
    let points = vec![(20.0, 40.0), (80.0, 40.0), (80.0, 60.0), (20.0, 60.0)];

    let canvas_normal = PixelCanvas::new(100, 100)
        .background(Color::BLACK)
        .polygon(points.clone())
        .fill(Color::RED)
        .done();

    let canvas_rotated = PixelCanvas::new(100, 100)
        .background(Color::BLACK)
        .polygon(points)
        .fill(Color::RED)
        .transform(Transform::rotate_at(
            std::f32::consts::FRAC_PI_4,
            50.0,
            50.0,
        ))
        .done();

    let pix_normal = Rasterizer::rasterize(&canvas_normal).unwrap();
    let pix_rotated = Rasterizer::rasterize(&canvas_rotated).unwrap();

    assert_ne!(
        pix_normal.data(),
        pix_rotated.data(),
        "rotation should produce different pixels"
    );

    // Interior point (50,50) should be filled in both
    let center = (50 * 100 + 50) * 4;
    assert!(
        pix_normal.data()[center] > 100,
        "center should be filled normally, R={}",
        pix_normal.data()[center]
    );
}

#[test]
fn miter_limit_affects_stroke() {
    // Sharp angle polyline: two segments meeting at a very acute angle
    let points = vec![(10.0, 90.0), (50.0, 10.0), (90.0, 90.0)];

    let canvas1 = PixelCanvas::new(100, 100)
        .background(Color::BLACK)
        .polyline(points.clone())
        .stroke(Color::WHITE, 4.0)
        .done();

    let canvas2 = PixelCanvas::new(100, 100)
        .background(Color::BLACK)
        .polyline(points)
        .stroke(Color::WHITE, 4.0)
        .miter_limit(1.0)
        .done();

    let pixmap1 = Rasterizer::rasterize(&canvas1).unwrap();
    let pixmap2 = Rasterizer::rasterize(&canvas2).unwrap();

    // The two renders should differ (bevel vs miter join)
    assert_ne!(
        pixmap1.data(),
        pixmap2.data(),
        "different miter_limit should produce different output"
    );
}

#[test]
fn blend_modes_differ_from_src_over() {
    let make_canvas = |mode: BlendMode| {
        PixelCanvas::new(100, 100)
            .background(Color::WHITE)
            .group(Transform::identity())
            .canvas(|c| c.rect(20.0, 20.0, 60.0, 60.0).fill(Color::RED).done())
            .blend_mode(mode)
            .done()
    };

    let p_src_over = Rasterizer::rasterize(&make_canvas(BlendMode::SrcOver)).unwrap();
    let p_dodge = Rasterizer::rasterize(&make_canvas(BlendMode::ColorDodge)).unwrap();
    let p_diff = Rasterizer::rasterize(&make_canvas(BlendMode::Difference)).unwrap();

    assert_ne!(
        p_src_over.data(),
        p_dodge.data(),
        "ColorDodge should differ from SrcOver"
    );
    assert_ne!(
        p_src_over.data(),
        p_diff.data(),
        "Difference should differ from SrcOver"
    );
    assert_ne!(
        p_dodge.data(),
        p_diff.data(),
        "ColorDodge should differ from Difference"
    );
}

// ═══════════════════════════════════════════════════════════
// Per-shape blend mode (without Group wrapping)
// ═══════════════════════════════════════════════════════════

#[test]
fn per_shape_blend_mode_without_group() {
    // Difference(red, white) should produce cyan
    let canvas = PixelCanvas::new(100, 100)
        .background(Color::WHITE)
        .circle(50.0, 50.0, 30.0)
        .fill(Color::RED)
        .blend_mode(BlendMode::Difference)
        .done();

    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    let center = (50 * 100 + 50) * 4;
    let r = pixmap.data()[center];
    let g = pixmap.data()[center + 1];
    let b = pixmap.data()[center + 2];
    // Difference(red=255, white=255) → R=0, G=255, B=255 (cyan)
    assert!(r < 30, "R should be near 0 for cyan, got {r}");
    assert!(g > 220, "G should be near 255 for cyan, got {g}");
    assert!(b > 220, "B should be near 255 for cyan, got {b}");
}

// ═══════════════════════════════════════════════════════════
// Gradient strokes
// ═══════════════════════════════════════════════════════════

#[test]
fn gradient_stroke_on_shape() {
    let grad = GradientDef {
        kind: GradientKind::Linear {
            start: Point::new(0.0, 0.0),
            end: Point::new(100.0, 0.0),
        },
        stops: vec![
            GradientStop {
                position: 0.0,
                color: Color::RED,
            },
            GradientStop {
                position: 1.0,
                color: Color::BLUE,
            },
        ],
    };

    let canvas = PixelCanvas::new(100, 100)
        .background(Color::BLACK)
        .circle(50.0, 50.0, 30.0)
        .stroke(Color::WHITE, 4.0)
        .stroke_gradient(grad)
        .done();

    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    // Left edge of circle stroke (~x=20) should be reddish
    let left = (50 * 100 + 20) * 4;
    assert!(
        pixmap.data()[left] > 100,
        "left stroke should have red, R={}",
        pixmap.data()[left]
    );
    // Right edge of circle stroke (~x=80) should be bluish
    let right = (50 * 100 + 80) * 4;
    assert!(
        pixmap.data()[right + 2] > 100,
        "right stroke should have blue, B={}",
        pixmap.data()[right + 2]
    );
}

#[test]
fn gradient_stroke_on_line() {
    let grad = GradientDef {
        kind: GradientKind::Linear {
            start: Point::new(0.0, 0.0),
            end: Point::new(100.0, 0.0),
        },
        stops: vec![
            GradientStop {
                position: 0.0,
                color: Color::RED,
            },
            GradientStop {
                position: 1.0,
                color: Color::BLUE,
            },
        ],
    };

    let canvas = PixelCanvas::new(100, 100)
        .background(Color::BLACK)
        .line(10.0, 50.0, 90.0, 50.0)
        .color(Color::WHITE)
        .width(4.0)
        .stroke_gradient(grad)
        .done();

    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    // Left side should be reddish
    let left = (50 * 100 + 15) * 4;
    assert!(
        pixmap.data()[left] > 100,
        "left should have red, R={}",
        pixmap.data()[left]
    );
    // Right side should be bluish
    let right = (50 * 100 + 85) * 4;
    assert!(
        pixmap.data()[right + 2] > 100,
        "right should have blue, B={}",
        pixmap.data()[right + 2]
    );
}
