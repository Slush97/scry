//! Property-based tests for scry-engine.
//!
//! These tests use `proptest` to verify invariants that must hold
//! for all possible inputs, catching edge cases that hand-written
//! tests might miss.

use proptest::prelude::*;

use scry_engine::rasterize::Rasterizer;
use scry_engine::scene::animation::{AnimationState, Easing, Keyframe, Keyframes, Transition};
use scry_engine::scene::{Color, PixelCanvas};

use std::time::Duration;

// ═══════════════════════════════════════════════════════════
// Easing function invariants
// ═══════════════════════════════════════════════════════════

/// All non-overshoot easing functions must satisfy:
/// - ease(0.0) == 0.0 (start at origin)
/// - ease(1.0) == 1.0 (end at target)
/// - output is finite for all inputs in [0, 1]
fn non_overshoot_easings() -> Vec<Easing> {
    vec![
        Easing::Linear,
        Easing::EaseInQuad,
        Easing::EaseOutQuad,
        Easing::EaseInOutQuad,
        Easing::EaseInCubic,
        Easing::EaseOutCubic,
        Easing::EaseInOutCubic,
        Easing::EaseInQuart,
        Easing::EaseOutQuart,
        Easing::EaseInOutQuart,
        Easing::EaseInQuint,
        Easing::EaseOutQuint,
        Easing::EaseInOutQuint,
        Easing::EaseInSine,
        Easing::EaseOutSine,
        Easing::EaseInOutSine,
        Easing::EaseInExpo,
        Easing::EaseOutExpo,
        Easing::EaseInOutExpo,
        Easing::EaseInCirc,
        Easing::EaseOutCirc,
        Easing::EaseInOutCirc,
    ]
}

proptest! {
    #[test]
    fn easing_boundary_zero(easing_idx in 0..22usize) {
        let easings = non_overshoot_easings();
        let easing = &easings[easing_idx];
        let result = easing.ease(0.0);
        prop_assert!(
            (result - 0.0).abs() < 1e-6,
            "ease(0.0) should be 0.0 for {:?}, got {result}",
            easing
        );
    }

    #[test]
    fn easing_boundary_one(easing_idx in 0..22usize) {
        let easings = non_overshoot_easings();
        let easing = &easings[easing_idx];
        let result = easing.ease(1.0);
        prop_assert!(
            (result - 1.0).abs() < 1e-6,
            "ease(1.0) should be 1.0 for {:?}, got {result}",
            easing
        );
    }

    #[test]
    fn easing_output_finite(easing_idx in 0..22usize, t in 0.0f32..=1.0f32) {
        let easings = non_overshoot_easings();
        let easing = &easings[easing_idx];
        let result = easing.ease(t);
        prop_assert!(result.is_finite(), "ease({t}) should be finite for {:?}, got {result}", easing);
    }

    #[test]
    fn easing_clamped_input(easing_idx in 0..22usize, t in -10.0f32..10.0f32) {
        let easings = non_overshoot_easings();
        let easing = &easings[easing_idx];
        // Should not panic with any float input
        let result = easing.ease(t);
        prop_assert!(result.is_finite(), "ease({t}) should be finite, got {result}");
    }
}

// ═══════════════════════════════════════════════════════════
// Lerp invariants
// ═══════════════════════════════════════════════════════════

proptest! {
    #[test]
    fn f32_lerp_at_zero_returns_start(a in -1000.0f32..1000.0f32, b in -1000.0f32..1000.0f32) {
        use scry_engine::scene::animation::Lerp;
        let result = a.lerp(&b, 0.0);
        prop_assert!(
            (result - a).abs() < 1e-4,
            "lerp({a}, {b}, 0.0) should be {a}, got {result}"
        );
    }

    #[test]
    fn f32_lerp_at_one_returns_end(a in -1000.0f32..1000.0f32, b in -1000.0f32..1000.0f32) {
        use scry_engine::scene::animation::Lerp;
        let result = a.lerp(&b, 1.0);
        prop_assert!(
            (result - b).abs() < 1e-4,
            "lerp({a}, {b}, 1.0) should be {b}, got {result}"
        );
    }

    #[test]
    fn f32_lerp_midpoint(a in -1000.0f32..1000.0f32, b in -1000.0f32..1000.0f32) {
        use scry_engine::scene::animation::Lerp;
        let result = a.lerp(&b, 0.5);
        let expected = (a + b) / 2.0;
        prop_assert!(
            (result - expected).abs() < 1e-3,
            "lerp({a}, {b}, 0.5) should be {expected}, got {result}"
        );
    }
}

// ═══════════════════════════════════════════════════════════
// Color lerp invariants
// ═══════════════════════════════════════════════════════════

proptest! {
    #[test]
    fn color_lerp_at_zero_returns_start(
        r in 0u8..=255, g in 0u8..=255, b in 0u8..=255, a in 0u8..=255
    ) {
        use scry_engine::scene::animation::Lerp;
        let start = Color::from_rgba8(r, g, b, a);
        let end = Color::from_rgba8(255 - r, 255 - g, 255 - b, a);
        let result = start.lerp(&end, 0.0);
        // Allow ±2 per channel for Oklab roundtrip
        prop_assert!((result.r - start.r).abs() < 0.01, "red channel mismatch");
        prop_assert!((result.g - start.g).abs() < 0.01, "green channel mismatch");
        prop_assert!((result.b - start.b).abs() < 0.01, "blue channel mismatch");
    }

    #[test]
    fn color_lerp_at_one_returns_end(
        r in 0u8..=255, g in 0u8..=255, b in 0u8..=255, a in 0u8..=255
    ) {
        use scry_engine::scene::animation::Lerp;
        let start = Color::from_rgba8(r, g, b, a);
        let end = Color::from_rgba8(255 - r, 255 - g, 255 - b, a);
        let result = start.lerp(&end, 1.0);
        prop_assert!((result.r - end.r).abs() < 0.01, "red channel mismatch");
        prop_assert!((result.g - end.g).abs() < 0.01, "green channel mismatch");
        prop_assert!((result.b - end.b).abs() < 0.01, "blue channel mismatch");
    }

    #[test]
    fn color_lerp_alpha_linear(
        a1 in 0u8..=255, a2 in 0u8..=255, t in 0.0f32..=1.0f32
    ) {
        use scry_engine::scene::animation::Lerp;
        let start = Color::from_rgba8(128, 128, 128, a1);
        let end = Color::from_rgba8(128, 128, 128, a2);
        let result = start.lerp(&end, t);
        let expected_alpha = start.a + (end.a - start.a) * t;
        prop_assert!(
            (result.a - expected_alpha).abs() < 0.02,
            "alpha should interpolate linearly: expected {expected_alpha}, got {}",
            result.a
        );
    }
}

// ═══════════════════════════════════════════════════════════
// Content hash properties
// ═══════════════════════════════════════════════════════════

proptest! {
    #[test]
    fn content_hash_deterministic(
        w in 1u32..200, h in 1u32..200,
        cx in 0.0f32..200.0, cy in 0.0f32..200.0, r in 1.0f32..100.0
    ) {
        let build = || {
            PixelCanvas::new(w, h)
                .circle(cx, cy, r)
                .fill(Color::RED)
                .done()
                .content_hash()
        };
        prop_assert_eq!(build(), build());
    }

    #[test]
    fn content_hash_sensitive_to_dimensions(
        w1 in 1u32..200, h1 in 1u32..200,
        w2 in 1u32..200, h2 in 1u32..200
    ) {
        prop_assume!(w1 != w2 || h1 != h2);
        let h1_hash = PixelCanvas::new(w1, h1)
            .circle(50.0, 50.0, 20.0)
            .fill(Color::RED)
            .done()
            .content_hash();
        let h2_hash = PixelCanvas::new(w2, h2)
            .circle(50.0, 50.0, 20.0)
            .fill(Color::RED)
            .done()
            .content_hash();
        prop_assert_ne!(h1_hash, h2_hash);
    }

    #[test]
    fn content_hash_sensitive_to_radius(
        r1 in 1.0f32..100.0, r2 in 1.0f32..100.0
    ) {
        prop_assume!((r1 - r2).abs() > 0.001);
        let h1 = PixelCanvas::new(100, 100)
            .circle(50.0, 50.0, r1)
            .fill(Color::RED)
            .done()
            .content_hash();
        let h2 = PixelCanvas::new(100, 100)
            .circle(50.0, 50.0, r2)
            .fill(Color::RED)
            .done()
            .content_hash();
        prop_assert_ne!(h1, h2);
    }

    #[test]
    fn content_hash_sensitive_to_color(
        r in 0u8..=254
    ) {
        let h1 = PixelCanvas::new(100, 100)
            .circle(50.0, 50.0, 20.0)
            .fill(Color::from_rgb8(r, 0, 0))
            .done()
            .content_hash();
        let h2 = PixelCanvas::new(100, 100)
            .circle(50.0, 50.0, 20.0)
            .fill(Color::from_rgb8(r + 1, 0, 0))
            .done()
            .content_hash();
        prop_assert_ne!(h1, h2);
    }
}

// ═══════════════════════════════════════════════════════════
// Rasterization safety properties
// ═══════════════════════════════════════════════════════════

proptest! {
    #[test]
    fn rasterize_never_panics(
        w in 1u32..500, h in 1u32..500,
        cx in -500.0f32..500.0, cy in -500.0f32..500.0,
        r in 0.1f32..500.0
    ) {
        let canvas = PixelCanvas::new(w, h)
            .background(Color::BLACK)
            .circle(cx, cy, r)
            .fill(Color::RED)
            .done();
        // Should never panic, even with out-of-bounds coordinates
        let result = Rasterizer::rasterize(&canvas);
        prop_assert!(result.is_ok());
    }

    #[test]
    fn rasterize_dimensions_match(w in 1u32..500, h in 1u32..500) {
        let canvas = PixelCanvas::new(w, h).background(Color::WHITE);
        let pixmap = Rasterizer::rasterize(&canvas).unwrap();
        prop_assert_eq!(pixmap.width(), w);
        prop_assert_eq!(pixmap.height(), h);
    }

    #[test]
    fn rasterize_data_length_correct(w in 1u32..500, h in 1u32..500) {
        let canvas = PixelCanvas::new(w, h).background(Color::WHITE);
        let pixmap = Rasterizer::rasterize(&canvas).unwrap();
        prop_assert_eq!(pixmap.data().len(), (w * h * 4) as usize);
    }

    #[test]
    fn rasterize_background_fills_entire_pixmap(
        r in 0u8..=255, g in 0u8..=255, b in 0u8..=255
    ) {
        let color = Color::from_rgb8(r, g, b);
        let canvas = PixelCanvas::new(10, 10).background(color);
        let pixmap = Rasterizer::rasterize(&canvas).unwrap();

        // Every pixel should match the background
        for pixel in pixmap.data().chunks_exact(4) {
            prop_assert_eq!(pixel[0], r, "red channel");
            prop_assert_eq!(pixel[1], g, "green channel");
            prop_assert_eq!(pixel[2], b, "blue channel");
            prop_assert_eq!(pixel[3], 255, "alpha channel");
        }
    }
}

// ═══════════════════════════════════════════════════════════
// Transition properties
// ═══════════════════════════════════════════════════════════

proptest! {
    #[test]
    fn transition_starts_at_from_value(from in -1000.0f32..1000.0f32, to in -1000.0f32..1000.0f32) {
        let t = Transition::new(from, to, Duration::from_secs(1));
        prop_assert!(
            (t.value() - from).abs() < 1e-4,
            "transition should start at {from}, got {}",
            t.value()
        );
    }

    #[test]
    fn transition_ends_at_to_value(from in -1000.0f32..1000.0f32, to in -1000.0f32..1000.0f32) {
        let mut t = Transition::new(from, to, Duration::from_secs(1));
        t.advance(Duration::from_secs(2)); // overshoot
        prop_assert!(
            (t.value() - to).abs() < 1e-4,
            "transition should end at {to}, got {}",
            t.value()
        );
        prop_assert!(t.is_complete());
    }

    #[test]
    fn transition_linear_midpoint(from in -100.0f32..100.0f32, to in -100.0f32..100.0f32) {
        let mut t = Transition::new(from, to, Duration::from_secs(2));
        t.advance(Duration::from_secs(1)); // exactly halfway
        let expected = (from + to) / 2.0;
        prop_assert!(
            (t.value() - expected).abs() < 1e-3,
            "midpoint should be {expected}, got {}",
            t.value()
        );
    }
}

// ═══════════════════════════════════════════════════════════
// Animation state properties
// ═══════════════════════════════════════════════════════════

proptest! {
    #[test]
    fn animation_state_starts_idle(
        _from in -100.0f32..100.0f32, _to in -100.0f32..100.0f32
    ) {
        let state = AnimationState::new();
        prop_assert!(state.is_idle(), "new animation state should be idle");
    }

    #[test]
    fn animation_state_completes_after_duration(
        from in -100.0f32..100.0f32, to in -100.0f32..100.0f32,
        duration_ms in 100u64..5000
    ) {
        let mut state = AnimationState::new();
        state.start("test", from, to, Duration::from_millis(duration_ms), Easing::Linear);
        prop_assert!(!state.is_idle(), "should not be idle after start");

        // Advance past the duration
        state.tick(Duration::from_millis(duration_ms + 100));

        prop_assert!(state.is_idle(), "should be idle after completion");
        let val: Option<f32> = state.get("test");
        // Animation is complete and removed, so it should be None
        // But the final value was reached before removal
        // AnimationState removes completed animations on tick
        prop_assert!(val.is_none(), "completed animation should be cleaned up");
    }
}

// ═══════════════════════════════════════════════════════════
// Keyframe properties
// ═══════════════════════════════════════════════════════════

proptest! {
    #[test]
    fn keyframes_first_value_at_zero(
        v0 in -100.0f32..100.0, v1 in -100.0f32..100.0, v2 in -100.0f32..100.0
    ) {
        let kf = Keyframes::new(vec![
            Keyframe { position: 0.0, value: v0, easing: Easing::Linear },
            Keyframe { position: 0.5, value: v1, easing: Easing::Linear },
            Keyframe { position: 1.0, value: v2, easing: Easing::Linear },
        ]);
        let result = kf.value_at(0.0);
        prop_assert!(
            (result - v0).abs() < 1e-4,
            "keyframes(0.0) should be {v0}, got {result}"
        );
    }

    #[test]
    fn keyframes_last_value_at_one(
        v0 in -100.0f32..100.0, v1 in -100.0f32..100.0, v2 in -100.0f32..100.0
    ) {
        let kf = Keyframes::new(vec![
            Keyframe { position: 0.0, value: v0, easing: Easing::Linear },
            Keyframe { position: 0.5, value: v1, easing: Easing::Linear },
            Keyframe { position: 1.0, value: v2, easing: Easing::Linear },
        ]);
        let result = kf.value_at(1.0);
        prop_assert!(
            (result - v2).abs() < 1e-4,
            "keyframes(1.0) should be {v2}, got {result}"
        );
    }
}

// ═══════════════════════════════════════════════════════════
// Dirty-tile detection properties
// ═══════════════════════════════════════════════════════════

#[test]
fn dirty_tiles_second_identical_frame_is_clean() {
    use scry_engine::rasterize::RasterCache;

    let mut cache = RasterCache::new();
    let canvas = PixelCanvas::new(200, 200).background(Color::BLUE);
    let pixmap = Rasterizer::rasterize(&canvas).unwrap();

    // Frame 1: everything dirty
    let dirty1 = cache.compute_dirty_tiles(&pixmap);
    assert!(!dirty1.is_empty());

    // Frame 2: identical → nothing dirty
    let dirty2 = cache.compute_dirty_tiles(&pixmap);
    assert!(
        dirty2.is_empty(),
        "identical frame should produce zero dirty tiles"
    );
}

#[test]
fn dirty_tiles_tracks_changed_regions() {
    use scry_engine::rasterize::RasterCache;

    let mut cache = RasterCache::new();

    // Frame 1: solid blue
    let canvas1 = PixelCanvas::new(256, 256).background(Color::BLUE);
    let pixmap1 = Rasterizer::rasterize(&canvas1).unwrap();
    let _ = cache.compute_dirty_tiles(&pixmap1);

    // Frame 2: blue + red circle in top-left corner
    let canvas2 = PixelCanvas::new(256, 256)
        .background(Color::BLUE)
        .circle(32.0, 32.0, 20.0)
        .fill(Color::RED)
        .done();
    let pixmap2 = Rasterizer::rasterize(&canvas2).unwrap();
    let dirty = cache.compute_dirty_tiles(&pixmap2);

    // Should have dirty tiles, but not all tiles
    assert!(!dirty.is_empty(), "changed frame should have dirty tiles");
    let total_tiles = (256 / 64) * (256 / 64); // 16 tiles
    assert!(
        dirty.len() < total_tiles,
        "only changed region should be dirty, got {} of {total_tiles}",
        dirty.len()
    );
}

// ═══════════════════════════════════════════════════════════
// Kitty backend properties
// ═══════════════════════════════════════════════════════════

#[cfg(feature = "kitty")]
mod kitty_tests {
    use super::*;
    use scry_engine::transport::backend::{FontSize, ProtocolBackend, TerminalPosition};
    use scry_engine::transport::kitty::{KittyBackend, TransmitFormat};
    use std::io::Cursor;

    #[test]
    fn kitty_output_contains_valid_escapes() {
        let canvas = PixelCanvas::new(64, 64).background(Color::RED);
        let pixmap = Rasterizer::rasterize(&canvas).unwrap();

        let writer = Cursor::new(Vec::new());
        let mut backend = KittyBackend::with_writer(writer, FontSize::default());
        let pos = TerminalPosition::new(0, 0, 8, 8);
        backend.transmit(&pixmap, pos, 0).unwrap();

        let output = backend.into_writer().into_inner();
        let output_str = String::from_utf8_lossy(&output);

        // Every APC must be properly terminated
        let apc_starts = output_str.matches("\x1b_G").count();
        let apc_ends = output_str.matches("\x1b\\").count();
        assert_eq!(
            apc_starts, apc_ends,
            "mismatched APC start/end: {apc_starts} starts, {apc_ends} ends"
        );
    }

    #[test]
    fn kitty_zlib_format_specified() {
        let canvas = PixelCanvas::new(64, 64).background(Color::BLUE);
        let pixmap = Rasterizer::rasterize(&canvas).unwrap();

        let writer = Cursor::new(Vec::new());
        let mut backend =
            KittyBackend::with_writer(writer, FontSize::default()).format(TransmitFormat::ZlibRgba);
        let pos = TerminalPosition::new(0, 0, 8, 8);
        backend.transmit(&pixmap, pos, 0).unwrap();

        let output = backend.into_writer().into_inner();
        let output_str = String::from_utf8_lossy(&output);
        assert!(output_str.contains("o=z"), "zlib mode should specify o=z");
        assert!(output_str.contains("f=32"), "should use f=32 format");
    }

    #[test]
    fn kitty_replace_reuses_image_id() {
        let canvas = PixelCanvas::new(64, 64).background(Color::RED);
        let pixmap = Rasterizer::rasterize(&canvas).unwrap();

        let writer = Cursor::new(Vec::new());
        let mut backend = KittyBackend::with_writer(writer, FontSize::default());
        let pos = TerminalPosition::new(0, 0, 8, 8);

        let handle1 = backend.transmit(&pixmap, pos, 0).unwrap();
        let handle2 = backend.replace(&handle1, &pixmap, pos, 0).unwrap();

        assert_eq!(
            handle1.id(),
            handle2.id(),
            "replace should reuse the same image ID"
        );
    }

    #[test]
    fn kitty_clear_all_emits_delete() {
        let canvas = PixelCanvas::new(32, 32).background(Color::GREEN);
        let pixmap = Rasterizer::rasterize(&canvas).unwrap();

        let writer = Cursor::new(Vec::new());
        let mut backend = KittyBackend::with_writer(writer, FontSize::default());
        let pos = TerminalPosition::new(0, 0, 4, 4);

        backend.transmit(&pixmap, pos, 0).unwrap();
        backend.clear_all().unwrap();

        let output = backend.into_writer().into_inner();
        let output_str = String::from_utf8_lossy(&output);
        assert!(
            output_str.contains("a=d"),
            "clear_all should emit delete action"
        );
    }

    #[test]
    fn kitty_synchronized_update_brackets() {
        let canvas = PixelCanvas::new(128, 128).background(Color::WHITE);
        let pixmap = Rasterizer::rasterize(&canvas).unwrap();

        let writer = Cursor::new(Vec::new());
        let mut backend = KittyBackend::with_writer(writer, FontSize::default());
        let pos = TerminalPosition::new(0, 0, 16, 16);

        let handle = backend.transmit(&pixmap, pos, 0).unwrap();
        backend.replace(&handle, &pixmap, pos, 0).unwrap();

        let output = backend.into_writer().into_inner();
        let output_str = String::from_utf8_lossy(&output);

        // Replace should use synchronized updates
        assert!(
            output_str.contains("\x1b[?2026h"),
            "should begin synchronized update"
        );
        assert!(
            output_str.contains("\x1b[?2026l"),
            "should end synchronized update"
        );
    }

    proptest! {
        #[test]
        fn kitty_various_sizes_never_panic(
            w in 1u32..300, h in 1u32..300
        ) {
            let canvas = PixelCanvas::new(w, h).background(Color::RED);
            let pixmap = Rasterizer::rasterize(&canvas).unwrap();

            let writer = Cursor::new(Vec::new());
            let mut backend = KittyBackend::with_writer(writer, FontSize::default());
            let pos = TerminalPosition::new(0, 0, 20, 20);
            let result = backend.transmit(&pixmap, pos, 0);
            prop_assert!(result.is_ok());
        }
    }
}

// ═══════════════════════════════════════════════════════════
// Rasterizer buffer reuse properties
// ═══════════════════════════════════════════════════════════

#[test]
fn rasterize_into_produces_same_output_as_rasterize() {
    use tiny_skia::Pixmap;

    let canvas = PixelCanvas::new(100, 100)
        .background(Color::from_rgb8(64, 128, 255))
        .circle(50.0, 50.0, 30.0)
        .fill(Color::RED)
        .done()
        .rect(10.0, 10.0, 40.0, 30.0)
        .fill(Color::GREEN)
        .done();

    let pixmap_alloc = Rasterizer::rasterize(&canvas).unwrap();

    let mut pixmap_reuse = Pixmap::new(100, 100).unwrap();
    Rasterizer::rasterize_into(&canvas, &mut pixmap_reuse);

    // Pixel-for-pixel identical
    assert_eq!(
        pixmap_alloc.data(),
        pixmap_reuse.data(),
        "rasterize() and rasterize_into() should produce identical output"
    );
}

// ═══════════════════════════════════════════════════════════
// Color conversion properties
// ═══════════════════════════════════════════════════════════

proptest! {
    #[test]
    fn color_from_rgba8_roundtrip(r in 0u8..=255, g in 0u8..=255, b in 0u8..=255, a in 0u8..=255) {
        let color = Color::from_rgba8(r, g, b, a);
        let r2 = (color.r * 255.0).round() as u8;
        let g2 = (color.g * 255.0).round() as u8;
        let b2 = (color.b * 255.0).round() as u8;
        let a2 = (color.a * 255.0).round() as u8;
        prop_assert_eq!(r, r2, "red roundtrip failed");
        prop_assert_eq!(g, g2, "green roundtrip failed");
        prop_assert_eq!(b, b2, "blue roundtrip failed");
        prop_assert_eq!(a, a2, "alpha roundtrip failed");
    }
}
