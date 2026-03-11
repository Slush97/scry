//! Integration test: full scene → rasterize → transport pipeline.
//!
//! This test exercises the complete rendering pipeline end-to-end,
//! verifying that a scene can be built, rasterized, and transmitted
//! through the Kitty backend without errors.

use scry_engine::rasterize::Rasterizer;
use scry_engine::scene::{Color, PixelCanvas};
use scry_engine::transport::backend::{FontSize, ProtocolBackend, TerminalPosition};

#[cfg(feature = "kitty")]
use scry_engine::transport::kitty::KittyBackend;

use scry_engine::transport::halfblock;

// ---------------------------------------------------------------------------
// Kitty pipeline
// ---------------------------------------------------------------------------

#[cfg(feature = "kitty")]
#[test]
fn full_pipeline_kitty_raw_rgba() {
    use std::io::Cursor;

    let canvas = PixelCanvas::new(100, 100)
        .background(Color::BLACK)
        .circle(50.0, 50.0, 30.0)
        .fill(Color::RED)
        .done()
        .rect(10.0, 10.0, 80.0, 80.0)
        .stroke(Color::WHITE, 2.0)
        .done();

    let pixmap = Rasterizer::rasterize(&canvas).expect("rasterization should succeed");
    assert_eq!(pixmap.width(), 100);
    assert_eq!(pixmap.height(), 100);

    // Transmit phase
    let writer = Cursor::new(Vec::new());
    let mut backend = KittyBackend::with_writer(writer, FontSize::default());
    let pos = TerminalPosition::new(0, 0, 10, 10);
    let handle = backend
        .transmit(&pixmap, pos, -1)
        .expect("transmission should succeed");

    // Remove phase
    backend.remove(&handle).expect("remove should succeed");

    // Extract written bytes after all operations
    let output = backend.into_writer().into_inner();
    let output_str = String::from_utf8_lossy(&output);

    // Should contain Kitty protocol escape sequences
    assert!(output_str.contains("\x1b_G"), "should contain APC start");
    assert!(output_str.contains("a=T"), "should contain transmit action");
    assert!(output_str.contains("f=32"), "should use raw RGBA format");
    assert!(output_str.contains("a=d"), "should contain delete action");
}

#[cfg(feature = "kitty")]
#[test]
fn full_pipeline_kitty_png_format() {
    use scry_engine::transport::kitty::TransmitFormat;
    use std::io::Cursor;

    let canvas = PixelCanvas::new(50, 50)
        .background(Color::BLUE)
        .circle(25.0, 25.0, 15.0)
        .fill(Color::WHITE)
        .done();

    let pixmap = Rasterizer::rasterize(&canvas).unwrap();

    let writer = Cursor::new(Vec::new());
    let mut backend =
        KittyBackend::with_writer(writer, FontSize::default()).format(TransmitFormat::Png);
    let pos = TerminalPosition::new(0, 0, 5, 5);
    backend.transmit(&pixmap, pos, 0).unwrap();

    let output = backend.into_writer().into_inner();
    let output_str = String::from_utf8_lossy(&output);
    assert!(output_str.contains("f=100"), "should use PNG format");
}

// ---------------------------------------------------------------------------
// Halfblock pipeline
// ---------------------------------------------------------------------------

#[test]
fn full_pipeline_halfblock() {
    let canvas = PixelCanvas::new(20, 20)
        .background(Color::from_rgb8(255, 0, 0))
        .circle(10.0, 10.0, 5.0)
        .fill(Color::from_rgb8(0, 255, 0))
        .done();

    let pixmap = Rasterizer::rasterize(&canvas).expect("rasterization should succeed");

    let cells = halfblock::render_to_cells(&pixmap);
    // 20px height → 10 halfblock rows
    assert_eq!(cells.len(), 10);
    // 20px width → 20 columns
    assert_eq!(cells[0].len(), 20);

    // Corner pixel should be the background color (red)
    assert_eq!(cells[0][0].fg, (255, 0, 0));
}

// ---------------------------------------------------------------------------
// Content hash stability
// ---------------------------------------------------------------------------

#[test]
fn content_hash_deterministic_across_builds() {
    let build = || {
        PixelCanvas::new(100, 100)
            .background(Color::BLACK)
            .circle(50.0, 50.0, 30.0)
            .fill(Color::RED)
            .done()
            .line(0.0, 0.0, 100.0, 100.0)
            .color(Color::WHITE)
            .width(2.0)
            .done()
            .content_hash()
    };

    assert_eq!(build(), build());
}

#[test]
fn different_scenes_produce_different_hashes() {
    let h1 = PixelCanvas::new(100, 100)
        .circle(50.0, 50.0, 30.0)
        .fill(Color::RED)
        .done()
        .content_hash();

    let h2 = PixelCanvas::new(100, 100)
        .circle(50.0, 50.0, 31.0) // slightly different radius
        .fill(Color::RED)
        .done()
        .content_hash();

    assert_ne!(h1, h2);
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn rasterize_tiny_canvas() {
    let canvas = PixelCanvas::new(1, 1).background(Color::WHITE);
    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    assert_eq!(pixmap.width(), 1);
    assert_eq!(pixmap.height(), 1);
    // White RGBA: 255, 255, 255, 255
    assert_eq!(pixmap.data()[0..4], [255, 255, 255, 255]);
}

#[test]
fn rasterize_zero_size_fails() {
    let canvas = PixelCanvas::new(0, 100);
    assert!(Rasterizer::rasterize(&canvas).is_err());
}

#[test]
fn empty_canvas_rasterizes() {
    // No draw commands, just background
    let canvas = PixelCanvas::new(50, 50).background(Color::from_rgba8(128, 64, 32, 255));
    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    // First pixel should be the background color
    assert_eq!(pixmap.data()[0], 128);
    assert_eq!(pixmap.data()[1], 64);
    assert_eq!(pixmap.data()[2], 32);
    assert_eq!(pixmap.data()[3], 255);
}

// ---------------------------------------------------------------------------
// Phase 4: New primitives
// ---------------------------------------------------------------------------

#[test]
fn ellipse_rasterizes() {
    let canvas = PixelCanvas::new(100, 100)
        .background(Color::BLACK)
        .ellipse(50.0, 50.0, 40.0, 20.0)
        .fill(Color::GREEN)
        .done();

    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    // Center pixel should be green (part of the ellipse)
    let idx = (50 * 100 + 50) * 4;
    assert_eq!(pixmap.data()[idx], 0); // R = 0
    assert_eq!(pixmap.data()[idx + 1], 255); // G = 255 (full green)
    assert_eq!(pixmap.data()[idx + 3], 255); // A = fully opaque
}

#[test]
fn rotated_ellipse_rasterizes() {
    let canvas = PixelCanvas::new(100, 100)
        .background(Color::BLACK)
        .ellipse(50.0, 50.0, 40.0, 10.0)
        .rotation(std::f32::consts::FRAC_PI_4)
        .fill(Color::RED)
        .done();

    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    // Should still produce a valid pixmap (non-zero pixels)
    let non_black = pixmap.data().chunks(4).any(|px| px[0] > 0);
    assert!(non_black, "rotated ellipse should have non-black pixels");
}

#[test]
fn polyline_rasterizes() {
    let canvas = PixelCanvas::new(100, 100)
        .background(Color::BLACK)
        .polyline(vec![(10.0, 10.0), (90.0, 10.0), (90.0, 90.0)])
        .stroke(Color::WHITE, 2.0)
        .done();

    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    // Should have white pixels along top edge
    let top_center_idx = (10 * 100 + 50) * 4; // y=10, x=50
    assert!(
        pixmap.data()[top_center_idx] > 200,
        "should have white pixels on the polyline"
    );
}

#[test]
fn polygon_rasterizes_filled() {
    let canvas = PixelCanvas::new(100, 100)
        .background(Color::BLACK)
        .polygon(vec![(10.0, 10.0), (90.0, 10.0), (50.0, 90.0)])
        .fill(Color::BLUE)
        .done();

    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    // Center should be blue (inside triangle)
    let center_idx = (40 * 100 + 50) * 4;
    assert!(pixmap.data()[center_idx + 2] > 100, "center should be blue");
}

#[test]
fn push_command_mutable_api() {
    let mut canvas = PixelCanvas::new(100, 100).background(Color::BLACK);
    canvas.push_command(scry_engine::scene::command::DrawCommand::Circle {
        cx: 50.0,
        cy: 50.0,
        radius: 25.0,
        style: scry_engine::scene::style::ShapeStyle {
            fill: Some(scry_engine::scene::style::FillStyle::Solid(Color::RED)),
            stroke: None,
            anti_alias: true,
            ..scry_engine::scene::style::ShapeStyle::default()
        },
    });
    assert_eq!(canvas.commands().len(), 1);
    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    let center_idx = (50 * 100 + 50) * 4;
    assert!(pixmap.data()[center_idx] > 200, "center should be red");
}

// ---------------------------------------------------------------------------
// Dirty-tile transmission
// ---------------------------------------------------------------------------

#[test]
fn dirty_tiles_detected_on_pixel_change() {
    use scry_engine::rasterize::{RasterCache, TILE_SIZE};

    let mut cache = RasterCache::new();

    // Frame 1: blue background
    let canvas1 = PixelCanvas::new(128, 128).background(Color::BLUE);
    let pixmap1 = Rasterizer::rasterize(&canvas1).unwrap();

    // First call: everything is dirty (no previous frame)
    let dirty1 = cache.compute_dirty_tiles(&pixmap1);
    let expected_tiles = (128_usize).div_ceil(TILE_SIZE) * (128_usize).div_ceil(TILE_SIZE);
    assert_eq!(dirty1.len(), expected_tiles, "first frame: all tiles dirty");

    // Frame 2: same scene — nothing dirty
    let dirty2 = cache.compute_dirty_tiles(&pixmap1);
    assert!(dirty2.is_empty(), "identical frame: no dirty tiles");

    // Frame 3: change one pixel in top-left tile
    let mut canvas3 = PixelCanvas::new(128, 128).background(Color::BLUE);
    canvas3.push_command(scry_engine::scene::command::DrawCommand::Circle {
        cx: 16.0,
        cy: 16.0,
        radius: 5.0,
        style: scry_engine::scene::style::ShapeStyle {
            fill: Some(scry_engine::scene::style::FillStyle::Solid(Color::RED)),
            stroke: None,
            anti_alias: true,
            ..scry_engine::scene::style::ShapeStyle::default()
        },
    });
    let pixmap3 = Rasterizer::rasterize(&canvas3).unwrap();
    let dirty3 = cache.compute_dirty_tiles(&pixmap3);

    // Only the first tile (0,0) should be dirty
    assert!(!dirty3.is_empty(), "modified frame should have dirty tiles");
    assert!(
        dirty3.len() < expected_tiles,
        "should be fewer than all tiles"
    );
    assert_eq!(dirty3[0].x, 0);
    assert_eq!(dirty3[0].y, 0);
}

#[cfg(feature = "kitty")]
#[test]
fn transmit_tiles_sends_multiple_images() {
    use scry_engine::rasterize::DirtyTile;
    use std::io::Cursor;

    let canvas = PixelCanvas::new(128, 128).background(Color::BLUE);
    let pixmap = Rasterizer::rasterize(&canvas).unwrap();

    let writer = Cursor::new(Vec::new());
    let mut backend = KittyBackend::with_writer(writer, FontSize::default());
    let pos = TerminalPosition::new(0, 0, 20, 20);

    // First: full transmit to establish the image
    let handle = backend.transmit(&pixmap, pos, -1).unwrap();

    // Two dirty tiles
    let tiles = vec![
        DirtyTile {
            x: 0,
            y: 0,
            width: 64,
            height: 64,
        },
        DirtyTile {
            x: 64,
            y: 0,
            width: 64,
            height: 64,
        },
    ];

    let _new_handle = backend
        .transmit_tiles(&handle, &pixmap, pos, -1, &tiles)
        .unwrap();

    let output = backend.into_writer().into_inner();
    let output_str = String::from_utf8_lossy(&output);

    // Should contain multiple transmit actions (one per tile)
    let transmit_count = output_str.matches("a=T").count();
    assert!(
        transmit_count >= 3,
        "expected at least 3 transmit actions (1 full + 2 tiles), got {transmit_count}"
    );

    // Should contain synchronized update markers
    assert!(
        output_str.contains("\x1b[?2026h"),
        "should begin sync update"
    );
    assert!(output_str.contains("\x1b[?2026l"), "should end sync update");
}

#[cfg(feature = "kitty")]
#[test]
fn transmit_tiles_empty_is_noop() {
    use std::io::Cursor;

    let canvas = PixelCanvas::new(64, 64).background(Color::RED);
    let pixmap = Rasterizer::rasterize(&canvas).unwrap();

    let writer = Cursor::new(Vec::new());
    let mut backend = KittyBackend::with_writer(writer, FontSize::default());
    let pos = TerminalPosition::new(0, 0, 5, 5);

    let handle = backend.transmit(&pixmap, pos, -1).unwrap();

    // Empty dirty tiles — should return same handle ID (no-op)
    let new_handle = backend
        .transmit_tiles(&handle, &pixmap, pos, -1, &[])
        .unwrap();

    assert_eq!(new_handle.id(), handle.id(), "should return same image ID");
}
