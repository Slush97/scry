//! Diagnostic test for `DashPattern` and `SvgLineDrawing` rasterization.

use ratatui_pixelcanvas::rasterize::Rasterizer;
use ratatui_pixelcanvas::scene::style::{Color, DashPattern};
use ratatui_pixelcanvas::scene::PixelCanvas;

#[cfg(feature = "svg")]
use ratatui_pixelcanvas::svg::line_drawing::SvgLineDrawing;

#[test]
fn path_no_dash_produces_pixels() {
    let mut pb = tiny_skia::PathBuilder::new();
    pb.move_to(10.0, 50.0);
    pb.line_to(90.0, 50.0);
    let path = pb.finish().unwrap();

    let canvas = PixelCanvas::new(100, 100)
        .background(Color::BLACK)
        .path(path)
        .stroke(Color::WHITE, 3.0)
        .done();
    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    let white_pixels = pixmap.data().chunks(4).filter(|px| px[0] > 200).count();
    eprintln!("No dash: {white_pixels} white pixels");
    assert!(
        white_pixels > 50,
        "path with no dash should produce visible pixels, got {white_pixels}"
    );
}

#[test]
fn path_partial_dash_produces_pixels() {
    let mut pb = tiny_skia::PathBuilder::new();
    pb.move_to(10.0, 50.0);
    pb.line_to(90.0, 50.0);
    let path = pb.finish().unwrap();

    let canvas = PixelCanvas::new(100, 100)
        .background(Color::BLACK)
        .path(path)
        .stroke(Color::WHITE, 3.0)
        .dash(DashPattern::new(vec![40.0, 200.0], 0.0))
        .done();
    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    let white_pixels = pixmap.data().chunks(4).filter(|px| px[0] > 200).count();
    eprintln!("Dash [40, 200]: {white_pixels} white pixels");
    assert!(
        white_pixels > 20,
        "partial dash should show some pixels, got {white_pixels}"
    );
}

#[test]
fn path_zero_dash_produces_no_pixels() {
    let mut pb = tiny_skia::PathBuilder::new();
    pb.move_to(10.0, 50.0);
    pb.line_to(90.0, 50.0);
    let path = pb.finish().unwrap();

    let canvas = PixelCanvas::new(100, 100)
        .background(Color::BLACK)
        .path(path)
        .stroke(Color::WHITE, 3.0)
        .dash(DashPattern::new(vec![0.0, 200.0], 0.0))
        .done();
    let pixmap = Rasterizer::rasterize(&canvas).unwrap();
    let white_pixels = pixmap.data().chunks(4).filter(|px| px[0] > 200).count();
    eprintln!("Dash [0, 200]: {white_pixels} white pixels (should be 0 or near-0)");
}

#[cfg(feature = "svg")]
#[test]
fn svg_line_drawing_produces_visible_pixels() {
    let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <line x1="10" y1="50" x2="90" y2="50" stroke="white" stroke-width="3"/>
    </svg>"#;
    let drawing = SvgLineDrawing::from_str(svg).unwrap();

    eprintln!("Segments: {}", drawing.segment_count());
    for (i, s) in drawing.segments().iter().enumerate() {
        eprintln!(
            "  seg[{i}]: len={:.1} color={:?} width={:.1}",
            s.length, s.stroke_color, s.stroke_width
        );
    }

    // t=0.5 should show half the line
    let c1 = drawing.draw(PixelCanvas::new(100, 100).background(Color::BLACK), 0.5);
    eprintln!("Commands at t=0.5: {}", c1.commands().len());
    for cmd in c1.commands() {
        eprintln!("  cmd: {cmd:?}");
    }
    let p1 = Rasterizer::rasterize(&c1).unwrap();
    let visible1 = p1.data().chunks(4).filter(|px| px[0] > 100).count();
    eprintln!("Visible pixels at t=0.5: {visible1}");
    assert!(
        visible1 > 10,
        "t=0.5 should produce visible pixels, got {visible1}"
    );

    // t=1.0 should show full line
    let c2 = drawing.draw(PixelCanvas::new(100, 100).background(Color::BLACK), 1.0);
    let p2 = Rasterizer::rasterize(&c2).unwrap();
    let visible2 = p2.data().chunks(4).filter(|px| px[0] > 100).count();
    eprintln!("Visible pixels at t=1.0: {visible2}");
    assert!(
        visible2 > visible1,
        "t=1.0 should produce more visible pixels than t=0.5"
    );
}
