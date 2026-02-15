//! Fuzz target: Rasterizer robustness.
//!
//! Builds `PixelCanvas` scenes with extreme/pathological coordinates and
//! rasterizes them. Verifies no panics and no out-of-bounds writes occur.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_engine::rasterize::Rasterizer;
use scry_engine::scene::style::Color;
use scry_engine::scene::PixelCanvas;

/// Extract an f32 from fuzz data at a given offset.
fn fuzz_f32(data: &[u8], offset: usize) -> f32 {
    if offset + 4 > data.len() {
        return 0.0;
    }
    let v = f32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]);
    // Clamp to a range that exercises edge cases without causing tiny-skia
    // to allocate massive internal path buffers (OOM) or hit internal assertion
    // panics in line_clipper.rs. Non-finite values (NaN, Inf) are replaced with
    // 0.0 because tiny-skia panics on them internally.
    if v.is_finite() { v.clamp(-10_000.0, 10_000.0) } else { 0.0 }
}

/// Extract a Color from fuzz data at a given offset.
fn fuzz_color(data: &[u8], offset: usize) -> Color {
    if offset + 4 > data.len() {
        return Color::RED;
    }
    Color::from_rgba8(data[offset], data[offset + 1], data[offset + 2], data[offset + 3])
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    // Canvas dimensions — keep small for rasterization speed
    let w = u16::from_le_bytes([data[0], data[1]]).max(1).min(128) as u32;
    let h = u16::from_le_bytes([data[2], data[3]]).max(1).min(128) as u32;

    let num_commands = (data[4] % 8) as usize;

    let mut canvas = PixelCanvas::new(w, h);
    canvas = canvas.background(fuzz_color(data, 5));

    let mut offset = 9;
    for _ in 0..num_commands {
        if offset + 21 > data.len() {
            break;
        }

        let shape_type = data[offset] % 8;
        offset += 1;

        let x = fuzz_f32(data, offset);
        offset += 4;
        let y = fuzz_f32(data, offset);
        offset += 4;
        let p1 = fuzz_f32(data, offset);
        offset += 4;
        let p2 = fuzz_f32(data, offset);
        offset += 4;
        let color = fuzz_color(data, offset);
        offset += 4;

        canvas = match shape_type {
            0 => {
                // Circle — radius can be negative, zero, NaN
                canvas.circle(x, y, p1).fill(color).done()
            }
            1 => {
                // Rectangle — negative dimensions
                canvas.rect(x, y, p1, p2).fill(color).done()
            }
            2 => {
                // Line
                canvas.line(x, y, p1, p2).color(color).width(p1.abs().min(100.0)).done()
            }
            3 => {
                // Ellipse with rotation
                canvas.ellipse(x, y, p1, p2).fill(color).done()
            }
            4 => {
                // Arc with extreme angles
                canvas.arc(x, y, p1, p2, x).fill(color).done()
            }
            5 => {
                // Rounded rect
                canvas
                    .rect(x, y, p1.abs().min(500.0), p2.abs().min(500.0))
                    .corner_radius(x.abs().min(100.0))
                    .fill(color)
                    .done()
            }
            6 => {
                // Polygon with potentially degenerate points
                let pts = vec![
                    (x, y),
                    (p1, p2),
                    (x + p1, y + p2),
                    (p2, p1),
                ];
                canvas.polygon(pts).fill(color).done()
            }
            _ => {
                // Polyline
                let pts = vec![(x, y), (p1, p2)];
                canvas.polyline(pts).stroke(color, 2.0).done()
            }
        };
    }

    // The critical test: rasterize must not panic or write out of bounds
    let result = Rasterizer::rasterize(&canvas);

    // For valid dimensions, rasterization should succeed
    if let Ok(pixmap) = result {
        // Verify pixmap dimensions match
        assert_eq!(pixmap.width(), w);
        assert_eq!(pixmap.height(), h);
    }

    // Also test rasterize_into (the in-place variant used in animation loops)
    if let Some(mut pixmap) = tiny_skia::Pixmap::new(w, h) {
        Rasterizer::rasterize_into(&canvas, &mut pixmap);
        assert_eq!(pixmap.width(), w);
        assert_eq!(pixmap.height(), h);
    }
});
