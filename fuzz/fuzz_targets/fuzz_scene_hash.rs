//! Fuzz target: Scene content hash determinism and robustness.
//!
//! Builds `PixelCanvas` scenes from fuzz data with various drawing commands,
//! then verifies that `content_hash()` is deterministic (same input → same hash)
//! and never panics.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_engine::scene::style::Color;
use scry_engine::scene::PixelCanvas;

/// Extract an f32 from fuzz data at a given offset, returning 0.0 if out of bounds.
fn fuzz_f32(data: &[u8], offset: usize) -> f32 {
    if offset + 4 > data.len() {
        return 0.0;
    }
    f32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]])
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

    // Canvas dimensions from first 4 bytes
    let w = u16::from_le_bytes([data[0], data[1]]).max(1).min(256) as u32;
    let h = u16::from_le_bytes([data[2], data[3]]).max(1).min(256) as u32;

    // Number of commands to add (from byte 4)
    let num_commands = (data[4] % 10) as usize;

    // Build canvas with fuzzed shapes
    let mut canvas = PixelCanvas::new(w, h);

    // Set background from fuzz data
    canvas = canvas.background(fuzz_color(data, 5));

    let mut offset = 9;
    for _ in 0..num_commands {
        if offset + 20 > data.len() {
            break;
        }

        let shape_type = data[offset] % 6;
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
            0 => canvas.circle(x, y, p1.abs()).fill(color).done(),
            1 => canvas.rect(x, y, p1.abs(), p2.abs()).fill(color).done(),
            2 => canvas.line(x, y, p1, p2).color(color).width(2.0).done(),
            3 => canvas
                .ellipse(x, y, p1.abs(), p2.abs())
                .fill(color)
                .done(),
            4 => canvas.arc(x, y, p1.abs(), p2, 1.0).fill(color).done(),
            _ => {
                // Polyline with a few points
                let points = vec![(x, y), (p1, p2), (x + p1, y + p2)];
                canvas.polyline(points).stroke(color, 2.0).done()
            }
        };
    }

    // Hash must be deterministic
    let h1 = canvas.content_hash();
    let h2 = canvas.content_hash();
    assert_eq!(h1, h2, "content_hash must be deterministic");

    // Clone and verify clone has same hash
    let canvas2 = canvas.clone();
    let h3 = canvas2.content_hash();
    assert_eq!(h1, h3, "cloned canvas must have same hash");
});
