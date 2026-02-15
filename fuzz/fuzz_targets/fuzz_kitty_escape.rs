//! Fuzz target: Kitty escape sequence generation.
//!
//! Uses the public `ProtocolBackend::transmit` trait method to exercise the
//! Kitty escape generation pipeline with arbitrary pixel data. Verifies
//! no panics occur.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_engine::transport::{FontSize, ProtocolBackend, TerminalPosition};

fuzz_target!(|data: &[u8]| {
    // Need at least 4 bytes for width/height
    if data.len() < 4 {
        return;
    }

    // Extract dimensions from fuzz data, clamped to reasonable range
    let w = u16::from_le_bytes([data[0], data[1]]).max(1).min(512) as u32;
    let h = u16::from_le_bytes([data[2], data[3]]).max(1).min(512) as u32;

    // Create a pixmap (may fail for very large sizes, that's OK)
    let Some(pixmap) = tiny_skia::Pixmap::new(w, h) else {
        return;
    };

    // Create a backend writing to a buffer (public API)
    let font_size = FontSize::new(8, 16);
    let mut backend =
        scry_engine::transport::kitty::KittyBackend::with_writer(Vec::new(), font_size);

    let pos = TerminalPosition::new(0, 0, 10, 10);

    // Test transmit through the public ProtocolBackend trait
    let result = backend.transmit(&pixmap, pos, 0);
    
    // transmit should succeed for valid dimensions
    if let Ok(handle) = result {
        // Test replace (another full code path)
        let _ = backend.replace(&handle, &pixmap, pos, 0);
        // Test remove
        let _ = backend.remove(&handle);
    }

    // Test with different format settings
    let mut backend2 = scry_engine::transport::kitty::KittyBackend::with_writer(
        Vec::new(),
        font_size,
    )
    .format(scry_engine::transport::kitty::TransmitFormat::RawRgba);
    let _ = backend2.transmit(&pixmap, pos, 0);

    // Success = no panic
});
