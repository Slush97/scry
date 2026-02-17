//! Renders a scene into a native OS window using the `window` backend.
//!
//! ```bash
//! cargo run --example window_demo --features window
//! ```

use scry_engine::rasterize::Rasterizer;
use scry_engine::scene::{Color, PixelCanvas};
use scry_engine::transport::window::run_loop;

fn main() {
    let width = 480;
    let height = 360;

    run_loop(width, height, "scry — window demo", move |backend| {
        let canvas = PixelCanvas::new(width, height)
            .background(Color::from_rgba8(25, 25, 35, 255))
            // Large circle
            .circle(240.0, 180.0, 100.0)
            .fill(Color::from_rgba8(70, 130, 180, 200))
            .done()
            // Overlapping smaller circle
            .circle(300.0, 140.0, 60.0)
            .fill(Color::from_rgba8(220, 80, 60, 180))
            .done()
            // Rectangle
            .rect(50.0, 250.0, 160.0, 80.0)
            .fill(Color::from_rgba8(60, 180, 90, 220))
            .done();

        let pixmap = Rasterizer::rasterize(&canvas).expect("rasterize failed");
        backend.blit(&pixmap).expect("blit failed");
    })
    .expect("window event loop failed");
}
