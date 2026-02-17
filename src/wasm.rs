// SPDX-License-Identifier: MIT OR Apache-2.0
//! WebAssembly bridge for rendering [`PixelCanvas`] scenes to HTML5 `<canvas>`.
//!
//! This module is only available when the `wasm` feature is enabled. It
//! provides [`WasmCanvas`], a JS-accessible handle around a [`PixelCanvas`],
//! and [`render_rgba_to_canvas`] (WASM-only), a free function that blits raw
//! RGBA pixel data onto an HTML5 canvas element.
//!
//! # Example (JavaScript)
//!
//! ```js
//! import init, { WasmCanvas, render_rgba_to_canvas } from './scry_engine.js';
//!
//! await init();
//!
//! const canvas = new WasmCanvas(400, 300);
//! console.log(`${canvas.width()}x${canvas.height()}`);
//!
//! // Get rasterized pixels and blit to an HTML <canvas>
//! canvas.render_to_canvas("my-canvas-id");
//! ```

use wasm_bindgen::prelude::*;

use crate::rasterize::Rasterizer;
use crate::scene::style::Color;
use crate::scene::PixelCanvas;

// ---------------------------------------------------------------------------
// WasmCanvas — JS-accessible handle wrapping PixelCanvas
// ---------------------------------------------------------------------------

/// A JS-accessible wrapper around [`PixelCanvas`].
///
/// `WasmCanvas` owns a `PixelCanvas` and provides methods to configure the
/// scene, rasterize it, and blit the result to an HTML5 `<canvas>` element.
#[wasm_bindgen]
pub struct WasmCanvas {
    canvas: PixelCanvas,
}

#[wasm_bindgen]
impl WasmCanvas {
    /// Create a new canvas with the given pixel dimensions.
    ///
    /// The canvas starts empty with a transparent background.
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            canvas: PixelCanvas::new(width, height),
        }
    }

    /// Get the canvas width in pixels.
    pub fn width(&self) -> u32 {
        self.canvas.width()
    }

    /// Get the canvas height in pixels.
    pub fn height(&self) -> u32 {
        self.canvas.height()
    }

    /// Set the background color (RGBA components, 0–255).
    pub fn set_background(&mut self, r: u8, g: u8, b: u8, a: u8) {
        // Rebuild canvas with the new background. PixelCanvas uses a fluent
        // builder, so we reconstruct preserving existing commands.
        let mut new_canvas = PixelCanvas::new(self.canvas.width(), self.canvas.height())
            .background(Color::from_rgba8(r, g, b, a));
        for cmd in self.canvas.commands() {
            new_canvas.push_command(cmd.clone());
        }
        self.canvas = new_canvas;
    }

    /// Add a circle to the scene.
    #[allow(clippy::too_many_arguments)]
    pub fn add_circle(&mut self, cx: f32, cy: f32, radius: f32, r: u8, g: u8, b: u8, a: u8) {
        let canvas = std::mem::replace(
            &mut self.canvas,
            PixelCanvas::new(1, 1), // placeholder
        );
        self.canvas = canvas
            .circle(cx, cy, radius)
            .fill(Color::from_rgba8(r, g, b, a))
            .done();
    }

    /// Add a filled rectangle to the scene.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rect(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        r: u8,
        g: u8,
        b: u8,
        a: u8,
    ) {
        let canvas = std::mem::replace(&mut self.canvas, PixelCanvas::new(1, 1));
        self.canvas = canvas
            .rect(x, y, width, height)
            .fill(Color::from_rgba8(r, g, b, a))
            .done();
    }

    /// Get the rasterized RGBA pixel data as a byte array.
    ///
    /// Returns a `Vec<u8>` of length `width * height * 4` in straight-alpha
    /// RGBA order, suitable for creating an `ImageData` in JavaScript.
    pub fn pixels(&self) -> Vec<u8> {
        rasterize_to_straight_rgba(&self.canvas)
    }

    /// Render the scene to an HTML5 `<canvas>` element by its DOM id.
    ///
    /// This rasterizes the scene, creates an `ImageData`, and draws it onto
    /// the 2D rendering context of the specified canvas element.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the canvas element is not found, the
    /// rendering context cannot be obtained, or the `ImageData` cannot be
    /// created.
    #[cfg(target_arch = "wasm32")]
    pub fn render_to_canvas(&self, canvas_id: &str) -> Result<(), JsValue> {
        let pixels = self.pixels();
        render_rgba_to_canvas_inner(
            &pixels,
            self.canvas.width(),
            self.canvas.height(),
            canvas_id,
        )
    }

    /// Clear all draw commands, keeping the canvas dimensions.
    pub fn clear(&mut self) {
        self.canvas.clear();
    }

    /// Return the number of draw commands in the scene.
    pub fn command_count(&self) -> usize {
        self.canvas.command_count()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Rasterize a `PixelCanvas` and convert from premultiplied to straight alpha.
fn rasterize_to_straight_rgba(canvas: &PixelCanvas) -> Vec<u8> {
    match Rasterizer::rasterize(canvas) {
        Ok(pixmap) => {
            // tiny-skia stores premultiplied RGBA; convert to straight
            // alpha for web ImageData compatibility.
            let data = pixmap.data();
            let mut out = Vec::with_capacity(data.len());
            for chunk in data.chunks_exact(4) {
                let (pr, pg, pb, a) = (chunk[0], chunk[1], chunk[2], chunk[3]);
                if a == 0 {
                    out.extend_from_slice(&[0, 0, 0, 0]);
                } else if a == 255 {
                    out.extend_from_slice(&[pr, pg, pb, 255]);
                } else {
                    // Un-premultiply: C = C_pre * 255 / A
                    let af = f32::from(a);
                    let r = (f32::from(pr) * 255.0 / af) as u8;
                    let g = (f32::from(pg) * 255.0 / af) as u8;
                    let b = (f32::from(pb) * 255.0 / af) as u8;
                    out.extend_from_slice(&[r, g, b, a]);
                }
            }
            out
        }
        Err(_) => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Free functions (WASM-only — require web-sys)
// ---------------------------------------------------------------------------

/// Render pre-computed RGBA pixel data to an HTML5 `<canvas>` element.
///
/// `rgba_data` must be exactly `width * height * 4` bytes in straight-alpha
/// RGBA order. This is the format produced by [`WasmCanvas::pixels`] and by
/// `scry-chart`'s `render_to_rgba`.
///
/// # Errors
///
/// Returns a `JsValue` error if the canvas element is not found or the pixel
/// data cannot be drawn.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn render_rgba_to_canvas(
    rgba_data: &[u8],
    width: u32,
    height: u32,
    canvas_id: &str,
) -> Result<(), JsValue> {
    render_rgba_to_canvas_inner(rgba_data, width, height, canvas_id)
}

/// Shared implementation for blitting RGBA data to an HTML canvas.
#[cfg(target_arch = "wasm32")]
fn render_rgba_to_canvas_inner(
    rgba_data: &[u8],
    width: u32,
    height: u32,
    canvas_id: &str,
) -> Result<(), JsValue> {
    let expected_len = (width as usize) * (height as usize) * 4;
    if rgba_data.len() != expected_len {
        return Err(JsValue::from_str(&format!(
            "pixel data length {} does not match {}x{}x4 = {}",
            rgba_data.len(),
            width,
            height,
            expected_len
        )));
    }

    // Obtain the document and locate the target <canvas>.
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no global window"))?;
    let document = window
        .document()
        .ok_or_else(|| JsValue::from_str("no document"))?;
    let element = document
        .get_element_by_id(canvas_id)
        .ok_or_else(|| JsValue::from_str(&format!("no element with id '{canvas_id}'")))?;

    let html_canvas: web_sys::HtmlCanvasElement = element
        .dyn_into()
        .map_err(|_| JsValue::from_str("element is not a <canvas>"))?;

    // Ensure the canvas element dimensions match our pixel data.
    html_canvas.set_width(width);
    html_canvas.set_height(height);

    let ctx = html_canvas
        .get_context("2d")?
        .ok_or_else(|| JsValue::from_str("failed to get 2d context"))?
        .dyn_into::<web_sys::CanvasRenderingContext2d>()?;

    // Create a Clamped<&[u8]> view and build ImageData.
    let clamped = wasm_bindgen::Clamped(rgba_data);
    let image_data = web_sys::ImageData::new_with_u8_clamped_array_and_sh(clamped, width, height)?;

    ctx.put_image_data(&image_data, 0.0, 0.0)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests (native-only, since we can't run web-sys in unit tests)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasm_canvas_dimensions() {
        let canvas = WasmCanvas::new(200, 100);
        assert_eq!(canvas.width(), 200);
        assert_eq!(canvas.height(), 100);
    }

    #[test]
    fn wasm_canvas_pixels_length() {
        let canvas = WasmCanvas::new(10, 10);
        let pixels = canvas.pixels();
        assert_eq!(pixels.len(), 10 * 10 * 4);
    }

    #[test]
    fn wasm_canvas_add_shapes() {
        let mut canvas = WasmCanvas::new(100, 100);
        assert_eq!(canvas.command_count(), 0);
        canvas.add_circle(50.0, 50.0, 25.0, 255, 0, 0, 255);
        assert_eq!(canvas.command_count(), 1);
        canvas.add_rect(10.0, 10.0, 30.0, 30.0, 0, 255, 0, 255);
        assert_eq!(canvas.command_count(), 2);
    }

    #[test]
    fn wasm_canvas_clear() {
        let mut canvas = WasmCanvas::new(100, 100);
        canvas.add_circle(50.0, 50.0, 25.0, 255, 0, 0, 255);
        assert_eq!(canvas.command_count(), 1);
        canvas.clear();
        assert_eq!(canvas.command_count(), 0);
    }

    #[test]
    fn wasm_canvas_set_background() {
        let mut canvas = WasmCanvas::new(50, 50);
        canvas.set_background(20, 20, 30, 255);
        let pixels = canvas.pixels();
        // Background should be non-transparent
        assert!(pixels.iter().any(|&b| b != 0));
    }

    #[test]
    fn wasm_canvas_pixels_with_shapes() {
        let mut canvas = WasmCanvas::new(100, 100);
        canvas.set_background(0, 0, 0, 255);
        canvas.add_circle(50.0, 50.0, 30.0, 255, 0, 0, 255);
        let pixels = canvas.pixels();
        assert_eq!(pixels.len(), 100 * 100 * 4);
        // Should have non-black pixels (from the red circle)
        let has_red = pixels.chunks_exact(4).any(|c| c[0] > 0 && c[2] == 0);
        assert!(has_red, "expected red pixels from circle");
    }
}
