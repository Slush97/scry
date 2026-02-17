// SPDX-License-Identifier: MIT OR Apache-2.0
//! Text rendering via fontdue.

#[cfg(feature = "text")]
use tiny_skia::{Pixmap, Transform as SkiaTransform};

#[cfg(feature = "text")]
use crate::scene::command::FontData;

use super::Rasterizer;

impl Rasterizer {
    /// Render text by rasterizing each glyph with `fontdue` and painting
    /// via tiny-skia's SIMD-optimized `draw_pixmap`.
    ///
    /// Font objects are cached per-thread by `FontData` pointer identity
    /// to avoid re-parsing TTF/OTF files on every text command.
    #[cfg(feature = "text")]
    #[allow(
        clippy::too_many_arguments,
        clippy::many_single_char_names,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_lossless
    )]
    pub(super) fn render_text(
        pixmap: &mut Pixmap,
        text: &str,
        x: f32,
        y: f32,
        font_size: f32,
        color: &crate::scene::style::Color,
        font_data: &FontData,
        parent_transform: SkiaTransform,
    ) {
        use std::cell::RefCell;
        use std::collections::HashMap;

        // Thread-local font cache keyed by Arc pointer identity.
        // Avoids re-parsing the entire TTF/OTF file (~1-5 ms) on
        // every DrawCommand::Text using the same font.
        thread_local! {
            static FONT_CACHE: RefCell<HashMap<usize, fontdue::Font>> =
                RefCell::new(HashMap::new());
        }

        let font_key = font_data.arc_ptr();

        let font_ok = FONT_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(font_key) {
                match fontdue::Font::from_bytes(font_data.bytes(), fontdue::FontSettings::default())
                {
                    Ok(font) => {
                        e.insert(font);
                    }
                    Err(_) => return false,
                }
            }
            true
        });
        if !font_ok {
            return;
        }

        let r = (color.r * 255.0) as u8;
        let g = (color.g * 255.0) as u8;
        let b = (color.b * 255.0) as u8;
        let a = (color.a * 255.0) as u8;

        let mut cursor_x = x;

        FONT_CACHE.with(|cache| {
            let cache = cache.borrow();
            // SAFETY: font was inserted into the cache immediately above.
            let font = cache.get(&font_key).expect("font was just inserted");

            for ch in text.chars() {
                let (metrics, bitmap) = font.rasterize(ch, font_size);

                if metrics.width > 0 && metrics.height > 0 {
                    let gw = metrics.width as u32;
                    let gh = metrics.height as u32;

                    // Build glyph pixmap with the text color × coverage alpha
                    if let Some(mut glyph_pm) = Pixmap::new(gw, gh) {
                        let glyph_data = glyph_pm.data_mut();
                        for (i, &coverage) in bitmap.iter().enumerate() {
                            if coverage == 0 {
                                continue;
                            }
                            let ca = ((a as u16) * (coverage as u16) / 255) as u8;
                            let idx = i * 4;
                            // Premultiplied RGBA (tiny-skia expects this)
                            glyph_data[idx] = ((r as u16 * ca as u16) / 255) as u8;
                            glyph_data[idx + 1] = ((g as u16 * ca as u16) / 255) as u8;
                            glyph_data[idx + 2] = ((b as u16 * ca as u16) / 255) as u8;
                            glyph_data[idx + 3] = ca;
                        }

                        // Position: baseline-relative
                        let gx = cursor_x as i32 + metrics.xmin;
                        let gy = y as i32 - metrics.height as i32 - metrics.ymin;

                        let paint = tiny_skia::PixmapPaint {
                            opacity: 1.0,
                            blend_mode: tiny_skia::BlendMode::SourceOver,
                            quality: tiny_skia::FilterQuality::Nearest,
                        };

                        // Use draw_pixmap for SIMD-optimized alpha blending.
                        // Also respects parent_transform (rotation, scale, etc.).
                        pixmap.draw_pixmap(
                            gx,
                            gy,
                            glyph_pm.as_ref(),
                            &paint,
                            parent_transform,
                            None,
                        );
                    }
                }

                cursor_x += metrics.advance_width;
            }
        });
    }
}
