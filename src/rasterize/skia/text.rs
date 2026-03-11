// SPDX-License-Identifier: MIT OR Apache-2.0
//! Text rendering via fontdue.

#[cfg(feature = "text")]
use tiny_skia::{Pixmap, Transform as SkiaTransform};

#[cfg(feature = "text")]
use crate::scene::command::{FontData, TextAlign, TextMetrics};

use super::Rasterizer;

// ---------------------------------------------------------------------------
// Default font (embedded Inter-Regular)
// ---------------------------------------------------------------------------

/// Embedded Inter-Regular font bytes for text rendering without user-provided fonts.
#[cfg(feature = "text")]
pub(crate) const DEFAULT_FONT_BYTES: &[u8] =
    include_bytes!("../../../crates/scry-chart/src/fonts/Inter-Regular.ttf");

/// Lazily initialized default [`FontData`].
#[cfg(feature = "text")]
static DEFAULT_FONT_CELL: std::sync::OnceLock<FontData> = std::sync::OnceLock::new();

/// Return the default embedded font ([Inter-Regular](https://rsms.me/inter/)).
///
/// The `FontData` is initialized once and reused for the lifetime of the process.
#[cfg(feature = "text")]
pub fn default_font() -> FontData {
    DEFAULT_FONT_CELL
        .get_or_init(|| FontData::new(DEFAULT_FONT_BYTES.to_vec()))
        .clone()
}

// ---------------------------------------------------------------------------
// Font fallback chain (lazy-loaded system fonts)
// ---------------------------------------------------------------------------

/// A parsed fallback font: the raw `FontData` + parsed `fontdue::Font`.
#[cfg(feature = "text")]
struct FallbackFont {
    /// Kept alive so the bytes backing `font` remain valid.
    #[allow(dead_code)]
    data: FontData,
    font: fontdue::Font,
}

/// Lazily initialized fallback font chain discovered from system font
/// directories.  Only loaded on the first glyph miss in the primary font,
/// so Latin-only text incurs zero extra cost.
#[cfg(feature = "text")]
static FALLBACK_CHAIN: std::sync::OnceLock<Vec<FallbackFont>> = std::sync::OnceLock::new();

/// Well-known system font directories (Linux, macOS, Windows).
#[cfg(feature = "text")]
const SYSTEM_FONT_DIRS: &[&str] = &[
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    "/System/Library/Fonts",
    "/Library/Fonts",
    "C:\\Windows\\Fonts",
];

/// Priority keywords for fallback font file names (checked in order).
/// Noto Sans variants are preferred because they cover most Unicode blocks.
#[cfg(feature = "text")]
const PRIORITY_KEYWORDS: &[&str] = &[
    "NotoSansCJK",
    "NotoSansArabic",
    "NotoSansHebrew",
    "NotoSansDevanagari",
    "NotoSans-Regular",
    "NotoSans-",
    "DroidSansFallback",
    "FreeSans",
    "DejaVuSans",
    "Arial",
    "Roboto",
];

/// Build the fallback chain by scanning system font directories.
///
/// Priority fonts (Noto Sans variants) are placed first, followed by any
/// remaining `.ttf` / `.otf` files found.  Each file is parsed once; files
/// that fail to parse are silently skipped.
#[cfg(feature = "text")]
fn discover_fallback_fonts() -> Vec<FallbackFont> {
    let mut priority: Vec<(usize, std::path::PathBuf)> = Vec::new();
    let mut others: Vec<std::path::PathBuf> = Vec::new();

    // Also check $HOME/.local/share/fonts
    let home_fonts = std::env::var("HOME")
        .ok()
        .map(|h| std::path::PathBuf::from(h).join(".local/share/fonts"));

    let dirs: Vec<&std::path::Path> = SYSTEM_FONT_DIRS
        .iter()
        .map(std::path::Path::new)
        .chain(home_fonts.as_deref())
        .collect();

    for dir in dirs {
        if !dir.is_dir() {
            continue;
        }
        let walker = walkdir(dir);
        for path in walker {
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let ext = std::path::Path::new(name)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");
            if !ext.eq_ignore_ascii_case("ttf") && !ext.eq_ignore_ascii_case("otf") {
                continue;
            }
            // Check priority
            if let Some(pos) = PRIORITY_KEYWORDS.iter().position(|kw| name.contains(kw)) {
                priority.push((pos, path));
            } else {
                others.push(path);
            }
        }
    }

    // Sort priority fonts by their keyword rank
    priority.sort_by_key(|(rank, _)| *rank);

    let mut chain = Vec::new();
    // Cap at a reasonable limit to avoid loading hundreds of fonts
    let candidates = priority.into_iter().map(|(_, p)| p).chain(others).take(12);

    for path in candidates {
        if let Ok(bytes) = std::fs::read(&path) {
            if let Ok(font) =
                fontdue::Font::from_bytes(bytes.as_slice(), fontdue::FontSettings::default())
            {
                chain.push(FallbackFont {
                    data: FontData::new(bytes),
                    font,
                });
            }
        }
    }
    chain
}

/// Simple recursive directory walk (avoids adding a `walkdir` crate dep).
#[cfg(feature = "text")]
fn walkdir(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut result = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                result.extend(walkdir(&path));
            } else {
                result.push(path);
            }
        }
    }
    result
}

/// Get the fallback font chain, initializing it on first call.
#[cfg(feature = "text")]
fn fallback_chain() -> &'static [FallbackFont] {
    FALLBACK_CHAIN.get_or_init(discover_fallback_fonts)
}

// ---------------------------------------------------------------------------
// Text measurement
// ---------------------------------------------------------------------------

/// Measure the pixel dimensions of `text` without rendering it.
///
/// If `font_data` is `None`, the embedded default font is used.
#[cfg(feature = "text")]
#[allow(clippy::cast_precision_loss)]
pub fn measure_text(text: &str, font_data: Option<&FontData>, font_size: f32) -> TextMetrics {
    use std::cell::RefCell;
    use std::collections::HashMap;

    thread_local! {
        static FONT_CACHE: RefCell<HashMap<usize, fontdue::Font>> =
            RefCell::new(HashMap::new());
    }

    let fd = font_data.cloned().unwrap_or_else(default_font);
    let font_key = fd.arc_ptr();

    FONT_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(font_key) {
            match fontdue::Font::from_bytes(fd.bytes(), fontdue::FontSettings::default()) {
                Ok(font) => {
                    e.insert(font);
                }
                Err(_) => {
                    return TextMetrics {
                        width: 0.0,
                        height: 0.0,
                        ascent: 0.0,
                        descent: 0.0,
                    };
                }
            }
        }

        let font = cache.get(&font_key).expect("font was just inserted");

        let mut width = 0.0_f32;
        for ch in text.chars() {
            if font.has_glyph(ch) {
                let (metrics, _) = font.rasterize(ch, font_size);
                width += metrics.advance_width;
            } else {
                // Try fallback chain
                let mut found = false;
                for fb in fallback_chain() {
                    if fb.font.has_glyph(ch) {
                        let (metrics, _) = fb.font.rasterize(ch, font_size);
                        width += metrics.advance_width;
                        found = true;
                        break;
                    }
                }
                if !found {
                    // No fallback has it — use primary font (tofu)
                    let (metrics, _) = font.rasterize(ch, font_size);
                    width += metrics.advance_width;
                }
            }
        }

        let line_metrics = font.horizontal_line_metrics(font_size);
        let (ascent, descent) = line_metrics.map_or((font_size * 0.8, font_size * 0.2), |lm| {
            (lm.ascent, -lm.descent)
        });

        TextMetrics {
            width,
            height: ascent + descent,
            ascent,
            descent,
        }
    })
}

// ---------------------------------------------------------------------------
// Text-to-image (for 3D compositing)
// ---------------------------------------------------------------------------

/// Rasterize text to a standalone [`ImageData`](crate::scene::command::ImageData).
///
/// The result can be composited onto a canvas via `canvas.image(img, x, y)` —
/// useful for putting text on SDF 3D scenes or applying transforms via groups.
#[cfg(feature = "text")]
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]
pub fn render_text_to_image(
    text: &str,
    font_data: Option<&FontData>,
    font_size: f32,
    color: &crate::scene::style::Color,
    padding: u32,
) -> crate::scene::command::ImageData {
    let metrics = measure_text(text, font_data, font_size);

    let w = (metrics.width.ceil() as u32).max(1) + padding * 2;
    let h = (metrics.height.ceil() as u32).max(1) + padding * 2;

    if let Some(mut pixmap) = Pixmap::new(w, h) {
        let fd = font_data.cloned().unwrap_or_else(default_font);
        Rasterizer::render_text(
            &mut pixmap,
            text,
            padding as f32,
            padding as f32 + metrics.ascent,
            font_size,
            color,
            &fd,
            SkiaTransform::identity(),
            TextAlign::Left,
        );
        crate::scene::command::ImageData::new(w, h, pixmap.data().to_vec())
    } else {
        crate::scene::command::ImageData::new(1, 1, vec![0; 4])
    }
}

// ---------------------------------------------------------------------------
// Core text rendering
// ---------------------------------------------------------------------------

/// 8 directions for outline rendering at unit distance.
#[cfg(feature = "text")]
const OUTLINE_OFFSETS: [(f32, f32); 8] = [
    (-1.0, 0.0),
    (1.0, 0.0),
    (0.0, -1.0),
    (0.0, 1.0),
    (-0.707, -0.707),
    (0.707, -0.707),
    (-0.707, 0.707),
    (0.707, 0.707),
];

impl Rasterizer {
    /// High-level text rendering entry point that handles outline, shadow,
    /// gradient fill, and plain text in the correct order.
    #[cfg(feature = "text")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn render_rich_text(
        pixmap: &mut Pixmap,
        text: &str,
        x: f32,
        y: f32,
        font_size: f32,
        color: &crate::scene::style::Color,
        font_data: &FontData,
        parent_transform: SkiaTransform,
        align: TextAlign,
        outline_color: Option<&crate::scene::style::Color>,
        outline_width: Option<f32>,
        fill_style: Option<&crate::scene::style::FillStyle>,
        shadow: Option<&crate::scene::command::TextShadow>,
        _grad_cache: &mut super::GradientCache,
    ) {
        // 1. Shadow (rendered first, behind everything)
        if let Some(shadow) = shadow {
            Self::render_text_shadow(
                pixmap,
                text,
                x,
                y,
                font_size,
                font_data,
                parent_transform,
                align,
                shadow,
            );
        }

        // 2. Outline (rendered behind the fill)
        if let (Some(oc), Some(ow)) = (outline_color, outline_width) {
            if ow > 0.0 {
                for &(dx, dy) in &OUTLINE_OFFSETS {
                    Self::render_text(
                        pixmap,
                        text,
                        dx.mul_add(ow, x),
                        dy.mul_add(ow, y),
                        font_size,
                        oc,
                        font_data,
                        parent_transform,
                        align,
                    );
                }
            }
        }

        // 3. Fill: gradient or solid color
        if let Some(fill) = fill_style {
            Self::render_text_gradient(
                pixmap,
                text,
                x,
                y,
                font_size,
                color,
                font_data,
                parent_transform,
                align,
                fill,
            );
        } else {
            Self::render_text(
                pixmap,
                text,
                x,
                y,
                font_size,
                color,
                font_data,
                parent_transform,
                align,
            );
        }
    }

    /// Render text with a gradient fill by rasterizing to an alpha mask,
    /// then painting the gradient through the mask.
    #[cfg(feature = "text")]
    #[allow(
        clippy::too_many_arguments,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss
    )]
    fn render_text_gradient(
        pixmap: &mut Pixmap,
        text: &str,
        x: f32,
        y: f32,
        font_size: f32,
        _color: &crate::scene::style::Color,
        font_data: &FontData,
        parent_transform: SkiaTransform,
        align: TextAlign,
        fill: &crate::scene::style::FillStyle,
    ) {
        let metrics = measure_text(text, Some(font_data), font_size);
        let pad = 2u32;
        let w = (metrics.width.ceil() as u32).max(1) + pad * 2;
        let h = (metrics.height.ceil() as u32).max(1) + pad * 2;

        // Render text as white-on-transparent to get the alpha mask
        let white = crate::scene::style::Color::WHITE;
        let Some(mut mask_pm) = Pixmap::new(w, h) else {
            return;
        };
        Self::render_text(
            &mut mask_pm,
            text,
            pad as f32,
            pad as f32 + metrics.ascent,
            font_size,
            &white,
            font_data,
            SkiaTransform::identity(),
            TextAlign::Left,
        );

        // Create gradient pixmap
        let Some(mut grad_pm) = Pixmap::new(w, h) else {
            return;
        };
        let bounds = crate::scene::style::Rect::new(0.0, 0.0, w as f32, h as f32);

        let grad = match fill {
            crate::scene::style::FillStyle::Solid(c) => {
                // Solid fill: just render normally
                Self::render_text(
                    pixmap,
                    text,
                    x,
                    y,
                    font_size,
                    c,
                    font_data,
                    parent_transform,
                    align,
                );
                return;
            }
            crate::scene::style::FillStyle::LinearGradient(g)
            | crate::scene::style::FillStyle::RadialGradient(g) => g,
        };

        let mut paint = Self::gradient_to_paint(grad, &bounds);
        paint.anti_alias = true;
        let skia_rect = tiny_skia::Rect::from_xywh(0.0, 0.0, w as f32, h as f32);
        if let Some(r) = skia_rect {
            grad_pm.fill_rect(r, &paint, SkiaTransform::identity(), None);
        }

        // Multiply gradient by text alpha mask
        let grad_data = grad_pm.data_mut();
        let mask_data = mask_pm.data();
        for i in (0..grad_data.len()).step_by(4) {
            let mask_a = mask_data[i + 3] as u16;
            grad_data[i] = ((grad_data[i] as u16 * mask_a) / 255) as u8;
            grad_data[i + 1] = ((grad_data[i + 1] as u16 * mask_a) / 255) as u8;
            grad_data[i + 2] = ((grad_data[i + 2] as u16 * mask_a) / 255) as u8;
            grad_data[i + 3] = ((grad_data[i + 3] as u16 * mask_a) / 255) as u8;
        }

        // Compute alignment offset
        let offset_x = match align {
            TextAlign::Left => 0.0,
            TextAlign::Center => -metrics.width / 2.0,
            TextAlign::Right => -metrics.width,
        };

        // Blit to target pixmap
        let blit_paint = tiny_skia::PixmapPaint {
            opacity: 1.0,
            blend_mode: tiny_skia::BlendMode::SourceOver,
            quality: tiny_skia::FilterQuality::Nearest,
        };
        let dest_x = (x + offset_x - pad as f32) as i32;
        let dest_y = (y - metrics.ascent - pad as f32) as i32;
        pixmap.draw_pixmap(
            dest_x,
            dest_y,
            grad_pm.as_ref(),
            &blit_paint,
            parent_transform,
            None,
        );
    }

    /// Render a text shadow (optionally blurred via box-downsample).
    #[cfg(feature = "text")]
    #[allow(
        clippy::too_many_arguments,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss
    )]
    fn render_text_shadow(
        pixmap: &mut Pixmap,
        text: &str,
        x: f32,
        y: f32,
        font_size: f32,
        font_data: &FontData,
        parent_transform: SkiaTransform,
        align: TextAlign,
        shadow: &crate::scene::command::TextShadow,
    ) {
        if shadow.blur_radius <= 0.5 {
            // Sharp shadow: just render at offset
            Self::render_text(
                pixmap,
                text,
                x + shadow.offset_x,
                y + shadow.offset_y,
                font_size,
                &shadow.color,
                font_data,
                parent_transform,
                align,
            );
        } else {
            // Blurred shadow: render at 2× size, then box-downsample
            let metrics = measure_text(text, Some(font_data), font_size);
            let pad = (shadow.blur_radius.ceil() as u32).max(2);
            let w = (metrics.width.ceil() as u32).max(1) + pad * 2;
            let h = (metrics.height.ceil() as u32).max(1) + pad * 2;
            let w2 = w * 2;
            let h2 = h * 2;

            let Some(mut big) = Pixmap::new(w2, h2) else {
                return;
            };
            Self::render_text(
                &mut big,
                text,
                (pad * 2) as f32,
                (pad * 2) as f32 + metrics.ascent * 2.0,
                font_size * 2.0,
                &shadow.color,
                font_data,
                SkiaTransform::identity(),
                TextAlign::Left,
            );

            // Box downsample 2×2
            let Some(mut small) = Pixmap::new(w, h) else {
                return;
            };
            let big_data = big.data();
            let small_data = small.data_mut();
            let stride2 = (w2 * 4) as usize;
            for sy in 0..h {
                for sx in 0..w {
                    let by = (sy * 2) as usize;
                    let bx = (sx * 2) as usize;
                    let mut sr = 0u32;
                    let mut sg = 0u32;
                    let mut sb = 0u32;
                    let mut sa = 0u32;
                    for dy in 0..2usize {
                        for dx in 0..2usize {
                            let off = (by + dy) * stride2 + (bx + dx) * 4;
                            sr += big_data[off] as u32;
                            sg += big_data[off + 1] as u32;
                            sb += big_data[off + 2] as u32;
                            sa += big_data[off + 3] as u32;
                        }
                    }
                    let off = ((sy * w + sx) * 4) as usize;
                    small_data[off] = (sr / 4) as u8;
                    small_data[off + 1] = (sg / 4) as u8;
                    small_data[off + 2] = (sb / 4) as u8;
                    small_data[off + 3] = (sa / 4) as u8;
                }
            }

            let offset_x = match align {
                TextAlign::Left => 0.0,
                TextAlign::Center => -metrics.width / 2.0,
                TextAlign::Right => -metrics.width,
            };

            let blit_paint = tiny_skia::PixmapPaint {
                opacity: 1.0,
                blend_mode: tiny_skia::BlendMode::SourceOver,
                quality: tiny_skia::FilterQuality::Bilinear,
            };
            let dest_x = (x + offset_x + shadow.offset_x - pad as f32) as i32;
            let dest_y = (y - metrics.ascent + shadow.offset_y - pad as f32) as i32;
            pixmap.draw_pixmap(
                dest_x,
                dest_y,
                small.as_ref(),
                &blit_paint,
                parent_transform,
                None,
            );
        }
    }
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
        align: TextAlign,
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

        // Compute alignment offset
        let offset_x = match align {
            TextAlign::Left => 0.0,
            TextAlign::Center | TextAlign::Right => {
                let metrics = measure_text(text, Some(font_data), font_size);
                match align {
                    TextAlign::Center => -metrics.width / 2.0,
                    TextAlign::Right => -metrics.width,
                    TextAlign::Left => unreachable!(),
                }
            }
        };

        let mut cursor_x = x + offset_x;

        FONT_CACHE.with(|cache| {
            let cache = cache.borrow();
            // SAFETY: font was inserted into the cache immediately above.
            let font = cache.get(&font_key).expect("font was just inserted");

            for ch in text.chars() {
                // Choose the best font for this character: primary or fallback.
                let (metrics, bitmap) = if font.has_glyph(ch) {
                    font.rasterize(ch, font_size)
                } else {
                    // Search fallback chain for a font that has this glyph
                    let mut fallback_result = None;
                    for fb in fallback_chain() {
                        if fb.font.has_glyph(ch) {
                            fallback_result = Some(fb.font.rasterize(ch, font_size));
                            break;
                        }
                    }
                    fallback_result.unwrap_or_else(|| font.rasterize(ch, font_size))
                };

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
