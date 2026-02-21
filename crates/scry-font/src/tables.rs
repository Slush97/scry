// SPDX-License-Identifier: MIT OR Apache-2.0
//! OpenType table assembly — converts glyph contours into a complete TTF.

use kurbo::BezPath;
use write_fonts::tables::cmap::Cmap;
use write_fonts::tables::glyf::{SimpleGlyph, GlyfLocaBuilder, Glyph};
use write_fonts::tables::head::{Head, Flags, MacStyle};
use write_fonts::tables::hhea::Hhea;
use write_fonts::tables::hmtx::Hmtx;
use write_fonts::tables::maxp::Maxp;
use write_fonts::tables::name::{Name, NameRecord};
use write_fonts::tables::os2::Os2;
use write_fonts::tables::post::Post;
use write_fonts::types::{FWord, Fixed, GlyphId, LongDateTime, NameId, UfWord, Version16Dot16};
use write_fonts::FontBuilder;

use crate::glyphs::glyph_contours;
use crate::params::FontParams;
use crate::primitives::Contour;

const CHARSET: &str = "abcdefghijklmnopqrstuvwxyz0123456789.,-_:!/|";

fn unique_chars() -> Vec<char> {
    let mut seen = std::collections::HashSet::new();
    CHARSET.chars().filter(|ch| seen.insert(*ch)).collect()
}

/// Convert our internal contours into a kurbo BezPath.
fn contours_to_bezpath(contours: &[Contour]) -> BezPath {
    let mut path = BezPath::new();
    for contour in contours {
        let pts = &contour.points;
        if pts.len() < 2 {
            continue;
        }
        let first_on = pts.iter().position(|p| p.on_curve).unwrap_or(0);
        let start = &pts[first_on];
        path.move_to((start.x as f64, start.y as f64));

        let n = pts.len();
        let mut i = 1;
        while i < n {
            let idx = (first_on + i) % n;
            let p = &pts[idx];
            if p.on_curve {
                path.line_to((p.x as f64, p.y as f64));
                i += 1;
            } else {
                let next_idx = (first_on + i + 1) % n;
                let next = &pts[next_idx];
                let end = if next.on_curve {
                    i += 2;
                    (next.x as f64, next.y as f64)
                } else {
                    i += 1;
                    ((p.x as f64 + next.x as f64) / 2.0, (p.y as f64 + next.y as f64) / 2.0)
                };
                path.quad_to((p.x as f64, p.y as f64), end);
            }
        }
        path.close_path();
    }
    path
}

fn notdef_bezpath(p: &FontParams) -> BezPath {
    let s = p.stroke_width as f64;
    let aw = p.advance_width as f64;
    let asc = p.ascender as f64;
    let mut path = BezPath::new();
    path.move_to((s, 0.0));
    path.line_to((aw - s, 0.0));
    path.line_to((aw - s, asc));
    path.line_to((s, asc));
    path.close_path();
    path.move_to((s * 2.0, asc - s));
    path.line_to((aw - s * 2.0, asc - s));
    path.line_to((aw - s * 2.0, s));
    path.line_to((s * 2.0, s));
    path.close_path();
    path
}

/// Generate the complete TTF font as a byte vector.
pub fn generate(p: &FontParams) -> Vec<u8> {
    let chars = unique_chars();
    let num_glyphs = chars.len() as u16 + 2;

    let mut builder = GlyfLocaBuilder::new();

    // GID 0: .notdef
    let notdef = SimpleGlyph::from_bezpath(&notdef_bezpath(p)).expect("notdef");
    builder.add_glyph(&notdef).unwrap();
    // GID 1: space
    builder.add_glyph(&Glyph::Empty).unwrap();

    for ch in &chars {
        if let Some(contours) = glyph_contours(*ch, p) {
            let bp = contours_to_bezpath(&contours);
            match SimpleGlyph::from_bezpath(&bp) {
                Ok(g) => { builder.add_glyph(&g).unwrap(); }
                Err(_) => { builder.add_glyph(&Glyph::Empty).unwrap(); }
            }
        } else {
            builder.add_glyph(&Glyph::Empty).unwrap();
        }
    }

    let (glyf, loca, loca_format) = builder.build();

    let head = Head {
        units_per_em: p.units_per_em,
        created: LongDateTime::new(0),
        modified: LongDateTime::new(0),
        lowest_rec_ppem: 8,
        flags: Flags::BASELINE_AT_Y_0,
        mac_style: MacStyle::empty(),
        font_direction_hint: 2,
        index_to_loc_format: loca_format as i16,
        ..Default::default()
    };

    let hhea = Hhea {
        ascender: FWord::new(p.ascender),
        descender: FWord::new(p.descender),
        line_gap: FWord::new(0),
        advance_width_max: UfWord::new(p.advance_width),
        min_left_side_bearing: FWord::new(0),
        min_right_side_bearing: FWord::new(0),
        x_max_extent: FWord::new(p.advance_width as i16),
        caret_slope_rise: 1,
        caret_slope_run: 0,
        caret_offset: 0,
        number_of_h_metrics: num_glyphs,
    };

    let metrics: Vec<_> = (0..num_glyphs)
        .map(|_| write_fonts::tables::hmtx::LongMetric {
            advance: p.advance_width,
            side_bearing: p.lsb(),
        })
        .collect();
    let hmtx = Hmtx::new(metrics, vec![]);
    let maxp = Maxp { num_glyphs, ..Default::default() };

    // cmap: uses char, not u32
    let mut map: Vec<(char, GlyphId)> = Vec::new();
    map.push((' ', GlyphId::new(1)));
    for (i, ch) in chars.iter().enumerate() {
        map.push((*ch, GlyphId::new(i as u32 + 2)));
    }
    let cmap = Cmap::from_mappings(map).unwrap();

    let post = Post {
        version: Version16Dot16::new(3, 0),
        italic_angle: Fixed::from_f64(0.0),
        underline_position: FWord::new(-100),
        underline_thickness: FWord::new(p.stroke_width),
        is_fixed_pitch: 1,
        ..Default::default()
    };

    let name = Name::new(build_name_records());
    let os2 = Os2 {
        x_avg_char_width: p.advance_width as i16,
        us_weight_class: 400,
        us_width_class: 5,
        fs_type: 0,
        y_strikeout_size: p.stroke_width,
        y_strikeout_position: p.x_height / 2,
        s_typo_ascender: p.ascender,
        s_typo_descender: p.descender,
        s_typo_line_gap: 0,
        us_win_ascent: p.ascender as u16,
        us_win_descent: (-p.descender) as u16,
        sx_height: Some(p.x_height),
        s_cap_height: Some(p.cap_height),
        ul_code_page_range_1: Some(1), // Latin 1
        ul_code_page_range_2: Some(0),
        us_default_char: Some(0),
        us_break_char: Some(32), // space
        us_max_context: Some(0),
        ..Default::default()
    };

    let mut fb = FontBuilder::new();
    fb.add_table(&head).expect("head");
    fb.add_table(&hhea).expect("hhea");
    fb.add_table(&hmtx).expect("hmtx");
    fb.add_table(&maxp).expect("maxp");
    fb.add_table(&cmap).expect("cmap");
    fb.add_table(&glyf).expect("glyf");
    fb.add_table(&loca).expect("loca");
    fb.add_table(&post).expect("post");
    fb.add_table(&name).expect("name");
    fb.add_table(&os2).expect("OS/2");
    fb.build()
}

fn build_name_records() -> Vec<NameRecord> {
    use write_fonts::OffsetMarker;
    let names = [
        (NameId::COPYRIGHT_NOTICE, "Copyright 2026 esoc. SIL OFL 1.1."),
        (NameId::FAMILY_NAME, "Sigil Mono"),
        (NameId::SUBFAMILY_NAME, "Regular"),
        (NameId::UNIQUE_ID, "esoc;SigilMono-Regular;2026"),
        (NameId::FULL_NAME, "Sigil Mono Regular"),
        (NameId::VERSION_STRING, "Version 0.1.0"),
        (NameId::POSTSCRIPT_NAME, "SigilMono-Regular"),
    ];
    names
        .iter()
        .map(|(id, text)| {
            NameRecord::new(0, 3, 0, *id, OffsetMarker::new(text.to_string()))
        })
        .collect()
}
