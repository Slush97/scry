// SPDX-License-Identifier: MIT OR Apache-2.0
//! Glyph definitions — each function returns contours for one character.
//!
//! Glyphs are built by composing shared primitives from [`crate::primitives`].

use crate::params::FontParams;
use crate::primitives::*;

/// Registry: returns contours for a given character code point.
pub fn glyph_contours(ch: char, p: &FontParams) -> Option<Vec<Contour>> {
    match ch {
        's' => Some(glyph_s(p)),
        'c' => Some(glyph_c(p)),
        'r' => Some(glyph_r(p)),
        'y' => Some(glyph_y(p)),
        'a' => Some(glyph_a(p)),
        'b' => Some(glyph_b(p)),
        'd' => Some(glyph_d(p)),
        'e' => Some(glyph_e(p)),
        'f' => Some(glyph_f(p)),
        'g' => Some(glyph_g(p)),
        'h' => Some(glyph_h(p)),
        'i' => Some(glyph_i(p)),
        'j' => Some(glyph_j(p)),
        'k' => Some(glyph_k(p)),
        'l' => Some(glyph_l(p)),
        'm' => Some(glyph_m(p)),
        'n' => Some(glyph_n(p)),
        'o' => Some(glyph_o(p)),
        'p' => Some(glyph_p(p)),
        'q' => Some(glyph_q(p)),
        't' => Some(glyph_t(p)),
        'u' => Some(glyph_u(p)),
        'v' => Some(glyph_v(p)),
        'w' => Some(glyph_w(p)),
        'x' => Some(glyph_x(p)),
        'z' => Some(glyph_z(p)),
        '0' => Some(glyph_0(p)),
        '1' => Some(glyph_1(p)),
        '2' => Some(glyph_2(p)),
        '3' => Some(glyph_3(p)),
        '4' => Some(glyph_4(p)),
        '5' => Some(glyph_5(p)),
        '6' => Some(glyph_6(p)),
        '7' => Some(glyph_7(p)),
        '8' => Some(glyph_8(p)),
        '9' => Some(glyph_9(p)),
        '.' => Some(glyph_period(p)),
        ',' => Some(glyph_comma(p)),
        '-' => Some(glyph_hyphen(p)),
        '_' => Some(glyph_underscore(p)),
        ':' => Some(glyph_colon(p)),
        '!' => Some(glyph_exclaim(p)),
        '/' => Some(glyph_slash(p)),
        '|' => Some(glyph_pipe(p)),
        _ => None,
    }
}

// ── Helper measurements ─────────────────────────────────────────────

fn left(p: &FontParams) -> i16 { p.lsb() }
fn right(p: &FontParams) -> i16 { p.advance_width as i16 - p.lsb() }
fn mid_x(p: &FontParams) -> f32 { p.advance_width as f32 / 2.0 }
fn sw(p: &FontParams) -> i16 { p.stroke_width }
fn xh(p: &FontParams) -> i16 { p.x_height }
fn half(p: &FontParams) -> i16 { p.x_height / 2 }

// ── Lowercase ───────────────────────────────────────────────────────

fn glyph_s(p: &FontParams) -> Vec<Contour> {
    // S-shape: top arc opening right, bottom arc opening left
    let l = left(p) as f32;
    let r = right(p) as f32;
    let mx = mid_x(p);
    let s = sw(p) as f32;
    let top = xh(p) as f32;
    let mid = top / 2.0;

    let outer_r = (r - l) / 2.0;
    let inner_r = outer_r - s;
    let top_cy = top - outer_r;
    let bot_cy = outer_r;

    // Top half: arc from ~0° to ~180° (opens right with terminal)
    let top_arc = thick_arc(mx, top_cy, outer_r, inner_r, 45.0, 200.0);
    // Bottom half: arc from ~180° to ~360° (opens left with terminal)
    let bot_arc = thick_arc(mx, bot_cy, outer_r, inner_r, 225.0, 20.0);
    vec![top_arc, bot_arc]
}

fn glyph_c(p: &FontParams) -> Vec<Contour> {
    let mx = mid_x(p);
    let s = sw(p) as f32;
    let top = xh(p) as f32;
    let cy = top / 2.0;
    let outer_r = (right(p) - left(p)) as f32 / 2.0;
    let inner_r = outer_r - s;
    // Open arc — wide aperture (symbol of openness)
    let arc = thick_arc(mx, cy, outer_r, inner_r, 40.0, 320.0);
    vec![arc]
}

fn glyph_r(p: &FontParams) -> Vec<Contour> {
    let l = left(p);
    let s = sw(p);
    // Vertical stem on the left
    let stem = vertical_stem(l, 0, xh(p), p);
    // Shoulder: horizontal bar at top + small arc
    let bar = horizontal_bar(l, right(p) - s, xh(p) - s, p);
    vec![stem, bar]
}

fn glyph_y(p: &FontParams) -> Vec<Contour> {
    let l = left(p) as f32;
    let r = right(p) as f32;
    let s = sw(p) as f32;
    let top = xh(p) as f32;
    let mid = top * 0.4;
    let desc = p.descender as f32;
    let mx = mid_x(p);
    // Left diagonal from top-left to middle
    let d1 = diagonal(l, top, mx, mid, s);
    // Right diagonal from top-right to middle
    let d2 = diagonal(r, top, mx, mid, s);
    // Descender stem from middle to below baseline
    let stem = rect(mx.round() as i16 - p.half_stroke(), desc as i16, s as i16, (mid - desc) as i16);
    vec![d1, d2, stem]
}

fn glyph_a(p: &FontParams) -> Vec<Contour> {
    let l = left(p);
    let r = right(p);
    let s = sw(p);
    // Right stem
    let stem = vertical_stem(r - s, 0, xh(p), p);
    // Bowl: thick arc on left side
    let mx = mid_x(p);
    let cy = xh(p) as f32 / 2.0;
    let outer_r = (r - l) as f32 / 2.0;
    let inner_r = outer_r - s as f32;
    let bowl = thick_arc(mx, cy, outer_r, inner_r, 90.0, 360.0 + 45.0);
    vec![stem, bowl]
}

fn glyph_b(p: &FontParams) -> Vec<Contour> {
    let l = left(p);
    let s = sw(p);
    let stem = vertical_stem(l, 0, p.ascender, p);
    let mx = mid_x(p);
    let cy = xh(p) as f32 / 2.0;
    let outer_r = (right(p) - l) as f32 / 2.0;
    let inner_r = outer_r - s as f32;
    let bowl = thick_arc(mx, cy, outer_r, inner_r, -45.0, 270.0);
    vec![stem, bowl]
}

fn glyph_d(p: &FontParams) -> Vec<Contour> {
    let r = right(p);
    let s = sw(p);
    let stem = vertical_stem(r - s, 0, p.ascender, p);
    let mx = mid_x(p);
    let cy = xh(p) as f32 / 2.0;
    let outer_r = (r - left(p)) as f32 / 2.0;
    let inner_r = outer_r - s as f32;
    let bowl = thick_arc(mx, cy, outer_r, inner_r, 90.0, 360.0 + 45.0);
    vec![stem, bowl]
}

fn glyph_e(p: &FontParams) -> Vec<Contour> {
    let mx = mid_x(p);
    let s = sw(p) as f32;
    let top = xh(p) as f32;
    let cy = top / 2.0;
    let outer_r = (right(p) - left(p)) as f32 / 2.0;
    let inner_r = outer_r - s;
    // Open arc + crossbar
    let arc = thick_arc(mx, cy, outer_r, inner_r, 20.0, 340.0);
    // Middle crossbar
    let bar = horizontal_bar(left(p), right(p), (cy - s / 2.0) as i16, p);
    vec![arc, bar]
}

fn glyph_f(p: &FontParams) -> Vec<Contour> {
    let l = left(p);
    let s = sw(p);
    let stem = vertical_stem(l + s, 0, p.ascender - s, p);
    // Top hook
    let hook = horizontal_bar(l + s, right(p), p.ascender - s, p);
    // Crossbar at x-height
    let bar = horizontal_bar(l, right(p) - s, xh(p) - s / 2, p);
    vec![stem, hook, bar]
}

fn glyph_g(p: &FontParams) -> Vec<Contour> {
    let r = right(p);
    let s = sw(p);
    let mx = mid_x(p);
    let cy = xh(p) as f32 / 2.0;
    let outer_r = (r - left(p)) as f32 / 2.0;
    let inner_r = outer_r - s as f32;
    let bowl = thick_arc(mx, cy, outer_r, inner_r, -45.0, 300.0);
    // Descender stem
    let stem = vertical_stem(r - s, p.descender, xh(p) / 2, p);
    vec![bowl, stem]
}

fn glyph_h(p: &FontParams) -> Vec<Contour> {
    let l = left(p);
    let r = right(p);
    let s = sw(p);
    let stem_l = vertical_stem(l, 0, p.ascender, p);
    let stem_r = vertical_stem(r - s, 0, xh(p), p);
    let bar = horizontal_bar(l, r, xh(p) - s, p);
    vec![stem_l, stem_r, bar]
}

fn glyph_i(p: &FontParams) -> Vec<Contour> {
    let mx = mid_x(p);
    let s = sw(p);
    let hs = p.half_stroke();
    let stem = vertical_stem(mx as i16 - hs, 0, xh(p), p);
    // Dot above
    let dot = circle(mx, (xh(p) + s + p.dot_radius) as f32, p.dot_radius as f32);
    vec![stem, dot]
}

fn glyph_j(p: &FontParams) -> Vec<Contour> {
    let r = right(p);
    let s = sw(p);
    let stem = vertical_stem(r - s - p.half_stroke(), p.descender, xh(p), p);
    let dot = circle((r - s / 2) as f32, (xh(p) + s + p.dot_radius) as f32, p.dot_radius as f32);
    vec![stem, dot]
}

fn glyph_k(p: &FontParams) -> Vec<Contour> {
    let l = left(p) as f32;
    let r = right(p) as f32;
    let s = sw(p) as f32;
    let top = xh(p) as f32;
    let asc = p.ascender as f32;
    let mid = top * 0.5;
    let stem = vertical_stem(l as i16, 0, p.ascender, p);
    let d_up = diagonal(l + s, mid, r, top, s);
    let d_dn = diagonal(l + s, mid, r, 0.0, s);
    vec![stem, d_up, d_dn]
}

fn glyph_l(p: &FontParams) -> Vec<Contour> {
    let mx = mid_x(p);
    let hs = p.half_stroke();
    // Stem with a small curved tail (hockey-stick base) for disambiguation
    let stem = vertical_stem(mx as i16 - hs, 0, p.ascender, p);
    vec![stem]
}

fn glyph_m(p: &FontParams) -> Vec<Contour> {
    let l = left(p);
    let r = right(p);
    let s = sw(p);
    let mx = mid_x(p) as i16;
    let top = xh(p);
    let stem_l = vertical_stem(l, 0, top, p);
    let stem_m = vertical_stem(mx - s / 2, 0, top, p);
    let stem_r = vertical_stem(r - s, 0, top, p);
    let bar = horizontal_bar(l, r, top - s, p);
    vec![stem_l, stem_m, stem_r, bar]
}

fn glyph_n(p: &FontParams) -> Vec<Contour> {
    let l = left(p);
    let r = right(p);
    let s = sw(p);
    let stem_l = vertical_stem(l, 0, xh(p), p);
    let stem_r = vertical_stem(r - s, 0, xh(p), p);
    let bar = horizontal_bar(l, r, xh(p) - s, p);
    vec![stem_l, stem_r, bar]
}

fn glyph_o(p: &FontParams) -> Vec<Contour> {
    let mx = mid_x(p);
    let cy = xh(p) as f32 / 2.0;
    let outer_r = (right(p) - left(p)) as f32 / 2.0;
    let inner_r = outer_r - sw(p) as f32;
    ring(mx, cy, outer_r, inner_r)
}

fn glyph_p(p: &FontParams) -> Vec<Contour> {
    let l = left(p);
    let s = sw(p);
    let stem = vertical_stem(l, p.descender, xh(p), p);
    let mx = mid_x(p);
    let cy = xh(p) as f32 / 2.0;
    let outer_r = (right(p) - l) as f32 / 2.0;
    let inner_r = outer_r - s as f32;
    let bowl = thick_arc(mx, cy, outer_r, inner_r, -45.0, 270.0);
    vec![stem, bowl]
}

fn glyph_q(p: &FontParams) -> Vec<Contour> {
    let r = right(p);
    let s = sw(p);
    let stem = vertical_stem(r - s, p.descender, xh(p), p);
    let mx = mid_x(p);
    let cy = xh(p) as f32 / 2.0;
    let outer_r = (r - left(p)) as f32 / 2.0;
    let inner_r = outer_r - s as f32;
    let bowl = thick_arc(mx, cy, outer_r, inner_r, 90.0, 360.0 + 45.0);
    vec![stem, bowl]
}

fn glyph_t(p: &FontParams) -> Vec<Contour> {
    let mx = mid_x(p);
    let hs = p.half_stroke();
    let stem = vertical_stem(mx as i16 - hs, 0, p.ascender, p);
    let bar = horizontal_bar(left(p), right(p), xh(p) - sw(p) / 2, p);
    vec![stem, bar]
}

fn glyph_u(p: &FontParams) -> Vec<Contour> {
    let l = left(p);
    let r = right(p);
    let s = sw(p);
    let stem_l = vertical_stem(l, 0, xh(p), p);
    let stem_r = vertical_stem(r - s, 0, xh(p), p);
    let bar = horizontal_bar(l, r, 0, p);
    vec![stem_l, stem_r, bar]
}

fn glyph_v(p: &FontParams) -> Vec<Contour> {
    let l = left(p) as f32;
    let r = right(p) as f32;
    let s = sw(p) as f32;
    let top = xh(p) as f32;
    let mx = mid_x(p);
    let d1 = diagonal(l, top, mx, 0.0, s);
    let d2 = diagonal(r, top, mx, 0.0, s);
    vec![d1, d2]
}

fn glyph_w(p: &FontParams) -> Vec<Contour> {
    let l = left(p) as f32;
    let r = right(p) as f32;
    let s = sw(p) as f32 * 0.8; // slightly thinner for w
    let top = xh(p) as f32;
    let mx = mid_x(p);
    let q1 = l + (r - l) * 0.25;
    let q3 = l + (r - l) * 0.75;
    let d1 = diagonal(l, top, q1, 0.0, s);
    let d2 = diagonal(q1, 0.0, mx, top * 0.6, s);
    let d3 = diagonal(mx, top * 0.6, q3, 0.0, s);
    let d4 = diagonal(q3, 0.0, r, top, s);
    vec![d1, d2, d3, d4]
}

fn glyph_x(p: &FontParams) -> Vec<Contour> {
    let l = left(p) as f32;
    let r = right(p) as f32;
    let s = sw(p) as f32;
    let top = xh(p) as f32;
    let d1 = diagonal(l, top, r, 0.0, s);
    let d2 = diagonal(r, top, l, 0.0, s);
    vec![d1, d2]
}

fn glyph_z(p: &FontParams) -> Vec<Contour> {
    let l = left(p);
    let r = right(p);
    let s = sw(p);
    let top = xh(p);
    let top_bar = horizontal_bar(l, r, top - s, p);
    let bot_bar = horizontal_bar(l, r, 0, p);
    let diag = diagonal(l as f32, s as f32, r as f32, (top - s) as f32, s as f32);
    vec![top_bar, bot_bar, diag]
}

// ── Digits ──────────────────────────────────────────────────────────

fn glyph_0(p: &FontParams) -> Vec<Contour> {
    let mx = mid_x(p);
    let cy = xh(p) as f32 / 2.0;
    let outer_r = (right(p) - left(p)) as f32 / 2.0;
    let inner_r = outer_r - sw(p) as f32;
    let mut contours = ring(mx, cy, outer_r, inner_r);
    // Dotted zero — central dot for disambiguation
    let dot = circle(mx, cy, p.dot_radius as f32);
    contours.push(dot);
    contours
}

fn glyph_1(p: &FontParams) -> Vec<Contour> {
    let mx = mid_x(p);
    let hs = p.half_stroke();
    let stem = vertical_stem(mx as i16 - hs, 0, xh(p), p);
    // Short flag at top-left
    let flag = diagonal(
        (mx as i16 - hs) as f32, xh(p) as f32,
        (mx as i16 - hs - sw(p)) as f32, (xh(p) - sw(p) * 2) as f32,
        sw(p) as f32,
    );
    vec![stem, flag]
}

fn digit_oval(p: &FontParams, y_off: i16, h_frac: f32) -> Vec<Contour> {
    let mx = mid_x(p);
    let h = (xh(p) as f32 * h_frac) as i16;
    let cy = y_off as f32 + h as f32 / 2.0;
    let rx = (right(p) - left(p)) as f32 / 2.0;
    let ry = h as f32 / 2.0;
    let outer_r = rx.min(ry);
    let inner_r = outer_r - sw(p) as f32;
    ring(mx, cy, outer_r, inner_r)
}

fn glyph_2(p: &FontParams) -> Vec<Contour> {
    let l = left(p);
    let r = right(p);
    let s = sw(p);
    let top = xh(p);
    let bot_bar = horizontal_bar(l, r, 0, p);
    let top_arc = thick_arc(mid_x(p), (top * 3 / 4) as f32, (r - l) as f32 / 2.0, (r - l) as f32 / 2.0 - s as f32, 0.0, 180.0);
    let diag = diagonal(l as f32, s as f32, r as f32, (top / 2) as f32, s as f32);
    vec![top_arc, diag, bot_bar]
}

fn glyph_3(p: &FontParams) -> Vec<Contour> {
    let mx = mid_x(p);
    let s = sw(p) as f32;
    let top = xh(p) as f32;
    let r = (right(p) - left(p)) as f32 / 2.0;
    let ir = r - s;
    let top_arc = thick_arc(mx, top * 0.75, r, ir, -30.0, 180.0);
    let bot_arc = thick_arc(mx, top * 0.25, r, ir, -180.0, 30.0);
    vec![top_arc, bot_arc]
}

fn glyph_4(p: &FontParams) -> Vec<Contour> {
    let r = right(p);
    let l = left(p);
    let s = sw(p);
    let top = xh(p);
    let mid = top * 45 / 100;
    let stem = vertical_stem(r - s - s, 0, top, p);
    let bar = horizontal_bar(l, r, mid, p);
    let diag = diagonal(l as f32, mid as f32 + s as f32, (r - s - s) as f32, top as f32, s as f32);
    vec![stem, bar, diag]
}

fn glyph_5(p: &FontParams) -> Vec<Contour> {
    let l = left(p);
    let r = right(p);
    let s = sw(p);
    let top = xh(p);
    let top_bar = horizontal_bar(l, r, top - s, p);
    let vert = vertical_stem(l, top / 2, top - s, p);
    let mx = mid_x(p);
    let bot_arc = thick_arc(mx, (top as f32) * 0.25, (r - l) as f32 / 2.0, (r - l) as f32 / 2.0 - s as f32, -180.0, 40.0);
    vec![top_bar, vert, bot_arc]
}

fn glyph_6(p: &FontParams) -> Vec<Contour> {
    let mx = mid_x(p);
    let s = sw(p) as f32;
    let top = xh(p) as f32;
    let r = (right(p) - left(p)) as f32 / 2.0;
    let ir = r - s;
    let bowl = ring(mx, r, r, ir);
    let stem = thick_arc(mx, top * 0.6, r, ir, 90.0, 200.0);
    let mut out: Vec<Contour> = bowl;
    out.push(stem);
    out
}

fn glyph_7(p: &FontParams) -> Vec<Contour> {
    let l = left(p);
    let r = right(p);
    let s = sw(p);
    let top = xh(p);
    let bar = horizontal_bar(l, r, top - s, p);
    let diag = diagonal(r as f32, (top - s) as f32, (l + s) as f32, 0.0, s as f32);
    vec![bar, diag]
}

fn glyph_8(p: &FontParams) -> Vec<Contour> {
    let mx = mid_x(p);
    let s = sw(p) as f32;
    let top = xh(p) as f32;
    let r = (right(p) - left(p)) as f32 / 2.0;
    let ir = r - s;
    let top_ring = ring(mx, top * 0.72, r * 0.85, ir * 0.85);
    let bot_ring = ring(mx, top * 0.28, r, ir);
    merge(vec![top_ring, bot_ring])
}

fn glyph_9(p: &FontParams) -> Vec<Contour> {
    let mx = mid_x(p);
    let s = sw(p) as f32;
    let top = xh(p) as f32;
    let r = (right(p) - left(p)) as f32 / 2.0;
    let ir = r - s;
    let bowl = ring(mx, top - r, r, ir);
    let stem = thick_arc(mx, top * 0.4, r, ir, -20.0, -110.0);
    let mut out = bowl;
    out.push(stem);
    out
}

// ── Punctuation ─────────────────────────────────────────────────────

fn glyph_period(p: &FontParams) -> Vec<Contour> {
    let mx = mid_x(p);
    vec![circle(mx, p.dot_radius as f32, p.dot_radius as f32)]
}

fn glyph_comma(p: &FontParams) -> Vec<Contour> {
    let mx = mid_x(p);
    let dr = p.dot_radius as f32;
    let dot = circle(mx, dr, dr);
    let tail = diagonal(mx, 0.0, mx - dr, -dr * 1.5, sw(p) as f32 * 0.6);
    vec![dot, tail]
}

fn glyph_hyphen(p: &FontParams) -> Vec<Contour> {
    vec![horizontal_bar(left(p), right(p), half(p) - p.half_stroke(), p)]
}

fn glyph_underscore(p: &FontParams) -> Vec<Contour> {
    vec![horizontal_bar(left(p), right(p), p.descender + sw(p), p)]
}

fn glyph_colon(p: &FontParams) -> Vec<Contour> {
    let mx = mid_x(p);
    let dr = p.dot_radius as f32;
    let top_dot = circle(mx, xh(p) as f32 * 0.7, dr);
    let bot_dot = circle(mx, xh(p) as f32 * 0.2, dr);
    vec![top_dot, bot_dot]
}

fn glyph_exclaim(p: &FontParams) -> Vec<Contour> {
    let mx = mid_x(p);
    let hs = p.half_stroke();
    let stem = vertical_stem(mx as i16 - hs, sw(p) * 2, xh(p), p);
    let dot = circle(mx, p.dot_radius as f32, p.dot_radius as f32);
    vec![stem, dot]
}

fn glyph_slash(p: &FontParams) -> Vec<Contour> {
    vec![diagonal(
        left(p) as f32, 0.0,
        right(p) as f32, xh(p) as f32,
        sw(p) as f32,
    )]
}

fn glyph_pipe(p: &FontParams) -> Vec<Contour> {
    let mx = mid_x(p);
    let hs = p.half_stroke();
    vec![vertical_stem(mx as i16 - hs, p.descender, p.ascender, p)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_registered_glyphs_produce_contours() {
        let p = FontParams::default();
        for ch in "scryabdefghijklmnopqtuvwxz0123456789.,-_:!/|".chars() {
            let contours = glyph_contours(ch, &p);
            assert!(contours.is_some(), "glyph '{}' should produce contours", ch);
            let contours = contours.unwrap();
            assert!(!contours.is_empty(), "glyph '{}' should have at least one contour", ch);
            for (i, c) in contours.iter().enumerate() {
                assert!(c.points.len() >= 3,
                    "glyph '{}' contour {} has only {} points",
                    ch, i, c.points.len());
            }
        }
    }
}
