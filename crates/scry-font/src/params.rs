// SPDX-License-Identifier: MIT OR Apache-2.0
//! Font design parameters — the single source of truth for Sigil Mono's appearance.
//!
//! Every glyph reads from [`FontParams`] so that changing one value (e.g. stroke
//! width) consistently updates every letter in the font. This is the typographic
//! equivalent of a design-token system.

/// Design parameters controlling the entire font's geometry.
///
/// All vertical measurements are in font units (UPM = units per em).
/// Horizontal measurements use the same unit space.
#[derive(Clone, Debug)]
pub struct FontParams {
    /// Units per em — defines the coordinate space. Standard values: 1000 or 2048.
    pub units_per_em: u16,
    /// Ascender height above baseline (positive).
    pub ascender: i16,
    /// Descender depth below baseline (negative).
    pub descender: i16,
    /// Capital letter height.
    pub cap_height: i16,
    /// Lowercase letter height (excluding ascenders/descenders).
    pub x_height: i16,
    /// Uniform stroke width for all stems, bars, and arcs.
    pub stroke_width: i16,
    /// Monospace advance width — every glyph occupies this horizontal space.
    pub advance_width: u16,
    /// Angle (in degrees) for the signature crystal-facet terminal cuts.
    pub terminal_angle: f32,
    /// Radius for circular dots (i, j, punctuation).
    pub dot_radius: i16,
    /// Overshoot: round glyphs extend this far past alignment zones
    /// for optical consistency.
    pub overshoot: i16,
    /// Baseline position (typically 0 in font coordinates).
    pub baseline: i16,
}

impl Default for FontParams {
    /// Sigil Mono Regular — the default parameter set.
    fn default() -> Self {
        Self {
            units_per_em: 1000,
            ascender: 800,
            descender: -200,
            cap_height: 700,
            x_height: 525,
            stroke_width: 80,
            advance_width: 600,
            terminal_angle: 45.0,
            dot_radius: 50,
            overshoot: 10,
            baseline: 0,
        }
    }
}

impl FontParams {
    /// Left side bearing — horizontal padding from the left edge of the
    /// advance width to where the glyph's strokes begin.
    pub fn lsb(&self) -> i16 {
        // Center the glyph body within the advance width.
        // Glyph body width = advance_width - 2 * lsb.
        // For a 600-unit advance with 80-unit stroke, leaves ~80 on each side.
        let body_width = self.advance_width as i16 - self.stroke_width * 2;
        (self.advance_width as i16 - body_width) / 2
    }

    /// Usable glyph body width (advance minus side bearings).
    pub fn body_width(&self) -> i16 {
        self.advance_width as i16 - 2 * self.lsb()
    }

    /// Half stroke width — convenience for centering strokes on guidelines.
    pub fn half_stroke(&self) -> i16 {
        self.stroke_width / 2
    }

    /// The inner radius for bowls/arcs, accounting for stroke width.
    pub fn inner_radius(&self, outer_radius: i16) -> i16 {
        (outer_radius - self.stroke_width).max(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_params_are_consistent() {
        let p = FontParams::default();
        assert!(p.x_height < p.cap_height, "x-height must be below cap height");
        assert!(p.cap_height < p.ascender, "cap height must be below ascender");
        assert!(p.descender < 0, "descender must be negative");
        assert!(p.body_width() > 0, "body width must be positive");
        assert!(p.lsb() > 0, "left side bearing must be positive");
    }
}
