// SPDX-License-Identifier: MIT OR Apache-2.0
//! Sigil Mono — a programmatic monospace font generator.
//!
//! Every glyph is defined as Rust code composing geometric primitives.
//! The font's "source" is readable, forkable code — embodying open source.
//!
//! # Usage
//!
//! ```no_run
//! let params = scry_font::FontParams::default();
//! let ttf_bytes = scry_font::generate_font(&params);
//! std::fs::write("SigilMono-Regular.ttf", &ttf_bytes).unwrap();
//! ```

pub mod glyphs;
pub mod params;
pub mod primitives;
pub mod tables;

pub use params::FontParams;

/// Generate a complete TTF font from the given design parameters.
///
/// Returns the raw bytes of a valid TrueType font file.
pub fn generate_font(params: &FontParams) -> Vec<u8> {
    tables::generate(params)
}
