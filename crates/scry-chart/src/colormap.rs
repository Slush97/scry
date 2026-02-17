// SPDX-License-Identifier: MIT OR Apache-2.0
//! Perceptual colormaps for heatmaps and continuous data visualization.
//!
//! Provides a [`Colormap`] trait and built-in implementations for
//! scientific-quality sequential and diverging palettes. Each colormap maps
//! a scalar `t ∈ [0.0, 1.0]` to a perceptually uniform color.
//!
//! # Examples
//!
//! ```
//! use scry_chart::colormap::{Colormap, Viridis};
//!
//! let cmap = Viridis;
//! let start = cmap.color_at(0.0);
//! let mid = cmap.color_at(0.5);
//! let end = cmap.color_at(1.0);
//! ```

use scry_engine::style::Color;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// A colormap that maps a normalized scalar to a color.
///
/// The input `t` is clamped to `[0.0, 1.0]`. Implementations should return
/// perceptually meaningful colors across the full range.
pub trait Colormap: std::fmt::Debug + Send + Sync {
    /// Map a scalar `t ∈ [0.0, 1.0]` to a color.
    fn color_at(&self, t: f32) -> Color;

    /// Human-readable name (e.g. `"viridis"`).
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Linearly interpolate through a table of (position, Color) stops.
fn table_lookup(stops: &[(f32, Color)], t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    if stops.is_empty() {
        return Color::BLACK;
    }
    if t <= stops[0].0 {
        return stops[0].1;
    }
    let last = stops.len() - 1;
    if t >= stops[last].0 {
        return stops[last].1;
    }
    for i in 0..last {
        let (p0, c0) = stops[i];
        let (p1, c1) = stops[i + 1];
        if t >= p0 && t <= p1 {
            let frac = (t - p0) / (p1 - p0);
            return c0.mix(c1, frac);
        }
    }
    stops[last].1
}

// ---------------------------------------------------------------------------
// Sequential colormaps
// ---------------------------------------------------------------------------

/// Viridis — perceptually uniform, yellow→teal→indigo.
///
/// The gold standard for scientific visualization. Colorblind-safe.
#[derive(Clone, Copy, Debug)]
pub struct Viridis;

impl Colormap for Viridis {
    fn color_at(&self, t: f32) -> Color {
        table_lookup(&VIRIDIS_STOPS, t)
    }
    fn name(&self) -> &'static str {
        "viridis"
    }
}

/// Plasma — perceptually uniform, yellow→magenta→indigo.
#[derive(Clone, Copy, Debug)]
pub struct Plasma;

impl Colormap for Plasma {
    fn color_at(&self, t: f32) -> Color {
        table_lookup(&PLASMA_STOPS, t)
    }
    fn name(&self) -> &'static str {
        "plasma"
    }
}

/// Inferno — perceptually uniform, yellow→red→black.
#[derive(Clone, Copy, Debug)]
pub struct Inferno;

impl Colormap for Inferno {
    fn color_at(&self, t: f32) -> Color {
        table_lookup(&INFERNO_STOPS, t)
    }
    fn name(&self) -> &'static str {
        "inferno"
    }
}

/// Magma — perceptually uniform, yellow→magenta→black.
#[derive(Clone, Copy, Debug)]
pub struct Magma;

impl Colormap for Magma {
    fn color_at(&self, t: f32) -> Color {
        table_lookup(&MAGMA_STOPS, t)
    }
    fn name(&self) -> &'static str {
        "magma"
    }
}

// ---------------------------------------------------------------------------
// Diverging colormaps
// ---------------------------------------------------------------------------

/// Red–Blue diverging colormap.
///
/// Neutral at the midpoint, red for high values, blue for low values.
#[derive(Clone, Copy, Debug)]
pub struct RdBu;

impl Colormap for RdBu {
    fn color_at(&self, t: f32) -> Color {
        table_lookup(&RDBU_STOPS, t)
    }
    fn name(&self) -> &'static str {
        "rdbu"
    }
}

/// Brown–Blue-Green diverging colormap.
#[derive(Clone, Copy, Debug)]
pub struct BrBG;

impl Colormap for BrBG {
    fn color_at(&self, t: f32) -> Color {
        table_lookup(&BRBG_STOPS, t)
    }
    fn name(&self) -> &'static str {
        "brbg"
    }
}

/// Pink–Yellow-Green diverging colormap.
#[derive(Clone, Copy, Debug)]
pub struct PiYG;

impl Colormap for PiYG {
    fn color_at(&self, t: f32) -> Color {
        table_lookup(&PIYG_STOPS, t)
    }
    fn name(&self) -> &'static str {
        "piyg"
    }
}

// ---------------------------------------------------------------------------
// Lookup by name
// ---------------------------------------------------------------------------

/// Look up a built-in colormap by name (case-insensitive).
///
/// Returns `None` if the name is not recognized.
///
/// # Examples
///
/// ```
/// use scry_chart::colormap::colormap_from_name;
///
/// let cmap = colormap_from_name("viridis").unwrap();
/// assert_eq!(cmap.name(), "viridis");
/// ```
#[must_use]
pub fn colormap_from_name(name: &str) -> Option<Box<dyn Colormap>> {
    match name.to_ascii_lowercase().as_str() {
        "viridis" => Some(Box::new(Viridis)),
        "plasma" => Some(Box::new(Plasma)),
        "inferno" => Some(Box::new(Inferno)),
        "magma" => Some(Box::new(Magma)),
        "rdbu" | "rd_bu" => Some(Box::new(RdBu)),
        "brbg" | "br_bg" => Some(Box::new(BrBG)),
        "piyg" | "pi_yg" => Some(Box::new(PiYG)),
        _ => None,
    }
}

// ===========================================================================
// Stop tables — sampled from matplotlib reference implementations
// ===========================================================================

const fn c(r: u8, g: u8, b: u8) -> Color {
    Color::from_rgb8(r, g, b)
}

static VIRIDIS_STOPS: [(f32, Color); 9] = [
    (0.000, c(68, 1, 84)),
    (0.125, c(72, 36, 117)),
    (0.250, c(64, 67, 135)),
    (0.375, c(52, 94, 141)),
    (0.500, c(33, 144, 140)),
    (0.625, c(43, 176, 99)),
    (0.750, c(121, 209, 81)),
    (0.875, c(189, 222, 38)),
    (1.000, c(253, 231, 37)),
];

static PLASMA_STOPS: [(f32, Color); 9] = [
    (0.000, c(13, 8, 135)),
    (0.125, c(75, 3, 161)),
    (0.250, c(125, 3, 168)),
    (0.375, c(168, 34, 150)),
    (0.500, c(203, 70, 121)),
    (0.625, c(229, 107, 93)),
    (0.750, c(248, 148, 65)),
    (0.875, c(253, 195, 40)),
    (1.000, c(240, 249, 33)),
];

static INFERNO_STOPS: [(f32, Color); 9] = [
    (0.000, c(0, 0, 4)),
    (0.125, c(40, 11, 84)),
    (0.250, c(101, 21, 110)),
    (0.375, c(159, 42, 99)),
    (0.500, c(212, 72, 66)),
    (0.625, c(245, 125, 21)),
    (0.750, c(250, 193, 39)),
    (0.875, c(252, 255, 164)),
    (1.000, c(252, 255, 164)),
];

static MAGMA_STOPS: [(f32, Color); 9] = [
    (0.000, c(0, 0, 4)),
    (0.125, c(28, 16, 68)),
    (0.250, c(79, 18, 123)),
    (0.375, c(136, 34, 106)),
    (0.500, c(186, 55, 85)),
    (0.625, c(227, 97, 76)),
    (0.750, c(249, 149, 103)),
    (0.875, c(254, 207, 165)),
    (1.000, c(252, 253, 191)),
];

static RDBU_STOPS: [(f32, Color); 7] = [
    (0.000, c(33, 102, 172)),
    (0.167, c(103, 169, 207)),
    (0.333, c(209, 229, 240)),
    (0.500, c(247, 247, 247)),
    (0.667, c(253, 219, 199)),
    (0.833, c(239, 138, 98)),
    (1.000, c(178, 24, 43)),
];

static BRBG_STOPS: [(f32, Color); 7] = [
    (0.000, c(0, 60, 48)),
    (0.167, c(53, 151, 143)),
    (0.333, c(199, 234, 229)),
    (0.500, c(245, 245, 245)),
    (0.667, c(223, 194, 125)),
    (0.833, c(166, 97, 26)),
    (1.000, c(84, 48, 5)),
];

static PIYG_STOPS: [(f32, Color); 7] = [
    (0.000, c(39, 100, 25)),
    (0.167, c(127, 188, 65)),
    (0.333, c(217, 240, 211)),
    (0.500, c(247, 247, 247)),
    (0.667, c(253, 224, 239)),
    (0.833, c(233, 163, 201)),
    (1.000, c(197, 27, 125)),
];

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viridis_endpoints() {
        let cmap = Viridis;
        let c0 = cmap.color_at(0.0);
        let c1 = cmap.color_at(1.0);
        // Should be distinct colors
        assert!((c0.r - c1.r).abs() > 0.1 || (c0.g - c1.g).abs() > 0.1);
    }

    #[test]
    fn plasma_midpoint() {
        let mid = Plasma.color_at(0.5);
        // Should return a valid color
        assert!(mid.r >= 0.0 && mid.r <= 1.0);
        assert!(mid.g >= 0.0 && mid.g <= 1.0);
        assert!(mid.b >= 0.0 && mid.b <= 1.0);
    }

    #[test]
    fn clamps_out_of_range() {
        let c_neg = Viridis.color_at(-0.5);
        let c_zero = Viridis.color_at(0.0);
        assert_eq!(c_neg.r, c_zero.r);
        assert_eq!(c_neg.g, c_zero.g);

        let c_over = Viridis.color_at(1.5);
        let c_one = Viridis.color_at(1.0);
        assert_eq!(c_over.r, c_one.r);
    }

    #[test]
    fn all_colormaps_have_names() {
        assert_eq!(Viridis.name(), "viridis");
        assert_eq!(Plasma.name(), "plasma");
        assert_eq!(Inferno.name(), "inferno");
        assert_eq!(Magma.name(), "magma");
        assert_eq!(RdBu.name(), "rdbu");
        assert_eq!(BrBG.name(), "brbg");
        assert_eq!(PiYG.name(), "piyg");
    }

    #[test]
    fn lookup_by_name() {
        assert!(colormap_from_name("viridis").is_some());
        assert!(colormap_from_name("PLASMA").is_some());
        assert!(colormap_from_name("rdbu").is_some());
        assert!(colormap_from_name("rd_bu").is_some());
        assert!(colormap_from_name("unknown").is_none());
    }

    #[test]
    fn diverging_midpoint_is_neutral() {
        let mid = RdBu.color_at(0.5);
        // RdBu midpoint is near-white
        assert!(mid.r > 0.9);
        assert!(mid.g > 0.9);
        assert!(mid.b > 0.9);
    }

    #[test]
    fn all_sequential_monotonic_lightness() {
        // Viridis should get brighter toward t=1
        let start = Viridis.color_at(0.0);
        let end = Viridis.color_at(1.0);
        let start_lum = 0.2126 * start.r + 0.7152 * start.g + 0.0722 * start.b;
        let end_lum = 0.2126 * end.r + 0.7152 * end.g + 0.0722 * end.b;
        assert!(end_lum > start_lum, "viridis should brighten toward t=1");
    }
}
