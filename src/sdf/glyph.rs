// SPDX-License-Identifier: MIT OR Apache-2.0
//! 3D text SDF from TTF font outlines.
//!
//! Pipeline: TTF glyph outlines → flatten curves → scanline rasterize →
//! Felzenszwalb/Huttenlocher EDT → 2D SDF grid → extrude to 3D via IQ formula.
//!
//! Caches `GlyphSdf` grids behind a global mutex for zero-copy sharing across
//! rayon threads and repeated text objects.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::math::{Vec2, Vec3};

// ── Outline extraction ──────────────────────────────────────────────

/// Collects flattened line edges from a TTF glyph outline.
struct OutlineCollector {
    edges: Vec<(f32, f32, f32, f32)>, // (x0, y0, x1, y1)
    cursor: (f32, f32),
    start: (f32, f32),
}

impl OutlineCollector {
    fn new() -> Self {
        Self {
            edges: Vec::new(),
            cursor: (0.0, 0.0),
            start: (0.0, 0.0),
        }
    }
}

impl ttf_parser::OutlineBuilder for OutlineCollector {
    fn move_to(&mut self, x: f32, y: f32) {
        self.cursor = (x, y);
        self.start = (x, y);
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.edges.push((self.cursor.0, self.cursor.1, x, y));
        self.cursor = (x, y);
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        // De Casteljau flattening — 16 segments for quadratic Bézier
        let (x0, y0) = self.cursor;
        let steps = 16;
        let mut prev = (x0, y0);
        for i in 1..=steps {
            let t = i as f32 / steps as f32;
            let inv = 1.0 - t;
            let px = inv * inv * x0 + 2.0 * inv * t * x1 + t * t * x;
            let py = inv * inv * y0 + 2.0 * inv * t * y1 + t * t * y;
            self.edges.push((prev.0, prev.1, px, py));
            prev = (px, py);
        }
        self.cursor = (x, y);
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        // De Casteljau flattening — 32 segments for cubic Bézier
        let (x0, y0) = self.cursor;
        let steps = 32;
        let mut prev = (x0, y0);
        for i in 1..=steps {
            let t = i as f32 / steps as f32;
            let inv = 1.0 - t;
            let px = inv * inv * inv * x0
                + 3.0 * inv * inv * t * x1
                + 3.0 * inv * t * t * x2
                + t * t * t * x;
            let py = inv * inv * inv * y0
                + 3.0 * inv * inv * t * y1
                + 3.0 * inv * t * t * y2
                + t * t * t * y;
            self.edges.push((prev.0, prev.1, px, py));
            prev = (px, py);
        }
        self.cursor = (x, y);
    }

    fn close(&mut self) {
        let (cx, cy) = self.cursor;
        let (sx, sy) = self.start;
        if (cx - sx).abs() > 1e-6 || (cy - sy).abs() > 1e-6 {
            self.edges.push((cx, cy, sx, sy));
        }
        self.cursor = self.start;
    }
}

// ── Scanline rasterization ──────────────────────────────────────────

/// Rasterize edges into a binary bitmap using even-odd fill rule.
#[cfg(test)]
fn rasterize_bitmap(
    edges: &[(f32, f32, f32, f32)],
    width: usize,
    height: usize,
    // Maps font-space coordinates to bitmap coordinates
    offset_x: f32,
    offset_y: f32,
    scale: f32,
) -> Vec<bool> {
    let mut bitmap = vec![false; width * height];

    for y in 0..height {
        let scan_y = (y as f32 + 0.5 - offset_y) / scale;
        // Collect x intersections for this scanline
        let mut intersections = Vec::new();

        for &(x0, y0, x1, y1) in edges {
            // Skip horizontal edges and edges that don't cross this scanline
            if (y0 - y1).abs() < 1e-10 {
                continue;
            }
            let (min_y, max_y) = if y0 < y1 { (y0, y1) } else { (y1, y0) };
            if scan_y < min_y || scan_y >= max_y {
                continue;
            }
            // Linear interpolation to find x at scan_y
            let t = (scan_y - y0) / (y1 - y0);
            let ix = x0 + t * (x1 - x0);
            intersections.push(ix);
        }

        intersections.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Even-odd fill: toggle inside/outside at each intersection
        for pair in intersections.chunks(2) {
            if pair.len() == 2 {
                let left = ((pair[0] * scale + offset_x) as isize).max(0) as usize;
                let right =
                    ((pair[1] * scale + offset_x).ceil() as usize).min(width);
                for x in left..right {
                    bitmap[y * width + x] = true;
                }
            }
        }
    }

    bitmap
}

// ── Felzenszwalb/Huttenlocher EDT ───────────────────────────────────

/// 1D squared distance transform (Felzenszwalb & Huttenlocher 2012).
///
/// Input: `f[0..n]` — function values (0 for boundary, large for non-boundary).
/// Output: `dt[0..n]` — squared distance to nearest boundary.
#[cfg(test)]
fn edt_1d(f: &[f32], dt: &mut [f32]) {
    let n = f.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        dt[0] = f[0];
        return;
    }

    let mut v = vec![0usize; n]; // locations of parabolas in lower envelope
    let mut z = vec![0.0_f32; n + 1]; // intersection points between parabolas
    let mut k = 0; // number of parabolas in lower envelope

    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;

    for q in 1..n {
        loop {
            let s = intersection(q, v[k], f);
            if s > z[k] {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f32::INFINITY;
                break;
            }
            if k == 0 {
                v[0] = q;
                z[0] = f32::NEG_INFINITY;
                z[1] = f32::INFINITY;
                break;
            }
            k -= 1;
        }
    }

    let mut k = 0;
    for q in 0..n {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let dq = q as f32 - v[k] as f32;
        dt[q] = dq * dq + f[v[k]];
    }
}

/// Intersection point of two parabolas in the EDT lower envelope.
#[cfg(test)]
#[inline]
fn intersection(q: usize, r: usize, f: &[f32]) -> f32 {
    let q_f = q as f32;
    let r_f = r as f32;
    (f[q] - f[r] + q_f * q_f - r_f * r_f) / (2.0 * (q_f - r_f))
}

/// Compute a 2D signed distance field from a binary bitmap.
///
/// Returns a grid of signed distances: negative inside, positive outside.
/// Values are in pixel units (grid cells).
#[cfg(test)]
fn compute_sdf(bitmap: &[bool], width: usize, height: usize) -> Vec<f32> {
    let inf = (width * width + height * height) as f32;

    // Compute distance transform for inside (true) and outside (false) separately
    let mut inside = vec![0.0_f32; width * height];
    let mut outside = vec![0.0_f32; width * height];

    for i in 0..bitmap.len() {
        if bitmap[i] {
            // Inside the shape: distance to nearest outside pixel
            inside[i] = inf;
            outside[i] = 0.0;
        } else {
            // Outside the shape: distance to nearest inside pixel
            inside[i] = 0.0;
            outside[i] = inf;
        }
    }

    edt_2d(&mut inside, width, height);
    edt_2d(&mut outside, width, height);

    // Combine: negative inside, positive outside
    let mut sdf = vec![0.0_f32; width * height];
    for i in 0..sdf.len() {
        sdf[i] = outside[i].sqrt() - inside[i].sqrt();
    }
    sdf
}

/// Apply 2D EDT by separable 1D passes (rows then columns).
#[cfg(test)]
fn edt_2d(grid: &mut [f32], width: usize, height: usize) {
    let mut temp = vec![0.0_f32; width.max(height)];
    let mut dt = vec![0.0_f32; width.max(height)];

    // Row pass
    for y in 0..height {
        let row_start = y * width;
        edt_1d(&grid[row_start..row_start + width], &mut dt[..width]);
        grid[row_start..row_start + width].copy_from_slice(&dt[..width]);
    }

    // Column pass
    for x in 0..width {
        for y in 0..height {
            temp[y] = grid[y * width + x];
        }
        edt_1d(&temp[..height], &mut dt[..height]);
        for y in 0..height {
            grid[y * width + x] = dt[y];
        }
    }
}

// ── GlyphSdf ────────────────────────────────────────────────────────

/// A cached 2D SDF grid for a single glyph.
///
/// Stores the signed distance field and the world-space bounds for mapping
/// sample coordinates back to grid coordinates.
#[derive(Debug)]
pub struct GlyphSdf {
    pub(super) grid: Vec<f32>,
    pub(super) width: usize,
    pub(super) height: usize,
    /// Glyph advance width in world units (for layout).
    pub advance: f32,
    /// Left side bearing in world units.
    pub lsb: f32,
    /// Glyph bounding box in world units.
    pub(super) bounds: (f32, f32, f32, f32),
}

impl GlyphSdf {
    /// Sample the SDF at world-space (x, y) with Catmull-Rom bicubic interpolation.
    ///
    /// Returns the signed distance in world units. Points outside the grid
    /// return a large positive distance. Bicubic (C1) interpolation produces
    /// smooth values AND smooth gradients, eliminating staircase artifacts.
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let (min_x, min_y, max_x, max_y) = self.bounds;
        let bw = max_x - min_x;
        let bh = max_y - min_y;

        if bw < 1e-10 || bh < 1e-10 {
            return 1e6;
        }

        // Map world coords to grid coords
        let gx = (x - min_x) / bw * (self.width as f32 - 1.0);
        let gy = (y - min_y) / bh * (self.height as f32 - 1.0);

        // Return large distance outside bounds (with some margin)
        if gx < -1.0 || gy < -1.0 || gx > self.width as f32 || gy > self.height as f32 {
            let dx = if x < min_x {
                min_x - x
            } else if x > max_x {
                x - max_x
            } else {
                0.0
            };
            let dy = if y < min_y {
                min_y - y
            } else if y > max_y {
                y - max_y
            } else {
                0.0
            };
            return dx.hypot(dy) + 0.01;
        }

        // Catmull-Rom bicubic interpolation (C1 continuous)
        let gx = gx.clamp(0.0, (self.width - 1) as f32);
        let gy = gy.clamp(0.0, (self.height - 1) as f32);

        let ix = gx as i32;
        let iy = gy as i32;
        let fx = gx - ix as f32;
        let fy = gy - iy as f32;

        // Catmull-Rom basis weights
        let fx2 = fx * fx;
        let fx3 = fx2 * fx;
        let wx = [
            -0.5 * fx3 + fx2 - 0.5 * fx,
             1.5 * fx3 - 2.5 * fx2 + 1.0,
            -1.5 * fx3 + 2.0 * fx2 + 0.5 * fx,
             0.5 * fx3 - 0.5 * fx2,
        ];

        let fy2 = fy * fy;
        let fy3 = fy2 * fy;
        let wy = [
            -0.5 * fy3 + fy2 - 0.5 * fy,
             1.5 * fy3 - 2.5 * fy2 + 1.0,
            -1.5 * fy3 + 2.0 * fy2 + 0.5 * fy,
             0.5 * fy3 - 0.5 * fy2,
        ];

        let w = self.width as i32;
        let h = self.height as i32;

        // 4×4 separable convolution with clamped boundary access
        let mut grid_dist = 0.0_f32;
        for jj in 0..4 {
            let sy = (iy + jj - 1).clamp(0, h - 1) as usize;
            let mut row_val = 0.0_f32;
            for ii in 0..4 {
                let sx = (ix + ii - 1).clamp(0, w - 1) as usize;
                row_val += self.grid[sy * self.width + sx] * wx[ii as usize];
            }
            grid_dist += row_val * wy[jj as usize];
        }

        // Convert from grid-cell units to world units
        let pixels_per_world = self.width as f32 / bw;
        grid_dist / pixels_per_world
    }

    /// Sample the SDF and compute the 2D gradient at world-space (x, y).
    ///
    /// Returns `(distance, ∂d/∂x, ∂d/∂y)` in world units. Uses central
    /// finite differences on the Catmull-Rom bicubic `sample()` with a
    /// half-grid-cell step, matching the GPU shader. This produces C1-smooth
    /// gradients without grid-stepping artifacts.
    pub fn sample_gradient(&self, x: f32, y: f32) -> (f32, f32, f32) {
        let d = self.sample(x, y);
        let (min_x, min_y, max_x, max_y) = self.bounds;
        let bw = max_x - min_x;
        let bh = max_y - min_y;
        if bw < 1e-10 || bh < 1e-10 {
            return (d, 0.0, 0.0);
        }
        // Half-grid-cell step in world units — matches GPU shader exactly.
        let hx = 0.5 * bw / self.width as f32;
        let hy = 0.5 * bh / self.height as f32;
        let dx = self.sample(x + hx, y) - self.sample(x - hx, y);
        let dy = self.sample(x, y + hy) - self.sample(x, y - hy);
        (d, dx / (2.0 * hx), dy / (2.0 * hy))
    }
}

// ── Direct edge-distance SDF ────────────────────────────────────────

/// Squared distance from point `(px, py)` to the closest point on
/// segment `(ax, ay)-(bx, by)`.
#[inline]
fn point_to_segment_dist_sq(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let dx = bx - ax;
    let dy = by - ay;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-12 {
        // Degenerate segment — distance to point A
        let ex = px - ax;
        let ey = py - ay;
        return ex * ex + ey * ey;
    }
    let t = ((px - ax) * dx + (py - ay) * dy) / len_sq;
    let t = t.clamp(0.0, 1.0);
    let cx = ax + t * dx - px;
    let cy = ay + t * dy - py;
    cx * cx + cy * cy
}

/// Compute the SDF grid directly from outline edges — no binary rasterization.
///
/// For each grid cell center, finds the minimum Euclidean distance to any
/// edge segment and determines inside/outside via ray casting (even-odd rule).
/// The result is sub-pixel-accurate: distances reflect the true outline
/// geometry, not jagged pixel boundaries.
///
/// Rows are computed in parallel via rayon for performance — a 1024² grid
/// with hundreds of edges per glyph is O(1M × edges) work.
fn compute_sdf_from_edges(
    edges: &[(f32, f32, f32, f32)],
    width: usize,
    height: usize,
) -> Vec<f32> {
    use rayon::prelude::*;
    let sdf: Vec<f32> = (0..height)
        .into_par_iter()
        .flat_map(|y| {
            let py = y as f32 + 0.5;
            (0..width)
                .map(move |x| {
                    let px = x as f32 + 0.5;
                    let mut min_dist_sq = f32::INFINITY;
                    let mut crossings = 0u32;
                    for &(ax, ay, bx, by) in edges {
                        let d = point_to_segment_dist_sq(px, py, ax, ay, bx, by);
                        if d < min_dist_sq {
                            min_dist_sq = d;
                        }
                        if (ay < py) != (by < py) {
                            let t = (py - ay) / (by - ay);
                            if px < ax + t * (bx - ax) {
                                crossings += 1;
                            }
                        }
                    }
                    let inside = (crossings & 1) == 1;
                    let dist = min_dist_sq.sqrt();
                    if inside { -dist } else { dist }
                })
                .collect::<Vec<f32>>()
        })
        .collect();
    sdf
}

// ── Cache ───────────────────────────────────────────────────────────

type CacheKey = (usize, ttf_parser::GlyphId, u32); // (font_ptr, glyph_id, resolution)

static GLYPH_CACHE: Mutex<Option<HashMap<CacheKey, Arc<GlyphSdf>>>> = Mutex::new(None);

fn get_cached_glyph(key: CacheKey) -> Option<Arc<GlyphSdf>> {
    let guard = GLYPH_CACHE.lock().ok()?;
    guard.as_ref()?.get(&key).cloned()
}

fn insert_cached_glyph(key: CacheKey, sdf: Arc<GlyphSdf>) {
    if let Ok(mut guard) = GLYPH_CACHE.lock() {
        let cache = guard.get_or_insert_with(HashMap::new);
        cache.insert(key, sdf);
    }
}

/// Build or retrieve a cached `GlyphSdf` for a single glyph.
fn build_glyph_sdf(
    face: &ttf_parser::Face<'_>,
    font_ptr: usize,
    glyph_id: ttf_parser::GlyphId,
    font_size: f32,
    resolution: u32,
) -> Option<Arc<GlyphSdf>> {
    let key = (font_ptr, glyph_id, resolution);

    if let Some(cached) = get_cached_glyph(key) {
        return Some(cached);
    }

    // Extract outline
    let mut collector = OutlineCollector::new();
    let bbox = face.outline_glyph(glyph_id, &mut collector)?;

    let units_per_em = face.units_per_em() as f32;
    let scale_to_world = font_size / units_per_em;

    // Advance width
    let advance = face
        .glyph_hor_advance(glyph_id)
        .map_or(0.0, |a| a as f32 * scale_to_world);
    let lsb = face
        .glyph_hor_side_bearing(glyph_id)
        .map_or(0.0, |b| b as f32 * scale_to_world);

    // Bounding box in font units
    let font_min_x = bbox.x_min as f32;
    let font_min_y = bbox.y_min as f32;
    let font_max_x = bbox.x_max as f32;
    let font_max_y = bbox.y_max as f32;
    let font_w = font_max_x - font_min_x;
    let font_h = font_max_y - font_min_y;

    if font_w < 1e-6 || font_h < 1e-6 {
        return None;
    }

    // Determine bitmap dimensions (maintain aspect ratio) with 2px padding
    // on each side so the EDT computes accurate distances near glyph edges.
    let pad = 2_usize;
    let res = resolution as usize;
    let (inner_w, inner_h) = if font_w >= font_h {
        (res, ((font_h / font_w) * res as f32).ceil() as usize)
    } else {
        (((font_w / font_h) * res as f32).ceil() as usize, res)
    };
    let inner_w = inner_w.max(2);
    let inner_h = inner_h.max(2);
    let bmp_w = inner_w + pad * 2;
    let bmp_h = inner_h + pad * 2;

    // Scale from font units to bitmap coordinates (based on inner area)
    let bitmap_scale = inner_w as f32 / font_w;
    let offset_x = -font_min_x * bitmap_scale + pad as f32;
    let offset_y = -font_min_y * bitmap_scale + pad as f32;

    // Transform edges from font coordinates to bitmap coordinates
    let bmp_edges: Vec<(f32, f32, f32, f32)> = collector
        .edges
        .iter()
        .map(|&(x0, y0, x1, y1)| {
            (
                x0 * bitmap_scale + offset_x,
                y0 * bitmap_scale + offset_y,
                x1 * bitmap_scale + offset_x,
                y1 * bitmap_scale + offset_y,
            )
        })
        .collect();

    // Compute SDF directly from outline edges — sub-pixel accurate,
    // no binary rasterization staircase artifacts.
    let sdf_grid = compute_sdf_from_edges(&bmp_edges, bmp_w, bmp_h);

    // World-space bounds — expanded by the padding so that sample()
    // maps world coords correctly into the padded grid.
    let pad_world = pad as f32 / bitmap_scale * scale_to_world;
    let world_min_x = font_min_x * scale_to_world - pad_world;
    let world_min_y = font_min_y * scale_to_world - pad_world;
    let world_max_x = font_max_x * scale_to_world + pad_world;
    let world_max_y = font_max_y * scale_to_world + pad_world;

    let glyph_sdf = Arc::new(GlyphSdf {
        grid: sdf_grid,
        width: bmp_w,
        height: bmp_h,
        advance,
        lsb,
        bounds: (world_min_x, world_min_y, world_max_x, world_max_y),
    });

    insert_cached_glyph(key, Arc::clone(&glyph_sdf));
    Some(glyph_sdf)
}

// ── TextSdfLayout ───────────────────────────────────────────────────

/// A laid-out line of 3D text: positioned glyphs along a baseline.
#[derive(Debug, Clone)]
pub struct TextSdfLayout {
    /// Each positioned glyph with its x offset.
    pub glyphs: Vec<(Arc<GlyphSdf>, f32)>,
    /// Total advance width of the text.
    pub total_width: f32,
    /// Maximum ascent above baseline.
    pub ascent: f32,
    /// Maximum descent below baseline (positive downward).
    pub descent: f32,
}

/// Lay out a string of text, building SDF grids for each glyph.
///
/// Returns `None` if the font cannot be parsed or contains no supported glyphs.
pub fn layout_text(
    font_bytes: &[u8],
    text: &str,
    font_size: f32,
    letter_spacing: f32,
    grid_resolution: u32,
) -> Option<Arc<TextSdfLayout>> {
    let face = ttf_parser::Face::parse(font_bytes, 0).ok()?;
    let font_ptr = font_bytes.as_ptr() as usize;

    let units_per_em = face.units_per_em() as f32;
    let scale = font_size / units_per_em;

    let ascent = face.ascender() as f32 * scale;
    let descent = -(face.descender() as f32 * scale); // make positive

    let mut glyphs = Vec::new();
    let mut cursor_x = 0.0_f32;

    for ch in text.chars() {
        let glyph_id = face.glyph_index(ch)?;

        if let Some(glyph_sdf) = build_glyph_sdf(&face, font_ptr, glyph_id, font_size, grid_resolution) {
            glyphs.push((glyph_sdf.clone(), cursor_x));
            cursor_x += glyph_sdf.advance + letter_spacing;
        } else {
            // Space or unsupported glyph — advance by a standard width
            let advance = face
                .glyph_hor_advance(glyph_id)
                .map_or(font_size * 0.25, |a| a as f32 * scale);
            cursor_x += advance + letter_spacing;
        }
    }

    Some(Arc::new(TextSdfLayout {
        glyphs,
        total_width: cursor_x - letter_spacing.max(0.0),
        ascent,
        descent,
    }))
}

// ── 3D SDF evaluation ───────────────────────────────────────────────

/// Evaluate the 3D extruded text SDF.
///
/// The text is centered at the origin, extruded along the Z axis by `depth`.
/// Uses the standard IQ extrusion formula:
/// ```text
/// d2d = min(glyph SDFs at (p.x, p.y))
/// dz  = |p.z| - depth/2
/// dist = length(max(vec2(d2d, dz), 0)) + min(max(d2d, dz), 0)
/// ```
pub fn sd_text3d(layout: &TextSdfLayout, p: Vec3, depth: f32) -> f32 {
    // Center the text: shift x so text is centered, y relative to baseline center
    let center_x = layout.total_width * 0.5;
    let center_y = (layout.ascent - layout.descent) * 0.5;
    let sample_x = p.x + center_x;
    let sample_y = p.y + center_y;

    // Find minimum 2D distance across all glyphs
    let mut d2d = f32::MAX;
    for (glyph, x_offset) in &layout.glyphs {
        let gx = sample_x - x_offset;
        let d = glyph.sample(gx, sample_y);
        d2d = d2d.min(d);
    }

    // Extrude along Z
    let dz = p.z.abs() - depth * 0.5;

    // IQ extrusion formula
    let w = Vec2::new(d2d, dz).max_comp(Vec2::ZERO);
    w.length() + d2d.max(dz).min(0.0)
}

/// Estimate the analytical surface normal for an extruded `Text3D` shape.
///
/// Uses central finite differences on the bilinear-interpolated 2D SDF
/// with a step size of ~2 grid cells. This naturally smooths across cell
/// boundaries, avoiding the moiré from bilinear gradient discontinuities.
pub fn estimate_text3d_normal(layout: &TextSdfLayout, p: Vec3, depth: f32) -> Vec3 {
    let center_x = layout.total_width * 0.5;
    let center_y = (layout.ascent - layout.descent) * 0.5;
    let sample_x = p.x + center_x;
    let sample_y = p.y + center_y;

    // Find closest glyph
    let mut d2d = f32::MAX;
    let mut best_glyph: Option<&GlyphSdf> = None;
    let mut best_gx = 0.0_f32;
    for (glyph, x_offset) in &layout.glyphs {
        let gx = sample_x - x_offset;
        let d = glyph.sample(gx, sample_y);
        if d < d2d {
            d2d = d;
            best_glyph = Some(glyph);
            best_gx = gx;
        }
    }

    let dz = p.z.abs() - depth * 0.5;
    let sign_z = if p.z >= 0.0 { 1.0 } else { -1.0 };

    // Central differences on the closest glyph's interpolated SDF
    let (dx, dy) = if let Some(glyph) = best_glyph {
        let (min_x, _, max_x, _) = glyph.bounds;
        let bw = max_x - min_x;
        // Step size: half grid cell in world units (matches GPU shader)
        let h = 0.5 * bw / glyph.width as f32;
        let dx = glyph.sample(best_gx + h, sample_y) - glyph.sample(best_gx - h, sample_y);
        let dy = glyph.sample(best_gx, sample_y + h) - glyph.sample(best_gx, sample_y - h);
        (dx, dy)
    } else {
        (0.0, 1.0)
    };

    // Normalize the 2D gradient direction
    let grad_len = dx.hypot(dy);
    let grad_dir = if grad_len > 1e-6 {
        (dx / grad_len, dy / grad_len)
    } else {
        (0.0, 1.0)
    };

    // Smooth blend between face normal and side normal
    let diff = d2d - dz;
    let t = ((diff + 0.03) / 0.06).clamp(0.0, 1.0);
    let blend = t * t * (3.0 - 2.0 * t); // smoothstep

    let face_n = Vec3::new(grad_dir.0, grad_dir.1, 0.0);
    let side_n = Vec3::new(0.0, 0.0, sign_z);

    let n = Vec3::new(
        side_n.x + (face_n.x - side_n.x) * blend,
        side_n.y + (face_n.y - side_n.y) * blend,
        side_n.z + (face_n.z - side_n.z) * blend,
    );
    n.normalize()
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edt_1d_basic() {
        // Distance from a single boundary point at index 2
        let f = [1e6, 1e6, 0.0, 1e6, 1e6];
        let mut dt = vec![0.0; 5];
        edt_1d(&f, &mut dt);

        // Squared distances: 4, 1, 0, 1, 4
        assert!((dt[0] - 4.0).abs() < 0.01);
        assert!((dt[1] - 1.0).abs() < 0.01);
        assert!(dt[2].abs() < 0.01);
        assert!((dt[3] - 1.0).abs() < 0.01);
        assert!((dt[4] - 4.0).abs() < 0.01);
    }

    #[test]
    fn edt_1d_empty() {
        let f: [f32; 0] = [];
        let mut dt: [f32; 0] = [];
        edt_1d(&f, &mut dt);
    }

    #[test]
    fn edt_1d_single() {
        let f = [42.0];
        let mut dt = [0.0];
        edt_1d(&f, &mut dt);
        assert!((dt[0] - 42.0).abs() < 0.01);
    }

    #[test]
    fn compute_sdf_simple_square() {
        // A 6x6 bitmap with a 2x2 filled square in the center
        let mut bitmap = vec![false; 36];
        bitmap[2 * 6 + 2] = true;
        bitmap[2 * 6 + 3] = true;
        bitmap[3 * 6 + 2] = true;
        bitmap[3 * 6 + 3] = true;

        let sdf = compute_sdf(&bitmap, 6, 6);

        // Inside the square: should be negative
        assert!(sdf[2 * 6 + 2] < 0.0, "inside should be negative");
        // Outside the square: should be positive
        assert!(sdf[0] > 0.0, "outside should be positive");
    }

    #[test]
    fn vec2_basics() {
        let a = Vec2::new(3.0, 4.0);
        assert!((a.length() - 5.0).abs() < 1e-5);

        let b = Vec2::new(1.0, 2.0);
        let sum = a + b;
        assert!((sum.x - 4.0).abs() < 1e-5);
        assert!((sum.y - 6.0).abs() < 1e-5);
    }

    #[test]
    fn glyph_sdf_sample_outside_bounds() {
        let sdf = GlyphSdf {
            grid: vec![0.0; 4],
            width: 2,
            height: 2,
            advance: 1.0,
            lsb: 0.0,
            bounds: (0.0, 0.0, 1.0, 1.0),
        };
        // Far outside should return large positive
        let d = sdf.sample(100.0, 100.0);
        assert!(d > 10.0);
    }
}
