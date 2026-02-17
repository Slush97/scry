// SPDX-License-Identifier: MIT OR Apache-2.0
//! Signed distance functions for basic shapes and domain combinators.

use super::math::Vec3;

// ── Primitive shapes ────────────────────────────────────────────────

/// Sphere centered at origin with given `radius`.
pub fn sd_sphere(p: Vec3, radius: f32) -> f32 {
    p.length() - radius
}

/// Infinite ground plane at y = 0 (normal pointing up).
pub fn sd_plane(p: Vec3) -> f32 {
    p.y
}

/// Axis-aligned box centered at origin with given `half_extents`.
pub fn sd_box(p: Vec3, half_extents: Vec3) -> f32 {
    let q = p.abs() - half_extents;
    let outside = q.max_comp(Vec3::ZERO).length();
    let inside = q.max_element().min(0.0);
    outside + inside
}

/// Torus centered at origin in the XZ plane.
pub fn sd_torus(p: Vec3, major: f32, minor: f32) -> f32 {
    let q_x = p.length_xz() - major;
    let q_y = p.y;
    q_x.hypot(q_y) - minor
}

/// Cylinder along the Y axis centered at origin.
pub fn sd_cylinder(p: Vec3, radius: f32, half_height: f32) -> f32 {
    let d_x = p.length_xz() - radius;
    let d_y = p.y.abs() - half_height;
    let outside = f32::max(d_x, 0.0).hypot(f32::max(d_y, 0.0));
    let inside = f32::max(d_x, d_y).min(0.0);
    outside + inside
}

// ── Combinators ─────────────────────────────────────────────────────

/// Boolean union (nearest of two surfaces).
pub fn op_union(d1: f32, d2: f32) -> f32 {
    d1.min(d2)
}

/// Boolean subtraction (carve `d2` from `d1`).
pub fn op_subtract(d1: f32, d2: f32) -> f32 {
    f32::max(d1, -d2)
}

/// Boolean intersection (region inside both).
pub fn op_intersect(d1: f32, d2: f32) -> f32 {
    f32::max(d1, d2)
}

/// Polynomial smooth minimum for organic blending.
///
/// `k` controls the blend radius (typical 0.1–1.0).
pub fn smooth_min(d1: f32, d2: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (d2 - d1) / k).clamp(0.0, 1.0);
    d2 + (d1 - d2) * h - k * h * (1.0 - h)
}

// ── Domain transforms ───────────────────────────────────────────────

/// Translate the SDF evaluation point (shifts the shape).
pub fn translate(p: Vec3, offset: Vec3) -> Vec3 {
    p - offset
}

/// Infinite repetition along all three axes with given `period`.
pub fn repeat(p: Vec3, period: Vec3) -> Vec3 {
    Vec3::new(
        p.x - (p.x / period.x).round() * period.x,
        p.y - (p.y / period.y).round() * period.y,
        p.z - (p.z / period.z).round() * period.z,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_distances() {
        // Center of sphere → negative (inside)
        assert!((sd_sphere(Vec3::ZERO, 1.0) - (-1.0)).abs() < 1e-6);
        // On surface → zero
        assert!((sd_sphere(Vec3::new(1.0, 0.0, 0.0), 1.0)).abs() < 1e-6);
        // Outside → positive
        assert!((sd_sphere(Vec3::new(2.0, 0.0, 0.0), 1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn plane_distances() {
        assert!((sd_plane(Vec3::new(0.0, 0.0, 0.0))).abs() < 1e-6);
        assert!((sd_plane(Vec3::new(0.0, 1.0, 0.0)) - 1.0).abs() < 1e-6);
        assert!((sd_plane(Vec3::new(0.0, -0.5, 0.0)) - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn box_distances() {
        let half = Vec3::new(1.0, 1.0, 1.0);
        // Center → inside
        assert!(sd_box(Vec3::ZERO, half) < 0.0);
        // On face → zero
        assert!((sd_box(Vec3::new(1.0, 0.0, 0.0), half)).abs() < 1e-6);
        // Outside → positive
        assert!((sd_box(Vec3::new(2.0, 0.0, 0.0), half) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn smooth_min_blends() {
        // When k=0, should behave like min
        let d = smooth_min(1.0, 2.0, 0.001);
        assert!((d - 1.0).abs() < 0.01);

        // With large k, result should be less than min
        let d = smooth_min(1.0, 1.0, 1.0);
        assert!(d < 1.0);
    }

    #[test]
    fn translate_shifts_point() {
        let p = translate(Vec3::new(3.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!((p.x - 2.0).abs() < 1e-6);
    }
}
