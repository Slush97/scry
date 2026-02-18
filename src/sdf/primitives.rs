// SPDX-License-Identifier: MIT OR Apache-2.0
//! Signed distance functions for basic shapes and domain combinators.

use super::math::Vec3;

// ── Primitive shapes ────────────────────────────────────────────────

/// Sphere centered at origin with given `radius`.
#[inline]
pub fn sd_sphere(p: Vec3, radius: f32) -> f32 {
    p.length() - radius
}

/// Infinite ground plane at y = 0 (normal pointing up).
#[inline]
pub fn sd_plane(p: Vec3) -> f32 {
    p.y
}

/// Axis-aligned box centered at origin with given `half_extents`.
#[inline]
pub fn sd_box(p: Vec3, half_extents: Vec3) -> f32 {
    let q = p.abs() - half_extents;
    let outside = q.max_comp(Vec3::ZERO).length();
    let inside = q.max_element().min(0.0);
    outside + inside
}

/// Torus centered at origin in the XZ plane.
#[inline]
pub fn sd_torus(p: Vec3, major: f32, minor: f32) -> f32 {
    let q_x = p.length_xz() - major;
    let q_y = p.y;
    q_x.hypot(q_y) - minor
}

/// Cylinder along the Y axis centered at origin.
#[inline]
pub fn sd_cylinder(p: Vec3, radius: f32, half_height: f32) -> f32 {
    let d_x = p.length_xz() - radius;
    let d_y = p.y.abs() - half_height;
    let outside = f32::max(d_x, 0.0).hypot(f32::max(d_y, 0.0));
    let inside = f32::max(d_x, d_y).min(0.0);
    outside + inside
}

/// Vertical capsule (line segment from −half_height to +half_height, swept by radius).
#[inline]
pub fn sd_capsule(p: Vec3, radius: f32, half_height: f32) -> f32 {
    let py = p.y - p.y.clamp(-half_height, half_height);
    let q = Vec3::new(p.x, py, p.z);
    q.length() - radius
}

/// Axis-aligned box with rounded edges.
#[inline]
pub fn sd_rounded_box(p: Vec3, half_extents: Vec3, radius: f32) -> f32 {
    sd_box(p, half_extents) - radius
}

/// Cone along the Y axis with tip at `(0, height, 0)` and base radius at `y=0`.
///
/// Uses the standard IQ bound-cone SDF (2D projection approach).
#[inline]
pub fn sd_cone(p: Vec3, radius: f32, height: f32) -> f32 {
    // 2D projection: (distance from axis, height)
    let q_len = p.length_xz();
    let q = (q_len, p.y);

    // Cone axis direction in 2D  (normalized)
    let tip = (0.0_f32, height);
    let base = (radius, 0.0_f32);
    // Vector along the cone surface: base - tip
    let cb = (base.0 - tip.0, base.1 - tip.1);
    let cb_len_sq = cb.0 * cb.0 + cb.1 * cb.1;

    // Project (q - tip) onto the cone surface edge
    let qp = (q.0 - tip.0, q.1 - tip.1);
    let t = (qp.0 * cb.0 + qp.1 * cb.1) / cb_len_sq;
    let t = t.clamp(0.0, 1.0);

    // Closest point on the cone surface edge
    let closest = (tip.0 + cb.0 * t, tip.1 + cb.1 * t);
    let dx = q.0 - closest.0;
    let dy = q.1 - closest.1;
    let dist_to_edge = (dx * dx + dy * dy).sqrt();

    // Also check distance to the base cap
    let base_dist = {
        let dy_base = -q.1; // distance below y=0
        let dr_base = (q.0 - radius).max(0.0);
        if q.1 < 0.0 {
            (dy_base * dy_base + dr_base * dr_base).sqrt()
        } else if q.0 > radius && q.1 < 0.01 {
            dr_base
        } else {
            f32::MAX
        }
    };

    let d = dist_to_edge.min(base_dist);

    // Sign: inside if below the slant line and above the base
    let cross = cb.0 * qp.1 - cb.1 * qp.0; // 2D cross product
    if cross <= 0.0 && q.1 >= 0.0 && q.1 <= height {
        -d
    } else {
        d
    }
}

// ── Combinators ─────────────────────────────────────────────────────

/// Boolean union (nearest of two surfaces).
#[inline]
pub fn op_union(d1: f32, d2: f32) -> f32 {
    d1.min(d2)
}

/// Boolean subtraction (carve `d2` from `d1`).
#[inline]
pub fn op_subtract(d1: f32, d2: f32) -> f32 {
    f32::max(d1, -d2)
}

/// Boolean intersection (region inside both).
#[inline]
pub fn op_intersect(d1: f32, d2: f32) -> f32 {
    f32::max(d1, d2)
}

/// Polynomial smooth minimum for organic blending.
///
/// `k` controls the blend radius (typical 0.1–1.0).
#[inline]
pub fn smooth_min(d1: f32, d2: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (d2 - d1) / k).clamp(0.0, 1.0);
    d2 + (d1 - d2) * h - k * h * (1.0 - h)
}

/// Round any SDF by subtracting a radius (inflates the shape).
#[inline]
pub fn op_round(d: f32, radius: f32) -> f32 {
    d - radius
}

/// Hollow out any SDF to create a shell of the given thickness.
#[inline]
pub fn op_onion(d: f32, thickness: f32) -> f32 {
    d.abs() - thickness
}

// ── Domain transforms ───────────────────────────────────────────────

/// Translate the SDF evaluation point (shifts the shape).
#[inline]
pub fn translate(p: Vec3, offset: Vec3) -> Vec3 {
    p - offset
}

/// Infinite repetition along all three axes with given `period`.
#[inline]
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

    #[test]
    fn capsule_distances() {
        // Center of capsule → inside (negative)
        assert!(sd_capsule(Vec3::ZERO, 1.0, 1.0) < 0.0);
        // On surface at equator → zero
        assert!((sd_capsule(Vec3::new(1.0, 0.0, 0.0), 1.0, 1.0)).abs() < 1e-6);
        // Outside → positive
        assert!((sd_capsule(Vec3::new(2.0, 0.0, 0.0), 1.0, 1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rounded_box_distances() {
        let half = Vec3::new(1.0, 1.0, 1.0);
        // Center → inside (more negative than unrounded box by radius)
        assert!(sd_rounded_box(Vec3::ZERO, half, 0.1) < sd_box(Vec3::ZERO, half));
        // Outside → positive but offset
        let outside = sd_rounded_box(Vec3::new(2.0, 0.0, 0.0), half, 0.1);
        assert!((outside - (1.0 - 0.1)).abs() < 1e-6);
    }

    #[test]
    fn cone_distances() {
        // Tip of cone at (0, height, 0) should be ~zero
        let at_tip = sd_cone(Vec3::new(0.0, 2.0, 0.0), 1.0, 2.0);
        assert!(at_tip.abs() < 0.1, "tip distance = {at_tip}");
        // Center base should be inside
        assert!(sd_cone(Vec3::new(0.0, 0.5, 0.0), 1.0, 2.0) < 0.0);
    }

    #[test]
    fn op_round_shrinks() {
        assert!((op_round(1.0, 0.1) - 0.9).abs() < 1e-6);
    }

    #[test]
    fn op_onion_makes_shell() {
        // Inside the original shape (d = -0.5) becomes positive shell
        let d = op_onion(-0.5, 0.1);
        assert!((d - 0.4).abs() < 1e-6);
    }
}
