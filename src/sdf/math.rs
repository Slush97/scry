// SPDX-License-Identifier: MIT OR Apache-2.0
//! Minimal 3D vector math for ray marching.
//!
//! All types are now defined in [`crate::math3d`] and re-exported here
//! for backwards compatibility — existing `use super::math::*` imports
//! in sibling SDF modules continue to work unchanged.

pub use crate::math3d::{fast_inv_sqrt, look_at, Vec3};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec3_basic_arithmetic() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);

        let sum = a + b;
        assert!((sum.x - 5.0).abs() < 1e-6);
        assert!((sum.y - 7.0).abs() < 1e-6);
        assert!((sum.z - 9.0).abs() < 1e-6);

        let diff = b - a;
        assert!((diff.x - 3.0).abs() < 1e-6);

        let scaled = a * 2.0;
        assert!((scaled.x - 2.0).abs() < 1e-6);

        let neg = -a;
        assert!((neg.x + 1.0).abs() < 1e-6);
    }

    #[test]
    fn vec3_dot_cross() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);

        assert!((a.dot(b)).abs() < 1e-6);

        let c = a.cross(b);
        assert!((c.x).abs() < 1e-6);
        assert!((c.y).abs() < 1e-6);
        assert!((c.z - 1.0).abs() < 1e-6);
    }

    #[test]
    fn vec3_normalize() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let n = v.normalize();
        assert!((n.length() - 1.0).abs() < 1e-6);
        assert!((n.x - 0.6).abs() < 1e-6);
        assert!((n.y - 0.8).abs() < 1e-6);

        // Zero vector stays zero
        let z = Vec3::ZERO.normalize();
        assert!((z.length()).abs() < 1e-6);
    }

    #[test]
    fn vec3_reflect() {
        // Reflect (1, -1, 0) off horizontal normal (0, 1, 0) → (1, 1, 0)
        let d = Vec3::new(1.0, -1.0, 0.0).normalize();
        let n = Vec3::new(0.0, 1.0, 0.0);
        let r = d.reflect(n);
        let expected = Vec3::new(1.0, 1.0, 0.0).normalize();
        assert!((r.x - expected.x).abs() < 1e-5);
        assert!((r.y - expected.y).abs() < 1e-5);
    }

    #[test]
    fn vec3_refract() {
        let d = Vec3::new(0.0, -1.0, 0.0); // straight down
        let n = Vec3::new(0.0, 1.0, 0.0); // surface normal up
        let r = d.refract(n, 1.0).unwrap(); // eta=1 → no bend
        assert!((r.x).abs() < 1e-5);
        assert!((r.y + 1.0).abs() < 1e-5);
    }

    #[test]
    fn look_at_produces_orthonormal_basis() {
        let eye = Vec3::new(0.0, 2.0, 5.0);
        let target = Vec3::ZERO;
        let (right, up, fwd) = look_at(eye, target, Vec3::UP);

        // All unit length
        assert!((right.length() - 1.0).abs() < 1e-5);
        assert!((up.length() - 1.0).abs() < 1e-5);
        assert!((fwd.length() - 1.0).abs() < 1e-5);

        // Mutually perpendicular
        assert!((right.dot(up)).abs() < 1e-5);
        assert!((right.dot(fwd)).abs() < 1e-5);
        assert!((up.dot(fwd)).abs() < 1e-5);
    }
}
