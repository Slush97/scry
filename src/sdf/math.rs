// SPDX-License-Identifier: MIT OR Apache-2.0
//! Minimal 3D vector math for ray marching.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// A 3-component `f32` vector used for positions, directions, and colors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
}

impl Vec3 {
    /// Zero vector.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Unit Y (up).
    pub const UP: Self = Self {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    /// Create a new vector.
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Dot product.
    #[inline]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Cross product.
    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    /// Squared Euclidean length (avoids sqrt).
    #[inline]
    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length.
    #[inline]
    pub fn length(self) -> f32 {
        self.length_sq().sqrt()
    }

    /// Normalize to unit length. Returns zero vector if length is near zero.
    #[inline]
    pub fn normalize(self) -> Self {
        let len_sq = self.length_sq();
        if len_sq < 1e-20 {
            Self::ZERO
        } else {
            self * (1.0 / len_sq.sqrt())
        }
    }

    /// Reflect direction `self` around normal `n`.
    #[inline]
    pub fn reflect(self, n: Self) -> Self {
        self - n * (2.0 * self.dot(n))
    }

    /// Refract direction `self` through surface with normal `n` and IOR ratio `eta`.
    /// Returns `None` for total internal reflection.
    #[inline]
    pub fn refract(self, n: Self, eta: f32) -> Option<Self> {
        let cos_i = -self.dot(n);
        let sin2_t = eta * eta * (1.0 - cos_i * cos_i);
        if sin2_t > 1.0 {
            return None; // Total internal reflection
        }
        let cos_t = (1.0 - sin2_t).sqrt();
        Some(self * eta + n * (eta * cos_i - cos_t))
    }

    /// Component-wise absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        }
    }

    /// Component-wise maximum.
    #[inline]
    pub fn max_comp(self, rhs: Self) -> Self {
        Self {
            x: self.x.max(rhs.x),
            y: self.y.max(rhs.y),
            z: self.z.max(rhs.z),
        }
    }

    /// Largest component.
    #[inline]
    pub fn max_element(self) -> f32 {
        self.x.max(self.y).max(self.z)
    }

    /// Component-wise minimum with a scalar.
    #[inline]
    pub fn min_scalar(self, v: f32) -> Self {
        Self {
            x: self.x.min(v),
            y: self.y.min(v),
            z: self.z.min(v),
        }
    }

    /// Length of the XZ projection.
    #[inline]
    pub fn length_xz(self) -> f32 {
        self.x.hypot(self.z)
    }
}

/// Fast reciprocal square root (Quake III style + two Newton–Raphson iterations).
///
/// Two iterations give ~46-bit accuracy — effectively full `f32` precision
/// while still avoiding the division in `1.0 / sqrt(x)`.
#[inline]
pub fn fast_inv_sqrt(x: f32) -> f32 {
    let half = 0.5 * x;
    let i = f32::to_bits(x);
    let i = 0x5f37_59df - (i >> 1); // magic constant
    let mut y = f32::from_bits(i);
    y = y * (1.5 - half * y * y); // first Newton–Raphson iteration
    y = y * (1.5 - half * y * y); // second iteration for full f32 precision
    y
}

/// Compute a camera basis from eye/target/up.
///
/// Returns `(right, up, forward)` orthonormal vectors.
pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> (Vec3, Vec3, Vec3) {
    let forward = (target - eye).normalize();
    let right = forward.cross(up).normalize();
    let cam_up = right.cross(forward);
    (right, cam_up, forward)
}

// ── Operator impls ──────────────────────────────────────────────────

impl Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;
    #[inline]
    fn mul(self, v: Vec3) -> Vec3 {
        v * self
    }
}

impl Div<f32> for Vec3 {
    type Output = Self;
    #[inline]
    fn div(self, s: f32) -> Self {
        self * (1.0 / s)
    }
}

impl Neg for Vec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

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
