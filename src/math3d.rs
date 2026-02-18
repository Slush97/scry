// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared 3D math primitives: vectors, quaternions, matrices.
//!
//! This module is the single source of truth for 3D math types used by both
//! the SDF ray marcher (`sdf`) and the chart 3D system (`scry-chart::chart3d`).
//! It is always available (no feature flag required).

use std::ops::{Add, Div, Mul, Neg, Sub};

// ── Vec3 ─────────────────────────────────────────────────────────────

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

    /// Unit vector along the X axis.
    pub const X: Self = Self {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };

    /// Unit vector along the Y axis.
    pub const Y: Self = Self {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    /// Unit vector along the Z axis.
    pub const Z: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    /// Unit Y (up) — alias for [`Y`](Self::Y).
    pub const UP: Self = Self::Y;

    /// Create a new vector.
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Scalar multiplication (same as `self * s`).
    #[inline]
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
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
    #[must_use]
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

// ── Vec3 operator impls ──────────────────────────────────────────────

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

// ── Quaternion ───────────────────────────────────────────────────────

/// A unit quaternion representing a 3D rotation.
///
/// Stored as `(w, x, y, z)` where `w` is the scalar part.
/// Using quaternions avoids gimbal lock and provides smooth interpolation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quaternion {
    /// Scalar part.
    pub w: f32,
    /// X component of the vector part.
    pub x: f32,
    /// Y component of the vector part.
    pub y: f32,
    /// Z component of the vector part.
    pub z: f32,
}

impl Quaternion {
    /// The identity rotation (no rotation).
    pub const IDENTITY: Self = Self {
        w: 1.0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Create a quaternion from an axis and angle (radians).
    ///
    /// The axis is normalized internally.
    #[must_use]
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let axis = axis.normalize();
        let half = angle * 0.5;
        let (s, c) = half.sin_cos();
        Self {
            w: c,
            x: axis.x * s,
            y: axis.y * s,
            z: axis.z * s,
        }
    }

    /// Normalize to unit quaternion.
    #[must_use]
    pub fn normalize(self) -> Self {
        let len = (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if len < 1e-10 {
            Self::IDENTITY
        } else {
            let inv = 1.0 / len;
            Self {
                w: self.w * inv,
                x: self.x * inv,
                y: self.y * inv,
                z: self.z * inv,
            }
        }
    }

    /// Conjugate (inverse for unit quaternions).
    #[must_use]
    pub fn conjugate(self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Rotate a vector by this quaternion.
    ///
    /// Computes `q * v * q⁻¹` (Hamilton product with pure quaternion).
    #[must_use]
    pub fn rotate_vec3(self, v: Vec3) -> Vec3 {
        let qv = Self {
            w: 0.0,
            x: v.x,
            y: v.y,
            z: v.z,
        };
        let result = (self * qv) * self.conjugate();
        Vec3::new(result.x, result.y, result.z)
    }

    /// Convert to a 4×4 rotation matrix (column-major, suitable for OpenGL conventions).
    #[must_use]
    pub fn to_rotation_matrix(self) -> Mat4 {
        let Self { w, x, y, z } = self;
        let x2 = x + x;
        let y2 = y + y;
        let z2 = z + z;
        let xx = x * x2;
        let xy = x * y2;
        let xz = x * z2;
        let yy = y * y2;
        let yz = y * z2;
        let zz = z * z2;
        let wx = w * x2;
        let wy = w * y2;
        let wz = w * z2;

        [
            [1.0 - (yy + zz), xy + wz, xz - wy, 0.0],
            [xy - wz, 1.0 - (xx + zz), yz + wx, 0.0],
            [xz + wy, yz - wx, 1.0 - (xx + yy), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }
}

impl Mul for Quaternion {
    type Output = Self;
    /// Quaternion multiplication (composition of rotations).
    ///
    /// `self * other` applies `other` first, then `self`.
    fn mul(self, other: Self) -> Self {
        Self {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }
}

// ── Mat4 ─────────────────────────────────────────────────────────────

/// 4×4 matrix type (row-major).
pub type Mat4 = [[f32; 4]; 4];

/// Multiply two 4×4 matrices.
#[must_use]
pub fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [[0.0_f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] =
                a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
        }
    }
    out
}

/// Multiply a 4×4 matrix by a 4-element vector.
#[must_use]
pub fn mat4_mul_vec4(m: &Mat4, v: [f32; 4]) -> [f32; 4] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + m[0][3] * v[3],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + m[1][3] * v[3],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + m[2][3] * v[3],
        m[3][0] * v[0] + m[3][1] * v[1] + m[3][2] * v[2] + m[3][3] * v[3],
    ]
}

/// Identity 4×4 matrix.
#[must_use]
pub fn mat4_identity() -> Mat4 {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

// ── Free functions ───────────────────────────────────────────────────

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

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    fn vec3_approx_eq(a: Vec3, b: Vec3) -> bool {
        approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z)
    }

    #[test]
    fn vec3_basic_arithmetic() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);

        let sum = a + b;
        assert!(approx_eq(sum.x, 5.0));
        assert!(approx_eq(sum.y, 7.0));
        assert!(approx_eq(sum.z, 9.0));

        let diff = b - a;
        assert!(approx_eq(diff.x, 3.0));

        let scaled = a * 2.0;
        assert!(approx_eq(scaled.x, 2.0));

        let neg = -a;
        assert!(approx_eq(neg.x, -1.0));
    }

    #[test]
    fn vec3_dot_cross() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);

        assert!(a.dot(b).abs() < 1e-6);

        let c = a.cross(b);
        assert!(approx_eq(c.x, 0.0));
        assert!(approx_eq(c.y, 0.0));
        assert!(approx_eq(c.z, 1.0));
    }

    #[test]
    fn vec3_normalize() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let n = v.normalize();
        assert!(approx_eq(n.length(), 1.0));
        assert!(approx_eq(n.x, 0.6));
        assert!(approx_eq(n.y, 0.8));

        // Zero vector stays zero
        let z = Vec3::ZERO.normalize();
        assert!(z.length().abs() < 1e-6);
    }

    #[test]
    fn vec3_reflect() {
        let d = Vec3::new(1.0, -1.0, 0.0).normalize();
        let n = Vec3::new(0.0, 1.0, 0.0);
        let r = d.reflect(n);
        let expected = Vec3::new(1.0, 1.0, 0.0).normalize();
        assert!(approx_eq(r.x, expected.x));
        assert!(approx_eq(r.y, expected.y));
    }

    #[test]
    fn vec3_refract() {
        let d = Vec3::new(0.0, -1.0, 0.0);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let r = d.refract(n, 1.0).unwrap();
        assert!(approx_eq(r.x, 0.0));
        assert!(approx_eq(r.y, -1.0));
    }

    #[test]
    fn vec3_constants() {
        assert!(vec3_approx_eq(Vec3::UP, Vec3::Y));
        assert!(vec3_approx_eq(Vec3::X.cross(Vec3::Y), Vec3::Z));
    }

    #[test]
    fn vec3_scale_matches_mul() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert!(vec3_approx_eq(v.scale(2.0), v * 2.0));
    }

    #[test]
    fn look_at_produces_orthonormal_basis() {
        let eye = Vec3::new(0.0, 2.0, 5.0);
        let target = Vec3::ZERO;
        let (right, up, fwd) = look_at(eye, target, Vec3::UP);

        assert!(approx_eq(right.length(), 1.0));
        assert!(approx_eq(up.length(), 1.0));
        assert!(approx_eq(fwd.length(), 1.0));

        assert!(right.dot(up).abs() < 1e-5);
        assert!(right.dot(fwd).abs() < 1e-5);
        assert!(up.dot(fwd).abs() < 1e-5);
    }

    #[test]
    fn quaternion_identity_rotates_nothing() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let result = Quaternion::IDENTITY.rotate_vec3(v);
        assert!(vec3_approx_eq(v, result));
    }

    #[test]
    fn quaternion_90deg_y_rotation() {
        let q = Quaternion::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_2);
        let v = Vec3::new(1.0, 0.0, 0.0);
        let result = q.rotate_vec3(v);
        assert!(vec3_approx_eq(result, Vec3::new(0.0, 0.0, -1.0)));
    }

    #[test]
    fn quaternion_composition() {
        let q1 = Quaternion::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_2);
        let q2 = Quaternion::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_2);
        let combined = (q1 * q2).normalize();
        let v = Vec3::new(1.0, 0.0, 0.0);
        let result = combined.rotate_vec3(v);
        assert!(vec3_approx_eq(result, Vec3::new(-1.0, 0.0, 0.0)));
    }

    #[test]
    fn mat4_identity_mul() {
        let id = mat4_identity();
        let m = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ];
        let result = mat4_mul(&id, &m);
        assert_eq!(result, m);
    }
}
