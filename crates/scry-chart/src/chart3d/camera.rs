//! 3D camera with arcball quaternion rotation.
//!
//! Provides [`Camera3D`] for positioning and orienting a virtual camera in 3D
//! space, using quaternion-based rotation to avoid gimbal lock. Supports orbit,
//! pan, and zoom operations suitable for interactive 3D chart viewing.

// ---------------------------------------------------------------------------
// Vec3 — minimal 3D vector
// ---------------------------------------------------------------------------

/// A point or direction in 3D space.
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
    /// The zero vector.
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

    /// Create a new vector.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Scalar multiplication.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    /// Dot product.
    #[must_use]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product.
    #[must_use]
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Euclidean length.
    #[must_use]
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    /// Normalize to unit length. Returns zero vector if length is near zero.
    #[must_use]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < 1e-10 {
            Self::ZERO
        } else {
            self.scale(1.0 / len)
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

// ---------------------------------------------------------------------------
// Quaternion — unit quaternion for rotation
// ---------------------------------------------------------------------------

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
    pub fn to_rotation_matrix(self) -> [[f32; 4]; 4] {
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

impl std::ops::Mul for Quaternion {
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

// ---------------------------------------------------------------------------
// Camera3D
// ---------------------------------------------------------------------------

/// Minimum distance from camera to target (prevents zooming through target).
const MIN_DISTANCE: f32 = 0.5;
/// Maximum distance from camera to target.
const MAX_DISTANCE: f32 = 100.0;

/// A 3D camera with arcball rotation for interactive visualization.
///
/// The camera orbits around a target point using quaternion rotation.
/// It supports orbit (rotate around target), pan (shift both position and
/// target), and zoom (move along the look-at axis).
///
/// # Example
///
/// ```
/// use scry_chart::chart3d::camera::{Camera3D, Vec3};
///
/// let mut cam = Camera3D::new(
///     Vec3::new(0.0, 0.0, 5.0),  // position
///     Vec3::ZERO,                 // target
///     Vec3::Y,                    // up
/// );
///
/// cam.orbit(0.1, 0.05);  // rotate
/// cam.zoom(-0.5);         // zoom in
///
/// let view = cam.view_matrix();
/// ```
#[derive(Clone, Debug)]
pub struct Camera3D {
    /// Camera position in world space.
    position: Vec3,
    /// Point the camera looks at.
    target: Vec3,
    /// World up direction.
    up: Vec3,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Near clipping plane distance.
    pub near: f32,
    /// Far clipping plane distance.
    pub far: f32,
    /// Accumulated rotation as a unit quaternion.
    orientation: Quaternion,
    /// Distance from camera to target.
    distance: f32,
}

impl Camera3D {
    /// Create a new camera looking at `target` from `position`.
    #[must_use]
    pub fn new(position: Vec3, target: Vec3, up: Vec3) -> Self {
        let distance = (position - target).length().max(MIN_DISTANCE);
        Self {
            position,
            target,
            up: up.normalize(),
            fov_y: std::f32::consts::FRAC_PI_4, // 45°
            near: 0.1,
            far: 200.0,
            orientation: Quaternion::IDENTITY,
            distance,
        }
    }

    /// Create a default camera positioned to view a unit-scale scene.
    #[must_use]
    pub fn default_for_scene(center: Vec3, radius: f32) -> Self {
        let distance = radius * 2.5;
        let position = center + Vec3::new(0.0, distance * 0.4, distance);
        Self::new(position, center, Vec3::Y)
    }

    /// Create a camera orbiting a target at spherical coordinates.
    ///
    /// - `target` — the point to look at
    /// - `distance` — distance from the target
    /// - `azimuth` — horizontal angle in radians (0 = +Z axis, positive = counterclockwise)
    /// - `elevation` — vertical angle in radians (0 = horizontal, positive = above)
    ///
    /// This is useful for the inline `show()` loop where arrow keys
    /// increment azimuth/elevation each frame.
    #[must_use]
    pub fn orbiting(target: Vec3, distance: f32, azimuth: f32, elevation: f32) -> Self {
        let dist = distance.max(MIN_DISTANCE);
        let (sa, ca) = azimuth.sin_cos();
        let (se, ce) = elevation.sin_cos();
        let position = target
            + Vec3::new(
                dist * ce * sa,
                dist * se,
                dist * ce * ca,
            );
        Self::new(position, target, Vec3::Y)
    }

    /// Orbit the camera around the target point.
    ///
    /// `dx` rotates around the world Y axis (horizontal drag).
    /// `dy` rotates around the camera's local right axis (vertical drag).
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        // Horizontal rotation: around world Y
        let yaw = Quaternion::from_axis_angle(Vec3::Y, -dx);
        // Vertical rotation: around camera's local right axis
        let right = self.right();
        let pitch = Quaternion::from_axis_angle(right, -dy);

        self.orientation = (yaw * pitch * self.orientation).normalize();
        self.update_position();
    }

    /// Zoom by moving along the look-at axis.
    ///
    /// Negative `delta` zooms in, positive zooms out.
    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance + delta).clamp(MIN_DISTANCE, MAX_DISTANCE);
        self.update_position();
    }

    /// Pan the camera and target by screen-space deltas.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let right = self.right().scale(dx);
        let up = self.up_dir().scale(dy);
        let offset = right + up;
        self.position = self.position + offset;
        self.target = self.target + offset;
    }

    /// Get the camera position.
    #[must_use]
    pub fn position(&self) -> Vec3 {
        self.position
    }

    /// Get the camera target.
    #[must_use]
    pub fn target(&self) -> Vec3 {
        self.target
    }

    /// Forward direction (from camera toward target).
    #[must_use]
    pub fn forward(&self) -> Vec3 {
        (self.target - self.position).normalize()
    }

    /// Right direction (perpendicular to forward and up).
    #[must_use]
    pub fn right(&self) -> Vec3 {
        self.forward().cross(self.up).normalize()
    }

    /// Up direction derived from forward and right (not world up).
    #[must_use]
    pub fn up_dir(&self) -> Vec3 {
        let right = self.right();
        right.cross(self.forward()).normalize()
    }

    /// Compute the 4×4 view matrix (world → camera space).
    ///
    /// Uses the standard look-at construction in row-major layout,
    /// consistent with [`mat4_mul_vec4`](crate::chart3d::projection::mat4_mul_vec4).
    #[must_use]
    pub fn view_matrix(&self) -> [[f32; 4]; 4] {
        let f = self.forward();
        let r = self.right();
        let u = self.up_dir();

        let tx = -r.dot(self.position);
        let ty = -u.dot(self.position);
        let tz = f.dot(self.position);

        [
            [r.x, r.y, r.z, tx],
            [u.x, u.y, u.z, ty],
            [-f.x, -f.y, -f.z, tz],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    /// Update position from orientation and distance.
    fn update_position(&mut self) {
        // Default camera direction is along +Z
        let offset = self.orientation.rotate_vec3(Vec3::Z).scale(self.distance);
        self.position = self.target + offset;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
    fn quaternion_identity_rotates_nothing() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let result = Quaternion::IDENTITY.rotate_vec3(v);
        assert!(vec3_approx_eq(v, result), "identity should not change vector");
    }

    #[test]
    fn quaternion_90deg_y_rotation() {
        let q = Quaternion::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_2);
        let v = Vec3::new(1.0, 0.0, 0.0);
        let result = q.rotate_vec3(v);
        // Rotating X-axis 90° around Y should give -Z
        assert!(
            vec3_approx_eq(result, Vec3::new(0.0, 0.0, -1.0)),
            "90° Y rotation of X should give -Z, got {:?}",
            result
        );
    }

    #[test]
    fn quaternion_composition() {
        let q1 = Quaternion::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_2);
        let q2 = Quaternion::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_2);
        let combined = (q1 * q2).normalize();
        // Two 90° rotations = 180° rotation
        let v = Vec3::new(1.0, 0.0, 0.0);
        let result = combined.rotate_vec3(v);
        assert!(
            vec3_approx_eq(result, Vec3::new(-1.0, 0.0, 0.0)),
            "two 90° Y rotations should give 180°, got {:?}",
            result
        );
    }

    #[test]
    fn quaternion_conjugate_is_inverse() {
        let q = Quaternion::from_axis_angle(Vec3::new(1.0, 1.0, 0.0), 0.7);
        let v = Vec3::new(3.0, -1.0, 2.0);
        let rotated = q.rotate_vec3(v);
        let back = q.conjugate().rotate_vec3(rotated);
        assert!(
            vec3_approx_eq(v, back),
            "conjugate should undo rotation: {:?} vs {:?}",
            v,
            back
        );
    }

    #[test]
    fn camera_creation() {
        let cam = Camera3D::new(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        assert!(approx_eq(cam.distance, 5.0));
        assert!(vec3_approx_eq(cam.forward(), Vec3::new(0.0, 0.0, -1.0)));
    }

    #[test]
    fn camera_orbit_changes_position() {
        let mut cam = Camera3D::new(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        let orig_pos = cam.position();
        cam.orbit(0.3, 0.0);
        let new_pos = cam.position();
        assert!(
            !vec3_approx_eq(orig_pos, new_pos),
            "orbit should change position"
        );
        // Distance to target should be preserved
        let new_dist = (new_pos - cam.target()).length();
        assert!(
            approx_eq(new_dist, 5.0),
            "orbit should preserve distance: {}",
            new_dist
        );
    }

    #[test]
    fn camera_zoom_clamps() {
        let mut cam = Camera3D::new(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        cam.zoom(-100.0); // zoom way in
        assert!(
            cam.distance >= MIN_DISTANCE,
            "zoom should clamp at MIN_DISTANCE"
        );
        cam.zoom(1000.0); // zoom way out
        assert!(
            cam.distance <= MAX_DISTANCE,
            "zoom should clamp at MAX_DISTANCE"
        );
    }

    #[test]
    fn camera_pan_shifts_both() {
        let mut cam = Camera3D::new(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        let orig_target = cam.target();
        cam.pan(1.0, 0.0);
        let new_target = cam.target();
        assert!(
            !vec3_approx_eq(orig_target, new_target),
            "pan should shift target"
        );
    }

    #[test]
    fn view_matrix_is_valid() {
        let cam = Camera3D::new(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        let view = cam.view_matrix();
        // Last row should be [0, 0, 0, 1] (affine, row-major)
        assert!(approx_eq(view[3][0], 0.0));
        assert!(approx_eq(view[3][1], 0.0));
        assert!(approx_eq(view[3][2], 0.0));
        assert!(approx_eq(view[3][3], 1.0));
    }

    #[test]
    fn vec3_cross_product() {
        let result = Vec3::X.cross(Vec3::Y);
        assert!(
            vec3_approx_eq(result, Vec3::Z),
            "X × Y should be Z, got {:?}",
            result
        );
    }

    #[test]
    fn vec3_normalize() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let n = v.normalize();
        assert!(approx_eq(n.length(), 1.0));
        assert!(approx_eq(n.x, 0.6));
        assert!(approx_eq(n.y, 0.8));
    }

    #[test]
    fn default_for_scene() {
        let cam = Camera3D::default_for_scene(Vec3::ZERO, 1.0);
        assert!(cam.distance > 1.0, "should be outside the scene");
        assert!(cam.position().y > 0.0, "should be elevated");
    }
}
