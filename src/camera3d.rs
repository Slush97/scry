// SPDX-License-Identifier: MIT OR Apache-2.0
//! Unified 3D camera with arcball quaternion rotation.
//!
//! [`Camera3D`] provides positioning and orienting a virtual camera in 3D
//! space, using quaternion-based rotation to avoid gimbal lock. Supports orbit,
//! pan, and zoom operations for interactive 3D visualization.
//!
//! Both the chart3d module and the SDF renderer can use this camera.

use crate::math3d::{Quaternion, Vec3};

/// Minimum distance from camera to target (prevents zooming through target).
pub const MIN_DISTANCE: f32 = 0.5;
/// Maximum distance from camera to target.
pub const MAX_DISTANCE: f32 = 100.0;

/// A 3D camera with arcball rotation for interactive visualization.
///
/// The camera orbits around a target point using quaternion rotation.
/// It supports orbit (rotate around target), pan (shift both position and
/// target), and zoom (move along the look-at axis).
///
/// # Example
///
/// ```
/// use scry_engine::camera3d::Camera3D;
/// use scry_engine::math3d::Vec3;
///
/// let mut cam = Camera3D::new(
///     Vec3::new(0.0, 0.0, 5.0),
///     Vec3::ZERO,
///     Vec3::Y,
/// );
///
/// cam.orbit(0.1, 0.05);
/// cam.zoom(-0.5);
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
    #[must_use]
    pub fn orbiting(target: Vec3, distance: f32, azimuth: f32, elevation: f32) -> Self {
        let dist = distance.max(MIN_DISTANCE);
        let (sa, ca) = azimuth.sin_cos();
        let (se, ce) = elevation.sin_cos();
        let position = target + Vec3::new(dist * ce * sa, dist * se, dist * ce * ca);
        Self::new(position, target, Vec3::Y)
    }

    /// Orbit the camera around the target point.
    ///
    /// `dx` rotates around the world Y axis (horizontal drag).
    /// `dy` rotates around the camera's local right axis (vertical drag).
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        let yaw = Quaternion::from_axis_angle(Vec3::Y, -dx);
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
    /// Uses the standard look-at construction in row-major layout.
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

    /// Vertical field of view in degrees.
    #[must_use]
    pub fn fov_degrees(&self) -> f32 {
        self.fov_y.to_degrees()
    }

    /// Update position from orientation and distance.
    fn update_position(&mut self) {
        let offset = self.orientation.rotate_vec3(Vec3::Z).scale(self.distance);
        self.position = self.target + offset;
    }
}

// ── SdfCamera conversions ────────────────────────────────────────────

/// Convert an [`SdfCamera`](crate::sdf::SdfCamera) to a [`Camera3D`].
#[cfg(feature = "sdf")]
impl From<crate::sdf::SdfCamera> for Camera3D {
    fn from(sdf: crate::sdf::SdfCamera) -> Self {
        let mut cam = Self::new(sdf.eye, sdf.target, Vec3::UP);
        cam.fov_y = sdf.fov.to_radians();
        cam
    }
}

/// Convert a [`Camera3D`] reference to an [`SdfCamera`](crate::sdf::SdfCamera).
#[cfg(feature = "sdf")]
impl From<&Camera3D> for crate::sdf::SdfCamera {
    fn from(cam: &Camera3D) -> Self {
        Self::new(cam.position(), cam.target(), cam.fov_degrees())
    }
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
    fn camera_creation() {
        let cam = Camera3D::new(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        assert!(approx_eq(cam.distance, 5.0));
        assert!(vec3_approx_eq(cam.forward(), Vec3::new(0.0, 0.0, -1.0)));
    }

    #[test]
    fn camera_fov_degrees() {
        let cam = Camera3D::new(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        assert!(approx_eq(cam.fov_degrees(), 45.0));
    }

    #[test]
    fn camera_orbit_preserves_distance() {
        let mut cam = Camera3D::new(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        cam.orbit(0.3, 0.0);
        let dist = (cam.position() - cam.target()).length();
        assert!(approx_eq(dist, 5.0));
    }

    #[test]
    fn camera_zoom_clamps() {
        let mut cam = Camera3D::new(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        cam.zoom(-100.0);
        assert!(cam.distance >= MIN_DISTANCE);
        cam.zoom(1000.0);
        assert!(cam.distance <= MAX_DISTANCE);
    }
}
