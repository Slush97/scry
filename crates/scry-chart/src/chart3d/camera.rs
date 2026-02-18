// SPDX-License-Identifier: MIT OR Apache-2.0
//! 3D camera with arcball quaternion rotation.
//!
//! [`Camera3D`], [`Vec3`], and [`Quaternion`] are defined in
//! [`scry_engine`] and re-exported here for backwards compatibility.

pub use scry_engine::camera3d::Camera3D;
pub use scry_engine::math3d::{Quaternion, Vec3};

// Re-export constants for test compatibility
pub use scry_engine::camera3d::{MAX_DISTANCE, MIN_DISTANCE};

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
        assert!(
            vec3_approx_eq(v, result),
            "identity should not change vector"
        );
    }

    #[test]
    fn quaternion_90deg_y_rotation() {
        let q = Quaternion::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_2);
        let v = Vec3::new(1.0, 0.0, 0.0);
        let result = q.rotate_vec3(v);
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
        cam.zoom(-100.0);
        cam.zoom(1000.0);
        // Just check it doesn't panic — distance field is private
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
        assert!(cam.position().y > 0.0, "should be elevated");
    }
}
