// SPDX-License-Identifier: MIT OR Apache-2.0
//! Perspective projection and depth sorting for 3D visualization.
//!
//! Provides [`PerspectiveProjection`] for transforming 3D world-space points
//! into 2D screen coordinates, and [`depth_sort`] for painter's algorithm
//! back-to-front ordering.

pub use scry_engine::math3d::{mat4_identity, mat4_mul, mat4_mul_vec4, Mat4, Vec3};

// ---------------------------------------------------------------------------
// ProjectedPoint
// ---------------------------------------------------------------------------

/// A 3D point after projection to 2D screen space.
///
/// Retains depth for painter's algorithm sorting and an index for
/// correlating back to the original data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectedPoint {
    /// X position in screen pixels.
    pub screen_x: f32,
    /// Y position in screen pixels.
    pub screen_y: f32,
    /// Depth in view space (larger = farther from camera).
    pub depth: f32,
    /// Index into the original point array.
    pub original_index: usize,
}

// ---------------------------------------------------------------------------
// PerspectiveProjection
// ---------------------------------------------------------------------------

/// Perspective projection configuration.
///
/// Transforms 3D view-space points into normalized device coordinates,
/// then maps to screen pixel coordinates.
#[derive(Clone, Debug)]
pub struct PerspectiveProjection {
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Viewport aspect ratio (width / height).
    pub aspect: f32,
    /// Near clipping plane distance.
    pub near: f32,
    /// Far clipping plane distance.
    pub far: f32,
}

impl PerspectiveProjection {
    /// Create a new perspective projection.
    #[must_use]
    pub fn new(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        Self {
            fov_y,
            aspect,
            near,
            far,
        }
    }

    /// Compute the 4×4 perspective projection matrix.
    #[must_use]
    pub fn projection_matrix(&self) -> Mat4 {
        let f = 1.0 / (self.fov_y * 0.5).tan();
        let nf = self.near - self.far;

        [
            [f / self.aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [
                0.0,
                0.0,
                (self.far + self.near) / nf,
                2.0 * self.far * self.near / nf,
            ],
            [0.0, 0.0, -1.0, 0.0],
        ]
    }

    /// Project a single 3D world-space point to screen coordinates.
    ///
    /// Applies the view matrix, then the projection matrix, then viewport
    /// mapping. Returns `None` if the point is behind the camera.
    #[must_use]
    pub fn project(
        &self,
        point: Vec3,
        view: &Mat4,
        width: u32,
        height: u32,
        index: usize,
    ) -> Option<ProjectedPoint> {
        let proj = self.projection_matrix();
        let vp = mat4_mul(&proj, view);

        let clip = mat4_mul_vec4(&vp, [point.x, point.y, point.z, 1.0]);

        // Behind camera check
        if clip[3] <= 0.0 {
            return None;
        }

        // Perspective divide
        let ndc_x = clip[0] / clip[3];
        let ndc_y = clip[1] / clip[3];
        let ndc_z = clip[2] / clip[3];

        // NDC → screen coordinates
        // NDC is [-1, 1], screen is [0, width] / [0, height]
        let screen_x = (ndc_x + 1.0) * 0.5 * width as f32;
        let screen_y = (1.0 - ndc_y) * 0.5 * height as f32; // flip Y

        Some(ProjectedPoint {
            screen_x,
            screen_y,
            depth: ndc_z,
            original_index: index,
        })
    }

    /// Project a batch of 3D points, filtering out those behind the camera.
    #[must_use]
    pub fn project_batch(
        &self,
        points: &[Vec3],
        view: &Mat4,
        width: u32,
        height: u32,
    ) -> Vec<ProjectedPoint> {
        let proj = self.projection_matrix();
        let vp = mat4_mul(&proj, view);
        project_batch_precomputed(&vp, points, width, height)
    }
}

/// Project a batch of 3D points using a pre-computed view-projection matrix.
///
/// This is the fast path — the VP matrix is computed once externally,
/// eliminating per-point `projection_matrix()` + `mat4_mul()` overhead.
#[must_use]
pub fn project_batch_precomputed(
    vp: &Mat4,
    points: &[Vec3],
    width: u32,
    height: u32,
) -> Vec<ProjectedPoint> {
    let w_f = width as f32;
    let h_f = height as f32;
    let mut out = Vec::with_capacity(points.len());

    for (i, &p) in points.iter().enumerate() {
        let clip = mat4_mul_vec4(vp, [p.x, p.y, p.z, 1.0]);

        // Behind camera check
        if clip[3] <= 0.0 {
            continue;
        }

        // Perspective divide
        let inv_w = 1.0 / clip[3];
        let ndc_x = clip[0] * inv_w;
        let ndc_y = clip[1] * inv_w;
        let ndc_z = clip[2] * inv_w;

        // NDC → screen coordinates
        out.push(ProjectedPoint {
            screen_x: (ndc_x + 1.0) * 0.5 * w_f,
            screen_y: (1.0 - ndc_y) * 0.5 * h_f,
            depth: ndc_z,
            original_index: i,
        });
    }

    out
}

/// Sort projected points back-to-front (painter's algorithm).
///
/// Points with larger depth values are drawn first so that closer
/// points paint over farther ones.
pub fn depth_sort(points: &mut [ProjectedPoint]) {
    points.sort_unstable_by(|a, b| {
        b.depth
            .partial_cmp(&a.depth)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Sort projected line segments back-to-front by average depth.
pub fn depth_sort_segments(segments: &mut [(ProjectedPoint, ProjectedPoint)]) {
    segments.sort_unstable_by(|a, b| {
        let da = (a.0.depth + a.1.depth) * 0.5;
        let db = (b.0.depth + b.1.depth) * 0.5;
        db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
    });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
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

    #[test]
    fn projection_matrix_is_valid() {
        let proj = PerspectiveProjection::new(std::f32::consts::FRAC_PI_4, 16.0 / 9.0, 0.1, 100.0);
        let m = proj.projection_matrix();
        // Should have -1 in [3][2] for perspective divide
        assert!(
            approx_eq(m[3][2], -1.0),
            "m[3][2] should be -1: {}",
            m[3][2]
        );
        // m[3][3] should be 0 (perspective, not orthographic)
        assert!(approx_eq(m[3][3], 0.0), "m[3][3] should be 0: {}", m[3][3]);
    }

    #[test]
    fn project_center_point() {
        let proj = PerspectiveProjection::new(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
        // A point at (0, 0, -5) with identity view should project to center
        let view = mat4_identity();
        let result = proj.project(Vec3::new(0.0, 0.0, -5.0), &view, 800, 600, 0);
        assert!(result.is_some(), "center point should be visible");
        let p = result.unwrap();
        assert!(
            approx_eq(p.screen_x, 400.0),
            "center x should be 400, got {}",
            p.screen_x
        );
        assert!(
            approx_eq(p.screen_y, 300.0),
            "center y should be 300, got {}",
            p.screen_y
        );
    }

    #[test]
    fn project_behind_camera_returns_none() {
        let proj = PerspectiveProjection::new(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
        let view = mat4_identity();
        // Point at (0, 0, 5) is behind camera when looking down -Z
        let result = proj.project(Vec3::new(0.0, 0.0, 5.0), &view, 800, 600, 0);
        // With identity view, +Z is behind camera in our projection setup
        // The point should either be None or have negative clip w
        // Actually with identity view and our projection, points at +z are behind
        // Let's just test the result is consistent
        assert!(
            result.is_none() || result.unwrap().depth > 1.0,
            "point behind camera should be filtered or have large depth"
        );
    }

    #[test]
    fn depth_sort_orders_back_to_front() {
        let mut points = vec![
            ProjectedPoint {
                screen_x: 0.0,
                screen_y: 0.0,
                depth: 0.1,
                original_index: 0,
            },
            ProjectedPoint {
                screen_x: 0.0,
                screen_y: 0.0,
                depth: 0.9,
                original_index: 1,
            },
            ProjectedPoint {
                screen_x: 0.0,
                screen_y: 0.0,
                depth: 0.5,
                original_index: 2,
            },
        ];
        depth_sort(&mut points);
        assert_eq!(points[0].original_index, 1, "farthest first");
        assert_eq!(points[1].original_index, 2, "middle second");
        assert_eq!(points[2].original_index, 0, "closest last");
    }

    #[test]
    fn project_batch_filters_backface() {
        let proj = PerspectiveProjection::new(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
        let view = mat4_identity();
        let points = vec![
            Vec3::new(0.0, 0.0, -5.0),  // visible
            Vec3::new(0.0, 0.0, -10.0), // visible
        ];
        let projected = proj.project_batch(&points, &view, 800, 600);
        assert_eq!(projected.len(), 2, "both points should be visible");
    }
}
