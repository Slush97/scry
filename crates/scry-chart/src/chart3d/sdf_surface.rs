// SPDX-License-Identifier: MIT OR Apache-2.0
//! Convert a [`Surface3D`] height-field into an [`SdfScene`] for lit rendering.
//!
//! The surface is approximated as a collection of small SDF box primitives
//! (one per grid cell) with Phong-shaded materials derived from the surface
//! colors or a default height-based colormap.

use scry_engine::camera3d::Camera3D;
use scry_engine::math3d::Vec3;
use scry_engine::scene::style::Color;
use scry_engine::sdf::{Material, SdfCamera, SdfLight, SdfObject, SdfScene, SdfShape};

use super::scene::Surface3D;

/// Build an [`SdfScene`] from a [`Surface3D`] height-field and camera.
///
/// Each grid cell becomes a thin box primitive positioned at the cell center
/// with height matching the surface value. A ground plane, two lights, and
/// Phong materials are added automatically.
///
/// # Arguments
///
/// * `surface` — The height-field grid.
/// * `camera` — The `Camera3D` to convert into an `SdfCamera`.
pub fn surface_to_sdf_scene(surface: &Surface3D, camera: &Camera3D) -> SdfScene {
    let rows = surface.rows;
    let cols = surface.cols;

    // Compute height range for normalization and colormap
    let mut h_min = f32::INFINITY;
    let mut h_max = f32::NEG_INFINITY;
    for &h in &surface.heights {
        if h.is_finite() {
            h_min = h_min.min(h);
            h_max = h_max.max(h);
        }
    }
    let h_range = (h_max - h_min).max(1e-6);

    let x_span = surface.extent_max.x - surface.extent_min.x;
    let z_span = surface.extent_max.z - surface.extent_min.z;
    let cell_w = x_span / cols.max(1) as f32;
    let cell_d = z_span / rows.max(1) as f32;

    let mut scene = SdfScene::new();

    // Ground plane
    scene = scene.object(
        SdfObject::new(
            SdfShape::Plane,
            Material::matte(Color::from_rgba8(60, 60, 70, 255)),
        )
        .at(Vec3::new(0.0, h_min - 0.01, 0.0)),
    );

    // Surface cells as boxes
    for row in 0..rows {
        for col in 0..cols {
            let h = surface.heights[row * cols + col];
            if !h.is_finite() {
                continue;
            }

            let x = surface.extent_min.x + (col as f32 + 0.5) * cell_w;
            let z = surface.extent_min.z + (row as f32 + 0.5) * cell_d;

            // Height of the box (from h_min to h)
            let box_h = ((h - h_min) * 0.5).max(0.005);
            let y_center = h_min + box_h;

            let color = if let Some(ref colors) = surface.colors {
                colors.get(row * cols + col).copied().unwrap_or(Color::WHITE)
            } else {
                height_colormap((h - h_min) / h_range)
            };

            scene = scene.object(
                SdfObject::new(
                    SdfShape::Box {
                        half_extents: Vec3::new(cell_w * 0.48, box_h, cell_d * 0.48),
                    },
                    Material::matte(color),
                )
                .at(Vec3::new(x, y_center, z)),
            );
        }
    }

    // Lighting: key + fill
    let center_x = (surface.extent_min.x + surface.extent_max.x) * 0.5;
    let center_z = (surface.extent_min.z + surface.extent_max.z) * 0.5;
    let light_h = h_max + h_range * 2.0;

    scene = scene
        .light(SdfLight::new(
            Vec3::new(center_x + x_span, light_h, center_z + z_span),
            Color::WHITE,
            1.0,
        ))
        .light(SdfLight::new(
            Vec3::new(center_x - x_span * 0.5, light_h * 0.6, center_z - z_span),
            Color::from_rgba8(180, 200, 255, 255),
            0.5,
        ));

    // Camera
    let sdf_cam: SdfCamera = camera.into();
    scene = scene.camera(sdf_cam);

    // Sky and ambient
    scene = scene
        .sky_color(Color::from_rgba8(30, 35, 50, 255))
        .ambient(0.12);

    scene
}

/// Simple height-based colormap (cool blue → warm red).
fn height_colormap(t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    // Blue at low, green at mid, red at high
    let r = (t * 2.0 - 0.5).clamp(0.0, 1.0);
    let g = (1.0 - (t * 2.0 - 1.0).abs()).clamp(0.0, 1.0);
    let b = (1.0 - t * 2.0).clamp(0.0, 1.0);
    Color::from_rgba(r * 0.9 + 0.1, g * 0.8 + 0.1, b * 0.8 + 0.1, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chart3d::scene::Surface3D;

    #[test]
    fn surface_to_sdf_produces_objects() {
        let heights = vec![0.0, 0.5, 0.3, 1.0];
        let surface = Surface3D::new(heights, 2, 2);
        let cam = Camera3D::new(Vec3::new(2.0, 3.0, 4.0), Vec3::new(0.5, 0.5, 0.5), Vec3::Y);
        let scene = surface_to_sdf_scene(&surface, &cam);

        // 1 ground plane + 4 cell boxes = 5 objects
        assert_eq!(scene.objects.len(), 5);
        // 2 lights
        assert_eq!(scene.lights.len(), 2);
    }

    #[test]
    fn height_colormap_bounds() {
        let low = height_colormap(0.0);
        let high = height_colormap(1.0);
        // Low should be bluish (b > r)
        assert!(low.b > low.r, "low should be cool/blue");
        // High should be reddish (r > b)
        assert!(high.r > high.b, "high should be warm/red");
    }
}
