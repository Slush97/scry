// SPDX-License-Identifier: MIT OR Apache-2.0
//! GPU-uploadable struct definitions and scene flattening for the SDF
//! compute shader.
//!
//! Converts the high-level `SdfScene` representation into flat, std140-aligned
//! GPU buffers: uniforms, object array, light array, and glyph data.

use crate::scene::style::Color;

use super::materials::Material;
use super::math::{self, Vec3};
use super::scene::{SdfScene, SdfShape};

use bytemuck::Zeroable;

// ── GPU-uploadable structs ─────────────────────────────────────────

/// Must match the WGSL `Uniforms` struct exactly (std140 layout).
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(super) struct GpuUniforms {
    pub(super) eye: [f32; 3],
    pub(super) _pad0: f32,
    pub(super) cam_right: [f32; 3],
    pub(super) _pad1: f32,
    pub(super) cam_up: [f32; 3],
    pub(super) _pad2: f32,
    pub(super) cam_forward: [f32; 3],
    pub(super) fov_scale: f32,

    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) aspect: f32,
    pub(super) time: f32,

    pub(super) sky_color: [f32; 4],
    pub(super) ambient: f32,
    pub(super) max_bounces: u32,
    pub(super) num_objects: u32,
    pub(super) num_lights: u32,
    pub(super) has_water: u32,
    pub(super) god_rays: u32,
    pub(super) god_ray_density: f32,
    pub(super) god_ray_samples: u32,
}

/// Must match the WGSL `GpuObject` struct.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(super) struct GpuObject {
    pub(super) position: [f32; 3],
    pub(super) shape_type: u32,
    pub(super) shape_params: [f32; 4],
    pub(super) blend_a_params: [f32; 4],
    pub(super) blend_b_params: [f32; 4],
    pub(super) blend_b_offset: [f32; 3],
    pub(super) material_type: u32,
    pub(super) material_params: [f32; 4],
    pub(super) material_color: [f32; 4],
    pub(super) bounding_radius: f32,
    pub(super) _pad2: [f32; 3],
    pub(super) orientation: [f32; 4],
}

/// Must match the WGSL `GpuLight` struct.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(super) struct GpuLight {
    pub(super) position: [f32; 3],
    pub(super) intensity: f32,
    pub(super) color: [f32; 4],
}

/// Glyph metadata for GPU text3d rendering (binding 4).
/// Must match the WGSL `GpuGlyphMeta` struct.
#[cfg(feature = "sdf-text")]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(super) struct GpuGlyphMeta {
    x_offset: f32,
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    grid_width: u32,
    grid_height: u32,
    grid_offset: u32,
}

// Shape type discriminants (must match WGSL constants)
const SHAPE_SPHERE: u32 = 0;
const SHAPE_BOX: u32 = 1;
const SHAPE_PLANE: u32 = 2;
const SHAPE_TORUS: u32 = 3;
const SHAPE_CYLINDER: u32 = 4;
const SHAPE_SMOOTH_BLEND: u32 = 5;
const SHAPE_CAPSULE: u32 = 6;
const SHAPE_ROUNDED_BOX: u32 = 7;
const SHAPE_CONE: u32 = 8;
#[cfg(feature = "sdf-text")]
const SHAPE_TEXT3D: u32 = 9;
const SHAPE_SUBTRACT: u32 = 10;
const SHAPE_MANDELBULB: u32 = 11;
const SHAPE_MENGER: u32 = 12;
const SHAPE_GYROID: u32 = 13;
const SHAPE_MORPH: u32 = 14;

// Material type discriminants
const MAT_SOLID: u32 = 0;
const MAT_WATER: u32 = 1;
const MAT_FIRE: u32 = 2;
const MAT_CHECKER: u32 = 3;
const MAT_GLASS: u32 = 4;
const MAT_RAINBOW: u32 = 5;
const MAT_SUBSURFACE: u32 = 6;

// ── Helper functions ───────────────────────────────────────────────

pub(super) fn vec3_to_arr(v: Vec3) -> [f32; 3] {
    [v.x, v.y, v.z]
}

pub(super) fn color_to_arr(c: Color) -> [f32; 4] {
    [c.r, c.g, c.b, c.a]
}

// ── Scene flattening ───────────────────────────────────────────────

/// Flatten an `SdfShape` into type discriminant + params.
/// For `SmoothBlend`, also populates `blend_a`/`blend_b` params and `b_offset`.
pub(super) fn flatten_shape(shape: &SdfShape) -> (u32, [f32; 4], [f32; 4], [f32; 4], [f32; 3]) {
    match shape {
        SdfShape::Sphere { radius } => (
            SHAPE_SPHERE,
            [*radius, 0.0, 0.0, 0.0],
            [0.0; 4],
            [0.0; 4],
            [0.0; 3],
        ),
        SdfShape::Box { half_extents } => (
            SHAPE_BOX,
            [half_extents.x, half_extents.y, half_extents.z, 0.0],
            [0.0; 4],
            [0.0; 4],
            [0.0; 3],
        ),
        SdfShape::Plane => (SHAPE_PLANE, [0.0; 4], [0.0; 4], [0.0; 4], [0.0; 3]),
        SdfShape::Torus { major, minor } => (
            SHAPE_TORUS,
            [*major, *minor, 0.0, 0.0],
            [0.0; 4],
            [0.0; 4],
            [0.0; 3],
        ),
        SdfShape::Cylinder {
            radius,
            half_height,
        } => (
            SHAPE_CYLINDER,
            [*radius, *half_height, 0.0, 0.0],
            [0.0; 4],
            [0.0; 4],
            [0.0; 3],
        ),
        SdfShape::SmoothBlend { a, b, b_offset, k } => {
            let (a_type, a_params, _, _, _) = flatten_shape(a);
            let (b_type, b_params, _, _, _) = flatten_shape(b);
            (
                SHAPE_SMOOTH_BLEND,
                [*k, a_type as f32, b_type as f32, 0.0],
                a_params,
                b_params,
                vec3_to_arr(*b_offset),
            )
        }
        SdfShape::Capsule {
            radius,
            half_height,
        } => (
            SHAPE_CAPSULE,
            [*radius, *half_height, 0.0, 0.0],
            [0.0; 4],
            [0.0; 4],
            [0.0; 3],
        ),
        SdfShape::RoundedBox {
            half_extents,
            radius,
        } => (
            SHAPE_ROUNDED_BOX,
            [half_extents.x, half_extents.y, half_extents.z, *radius],
            [0.0; 4],
            [0.0; 4],
            [0.0; 3],
        ),
        SdfShape::Cone { radius, height } => (
            SHAPE_CONE,
            [*radius, *height, 0.0, 0.0],
            [0.0; 4],
            [0.0; 4],
            [0.0; 3],
        ),
        SdfShape::Subtract { a, b, b_offset } => {
            let (a_type, a_params, _, _, _) = flatten_shape(a);
            let (b_type, b_params, _, _, _) = flatten_shape(b);
            (
                SHAPE_SUBTRACT,
                [0.0, a_type as f32, b_type as f32, 0.0],
                a_params,
                b_params,
                vec3_to_arr(*b_offset),
            )
        }
        SdfShape::Mandelbulb { power, iterations } => (
            SHAPE_MANDELBULB,
            [*power, *iterations as f32, 0.0, 0.0],
            [0.0; 4],
            [0.0; 4],
            [0.0; 3],
        ),
        SdfShape::MengerSponge { iterations } => (
            SHAPE_MENGER,
            [*iterations as f32, 0.0, 0.0, 0.0],
            [0.0; 4],
            [0.0; 4],
            [0.0; 3],
        ),
        SdfShape::Gyroid {
            scale,
            thickness,
            bound,
        } => (
            SHAPE_GYROID,
            [*scale, *thickness, *bound, 0.0],
            [0.0; 4],
            [0.0; 4],
            [0.0; 3],
        ),
        SdfShape::Morph { a, b, t } => {
            let (a_type, a_params, _, _, _) = flatten_shape(a);
            let (b_type, b_params, _, _, _) = flatten_shape(b);
            (
                SHAPE_MORPH,
                [*t, a_type as f32, b_type as f32, 0.0],
                a_params,
                b_params,
                [0.0; 3],
            )
        }
        // Text3D: encode layout params and glyph indices for GPU evaluation.
        // glyph_start is set to 0 here; the caller (build_objects) patches it.
        #[cfg(feature = "sdf-text")]
        SdfShape::Text3D { layout, depth } => (
            SHAPE_TEXT3D,
            [*depth, layout.total_width, layout.ascent, layout.descent],
            [0.0, layout.glyphs.len() as f32, 0.0, 0.0], // [glyph_start, glyph_count, -, -]
            [0.0; 4],
            [0.0; 3],
        ),
    }
}

/// Flatten a `Material` into type discriminant + params + color.
pub(super) fn flatten_material(mat: &Material) -> (u32, [f32; 4], [f32; 4]) {
    match mat {
        Material::Solid {
            color,
            reflectivity,
            specular,
        } => (
            MAT_SOLID,
            [*reflectivity, *specular, 0.0, 0.0],
            color_to_arr(*color),
        ),
        Material::Water {
            tint,
            ior,
            amplitude,
            frequency,
        } => (
            MAT_WATER,
            [*ior, *amplitude, *frequency, 0.0],
            color_to_arr(*tint),
        ),
        Material::Fire {
            intensity,
            noise_scale,
            speed,
        } => (
            MAT_FIRE,
            [*intensity, *noise_scale, *speed, 0.0],
            color_to_arr(Color::WHITE),
        ),
        Material::Checkerboard {
            color_a,
            color_b: _,
            scale,
            reflectivity,
            specular,
        } => (
            MAT_CHECKER,
            [*reflectivity, *specular, *scale, 0.0],
            color_to_arr(*color_a),
        ),
        Material::Glass {
            tint,
            ior,
            opacity,
            dispersion,
        } => (
            MAT_GLASS,
            [*ior, *opacity, *dispersion, 0.0],
            color_to_arr(*tint),
        ),
        Material::Rainbow {
            saturation,
            lightness,
            hue_offset,
            specular,
        } => (
            MAT_RAINBOW,
            [*saturation, *lightness, *hue_offset, *specular],
            [0.5, 0.5, 0.5, 1.0], // base color unused (computed from angle)
        ),
        Material::Subsurface {
            color,
            scatter_color: _,
            thickness,
            specular,
        } => (
            MAT_SUBSURFACE,
            [*thickness, *specular, 0.0, 0.0],
            color_to_arr(*color),
        ),
    }
}

pub(super) fn build_uniforms(scene: &SdfScene, width: u32, height: u32, time: f32) -> GpuUniforms {
    let (cam_right, cam_up, cam_fwd) =
        math::look_at(scene.camera.eye, scene.camera.target, Vec3::UP);
    let fov_scale = (scene.camera.fov.to_radians() * 0.5).tan();
    let aspect = width as f32 / height as f32;

    GpuUniforms {
        eye: vec3_to_arr(scene.camera.eye),
        _pad0: 0.0,
        cam_right: vec3_to_arr(cam_right),
        _pad1: 0.0,
        cam_up: vec3_to_arr(cam_up),
        _pad2: 0.0,
        cam_forward: vec3_to_arr(cam_fwd),
        fov_scale,
        width,
        height,
        aspect,
        time,
        sky_color: color_to_arr(scene.sky_color),
        ambient: scene.ambient,
        max_bounces: scene.max_bounces,
        num_objects: scene.objects.len() as u32,
        num_lights: scene.lights.len() as u32,
        has_water: u32::from(scene.has_water || scene.has_glass),
        god_rays: u32::from(scene.god_rays),
        god_ray_density: scene.god_ray_density,
        god_ray_samples: scene.god_ray_samples,
    }
}

pub(super) fn build_objects(scene: &SdfScene) -> Vec<GpuObject> {
    let mut result = Vec::with_capacity(scene.objects.len());
    #[allow(unused_variables, unused_mut)]
    let mut glyph_cursor: u32 = 0;

    for obj in &scene.objects {
        let (shape_type, shape_params, blend_a, blend_b, blend_b_off) = flatten_shape(&obj.shape);
        let (material_type, material_params, material_color) = flatten_material(&obj.material);

        // For checkerboard materials, pack color_b into blend_a_params
        // (safe because checkerboard is always on Plane, never SmoothBlend).
        #[allow(unused_mut)]
        let mut blend_a = if let Material::Checkerboard { color_b, .. } = &obj.material {
            color_to_arr(*color_b)
        } else if let Material::Subsurface { scatter_color, .. } = &obj.material {
            color_to_arr(*scatter_color)
        } else {
            blend_a
        };

        // Patch glyph_start index for Text3D objects
        #[cfg(feature = "sdf-text")]
        if let SdfShape::Text3D { layout, .. } = &obj.shape {
            blend_a[0] = glyph_cursor as f32;
            glyph_cursor += layout.glyphs.len() as u32;
        }

        // Compute conjugated quaternion for inverse rotation on GPU.
        // Quaternion orientation takes precedence over Y-axis rotation.
        let orientation = if let Some(q) = obj.orientation {
            let c = q.conjugate();
            [c.x, c.y, c.z, c.w]
        } else if let Some((cos_y, sin_y)) = obj.rotation_y {
            // Y-axis rotation quaternion: q = (0, sin(θ/2), 0, cos(θ/2))
            // We need the conjugate for inverse rotation.
            // From cos/sin of full angle: half-angle via cos(θ/2) = sqrt((1+cosθ)/2),
            // sin(θ/2) = sqrt((1-cosθ)/2) * sign(sinθ)
            let half_cos = ((1.0 + cos_y) * 0.5).sqrt();
            let half_sin = ((1.0 - cos_y) * 0.5).sqrt().copysign(sin_y);
            // Conjugate negates the vector part
            [0.0, -half_sin, 0.0, half_cos]
        } else {
            [0.0, 0.0, 0.0, 1.0] // identity
        };

        result.push(GpuObject {
            position: vec3_to_arr(obj.position),
            shape_type,
            shape_params,
            blend_a_params: blend_a,
            blend_b_params: blend_b,
            blend_b_offset: blend_b_off,
            material_type,
            material_params,
            material_color,
            bounding_radius: obj.bounding_radius,
            _pad2: [0.0; 3],
            orientation,
        });
    }

    result
}

pub(super) fn build_lights(scene: &SdfScene) -> Vec<GpuLight> {
    scene
        .lights
        .iter()
        .map(|light| GpuLight {
            position: vec3_to_arr(light.position),
            intensity: light.intensity,
            color: color_to_arr(light.color),
        })
        .collect()
}

/// Collect glyph metadata and grids from all `Text3D` objects in the scene.
///
/// Returns `(meta_bytes, grids_bytes)` ready for GPU upload. When no `Text3D`
/// objects exist, returns 1-element placeholder buffers (same pattern as
/// objects/lights).
pub(super) fn build_glyph_data(scene: &SdfScene) -> (Vec<u8>, Vec<u8>) {
    #[cfg(feature = "sdf-text")]
    {
        let mut metas: Vec<GpuGlyphMeta> = Vec::new();
        let mut grids: Vec<f32> = Vec::new();

        for obj in &scene.objects {
            if let SdfShape::Text3D { layout, .. } = &obj.shape {
                for (glyph, x_offset) in &layout.glyphs {
                    let (min_x, min_y, max_x, max_y) = glyph.bounds;
                    metas.push(GpuGlyphMeta {
                        x_offset: *x_offset,
                        min_x,
                        min_y,
                        max_x,
                        max_y,
                        grid_width: glyph.width as u32,
                        grid_height: glyph.height as u32,
                        grid_offset: grids.len() as u32,
                    });
                    grids.extend_from_slice(&glyph.grid);
                }
            }
        }

        if metas.is_empty() {
            let placeholder_meta = [GpuGlyphMeta::zeroed()];
            let placeholder_grid = [0.0_f32];
            return (
                bytemuck::cast_slice(&placeholder_meta).to_vec(),
                bytemuck::cast_slice(&placeholder_grid).to_vec(),
            );
        }

        (
            bytemuck::cast_slice(&metas).to_vec(),
            bytemuck::cast_slice(&grids).to_vec(),
        )
    }

    #[cfg(not(feature = "sdf-text"))]
    {
        let _ = scene;
        // Placeholder buffers when sdf-text feature is disabled
        let placeholder_meta = [0u8; 32]; // size of one GpuGlyphMeta
        let placeholder_grid = [0u8; 4]; // one f32
        (placeholder_meta.to_vec(), placeholder_grid.to_vec())
    }
}
