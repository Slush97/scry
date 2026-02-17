// SPDX-License-Identifier: MIT OR Apache-2.0
//! Core SDF ray marching renderer.
//!
//! Renders an [`SdfScene`] to a `tiny_skia::Pixmap` or [`ImageData`] by sphere-
//! tracing each pixel ray through the signed distance field. Supports Phong
//! shading, mirror reflections, animated water (Fresnel + refraction), and
//! volumetric fire (front-to-back FBM compositing).

use rayon::prelude::*;
use tiny_skia::Pixmap;

use crate::scene::command::ImageData;
use crate::scene::style::Color;
use crate::PixelCanvasError;

use super::materials::{self, Material};
use super::math::{self, Vec3};
use super::noise;
use super::primitives;
use super::profiler::{RowProfile, SdfProfile, SdfStage};
use super::scene::{SdfObject, SdfScene, SdfShape};

// ── Constants ───────────────────────────────────────────────────────

const MAX_STEPS: u32 = 48;
const SHADOW_MAX_STEPS: u32 = 24;
const MAX_DIST: f32 = 50.0;
const SURF_DIST: f32 = 0.002;
const NORMAL_EPS: f32 = 0.002;

// ── Public API ──────────────────────────────────────────────────────

/// Stateless renderer that ray-marches an [`SdfScene`].
pub struct SdfRenderer;

impl SdfRenderer {
    /// Render the scene to a `tiny_skia::Pixmap`.
    pub fn render_to_pixmap(
        scene: &SdfScene,
        width: u32,
        height: u32,
        time: f32,
    ) -> Result<Pixmap, PixelCanvasError> {
        let mut pixmap = Pixmap::new(width, height).ok_or_else(|| {
            PixelCanvasError::PixmapCreation(format!("failed to create {width}x{height} pixmap"))
        })?;

        let (cam_right, cam_up, cam_fwd) =
            math::look_at(scene.camera.eye, scene.camera.target, Vec3::UP);
        let fov_scale = (scene.camera.fov.to_radians() * 0.5).tan();
        let aspect = width as f32 / height as f32;

        let pixels = pixmap.pixels_mut();

        // Parallel: each row is traced independently
        let row_pixels: Vec<Vec<tiny_skia::PremultipliedColorU8>> = (0..height)
            .into_par_iter()
            .map(|y| {
                let ndc_y = (1.0 - 2.0 * (y as f32 + 0.5) / height as f32) * fov_scale;
                (0..width)
                    .map(|x| {
                        let ndc_x =
                            (2.0 * (x as f32 + 0.5) / width as f32 - 1.0) * aspect * fov_scale;
                        let dir = (cam_right * ndc_x + cam_up * ndc_y + cam_fwd).normalize();
                        let color = shade_ray(scene, scene.camera.eye, dir, time, 0);
                        tiny_skia::PremultipliedColorU8::from_rgba(
                            (color.r.clamp(0.0, 1.0) * 255.0) as u8,
                            (color.g.clamp(0.0, 1.0) * 255.0) as u8,
                            (color.b.clamp(0.0, 1.0) * 255.0) as u8,
                            255,
                        )
                        .unwrap()
                    })
                    .collect()
            })
            .collect();

        for (y, row) in row_pixels.into_iter().enumerate() {
            let offset = y * width as usize;
            pixels[offset..offset + width as usize].copy_from_slice(&row);
        }

        Ok(pixmap)
    }

    /// Render the scene to a `tiny_skia::Pixmap` with per-stage profiling.
    ///
    /// Same output as [`render_to_pixmap`](Self::render_to_pixmap), but each
    /// rendering stage is timed and returned as an [`SdfProfile`].
    pub fn render_to_pixmap_profiled(
        scene: &SdfScene,
        width: u32,
        height: u32,
        time: f32,
    ) -> Result<(Pixmap, SdfProfile), PixelCanvasError> {
        let frame_start = std::time::Instant::now();

        let mut pixmap = Pixmap::new(width, height).ok_or_else(|| {
            PixelCanvasError::PixmapCreation(format!("failed to create {width}x{height} pixmap"))
        })?;

        let (cam_right, cam_up, cam_fwd) =
            math::look_at(scene.camera.eye, scene.camera.target, Vec3::UP);
        let fov_scale = (scene.camera.fov.to_radians() * 0.5).tan();
        let aspect = width as f32 / height as f32;

        let pixels = pixmap.pixels_mut();

        let row_results: Vec<(Vec<tiny_skia::PremultipliedColorU8>, RowProfile)> = (0..height)
            .into_par_iter()
            .map(|y| {
                let ndc_y = (1.0 - 2.0 * (y as f32 + 0.5) / height as f32) * fov_scale;
                let mut row_profile = RowProfile::new();
                let row_pixels: Vec<_> = (0..width)
                    .map(|x| {
                        let ndc_x =
                            (2.0 * (x as f32 + 0.5) / width as f32 - 1.0) * aspect * fov_scale;
                        let dir = (cam_right * ndc_x + cam_up * ndc_y + cam_fwd).normalize();
                        let color = shade_ray_profiled(
                            scene,
                            scene.camera.eye,
                            dir,
                            time,
                            0,
                            &mut row_profile,
                        );
                        tiny_skia::PremultipliedColorU8::from_rgba(
                            (color.r.clamp(0.0, 1.0) * 255.0) as u8,
                            (color.g.clamp(0.0, 1.0) * 255.0) as u8,
                            (color.b.clamp(0.0, 1.0) * 255.0) as u8,
                            255,
                        )
                        .unwrap()
                    })
                    .collect();
                (row_pixels, row_profile)
            })
            .collect();

        let mut row_profiles = Vec::with_capacity(height as usize);
        for (y, (row, profile)) in row_results.into_iter().enumerate() {
            let offset = y * width as usize;
            pixels[offset..offset + width as usize].copy_from_slice(&row);
            row_profiles.push(profile);
        }

        let total_us = frame_start.elapsed().as_micros() as u64;
        let sdf_profile = SdfProfile::from_rows(&row_profiles, total_us, width, height);

        Ok((pixmap, sdf_profile))
    }

    /// Render at a reduced internal resolution, then bicubic-upscale to the target size.
    ///
    /// `render_scale` controls the internal resolution as a fraction of the target:
    /// - `1.0` — full resolution (no upscale, same as [`render_to_pixmap`](Self::render_to_pixmap))
    /// - `0.5` — half resolution (¼ the pixels, ~4× faster)
    /// - `0.25` — quarter resolution (1/16 the pixels, ~16× faster)
    ///
    /// Values are clamped to `[0.1, 1.0]`.
    pub fn render_to_pixmap_upscaled(
        scene: &SdfScene,
        target_width: u32,
        target_height: u32,
        render_scale: f32,
        time: f32,
    ) -> Result<Pixmap, PixelCanvasError> {
        let scale = render_scale.clamp(0.1, 1.0);
        let internal_w = ((target_width as f32 * scale) as u32).max(1);
        let internal_h = ((target_height as f32 * scale) as u32).max(1);

        let small = Self::render_to_pixmap(scene, internal_w, internal_h, time)?;

        if internal_w == target_width && internal_h == target_height {
            return Ok(small);
        }

        let upscaled = super::upscale::upscale_bicubic(
            small.data(),
            internal_w,
            internal_h,
            target_width,
            target_height,
        );

        let mut pixmap = Pixmap::new(target_width, target_height).ok_or_else(|| {
            PixelCanvasError::PixmapCreation(format!(
                "failed to create {target_width}x{target_height} pixmap"
            ))
        })?;
        pixmap.data_mut().copy_from_slice(&upscaled);
        Ok(pixmap)
    }

    /// Upscaled render with per-stage profiling.
    ///
    /// Same as [`render_to_pixmap_upscaled`](Self::render_to_pixmap_upscaled) but
    /// returns an [`SdfProfile`] covering the internal (lower-resolution) render.
    /// The upscale pass cost is included in `SdfProfile::total_us`.
    pub fn render_to_pixmap_upscaled_profiled(
        scene: &SdfScene,
        target_width: u32,
        target_height: u32,
        render_scale: f32,
        time: f32,
    ) -> Result<(Pixmap, SdfProfile), PixelCanvasError> {
        let scale = render_scale.clamp(0.1, 1.0);
        let internal_w = ((target_width as f32 * scale) as u32).max(1);
        let internal_h = ((target_height as f32 * scale) as u32).max(1);

        let (small, profile) =
            Self::render_to_pixmap_profiled(scene, internal_w, internal_h, time)?;

        if internal_w == target_width && internal_h == target_height {
            return Ok((small, profile));
        }

        let upscaled = super::upscale::upscale_bicubic(
            small.data(),
            internal_w,
            internal_h,
            target_width,
            target_height,
        );

        let mut pixmap = Pixmap::new(target_width, target_height).ok_or_else(|| {
            PixelCanvasError::PixmapCreation(format!(
                "failed to create {target_width}x{target_height} pixmap"
            ))
        })?;
        pixmap.data_mut().copy_from_slice(&upscaled);
        Ok((pixmap, profile))
    }

    /// Render the scene to an [`ImageData`] for compositing onto a `PixelCanvas`.
    pub fn render_to_image(scene: &SdfScene, width: u32, height: u32, time: f32) -> ImageData {
        let (cam_right, cam_up, cam_fwd) =
            math::look_at(scene.camera.eye, scene.camera.target, Vec3::UP);
        let fov_scale = (scene.camera.fov.to_radians() * 0.5).tan();
        let aspect = width as f32 / height as f32;

        // Parallel: each row traced independently, then flattened
        let data: Vec<u8> = (0..height)
            .into_par_iter()
            .flat_map(|y| {
                let ndc_y = (1.0 - 2.0 * (y as f32 + 0.5) / height as f32) * fov_scale;
                let mut row = Vec::with_capacity(width as usize * 4);
                for x in 0..width {
                    let ndc_x = (2.0 * (x as f32 + 0.5) / width as f32 - 1.0) * aspect * fov_scale;
                    let dir = (cam_right * ndc_x + cam_up * ndc_y + cam_fwd).normalize();
                    let color = shade_ray(scene, scene.camera.eye, dir, time, 0);
                    row.push((color.r.clamp(0.0, 1.0) * 255.0) as u8);
                    row.push((color.g.clamp(0.0, 1.0) * 255.0) as u8);
                    row.push((color.b.clamp(0.0, 1.0) * 255.0) as u8);
                    row.push(255);
                }
                row
            })
            .collect();

        ImageData::new(width, height, data)
    }
}

// ── Internal ray marching ───────────────────────────────────────────

/// Evaluate the entire scene SDF at `point`, returning `(distance, object_index)`.
fn scene_sdf(scene: &SdfScene, point: Vec3, time: f32) -> (f32, usize) {
    let mut min_dist = MAX_DIST;
    let mut closest = 0;

    for (i, obj) in scene.objects.iter().enumerate() {
        let d = object_sdf(obj, point, time);
        if d < min_dist {
            min_dist = d;
            closest = i;
        }
    }

    (min_dist, closest)
}

/// Evaluate the SDF for a single object, handling water displacement.
fn object_sdf(obj: &SdfObject, point: Vec3, time: f32) -> f32 {
    let local = primitives::translate(point, obj.position);

    let base_dist = shape_sdf(&obj.shape, local);

    // Water gets animated surface displacement
    if let Material::Water {
        amplitude,
        frequency,
        ..
    } = &obj.material
    {
        if matches!(obj.shape, SdfShape::Plane) {
            // Early-out: max displacement is bounded by amplitude * 2.3
            // (sum of wave coefficients: 1.0 + 0.6 + 0.3 + 0.4 = 2.3).
            // If we're farther than that, the displacement can't affect the result.
            let max_disp = *amplitude * 2.3;
            if base_dist.abs() > max_disp {
                return base_dist;
            }
            let disp = water_displacement(point.x, point.z, time, *amplitude, *frequency);
            return base_dist - disp;
        }
    }

    base_dist
}

/// Evaluate the raw SDF for a shape.
fn shape_sdf(shape: &SdfShape, p: Vec3) -> f32 {
    match shape {
        SdfShape::Sphere { radius } => primitives::sd_sphere(p, *radius),
        SdfShape::Box { half_extents } => primitives::sd_box(p, *half_extents),
        SdfShape::Plane => primitives::sd_plane(p),
        SdfShape::Torus { major, minor } => primitives::sd_torus(p, *major, *minor),
        SdfShape::Cylinder {
            radius,
            half_height,
        } => primitives::sd_cylinder(p, *radius, *half_height),
        SdfShape::SmoothBlend { a, b, b_offset, k } => {
            let da = shape_sdf(a, p);
            let db = shape_sdf(b, p - *b_offset);
            primitives::smooth_min(da, db, *k)
        }
    }
}

/// Estimate the surface normal via the tetrahedron technique.
///
/// For water planes we use an optimised displacement-gradient path
/// (4 cheap displacement evals instead of 4 full `scene_sdf` evals).
/// Everything else uses the numerical tetrahedron technique on the
/// *combined* scene SDF so that normals are correct at inter-object
/// boundaries (e.g. where a sphere meets the ground plane).
fn estimate_normal(scene: &SdfScene, point: Vec3, obj_idx: usize, time: f32) -> Vec3 {
    let obj = &scene.objects[obj_idx];

    // Water plane: compute normal from displacement gradient directly
    if let Material::Water {
        amplitude,
        frequency,
        ..
    } = &obj.material
    {
        if matches!(obj.shape, SdfShape::Plane) {
            return water_normal(point.x, point.z, time, *amplitude, *frequency);
        }
    }

    // Numerical normal via the combined scene SDF — correct everywhere.
    estimate_normal_numerical(scene, point, time)
}

/// Tetrahedron technique fallback for normals (4 SDF evals).
fn estimate_normal_numerical(scene: &SdfScene, point: Vec3, time: f32) -> Vec3 {
    let e = NORMAL_EPS;
    let k0 = Vec3::new(1.0, -1.0, -1.0);
    let k1 = Vec3::new(-1.0, -1.0, 1.0);
    let k2 = Vec3::new(-1.0, 1.0, -1.0);
    let k3 = Vec3::new(1.0, 1.0, 1.0);
    let n = k0 * scene_sdf(scene, point + k0 * e, time).0
        + k1 * scene_sdf(scene, point + k1 * e, time).0
        + k2 * scene_sdf(scene, point + k2 * e, time).0
        + k3 * scene_sdf(scene, point + k3 * e, time).0;
    n.normalize()
}

/// Compute water surface normal from the displacement gradient.
///
/// The water surface is `y = displacement(x, z)`, so the normal is
/// `normalize(-∂d/∂x, 1.0, -∂d/∂z)`. Uses central differences on the
/// displacement function alone (4 displacement evals instead of 4 full scene evals).
fn water_normal(x: f32, z: f32, time: f32, amplitude: f32, frequency: f32) -> Vec3 {
    let e = 0.002;
    let dx = water_displacement(x + e, z, time, amplitude, frequency)
        - water_displacement(x - e, z, time, amplitude, frequency);
    let dz = water_displacement(x, z + e, time, amplitude, frequency)
        - water_displacement(x, z - e, time, amplitude, frequency);
    Vec3::new(-dx / (2.0 * e), 1.0, -dz / (2.0 * e)).normalize()
}

/// Sphere-trace from `origin` along `dir`. Returns `Some((hit_point, obj_index, total_dist))`.
fn ray_march(scene: &SdfScene, origin: Vec3, dir: Vec3, time: f32) -> Option<(Vec3, usize, f32)> {
    let mut t = 0.0_f32;
    for _ in 0..MAX_STEPS {
        let p = origin + dir * t;
        let (d, idx) = scene_sdf(scene, p, time);
        if d < SURF_DIST {
            return Some((p, idx, t));
        }
        t += d;
        if t > MAX_DIST {
            break;
        }
    }
    None
}

/// Fast shadow test — only needs to know if *anything* is hit before `max_t`.
fn shadow_march(scene: &SdfScene, origin: Vec3, dir: Vec3, max_t: f32, time: f32) -> bool {
    let mut t = SURF_DIST * 4.0; // start past the surface
    for _ in 0..SHADOW_MAX_STEPS {
        let p = origin + dir * t;
        let (d, _) = scene_sdf(scene, p, time);
        if d < SURF_DIST {
            return true; // in shadow
        }
        t += d;
        if t > max_t {
            break;
        }
    }
    false
}

/// Top-level per-ray shading: march, hit → shade surface, miss → sky.
fn shade_ray(scene: &SdfScene, origin: Vec3, dir: Vec3, time: f32, bounce: u32) -> Color {
    // Only check fire on primary rays and only if fire objects exist
    if bounce == 0 && scene.has_fire {
        for obj in &scene.objects {
            if let Material::Fire { .. } = &obj.material {
                let fire_color = march_fire_volume(origin, dir, obj, time);
                if fire_color.a > 0.01 {
                    let bg = if let Some((hit, idx, _)) = ray_march(scene, origin, dir, time) {
                        shade_surface(scene, hit, dir, idx, time, bounce)
                    } else {
                        scene.sky_color
                    };
                    return blend_over(fire_color, bg);
                }
            }
        }
    }

    match ray_march(scene, origin, dir, time) {
        Some((hit, idx, _)) => shade_surface(scene, hit, dir, idx, time, bounce),
        None => scene.sky_color,
    }
}

/// Shade a surface hit point with Phong lighting, reflections, and water effects.
fn shade_surface(
    scene: &SdfScene,
    hit: Vec3,
    ray_dir: Vec3,
    obj_idx: usize,
    time: f32,
    bounce: u32,
) -> Color {
    let obj = &scene.objects[obj_idx];
    let normal = estimate_normal(scene, hit, obj_idx, time);

    match &obj.material {
        Material::Solid {
            color,
            reflectivity,
            specular,
        } => {
            let mut result = phong(scene, hit, normal, ray_dir, *color, *specular, time);

            // Reflections
            if *reflectivity > 0.01 && bounce < scene.max_bounces {
                let refl_dir = ray_dir.reflect(normal);
                let refl_origin = hit + normal * (SURF_DIST * 2.0);
                let refl_color = shade_ray(scene, refl_origin, refl_dir, time, bounce + 1);
                result = lerp_color(result, refl_color, *reflectivity);
            }

            result
        }
        Material::Water { tint, ior, .. } => {
            let cos_theta = (-ray_dir).dot(normal).max(0.0);
            let f = materials::fresnel(cos_theta, *ior);

            let mut color = *tint;
            color.a = 1.0;

            // Reflection
            if bounce < scene.max_bounces {
                let refl_dir = ray_dir.reflect(normal);
                let refl_origin = hit + normal * (SURF_DIST * 2.0);
                let refl_color = shade_ray(scene, refl_origin, refl_dir, time, bounce + 1);

                // Refraction
                let eta = 1.0 / ior;
                let refr_color = if let Some(refr_dir) = ray_dir.refract(normal, eta) {
                    let refr_origin = hit - normal * (SURF_DIST * 2.0);
                    let mut rc = shade_ray(scene, refr_origin, refr_dir, time, bounce + 1);
                    // Tint the refracted color
                    rc.r *= tint.r;
                    rc.g *= tint.g;
                    rc.b *= tint.b;
                    rc
                } else {
                    refl_color // Total internal reflection
                };

                color = lerp_color(refr_color, refl_color, f);
            }

            // Add specular highlights from lights
            let spec = phong_specular(scene, hit, normal, ray_dir, 128.0);
            color.r = (color.r + spec.r).min(1.0);
            color.g = (color.g + spec.g).min(1.0);
            color.b = (color.b + spec.b).min(1.0);

            color
        }
        Material::Fire { .. } => {
            // Fire objects are rendered volumetrically, but if we hit the surface
            // directly, show a faint glow
            let glow = materials::fire_color_ramp(0.3);
            Color {
                r: glow.r * 0.5,
                g: glow.g * 0.5,
                b: glow.b * 0.5,
                a: 1.0,
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
/// Phong lighting: ambient + diffuse + specular from all lights.
fn phong(
    scene: &SdfScene,
    hit: Vec3,
    normal: Vec3,
    ray_dir: Vec3,
    base_color: Color,
    spec_power: f32,
    time: f32,
) -> Color {
    let mut r = base_color.r * scene.ambient;
    let mut g = base_color.g * scene.ambient;
    let mut b = base_color.b * scene.ambient;

    for light in &scene.lights {
        let to_light = (light.position - hit).normalize();
        let light_dist = (light.position - hit).length();

        // Fast shadow check
        let shadow_origin = hit + normal * (SURF_DIST * 4.0);
        if shadow_march(scene, shadow_origin, to_light, light_dist, time) {
            continue; // In shadow
        }

        let intensity = light.intensity;

        // Diffuse
        let diff = normal.dot(to_light).max(0.0) * intensity;
        r += base_color.r * light.color.r * diff;
        g += base_color.g * light.color.g * diff;
        b += base_color.b * light.color.b * diff;

        // Specular (Blinn-Phong)
        let half = (to_light - ray_dir).normalize();
        let spec = normal.dot(half).max(0.0).powf(spec_power) * intensity;
        r += light.color.r * spec * 0.5;
        g += light.color.g * spec * 0.5;
        b += light.color.b * spec * 0.5;
    }

    Color {
        r: r.min(1.0),
        g: g.min(1.0),
        b: b.min(1.0),
        a: 1.0,
    }
}

/// Just the specular component of Phong (for adding highlights to water).
fn phong_specular(
    scene: &SdfScene,
    hit: Vec3,
    normal: Vec3,
    ray_dir: Vec3,
    spec_power: f32,
) -> Color {
    let mut r = 0.0_f32;
    let mut g = 0.0_f32;
    let mut b = 0.0_f32;

    for light in &scene.lights {
        let to_light = (light.position - hit).normalize();
        let half = (to_light - ray_dir).normalize();
        let spec = normal.dot(half).max(0.0).powf(spec_power) * light.intensity;
        r += light.color.r * spec;
        g += light.color.g * spec;
        b += light.color.b * spec;
    }

    Color {
        r: r.min(1.0),
        g: g.min(1.0),
        b: b.min(1.0),
        a: 1.0,
    }
}

/// Front-to-back ray marching through a fire volume.
fn march_fire_volume(origin: Vec3, dir: Vec3, obj: &SdfObject, time: f32) -> Color {
    let (intensity, noise_scale, speed) = match &obj.material {
        Material::Fire {
            intensity,
            noise_scale,
            speed,
        } => (*intensity, *noise_scale, *speed),
        _ => return Color::from_rgba8(0, 0, 0, 0),
    };

    // Intersect bounding volume: sphere or cylinder around the object
    let bounding_radius = match &obj.shape {
        SdfShape::Sphere { radius } => *radius,
        SdfShape::Cylinder {
            radius,
            half_height,
        } => radius.max(*half_height),
        _ => 2.0,
    };

    // Simple sphere bounding volume
    let oc = origin - obj.position;
    let b_val = oc.dot(dir);
    let c_val = oc.dot(oc) - bounding_radius * bounding_radius;
    let disc = b_val * b_val - c_val;
    if disc < 0.0 {
        return Color::from_rgba8(0, 0, 0, 0);
    }

    let sqrt_disc = disc.sqrt();
    let t_near = (-b_val - sqrt_disc).max(0.0);
    let t_far = -b_val + sqrt_disc;
    if t_far < 0.0 {
        return Color::from_rgba8(0, 0, 0, 0);
    }

    // Step through volume (larger steps = faster, slightly coarser)
    let step_size = bounding_radius * 0.1;
    let mut t = t_near;
    let mut accum_r = 0.0_f32;
    let mut accum_g = 0.0_f32;
    let mut accum_b = 0.0_f32;
    let mut accum_a = 0.0_f32;

    while t < t_far && accum_a < 0.95 {
        let p = origin + dir * t;
        let local = p - obj.position;

        // Density from FBM noise, modulated by height and shape distance
        let noise_p = Vec3::new(
            local.x * noise_scale,
            local.y * noise_scale - time * speed,
            local.z * noise_scale,
        );
        let raw_density = noise::fbm3d(noise_p, 3);

        // Fade by distance from center and height
        let dist_from_center = local.length_xz() / bounding_radius;
        let height_factor = (1.0 - (local.y / bounding_radius).abs()).max(0.0);
        let shape_factor = (1.0 - dist_from_center).max(0.0) * height_factor;

        let density = (raw_density * shape_factor * intensity - 0.2).max(0.0);

        if density > 0.001 {
            // Temperature from height + noise
            let temp = ((local.y / bounding_radius + 0.5).clamp(0.0, 1.0) * 0.7
                + raw_density * 0.3)
                .clamp(0.0, 1.0);
            let fire_color = materials::fire_color_ramp(temp);

            // Front-to-back compositing
            let alpha = (density * step_size * 8.0).min(1.0);
            let contrib = alpha * (1.0 - accum_a);
            accum_r += fire_color.r * contrib;
            accum_g += fire_color.g * contrib;
            accum_b += fire_color.b * contrib;
            accum_a += contrib;
        }

        t += step_size;
    }

    Color {
        r: accum_r,
        g: accum_g,
        b: accum_b,
        a: accum_a,
    }
}

/// Animated water surface displacement (superposed sine waves).
fn water_displacement(x: f32, z: f32, time: f32, amplitude: f32, frequency: f32) -> f32 {
    let w1 = (x * frequency + time * 2.0).sin() * amplitude;
    let w2 = (z * frequency * 0.7 + time * 1.5).sin() * amplitude * 0.6;
    let w3 = ((x + z) * frequency * 1.3 + time * 2.5).sin() * amplitude * 0.3;
    let n = noise::fbm2d(x * 0.5, z * 0.5 + time * 0.3, 2) * amplitude * 0.4;
    w1 + w2 + w3 + n
}

/// Linearly interpolate between two colors.
fn lerp_color(a: Color, b: Color, t: f32) -> Color {
    Color {
        r: a.r + (b.r - a.r) * t,
        g: a.g + (b.g - a.g) * t,
        b: a.b + (b.b - a.b) * t,
        a: a.a + (b.a - a.a) * t,
    }
}

/// Composite `foreground` over `background` (premultiplied-style).
fn blend_over(fg: Color, bg: Color) -> Color {
    let inv_a = 1.0 - fg.a;
    Color {
        r: fg.r + bg.r * inv_a,
        g: fg.g + bg.g * inv_a,
        b: fg.b + bg.b * inv_a,
        a: fg.a + bg.a * inv_a,
    }
}

// ── Profiled variants ───────────────────────────────────────────

/// Profiled variant of [`shade_ray`]. Times march, fire, and delegates to
/// profiled surface shading.
fn shade_ray_profiled(
    scene: &SdfScene,
    origin: Vec3,
    dir: Vec3,
    time: f32,
    bounce: u32,
    prof: &mut RowProfile,
) -> Color {
    // Fire volume (primary rays only)
    if bounce == 0 && scene.has_fire {
        for obj in &scene.objects {
            if let Material::Fire { .. } = &obj.material {
                let t0 = std::time::Instant::now();
                let fire_color = march_fire_volume(origin, dir, obj, time);
                prof.record(SdfStage::Fire, t0);

                if fire_color.a > 0.01 {
                    let t1 = std::time::Instant::now();
                    let hit = ray_march(scene, origin, dir, time);
                    prof.record(SdfStage::March, t1);

                    let bg = if let Some((hit_pt, idx, _)) = hit {
                        shade_surface_profiled(scene, hit_pt, dir, idx, time, bounce, prof)
                    } else {
                        scene.sky_color
                    };
                    return blend_over(fire_color, bg);
                }
            }
        }
    }

    let t0 = std::time::Instant::now();
    let hit = ray_march(scene, origin, dir, time);
    prof.record(SdfStage::March, t0);

    match hit {
        Some((hit_pt, idx, _)) => {
            shade_surface_profiled(scene, hit_pt, dir, idx, time, bounce, prof)
        }
        None => scene.sky_color,
    }
}

/// Profiled variant of [`shade_surface`]. Times normal estimation,
/// shading, and reflections.
fn shade_surface_profiled(
    scene: &SdfScene,
    hit: Vec3,
    ray_dir: Vec3,
    obj_idx: usize,
    time: f32,
    bounce: u32,
    prof: &mut RowProfile,
) -> Color {
    let obj = &scene.objects[obj_idx];

    let t0 = std::time::Instant::now();
    let normal = estimate_normal(scene, hit, obj_idx, time);
    prof.record(SdfStage::Normal, t0);

    match &obj.material {
        Material::Solid {
            color,
            reflectivity,
            specular,
        } => {
            let result = phong_profiled(scene, hit, normal, ray_dir, *color, *specular, time, prof);

            if *reflectivity > 0.01 && bounce < scene.max_bounces {
                let t1 = std::time::Instant::now();
                let refl_dir = ray_dir.reflect(normal);
                let refl_origin = hit + normal * (SURF_DIST * 2.0);
                let refl_color =
                    shade_ray_profiled(scene, refl_origin, refl_dir, time, bounce + 1, prof);
                prof.record(SdfStage::Reflection, t1);
                lerp_color(result, refl_color, *reflectivity)
            } else {
                result
            }
        }
        Material::Water { tint, ior, .. } => {
            let t_shade = std::time::Instant::now();
            let cos_theta = (-ray_dir).dot(normal).max(0.0);
            let f = materials::fresnel(cos_theta, *ior);
            let mut color = *tint;
            color.a = 1.0;

            if bounce < scene.max_bounces {
                let refl_dir = ray_dir.reflect(normal);
                let refl_origin = hit + normal * (SURF_DIST * 2.0);
                prof.record(SdfStage::Shading, t_shade);

                let t_refl = std::time::Instant::now();
                let refl_color =
                    shade_ray_profiled(scene, refl_origin, refl_dir, time, bounce + 1, prof);
                prof.record(SdfStage::Reflection, t_refl);

                let eta = 1.0 / ior;
                let t_refr = std::time::Instant::now();
                let refr_color = if let Some(refr_dir) = ray_dir.refract(normal, eta) {
                    let refr_origin = hit - normal * (SURF_DIST * 2.0);
                    let mut rc =
                        shade_ray_profiled(scene, refr_origin, refr_dir, time, bounce + 1, prof);
                    rc.r *= tint.r;
                    rc.g *= tint.g;
                    rc.b *= tint.b;
                    rc
                } else {
                    refl_color
                };
                prof.record(SdfStage::Reflection, t_refr);

                color = lerp_color(refr_color, refl_color, f);
            } else {
                prof.record(SdfStage::Shading, t_shade);
            }

            let t_spec = std::time::Instant::now();
            let spec = phong_specular(scene, hit, normal, ray_dir, 128.0);
            prof.record(SdfStage::Shading, t_spec);

            color.r = (color.r + spec.r).min(1.0);
            color.g = (color.g + spec.g).min(1.0);
            color.b = (color.b + spec.b).min(1.0);
            color
        }
        Material::Fire { .. } => {
            let t_shade = std::time::Instant::now();
            let glow = materials::fire_color_ramp(0.3);
            prof.record(SdfStage::Shading, t_shade);
            Color {
                r: glow.r * 0.5,
                g: glow.g * 0.5,
                b: glow.b * 0.5,
                a: 1.0,
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
/// Profiled variant of [`phong`]. Times shadow marches separately from
/// the diffuse/specular computation.
fn phong_profiled(
    scene: &SdfScene,
    hit: Vec3,
    normal: Vec3,
    ray_dir: Vec3,
    base_color: Color,
    spec_power: f32,
    time: f32,
    prof: &mut RowProfile,
) -> Color {
    let t_shade = std::time::Instant::now();
    let mut r = base_color.r * scene.ambient;
    let mut g = base_color.g * scene.ambient;
    let mut b = base_color.b * scene.ambient;
    prof.record(SdfStage::Shading, t_shade);

    for light in &scene.lights {
        let to_light = (light.position - hit).normalize();
        let light_dist = (light.position - hit).length();

        let shadow_origin = hit + normal * (SURF_DIST * 4.0);
        let t_shadow = std::time::Instant::now();
        let in_shadow = shadow_march(scene, shadow_origin, to_light, light_dist, time);
        prof.record(SdfStage::Shadow, t_shadow);

        if in_shadow {
            continue;
        }

        let t_shade2 = std::time::Instant::now();
        let intensity = light.intensity;
        let diff = normal.dot(to_light).max(0.0) * intensity;
        r += base_color.r * light.color.r * diff;
        g += base_color.g * light.color.g * diff;
        b += base_color.b * light.color.b * diff;

        let half = (to_light - ray_dir).normalize();
        let spec = normal.dot(half).max(0.0).powf(spec_power) * intensity;
        r += light.color.r * spec * 0.5;
        g += light.color.g * spec * 0.5;
        b += light.color.b * spec * 0.5;
        prof.record(SdfStage::Shading, t_shade2);
    }

    Color {
        r: r.min(1.0),
        g: g.min(1.0),
        b: b.min(1.0),
        a: 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf::scene::{SdfCamera, SdfLight};

    fn simple_sphere_scene() -> SdfScene {
        SdfScene::new()
            .object(
                SdfObject::new(
                    SdfShape::Sphere { radius: 1.0 },
                    Material::matte(Color::from_rgba8(200, 50, 50, 255)),
                )
                .at(Vec3::new(0.0, 1.0, 0.0)),
            )
            .light(SdfLight::new(Vec3::new(5.0, 10.0, 5.0), Color::WHITE, 1.0))
            .camera(SdfCamera::new(
                Vec3::new(0.0, 3.0, 6.0),
                Vec3::new(0.0, 1.0, 0.0),
                45.0,
            ))
    }

    #[test]
    fn render_produces_non_black_pixmap() {
        let scene = simple_sphere_scene();
        let pixmap = SdfRenderer::render_to_pixmap(&scene, 64, 48, 0.0).unwrap();

        // Check that we have some non-black, non-uniform pixels
        let pixels = pixmap.pixels();
        let first = pixels[0];
        let has_variation = pixels.iter().any(|p| *p != first);
        assert!(has_variation, "render produced a uniform image");

        let has_nonblack = pixels
            .iter()
            .any(|p| p.red() > 0 || p.green() > 0 || p.blue() > 0);
        assert!(has_nonblack, "render produced an all-black image");
    }

    #[test]
    fn render_to_image_matches_dimensions() {
        let scene = simple_sphere_scene();
        let img = SdfRenderer::render_to_image(&scene, 32, 24, 0.0);
        assert_eq!(img.width(), 32);
        assert_eq!(img.height(), 24);
        assert_eq!(img.data().len(), 32 * 24 * 4);
    }

    #[test]
    fn scene_sdf_finds_sphere() {
        let scene = simple_sphere_scene();
        // Point at center of sphere (0, 1, 0) should be inside (negative distance)
        let (d, _) = scene_sdf(&scene, Vec3::new(0.0, 1.0, 0.0), 0.0);
        assert!(d < 0.0, "center of sphere should be inside: {d}");
    }

    #[test]
    fn ray_march_hits_sphere() {
        let scene = simple_sphere_scene();
        let origin = Vec3::new(0.0, 1.0, 6.0);
        let dir = Vec3::new(0.0, 0.0, -1.0);
        let hit = ray_march(&scene, origin, dir, 0.0);
        assert!(hit.is_some(), "ray should hit sphere");
        let (p, _, _) = hit.unwrap();
        // Hit point Z should be near the sphere surface at z ≈ 1.0
        assert!((p.z - 1.0).abs() < 0.1, "hit z={} expected near 1.0", p.z);
    }
}
