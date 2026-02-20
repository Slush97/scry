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

const MAX_DIST: f32 = 50.0;
const SURF_DIST: f32 = 0.002;
const NORMAL_EPS: f32 = 0.002;

/// Over-relaxation factor for enhanced sphere tracing (Keinert et al. 2014).
const OMEGA: f32 = 1.6;

/// Distance threshold: below this, fall back to conservative stepping.
const RELAX_DIST: f32 = 0.1;

/// Soft shadow penumbra sharpness.
const SHADOW_K: f32 = 16.0;

/// Per-bounce quality budget that degrades gracefully for deeper bounces.
struct RayBudget {
    march_steps: u32,
    shadow_steps: u32,
    do_shadows: bool,
}

impl RayBudget {
    fn for_bounce(bounce: u32) -> Self {
        match bounce {
            0 => Self {
                march_steps: 128,
                shadow_steps: 32,
                do_shadows: true,
            },
            1 => Self {
                march_steps: 64,
                shadow_steps: 16,
                do_shadows: true,
            },
            _ => Self {
                march_steps: 48,
                shadow_steps: 0,
                do_shadows: false,
            },
        }
    }
}

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

        let black = tiny_skia::PremultipliedColorU8::from_rgba(0, 0, 0, 255).unwrap();
        let mut pixel_buf = vec![black; (width * height) as usize];

        pixel_buf
            .par_chunks_mut(width as usize)
            .enumerate()
            .for_each(|(y, row)| {
                let ndc_y = (1.0 - 2.0 * (y as f32 + 0.5) / height as f32) * fov_scale;
                for (x, pixel) in row.iter_mut().enumerate() {
                    let ndc_x = (2.0 * (x as f32 + 0.5) / width as f32 - 1.0) * aspect * fov_scale;
                    let dir = (cam_right * ndc_x + cam_up * ndc_y + cam_fwd).normalize();
                    let color = shade_ray(scene, scene.camera.eye, dir, time, 0, None);
                    let (r, g, b) = if scene.tone_map {
                        (tone_map_reinhard(color.r), tone_map_reinhard(color.g), tone_map_reinhard(color.b))
                    } else {
                        (color.r, color.g, color.b)
                    };
                    let a = (color.a.clamp(0.0, 1.0) * 255.0) as u8;
                    // Premultiplied constraint: R,G,B must not exceed A.
                    // Scale RGB by alpha so transparent pixels don't
                    // violate the invariant (glass + transparent sky
                    // can produce non-zero RGB with near-zero alpha).
                    let r8 = gamma_encode(r);
                    let g8 = gamma_encode(g);
                    let b8 = gamma_encode(b);
                    let (r8, g8, b8) = if a < 255 {
                        (r8.min(a), g8.min(a), b8.min(a))
                    } else {
                        (r8, g8, b8)
                    };
                    *pixel = tiny_skia::PremultipliedColorU8::from_rgba(r8, g8, b8, a)
                    .unwrap();
                }
            });

        pixmap.pixels_mut().copy_from_slice(&pixel_buf);

        // Composite billboard text labels on top of the rendered scene.
        #[cfg(feature = "text")]
        Self::composite_text_labels(&mut pixmap, scene, width, height);

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
                            None,
                            &mut row_profile,
                        );
                        let (r, g, b) = if scene.tone_map {
                            (tone_map_reinhard(color.r), tone_map_reinhard(color.g), tone_map_reinhard(color.b))
                        } else {
                            (color.r, color.g, color.b)
                        };
                        let a = (color.a.clamp(0.0, 1.0) * 255.0) as u8;
                        let r8 = gamma_encode(r);
                        let g8 = gamma_encode(g);
                        let b8 = gamma_encode(b);
                        let (r8, g8, b8) = if a < 255 {
                            (r8.min(a), g8.min(a), b8.min(a))
                        } else {
                            (r8, g8, b8)
                        };
                        tiny_skia::PremultipliedColorU8::from_rgba(r8, g8, b8, a)
                        .unwrap()
                    })
                    .collect();
                (row_pixels, row_profile)
            })
            .collect();

        let pixels = pixmap.pixels_mut();
        let mut row_profiles = Vec::with_capacity(height as usize);
        for (y, (row, profile)) in row_results.into_iter().enumerate() {
            let offset = y * width as usize;
            pixels[offset..offset + width as usize].copy_from_slice(&row);
            row_profiles.push(profile);
        }

        let total_us = frame_start.elapsed().as_micros() as u64;
        let sdf_profile = SdfProfile::from_rows(&row_profiles, total_us, width, height);

        #[cfg(feature = "text")]
        Self::composite_text_labels(&mut pixmap, scene, width, height);

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

        #[cfg(feature = "text")]
        Self::composite_text_labels(&mut pixmap, scene, target_width, target_height);

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

        #[cfg(feature = "text")]
        Self::composite_text_labels(&mut pixmap, scene, target_width, target_height);

        Ok((pixmap, profile))
    }

    /// Render the scene to an [`ImageData`] for compositing onto a `PixelCanvas`.
    pub fn render_to_image(scene: &SdfScene, width: u32, height: u32, time: f32) -> ImageData {
        let (cam_right, cam_up, cam_fwd) =
            math::look_at(scene.camera.eye, scene.camera.target, Vec3::UP);
        let fov_scale = (scene.camera.fov.to_radians() * 0.5).tan();
        let aspect = width as f32 / height as f32;

        let mut data = vec![0u8; (width * height * 4) as usize];

        data.par_chunks_mut(width as usize * 4)
            .enumerate()
            .for_each(|(y, row)| {
                let ndc_y = (1.0 - 2.0 * (y as f32 + 0.5) / height as f32) * fov_scale;
                for x in 0..width as usize {
                    let ndc_x = (2.0 * (x as f32 + 0.5) / width as f32 - 1.0) * aspect * fov_scale;
                    let dir = (cam_right * ndc_x + cam_up * ndc_y + cam_fwd).normalize();
                    let color = shade_ray(scene, scene.camera.eye, dir, time, 0, None);
                    let (r, g, b) = if scene.tone_map {
                        (tone_map_reinhard(color.r), tone_map_reinhard(color.g), tone_map_reinhard(color.b))
                    } else {
                        (color.r, color.g, color.b)
                    };
                    let a = (color.a.clamp(0.0, 1.0) * 255.0) as u8;
                    let off = x * 4;
                    row[off] = gamma_encode(r);
                    row[off + 1] = gamma_encode(g);
                    row[off + 2] = gamma_encode(b);
                    row[off + 3] = a;
                }
            });

        ImageData::new(width, height, data)
    }

    /// Composite billboard text labels onto a rendered SDF pixmap.
    ///
    /// For each label, projects the world-space position through the camera's
    /// view + perspective matrices to screen space, then renders the text at
    /// that position using the Phase 1 rich text pipeline.
    #[cfg(feature = "text")]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn composite_text_labels(
        pixmap: &mut Pixmap,
        scene: &SdfScene,
        width: u32,
        height: u32,
    ) {
        if scene.text_labels.is_empty() {
            return;
        }

        let (cam_right, cam_up, cam_fwd) =
            math::look_at(scene.camera.eye, scene.camera.target, Vec3::UP);
        let fov_scale = (scene.camera.fov.to_radians() * 0.5).tan();
        let aspect = width as f32 / height as f32;

        // Collect labels with their depth for back-to-front sorting
        let mut label_depths: Vec<(usize, f32)> = Vec::with_capacity(scene.text_labels.len());
        for (i, label) in scene.text_labels.iter().enumerate() {
            let to_label = label.position - scene.camera.eye;
            let depth = to_label.dot(cam_fwd);
            if depth > 0.1 {
                label_depths.push((i, depth));
            }
        }

        // Sort back-to-front (furthest first)
        label_depths.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut gc = std::collections::HashMap::new();
        for (idx, depth) in label_depths {
            let label = &scene.text_labels[idx];
            let to_label = label.position - scene.camera.eye;

            // Project to NDC
            let ndc_x = to_label.dot(cam_right) / (depth * aspect * fov_scale);
            let ndc_y = to_label.dot(cam_up) / (depth * fov_scale);

            // NDC → screen
            let screen_x = (ndc_x + 1.0) * 0.5 * width as f32;
            let screen_y = (1.0 - ndc_y) * 0.5 * height as f32;

            // Perspective-scaled font size
            let font_size = if label.perspective_scale {
                (label.font_size / depth).clamp(4.0, label.font_size * 4.0)
            } else {
                label.font_size
            };

            let fd = label
                .font_data
                .clone()
                .unwrap_or_else(crate::rasterize::skia::text::default_font);

            // Measure to center the label horizontally
            let metrics =
                crate::rasterize::skia::text::measure_text(&label.text, Some(&fd), font_size);
            let draw_x = screen_x - metrics.width * 0.5;
            let draw_y = screen_y;

            crate::rasterize::skia::Rasterizer::render_rich_text(
                pixmap,
                &label.text,
                draw_x,
                draw_y,
                font_size,
                &label.color,
                &fd,
                tiny_skia::Transform::identity(),
                crate::scene::command::TextAlign::Left,
                label.outline.as_ref().map(|(c, _)| c),
                label.outline.map(|(_, w)| w),
                label.fill_style.as_ref(),
                None,
                &mut gc,
            );
        }
    }
}

// ── Gamma correction ────────────────────────────────────────────────

/// Encode a linear-space color channel to sRGB (gamma 1/2.2), returning a u8.
#[inline]
fn gamma_encode(linear: f32) -> u8 {
    (linear.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8
}

/// Reinhard tone mapping: maps HDR [0, ∞) to LDR [0, 1).
#[inline]
fn tone_map_reinhard(c: f32) -> f32 {
    c / (1.0 + c)
}

/// Apply distance fog: blends `color` toward `fog_color` based on distance.
#[inline]
fn apply_fog(color: Color, fog_color: Color, fog_density: f32, distance: f32) -> Color {
    let fog_factor = (-fog_density * distance).exp();
    lerp_color(fog_color, color, fog_factor)
}

// ── Internal ray marching ───────────────────────────────────────────

/// Evaluate the entire scene SDF at `point`, returning `(distance, object_index)`.
///
/// Uses per-object bounding spheres for early culling: if the distance from
/// `point` to the object's center minus the bounding radius already exceeds
/// `min_dist`, the full SDF evaluation is skipped. Planes have
/// `bounding_radius = INFINITY` so they are never culled.
#[inline]
fn scene_sdf(scene: &SdfScene, point: Vec3, time: f32) -> (f32, usize) {
    let mut min_dist = MAX_DIST;
    let mut closest = 0;

    for (i, obj) in scene.objects.iter().enumerate() {
        // Bounding-sphere cull (squared-distance version — avoids sqrt).
        // The SDF of a shape can never be less than
        // (center_distance - bounding_radius). If that already exceeds
        // `min_dist`, this object can't be the closest — skip it.
        let delta = point - obj.position;
        let dist_sq = delta.length_sq();
        let threshold = min_dist + obj.bounding_radius;
        if dist_sq > threshold * threshold {
            continue;
        }

        let d = object_sdf(obj, point, time);
        if d < min_dist {
            min_dist = d;
            closest = i;
        }
    }

    (min_dist, closest)
}

/// Like [`scene_sdf`] but skips the object at `exclude_idx` (when `Some`).
/// Used by glass refraction rays to avoid self-intersecting the glass body.
#[inline]
fn scene_sdf_exclude(
    scene: &SdfScene,
    point: Vec3,
    time: f32,
    exclude_idx: Option<usize>,
) -> (f32, usize) {
    let Some(exclude) = exclude_idx else {
        return scene_sdf(scene, point, time);
    };

    let mut min_dist = MAX_DIST;
    let mut closest = 0;

    for (i, obj) in scene.objects.iter().enumerate() {
        if i == exclude {
            continue;
        }
        let delta = point - obj.position;
        let dist_sq = delta.length_sq();
        let threshold = min_dist + obj.bounding_radius;
        if dist_sq > threshold * threshold {
            continue;
        }
        let d = object_sdf(obj, point, time);
        if d < min_dist {
            min_dist = d;
            closest = i;
        }
    }

    (min_dist, closest)
}

/// Evaluate the SDF for a single object, handling water displacement and rotation.
#[inline]
fn object_sdf(obj: &SdfObject, point: Vec3, time: f32) -> f32 {
    let mut local = primitives::translate(point, obj.position);

    // Apply inverse rotation to the evaluation point (domain rotation).
    // Quaternion orientation takes precedence over Y-axis rotation.
    if let Some(q) = obj.orientation {
        local = q.conjugate().rotate_vec3(local);
    } else if let Some((cos_y, sin_y)) = obj.rotation_y {
        // Inverse rotation: rotate by -angle (swap sin sign)
        let rx = local.x * cos_y - local.z * sin_y;
        let rz = local.x * sin_y + local.z * cos_y;
        local = Vec3::new(rx, local.y, rz);
    }

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
#[inline]
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
        SdfShape::Subtract { a, b, b_offset } => {
            let da = shape_sdf(a, p);
            let db = shape_sdf(b, p - *b_offset);
            primitives::op_subtract(da, db)
        }
        SdfShape::Capsule {
            radius,
            half_height,
        } => primitives::sd_capsule(p, *radius, *half_height),
        SdfShape::RoundedBox {
            half_extents,
            radius,
        } => primitives::sd_rounded_box(p, *half_extents, *radius),
        SdfShape::Cone { radius, height } => primitives::sd_cone(p, *radius, *height),
        SdfShape::Mandelbulb { power, iterations } => {
            primitives::sd_mandelbulb(p, *power, *iterations)
        }
        SdfShape::MengerSponge { iterations } => {
            primitives::sd_menger_sponge(p, *iterations)
        }
        SdfShape::Gyroid { scale, thickness, bound } => {
            primitives::sd_gyroid(p, *scale, *thickness, *bound)
        }
        #[cfg(feature = "sdf-text")]
        SdfShape::Text3D { layout, depth } => super::glyph::sd_text3d(layout, p, *depth),
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
#[inline]
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
///
/// Uses enhanced sphere tracing (over-relaxation ω=1.6) for ~30-40% fewer
/// steps on average. Step budget comes from `RayBudget` based on bounce depth.
/// Distance-adaptive relaxation: uses ω=1.6 far from surfaces, ω=1.0 near
/// them. Includes rewind-on-overshoot fallback for correctness.
#[inline]
fn ray_march(
    scene: &SdfScene,
    origin: Vec3,
    dir: Vec3,
    time: f32,
    budget: &RayBudget,
    exclude_idx: Option<usize>,
) -> Option<(Vec3, usize, f32)> {
    let mut t = SURF_DIST;
    let mut prev_d = 0.0_f32;
    let mut prev_step = 0.0_f32;
    // Primary rays on non-water scenes: always use full relaxation.
    let always_relax = !scene.has_water;
    let mut omega = if always_relax { OMEGA } else { 1.0_f32 };

    for _ in 0..budget.march_steps {
        let p = origin + dir * t;
        let (d, idx) = scene_sdf_exclude(scene, p, time, exclude_idx);
        if d < SURF_DIST {
            return Some((p, idx, t));
        }

        if !always_relax {
            // Distance-adaptive: relax when far from surfaces, conservative when close.
            omega = if d > RELAX_DIST { OMEGA } else { 1.0 };
        }

        // Over-relaxation with automatic fallback: if the two successive SDF
        // balls don't cover the step we took, we may have overshot a surface.
        // Rewind to the safe conservative position and go conservative (ω=1)
        // for the rest of the march.
        if omega > 1.0 && prev_step > 0.0 && d + prev_d < prev_step {
            t -= prev_step - prev_d; // rewind to prev_pos + prev_d (safe)
            omega = 1.0;
            prev_d = 0.0;
            prev_step = 0.0;
            continue; // re-evaluate SDF at the safe position
        }

        let step = d * omega;
        prev_d = d;
        prev_step = step;
        t += step;
        if t > MAX_DIST {
            break;
        }
    }
    None
}

/// Soft shadow factor using IQ's penumbra technique.
/// Returns 1.0 (fully lit) to 0.0 (fully shadowed).
/// Step count comes from the `RayBudget`.
#[inline]
fn soft_shadow(
    scene: &SdfScene,
    origin: Vec3,
    dir: Vec3,
    max_t: f32,
    time: f32,
    shadow_steps: u32,
) -> f32 {
    // Fast-path: if the shadow ray misses the scene bounding sphere entirely,
    // no finite object can cast a shadow — return fully lit immediately.
    // (Planes don't cast shadows on themselves in typical scenes.)
    if scene.scene_radius > 0.0 {
        let oc = origin - scene.scene_center;
        let b = oc.dot(dir);
        let c = oc.dot(oc) - scene.scene_radius * scene.scene_radius;
        let disc = b * b - c;
        if disc < 0.0 {
            return 1.0; // ray misses all finite objects
        }
        let sqrt_disc = disc.sqrt();
        let t_enter = (-b - sqrt_disc).max(0.0);
        if t_enter > max_t {
            return 1.0; // intersection is beyond the light
        }
    }

    let mut t = SURF_DIST * 4.0;
    let mut res = 1.0_f32;
    for _ in 0..shadow_steps {
        let p = origin + dir * t;
        let (d, _) = scene_sdf(scene, p, time);
        if d < SURF_DIST * 0.5 {
            return 0.0;
        }
        res = res.min(SHADOW_K * d / t);
        t += d.clamp(0.01, 1.0);
        if t > max_t {
            break;
        }
    }
    res.clamp(0.0, 1.0)
}

/// SDF-based ambient occlusion (5 samples along normal).
#[inline]
fn ambient_occlusion(scene: &SdfScene, hit: Vec3, normal: Vec3, time: f32) -> f32 {
    let mut occ = 0.0_f32;
    let mut scale = 1.0_f32;
    for i in 1..=5 {
        let dist = 0.02 * i as f32;
        let (d, _) = scene_sdf(scene, hit + normal * dist, time);
        occ += (dist - d) * scale;
        scale *= 0.75;
    }
    (1.0 - occ.clamp(0.0, 1.0)).max(0.0)
}

/// Phong lighting without shadow marches (used for reflection bounces).
#[inline]
fn phong_no_shadows(
    scene: &SdfScene,
    hit: Vec3,
    normal: Vec3,
    ray_dir: Vec3,
    base_color: Color,
    spec_power: f32,
) -> Color {
    let mut r = base_color.r * scene.ambient;
    let mut g = base_color.g * scene.ambient;
    let mut b = base_color.b * scene.ambient;

    for light in &scene.lights {
        let delta = light.position - hit;
        let light_dist = delta.length();
        let to_light = delta * (1.0 / light_dist);

        // Backface cull
        let n_dot_l = normal.dot(to_light);
        if n_dot_l <= 0.0 {
            continue;
        }

        let intensity = light.intensity;

        // Diffuse
        let diff = n_dot_l * intensity;
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
#[inline]
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
#[inline]
fn lerp_color(a: Color, b: Color, t: f32) -> Color {
    Color {
        r: a.r + (b.r - a.r) * t,
        g: a.g + (b.g - a.g) * t,
        b: a.b + (b.b - a.b) * t,
        a: a.a + (b.a - a.a) * t,
    }
}

/// Composite `foreground` over `background` (premultiplied-style).
#[inline]
fn blend_over(fg: Color, bg: Color) -> Color {
    let inv_a = 1.0 - fg.a;
    Color {
        r: fg.r + bg.r * inv_a,
        g: fg.g + bg.g * inv_a,
        b: fg.b + bg.b * inv_a,
        a: fg.a + bg.a * inv_a,
    }
}

// ── Tracing infrastructure ──────────────────────────────────────────
//
// Instead of duplicating every shading function with a `_profiled`
// variant, we parameterise by a zero-cost `SdfTracer` trait. The
// `NoOpTracer` implementation compiles to nothing; `ProfilingTracer`
// records per-stage timings into a `RowProfile`.

/// Compile-time hooks for optional per-stage timing.
trait SdfTracer {
    /// Called before a rendering stage begins.
    fn begin(&mut self, stage: SdfStage);
    /// Called after a rendering stage ends (records elapsed time since `begin`).
    fn end(&mut self, stage: SdfStage);
}

/// Zero-cost tracer — all calls are eliminated by the compiler.
struct NoOpTracer;

impl SdfTracer for NoOpTracer {
    #[inline(always)]
    fn begin(&mut self, _stage: SdfStage) {}
    #[inline(always)]
    fn end(&mut self, _stage: SdfStage) {}
}

/// Profiling tracer that records per-stage wall-clock time.
struct ProfilingTracer<'a> {
    prof: &'a mut RowProfile,
    starts: [Option<std::time::Instant>; 6],
}

impl<'a> ProfilingTracer<'a> {
    fn new(prof: &'a mut RowProfile) -> Self {
        Self {
            prof,
            starts: [None; 6],
        }
    }
}

impl SdfTracer for ProfilingTracer<'_> {
    #[inline]
    fn begin(&mut self, stage: SdfStage) {
        self.starts[stage.index()] = Some(std::time::Instant::now());
    }
    #[inline]
    fn end(&mut self, stage: SdfStage) {
        if let Some(start) = self.starts[stage.index()].take() {
            self.prof.record(stage, start);
        }
    }
}

// ── Unified shading functions ───────────────────────────────────────

/// Top-level per-ray shading: march, hit → shade surface, miss → sky.
fn shade_ray(scene: &SdfScene, origin: Vec3, dir: Vec3, time: f32, bounce: u32, exclude_idx: Option<usize>) -> Color {
    shade_ray_traced(scene, origin, dir, time, bounce, exclude_idx, &mut NoOpTracer)
}

/// Profiled variant of [`shade_ray`].
fn shade_ray_profiled(
    scene: &SdfScene,
    origin: Vec3,
    dir: Vec3,
    time: f32,
    bounce: u32,
    exclude_idx: Option<usize>,
    prof: &mut RowProfile,
) -> Color {
    shade_ray_traced(
        scene,
        origin,
        dir,
        time,
        bounce,
        exclude_idx,
        &mut ProfilingTracer::new(prof),
    )
}

/// Generic shade_ray parameterised by tracer.
fn shade_ray_traced<T: SdfTracer>(
    scene: &SdfScene,
    origin: Vec3,
    dir: Vec3,
    time: f32,
    bounce: u32,
    exclude_idx: Option<usize>,
    tracer: &mut T,
) -> Color {
    let budget = RayBudget::for_bounce(bounce);

    // Sky fast-path for bounce rays: if the ray misses the scene bounding
    // sphere entirely, skip the march and return sky immediately.
    // Disabled for water scenes — upward bounce rays can still graze the
    // displaced water surface just above y=0.
    if bounce > 0 && !scene.has_water && !scene.has_glass && scene.scene_radius > 0.0 {
        let oc = scene.scene_center - origin;
        let proj = oc.dot(dir);
        let oc_sq = oc.dot(oc);
        let perp_sq = oc_sq - proj * proj;
        let r_sq = scene.scene_radius * scene.scene_radius;
        if perp_sq > r_sq || (proj < 0.0 && oc_sq > r_sq) {
            // Ray misses the bounding sphere entirely, or sphere is behind us
            // and we're outside it — only sky (or ground plane) ahead.
            if dir.y > 0.0 {
                return scene.sky_color;
            }
            // Downward ray: will hit the ground plane. Let the march handle it
            // but it will converge quickly since plane SDF is cheap.
        }
    }

    // Fire volume (primary rays only)
    if bounce == 0 && scene.has_fire {
        for obj in &scene.objects {
            if let Material::Fire { .. } = &obj.material {
                tracer.begin(SdfStage::Fire);
                let fire_color = march_fire_volume(origin, dir, obj, time);
                tracer.end(SdfStage::Fire);

                if fire_color.a > 0.01 {
                    tracer.begin(SdfStage::March);
                    let hit = ray_march(scene, origin, dir, time, &budget, None);
                    tracer.end(SdfStage::March);

                    let bg = if let Some((hit_pt, idx, _)) = hit {
                        shade_surface_traced(scene, hit_pt, dir, idx, time, bounce, &budget, tracer)
                    } else {
                        scene.sky_color
                    };
                    return blend_over(fire_color, bg);
                }
            }
        }
    }

    tracer.begin(SdfStage::March);
    let hit = ray_march(scene, origin, dir, time, &budget, exclude_idx);
    tracer.end(SdfStage::March);

    match hit {
        Some((hit_pt, idx, dist)) => {
            let mut color =
                shade_surface_traced(scene, hit_pt, dir, idx, time, bounce, &budget, tracer);
            // Apply distance fog (only on primary rays)
            if bounce == 0 && scene.fog_density > 0.0 {
                color = apply_fog(color, scene.fog_color, scene.fog_density, dist);
            }
            color
        }
        None => scene.sky_color,
    }
}

/// Shade a surface hit point with Phong lighting, reflections, and water effects.
fn shade_surface_traced<T: SdfTracer>(
    scene: &SdfScene,
    hit: Vec3,
    ray_dir: Vec3,
    obj_idx: usize,
    time: f32,
    bounce: u32,
    budget: &RayBudget,
    tracer: &mut T,
) -> Color {
    let obj = &scene.objects[obj_idx];

    tracer.begin(SdfStage::Normal);
    let normal = estimate_normal(scene, hit, obj_idx, time);
    tracer.end(SdfStage::Normal);

    match &obj.material {
        Material::Solid {
            color,
            reflectivity,
            specular,
        } => {
            // Skip expensive shadow marches on reflection bounces
            let result = if bounce > 0 {
                tracer.begin(SdfStage::Shading);
                let r = phong_no_shadows(scene, hit, normal, ray_dir, *color, *specular);
                tracer.end(SdfStage::Shading);
                r
            } else {
                phong_traced(scene, hit, normal, ray_dir, *color, *specular, time, budget, tracer)
            };

            // Reflections
            if *reflectivity > 0.01 && bounce < scene.max_bounces {
                tracer.begin(SdfStage::Reflection);
                let refl_dir = ray_dir.reflect(normal);
                let refl_origin = hit + normal * (SURF_DIST * 2.0);
                let refl_color =
                    shade_ray_traced(scene, refl_origin, refl_dir, time, bounce + 1, None, tracer);
                tracer.end(SdfStage::Reflection);
                lerp_color(result, refl_color, *reflectivity)
            } else {
                result
            }
        }
        Material::Water { tint, ior, .. } => {
            tracer.begin(SdfStage::Shading);
            let cos_theta = (-ray_dir).dot(normal).max(0.0);
            let f = materials::fresnel(cos_theta, *ior);
            let mut color = *tint;
            color.a = 1.0;

            if bounce < scene.max_bounces {
                let refl_dir = ray_dir.reflect(normal);
                let refl_origin = hit + normal * (SURF_DIST * 2.0);
                tracer.end(SdfStage::Shading);

                tracer.begin(SdfStage::Reflection);
                let refl_color =
                    shade_ray_traced(scene, refl_origin, refl_dir, time, bounce + 1, None, tracer);
                tracer.end(SdfStage::Reflection);

                let eta = 1.0 / ior;
                tracer.begin(SdfStage::Reflection);
                let refr_color = if let Some(refr_dir) = ray_dir.refract(normal, eta) {
                    let refr_origin = hit - normal * (SURF_DIST * 2.0);
                    let mut rc =
                        shade_ray_traced(scene, refr_origin, refr_dir, time, bounce + 1, None, tracer);
                    rc.r *= tint.r;
                    rc.g *= tint.g;
                    rc.b *= tint.b;
                    rc
                } else {
                    refl_color // Total internal reflection
                };
                tracer.end(SdfStage::Reflection);

                color = lerp_color(refr_color, refl_color, f);
            } else {
                tracer.end(SdfStage::Shading);
            }

            // Add specular highlights from lights
            tracer.begin(SdfStage::Shading);
            let spec = phong_specular(scene, hit, normal, ray_dir, 128.0);
            tracer.end(SdfStage::Shading);

            color.r = (color.r + spec.r).min(1.0);
            color.g = (color.g + spec.g).min(1.0);
            color.b = (color.b + spec.b).min(1.0);
            color
        }
        Material::Fire { .. } => {
            tracer.begin(SdfStage::Shading);
            let glow = materials::fire_color_ramp(0.3);
            tracer.end(SdfStage::Shading);
            Color {
                r: glow.r * 0.5,
                g: glow.g * 0.5,
                b: glow.b * 0.5,
                a: 1.0,
            }
        }
        Material::Checkerboard {
            color_a,
            color_b,
            scale,
            reflectivity,
            specular,
        } => {
            // Determine which square we're in
            let checker = ((hit.x / scale).floor() as i32 + (hit.z / scale).floor() as i32) & 1;
            let base_color = if checker == 0 { *color_a } else { *color_b };

            let result = if bounce > 0 {
                tracer.begin(SdfStage::Shading);
                let r = phong_no_shadows(scene, hit, normal, ray_dir, base_color, *specular);
                tracer.end(SdfStage::Shading);
                r
            } else {
                phong_traced(scene, hit, normal, ray_dir, base_color, *specular, time, budget, tracer)
            };

            if *reflectivity > 0.01 && bounce < scene.max_bounces {
                tracer.begin(SdfStage::Reflection);
                let refl_dir = ray_dir.reflect(normal);
                let refl_origin = hit + normal * (SURF_DIST * 2.0);
                let refl_color =
                    shade_ray_traced(scene, refl_origin, refl_dir, time, bounce + 1, None, tracer);
                tracer.end(SdfStage::Reflection);
                lerp_color(result, refl_color, *reflectivity)
            } else {
                result
            }
        }
        Material::Glass {
            tint,
            ior,
            opacity,
            dispersion,
        } => {
            tracer.begin(SdfStage::Shading);
            let cos_theta = (-ray_dir).dot(normal).max(0.0);
            let f = materials::fresnel(cos_theta, *ior);
            tracer.end(SdfStage::Shading);

            if bounce < scene.max_bounces {
                // Reflection
                let refl_dir = ray_dir.reflect(normal);
                let refl_origin = hit + normal * (SURF_DIST * 2.0);
                tracer.begin(SdfStage::Reflection);
                let refl_color =
                    shade_ray_traced(scene, refl_origin, refl_dir, time, bounce + 1, None, tracer);
                tracer.end(SdfStage::Reflection);

                // Refraction (with optional chromatic dispersion)
                tracer.begin(SdfStage::Reflection);
                let refr_color = if *dispersion > 0.001 {
                    // Chromatic aberration: different IOR per channel
                    let ior_r = 1.0 / (*ior - *dispersion);
                    let ior_g = 1.0 / *ior;
                    let ior_b = 1.0 / (*ior + *dispersion);
                    let refr_origin = hit - normal * (SURF_DIST * 2.0);

                    let exclude = Some(obj_idx);
                    let shade_channel = |eta: f32| -> f32 {
                        if let Some(refr_dir) = ray_dir.refract(normal, eta) {
                            let rc = shade_ray(scene, refr_origin, refr_dir, time, bounce + 1, exclude);
                            // Return luminance-ish single channel (we pick per-channel below)
                            rc.r * 0.33 + rc.g * 0.34 + rc.b * 0.33
                        } else {
                            // Total internal reflection fallback
                            let rc = shade_ray(scene, refl_origin, refl_dir, time, bounce + 1, None);
                            rc.r * 0.33 + rc.g * 0.34 + rc.b * 0.33
                        }
                    };

                    // Refract each channel separately for prismatic effect
                    let mut cr = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
                    if let Some(dir_r) = ray_dir.refract(normal, ior_r) {
                        let c = shade_ray(scene, hit - normal * (SURF_DIST * 2.0), dir_r, time, bounce + 1, exclude);
                        cr.r = c.r * tint.r;
                    } else {
                        cr.r = shade_channel(ior_r) * tint.r;
                    }
                    if let Some(dir_g) = ray_dir.refract(normal, ior_g) {
                        let c = shade_ray(scene, hit - normal * (SURF_DIST * 2.0), dir_g, time, bounce + 1, exclude);
                        cr.g = c.g * tint.g;
                    } else {
                        cr.g = shade_channel(ior_g) * tint.g;
                    }
                    if let Some(dir_b) = ray_dir.refract(normal, ior_b) {
                        let c = shade_ray(scene, hit - normal * (SURF_DIST * 2.0), dir_b, time, bounce + 1, exclude);
                        cr.b = c.b * tint.b;
                    } else {
                        cr.b = shade_channel(ior_b) * tint.b;
                    }
                    cr
                } else {
                    // Single IOR refraction
                    let eta = 1.0 / ior;
                    if let Some(refr_dir) = ray_dir.refract(normal, eta) {
                        let refr_origin = hit - normal * (SURF_DIST * 2.0);
                        let mut rc =
                            shade_ray_traced(scene, refr_origin, refr_dir, time, bounce + 1, Some(obj_idx), tracer);
                        rc.r *= tint.r;
                        rc.g *= tint.g;
                        rc.b *= tint.b;
                        rc
                    } else {
                        refl_color // Total internal reflection
                    }
                };
                tracer.end(SdfStage::Reflection);

                // Blend reflection and refraction via Fresnel
                let mut color = lerp_color(refr_color, refl_color, f);

                // Opacity: blend in tint color
                if *opacity > 0.001 {
                    color = lerp_color(color, *tint, *opacity);
                }

                // Add specular highlights
                tracer.begin(SdfStage::Shading);
                let spec = phong_specular(scene, hit, normal, ray_dir, 128.0);
                tracer.end(SdfStage::Shading);
                color.r = (color.r + spec.r).min(1.0);
                color.g = (color.g + spec.g).min(1.0);
                color.b = (color.b + spec.b).min(1.0);
                color
            } else {
                // Bounce budget exhausted: return tinted ambient
                tracer.begin(SdfStage::Shading);
                let base = Color {
                    r: tint.r * scene.ambient,
                    g: tint.g * scene.ambient,
                    b: tint.b * scene.ambient,
                    a: 1.0,
                };
                tracer.end(SdfStage::Shading);
                base
            }
        }
        Material::Rainbow {
            saturation,
            lightness,
            hue_offset,
            specular,
        } => {
            // Compute local-space position for angular color mapping
            let local = hit - obj.position;
            let angle = local.z.atan2(local.x); // [-π, π]
            let hue = angle / std::f32::consts::TAU + 0.5 + hue_offset / std::f32::consts::TAU;
            let base_color = materials::hsl_to_color(hue, *saturation, *lightness);

            if bounce > 0 {
                tracer.begin(SdfStage::Shading);
                let r = phong_no_shadows(scene, hit, normal, ray_dir, base_color, *specular);
                tracer.end(SdfStage::Shading);
                r
            } else {
                phong_traced(scene, hit, normal, ray_dir, base_color, *specular, time, budget, tracer)
            }
        }
    }
}

/// Phong lighting: ambient + diffuse + specular from all lights.
#[allow(clippy::too_many_arguments)]
fn phong_traced<T: SdfTracer>(
    scene: &SdfScene,
    hit: Vec3,
    normal: Vec3,
    ray_dir: Vec3,
    base_color: Color,
    spec_power: f32,
    time: f32,
    budget: &RayBudget,
    tracer: &mut T,
) -> Color {
    tracer.begin(SdfStage::Shading);
    let ao = ambient_occlusion(scene, hit, normal, time);
    let mut r = base_color.r * scene.ambient * ao;
    let mut g = base_color.g * scene.ambient * ao;
    let mut b = base_color.b * scene.ambient * ao;
    tracer.end(SdfStage::Shading);

    for light in &scene.lights {
        let delta = light.position - hit;
        let light_dist = delta.length();
        let to_light = delta * (1.0 / light_dist);

        // Backface cull: surface faces away from light → no contribution
        let n_dot_l = normal.dot(to_light);
        if n_dot_l <= 0.0 {
            continue;
        }

        // Soft shadow (skipped for deep bounces where budget says no shadows)
        let shadow = if budget.do_shadows {
            let shadow_origin = hit + normal * (SURF_DIST * 4.0);
            tracer.begin(SdfStage::Shadow);
            let s = soft_shadow(scene, shadow_origin, to_light, light_dist, time, budget.shadow_steps);
            tracer.end(SdfStage::Shadow);
            if s < 0.001 {
                continue;
            }
            s
        } else {
            1.0
        };

        tracer.begin(SdfStage::Shading);
        let intensity = light.intensity;

        // Diffuse (modulated by shadow factor)
        let diff = n_dot_l * intensity * shadow;
        r += base_color.r * light.color.r * diff;
        g += base_color.g * light.color.g * diff;
        b += base_color.b * light.color.b * diff;

        // Specular (Blinn-Phong, modulated by shadow factor)
        let half = (to_light - ray_dir).normalize();
        let spec = normal.dot(half).max(0.0).powf(spec_power) * intensity * shadow;
        r += light.color.r * spec * 0.5;
        g += light.color.g * spec * 0.5;
        b += light.color.b * spec * 0.5;
        tracer.end(SdfStage::Shading);
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
        let budget = RayBudget::for_bounce(0);
        let hit = ray_march(&scene, origin, dir, 0.0, &budget, None);
        assert!(hit.is_some(), "ray should hit sphere");
        let (p, _, _) = hit.unwrap();
        // Hit point Z should be near the sphere surface at z ≈ 1.0
        assert!((p.z - 1.0).abs() < 0.1, "hit z={} expected near 1.0", p.z);
    }
}
