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
use crate::PixelCanvasError;

use super::lighting::{gamma_encode, tone_map_reinhard};
use super::math::{self, Vec3};
use super::profiler::{RowProfile, SdfProfile};
use super::scene::SdfScene;
use super::shading::{shade_ray, shade_ray_profiled};

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
                        (
                            tone_map_reinhard(color.r),
                            tone_map_reinhard(color.g),
                            tone_map_reinhard(color.b),
                        )
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
                    *pixel = tiny_skia::PremultipliedColorU8::from_rgba(r8, g8, b8, a).unwrap();
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
                            (
                                tone_map_reinhard(color.r),
                                tone_map_reinhard(color.g),
                                tone_map_reinhard(color.b),
                            )
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
                        tiny_skia::PremultipliedColorU8::from_rgba(r8, g8, b8, a).unwrap()
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
                        (
                            tone_map_reinhard(color.r),
                            tone_map_reinhard(color.g),
                            tone_map_reinhard(color.b),
                        )
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
    fn composite_text_labels(pixmap: &mut Pixmap, scene: &SdfScene, width: u32, height: u32) {
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

#[cfg(test)]
mod tests {
    use super::super::materials::Material;
    use super::super::ray_march::{ray_march, scene_sdf, RayBudget};
    use super::super::scene::{SdfCamera, SdfLight, SdfObject, SdfShape};
    use super::*;
    use crate::scene::style::Color;

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
