// SPDX-License-Identifier: MIT OR Apache-2.0
//! Per-ray and per-surface shading dispatch for the CPU SDF renderer.
//!
//! Contains the tracer abstraction (`SdfTracer`, `NoOpTracer`,
//! `ProfilingTracer`), the top-level `shade_ray` entry points, and
//! the per-material `shade_surface_traced` dispatcher.

use crate::scene::style::Color;

use super::lighting::{
    apply_fog, blend_over, estimate_normal, lerp_color, march_fire_volume, phong_no_shadows,
    phong_specular, phong_traced, volumetric_light,
};
use super::materials::{self, Material};
use super::math::Vec3;
use super::profiler::{RowProfile, SdfStage};
use super::ray_march::{ray_march, scene_sdf, RayBudget, MAX_DIST, SURF_DIST};
use super::scene::{SdfScene, SdfShape};

// ── Tracing infrastructure ──────────────────────────────────────────
//
// Instead of duplicating every shading function with a `_profiled`
// variant, we parameterise by a zero-cost `SdfTracer` trait. The
// `NoOpTracer` implementation compiles to nothing; `ProfilingTracer`
// records per-stage timings into a `RowProfile`.

/// Compile-time hooks for optional per-stage timing.
pub(super) trait SdfTracer {
    /// Called before a rendering stage begins.
    fn begin(&mut self, stage: SdfStage);
    /// Called after a rendering stage ends (records elapsed time since `begin`).
    fn end(&mut self, stage: SdfStage);
}

/// Zero-cost tracer — all calls are eliminated by the compiler.
pub(super) struct NoOpTracer;

impl SdfTracer for NoOpTracer {
    #[inline(always)]
    fn begin(&mut self, _stage: SdfStage) {}
    #[inline(always)]
    fn end(&mut self, _stage: SdfStage) {}
}

/// Profiling tracer that records per-stage wall-clock time.
pub(super) struct ProfilingTracer<'a> {
    prof: &'a mut RowProfile,
    starts: [Option<std::time::Instant>; 6],
}

impl<'a> ProfilingTracer<'a> {
    pub(super) fn new(prof: &'a mut RowProfile) -> Self {
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

// ── Unified shading entry points ────────────────────────────────────

/// Top-level per-ray shading: march, hit → shade surface, miss → sky.
pub(super) fn shade_ray(
    scene: &SdfScene,
    origin: Vec3,
    dir: Vec3,
    time: f32,
    bounce: u32,
    exclude_idx: Option<usize>,
) -> Color {
    shade_ray_traced(
        scene,
        origin,
        dir,
        time,
        bounce,
        exclude_idx,
        &mut NoOpTracer,
    )
}

/// Profiled variant of [`shade_ray`].
pub(super) fn shade_ray_profiled(
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

/// Generic `shade_ray` parameterised by tracer.
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

    if let Some((hit_pt, idx, dist)) = hit {
        let mut color =
            shade_surface_traced(scene, hit_pt, dir, idx, time, bounce, &budget, tracer);
        // Apply distance fog (only on primary rays)
        if bounce == 0 && scene.fog_density > 0.0 {
            color = apply_fog(color, scene.fog_color, scene.fog_density, dist);
        }
        // Volumetric god rays: accumulate in-scatter along the camera ray
        if bounce == 0 && scene.god_rays {
            let god_ray_color = volumetric_light(scene, origin, dir, dist, time);
            color.r = (color.r + god_ray_color.r).min(1.0);
            color.g = (color.g + god_ray_color.g).min(1.0);
            color.b = (color.b + god_ray_color.b).min(1.0);
        }
        color
    } else {
        let mut color = scene.sky_color;
        // God rays apply even to sky rays (light shafts visible against sky)
        if bounce == 0 && scene.god_rays {
            let god_ray_color = volumetric_light(scene, origin, dir, MAX_DIST, time);
            color.r = (color.r + god_ray_color.r).min(1.0);
            color.g = (color.g + god_ray_color.g).min(1.0);
            color.b = (color.b + god_ray_color.b).min(1.0);
        }
        color
    }
}

/// Shade a surface hit point with Phong lighting, reflections, and water effects.
#[allow(clippy::too_many_arguments)]
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
                phong_traced(
                    scene, hit, normal, ray_dir, *color, *specular, time, budget, tracer,
                )
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
                    let mut rc = shade_ray_traced(
                        scene,
                        refr_origin,
                        refr_dir,
                        time,
                        bounce + 1,
                        None,
                        tracer,
                    );
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
                phong_traced(
                    scene, hit, normal, ray_dir, base_color, *specular, time, budget, tracer,
                )
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
                            let rc =
                                shade_ray(scene, refr_origin, refr_dir, time, bounce + 1, exclude);
                            // Return luminance-ish single channel (we pick per-channel below)
                            rc.r * 0.33 + rc.g * 0.34 + rc.b * 0.33
                        } else {
                            // Total internal reflection fallback
                            let rc =
                                shade_ray(scene, refl_origin, refl_dir, time, bounce + 1, None);
                            rc.r * 0.33 + rc.g * 0.34 + rc.b * 0.33
                        }
                    };

                    // Refract each channel separately for prismatic effect
                    let mut cr = Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    };
                    if let Some(dir_r) = ray_dir.refract(normal, ior_r) {
                        let c = shade_ray(
                            scene,
                            hit - normal * (SURF_DIST * 2.0),
                            dir_r,
                            time,
                            bounce + 1,
                            exclude,
                        );
                        cr.r = c.r * tint.r;
                    } else {
                        cr.r = shade_channel(ior_r) * tint.r;
                    }
                    if let Some(dir_g) = ray_dir.refract(normal, ior_g) {
                        let c = shade_ray(
                            scene,
                            hit - normal * (SURF_DIST * 2.0),
                            dir_g,
                            time,
                            bounce + 1,
                            exclude,
                        );
                        cr.g = c.g * tint.g;
                    } else {
                        cr.g = shade_channel(ior_g) * tint.g;
                    }
                    if let Some(dir_b) = ray_dir.refract(normal, ior_b) {
                        let c = shade_ray(
                            scene,
                            hit - normal * (SURF_DIST * 2.0),
                            dir_b,
                            time,
                            bounce + 1,
                            exclude,
                        );
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
                        let mut rc = shade_ray_traced(
                            scene,
                            refr_origin,
                            refr_dir,
                            time,
                            bounce + 1,
                            Some(obj_idx),
                            tracer,
                        );
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
            // Compute local-space position for color mapping
            let local = hit - obj.position;

            // Text3D: smooth x-position gradient; other shapes: angular mapping
            #[cfg(feature = "sdf-text")]
            let hue = if let SdfShape::Text3D { layout, .. } = &obj.shape {
                let half_w = layout.total_width * 0.5;
                (local.x + half_w) / layout.total_width.max(0.001)
                    + hue_offset / std::f32::consts::TAU
            } else {
                let angle = local.z.atan2(local.x);
                angle / std::f32::consts::TAU + 0.5 + hue_offset / std::f32::consts::TAU
            };
            #[cfg(not(feature = "sdf-text"))]
            let hue = {
                let angle = local.z.atan2(local.x);
                angle / std::f32::consts::TAU + 0.5 + hue_offset / std::f32::consts::TAU
            };
            let base_color = materials::hsl_to_color(hue, *saturation, *lightness);

            if bounce > 0 {
                tracer.begin(SdfStage::Shading);
                let r = phong_no_shadows(scene, hit, normal, ray_dir, base_color, *specular);
                tracer.end(SdfStage::Shading);
                r
            } else {
                phong_traced(
                    scene, hit, normal, ray_dir, base_color, *specular, time, budget, tracer,
                )
            }
        }
        Material::Subsurface {
            color,
            scatter_color,
            thickness,
            specular,
        } => {
            // Standard Phong for front-lit surface shading
            let front_shaded = if bounce > 0 {
                tracer.begin(SdfStage::Shading);
                let r = phong_no_shadows(scene, hit, normal, ray_dir, *color, *specular);
                tracer.end(SdfStage::Shading);
                r
            } else {
                phong_traced(
                    scene, hit, normal, ray_dir, *color, *specular, time, budget, tracer,
                )
            };

            // Subsurface scattering: evaluate SDF thickness for back-illumination.
            // For each light, march a small distance into the surface toward the light
            // and measure how thick the object is. Thin areas glow with scatter_color.
            tracer.begin(SdfStage::Shading);
            let mut sss_r = 0.0_f32;
            let mut sss_g = 0.0_f32;
            let mut sss_b = 0.0_f32;
            for light in &scene.lights {
                let to_light = (light.position - hit).normalize();
                // Sample SDF at a point offset into the surface toward the light
                let sample_pt = hit + to_light * *thickness;
                let (d, _) = scene_sdf(scene, sample_pt, time);
                // Exponential decay: thin surfaces let more light through
                let sss_factor = (-d.abs() * 3.0).exp();
                // Back-illumination: contribution from light shining "through"
                let n_dot_l_inv = (-normal).dot(to_light).max(0.0);
                let contribution = sss_factor * n_dot_l_inv * light.intensity;
                sss_r += scatter_color.r * light.color.r * contribution;
                sss_g += scatter_color.g * light.color.g * contribution;
                sss_b += scatter_color.b * light.color.b * contribution;
            }
            tracer.end(SdfStage::Shading);

            Color {
                r: (front_shaded.r + sss_r).min(1.0),
                g: (front_shaded.g + sss_g).min(1.0),
                b: (front_shaded.b + sss_b).min(1.0),
                a: 1.0,
            }
        }
    }
}
