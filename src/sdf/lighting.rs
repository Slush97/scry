// SPDX-License-Identifier: MIT OR Apache-2.0
//! Lighting, shadows, normals, and post-processing for the CPU SDF renderer.
//!
//! Contains Phong shading, soft shadows, ambient occlusion, surface normal
//! estimation (including analytical water and text3d normals), fog, tone
//! mapping, gamma correction, volumetric light scattering, and fire volume
//! ray marching.

use crate::scene::style::Color;

use super::materials::{self, Material};
use super::math::Vec3;
use super::noise;
use super::primitives;
use super::profiler::SdfStage;
use super::ray_march::{
    scene_sdf, water_displacement, MAX_DIST, RayBudget, SURF_DIST,
};
use super::scene::{SdfObject, SdfScene, SdfShape};

// ── Constants ───────────────────────────────────────────────────────

pub(super) const NORMAL_EPS: f32 = 0.005;

/// Soft shadow penumbra sharpness.
const SHADOW_K: f32 = 16.0;

// ── Gamma correction / tone mapping ─────────────────────────────────

/// Encode a linear-space color channel to sRGB, returning a u8.
///
/// Uses the standard piecewise sRGB OETF (via `style::linear_to_srgb`)
/// for consistency with the rest of the engine. This avoids the ±4/255
/// per-channel error of the `powf(1/2.2)` approximation.
#[inline]
pub(super) fn gamma_encode(linear: f32) -> u8 {
    (crate::scene::style::linear_to_srgb(linear.clamp(0.0, 1.0)) * 255.0) as u8
}

/// Reinhard tone mapping: maps HDR [0, ∞) to LDR [0, 1).
#[inline]
pub(super) fn tone_map_reinhard(c: f32) -> f32 {
    c / (1.0 + c)
}

/// Apply distance fog: blends `color` toward `fog_color` based on distance.
#[inline]
pub(super) fn apply_fog(color: Color, fog_color: Color, fog_density: f32, distance: f32) -> Color {
    let fog_factor = (-fog_density * distance).exp();
    lerp_color(fog_color, color, fog_factor)
}

// ── Color blending ──────────────────────────────────────────────────

/// Linearly interpolate between two colors.
#[inline]
pub(super) fn lerp_color(a: Color, b: Color, t: f32) -> Color {
    Color {
        r: a.r + (b.r - a.r) * t,
        g: a.g + (b.g - a.g) * t,
        b: a.b + (b.b - a.b) * t,
        a: a.a + (b.a - a.a) * t,
    }
}

/// Composite `foreground` over `background` (premultiplied-style).
#[inline]
pub(super) fn blend_over(fg: Color, bg: Color) -> Color {
    let inv_a = 1.0 - fg.a;
    Color {
        r: fg.r + bg.r * inv_a,
        g: fg.g + bg.g * inv_a,
        b: fg.b + bg.b * inv_a,
        a: fg.a + bg.a * inv_a,
    }
}

// ── Surface normal estimation ───────────────────────────────────────

/// Estimate the surface normal via the tetrahedron technique.
///
/// For water planes we use an optimised displacement-gradient path
/// (4 cheap displacement evals instead of 4 full `scene_sdf` evals).
/// Everything else uses the numerical tetrahedron technique on the
/// *combined* scene SDF so that normals are correct at inter-object
/// boundaries (e.g. where a sphere meets the ground plane).
pub(super) fn estimate_normal(scene: &SdfScene, point: Vec3, obj_idx: usize, time: f32) -> Vec3 {
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

    // Text3D: use analytical normals from bilinear-interpolated SDF gradient.
    // This completely avoids finite-difference artifacts on the discrete grid.
    #[cfg(feature = "sdf-text")]
    if let SdfShape::Text3D { layout, depth } = &obj.shape {
        // Transform hit point to object-local space
        let mut local = primitives::translate(point, obj.position);
        if let Some(q) = obj.orientation {
            local = q.conjugate().rotate_vec3(local);
        } else if let Some((cos_y, sin_y)) = obj.rotation_y {
            let rx = local.x * cos_y - local.z * sin_y;
            let rz = local.x * sin_y + local.z * cos_y;
            local = Vec3::new(rx, local.y, rz);
        }

        let local_n = super::glyph::estimate_text3d_normal(layout, local, *depth);

        // Rotate local normal back to world space
        if let Some(q) = obj.orientation {
            return q.rotate_vec3(local_n).normalize();
        } else if let Some((cos_y, sin_y)) = obj.rotation_y {
            // Forward rotation (inverse of inverse)
            let rx = local_n.x * cos_y + local_n.z * sin_y;
            let rz = -local_n.x * sin_y + local_n.z * cos_y;
            return Vec3::new(rx, local_n.y, rz).normalize();
        }
        return local_n;
    }

    // Numerical normal via the combined scene SDF — correct everywhere.
    estimate_normal_numerical(scene, point, time)
}

/// Tetrahedron technique fallback for normals (4 SDF evals).
#[inline]
pub(super) fn estimate_normal_numerical(scene: &SdfScene, point: Vec3, time: f32) -> Vec3 {
    estimate_normal_numerical_eps(scene, point, time, NORMAL_EPS)
}

/// Tetrahedron technique with custom epsilon.
#[inline]
fn estimate_normal_numerical_eps(scene: &SdfScene, point: Vec3, time: f32, e: f32) -> Vec3 {
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

// ── Shadow computation ──────────────────────────────────────────────

/// Soft shadow factor using IQ's penumbra technique.
/// Returns 1.0 (fully lit) to 0.0 (fully shadowed).
/// Step count comes from the `RayBudget`.
#[inline]
pub(super) fn soft_shadow(
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
pub(super) fn ambient_occlusion(scene: &SdfScene, hit: Vec3, normal: Vec3, time: f32) -> f32 {
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

// ── Phong lighting ──────────────────────────────────────────────────

/// Phong lighting without shadow marches (used for reflection bounces).
#[inline]
pub(super) fn phong_no_shadows(
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
pub(super) fn phong_specular(
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

/// Phong lighting: ambient + diffuse + specular from all lights.
#[allow(clippy::too_many_arguments)]
pub(super) fn phong_traced<T: super::shading::SdfTracer>(
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

// ── Fire volume ─────────────────────────────────────────────────────

/// Front-to-back ray marching through a fire volume.
pub(super) fn march_fire_volume(origin: Vec3, dir: Vec3, obj: &SdfObject, time: f32) -> Color {
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

// ── Volumetric lighting (god rays) ──────────────────────────────────

/// Volumetric light scattering (god rays): march along the camera ray,
/// accumulate light contribution at each sample point that isn't in shadow.
pub(super) fn volumetric_light(
    scene: &SdfScene,
    origin: Vec3,
    dir: Vec3,
    max_t: f32,
    time: f32,
) -> Color {
    let samples = scene.god_ray_samples;
    let density = scene.god_ray_density;
    let step = max_t.min(MAX_DIST) / samples as f32;

    let mut accum_r = 0.0_f32;
    let mut accum_g = 0.0_f32;
    let mut accum_b = 0.0_f32;

    for i in 0..samples {
        let t = step * (i as f32 + 0.5);
        let sample_pt = origin + dir * t;

        // Check if this point is lit by any light
        for light in &scene.lights {
            let to_light = light.position - sample_pt;
            let light_dist = to_light.length();
            let light_dir = to_light * (1.0 / light_dist);

            // Cheap shadow test: just check if the SDF is positive at a few points
            let (d, _) = scene_sdf(scene, sample_pt, time);
            if d < 0.0 {
                // Inside geometry — no contribution
                continue;
            }

            // Quick 8-step shadow check
            let shadow = soft_shadow(scene, sample_pt, light_dir, light_dist, time, 8);
            if shadow > 0.01 {
                let contribution = density * step * shadow * light.intensity * 0.02;
                accum_r += light.color.r * contribution;
                accum_g += light.color.g * contribution;
                accum_b += light.color.b * contribution;
            }
        }
    }

    Color {
        r: accum_r.min(1.0),
        g: accum_g.min(1.0),
        b: accum_b.min(1.0),
        a: 0.0,
    }
}
