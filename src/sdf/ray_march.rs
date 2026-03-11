// SPDX-License-Identifier: MIT OR Apache-2.0
//! Ray marching and SDF evaluation for the CPU renderer.
//!
//! Contains the sphere-tracing loop, scene SDF evaluation with bounding-sphere
//! culling, per-object SDF evaluation (including water displacement and
//! rotation), and the raw shape SDF dispatcher.

use super::materials::Material;
use super::math::Vec3;
use super::noise;
use super::primitives;
use super::scene::{SdfObject, SdfScene, SdfShape};

// ── Constants ───────────────────────────────────────────────────────

pub(super) const MAX_DIST: f32 = 50.0;
pub(super) const SURF_DIST: f32 = 0.002;

/// Over-relaxation factor for enhanced sphere tracing (Keinert et al. 2014).
const OMEGA: f32 = 1.6;

/// Distance threshold: below this, fall back to conservative stepping.
const RELAX_DIST: f32 = 0.02;

/// Per-bounce quality budget that degrades gracefully for deeper bounces.
pub(super) struct RayBudget {
    pub(super) march_steps: u32,
    pub(super) shadow_steps: u32,
    pub(super) do_shadows: bool,
}

impl RayBudget {
    pub(super) fn for_bounce(bounce: u32) -> Self {
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

// ── Scene SDF evaluation ────────────────────────────────────────────

/// Evaluate the entire scene SDF at `point`, returning `(distance, object_index)`.
///
/// Uses per-object bounding spheres for early culling: if the distance from
/// `point` to the object's center minus the bounding radius already exceeds
/// `min_dist`, the full SDF evaluation is skipped. Planes have
/// `bounding_radius = INFINITY` so they are never culled.
#[inline]
pub(super) fn scene_sdf(scene: &SdfScene, point: Vec3, time: f32) -> (f32, usize) {
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
pub(super) fn scene_sdf_exclude(
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
pub(super) fn object_sdf(obj: &SdfObject, point: Vec3, time: f32) -> f32 {
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
pub(super) fn shape_sdf(shape: &SdfShape, p: Vec3) -> f32 {
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
        SdfShape::MengerSponge { iterations } => primitives::sd_menger_sponge(p, *iterations),
        SdfShape::Gyroid {
            scale,
            thickness,
            bound,
        } => primitives::sd_gyroid(p, *scale, *thickness, *bound),
        SdfShape::Morph { a, b, t } => {
            let da = shape_sdf(a, p);
            let db = shape_sdf(b, p);
            da + (db - da) * *t
        }
        #[cfg(feature = "sdf-text")]
        SdfShape::Text3D { layout, depth } => super::glyph::sd_text3d(layout, p, *depth),
    }
}

/// Animated water surface displacement (superposed sine waves).
pub(super) fn water_displacement(x: f32, z: f32, time: f32, amplitude: f32, frequency: f32) -> f32 {
    let w1 = (x * frequency + time * 2.0).sin() * amplitude;
    let w2 = (z * frequency * 0.7 + time * 1.5).sin() * amplitude * 0.6;
    let w3 = ((x + z) * frequency * 1.3 + time * 2.5).sin() * amplitude * 0.3;
    let n = noise::fbm2d(x * 0.5, z * 0.5 + time * 0.3, 2) * amplitude * 0.4;
    w1 + w2 + w3 + n
}

// ── Ray marching ────────────────────────────────────────────────────

/// Sphere-trace from `origin` along `dir`. Returns `Some((hit_point, obj_index, total_dist))`.
///
/// Uses enhanced sphere tracing (over-relaxation ω=1.6) for ~30-40% fewer
/// steps on average. Step budget comes from `RayBudget` based on bounce depth.
/// Distance-adaptive relaxation: uses ω=1.6 far from surfaces, ω=1.0 near
/// them. Includes rewind-on-overshoot fallback for correctness.
#[inline]
pub(super) fn ray_march(
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
            // Bisection refinement: binary search between prev and current t
            // for sub-grid-cell precision (4 iterations ≈ 1/16 step accuracy)
            let mut lo = t - prev_step;
            let mut hi = t;
            let mut mid_idx = idx;
            for _ in 0..8 {
                let mid = (lo + hi) * 0.5;
                let mp = origin + dir * mid;
                let (md, mi) = scene_sdf_exclude(scene, mp, time, exclude_idx);
                mid_idx = mi;
                if md < SURF_DIST {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            let final_t = (lo + hi) * 0.5;
            let final_p = origin + dir * final_t;
            return Some((final_p, mid_idx, final_t));
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
