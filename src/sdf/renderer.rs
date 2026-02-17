// SPDX-License-Identifier: MIT OR Apache-2.0
//! Core SDF ray marching renderer.
//!
//! Renders an [`SdfScene`] to a `tiny_skia::Pixmap` or [`ImageData`] by sphere-
//! tracing each pixel ray through the signed distance field. Supports Phong
//! shading, mirror reflections, animated water (Fresnel + refraction), and
//! volumetric fire (front-to-back FBM compositing).

use tiny_skia::Pixmap;

use crate::scene::command::ImageData;
use crate::scene::style::Color;
use crate::PixelCanvasError;

use super::materials::{self, Material};
use super::math::{self, Vec3};
use super::noise;
use super::primitives;
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
        for y in 0..height {
            for x in 0..width {
                // Normalized device coordinates [-1, 1]
                let ndc_x = (2.0 * (x as f32 + 0.5) / width as f32 - 1.0) * aspect * fov_scale;
                let ndc_y = (1.0 - 2.0 * (y as f32 + 0.5) / height as f32) * fov_scale;

                let dir = (cam_right * ndc_x + cam_up * ndc_y + cam_fwd).normalize();
                let color = shade_ray(scene, scene.camera.eye, dir, time, 0);

                let idx = (y * width + x) as usize;
                pixels[idx] = tiny_skia::PremultipliedColorU8::from_rgba(
                    (color.r.clamp(0.0, 1.0) * 255.0) as u8,
                    (color.g.clamp(0.0, 1.0) * 255.0) as u8,
                    (color.b.clamp(0.0, 1.0) * 255.0) as u8,
                    255,
                )
                .unwrap();
            }
        }

        Ok(pixmap)
    }

    /// Render the scene to an [`ImageData`] for compositing onto a `PixelCanvas`.
    pub fn render_to_image(scene: &SdfScene, width: u32, height: u32, time: f32) -> ImageData {
        let (cam_right, cam_up, cam_fwd) =
            math::look_at(scene.camera.eye, scene.camera.target, Vec3::UP);
        let fov_scale = (scene.camera.fov.to_radians() * 0.5).tan();
        let aspect = width as f32 / height as f32;

        let mut data = vec![0u8; (width as usize) * (height as usize) * 4];

        for y in 0..height {
            for x in 0..width {
                let ndc_x = (2.0 * (x as f32 + 0.5) / width as f32 - 1.0) * aspect * fov_scale;
                let ndc_y = (1.0 - 2.0 * (y as f32 + 0.5) / height as f32) * fov_scale;

                let dir = (cam_right * ndc_x + cam_up * ndc_y + cam_fwd).normalize();
                let color = shade_ray(scene, scene.camera.eye, dir, time, 0);

                let idx = ((y * width + x) * 4) as usize;
                data[idx] = (color.r.clamp(0.0, 1.0) * 255.0) as u8;
                data[idx + 1] = (color.g.clamp(0.0, 1.0) * 255.0) as u8;
                data[idx + 2] = (color.b.clamp(0.0, 1.0) * 255.0) as u8;
                data[idx + 3] = 255;
            }
        }

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

/// Estimate the surface normal via tetrahedron technique (4 SDF evals instead of 6).
fn estimate_normal(scene: &SdfScene, point: Vec3, time: f32) -> Vec3 {
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

/// Check if the scene has any fire objects (cached per-scene).
fn has_fire_objects(scene: &SdfScene) -> bool {
    scene
        .objects
        .iter()
        .any(|o| matches!(o.material, Material::Fire { .. }))
}

/// Top-level per-ray shading: march, hit → shade surface, miss → sky.
fn shade_ray(scene: &SdfScene, origin: Vec3, dir: Vec3, time: f32, bounce: u32) -> Color {
    // Only check fire on primary rays and only if fire objects exist
    if bounce == 0 && has_fire_objects(scene) {
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
    let normal = estimate_normal(scene, hit, time);

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
