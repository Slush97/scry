// SPDX-License-Identifier: MIT OR Apache-2.0
//! Scene builder types for the SDF ray marcher.
//!
//! ```no_run
//! use scry_engine::sdf::*;
//! use scry_engine::scene::style::Color;
//!
//! let scene = SdfScene::new()
//!     .object(SdfObject::new(SdfShape::Sphere { radius: 1.0 }, Material::mirror(Color::WHITE, 0.8))
//!         .at(Vec3::new(0.0, 1.0, 0.0)))
//!     .object(SdfObject::new(SdfShape::Plane, Material::matte(Color::from_rgba8(180, 180, 180, 255))))
//!     .light(SdfLight::new(Vec3::new(5.0, 10.0, 5.0), Color::WHITE, 1.0))
//!     .camera(SdfCamera::new(Vec3::new(0.0, 3.0, 6.0), Vec3::ZERO, 45.0))
//!     .sky_color(Color::from_rgba8(40, 60, 100, 255));
//! ```

use crate::scene::style::Color;

use super::materials::Material;
use super::math::Vec3;

// ── Shapes ──────────────────────────────────────────────────────────

/// An SDF primitive or composite shape.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum SdfShape {
    /// Sphere centered at the object's position.
    Sphere {
        /// Radius.
        radius: f32,
    },
    /// Axis-aligned box.
    Box {
        /// Half-extents along each axis.
        half_extents: Vec3,
    },
    /// Infinite ground plane at y = 0 (before positioning).
    Plane,
    /// Torus in the XZ plane.
    Torus {
        /// Major radius (ring center).
        major: f32,
        /// Minor radius (tube).
        minor: f32,
    },
    /// Cylinder along the Y axis.
    Cylinder {
        /// Radius.
        radius: f32,
        /// Half-height.
        half_height: f32,
    },
    /// Smooth blend of two sub-shapes.
    SmoothBlend {
        /// First shape.
        a: std::boxed::Box<Self>,
        /// Second shape.
        b: std::boxed::Box<Self>,
        /// Offset of shape B relative to the object position.
        b_offset: Vec3,
        /// Blend radius (higher = smoother).
        k: f32,
    },
}

// ── Object ──────────────────────────────────────────────────────────

/// A shape + material + position in the scene.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SdfObject {
    /// The SDF shape.
    pub shape: SdfShape,
    /// Surface or volumetric material.
    pub material: Material,
    /// World-space position (the shape is centered here).
    pub position: Vec3,
}

impl SdfObject {
    /// Create a new object with the given shape and material at the origin.
    pub fn new(shape: SdfShape, material: Material) -> Self {
        Self {
            shape,
            material,
            position: Vec3::ZERO,
        }
    }

    /// Set the object's world-space position.
    pub fn at(mut self, pos: Vec3) -> Self {
        self.position = pos;
        self
    }
}

// ── Light ───────────────────────────────────────────────────────────

/// A point light source.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SdfLight {
    /// World position.
    pub position: Vec3,
    /// Light color.
    pub color: Color,
    /// Intensity multiplier.
    pub intensity: f32,
}

impl SdfLight {
    /// Create a new point light.
    pub fn new(position: Vec3, color: Color, intensity: f32) -> Self {
        Self {
            position,
            color,
            intensity,
        }
    }
}

// ── Camera ──────────────────────────────────────────────────────────

/// A perspective camera defined by position, look-at target, and field of view.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SdfCamera {
    /// Eye position.
    pub eye: Vec3,
    /// Look-at target.
    pub target: Vec3,
    /// Vertical field of view in degrees.
    pub fov: f32,
}

impl SdfCamera {
    /// Create a new camera.
    pub fn new(eye: Vec3, target: Vec3, fov: f32) -> Self {
        Self { eye, target, fov }
    }
}

impl Default for SdfCamera {
    fn default() -> Self {
        Self {
            eye: Vec3::new(0.0, 3.0, 6.0),
            target: Vec3::ZERO,
            fov: 45.0,
        }
    }
}

// ── Scene ───────────────────────────────────────────────────────────

/// Top-level scene description for the SDF ray marcher.
///
/// Build with a fluent API, then pass to
/// [`SdfRenderer::render_to_pixmap`](super::SdfRenderer::render_to_pixmap).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SdfScene {
    /// Objects in the scene.
    pub objects: Vec<SdfObject>,
    /// Point lights.
    pub lights: Vec<SdfLight>,
    /// Camera.
    pub camera: SdfCamera,
    /// Background / sky color for rays that miss all geometry.
    pub sky_color: Color,
    /// Maximum reflection bounces (default 2).
    pub max_bounces: u32,
    /// Ambient light contribution (0–1, default 0.05).
    pub ambient: f32,
    /// Cached: whether any object uses a `Fire` material (avoids per-ray scan).
    pub has_fire: bool,
    /// Cached: whether any object uses a `Water` material (disables over-relaxation + sky fast-path).
    pub has_water: bool,
    /// Center of the bounding sphere around all finite (non-Plane) objects.
    pub scene_center: Vec3,
    /// Radius of the bounding sphere around all finite objects.
    pub scene_radius: f32,
}

impl SdfScene {
    /// Create an empty scene with default camera and sky.
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            lights: Vec::new(),
            camera: SdfCamera::default(),
            sky_color: Color::from_rgba8(40, 50, 70, 255),
            max_bounces: 2,
            ambient: 0.05,
            has_fire: false,
            has_water: false,
            scene_center: Vec3::ZERO,
            scene_radius: 0.0,
        }
    }

    /// Add an object to the scene.
    pub fn object(mut self, obj: SdfObject) -> Self {
        if matches!(obj.material, Material::Fire { .. }) {
            self.has_fire = true;
        }
        if matches!(obj.material, Material::Water { .. }) {
            self.has_water = true;
        }

        // Update bounding sphere for finite (non-Plane) objects
        let obj_radius = match &obj.shape {
            SdfShape::Plane => None,
            SdfShape::Sphere { radius } => Some(*radius),
            SdfShape::Box { half_extents } => Some(half_extents.length()),
            SdfShape::Torus { major, minor } => Some(*major + *minor),
            SdfShape::Cylinder {
                radius,
                half_height,
            } => Some(radius.hypot(*half_height)),
            SdfShape::SmoothBlend { .. } => Some(3.0), // conservative estimate
        };

        if let Some(r) = obj_radius {
            self.objects.push(obj);
            self.recompute_bounds(r);
        } else {
            self.objects.push(obj);
        }

        self
    }

    /// Recompute the bounding sphere to encompass all finite objects.
    fn recompute_bounds(&mut self, new_obj_radius: f32) {
        // Simple approach: average position of finite objects as center,
        // max distance + object radius as scene radius.
        let mut center = Vec3::ZERO;
        let mut count = 0u32;
        for o in &self.objects {
            if !matches!(o.shape, SdfShape::Plane) {
                center = center + o.position;
                count += 1;
            }
        }
        if count > 0 {
            self.scene_center = center * (1.0 / count as f32);

            let mut max_r = 0.0_f32;
            for o in &self.objects {
                let r = match &o.shape {
                    SdfShape::Plane => continue,
                    SdfShape::Sphere { radius } => *radius,
                    SdfShape::Box { half_extents } => half_extents.length(),
                    SdfShape::Torus { major, minor } => *major + *minor,
                    SdfShape::Cylinder {
                        radius,
                        half_height,
                    } => radius.hypot(*half_height),
                    SdfShape::SmoothBlend { .. } => 3.0,
                };
                let dist = (o.position - self.scene_center).length() + r;
                max_r = max_r.max(dist);
            }
            self.scene_radius = max_r;
        } else {
            // Only planes, no finite objects
            self.scene_center = Vec3::ZERO;
            self.scene_radius = 0.0;
        }
        let _ = new_obj_radius; // used implicitly via objects list
    }

    /// Add a point light.
    pub fn light(mut self, light: SdfLight) -> Self {
        self.lights.push(light);
        self
    }

    /// Set the camera.
    pub fn camera(mut self, cam: SdfCamera) -> Self {
        self.camera = cam;
        self
    }

    /// Set the sky / background color.
    pub fn sky_color(mut self, color: Color) -> Self {
        self.sky_color = color;
        self
    }

    /// Set maximum reflection bounces.
    pub fn max_bounces(mut self, n: u32) -> Self {
        self.max_bounces = n;
        self
    }

    /// Set ambient light level.
    pub fn ambient(mut self, a: f32) -> Self {
        self.ambient = a;
        self
    }
}

impl Default for SdfScene {
    fn default() -> Self {
        Self::new()
    }
}
