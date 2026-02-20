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

#[cfg(feature = "text")]
use crate::scene::command::FontData;
#[cfg(feature = "text")]
use crate::scene::style::FillStyle;

use std::sync::Arc;

use super::materials::Material;
use super::math::Vec3;

#[cfg(feature = "sdf-text")]
use super::glyph::{self, TextSdfLayout};

use crate::math3d::Quaternion;

// ── Text Labels ─────────────────────────────────────────────────────

/// A billboard text label placed at a 3D position in an SDF scene.
///
/// After the ray marcher produces a pixmap, each label's world position is
/// projected through the camera to screen space, and the text is composited
/// on top. Labels always face the camera (billboard mode).
#[cfg(feature = "text")]
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SdfTextLabel {
    /// World-space anchor position.
    pub position: Vec3,
    /// The text string.
    pub text: String,
    /// Font size in pixels (at depth=1.0; scaled by 1/depth for perspective).
    pub font_size: f32,
    /// Text color.
    pub color: Color,
    /// Optional custom font data. Uses the default font if `None`.
    pub font_data: Option<FontData>,
    /// Optional outline (color, width).
    pub outline: Option<(Color, f32)>,
    /// Optional gradient fill style.
    pub fill_style: Option<FillStyle>,
    /// Whether to scale font size by inverse depth (default: true).
    pub perspective_scale: bool,
}

#[cfg(feature = "text")]
impl SdfTextLabel {
    /// Create a new text label at the given world position.
    pub fn new(position: Vec3, text: impl Into<String>) -> Self {
        Self {
            position,
            text: text.into(),
            font_size: 24.0,
            color: Color::WHITE,
            font_data: None,
            outline: None,
            fill_style: None,
            perspective_scale: true,
        }
    }

    /// Set the font size.
    pub fn font_size(mut self, size: f32) -> Self {
        self.font_size = size;
        self
    }

    /// Set the text color.
    pub fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Set custom font data.
    pub fn font(mut self, font_data: FontData) -> Self {
        self.font_data = Some(font_data);
        self
    }

    /// Set an outline.
    pub fn outline(mut self, color: Color, width: f32) -> Self {
        self.outline = Some((color, width));
        self
    }

    /// Set a gradient fill style.
    pub fn fill(mut self, fill_style: FillStyle) -> Self {
        self.fill_style = Some(fill_style);
        self
    }

    /// Enable or disable perspective scaling (default: true).
    pub fn perspective_scale(mut self, enabled: bool) -> Self {
        self.perspective_scale = enabled;
        self
    }
}

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
    /// Boolean subtraction: carve shape B from shape A.
    Subtract {
        /// Base shape (kept).
        a: std::boxed::Box<Self>,
        /// Carving shape (removed from A).
        b: std::boxed::Box<Self>,
        /// Offset of shape B relative to the object position.
        b_offset: Vec3,
    },
    /// Vertical capsule (line segment swept by radius).
    Capsule {
        /// Radius of the capsule.
        radius: f32,
        /// Half-height of the line segment.
        half_height: f32,
    },
    /// Axis-aligned box with rounded edges.
    RoundedBox {
        /// Half-extents along each axis.
        half_extents: Vec3,
        /// Corner rounding radius.
        radius: f32,
    },
    /// Cone along the Y axis.
    Cone {
        /// Base radius.
        radius: f32,
        /// Height (tip at y = height).
        height: f32,
    },
    /// Mandelbulb fractal.
    Mandelbulb {
        /// Fractal power (8.0 = classic bulb).
        power: f32,
        /// Iteration count (10–15 for good detail).
        iterations: u32,
    },
    /// Menger sponge fractal (unit cube with recursive cross holes).
    MengerSponge {
        /// Recursion depth (3–5 iterations).
        iterations: u32,
    },
    /// Gyroid triply periodic minimal surface, clipped to a bounding sphere.
    Gyroid {
        /// Spatial frequency scale.
        scale: f32,
        /// Surface thickness.
        thickness: f32,
        /// Bounding sphere radius (0 = unbounded).
        bound: f32,
    },
    /// 3D extruded text from TTF font outlines.
    ///
    /// The text is centered at the object's position and extruded along the
    /// Z axis by `depth`. Participates fully in the 3D scene: receives
    /// lighting, casts/receives shadows, reflects in mirrors, and supports
    /// all materials.
    #[cfg(feature = "sdf-text")]
    Text3D {
        /// Pre-computed SDF layout for the text string.
        layout: Arc<TextSdfLayout>,
        /// Extrusion depth along Z.
        depth: f32,
    },
}

#[cfg(feature = "sdf-text")]
impl SdfShape {
    /// Create a 3D extruded text shape from font data and a string.
    ///
    /// `font_size` controls the world-space height of the text.
    /// `depth` controls how far the text is extruded along Z.
    /// Uses a 64×64 SDF grid per glyph and zero letter spacing.
    pub fn text_3d(font_data: &[u8], text: &str, font_size: f32, depth: f32) -> Option<Self> {
        let layout = glyph::layout_text(font_data, text, font_size, 0.0, 64)?;
        Some(Self::Text3D { layout, depth })
    }

    /// Create a 3D extruded text shape with full control over layout parameters.
    ///
    /// `letter_spacing` adds extra space between glyphs (in world units).
    /// `grid_resolution` controls the SDF grid size per glyph (higher = sharper edges,
    /// default 64).
    pub fn text_3d_with_options(
        font_data: &[u8],
        text: &str,
        font_size: f32,
        depth: f32,
        letter_spacing: f32,
        grid_resolution: u32,
    ) -> Option<Self> {
        let layout = glyph::layout_text(font_data, text, font_size, letter_spacing, grid_resolution)?;
        Some(Self::Text3D { layout, depth })
    }
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
    /// Bounding sphere radius for early culling in `scene_sdf`.
    ///
    /// Set to `f32::INFINITY` for infinite primitives (planes) so
    /// they are never skipped.
    pub bounding_radius: f32,
    /// Y-axis rotation in radians (applied in local space before SDF eval).
    ///
    /// Cached as `(cos, sin)` for fast domain rotation.
    pub rotation_y: Option<(f32, f32)>,
    /// Full 3D rotation via quaternion (applied in local space before SDF eval).
    ///
    /// When set, takes precedence over `rotation_y`. Use `.orient(q)` or
    /// `.rotate(axis, angle)` to set.
    pub orientation: Option<Quaternion>,
}

impl SdfObject {
    /// Create a new object with the given shape and material at the origin.
    pub fn new(shape: SdfShape, material: Material) -> Self {
        Self {
            shape,
            material,
            position: Vec3::ZERO,
            bounding_radius: f32::INFINITY,
            rotation_y: None,
            orientation: None,
        }
    }

    /// Set the object's world-space position.
    pub fn at(mut self, pos: Vec3) -> Self {
        self.position = pos;
        self
    }

    /// Rotate the object around the Y axis by `angle` radians.
    pub fn rotate_y(mut self, angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        self.rotation_y = Some((c, s));
        self
    }

    /// Set the object's 3D orientation via a quaternion.
    ///
    /// Takes precedence over `rotation_y` when both are set.
    pub fn orient(mut self, q: Quaternion) -> Self {
        self.orientation = Some(q);
        self
    }

    /// Rotate the object around an arbitrary axis by `angle` radians.
    ///
    /// This is a convenience wrapper that builds a quaternion from the axis
    /// and angle, then sets it as the object's orientation.
    pub fn rotate(mut self, axis: Vec3, angle: f32) -> Self {
        self.orientation = Some(Quaternion::from_axis_angle(axis, angle));
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
    /// Cached: whether any object uses a `Glass` material (disables sky fast-path for bounce rays).
    pub has_glass: bool,
    /// Center of the bounding sphere around all finite (non-Plane) objects.
    pub scene_center: Vec3,
    /// Radius of the bounding sphere around all finite objects.
    pub scene_radius: f32,
    /// Exponential fog density (0.0 = no fog).
    pub fog_density: f32,
    /// Fog color (blends toward this at distance).
    pub fog_color: Color,
    /// Enable Reinhard tone mapping before gamma.
    pub tone_map: bool,
    /// Billboard text labels composited after ray marching.
    #[cfg(feature = "text")]
    pub text_labels: Vec<SdfTextLabel>,
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
            has_glass: false,
            scene_center: Vec3::ZERO,
            scene_radius: 0.0,
            fog_density: 0.0,
            fog_color: Color::from_rgba8(40, 50, 70, 255),
            tone_map: false,
            #[cfg(feature = "text")]
            text_labels: Vec::new(),
        }
    }

    /// Add an object to the scene.
    pub fn object(mut self, mut obj: SdfObject) -> Self {
        if matches!(obj.material, Material::Fire { .. }) {
            self.has_fire = true;
        }
        if matches!(obj.material, Material::Water { .. }) {
            self.has_water = true;
        }
        if matches!(obj.material, Material::Glass { .. }) {
            self.has_glass = true;
        }

        // Compute bounding sphere radius for finite (non-Plane) objects.
        // Planes keep `f32::INFINITY` so they are never culled.
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
            SdfShape::Subtract { .. } => Some(3.0), // conservative estimate
            SdfShape::Capsule {
                radius,
                half_height,
            } => Some(*radius + *half_height),
            SdfShape::RoundedBox {
                half_extents,
                radius,
            } => Some(half_extents.length() + *radius),
            SdfShape::Cone { radius, height } => Some(radius.hypot(*height)),
            SdfShape::Mandelbulb { .. } => Some(1.5), // bulb fits roughly in r=1.5
            SdfShape::MengerSponge { .. } => Some(1.8), // unit cube diagonal ≈ 1.73
            SdfShape::Gyroid { bound, .. } => if *bound > 0.0 { Some(*bound) } else { Some(5.0) },
            #[cfg(feature = "sdf-text")]
            SdfShape::Text3D { layout, depth } => {
                let half_w = layout.total_width * 0.5;
                let half_h = (layout.ascent + layout.descent) * 0.5;
                let half_d = depth * 0.5;
                Some((half_w * half_w + half_h * half_h + half_d * half_d).sqrt())
            }
        };

        if let Some(r) = obj_radius {
            obj.bounding_radius = r;
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
                    SdfShape::Subtract { .. } => 3.0,
                    SdfShape::Capsule {
                        radius,
                        half_height,
                    } => *radius + *half_height,
                    SdfShape::RoundedBox {
                        half_extents,
                        radius,
                    } => half_extents.length() + *radius,
                    SdfShape::Cone { radius, height } => radius.hypot(*height),
                    SdfShape::Mandelbulb { .. } => 1.5,
                    SdfShape::MengerSponge { .. } => 1.8,
                    SdfShape::Gyroid { bound, .. } => if *bound > 0.0 { *bound } else { 5.0 },
                    #[cfg(feature = "sdf-text")]
                    SdfShape::Text3D { layout, depth } => {
                        let half_w = layout.total_width * 0.5;
                        let half_h = (layout.ascent + layout.descent) * 0.5;
                        let half_d = depth * 0.5;
                        (half_w * half_w + half_h * half_h + half_d * half_d).sqrt()
                    }
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

    /// Add a billboard text label to the scene.
    ///
    /// Labels are composited onto the rendered pixmap after ray marching.
    /// They always face the camera and are perspective-scaled by default.
    #[cfg(feature = "text")]
    pub fn text_label(mut self, label: SdfTextLabel) -> Self {
        self.text_labels.push(label);
        self
    }
}

impl Default for SdfScene {
    fn default() -> Self {
        Self::new()
    }
}
