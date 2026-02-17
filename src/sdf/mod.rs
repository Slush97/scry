// SPDX-License-Identifier: MIT OR Apache-2.0
//! SDF ray marching renderer.
//!
//! Renders 3D scenes described as signed distance fields via sphere tracing.
//! Supports Phong shading, mirror reflections, animated water with Fresnel
//! effects, and volumetric fire. Output is a `Pixmap` or `ImageData` that
//! feeds into the existing transport or `PixelCanvas` compositing pipeline.
//!
//! # Example
//!
//! ```
//! use scry_engine::sdf::*;
//! use scry_engine::scene::style::Color;
//!
//! let scene = SdfScene::new()
//!     .object(SdfObject::new(SdfShape::Sphere { radius: 1.0 },
//!                            Material::mirror(Color::WHITE, 0.8))
//!         .at(Vec3::new(0.0, 1.0, 0.0)))
//!     .object(SdfObject::new(SdfShape::Plane,
//!                            Material::matte(Color::from_rgba8(180, 180, 180, 255))))
//!     .light(SdfLight::new(Vec3::new(5.0, 10.0, 5.0), Color::WHITE, 1.0))
//!     .camera(SdfCamera::new(Vec3::new(0.0, 3.0, 6.0), Vec3::ZERO, 45.0));
//!
//! let pixmap = SdfRenderer::render_to_pixmap(&scene, 200, 150, 0.0).unwrap();
//! assert_eq!(pixmap.width(), 200);
//! ```

pub mod materials;
pub mod math;
pub mod noise;
pub mod primitives;
pub mod renderer;
pub mod scene;

pub use materials::Material;
pub use math::Vec3;
pub use renderer::SdfRenderer;
pub use scene::{SdfCamera, SdfLight, SdfObject, SdfScene, SdfShape};
