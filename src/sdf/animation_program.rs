// SPDX-License-Identifier: MIT OR Apache-2.0
//! Preset-based animation programs for terminal-side SDF rendering.
//!
//! An [`AnimationProgram`] is a serializable identifier for a named SDF
//! animation preset. The terminal deserializes it from IPC and evaluates
//! `build_scene(time)` each frame to produce an [`SdfScene`] autonomously,
//! without needing the CLI to stay alive.
//!
//! Each variant mirrors a `PlayPreset` from `scry-cli` but is decoupled
//! from the CLI crate so it can live in `scry-engine`.

use serde::{Deserialize, Serialize};

use crate::math3d::Quaternion;
use crate::scene::style::Color;

use super::materials::Material;
use super::math::Vec3;
use super::scene::{SdfCamera, SdfLight, SdfObject, SdfScene, SdfShape};

/// A serializable animation program identifier.
///
/// Sent over IPC from the CLI to the terminal. The terminal evaluates
/// `build_scene(t)` each frame to produce the SDF scene for that instant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnimationProgram {
    /// Spinning 3D cube with rainbow gradient.
    Cube,
    /// Hypnotic toroidal vortex (glass torus + fire core).
    Vortex,
    /// Breathing organic orb (smooth-blended spheres).
    Pulse,
    /// Cosmic orbital system (mirror sphere + orbiting bodies).
    Orbit,
    /// Rainbow chrome torus sliced to reveal trippy swirl.
    Torus,
    /// Mandelbulb fractal in rainbow chrome.
    Mandelbulb,
    /// Glass Menger sponge with fire inside.
    Menger,
    /// Gyroid minimal surface in rainbow.
    Gyroid,
    /// Volumetric god rays through Menger sponge.
    GodRays,
    /// Translucent subsurface scattering demo.
    Sss,
    /// Animated SDF shape morphing (sphere ↔ torus).
    Morph,
}

impl AnimationProgram {
    /// Build the SDF scene for a given time `t` (seconds since animation start).
    #[allow(clippy::cast_precision_loss)]
    pub fn build_scene(self, t: f32) -> SdfScene {
        let transparent = Color {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        };

        match self {
            Self::Cube => build_cube(t, transparent),
            Self::Vortex => build_vortex(t, transparent),
            Self::Pulse => build_pulse(t, transparent),
            Self::Orbit => build_orbit(t, transparent),
            Self::Torus => build_torus(t, transparent),
            Self::Mandelbulb => build_mandelbulb(t, transparent),
            Self::Menger => build_menger(t, transparent),
            Self::Gyroid => build_gyroid(t, transparent),
            Self::GodRays => build_godrays(t, transparent),
            Self::Sss => build_sss(t, transparent),
            Self::Morph => build_morph(t, transparent),
        }
    }
}

// ---------------------------------------------------------------------------
// Scene builders — extracted from scry-cli play.rs
// ---------------------------------------------------------------------------

fn orbiting_camera(t: f32, radius: f32, base_y: f32, y_amp: f32, speed: f32) -> Vec3 {
    let angle = t * speed;
    let y = base_y + (t * 0.1).sin() * y_amp;
    Vec3::new(angle.cos() * radius, y, angle.sin() * radius)
}

fn build_cube(t: f32, sky: Color) -> SdfScene {
    let qy = Quaternion::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), t * 0.5);
    let qx = Quaternion::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), t * 0.3);
    let orientation = qy * qx;

    let cube = SdfObject::new(
        SdfShape::Box {
            half_extents: Vec3::new(1.0, 1.0, 1.0),
        },
        Material::Rainbow {
            saturation: 1.0,
            lightness: 0.5,
            hue_offset: t * 0.5,
            specular: 64.0,
        },
    )
    .at(Vec3::new(0.0, 1.0, 0.0))
    .orient(orientation);

    let ground = SdfObject::new(
        SdfShape::Plane,
        Material::matte(Color::from_rgba8(60, 60, 60, 255)),
    );

    let eye = orbiting_camera(t, 6.0, 3.0, 0.5, 0.15);

    SdfScene::new()
        .object(cube)
        .object(ground)
        .light(SdfLight::new(
            Vec3::new(5.0, 8.0, 5.0),
            Color::WHITE,
            1.0,
        ))
        .light(SdfLight::new(
            Vec3::new(-3.0, 4.0, -2.0),
            Color::from_rgba8(100, 150, 255, 255),
            0.5,
        ))
        .camera(SdfCamera::new(eye, Vec3::new(0.0, 1.0, 0.0), 45.0))
        .sky_color(sky)
        .ambient(0.06)
}

fn build_vortex(t: f32, sky: Color) -> SdfScene {
    let qy = Quaternion::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), t * 0.4);
    let qx = Quaternion::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), t * 0.25);
    let orientation = qy * qx;

    let torus = SdfObject::new(
        SdfShape::Torus {
            major: 1.5,
            minor: 0.4 + (t * 1.5).sin() * 0.1,
        },
        Material::glass(Color::from_rgba8(200, 220, 255, 255), 1.4),
    )
    .at(Vec3::ZERO)
    .orient(orientation);

    let fire_r = 0.5 + (t * 2.0).sin() * 0.15;
    let fire = SdfObject::new(
        SdfShape::Sphere { radius: fire_r },
        Material::Fire {
            intensity: 3.5,
            noise_scale: 5.0,
            speed: 2.0,
        },
    )
    .at(Vec3::ZERO);

    let eye = orbiting_camera(t, 5.0, 2.0, 0.5, 0.2);

    SdfScene::new()
        .object(torus)
        .object(fire)
        .light(SdfLight::new(
            Vec3::new(5.0, 7.0, 4.0),
            Color::WHITE,
            1.2,
        ))
        .light(SdfLight::new(
            Vec3::new(-4.0, 3.0, -3.0),
            Color::from_rgba8(255, 180, 100, 255),
            0.5,
        ))
        .camera(SdfCamera::new(eye, Vec3::ZERO, 45.0))
        .sky_color(sky)
        .ambient(0.08)
        .max_bounces(2)
}

fn build_pulse(t: f32, sky: Color) -> SdfScene {
    let pulse = 1.0 + (t * 1.2).sin() * 0.3;
    let r1 = 0.8 * pulse;
    let r2 = 0.6 * (1.0 + (t * 1.5 + 1.0).sin() * 0.3);

    let orb = SdfObject::new(
        SdfShape::SmoothBlend {
            a: Box::new(SdfShape::Sphere { radius: r1 }),
            b: Box::new(SdfShape::Sphere { radius: r2 }),
            b_offset: Vec3::new(
                (t * 0.7).sin() * 0.5,
                (t * 0.9).cos() * 0.4,
                (t * 0.5).sin() * 0.3,
            ),
            k: 0.8,
        },
        Material::Subsurface {
            color: Color::from_rgba8(220, 180, 255, 255),
            scatter_color: Color::from_rgba8(60, 20, 80, 255),
            thickness: 0.3,
            specular: 32.0,
        },
    )
    .at(Vec3::ZERO);

    let eye = orbiting_camera(t, 4.5, 1.5, 0.3, 0.15);

    SdfScene::new()
        .object(orb)
        .light(SdfLight::new(
            Vec3::new(4.0, 6.0, 4.0),
            Color::WHITE,
            1.3,
        ))
        .light(SdfLight::new(
            Vec3::new(-3.0, 2.0, -4.0),
            Color::from_rgba8(180, 100, 255, 255),
            0.6,
        ))
        .camera(SdfCamera::new(eye, Vec3::ZERO, 45.0))
        .sky_color(sky)
        .ambient(0.08)
}

fn build_orbit(t: f32, sky: Color) -> SdfScene {
    let pi = std::f32::consts::PI;

    // Central mirror sphere
    let center = SdfObject::new(
        SdfShape::Sphere { radius: 1.0 },
        Material::mirror(Color::from_rgba8(240, 240, 255, 255), 0.9),
    )
    .at(Vec3::ZERO);

    // Orbiter 1
    let a1 = t * 0.8;
    let r1 = 2.2;
    let pos1 = Vec3::new(a1.cos() * r1, (t * 0.5).sin() * 0.5, a1.sin() * r1);
    let orbiter1 = SdfObject::new(
        SdfShape::Sphere { radius: 0.4 },
        Material::Rainbow {
            saturation: 1.0,
            lightness: 0.5,
            hue_offset: t * 0.3,
            specular: 32.0,
        },
    )
    .at(pos1);

    // Orbiter 2
    let a2 = t * 0.6 + pi * 2.0 / 3.0;
    let r2 = 2.6;
    let tilt2 = t * 0.3;
    let pos2 = Vec3::new(
        a2.cos() * r2,
        tilt2.sin() * 1.2,
        a2.sin() * r2 * tilt2.cos(),
    );
    let orbiter2 = SdfObject::new(
        SdfShape::Sphere { radius: 0.35 },
        Material::glass_dispersive(Color::from_rgba8(180, 240, 255, 255), 1.5, 0.03),
    )
    .at(pos2);

    // Orbiter 3
    let a3 = t * 0.4 + pi * 4.0 / 3.0;
    let r3 = 2.8;
    let pos3 = Vec3::new(a3.sin() * r3 * 0.3, a3.cos() * r3, a3.sin() * r3 * 0.95);
    let qr3 = Quaternion::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), a3);
    let orbiter3 = SdfObject::new(
        SdfShape::Torus {
            major: 0.25,
            minor: 0.08,
        },
        Material::rainbow_animated(t * 0.5),
    )
    .at(pos3)
    .orient(qr3);

    let eye = orbiting_camera(t, 6.0, 3.0, 0.5, 0.15);

    SdfScene::new()
        .object(center)
        .object(orbiter1)
        .object(orbiter2)
        .object(orbiter3)
        .light(SdfLight::new(
            Vec3::new(6.0, 8.0, 4.0),
            Color::WHITE,
            1.3,
        ))
        .light(SdfLight::new(
            Vec3::new(-5.0, 3.0, -3.0),
            Color::from_rgba8(100, 140, 255, 255),
            0.6,
        ))
        .camera(SdfCamera::new(eye, Vec3::ZERO, 45.0))
        .sky_color(sky)
        .ambient(0.06)
        .max_bounces(3)
}

fn build_torus(t: f32, sky: Color) -> SdfScene {
    let qy = Quaternion::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), t * 0.3);
    let qx = Quaternion::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), t * 0.2);
    let orientation = qy * qx;

    let minor = 0.5 + (t * 0.8).sin() * 0.1;
    let torus = SdfObject::new(
        SdfShape::Torus {
            major: 1.5,
            minor,
        },
        Material::Rainbow {
            saturation: 1.0,
            lightness: 0.45,
            hue_offset: t * 0.4,
            specular: 96.0,
        },
    )
    .at(Vec3::ZERO)
    .orient(orientation);

    let eye = orbiting_camera(t, 5.0, 2.5, 0.5, 0.18);

    SdfScene::new()
        .object(torus)
        .light(SdfLight::new(
            Vec3::new(5.0, 7.0, 4.0),
            Color::WHITE,
            1.2,
        ))
        .light(SdfLight::new(
            Vec3::new(-4.0, 3.0, -3.0),
            Color::from_rgba8(255, 160, 80, 255),
            0.5,
        ))
        .camera(SdfCamera::new(eye, Vec3::ZERO, 45.0))
        .sky_color(sky)
        .ambient(0.07)
}

fn build_mandelbulb(t: f32, sky: Color) -> SdfScene {
    let qy = Quaternion::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), t * 0.3);
    let power = 8.0 + (t * 0.2).sin() * 0.5;

    let bulb = SdfObject::new(
        SdfShape::Mandelbulb {
            power,
            iterations: 10,
        },
        Material::Rainbow {
            saturation: 0.9,
            lightness: 0.45,
            hue_offset: t * 0.3,
            specular: 96.0,
        },
    )
    .at(Vec3::ZERO)
    .orient(qy);

    let eye = orbiting_camera(t, 3.5, 1.5, 0.5, 0.15);

    SdfScene::new()
        .object(bulb)
        .light(SdfLight::new(
            Vec3::new(4.0, 6.0, 4.0),
            Color::WHITE,
            1.2,
        ))
        .light(SdfLight::new(
            Vec3::new(-3.0, 2.0, -4.0),
            Color::from_rgba8(180, 120, 255, 255),
            0.6,
        ))
        .light(SdfLight::new(
            Vec3::new(2.0, -1.0, 5.0),
            Color::from_rgba8(120, 255, 180, 255),
            0.4,
        ))
        .camera(SdfCamera::new(eye, Vec3::ZERO, 45.0))
        .sky_color(sky)
        .ambient(0.06)
        .max_bounces(1)
}

fn build_menger(t: f32, sky: Color) -> SdfScene {
    let qx = Quaternion::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), t * 0.15);
    let qy = Quaternion::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), t * 0.25);
    let orientation = qy * qx;

    let sponge = SdfObject::new(
        SdfShape::MengerSponge { iterations: 4 },
        Material::glass(Color::from_rgba8(220, 230, 255, 255), 1.35),
    )
    .at(Vec3::ZERO)
    .orient(orientation);

    let fire_r = 0.3 + (t * 1.2).sin() * 0.08;
    let fire_core = SdfObject::new(
        SdfShape::Sphere { radius: fire_r },
        Material::Fire {
            intensity: 3.0,
            noise_scale: 4.0,
            speed: 1.5,
        },
    )
    .at(Vec3::ZERO);

    let eye = orbiting_camera(t, 4.0, 2.0, 0.6, 0.2);

    SdfScene::new()
        .object(sponge)
        .object(fire_core)
        .light(SdfLight::new(
            Vec3::new(5.0, 7.0, 4.0),
            Color::WHITE,
            1.4,
        ))
        .light(SdfLight::new(
            Vec3::new(-3.0, 3.0, -5.0),
            Color::from_rgba8(255, 180, 100, 255),
            0.5,
        ))
        .camera(SdfCamera::new(eye, Vec3::ZERO, 45.0))
        .sky_color(sky)
        .ambient(0.08)
        .max_bounces(2)
}

fn build_gyroid(t: f32, sky: Color) -> SdfScene {
    let qx = Quaternion::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), t * 0.2);
    let qy = Quaternion::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), t * 0.35);
    let qz = Quaternion::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), t * 0.1);
    let orientation = qz * qy * qx;

    let scale = 4.0 + (t * 0.15).sin() * 0.5;

    let gyroid = SdfObject::new(
        SdfShape::Gyroid {
            scale,
            thickness: 0.25,
            bound: 1.5,
        },
        Material::Rainbow {
            saturation: 1.0,
            lightness: 0.5,
            hue_offset: t * 0.4,
            specular: 64.0,
        },
    )
    .at(Vec3::ZERO)
    .orient(orientation);

    let eye = orbiting_camera(t, 4.5, 2.0, 0.5, 0.18);

    SdfScene::new()
        .object(gyroid)
        .light(SdfLight::new(
            Vec3::new(5.0, 7.0, 3.0),
            Color::WHITE,
            1.3,
        ))
        .light(SdfLight::new(
            Vec3::new(-4.0, 2.0, -4.0),
            Color::from_rgba8(255, 130, 200, 255),
            0.5,
        ))
        .camera(SdfCamera::new(eye, Vec3::ZERO, 45.0))
        .sky_color(sky)
        .ambient(0.06)
}

fn build_godrays(t: f32, sky: Color) -> SdfScene {
    let qx = Quaternion::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), t * 0.12);
    let qy = Quaternion::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), t * 0.2);
    let orientation = qy * qx;

    let sponge = SdfObject::new(
        SdfShape::MengerSponge { iterations: 3 },
        Material::matte(Color::from_rgba8(180, 180, 180, 255)),
    )
    .at(Vec3::ZERO)
    .orient(orientation);

    let cam_angle = t * 0.15;
    let cam_r = 5.0;
    let eye = Vec3::new(
        cam_angle.cos() * cam_r,
        2.5 + (t * 0.1).sin() * 0.5,
        cam_angle.sin() * cam_r,
    );

    SdfScene::new()
        .object(sponge)
        .light(SdfLight::new(
            Vec3::new(4.0, 8.0, 3.0),
            Color::from_rgba8(255, 240, 200, 255),
            1.5,
        ))
        .camera(SdfCamera::new(eye, Vec3::ZERO, 45.0))
        .sky_color(sky)
        .ambient(0.03)
        .god_rays(0.4, 32)
}

fn build_sss(t: f32, sky: Color) -> SdfScene {
    let orb = SdfObject::new(
        SdfShape::Sphere {
            radius: 1.2 + (t * 0.8).sin() * 0.1,
        },
        Material::Subsurface {
            color: Color::from_rgba8(180, 220, 160, 255),
            scatter_color: Color::from_rgba8(30, 60, 20, 255),
            thickness: 0.4,
            specular: 32.0,
        },
    )
    .at(Vec3::ZERO);

    let eye = orbiting_camera(t, 4.0, 1.5, 0.3, 0.15);

    SdfScene::new()
        .object(orb)
        .light(SdfLight::new(
            Vec3::new(4.0, 6.0, 4.0),
            Color::WHITE,
            1.4,
        ))
        .light(SdfLight::new(
            Vec3::new(-3.0, 2.0, -3.0),
            Color::from_rgba8(255, 200, 150, 255),
            0.6,
        ))
        .camera(SdfCamera::new(eye, Vec3::ZERO, 45.0))
        .sky_color(sky)
        .ambient(0.06)
}

fn build_morph(t: f32, sky: Color) -> SdfScene {
    // Ping-pong morph factor between 0 and 1
    let cycle = (t * 0.3).sin() * 0.5 + 0.5;

    let morph_obj = SdfObject::new(
        SdfShape::Morph {
            a: Box::new(SdfShape::Sphere { radius: 1.2 }),
            b: Box::new(SdfShape::Torus {
                major: 1.0,
                minor: 0.4,
            }),
            t: cycle,
        },
        Material::Rainbow {
            saturation: 1.0,
            lightness: 0.5,
            hue_offset: t * 0.4,
            specular: 64.0,
        },
    )
    .at(Vec3::ZERO)
    .rotate_y(t * 0.3);

    let eye = orbiting_camera(t, 5.0, 2.0, 0.5, 0.15);

    SdfScene::new()
        .object(morph_obj)
        .light(SdfLight::new(
            Vec3::new(5.0, 7.0, 4.0),
            Color::WHITE,
            1.2,
        ))
        .light(SdfLight::new(
            Vec3::new(-4.0, 3.0, -3.0),
            Color::from_rgba8(120, 180, 255, 255),
            0.5,
        ))
        .camera(SdfCamera::new(eye, Vec3::ZERO, 45.0))
        .sky_color(sky)
        .ambient(0.06)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_programs_build_scenes() {
        let programs = [
            AnimationProgram::Cube,
            AnimationProgram::Vortex,
            AnimationProgram::Pulse,
            AnimationProgram::Orbit,
            AnimationProgram::Torus,
            AnimationProgram::Mandelbulb,
            AnimationProgram::Menger,
            AnimationProgram::Gyroid,
            AnimationProgram::GodRays,
            AnimationProgram::Sss,
            AnimationProgram::Morph,
        ];

        for prog in programs {
            let scene = prog.build_scene(0.0);
            assert!(!scene.objects.is_empty(), "{prog:?} should have objects");
            assert!(!scene.lights.is_empty(), "{prog:?} should have lights");

            // Also test with non-zero time
            let scene2 = prog.build_scene(2.5);
            assert!(!scene2.objects.is_empty());
        }
    }

    #[test]
    fn serialization_roundtrip() {
        let program = AnimationProgram::Mandelbulb;
        let bytes = postcard::to_allocvec(&program).unwrap();
        let recovered: AnimationProgram = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(program, recovered);
    }
}
