// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Scry Logo** — SDF-rendered logo variants using the scry engine's own ray marcher.
//!
//! Renders multiple logo concepts from clean to psychedelic, using fractals,
//! gyroids, fire, glass, and infinite reflections.
//!
//! Run with:
//!   `cargo run --example scry_logo --features sdf --release`

#![allow(
    clippy::cast_precision_loss,
    clippy::suboptimal_flops,
    clippy::similar_names,
    clippy::too_many_lines
)]

use scry_engine::scene::style::Color;
use scry_engine::sdf::{
    Material, SdfCamera, SdfLight, SdfObject, SdfRenderer, SdfScene, SdfShape, Vec3,
};

use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

fn main() {
    let out_dir = "/tmp/scry_logo";
    std::fs::create_dir_all(out_dir).unwrap();
    let size: u32 = 1024;

    println!("⟡ scry logo — rendering {size}×{size} variants...");
    println!();

    let variants: Vec<(&str, fn() -> SdfScene)> = vec![
        ("01_obsidian_mirror", build_obsidian_mirror),
        ("02_recursive_eye", build_recursive_eye),
        ("03_loom_of_threads", build_loom_of_threads),
        ("04_ember_door", build_ember_door),
        ("05_nested_glass", build_nested_glass),
        ("06_mirror_corridor", build_mirror_corridor),
    ];

    for (name, builder) in &variants {
        print!("  ⟐ {name}... ");
        let scene = builder();
        let pixmap =
            SdfRenderer::render_to_pixmap(&scene, size, size, 0.0).expect("SDF render failed");
        let path = format!("{out_dir}/{name}.png");
        pixmap.save_png(&path).expect("PNG save failed");
        println!("✓");
    }

    println!();
    println!("✓ All variants saved to {out_dir}/");
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. OBSIDIAN MIRROR — the original, refined
// ═══════════════════════════════════════════════════════════════════════════

fn build_obsidian_mirror() -> SdfScene {
    let obsidian = Color::from_rgba8(15, 15, 20, 255);
    let silver = Color::from_rgba8(180, 185, 200, 255);
    let gold_warm = Color::from_rgba8(200, 170, 100, 255);
    let deep_sky = Color::from_rgba8(5, 5, 15, 255);
    let blue = Color::from_rgba8(96, 165, 250, 255);
    let violet = Color::from_rgba8(167, 139, 250, 255);
    let green = Color::from_rgba8(74, 222, 128, 255);
    let orange = Color::from_rgba8(251, 146, 60, 255);
    let red = Color::from_rgba8(248, 113, 113, 255);

    let pillar_data: [(f32, f32, f32, Color); 5] = [
        (-2.8, 0.9, -1.5, blue),
        (-2.0, 1.5, -2.2, violet),
        (0.0, 2.2, -2.8, green),
        (2.0, 3.0, -2.2, orange),
        (2.8, 4.0, -1.5, red),
    ];

    let mut scene = SdfScene::new()
        .object(SdfObject::new(
            SdfShape::Plane,
            Material::Solid {
                color: Color::from_rgba8(8, 8, 14, 255),
                reflectivity: 0.35,
                specular: 48.0,
            },
        ))
        // Mirror disc
        .object(
            SdfObject::new(
                SdfShape::Cylinder {
                    radius: 2.0,
                    half_height: 0.06,
                },
                Material::Solid {
                    color: obsidian,
                    reflectivity: 0.85,
                    specular: 128.0,
                },
            )
            .at(Vec3::new(0.0, 1.8, 0.0))
            .rotate(Vec3::new(1.0, 0.0, 0.0), 0.52),
        )
        // Rim
        .object(
            SdfObject::new(
                SdfShape::Torus {
                    major: 2.0,
                    minor: 0.1,
                },
                Material::Solid {
                    color: silver,
                    reflectivity: 0.5,
                    specular: 64.0,
                },
            )
            .at(Vec3::new(0.0, 1.8, 0.0))
            .rotate(Vec3::new(1.0, 0.0, 0.0), 0.52),
        )
        // Handle
        .object(
            SdfObject::new(
                SdfShape::Capsule {
                    radius: 0.18,
                    half_height: 1.2,
                },
                Material::Solid {
                    color: gold_warm,
                    reflectivity: 0.3,
                    specular: 48.0,
                },
            )
            .at(Vec3::new(0.0, -0.2, 0.7)),
        )
        // Pommel
        .object(
            SdfObject::new(
                SdfShape::Sphere { radius: 0.28 },
                Material::Solid {
                    color: gold_warm,
                    reflectivity: 0.4,
                    specular: 64.0,
                },
            )
            .at(Vec3::new(0.0, -1.45, 0.7)),
        )
        // Glass orb
        .object(
            SdfObject::new(
                SdfShape::Sphere { radius: 0.6 },
                Material::glass_dispersive(Color::from_rgba8(220, 225, 255, 255), 1.52, 0.03),
            )
            .at(Vec3::new(0.0, 2.35, -0.2)),
        )
        // Rainbow disc
        .object(
            SdfObject::new(
                SdfShape::Cylinder {
                    radius: 1.6,
                    half_height: 0.008,
                },
                Material::Rainbow {
                    saturation: 0.85,
                    lightness: 0.45,
                    hue_offset: 0.0,
                    specular: 48.0,
                },
            )
            .at(Vec3::new(0.0, 1.84, 0.0))
            .rotate(Vec3::new(1.0, 0.0, 0.0), 0.52),
        );

    for (x, height, z, color) in &pillar_data {
        scene = scene.object(
            SdfObject::new(
                SdfShape::RoundedBox {
                    half_extents: Vec3::new(0.22, height / 2.0, 0.18),
                    radius: 0.07,
                },
                Material::Solid {
                    color: *color,
                    reflectivity: 0.2,
                    specular: 48.0,
                },
            )
            .at(Vec3::new(*x, height / 2.0, *z)),
        );
    }

    let accents = [
        (Vec3::new(-2.5, 3.8, 0.5), 0.12, blue),
        (Vec3::new(2.8, 4.2, 0.2), 0.10, violet),
        (Vec3::new(-1.5, 4.5, -1.0), 0.08, green),
        (Vec3::new(1.2, 5.0, -0.5), 0.09, orange),
    ];
    for (pos, radius, color) in &accents {
        scene = scene.object(
            SdfObject::new(
                SdfShape::Sphere { radius: *radius },
                Material::Solid {
                    color: *color,
                    reflectivity: 0.6,
                    specular: 128.0,
                },
            )
            .at(*pos),
        );
    }

    scene
        .light(SdfLight::new(
            Vec3::new(5.0, 10.0, 5.0),
            Color::from_rgba8(255, 240, 210, 255),
            0.85,
        ))
        .light(SdfLight::new(
            Vec3::new(-6.0, 7.0, 3.0),
            Color::from_rgba8(100, 130, 220, 255),
            0.4,
        ))
        .light(SdfLight::new(
            Vec3::new(0.0, 6.0, -5.0),
            Color::from_rgba8(180, 160, 220, 255),
            0.5,
        ))
        .camera(SdfCamera::new(
            Vec3::new(0.0, 4.5, 8.0),
            Vec3::new(0.0, 1.6, 0.0),
            44.0,
        ))
        .max_bounces(3)
        .ambient(0.05)
        .sky_color(deep_sky)
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. RECURSIVE EYE — "empty recursion span the bits" / "the face of the
//    order peer at me" / "wheel turn again" / Mandelbulb as the all-seeing
//    fractal iris, Menger sponge pupil, concentric torus rings, all on a
//    reflective checkerboard — infinite regression in every direction.
// ═══════════════════════════════════════════════════════════════════════════

fn build_recursive_eye() -> SdfScene {
    let void = Color::from_rgba8(2, 2, 8, 255);
    let iris_hue = Color::from_rgba8(40, 180, 160, 255); // teal-cyan
    let pupil = Color::from_rgba8(8, 8, 20, 255);
    let gold = Color::from_rgba8(220, 185, 100, 255);

    SdfScene::new()
        // Checkerboard plane — "checkmate if empty"
        .object(SdfObject::new(
            SdfShape::Plane,
            Material::Checkerboard {
                color_a: Color::from_rgba8(12, 12, 18, 255),
                color_b: Color::from_rgba8(35, 35, 50, 255),
                scale: 1.5,
                reflectivity: 0.6,
                specular: 64.0,
            },
        ))
        // Mandelbulb — the fractal iris, the all-seeing recursive eye
        .object(
            SdfObject::new(
                SdfShape::Mandelbulb {
                    power: 8.0,
                    iterations: 12,
                },
                Material::Solid {
                    color: iris_hue,
                    reflectivity: 0.4,
                    specular: 96.0,
                },
            )
            .at(Vec3::new(0.0, 2.0, 0.0)),
        )
        // Menger sponge pupil — nested inside, recursion within recursion
        .object(
            SdfObject::new(
                SdfShape::MengerSponge { iterations: 4 },
                Material::Solid {
                    color: pupil,
                    reflectivity: 0.7,
                    specular: 128.0,
                },
            )
            .at(Vec3::new(0.0, 2.0, 0.0)),
        )
        // Outer ring 1 — "wheel turn again"
        .object(
            SdfObject::new(
                SdfShape::Torus {
                    major: 2.2,
                    minor: 0.08,
                },
                Material::Solid {
                    color: gold,
                    reflectivity: 0.6,
                    specular: 96.0,
                },
            )
            .at(Vec3::new(0.0, 2.0, 0.0))
            .rotate(Vec3::new(1.0, 0.0, 0.0), FRAC_PI_4),
        )
        // Outer ring 2 — perpendicular
        .object(
            SdfObject::new(
                SdfShape::Torus {
                    major: 2.5,
                    minor: 0.06,
                },
                Material::Rainbow {
                    saturation: 0.9,
                    lightness: 0.5,
                    hue_offset: 0.0,
                    specular: 64.0,
                },
            )
            .at(Vec3::new(0.0, 2.0, 0.0))
            .rotate(Vec3::new(0.0, 0.0, 1.0), FRAC_PI_4),
        )
        // Outer ring 3 — tilted
        .object(
            SdfObject::new(
                SdfShape::Torus {
                    major: 2.8,
                    minor: 0.05,
                },
                Material::Solid {
                    color: Color::from_rgba8(160, 80, 200, 255),
                    reflectivity: 0.5,
                    specular: 64.0,
                },
            )
            .at(Vec3::new(0.0, 2.0, 0.0))
            .rotate(Vec3::new(0.5, 1.0, 0.0), 0.9),
        )
        // Glass enclosure — the lens through which the order peers
        .object(
            SdfObject::new(
                SdfShape::Sphere { radius: 3.2 },
                Material::Glass {
                    tint: Color::from_rgba8(200, 220, 255, 255),
                    ior: 1.3,
                    opacity: 0.0,
                    dispersion: 0.06,
                },
            )
            .at(Vec3::new(0.0, 2.0, 0.0)),
        )
        // Dramatic rim lighting
        .light(SdfLight::new(
            Vec3::new(6.0, 8.0, 6.0),
            Color::from_rgba8(255, 220, 180, 255),
            0.8,
        ))
        .light(SdfLight::new(
            Vec3::new(-5.0, 10.0, -3.0),
            Color::from_rgba8(120, 160, 255, 255),
            0.6,
        ))
        .light(SdfLight::new(
            Vec3::new(0.0, 0.5, 8.0),
            Color::from_rgba8(200, 100, 255, 255),
            0.3,
        ))
        .camera(SdfCamera::new(
            Vec3::new(0.0, 4.0, 8.0),
            Vec3::new(0.0, 2.0, 0.0),
            40.0,
        ))
        .max_bounces(4)
        .ambient(0.03)
        .sky_color(void)
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. LOOM OF THREADS — "cast copper threads into a loom of dancing thread"
//    / "energy weaving windfalls" / "a still a still a still" — Gyroid as
//    the woven fabric of reality, rainbow spectral threads, glass capsules
//    as spindles, all floating in contemplative darkness.
// ═══════════════════════════════════════════════════════════════════════════

fn build_loom_of_threads() -> SdfScene {
    let copper = Color::from_rgba8(184, 115, 51, 255);
    let deep_void = Color::from_rgba8(3, 3, 12, 255);

    let mut scene = SdfScene::new()
        // Reflective dark plane
        .object(SdfObject::new(
            SdfShape::Plane,
            Material::Solid {
                color: Color::from_rgba8(5, 5, 12, 255),
                reflectivity: 0.5,
                specular: 64.0,
            },
        ))
        // Central gyroid — the woven fabric of reality
        .object(
            SdfObject::new(
                SdfShape::Gyroid {
                    scale: 4.0,
                    thickness: 0.08,
                    bound: 2.0,
                },
                Material::Rainbow {
                    saturation: 0.9,
                    lightness: 0.5,
                    hue_offset: 0.0,
                    specular: 64.0,
                },
            )
            .at(Vec3::new(0.0, 2.5, 0.0))
            .rotate(Vec3::new(1.0, 1.0, 0.0), 0.4),
        )
        // Inner gyroid — finer weave, copper
        .object(
            SdfObject::new(
                SdfShape::Gyroid {
                    scale: 8.0,
                    thickness: 0.04,
                    bound: 1.2,
                },
                Material::Solid {
                    color: copper,
                    reflectivity: 0.5,
                    specular: 96.0,
                },
            )
            .at(Vec3::new(0.0, 2.5, 0.0))
            .rotate(Vec3::new(0.0, 1.0, 1.0), 0.7),
        )
        // Glass enclosure — the still, the distillation vessel
        .object(
            SdfObject::new(
                SdfShape::Sphere { radius: 2.5 },
                Material::Glass {
                    tint: Color::from_rgba8(240, 230, 255, 255),
                    ior: 1.15,
                    opacity: 0.0,
                    dispersion: 0.05,
                },
            )
            .at(Vec3::new(0.0, 2.5, 0.0)),
        );

    // Spindle capsules — "a still a still a still" — three vertical spindles
    let spindle_positions = [
        (Vec3::new(-3.0, 1.5, -1.0), 0.3),
        (Vec3::new(3.0, 1.5, -1.0), -0.3),
        (Vec3::new(0.0, 1.5, -3.5), 0.0),
    ];
    for (pos, tilt) in &spindle_positions {
        scene = scene.object(
            SdfObject::new(
                SdfShape::Capsule {
                    radius: 0.12,
                    half_height: 1.8,
                },
                Material::Solid {
                    color: copper,
                    reflectivity: 0.4,
                    specular: 64.0,
                },
            )
            .at(*pos)
            .rotate(Vec3::new(0.0, 0.0, 1.0), *tilt),
        );
    }

    // Thread connections — thin tori linking spindles to center
    let thread_angles = [0.0, PI * 2.0 / 3.0, PI * 4.0 / 3.0];
    for angle in &thread_angles {
        scene = scene.object(
            SdfObject::new(
                SdfShape::Torus {
                    major: 3.0,
                    minor: 0.02,
                },
                Material::Solid {
                    color: Color::from_rgba8(200, 160, 80, 255),
                    reflectivity: 0.3,
                    specular: 48.0,
                },
            )
            .at(Vec3::new(0.0, 2.5, 0.0))
            .rotate(Vec3::new(0.0, 1.0, 0.0), *angle),
        );
    }

    scene
        .light(SdfLight::new(
            Vec3::new(4.0, 10.0, 6.0),
            Color::from_rgba8(255, 200, 150, 255),
            0.9,
        ))
        .light(SdfLight::new(
            Vec3::new(-5.0, 6.0, -2.0),
            Color::from_rgba8(150, 100, 255, 255),
            0.5,
        ))
        .light(SdfLight::new(
            Vec3::new(0.0, 3.0, 5.0),
            Color::from_rgba8(100, 200, 180, 255),
            0.3,
        ))
        .camera(SdfCamera::new(
            Vec3::new(0.0, 5.0, 8.0),
            Vec3::new(0.0, 2.5, 0.0),
            46.0,
        ))
        .max_bounces(4)
        .ambient(0.04)
        .sky_color(deep_void)
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. EMBER DOOR — "branded in the coal covered ceiling" / "the handle is
//    missing" / "enthrone my desires" / "concresco of cascaded meaning" —
//    A Menger sponge doorframe with fire inside (the missing door), a
//    mirror throne reflecting it all, ember glow, smoke and void.
// ═══════════════════════════════════════════════════════════════════════════

fn build_ember_door() -> SdfScene {
    let charcoal = Color::from_rgba8(25, 20, 18, 255);
    let ember_sky = Color::from_rgba8(15, 5, 3, 255);
    let bone = Color::from_rgba8(200, 190, 170, 255);

    SdfScene::new()
        // Checkerboard floor — karmic scales
        .object(SdfObject::new(
            SdfShape::Plane,
            Material::Checkerboard {
                color_a: Color::from_rgba8(15, 12, 10, 255),
                color_b: Color::from_rgba8(40, 30, 25, 255),
                scale: 1.0,
                reflectivity: 0.4,
                specular: 32.0,
            },
        ))
        // Menger sponge doorframe — the recursive structure, the door
        // with holes but no handle. "I'm trying to open the door and
        // the handle is missing" — the sponge IS the door: full of
        // openings but no way through.
        .object(
            SdfObject::new(
                SdfShape::MengerSponge { iterations: 4 },
                Material::Solid {
                    color: charcoal,
                    reflectivity: 0.25,
                    specular: 32.0,
                },
            )
            .at(Vec3::new(0.0, 2.5, 0.0)),
        )
        // Fire core — the ember burning inside the recursive door
        // "branded in the coal covered ceiling"
        .object(
            SdfObject::new(
                SdfShape::Sphere { radius: 0.6 },
                Material::Fire {
                    intensity: 2.0,
                    noise_scale: 3.0,
                    speed: 0.8,
                },
            )
            .at(Vec3::new(0.0, 2.5, 0.0)),
        )
        // Mirror throne behind — "let me enthrone my desires"
        // A tall mirror slab that reflects everything back
        .object(
            SdfObject::new(
                SdfShape::Box {
                    half_extents: Vec3::new(2.5, 3.0, 0.08),
                },
                Material::Solid {
                    color: Color::from_rgba8(20, 20, 30, 255),
                    reflectivity: 0.9,
                    specular: 128.0,
                },
            )
            .at(Vec3::new(0.0, 3.0, -4.0)),
        )
        // Bone-white pillars flanking — the columns of the throne room
        .object(
            SdfObject::new(
                SdfShape::Cylinder {
                    radius: 0.2,
                    half_height: 3.0,
                },
                Material::Solid {
                    color: bone,
                    reflectivity: 0.15,
                    specular: 24.0,
                },
            )
            .at(Vec3::new(-3.0, 3.0, -3.0)),
        )
        .object(
            SdfObject::new(
                SdfShape::Cylinder {
                    radius: 0.2,
                    half_height: 3.0,
                },
                Material::Solid {
                    color: bone,
                    reflectivity: 0.15,
                    specular: 24.0,
                },
            )
            .at(Vec3::new(3.0, 3.0, -3.0)),
        )
        // Floating glass sphere — the weight on the karmic scale
        .object(
            SdfObject::new(
                SdfShape::Sphere { radius: 0.4 },
                Material::glass_dispersive(Color::from_rgba8(255, 200, 150, 255), 1.45, 0.04),
            )
            .at(Vec3::new(2.0, 4.5, 1.0)),
        )
        // Small obsidian sphere — counterweight
        .object(
            SdfObject::new(
                SdfShape::Sphere { radius: 0.3 },
                Material::Solid {
                    color: Color::from_rgba8(10, 10, 15, 255),
                    reflectivity: 0.8,
                    specular: 128.0,
                },
            )
            .at(Vec3::new(-2.0, 4.0, 1.5)),
        )
        // Scale beam — thin cylinder connecting the two spheres
        .object(
            SdfObject::new(
                SdfShape::Cylinder {
                    radius: 0.03,
                    half_height: 2.5,
                },
                Material::Solid {
                    color: Color::from_rgba8(160, 140, 100, 255),
                    reflectivity: 0.3,
                    specular: 32.0,
                },
            )
            .at(Vec3::new(0.0, 4.3, 1.25))
            .rotate(Vec3::new(0.0, 0.0, 1.0), 0.12),
        )
        // Warm ember light from below — coal glow
        .light(SdfLight::new(
            Vec3::new(0.0, 0.3, 0.0),
            Color::from_rgba8(255, 120, 40, 255),
            0.7,
        ))
        // Cool overhead
        .light(SdfLight::new(
            Vec3::new(0.0, 12.0, 4.0),
            Color::from_rgba8(180, 160, 200, 255),
            0.5,
        ))
        // Side accent
        .light(SdfLight::new(
            Vec3::new(-6.0, 5.0, 3.0),
            Color::from_rgba8(255, 80, 30, 255),
            0.3,
        ))
        .camera(SdfCamera::new(
            Vec3::new(0.0, 3.5, 8.0),
            Vec3::new(0.0, 2.5, 0.0),
            42.0,
        ))
        .max_bounces(4)
        .ambient(0.03)
        .sky_color(ember_sky)
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. NESTED GLASS — Concentric glass spheres with varying IOR and
//    dispersion. Each shell bends light differently, creating layered
//    prismatic halos. At the center: a Mandelbulb core, visible only
//    through the compounded refractions of every shell.
// ═══════════════════════════════════════════════════════════════════════════

fn build_nested_glass() -> SdfScene {
    let void = Color::from_rgba8(3, 3, 10, 255);

    // Each shell: (radius, ior, dispersion, tint)
    let shells: [(f32, f32, f32, Color); 5] = [
        (2.8, 1.10, 0.08, Color::from_rgba8(255, 200, 220, 255)), // outermost — rose
        (2.2, 1.20, 0.06, Color::from_rgba8(200, 220, 255, 255)), // blue-white
        (1.6, 1.35, 0.05, Color::from_rgba8(220, 255, 220, 255)), // pale green
        (1.1, 1.50, 0.04, Color::from_rgba8(255, 240, 200, 255)), // warm gold
        (0.6, 1.70, 0.10, Color::from_rgba8(230, 200, 255, 255)), // violet — innermost, strongest dispersion
    ];

    let mut scene = SdfScene::new()
        // Reflective floor — catches the caustic light
        .object(SdfObject::new(
            SdfShape::Plane,
            Material::Solid {
                color: Color::from_rgba8(6, 6, 14, 255),
                reflectivity: 0.5,
                specular: 64.0,
            },
        ))
        // Core — a small Mandelbulb at the very center, the seed of
        // all this refraction, visible only through layers of glass
        .object(
            SdfObject::new(
                SdfShape::Mandelbulb {
                    power: 7.0,
                    iterations: 10,
                },
                Material::Rainbow {
                    saturation: 1.0,
                    lightness: 0.55,
                    hue_offset: 0.0,
                    specular: 96.0,
                },
            )
            .at(Vec3::new(0.0, 3.0, 0.0)),
        );

    // Add concentric glass shells
    for (radius, ior, dispersion, tint) in &shells {
        scene = scene.object(
            SdfObject::new(
                SdfShape::Sphere { radius: *radius },
                Material::Glass {
                    tint: *tint,
                    ior: *ior,
                    opacity: 0.0,
                    dispersion: *dispersion,
                },
            )
            .at(Vec3::new(0.0, 3.0, 0.0)),
        );
    }

    // Orbiting accent tori — thin rings at different tilts around the
    // glass assembly, like electron orbits
    let orbit_data: [(f32, f32, Vec3, Color); 3] = [
        (
            3.5,
            0.04,
            Vec3::new(1.0, 0.3, 0.0),
            Color::from_rgba8(96, 165, 250, 255),
        ),
        (
            4.0,
            0.03,
            Vec3::new(0.0, 0.5, 1.0),
            Color::from_rgba8(167, 139, 250, 255),
        ),
        (
            4.5,
            0.025,
            Vec3::new(0.7, 1.0, 0.3),
            Color::from_rgba8(74, 222, 128, 255),
        ),
    ];
    for (major, minor, axis, color) in &orbit_data {
        scene = scene.object(
            SdfObject::new(
                SdfShape::Torus {
                    major: *major,
                    minor: *minor,
                },
                Material::Solid {
                    color: *color,
                    reflectivity: 0.4,
                    specular: 64.0,
                },
            )
            .at(Vec3::new(0.0, 3.0, 0.0))
            .rotate(*axis, 0.8),
        );
    }

    scene
        // Key light — bright white from above-right
        .light(SdfLight::new(Vec3::new(6.0, 12.0, 6.0), Color::WHITE, 0.9))
        // Cool fill from left
        .light(SdfLight::new(
            Vec3::new(-8.0, 6.0, 4.0),
            Color::from_rgba8(120, 140, 255, 255),
            0.5,
        ))
        // Warm accent from behind — highlights shell edges
        .light(SdfLight::new(
            Vec3::new(0.0, 4.0, -8.0),
            Color::from_rgba8(255, 200, 130, 255),
            0.4,
        ))
        .camera(SdfCamera::new(
            Vec3::new(0.0, 5.5, 10.0),
            Vec3::new(0.0, 3.0, 0.0),
            38.0, // tighter FOV to fill frame with the sphere cluster
        ))
        .max_bounces(5) // extra bounces for multi-shell refraction
        .ambient(0.03)
        .sky_color(void)
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. MIRROR CORRIDOR — Two facing mirror walls creating infinite
//    regression. Between them: rainbow objects, glass spheres, and a
//    checkerboard floor stretching into recursive infinity.
// ═══════════════════════════════════════════════════════════════════════════

fn build_mirror_corridor() -> SdfScene {
    let void = Color::from_rgba8(2, 2, 6, 255);

    let mut scene = SdfScene::new()
        // Checkerboard floor — extends into both mirror reflections
        .object(SdfObject::new(
            SdfShape::Plane,
            Material::Checkerboard {
                color_a: Color::from_rgba8(10, 10, 18, 255),
                color_b: Color::from_rgba8(30, 30, 45, 255),
                scale: 1.0,
                reflectivity: 0.45,
                specular: 48.0,
            },
        ))
        // LEFT MIRROR WALL — massive, near-perfect reflector
        .object(
            SdfObject::new(
                SdfShape::Box {
                    half_extents: Vec3::new(0.05, 5.0, 8.0),
                },
                Material::Solid {
                    color: Color::from_rgba8(12, 12, 20, 255),
                    reflectivity: 0.95,
                    specular: 200.0,
                },
            )
            .at(Vec3::new(-4.0, 5.0, 0.0)),
        )
        // RIGHT MIRROR WALL — facing the left one
        .object(
            SdfObject::new(
                SdfShape::Box {
                    half_extents: Vec3::new(0.05, 5.0, 8.0),
                },
                Material::Solid {
                    color: Color::from_rgba8(12, 12, 20, 255),
                    reflectivity: 0.95,
                    specular: 200.0,
                },
            )
            .at(Vec3::new(4.0, 5.0, 0.0)),
        )
        // Central rainbow sphere — the main subject reflected infinitely
        .object(
            SdfObject::new(
                SdfShape::Sphere { radius: 0.8 },
                Material::Rainbow {
                    saturation: 1.0,
                    lightness: 0.5,
                    hue_offset: 0.0,
                    specular: 96.0,
                },
            )
            .at(Vec3::new(0.0, 1.5, 0.0)),
        )
        // Glass sphere — offset, creates asymmetric reflections
        .object(
            SdfObject::new(
                SdfShape::Sphere { radius: 0.5 },
                Material::glass_dispersive(Color::from_rgba8(230, 240, 255, 255), 1.5, 0.06),
            )
            .at(Vec3::new(1.2, 0.8, -1.5)),
        )
        // Small mirror sphere — creates recursive reflection-in-reflection
        .object(
            SdfObject::new(
                SdfShape::Sphere { radius: 0.35 },
                Material::Solid {
                    color: Color::from_rgba8(20, 20, 30, 255),
                    reflectivity: 0.9,
                    specular: 128.0,
                },
            )
            .at(Vec3::new(-1.0, 0.6, 1.0)),
        );

    // Vertical accent pillars — thin columns along the corridor
    let pillar_colors = [
        Color::from_rgba8(96, 165, 250, 255),  // blue
        Color::from_rgba8(167, 139, 250, 255), // violet
        Color::from_rgba8(248, 113, 113, 255), // red
        Color::from_rgba8(74, 222, 128, 255),  // green
    ];
    let pillar_z = [-3.0, -1.0, 1.0, 3.0];
    for (z, color) in pillar_z.iter().zip(pillar_colors.iter()) {
        // Pillars on both sides of the corridor
        scene = scene
            .object(
                SdfObject::new(
                    SdfShape::Cylinder {
                        radius: 0.08,
                        half_height: 2.5,
                    },
                    Material::Solid {
                        color: *color,
                        reflectivity: 0.3,
                        specular: 48.0,
                    },
                )
                .at(Vec3::new(-3.5, 2.5, *z)),
            )
            .object(
                SdfObject::new(
                    SdfShape::Cylinder {
                        radius: 0.08,
                        half_height: 2.5,
                    },
                    Material::Solid {
                        color: *color,
                        reflectivity: 0.3,
                        specular: 48.0,
                    },
                )
                .at(Vec3::new(3.5, 2.5, *z)),
            );
    }

    // Horizontal torus gate — archway in the corridor
    scene = scene.object(
        SdfObject::new(
            SdfShape::Torus {
                major: 3.0,
                minor: 0.06,
            },
            Material::Solid {
                color: Color::from_rgba8(200, 180, 120, 255),
                reflectivity: 0.5,
                specular: 64.0,
            },
        )
        .at(Vec3::new(0.0, 3.0, 0.0))
        .rotate(Vec3::new(0.0, 0.0, 1.0), FRAC_PI_2),
    );

    scene
        // Bright overhead strip — runs down the corridor
        .light(SdfLight::new(Vec3::new(0.0, 8.0, 0.0), Color::WHITE, 0.8))
        // Warm side light — asymmetric to create depth
        .light(SdfLight::new(
            Vec3::new(2.0, 3.0, 5.0),
            Color::from_rgba8(255, 200, 150, 255),
            0.5,
        ))
        // Cool accent from the other end
        .light(SdfLight::new(
            Vec3::new(-1.0, 2.0, -6.0),
            Color::from_rgba8(100, 150, 255, 255),
            0.4,
        ))
        // Purple uplighting for drama
        .light(SdfLight::new(
            Vec3::new(0.0, 0.2, 0.0),
            Color::from_rgba8(180, 100, 255, 255),
            0.2,
        ))
        .camera(SdfCamera::new(
            Vec3::new(2.5, 2.5, 6.0),  // offset camera — not centered, so you
            Vec3::new(0.0, 1.5, -1.0), // see the reflections recede at an angle
            48.0,
        ))
        .max_bounces(6) // maximum bounces for deep infinite reflection
        .ambient(0.03)
        .sky_color(void)
}
