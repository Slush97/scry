// Quick test: render the masonic_mirror scene to a tiny pixmap and check
use scry_engine::sdf::pipeline::SdfPipeline;
use scry_engine::sdf::{Material, SdfCamera, SdfLight, SdfObject, SdfScene, SdfShape, Vec3};
use scry_engine::style::Color;

fn build_sdf_scene(time: f32) -> SdfScene {
    let angle = time * 0.15;
    let cam_radius = 6.0;
    let cam_height = 3.0;
    let disc_angle = time * 1.2;
    let hue_spin = time * 0.8;
    SdfScene::new()
        .object(SdfObject::new(
            SdfShape::Plane,
            Material::Checkerboard {
                color_a: Color::from_rgba8(20, 20, 25, 255),
                color_b: Color::from_rgba8(180, 180, 190, 255),
                scale: 0.8,
                reflectivity: 0.3,
                specular: 32.0,
            },
        ))
        .object(
            SdfObject::new(
                SdfShape::Sphere { radius: 1.2 },
                Material::glass_dispersive(Color::from_rgba8(230, 230, 255, 255), 1.45, 0.04),
            )
            .at(Vec3::new(0.0, 1.2, 0.0)),
        )
        .object(
            SdfObject::new(
                SdfShape::Cylinder {
                    radius: 0.95,
                    half_height: 0.015,
                },
                Material::rainbow_animated(hue_spin),
            )
            .at(Vec3::new(0.0, 1.2, 0.0))
            .rotate_y(disc_angle),
        )
        .object(
            SdfObject::new(
                SdfShape::Cylinder {
                    radius: 0.85,
                    half_height: 0.012,
                },
                Material::Rainbow {
                    saturation: 0.85,
                    lightness: 0.45,
                    hue_offset: hue_spin + std::f32::consts::FRAC_PI_2,
                    specular: 48.0,
                },
            )
            .at(Vec3::new(0.0, 1.2, 0.0))
            .rotate_y(disc_angle + std::f32::consts::FRAC_PI_2),
        )
        .object(
            SdfObject::new(
                SdfShape::Cylinder {
                    radius: 0.25,
                    half_height: 2.0,
                },
                Material::Solid {
                    color: Color::from_rgba8(180, 160, 120, 255),
                    reflectivity: 0.15,
                    specular: 16.0,
                },
            )
            .at(Vec3::new(-2.5, 2.0, 0.0)),
        )
        .object(
            SdfObject::new(
                SdfShape::Cylinder {
                    radius: 0.25,
                    half_height: 2.0,
                },
                Material::Solid {
                    color: Color::from_rgba8(60, 60, 80, 255),
                    reflectivity: 0.15,
                    specular: 16.0,
                },
            )
            .at(Vec3::new(2.5, 2.0, 0.0)),
        )
        .light(SdfLight::new(
            Vec3::new(0.0, 10.0, 3.0),
            Color::from_rgba8(255, 240, 200, 255),
            0.9,
        ))
        .light(SdfLight::new(
            Vec3::new(-4.0, 5.0, -2.0),
            Color::from_rgba8(100, 120, 200, 255),
            0.3,
        ))
        .camera(SdfCamera::new(
            Vec3::new(
                angle.cos() * cam_radius,
                cam_height,
                angle.sin() * cam_radius,
            ),
            Vec3::new(0.0, 1.0, 0.0),
            50.0,
        ))
        .max_bounces(3)
        .sky_color(Color::from_rgba8(5, 5, 15, 255))
}

fn main() {
    let scene = build_sdf_scene(0.0);
    eprintln!(
        "Scene: {} objects, {} lights, has_glass={}, has_water={}, max_bounces={}",
        scene.objects.len(),
        scene.lights.len(),
        scene.has_glass,
        scene.has_water,
        scene.max_bounces
    );

    // Test CPU render at small resolution
    let start = std::time::Instant::now();
    let mut pipeline = SdfPipeline::cpu_only();
    let result = pipeline.render(&scene, 64, 48, 0.0);
    let elapsed = start.elapsed();
    eprintln!(
        "CPU 64x48 render: {:?}, backend={:?}",
        elapsed, result.backend
    );

    let data = result.image.data();
    let total_pixels = 64 * 48;
    let mut non_black = 0;
    let mut non_skycolor = 0;
    let mut max_r = 0u8;
    let mut max_g = 0u8;
    let mut max_b = 0u8;
    for i in 0..total_pixels {
        let r = data[i * 4];
        let g = data[i * 4 + 1];
        let b = data[i * 4 + 2];
        if r > 0 || g > 0 || b > 0 {
            non_black += 1;
        }
        // Sky color gamma-encoded: ~28, ~28, ~47 (from 5/255, 5/255, 15/255)
        if r > 30 || g > 30 || b > 50 {
            non_skycolor += 1;
        }
        max_r = max_r.max(r);
        max_g = max_g.max(g);
        max_b = max_b.max(b);
    }
    eprintln!("Non-black pixels: {non_black}/{total_pixels}");
    eprintln!("Non-sky pixels: {non_skycolor}/{total_pixels}");
    eprintln!("Max RGB: ({max_r}, {max_g}, {max_b})");

    if non_skycolor == 0 {
        eprintln!("ERROR: All pixels are sky color or darker — scene appears blank!");
        std::process::exit(1);
    }

    // Test at larger resolution
    let start = std::time::Instant::now();
    let result2 = pipeline.render(&scene, 320, 240, 0.0);
    let elapsed2 = start.elapsed();
    eprintln!("CPU 320x240 render: {:?}", elapsed2);

    let data2 = result2.image.data();
    let mut non_skycolor2 = 0;
    for i in 0..(320 * 240) {
        let r = data2[i * 4];
        let g = data2[i * 4 + 1];
        let b = data2[i * 4 + 2];
        if r > 30 || g > 30 || b > 50 {
            non_skycolor2 += 1;
        }
    }
    eprintln!("320x240 non-sky pixels: {non_skycolor2}/{}", 320 * 240);

    // Test GPU pipeline if available
    let mut gpu_pipeline = SdfPipeline::new();
    eprintln!("GPU pipeline backend: {}", gpu_pipeline.backend_name());
    let start = std::time::Instant::now();
    let gpu_result = gpu_pipeline.render(&scene, 64, 48, 0.0);
    let elapsed3 = start.elapsed();
    eprintln!(
        "Pipeline 64x48 render (frame 1): {:?}, backend={:?}",
        elapsed3, gpu_result.backend
    );

    // Frame 2 should use GPU if available
    let start = std::time::Instant::now();
    let gpu_result2 = gpu_pipeline.render(&scene, 64, 48, 0.1);
    let elapsed4 = start.elapsed();
    eprintln!(
        "Pipeline 64x48 render (frame 2): {:?}, backend={:?}",
        elapsed4, gpu_result2.backend
    );

    // Check GPU result
    let gdata = gpu_result2.image.data();
    let mut gpu_non_sky = 0;
    for i in 0..total_pixels {
        let r = gdata[i * 4];
        let g = gdata[i * 4 + 1];
        let b = gdata[i * 4 + 2];
        if r > 30 || g > 30 || b > 50 {
            gpu_non_sky += 1;
        }
    }
    eprintln!("GPU frame 2 non-sky pixels: {gpu_non_sky}/{total_pixels}");

    eprintln!("\nAll checks passed.");
}
