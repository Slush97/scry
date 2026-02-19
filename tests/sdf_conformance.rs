// SPDX-License-Identifier: MIT OR Apache-2.0
//! SDF conformance tests — compare CPU pipeline output for consistency.
//!
//! When the `sdf-gpu` feature is enabled AND a GPU is available, these tests
//! also compare GPU vs CPU output to ensure visual equivalence within an
//! RMSE threshold.
#![cfg(feature = "sdf")]

use scry_engine::sdf::pipeline::{SdfBackend, SdfPipeline};
use scry_engine::sdf::*;
use scry_engine::scene::style::Color;

// ── Helpers ────────────────────────────────────────────────────────

/// Compute the per-channel RMSE between two RGBA byte slices.
fn rgba_rmse(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "buffers must be the same length");
    if a.is_empty() {
        return 0.0;
    }
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = (x as f64) - (y as f64);
            d * d
        })
        .sum();
    (sum / a.len() as f64).sqrt()
}

fn test_scene() -> SdfScene {
    SdfScene::new()
        .object(
            SdfObject::new(
                SdfShape::Sphere { radius: 1.0 },
                Material::Solid {
                    color: Color::RED,
                    reflectivity: 0.3,
                    specular: 0.5,
                },
            )
            .at(Vec3::new(0.0, 1.0, 0.0)),
        )
        .object(SdfObject::new(
            SdfShape::Plane,
            Material::Checkerboard {
                color_a: Color::WHITE,
                color_b: Color::from_rgba8(80, 80, 80, 255),
                scale: 1.0,
                reflectivity: 0.1,
                specular: 0.2,
            },
        ))
        .light(SdfLight::new(
            Vec3::new(5.0, 10.0, 5.0),
            Color::WHITE,
            1.0,
        ))
        .camera(SdfCamera::new(
            Vec3::new(0.0, 3.0, 6.0),
            Vec3::ZERO,
            45.0,
        ))
}

fn multi_shape_scene() -> SdfScene {
    SdfScene::new()
        .object(
            SdfObject::new(
                SdfShape::Sphere { radius: 0.6 },
                Material::matte(Color::BLUE),
            )
            .at(Vec3::new(-1.5, 0.6, 0.0)),
        )
        .object(
            SdfObject::new(
                SdfShape::Box {
                    half_extents: Vec3::new(0.5, 0.5, 0.5),
                },
                Material::matte(Color::GREEN),
            )
            .at(Vec3::new(1.5, 0.5, 0.0)),
        )
        .object(SdfObject::new(
            SdfShape::Plane,
            Material::matte(Color::from_rgba8(200, 200, 200, 255)),
        ))
        .light(SdfLight::new(
            Vec3::new(3.0, 8.0, 4.0),
            Color::WHITE,
            1.0,
        ))
        .camera(SdfCamera::new(
            Vec3::new(0.0, 3.0, 6.0),
            Vec3::ZERO,
            45.0,
        ))
}

// ── CPU determinism tests ──────────────────────────────────────────

#[test]
fn cpu_render_is_deterministic() {
    let scene = test_scene();
    let w = 100;
    let h = 75;

    let pm1 = SdfRenderer::render_to_pixmap(&scene, w, h, 0.0).expect("render 1");
    let pm2 = SdfRenderer::render_to_pixmap(&scene, w, h, 0.0).expect("render 2");

    assert_eq!(
        pm1.data(),
        pm2.data(),
        "CPU renders of the same scene should be bit-identical"
    );
}

#[test]
fn cpu_pipeline_matches_direct_renderer() {
    let scene = test_scene();
    let w = 80;
    let h = 60;

    let direct = SdfRenderer::render_to_pixmap(&scene, w, h, 0.0).expect("direct render");

    let mut pipeline = SdfPipeline::cpu_only();
    let result = pipeline.render_sync(&scene, w, h, 0.0);

    assert_eq!(result.backend, SdfBackend::Cpu);
    assert_eq!(result.width, w);
    assert_eq!(result.height, h);

    let rmse = rgba_rmse(direct.data(), result.image.data());
    assert!(
        rmse < 0.01,
        "CPU pipeline should match direct renderer exactly, got RMSE={rmse:.4}"
    );
}

#[test]
fn cpu_multi_shape_consistency() {
    let scene = multi_shape_scene();
    let w = 120;
    let h = 90;

    let pm1 = SdfRenderer::render_to_pixmap(&scene, w, h, 0.0).expect("render 1");
    let pm2 = SdfRenderer::render_to_pixmap(&scene, w, h, 0.0).expect("render 2");

    assert_eq!(
        pm1.data(),
        pm2.data(),
        "Multi-shape CPU renders should be bit-identical"
    );
}

#[test]
fn cpu_different_camera_produces_different_output() {
    // Same scene rendered from two different camera positions
    let base = SdfScene::new()
        .object(
            SdfObject::new(SdfShape::Sphere { radius: 1.0 }, Material::matte(Color::RED))
                .at(Vec3::new(0.0, 1.0, 0.0)),
        )
        .object(SdfObject::new(SdfShape::Plane, Material::matte(Color::WHITE)))
        .light(SdfLight::new(Vec3::new(5.0, 10.0, 5.0), Color::WHITE, 1.0));

    let scene_a = base
        .clone()
        .camera(SdfCamera::new(Vec3::new(0.0, 3.0, 6.0), Vec3::ZERO, 45.0));
    let scene_b = base
        .clone()
        .camera(SdfCamera::new(Vec3::new(4.0, 3.0, 4.0), Vec3::ZERO, 45.0));

    let pm_a = SdfRenderer::render_to_pixmap(&scene_a, 100, 75, 0.0).expect("cam A");
    let pm_b = SdfRenderer::render_to_pixmap(&scene_b, 100, 75, 0.0).expect("cam B");

    assert_ne!(
        pm_a.data(),
        pm_b.data(),
        "Different camera positions should produce different output"
    );
}

// ── Pipeline API tests ─────────────────────────────────────────────

#[test]
fn pipeline_render_scale_produces_correct_dimensions() {
    let mut pipeline = SdfPipeline::cpu_only().render_scale(0.5);
    let scene = test_scene();
    let result = pipeline.render_sync(&scene, 200, 150, 0.0);
    // Output should be full requested size (upscaled from 50% render)
    assert_eq!(result.width, 200);
    assert_eq!(result.height, 150);
}

#[test]
fn pipeline_handles_zero_dimensions() {
    let mut pipeline = SdfPipeline::cpu_only();
    let scene = test_scene();
    let result = pipeline.render(&scene, 0, 0, 0.0);
    // Should not panic, returns a minimal image
    assert!(result.width >= 1);
    assert!(result.height >= 1);
}

// ── GPU vs CPU conformance (only when GPU is available) ────────────

#[cfg(feature = "sdf-gpu")]
mod gpu_conformance {
    use super::*;

    /// Compare GPU and CPU output for the same scene.
    ///
    /// RMSE threshold of 5.0 per channel allows for differences in
    /// floating-point precision between GPU (often FP16/FP32 with less
    /// precise built-in math) and CPU (full FP32/FP64 depending on LLVM).
    const RMSE_THRESHOLD: f64 = 5.0;

    fn gpu_available() -> bool {
        scry_engine::gpu::GpuDevice::is_available()
    }

    #[test]
    fn gpu_cpu_sphere_conformance() {
        if !gpu_available() {
            eprintln!("skipping GPU conformance test: no GPU available");
            return;
        }

        let scene = test_scene();
        let w = 100;
        let h = 75;

        // CPU reference
        let cpu_pm = SdfRenderer::render_to_pixmap(&scene, w, h, 0.0).expect("CPU render");

        // GPU render
        let mut pipeline = SdfPipeline::new();
        let gpu_result = pipeline.render_sync(&scene, w, h, 0.0);

        if gpu_result.backend == SdfBackend::Cpu {
            eprintln!("GPU pipeline fell back to CPU, skipping conformance comparison");
            return;
        }

        let rmse = rgba_rmse(cpu_pm.data(), gpu_result.image.data());
        assert!(
            rmse < RMSE_THRESHOLD,
            "GPU vs CPU RMSE={rmse:.2} exceeds threshold {RMSE_THRESHOLD} — \
             output diverges beyond acceptable floating-point tolerance"
        );
        eprintln!("✓ GPU vs CPU sphere scene RMSE = {rmse:.2} (threshold: {RMSE_THRESHOLD})");
    }

    #[test]
    fn gpu_cpu_multi_shape_conformance() {
        if !gpu_available() {
            eprintln!("skipping GPU conformance test: no GPU available");
            return;
        }

        let scene = multi_shape_scene();
        let w = 120;
        let h = 90;

        let cpu_pm = SdfRenderer::render_to_pixmap(&scene, w, h, 0.0).expect("CPU render");

        let mut pipeline = SdfPipeline::new();
        let gpu_result = pipeline.render_sync(&scene, w, h, 0.0);

        if gpu_result.backend == SdfBackend::Cpu {
            eprintln!("GPU pipeline fell back to CPU, skipping conformance comparison");
            return;
        }

        let rmse = rgba_rmse(cpu_pm.data(), gpu_result.image.data());
        assert!(
            rmse < RMSE_THRESHOLD,
            "GPU vs CPU multi-shape RMSE={rmse:.2} exceeds threshold {RMSE_THRESHOLD}"
        );
        eprintln!(
            "✓ GPU vs CPU multi-shape scene RMSE = {rmse:.2} (threshold: {RMSE_THRESHOLD})"
        );
    }
}
