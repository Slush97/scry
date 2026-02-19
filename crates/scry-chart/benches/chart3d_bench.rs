//! Criterion benchmarks for Chart3D rendering performance.
//!
//! Measures render time for 1K / 5K / 10K point scatter plots at various
//! resolutions. Sprint 8.5D targets:
//!
//! | Scenario | Target |
//! |----------|--------|
//! | 1K pts, 800×600 | <16.7ms (60 fps) |
//! | 5K pts, 1920×1080 | <33.3ms (30 fps) |
//! | 10K pts, 1920×1080 | <66.7ms (15 fps) |

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use scry_chart::chart3d::Chart3D;

/// Generate deterministic random-ish data (no external RNG needed for benchmarks).
fn gen_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);

    // Simple LCG for reproducible pseudo-random data
    let mut seed: u64 = 42;
    for _ in 0..n {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let v1 = ((seed >> 33) as f64) / (u32::MAX as f64);
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let v2 = ((seed >> 33) as f64) / (u32::MAX as f64);
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let v3 = ((seed >> 33) as f64) / (u32::MAX as f64);
        x.push(v1 * 10.0);
        y.push(v2 * 10.0);
        z.push(v3 * 10.0);
    }
    (x, y, z)
}

fn bench_render_1k_800x600(c: &mut Criterion) {
    let (x, y, z) = gen_data(1_000);
    let chart = Chart3D::scatter(&x, &y, &z).title("1K Scatter");

    c.bench_function("chart3d_render_1k_800x600", |b| {
        b.iter(|| {
            let _ = black_box(chart.render(800, 600));
        });
    });
}

fn bench_render_5k_1080p(c: &mut Criterion) {
    let (x, y, z) = gen_data(5_000);
    let chart = Chart3D::scatter(&x, &y, &z).title("5K Scatter");

    c.bench_function("chart3d_render_5k_1080p", |b| {
        b.iter(|| {
            let _ = black_box(chart.render(1920, 1080));
        });
    });
}

fn bench_render_10k_1080p(c: &mut Criterion) {
    let (x, y, z) = gen_data(10_000);
    let chart = Chart3D::scatter(&x, &y, &z).title("10K Scatter");

    c.bench_function("chart3d_render_10k_1080p", |b| {
        b.iter(|| {
            let _ = black_box(chart.render(1920, 1080));
        });
    });
}

fn bench_render_cpu_50k_1080p(c: &mut Criterion) {
    let (x, y, z) = gen_data(50_000);
    let chart = Chart3D::scatter(&x, &y, &z).title("50K CPU");

    c.bench_function("chart3d_render_cpu_50k_1080p", |b| {
        b.iter(|| {
            let _ = black_box(chart.render(1920, 1080));
        });
    });
}

fn bench_render_cpu_100k_1080p(c: &mut Criterion) {
    let (x, y, z) = gen_data(100_000);
    let chart = Chart3D::scatter(&x, &y, &z).title("100K CPU");

    c.bench_function("chart3d_render_cpu_100k_1080p", |b| {
        b.iter(|| {
            let _ = black_box(chart.render(1920, 1080));
        });
    });
}

#[cfg(feature = "gpu")]
fn bench_render_gpu_50k_1080p(c: &mut Criterion) {
    let (x, y, z) = gen_data(50_000);
    let chart = Chart3D::scatter(&x, &y, &z).title("50K GPU");

    c.bench_function("chart3d_render_gpu_50k_1080p", |b| {
        b.iter(|| {
            let _ = black_box(chart.render_gpu(1920, 1080));
        });
    });
}

#[cfg(feature = "gpu")]
fn bench_render_gpu_100k_1080p(c: &mut Criterion) {
    let (x, y, z) = gen_data(100_000);
    let chart = Chart3D::scatter(&x, &y, &z).title("100K GPU");

    c.bench_function("chart3d_render_gpu_100k_1080p", |b| {
        b.iter(|| {
            let _ = black_box(chart.render_gpu(1920, 1080));
        });
    });
}

#[cfg(feature = "gpu")]
fn bench_render_gpu_cached_50k_1080p(c: &mut Criterion) {
    use scry_engine::gpu::GpuDevice;

    let (x, y, z) = gen_data(50_000);
    let chart = Chart3D::scatter(&x, &y, &z).title("50K GPU Cached");
    let gpu = GpuDevice::global().expect("GpuDevice init");

    c.bench_function("chart3d_render_gpu_cached_50k_1080p", |b| {
        b.iter(|| {
            let _ = black_box(chart.render_gpu_with_device(gpu, 1920, 1080));
        });
    });
}

#[cfg(feature = "gpu")]
fn bench_render_gpu_cached_100k_1080p(c: &mut Criterion) {
    use scry_engine::gpu::GpuDevice;

    let (x, y, z) = gen_data(100_000);
    let chart = Chart3D::scatter(&x, &y, &z).title("100K GPU Cached");
    let gpu = GpuDevice::global().expect("GpuDevice init");

    c.bench_function("chart3d_render_gpu_cached_100k_1080p", |b| {
        b.iter(|| {
            let _ = black_box(chart.render_gpu_with_device(gpu, 1920, 1080));
        });
    });
}

#[cfg(not(feature = "gpu"))]
criterion_group!(
    benches,
    bench_render_1k_800x600,
    bench_render_5k_1080p,
    bench_render_10k_1080p,
    bench_render_cpu_50k_1080p,
    bench_render_cpu_100k_1080p,
);

#[cfg(feature = "gpu")]
criterion_group!(
    benches,
    bench_render_1k_800x600,
    bench_render_5k_1080p,
    bench_render_10k_1080p,
    bench_render_cpu_50k_1080p,
    bench_render_cpu_100k_1080p,
    bench_render_gpu_50k_1080p,
    bench_render_gpu_100k_1080p,
    bench_render_gpu_cached_50k_1080p,
    bench_render_gpu_cached_100k_1080p,
);

criterion_main!(benches);
