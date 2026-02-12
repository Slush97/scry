//! Rasterization benchmarks for `ratatui-pixelcanvas`.
//!
//! Run with: `cargo bench`

use criterion::{criterion_group, criterion_main, Criterion, black_box};
use ratatui_pixelcanvas::rasterize::Rasterizer;
use ratatui_pixelcanvas::scene::PixelCanvas;
use ratatui_pixelcanvas::scene::style::{Color, Point};

fn simple_circle(c: &mut Criterion) {
    let canvas = PixelCanvas::new(400, 400)
        .background(Color::BLACK)
        .circle(200.0, 200.0, 100.0)
        .fill(Color::RED)
        .stroke(Color::WHITE, 2.0)
        .done();

    c.bench_function("rasterize_circle_400x400", |b| {
        b.iter(|| {
            let _ = Rasterizer::rasterize(black_box(&canvas)).unwrap();
        });
    });
}

fn complex_scene(c: &mut Criterion) {
    let canvas = PixelCanvas::new(800, 600)
        .background(Color::from_rgba8(18, 18, 28, 255))
        // Circles
        .circle(200.0, 200.0, 80.0)
        .fill(Color::from_rgba8(100, 149, 237, 220))
        .stroke(Color::WHITE, 2.0)
        .done()
        .circle(600.0, 200.0, 60.0)
        .fill(Color::from_rgba8(255, 100, 100, 200))
        .done()
        // Rectangle
        .rect(300.0, 100.0, 200.0, 150.0)
        .fill(Color::from_rgba8(60, 179, 113, 200))
        .corner_radius(10.0)
        .done()
        // Ellipse
        .ellipse(400.0, 400.0, 150.0, 60.0)
        .fill(Color::from_rgba8(255, 165, 0, 200))
        .done()
        // Rotated ellipse
        .ellipse(600.0, 400.0, 100.0, 30.0)
        .rotation(0.7)
        .fill(Color::from_rgba8(186, 85, 211, 200))
        .done()
        // Lines
        .line(0.0, 0.0, 800.0, 600.0)
        .stroke(Color::from_rgba8(255, 215, 0, 255), 2.0)
        .done()
        .line(800.0, 0.0, 0.0, 600.0)
        .stroke(Color::from_rgba8(255, 100, 100, 255), 2.0)
        .done()
        // Polyline
        .polyline(vec![
            (50.0, 500.0), (150.0, 350.0), (250.0, 500.0),
            (350.0, 350.0), (450.0, 500.0),
        ])
        .stroke(Color::from_rgba8(0, 255, 255, 255), 2.5)
        .done()
        // Polygon
        .polygon(vec![
            (550.0, 350.0), (500.0, 550.0), (700.0, 550.0),
        ])
        .fill(Color::from_rgba8(255, 99, 71, 200))
        .done()
        // Gradient
        .gradient(50.0, 20.0, 700.0, 40.0)
        .linear(Point::new(50.0, 20.0), Point::new(750.0, 20.0))
        .stop(0.0, Color::from_rgba8(255, 0, 128, 255))
        .stop(0.5, Color::from_rgba8(128, 0, 255, 255))
        .stop(1.0, Color::from_rgba8(0, 128, 255, 255))
        .done();

    c.bench_function("rasterize_complex_800x600", |b| {
        b.iter(|| {
            let _ = Rasterizer::rasterize(black_box(&canvas)).unwrap();
        });
    });
}

fn content_hash(c: &mut Criterion) {
    let canvas = PixelCanvas::new(800, 600)
        .background(Color::BLACK)
        .circle(400.0, 300.0, 200.0)
        .fill(Color::RED)
        .done()
        .rect(100.0, 100.0, 200.0, 150.0)
        .fill(Color::BLUE)
        .done()
        .ellipse(600.0, 400.0, 100.0, 50.0)
        .fill(Color::GREEN)
        .done()
        .polygon(vec![(300.0, 100.0), (200.0, 500.0), (700.0, 500.0)])
        .fill(Color::WHITE)
        .done();

    c.bench_function("content_hash_complex", |b| {
        b.iter(|| {
            black_box(canvas.content_hash());
        });
    });
}

fn resolution_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterize_circle_scaling");
    for size in [100, 200, 400, 800, 1600] {
        let s = size as f32;
        let canvas = PixelCanvas::new(size, size)
            .background(Color::BLACK)
            .circle(s / 2.0, s / 2.0, s * 0.3)
            .fill(Color::RED)
            .done();

        group.bench_function(format!("{size}x{size}"), |b| {
            b.iter(|| {
                let _ = Rasterizer::rasterize(black_box(&canvas)).unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    simple_circle,
    complex_scene,
    content_hash,
    resolution_scaling,
);
criterion_main!(benches);
