//! Efficiency benchmarks for the animation system, hashing strategies,
//! and rasterization pipeline.
//!
//! Run with: `cargo bench --bench efficiency`

#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration as StdDuration;
use std::time::Duration;

use ratatui_pixelcanvas::rasterize::Rasterizer;
use ratatui_pixelcanvas::scene::animation::{
    AnimationState, Easing, Keyframe, Keyframes, Lerp, Transition,
};
use ratatui_pixelcanvas::scene::command::{ImageData, PathData};
use ratatui_pixelcanvas::scene::style::{Color, Point, Transform};
use ratatui_pixelcanvas::scene::PixelCanvas;

// ─────────────────────────────────────────────────────────────────
// 1. Easing curve evaluation
// ─────────────────────────────────────────────────────────────────

fn bench_easing_curves(c: &mut Criterion) {
    let mut group = c.benchmark_group("easing_curves");

    let curves: Vec<(&str, Easing)> = vec![
        ("linear", Easing::Linear),
        ("ease_out_cubic", Easing::EaseOutCubic),
        ("ease_in_out_quart", Easing::EaseInOutQuart),
        ("bounce", Easing::Bounce),
        ("elastic", Easing::Elastic),
        ("spring", Easing::BACK),
        ("cubic_bezier_css_ease", Easing::CSS_EASE),
    ];

    for (name, easing) in &curves {
        group.bench_with_input(
            BenchmarkId::new("single_eval", name),
            easing,
            |b, easing| {
                b.iter(|| black_box(easing.ease(black_box(0.5))));
            },
        );
    }

    // Evaluate all curves across 1000 samples — simulates a full animation
    group.bench_function("all_curves_1000_samples", |b| {
        let all: Vec<Easing> = vec![
            Easing::Linear,
            Easing::EaseInQuad,
            Easing::EaseOutQuad,
            Easing::EaseInOutQuad,
            Easing::EaseInCubic,
            Easing::EaseOutCubic,
            Easing::EaseInOutCubic,
            Easing::EaseInSine,
            Easing::EaseOutSine,
            Easing::EaseInOutSine,
            Easing::EaseInExpo,
            Easing::EaseOutExpo,
            Easing::EaseInOutExpo,
            Easing::EaseInCirc,
            Easing::EaseOutCirc,
            Easing::EaseInOutCirc,
            Easing::Bounce,
            Easing::Elastic,
            Easing::BACK,
            Easing::CSS_EASE,
            Easing::CSS_EASE_IN,
            Easing::CSS_EASE_OUT,
        ];
        b.iter(|| {
            for easing in &all {
                for i in 0..1000 {
                    black_box(easing.ease(i as f32 / 999.0));
                }
            }
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// 2. Lerp performance
// ─────────────────────────────────────────────────────────────────

fn bench_lerp(c: &mut Criterion) {
    let mut group = c.benchmark_group("lerp");

    group.bench_function("f32", |b| {
        b.iter(|| black_box(0.0_f32.lerp(&100.0, black_box(0.5))));
    });

    group.bench_function("color", |b| {
        let a = Color::from_rgba8(255, 0, 0, 255);
        let z = Color::from_rgba8(0, 0, 255, 255);
        b.iter(|| black_box(a.lerp(&z, black_box(0.5))));
    });

    group.bench_function("point", |b| {
        let a = Point::new(0.0, 0.0);
        let z = Point::new(100.0, 200.0);
        b.iter(|| black_box(a.lerp(&z, black_box(0.5))));
    });

    group.bench_function("transform", |b| {
        let a = Transform::identity();
        let z = Transform::translate(100.0, 50.0);
        b.iter(|| black_box(a.lerp(&z, black_box(0.5))));
    });

    // Bulk: 10000 color lerps — simulates gradient rendering
    group.bench_function("color_10k", |b| {
        let a = Color::from_rgba8(255, 0, 0, 255);
        let z = Color::from_rgba8(0, 0, 255, 255);
        b.iter(|| {
            for i in 0..10_000 {
                black_box(a.lerp(&z, i as f32 / 9999.0));
            }
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// 3. Transition tick performance
// ─────────────────────────────────────────────────────────────────

fn bench_transition(c: &mut Criterion) {
    let mut group = c.benchmark_group("transition");

    group.bench_function("advance_f32", |b| {
        let mut t = Transition::new(0.0_f32, 100.0_f32, Duration::from_secs(1))
            .easing(Easing::EaseOutCubic);
        let dt = Duration::from_millis(16);
        b.iter(|| {
            t.advance(black_box(dt));
            black_box(t.value());
        });
    });

    group.bench_function("advance_color", |b| {
        let from = Color::from_rgba8(255, 0, 0, 255);
        let to = Color::from_rgba8(0, 0, 255, 255);
        let mut t =
            Transition::new(from, to, Duration::from_secs(1)).easing(Easing::EaseInOutCubic);
        let dt = Duration::from_millis(16);
        b.iter(|| {
            t.advance(black_box(dt));
            black_box(t.value());
        });
    });

    group.bench_function("advance_point_elastic", |b| {
        let from = Point::new(0.0, 0.0);
        let to = Point::new(800.0, 600.0);
        let mut t = Transition::new(from, to, Duration::from_secs(2)).easing(Easing::Elastic);
        let dt = Duration::from_millis(16);
        b.iter(|| {
            t.advance(black_box(dt));
            black_box(t.value());
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// 4. Keyframe evaluation
// ─────────────────────────────────────────────────────────────────

fn bench_keyframes(c: &mut Criterion) {
    let mut group = c.benchmark_group("keyframes");

    // Simple 3-stop
    let kf3 = Keyframes::new(vec![
        Keyframe {
            position: 0.0,
            value: 0.0_f32,
            easing: Easing::Linear,
        },
        Keyframe {
            position: 0.5,
            value: 100.0,
            easing: Easing::EaseOutCubic,
        },
        Keyframe {
            position: 1.0,
            value: 50.0,
            easing: Easing::Linear,
        },
    ]);

    group.bench_function("3_stops_single_eval", |b| {
        b.iter(|| black_box(kf3.value_at(black_box(0.7))));
    });

    // Complex 10-stop
    let kf10 = Keyframes::new(
        (0..10)
            .map(|i| Keyframe {
                position: i as f32 / 9.0,
                value: Color::from_hsl(360.0 * i as f32 / 9.0, 0.8, 0.5),
                easing: Easing::EaseInOutCubic,
            })
            .collect(),
    );

    group.bench_function("10_stops_color_single_eval", |b| {
        b.iter(|| black_box(kf10.value_at(black_box(0.35))));
    });

    // Dense 50-stop, evaluated 1000 times
    let kf50 = Keyframes::new(
        (0..50)
            .map(|i| Keyframe {
                position: i as f32 / 49.0,
                value: i as f32 * 10.0,
                easing: Easing::EaseOutCubic,
            })
            .collect(),
    );

    group.bench_function("50_stops_1000_evals", |b| {
        b.iter(|| {
            for i in 0..1000 {
                black_box(kf50.value_at(i as f32 / 999.0));
            }
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// 5. AnimationState orchestration
// ─────────────────────────────────────────────────────────────────

fn bench_animation_state(c: &mut Criterion) {
    let mut group = c.benchmark_group("animation_state");
    let dt = Duration::from_millis(16);

    group.bench_function("tick_5_animations", |b| {
        let mut state = AnimationState::new();
        b.iter(|| {
            // Reset 5 animations each iteration to keep them alive
            if state.is_idle() {
                state.start("a", 0.0_f32, 1.0, Duration::from_secs(1), Easing::Linear);
                state.start(
                    "b",
                    0.0_f32,
                    1.0,
                    Duration::from_secs(2),
                    Easing::EaseOutCubic,
                );
                state.start("c", 0.0_f32, 1.0, Duration::from_secs(3), Easing::Bounce);
                state.start(
                    "d",
                    Point::new(0.0, 0.0),
                    Point::new(100.0, 100.0),
                    Duration::from_secs(2),
                    Easing::Elastic,
                );
                state.start(
                    "e",
                    Color::RED,
                    Color::BLUE,
                    Duration::from_secs(1),
                    Easing::CSS_EASE,
                );
            }
            state.tick(black_box(dt));
            black_box(state.get::<f32>("a"));
            black_box(state.get::<f32>("b"));
            black_box(state.get::<f32>("c"));
            black_box(state.get::<Point>("d"));
            black_box(state.get::<Color>("e"));
        });
    });

    group.bench_function("tick_20_animations", |b| {
        let mut state = AnimationState::new();
        b.iter(|| {
            if state.is_idle() {
                for i in 0..20 {
                    let names: [&str; 20] = [
                        "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "b0", "b1",
                        "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9",
                    ];
                    state.start(
                        names[i],
                        0.0_f32,
                        1.0,
                        Duration::from_millis(500 + i as u64 * 200),
                        Easing::EaseOutCubic,
                    );
                }
            }
            state.tick(black_box(dt));
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// 6. Content hashing — lazy PathData vs sampled ImageData
// ─────────────────────────────────────────────────────────────────

fn bench_hashing(c: &mut Criterion) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut group = c.benchmark_group("content_hashing");

    // PathData — lazy hash (first call computes, subsequent calls are cached)
    let pb = tiny_skia::PathBuilder::from_circle(200.0, 200.0, 100.0).unwrap();
    let path_data = PathData::new(pb);

    group.bench_function("path_data_hash_cold", |b| {
        b.iter_batched(
            || {
                let pb = tiny_skia::PathBuilder::from_circle(200.0, 200.0, 100.0).unwrap();
                PathData::new(pb)
            },
            |pd| {
                let mut h = DefaultHasher::new();
                pd.hash(&mut h);
                black_box(h.finish())
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("path_data_hash_warm", |b| {
        // Prime the lazy cache
        let mut h = DefaultHasher::new();
        path_data.hash(&mut h);
        b.iter(|| {
            let mut h = DefaultHasher::new();
            path_data.hash(&mut h);
            black_box(h.finish())
        });
    });

    // ImageData — sampled hash at different sizes
    for &size in &[64u32, 256, 1024, 4096] {
        let pixels = size * size;
        let data: Vec<u8> = (0..pixels * 4).map(|i| (i % 256) as u8).collect();
        let img = ImageData::new(size, size, data);

        group.bench_function(
            BenchmarkId::new("image_data_hash", format!("{size}x{size}")),
            |b| {
                b.iter(|| {
                    let mut h = DefaultHasher::new();
                    img.hash(&mut h);
                    black_box(h.finish())
                });
            },
        );
    }

    // Full scene content_hash
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
        .done();

    group.bench_function("scene_content_hash_5_cmds", |b| {
        b.iter(|| black_box(canvas.content_hash()));
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// 7. Rasterize vs rasterize_into (pixmap reuse)
// ─────────────────────────────────────────────────────────────────

fn bench_rasterize_into(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterize_reuse");

    for &size in &[200u32, 400] {
        let s = size as f32;
        let canvas = PixelCanvas::new(size, size)
            .background(Color::from_rgba8(18, 18, 28, 255))
            .circle(s / 2.0, s / 2.0, s * 0.3)
            .fill(Color::from_rgba8(100, 149, 237, 220))
            .stroke(Color::WHITE, 2.0)
            .done()
            .rect(s * 0.1, s * 0.1, s * 0.3, s * 0.3)
            .fill(Color::from_rgba8(60, 179, 113, 200))
            .corner_radius(8.0)
            .done()
            .ellipse(s * 0.7, s * 0.7, s * 0.15, s * 0.08)
            .fill(Color::from_rgba8(255, 165, 0, 200))
            .done();

        // Baseline: allocate a new Pixmap each time
        group.bench_with_input(
            BenchmarkId::new("rasterize_alloc", format!("{size}x{size}")),
            &canvas,
            |b, canvas| {
                b.iter(|| {
                    let _ = Rasterizer::rasterize(black_box(canvas)).unwrap();
                });
            },
        );

        // Optimized: reuse an existing Pixmap
        group.bench_with_input(
            BenchmarkId::new("rasterize_reuse", format!("{size}x{size}")),
            &canvas,
            |b, canvas| {
                let mut pixmap = tiny_skia::Pixmap::new(size, size).unwrap();
                b.iter(|| {
                    Rasterizer::rasterize_into(black_box(canvas), &mut pixmap);
                });
            },
        );
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// 8. Transform math
// ─────────────────────────────────────────────────────────────────

fn bench_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("transform_math");

    let t = Transform::identity()
        .concat(Transform::translate(100.0, 50.0))
        .concat(Transform::rotate(0.7))
        .concat(Transform::scale(2.0));

    group.bench_function("determinant", |b| {
        b.iter(|| black_box(black_box(t).determinant()));
    });

    group.bench_function("inverse", |b| {
        b.iter(|| black_box(black_box(t).inverse()));
    });

    group.bench_function("apply_point", |b| {
        let p = Point::new(50.0, 75.0);
        b.iter(|| black_box(black_box(t).apply_point(black_box(p))));
    });

    group.bench_function("concat_chain_10", |b| {
        b.iter(|| {
            let mut result = Transform::identity();
            for i in 0..10 {
                result = result.concat(Transform::rotate(i as f32 * 0.1));
            }
            black_box(result)
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// 9. Color utilities
// ─────────────────────────────────────────────────────────────────

fn bench_color_utils(c: &mut Criterion) {
    let mut group = c.benchmark_group("color_utils");

    group.bench_function("from_hsla", |b| {
        b.iter(|| black_box(Color::from_hsla(black_box(210.0), 0.8, 0.5, 1.0)));
    });

    group.bench_function("mix", |b| {
        let a = Color::RED;
        let z = Color::BLUE;
        b.iter(|| black_box(black_box(a).mix(black_box(z), 0.5)));
    });

    group.bench_function("with_lightness", |b| {
        let c = Color::from_rgba8(100, 149, 237, 255);
        b.iter(|| black_box(black_box(c).with_lightness(1.2)));
    });

    // Bulk: 1000 HSL conversions — simulates color wheel generation
    group.bench_function("from_hsl_1000", |b| {
        b.iter(|| {
            for i in 0..1000 {
                black_box(Color::from_hsl(360.0 * i as f32 / 999.0, 0.8, 0.5));
            }
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────

fn fast_config() -> Criterion {
    Criterion::default()
        .warm_up_time(StdDuration::from_secs(1))
        .measurement_time(StdDuration::from_secs(2))
        .sample_size(50)
}

criterion_group! {
    name = benches;
    config = fast_config();
    targets =
        bench_easing_curves,
        bench_lerp,
        bench_transition,
        bench_keyframes,
        bench_animation_state,
        bench_hashing,
        bench_rasterize_into,
        bench_transform,
        bench_color_utils,
}
criterion_main!(benches);
