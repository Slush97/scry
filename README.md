# scry-engine

**A vector graphics engine for the terminal.**

Anti-aliased shapes, paths, gradients, and text — rasterized with `tiny-skia`,
then shipped to the screen via Kitty, Sixel, iTerm2, or Unicode halfblocks.
Each layer works on its own: draw without a terminal, transmit without Ratatui,
or plug in as a Ratatui widget with caching and dirty-tile diffing.

## Features

- Builder API — `canvas.circle(50, 50, 30).fill(Color::RED).done()`
- Shapes — circles, rectangles, ellipses, lines, polylines, polygons, arcs, paths, groups
- Gradients — linear and radial with arbitrary color stops
- Full RGBA — alpha compositing and blend modes
- Protocols — Kitty (zlib, SHM zero-copy), Sixel (median-cut 256-color), iTerm2 (inline PNG), halfblock fallback
- Auto-detection — `Picker` probes the terminal and picks the best backend
- Ratatui widget — `StatefulWidget` with content-hash caching and incremental tile updates
- Animation — 20+ easing curves, keyframe timelines, Oklab color interpolation
- Text — glyph rasterization via fontdue (optional)
- SVG — parse and render via resvg, with line-drawing animation (optional)

## Workspace

| Crate | What it does |
|-------|--------------|
| [`scry-engine`](.) | Drawing, rasterization, and terminal transport |
| [`scry-chart`](crates/scry-chart) | 18 chart types, 3D interactive, 6 themes, streaming, PNG/SVG export |
| [`scry-cli`](crates/scry-cli) | CLI — render charts from JSON/CSV, show images inline |
| [`scry-learn`](crates/scry-learn) | ML — CART, Random Forest, Gradient Boosting, Linear/Logistic Regression, Lasso, ElasticNet, KNN, Naive Bayes, SVM, K-Means, DBSCAN, preprocessing, cross-validation |
| [`scry-pipe`](crates/scry-pipe) | Feature pipeline IR + codegen |

## Quick start

```rust
use scry_engine::prelude::*;

let canvas = PixelCanvas::new(200, 200)
    .background(Color::from_rgba8(20, 20, 30, 255))
    .circle(100.0, 100.0, 60.0)
        .fill(Color::from_rgba8(70, 130, 180, 255))
        .stroke(Color::from_rgba8(255, 255, 255, 200), 2.0)
        .done()
    .line(20.0, 180.0, 180.0, 20.0)
        .color(Color::from_rgba8(255, 100, 100, 255))
        .width(3.0)
        .done();

// Ratatui widget (Layer 3)
frame.render_stateful_widget(
    PixelCanvasWidget::new(canvas),
    area,
    &mut pixel_canvas_state,
);
```

## Architecture

```
┌──────────────────────────────────────────────────┐
│              Your app / TUI                      │
├──────────────────────────────────────────────────┤
│  3. Widget    Ratatui StatefulWidget, caching    │
├──────────────────────────────────────────────────┤
│  2. Transport Kitty / Sixel / iTerm2 / Halfblock │
├──────────────────────────────────────────────────┤
│  1. Drawing   Scene → tiny-skia → Pixmap         │
│     SVG       resvg → Pixmap                     │
└──────────────────────────────────────────────────┘
```

Each layer is independent. You can draw without a terminal, transmit without
Ratatui, or use all three together.

## 📊 scry-chart

The `scry-chart` crate provides a high-level charting API on top of `scry-engine`:

- **16 2D chart types** — scatter, line, bar, histogram, box plot, heatmap, pie, radar, candlestick, bubble, violin, sparkline, waterfall, funnel, gauge, lollipop
- **3D scatter** — interactive rotation/zoom in the terminal
- **Streaming charts** — ring-buffer backed, auto-scrolling time axis
- **Interactive features** — zoom, pan, crosshair cursor, and tooltips
- **Export** — PNG and SVG output
- **Themes** — `dark`, `light`, `ocean`, `forest`, `pastel`, and colorblind-safe themes

## 🎨 Examples

The engine ships with 30+ examples. A few highlights:

```bash
# Basic shapes and gradients
cargo run --example simple_shapes
cargo run --example new_features

# Animations
cargo run --example animation_demo
cargo run --example sacred_geometry
cargo run --example wave_interference
cargo run --example fluid_symphony

# 3D rendering
cargo run --example cube_3d

# SVG line drawing (requires svg feature)
cargo run --example line_drawing --features svg

# scry-chart showcase
cargo run -p scry-chart --example showcase
```

See all examples with `ls examples/`.

## ⚙️ Feature Flags

| Flag      | Default | Description                                      |
|-----------|---------|--------------------------------------------------|
| `kitty`   | ✅      | Kitty graphics protocol backend                  |
| `widget`  | ✅      | Ratatui `StatefulWidget` integration              |
| `gpu`     | ✅      | wgpu GPU-accelerated rasterization                |
| `sixel`   | ❌      | Sixel graphics protocol backend                  |
| `iterm2`  | ❌      | iTerm2 inline image protocol backend              |
| `text`    | ❌      | Text rendering via `fontdue`                      |
| `shm`     | ❌      | Zero-copy Kitty transmission via POSIX shared mem |
| `svg`     | ❌      | SVG rendering via `resvg`                         |
| `wasm`    | ❌      | WebAssembly canvas bridge                         |
| `sdf`     | ❌      | SDF shape rendering                               |
| `window`  | ❌      | Standalone window transport                       |
| `serde`   | ❌      | Serialize/Deserialize derives                     |

## 🔧 Minimum Supported Rust Version

1.83.0

## Crate Maturity

| Crate | Maturity | Breaking Changes |
|-------|----------|-----------------|
| scry-engine | Stable | Semver-protected |
| scry-chart | Stable | Semver-protected |
| scry-learn | Beta | API may evolve |
| scry-cli | Beta | Commands may change |
| scry-pipe | Beta | Phase 1 shipped, API may evolve |

## 📄 License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT License](LICENSE-MIT) at your option.
