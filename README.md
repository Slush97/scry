# ratatui-pixelcanvas

**Pixel-perfect vector graphics for [Ratatui](https://ratatui.rs) via Kitty, Sixel, and Unicode fallbacks.**

Draw anti-aliased circles, lines, paths, gradients, and complex shapes in your TUI —
rendered as actual pixels when the terminal supports it, with graceful degradation to
text-based drawing when it doesn't.

## ✨ Features

- **Fluent drawing API** — `canvas.circle(50, 50, 30).fill(Color::RED).done()`
- **Rich primitives** — circles, rectangles, ellipses, lines, polylines, polygons, arcs, paths, and groups
- **Gradient fills** — linear and radial gradients with arbitrary color stops
- **Alpha compositing** — full RGBA transparency and blend modes
- **Kitty graphics protocol** — pixel-perfect rendering with transparency and zlib compression
- **POSIX shared memory** — optional zero-copy Kitty transmission via `shm` feature
- **Text rendering** — optional glyph rasterization via `fontdue` (`text` feature)
- **Automatic fallback** — Kitty → Sixel → Unicode halfblocks
- **Ratatui `StatefulWidget`** — integrates seamlessly into any Ratatui layout
- **Content-hash caching** — skips redundant re-renders when the scene hasn't changed
- **Layered architecture** — use the drawing API standalone, without Ratatui or any terminal protocol

## 📦 Workspace

This is a Cargo workspace containing:

| Crate | Description |
|-------|-------------|
| [`ratatui-pixelcanvas`](.) | Core drawing engine, rasterizer, and terminal backends |
| [`pixelchart`](crates/pixelchart) | High-level charting library — scatter, line, bar, histogram, boxplot, heatmap |

## 🚀 Quick Start

```rust
use ratatui_pixelcanvas::prelude::*;

// Build a scene
let canvas = PixelCanvas::new(200, 200)
    .background(Color::from_rgba8(20, 20, 30, 255))
    .circle(100.0, 100.0, 60.0)
        .fill(Color::from_rgba8(70, 130, 180, 255))
        .stroke(Color::from_rgba8(255, 255, 255, 200), 2.0)
        .done()
    .line(20.0, 180.0, 180.0, 20.0)
        .stroke_color(Color::from_rgba8(255, 100, 100, 255))
        .width(3.0)
        .done();

// Use as a Ratatui widget
frame.render_stateful_widget(
    PixelCanvasWidget::new(canvas),
    area,
    &mut pixel_canvas_state,
);
```

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────┐
│                 Your TUI App                     │
├──────────────────────────────────────────────────┤
│  Layer 3: Widget    (StatefulWidget, lifecycle)  │
├──────────────────────────────────────────────────┤
│  Layer 2: Transport (Kitty / Sixel / Halfblock)  │
├──────────────────────────────────────────────────┤
│  Layer 1: Drawing   (Scene → tiny-skia → Pixmap) │
└──────────────────────────────────────────────────┘
```

Each layer is independently usable. The drawing API has zero dependency on
Ratatui or any terminal protocol.

## 📊 Pixelchart

The `pixelchart` crate provides a high-level charting API on top of `ratatui-pixelcanvas`:

- **Scatter plots** — with customizable markers and colors
- **Line charts** — with fills, reference lines, and multiple series
- **Bar charts** — grouped and stacked variants
- **Histograms** — frequency and density modes
- **Box plots** — with whiskers, outliers, and statistical annotations
- **Heatmaps** — with configurable color scales
- **Interactive features** — zoom, pan, crosshair cursor, and tooltips
- **Themes** — built-in dark/light themes with full customization

## 🎨 Examples

```bash
# Basic shapes
cargo run --example simple_shapes

# 3D rotating cube
cargo run --example cube_3d

# Visual illusions
cargo run --example illusions

# New features showcase
cargo run --example new_features

# Animation demo
cargo run --example animation_demo

# Pixelchart showcase
cargo run -p pixelchart --example showcase
```

## ⚙️ Feature Flags

| Flag      | Default | Description                                      |
|-----------|---------|--------------------------------------------------|
| `kitty`   | ✅      | Kitty graphics protocol backend                  |
| `sixel`   | ❌      | Sixel graphics protocol backend                  |
| `widget`  | ✅      | Ratatui `StatefulWidget` integration              |
| `text`    | ❌      | Text rendering via `fontdue`                      |
| `shm`     | ❌      | Zero-copy Kitty transmission via POSIX shared mem |

## 🔧 Minimum Supported Rust Version

1.83.0

## 📄 License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT License](LICENSE-MIT) at your option.
