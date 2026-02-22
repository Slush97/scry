# Architecture

## Overview

scry-engine is a layered graphics engine for terminal rendering.
It builds anti-aliased 2D scenes, optionally renders 3D SDF objects,
and ships pixels to the screen via protocol-specific backends.

## Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│   Scene     │────▶│  Rasterize  │────▶│  Transport   │
│  (builder)  │     │  (GPU/CPU)  │     │  (protocol)  │
└─────────────┘     └─────────────┘     └──────────────┘
     │                    │                     │
 DrawCommand[]        Pixmap (RGBA)        escape sequences
```

## Layers

### 1. Scene (`src/scene/`)

Fluent builder API producing a `Vec<DrawCommand>` display list.
No I/O, no pixel work — purely declarative.

- **builder.rs** — `PixelCanvas` fluent API (shapes, gradients, text, groups)
- **command.rs** — `DrawCommand` enum (every drawable type)
- **style.rs** — Color (Oklab, HSL, sRGB), Fill, Stroke, Transform
- **animation.rs** — 20+ easing curves, `Transition`, `Keyframes`, `Spring`

### 2. Rasterize (`src/rasterize/`)

Converts display lists into `tiny_skia::Pixmap` (RGBA pixel buffers).

- **skia/** — CPU rasterizer wrapping `tiny-skia`
- **wgpu.rs** — GPU rasterizer via `wgpu` (shapes, lines, gradients, mesh)
- **backend.rs** — `RasterBackend` trait, `AutoBackend` (GPU→CPU fallback)
- **pipeline.rs** — `RasterPipeline` (caching + backend selection)
- **cache.rs** — Content-hash dirty-tile diffing (`RasterCache`)
- **batch.rs** — Command batching for GPU instancing
- **profiler.rs** — Per-command-type timing instrumentation

### 3. Transport (`src/transport/`)

Serializes pixmaps into terminal protocol escape sequences.

- **Kitty** — zlib-compressed PNG payload + SHM zero-copy variant
- **Sixel** — Median-cut 256-color quantization
- **iTerm2** — Inline Base64-PNG
- **Halfblock** — Unicode half-block fallback (no protocol required)
- **probe.rs** — Active terminal detection (XTVERSION, DA1, DA2)
- **window.rs** — Native OS window via `winit` + `softbuffer`

### 4. GPU (`src/gpu/`)

Shared GPU device management and pipeline compilation.

- **device.rs** — `GpuDevice` singleton (`OnceLock` + 3s timeout)
- **pipeline_registry.rs** — Lazy `OnceLock`-backed pipeline sets (2D, SDF, 3D)

### 5. SDF (`src/sdf/`)

Signed Distance Field 3D renderer with CPU ray marching and GPU compute.

- **renderer.rs** — CPU ray marcher
- **gpu_renderer.rs** — GPU compute shader dispatch + readback
- **pipeline.rs** — `SdfPipeline` with double-buffered GPU→display pipeline

### 6. Widget (`src/widget/`)

Ratatui `StatefulWidget` integration with two-phase render/flush lifecycle.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `AutoBackend` GPU→CPU fallback | Transparent degradation on headless/CI |
| Content-hash caching | Skip redundant rasterization in animation loops |
| `OnceLock` pipeline registry | One-time compilation, shared across contexts |
| `#![deny(unsafe_code)]` at root | Only 14 unsafe blocks in FFI boundary modules |
| Protocol auto-detection | Works across Kitty, iTerm2, WezTerm, etc. |

## Feature Flags

Default: `kitty,widget,gpu`.
Full: adds `text,svg,sdf,sdf-gpu,sdf-text,input,window,logging`.

## Workspace Dependency DAG

```
scry-engine (core) ← scry-chart ← scry-cli
scry-learn (independent)
scry-llm (independent)
scry-pipe → scry-learn
scry-terminal (uses scry-engine's wgpu/winit, but not scry-engine crate)
```

| Crate | Depends On |
|-------|-----------|
| `scry-engine` | *(core — no workspace deps)* |
| `scry-chart` | `scry-engine` |
| `scry-cli` | `scry-engine`, `scry-chart` |
| `scry-learn` | *(independent)* |
| `scry-llm` | *(independent)* |
| `scry-pipe` | `scry-learn` |
| `scry-terminal` | `wgpu`, `winit` (shared deps, not `scry-engine` crate) |

