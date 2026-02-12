# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Ellipse primitive** (`ellipse(cx, cy, rx, ry)`) with optional rotation.
- **Polyline primitive** (`polyline(points)`) for connected open line segments.
- **Polygon primitive** (`polygon(points)`) for closed filled shapes.
- **Mutable composition** via `PixelCanvas::push_command()` for conditional drawing.
- **Color conversion** between `ratatui::style::Color` and `pixelcanvas::Color`
  (bidirectional `From` impls, gated on `widget` feature).
- **`LineBuilder::stroke(color, width)`** for API consistency with `ShapeBuilder`.
- **`KittyBackend::into_writer()`** for extracting the underlying writer.
- **`TransmitFormat`** enum on `KittyBackend` — choose between raw RGBA and PNG
  transmission.
- **macOS support** for font-size detection (`TIOCGWINSZ` constant).
- **Halfblock rendering** actually works now — the widget correctly renders
  halfblock cells for terminals without graphics protocol support.
- Integration test suite (`tests/pipeline_test.rs`) covering the full
  scene → rasterize → transport pipeline.
- Unit tests for halfblock backend (cell rendering, alpha compositing, protocol
  trait methods).

### Fixed

- **`PathData` hash collisions** — hashing now includes full path geometry
  (verbs + points) instead of just bounding box and segment count.
- **Halfblock alpha compositing** — semi-transparent pixels now correctly
  composite against a black background.
- **`pixel_at` bounds check** — corrected from `idx + 2` to `idx + 3` for
  RGBA data.
- **Kitty image ID 0** — `next_id()` now skips reserved ID 0.
- **Redundant `#[cfg(feature = "widget")]`** in `widget/mod.rs` removed (the
  parent module is already gated).
- **Redundant doc link targets** in `skia.rs`.
- **Dead code warning** resolved by introducing `TransmitFormat` enum.

### Changed

- **MSRV bumped to 1.83.0** for stable `const fn` float operations.
- **`Color` documentation** corrected — stores straight RGBA, not premultiplied.
- **Base64 encoding buffer** in `KittyBackend` is now reused across frames.

## [0.1.0] — Initial Release

- Scene builder API with circle, rectangle, line, path, gradient, and group
  commands.
- Rasterization via `tiny-skia`.
- Kitty graphics protocol backend.
- Halfblock Unicode fallback backend.
- Ratatui `StatefulWidget` integration with content-hash caching.
- Protocol auto-detection via `Picker`.
