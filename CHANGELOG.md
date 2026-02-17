# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added — `scry-chart` (new crate)

- **10 chart types** — line, line XY, scatter, bar, histogram, box plot,
  heatmap, pie, radar, and candlestick/OHLC.
- **Formatter system** — `AutoFormatter`, `SiFormatter`, `FixedFormatter`,
  `PercentFormatter`, `CurrencyFormatter`, `ThousandsFormatter`,
  `DateTimeFormatter` with batch formatting for uniform precision.
- **Locale support** — US, European, Swiss, and Indian number formatting
  conventions (decimal/thousands separators).
- **Interactive features** — zoom, pan, crosshair cursor, and tooltips via
  `ChartState`.
- **Legends** — configurable position (inside/outside), orientation
  (horizontal/vertical), multi-column layout, and titles.
- **Annotations** — data point labels, trend lines, reference lines
  (horizontal/vertical) with custom colors.
- **Tick rotation** — horizontal, diagonal (45°), vertical (90°), and custom
  angle rotation for X-axis tick labels.
- **Export** — PNG (`save_png`) and SVG (`render_to_svg` / `save_svg`) output.
- **Themes** — `dark`, `light`, `ocean`, `forest`, `pastel` built-in themes
  with full customization via `Theme` builder.
- **Builder API** — fluent builder pattern for all chart types with
  `#[must_use]` annotations.
- **Scales** — `LinearScale`, `LogScale`, `CategoricalScale`, `TimeScale`
  with nice domain rounding and adaptive tick generation.

### Added — `scry-cli` (new crate)

- **`render` command** — generate charts from JSON (stdin or `--data`).
- **`plot` command** — generate charts from CSV with column selection,
  delimiter config, header detection, axis bounds, sorting, and binning.
- **`example` command** — render built-in example charts inline or to file.
- **`show` command** — display existing PNG images inline in the terminal.
- **`info` command** — print terminal capabilities and supported chart types.

### Added — `scry-engine`

- **Sixel graphics protocol** (`--features sixel`) — full DEC Sixel backend with
  median-cut color quantization (256 colors), run-length encoding, and cursor
  positioning.
- **iTerm2 inline image protocol** (`--features iterm2`) — OSC 1337 backend with
  a zero-dependency PNG encoder and CRC32 checksums for iTerm2, WezTerm, and Mintty.
- **POSIX shared memory** (`--features shm`) — zero-copy Kitty transmission via
  `shm_open` / `mmap`. Sends a ~200 byte control sequence instead of megabytes
  of base64.
- **SVG rendering** (`--features svg`) — parse and rasterize SVG via `resvg`.
  `SvgImage::render()` with aspect-ratio preservation. `SvgWidget` for Ratatui.
- **SVG line drawing animation** — `SvgLineDrawing` progressively reveals SVG
  paths with animated dash patterns, pen pressure simulation, and glowing pen tip.
- **Animation system** — 20+ easing curves (CSS-standard + spring, elastic,
  bounce), `Lerp` trait with impls for `f32`, `f64`, `Color` (Oklab), `Point`,
  and `Transform` (decomposed interpolation). `Transition` orchestrator,
  `Keyframes` timeline, `AnimationState` frame-level controller.
- **Oklab color interpolation** — `Color::mix()` uses perceptual Oklab space
  for smooth gradients. `Color::from_hsla()`, `Color::with_lightness()`.
- **Command batching** — consecutive same-style shape commands are merged into
  compound paths, reducing `fill_path`/`stroke_path` call overhead.
- **Pipeline profiling** — `ProfiledRasterizer` with per-command-type timing,
  `ProfileHistory` with rolling median/P95, and `TransportProfile` breaking down
  compress/encode/I/O time.
- **Dirty-tile transmission** — `RasterCache::compute_dirty_tiles()` detects
  changed 64×64 pixel regions; `PixelCanvasWidget::incremental()` transmits
  only dirty tiles, reducing bandwidth for partially-animated scenes.
- **Ellipse primitive** (`ellipse(cx, cy, rx, ry)`) with optional rotation.
- **Polyline primitive** (`polyline(points)`) for connected open line segments.
- **Polygon primitive** (`polygon(points)`) for closed filled shapes.
- **Arc primitive** (`arc(cx, cy, r, start, sweep)`) via cubic Bézier approximation.
- **`PixelCanvas::push_command()`** for mutable conditional composition.
- **`PixelCanvas::clear()`** for animation loops.
- **Color conversion** between `ratatui::style::Color` and `scry-engine::Color`
  (bidirectional `From` impls, gated on `widget` feature).
- **`LineBuilder::stroke(color, width)`** for API consistency with `ShapeBuilder`.
- **`KittyBackend::into_writer()`** for extracting the underlying writer.
- **`TransmitFormat`** enum on `KittyBackend` — choose between raw RGBA, zlib
  RGBA, PNG, and shared memory transmission.
- **macOS support** for font-size detection (`TIOCGWINSZ` constant).
- **Halfblock rendering** — the widget correctly renders halfblock cells for
  terminals without graphics protocol support, with alpha compositing and
  flat-buffer reuse.
- **Semver safety** — `#[non_exhaustive]` on all public enums and structs.
- Integration test suite (`tests/pipeline_test.rs`) covering the full
  scene → rasterize → transport pipeline.
- Property tests (`tests/property_tests.rs`) via `proptest`.
- Benchmark suite (`benches/`) with rasterization, efficiency, and chart benchmarks.
- Fuzz testing harness (`fuzz/`) with 6 fuzz targets.

### Added — `scry-learn`

- **DenseMatrix** — contiguous column-major `Vec<f64>` storage with zero-cost
  `col(j)` slice access. All models migrated from `Vec<Vec<f64>>`.
- **SVD/QR solvers** — Golub-Kahan SVD and Householder QR for linear regression.
  `LinRegSolver::Svd`, `LinRegSolver::Qr`, `LinRegSolver::Normal` selection.
- **CART builder optimization** — pre-filtered index arrays, incremental variance
  computation, and buffer reuse replace membership bitsets.
- **MLP neural networks** — `MLPClassifier` and `MLPRegressor` with configurable
  hidden layers, activations (ReLU, Sigmoid, Tanh), and optimizers (SGD, Adam).
- **Sparse matrix support** — `CsrMatrix` and `CscMatrix` with `from_triplets`,
  row/col views, `dot_vec`, CSR↔CSC conversion, and arithmetic ops.
- **Sparse-aware algorithms** — `fit_sparse`/`predict_sparse` for LinearRegression,
  LogisticRegression, Lasso, ElasticNet, GaussianNB, MultinomialNB, KNN.
  `fit_sparse`/`transform_sparse` for StandardScaler, MinMaxScaler.
- **Sparse dataset integration** — `Storage` enum (Dense/Sparse),
  `Dataset::from_sparse()`, sparse-aware `subset()` and `train_test_split`.
- **Incremental learning** — `PartialFit` trait with `partial_fit(&mut self, &Dataset)`.
  Implemented for LogisticRegression, GaussianNB, MiniBatchKMeans, MLPClassifier,
  MLPRegressor.
- **Large-scale benchmarks** — Criterion benchmarks for PCA, LinearRegression, and
  tree models at 100K/1M row scale with throughput metrics.

### Added — `scry-pipe` (new crate)

- **Pipeline IR** — `PipelineDef`, `PipelineStep`, `TransformOp` with JSON
  serialization. 10 transform operations with all fitted parameters baked in.
- **Execution engine** — `PipelineEngine` for runtime pipeline evaluation.
- **Rust codegen** — compile pipeline definitions to standalone Rust code
  (feature `codegen`).
- **Fuzz targets** — `fuzz_ir_roundtrip` and `fuzz_pipeline_transform`.

### Fixed

- **LineChart builder** — added missing `margin()` and `y_inverted()` methods.
- **Axis label/tick collision** — fixed spacing in common overlays layout.
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
