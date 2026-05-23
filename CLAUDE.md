# scry

Vector graphics engine for the terminal + charting library.

## Workspace Layout

| Crate | Path | Purpose |
|-------|------|---------|
| `scry-engine` | `src/` | Core engine: scene builder, rasterizer (tiny-skia), transport (Kitty/Sixel/iTerm2/halfblock) |
| `scry-chart` | `crates/scry-chart/` | 18 chart types, 6 themes, PNG/SVG export, 3D interactive viz |
| `scry-cli` | `crates/scry-cli/` | CLI tool (`scry` binary) |
| `scry-pipe` | `crates/scry-pipe/` | Feature pipeline IR + codegen |
| `examples/` | `examples/` | 30+ demo programs for the core engine |
| `fuzz/` | `fuzz/` | libfuzzer targets (chart) |

## Commands

```bash
# Build & verify
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets
cargo fmt --all -- --check

# Crate-specific testing
cargo test -p scry-chart --release
cargo test -p scry-engine --release

# Documentation
cargo doc -p scry-engine --all-features --open
```

## Stack

- **Rust** (MSRV 1.83.0)
- **tiny-skia** — 2D rasterization
- **fontdue** — text rendering (engine feature `text`, always-on in scry-chart)
- **ratatui** — widget integration (feature `widget`)
- **rayon** — parallel computation (scry-pipe, scry-engine)
- **wgpu** — GPU rasterization (feature `gpu`)
- **clap** — CLI parsing (scry-cli only)
- **serde** — serialization (optional in scry-engine)

## Architecture Rules

### Code Conventions
- `#[non_exhaustive]` on public enums and error types
- Builder pattern for configuration
- Deterministic RNG: `fastrand::Rng::with_seed(42)` for all data generation in tests/benchmarks
- Feature flags (scry-engine): `kitty` (default), `widget` (default), `gpu` (default), `sixel`, `iterm2`, `text`, `shm`, `svg`, `wasm`, `sdf`, `window`, `serde`

### Test & Benchmark Integrity
- **No marketing language in test/benchmark files.** Output only measured numbers.
- **Use `std::hint::black_box()`** for all timing measurements to prevent compiler elision.
- **Warmup iterations** (2+) before timing loops.

## Known Issues

- `scry-chart/src/formatter/mod.rs` — 665 lines, locale extracted to `formatter/locale.rs`
