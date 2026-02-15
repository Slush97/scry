# scry

Vector graphics engine for the terminal. Rasterizes with tiny-skia, transmits
via Kitty/Sixel/iTerm2/halfblock. Optional Ratatui widget layer.

## Workspace

- `src/` — Core engine (`scry-engine`): scene builder, rasterizer, transport backends
- `crates/scry-chart/` — Charting library (10 chart types, themes, PNG/SVG export)
- `crates/scry-cli/` — CLI tool (`scry` binary)
- `crates/scry-learn/` — ML library (linear regression, CART, Random Forest)
- `examples/` — 25 demo programs for the core engine
- `crates/scry-chart/examples/` — Chart examples (`showcase.rs`, `interactive.rs`)

## Commands

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets
cargo doc -p scry-engine --all-features --open

cargo run -p scry-chart --example showcase
cargo run --example animation_demo

# CLI
scry chart render < chart.json
scry chart plot --csv data.csv -y revenue
scry chart example line
scry info
```

## Stack

- **Rust** (MSRV 1.83.0)
- **tiny-skia** — 2D rasterization
- **fontdue** — text rendering (feature `text`)
- **resvg** — SVG rendering (feature `svg`)
- **ratatui** — widget integration (feature `widget`)
- **serde/clap** — CLI parsing

## Conventions

- `#[non_exhaustive]` on all public types
- Builder pattern everywhere: `canvas.circle(…).fill(…).done()`
- Three layers: Drawing → Transport → Widget
- Feature flags: `kitty` (default), `sixel`, `iterm2`, `widget` (default), `text`, `shm`, `svg`
