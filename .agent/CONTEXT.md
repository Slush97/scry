# scry — Agent Context Sheet

> Read this FIRST before any implementation work. Then read the TRACK file for your crate.

## Workspace

```
scry/                          ← Rust workspace, MSRV 1.83.0, edition 2021
├── src/                       ← scry-engine (terminal vector graphics)
├── crates/
│   ├── scry-chart/            ← 17 chart types, themes, PNG/SVG export
│   ├── scry-learn/            ← 15+ ML models, preprocessing, metrics, viz
│   ├── scry-cli/              ← Unified CLI binary (`scry`)
│   └── scry-pipe/             ← Feature engineering compiler (proposal only)
├── .agent/
│   ├── CONTEXT.md             ← THIS FILE
│   ├── ROADMAP.md             ← Cross-product priorities & sprint plan
│   ├── TRACK_ENGINE.md        ← Engine status & roadmap
│   ├── TRACK_CHART.md         ← Chart status & roadmap
│   ├── TRACK_LEARN.md         ← ML status & roadmap
│   └── workflows/             ← Detailed session-by-session implementation guides
├── benches/                   ← Criterion benchmarks
├── tests/                     ← Integration tests, property tests
├── fuzz/                      ← 6 fuzz targets (cargo-fuzz)
└── examples/                  ← 27+ examples
```

## Dependency Graph

```
scry-engine          (foundation — no scry deps)
    ↑
scry-chart           (depends on scry-engine)
    ↑
scry-learn           (depends on scry-chart)

scry-cli             (depends on scry-engine + scry-chart)
scry-pipe            (standalone, proposed integration with scry-learn)
```

## Conventions

### Code Quality
- `#![deny(unsafe_code)]` on ALL crates (engine opts in per-module for FFI only)
- `#![warn(missing_docs)]` — every public item gets a doc comment
- Clippy: `all + pedantic + nursery` warnings enabled
- `#[non_exhaustive]` on all public enums and error types

### Patterns
- **Builder pattern** for all models and configs: `.param(value).param(value)`
- **Prelude module** in every crate — all public API re-exported via `crate::prelude::*`
- **Error type** per crate: `PixelCanvasError`, `ChartError`, `ScryLearnError`
- **Feature flags** for optional deps: `serde`, `widget`, `text`, `svg`, `shm`
- **Serde opt-in**: `#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]`

### Testing
- Unit tests: inline `#[cfg(test)] mod tests` in each source file
- Correctness proofs: `tests/correctness.rs` (scry-learn) — verify against known sklearn outputs
- Benchmarks: `benches/` with Criterion — always compare against linfa, smartcore, scikit-learn
- Fuzz: `fuzz/fuzz_targets/` — 6 targets covering parsing, rendering, scaling
- Miri: 125/126 core tests pass, 9/9 chart tests pass

### Dependencies (hard rules)
- **No BLAS/LAPACK** in production deps — all math is pure Rust
- **No nalgebra/ndarray** in production deps — custom matrix ops only
- `tiny-skia` is the rasterization backend (not skia-safe, not wgpu)
- `rayon` for parallelism in scry-learn (RF predict, future GBT)

## Agent Scoping Rules

1. **Read your TRACK file** to understand what's done and what's next
2. **Read only the prelude** of adjacent crates, not their internals
3. **Read the specific workflow session** you're implementing, not all sessions
4. **Update your TRACK file** after completing work
5. **Run verification commands** listed in the workflow before declaring done

### Context Loading Order
```
1. .agent/CONTEXT.md           ← you are here
2. .agent/TRACK_{YOUR_CRATE}.md
3. .agent/workflows/{relevant}.md  (specific session only)
4. Files listed in that session's "Files to modify"
```

## Quick Reference

| Item | Value |
|------|-------|
| Repo | `github.com/Slush97/scry` |
| License | MIT OR Apache-2.0 |
| Rust edition | 2021 |
| MSRV | 1.83.0 |
| All versions | 0.1.0 (pre-1.0) |
| Test command | `cargo test -p {crate}` |
| Clippy command | `cargo clippy -p {crate} -- -D warnings` |
| Workspace check | `cargo check --workspace` |
| Benchmark | `cargo bench --bench {name} -p {crate}` |
