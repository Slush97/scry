# scry — Agent Context Sheet

> Read this FIRST before any implementation work. Then read ROADMAP.md for priorities.

## Workspace

```
scry/                          ← Rust workspace, MSRV 1.83.0, edition 2021
├── src/                       ← scry-engine (terminal vector graphics)
│   └── wasm.rs                ← WASM bridge (WasmCanvas, render_to_canvas)
├── crates/
│   ├── scry-chart/            ← 19 chart types, 6 themes, PNG/SVG export
│   ├── scry-learn/            ← 23+ ML models, preprocessing, metrics, viz
│   ├── scry-cli/              ← Unified CLI binary (`scry`)
│   └── scry-pipe/             ← Feature engineering compiler (Phase 1 done)
├── .agent/
│   ├── CONTEXT.md             ← THIS FILE
│   ├── ROADMAP.md             ← Sprints 8+ priorities (benchmarks, CUDA, gaps)
│   └── workflows/             ← Active workflow instructions
├── benches/                   ← Criterion benchmarks
├── tests/                     ← Integration tests, property tests
├── fuzz/                      ← 6 fuzz targets (cargo-fuzz)
└── examples/                  ← 27+ examples (including wasm_demo/)
```

## Dependency Graph

```
scry-engine          (foundation — no scry deps)
    ↑
scry-chart           (depends on scry-engine)
    ↑
scry-learn           (depends on scry-chart)

scry-cli             (depends on scry-engine + scry-chart)
scry-pipe            (standalone — no scry deps)
```

## Current Stats (v0.7.0)

| Metric | Value |
|--------|-------|
| Total tests | 428 |
| Clippy warnings | 0 |
| ML models | 23+ |
| Chart types | 19 |
| Viz functions | 19 |
| Fuzz targets | 6 |
| Miri tests passing | 134/135 |

## Conventions

### Code Quality
- `#![deny(unsafe_code)]` on ALL crates (engine opts in per-module for FFI only)
- `#![warn(missing_docs)]` — every public item gets a doc comment
- Clippy: `all + pedantic + nursery` warnings enabled
- `#[non_exhaustive]` on all public enums and error types

### Patterns
- **Builder pattern** for all models and configs: `.param(value).param(value)`
- **Prelude module** in every crate — all public API re-exported via `crate::prelude::*`
- **Error type** per crate: `PixelCanvasError`, `ChartError`, `ScryLearnError`, `PipeError`
- **Feature flags** for optional deps: `serde`, `widget`, `text`, `svg`, `shm`, `wasm`
- **Serde opt-in**: `#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]`

### Testing
- Unit tests: inline `#[cfg(test)] mod tests` in each source file
- Correctness proofs: `tests/correctness.rs` (scry-learn) — verify against sklearn JSON refs
- Benchmarks: `benches/` with Criterion — compare against linfa, smartcore, scikit-learn
- Fuzz: `fuzz/fuzz_targets/` — 6 targets covering parsing, rendering, scaling
- Miri: 134/135 tests pass

### Dependencies (hard rules)
- **No BLAS/LAPACK** in production deps — all math is pure Rust (CUDA is opt-in feature only)
- **No nalgebra/ndarray** in production deps — custom matrix ops only
- `tiny-skia` is the rasterization backend (not skia-safe, not wgpu)
- `rayon` for parallelism in scry-learn and scry-pipe

## Agent Scoping Rules

1. **Read ROADMAP.md** to understand current priorities (Sprint 8+)
2. **Read only the prelude** of adjacent crates, not their internals
3. **Read the specific workflow** you're implementing, not all workflows
4. **Run verification commands** before declaring done
5. **Never break existing tests** — all 428 must continue to pass

### Context Loading Order
```
1. .agent/CONTEXT.md                  ← you are here
2. .agent/ROADMAP.md                  ← current priorities
3. .agent/workflows/{relevant}.md     ← specific session only
4. Files listed in that session
```

## Quick Reference

| Item | Value |
|------|-------|
| Repo | `github.com/Slush97/scry` |
| License | MIT OR Apache-2.0 |
| Rust edition | 2021 |
| MSRV | 1.83.0 |
| Current version | 0.7.0 (pre-1.0) |
| Test command | `cargo test -p {crate}` |
| Clippy command | `cargo clippy -p {crate} -- -D warnings` |
| Workspace check | `cargo check --workspace` |
| Benchmark | `cargo bench --bench {name} -p {crate}` |
