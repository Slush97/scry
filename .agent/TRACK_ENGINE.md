# scry-engine — Development Tracker

> **Crate**: root (`src/`) | **Tier**: Production-Ready | **Version**: 0.1.0
> **Updated**: 2026-02-14

## Status

Stable core with full protocol coverage (Kitty, Sixel, iTerm2, halfblock).
Unsafe audit complete. 6 fuzz targets. 125/126 Miri tests pass.
Animation system with 20+ easing curves, Oklab color interpolation, keyframe timelines.

## Roadmap (Priority Order)

### P0 — Ship Blockers
- [ ] CI/CD pipeline (GitHub Actions: test, clippy, fuzz, Miri)
- [ ] `NO_COLOR` / `TERM=dumb` fallback behavior
- [ ] Bump to 0.2.0 with semver contract (public API freeze)

### P1 — High Value
- [ ] WASM rasterization target (`tiny-skia` → `wasm32-unknown-unknown`)
- [ ] GPU-accelerated rasterization path (optional `wgpu` backend)
- [ ] Headless PNG export without terminal dependency

### P2 — Polish
- [ ] Pywal integration (`~/.cache/wal/colors.json` palette auto-detection)
- [ ] Multiple animation presets (`wave`, `particles`, `minimal`, `geometry`)
- [ ] `tarpaulin-report.json` cleanup (stale old-name references)

## Key Files
- `src/scene/` — builder API, animation, style
- `src/rasterize/` — skia backend, batch, cache, profiler
- `src/transport/` — kitty, sixel, iterm2, halfblock, shm, picker
- `src/widget/` — Ratatui `StatefulWidget` integration
