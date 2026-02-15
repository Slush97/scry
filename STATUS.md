# scry — Current Status

> **Date:** 2026-02-14 | **Repo:** `Slush97/scry` | **Local:** `/home/esoc/code/scry`

## ✅ Completed This Session

### 1. Looping Background Startup Animation (`examples/startup_anim.rs`)
- Forks to background via `libc::fork()` — shell prompt appears immediately
- Animation loops infinitely (6s cycle, sacred geometry)
- Auto-exits on stdout write failure (broken pipe = terminal closed)
- 120s safety-net timeout as fallback
- Colors matched to user's `fastfetch config.jsonc` + `kitty.conf` palette
- Kitty terminal accent colors (`#4A76E6`, `#4AA3A3`, `#B04AA2`) in palette
- Floating sparkle particles, thicker geometry strokes
- Shell script at `examples/startup_anim.sh` for `.bashrc` integration
- `libc` added as dev-dependency

### 2. Crate Rename: `ratatui-pixelcanvas` → `scry-engine`
- Package name updated in all `Cargo.toml` files (root, scry-chart, scry-cli, fuzz)
- All `use ratatui_pixelcanvas::` → `use scry_engine::` across ~50 source files
- All docs (README, CHANGELOG, CLAUDE.md, SAFETY.md) updated
- Description: "Pixel-perfect vector graphics for **terminals**" (not "for Ratatui")
- GitHub repo renamed: `Slush97/scry`
- Local directory: `/home/esoc/code/scry`
- Kitty `startup.session` path updated
- **Verified:** `cargo build --workspace` ✅ | `cargo clippy` ✅ | `cargo test` 90/91 ✅

## 📋 Planned Next Steps (Not Yet Started)

### 3. Expand CLI to unified `scry` binary
Currently `scry-cli` only does charts. Plan to expand into a unified CLI:
```
scry chart line data.csv       # existing chart functionality
scry splash --preset geometry  # productized startup_anim
scry render image.png          # display images via Kitty
scry play --preset wave        # animation presets
```

### 4. Ricer-Friendly Features
- CLI flags: `--rows`, `--preset`, `--palette`, `--duration`
- Pywal integration (read `~/.cache/wal/colors.json`)
- Multiple animation presets (wave, particles, minimal, geometry)
- `NO_COLOR` / screenshot mode support

## 🐛 Known Issues
- 1 pre-existing test failure: `axis::tests::auto_skip_y_axis_uses_vertical_spacing` (scry-chart, unrelated to rename)
- `tarpaulin-report.json` still contains old name references (auto-generated, will refresh on next coverage run)

## Product Architecture
```
scry-engine (engine crate)          ← core rendering engine
├── scry-chart (charting library)   ← 10 chart types, themes, export
└── scry-cli (CLI tool)             ← to be expanded as unified scry CLI
```
