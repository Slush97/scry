---
description: Continue the production-grade chart formatting overhaul for pixelchart
---

# Chart Formatting Overhaul — Cross-Session Workflow

## Progress Tracker

The canonical progress file is at:
`crates/pixelchart/FORMATTING_PROGRESS.md`

**Every agent working on this project MUST:**
1. Read `crates/pixelchart/FORMATTING_PROGRESS.md` first to see what's done
2. Update it after completing each item
3. Run verification after each phase

## Context Files to Read

Before starting work, read these files to understand the architecture:
- `crates/pixelchart/src/formatter.rs` — TickFormatter trait + built-in formatters
- `crates/pixelchart/src/scale.rs` — Tick generation, nice_ticks, format_tick_adaptive
- `crates/pixelchart/src/axis.rs` — Axis rendering, draw_axis, AxisConfig
- `crates/pixelchart/src/layout/mod.rs` — Layout engine, label measurement
- `crates/pixelchart/src/legend.rs` — Legend rendering
- `crates/pixelchart/src/theme.rs` — Theme tokens (GridTheme, AxisTheme, etc.)
- `crates/pixelchart/src/cursor.rs` — Tooltip formatting
- `crates/pixelchart/tests/axis_diagnostic.rs` — Diagnostic test suite

## Verification Commands

// turbo-all

1. Run unit tests:
```bash
cargo test -p pixelchart
```

2. Run axis diagnostic test:
```bash
cargo test -p pixelchart --test axis_diagnostic -- --nocapture
```

3. Run full workspace check:
```bash
cargo check --workspace
```

4. Run clippy:
```bash
cargo clippy -p pixelchart -- -D warnings
```
