# scry-chart — Development Tracker

> **Crate**: `crates/scry-chart` | **Tier**: Advanced Beta | **Version**: 0.1.0
> **Updated**: 2026-02-14

## Status

17 chart types shipped. Sessions 1–5 of formatting overhaul complete.
Full progress log: `crates/scry-chart/FORMATTING_PROGRESS.md`

### Chart Types (17)
Line, Line-XY, Scatter, Bubble, Bar, Histogram, Box Plot, Heatmap, Pie, Radar, Candlestick, Funnel, Waterfall, Gauge, Lollipop, Sparkline, Violin

### Completed Features
- **Formatters**: Auto, SI, FixedDecimal, Percent, Currency, Thousands, DateTime, Scientific, BinarySI, Engineering, SemanticZoom, Null, Fn — plus LocaleFormatter decorator
- **Locale**: US, European, Swiss, Indian number formatting
- **Axes**: Tick rotation (horizontal/45°/90°/custom angle), auto-skip, zero line, semantic zoom, shared axes
- **Legends**: Title, horizontal/vertical, multi-column, inside/outside positioning
- **Annotations**: Data labels, trend lines, reference lines, conversion rates
- **Export**: PNG + SVG, configurable DPI, subplot grid export
- **Themes**: dark, light, ocean, forest, pastel, plus custom palette builder
- **Serde**: All chart configs serializable (opt-in `--features serde`)
- **Subplots**: SubplotGrid with shared axes (SharedAxisMode: None/ShareX/ShareY/ShareBoth)
- **Visual polish**: Funnel trapezoids, waterfall connectors, compact K/M formatters

### Test Coverage
- 120+ unit tests, 57+ integration tests, 22+ doc tests
- Render tests for subplot layouts

## Roadmap (Priority Order)

### P0 — Quality Hardening (from audit)
> **Workflow:** `.agent/workflows/chart-quality-roadmap.md` (4 sessions)
- [x] Session 1: Heatmap RenderContext migration + `contrast_text_color()` helper
- [x] Session 2: Font scaling consistency + proportional offsets (gauge, radar, y-label)
- [x] Session 3: Code hygiene (AVG_CHAR_WIDTH dedup) + SVG standards compliance
- [x] Session 4: Theme-aware funnel colors + quality audit test suite
- [x] Kill all 33 pre-existing clippy warnings
- [x] CI/CD pipeline (shared with workspace)

### P1 — High Value Features
- [x] Null / gap handling in lines (`GapPolicy` enum — skip, interpolate, zero)
- [ ] Running average / smoothing (SMA, EMA, LOESS overlays)
- [ ] Statistical bands (confidence intervals, prediction bands)
- [ ] Interactive legend toggle (click to show/hide series)
- [ ] Color bar / gradient legend for heatmaps

### P2 — Innovation
- [ ] Perceptual emphasis axis (Oklab-weighted tick distribution)
- [ ] Density-adaptive grid (map-style zoom levels)
- [ ] Streaming-aware axis (tick anchoring, hysteresis for live data)
- [ ] Animated label transitions
- [ ] Micro-chart mode (sparkline-quality at any size)

### P3 — Polish
- [ ] SVG accessibility (ARIA roles, alt text)
- [ ] High contrast theme
- [ ] Theme inheritance (extend base themes)
- [ ] Gallery / cookbook documentation
- [ ] Base64 inline PNG output for notebooks

## Key Files
- `src/chart/` — 18 builder modules + unified `ChartConfig`
- `src/layout/` — 17 renderers (one per chart type)
- `src/formatter.rs` — 13 tick formatters + SemanticZoom + locale support
- `src/subplot.rs` — SubplotGrid, SharedAxisMode
- `src/theme.rs` — 6 built-in themes
- `src/export.rs` / `src/svg_export.rs` — PNG + SVG + subplot export
- `src/chart/extent.rs` — data extent extraction for shared axes
